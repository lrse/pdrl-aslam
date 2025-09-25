# ros2_training_wrapper.py
# Keep ROS2 callbacks running in a background executor (for covariance subs),
# but publish TF/stereo ONLY from the sim thread after each env.step().

import time
import threading
import gymnasium as gym

import rclpy
from rclpy.utilities import ok as ros_ok
from rclpy.executors import MultiThreadedExecutor


class Ros2TrainingWrapper(gym.Wrapper):
    """Runs RobotDataManager in a background executor and enforces a per-env
    'fresh covariance per step' contract with warm-up after resets.

    Contract (per env):
      - After any reset (full or partial), allow `warmup_steps` ungated steps for that env.
      - Once that env produces its first post-reset covariance, mark it ARMED.
      - On each training step, block until all ARMED envs have provided NEW covariance
        relative to the last step. Newly reset (unarmed) envs are temporarily exempt.
      - Skip the very next pre-step wait once right after an env becomes ARMED
        (to avoid waiting before we publish new frames).

    Args:
        env: gym.Env
        dm_ctor: callable -> rclpy.Node (RobotDataManager)
        dm_kwargs: kwargs for dm_ctor
        subs_wait_timeout: seconds to wait once for cuVSLAM subscribers (0 = no wait)
        require_subs: if True, enforce subscriber wait
        cov_per_step_timeout: None => wait forever; float => max seconds per wait before raising
        warmup_steps: number of ungated steps per env after it resets (typically 1–2)
    """

    def __init__(
        self,
        env,
        dm_ctor,
        dm_kwargs=None,
        *,
        subs_wait_timeout: float = 0.0,
        require_subs: bool = False,
        cov_per_step_timeout: float | None = None,  # None => infinite wait
        warmup_steps: int = 1,
        warmup_time_s: float | None = None,         # reserved / unused
    ):
        super().__init__(env)

        # Initialize ROS if needed
        self._owns_rclpy = False
        if not ros_ok():
            rclpy.init(args=[])
            self._owns_rclpy = True

        # Bridge node
        self._dm = dm_ctor(**(dm_kwargs or {}))
        # Expose for obs/reward hooks
        self.env.unwrapped.data_manager = self._dm

        # Background executor to service subscriptions (NO timers touching GPU)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._dm)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # One-time subscriber wait (optional)
        self._subs_wait_timeout = float(subs_wait_timeout)
        self._require_subs = bool(require_subs)
        self._did_wait_subs = False

        # Timing / gating
        self._cov_per_step_timeout = cov_per_step_timeout
        self._warmup_steps = int(warmup_steps)
        self._warmup_time_s = warmup_time_s  # not used

        # Bookkeeping (filled in reset())
        self._num_envs = getattr(self._dm, "num_envs", 1)
        self._armed = [False] * self._num_envs
        self._cooldown = [self._warmup_steps] * self._num_envs
        self._seq_at_reset = [0] * self._num_envs
        self._last_cov_seq = [0] * self._num_envs
        self._skip_wait_once = [False] * self._num_envs  # one-shot after (re)arming

    # ----------------
    # Helper utilities
    # ----------------
    def _wait_for_cuvslam_subs(self, timeout: float) -> bool:
        if timeout <= 0.0:
            return True
        t0 = time.monotonic()

        def topics_for_env(i: int):
            ns = "unitree_go2" if self._num_envs == 1 else f"unitree_go2_{i}"
            return [
                f"{ns}/visual_slam/image_0",
                f"{ns}/visual_slam/camera_info_0",
                f"{ns}/visual_slam/image_1",  
                f"{ns}/visual_slam/camera_info_1",    
                # f"{ns}/visual_slam/imu", #TRYING IMU
            ]

        while (time.monotonic() - t0) < timeout:
            all_ok = True
            for i in range(self._num_envs):
                for topic in topics_for_env(i):
                    try:
                        if len(self._dm.get_subscriptions_info_by_topic(topic)) == 0:
                            all_ok = False
                            break
                    except Exception:
                        all_ok = False
                        break
                if not all_ok:
                    break
            if all_ok:
                try:
                    self._dm.get_logger().info("cuVSLAM subscribers detected on stereo topics for ALL envs.")
                except Exception:
                    pass
                return True
            time.sleep(0.05)

        try:
            self._dm.get_logger().warn(
                f"No cuVSLAM subscribers for all envs within {timeout:.1f}s. Continuing anyway."
            )
        except Exception:
            pass
        return False

    def _tick_ros(self):
        """Publish once from sim thread (safe) and let executor handle callbacks."""
        self._dm.pub_ros2_data()

    def _env_bool_list(self, x):
        """Convert terminated/truncated to a list[bool] of length N."""
        try:
            import numpy as np
            if hasattr(x, "detach"):
                arr = x.detach().cpu().numpy()
            elif hasattr(x, "__array__"):
                arr = np.asarray(x)
            elif isinstance(x, (list, tuple)):
                arr = x
            else:
                return [bool(x)] * self._num_envs
            return [bool(v) for v in arr.tolist()]
        except Exception:
            if isinstance(x, (list, tuple)):
                return [bool(v) for v in x]
            return [bool(x)] * self._num_envs

    def _poll_cov_seqs(self):
        return [self._dm.get_cov_seq(i) for i in range(self._num_envs)]

    def _wait_for_new_cov_on_armed(self) -> bool:
        """Block until every ARMED env (except those skipping once) has seq > last_cov_seq.
           Unarmed envs are ignored."""
        # Determine which envs we must actually wait on
        active = [i for i, a in enumerate(self._armed) if a and not self._skip_wait_once[i]]
        if not active:
            # Consume one-shot skips so that next loop will start waiting normally
            for i in range(self._num_envs):
                if self._skip_wait_once[i] and self._armed[i]:
                    self._skip_wait_once[i] = False
            return True

        t0 = time.time()
        while True:
            seqs = self._poll_cov_seqs()
            ready = all(seqs[i] > self._last_cov_seq[i] for i in active)
            if ready:
                for i in active:
                    self._last_cov_seq[i] = seqs[i]
                # Also consume any pending one-shot skips on ARMED envs
                for i in range(self._num_envs):
                    if self._skip_wait_once[i] and self._armed[i]:
                        self._skip_wait_once[i] = False
                return True

            if (self._cov_per_step_timeout is not None) and ((time.time() - t0) >= self._cov_per_step_timeout):
                return False
            time.sleep(0.001)

    # -------------
    # Gym API
    # -------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Optional: wait once for subscribers (does not touch GPU)
        if self._require_subs and not self._did_wait_subs and self._subs_wait_timeout > 0.0:
            self._wait_for_cuvslam_subs(timeout=self._subs_wait_timeout)
            self._did_wait_subs = True

        # Re-init per-env gating
        self._num_envs = getattr(self._dm, "num_envs", 1)
        seqs = self._poll_cov_seqs()
        self._seq_at_reset = seqs[:]                 # baseline at full reset
        self._last_cov_seq = seqs[:]
        self._armed = [False] * self._num_envs       # everyone starts unarmed
        self._cooldown = [self._warmup_steps] * self._num_envs
        self._skip_wait_once = [False] * self._num_envs

        # Do NOT publish here; first _tick_ros happens after first step
        return obs, info

    def step(self, action):
        # 1) Before stepping: wait for NEW covariance on all ARMED envs
        #    (but skip those that just armed on the previous step)
        if not self._wait_for_new_cov_on_armed():
            raise RuntimeError("Per-step covariance wait timed out. SLAM did not provide fresh pose(s).")

        # 2) Do the actual environment step
        obs, rew, terminated, truncated, info = self.env.step(action)

        # 3) Detect which envs reset this step (partial resets)
        term_b = self._env_bool_list(terminated)
        trunc_b = self._env_bool_list(truncated)
        just_reset = [bool(term_b[i] or trunc_b[i]) for i in range(self._num_envs)]
        if any(just_reset):
            seqs_now = self._poll_cov_seqs()
            for i, jr in enumerate(just_reset):
                if jr:
                    self._armed[i] = False
                    self._cooldown[i] = self._warmup_steps
                    self._seq_at_reset[i] = seqs_now[i]
                    self._last_cov_seq[i] = seqs_now[i]
                    self._skip_wait_once[i] = False
                    try:
                        self._dm.get_logger().info(f"Env {i} RESET: baseline seq={seqs_now[i]}")
                    except Exception:
                        pass

        # 4) Publish frames for SLAM (these will produce covariance for the NEXT wait)
        self._tick_ros()

        # 5) Post-step: count down warmup and (re)arm envs that have produced first cov
        seqs_after = self._poll_cov_seqs()
        for i in range(self._num_envs):
            if not self._armed[i]:
                if self._cooldown[i] > 0:
                    self._cooldown[i] -= 1
                if self._cooldown[i] == 0 and (seqs_after[i] > self._seq_at_reset[i]):
                    self._armed[i] = True
                    self._last_cov_seq[i] = seqs_after[i]
                    self._skip_wait_once[i] = True  # skip the very next pre-step wait once
                    try:
                        self._dm.get_logger().info(f"Env {i} ARMED (post-reset), seq={seqs_after[i]}")
                    except Exception:
                        pass

        return obs, rew, terminated, truncated, info

    def close(self):
        try:
            try:
                self._executor.remove_node(self._dm)
            except Exception:
                pass
            try:
                self._executor.shutdown()
            except Exception:
                pass
            if getattr(self, "_spin_thread", None):
                self._spin_thread.join(timeout=1.0)
            try:
                self._dm.destroy_node()
            except Exception:
                pass
        finally:
            if self._owns_rclpy and ros_ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
        return super().close()
