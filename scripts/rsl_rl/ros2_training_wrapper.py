"""This module contains the wrapper configuration to incorporate SLAM package
during training."""

from __future__ import annotations

import numpy as np
import torch

import time
import threading
import gymnasium as gym

import rclpy
from rclpy.utilities import ok as ros_ok
from rclpy.executors import MultiThreadedExecutor

from ros2_bridge import RobotDataManager

# To later avoid the one-frame lag.
OBS_XY_SLICE = slice(100, 102)


class Ros2TrainingWrapper(gym.Wrapper):
    """Runs RobotDataManager in a background executor and forces a per-environment
    fresh odometry per step.

    Overview:
    - Once that env produces its first post-reset odometry, mark it as "ARMED".
    - On each training step, block until all envs have provided new odometry."""

    def __init__(
        self,
        env
    ):
        super().__init__(env)

        # Initialize ROS. Track whether we created it, so close() can safely
        # call rclpy.shutdown() without killing other environments' ROS.
        self._owns_rclpy = False
        if not ros_ok():
            rclpy.init(args=[])
            self._owns_rclpy = True

        # Bridge node.
        self._dm = RobotDataManager(env=self.env.unwrapped)

        # Creates data_manager inside the base environment and inserts the node onto
        # it so we can use live robot data.
        self.env.unwrapped.data_manager = self._dm

        # Background executor to service subscriptions (no timers touching GPU).
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._dm)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._num_envs = getattr(self._dm, "num_envs", 1)
        self._armed = [False] * self._num_envs
        self._seq_at_reset = [0] * self._num_envs
        self._last_odom_seq = [0] * self._num_envs
        self._skip_wait_once = [False] * self._num_envs 

##
# Config helpers.
##

    def _fetch_slam_xy(self, i: int):
        try:
            xy = self._dm.get_slam_xy_at_last_odom(env_idx=i)
            if xy is None:
                return None
            x, y = float(xy[0]), float(xy[1])
            return x, y
        except Exception:
            return None

    # Normalize vectors.
    def _env_bool_list(self, x):
        try:
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

    def _poll_odom_seqs(self):
        return self._dm._snapshot_odom_seqs()

    # Blocks until new odometry messages arrive for ARMED environments.
    def _wait_for_new_odom_on_armed(self):
        active = [i for i in range(self._num_envs) if self._armed[i] and not self._skip_wait_once[i]]
        if not active:
            self._skip_wait_once = [False] * self._num_envs     # No skipping anymore but keeping it as a reminder.

            return self._poll_odom_seqs()

        while True:
            seqs = self._poll_odom_seqs()
            if all(seqs[i] > self._last_odom_seq[i] for i in active):
                for i in active:
                    self._last_odom_seq[i] = seqs[i]
                self._skip_wait_once = [False] * self._num_envs
                return seqs

            if not ros_ok() or (self._spin_thread and not self._spin_thread.is_alive()):
                raise RuntimeError("ROS node or executor stopped.")
            time.sleep(0.001)

##
# Gym API.
##

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._num_envs = self._dm.num_envs
        seqs = self._poll_odom_seqs()
        self._seq_at_reset = seqs[:]
        self._last_odom_seq = seqs[:]
        self._armed = [False] * self._num_envs
        self._skip_wait_once = [False] * self._num_envs

        # Delete any pre-reset SLAM (x,y) so we never reuse it.
        try:
            with self._dm._odom_lock:
                for i in range(self._num_envs):
                    self._dm._xy_at_odom_seq[i] = None
        except Exception:
            pass

        # Preset start pose since it starts at (0,0).
        target = obs["policy"] if (isinstance(obs, dict) and "policy" in obs) else obs
        try:
            if hasattr(target, "detach"):
                device, dtype = target.device, target.dtype
                with torch.no_grad():
                    for i in range(self._num_envs):
                        target[i, OBS_XY_SLICE] = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
            else:
                arr = np.asarray(target)
                for i in range(self._num_envs):
                    arr[i, OBS_XY_SLICE] = (0.0, 0.0)
                if isinstance(obs, dict) and "policy" in obs:
                    obs["policy"] = arr
                else:
                    obs = arr
        except Exception:
            pass

        # Wait until every environment has produced first odometry.
        while True:
            self._dm.pub_ros2_data()
            seqs = self._poll_odom_seqs()
            if all(seqs[i] > self._seq_at_reset[i] for i in range(self._num_envs)):
                self._armed = [True] * self._num_envs
                self._last_odom_seq = seqs[:]
                self._skip_wait_once = [False] * self._num_envs
                break
            if not ros_ok() or (self._spin_thread and not self._spin_thread.is_alive()):
                raise RuntimeError("ROS node or executor stopped during bootstrap.")
            time.sleep(0.001)

        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        term_b = self._env_bool_list(terminated)
        trunc_b = self._env_bool_list(truncated)
        just_reset = [bool(term_b[i] or trunc_b[i]) for i in range(self._num_envs)]
        if any(just_reset):
            seqs_now = self._poll_odom_seqs()
            for i, jr in enumerate(just_reset):
                if jr:
                    self._armed[i] = False
                    self._seq_at_reset[i] = seqs_now[i]
                    self._last_odom_seq[i] = seqs_now[i]
                    self._skip_wait_once[i] = False
                    # Delete any pre-reset SLAM (x,y) so we never reuse it.
                    try:
                        with self._dm._odom_lock:
                            self._dm._xy_at_odom_seq[i] = None
                    except Exception:
                        pass
                    try:
                        self._dm.get_logger().info(f"Env {i} RESET: baseline seq={seqs_now[i]}")
                    except Exception:
                        pass

        # Publish images for the new state.
        self._dm.pub_ros2_data()

        # Wait and get the exact seqs that satisfied the wait.
        ready_seqs = self._wait_for_new_odom_on_armed()

        for _ in range(8):
            snapshots = [self._dm.get_odom_seq_and_xy(i) for i in range(self._num_envs)]
            if all(
                (not self._armed[i] or self._skip_wait_once[i]) or
                (snapshots[i][0] >= ready_seqs[i])
                for i in range(self._num_envs)
            ):
                break
            time.sleep(0.0005)

        # Patch observation with SLAM (x,y).
        target = obs["policy"] if (isinstance(obs, dict) and "policy" in obs) else obs
        try:
            if hasattr(target, "detach"):
                device, dtype = target.device, target.dtype
                with torch.no_grad():
                    for i in range(self._num_envs):
                        seq_i, xy = snapshots[i]
                        if not self._armed[i]:
                            target[i, OBS_XY_SLICE] = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
                            continue
                        if xy is None:
                            target[i, OBS_XY_SLICE] = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
                            continue
                        target[i, OBS_XY_SLICE] = torch.tensor(xy, dtype=dtype, device=device)
            else:
                arr = np.asarray(target)
                for i in range(self._num_envs):
                    seq_i, xy = snapshots[i]
                    if not self._armed[i]:
                        arr[i, OBS_XY_SLICE] = (0.0, 0.0)
                        continue
                    if xy is None:
                        arr[i, OBS_XY_SLICE] = (0.0, 0.0)
                        continue
                    arr[i, OBS_XY_SLICE] = xy
                if isinstance(obs, dict) and "policy" in obs:
                    obs["policy"] = arr
                else:
                    obs = arr
        except Exception:
            pass

        self._last_ready_seqs = ready_seqs
        self._last_snapshots = snapshots

        # Arm any environment that produced first post-reset odometry.
        seqs_after = self._poll_odom_seqs()
        for i in range(self._num_envs):
            if (not self._armed[i]) and (seqs_after[i] > self._seq_at_reset[i]):
                self._armed[i] = True
                self._last_odom_seq[i] = seqs_after[i]
                self._skip_wait_once[i] = False
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
