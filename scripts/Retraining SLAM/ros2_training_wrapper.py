from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import ok as ros_ok

from ros2_bridge import RobotDataManager


class Ros2TrainingWrapper(gym.Wrapper):
    """Thin ROS bridge wrapper.

    The reward term now owns the SLAM-toolbox synchronization and the blend schedule.
    This wrapper only:
      1) creates the ROS bridge node,
      2) keeps the executor spinning,
      3) mirrors env.unwrapped._slam_U_norm into the chosen observation scalar.
    """

    def __init__(
        self,
        env,
        *,
        obs_scalar_index: int = -1,
        publish_trajectory: bool = False,
        attempt_hard_reset_on_env_reset: bool = True,
        clear_queue_on_env_reset: bool = True,
    ):
        super().__init__(env)

        self._obs_scalar_index = int(obs_scalar_index)
        self._owns_rclpy = False
        if not ros_ok():
            rclpy.init(args=[])
            self._owns_rclpy = True

        self._dm = RobotDataManager(
            env=self.env.unwrapped,
            attempt_hard_reset_on_env_reset=bool(attempt_hard_reset_on_env_reset),
            clear_queue_on_env_reset=bool(clear_queue_on_env_reset),
            publish_trajectory=bool(publish_trajectory),
        )
        self.env.unwrapped.data_manager = self._dm

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._dm)
        self._spin_thread = __import__("threading").Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _policy_obs_ref(self, obs):
        return obs["policy"] if (isinstance(obs, dict) and "policy" in obs) else obs

    def _write_scalar_obs(self, obs, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        target = self._policy_obs_ref(obs)
        try:
            if hasattr(target, "detach"):
                device, dtype = target.device, target.dtype
                idx = self._obs_scalar_index if self._obs_scalar_index >= 0 else target.shape[1] + self._obs_scalar_index
                with torch.no_grad():
                    target[:, idx] = torch.as_tensor(values, dtype=dtype, device=device)
            else:
                arr = np.asarray(target)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                idx = self._obs_scalar_index if self._obs_scalar_index >= 0 else arr.shape[1] + self._obs_scalar_index
                arr[:, idx] = values
                if isinstance(obs, dict) and "policy" in obs:
                    obs["policy"] = arr
                else:
                    obs = arr
        except Exception:
            pass
        return obs

    def _patch_obs(self, obs):
        env_u = getattr(self.env, "unwrapped", self.env)
        u = getattr(env_u, "_slam_U_norm", None)
        if u is None:
            return obs
        if hasattr(u, "detach"):
            u = u.detach().cpu().numpy()
        else:
            u = np.asarray(u)
        return self._write_scalar_obs(obs, np.asarray(u, dtype=np.float32).reshape(-1))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._patch_obs(obs), info

    def get_observations(self):
        obs = self.env.get_observations()
        return self._patch_obs(obs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return self._patch_obs(obs), rew, terminated, truncated, info

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