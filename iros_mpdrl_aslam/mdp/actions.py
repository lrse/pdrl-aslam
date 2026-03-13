"""This module contains the RL action space configuration."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg, ActionTerm


@configclass
class DiffDriveContCfg(ActionTermCfg):
    class_type: type["DiffDriveCont"] = None
    asset_name: str = "turtlebot3_burger"
    left_joint: str = "wheel_left_joint"
    right_joint: str = "wheel_right_joint"
    wheel_radius: float = 0.033
    axle_length: float = 0.160

    v_bounds: tuple[float, float] = (0, 0.4)     # m/s.
    w_bounds: tuple[float, float] = (-1, 1)     # rad/s.

    squash: bool = False


class DiffDriveCont(ActionTerm):
    cfg: DiffDriveContCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        l_ids, _ = self._asset.find_joints(cfg.left_joint)
        r_ids, _ = self._asset.find_joints(cfg.right_joint)
        self._joint_ids = [l_ids[0], r_ids[0]]

        self._raw = torch.zeros((self.num_envs, 2), device=self.device)
        self._processed = torch.zeros((self.num_envs, 2), device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        """
        Map policy outputs in [-1,1]^2 to wheel angular velocities.
        """

        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if actions.shape[-1] == 1:
            actions = torch.cat([actions, torch.zeros_like(actions)], dim=-1)

        a = actions.to(device=self.device, dtype=torch.float32)
        self._raw = a

        if self.cfg.squash:
            a = torch.tanh(a)
        a = torch.clamp(a, -1.0, 1.0)

        av = a[:, 0]
        aw = a[:, 1]

        vmin, vmax = self.cfg.v_bounds
        wmin, wmax = self.cfg.w_bounds

        vx = (av + 1.0) * 0.5 * (vmax - vmin) + vmin
        wz = (aw + 1.0) * 0.5 * (wmax - wmin) + wmin

        r, L = self.cfg.wheel_radius, self.cfg.axle_length
        wL = (vx - 0.5 * L * wz) / r
        wR = (vx + 0.5 * L * wz) / r
        self._processed = torch.stack([wL, wR], dim=-1)

    def apply_actions(self):
        self._processed = torch.clamp(self._processed, -16.0, 16.0)
        self._asset.set_joint_velocity_target(self._processed, joint_ids=self._joint_ids)

        self._env.unwrapped._last_wheel_cmd = self._processed

        r, L = self.cfg.wheel_radius, self.cfg.axle_length
        wL = self._processed[:, 0]
        wR = self._processed[:, 1]
        vx = 0.5 * r * (wL + wR)
        wz = (r / L) * (wR - wL)
        self._env.unwrapped._last_cmd_vw = torch.stack([vx, wz], dim=-1)


DiffDriveContCfg.class_type = DiffDriveCont
