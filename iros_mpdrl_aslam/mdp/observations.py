"""This module contains the RL observation space configuration."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, Imu


# 2D Lidar auxiliar function.
def horizontal_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0, max_range: float = None) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    if max_range is None:
        max_range = float(sensor.cfg.max_distance)

    # XY positions.
    sensor_xy = sensor.data.pos_w[:, :2]
    hit_xy = sensor.data.ray_hits_w[..., :2]

    # Distances in plane.
    d_xy = torch.linalg.norm(sensor_xy.unsqueeze(1) - hit_xy, dim=-1)

    # Treat no-hits (NaN/Inf) as max_range.
    d_xy = torch.nan_to_num(d_xy, nan=max_range, posinf=max_range, neginf=max_range)

    # Clamp to keep strictly non-negative in case.
    ranges = torch.clamp(d_xy - offset, min=0.0, max=max_range)

    return ranges


# Imu (yaw).
def imu_yaw_rate(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu")):
    imu: Imu = env.scene.sensors[sensor_cfg.name]
    return imu.data.ang_vel_b[:, 2:3].float()   # (N, 1)


# SLAM-based terms.
def slam_U(env: ManagerBasedEnv) -> torch.Tensor:
    """Returns SLAM uncertainty for the observation.

    Normalized uncertainty in [0, 1].

    Falls back to legacy:
      - env.unwrapped._slam_U

    Returns shape (num_envs, 1) on env.device.
    """

    v = getattr(env.unwrapped, "_slam_U_norm", None)
    got_norm = v is not None

    if v is None:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

    # Convert to torch if required.
    if not torch.is_tensor(v):
        v = torch.as_tensor(v, device=env.device)
    else:
        v = v.to(device=env.device)

    v = v.to(dtype=torch.float32)

    # Flatten.
    v = v.view(-1)

    # Broadcast scalar -> per-env.
    if v.numel() == 1:
        v = v.repeat(env.num_envs)

    # If length mismatch, best-effort broadcast or truncate.
    if v.shape[0] != env.num_envs:
        if v.shape[0] == 1:
            v = v.repeat(env.num_envs)
        else:
            v = v[: env.num_envs]

    # Clamp ONLY if we know it is normalized, or if it already looks like [0,1].
    if got_norm or (v.min() >= -1e-3 and v.max() <= 1.0 + 1e-3):
        v = torch.clamp(v, 0.0, 1.0)

    return v.view(-1, 1)