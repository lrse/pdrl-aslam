"""This module contains the RL terminations configuration."""

from __future__ import annotations

import torch

from isaaclab.sensors import RayCaster
from isaaclab.managers import SceneEntityCfg


def compute_beam_distances(env, sensor_cfg: SceneEntityCfg):
    scene = env.unwrapped.scene
    sensor: RayCaster = scene.sensors[sensor_cfg.name]
    hits_w = sensor.data.ray_hits_w
    origins = sensor.data.pos_w.unsqueeze(1).expand_as(hits_w)
    dists = torch.linalg.norm(hits_w - origins, dim=2)
    max_range = sensor.cfg.max_distance
    dists = torch.where(torch.isfinite(dists), dists, max_range)
    return dists


def is_too_close(env, sensor_cfg: SceneEntityCfg, safe_distance: float = 0.2):
    """Returns True if any beam < safe_distance."""

    # 0.15 is the minimum distance for which the robot resets when close to the wall.
    # We want this offset for learning instead of having it on the lidar.
    turtlebot_and_maze1_offset = 0.15
    safe_distance = turtlebot_and_maze1_offset + safe_distance
    dists = compute_beam_distances(env, sensor_cfg)
    return dists.amin(dim=1) < safe_distance
