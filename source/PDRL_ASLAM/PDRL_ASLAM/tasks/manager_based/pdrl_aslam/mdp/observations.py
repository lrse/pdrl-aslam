"""This module contains the RL observation space configuration."""

from __future__ import annotations

import torch
import numpy as np
import math

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from .slam_helpers import slam_alpha, slam_xyb_core, scale_m11, disk_offsets, batch_odom_seq_and_xy

from ..config import DEBUG


BOUNDS_XY = (-10.0, 10.0, -10.0, 10.0)
RESOLUTION = 0.10
FOOTPRINT_R = 0.10


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
    ranges = torch.round(ranges * 100.0) / 100.0    # As in Placed/Castellanos
    return ranges


# SLAM-based terms.
# Normalization using map bounds.
BOUNDS_X = (-10, 10)
BOUNDS_Y = (-10, 10)


def slam_x_term(env: ManagerBasedEnv) -> torch.Tensor:
    a = slam_alpha(env)
    if a <= 0.0:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    X, _, _ = slam_xyb_core(env)
    # DEBUG mode to check if the bridge is working correctly.
    Xn = X if DEBUG == "yes" else scale_m11(X, BOUNDS_X[0], BOUNDS_X[1])
    return a * Xn


def slam_y_term(env: ManagerBasedEnv) -> torch.Tensor:
    a = slam_alpha(env)
    if a <= 0.0:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    _, Y, _ = slam_xyb_core(env)
    # DEBUG mode to check if the bridge is working correctly.
    Yn = Y if DEBUG == "yes" else scale_m11(Y, BOUNDS_Y[0], BOUNDS_Y[1])
    return a * Yn


def _ensure_cov2d_state(env: ManagerBasedEnv):
    """Ensure coverage state exists but does not modify "visited" grid."""

    if not hasattr(env.unwrapped, "_pose_cov_state"):
        env.unwrapped._pose_cov_state = {}
    S = env.unwrapped._pose_cov_state
    if "cov2d" not in S:
        min_x, max_x, min_y, max_y = BOUNDS_XY
        W = int(math.ceil((max_x - min_x) / RESOLUTION))
        H = int(math.ceil((max_y - min_y) / RESOLUTION))
        r_cells = max(1, int(math.ceil(FOOTPRINT_R / RESOLUTION)))
        S["cov2d"] = {
            "min_x": float(min_x), "max_x": float(max_x),
            "min_y": float(min_y), "max_y": float(max_y),
            "res": float(RESOLUTION), "W": W, "H": H, "r_cells": r_cells,
            "disk": disk_offsets(r_cells),
            "visited": [np.zeros((H, W), dtype=np.bool_) for _ in range(env.num_envs)],
            "first_seen_seq": [np.full((H, W), -1, dtype=np.int32) for _ in range(env.num_envs)],
            "total": float(W * H),
            "prev_cov": np.zeros(env.num_envs, dtype=np.float64),
            "newly_known_mask": [None for _ in range(env.num_envs)],
        }
    return S["cov2d"]


def slam_b_term(env: ManagerBasedEnv) -> torch.Tensor:
    """0 when the current step would discover a new cell.
    Just observational."""

    a = slam_alpha(env)
    N = env.num_envs
    out = torch.ones((N, 1), device=env.device, dtype=torch.float32)
    if a <= 0.0:
        return a * out

    C = _ensure_cov2d_state(env)
    res, min_x, min_y = C["res"], C["min_x"], C["min_y"]
    W, H, disk = C["W"], C["H"], C["disk"]
    snaps = batch_odom_seq_and_xy(env)

    for i in range(N):
        seq, xy = snaps[i]

        if xy is None:
            out[i, 0] = 0.0
            continue

        x, y = float(xy[0]), float(xy[1])
        ix = math.floor((x - min_x) / res)
        iy = math.floor((y - min_y) / res)
        if not (0 <= ix < W and 0 <= iy < H):
            out[i, 0] = 0.0
            continue

        mask = np.zeros((H, W), dtype=np.bool_)
        for dx, dy in disk:
            cx, cy = ix + dx, iy + dy
            if 0 <= cx < W and 0 <= cy < H:
                mask[cy, cx] = True

        fst_list = C.get("first_seen_seq", None)
        if fst_list is not None:
            fst = fst_list[i]

            if np.any((fst < 0) & mask) or (
                seq is not None and np.any((fst == int(seq)) & mask)
            ):
                out[i, 0] = 0.0
                continue

        vis = C.get("visited", None)
        if vis is not None:
            newly = (~vis[i]) & mask
            if np.any(newly):
                out[i, 0] = 0.0
                continue

    return a * out
