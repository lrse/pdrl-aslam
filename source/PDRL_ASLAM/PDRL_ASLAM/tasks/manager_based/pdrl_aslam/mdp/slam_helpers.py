"""This module contains SLAM helper functions."""

from __future__ import annotations

import torch
import math

from isaaclab.envs import ManagerBasedEnv


def slam_alpha(env: ManagerBasedEnv) -> float:
    cfg = env.unwrapped.cfg
    if not getattr(cfg, "use_slam_obs", False):
        return 0.0
    warmup = int(getattr(cfg, "slam_warmup_env_steps", 0))
    if warmup <= 0:
        return 1.0
    step = int(getattr(env, "common_step_counter", 0))
    return float(min(1.0, step / float(warmup)))


def get_slam_xy(env: ManagerBasedEnv, env_idx: int):
    dm = getattr(env.unwrapped, "data_manager", None)
    if dm is None or not hasattr(dm, "get_slam_xy_at_last_odom"):
        return None
    return dm.get_slam_xy_at_last_odom(env_idx)


def batch_odom_seq_and_xy(env: ManagerBasedEnv):
    dm = getattr(env.unwrapped, "data_manager", None)
    if dm is None or not hasattr(dm, "get_odom_seq_and_xy"):
        return [(0, None)] * env.num_envs
    return [dm.get_odom_seq_and_xy(i) for i in range(env.num_envs)]


def slam_xyb_core(env: ManagerBasedEnv):
    N = env.num_envs
    device = env.device
    X = torch.zeros((N, 1), device=device, dtype=torch.float32)
    Y = torch.zeros((N, 1), device=device, dtype=torch.float32)
    B = torch.zeros((N, 1), device=device, dtype=torch.float32)

    snaps = batch_odom_seq_and_xy(env)

    S = getattr(env.unwrapped, "_pose_cov_state", None)
    C = (S.get("cov2d") if (S and "cov2d" in S) else None)

    for i in range(N):
        seq, xy = snaps[i]
        if xy is None:
            continue

        x, y = float(xy[0]), float(xy[1])
        X[i, 0], Y[i, 0] = x, y

        if C is None:
            continue

        res = C["res"]
        min_x = C["min_x"]
        min_y = C["min_y"]
        W = C["W"]
        H = C["H"]

        ix = math.floor((x - min_x) / res)
        iy = math.floor((y - min_y) / res)
        if not (0 <= ix < W and 0 <= iy < H):
            continue

        fst_list = C.get("first_seen_seq", None)
        if fst_list is not None:
            t_first = fst_list[i][iy, ix]

            if (seq is not None) and (t_first >= 0) and (t_first < int(seq)):
                B[i, 0] = 1.0
            else:
                B[i, 0] = 0.0
        else:
            visited = C.get("visited", None)
            if visited is not None and visited[i][iy, ix]:
                B[i, 0] = 1.0
            else:
                B[i, 0] = 0.0

    return X, Y, B


def scale_m11(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    x = 2.0 * (x - lo) / max(hi - lo, 1e-6) - 1.0
    return torch.clamp(x, -1.0, 1.0)


def disk_offsets(radius_cells: int):
    offs = []
    r2 = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= r2:
                offs.append((dx, dy))
    return offs
