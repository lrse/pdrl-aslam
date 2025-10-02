"""This module contains the RL rewards configuration."""

from __future__ import annotations

import torch
import numpy as np
import math

from isaaclab.envs import ManagerBasedEnv

from .slam_helpers import slam_alpha, batch_odom_seq_and_xy, disk_offsets, slam_xyb_core


# One-shot penalty
def collision_impulse_on_done(env, term_name: str = "obstacle_too_close"):
    hit = env.termination_manager.get_term(term_name)
    # RewardManager will multiply by dt, so we perform 1/dt to make the weight comparable to Placed-Castellanos.
    return hit.float() / env.step_dt


# Inspired by Placed-Castellanos reward function.
def rf(
    env: ManagerBasedEnv,
    wheel_radius: float = 0.033,
    track_width: float = 0.160,
    eps_w: float = 0.075,      # Threshold for going “straight”.
    k_turn: float = 0.5,
    term_name: str = "obstacle_too_close",
) -> torch.Tensor:
    hit = env.termination_manager.get_term(term_name).to(torch.float32)
    w = getattr(env.unwrapped, "_last_wheel_cmd", None)
    if w is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    wL, wR = w[:, 0], w[:, 1]
    # Cinematic.
    vx = 0.5 * wheel_radius * (wL + wR)     # m/s.
    wz = (wR - wL) * wheel_radius / track_width     # rad/s.

    vmax = 0.3
    forward_bonus = torch.clamp(vx, min=0.0, max=vmax) / vmax

    # Penalty for turning more than eps_w.
    wmax = 0.3
    excess = torch.clamp(torch.abs(wz) - eps_w, min=0.0)
    turn_pen = (excess / max(wmax - eps_w, 1e-6)).clamp(0.0, 1.0)

    base = forward_bonus - k_turn * turn_pen

    norm = 500.0 / 2000     # To compare with Placed-Castellanos 500 episode timesteps vs ours 2000.

    return (1.0 - hit) * base / env.step_dt * norm


def pose_coverage2d_delta(
    env: ManagerBasedEnv,
    bounds_xy: tuple[float, float, float, float],
    resolution: float = 0.10,
    footprint_radius_m: float = 0.1,
) -> torch.Tensor:

    N = env.num_envs
    device = env.device
    out = torch.zeros(N, device=device, dtype=torch.float32)

    if not hasattr(env.unwrapped, "_pose_cov_state"):
        env.unwrapped._pose_cov_state = {}
    S = env.unwrapped._pose_cov_state

    # init state
    if "cov2d" not in S:
        min_x, max_x, min_y, max_y = bounds_xy
        W = int(math.ceil((max_x - min_x) / resolution))
        H = int(math.ceil((max_y - min_y) / resolution))
        r_cells = max(1, int(math.ceil(footprint_radius_m / resolution)))
        S["cov2d"] = {
            "min_x": float(min_x), "max_x": float(max_x),
            "min_y": float(min_y), "max_y": float(max_y),
            "res": float(resolution), "W": W, "H": H, "r_cells": r_cells,
            "disk": disk_offsets(r_cells),
            "visited": [np.zeros((H, W), dtype=np.bool_) for _ in range(N)],
            "first_seen_seq": [np.full((H, W), -1, dtype=np.int32) for _ in range(N)],
            "newly_known_mask": [None for _ in range(N)],
            "prev_cov": np.zeros(N, dtype=np.float64),
            "total": float(W * H),
            "paint_radius_m": float(footprint_radius_m),
        }

    C = S["cov2d"]
    res = C["res"]
    min_x = C["min_x"]
    min_y = C["min_y"]
    W = C["W"]
    H = C["H"]
    disk = C["disk"]
    total = C["total"]

    snaps = batch_odom_seq_and_xy(env)

    _, _, B = slam_xyb_core(env)
    exploring_gate = (B.squeeze(-1) <= 0.5).float()

    for i in range(N):
        seq, xy = snaps[i]
        if xy is None:
            C["newly_known_mask"][i] = None
            continue

        x, y = float(xy[0]), float(xy[1])
        ix = math.floor((x - min_x) / res)
        iy = math.floor((y - min_y) / res)

        if not (0 <= ix < W and 0 <= iy < H):
            C["newly_known_mask"][i] = None
            continue

        vis = C["visited"][i]
        fst = C["first_seen_seq"][i]

        # Build mask for this step.
        mask = np.zeros((H, W), dtype=np.bool_)
        for dx, dy in disk:
            cx, cy = ix + dx, iy + dy
            if 0 <= cx < W and 0 <= cy < H:
                mask[cy, cx] = True

        # Cells that become known this step.
        newly = (~vis) & mask
        if np.any(newly):
            fst[newly] = int(seq) if seq is not None else -1

        # Update visited and coverage.
        vis |= mask
        cov = float(vis.sum()) / total

        delta_cov = float(newly.sum()) / total
        out[i] = exploring_gate[i] * (delta_cov / max(1e-6, float(getattr(env, "step_dt", 1.0))))

        C["newly_known_mask"][i] = newly
        C["prev_cov"][i] = cov

    out *= float(slam_alpha(env))

    return out
