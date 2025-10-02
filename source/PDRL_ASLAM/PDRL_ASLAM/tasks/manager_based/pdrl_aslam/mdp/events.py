"""This module contains the RL events configuration, which
in our case are mostly resets."""

from __future__ import annotations

import numpy as np

from isaaclab.envs import ManagerBasedEnv

from ..config import PHASE, ENVIRONMENT


def _coverage_counts(env: ManagerBasedEnv, i: int):
    S = getattr(env.unwrapped, "_pose_cov_state", None)
    if not S or "cov2d" not in S:
        return None
    C = S["cov2d"]
    vis = C["visited"][i]
    covered = int(vis.sum())
    total = int(C["total"])
    frac = float(covered / total) if total > 0 else 0.0
    return covered, total, frac


def _ensure_last_episode_cov_store(env: ManagerBasedEnv):
    if not hasattr(env.unwrapped, "_last_episode_cov"):
        N = env.num_envs
        env.unwrapped._last_episode_cov = {
            "covered": np.full(N, -1, dtype=np.int32),
            "total": np.full(N, -1, dtype=np.int32),
            "frac": np.full(N, np.nan, dtype=np.float32),
        }
    return env.unwrapped._last_episode_cov


def _reset_pose_coverage_state(env: ManagerBasedEnv, env_ids=None):
    """Clear visited masks for the given environments."""
    S = getattr(env.unwrapped, "_pose_cov_state", None)
    if S is None:
        return
    C = S.get("cov2d", None)
    if C is None:
        return

    if env_ids is None:
        ids = range(env.num_envs)
    else:
        try:
            ids = env_ids.tolist()
        except AttributeError:
            ids = list(env_ids)

    for i in ids:
        C["visited"][i][:] = False      # Grid to 0.
        if "first_seen_seq" in C:
            C["first_seen_seq"][i][:] = -1
        if "prev_cov" in C:
            C["prev_cov"][i] = 0.0
        if "newly_known_mask" in C:
            C["newly_known_mask"][i] = None


def reset_visit_event(env: ManagerBasedEnv, env_ids=None):
    """Reset visited masks and reset SLAM. Before reseting,
    snapshot episode-end coverage for logging."""

    S = getattr(env.unwrapped, "_pose_cov_state", None)
    if S and "cov2d" in S:
        store = _ensure_last_episode_cov_store(env)
        if env_ids is None:
            ids = range(env.num_envs)
        else:
            try:
                ids = env_ids.tolist()
            except AttributeError:
                ids = list(env_ids)
        for i in ids:
            res = _coverage_counts(env, i)
            if res is not None:
                covered, total, frac = res
                store["covered"][i] = covered
                store["total"][i] = total
                store["frac"][i] = frac

    _reset_pose_coverage_state(env, env_ids)

    dm = getattr(env.unwrapped, "data_manager", None)
    if dm is not None and hasattr(dm, "reset_slam"):
        dm.reset_slam(env_ids)


# Starting position training vs playing. Mostly for testing.
POSE_RANGE = {"x": (0, 0), "y": (3, 3), "z": (-0.5, -0.5), "yaw": (-3.14, 3.14)}

if PHASE == "play":
    if ENVIRONMENT == "env3":
        POSE_RANGE = {"x": (3, 3), "y": (0, 0), "z": (-0.5, -0.5), "yaw": (-1, -1)}
    elif ENVIRONMENT == "env2":
        POSE_RANGE = {"x": (0, 0), "y": (-3, -3), "z": (-0.5, -0.5), "yaw": (-3.14, -3.14)}
