from __future__ import annotations


# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint or run a fixed-horizon evaluation for an RSL-RL agent with slam_toolbox covariance injected."""

import argparse
import csv
import importlib.metadata as metadata
import inspect
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from packaging import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _maybe_set_term_param(term_cfg, key: str, value) -> bool:
    """Set a reward-term param only if the callable signature accepts it."""
    try:
        func = term_cfg.func
        if callable(func) and not inspect.isfunction(func):
            sig = inspect.signature(func.__call__)
        else:
            sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = None

    if allowed is None or key in allowed:
        term_cfg.params[key] = value
        return True
    return False


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: str, headers: list[str], row: dict[str, object]) -> None:
    _ensure_parent_dir(path)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _to_float_list(value: object, expected_len: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        vals = value.detach().reshape(-1).to("cpu").tolist()
        vals = [float(v) for v in vals]
    elif isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
    else:
        vals = [float(value)]

    if len(vals) == expected_len:
        return vals

    if expected_len == 1 and len(vals) >= 1:
        return [float(vals[0])]

    raise RuntimeError(
        f"Expected metric vector of length {expected_len}, but got length {len(vals)}. "
        "Please make sure the reward term publishes per-environment tensors."
    )


def _read_eval_metric_vector(env, attr_name: str, expected_len: int) -> list[float]:
    if not hasattr(env.unwrapped, attr_name):
        raise RuntimeError(
            f"Missing evaluation metric '{attr_name}' on env.unwrapped. "
            "Please update SLAMTOOLBOXActiveReward with the evaluation-tracking version."
        )
    return _to_float_list(getattr(env.unwrapped, attr_name), expected_len)


def _rebuild_summary_csv(runs_csv_path: str, summary_csv_path: str) -> None:
    grouped: dict[tuple[str, str], dict[str, float]] = {}

    if not os.path.exists(runs_csv_path):
        _ensure_parent_dir(summary_csv_path)
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADERS)
            writer.writeheader()
        return

    with open(runs_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            finished = str(row.get("Finished full horizon", "1")).strip().lower()
            if finished not in ("1", "true", "yes", "y"):
                continue

            key = (row["Environment"], row["Agent"])
            if key not in grouped:
                grouped[key] = {
                    "count": 0.0,
                    "Visited cells (unique)": 0.0,
                    "Visited cells (total)": 0.0,
                    "Collision rate (%)": 0.0,
                    "Average linear speed (m/s)": 0.0,
                }

            grouped[key]["count"] += 1.0
            grouped[key]["Visited cells (unique)"] += float(row["Visited cells (unique)"])
            grouped[key]["Visited cells (total)"] += float(row["Visited cells (total)"])
            grouped[key]["Collision rate (%)"] += float(row["Collision rate (%)"])
            grouped[key]["Average linear speed (m/s)"] += float(row["Average linear speed (m/s)"])

    _ensure_parent_dir(summary_csv_path)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADERS)
        writer.writeheader()
        for (environment_name, agent_name), agg in sorted(grouped.items()):
            count = max(agg["count"], 1.0)
            writer.writerow(
                {
                    "Environment": environment_name,
                    "Agent": agent_name,
                    "Visited cells (unique)": agg["Visited cells (unique)"] / count,
                    "Visited cells (total)": agg["Visited cells (total)"] / count,
                    "Collision rate (%)": agg["Collision rate (%)"] / count,
                    "Average linear speed (m/s)": agg["Average linear speed (m/s)"] / count,
                }
            )


def _reset_policy_hidden_state(policy_obj, policy_nn, device: torch.device, num_envs: int) -> None:
    done_reset = torch.ones(num_envs, dtype=torch.bool, device=device)

    try:
        policy_obj.reset(done_reset)
        return
    except Exception:
        pass

    try:
        policy_nn.reset(done_reset)
    except Exception:
        pass


def _get_data_manager_from_env(env):
    dm = getattr(env.unwrapped, "data_manager", None)
    if dm is None:
        raise RuntimeError(
            "Ros2TrainingWrapper did not attach env.unwrapped.data_manager. "
            "Make sure the ROS2 wrapper is applied before RslRlVecEnvWrapper."
        )
    return dm


def _prime_slam_toolbox(env, bootstrap_timeout_s: float, require_fresh: bool) -> None:
    """Warm up the ROS graph so the first policy step does not block for a long time.

    We force one scan/odom publication per environment and wait until every env has
    produced at least one newer slam_toolbox /pose message with covariance.
    """
    dm = _get_data_manager_from_env(env)
    num_envs = int(env.unwrapped.num_envs)
    env_ids = list(range(num_envs))

    print("[INFO] Waiting for slam_toolbox graph connectivity...", flush=True)
    ready, status = dm.wait_for_toolbox_graph(env_ids, timeout_s=float(bootstrap_timeout_s))
    if not ready:
        raise RuntimeError(
            "slam_toolbox graph is not ready for play. "
            f"Status: {status}. "
            "Check that the namespaces match the environment count and that each slam_toolbox node is active."
        )

    baseline = [int(dm.get_odom_seq_xy_yaw_cov(i)[0]) for i in env_ids]
    pending = set(env_ids)
    deadline = time.time() + max(float(bootstrap_timeout_s), 1e-3)

    print("[INFO] Priming slam_toolbox with an initial forced scan/odom publication...", flush=True)
    while pending and time.time() < deadline:
        force_ids = sorted(pending)
        dm.pub_ros2_data(force_scan_envs=force_ids)
        remaining = max(0.0, deadline - time.time())
        dm.wait_for_new_odom(force_ids, baseline, min(0.10, remaining))

        next_pending = set()
        for i in force_ids:
            seq_i, _xy_i, _yaw_i, cov_i = dm.get_odom_seq_xy_yaw_cov(i)
            if int(seq_i) <= baseline[i] or cov_i is None:
                next_pending.add(i)
        pending = next_pending

    if pending:
        msg = (
            "slam_toolbox did not publish an initial fresh pose/covariance before play started. "
            f"Missing env ids: {sorted(pending)}."
        )
        if require_fresh:
            raise RuntimeError(msg)
        print(f"[WARNING] {msg} Falling back to non-strict startup.", flush=True)
    else:
        print("[INFO] slam_toolbox warm-up complete.", flush=True)


def _do_initial_reset_and_prime(env, policy, policy_nn, use_bridge: bool, bootstrap_timeout_s: float) -> object:
    """Reset once more after all wrappers exist, then prime slam_toolbox before the first visible step."""
    env.reset()
    if use_bridge:
        _prime_slam_toolbox(
            env,
            bootstrap_timeout_s=float(bootstrap_timeout_s),
            require_fresh=True,
        )
    obs = env.get_observations()
    _reset_policy_hidden_state(policy, policy_nn, env.unwrapped.device, int(env.unwrapped.num_envs))
    return obs


# add argparse arguments
parser = argparse.ArgumentParser(description="Play or evaluate an RL agent with RSL-RL + slam_toolbox covariance.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training/play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# --- SLAM / ROS2 bridge args ---
parser.add_argument(
    "--obs_scalar_index",
    type=int,
    default=-1,
    help=(
        "Index in the flattened policy observation vector where the SLAM uncertainty scalar is written. "
        "Use -1 to overwrite the last element (recommended)."
    ),
)
parser.add_argument(
    "--use_slam_toolbox_bridge",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable slam_toolbox covariance/pose synchronization and inject uncertainty into observations.",
)
parser.add_argument(
    "--slam_blend_steps",
    type=int,
    default=1,
    help=(
        "Total environment steps used to blend fixed-lag -> slam_toolbox uncertainty. "
        "Use 1 for effectively immediate full slam_toolbox after the first primed step."
    ),
)
parser.add_argument(
    "--slam_bridge_mode",
    type=str,
    default="convex",
    choices=["convex", "inject_only"],
    help="Uncertainty bridge blend mode (must match what slam_active supports).",
)
parser.add_argument(
    "--slam_bridge_bootstrap_timeout",
    type=float,
    default=20.0,
    help="Seconds to wait after reset for the initial slam_toolbox graph + first fresh pose/covariance.",
)
parser.add_argument(
    "--slam_bridge_step_wait_timeout",
    type=float,
    default=2.0,
    help="Per-step maximum wait for a fresh slam_toolbox pose/covariance during play.",
)
parser.add_argument(
    "--slam_bridge_reset_quarantine_updates",
    type=int,
    default=0,
    help="Ignore this many slam_toolbox updates right after reset. Forced to 0 in strict pure-toolbox play.",
)
parser.add_argument(
    "--slam_bridge_require_fresh_when_full",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require a fresh slam_toolbox covariance every step once the blend reaches full toolbox weight.",
)
parser.add_argument(
    "--slam_bridge_require_fresh_always",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require a fresh slam_toolbox covariance on every env step from the start of play.",
)
parser.add_argument(
    "--slam_attempt_hard_reset_on_env_reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If True, call slam_toolbox reset/clear services on env reset (more robust, slightly slower).",
)
parser.add_argument(
    "--slam_clear_queue_on_env_reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If True, clear internal ROS2 message queues on env reset to avoid stale covariances.",
)
parser.add_argument(
    "--slam_publish_trajectory",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="If True, publish trajectory/TF topics for visualization.",
)

# evaluation arguments
parser.add_argument(
    "--evaluate",
    action="store_true",
    default=False,
    help="Run fixed-horizon evaluation and write CSV files instead of free-running play mode.",
)
parser.add_argument(
    "--eval_duration_s",
    type=float,
    default=300.0,
    help="Evaluation duration in simulated seconds. Default: 300 seconds (5 minutes).",
)
parser.add_argument(
    "--eval_runs",
    type=int,
    default=1,
    help="Number of evaluation batches to execute.",
)
parser.add_argument(
    "--environment_label",
    type=str,
    default="unknown_environment",
    help="Value written to the 'Environment' CSV column.",
)
parser.add_argument(
    "--agent_label",
    type=str,
    default="policy",
    help="Value written to the 'Agent' CSV column.",
)
parser.add_argument(
    "--results_csv",
    type=str,
    default=None,
    help="Path to the per-run CSV. If omitted, it is written inside the checkpoint folder.",
)
parser.add_argument(
    "--summary_csv",
    type=str,
    default=None,
    help="Path to the grouped summary CSV. If omitted, it is written beside the per-run CSV.",
)
parser.add_argument(
    "--collision_distance",
    type=float,
    default=0.02,
    help="Collision threshold in meters used only for evaluation metrics. A step is counted as collision when min_lidar < this value.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video and not args_cli.evaluate:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version):
        # Older Isaac Lab builds do not expose this compatibility helper.
        # For rsl-rl 2.x, returning the config unchanged is the right fallback.
        return agent_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from ros2_training_wrapper import Ros2TrainingWrapper


RUN_CSV_HEADERS = [
    "Run",
    "Parallel env id",
    "Environment",
    "Agent",
    "Visited cells (unique)",
    "Visited cells (total)",
    "Collision rate (%)",
    "Average linear speed (m/s)",
    "Checkpoint",
    "Eval duration (seconds)",
    "Seed",
    "Finished full horizon",
    "Timestamp",
]

SUMMARY_CSV_HEADERS = [
    "Environment",
    "Agent",
    "Visited cells (unique)",
    "Visited cells (total)",
    "Collision rate (%)",
    "Average linear speed (m/s)",
]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with or evaluate an RSL-RL agent, synchronously consuming slam_toolbox covariance."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.evaluate:
        env_cfg.episode_length_s = float(args_cli.eval_duration_s) + 5.0

        if hasattr(env_cfg, "terminations") and getattr(env_cfg.terminations, "obstacle_too_close", None) is not None:
            env_cfg.terminations.obstacle_too_close.params["safe_distance"] = -1.0

        if hasattr(env_cfg, "rewards") and getattr(env_cfg.rewards, "slam_active", None) is not None:
            _maybe_set_term_param(
                env_cfg.rewards.slam_active,
                "eval_collision_distance",
                float(args_cli.collision_distance),
            )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    use_bridge = bool(args_cli.use_slam_toolbox_bridge)
    effective_blend_steps = max(int(args_cli.slam_blend_steps), 1)
    effective_quarantine = int(args_cli.slam_bridge_reset_quarantine_updates)
    if bool(args_cli.slam_bridge_require_fresh_always) and effective_quarantine > 0:
        print(
            "[INFO] slam_bridge_require_fresh_always=True is incompatible with reset quarantine > 0. "
            f"Forcing slam_bridge_reset_quarantine_updates from {effective_quarantine} to 0.",
            flush=True,
        )
        effective_quarantine = 0

    if hasattr(env_cfg, "rewards") and getattr(env_cfg.rewards, "slam_active", None) is not None:
        slam_term = env_cfg.rewards.slam_active
        _maybe_set_term_param(slam_term, "use_slam_toolbox_bridge", use_bridge)
        _maybe_set_term_param(slam_term, "bridge_blend_steps", effective_blend_steps)
        _maybe_set_term_param(slam_term, "bridge_mode", str(args_cli.slam_bridge_mode))
        _maybe_set_term_param(slam_term, "bridge_bootstrap_timeout_s", float(args_cli.slam_bridge_bootstrap_timeout))
        _maybe_set_term_param(slam_term, "bridge_step_wait_timeout_s", float(args_cli.slam_bridge_step_wait_timeout))
        _maybe_set_term_param(slam_term, "bridge_reset_quarantine_updates", effective_quarantine)
        _maybe_set_term_param(
            slam_term, "bridge_require_fresh_when_full", bool(args_cli.slam_bridge_require_fresh_when_full)
        )
        _maybe_set_term_param(
            slam_term, "bridge_require_fresh_always", bool(args_cli.slam_bridge_require_fresh_always)
        )

        print(
            "[INFO] slam_active configured for play: "
            f"use_slam_toolbox_bridge={use_bridge} | "
            f"blend_steps={effective_blend_steps} | "
            f"mode={str(args_cli.slam_bridge_mode)} | "
            f"bootstrap_timeout={float(args_cli.slam_bridge_bootstrap_timeout):.3f}s | "
            f"step_wait_timeout={float(args_cli.slam_bridge_step_wait_timeout):.3f}s | "
            f"require_fresh_when_full={bool(args_cli.slam_bridge_require_fresh_when_full)} | "
            f"require_fresh_always={bool(args_cli.slam_bridge_require_fresh_always)} | "
            f"reset_quarantine_updates={effective_quarantine}",
            flush=True,
        )
    else:
        if use_bridge:
            raise RuntimeError(
                "env_cfg.rewards.slam_active was not found. "
                "This play script needs that reward term because it owns the slam_toolbox synchronization "
                "and writes the SLAM uncertainty scalar into env.unwrapped._slam_U_norm."
            )
        print(
            "[WARNING] env_cfg.rewards.slam_active was not found. "
            "No SLAM uncertainty will be injected into observations.",
            flush=True,
        )

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video and not args_cli.evaluate else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = Ros2TrainingWrapper(
        env,
        obs_scalar_index=int(args_cli.obs_scalar_index),
        attempt_hard_reset_on_env_reset=bool(args_cli.slam_attempt_hard_reset_on_env_reset),
        clear_queue_on_env_reset=bool(args_cli.slam_clear_queue_on_env_reset),
        publish_trajectory=bool(args_cli.slam_publish_trajectory),
    )

    if args_cli.video and not args_cli.evaluate:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    elif args_cli.video and args_cli.evaluate:
        print("[INFO] --video is ignored in --evaluate mode.")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
        policy_nn = getattr(runner.alg, "policy", None)
    else:
        policy_nn = getattr(runner.alg, "policy", None)
        if policy_nn is None:
            policy_nn = runner.alg.actor_critic

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    if args_cli.evaluate:
        results_csv = args_cli.results_csv or os.path.join(log_dir, "evaluation", "evaluation_runs.csv")
        summary_csv = args_cli.summary_csv or os.path.join(log_dir, "evaluation", "evaluation_summary.csv")
        eval_steps = max(1, int(round(float(args_cli.eval_duration_s) / float(dt))))
        eval_num_envs = int(env.unwrapped.num_envs)

        print(
            f"[INFO] Evaluation mode enabled: {args_cli.eval_runs} batch(es), "
            f"{eval_num_envs} env(s)/batch, {eval_steps} steps/env, dt={dt:.4f}s"
        )
        print(f"[INFO] Total rollouts to collect: {int(args_cli.eval_runs) * eval_num_envs}")
        print(
            "[INFO] Visited-cells metrics: (1) unique cells, (2) total cell-entry visits "
            "(includes revisits, counts when cell index changes)."
        )
        print(f"[INFO] Collision metric threshold: min_lidar < {float(args_cli.collision_distance):.4f} m")
        print("[INFO] Linear speed metric: mean(|v_cmd|) over horizon, where v_cmd = actions[:,0]")
        print(f"[INFO] Per-run CSV : {results_csv}")
        print(f"[INFO] Summary CSV : {summary_csv}")

        for run_idx in range(1, int(args_cli.eval_runs) + 1):
            obs = _do_initial_reset_and_prime(
                env,
                policy,
                policy_nn,
                use_bridge=use_bridge,
                bootstrap_timeout_s=float(args_cli.slam_bridge_bootstrap_timeout),
            )

            speed_integral = torch.zeros(eval_num_envs, device=env.unwrapped.device, dtype=torch.float32)
            time_integral = 0.0

            batch_finished_full_horizon = True
            completed_steps = 0

            for _ in range(eval_steps):
                start_time = time.time()

                with torch.inference_mode():
                    actions = policy(obs)

                    v_cmd = actions[:, 0].detach().to(torch.float32)
                    speed_integral += torch.abs(v_cmd) * float(dt)
                    time_integral += float(dt)

                    obs, _, dones, _ = env.step(actions)
                    dones_bool = dones.to(dtype=torch.bool)
                    try:
                        policy.reset(dones_bool)
                    except Exception:
                        try:
                            policy_nn.reset(dones_bool)
                        except Exception:
                            pass

                completed_steps += 1

                if torch.any(dones_bool):
                    done_ids = torch.nonzero(dones_bool, as_tuple=False).reshape(-1).tolist()
                    print(
                        "[WARNING] Unexpected termination/reset during evaluation in parallel env ids "
                        f"{done_ids}. Aborting this evaluation batch early."
                    )
                    batch_finished_full_horizon = False
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            if completed_steps < eval_steps:
                batch_finished_full_horizon = False

            visited_unique_vec = _read_eval_metric_vector(env, "_eval_visited_cells", eval_num_envs)
            visited_total_vec = _read_eval_metric_vector(env, "_eval_visited_cells_total", eval_num_envs)
            collision_rate_vec = _read_eval_metric_vector(env, "_eval_collision_rate_pct", eval_num_envs)

            denom_t = max(float(time_integral), 1e-9)
            avg_speed_vec = (speed_integral / denom_t).detach().to("cpu").tolist()
            avg_speed_vec = [float(x) for x in avg_speed_vec]

            for env_idx in range(eval_num_envs):
                run_row = {
                    "Run": run_idx,
                    "Parallel env id": env_idx,
                    "Environment": args_cli.environment_label,
                    "Agent": args_cli.agent_label,
                    "Visited cells (unique)": int(round(float(visited_unique_vec[env_idx]))),
                    "Visited cells (total)": int(round(float(visited_total_vec[env_idx]))),
                    "Collision rate (%)": collision_rate_vec[env_idx],
                    "Average linear speed (m/s)": avg_speed_vec[env_idx],
                    "Checkpoint": resume_path,
                    "Eval duration (seconds)": float(args_cli.eval_duration_s),
                    "Seed": agent_cfg.seed,
                    "Finished full horizon": int(bool(batch_finished_full_horizon)),
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                _append_csv_row(results_csv, RUN_CSV_HEADERS, run_row)

            mean_unique = sum(visited_unique_vec) / max(len(visited_unique_vec), 1)
            mean_total = sum(visited_total_vec) / max(len(visited_total_vec), 1)
            mean_collision = sum(collision_rate_vec) / max(len(collision_rate_vec), 1)
            mean_speed = sum(avg_speed_vec) / max(len(avg_speed_vec), 1)
            print(
                f"[EVAL][batch {run_idx}/{args_cli.eval_runs}] "
                f"mean_unique_cells={mean_unique:.3f} | mean_total_cell_entries={mean_total:.3f} | "
                f"mean_collision_rate={mean_collision:.3f}% | mean_avg_speed={mean_speed:.3f} m/s | "
                f"full_horizon={int(batch_finished_full_horizon)}"
            )

        _rebuild_summary_csv(results_csv, summary_csv)
        print(f"[INFO] Wrote evaluation CSV: {results_csv}")
        print(f"[INFO] Wrote summary CSV   : {summary_csv}")

    else:
        obs = _do_initial_reset_and_prime(
            env,
            policy,
            policy_nn,
            use_bridge=use_bridge,
            bootstrap_timeout_s=float(args_cli.slam_bridge_bootstrap_timeout),
        )

        timestep = 0

        # One hidden priming step makes bridge alpha reach 1 quickly when blend_steps == 1,
        # so the visible play loop uses slam_toolbox immediately after startup.
        if use_bridge and max(int(args_cli.slam_blend_steps), 1) <= 1:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                dones_bool = dones.to(dtype=torch.bool)
                try:
                    policy.reset(dones_bool)
                except Exception:
                    try:
                        policy_nn.reset(dones_bool)
                    except Exception:
                        pass

        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                dones_bool = dones.to(dtype=torch.bool)
                try:
                    policy.reset(dones_bool)
                except Exception:
                    try:
                        policy_nn.reset(dones_bool)
                    except Exception:
                        pass

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
