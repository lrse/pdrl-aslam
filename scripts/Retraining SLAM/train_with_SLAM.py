# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL agent while using slam_toolbox uncertainty inside the reward and observation.

The SLAM reward term owns:
- per-step synchronization with slam_toolbox,
- the fixed-lag -> slam_toolbox blend schedule,
- the scalar uncertainty that is fed to both reward and observation.

This version adds a strict option so training can require a *fresh* slam_toolbox
covariance on every env step for the whole blend, not just after alpha reaches 1.0.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Train an RL agent with RSL-RL while progressively replacing fixedlag SLAM uncertainty with slam_toolbox uncertainty."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

parser.add_argument(
    "--slam_blend_steps",
    type=int,
    default=204800,
    help="Total environment transitions over which to replace the fixedlag uncertainty with slam_toolbox.",
)
parser.add_argument(
    "--slam_bridge_mode",
    type=str,
    default="convex",
    choices=["convex", "inject_only"],
    help="Blend mode used inside the reward term.",
)
parser.add_argument(
    "--slam_bridge_obs_index",
    type=int,
    default=-1,
    help="Index of the policy observation scalar carrying SLAM uncertainty.",
)
parser.add_argument(
    "--slam_bridge_bootstrap_timeout",
    type=float,
    default=60.0,
    help="Seconds to wait for the first fresh slam_toolbox pose/covariance after each reset.",
)
parser.add_argument(
    "--slam_bridge_step_wait_timeout",
    type=float,
    default=60.0,
    help="Per-step maximum wait for a fresh slam_toolbox pose after publishing the new scan.",
)
parser.add_argument(
    "--slam_bridge_yaw_length_m",
    type=float,
    default=1.0,
    help="Scale factor used when converting yaw covariance into the planar 3x3 covariance determinant.",
)
parser.add_argument(
    "--slam_reset_quarantine_updates",
    type=int,
    default=1,
    help="Ignore slam_toolbox covariance until this many fresh post-reset updates have arrived."
         " When strict fresh-per-step mode is enabled this is forced to 0.",
)
parser.add_argument(
    "--slam_bridge_require_fresh_always",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require a fresh slam_toolbox covariance on every env step during the whole blend, not only after alpha reaches 1.0.",
)
parser.add_argument(
    "--no-slam-hard-reset-on-env-reset",
    action="store_true",
    default=False,
    help="Disable calling /<ns>/slam_toolbox/reset when that optional service exists.",
)
parser.add_argument(
    "--no-slam-clear-queue-on-env-reset",
    action="store_true",
    default=False,
    help="Disable calling /<ns>/slam_toolbox/clear_queue on env resets.",
)
parser.add_argument(
    "--slam_publish_trajectory",
    action="store_true",
    default=False,
    help="Publish /trajectory for RViz. Keep this OFF for training throughput.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os
import platform
import pickle
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

try:
    from isaaclab.utils.io import dump_pickle  # type: ignore
except ImportError:
    def dump_pickle(filename, data):
        filename = str(filename)
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from ros2_training_wrapper import Ros2TrainingWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # ------------------------------------------------------------------
    # Push the bridge/blend config into the reward term.
    # ------------------------------------------------------------------
    slam_term = env_cfg.rewards.slam_active
    slam_term.params["use_slam_toolbox_bridge"] = True
    slam_term.params["bridge_blend_steps"] = int(args_cli.slam_blend_steps)
    slam_term.params["bridge_mode"] = str(args_cli.slam_bridge_mode)
    slam_term.params["bridge_bootstrap_timeout_s"] = float(args_cli.slam_bridge_bootstrap_timeout)
    slam_term.params["bridge_step_wait_timeout_s"] = float(args_cli.slam_bridge_step_wait_timeout)

    effective_quarantine = int(args_cli.slam_reset_quarantine_updates)
    if args_cli.slam_bridge_require_fresh_always and effective_quarantine > 0:
        print(
            f"[INFO] bridge_require_fresh_always=True is incompatible with reset quarantine > 0. "
            f"Forcing slam_reset_quarantine_updates from {effective_quarantine} to 0.",
            flush=True,
        )
        effective_quarantine = 0
    slam_term.params["bridge_reset_quarantine_updates"] = effective_quarantine
    slam_term.params["bridge_require_fresh_when_full"] = True
    slam_term.params["bridge_require_fresh_always"] = bool(args_cli.slam_bridge_require_fresh_always)
    slam_term.params["yaw_length_m"] = float(args_cli.slam_bridge_yaw_length_m)

    print(
        "[INFO] SLAM bridge configuration: "
        f"blend_steps={args_cli.slam_blend_steps}  mode={args_cli.slam_bridge_mode}  "
        f"obs_index={args_cli.slam_bridge_obs_index}  step_wait_timeout={args_cli.slam_bridge_step_wait_timeout:.3f}s  "
        f"require_fresh_always={bool(args_cli.slam_bridge_require_fresh_always)}  "
        f"reset_quarantine_updates={effective_quarantine}"
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Ros2TrainingWrapper(
        env,
        obs_scalar_index=args_cli.slam_bridge_obs_index,
        publish_trajectory=args_cli.slam_publish_trajectory,
        attempt_hard_reset_on_env_reset=not args_cli.no_slam_hard_reset_on_env_reset,
        clear_queue_on_env_reset=not args_cli.no_slam_clear_queue_on_env_reset,
    )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

        # Fine-tuning with slam_toolbox should not reuse the Adam moments from the
        # original fixed-lag training run. Keep the model weights, but reset the
        # optimizer state and force the configured learning rate.
        if hasattr(runner.alg, "actor_critic"):
            optim_params = runner.alg.actor_critic.parameters()
        elif hasattr(runner.alg, "policy"):
            optim_params = runner.alg.policy.parameters()
        else:
            raise RuntimeError("Could not find policy parameters to rebuild optimizer for fine-tuning.")

        runner.alg.optimizer = torch.optim.Adam(optim_params, lr=float(agent_cfg.algorithm.learning_rate))
        if hasattr(runner.alg, "learning_rate"):
            runner.alg.learning_rate = float(agent_cfg.algorithm.learning_rate)

        print(
            f"[INFO] Fine-tuning optimizer reset. "
            f"LR = {float(agent_cfg.algorithm.learning_rate):.3e}",
            flush=True,
        )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()