# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL. It incorporates our bridge for training with SLAM plus debugging options."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
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
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import PDRL_ASLAM.tasks  # noqa: F401

# Bridge import.
from ros2_training_wrapper import Ros2TrainingWrapper

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
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

    # Bridge wrapper.
    env = Ros2TrainingWrapper(
        env
    )


##
# DEBUGGER
##

    import numpy as np
    import pathlib
    import sys

    np.set_printoptions(precision=4, suppress=True)

    ROOT = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))

    from source.PDRL_ASLAM.PDRL_ASLAM.tasks.manager_based.pdrl_aslam.config import DEBUG, PROFILE

    if DEBUG == "yes":
        obs, info = env.reset()
        dm = env.unwrapped.data_manager
        N = dm.num_envs

        lidar = env.unwrapped.scene.sensors["horizontal_scanner_1"]
        num_rays = int(lidar.data.ray_hits_w.shape[1])
        X_IDX, Y_IDX = num_rays, num_rays + 1

        has_slam_b = (PROFILE == "SLAM_and_occupancy_grid")
        T_IDX = (num_rays + 2) if has_slam_b else None
        term_label = "slam_b" if has_slam_b else "no_slam"

        print(f"[DEBUG] num_envs={N}  rays={num_rays}  X_IDX={X_IDX}  Y_IDX={Y_IDX}"
              + (f"  T_IDX={T_IDX} ({term_label})" if has_slam_b else f"  ({term_label})"))

        def _to_numpy_2d(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.ndim == 2 and x.shape[0] == N:
                return x
            if x.ndim == 1 and x.size % N == 0:
                return x.reshape(N, -1)
            if isinstance(x, (list, tuple)) and len(x) == N:
                parts = [np.asarray(p.detach().cpu().numpy() if hasattr(p, "detach") else p) for p in x]
                return np.stack(parts, axis=0)
            raise ValueError(f"Cannot coerce obs to (N, D); got shape={x.shape}, type={type(x)}")

        def split_xy_term(obs_any):
            base = _to_numpy_2d(obs_any["policy"] if (isinstance(obs_any, dict) and "policy" in obs_any) else obs_any)
            D = base.shape[1]
            if max(X_IDX, Y_IDX) >= D:
                raise IndexError(f"(x,y) indices out of range for obs dim D={D}.")
            xy = base[:, [X_IDX, Y_IDX]].astype(float)
            if has_slam_b:
                if T_IDX >= D:
                    raise IndexError(f"slam_b index {T_IDX} out of range for obs dim D={D}.")
                term = base[:, T_IDX].astype(float)
            else:
                term = np.full((N,), np.nan, dtype=float)
            return xy, term

        if isinstance(obs, dict) and "policy" in obs:
            pol = _to_numpy_2d(obs["policy"])
            lo = max(0, X_IDX - 3)
            hi = min(pol.shape[1], (T_IDX if has_slam_b else Y_IDX) + 3)
            print(f"[DEBUG] policy[0] dims {lo}..{hi-1}:\n{pol[0, lo:hi]}")

        PRINT_EVERY = 1
        DEBUG_STEPS = 1000
        env_device = getattr(getattr(env, "unwrapped", env), "device", None)

        for t in range(DEBUG_STEPS):
            act = env.action_space.sample()
            if not hasattr(act, "to"):
                import torch
                act = torch.as_tensor(act, dtype=torch.float32, device=env_device) if env_device else torch.as_tensor(act, dtype=torch.float32)

            obs, rew, terminated, truncated, info = env.step(act)

            if t % PRINT_EVERY == 0:
                obs_xy, obs_term = split_xy_term(obs)
                snapshots = getattr(env, "_last_snapshots", None)

                for i in range(N):
                    seq, xy = snapshots[i] if snapshots is not None else (None, None)
                    diff_xy = None if xy is None else (obs_xy[i] - np.asarray(xy, dtype=float))
                    term_val = obs_term[i]
                    print(
                        f"t={t} env{i}: seq={seq}  "
                        f"obs_xy={obs_xy[i]}  slam_xy={xy}  diff={diff_xy}  "
                        f"{term_label}={term_val if np.isfinite(term_val) else 'n/a'}"
                    )

        # Reset again, so training starts clean.
        obs, info = env.reset()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
