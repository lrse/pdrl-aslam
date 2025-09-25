# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Script to play a checkpoint if an RL agent from RSL-RL."""

# """Launch Isaac Sim Simulator first."""

# import argparse

# from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
# parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import gymnasium as gym
# import os
# import time
# import torch

# from rsl_rl.runners import OnPolicyRunner

# from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


# from ros2_training_wrapper import Ros2TrainingWrapper
# from go2_ros2_bridge import RobotDataManager

# import PDRL_ASLAM.tasks  # noqa: F401


# def main():
#     """Play with RSL-RL agent."""
#     task_name = args_cli.task.split(":")[-1]
#     # parse configuration
#     env_cfg = parse_env_cfg(
#         args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
#     )
#     agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     if args_cli.use_pretrained_checkpoint:
#         resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
#         if not resume_path:
#             print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
#             return
#     elif args_cli.checkpoint:
#         resume_path = retrieve_file_path(args_cli.checkpoint)
#     else:
#         resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

#     log_dir = os.path.dirname(resume_path)

#     # create isaac environment
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "play"),
#             "step_trigger": lambda step: step == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     dm_kwargs = {
#         "env": env.unwrapped,
#         "lidar_annotators": [],   # not used by this minimal bridge
#         "cameras": [],
#         "cfg": None
#     }

#     env = Ros2TrainingWrapper(
#         env,
#         dm_ctor=lambda **kw: RobotDataManager(**kw),
#         dm_kwargs=dm_kwargs,
#         require_subs=True,      #CHANGED FROM TRUE
#         subs_wait_timeout=10.0,    # e.g., give cuVSLAM 10s to connect     #CHANGED FROM 10.0
#         # slam_wait_timeout=10.0,                                 #CHANGED FROM 10.0
#         # slam_require_all=True,     # wait for pose on all envs (optional but consistent)         #CHANGED FROM TRUE
#     )

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # load previously trained model
#     ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
#     ppo_runner.load(resume_path)

#     # obtain the trained policy for inference
#     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

#     # extract the neural network module
#     # we do this in a try-except to maintain backwards compatibility.
#     try:
#         # version 2.3 onwards
#         policy_nn = ppo_runner.alg.policy
#     except AttributeError:
#         # version 2.2 and below
#         policy_nn = ppo_runner.alg.actor_critic

#     # export policy to onnx/jit
#     export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
#     export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
#     export_policy_as_onnx(
#         policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
#     )

#     dt = env.unwrapped.step_dt

#     # reset environment
#     obs, _ = env.get_observations()
#     timestep = 0
#     # simulate environment
#     while simulation_app.is_running():
#         start_time = time.time()
#         # run everything in inference mode
#         with torch.inference_mode():
#             # agent stepping
#             actions = policy(obs)
#             # env stepping
#             obs, _, _, _ = env.step(actions)
#         if args_cli.video:
#             timestep += 1
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - start_time)
#         if args_cli.real_time and sleep_time > 0:
#             time.sleep(sleep_time)

#     # close the simulator
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


########################### ABOVE IS THE OLDER FILE ###################################################################


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import os

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ---- added CSV logging args ----
parser.add_argument("--reward_log", type=str, default="rewards.csv",
                    help="Path to CSV file to append per-step rewards (default: rewards.csv).")
parser.add_argument("--reward_log_interval", type=int, default=100,
                    help="Flush the reward log to disk every N steps (default: 100).")
parser.add_argument("--episode_log", type=str, default="episodes.csv",
                    help="Path to CSV file to append per-episode summaries (default: episodes.csv).")
# --------------------------------

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from ros2_training_wrapper import Ros2TrainingWrapper
from go2_ros2_bridge import RobotDataManager

# ---- added for CSV logging ----
import csv
import numpy as np

def _read_cov_snapshot(env, env_id: int):
    """
    Preferred: read coverage snapshot captured by the env at reset time
    into env.unwrapped._last_episode_cov. Falls back to (-1,-1,nan) if missing.
    """
    cov = getattr(env.unwrapped, "_last_episode_cov", None)
    if cov is None:
        return -1, -1, float("nan")
    try:
        cells = int(cov["covered"][env_id])
        total = int(cov["total"][env_id])
        frac  = float(cov["frac"][env_id])
        # clear after reading so we don't reuse on next episode
        cov["covered"][env_id] = -1
        cov["total"][env_id]   = -1
        cov["frac"][env_id]    = np.nan
        return cells, total, frac
    except Exception:
        return -1, -1, float("nan")


def _coverage_from_grid_now(env, env_id: int):
    """
    Fallback: try to read current visited grid immediately on 'done' step.
    This may be 0 if the wrapper already auto-reset.
    """
    S = getattr(env.unwrapped, "_pose_cov_state", None)
    if not S or "cov2d" not in S:
        return None
    C = S["cov2d"]
    vis = C["visited"][env_id]
    covered = int(np.count_nonzero(vis))
    total   = int(C["total"])
    frac    = float(covered / total) if total > 0 else 0.0
    return covered, total, frac


log_writer = None
log_file = None
ep_log_writer = None
ep_log_file = None

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# initialize CSV writers (module-level; we only *use* them in main)
if args_cli.reward_log:
    parent = os.path.dirname(args_cli.reward_log)
    if parent:
        os.makedirs(parent, exist_ok=True)
    log_file = open(args_cli.reward_log, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "env_id", "reward"])  # per-step schema

if args_cli.episode_log:
    parent = os.path.dirname(args_cli.episode_log)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ep_log_file = open(args_cli.episode_log, "w", newline="")
    ep_log_writer = csv.writer(ep_log_file)
    ep_log_writer.writerow([
        "step_end", "env_id", "episode_return", "episode_length",
        "cov_cells", "cov_total", "cov_frac"
    ])  # per-episode schema

# --------------------------------

import PDRL_ASLAM.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    dm_kwargs = {
        "env": env.unwrapped,
        "lidar_annotators": [],   # not used by this minimal bridge
        "cameras": [],
        "cfg": None
    }

    env = Ros2TrainingWrapper(
        env,
        dm_ctor=lambda **kw: RobotDataManager(**kw),
        dm_kwargs=dm_kwargs,
        require_subs=True,           # CHANGED FROM TRUE (kept as provided)
        subs_wait_timeout=10.0,      # CHANGED FROM 10.0 (kept as provided)
        # slam_wait_timeout=10.0,
        # slam_require_all=True,
    )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # ---- episode accumulators (local to avoid UnboundLocalError) ----
    ep_ret = None
    ep_len = None
    # ----------------------------------------------------------------

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, reward, dones, info = env.step(actions)

        # ----- logging (per-step + per-episode) -----
        r = _to_numpy(reward).reshape(-1)  # works for single or vectorized envs

        # lazy init after first step (know num_envs now)
        if ep_ret is None:
            ep_ret = np.zeros(r.shape[0], dtype=np.float32)
            ep_len = np.zeros(r.shape[0], dtype=np.int32)

        # update accumulators
        ep_ret += r
        ep_len += 1

        # per-step rewards
        if log_writer is not None:
            for env_id, ri in enumerate(r):
                log_writer.writerow([timestep, env_id, float(ri)])
            if timestep % args_cli.reward_log_interval == 0:
                log_file.flush()

        # per-episode summaries when done
# per-episode summaries when done (with coverage columns)
        if dones is not None and ep_log_writer is not None:
            d = _to_numpy(dones).astype(bool).reshape(-1)
            for env_id, done in enumerate(d):
                if not done:
                    continue

                # 1) preferred: snapshot saved by env at reset
                cov_cells, cov_total, cov_frac = _read_cov_snapshot(env, env_id)

                # 2) fallback: try to read current grid immediately (may be zeroed if auto-reset)
                if cov_cells < 0:
                    alt = _coverage_from_grid_now(env, env_id)
                    if alt is not None:
                        cov_cells, cov_total, cov_frac = alt

                ep_log_writer.writerow([
                    timestep, env_id, float(ep_ret[env_id]), int(ep_len[env_id]),
                    int(cov_cells), int(cov_total), float(cov_frac)
                ])

                # reset for next episode
                ep_ret[env_id] = 0.0
                ep_len[env_id] = 0

            if timestep % args_cli.reward_log_interval == 0:
                ep_log_file.flush()

        # ----- logging end -----

        # advance time step every loop
        timestep += 1

        # video stop condition
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # close CSV files
    if log_file is not None:
        log_file.flush()
        log_file.close()
    if ep_log_file is not None:
        ep_log_file.flush()
        ep_log_file.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
