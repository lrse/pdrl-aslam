# Massive Parallel Deep Reinforcement Learning for Active SLAM


## Overview ##
This repository provides the code used for IROS 2026 submission.

**Authors**: Private.

**Maintainer**: Private.

**Affiliation**: Private.

**Contact**: Private.

---

## Requirements ##

This software stack was tested on:

1. Ubuntu 22.04.
2. NVIDIA driver 570.172.08.
3. CUDA 12.8.
4. Python 3.10.19

## Installation ##

1. Install ROS2 Humble following the [documentation instructions](https://docs.ros.org/en/humble/Installation.html). Remember to always “source /opt/ros/humble/setup.bash” on any new CLI.
2. Create a new Conda environment with Python 3.10.19.
3. Install Isaac Sim 4.5 and Isaac Lab 2.3.2 following the [documentation instructions](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). Make sure to include the installation of rsl_rl since this is the RL library we will use for training.
4. Clone this repository.
5. If you want to test our training bridge wrapper, we recommend downloading slam_toolbox following the [documentation instructions](https://docs.ros.org/en/humble/p/slam_toolbox/).


## Usage ##

- We will only modify the config.py file in our folder to switch between our four environments.
- Training with our fixed-lag SLAM is designed to work with 750 environments in parallel in our machine (Intel Core i7 (13th generation) CPU, an NVIDIA RTX 4060 GPU 8GB VRAM, and 32GB RAM). However, retraining with slam_toolbox works well with 4 agents.
- If anything does not work, first reboot and retry, since that usually resolves common issues.

**0. Activate** your conda environment (generally this is: ```conda activate env_isaaclab``` if you followed the installation guidelines) and source ROS2: 

```source /opt/ros/humble/setup.bash```

**1. Train with fixed-lag SLAM:**
- Run:
```python scripts/rsl_rl/train.py   --task Isaac-Iros-Mpdrl-Aslam-v0  --num_envs 750 --headless  --kit_args "--enable isaacsim.asset.importer.urdf```

This will run the initial training with 750 environments in parallel and no visualization. 

- If visualization is desired, run instead: 

```python scripts/rsl_rl/train.py   --task Isaac-Iros-Mpdrl-Aslam-v0  --num_envs 750  --kit_args "--enable isaacsim.asset.importer.urdf```


**2. Play with fixed-lag SLAM:**

By default, the loaded policy is the last model of the last run of the experiment folder logs/rsl_rl/iros_mpdrl_aslam .

However, an already trained policy is saved in the demos folder.

To run our already trained agents, run:

```python scripts/rsl_rl/play.py --task Isaac-Iros-Mpdrl-Aslam-v0 --num_envs 1 --checkpoint demos/PPO_exploratory.pt  --kit_args "--enable isaacsim.asset.importer.urdf```

or

```python scripts/rsl_rl/play.py --task Isaac-Iros-Mpdrl-Aslam-v0 --num_envs 1 --checkpoint demos/PPO_uncertainty.pt  --kit_args "--enable isaacsim.asset.importer.urdf```

Different environments can be tested by changing the variables ENVIRONMENT and warehouse_bool in the config.py file.


**3. Retrain with slam_toolbox SLAM:** 

- Open the config.py file and change the default settings so that:

PHASE = “retrain”

The difference between “train” and “retrain” is due to the different PPO hyperparameters in each case. These can be seen in the rsl_rl_ppo_cfg.py file.

- Run the slam_toolbox_launcher.py file and wait until everything is set. The command to run it is: 

```python3 slam_toolbox_launcher.py   --num-envs 4   --mode async   --spawn-mode background   --logs-dir ~/slam_toolbox_logs```

Then run:

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/Retraining\ SLAM/train_with_SLAM.py   --task Isaac-Iros-Mpdrl-Aslam-v0   --num_envs 4 --resume   agent.load_run=demos   agent.load_checkpoint=PPO_uncertainty.pt   --headless   --kit_args "--enable isaacsim.asset.importer.urdf"
```

This will run the retraining with slam_toolbox SLAM with 4 environments in parallel and no visualization.

- If visualization is desired, run instead:

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/Retraining\ SLAM/train_with_SLAM.py   --task Isaac-Iros-Mpdrl-Aslam-v0   --num_envs 4 --resume   agent.load_run=demos   agent.load_checkpoint=PPO_uncertainty.pt   --kit_args "--enable isaacsim.asset.importer.urdf"
```

**4. Play with slam_toolbox SLAM:**

- Run the slam_toolbox_launcher.py file and wait until everything is set. The command to run it is:

```python3 slam_toolbox_launcher.py   --num-envs 1   --mode async   --spawn-mode background   --logs-dir ~/slam_toolbox_logs```

By default, the loaded policy is the last model of the last run of the experiment folder logs/rsl_rl/iros_mpdrl_aslam .

However, an already trained policy is saved in the demos folder.

- To run our already trained agent, run:

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/Retraining\ SLAM/play_with_SLAM.py   --task Isaac-Iros-Mpdrl-Aslam-v0   --num_envs 1   --resume   agent.load_run=demos   agent.load_checkpoint=PPO_slam_toolbox.pt   --kit_args "--enable isaacsim.asset.importer.urdf"
```

Different environments can be tested by changing the variables ENVIRONMENT and warehouse_bool in the config.py file.


**5. Visualizing with rviz2:**

If we want to visualize the trajectory, we can expose the trajectory topic with --slam_publish_trajectory. Also, if we want to view an appealing occupancy grid we recommend modifying the slam_toolbox_launcher.py parameters for that purpose, since the current ones are set to optimize the retraining step.

## Submission video ##

https://github.com/user-attachments/assets/23c5ee13-6702-46b4-ac61-b9c4d41755e7













