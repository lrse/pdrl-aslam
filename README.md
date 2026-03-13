# Massive Parallel Deep Reinforcement Learning for Active SLAM


## Overview ##
This repository provides the codes used for IROS 2026 submission.

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

1. Install ROS2 Humble following the [documentation's instructions](https://docs.ros.org/en/humble/Installation.html). Remember to always “source /opt/ros/humble/setup.bash” on any new CLI.
2. Create a new conda env with Python 3.10.19.
3. Install Isaac Sim 4.5 and Isaac Lab 2.3.2 following the [documentation's instructions](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). Make sure to include the installation of rsl_rl since this is the RL library we will use for training.
4. Clone this repository.
5. In case of wanting to test our training bridge wrapper, we recommend downloading slam_toolbox following the [documentation's instructions](https://docs.ros.org/en/humble/p/slam_toolbox/).


## Usage ##

- We will only modify the config.py file in our folder to change between our four environments.
- Training with our fixed-lag SLAM is designed to work with 750 in parallel in our machine (Intel Core i7 CPU, an NVIDIA RTX 4060 GPU 8GB VRAM, and 32GB RAM). However, retraining with slam_toolbox works well with 4 agents.
- If anything does not work, first reboot and retry, since that usually resolves common issues.

**0. Activate** your conda environment (generally this is: ```conda activate env_isaaclab_232``` if you followed the installation guidelines) and source ROS 2: 

```source /opt/ros/humble/setup.bash```


**1. Train with fixed-lag SLAM:**
- Run:
```python scripts/rsl_rl/train.py   --task Isaac-Iros-Mpdrl-Aslam-v0  --num_envs 750 --headless```

This will run the initial training with 750 environments in parallel and no visualization. 

- If visualization is wanted run instead: 

```python scripts/rsl_rl/train.py   --task Isaac-Iros-Mpdrl-Aslam-v0  --num_envs 750```


**2. Play with fixed-lag SLAM:**

By default, the loaded policy is the last model of the last run of the experiment folder logs/rsl_rl/iros_mpdrl_aslam

However, an already trained policy is saved in demos folder.

To run this already trained agent run:

```python scripts/rsl_rl/play.py --task Isaac-Iros-Mpdrl-Aslam-v0 --num_envs 1 --checkpoint demos/PPO_uncertainty.pt```

Different environments can be tested by changing the variables ENVIRONMENT and warehouse_bool in the config.py file.


**3. Retrain with slam_toolbox SLAM: TODO** 

- Open the config.py file and change the default settings so that:

PHASE = “retrain”

The difference between “train” and “retrain” is due to the different PPO hyperparameters in each case. These can be noticed on the rsl_rl_ppo_cfg.py file.

- Run the cuvslam_launcher.py file and wait until everything is set. The command to run it is: 

```python3 cuvslam_launcher.py```

Then run:

```
TODO
```

This will run the retraining with slam_toolbox SLAM with 4 environments in parallel and no visualization.

- If visualization is wanted, run instead:

```
TODO
```

**4. Play with slam_toolbox SLAM: TODO**

- Run the cuvslam_launcher.py file and wait until everything is set. The command to run it is:

```python3 cuvslam_launcher.py --single```

Notice that now we are using the flag --single since we want to play only one agent for visualization.

By default, the loaded policy is the last model of the last run of the experiment folder logs/rsl_rl/TODO

However, an already trained policy is saved in demos

- To run this already trained agent run:

```
TODO
```

Different environments can be tested by changing the variables ENVIRONMENT and warehouse_bool in the config.py file.


## Submission video ##

https://github.com/user-attachments/assets/23c5ee13-6702-46b4-ac61-b9c4d41755e7













