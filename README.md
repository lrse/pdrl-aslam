# Parallel Deep Reinforcement Learning for Active SLAM


## Overview ##
This repository provides the codes used for ICRA 2026 submission. It includes the bridge to train RL agents on Isaac Lab with ROS2 topics data, the Isaac ROS Visual SLAM configuration, the environments, and some obtained policies.


**Maintainer**: Private.

**Affiliation**: Private.

**Contact**: Private.

---

## Requirements ##

This software stack was tested on:

1. Ubuntu 22.04.
2. NVIDIA driver 570.172.08.
3. CUDA 12.8.
4. Python 3.10.18

## Installation ##

1. Install ROS2 Humble following the [documentation's instructions](https://docs.ros.org/en/humble/Installation.html). Remember to always “source /opt/ros/humble/setup.bash” on any new CLI.
2. Create a new conda env with python 3.10.18.
3. Install Isaac Gym 4.5 and Isaac Lab 2.1 following the [documentation's instructions](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). Make sure to include the installation of rsl_rl since this is the RL library we will use for training.
4. Download Isaac ROS Visual SLAM container following the [documentation's instructions](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html).
5. Clone this repository.
6. In case of wanting to visualize an occupancy grid, as in our video, we recommend downloading slam_toolbox following the [documentation's instructions](https://docs.ros.org/en/humble/p/slam_toolbox/). The LiDAR topics are already configured to be subscribed for this package.


## Usage ##

- We will always assume that we are inside the PDRL_ASLAM folder.
- We will only modify the config.py file.
- The training/testing with SLAM is designed to work with one or two environments in parallel in our machine (Intel Core i7 CPU, an NVIDIA RTX 4060 GPU, and 32 GB of RAM). However, the default settings consider two environments. 
- More agents can be easily added with minimal additions if the hardware allows it. 
- If anything does not work, first reboot and retry, since that usually resolves common issues.

**0. Activate** your conda environment (generally this is: ```conda activate env_isaaclab``` if you followed the installation guidelines) and source ROS 2: 

```source /opt/ros/humble/setup.bash```


**1. Train without SLAM:**
- Run:
```python scripts/rsl_rl/train.py   --task Template-Pdrl-Aslam-v0  --num_envs 512 --headless```

This will run the initial training without SLAM with 512 environments in parallel and no visualization. 

- If visualization is wanted run instead: 

```python scripts/rsl_rl/train.py   --task Template-Pdrl-Aslam-v0  --num_envs 512```


**2. Play without SLAM:**

By default, the loaded policy is the last model of the last run of the experiment folder PDRL_ASLAM/logs/rsl_rl/PDRL_ASLAM_v0

However, an already trained policy is saved in PDRL_ASLAM/demos

To run this already trained agent run:

```python scripts/rsl_rl/play.py --task Template-Pdrl-Aslam-v0 --num_envs 1 --checkpoint demos/trained_no_SLAM_agent.pt```

Different environments can be tested by changing the variable ENVIRONMENT in the config.py file.


**3. Retrain with SLAM:**

- Open the config.py file and change the default settings so that:

PROFILE = “SLAM_and_occupancy_grid”

PHASE = “retrain”

The difference between “train” and “retrain” is due to the different PPO hyperparameters in each case. These can be noticed on the rsl_rl_ppo_cfg.py file.

- Run the cuvslam_launcher.py file and wait until everything is set. The command to run it is: 

```python3 cuvslam_launcher.py```

Then run:

```
python scripts/rsl_rl/train_with_SLAM.py \
  --task Template-Pdrl-Aslam-v0  --num_envs 2 --headless \
  --resume \
  agent.load_run=Trained_NO_SLAM \
  agent.load_checkpoint=trained_no_SLAM.pt \
--enable_cameras
```

This will run the retraining with SLAM with 2 environments in parallel and no visualization.

- If visualization is wanted, run instead:

```
python scripts/rsl_rl/train_with_SLAM.py \
  --task Template-Pdrl-Aslam-v0  --num_envs 2 \
  --resume \
  agent.load_run=Trained_NO_SLAM \
  agent.load_checkpoint=trained_no_SLAM.pt \
--enable_cameras
```

**4. Play with SLAM:**

- Open the config.py file and change the default settings so that:

PROFILE = “SLAM_and_occupancy_grid”

PHASE = “play”

- Run the cuvslam_launcher.py file and wait until everything is set. The command to run it is:

```python3 cuvslam_launcher.py --single```

Notice that now we are using the flag --single since we want to play only one agent for visualization.

By default, the loaded policy is the last model of the last run of the experiment folder PDRL_ASLAM/logs/rsl_rl/PDRL_ASLAM_v0

However, an already trained policy is saved in 
PDRL_ASLAM/demos

- To run this already trained agent run:

```
python scripts/rsl_rl/play_with_SLAM.py --task Template-Pdrl-Aslam-v0 --num_envs 1 --checkpoint demos/retrained_with_SLAM_agent.pt --enable_cameras
```

Different environments can be tested by changing the variable ENVIRONMENT in the config.py file.

**5. Training debug:**

Open the config.py file and change the settings so that:

DEBUG = “yes”

Run the cuvslam_launcher.py file and wait until everything is set.

```python scripts/rsl_rl/train_with_SLAM.py --task Template-Pdrl-Aslam-v0  --num_envs 2```

## Submission video ##

<video controls width="600" muted>
  <source src="video/ICRA26_3061_VI_i.mp4" type="video/mp4" />
  Sorry, your browser doesn’t support embedded videos.
</video>









