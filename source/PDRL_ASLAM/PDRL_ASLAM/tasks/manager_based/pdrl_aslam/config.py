"""This module contains the configuration options for our work. For ease of
testing/debugging we restrict configuration options to either NO SLAM or
SLAM with occupancy grid reward function. Other options, including settings
from different ROS 2 packages, can be set in a similar manner."""


# Choose from: "NO_SLAM", "SLAM_and_occupancy_grid".
PROFILE = "NO_SLAM"

# Choose from: env1, env2, env3.
ENVIRONMENT = "env1"

# Choose from: "train", "retrain", "play".
PHASE = "train"

# Choose from: "no", "yes".
# This is just to verify that bridge is working correctly during training (not playing).
DEBUG = "no"


# -----------------------------------------------------------------------------------

##
# DO NOT CHANGE THE CODE BELOW
##

if PHASE == "train":
    # During initial training we do not want to use SLAM and we want PPO config for
    # massive parallelism.
    PROFILE = "NO_SLAM"
if DEBUG == "yes":
    PROFILE = "SLAM_and_occupancy_grid"
    PHASE = "retrain"


if ENVIRONMENT not in ("env1", "env2", "env3"):
    raise ValueError(f"Invalid ENVIRONMENT: {ENVIRONMENT}")
if PROFILE not in ("NO_SLAM", "SLAM_and_occupancy_grid"):
    raise ValueError(f"Invalid PROFILE: {PROFILE}")
if PHASE not in ("train", "retrain", "play"):
    raise ValueError(f"Invalid PHASE: {PHASE}")
if DEBUG not in ("no", "yes"):
    raise ValueError(f"Invalid DEBUG: {DEBUG}")
