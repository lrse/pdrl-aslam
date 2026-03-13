"""This module contains the configuration options for our work. For ease of
testing/debugging we restrict configuration options to either NO SLAM or
SLAM with occupancy grid reward function. Other options, including settings
from different ROS 2 packages, can be set in a similar manner."""


# Choose from: env1, env2, env3.
ENVIRONMENT = "env1"

# Choose from: True, False.
warehouse_bool = False      # Setting it to True cancels the ENVIRONMENT setting.

# Choose from: train, retrain.
PHASE = "train"


##
# DO NOT CHANGE THE CODE BELOW
##


if ENVIRONMENT not in ("env1", "env2", "env3"):
    raise ValueError(f"Invalid ENVIRONMENT: {ENVIRONMENT}")

if PHASE not in ("train", "retrain"):
    raise ValueError(f"Invalid PHASE: {PHASE}")
