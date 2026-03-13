"""This module contains the RL events configuration, which
in our case are mostly resets."""

from __future__ import annotations

# Starting training position.
POSE_RANGE = {"x": (-4, 4), "y": (4, 4), "z": (-0.5, -0.5), "yaw": (-3.14, 3.14)}


# Testing pose for videos.
# POSE_RANGE = {"x": (0.75, 0.75), "y": (4, 4), "z": (-0.5, -0.5), "yaw": (1.57, 1.57)}
