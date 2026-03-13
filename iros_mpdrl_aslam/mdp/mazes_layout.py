"""This module contains the RL environments (mazes) configuration. It
can be used to change environment sizes without affecting the global
shape."""

from .mazes import (
    HfRectMazeTerrainCfg,
    make_env1_rects,
    make_env2_smooth,
    make_env3_rects,
)

# Scaling helpers.
BASE_SIZE = (10.0, 10.0)
TARGET_SIZE = (10, 10)


def _scale_rects(rects, from_size, to_size):
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    return [(x * sx, y * sy, w * sx, h * sy) for (x, y, w, h) in rects]


def _scale_tris(tris, from_size, to_size):
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    out = []
    for (x0, y0, x1, y1, x2, y2) in tris:
        out.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return out


##
# Env1.
##


ENV1_RECTS_10 = make_env1_rects(size=BASE_SIZE, wall_t=0.26, cube=1.5, clear=2.5)
ENV1_RECTS = _scale_rects(ENV1_RECTS_10, BASE_SIZE, TARGET_SIZE)


env1_maze = HfRectMazeTerrainCfg(
    size=TARGET_SIZE,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    wall_height=0.50,
    border_thickness=0.26 * (TARGET_SIZE[0] / BASE_SIZE[0]),
    rects=ENV1_RECTS,

    # NEW:
    wall_edge_roughness_m=0.10,       # 1 cell at 0.05m resolution
    wall_edge_bump_length_m=0.10,     # ~2 cells
    wall_edge_bump_prob=0.35,
    wall_edge_seed=1,
)


##
# Env2.
##


TARGET_SIZE_2 = (10, 10)

ENV2_RECTS_10, ENV2_TRIS_10, _ = make_env2_smooth(size=BASE_SIZE, center_vertically=True)

ENV2_RECTS = _scale_rects(ENV2_RECTS_10, BASE_SIZE, TARGET_SIZE_2)
ENV2_TRIS = _scale_tris(ENV2_TRIS_10, BASE_SIZE, TARGET_SIZE_2)

env2_maze = HfRectMazeTerrainCfg(
    size=TARGET_SIZE_2,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    wall_height=0.50,
    border_thickness=0.26 * (TARGET_SIZE_2[0] / BASE_SIZE[0]),
    rects=ENV2_RECTS,
    tris=ENV2_TRIS,
)


##
# Env3.
##


TARGET_SIZE_3 = (10, 10)

ENV3_RECTS_10 = make_env3_rects(
    size=BASE_SIZE,
    wall_t=0.26,
    t=0.26,
)

ENV3_RECTS = _scale_rects(ENV3_RECTS_10, BASE_SIZE, TARGET_SIZE_3)

env3_maze = HfRectMazeTerrainCfg(
    size=TARGET_SIZE_3,
    horizontal_scale=0.04,
    vertical_scale=0.005,
    wall_height=0.50,
    border_thickness=0.26 * (TARGET_SIZE_3[0] / BASE_SIZE[0]),
    rects=ENV3_RECTS,
)
