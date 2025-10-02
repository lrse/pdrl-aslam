"""This module contains the RL environments design. We recommend not
modifying this file."""


from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.hf_terrains import height_field_to_mesh

import numpy as np


# Helpers
Rect = tuple[float, float, float, float]
Tri = tuple[float, float, float, float, float, float]


def _m2px(x: float, scale: float) -> int:
    return max(0, int(round(x / scale)))


def R(x: float, y: float, w: float, h: float) -> Rect:
    return (x, y, w, h)


def T(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> Tri:
    return (x0, y0, x1, y1, x2, y2)


@height_field_to_mesh
def rect_maze_terrain(difficulty: float, cfg: "HfRectMazeTerrainCfg") -> np.ndarray:
    """Builds a maze with rectangular and triangular blocks."""

    Wpx = max(8, _m2px(cfg.size[0], cfg.horizontal_scale))
    Lpx = max(8, _m2px(cfg.size[1], cfg.horizontal_scale))
    hf = np.zeros((Wpx, Lpx), dtype=np.int16)
    H = max(1, int(cfg.wall_height / cfg.vertical_scale))

    def stamp_rect(x: float, y: float, w: float, h: float):
        x0 = np.clip(_m2px(x, cfg.horizontal_scale), 0, Wpx - 1)
        y0 = np.clip(_m2px(y, cfg.horizontal_scale), 0, Lpx - 1)
        x1 = np.clip(_m2px(x + w, cfg.horizontal_scale), 0, Wpx)
        y1 = np.clip(_m2px(y + h, cfg.horizontal_scale), 0, Lpx)
        if x1 > x0 and y1 > y0:
            hf[x0:x1, y0:y1] = H

    def stamp_triangle(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float):
        xs = [x0, x1, x2]
        ys = [y0, y1, y2]
        px0 = int(np.clip(_m2px(min(xs), cfg.horizontal_scale), 0, Wpx - 1))
        px1 = int(np.clip(_m2px(max(xs), cfg.horizontal_scale), 0, Wpx - 1))
        py0 = int(np.clip(_m2px(min(ys), cfg.horizontal_scale), 0, Lpx - 1))
        py1 = int(np.clip(_m2px(max(ys), cfg.horizontal_scale), 0, Lpx - 1))

        sx = cfg.horizontal_scale
        ax, ay = x0, y0
        bx, by = x1, y1
        cx, cy = x2, y2

        def edge_sign(px, py, qx, qy, rx, ry):
            return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)

        for ix in range(px0, px1 + 1):
            xc = (ix + 0.5) * sx
            for iy in range(py0, py1 + 1):
                yc = (iy + 0.5) * sx
                b1 = edge_sign(xc, yc, ax, ay, bx, by) <= 0.0
                b2 = edge_sign(xc, yc, bx, by, cx, cy) <= 0.0
                b3 = edge_sign(xc, yc, cx, cy, ax, ay) <= 0.0
                if (b1 == b2) and (b2 == b3):
                    hf[ix, iy] = H

    if cfg.border_thickness > 0.0:
        t = cfg.border_thickness
        sx, sy = cfg.size
        for x, y, w, h in (
            (0.0, 0.0, sx, t),
            (0.0, sy - t, sx, t),
            (0.0, 0.0, t, sy),
            (sx - t, 0.0, t, sy),
        ):
            stamp_rect(x, y, w, h)

    for (x, y, w, h) in cfg.rects:
        stamp_rect(x, y, w, h)

    for tri in getattr(cfg, "tris", ()):
        stamp_triangle(*tri)

    return hf

##
# Env1.
##


def make_env1_rects(size=(10.0, 10.0), wall_t=0.26, cube=1.6, clear=1.2):

    sx, sy = size
    rects: list[Rect] = []

    x_col = sx - wall_t - clear - cube
    y_row = sy - wall_t - clear - cube

    rects += [
        R(x_col - 2 * cube, y_row, cube, cube),
        R(x_col - 1 * cube, y_row, cube, cube),
        R(x_col, y_row, cube, cube),
    ]

    rects += [
        R(x_col, y_row - 1 * cube, cube, cube),
        R(x_col, y_row - 2 * cube, cube, cube),
    ]

    big_side = 3.5
    rects.append(R(wall_t, wall_t, big_side, big_side))
    return rects


##
# Env2.
##


def make_env2_smooth(
    size: tuple[float, float] = (10.0, 10.0),
    wall_t: float = 0.26,
    t: float = 0.26,     # Inner walls width.
    x_vert: float = 5.25,
    y_top: float = 8.3,
    y_base: float = 3.4,
    top_len_left: float = 2.55,
    x_diag_top: float = 7.15,
    left_stub_y: float = 5.0,
    left_stub_len: float = 2.4,     # Size of the left bar that it is on its own.
    center_vertically: bool = True,
    seven_dy: float = 0.0,
) -> tuple[list[Rect], list[Tri], tuple[float, float]]:

    sx, sy = size
    if center_vertically:
        seven_dy = (sy - y_top - y_base) / 2.0

    rects: list[Rect] = []
    tris: list[Tri] = []

    y_top7 = y_top + seven_dy
    y_base7 = y_base + seven_dy

    rects.append(R(x_vert - t, y_base7, t, y_top7 - y_base7))
    rects.append(R(x_vert - top_len_left, y_top7, top_len_left, t))
    rects.append(R(x_vert - top_len_left, y_base7 - t, top_len_left, t))
    rects.append(R(wall_t, left_stub_y - 0.5 * t, left_stub_len, t))

    tris.append(T(x_vert, y_top7 + t, x_diag_top, y_top7 + t, x_vert, y_base7))
    return rects, tris, size


##
# Env3.
##


def make_env3_rects(
    size: tuple[float, float] = (10.0, 10.0),
    wall_t: float = 0.26,
    t: float = 0.26,
    left_col_x: float = 2.00,
    left_three_h: float = 1.50,
    mid_dy: float = -1.50,
    top_bar_len: float = 4.50,
    pillar_len: float = 3.00,
    tiny_vertical_h: float = 1.50,
    midbar_down_from_top: float = 1.50,
    midbar_len: float = 1.50,
    right_stub_len: float = 1.50,
    right_vert_bottom_y: float = 5.05,
    right_vert_h: float = 3.00,
    right_vert_dx: float = 0.20,
    top_right_len: float = 1.50,
) -> list[Rect]:

    sx, sy = size
    rects: list[Rect] = []

    def _add(x: float, y: float, w: float, h: float) -> None:
        if w > 1e-6 and h > 1e-6:
            rects.append(R(x, y, w, h))

    _add(left_col_x - 0.5 * t, sy - wall_t - left_three_h, t, left_three_h)
    _add(left_col_x - 0.5 * t, wall_t, t, left_three_h)

    # Central structure.
    C_top_y = 7.20 + mid_dy
    CL_x = left_col_x
    CR_x = left_col_x + top_bar_len

    _add(CL_x - 0.5 * t, C_top_y - left_three_h, t, left_three_h)
    top_left_x = CL_x - 0.5 * t
    top_right_x = CR_x + 0.5 * t
    _add(top_left_x, C_top_y, top_right_x - top_left_x, t)          # Main top bar.

    pillar_bottom_y = C_top_y - pillar_len
    _add(CR_x - 0.5 * t, pillar_bottom_y, t, pillar_len)

    Tmid_x = 4.10
    _add(Tmid_x - 0.5 * t, C_top_y + t, t, tiny_vertical_h)
    y_mid_center = C_top_y - midbar_down_from_top
    _add(CR_x - midbar_len, y_mid_center - 0.5 * t, midbar_len, t)

    right_inner_x = sx - wall_t
    stub_left = right_inner_x - right_stub_len
    _add(stub_left, pillar_bottom_y, right_stub_len, t)              # Right-most horizontal.

    right_vert_left_x = (stub_left - t) + right_vert_dx
    _add(right_vert_left_x, right_vert_bottom_y, t, right_vert_h)    # Right-most vertical.

    vert_top_y = right_vert_bottom_y + right_vert_h
    vert_right_edge_x = right_vert_left_x + t
    _add(vert_right_edge_x - top_right_len, vert_top_y, top_right_len, t)  # Top-most horizontal.

    _add(Tmid_x - 0.5 * t, wall_t, t, 1.45)

    return rects


##
# Config class.
##


@configclass
class HfRectMazeTerrainCfg(HfTerrainBaseCfg):

    function = rect_maze_terrain

    rects: list[Rect] = ()
    tris: list[Tri] = ()
    wall_height: float = 0.50
    border_thickness: float = 0.20
