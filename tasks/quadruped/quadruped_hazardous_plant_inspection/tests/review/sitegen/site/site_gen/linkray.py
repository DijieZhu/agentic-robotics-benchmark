"""The radio link as the cell decides it (mission/world.py link_visible), for the generator's proofs:
the link is up at a point when a working node (the base station's aerial, a working repeater)
is within its range and a ray from the point to the node hits no geom of group 0 (structure, door leaves,
equipment) before the last end_margin_m of its length. The robot's point is its lidar site; the heights below are
that site's for the postures a program uses near the pad or at a reading place.

The 2-D coverage grid of site_gen.link (radio-opaque prims only) does not see an open door leaf: a documents review
found the airlock's inner leaf shading the corridor and part of the pad circle on a quarter of the candidates. The
two proof rows that guard it use this module.
"""

from __future__ import annotations

import mujoco
import numpy as np

# lidar site above the body origin (robot.xml: body 'lidar' at z 0.06, site at 0.006) and body heights from the
# gait's capability.json v3 notes (walk 0.242 m, crouch_walk 0.190 m) and the resting sit (0.04 m)
LIDAR_ABOVE_BODY = 0.066
POSTURE_BODY_Z = {"walk": 0.242, "crouch_walk": 0.190, "sit": 0.040}


def lidar_heights(postures=("walk", "crouch_walk", "sit")):
    return {p: round(POSTURE_BODY_Z[p] + LIDAR_ABOVE_BODY, 4) for p in postures}


def cell_link_config(cfg) -> dict:
    """The cell's link settings (mission.config: link.base_range_m, repeater_range_m, end_margin_m), read as the cell
    reads them (one definition)."""
    from . import accept  # noqa: F401  (puts the library root on sys.path for mission)
    from mission.config import load_cfg
    from .config import acceptance_path
    return dict(load_cfg(acceptance=acceptance_path(cfg))["link"])


def nodes_of(base_station, repeaters, lk) -> list:
    """[(xyz, range)] of the working nodes, as mission/world.py builds them."""
    out = [(np.asarray(base_station["antenna"], float), float(lk["base_range_m"]))]
    for r in repeaters or []:
        if r.get("working", False):
            out.append((np.asarray(r["pos"], float), float(lk["repeater_range_m"])))
    return out


def link_up(model, data, P, nodes, margin) -> np.ndarray:
    """(N,) bool: the cell's link test from each point of P (N, 3)."""
    P = np.atleast_2d(np.asarray(P, float))
    out = np.zeros(len(P), bool)
    gg = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    gid = np.zeros(1, np.int32)
    for i, p in enumerate(P):
        for q, rng in nodes:
            v = q - p
            dist = float(np.linalg.norm(v))
            if dist > rng or dist < 1e-9:
                continue
            hit = mujoco.mj_ray(model, data, p, v / dist, gg, 1, -1, gid)
            if hit < 0 or hit >= dist - margin:
                out[i] = True
                break
    return out


def link_mask(model, data, grid, cells, nodes, margin, z) -> np.ndarray:
    """Bool grid: cells of `cells` (bool grid) where the link is up with the robot's lidar site at height z."""
    out = np.zeros(grid.shape, bool)
    iy, ix = np.nonzero(cells)
    if not len(iy):
        return out
    X, Y = grid.centers()
    P = np.c_[X[iy, ix], Y[iy, ix], np.full(len(iy), float(z))]
    ok = link_up(model, data, P, nodes, margin)
    out[iy[ok], ix[ok]] = True
    return out


def disc(grid, centre, r) -> np.ndarray:
    X, Y = grid.centers()
    return (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= r * r + 1e-12
