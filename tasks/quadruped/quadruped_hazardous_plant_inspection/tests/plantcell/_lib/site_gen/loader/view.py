"""Rendering options every camera of the site needs (the classic renderer).

Geom groups: 0 structure / equipment (collision), 1 decor, 2 ceiling and overhead decor, 3 traversal elements
(collision). MuJoCo hides groups 3-5 by default: robot cameras must render with scene_option() so sills, kerbs,
hoses and floor pipes are drawn. The classic renderer draws at most 8 lights (the headlight included): before each
frame, lights_for_view() switches on the room lights nearest to the camera.
"""

from __future__ import annotations

import mujoco
import numpy as np


def lights_for_view(model, pos, n=7, fills=False):
    """Activate the n room lights nearest to `pos` (and the fill lights if asked); returns the active light ids."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or "" for i in range(model.nlight)]
    d = [(np.linalg.norm(model.light_pos[i][:2] - np.asarray(pos)[:2]), i) for i in range(model.nlight)
         if not names[i].startswith("light/fill")]
    d.sort()
    act = [i for _, i in d[:n]]
    for i in range(model.nlight):
        model.light_active[i] = 1 if (i in act or (fills and names[i].startswith("light/fill"))) else 0
    return act


def scene_option(show_overhead=True):
    """MjvOption with the site's geom groups visible: 0 structure, 1 decor, 2 overhead (optional), 3 traversal
    elements."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[3] = 1
    opt.geomgroup[2] = 1 if show_overhead else 0
    return opt
