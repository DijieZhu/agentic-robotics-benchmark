"""Render-twin helpers of the cell's camera session (agent_render), for your own twin of the cell: the cell's own code.

  load_look(path=None), kit_materials(look=None)   the look file next to this module (agent_look.json) and its
                                                   `kit_materials` block: the palette of material classes (rgba,
                                                   specular, shininess), their physically based terms (`pbr`), the
                                                   roughness clamp of the opaque classes, `transparent_view`
  material_name(class)                             the material name the session's look rules match for a class:
                                                   "mat_vis_<class>"
  transparent_keys()                               the classes with alpha below 1: no depth return
  sync_map(src, dst), apply_sync(pairs, ...)        a state copy between two models: every joint of `dst` found by name
                                                   in `src`; a free joint of `dst` not found by name is matched through
                                                   its body's name (the source body's own free joint)
  TwinSession(backend, scene), TwinBackend()       a MuJoCo session (render.backends.mujoco_backend) of a twin with the
                                                   no-return rule on its depth: a pixel whose first surface on its ray
                                                   is a geom of a transparent class (by its material name) has depth 0,
                                                   set before the sensor model; `no_return = False` gives the exact
                                                   depth of every first surface
"""

import json
import os

import numpy as np

__all__ = ["LOOK_FILE", "VIS_GROUP", "load_look", "kit_materials", "transparent_keys", "material_name",
           "sync_map", "apply_sync", "TwinSession", "TwinBackend"]

HERE = os.path.dirname(os.path.abspath(__file__))
LOOK_FILE = os.path.join(HERE, "agent_look.json")
VIS_GROUP = 2
_LOOK = {}
_CLASSES = []


def load_look(path=None):
    p = path or LOOK_FILE
    if p not in _LOOK:
        with open(p) as f:
            _LOOK[p] = json.load(f)
    return _LOOK[p]


def kit_materials(look=None):
    return (look or load_look())["kit_materials"]


def transparent_keys(km=None):
    """palette classes with alpha < 1: no depth return"""
    km = km or kit_materials()
    return sorted(k for k, v in km["palette"].items() if float(v["rgba"][3]) < 1.0)


def material_name(pkey):
    return "mat_vis_" + pkey


def sync_map(src, dst):
    """[(src qpos address, dst qpos address, width)]: every joint of `dst` found by name in `src`; a free joint of
    `dst` not found by name is matched through its body's name (the source body's own free joint)"""
    import mujoco as mj
    pairs, done = [], set()
    width = {0: 7, 1: 4, 2: 1, 3: 1}
    for j in range(dst.njnt):
        nm = mj.mj_id2name(dst, mj.mjtObj.mjOBJ_JOINT, j)
        k = mj.mj_name2id(src, mj.mjtObj.mjOBJ_JOINT, nm) if nm else -1
        if k >= 0 and int(src.jnt_type[k]) == int(dst.jnt_type[j]):
            pairs.append((int(src.jnt_qposadr[k]), int(dst.jnt_qposadr[j]), width[int(dst.jnt_type[j])]))
            done.add(j)
    src_free = {}
    for k in range(src.njnt):
        if int(src.jnt_type[k]) == 0:
            src_free[mj.mj_id2name(src, mj.mjtObj.mjOBJ_BODY, int(src.jnt_bodyid[k]))] = int(src.jnt_qposadr[k])
    for j in range(dst.njnt):
        if j in done or int(dst.jnt_type[j]) != 0:
            continue
        bn = mj.mj_id2name(dst, mj.mjtObj.mjOBJ_BODY, int(dst.jnt_bodyid[j]))
        if bn in src_free:
            pairs.append((src_free[bn], int(dst.jnt_qposadr[j]), 7))
    return pairs


def apply_sync(pairs, src_data, dst_model, dst_data):
    import mujoco as mj
    for a, b, n in pairs:
        dst_data.qpos[b:b + n] = src_data.qpos[a:a + n]
    mj.mj_kinematics(dst_model, dst_data)
    mj.mj_comPos(dst_model, dst_data)
    mj.mj_camlight(dst_model, dst_data)


def _no_return_ids(model, keys):
    import mujoco as mj
    mats = set(mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, material_name(k)) for k in keys)
    mats.discard(-1)
    return np.array(sorted(g for g in range(model.ngeom) if int(model.geom_matid[g]) in mats), np.int32)


def _session_classes():
    if _CLASSES:
        return _CLASSES[0]
    from render.backends.base import RawView, raw_outputs
    from render.backends.mujoco_backend import MujocoBackend, MujocoSession

    class _TwinSession(MujocoSession):
        """A MuJoCo session of the twin with the no-return rule on its depth (module docstring) and the state copy of
        `sync_map`. no_return=False renders the exact depth of every first surface (a dev tool's switch)."""

        def __init__(self, backend, scene, no_return_keys=None):
            MujocoSession.__init__(self, backend, scene)
            self.no_return_keys = list(transparent_keys() if no_return_keys is None else no_return_keys)
            self.no_return_ids = _no_return_ids(self.model, self.no_return_keys)
            self.no_return = True
            self._pairs = None
            self.last_no_return_px = []

        def sync_from(self, model, data):
            if self._pairs is None or self._pairs[0] is not model:
                self._pairs = (model, sync_map(model, self.model))
            apply_sync(self._pairs[1], data, self.model, self.data)

        def render_raw_views(self, cameras, outputs=("rgb",)):
            outs = raw_outputs(outputs)
            want_d, want_s = "depth" in outs, "segmentation" in outs
            mask = want_d and self.no_return and len(self.no_return_ids) > 0
            need = list(outs) + (["segmentation"] if mask and not want_s else [])
            views = MujocoSession.render_raw_views(self, cameras, need)
            out = []
            self.last_no_return_px = []
            for v in views:
                d = v.depth
                if mask:
                    hit = np.isin(v.seg, self.no_return_ids)
                    d = np.where(hit, np.float32(0.0), d).astype(np.float32)
                    self.last_no_return_px.append(int(hit.sum()))
                out.append(RawView(rgb=v.rgb, depth=d, seg=v.seg if want_s else None))
            return out

    class _TwinBackend(MujocoBackend):
        def open(self, scene):
            return _TwinSession(self, scene)

    _CLASSES.append((_TwinSession, _TwinBackend))
    return _CLASSES[0]


def TwinSession(backend, scene, no_return_keys=None):
    return _session_classes()[0](backend, scene, no_return_keys)


def TwinBackend(gl=None):
    return _session_classes()[1](gl)
