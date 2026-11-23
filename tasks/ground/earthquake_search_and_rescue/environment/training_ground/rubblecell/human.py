"""People-shaped things of the site (cell internals): the survivor (warm, alive, dusty clothes) and the decoys a camera
could take for one (a shop mannequin, a heap of clothes). A figure is a real body in a lying pose (figures.json: the
joints; the look pack: the posed body's mesh, which every sensor sees) and, for contact, capsules along its bones,
each tagged with what it is made of (skin, shirt, trousers, shoes, plastic).

Local frame of a lying figure: x along the body (head at +x), y to its left, z up; the floor at z = 0.
"""

import math

import numpy as np

__all__ = ["figure_parts", "figure_key", "figure_layout", "figures", "SKIN", "SHIRT", "TROUSERS"]

SKIN = [(0.87, 0.70, 0.58), (0.76, 0.57, 0.44), (0.58, 0.41, 0.30), (0.40, 0.28, 0.20)]
HAIR = [(0.10, 0.08, 0.06), (0.30, 0.20, 0.12), (0.55, 0.45, 0.30), (0.25, 0.25, 0.25)]
SHIRT = [(0.72, 0.18, 0.15), (0.20, 0.35, 0.62), (0.85, 0.80, 0.55), (0.25, 0.48, 0.30), (0.55, 0.55, 0.58)]
TROUSERS = [(0.20, 0.24, 0.35), (0.30, 0.28, 0.25), (0.15, 0.15, 0.16), (0.45, 0.40, 0.30)]


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _seg(a, b, r, what, name):
    """A capsule part from a to b (local coordinates)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return {"type": "capsule", "from": a, "to": b, "r": float(r), "what": what, "name": name}


def _ell(c, radii, yaw, what, name, pitch=0.0):
    return {"type": "ellipsoid", "pos": np.asarray(c, float), "size": tuple(float(v) for v in radii), "yaw": float(yaw),
            "pitch": float(pitch), "what": what, "name": name}


_FIG = {}


def figures():
    """The posed figures (figures.json beside this file): per key ("side", "back", "prone": the survivor's three lying
    poses; "mannequin": the shop mannequin, face down) the joints of a real body lying in that pose, in the figure's
    local frame, its bounds, where its lower legs lie and how high. The poses are the last frames of motion-capture
    clips of a person falling and lying still, transferred to a clothed body model and settled on the floor; the same
    posed body is what the cameras, the depth camera, the thermal camera and the lidar see (the look pack's
    avatars/<key>.npz, scene.py), and the capsules built here from its joints are what the robot collides with."""
    if not _FIG:
        import json
        import os
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures.json")) as f:
            _FIG.update(json.load(f))
    return _FIG


def figure_key(pose, mannequin=False):
    return "mannequin" if mannequin else pose


# the capsules of a figure: (name, from joint, to joint, radius, what it is made of)
_SEGS = [("torso", "pelvis", "neck", 0.125, "shirt"), ("neck", "neck", "head", 0.055, "skin"),
         ("head", "head", "headtop", 0.095, "skin"),
         ("uarm_l", "uarm_l", "farm_l", 0.048, "shirt"), ("farm_l", "farm_l", "hand_l", 0.040, "skin"),
         ("uarm_r", "uarm_r", "farm_r", 0.048, "shirt"), ("farm_r", "farm_r", "hand_r", 0.040, "skin"),
         ("thigh_l", "thigh_l", "shin_l", 0.078, "trousers"), ("shin_l", "shin_l", "foot_l", 0.055, "trousers"),
         ("thigh_r", "thigh_r", "shin_r", 0.078, "trousers"), ("shin_r", "shin_r", "foot_r", 0.055, "trousers"),
         ("foot_l", "foot_l", "toe_l", 0.045, "shoes"), ("foot_r", "foot_r", "toe_r", 0.045, "shoes")]


def figure_parts(pose="back", scale=1.0, rng=None, mannequin=False):
    """The collision parts of a lying adult in pose "back" (face up), "side" (on the side, knees drawn up) or "prone"
    (face down), or of the mannequin: capsules along the posed body's bones (and one from each wrist along the forearm:
    the hand). Returns (parts, head points, hand points) in the local frame, scaled by `scale`."""
    s = float(scale)
    fig = figures()[figure_key(pose, mannequin)]
    J = {k: np.array(v, float) * s for k, v in fig["joints"].items()}
    P = []
    for name, j0, j1, r, what in _SEGS:
        a, b = J[j0].copy(), J[j1].copy()
        a[2], b[2] = max(a[2], r * s), max(b[2], r * s)          # a capsule never reaches under the floor
        P.append(_seg(a, b, r * s, what, name))
    hands = []
    for side in ("l", "r"):
        w, e = J["hand_" + side], J["farm_" + side]
        u = (w - e) / max(1e-6, float(np.linalg.norm(w - e)))
        tip = w + u * 0.10 * s
        a, b = w.copy(), tip.copy()
        a[2], b[2] = max(a[2], 0.03 * s), max(b[2], 0.03 * s)
        P.append(_seg(a, b, 0.030 * s, "skin", "hand_" + side))
        hands.append(w + u * 0.05 * s)
    heads = [0.5 * (J["head"] + J["headtop"])]
    if mannequin:
        for p in P:
            p["what"] = "plastic"
    return P, heads, hands


def figure_layout(pose, scale=1.0, mannequin=False):
    """Where things around a figure go (local frame): the middle of the torso, the middle of the lower legs and their
    top, the body's reach to its left (+y) and right (-y)."""
    s = float(scale)
    fig = figures()[figure_key(pose, mannequin)]
    J = fig["joints"]
    torso = 0.5 * (np.array(J["pelvis"], float) + np.array(J["neck"], float)) * s
    lo, hi = np.array(fig["bounds"][0], float) * s, np.array(fig["bounds"][1], float) * s
    return {"torso": torso, "legs": np.array(fig["legs_centre"], float) * s, "legs_top": float(fig["legs_top"]) * s,
            "left": float(hi[1]), "right": float(-lo[1]), "x0": float(lo[0]), "x1": float(hi[0]), "top": float(hi[2])}


def place_parts(parts, x, y, yaw, z0=0.0):
    """World placement of a figure's parts: rotate by yaw about z, translate to (x, y, z0)."""
    R = _rot(yaw)
    t = np.array([x, y, z0])
    out = []
    for p in parts:
        q = dict(p)
        if p["type"] == "capsule":
            q["from"] = R @ p["from"] + t
            q["to"] = R @ p["to"] + t
        else:
            q["pos"] = R @ p["pos"] + t
            q["yaw"] = p["yaw"] + yaw
        out.append(q)
    return out
