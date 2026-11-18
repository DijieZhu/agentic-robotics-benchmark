"""Acceptance of an evidence frame, at region scale.

The authority for whether a frame is evidence is the grader side's geometric record, mission/evidence.py
(EvidenceModel.record + mission/scoring.evidence_failures), with the thresholds of acceptance_v1.yaml as the
cell reads them (mission.config). This module does not define acceptance again:

  * EvGeom takes an instrument's geometry from mission.evidence.EvidenceModel itself on the compiled site: the face
    centre and frame, the feature size, the in-frame points (feature_points), the tag points of a lever, the sample
    points of the visible share (sample_points), the instrument's own bodies (or, for a temperature point, the
    component geom), so the point sets are the cell's by construction;
  * evaluate() is record() + evidence_failures() written for many camera poses at once: the same distance,
    off-axis, apparent size (thermal: the projected box x sqrt(visible), acceptance_v1's size rule), in-frame
    shares, in-front test and visible share (MuJoCo rays from the camera to each sample point, geom groups 0-3,
    a hit of the instrument itself or within 12 mm of the point counts as clear), with the record's own rounding,
    then the same conditions. A development test compares it with EvidenceModel.record on 10,000
    random camera poses per instrument type and requires no disagreement.
  * The ray casts mask, per 0.5 m tile of cameras, every geom that cannot meet a segment from the tile to the
    instrument (bounding spheres against a capsule); masking changes no hit (tested bit for bit).

Two regions derive from the exact test, for the slot proofs:
  generous  (camera cells; "no accepted frame from here") a set that holds every pose the robot can take there:
            only the position conditions (distance, off-axis, apparent size; thermal: 1.35 x the centred box, the
            most a pinhole camera of 50 deg can enlarge it off-centre), visibility relaxed to "one of the k + 1 key
            sample points is clear" where k is the most occluded points the exact test allows, camera heights over
            the robot's whole range, aim inside the head's tilt range widened by the body pitch and half the
            vertical field of view; no in-frame test.
  margin    (BODY cells; "an accepted frame is possible from here") the robot stands with its body centre in the
            cell facing the instrument, the camera 0.18 m ahead at the real camera height
            (0.295 m standing, 0.24 m crouching), aimed at the feature, and the exact test passes with margins:
            1.3 x the pixel size, 5 deg inside the off-axis limit, 2 deg inside the tilt limits, 0.05 m inside the
            distance limits, every sample point visible.

feature_of() keeps the analytic feature of a slot (no compiled instrument yet) for the cheap pre-checks during
placement; the proofs always use EvGeom on the compiled site.
"""

from __future__ import annotations

import math
import os
import sys

import mujoco
import numpy as np

from .config import SKETCH_DIR, acceptance_path, load_acceptance
from .util import face_quat, mat_from_quat

TAG_PLATE = (0.1908, 0.0792)          # bricks TagSpec default (60 mm marker): plate width, height
TAG_MARKER = 0.060
TAG_MARKER_DY = -0.0558               # marker centre from the plate centre along the face's +y
LEVER_LEN = 0.18
GATE_PLATE = (0.036, 0.13)
GATE_STROKE = 0.10
LEVEL_PLATE_W = 0.05
LEVEL_PLATE_Y = 0.012 + 0.016 + 0.025
LAMP_PITCH = 0.065
LAMP_LENS_D = 0.023
LAMP_PANEL_H = 0.20
EV_GROUPS = np.array([1, 1, 1, 1, 0, 0], np.uint8)      # evidence.py's occluders: geom groups 0-3
OCCLUDER_GROUPS = EV_GROUPS                              # (kept name) line-of-sight tests use the same groups
SENSOR_OF = {"gauge": "inspect", "level": "inspect", "valve_lever": "inspect", "valve_handwheel": "inspect",
             "lamps": "inspect", "thermal": "thermal"}
THERMAL_OFFCENTRE_GAIN = 1.35
TILE_M = 0.5
IN_FRAME_MARGIN_DEG = 0.7      # margin region: feature points this far inside the image border (49 px inspect, 2 px
                               # thermal)

if SKETCH_DIR not in sys.path:
    sys.path.insert(0, SKETCH_DIR)

from mission.evidence import first_hits  # noqa: E402  (the cell's ray test: one definition)


def acc_type(item) -> str:
    return {"gauge": "gauge", "level": "level", "valve_lever": "valve_lever", "valve_handwheel": "valve_handwheel",
            "lamps": "lamps", "thermal": "thermal"}[item["type"]]


def camera(cfg, name):
    """Pixel geometry of a camera (the cell's bricks CameraModel: f from the vertical field of view)."""
    c = cfg["ladder"]["cameras"][name]
    w, h, hf = int(c["width"]), int(c["height"]), float(c["hfov_deg"])
    th = math.tan(math.radians(hf) / 2)
    tv = th * h / float(w)
    try:
        from bricks.sensors.camera import CameraModel          # the cell's intrinsics, bit for bit
        K = np.asarray(CameraModel(name, w, h, hf).K, float)
    except Exception:  # noqa: BLE001
        f0 = (h / 2.0) / tv
        K = np.array([[f0, 0.0, (w - 1) / 2.0], [0.0, f0, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
    return {"name": name, "w": w, "h": h, "f": float(K[0, 0]), "tan_h": th, "tan_v": tv, "K": K}


def evidence_config(cfg) -> dict:
    """The evidence gate exactly as the cell builds it: mission.config.load_cfg() on the generator's acceptance
    file (acceptance_v1.yaml by default) -> cfg['evidence']."""
    from mission.config import load_cfg
    return load_cfg(acceptance=acceptance_path(cfg))["evidence"]


# ============================================================================================ analytic features
class Feature:
    """Geometry of what an evidence frame must show (world frame), from a slot pose (no compiled instrument)."""

    def __init__(self, typ, size, centre, normal, frame_pts, vis_pts, tag=None, aim=None, tag_pts=None):
        self.type = typ
        self.size = float(size)
        self.centre = np.asarray(centre, float)
        self.normal = np.asarray(normal, float) / max(np.linalg.norm(normal), 1e-12)
        self.frame_pts = np.asarray(frame_pts, float)
        self.vis_pts = np.asarray(vis_pts, float)
        self.aim = np.asarray(aim if aim is not None else self.frame_pts.mean(0), float)
        self.tag = tag            # {"marker": xyz, "plate": xyz, "normal": n, "ey", "ez", "side": m} or None
        self.tag_pts = None if tag_pts is None else np.asarray(tag_pts, float)

    @property
    def plate(self):
        """The tag plate (version 2: the text plate) as a plate frame, or None."""
        t = self.tag
        if t is None or "ey" not in t:
            return None
        return {"c": np.asarray(t["plate"], float), "n": np.asarray(t["normal"], float),
                "ey": np.asarray(t["ey"], float), "ez": np.asarray(t["ez"], float)}


# ============================================================================================ text plates (version 2)
# The plate standard (bricks.lettering): 190.8 x 79.2 mm, capitals 25 mm high at a pitch of 20 mm.
# Its legibility rule (acceptance_v1.yaml `plate_readable`): the cap height and the pitch at the
# plate's centre in the inspection camera, the angle off the plate's normal, the plate whole in a frame aimed at it,
# a 12 x 6 grid over it clear.
PLATE_W, PLATE_H = 0.1908, 0.0792
CAP_M, PITCH_M = 0.025, 0.020


def plate_rule(cfg):
    """The plate_readable block of the acceptance file (version 2), or None (version 1: no text plates)."""
    if not (cfg.get("ladder") or {}).get("v2"):
        return None
    return load_acceptance(cfg).get("plate_readable")


def plate_compiled(model, data, body):
    """An instrument's tag plate in the compiled site: the plate body `<body>/tag` (bricks.targets.tag: its origin is
    the centre of the plate's front face, x out, y the viewer's right, z up). None when it has none."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s/tag" % body)
    if bid < 0:
        return None
    R = np.asarray(data.xmat[bid], float).reshape(3, 3)
    return {"c": np.asarray(data.xpos[bid], float).copy(), "n": R[:, 0].copy(), "ey": R[:, 1].copy(),
            "ez": R[:, 2].copy(), "body": int(bid)}


PLATE_GRID_INSET_M = 0.0005     # acceptance_v1.yaml plate_readable.clear_grid (since 2026-11-16 03:15): edge to edge, 0.5 mm in


def plate_grid(pl, nu=12, nv=6, ahead=0.0015):
    """nu x nv points over the plate's front face, `ahead` in front of it (evidence.py's grids lie 1.5 mm in front of a
    face), placed edge to edge as acceptance_v1.yaml's clear_grid says since 2026-11-16 03:15 (y = linspace(-w/2, w/2, nu),
    z = linspace(-h/2, h/2, nv), 0.5 mm in from the edge: the border line is where thin occluders hid; before that the
    points sat at the cell centres)."""
    uu = np.linspace(-PLATE_W / 2 + PLATE_GRID_INSET_M, PLATE_W / 2 - PLATE_GRID_INSET_M, int(nu))
    vv = np.linspace(-PLATE_H / 2 + PLATE_GRID_INSET_M, PLATE_H / 2 - PLATE_GRID_INSET_M, int(nv))
    c = pl["c"] + float(ahead) * pl["n"]
    return np.asarray([c + u * pl["ey"] + v * pl["ez"] for u in uu for v in vv], float)


def plate_corners(pl):
    return np.asarray([pl["c"] + sy * PLATE_W / 2 * pl["ey"] + sz * PLATE_H / 2 * pl["ez"]
                       for sy, sz in ((-1, 1), (1, 1), (1, -1), (-1, -1))], float)


# ---------------------------------------------------------------------------------- nothing flush with a plate
PLATE_COPLANAR_M = 0.002      # since 2026-11-16: no other surface within 2 mm of a plate's face plane over its
                              # outline (a switchgear door seam flush with the plate's face ran through the lettering
                              # of lamp-panel plates on three practice buildings: no ray grid can see it)
_COPLANAR_EPS = 1e-5          # a surface exactly 2 mm behind the face (the plate's own back face, its mount) is not
                              # "within" 2 mm


def _obb_overlap(c1, R1, h1, c2, R2, h2, eps=_COPLANAR_EPS):
    """Separating-axis test of two oriented boxes (centre, rotation with the box axes as columns, half-sizes): True
    when they overlap by more than eps along every one of the 15 axes."""
    A = [R1[:, i] for i in range(3)]
    B = [R2[:, i] for i in range(3)]
    T = np.asarray(c2, float) - np.asarray(c1, float)
    axes = A + B + [np.cross(a, b) for a in A for b in B]
    for ax in axes:
        nrm = float(np.linalg.norm(ax))
        if nrm < 1e-9:
            continue
        ax = ax / nrm
        r1 = sum(float(h1[i]) * abs(float(A[i] @ ax)) for i in range(3))
        r2 = sum(float(h2[i]) * abs(float(B[i] @ ax)) for i in range(3))
        if abs(float(T @ ax)) >= r1 + r2 - eps:
            return False
    return True


def _mesh_meets_box(model, data, g, c, Rq, hq, eps=_COPLANAR_EPS):
    """Does a triangle of mesh geom g meet the box (centre c, axes Rq as columns, half-sizes hq)? Separating axes of
    triangle and box (the box's three axes, the triangle's normal, the nine edge cross products), every triangle."""
    mid = int(model.geom_dataid[g])
    if mid < 0:
        return True
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    V = np.asarray(model.mesh_vert[va:va + vn], float)
    F = np.asarray(model.mesh_face[fa:fa + fn], int)
    Rg = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
    Vw = np.asarray(data.geom_xpos[g], float)[None, :] + V @ Rg.T
    Vq = (Vw - np.asarray(c, float)[None, :]) @ np.asarray(Rq, float)          # in the box's frame
    T = Vq[F]                                                                  # (n, 3 vertices, 3)
    h = np.asarray(hq, float) - eps
    keep = np.all(T.min(1) < h[None, :], axis=1) & np.all(T.max(1) > -h[None, :], axis=1)
    T = T[keep]
    if not len(T):
        return False
    E = [T[:, 1] - T[:, 0], T[:, 2] - T[:, 1], T[:, 0] - T[:, 2]]
    axes = [np.cross(E[0], E[1])]
    for e in E:
        for i in range(3):
            u = np.zeros(3)
            u[i] = 1.0
            axes.append(np.cross(np.broadcast_to(u, e.shape), e))
    sep = np.zeros(len(T), bool)
    for ax in axes:
        nrm = np.linalg.norm(ax, axis=1)
        ok = nrm > 1e-12
        a = np.where(ok[:, None], ax / np.maximum(nrm, 1e-12)[:, None], 0.0)
        p = np.einsum("tij,tj->ti", T, a)
        r = np.abs(a) @ h
        sep |= ok & ((p.min(1) > r) | (p.max(1) < -r))
    return bool((~sep).any())


def _inside_geom(model, data, g, P):
    """Bool per point P (N, 3): inside geom g (sphere, capsule, cylinder, ellipsoid, box exactly)."""
    t = int(model.geom_type[g])
    s = np.asarray(model.geom_size[g], float)
    R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
    q = (np.asarray(P, float) - np.asarray(data.geom_xpos[g], float)[None, :]) @ R
    G = mujoco.mjtGeom
    if t == int(G.mjGEOM_SPHERE):
        return np.linalg.norm(q, axis=1) < s[0]
    if t == int(G.mjGEOM_CAPSULE):
        z = np.clip(q[:, 2], -s[1], s[1])
        return np.linalg.norm(q - np.c_[np.zeros(len(q)), np.zeros(len(q)), z], axis=1) < s[0]
    if t == int(G.mjGEOM_CYLINDER):
        return (np.hypot(q[:, 0], q[:, 1]) < s[0]) & (np.abs(q[:, 2]) < s[1])
    if t == int(G.mjGEOM_ELLIPSOID):
        return ((q / np.maximum(s[None, :], 1e-12)) ** 2).sum(1) < 1.0
    a = np.asarray(model.geom_aabb[g], float)          # box (exact) and mesh (its box in the geom frame)
    return np.all(np.abs(q - a[None, :3]) < a[None, 3:], axis=1)


def plate_coplanar(model, data, pl, own_bodies=(), tol_m=PLATE_COPLANAR_M, groups=4):
    """The coplanar test of one tag plate (a plate frame: c = centre of the front face, n out, ey, ez): the visible
    geoms (groups 0-3, not transparent) other than the plate itself whose volume reaches into the slab from tol_m
    behind to tol_m in front of the plate's face plane over the plate's outline, i.e. whose surface lies within
    tol_m of the face plane there (or which bury the plate). The plate's own body (pl["body"], `<instrument>/tag`
    from plate_compiled, and everything under it) is left out; the instrument's own geoms (own_bodies: body ids)
    count only where they reach in front of the face plane: what lies behind the plate's face (its hanger, its mount)
    is hidden by the plate. A box is tested exactly (separating axes), a sphere, capsule, cylinder or ellipsoid by
    points 2 mm apart in the slab, a mesh by its triangles. Returns [{"geom", "dmin_mm", "dmax_mm"}] (the geom's extent
    along n from the face plane), empty when clear. On the pre-model of the placement (no instrument yet) the plate
    is the analytic one (Feature.plate) and nothing is left out."""
    c = np.asarray(pl["c"], float)
    n, ey, ez = (np.asarray(pl[k], float) for k in ("n", "ey", "ez"))
    Rq = np.c_[n, ey, ez]
    hq = np.array([tol_m, PLATE_W / 2, PLATE_H / 2])
    own = set(int(b) for b in own_bodies)
    pb = int(pl.get("body", -1))
    gx = np.asarray(data.geom_xpos, float)
    rb = np.asarray(model.geom_rbound, float)
    reach = float(np.linalg.norm(hq))
    near = np.nonzero(np.linalg.norm(gx - c[None, :], axis=1) <= rb + reach + 1e-6)[0]
    G = mujoco.mjtGeom
    out = []
    # points of the slab over the outline (2 mm apart; five depths) for the round geoms
    uu = np.arange(-PLATE_W / 2, PLATE_W / 2 + 1e-9, 0.002)
    vv = np.arange(-PLATE_H / 2, PLATE_H / 2 + 1e-9, 0.002)
    aa = np.linspace(-tol_m + _COPLANAR_EPS, tol_m - _COPLANAR_EPS, 5)
    Pq = None
    for g in near:
        g = int(g)
        t = int(model.geom_type[g])
        if t in (int(G.mjGEOM_PLANE), int(G.mjGEOM_HFIELD)) or int(model.geom_group[g]) >= groups:
            continue
        mid = int(model.geom_matid[g])
        alpha = float(model.mat_rgba[mid][3]) if mid >= 0 else float(model.geom_rgba[g][3])
        if alpha <= 0.0:
            continue
        b = int(model.geom_bodyid[g])
        if pb >= 0 and _is_under(model, b, pb):
            continue                    # the plate itself
        a = np.asarray(model.geom_aabb[g], float)
        Rg = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        cg = gx[g] + Rg @ a[:3]
        ext = float(np.abs(Rg.T @ n) @ a[3:])
        d0, d1 = float((cg - c) @ n) - ext, float((cg - c) @ n) + ext
        if b in own and d1 < -_COPLANAR_EPS:
            continue                    # the instrument's own geom behind the plate's face (hanger, mount)
        if not _obb_overlap(c, Rq, hq, cg, Rg, a[3:]):
            continue
        if t == int(G.mjGEOM_MESH):
            # a mesh by its triangles (the dressing's decals and skins are single meshes spanning whole rooms)
            if not _mesh_meets_box(model, data, g, c, Rq, hq):
                continue
        elif t != int(G.mjGEOM_BOX):
            if Pq is None:
                U, V, A_ = np.meshgrid(uu, vv, aa, indexing="ij")
                Pq = c[None, :] + A_.reshape(-1, 1) * n[None, :] + U.reshape(-1, 1) * ey[None, :] + \
                    V.reshape(-1, 1) * ez[None, :]
            if not _inside_geom(model, data, g, Pq).any():
                continue
        out.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "geom%d" % g,
                    "dmin_mm": round(1000 * d0, 2), "dmax_mm": round(1000 * d1, 2)})
    return out


_SEEN_COPLANAR = {}          # plate_seen's coplanar results per (model, instrument, plate) within a process


def plate_coplanar_cached(model, data, eg, pl):
    """plate_coplanar for the plate `pl` of instrument `eg` (its own bodies: eg.own), computed once per (model,
    plate) and kept on eg."""
    key = (id(model), int(model.ngeom), float(np.asarray(data.geom_xpos).sum()),
           tuple(np.round(np.asarray(pl["c"], float), 7)), tuple(np.round(np.asarray(pl["n"], float), 7)))
    cache = getattr(eg, "_coplanar", None)
    if cache is None:
        cache = {}
        try:
            eg._coplanar = cache
        except AttributeError:
            pass
    if key not in cache:
        own = getattr(eg, "own", None)
        cache[key] = plate_coplanar(model, data, pl, own_bodies=() if own is None else [int(b) for b in own])
    return cache[key]


def subtree(model, root):
    """Body ids of `root` and every body under it."""
    root = int(root)
    return [b for b in range(model.nbody) if _is_under(model, b, root)] if root >= 0 else []


def _is_under(model, b, root):
    """Is body b the body `root` or under it?"""
    p, root = int(b), int(root)
    while p > 0:
        if p == root:
            return True
        p = int(model.body_parentid[p])
    return p == root


def plate_sizes(cfg, pl, C):
    """(cap_px, pitch_px, off_axis_deg, distance) of the plate at its centre, seen from cameras C (N, 3) of the
    inspection camera."""
    cam = camera(cfg, "inspect")
    C = np.atleast_2d(np.asarray(C, float))
    v = C - pl["c"][None, :]
    d = np.linalg.norm(v, axis=1)
    cosang = np.clip((v @ pl["n"]) / np.maximum(d, 1e-9), -1.0, 1.0)
    off = np.degrees(np.arccos(cosang))
    cap = cam["f"] * CAP_M / np.maximum(d, 1e-9)
    pitch = cam["f"] * PITCH_M * np.maximum(cosang, 0.0) / np.maximum(d, 1e-9)
    return cap, pitch, off, d


def plate_legible(acc, caster, eg, pl, C, cap_px, pitch_px, max_off, whole=True, clear=True, tilt_margin_deg=0.0):
    """Bool per camera C (N, 3) of the inspection camera aimed at the plate's centre: cap and pitch at least the
    given sizes, at most `max_off` off the plate, the aim inside the head's tilt limits, the whole plate inside the
    frame (IN_FRAME_MARGIN_DEG from the border) and every point of the plate grid clear (the instrument's own
    convention of the visible share: `eg.own`). With clear=True also the coplanar test (plate_coplanar_cached):
    a plate with another surface flush with its face is legible from nowhere."""
    cfg = acc.cfg
    C = np.atleast_2d(np.asarray(C, float))
    if clear and plate_coplanar_cached(caster.m, caster.d, eg, pl):
        return np.zeros(len(C), bool)
    cap, pitch, off, d = plate_sizes(cfg, pl, C)
    ok = (cap >= cap_px - 1e-9) & (pitch >= pitch_px - 1e-9) & (off <= max_off + 1e-9)
    a = pl["c"][None, :] - C
    el = np.arctan2(a[:, 2], np.maximum(np.linalg.norm(a[:, :2], axis=1), 1e-9))
    tm = math.radians(tilt_margin_deg)
    ok &= (el >= acc.tilt[0] + tm) & (el <= acc.tilt[1] - tm)
    k = np.nonzero(ok)[0]
    if whole and len(k):
        cam = acc.cams["inspect"]
        T = look_at(C[k], pl["c"])
        pm = max(1.0, float(cam["f"]) * math.tan(math.radians(IN_FRAME_MARGIN_DEG)))
        inf = _in_frame(cam["K"], np.linalg.inv(T), plate_corners(pl), cam["w"], cam["h"], margin=pm)
        ok[k[~(inf >= 1.0 - 1e-9)]] = False
        k = np.nonzero(ok)[0]
    if clear and len(k):
        g = acc.plate_rule.get("clear_grid", [12, 6]) if acc.plate_rule else [12, 6]
        clr = caster.clear(eg, C[k], pts=plate_grid(pl, int(g[0]), int(g[1]))).all(axis=1)
        ok[k[~clr]] = False
    return ok


def plate_readable_cells(acc, caster, eg, pl, B, height):
    """`plate_readable` for body cells B (N, 2): the robot facing the plate with its camera at `height`
    (the inspection camera) takes a frame aimed at the plate that has the whole plate in it, its 12 x 6 grid clear,
    cap height and pitch at least margin x the legible size, at most max_off_axis off the plate."""
    R = acc.plate_rule
    C = camera_from_body(acc.cfg, B, pl["c"], height, "inspect")
    return plate_legible(acc, caster, eg, pl, C, float(R["margin"]) * float(R["cap_px"]),
                         float(R["margin"]) * float(R["pitch_px"]), float(R["max_off_axis"]),
                         whole=bool(R.get("whole_plate_in_frame", True)), clear=True,
                         tilt_margin_deg=float(acc.margin["tilt_deg"]))


def _frame(slot):
    R = mat_from_quat(face_quat(slot["yaw"], slot.get("pitch", 0.0)))
    return np.asarray(slot["pos"], float), R[:, 0], R[:, 1], R[:, 2]


def _tag(P, ex, x, y, z):
    o = np.asarray(P(0.0, 0.0, 0.0), float)
    ey = np.asarray(P(0.0, 1.0, 0.0), float) - o
    ez = np.asarray(P(0.0, 0.0, 1.0), float) - o
    return {"marker": P(x + 0.0005, y + TAG_MARKER_DY, z), "plate": P(x, y, z), "normal": ex, "side": TAG_MARKER,
            "ey": ey, "ez": ez}


def feature_of(item) -> Feature:
    """The acceptance feature of an instrument record (slot pose + params + truth), laid out like the bricks
    builders that ip_models uses (face frame: +x toward the viewer, +y the viewer's right, +z up)."""
    s = item["slot"]
    prm = item.get("params") or {}
    tr = item.get("truth") or {}
    t = item["type"]
    o, ex, ey, ez = _frame(s)

    def P(x, y, z):
        return o + x * ex + y * ey + z * ez
    tw, th = TAG_PLATE
    if t == "gauge":
        D = float(prm.get("case_mm", 100)) / 1000.0
        Rd = 0.45 * D
        ang = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        rim = [P(0.0, Rd * math.cos(a), Rd * math.sin(a)) for a in ang]
        vis = [P(0.0015, 0, 0)] + [P(0.0015, f * Rd * math.cos(a), f * Rd * math.sin(a))
                                   for f in (0.25, 0.5, 0.75, 0.92) for a in np.linspace(0, 2 * math.pi, 8,
                                                                                          endpoint=False)]
        tag = _tag(P, ex, -0.17 * D + 0.015, 0.0, -0.5 * D - 0.015 - th / 2)
        return Feature("gauge", 0.9 * D, P(0, 0, 0), ex, rim, vis, tag)
    if t == "level":
        L = float(prm.get("window_m", (prm["span"][1] - prm["span"][0]) - 0.12 if "span" in prm else 0.4))
        c0 = P(0, 0, 0)
        hp = L / 2 + 0.025
        fr = [P(0, LEVEL_PLATE_Y + sy * LEVEL_PLATE_W / 2, sz * hp) for sy in (-1, 1) for sz in (-1, 1)]
        fr += [P(0, 0, -L / 2), P(0, 0, L / 2)]
        vis = [P(0.02, 0, z) for z in np.linspace(-0.4 * L, 0.4 * L, 5)]
        vis += [P(0.01, LEVEL_PLATE_Y, z) for z in (-0.4 * L, -0.13 * L, 0.13 * L, 0.4 * L)]
        tag = _tag(P, ex, 0.0, LEVEL_PLATE_Y + LEVEL_PLATE_W / 2 + 0.02 + tw / 2, -L / 2 + th / 2)
        return Feature("level", L, c0, ex, fr, vis, tag)
    if t == "valve_lever":
        rp = float(prm.get("pipe_r", 0.04)) - 0.002
        od = 2 * rp
        lx = 0.62 * od + 0.035
        ax = np.asarray(prm.get("pipe_axis", [1, 0, 0]), float)
        vert = abs(ax[2]) / max(np.linalg.norm(ax), 1e-9) > 0.7
        a = math.radians(float(tr.get("angle_deg", 0.0)))
        if not vert:
            tip = (0.9 * LEVER_LEN * math.cos(a), 0.9 * LEVER_LEN * math.sin(a))      # (y, z)
            env = [(0.9 * LEVER_LEN, 0.0), (0.0, 0.9 * LEVER_LEN)]
            ty, tz = 0.0, -(1.75 * rp + 0.02 + th / 2)
        else:
            tip = (-0.9 * LEVER_LEN * math.sin(a), 0.9 * LEVER_LEN * math.cos(a))
            env = [(0.0, 0.9 * LEVER_LEN), (-0.9 * LEVER_LEN, 0.0)]
            ty, tz = 1.75 * rp + 0.03 + tw / 2, -(0.0135 + 0.01 + th / 2)
        tx = 1.3 * rp
        fr = [P(lx, 0, 0)] + [P(lx, y, z) for y, z in env] + [P(lx, tip[0], tip[1])]
        tp = [P(tx, ty + sy * tw / 2, tz + sz * th / 2) for sy in (-1, 1) for sz in (-1, 1)]
        vis = [P(lx + 0.02, f * tip[0], f * tip[1]) for f in (0.15, 0.4, 0.65, 0.9)]
        vis += [P(tx + 0.01, ty + sy * 0.4 * tw, tz + sz * 0.3 * th) for sy in (-1, 1) for sz in (-1, 1)]
        vis += [P(tx + 0.01, ty, tz)]
        tag = _tag(P, ex, tx, ty, tz)
        return Feature("valve_lever", LEVER_LEN, P(lx, 0, 0), ex, fr, vis, tag,
                       aim=np.mean([P(lx, 0, 0), P(tx, ty, tz)], 0), tag_pts=tp)
    if t == "valve_handwheel":
        o2 = o + 0.42 * ez                              # ip_models builds the gate valve with its plate 0.42 m up

        def Q(x, y, z):
            return o2 + x * ex + y * ey + z * ez
        pw, ph = GATE_PLATE
        fr = [Q(0, sy * pw / 2, sz * ph / 2) for sy in (-1, 1) for sz in (-1, 1)]
        vis = [Q(0.012, sy * 0.35 * pw, z) for sy in (-1, 1) for z in (-0.045, 0.0, 0.045)]
        vis += [Q(0.012, 0, -0.03), Q(0.012, 0, 0.03), Q(0.012, 0, 0.0)]
        tag = _tag(lambda x, y, z: Q(x, y, z), ex, 0.0, pw / 2 + 0.02 + tw / 2, 0.0)
        return Feature("valve_handwheel", GATE_STROKE, Q(0, 0, 0), ex, fr, vis, tag)
    if t == "lamps":
        n = len(prm.get("colors", [])) or 4
        w = n * LAMP_PITCH + 0.05
        zl = 0.025
        lens = [P(0.0095, (k - (n - 1) / 2) * LAMP_PITCH, zl) for k in range(n)]
        fr = [P(0, sy * w / 2, -0.005 + sz * LAMP_PANEL_H / 2) for sy in (-1, 1) for sz in (-1, 1)]
        ks = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(n, 5))})
        vis = [P(0.02, (k - (n - 1) / 2) * LAMP_PITCH, zl) for k in ks]
        vis += [P(0.01, sy * 0.4 * w, -0.005 + sz * 0.4 * LAMP_PANEL_H) for sy in (-1, 1) for sz in (-1, 1)]
        tag = _tag(P, ex, 0.003, 0.0, zl - 0.03 - LAMP_PANEL_H / 2 - 0.012 - th / 2)
        return Feature("lamps", LAMP_LENS_D, P(0, 0, zl), ex, fr + lens, vis, tag)
    if t == "thermal":
        fw, fh = float(prm.get("face_w", 0.3)), float(prm.get("face_h", 0.3))
        fr = [P(0, sy * fw / 2, sz * fh / 2) for sy in (-1, 1) for sz in (-1, 1)] + [P(0, 0, 0)]
        vis = [P(0.02, sy * 0.4 * fw, sz * 0.4 * fh) for sy in (-1, 1) for sz in (-1, 1)]
        vis += [P(0.02, 0, 0), P(0.02, 0.4 * fw, 0), P(0.02, -0.4 * fw, 0), P(0.02, 0, 0.4 * fh),
                P(0.02, 0, -0.4 * fh)]
        tg = prm.get("tag_offset")
        tag = _tag(P, ex, 0.0, tg[0], tg[1]) if tg else None
        return Feature("thermal", min(fw, fh), P(0, 0, 0), ex, fr, vis, tag)
    raise KeyError(t)


# ============================================================================================ evidence geometry
class EvGeom:
    """What the evidence test needs of one instrument (world frame).

    type, tag, sensor        the acceptance file's type; the camera the type needs
    c, R, n                  face centre, face frame (None for a component without a face normal), normal R[:, 0]
    size                     nominal feature size (m) of the apparent-size rule (non-thermal)
    fpts, tpts, spts         in-frame points, the lever's tag points (or None), visible-share sample points
    own                      body ids that count as the instrument itself in the visible share (non-thermal)
    own_geom                 the component geom that counts as itself (thermal)
    aim                      where the margin region aims the camera (centre of the in-frame points' box)
    source                   'evidence' (EvidenceModel on the compiled site) or 'analytic' (feature_of)
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def radius(self) -> float:
        P = [self.spts, self.fpts] + ([self.tpts] if self.tpts is not None else [])
        P = np.vstack([p for p in P if p is not None and len(p)])
        return float(np.linalg.norm(P - self.centre_all[None, :], axis=1).max()) if len(P) else 0.1

    @property
    def centre_all(self) -> np.ndarray:
        P = [self.spts, self.fpts] + ([self.tpts] if self.tpts is not None else [])
        P = np.vstack([p for p in P if p is not None and len(p)])
        return 0.5 * (P.min(0) + P.max(0))


def _aim_of(fpts, tpts):
    P = fpts if tpts is None else np.vstack([fpts, tpts])
    return 0.5 * (P.min(0) + P.max(0))


def evidence_model(model, data, records, ev_cfg, exclude_body=-1):
    """mission.evidence.EvidenceModel over a compiled site (records: site.json ips / decoys format)."""
    from mission.evidence import EvidenceModel
    return EvidenceModel(model, data, {"ips": list(records)}, {"evidence": ev_cfg}, exclude_body=exclude_body)


GRID_AHEAD_M = 0.0015                  # evidence.py's sample grids lie this far in front of a face
# denser grids over the plates the readers need whole (nothing in front of a level gauge's
# scale plate or a lever valve's tag plate from the accepted reading places): the margin region requires every point
# of them clear, besides every evidence sample point. (nu along the face's longer side, nv across it)
PLATE_GRID = {"level": ("/plate", 30, 4), "valve_lever": ("/tag/plate", 12, 6)}


def face_grid(model, data, gid, n, nu, nv, ahead=GRID_AHEAD_M) -> np.ndarray:
    """nu x nv points at the cell centres of a box geom's face whose outward normal is closest to n, `ahead` in front
    of it: mission/evidence.py EvidenceModel._face_grid written again (a development test checks that both give the
    same points at evidence.py's own grid sizes)."""
    if gid < 0 or int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.zeros((0, 3))
    s = np.asarray(model.geom_size[gid], float)
    p = np.asarray(data.geom_xpos[gid], float)
    Rg = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
    n = np.asarray(n, float)
    a, sg = max(((a, sg) for a in range(3) for sg in (-1.0, 1.0)), key=lambda k: k[1] * float(Rg[:, k[0]] @ n))
    o = sorted((i for i in range(3) if i != a), key=lambda i: -s[i])
    fc = p + sg * s[a] * Rg[:, a] + ahead * sg * Rg[:, a]
    uu = (2.0 * np.arange(nu) + 1.0) / nu - 1.0
    vv = (2.0 * np.arange(nv) + 1.0) / nv - 1.0
    return np.asarray([fc + u * s[o[0]] * Rg[:, o[0]] + v * s[o[1]] * Rg[:, o[1]] for u in uu for v in vv], float)


def plate_points(model, data, rec, n) -> np.ndarray | None:
    """The dense plate grid (PLATE_GRID) of an EvidenceModel record, or None for the other types."""
    spec = PLATE_GRID.get(rec["type"])
    if spec is None:
        return None
    suffix, nu, nv = spec
    names = rec.get("tag_geoms") if rec["type"] == "valve_lever" else rec.get("face_geoms")
    pts = [face_grid(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g), n, nu, nv)
           for g in (names or []) if g.endswith(suffix)]
    pts = [p for p in pts if len(p)]
    return np.vstack(pts) if pts else None


def thermal_hot_geom(model, ip) -> str | None:
    """The hot face of a temperature point whose truth has no unit list, as the cell heats it (mission/world.py: the
    truth's `geom`, else `params.hot_geom`), when the compiled model has that geom; else None. The ladder's decoy at a
    thermal C-moved item's recorded place is such a point (truth.temp_C on the station's hot face, no units)."""
    tr = ip.get("truth") or {}
    hg = tr.get("geom") or (ip.get("params") or {}).get("hot_geom")
    if hg and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, hg) >= 0:
        return hg
    return None


def evgeoms(model, data, records, ev_cfg) -> dict:
    """{tag: EvGeom} of the instruments among `records`, from the cell's EvidenceModel on the compiled site.

    A temperature point without a unit list (the decoy at a thermal C-moved item's recorded place) has no hot geom in
    the EvidenceModel's record; its hot face is taken as the cell heats it (`thermal_hot_geom`) and passed to the
    EvidenceModel's face, feature and sample points (earlier, on such decoys this raised "zero-size array",
    on two development sites). Listed instruments carry their units: unchanged. An instrument with no
    feature point at all raises ValueError naming it."""
    em = evidence_model(model, data, records, ev_cfg)
    out = {}
    body_of = {r.get("tag"): r.get("body") for r in records}
    for tag, rec in em.items.items():
        typ = rec["type"]
        hg = None
        own_geom = None
        if typ == "thermal":
            hg = rec.get("hot_geom") or thermal_hot_geom(model, rec["ip"])
            own_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, hg) if hg else -1
        c, R = em.face_frame(rec, hg)
        fpts, tpts = em.feature_points(rec, hg)
        spts = em.sample_points(rec, hg)
        if not len(fpts):
            raise ValueError("evgeoms: no feature points for %s (%s): no geometry the evidence test can use" %
                             (tag, typ))
        out[tag] = EvGeom(type=typ, tag=tag, sensor=SENSOR_OF[typ], c=np.asarray(c, float),
                          R=None if R is None else np.asarray(R, float),
                          n=None if R is None else np.asarray(R, float)[:, 0].copy(),
                          size=float(rec.get("size_m", 0.0) or 0.0), fpts=np.asarray(fpts, float),
                          tpts=None if tpts is None else np.asarray(tpts, float), spts=np.asarray(spts, float),
                          own=np.asarray(rec["bodies"], int), own_geom=own_geom, aim=_aim_of(fpts, tpts),
                          rule=ev_cfg["types"][typ], source="evidence", model_ref=(model, data),
                          plate_pts=None if R is None else plate_points(model, data, rec, np.asarray(R, float)[:, 0]),
                          plate=plate_compiled(model, data, body_of[tag]) if body_of.get(tag) else None)
    return out


def evgeom_from_feature(feat: Feature, ev_cfg) -> EvGeom:
    """The analytic feature of a slot as an EvGeom (placement pre-checks on the model without the instrument)."""
    typ = feat.type
    n = feat.normal
    ey = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(ey) < 1e-9:
        ey = np.array([0.0, 1.0, 0.0])
    ey /= np.linalg.norm(ey)
    R = np.column_stack([n, -ey, np.cross(n, -ey)])
    return EvGeom(type=typ, tag="", sensor=SENSOR_OF[typ], c=feat.centre, R=R, n=n, size=feat.size,
                  fpts=feat.frame_pts, tpts=feat.tag_pts, spts=feat.vis_pts, own=np.zeros(0, int), own_geom=None,
                  aim=feat.aim if typ != "thermal" else feat.centre, rule=ev_cfg["types"][typ], source="analytic",
                  model_ref=None, plate=feat.plate)


# ============================================================================================ visibility
class RayCaster:
    """evidence.py's visible share for many cameras: rays from each camera to the sample points (geom groups 0-3,
    static geoms included); clear when nothing is hit, the hit is at or beyond the point less 12 mm, or the hit geom
    is the instrument's own. Geoms that cannot meet any segment from a tile of cameras are masked by group."""

    def __init__(self, model, data, exclude_body=-1):
        self.m, self.d = model, data
        self.exclude_body = int(exclude_body)
        self.grp0 = np.asarray(model.geom_group).copy()
        self.always = np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE)
        self.n_calls = 0

    def clear(self, eg: EvGeom, C, pts=None, tile=TILE_M):
        """(N, k) bool: point k is clear from camera n."""
        m, d = self.m, self.d
        P = eg.spts if pts is None else np.asarray(pts, float)
        C = np.atleast_2d(np.asarray(C, float))
        N, k = len(C), len(P)
        out = np.zeros((N, k), bool)
        if N == 0 or k == 0:
            return out
        gx = np.asarray(d.geom_xpos)
        gr = np.asarray(m.geom_rbound)
        ctr = 0.5 * (P.min(0) + P.max(0))
        rho = float(np.linalg.norm(P - ctr[None, :], axis=1).max())
        key = np.floor(C[:, :2] / tile).astype(np.int64)
        order = np.lexsort((key[:, 1], key[:, 0]))
        ks = key[order]
        brk = np.nonzero(np.any(np.diff(ks, axis=0) != 0, axis=1))[0] + 1
        groups = np.split(order, brk)
        gid = np.zeros(k, np.int32)
        hit = np.zeros(k, np.float64)
        own_bodies = eg.own
        try:
            for idx in groups:
                Ct = C[idx]
                lo, hi = Ct.min(0), Ct.max(0)
                tc = 0.5 * (lo + hi)
                rt = 0.5 * float(np.linalg.norm(hi - lo))
                ab = ctr - tc
                L2 = float(ab @ ab)
                w = gx - tc[None, :]
                lam = np.clip((w @ ab) / max(L2, 1e-12), 0.0, 1.0)
                dd = np.linalg.norm(w - lam[:, None] * ab[None, :], axis=1)
                keep = (dd <= gr + max(rt, rho) + 1e-6) | self.always
                m.geom_group[:] = np.where(keep, self.grp0, 5)
                for j, i in enumerate(idx):
                    vec = P - C[i][None, :]
                    dist = np.linalg.norm(vec, axis=1)
                    vec = vec / np.maximum(dist[:, None], 1e-9)
                    # the cell's own ray test (mission/evidence.py first_hits: mj_ray once per ray; mj_multiRay's
                    # culling misses geoms of static bodies holding several geoms in MuJoCo 3.14: found
                    # when the cell was integrated)
                    gid[:], hit[:] = first_hits(m, d, C[i], vec, EV_GROUPS, self.exclude_body)
                    self.n_calls += 1
                    if eg.type == "thermal":
                        own = (gid == eg.own_geom) if eg.own_geom is not None else np.zeros(k, bool)
                    else:
                        own = np.isin(m.geom_bodyid[np.maximum(gid, 0)], own_bodies) & (gid >= 0)
                    out[i] = (gid < 0) | (hit < 0) | (hit >= dist - 0.012) | own
        finally:
            m.geom_group[:] = self.grp0
        return out


# ============================================================================================ the exact test
_pyround = np.frompyfunc(lambda x, n: round(float(x), int(n)), 2, 1)


def _r(a, n):
    """Python's round() elementwise (EvidenceModel.record rounds with it before deciding)."""
    return np.asarray(_pyround(np.asarray(a, float), n), float)


def look_at(C, A):
    """(N, 4, 4) optical-frame poses (x right, y down, z forward) at C looking at A, upright: mission.truth's
    look_at_optical for many cameras."""
    C = np.atleast_2d(np.asarray(C, float))
    A = np.broadcast_to(np.asarray(A, float), C.shape)
    z = A - C
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    x = np.cross(z, np.array([0.0, 0.0, 1.0])[None, :])
    nx = np.linalg.norm(x, axis=1, keepdims=True)
    bad = nx[:, 0] < 1e-9
    x = np.where(bad[:, None], np.array([[1.0, 0.0, 0.0]]), x / np.maximum(nx, 1e-12))
    y = np.cross(z, x)
    T = np.zeros((len(C), 4, 4))
    T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = x, y, z, C
    T[:, 3, 3] = 1.0
    return T


def _project(K, Tinv, pts):
    """u, v, z (N, k) of world points pts (k, 3) in the cameras with inverse poses Tinv (N, 4, 4)."""
    pc = np.einsum("nij,kj->nki", Tinv[:, :3, :3], pts) + Tinv[:, None, :3, 3]
    z = pc[..., 2]
    u = K[0, 0] * pc[..., 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * pc[..., 1] / np.maximum(z, 1e-9) + K[1, 2]
    return u, v, z


def _in_frame(K, Tinv, pts, w, h, margin=1.0):
    if pts is None or len(pts) == 0:
        return np.full(len(Tinv), np.nan)
    u, v, z = _project(K, Tinv, pts)
    ok = (z > 1e-3) & (u >= margin) & (u <= w - 1 - margin) & (v >= margin) & (v <= h - 1 - margin)
    return ok.mean(axis=1)


class Acceptance:
    """The evidence gate of the cell (acceptance_v1.yaml via mission.config) plus the ladder's margins and the
    robot's head model."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.A = load_acceptance(cfg)
        self.ev = evidence_config(cfg)
        L = cfg["ladder"]
        self.cams = {k: camera(cfg, k) for k in L["cameras"]}
        H = L["head"]
        self.H = H
        self.tilt = tuple(math.radians(float(v)) for v in H["tilt_deg"])
        self.margin = L["margin"]
        self.max_dist = float(self.ev["common"]["max_distance_m"])
        self.plate_rule = plate_rule(cfg)          # version 2: the text plate's legibility (None in version 1)

    def type_rule(self, typ):
        """The acceptance file's entry of a type (raw yaml: feature_px, max_off_axis, min_dist, ...)."""
        return self.A["types"][typ]

    def ev_rule(self, typ):
        return self.ev["types"][typ]

    # ------------------------------------------------------------------ the record, many poses at once
    def evaluate(self, eg: EvGeom, T, caster: RayCaster, sensor=None, vis=None):
        """EvidenceModel.record() + evidence_failures() for camera poses T (N, 4, 4) of `sensor` (default the
        type's camera). Returns a dict of arrays: distance_m, off_axis_deg, app_px, in_frame, tag_in_frame (lever),
        in_front, visible, ok, and fail flags per condition. `vis` (N, k) may pass precomputed clear flags."""
        T = np.asarray(T, float)
        if T.ndim == 2:
            T = T[None]
        N = len(T)
        sensor = sensor or eg.sensor
        cam = self.cams[sensor]
        K, w, h = cam["K"], cam["w"], cam["h"]
        rule = eg.rule
        com = self.ev["common"]
        C = T[:, :3, 3]
        to_cam = C - eg.c[None, :]
        dist = np.linalg.norm(to_cam, axis=1)
        out = {"distance_m": _r(dist, 4)}
        if eg.R is not None:
            off = np.degrees(np.arccos(np.clip((to_cam @ eg.n) / np.maximum(dist, 1e-9), -1.0, 1.0)))
        else:
            off = np.zeros(N)
        out["off_axis_deg"] = _r(off, 2)
        Tinv = np.linalg.inv(T)
        out["in_frame"] = _r(_in_frame(K, Tinv, eg.fpts, w, h), 4)
        if eg.tpts is not None:
            out["tag_in_frame"] = _r(_in_frame(K, Tinv, eg.tpts, w, h), 4)
        if eg.type == "thermal":
            u, v, z = _project(K, Tinv, eg.fpts)
            okz = z > 1e-3
            uc = np.clip(u, 0, w - 1)
            vc = np.clip(v, 0, h - 1)
            any_ = okz.any(axis=1)
            bw = np.where(any_, np.where(okz, uc, -np.inf).max(1) - np.where(okz, uc, np.inf).min(1), 0.0)
            bh = np.where(any_, np.where(okz, vc, -np.inf).max(1) - np.where(okz, vc, np.inf).min(1), 0.0)
            app = _r(np.minimum(bw, bh), 2)
        else:
            app = _r(float(K[0, 0]) * eg.size / np.maximum(dist, 1e-9), 2)
        _, _, z0 = _project(K, Tinv, eg.c[None, :])
        out["in_front"] = z0[:, 0] > 0
        if vis is None:
            vis = caster.clear(eg, C)
        vshare = vis.mean(axis=1) if vis.shape[1] else np.full(N, np.nan)
        out["visible"] = _r(vshare, 4)
        if eg.type == "thermal":
            app = _r(app * np.sqrt(np.maximum(out["visible"], 0.0)), 2)
        out["app_px"] = app
        # evidence_failures, condition by condition
        fl = {}
        fl["sensor"] = np.full(N, sensor != rule["sensor"])
        fl["behind"] = ~out["in_front"]
        fl["size"] = out["app_px"] < float(rule["min_px"])
        angle_applies = not (eg.type == "thermal" and rule.get("off_axis_needs_face_normal", True) and eg.R is None)
        fl["off_axis"] = (out["off_axis_deg"] > float(rule["max_off_axis_deg"])) if angle_applies else np.zeros(N, bool)
        inf = out["in_frame"]
        fl["in_frame"] = ~(np.isfinite(inf) & (inf >= float(rule["min_in_frame"]) - 1e-9))
        if "min_tag_in_frame" in rule:
            tif = out.get("tag_in_frame", np.zeros(N))
            fl["tag_in_frame"] = ~(np.nan_to_num(tif, nan=-1.0) >= float(rule["min_tag_in_frame"]) - 1e-9)
        fl["far"] = out["distance_m"] > float(com["max_distance_m"])
        dmin = float(rule.get("min_distance_m", 0.0))
        fl["near"] = (out["distance_m"] < dmin) if dmin > 0.0 else np.zeros(N, bool)
        vmin = float(rule.get("min_visible_share", com.get("min_visible_share", 0.9)))
        vv = out["visible"]
        fl["visible"] = ~(np.isfinite(vv) & (vv >= vmin))
        bad = np.zeros(N, bool)
        for v_ in fl.values():
            bad |= v_
        out["fail"] = fl
        out["ok"] = ~bad
        return out

    # ------------------------------------------------------------------ allowed occlusion
    def allowed_occluded(self, eg: EvGeom) -> int:
        """The most sample points that can be occluded in an accepted frame (the record's rounding included)."""
        S = len(eg.spts)
        rule = eg.rule
        vmin = float(rule.get("min_visible_share", self.ev["common"].get("min_visible_share", 0.9)))
        k = 0
        while k + 1 <= S and round((S - (k + 1)) / S, 4) >= vmin:
            k += 1
        return k

    def key_points(self, eg: EvGeom) -> np.ndarray:
        """allowed_occluded + 1 sample points spread over the set (farthest-point order from the aim): in an
        accepted frame at least one of them is clear."""
        S = eg.spts
        need = min(len(S), self.allowed_occluded(eg) + 1)
        chosen = [int(np.argmin(np.linalg.norm(S - eg.aim[None, :], axis=1)))]
        d = np.linalg.norm(S - S[chosen[0]][None, :], axis=1)
        while len(chosen) < need:
            j = int(np.argmax(d))
            if d[j] <= 0.0:
                break
            chosen.append(j)
            d = np.minimum(d, np.linalg.norm(S - S[j][None, :], axis=1))
        idx = sorted(set(chosen))
        if len(idx) < need:                       # repeated sample points: every point then
            return S
        return S[idx]


# ============================================================================================ head model
def nominal_head(cfg, which):
    """Camera height (m) of the robot standing ('stand') or crouching ('crouch')."""
    H = cfg["ladder"]["head"]
    v = H.get("%s_nominal" % which)
    if v is None:
        v = cfg["modes"]["walk" if which == "stand" else "crouch_walk"]["head_height"]
    return float(v)


def heights(cfg, which):
    """Camera heights: the nominal one ('stand' / 'crouch') or a sampling of the robot's whole range ('range')."""
    H = cfg["ladder"]["head"]
    if which in ("stand", "crouch"):
        return [nominal_head(cfg, which)]
    lo, hi = [float(v) for v in H["camera_z_range"]]
    n = int(H.get("range_samples", 7))
    return [float(v) for v in np.linspace(lo, hi, max(2, n))]


def camera_from_body(cfg, B, aim, height, sensor):
    """Camera positions (N, 3) of a robot whose body centre stands at B (N, 2) facing `aim` (3,), camera tilt
    axis at `height` (m): the pan axis mast_x ahead of the body centre, the camera cam_fwd ahead of the tilt axis
    along the line of sight and cam_lat to the side (the inspection camera on the right, the thermal on the left)."""
    H = cfg["ladder"]["head"]
    B = np.atleast_2d(np.asarray(B, float))[:, :2]
    a = np.asarray(aim, float)
    u = a[None, :2] - B
    du = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / np.maximum(du, 1e-9)
    perp = np.c_[-u[:, 1], u[:, 0]]
    mx, fw = float(H["mast_x"]), float(H["cam_fwd"])
    lat = float(H["cam_lat"][sensor])
    horiz = np.maximum(du[:, 0] - mx, 1e-6)
    el = np.arctan2(a[2] - height, horiz)
    xy = B + (mx + fw * np.cos(el))[:, None] * u + lat * perp
    z = height + fw * np.sin(el)
    return np.c_[xy, z]


# ============================================================================================ regions
def _cells_near(grid, cells, centre, r):
    X, Y = grid.centers()
    return cells & ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= r * r)


def generous_region(acc: Acceptance, caster: RayCaster, grid, eg: EvGeom, cells, hs=None, early_any=False):
    """Camera cells of `cells` from which the robot could take an accepted frame (a set that holds the exact test's cells over
    every pose: see the module doc). early_any: stop at the first accepted cell (an exclusion test)."""
    cfg = acc.cfg
    hs = hs if hs is not None else heights(cfg, "range")
    out = np.zeros(grid.shape, bool)
    rule = eg.rule
    com = acc.ev["common"]
    sel = _cells_near(grid, cells, eg.c, acc.max_dist + 0.05)
    iy, ix = np.nonzero(sel)
    if not len(iy):
        return out
    X, Y = grid.centers()
    cam = acc.cams[eg.sensor]
    H = acc.H
    # aim limits: head tilt widened by the body pitch and half the vertical field of view (+2 deg)
    pitch = math.radians(float(H.get("body_pitch_deg", 14.3)))
    half_v = math.atan(cam["tan_v"])
    e_lo = acc.tilt[0] - pitch - half_v - math.radians(2.0)
    e_hi = acc.tilt[1] + pitch + half_v + math.radians(2.0)
    keys = acc.key_points(eg)
    dmin = float(rule.get("min_distance_m", 0.0))
    for h in hs:
        C = np.stack([X[iy, ix], Y[iy, ix], np.full(len(iy), h)], 1)
        v = C - eg.c[None, :]
        d = np.linalg.norm(v, axis=1)
        ok = (d <= float(com["max_distance_m"]) + 1e-4) & ((d >= dmin - 1e-4) if dmin > 0 else True)
        if eg.n is not None:                      # a thermal component without a face normal has no angle rule
            off = np.degrees(np.arccos(np.clip((v @ eg.n) / np.maximum(d, 1e-9), -1, 1)))
            ok &= off <= float(rule["max_off_axis_deg"]) + 0.01
        a = eg.aim[None, :] - C
        el = np.arctan2(a[:, 2], np.maximum(np.linalg.norm(a[:, :2], axis=1), 1e-9))
        ok &= (el >= e_lo) & (el <= e_hi)
        if eg.type != "thermal":
            ok &= cam["f"] * eg.size / np.maximum(d, 1e-9) >= float(rule["min_px"]) - 0.01
        ok &= ~out[iy, ix]
        k = np.nonzero(ok)[0]
        if not len(k):
            continue
        if eg.type == "thermal":
            T = look_at(C[k], eg.c)
            Tinv = np.linalg.inv(T)
            u, vv, z = _project(cam["K"], Tinv, eg.fpts)
            okz = z > 1e-3
            bw = np.where(okz, u, -np.inf).max(1) - np.where(okz, u, np.inf).min(1)
            bh = np.where(okz, vv, -np.inf).max(1) - np.where(okz, vv, np.inf).min(1)
            box = np.where(okz.any(1), np.minimum(bw, bh), 0.0) * THERMAL_OFFCENTRE_GAIN
            # visible share needed for the size: box x sqrt(share) >= min_px - 0.01
            need_share = np.clip((float(rule["min_px"]) - 0.01) / np.maximum(box, 1e-9), 0, 10.0) ** 2
            clr = caster.clear(eg, C[k])
            share = clr.mean(1)
            vmin = float(rule.get("min_visible_share", com.get("min_visible_share", 0.9)))
            good = (share + 1e-4 >= np.maximum(need_share, vmin))
        else:
            clr = caster.clear(eg, C[k], pts=keys)
            good = clr.any(axis=1)
        out[iy[k[good]], ix[k[good]]] = True
        if early_any and out.any():
            return out
    return out


def margin_region(acc: Acceptance, caster: RayCaster, grid, eg: EvGeom, body_cells, height, return_poses=False):
    """BODY cells of `body_cells` from which the robot, facing the instrument with its camera at `height`, takes an
    accepted frame with the margins of the module doc (exact test, all sample points visible)."""
    cfg = acc.cfg
    out = np.zeros(grid.shape, bool)
    rule = eg.rule
    com = acc.ev["common"]
    Mg = acc.margin
    sel = _cells_near(grid, body_cells, eg.c, acc.max_dist + 0.3)
    iy, ix = np.nonzero(sel)
    if not len(iy):
        return (out, None) if return_poses else out
    X, Y = grid.centers()
    B = np.stack([X[iy, ix], Y[iy, ix]], 1)
    C = camera_from_body(cfg, B, eg.aim, height, eg.sensor)
    v = C - eg.c[None, :]
    d = np.linalg.norm(v, axis=1)
    dm = float(Mg.get("dist_m", 0.05))
    dmin = float(rule.get("min_distance_m", 0.0))
    ok = (d <= float(com["max_distance_m"]) - dm) & (d >= dmin + dm)
    angle_applies = not (eg.type == "thermal" and rule.get("off_axis_needs_face_normal", True) and eg.R is None)
    if angle_applies:
        off = np.degrees(np.arccos(np.clip((v @ eg.n) / np.maximum(d, 1e-9), -1, 1)))
        ok &= off <= float(rule["max_off_axis_deg"]) - float(Mg["angle_deg"])
    a = eg.aim[None, :] - C
    el = np.arctan2(a[:, 2], np.maximum(np.linalg.norm(a[:, :2], axis=1), 1e-9))
    tm = math.radians(float(Mg["tilt_deg"]))
    ok &= (el >= acc.tilt[0] + tm) & (el <= acc.tilt[1] - tm)
    cam = acc.cams[eg.sensor]
    if eg.type != "thermal":
        ok &= cam["f"] * eg.size / np.maximum(d, 1e-9) >= float(rule["min_px"]) * float(Mg["size_factor"])
    k = np.nonzero(ok)[0]
    if not len(k):
        return (out, None) if return_poses else out
    T = look_at(C[k], eg.aim)
    # the cheapest test first: the in-frame shares at the aimed pose, with a margin of IN_FRAME_MARGIN_DEG inside the
    # image border (found when the cell was integrated: a real frame from the nearest lever cells lost a
    # corner of the tag plate; the record's own test uses 1 px)
    Tinv = np.linalg.inv(T)
    pm = max(1.0, float(cam["f"]) * math.tan(math.radians(IN_FRAME_MARGIN_DEG)))
    inf = _r(_in_frame(cam["K"], Tinv, eg.fpts, cam["w"], cam["h"], margin=pm), 4)
    keep = np.isfinite(inf) & (inf >= float(rule["min_in_frame"]) - 1e-9)
    if eg.tpts is not None and "min_tag_in_frame" in rule:
        tif = _r(_in_frame(cam["K"], Tinv, eg.tpts, cam["w"], cam["h"], margin=pm), 4)
        keep &= tif >= float(rule["min_tag_in_frame"]) - 1e-9
    k, T = k[keep], T[keep]
    if not len(k):
        return (out, None) if return_poses else out
    clr = caster.clear(eg, C[k])
    allvis = clr.all(axis=1)
    k, T, clr = k[allvis], T[allvis], clr[allvis]
    if not len(k):
        return (out, None) if return_poses else out
    pp = getattr(eg, "plate_pts", None)
    if pp is not None and len(pp):
        # nothing in front of the scale plate / the tag plate (PLATE_GRID): every point of the dense grid clear
        whole = caster.clear(eg, C[k], pts=pp).all(axis=1)
        k, T, clr = k[whole], T[whole], clr[whole]
        if not len(k):
            return (out, None) if return_poses else out
    rec = acc.evaluate(eg, T, caster, vis=clr)
    good = rec["ok"].copy()
    if eg.type == "thermal":
        good &= rec["app_px"] >= float(rule["min_px"]) * float(Mg["size_factor"])
    pl = getattr(eg, "plate", None)
    if acc.plate_rule is not None and pl is not None and good.any():
        # version 2 (plate_readable): from the same body cell a frame aimed at the instrument's text plate is
        # legible with the margin, so that every *_observable row means "record photo and plate"
        g2 = np.nonzero(good)[0]
        good[g2] = plate_readable_cells(acc, caster, eg, pl, B[k[g2]], height)
    out[iy[k[good]], ix[k[good]]] = True
    if return_poses:
        return out, {"cells": np.c_[iy[k[good]], ix[k[good]]], "cams": C[k[good]], "T": T[good]}
    return out


# ============================================================================================ line of sight, tags
def los(model, data, P, C, bodyexclude=-1, slack=0.04):
    """Line of sight from point P to each camera position C (N, 3) (geom groups 0-3)."""
    C = np.atleast_2d(np.asarray(C, float))
    n = len(C)
    if n == 0:
        return np.zeros(0, bool)
    v = C - np.asarray(P, float)[None, :]
    d = np.linalg.norm(v, axis=1)
    vec = (v / np.maximum(d[:, None], 1e-12)).astype(np.float64)
    gid, dist = first_hits(model, data, P, vec, OCCLUDER_GROUPS, int(bodyexclude))     # the exact ray test
    return (gid < 0) | (dist < 0) | (dist >= d - slack)


def tag_seen(cfg, model, data, tag, C, bodyexclude=-1):
    """Bool per camera position: the tag's marker is decodable in an inspect frame (>= tag_px pixels on its
    foreshortened side, off-axis <= max_off_deg, inside the tilt limits, line of sight)."""
    F = cfg["ladder"]["findable"]
    cam = camera(cfg, "inspect")
    C = np.atleast_2d(np.asarray(C, float))
    M = np.asarray(tag["marker"], float)
    n = np.asarray(tag["normal"], float)
    v = C - M[None, :]
    d = np.linalg.norm(v, axis=1)
    cosang = (v @ n) / np.maximum(d, 1e-9)
    ok = cosang >= math.cos(math.radians(float(F["max_off_deg"])))
    ok &= cam["f"] * float(tag["side"]) * np.maximum(cosang, 0.0) / np.maximum(d, 1e-9) >= float(F["tag_px"])
    tl = [math.radians(float(x)) for x in cfg["ladder"]["head"]["tilt_deg"]]
    el = np.arctan2(-v[:, 2], np.maximum(np.linalg.norm(v[:, :2], axis=1), 1e-9))
    ok &= (el >= tl[0]) & (el <= tl[1])
    if ok.any():
        k = np.nonzero(ok)[0]
        ok[k] = los(model, data, M + 0.004 * n, C[k], bodyexclude)
    return ok


def plate_seen(cfg, model, data, pl, C, bodyexclude=-1, grid=(12, 6)):
    """Version 2 (findable, search_budget): bool per camera position C (N, 3): the text plate is legible in an
    inspection frame aimed at it (cap >= cap_px, pitch >= pitch_px at the plate's centre, at most findable_off_axis
    off the plate, the aim inside the tilt limits, every point of its grid in line of sight), and the coplanar test
    (nothing flush with its face over its outline; the instrument's own body `bodyexclude` and what is under it count
    only in front of the face)."""
    R = load_acceptance(cfg)["plate_readable"]
    C = np.atleast_2d(np.asarray(C, float))
    key = (id(model), int(model.ngeom), int(model.nbody), float(np.asarray(data.geom_xpos).sum()), int(bodyexclude),
           tuple(np.round(np.asarray(pl["c"], float), 7)))       # a model id can be reused: its geometry's sum too
    if key not in _SEEN_COPLANAR:
        if len(_SEEN_COPLANAR) > 4096:
            _SEEN_COPLANAR.clear()
        _SEEN_COPLANAR[key] = bool(plate_coplanar(model, data, pl, own_bodies=subtree(model, bodyexclude)
                                                  if bodyexclude >= 0 else ()))
    if _SEEN_COPLANAR[key]:
        return np.zeros(len(C), bool)
    cap, pitch, off, d = plate_sizes(cfg, pl, C)
    ok = (cap >= float(R["cap_px"]) - 1e-9) & (pitch >= float(R["pitch_px"]) - 1e-9) & \
        (off <= float(R["findable_off_axis"]) + 1e-9)
    tl = [math.radians(float(x)) for x in cfg["ladder"]["head"]["tilt_deg"]]
    v = C - pl["c"][None, :]
    el = np.arctan2(-v[:, 2], np.maximum(np.linalg.norm(v[:, :2], axis=1), 1e-9))
    ok &= (el >= tl[0]) & (el <= tl[1])
    k = np.nonzero(ok)[0]
    if len(k):
        pts = plate_grid(pl, int(grid[0]), int(grid[1]), ahead=0.004)
        allc = np.ones(len(k), bool)
        for p in pts:
            allc &= los(model, data, p, C[k], bodyexclude)
            if not allc.any():
                break
        ok[k] = allc
    return ok


def plate_nav_seen(cfg, model, data, pl, C, bodyexclude=-1):
    """Version 2: bool per camera position C (N, 3): the yellow plate is at least nav_blob_px wide in a navigation
    frame (640 px over 90 deg) and its centre in line of sight."""
    R = load_acceptance(cfg)["plate_readable"]
    nav = camera(cfg, "nav")
    C = np.atleast_2d(np.asarray(C, float))
    v = C - pl["c"][None, :]
    d = np.linalg.norm(v, axis=1)
    cosang = np.clip((v @ pl["n"]) / np.maximum(d, 1e-9), 0.0, 1.0)
    ok = nav["f"] * PLATE_W * cosang / np.maximum(d, 1e-9) >= float(R["nav_blob_px"]) - 1e-9
    k = np.nonzero(ok)[0]
    if len(k):
        ok[k] = los(model, data, pl["c"] + 0.004 * pl["n"], C[k], bodyexclude)
    return ok


def acceptance_file(cfg) -> str:
    return os.path.abspath(acceptance_path(cfg))
