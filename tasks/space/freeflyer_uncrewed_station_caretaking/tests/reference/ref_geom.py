"""ref_geom.py -- pure geometry of the reference program (no reads of the cell): the pre-grasp poses of a bar, the
capture hint, the deposit plans of a held item on a suction face.

* ``pregrasp_candidates``: the bar ``dist`` in front of the palm (world.pregrasp.distance_m sampled), both cage rolls
  (palm +z = +- bar axis): the construction of ``spacekit.primitives.GraspFeature._target`` as the site generator's
  reach check uses it (tests/review/sitegen/gen/pregrasp.py,
  ``robot_pose_for_bar``), ordered so that the aft
  depth camera sees the bar from 0.45-0.60 m first (reference.hint_range_m).
* ``hint``: a world bar (centre, axis) and a world palm direction expressed in the body frame of a pose.
* ``plan_pouch``: the pouch's pose in the
  palm frame (privilege 4) -> the palm pose that puts the POUCH CENTRE on the pad centre (deposit rule 1),
  its long side (the handle bar) along the pad's b, its back face ``standoff`` off the face plane, palm +z toward the
  pad plate's side; the palm is squared to the pad, not turned by the pouch's in-hand attitude (align).
* ``plan_sheet``: the soft sheet on the holding screen: the palm square to the screen with
  the bar kept in its in-plane direction (the smallest turn; with the inlet and the screen on parallel rack faces a
  turn about the bar only), the sheet's flat-layout centre on the screen
  centre (deposit rule 1; for a loop on an end strip this is an offset of 0.12 m), the held strip's back
  ``standoff`` off the face plane; ``z_choice`` turns the bar to the screen's +-b / +-a axes instead (the fallback
  when no plate is seen).  The items turn freely about the grasped bar in the cage, so the plans
  square the palm, not the item, and the item's attitude is checked at the press (``item_normal_dev_deg``).
"""
from __future__ import annotations

import math

import numpy as np

from spacekit.primitives import pose_for_palm
from spacekit.rotations import mat_to_quat

BAR_IN_PALM_X = -0.090       # bar centre in the palm frame after the close (89-91 mm measured)


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def palm_z_pref(n) -> np.ndarray:
    """spacekit FirmwareLayer.palm_z_pref (the site generator's copy)."""
    return np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])


def robot_pose_for_bar(geom: dict, c, a, n, depth: float, extra: float = 0.0, zsign: float | None = None):
    """(body position, quaternion, palm +z) that put a bar (centre c, axis a, face normal n toward the robot) at palm
    depth ``depth`` plus ``extra`` out along n (depth = -dist: the bar dist in front of the palm)."""
    a = unit(a)
    n = unit(np.asarray(n, float) - (np.asarray(n, float) @ a) * a)
    if zsign is None:
        z = a if (a @ palm_z_pref(n)) >= 0 else -a
    else:
        z = float(zsign) * a
    x = n
    y = np.cross(z, x)
    Rp = np.column_stack([x, y, z])
    pp = np.asarray(c, float) - Rp @ np.array([depth, 0.0, geom["z_plane"]]) + n * extra
    p, q = pose_for_palm((geom["pb"], geom["Rb"]), pp, Rp)
    return np.asarray(p, float), np.asarray(q, float), z


def pregrasp_candidates(geom: dict, c, a, n, distances, zsigns=(1.0, -1.0)) -> list:
    """[dict(pos, quat, palm_z, dist, zsign)] in the order given (distances, then the two rolls)."""
    out = []
    for dist in distances:
        for zs in zsigns:
            p, q, z = robot_pose_for_bar(geom, c, a, n, -float(dist), 0.0, zs)
            out.append(dict(pos=p, quat=q, palm_z=z, dist=float(dist), zsign=float(zs)))
    return out


def hint(pos, R, c_world, a_world, palm_dir_world=None):
    """(bar centre, bar axis, palm direction) in the body frame of the pose (pos, R body -> world)."""
    R = np.asarray(R, float)
    cb = R.T @ (np.asarray(c_world, float) - np.asarray(pos, float))
    ab = R.T @ unit(a_world)
    pd = None if palm_dir_world is None else R.T @ unit(palm_dir_world)
    return cb, ab, pd


def body_pose_for_palm(geom: dict, palm_pos, R_palm):
    p, q = pose_for_palm((geom["pb"], geom["Rb"]), np.asarray(palm_pos, float), np.asarray(R_palm, float))
    return np.asarray(p, float), np.asarray(q, float)


def _plane_axes(face):
    return unit(face.n), unit(face.a), unit(face.b)


def _rz(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def inhand_angle(held: dict, R_item_palm) -> float:
    """The item's turn about the grasped bar (palm +z) in the hand [rad]: 0 = its top normal along +palm x."""
    z = np.asarray(R_item_palm, float)[:, 2]
    return math.atan2(float(z[1]), float(z[0]))


def plan_pouch(geom: dict, face, held: dict, plate_side: str = "+b", standoff: float = -0.008, align: str = "clamped",
               clamp_deg: float = 35.0, turn_deg: float | None = None) -> dict:
    """The pouch-pad plan (module doc).  held: Truth.held_pose (bodies [(p, R)] in the palm frame, half sizes).
    align 'clamped' (default): the palm turned about the bar by the pouch's in-hand turn so that
    the pouch faces the pad (the cage carries it: 30 / 30 on a test bench, 6 / 6 in the cell when not refused) -- with the
    turn clamped to clamp_deg (a pouch turned edge-on would swing the robot's body toward the face: the zone check
    refused those inserts), the centre placed where the pouch sits once the pad squares the rest of the turn;
    'palm': the palm square to the pad, no turn; 'item': the item square to the pad, unclamped."""
    n, fa, fb = _plane_axes(face)
    p_rel, R_rel = held["bodies"][0]
    hz = float(held["half"][2])
    side = 1.0 if plate_side.startswith("+") else -1.0
    if align == "clamped":
        z = side * fb
        R_sq = np.column_stack([n, np.cross(z, n), z])
        phi = inhand_angle(held, R_rel)
        if turn_deg is None:
            al = float(np.clip(phi, -math.radians(clamp_deg), math.radians(clamp_deg)))
        else:                                                   # an explicit palm turn (the closed-loop alignment)
            al = math.radians(float(turn_deg))
        R_palm = R_sq @ _rz(-al)
        sg = side
        p_use = squared_offset(held, p_rel)                     # the centre at in-hand angle 0 ...
        bars = held.get("bars") or {}
        if bars:
            name = min(bars, key=lambda k: float(np.linalg.norm(bars[k][0] - np.array([BAR_IN_PALM_X, 0.0, 0.0]))))
            bc = np.asarray(bars[name][0], float)
            p_use = bc + _rz(al) @ (p_use - bc)                  # ... turned to the in-hand angle the plan keeps
        c_des = face.c + n * (hz + float(standoff))
        palm_pos = c_des - R_palm @ p_use
        p, q = body_pose_for_palm(geom, palm_pos, R_palm)
        return dict(p_dep=p, q_dep=q, n=n, palm_z=R_palm[:, 2].copy(), R_palm=R_palm, sign=sg, kind="pouch", align=align,
                    inhand_deg=math.degrees(phi), turn_deg=math.degrees(al),
                    item_normal_dev_deg=item_normal_dev_deg(R_palm @ R_rel[:, 2], n))
    if align == "item":
        best = None
        for sg in (1.0, -1.0):
            x_p = sg * fb
            R_item = np.column_stack([x_p, np.cross(n, x_p), n])
            R_palm = R_item @ R_rel.T
            score = float(R_palm[:, 2] @ (side * fb))
            if best is None or score > best[0]:
                best = (score, R_palm, sg)
        _, R_palm, sg = best
    else:
        z = side * fb
        R_palm = np.column_stack([n, np.cross(z, n), z])
        sg = side
    c_des = face.c + n * (hz + float(standoff))
    p_use = squared_offset(held, p_rel) if align == "palm" else p_rel
    palm_pos = c_des - R_palm @ p_use
    p, q = body_pose_for_palm(geom, palm_pos, R_palm)
    return dict(p_dep=p, q_dep=q, n=n, palm_z=R_palm[:, 2].copy(), R_palm=R_palm, sign=sg, kind="pouch", align=align,
                item_normal_dev_deg=item_normal_dev_deg(R_palm @ R_rel[:, 2], n))


def squared_offset(held: dict, centre_palm):
    """Where an item's centre sits in the palm frame once it is squared to the face (turned about the grasped bar
    until its top normal is +palm x): the bar's palm (x, y) moved back along -x by the centre's distance from the bar
    axis, the centre's own position along the bar kept.  The items turn freely about the bar in the cage, so a plan on
    the momentary pose would plunge the robot when the item happens to be edge-on (the zone check refuses that)."""
    bars = held.get("bars") or {}
    if not bars:
        return np.asarray(centre_palm, float)
    name = min(bars, key=lambda k: float(np.linalg.norm(bars[k][0] - np.array([BAR_IN_PALM_X, 0.0, 0.0]))))
    bc, ba = np.asarray(bars[name][0], float), unit(bars[name][1])
    c = np.asarray(centre_palm, float)
    d = c - bc
    along = float(d @ ba)
    radial = d - along * ba
    r = float(np.linalg.norm(radial))
    # squared: the centre straight 'below' the bar toward the face (palm -x), at the same radius
    xdir = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ ba) * ba
    xdir = unit(xdir)
    return bc + ba * along - xdir * r


def held_item_normal(held: dict, kind: str):
    """The palm-frame normal of the item's face that will seal (sheet: the strip carrying the grasped bar; pouch: its
    body) and its in-hand angle about the bar (palm z) [deg] -- from the held pose (privilege P4)."""
    if kind == "sheet":
        k = sheet_held_strip(held)
        z = np.asarray(held["bodies"][k][1], float)[:, 2]
    else:
        z = np.asarray(held["bodies"][0][1], float)[:, 2]
    return z, math.degrees(math.atan2(float(z[1]), float(z[0])))


def item_normal_dev_deg(item_z_world, n) -> float:
    """Angle between an item's face plane and the face plane, 0-90 deg (either large face can seal: the suction takes
    the one facing it): 0 = parallel, 90 = edge-on."""
    c = abs(float(np.clip(unit(item_z_world) @ unit(n), -1.0, 1.0)))
    return math.degrees(math.acos(c))


def sheet_held_strip(held: dict) -> int:
    """Index of the strip that carries the grasped bar (the bar nearest the palm cage)."""
    bars = held["bars"]
    if not bars:
        return 0
    name = min(bars, key=lambda k: float(np.linalg.norm(bars[k][0] - np.array([BAR_IN_PALM_X, 0.0, 0.0]))))
    pb = bars[name][0]
    d = [float(np.linalg.norm(p - pb)) for p, _ in held["bodies"]]
    return int(np.argmin(d))


def plan_sheet(geom: dict, face, held: dict, R_palm_world, standoff: float = -0.008, z_choice: str | None = None,
               turn_deg: float = 0.0) -> dict:
    """The holding-screen plan (module doc).  held: Truth.held_pose; R_palm_world: the palm attitude now (the body
    estimate's attitude times the palm's constant rotation), only used to pick the smallest rotation.  The palm square
    to the screen (palm x = the face normal), palm +z = the bar: z_choice None keeps the bar's
    in-plane direction (the smallest turn: a turn about the bar only when the inlet and the screen are on parallel
    faces) or '+b' / '-b' / '+a' / '-a'.  The sheet's flat-layout centre (the middle strip, from
    the held strip and the chain direction where they sit in the hand now) on the screen centre; the held strip's
    back ``standoff`` off the face plane."""
    n, fa, fb = _plane_axes(face)
    k = sheet_held_strip(held)
    nb = len(held["bodies"])
    hx, _hy, hz = (float(v) for v in held["half"])
    p_k, R_k = held["bodies"][k]                       # the held strip in the palm frame
    if z_choice is None:
        zw = np.asarray(R_palm_world, float)[:, 2]
        zp = zw - (zw @ n) * n
        z_des = unit(zp) if float(np.linalg.norm(zp)) > 1e-6 else fb
    else:
        z_des = (1.0 if z_choice[0] == "+" else -1.0) * (fb if z_choice[1] == "b" else fa)
    R_sq = np.column_stack([n, np.cross(z_des, n), z_des])
    al = math.radians(float(turn_deg))
    R_palm = R_sq @ _rz(-al)                           # turn_deg: the palm turned about the bar so that a sheet at the
                                                       # in-hand angle +turn_deg lies flat (the closed-loop alignment)
    # the flat layout: strip j's centre at (j - k) * 2 hx * s along the held strip's x (s from the chain's order)
    s = 1.0
    if nb > 1:
        j2 = k + 1 if k + 1 < nb else k - 1
        d = held["bodies"][j2][0] - p_k
        s = 1.0 if float(d @ R_k[:, 0]) * float(j2 - k) >= 0 else -1.0
    off = (0.5 * (nb - 1) - k) * 2.0 * hx * s          # flat centre relative to the held strip, along its x
    p_sq = squared_offset(held, p_k)                   # the held strip once squared to the face
    p_mid = p_sq + R_k[:, 0] * off                     # the flat layout's centre, in the palm frame
    if al != 0.0:
        bars = held.get("bars") or {}
        if bars:                                       # ... at the in-hand angle the turned palm keeps (plan_pouch's way)
            name = min(bars, key=lambda kk: float(np.linalg.norm(bars[kk][0] - np.array([BAR_IN_PALM_X, 0.0, 0.0]))))
            bc = np.asarray(bars[name][0], float)
            p_mid = bc + _rz(al) @ (p_mid - bc)
            p_sq = bc + _rz(al) @ (p_sq - bc)
    palm_pos = face.c - R_palm @ p_mid
    dz = (hz + float(standoff)) - float((palm_pos + R_palm @ p_sq - face.c) @ n)
    palm_pos = palm_pos + n * dz
    p, q = body_pose_for_palm(geom, palm_pos, R_palm)
    return dict(p_dep=p, q_dep=q, n=n, palm_z=R_palm[:, 2].copy(), R_palm=R_palm, held_strip=k, centre_offset_m=float(off),
                kind="sheet", z_choice=z_choice, turn_deg=float(turn_deg),
                item_normal_dev_deg=item_normal_dev_deg(R_palm @ R_k[:, 2], n))


def _axis_angle(axis, ang: float) -> np.ndarray:
    a = unit(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def plan_aligned(geom: dict, face, held: dict, kind: str, base: dict, standoff: float, max_deg: float) -> dict:
    """The deposit plan with the palm turned so that the held item's sealing face lies in the face plane (the
    closed-loop alignment): ``base`` is the squared plan (plan_sheet / plan_pouch); the smallest rotation Q
    about the item that takes the item's face normal (its pose in the hand now, P4) onto the face normal (either face
    seals), clamped to max_deg, turns the palm: R_palm = Q R_base; the item's centre (a sheet's flat-layout centre from
    its held strip, a pouch's centre) at its pose in the hand now goes on the face centre, its back ``standoff`` off
    the face plane.  The residual deviation after the clamp is item_normal_dev_deg."""
    n = unit(face.n)
    R_b = np.asarray(base["R_palm"], float)
    if kind == "sheet":
        k = sheet_held_strip(held)
        p_k, R_k = held["bodies"][k]
        nb = len(held["bodies"])
        hx, _hy, hz = (float(v) for v in held["half"])
        s = 1.0
        if nb > 1:
            j2 = k + 1 if k + 1 < nb else k - 1
            d = held["bodies"][j2][0] - p_k
            s = 1.0 if float(d @ R_k[:, 0]) * float(j2 - k) >= 0 else -1.0
        off = (0.5 * (nb - 1) - k) * 2.0 * hx * s
        p_c = np.asarray(p_k, float) + R_k[:, 0] * off            # the flat-layout centre at the pose in the hand now
        p_back = np.asarray(p_k, float)
        z_p = R_k[:, 2]
    else:
        p_c, R_rel = held["bodies"][0]
        p_c = np.asarray(p_c, float)
        p_back = p_c
        hz = float(held["half"][2])
        z_p = np.asarray(R_rel, float)[:, 2]
    nu = R_b @ z_p
    tgt = n if float(nu @ n) >= 0.0 else -n
    ax = np.cross(nu, tgt)
    sa, ca = float(np.linalg.norm(ax)), float(nu @ tgt)
    ang = math.atan2(sa, ca)
    if sa < 1e-9:
        Q = np.eye(3)
        ang = 0.0
    else:
        ang = min(ang, math.radians(float(max_deg)))
        Q = _axis_angle(ax / sa, ang)
    R_palm = Q @ R_b
    palm_pos = face.c - R_palm @ p_c
    dz = (hz + float(standoff)) - float((palm_pos + R_palm @ p_back - face.c) @ n)
    palm_pos = palm_pos + n * dz
    p, q = body_pose_for_palm(geom, palm_pos, R_palm)
    out = dict(base)
    out.update(p_dep=p, q_dep=q, n=n, palm_z=R_palm[:, 2].copy(), R_palm=R_palm, aligned_deg=math.degrees(ang),
               item_normal_dev_deg=item_normal_dev_deg(R_palm @ z_p, n))
    return out


def quat_dist_deg(qa, qb) -> float:
    from spacekit.rotations import quat_to_mat
    Ra, Rb = quat_to_mat(np.asarray(qa, float)), quat_to_mat(np.asarray(qb, float))
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def quat(R) -> np.ndarray:
    return np.asarray(mat_to_quat(np.asarray(R, float)), float)
