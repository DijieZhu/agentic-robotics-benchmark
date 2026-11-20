"""Per-episode physical draws of the strut assembly cell (the layout family is the hidden world;
grader side): where the cradle and the trays stand, which unit lies in which nest and how each one lies there.

`episode_layout(seed, kit, cfg, units, nominal=False, scope=...)` -> {"fixture": {"xy"}, "trays": {tray: {"xy",
"yaw_deg"}}, "nests": {unit: nest}, "parts": {unit: (pos, quat wxyz)}, "rolls_deg", "lid_flip", "lid_flips", "jitter":
{...}}, a pure function of its inputs:
  * the cradle at its nominal place +- jitter_m per axis, never turned (fixture.py);
  * every tray at its nominal place +- xy_m per axis and +- jitter_yaw_deg about the vertical, ALL trays redrawn
    together when one draw fails: each footprint on the bench, clear of the cradle and the struts on
    it by `clear_m`, of the Panda's base column by `base_clear_m`, of the bench board and of the other trays, every
    nest centre 0.30-0.62 m from the Panda's base axis; the amplitudes halve every `layouts` failures, the nominal
    trays last;
  * the units: a random permutation of the scope's motor modules over all the motor nests (both builds are one
    design), of its end caps over all the cap nests (the cover colours only read in the images), the lids in the lid
    nests of the plan;
  * each motor module's roll about its own axis = the roll that a straight transfer would put on the rod at the end
    its nest feeds in the nominal plan (`nest_end`; seen in the nest's tray: a module that has to turn half a turn
    about the vertical on its way, because its tray's +y does not point out of that end, is counted before the turn;
    the authors' two motor modules of a strut sit 88.2 deg apart about the rod after that turn, so the two ends' bases
    differ) + the nest's hidden offset (+-roll_offset_deg, alternating over the nests: nest_offset_sign) + a uniform
    +-roll_jitter_deg: 6-16 deg off a straight transfer to its nest's end, 72-104 deg off the other end; each end
    cap at a uniform roll (its roll is free); every lid outer face up, turned 180 deg about the vertical with
    probability 0.5 (each lid on its own);
  * every unit's pose = its tray's pose x its resting pose in its nest (trays.nest_rest: from its collision hulls).
nominal=True (the privileged oracle, `oracle_nominal_layout`): the nominal cradle and trays, the units in the plan's
nests, the nests' offsets without jitter, no lid turned.

`defaults(scope)` = the generic keys + the scope's trays, nominal plan (plan_nests), nest offsets and nest ends.
"""
import copy
import math

import numpy as np

from . import fixture as FX
from . import trays as TR

__all__ = ["DEFAULT_LAYOUT", "SCOPE_LAYOUT", "defaults", "episode_layout", "check_layout", "design_roll_deg", "tray_T"]

PANDA_XY = (-0.45, 0.0)

DEFAULT_LAYOUT = {
    # every tray's draw about its nominal pose
    "tray": {"xy_m": 0.020, "jitter_yaw_deg": 10.0},
    "nest": {"roll_offset_deg": 11.0, "roll_jitter_deg": 5.0, "lid_flip": True, "permute_caps": True, "permute_motor_builds": True},
    "reach_m": [0.30, 0.62],
    "base_clear_m": 0.16,
    "clear_m": 0.03,
    "bench_xy": [[-0.798, 0.798], [-0.498, 0.498]],
    # the bench board's footprint (SO-101's board at (0.335, 0.005), 9 x 6 squares of 20 mm, 20 mm margin) + 10 mm
    "board_box": [[0.215, 0.455], [-0.165, 0.175]],
    "tries": 200,
    "layouts": 50,
}

# the scopes' trays (nominal place and turn), nominal plan (which unit in which nest), the motor nests' hidden roll
# offsets (sign x nest.roll_offset_deg, alternating) and the end each motor / cap nest feeds in the nominal plan (its
# base roll is that end's straight-transfer roll; an end cap's roll is free, its nest end only says which end it faces)
SCOPE_LAYOUT = {
    "single": {
        "trays": {"tray_l": {"nominal_xy": [-0.165, 0.36], "yaw_deg": 0.0}},
        "plan_nests": {"M1_1": "motor_a", "M1_2": "motor_b", "M2_1": "lid", "M3_1": "cap_a", "M3_2": "cap_b"},
        "nest_offset_sign": {"motor_a": 1.0, "motor_b": -1.0},
        "nest_end": {"motor_a": "p", "motor_b": "n"},
    },
    "default": {
        # tray_l left of the robot (+y), tray_r right (-y) turned half a turn so that its -y (the leads' and bore
        # mouths' side) also faces the cradle: the left tray feeds the struts' p ends (+y) and the lids, the right tray
        # the n ends, every motor module and end cap by a straight transfer
        "trays": {"tray_l": {"nominal_xy": [-0.33, 0.36], "yaw_deg": 0.0}, "tray_r": {"nominal_xy": [-0.30, -0.36], "yaw_deg": 180.0}},
        "plan_nests": {"M1_1": "l_motor1", "M1_3": "l_motor2", "M1_5": "l_motor3", "M1_2": "r_motor3", "M1_4": "r_motor2", "M1_6": "r_motor1",
                       "M2_1": "l_lid1", "M2_2": "l_lid2", "M2_3": "l_lid3",
                       "M3_1": "l_cap1", "M3_3": "l_cap2", "M3_5": "l_cap3", "M3_2": "r_cap3", "M3_4": "r_cap2", "M3_6": "r_cap1"},
        "nest_offset_sign": {"l_motor1": 1.0, "l_motor2": -1.0, "l_motor3": 1.0, "r_motor1": -1.0, "r_motor2": 1.0, "r_motor3": -1.0},
        "nest_end": {"l_motor1": "p", "l_motor2": "p", "l_motor3": "p", "r_motor1": "n", "r_motor2": "n", "r_motor3": "n",
                     "l_cap1": "p", "l_cap2": "p", "l_cap3": "p", "r_cap1": "n", "r_cap2": "n", "r_cap3": "n"},
    },
}


def _update(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def defaults(scope="single"):
    """the layout config of a scope: the generic keys + the scope's trays and nominal plan"""
    return _update(DEFAULT_LAYOUT, SCOPE_LAYOUT[scope])


def merged(cfg=None, scope="single"):
    """the scope's defaults with `cfg` over them (a config that already holds the scope's keys stays as it is)"""
    return _update(defaults(scope), cfg)


def _rng(seed, tag):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, tag])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def design_roll_deg(kit, scope="single", fixture_cfg=None, end="p", tray_yaw_deg=0.0):
    """the motor module's roll in a tray frame (the module's axis along the tray's +y) that a straight transfer
    carries onto the rod at the design of `end`, for a tray turned tray_yaw_deg about the vertical (a module whose
    design axis points along the tray's -y turns 180 deg about the vertical on its way: counted before the turn)"""
    mt = [m for m in kit.mates(scope) if m["kind"] == "rod_sleeve" and m["part_b_type"] == "M1" and m["id"].endswith("_" + end)][0]
    Td = np.array(mt["design_pose_b_in_a"], float)
    R = FX.seat_frame(FX.merged(fixture_cfg)["seat_azimuth_deg"]) @ Td[:3, :3]      # module axes in the cell, at the design
    R = _rotz(math.radians(float(tray_yaw_deg))).T @ R                              # ... in the tray's frame
    if float(R[1, 2]) < 0:
        R = _rotz(math.pi).T @ R                         # as it lies in the tray before the half turn
    x = R[:, 0]
    # module_orientation: x -> (cos r, 0, -sin r) for a roll r about +y
    return math.degrees(math.atan2(-x[2], x[0]))


def tray_T(xy, yaw_deg, tray_cfg=None, plate=None):
    T = np.eye(4)
    T[:3, :3] = _rotz(math.radians(float(yaw_deg)))
    T[:3, 3] = [float(xy[0]), float(xy[1]), float((plate or TR.merged(tray_cfg)["plate"])[2])]
    return T


def _footprint(T_tray, plate):
    px, py = float(plate[0]), float(plate[1])
    cs = np.array([[sx * px / 2, sy * py / 2, 0.0, 1.0] for sx in (-1, 1) for sy in (-1, 1)])
    return (T_tray @ cs.T).T[:, :2]


def _rect_sep(c, ex, ey, hx, hy, box, g):
    """True when the rectangle (centre c, axes ex / ey, half sizes hx / hy) is separated from the axis-aligned box
    (x0, x1, y0, y1) grown by g"""
    x0, x1, y0, y1 = box[0] - g, box[1] + g, box[2] - g, box[3] + g
    bc = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    bh = np.array([0.5 * (x1 - x0), 0.5 * (y1 - y0)])
    for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), ex, ey):
        r_t = abs(hx * float(ex @ ax)) + abs(hy * float(ey @ ax))
        r_b = abs(bh[0] * ax[0]) + abs(bh[1] * ax[1])
        if abs(float((c - bc) @ ax)) > r_t + r_b:
            return True
    return False


def _rects_sep(A, B, g):
    """True when two rectangles (c, ex, ey, hx, hy) are more than g apart (separating axis on their four axes)"""
    (ca, exa, eya, hxa, hya), (cb, exb, eyb, hxb, hyb) = A, B
    for ax in (exa, eya, exb, eyb):
        ra = abs(hxa * float(exa @ ax)) + abs(hya * float(eya @ ax))
        rb = abs(hxb * float(exb @ ax)) + abs(hyb * float(eyb @ ax))
        if abs(float((ca - cb) @ ax)) > ra + rb + g:
            return True
    return False


def _rect(T_tray, plate):
    return (T_tray[:2, 3].copy(), T_tray[:2, 0].copy(), T_tray[:2, 1].copy(), 0.5 * float(plate[0]), 0.5 * float(plate[1]))


def _tray_ok(T_tray, cradle_xy, L, tray, fixture_cfg, struts):
    """(ok, why) of one tray's pose: `tray` = {"plate", "nests"} (trays.trays_of)"""
    plate = tray["plate"]
    P = _footprint(T_tray, plate)
    (bx0, bx1), (by0, by1) = L["bench_xy"]
    if P[:, 0].min() < bx0 or P[:, 0].max() > bx1 or P[:, 1].min() < by0 or P[:, 1].max() > by1:
        return False, "off the bench"
    # the cradle plate and the struts on it (caps reach +-0.163 m along y, motor modules +-21.5 mm across)
    fx = FX.merged(fixture_cfg)
    hx, hy = 0.5 * float(fx["plate"][0]), 0.5 * float(fx["plate"][1])
    boxes = [(cradle_xy[0] - hx, cradle_xy[0] + hx, cradle_xy[1] - hy, cradle_xy[1] + hy)]
    for s in struts:
        x = cradle_xy[0] + float(FX.seat_x(fx, s))
        boxes.append((x - 0.0215, x + 0.0215, cradle_xy[1] - 0.163, cradle_xy[1] + 0.163))
    bb = L["board_box"]
    boxes.append((bb[0][0], bb[0][1], bb[1][0], bb[1][1]))
    g = float(L["clear_m"])
    rc = _rect(T_tray, plate)
    for box in boxes:
        if not _rect_sep(rc[0], rc[1], rc[2], rc[3], rc[4], box, g):
            return False, "overlaps a cradle / strut / board box"
    if min(math.hypot(p[0] - PANDA_XY[0], p[1] - PANDA_XY[1]) for p in P) < float(L["base_clear_m"]):
        return False, "inside the Panda's base column"
    r0, r1 = (float(v) for v in L["reach_m"])
    for _n, _k, x, y in tray["nests"]:
        q = T_tray @ np.array([float(x), float(y), 0.0, 1.0])
        r = math.hypot(q[0] - PANDA_XY[0], q[1] - PANDA_XY[1])
        if r < r0 or r > r1:
            return False, "a nest %.3f m from the Panda's axis" % r
    return True, ""


def _trays_ok(poses, cradle_xy, L, trays, fixture_cfg, struts):
    for tn, T in poses.items():
        ok, why = _tray_ok(T, cradle_xy, L, trays[tn], fixture_cfg, struts)
        if not ok:
            return False, "%s: %s" % (tn, why)
    names = sorted(poses)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not _rects_sep(_rect(poses[a], trays[a]["plate"]), _rect(poses[b], trays[b]["plate"]), float(L["clear_m"])):
                return False, "%s overlaps %s" % (a, b)
    return True, ""


def nest_tray(trays, nest):
    for tn, t in trays.items():
        for n, _k, _x, _y in t["nests"]:
            if n == nest:
                return tn
    raise KeyError(nest)


def episode_layout(seed, kit, cfg=None, units=None, nominal=False, scope="single", tray_cfg=None, fixture_cfg=None, struts=None,
                   hullsets=None):
    L = merged(cfg, scope)
    plan = dict(L["plan_nests"])
    units = list(units or plan.keys())
    struts = tuple(struts or sorted(set(m["part_a"].split(".")[0] for m in kit.mates(scope))))
    trays = TR.trays_of(tray_cfg, scope)
    cxy = FX.cradle_xy(seed, fixture_cfg, nominal=nominal)
    tc = L["tray"]
    nom = {tn: ([float(v) for v in L["trays"][tn]["nominal_xy"]], float(L["trays"][tn]["yaw_deg"])) for tn in trays}
    jitter = {"nominal": bool(nominal), "amp": 0.0, "layouts_tried": 0}
    poses = {tn: tray_T(nom[tn][0], nom[tn][1], tray_cfg, trays[tn]["plate"]) for tn in trays}
    tray_xy = {tn: list(nom[tn][0]) for tn in trays}
    tray_yaw = {tn: nom[tn][1] for tn in trays}
    if not nominal:
        r = _rng(seed, 0x7A7)
        amp, chosen = 1.0, None
        for attempt in range(4 * int(L["layouts"])):
            if attempt and attempt % int(L["layouts"]) == 0:
                amp *= 0.5
            cand = {}
            for tn in sorted(trays):
                dx, dy = (float(v) for v in r.uniform(-1, 1, 2) * float(tc["xy_m"]) * amp)
                dyaw = float(r.uniform(-1, 1)) * float(tc["jitter_yaw_deg"]) * amp
                xy = [nom[tn][0][0] + dx, nom[tn][0][1] + dy]
                cand[tn] = (tray_T(xy, nom[tn][1] + dyaw, tray_cfg, trays[tn]["plate"]), xy, nom[tn][1] + dyaw)
            ok, _why = _trays_ok({tn: v[0] for tn, v in cand.items()}, cxy, L, trays, fixture_cfg, struts)
            if ok:
                chosen = (cand, attempt + 1, amp)
                break
        if chosen is None:
            jitter.update({"amp": 0.0, "layouts_tried": -1})
        else:
            cand, n, a = chosen
            for tn, (T, xy, yaw) in cand.items():
                poses[tn], tray_xy[tn], tray_yaw[tn] = T, xy, yaw
            jitter.update({"amp": a, "layouts_tried": n})
    # units -> nests
    nests_of_kind = {}
    for u in units:
        nests_of_kind.setdefault(kit.units[u]["type"], []).append(u)
    nests = {}
    rr = _rng(seed, 0x4E57)
    for t, us in nests_of_kind.items():
        slots = sorted(plan[u] for u in us)
        perm = list(range(len(us)))
        if not nominal and ((t == "M1" and L["nest"]["permute_motor_builds"]) or (t == "M3" and L["nest"]["permute_caps"])):
            perm = [int(v) for v in rr.permutation(len(us))]
        ordered = sorted(us, key=lambda u: plan[u])
        for i, u in enumerate(ordered):
            nests[u] = slots[perm[i]]
    # rolls, flips
    rd = _rng(seed, 0x2011)
    base = {}
    for tn in trays:
        for e in ("p", "n"):
            base[(tn, e)] = design_roll_deg(kit, scope, fixture_cfg, e, nom[tn][1])
    rolls, flips = {}, {}
    parts = {}
    for u in units:
        t = kit.units[u]["type"]
        nest = nests[u]
        tn = nest_tray(trays, nest)
        flip = False
        if t == "M1":
            sgn = float(L["nest_offset_sign"].get(nest, 1.0))
            j = 0.0 if nominal else float(rd.uniform(-1, 1)) * float(L["nest"]["roll_jitter_deg"])
            roll = base[(tn, L["nest_end"].get(nest, "p"))] + sgn * float(L["nest"]["roll_offset_deg"]) + j
        elif t == "M3":
            roll = 0.0 if nominal else float(rd.uniform(0.0, 90.0))
        else:
            roll = 0.0
            flip = (not nominal) and bool(L["nest"]["lid_flip"]) and bool(rd.uniform() < 0.5)
            flips[u] = flip
        rolls[u] = roll
        Tn, _info = TR.nest_rest(kit, tray_cfg, nest, kit.units[u]["build"], math.radians(roll), flip=flip, hullsets=hullsets, scope=scope)
        Tw = poses[tn] @ Tn
        parts[u] = ([float(v) for v in Tw[:3, 3]], [float(v) for v in TR._quat_from_R(Tw[:3, :3])])
    by_end = {}
    for (tn, e), v in base.items():
        by_end.setdefault(tn, {})[e] = round(float(v), 6)
    first = sorted(trays)[0]
    return {"fixture": {"xy": [float(v) for v in cxy]},
            "trays": {tn: {"xy": [float(v) for v in tray_xy[tn]], "yaw_deg": float(tray_yaw[tn])} for tn in sorted(trays)},
            "nests": nests, "parts": parts, "rolls_deg": {u: round(float(v), 6) for u, v in rolls.items()},
            "lid_flip": bool(any(flips.values())), "lid_flips": {u: bool(v) for u, v in flips.items()},
            "design_roll_deg": round(float(base[(first, "p")]), 6), "design_roll_deg_by_end": dict(by_end[first]),
            "design_roll_deg_by_tray_end": by_end, "nest_end": dict(L["nest_end"]), "nest_tray": {u: nest_tray(trays, nests[u]) for u in units},
            "scope": scope, "jitter": jitter}


def check_layout(lay, kit, cfg=None, tray_cfg=None, fixture_cfg=None, struts=None, scope=None):
    """(ok, problems): every tray inside the bench, clear of the cradle, struts, board, the other trays and the Panda's
    column, every nest 0.30-0.62 m from the Panda's axis; every motor module 6-16 deg off a straight transfer to its
    nest's end, >= 6 deg off the other end"""
    scope = scope or lay.get("scope", "single")
    L = merged(cfg, scope)
    probs = []
    trays = TR.trays_of(tray_cfg, scope)
    struts = tuple(struts or sorted(set(m["part_a"].split(".")[0] for m in kit.mates(scope))))
    poses = {tn: tray_T(t["xy"], t["yaw_deg"], tray_cfg, trays[tn]["plate"]) for tn, t in lay["trays"].items()}
    ok, why = _trays_ok(poses, lay["fixture"]["xy"], L, trays, fixture_cfg, struts)
    if not ok:
        probs.append("trays: " + why)
    bases = lay.get("design_roll_deg_by_tray_end") or {sorted(trays)[0]: lay["design_roll_deg_by_end"]}
    for u, r in lay["rolls_deg"].items():
        if kit.units[u]["type"] == "M1":
            nest = lay["nests"][u]
            tn = (lay.get("nest_tray") or {}).get(u) or nest_tray(trays, nest)
            e = lay["nest_end"].get(nest, "p")
            b = bases[tn]
            off = abs(((float(r) - float(b[e])) + 180.0) % 360.0 - 180.0)
            other = abs(((float(r) - float(b["n" if e == "p" else "p"])) + 180.0) % 360.0 - 180.0)
            if not (6.0 - 1e-9 <= off <= 16.0 + 1e-9):
                probs.append("%s roll %.2f deg off a straight transfer to its nest's end" % (u, off))
            if other < 6.0:
                probs.append("%s roll %.2f deg off a straight transfer to the other end" % (u, other))
    return not probs, probs
