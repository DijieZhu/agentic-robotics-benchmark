"""The whole-arm assembly plan (schema `arm-assembly-plan/1`): schema check and grader (assembly.md 6.2, 7.4; PUBLIC CODE).

The grader half runs on a GT (`gt_plan.json` + `symmetries.json` + `gt_axes.json` of a kit): practice kits ship theirs, the
hidden kit's live only in the task's tests/. Plan (millimetres, radians; every pose = the part's delivered mesh frame in
the carrier's delivered mesh frame; servos in the frame of parts/servo.stl):

    {"schema": "arm-assembly-plan/1", "units": {"length": "mm", "angle": "rad"},
     "mates": [{"kind": "servo_seat", "carrier": "P05", "T_servo_in_carrier": 4x4, "insertion_dir_in_carrier": [3], "step": 4},
               {"kind": "bracket_seat", "carrier": "P03", "part": "P04", "T_part_in_carrier": 4x4, "insertion_dir_in_carrier": [3], "step": 2},
               {"kind": "horn_mate", "carrier": "P05", "link": "P06", "T_link_in_carrier_at_zero": 4x4, "step": 4}]}

A servo seat is keyed by its carrier (the servos are interchangeable), a bracket seat by (carrier, part), a horn mate by
(carrier, link) where the carrier is the printed part that holds the DRIVING servo and the link the child link's printed
part, at the joint's design zero (every servo at bus reading 0: the manual's calibration pose). Duplicates count as extra.

Score (b(x; full, zero) = 1 at x <= full, 0 at x >= zero, linear between; every error = the minimum over the GT
self-symmetries of the carrier and of the part):
    seat   = min(b(e_pos; 0.5, 5.0 mm), b(e_ang; 1, 10 deg)) x (0.7 + 0.3 b(e_insertion_dir; 10, 45 deg))
    horn   = min(b(axis line distance at the horn; 0.5, 5.0 mm), b(axis angle; 1, 10 deg), b(axial offset; 0.5, 5.0 mm),
                 b(design-zero error, exact; f_j, z_j))
    plan   = (sum_m w_m score_m / sum_m w_m) x precision,   w = 1.0 seat, 1.0 bracket, 3.0 horn (27.0 for 15 mates),
             precision = matched declared mates / declared mates

The zero band of joint j is relative to what the practice kits give without a photograph (PLAN 21.3, template-copy credit;
2026-09-28): with e0_j = the smallest zero error among the no-photograph estimates of `zero_candidates` (every template
kit's zero, their mean, 0, and -- when the template kits agree in sign -- that sign times the kits' mean magnitude, the
middle of their magnitudes + 10 deg, the larger magnitude + 15 deg and 45 deg), z_j = min(15, e0_j) and
f_j = min(2, z_j / 2) (floors 0.2 / 0.1 deg): copying any of those estimates earns 0 on the zero, the exact zero earns 1. The templates are the
practice kits OTHER than the graded kit (the hidden kit: P1 and P2; practice kit P1: P2; P2: P1). Without templates (the
e2e mode) the band is (2, 15).
"""

import json
import math

import numpy as np

__all__ = ["SCHEMA", "BANDS", "WEIGHTS", "check_schema", "grade_plan", "band", "load_plan", "zero_candidates", "zero_bands",
           "kit_zeros"]

SCHEMA = "arm-assembly-plan/1"
KINDS = ("servo_seat", "bracket_seat", "horn_mate")
# 2026-09-28 (review of the practice carry-over): the seat items are geometry, exactly derivable from the kit's meshes
# (E-ASM-12); the horn mates carry the information-bound zero angles; horn 3.0 puts 2/3 of the weight on them. With the zero
# band relative to the practice templates (zero_bands below; round-3 review: a practice-mean wrist-roll zero earned 0.79 of
# its horn mate under the fixed (2, 15) band), a plan whose zeros come from the practice kits alone scores at most the seat
# share 9 / 27 = 0.333 on geometry that is exact
WEIGHTS = {"servo_seat": 1.0, "bracket_seat": 1.0, "horn_mate": 3.0}
BANDS = {"seat_pos_mm": (0.5, 5.0), "seat_ang_deg": (1.0, 10.0), "seat_ins_deg": (10.0, 45.0),
         "horn_line_mm": (0.5, 5.0), "horn_axis_deg": (1.0, 10.0), "horn_axial_mm": (0.5, 5.0), "horn_zero_deg": (2.0, 15.0)}
PART_IDS = ["P%02d" % i for i in range(1, 11)]
MAX_MATES = 64


def band(x, full, zero):
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    return float((zero - x) / (zero - full))


# ------------------------------------------------------------------------------------------------ template-relative zero band
JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "jaw")
ZERO_RULE = "arm-assembly-plan-zero-band/2"          # /2 (task version 3): the magnitude estimates with both signs
ZERO_FLOOR_DEG = 0.1        # the tightest zero band: full 0.1 deg, zero 0.2 deg


def _wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def kit_zeros(variant):
    """{joint: bus zero (deg, design-angle units)} of a kit's variant.json (practice answer keys carry theirs)."""
    return {j: float(v) for j, v in (variant.get("C") or {}).get("bus_zero_deg", {}).items()}


def zero_candidates(template_zeros):
    """{joint: [deg, ...]}: the no-photograph estimates of every joint's zero from the template kits' zeros
    ({kit: {joint: deg}}): each kit's zero, their mean, 0 (the nominal arm's zero), and, for BOTH signs s = +1 and -1,
    s x the mean magnitude, s x (the middle of the magnitudes + 10), s x (the largest magnitude + 15) and s x 45.
    Task version 3 (a before-round tightening, PLAN 21.3): version 2 listed the magnitude estimates only with the kits' common
    sign, so on a joint whose practice kits disagree in sign a guessed sign earned credit (a fixed '-45 deg on every joint'
    plan scored 0.43 on the GT's exact geometry, the seat share being 0.333)."""
    out = {}
    kits = [k for k in sorted(template_zeros)]
    for j in JOINTS:
        z = [float(template_zeros[k][j]) for k in kits if j in template_zeros[k]]
        if not z:
            continue
        c = list(z) + [float(np.mean(z)), 0.0]
        m = [abs(x) for x in z]
        for s in (1.0, -1.0):
            c += [s * float(np.mean(m)), s * (0.5 * (min(m) + max(m)) + 10.0), s * (max(m) + 15.0), s * 45.0]
        out[j] = [round(x, 6) for x in c]
    return out


def zero_bands(gt_zeros, template_zeros, default=None):
    """{joint: {"full": f, "zero": z, "e0": e0, "candidate": the best estimate}}: z = max(min(default zero, e0), 0.2), f =
    max(min(default full, z / 2), 0.1) with e0 = the smallest |wrapped difference| between the joint's GT zero and any of its
    zero_candidates. Joints without a template keep the default band."""
    full0, zero0 = default or BANDS["horn_zero_deg"]
    out = {}
    cands = zero_candidates(template_zeros or {})
    for j, g in (gt_zeros or {}).items():
        cs = cands.get(j)
        if not cs:
            continue
        errs = [abs(_wrap180(c - float(g))) for c in cs]
        i = int(np.argmin(errs))
        e0 = float(errs[i])
        # numeric floor (T7: a full end never below the GT floor): the exact zero keeps full credit
        z = max(min(float(zero0), e0), 2.0 * ZERO_FLOOR_DEG)
        out[j] = {"full": max(ZERO_FLOOR_DEG, min(float(full0), z / 2.0)), "zero": z, "e0": e0, "candidate": float(cs[i])}
    return out


def _pose(v):
    T = np.asarray(v, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("not a finite 4x4 matrix")
    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 1e-3 or np.max(np.abs(R.T @ R - np.eye(3))) > 1e-3 or np.max(np.abs(T[3] - [0, 0, 0, 1])) > 1e-9:
        raise ValueError("not a rigid transform (rotation not orthonormal within 1e-3, or last row != 0 0 0 1)")
    # re-orthonormalise (a plan written with 6 digits is accepted)
    U, _s, Vt = np.linalg.svd(R)
    T = T.copy()
    T[:3, :3] = U @ Vt
    return T


def _unit(v):
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9:
        raise ValueError("not a non-zero 3-vector")
    return v / np.linalg.norm(v)


def check_schema(plan):
    """(ok, problems, parsed mates). Public rules only (the practice checker and the grader use the same)."""
    probs = []
    parsed = []
    if not isinstance(plan, dict):
        return False, ["the plan is not a JSON object"], []
    if plan.get("schema") != SCHEMA:
        probs.append("schema must be %r" % SCHEMA)
    u = plan.get("units") or {}
    if u.get("length") != "mm" or u.get("angle") != "rad":
        probs.append('units must be {"length": "mm", "angle": "rad"}')
    mates = plan.get("mates")
    if not isinstance(mates, list) or not mates:
        probs.append("mates must be a non-empty list")
        return False, probs, []
    if len(mates) > MAX_MATES:
        probs.append("more than %d mates" % MAX_MATES)
    for i, m in enumerate(mates[:MAX_MATES]):
        where = "mates[%d]" % i
        if not isinstance(m, dict) or m.get("kind") not in KINDS:
            probs.append("%s: kind must be one of %s" % (where, KINDS))
            continue
        try:
            e = {"kind": m["kind"], "carrier": str(m["carrier"]), "step": m.get("step")}
            if e["carrier"] not in PART_IDS:
                raise ValueError("carrier %r is not a kit part id (P01..P10)" % e["carrier"])
            if m["kind"] == "servo_seat":
                e["T"] = _pose(m["T_servo_in_carrier"])
                e["dir"] = _unit(m["insertion_dir_in_carrier"])
                e["key"] = ("servo_seat", e["carrier"])
            elif m["kind"] == "bracket_seat":
                e["part"] = str(m["part"])
                if e["part"] not in PART_IDS:
                    raise ValueError("part %r is not a kit part id" % e["part"])
                e["T"] = _pose(m["T_part_in_carrier"])
                e["dir"] = _unit(m["insertion_dir_in_carrier"])
                e["key"] = ("bracket_seat", e["carrier"], e["part"])
            else:
                e["link"] = str(m["link"])
                if e["link"] not in PART_IDS:
                    raise ValueError("link %r is not a kit part id" % e["link"])
                e["T"] = _pose(m["T_link_in_carrier_at_zero"])
                e["key"] = ("horn_mate", e["carrier"], e["link"])
            if e["step"] is not None and not isinstance(e["step"], int):
                raise ValueError("step must be an integer")
        except (KeyError, TypeError, ValueError) as exc:
            probs.append("%s: %s" % (where, exc))
            continue
        parsed.append(e)
    return (not probs), probs, parsed


def load_plan(path):
    with open(path) as f:
        return json.load(f)


def _rot_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def _vec_angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))))


def _twist_deg(R, axis):
    """The rotation of R about `axis` (swing-twist decomposition), degrees, signed."""
    w = math.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    if w < 1e-9:
        v = np.array([math.sqrt(max(0.0, (1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4)), math.sqrt(max(0.0, (1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4)),
                      math.sqrt(max(0.0, (1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4))])
    else:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4.0 * w)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return math.degrees(2.0 * math.atan2(float(v @ a), w))


def _seat_errors(decl, gt_T, gt_dir, sym_c, sym_p):
    best = None
    for Sc in sym_c:
        for Sp in sym_p:
            G = Sc @ gt_T @ Sp
            d = Sc[:3, :3] @ gt_dir
            e_pos = float(np.linalg.norm(decl["T"][:3, 3] - G[:3, 3]))
            e_ang = _rot_angle_deg(G[:3, :3].T @ decl["T"][:3, :3])
            e_ins = _vec_angle_deg(decl["dir"], d)
            s = min(band(e_pos, *BANDS["seat_pos_mm"]), band(e_ang, *BANDS["seat_ang_deg"])) * (0.7 + 0.3 * band(e_ins, *BANDS["seat_ins_deg"]))
            if best is None or s > best["score"]:
                best = {"score": s, "pos_mm": e_pos, "ang_deg": e_ang, "ins_deg": e_ins}
    return best


def _horn_errors(decl, gt_T, axis_c, sym_c, sym_p, zband=None):
    """axis_c = {"point": horn centre on the axis (carrier frame), "dir": joint axis (carrier frame)}; zband = (full, zero)
    of the design-zero error (default BANDS["horn_zero_deg"])."""
    zband = tuple(zband) if zband else BANDS["horn_zero_deg"]
    best = None
    p_h, u = np.asarray(axis_c["point"], float), np.asarray(axis_c["dir"], float)
    for Sc in sym_c:
        for Sp in sym_p:
            G = Sc @ gt_T @ Sp
            ph = Sc[:3, :3] @ p_h + Sc[:3, 3]
            uu = Sc[:3, :3] @ u
            # the child's axis in its own frame (GT), carried by the declared placement
            Gi = np.linalg.inv(G)
            pc = Gi[:3, :3] @ ph + Gi[:3, 3]
            uc = Gi[:3, :3] @ uu
            D = decl["T"]
            p_d = D[:3, :3] @ pc + D[:3, 3]
            u_d = D[:3, :3] @ uc
            axis_ang = _vec_angle_deg(uu, u_d)
            w = ph - p_d
            line = float(np.linalg.norm(w - u_d * float(w @ u_d) / float(u_d @ u_d)))
            axial = abs(float((p_d - ph) @ uu))
            zero = abs(_twist_deg(G[:3, :3].T @ D[:3, :3], uc))
            s = min(band(line, *BANDS["horn_line_mm"]), band(axis_ang, *BANDS["horn_axis_deg"]), band(axial, *BANDS["horn_axial_mm"]),
                    band(zero, *zband))
            if best is None or s > best["score"]:
                best = {"score": s, "line_mm": line, "axis_deg": axis_ang, "axial_mm": axial, "zero_deg": zero}
    return best


def grade_plan(plan, gt, symmetries=None, public=None, weights=None, zbands=None):
    """(value in [0, 1], details). `gt` = the kit's gt_plan.json (+ "axes" per horn mate: gt_axes.json merged by the caller
    or present in the entries as "axis"); `symmetries` = symmetries.json ({"parts": {pid | "servo": [4x4, ...]}});
    `public` = an optional plan (the public-arm counterpart) for the memorisation stat; `weights` = per-kind weights
    (default WEIGHTS); `zbands` = zero_bands(...) of the graded kit (default: the fixed band BANDS["horn_zero_deg"])."""
    WEIGHTS = dict(globals()["WEIGHTS"], **(weights or {}))
    zbands = zbands or {}
    ok, probs, parsed = check_schema(plan)
    det = {"schema_ok": bool(ok), "schema_problems": probs[:10], "declared": len(parsed), "matched": 0, "per_mate": {}}
    sym = (symmetries or {}).get("parts", {})

    def S(pid):
        return [np.asarray(T, float) for T in sym.get(pid, [])] or [np.eye(4)]
    gts = {}
    for m in gt["mates"]:
        if m["kind"] == "servo_seat":
            gts[("servo_seat", m["carrier"])] = m
        elif m["kind"] == "bracket_seat":
            gts[("bracket_seat", m["carrier"], m["part"])] = m
        else:
            gts[("horn_mate", m["carrier"], m["link"])] = m
    W = sum(WEIGHTS[m["kind"]] for m in gt["mates"])
    if not parsed:
        det["value"] = 0.0
        return 0.0, det
    used = set()
    total = 0.0
    kinds = {k: [] for k in KINDS}
    for e in parsed:
        g = gts.get(e["key"])
        if g is None or e["key"] in used:
            continue
        used.add(e["key"])
        if e["kind"] == "servo_seat":
            r = _seat_errors(e, np.asarray(g["T_servo_in_carrier"], float), np.asarray(g["insertion_dir_in_carrier"], float), S(e["carrier"]), S("servo"))
        elif e["kind"] == "bracket_seat":
            r = _seat_errors(e, np.asarray(g["T_part_in_carrier"], float), np.asarray(g["insertion_dir_in_carrier"], float), S(e["carrier"]), S(e["part"]))
        else:
            zb = zbands.get(g.get("joint"))
            r = _horn_errors(e, np.asarray(g["T_link_in_carrier_at_zero"], float), g["axis"], S(e["carrier"]), S(e["link"]),
                             zband=(zb["full"], zb["zero"]) if zb else None)
            if zb:
                r["zero_band_full_deg"], r["zero_band_zero_deg"] = zb["full"], zb["zero"]
        total += WEIGHTS[e["kind"]] * r["score"]
        kinds[e["kind"]].append(r["score"])
        det["per_mate"]["/".join(e["key"])] = dict({k: round(float(v), 4) for k, v in r.items()}, joint=g.get("joint"))
    det["matched"] = len(used)
    precision = len(used) / float(len(parsed))
    det["precision"] = precision
    det["recall"] = len(used) / float(len(gts))
    det["per_kind_mean"] = {k: (float(np.mean(v)) if v else 0.0) for k, v in kinds.items()}
    value = (total / W) * precision
    det["value"] = float(value)
    # step order (stat): a seat before its horn mate, a clamp not before the seat of its link
    det["step_order_ok"] = _step_order_ok(parsed, gt)
    if public is not None:
        det["public_arm_hits"] = _public_hits(parsed, gt, public, S)
    return float(value), det


def _step_order_ok(parsed, gt):
    seat_step = {e["carrier"]: e["step"] for e in parsed if e["kind"] == "servo_seat" and e["step"] is not None}
    for e in parsed:
        if e["step"] is None:
            continue
        if e["kind"] == "horn_mate" and e["carrier"] in seat_step and e["step"] < seat_step[e["carrier"]]:
            return False
        if e["kind"] == "bracket_seat" and e["carrier"] in seat_step and e["step"] < seat_step[e["carrier"]]:
            return False
    return True


def _public_hits(parsed, gt, public, S):
    """Declared mates within the full band of the public-arm counterpart's value but outside the GT's (memorisation)."""
    ok, _p, pub = check_schema(public)
    pub = {e["key"]: e for e in pub}
    gts = {}
    for m in gt["mates"]:
        key = ("servo_seat", m["carrier"]) if m["kind"] == "servo_seat" else (("bracket_seat", m["carrier"], m["part"]) if m["kind"] == "bracket_seat" else ("horn_mate", m["carrier"], m["link"]))
        gts[key] = m
    hits = 0
    for e in parsed:
        p, g = pub.get(e["key"]), gts.get(e["key"])
        if p is None or g is None:
            continue
        if e["kind"] == "horn_mate":
            rp = _horn_errors(e, p["T"], g["axis"], [np.eye(4)], [np.eye(4)])
            rg = _horn_errors(e, np.asarray(g["T_link_in_carrier_at_zero"], float), g["axis"], [np.eye(4)], [np.eye(4)])
            if rp["zero_deg"] <= BANDS["horn_zero_deg"][0] and rg["zero_deg"] > BANDS["horn_zero_deg"][0]:
                hits += 1
        else:
            key = "T_servo_in_carrier" if e["kind"] == "servo_seat" else "T_part_in_carrier"
            rp = float(np.linalg.norm(e["T"][:3, 3] - p["T"][:3, 3]))
            rg = float(np.linalg.norm(e["T"][:3, 3] - np.asarray(g[key], float)[:3, 3]))
            if rp <= BANDS["seat_pos_mm"][0] and rg > BANDS["seat_pos_mm"][0]:
                hits += 1
    return hits
