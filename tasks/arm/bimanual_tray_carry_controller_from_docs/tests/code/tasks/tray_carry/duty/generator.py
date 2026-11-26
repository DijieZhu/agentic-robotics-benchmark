import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
for _p in (os.path.join(TASK_DIR, "ref"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shift as SH
import numpy as np
from openarm_unit import rng as urng

import sheet as SHEET

K = SH.K
CONTACT_EVERY = 5
DECLINE_DRAWS_MAX = SH.DECLINE_DRAWS_MAX
NO_DECLINE_TRAY = SH.NO_DECLINE_TRAY
REASON_JOB = "no {kind} job passes the generator's checks on this unit"
REASON_DECLINE_JOB = "the decline slot's job fails a check other than the torques"
REASON_CARRY_COMMAND = "a carry job's torque by command on the true unit is above the generator's share"
GENERATOR_VERSION = 1


def tray_ranges():
    with open(K.PROFILE_PATH, encoding="utf-8") as f:
        prof = json.load(f)
    tq = {t["quantity"]: t for t in prof["task_quantities"] if "quantity" in t}
    r = tq["tray_mass_and_centre"]["range"]
    return {"classes_kg": {k: tuple(float(x) for x in v) for k, v in r["classes_kg"].items()},
            "along_m": tuple(float(x) for x in r["centre_along_m"]),
            "fore_aft_m": tuple(float(x) for x in r["centre_fore_aft_m"]),
            "below_m": tuple(float(x) for x in r["centre_below_m"]),
            "decline_along_abs_m": tuple(float(x) for x in r["decline_centre_along_abs_m"])}


def bracket_pad_friction():
    with open(K.PROFILE_PATH, encoding="utf-8") as f:
        prof = json.load(f)
    return float(prof["bracket"]["U_worst"]["pad_friction"])


def decline_counts(key):
    lo, hi = K.PER_UNIT["decline"]
    n_units = K.N_UNITS
    n_hi = 4 - lo * n_units
    if n_hi != 1 or hi != lo + 1:
        raise ValueError("decline_counts: the contract's counts are not (1, 1, 2)")
    place = int(math.floor(3.0 * urng.uniform(bytes(key), "duty/decline_counts", 0)))
    return tuple(hi if i == place else lo for i in range(n_units))


def roles(n_decline):
    lo, hi = K.PER_UNIT["decline"]
    if not lo <= int(n_decline) <= hi:
        raise ValueError(f"n_decline: {lo} to {hi}")
    out = []
    n_short, n_proof = K.PER_UNIT["short"], K.PER_UNIT["short_proof"]
    for i in range(n_short - n_proof):
        out.append((f"short_band_{i + 1}", "short", "band"))
    for i in range(n_proof):
        out.append((f"short_proof_{i + 1}", "short", "proof"))
    for i in range(K.PER_UNIT["medium"]):
        out.append((f"medium_band_{i + 1}", "medium", "band"))
    for i in range(K.PER_UNIT["lap_band"]):
        out.append((f"lap_band_{i + 1}", "lap", "band"))
    for i in range(int(n_decline)):
        out.append((f"lap_decline_{i + 1}", "lap", "decline"))
    for _, kind, cls in out:
        if kind not in K.TRAY_CLASSES[cls]["jobs"]:
            raise ValueError(f"class {cls} is not offered on a {kind} job")
    return out


def _is_lap_slot(role):
    return role[1] == "lap"


def slot_order(key, label, rl):
    stream = f"duty/{label}/order"
    c = [0]

    def u():
        v = urng.uniform(bytes(key), stream, c[0])
        c[0] += 1
        return v
    ends = [r for r in rl if not _is_lap_slot(r)]
    if len(ends) < 2:
        raise ValueError("fewer than two slots that may be first or last")
    first = ends.pop(int(math.floor(u() * len(ends))))
    last = ends.pop(int(math.floor(u() * len(ends))))
    mid = [r for r in rl if r is not first and r is not last]
    for i in range(len(mid) - 1, 0, -1):
        j = int(math.floor(u() * (i + 1)))
        mid[i], mid[j] = mid[j], mid[i]
    return [first] + mid + [last]


def carry_tray_draw(key, label, role, cls, k):
    r = tray_ranges()

    def u(q, lo, hi):
        return float(urng.uniform_in(bytes(key), f"duty/{label}/tray/{role}/{q}", int(k), lo, hi))
    return {"mass_kg": u("mass_kg", *r["classes_kg"][cls]), "along_m": u("along_m", *r["along_m"]),
            "fore_aft_m": u("fore_aft_m", *r["fore_aft_m"]), "below_m": u("below_m", *r["below_m"]), "draw": int(k)}


def decline_entropy(key, label, role):
    return urng.digest(bytes(key), f"duty/{label}/tray/{role}", 0)


def decline_tray_draw(entropy, k):
    r = tray_ranges()
    key = bytes(entropy) if len(entropy) == 32 else hashlib.sha256(bytes(entropy)).digest()

    def u(q, lo, hi):
        return float(urng.uniform_in(key, f"decline_tray/{q}", int(k), lo, hi))
    side = 1.0 if u("handle", 0.0, 1.0) < 0.5 else -1.0
    return {"mass_kg": u("mass_kg", *r["classes_kg"]["decline"]),
            "along_m": side * u("along_abs_m", *r["decline_along_abs_m"]),
            "fore_aft_m": u("fore_aft_m", *r["fore_aft_m"]), "below_m": u("below_m", *r["below_m"]), "draw": int(k)}


def tray_of(d):
    return {"mass_kg": float(d["mass_kg"]), "com": (float(d["fore_aft_m"]), float(d["along_m"]), -float(d["below_m"]))}


def job_entropy(key, label, role):
    return urng.digest(bytes(key), f"duty/{label}/job/{role}", 0)


def plain(o):
    return json.loads(json.dumps(o, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def failing_checks(res):
    if not isinstance(res, dict):
        return ["targets: " + str(res)]
    out = []
    if res.get("margin_min_deg", 1e9) < K.GENERATOR_STOP_MARGIN_DEG:
        out.append("stop_margin")
    if res.get("fk_err_mm", 0.0) > 0.2 or res.get("fk_err_deg", 0.0) > 0.05:
        out.append("solve")
    if res.get("speed_share", 0.0) > K.GENERATOR_SPEED_SHARE_MAX:
        out.append("speed")
    if res.get("contacts"):
        out.append("contact")
    if res.get("tray_to_S_min_m", 1e9) < K.GENERATOR_TRAY_TO_S_MIN_M:
        out.append("tray_to_S")
    if "torque_share" in res:
        out.append("torque?")
    return out


def rejection_reasons(rejections, label):
    lim = 1.0 if label == "decline" else K.GENERATOR_TORQUE_SHARE_MAX
    out = {}
    for r in rejections:
        if isinstance(r.get("reason"), str):
            k = "draw: " + r["reason"].split(" (")[0].split(" of ")[0]
            out[k] = out.get(k, 0) + 1
            continue
        res = dict(r["reason"])
        if isinstance(res.get("reason"), str):
            k = f"{r.get('on', '?')}: {res['reason']}"
            out[k] = out.get(k, 0) + 1
            continue
        res["contacts"] = r.get("contacts")
        names = [n for n in failing_checks(res) if n != "torque?"]
        if res.get("torque_share", 0.0) > lim:
            names.append("torque")
        for n in names or ["other"]:
            k = f"{r.get('on', '?')}: {n}"
            out[k] = out.get(k, 0) + 1
    return out


def _torques_only(rej):
    if not rej or rej.get("on") != "true unit" or isinstance(rej.get("reason"), str):
        return False
    res = dict(rej["reason"])
    res["contacts"] = rej.get("contacts")
    return [n for n in failing_checks(res) if n != "torque?"] == [] and res.get("torque_share", 0.0) > 1.0


_CACHE = {}


def nominal_kin():
    if "kin" not in _CACHE:
        _CACHE["kin"] = SH.Kin()
    return _CACHE["kin"]


def lap_list():
    if "lap" not in _CACHE:
        tg, rec = SH.lap_targets(nominal_kin())
        _CACHE["lap"] = ([list(t) for t in tg], rec)
    return _CACHE["lap"]


def nominal_basis(targets, swivel=None):
    k = ("basis", json.dumps(targets), swivel)
    if k not in _CACHE:
        _CACHE[k] = SH.TorqueBasis(nominal_kin(), targets, swivel=swivel)
    return _CACHE[k]


def true_check_of(state, session, unit):
    return (SH.true_kin(state), SH.CellModel(unit), tuple(float(x) for x in state.cal.k), "command",
            SH.true_offsets(state, session), SH.module_clamps(unit))


def take_up(entropy, basis_nom, basis_true, scale, offset, clamp, start=0, draws=DECLINE_DRAWS_MAX):
    tries = []
    for k in range(int(start), int(start) + int(draws)):
        d = decline_tray_draw(entropy, k)
        t = tray_of(d)
        sn = basis_nom.share(t)
        stt = basis_true.share(t, scale, offset, clamp)
        tries.append({"draw": k, "share_nominal": sn, "share_true": stt})
        if sn <= 1.0 and stt <= 1.0:
            return {"tray": d, "tried": len(tries), "tries": tries, "reason": None}
    return {"tray": None, "tried": len(tries), "tries": tries, "reason": NO_DECLINE_TRAY}


def generate_shift(key, label, state, session, unit, n_decline, tray_index=None, cell=None,
                   contact_every=CONTACT_EVERY, check_true=True):
    t0, c0 = time.perf_counter(), time.process_time()
    cell = SH.CELL_DEFAULT if cell is None else cell
    ti = dict(tray_index or {})
    rl = roles(n_decline)
    order = slot_order(key, label, rl)
    kin = nominal_kin()
    cm_nom = SH.nominal_cell_model(cell)
    tc = true_check_of(state, session, unit) if check_true else None
    out = {"ok": False, "reason": None, "label": label, "n_decline": int(n_decline), "cell": cell,
           "order": [r[0] for r in order], "slots": [], "jobs": [], "sheet": None, "sheet_digest": None,
           "shift_digest": None, "generator_version": GENERATOR_VERSION, "contact_every": int(contact_every),
           "check_true": bool(check_true)}
    true_basis = {}

    def tbasis(targets):
        kk = json.dumps(targets)
        if kk not in true_basis:
            true_basis[kk] = SH.TorqueBasis(tc[0], targets)
        return true_basis[kk]

    def finish(reason=None, detail=None):
        out["reason"] = reason
        if detail is not None:
            out["rejection_detail"] = plain(detail)
        out["ok"] = reason is None
        if out["ok"]:
            sh = SHEET.build_sheet([(sl["job"], j["targets"]) for sl, j in zip(out["slots"], out["jobs"])])
            SHEET.validate_sheet(sh)
            out["sheet"] = sh
            out["sheet_digest"] = SHEET.digest(sh)
            out["shift_digest"] = shift_digest(out)
        out["cpu_s"] = time.process_time() - c0
        out["wall_s"] = time.perf_counter() - t0
        return out

    for pos, (role, kind, cls) in enumerate(order):
        k = int(ti.get(role, 0))
        ent = job_entropy(key, label, role)
        sl = {"slot": pos + 1, "name": role, "job": kind, "label": "decline" if cls == "decline" else "carry",
              "class": cls, "tray_index": k, "job_entropy": ent.hex()}
        if cls == "decline":
            targets = lap_list()[0]
            bn = nominal_basis(targets)
            if tc is not None:
                bt = tbasis(targets)
                tu = take_up(decline_entropy(key, label, role), bn, bt, tc[2], tc[4], tc[5], start=k)
            else:
                tu = take_up(decline_entropy(key, label, role), bn, bn, None, None, None, start=k)
            sl["take_up"] = {"start": k, "tried": tu["tried"], "reason": tu["reason"],
                             "first_share_true": tu["tries"][0]["share_true"],
                             "best_share_true": min(x["share_true"] for x in tu["tries"]),
                             "first_share_nominal": tu["tries"][0]["share_nominal"],
                             "taken_draw": None if tu["tray"] is None else tu["tray"]["draw"]}
            if tu["tray"] is None:
                out["slots"].append(plain(sl))
                return finish(NO_DECLINE_TRAY, {"slot": pos + 1, "role": role, "tries": tu["tries"]})
            d = tu["tray"]
        else:
            d = carry_tray_draw(key, label, role, cls, k)
        sl.update(mass_kg=d["mass_kg"], along_m=d["along_m"], fore_aft_m=d["fore_aft_m"], below_m=d["below_m"])
        tray = tray_of(d)
        j = SH.generate_job(kind, ent, kin, tray, cm_nom, label=sl["label"], contact_every=contact_every,
                            true_check=tc)
        jr = {"targets": j["targets"], "swivels": j.get("swivels"), "record": j["record"],
              "rejections": j["rejections"], "check": j["check"], "check_true": j["check_true"],
              "rejection_reasons": rejection_reasons(j["rejections"], sl["label"])}
        out["slots"].append(plain(sl))
        if j["targets"] is None:
            out["jobs"].append(plain(jr))
            if cls == "decline":
                return finish(REASON_DECLINE_JOB, {"slot": pos + 1, "role": role,
                                                   "rejection": j["rejections"][-1] if j["rejections"] else None,
                                                   "torques_only": _torques_only(j["rejections"][-1]
                                                                                 if j["rejections"] else None)})
            return finish(REASON_JOB.format(kind=kind), {"slot": pos + 1, "role": role,
                                                          "rejections": j["rejections"]})
        if cls != "decline" and tc is not None:
            sh_c = tbasis(jr["targets"]).share(tray, tc[2], tc[4], tc[5])
            jr["torque_by_command"] = {"share": sh_c, "limit": K.GENERATOR_TORQUE_SHARE_MAX}
            if sh_c > K.GENERATOR_TORQUE_SHARE_MAX:
                out["jobs"].append(plain(jr))
                return finish(REASON_CARRY_COMMAND, {"slot": pos + 1, "role": role, "share": sh_c})
        out["jobs"].append(plain(jr))
    return finish(None)


def shift_digest(shift):
    body = {"label": shift["label"], "n_decline": shift["n_decline"], "order": shift["order"], "slots": shift["slots"],
            "jobs": [{"targets": j["targets"], "swivels": j["swivels"]} for j in shift["jobs"]],
            "sheet": shift["sheet"]}
    return SHEET.digest(plain(body))


def run_slots(shift):
    return [dict(s) for s in shift["slots"]]


def run_jobs(shift):
    return [dict(j) for j in shift["jobs"]]


CODE_FILES = ("duty/generator.py", "duty/sheet.py", "duty/label.py", "duty/tells.py", "duty/dev.py", "ref/shift.py",
              "ref/minimal.py", "ref/quantities.py", "contract/constants.py")


def code_digest(root=None):
    root = TASK_DIR if root is None else root
    h = hashlib.sha256()
    for rel in CODE_FILES:
        h.update(rel.encode("ascii") + b"\0")
        with open(os.path.join(root, rel), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def certification_rows(shift):
    import quantities as QT
    L = (K.F_REFERENCE_POINT[0], K.F_REFERENCE_POINT[1], K.F_REFERENCE_POINT[2] + K.LIFTED_POINT_M)
    v = []
    for sl, j in zip(shift["slots"], shift["jobs"]):
        if sl["label"] != "carry":
            continue
        vs = QT.v_strategies(j["targets"], L)
        v.append({"slot": sl["slot"], "job": sl["job"], **{k: {"V": vs[k]["V"], "visited": vs[k]["visited"],
                                                               "n": vs[k]["n"]} for k in vs}})
    jl = QT.j_enumeration_hold_still(shift["slots"], shift["jobs"], L)
    return {"V": v, "J_hold_still": jl, "J_hold_still_max": max(x["J"] for x in jl) if jl else None}
