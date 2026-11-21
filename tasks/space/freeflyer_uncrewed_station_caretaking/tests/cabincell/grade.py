"""grade.py -- the per-site grader of freeflyer_uncrewed_station_caretaking (the task's scoring rules
and the aggregate over the hidden set).

    grade(record, site, ref_energy_wh=None, config=None, runtime=None) -> the site's verdict
    margins(record, graded, config=None, runtime=None)                 -> the selection margins of the hidden-set rule
    chain_pass(graded, margins_out)                                    -> full / chain pass of a reference run
    mean_score(site_scores, contract_ok)                               -> caretaking_score = contract_ok x mean

Inputs, and nothing else: the cell's record of one run (the record format, caretaking.record/1: world state
every cell.record_period_s up to 'end' = program end + the settle, and events), the hidden site description
(station_site/2, its 'truth'), tests/grader_config.json (every threshold, generated value by value and never
edited by hand) and the runtime copy tests/cabincell/data/site_spec.json (the record period and the
budgets).  Imports the standard library and predicates.py only, never the simulation package.  Every rule reads
the world state or an event; the program's own word enters only through report().

Site score = R x (D + U1 + H + U2 + V + K + L + E), weights 1.00 in total:
  D   first report: inlet right; then register item and onset (within the tolerance) count; the cited frame
      (FIC / FWC, taken before the report) shows the inlet by the evidence rule while the blocker still covers it.
  U1  the blocked inlet clear (covered fraction <= max) for a continuous run of at least min_s, at any time.
  H   gate U1; at least the minimum charge at the DOCKED moment (the lowest over every docking when the robot
      undocked and docked again: charging on the berth earns nothing), docked at the end, no item in the gripper
      from the final docking on, no item touching both robot and dock.
  U2  over the last window_s before the end: the inlet clear, and the blocker under control throughout -- fixed on
      a holding screen the whole window, or in the closed gripper the whole window (one window for both).
  V   gate U2; all units running at the end; no flight-rule violation; every inlet of the blocked unit covered no
      more than the V limit at the end.
  K   the blocker fixed on a holding screen (sealed area on one >= the K minimum, not in the gripper) throughout
      the window.
  L   weight x g x max(0, n_on - penalty x n_lost_unstowed) / n_total; a pouch is on a pouch pad when, throughout
      the window, it is not in the gripper and its sealed area on one pad is at least the criterion's minimum:
      'own_footprint_fraction' (default: >= min_own_fraction of its own footprint) or 'abs_seal_area'
      (>= min_seal_m2_alt); both counts go to the metrics.  Lost = loss of control (the record's event, or the same
      rule re-applied to the state samples).  g = 1 with U1, else g_without_U1.
  E   gate K and every pouch on a pad: w_elec x e_elec (energy over the site's reference energy) + w_wind x e_wind
      (total fan-stop seconds), each linear between its full and zero lines.
  R   factor when hard collisions exceed the maximum (normal speed over the speed line, or force over the force line
      on a hatch seal, a lamp or a fiducial plate) or the keep-out time reaches its limit; else 1.
'End' = times.end (the end of the settle after the program).  No negative score; metrics hold finite numbers only;
missing or non-finite record values never raise a score (predicates.py: conservative readers).

The eight items, R, margins and chain_pass: holding screens and pouch pads by sealed area, the 60 s window for K,
L and both U2 branches, e_wind lines 35 / 120 s, hard-collision thresholds and every other number from
grader_config.json, every docking for H, the report check of non-finite values, frames looked up by id, a loss of
control re-checked on the samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(TESTS, "grader_config.json")
RUNTIME_PATH = os.path.join(HERE, "data", "site_spec.json")
RUNTIME_SHA_PATH = os.path.join(HERE, "data", "site_spec.sha256")

if __package__:
    from . import predicates as P
else:                                                     # loaded by path (verify scripts, the replay runner, tools)
    import importlib.util as _ilu
    _pp = os.path.join(HERE, "predicates.py")
    _pn = "caretaking_predicates_" + hashlib.sha256(_pp.encode()).hexdigest()[:12]
    P = sys.modules.get(_pn)
    if P is None:
        _spec = _ilu.spec_from_file_location(_pn, _pp)
        P = _ilu.module_from_spec(_spec)
        sys.modules[_pn] = P
        _spec.loader.exec_module(P)

GRADER = "caretaking.grade/1"
CONFIG_SCHEMA = "caretaking.grader_config/1"
RUNTIME_SCHEMA = "caretaking.site_spec_runtime/1"
ITEMS = ("D", "U1", "H", "U2", "V", "K", "L", "E")
FORCE_CLASSES = ("hatch_seal", "lamp", "fiducial_plate")          # the force line's contact classes
CRITERIA = ("own_footprint_fraction", "abs_seal_area")
REQUIRED_KEYS = (
    "grader.weights", "grader.window_s", "grader.D.parts", "grader.D.evidence", "grader.D.onset_tol_min", "grader.U1",
    "grader.H.min_battery_pct", "grader.U2.max_cover", "grader.V.max_unit_inlet_cover", "grader.K.min_seal_m2",
    "grader.L.criterion", "grader.L.min_own_fraction", "grader.L.min_seal_m2_alt", "grader.L.formula", "grader.E",
    "grader.R.factor", "grader.R.hard_speed_m_s", "grader.R.hard_force_N", "grader.R.max_hard_collisions",
    "grader.R.keepout_s", "grader.loss_of_control", "grader.selection", "grader.seal.max_gap_mm", "grader.covered",
    "grader.in_gripper_cage_mm", "grader.aggregate")
RUNTIME_KEYS = ("cell.record_period_s", "cell.budget.colour_frames", "cell.budget.depth_frames", "cell.budget.sim_time_s",
                "cell.budget.capture_calls")
NA = -1.0                                                          # metric value for 'not applicable'


# ------------------------------------------------------------------------------------------------ inputs
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_config(path: str = CONFIG_PATH) -> dict:
    """tests/grader_config.json: {'values': {key: value}, 'sha256', ...}; raises when a threshold is missing."""
    raw = open(path, "rb").read()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} is not {CONFIG_SCHEMA}")
    vals = doc.get("values") or {}
    missing = [k for k in REQUIRED_KEYS if k not in vals]
    if missing:
        raise KeyError(f"{path}: missing grader keys {missing}")
    if vals["grader.L.criterion"] not in CRITERIA:
        raise ValueError(f"{path}: unknown grader.L.criterion {vals['grader.L.criterion']!r}")
    return dict(values=vals, sha256=_sha(raw), contract_version=doc.get("contract_version"),
                contract_sha256=doc.get("contract_sha256"), path=path)


def load_runtime(path: str = RUNTIME_PATH, sha_path: str = RUNTIME_SHA_PATH) -> dict:
    """The runtime copy of the task's numbers, checked against its sha256 file (as the cell does at start)."""
    raw = open(path, "rb").read()
    want = open(sha_path, "rb").read().decode("ascii").split()[0]
    if _sha(raw) != want:
        raise RuntimeError(f"{path}: sha256 {_sha(raw)} != {want}")
    doc = json.loads(raw.decode("ascii"))
    if doc.get("schema") != RUNTIME_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}")
    vals = doc["values"]
    missing = [k for k in RUNTIME_KEYS if k not in vals]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    return dict(values=vals, sha256=_sha(raw), path=path)


def load_site(path: str) -> dict:
    """A site description from a .json file or from the 'site.json' member of a .site.tar.xz package."""
    if path.endswith(".tar.xz"):
        import tarfile
        with tarfile.open(path, "r:xz") as tf:
            return json.loads(tf.extractfile("site.json").read().decode("utf-8"))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _report_valid(rep) -> tuple:
    """A report the grader can use: a dict with a finite time, an inlet string, an integer (or no) frame id and a
    finite (or no) onset.  A non-finite number voids the whole call (the robot API: INVALID_ARGS, not counted)."""
    if not isinstance(rep, dict):
        return False, "no report"
    if P.fnum(rep.get("t")) is None:
        return False, "report time not finite"
    on = rep.get("onset_min")
    if on is not None and P.fnum(on) is None:
        return False, "onset not a finite number"
    fid = rep.get("frame_id")
    if fid is not None and (isinstance(fid, bool) or not isinstance(fid, int)):
        return False, "frame id not an integer"
    if not isinstance(rep.get("inlet"), str):
        return False, "no inlet"
    ir = rep.get("item_ref")
    if ir is not None and not isinstance(ir, str):
        return False, "item reference not a string"
    return True, "ok"


def _frame(rec: dict, fid):
    if fid is None or isinstance(fid, bool) or not isinstance(fid, int):
        return None
    for f in _l(rec.get("frames")):
        if isinstance(f, dict) and not isinstance(f.get("frame_id"), bool) and f.get("frame_id") == fid:
            return f
    return None


def _ref(s) -> str | None:
    return s.strip().upper() if isinstance(s, str) else None


def _footprint(site: dict, rec: dict, name: str):
    """The pouch's own projected area: the site's size (truth), else the record's footprint."""
    for it in _l(site.get("small_items")):
        if isinstance(it, dict) and it.get("id") == name:
            sz = _l(it.get("size"))
            if len(sz) >= 2:
                a, b = P.fnum(sz[0]), P.fnum(sz[1])
                if a is not None and b is not None and a > 0 and b > 0:
                    return a * b, "site"
    fp = P.fnum(_d(_d(rec.get("items")).get(name)).get("footprint_m2"))
    return (fp, "record") if (fp is not None and fp > 0) else (None, "none")


def _unit_inlets(rec: dict, unit, blk) -> list:
    mod = unit[4:] if isinstance(unit, str) and unit.startswith("cau_") else P.module_of_inlet(blk or "")
    out = set([blk] if blk else [])
    for f, meta in _d(rec.get("faces")).items():
        if P.face_kind(f) == "return_inlet" and (_d(meta).get("owner") == unit or P.module_of_inlet(f) == mod):
            out.add(f)
    for f in _d(_d(rec.get("final")).get("inlets")):
        if P.module_of_inlet(f) == mod:
            out.add(f)
    return sorted(out)


def _docking(rec: dict, smp: list, period: float, names: list) -> dict:
    """The docking H judges.  Charge: the lowest over every DOCKED moment -- the record's first DOCKED and each later
    docking seen in the samples (an undocked -> docked transition, its charge read from that sample), so charging on
    the berth between two dockings earns nothing.  Docked at the end by every indicator.  No item in the gripper from
    the final docking on; no item touching both robot and dock from the first DOCKED on (the record's flag)."""
    dk = _d(rec.get("dock"))
    out = dict(t=P.fnum(dk.get("t_docked")), t_final=None, battery=P.fnum(dk.get("battery_pct_at_docked")), redocked=False,
               docked_end=False, grip_after=False, contact_after=False, ok=False)
    if out["t"] is None:
        return out
    t_d = out["t"]
    out["t_final"] = t_d
    fin_docked = _d(rec.get("final")).get("docked")
    out["docked_end"] = bool(dk.get("docked_at_end") is True and fin_docked is not False
                             and (not smp or smp[-1].get("docked") is True))
    bats, was = [out["battery"]], True
    for s in smp:
        t = float(s["t"])
        if t <= t_d + period + 1e-6:
            continue
        d = s.get("docked") is True
        if not d:
            out["redocked"] = True
        elif not was:
            out["t_final"] = t
            bats.append(P.fnum(s.get("battery_pct")))
        was = d
    out["battery"] = None if any(b is None for b in bats) else min(bats)
    t0 = out["t_final"]
    out["grip_after"] = any(P.in_grip(P.item_state(s, n)) for s in smp if float(s["t"]) >= t0 - 1e-9 for n in names)
    out["contact_after"] = any(v is not False for v in _d(dk.get("item_contacts_after_dock")).values())
    out["ok"] = bool(out["docked_end"] and out["battery"] is not None and not out["grip_after"] and not out["contact_after"])
    return out


# ------------------------------------------------------------------------------------------------ the grade
def grade(record, site, ref_energy_wh=None, config: dict | None = None, runtime: dict | None = None) -> dict:
    """The verdict of one site: score (R x the item sum), items, R, binary outcomes, details, finite metrics."""
    C = config or load_config()
    RT = runtime or load_runtime()
    V = C["values"]
    W = V["grader.weights"]
    rec = _d(record)
    site = _d(site)
    T = _d(site.get("truth"))
    period = float(RT["values"]["cell.record_period_s"])
    w = float(V["grader.window_s"])
    smp = P.samples(rec)
    t_last = float(smp[-1]["t"]) if smp else 0.0
    t_end = P.fnum(_d(rec.get("times")).get("end"), t_last)
    blk = T.get("blocked_inlet") if isinstance(T.get("blocked_inlet"), str) else None
    if blk is None or P.face_kind(blk) != "return_inlet":
        raise ValueError("the site description has no hidden truth (truth.blocked_inlet): a grader input error, not a score")
    unit = T.get("blocked_unit")
    pouches = [p for p in _l(T.get("pouches")) if isinstance(p, str)]
    names = sorted(set(["blocker"] + pouches + [k for k in _d(rec.get("items")) if isinstance(k, str)]))
    win = P.window(smp, t_end, w)
    complete = P.window_complete(win, t_end, w, period)
    det, met = {}, {}
    met.update(record_samples=len(smp), record_period_s=period, window_s=w, window_samples=len(win),
               window_complete=complete, end_s=t_end)

    # ---- D
    Dp, Ev = V["grader.D.parts"], V["grader.D.evidence"]
    rep = rec.get("report")
    rep_ok, rep_why = _report_valid(rep)
    inl = P.inlet_from_report(rep.get("inlet")) if rep_ok else None
    ev_ok, ev_why, entry = False, rep_why, None
    if rep_ok:
        fr = _frame(rec, rep.get("frame_id"))
        if fr is None:
            ev_why = "the cited frame is not in the record"
        elif fr.get("camera") not in Ev["cameras"]:
            ev_why = f"camera {fr.get('camera')!r} is not an evidence camera"
        elif not (P.fnum(fr.get("t"), P.INF) <= P.fnum(rep.get("t"), -P.INF) + 1e-6):
            ev_why = "the frame was taken after the report"
        else:
            entry = _d(fr.get("evidence")).get(inl) if inl else None
            ev_ok, ev_why = P.evidence_ok(entry, Ev, need_cover=(inl is not None and inl == blk))
    d_inlet = bool(rep_ok and ev_ok and blk is not None and inl == blk)
    d_item = bool(d_inlet and _ref(T.get("blocker_ref")) is not None and _ref(rep.get("item_ref")) == _ref(T.get("blocker_ref")))
    onset_err = None
    if d_inlet:
        o, to = P.fnum(rep.get("onset_min")), P.fnum(T.get("onset_min"))
        onset_err = (o - to) if (o is not None and to is not None) else None
    d_onset = bool(d_inlet and onset_err is not None and abs(onset_err) <= float(V["grader.D.onset_tol_min"]) + 1e-9)
    D = (float(Dp["inlet"]) if d_inlet else 0.0) + (float(Dp["item"]) if d_item else 0.0) + (float(Dp["onset"]) if d_onset else 0.0)
    det["D"] = dict(report_valid=rep_ok, report_why=rep_why, inlet_reported=inl, evidence=ev_why, inlet_ok=d_inlet,
                    item_ok=d_item, onset_err_min=onset_err)
    e = _d(entry)
    met.update(D_report_valid=rep_ok, D_evidence_ok=ev_ok, D_inlet_ok=d_inlet, D_item_ok=d_item, D_onset_ok=d_onset,
               D_onset_err_min=(abs(onset_err) if onset_err is not None else NA),
               D_evidence_dist_m=P.fnum(e.get("dist_m"), NA), D_evidence_incidence_deg=P.fnum(e.get("incidence_deg"), NA),
               D_evidence_long_px=P.fnum(e.get("long_px"), NA), D_evidence_occluded=P.fnum(e.get("occluded"), NA),
               D_evidence_blocker_cover=P.fnum(e.get("blocker_cover"), NA))

    # ---- U1
    U1c = V["grader.U1"]
    if blk:
        dur, a, _b = P.longest_run(smp, lambda s: P.cover(s, blk) <= float(U1c["max_cover"]) + 1e-12, period)
    else:
        dur, a = 0.0, None
    U1 = bool(dur >= float(U1c["min_s"]) - 1e-9)
    met.update(U1_clear_longest_s=dur, U1_first_clear_s=(a if a is not None else NA))

    # ---- H
    Hd = _docking(rec, smp, period, names)
    H = bool(U1 and Hd["ok"] and Hd["battery"] >= float(V["grader.H.min_battery_pct"]) - 1e-9)
    det["H"] = dict(Hd)
    met.update(H_docked_at_s=(Hd["t"] if Hd["t"] is not None else NA), H_final_docked_at_s=(Hd["t_final"] if Hd["t_final"] is not None else NA),
               H_battery_pct=(Hd["battery"] if Hd["battery"] is not None else NA),
               H_docked_end=Hd["docked_end"], H_redocked=Hd["redocked"], H_grip_after_dock=Hd["grip_after"],
               H_item_contact_dock=Hd["contact_after"])

    # ---- U2 and K (one window)
    kmin = float(V["grader.K.min_seal_m2"])

    def blk_state(s):
        return P.item_state(s, "blocker")

    def on_screen(s):
        st = blk_state(s)
        return (not P.in_grip(st)) and P.max_seal(st, "holding_screen")[0] >= kmin - 1e-12

    clear_last = bool(complete and blk and all(P.cover(s, blk) <= float(V["grader.U2.max_cover"]) + 1e-12 for s in win))
    screen_branch = bool(complete and P.holds_throughout(win, on_screen))
    grip_branch = bool(complete and P.holds_throughout(win, lambda s: P.in_grip(blk_state(s))))
    U2 = bool(clear_last and (screen_branch or grip_branch))
    K = screen_branch
    k_seal = [0.0 if P.in_grip(blk_state(s)) else P.max_seal(blk_state(s), "holding_screen")[0] for s in win]
    met.update(U2_clear_last=clear_last, U2_screen_branch=screen_branch, U2_grip_branch=grip_branch,
               K_seal_min_m2=(min(k_seal) if k_seal else NA), K_seal_end_m2=(k_seal[-1] if k_seal else NA))

    # ---- V
    fin = _d(rec.get("final"))
    fans = fin.get("fans")
    units = set(k for k in _d(site.get("cabin_air_units")) if isinstance(k, str))
    last_fans = _d(smp[-1].get("fans")) if smp else {}
    fans_ok = bool(isinstance(fans, dict) and fans and all(v is True for v in fans.values()) and units <= set(fans)
                   and all(v is True for v in last_fans.values()))
    viol = rec.get("violations")
    n_viol = len(viol) if isinstance(viol, list) else 1
    u_inlets = _unit_inlets(rec, unit, blk)
    end_cov = _d(fin.get("inlets"))
    last = smp[-1] if smp else {}

    def end_cover(i):                                   # final.inlets, else the last sample; a bad final value reads as covered
        if i in end_cov:
            v = P.fnum(end_cov.get(i))
            return v if (v is not None and 0.0 <= v <= 1.0) else 1.0
        return P.cover(last, i)

    cov_max = max([end_cover(i) for i in u_inlets], default=1.0)
    inlets_ok = bool(u_inlets and cov_max <= float(V["grader.V.max_unit_inlet_cover"]) + 1e-12)
    Vok = bool(U2 and fans_ok and n_viol == 0 and inlets_ok)
    det["V"] = dict(fans_running=fans_ok, violations=n_viol, unit_inlets=u_inlets, unit_inlet_cover_max=cov_max)
    met.update(V_fans_running=fans_ok, V_violations=n_viol, V_unit_inlet_cover_max=cov_max)

    # ---- L
    crit = V["grader.L.criterion"]
    fmin, amin = float(V["grader.L.min_own_fraction"]), float(V["grader.L.min_seal_m2_alt"])
    LF, LC = V["grader.L.formula"], V["grader.loss_of_control"]
    rec_loss = _d(rec.get("loss_of_control"))
    on, on_frac, on_abs, lost, Ld = {}, {}, {}, {}, {}
    for p in pouches:
        fp, fp_src = _footprint(site, rec, p)
        pad = [0.0 if P.in_grip(P.item_state(s, p)) else P.max_seal(P.item_state(s, p), "pouch_pad")[0] for s in win]
        on_abs[p] = bool(complete and pad and min(pad) >= amin - 1e-12)
        on_frac[p] = bool(complete and pad and fp is not None and min(pad) / fp >= fmin - 1e-12)
        on[p] = on_frac[p] if crit == "own_footprint_fraction" else on_abs[p]
        t_lost = P.loss_of_control(smp, p, float(LC["disp_m"]), float(LC["min_s"]))
        lost[p] = bool(t_lost is not None or p in rec_loss)
        Ld[p] = dict(footprint_m2=fp, footprint_from=fp_src, pad_seal_min_m2=(min(pad) if pad else None),
                     own_fraction_min=(min(pad) / fp if (pad and fp) else None), on_frac80=on_frac[p], on_abs=on_abs[p],
                     lost_from_samples=t_lost, lost_in_record=(p in rec_loss))
        met[f"L_{p}_seal_min_m2"] = min(pad) if pad else NA
        met[f"L_{p}_own_fraction_min"] = (min(pad) / fp) if (pad and fp) else NA
        met[f"L_{p}_on_pad"] = on[p]
        met[f"L_{p}_lost"] = lost[p]
    n_tot = len(pouches)
    n_on = sum(1 for p in pouches if on[p])
    n_lost = sum(1 for p in pouches if lost[p] and not on[p])
    g = 1.0 if U1 else float(LF["g_without_U1"])
    L = (float(W["L"]) * g * max(0.0, n_on - float(LF["lost_unstowed_penalty"]) * n_lost) / n_tot) if n_tot else 0.0
    det["L"] = dict(criterion=crit, n_total=n_tot, n_on_pad=n_on, n_lost_unstowed=n_lost, g=g, pouches=Ld)
    met.update(L_n_total=n_tot, L_n_on_pad=n_on, L_n_on_pad_frac80=sum(on_frac.values()), L_n_on_pad_abs=sum(on_abs.values()),
               L_n_lost_unstowed=n_lost, L_g=g, L_criterion_own_fraction=(crit == "own_footprint_fraction"),
               L_site_without_pouches=(n_tot == 0))

    # ---- E
    Ec = V["grader.E"]
    gate = bool(K and n_tot > 0 and n_on == n_tot)
    fs = P.fan_stop_total(rec.get("fan_stops"), t_end)
    off_samples = sum(1 for x in smp for v in _d(x.get("fans")).values() if v is not True)   # cross-check only (not scored)
    e_wind = P.linear_credit(fs, float(Ec["wind_full_s"]), float(Ec["wind_zero_s"]))
    energy, ref = P.fnum(rec.get("energy_wh")), P.fnum(ref_energy_wh)
    ratio = energy / ref if (energy is not None and energy >= 0.0 and ref is not None and ref > 0.0) else None
    e_elec = P.linear_credit(ratio, float(Ec["elec_full_ratio"]), float(Ec["elec_zero_ratio"])) if ratio is not None else 0.0
    E = (float(Ec["w_elec"]) * e_elec + float(Ec["w_wind"]) * e_wind) if gate else 0.0
    det["E"] = dict(gate=gate, fan_stop_s=fs, e_wind=e_wind, energy_ratio=ratio, e_elec=e_elec)
    met.update(E_gate=gate, E_fan_stop_s=fs, E_fan_off_s_from_samples=off_samples * period, E_e_wind=e_wind, E_e_elec=e_elec,
               E_energy_ratio=(ratio if ratio is not None else NA),
               E_energy_wh=(energy if energy is not None else NA), E_ref_energy_wh=(ref if ref is not None else NA))

    # ---- R
    hard = P.hard_collisions(rec.get("hard_contacts"), float(V["grader.R.hard_speed_m_s"]), float(V["grader.R.hard_force_N"]), FORCE_CLASSES)
    keep = P.fnum(rec.get("keepout_s"), P.INF)
    R = float(V["grader.R.factor"]) if (hard > int(V["grader.R.max_hard_collisions"]) or keep >= float(V["grader.R.keepout_s"]) - 1e-9) else 1.0
    det["R"] = dict(hard_collisions=hard, keepout_s=(keep if math.isfinite(keep) else None))
    met.update(R_hard_collisions=hard, R_keepout_s=(keep if math.isfinite(keep) else NA), R_contacts=len(_l(rec.get("hard_contacts"))))

    # ---- sum
    items = dict(D=D, U1=float(W["U1"]) if U1 else 0.0, H=float(W["H"]) if H else 0.0, U2=float(W["U2"]) if U2 else 0.0,
                 V=float(W["V"]) if Vok else 0.0, K=float(W["K"]) if K else 0.0, L=L, E=E)
    raw = sum(items[k] for k in ITEMS)
    score = min(1.0, max(0.0, round(R * raw, 12)))
    for k in ITEMS:
        met[f"item_{k}"] = items[k]
    met.update(score=score, raw=raw, R=R)
    for k, v in _d(rec.get("metrics")).items():
        f = P.fnum(v)
        if f is not None and isinstance(k, str):
            met["cell_" + k] = f
    binary = dict(D_inlet=d_inlet, D_item=d_item, D_onset=d_onset, U1=U1, H=H, U2=U2, V=Vok, K=K, R=(R == 1.0), E_gate=gate,
                  e_elec_full=e_elec >= 1.0 - 1e-12, e_wind_full=e_wind >= 1.0 - 1e-12)
    for p in pouches:
        binary[f"pad:{p}"] = on[p]
        binary[f"lost:{p}"] = lost[p]
    return dict(score=score, raw=round(raw, 12), R=R, items={k: round(items[k], 12) for k in ITEMS}, binary=binary,
                weights={k: float(W[k]) for k in ITEMS}, detail=det, metrics=P.finite_metrics(met),
                provenance=dict(grader=GRADER, grader_config_sha256=C.get("sha256"), contract_version=C.get("contract_version"),
                                runtime_sha256=RT.get("sha256"), record_schema=rec.get("schema"), site_id=rec.get("site_id"),
                                L_criterion=crit))


# ------------------------------------------------------------------------------------------------ selection
def margins(record, graded: dict, config: dict | None = None, runtime: dict | None = None) -> dict:
    """Selection margins of a reference run: time to DOCKED (else program end), charge at DOCKED,
    frames as a fraction of each budget, capture calls, hard collisions, keep-out seconds, fan-stop seconds."""
    C = config or load_config()
    RT = runtime or load_runtime()
    S, RV = C["values"]["grader.selection"], RT["values"]
    rec = _d(record)
    m = _d(graded.get("metrics"))
    bu = _d(rec.get("budgets_used"))
    t_done = P.fnum(_d(rec.get("dock")).get("t_docked"))
    if t_done is None:
        t_done = P.fnum(_d(rec.get("times")).get("program_end"))
    lim_t = min(float(S["time_s"]), float(S["time_frac"]) * float(RV["cell.budget.sim_time_s"]))
    bat = m.get("H_battery_pct", NA)
    rows = dict(
        time_s=(t_done, lim_t, t_done is not None and t_done <= lim_t + 1e-9),
        dock_battery_pct=(None if bat == NA else bat, float(S["dock_battery_pct"]), bat != NA and bat >= float(S["dock_battery_pct"]) - 1e-9),
        colour_frames=(P.fnum(bu.get("colour_frames")), float(S["frames_frac"]) * float(RV["cell.budget.colour_frames"]), None),
        depth_frames=(P.fnum(bu.get("depth_frames")), float(S["frames_frac"]) * float(RV["cell.budget.depth_frames"]), None),
        captures=(P.fnum(bu.get("capture_calls")), float(S["captures_max"]), None),
        hard_collisions=(m.get("R_hard_collisions"), float(S["hard_collisions_max"]), None),
        keepout_s=(P.fnum(rec.get("keepout_s")), float(S["keepout_s_max"]), None),
        fan_stop_s=(m.get("E_fan_stop_s"), float(S["fan_stop_s_max"]), None))
    out = {}
    for k, (v, lim, ok) in rows.items():
        if ok is None:
            ok = v is not None and v <= lim + 1e-9
        out[k] = dict(value=v, limit=lim, ok=bool(ok))
    return out


def chain_pass(graded: dict, margins_out: dict) -> dict:
    """Pass under the rules (raw score 1 and every margin) and the 'chain' pass that leaves out what the blocker's
    stowage decides (K, U2, V that hangs on U2, E's gate): diagnosis, removal, docking, every pouch, R, fans,
    no violation, e_wind full and the margins."""
    it, b, W, m = graded["items"], graded["binary"], graded["weights"], graded["metrics"]
    mok = all(v["ok"] for v in margins_out.values())
    full = abs(graded["score"] - 1.0) < 1e-9 and mok
    chain = bool(abs(it["D"] - W["D"]) < 1e-9 and b["U1"] and b["H"] and abs(it["L"] - W["L"]) < 1e-9 and b["R"]
                 and m.get("V_fans_running") == 1.0 and m.get("V_violations") == 0.0 and b["e_wind_full"] and mok)
    return dict(full=bool(full), chain=chain, margins_ok=bool(mok))


def mean_score(site_scores, contract_ok) -> float:
    """caretaking_score = contract_ok x the mean of the site scores (each clipped to [0, 1]; non-finite reads as 0)."""
    sc = [min(1.0, max(0.0, P.fnum(s, 0.0))) for s in (site_scores or [])]
    return (1.0 if contract_ok else 0.0) * (sum(sc) / len(sc) if sc else 0.0)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="grade one cell record against its site")
    ap.add_argument("record")
    ap.add_argument("site", help="site.json or <site>.site.tar.xz")
    ap.add_argument("--ref-energy", type=float, default=None, help="the reference's energy on this site [Wh]")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--runtime", default=RUNTIME_PATH)
    ap.add_argument("--margins", action="store_true")
    a = ap.parse_args(argv)
    C = load_config(a.config)
    RT = load_runtime(a.runtime, os.path.join(os.path.dirname(a.runtime), "site_spec.sha256"))
    with open(a.record, encoding="utf-8") as fh:
        rec = json.load(fh)
    g = grade(rec, load_site(a.site), a.ref_energy, C, RT)
    out = dict(score=g["score"], items=g["items"], R=g["R"], binary=g["binary"], metrics=g["metrics"], provenance=g["provenance"])
    if a.margins:
        mg = margins(rec, g, C, RT)
        out.update(margins=mg, chain_pass=chain_pass(g, mg))
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
