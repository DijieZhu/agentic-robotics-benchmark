"""The 40 s constant-force fixture of the site generator (the hold-force rule of the sites) and
one blocker candidate's fixture record.

Here: the conditions of the final hold-force rule, the pull-release-pull record of the P tier (recorded only, not a
condition), the sheet of the candidate (``cabincell.site.items``), the site's angle convention (``bench``), the record in station_site/2 form
(``fixture_records.blocker``), CPU time per candidate.

One run (``run_const``): the sheet settles ``SETTLE_S`` under suction, then a constant force F acts at the handle's
site along the pull direction (the bar is free to turn) for 40 s; removed = the sealed area is at most 10 % of its
value at the force start; a removed run stops there.

Conditions (site_spec world.blocker.fixture):
  S tier (element type, 5.5-8.5 N): every pull -- strap normal, strap oblique 30 / 45 / 60 deg toward both chain ends,
         the along-chain drag toward both ends, loop normal, loop outward 45 deg -- at 0.61 N must NOT remove;
  P tier (pad type, 1.15-1.60 N): strap normal 0.37 N must NOT remove; loop normal 0.22 N MUST remove.
  P tier record only (world.blocker.fixture_repeat_record): strap normal 0.37 N for 20 s, released 5 s, 20 s again.
Every condition runs on every candidate (no short circuit) so the record says what a rejected candidate violates.

A candidate: dict(tier 'S' | 'P', theta_deg, offset_m [a, b], size, mass, n_inlets, r_face, flow_cfm (40), and
optionally family, hold_target).  ``draw_candidate`` = the natural candidate stream (natural
placement, a family sheet, a two- or three-inlet unit with equal odds; P: a hold target drawn in the window and r_face
solved on the bench's settled sealed area, then one fixed-point step; S: r_face uniform in the element range).
``evaluate`` -> the result with the record (``record``) and the decision (``accepted``, ``rejected_by``): rejected if
the settled sealed area is outside 0.026-0.036 m^2, the measured hold outside the tier window, a P candidate's r_face
outside the pad range of its unit size, or a condition fails.

usage: python -I -B fixture.py gen <S|P> <first> <count> <out.jsonl> [--label L]
       python -I -B fixture.py one <candidate.json> <out.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
    __package__ = "review.sitegen.fixture"
    import review.sitegen.fixture  # noqa: F401

import numpy as np

from . import bench as B
from . import VERSION

import mujoco  # noqa: E402

from cabincell.site import items as IT  # noqa: E402
from cabincell.site import placement as PL  # noqa: E402
from cabincell.site import spec as SPEC  # noqa: E402

DEFAULT_LABEL = "fixture"
RECORD_PULL = {"normal": "strap_normal", "drag_e0": "drag_along_chain", "drag_e4": "drag_along_chain"}


def _record_pull(where: str, pull: str) -> str:
    if where == "loop":
        return "loop_normal" if pull == "normal" else "loop_outward"
    if pull.startswith("tilt"):
        return "strap_oblique"
    return RECORD_PULL[pull]


def fixture_values(values=None) -> dict:
    v = values or SPEC.values()
    return dict(fix=v["world.blocker.fixture"], rep=v["world.blocker.fixture_repeat_record"], hold=v["world.blocker.hold_N"],
                cover=v["world.blocker.placement"]["covered_m2"])


def conditions(tier: str, values=None) -> list:
    """[(where, pull, F [N], must_remove)] of a tier (see the module doc)."""
    f = fixture_values(values)["fix"]
    if tier == "P":
        return [("strap", "normal", float(f["P_strap_normal_not_removed_N"]), False),
                ("loop", "normal", float(f["P_loop_normal_removed_N"]), True)]
    if tier == "S":
        F = float(f["S_any_pull_not_removed_N"])
        return [("strap", p, F, False) for p in B.PULLS_STRAP] + [("loop", p, F, False) for p in B.PULLS_LOOP]
    raise ValueError(tier)


def run_const(cand: dict, where: str, pull: str, F: float, T: float | None = None, settle: float = B.SETTLE_S,
              schedule=None) -> dict:
    """One constant-force run (see the module doc).  ``schedule``: [(t0, t1)] force-on intervals after the settle
    (default the whole T); returns dict(removed, t_removed_s, sealed_min_frac, sealed_end_frac, hold0_N, sealed0_m2,
    Pc0_Pa, site_moved_m)."""
    T = float(fixture_values()["fix"]["duration_s"]) if T is None else float(T)
    on = schedule or [(0.0, T)]
    t_end = max(t1 for _, t1 in on)
    xml, excl, half, R = B.sheet_xml(cand)
    m = B.item_model(xml, excl)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "blk_grasp_centre" if where == "strap" else "blk_grasp_corner")
    d = mujoco.MjData(m)
    net = B.make_network(float(cand["r_face"]), int(cand["n_inlets"]), float(cand.get("flow_cfm", 40.0)))
    suc, bodies = B.make_suction(m, net, half)
    sb = int(m.site_bodyid[site])
    u = B.pull_dir(pull, R)
    W = np.zeros((m.nbody, 6))
    dt = m.opt.timestep
    k, t = 0, 0.0
    cov0 = None
    res = dict(where=where, pull=pull, F_N=float(F), removed=False, t_removed_s=None)
    site0 = None
    smin = 1.0
    nmax = int(round((settle + t_end) / dt))
    state = {}

    def force(W_):
        if t >= settle:
            ts = t - settle
            if any(t0 <= ts < t1 for t0, t1 in on):
                f = float(F) * u
                W_[sb, :3] += f
                W_[sb, 3:] += np.cross(d.site_xpos[site] - d.xipos[sb], f)
    while k < nmax:
        if t >= settle and cov0 is None:
            # the force-start state is read after the step's suction evaluation below; flag it here
            state["start"] = True
        mujoco.mj_step1(m, d)
        W[:] = 0.0
        suc.accumulate(m, d, W)
        if state.pop("start", False):
            cov0 = float(suc.covered["g0"])
            res.update(sealed0_m2=round(cov0, 6), hold0_N=round(float(suc.hold_force(bodies)), 5), Pc0_Pa=round(float(net.Pc), 3))
            site0 = d.site_xpos[site].copy()
        force(W)
        d.xfrc_applied[:] = W
        mujoco.mj_step2(m, d)
        k += 1
        t = k * dt
        if k % 5 == 0:
            net.advance(0.01)
            suc.update_networks()
        if cov0 is not None and k % 10 == 0:
            fr = float(suc.covered["g0"]) / max(cov0, 1e-12)
            smin = min(smin, fr)
            if fr <= 0.10:
                res.update(removed=True, t_removed_s=round(t - settle, 3))
                break
    res["sealed_min_frac"] = round(smin, 4)
    res["sealed_end_frac"] = round(float(suc.covered["g0"]) / max(cov0 or 1e-12, 1e-12), 4)
    res["site_moved_m"] = round(float(np.linalg.norm(d.site_xpos[site] - site0)), 5) if site0 is not None else None
    res["final_qpos_sha256"] = hashlib.sha256(np.ascontiguousarray(d.qpos).tobytes()).hexdigest()
    return res


def repeat_pull(cand: dict, values=None) -> dict:
    """The P tier's pull-release-pull record: strap normal at pull_N for pull_s, off for release_s, on for pull_s."""
    r = fixture_values(values)["rep"]
    p, s = float(r["pull_s"]), float(r["release_s"])
    out = run_const(cand, "strap", "normal", float(r["pull_N"]), T=2 * p + s, schedule=[(0.0, p), (p + s, 2 * p + s)])
    out.update(pull_N=float(r["pull_N"]), pull_s=p, release_s=s)
    return out


# ------------------------------------------------------------------------------------------------ candidates
def seed_of(*parts) -> int:
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def draw_candidate(tier: str, k: int, label: str = DEFAULT_LABEL, values=None) -> dict:
    """The k-th natural candidate of a tier (deterministic in (label, tier, k)); see the module doc."""
    v = values or SPEC.values()
    rng = np.random.default_rng(seed_of(label, tier, k))
    sheet = IT.draw_sheet(rng)
    pl = PL.natural_placement(rng, v)
    n = int(rng.choice([2, 3]))
    c = dict(tier=tier, k=int(k), label=label, family=sheet["family"], size=sheet["size"], mass=sheet["mass"], colour=sheet["colour"],
             theta_deg=pl["theta_deg"], offset_m=pl["offset_m"], n_inlets=n, flow_cfm=40.0)
    if tier == "P":
        c["hold_target"] = float(rng.uniform(*PL.tier_window("P", v)))
    else:
        c["r_face"] = float(rng.uniform(*PL.r_face_range("element", n, v)))
    return c


def solve_r_face(cand: dict) -> tuple:
    """P: r_face for the hold target on the bench's settled sealed area (one fixed-point step).  Returns
    (r_face, settled dict)."""
    s0 = B.settle(cand, 66.0)
    A = s0["A_sealed"]
    if A <= 1e-4:
        return float("nan"), s0
    n, q = int(cand["n_inlets"]), float(cand.get("flow_cfm", 40.0))
    r = B.r_face_for_hold(cand["hold_target"], A, n, q)
    s1 = B.settle(dict(cand, r_face=r), r)
    if abs(s1["A_sealed"] - A) > 1e-4:
        A = s1["A_sealed"]
        r = B.r_face_for_hold(cand["hold_target"], A, n, q)
        s1 = B.settle(dict(cand, r_face=r), r)
    return r, s1


def evaluate(cand: dict, values=None) -> dict:
    """Settle (P: solve r_face), run every condition (and the P tier's repeat record), decide.  Returns the candidate
    with ``record`` (station_site/2 fixture_records.blocker), ``runs`` (every run's detail), ``accepted``,
    ``rejected_by``, timing."""
    v = values or SPEC.values()
    c0, w0 = time.process_time(), time.time()
    tier = cand["tier"]
    out = dict(cand)
    out["fixture_version"] = VERSION
    rej = []
    if tier == "P" and cand.get("r_face") is None:
        r, s = solve_r_face(cand)
        out["r_face"] = r
    else:
        r = float(cand["r_face"])
        s = B.settle(cand, r)
    A = s["A_sealed"]
    out.update(A_settled_m2=A)
    lo, hi = fixture_values(v)["cover"]
    if not (lo <= A <= hi):
        rej.append("cover")
    if tier == "P" and A > 1e-4:
        rr = PL.r_face_range("pad", int(cand["n_inlets"]), v)
        if not (rr[0] <= out["r_face"] <= rr[1]):
            rej.append("r_face_range")
    runs, conds = [], []
    if A > 1e-4:
        for where, pull, F, must in conditions(tier, v):
            rr_ = run_const(dict(out), where, pull, F)
            rr_["must_remove"] = must
            rr_["ok"] = bool(rr_["removed"] == must)
            runs.append(rr_)
            conds.append(dict(pull=_record_pull(where, pull), force_N=float(F), duration_s=float(fixture_values(v)["fix"]["duration_s"]),
                              want="removed" if must else "not_removed", removed=bool(rr_["removed"]), t_removed_s=rr_["t_removed_s"]))
            if not rr_["ok"]:
                rej.append(f"{where}:{pull}")
        h0 = runs[0]["hold0_N"]
        out["hold_measured_N"] = h0
        wlo, whi = (float(x) for x in fixture_values(v)["hold"][tier])
        if not (wlo <= h0 <= whi):
            rej.append("hold")
        hs, pcb, pc0 = B.hold_static(out["r_face"], A, int(cand["n_inlets"]), float(cand.get("flow_cfm", 40.0)))
        out.update(hold_static_N=hs, Pc_blocked_Pa=pcb, Pc_clear_Pa=pc0, dPc_Pa=pcb - pc0)
    else:
        rej.append("no_cover")
    rep = None
    if tier == "P" and A > 1e-4:
        rep = repeat_pull(dict(out), v)
        out["repeat_run"] = rep
    out["runs"] = runs
    out["rejected_by"] = rej
    out["accepted"] = not rej
    out["record"] = dict(tier=tier, conditions=conds,
                         repeat_pull=(dict(pull_N=rep["pull_N"], pull_s=rep["pull_s"], release_s=rep["release_s"], removed=bool(rep["removed"]))
                                      if rep is not None else None),
                         passed=not rej)
    out["chain_fold_deg"] = PL.chain_angle_fold(cand["theta_deg"])
    out["loop_over_opening"] = B.loop_over_opening(cand)
    out["cpu_s"] = round(time.process_time() - c0, 2)
    out["wall_s"] = round(time.time() - w0, 2)
    out["record_sha256"] = record_digest(out)
    return out


def record_digest(res: dict) -> str:
    """sha256 of everything a candidate's result says except its timing (the bit-for-bit comparison of two runs)."""
    keep = {k: v for k, v in res.items() if k not in ("cpu_s", "wall_s", "record_sha256", "host")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=float).encode()).hexdigest()


def candidate_from_site(site: dict, feats: dict | None = None) -> dict:
    """The fixture candidate of a station_site/2 site's blocker (its unit's size, r_face and flow scale)."""
    from station import data as SD
    feats = feats or SD.features(site.get("scene", "mission"))
    Bk = site["blocker"]
    unit = PL.unit_of(feats, Bk["inlet"])
    U = site["cabin_air_units"][unit]
    tier = "S" if U["build"] == "element" else "P"
    return dict(tier=tier, family=Bk.get("family"), size=list(Bk["size"]), mass=float(Bk["mass"]), theta_deg=float(Bk["theta_deg"]),
                offset_m=list(Bk["offset_m"]), n_inlets=len(PL.unit_inlets(feats, unit)), r_face=float(U["r_face"]),
                flow_cfm=40.0 * float(U.get("flow_scale", 1.0)), hinge_k=float(Bk.get("hinge_k", 0.002)),
                n_strips=int(Bk.get("n_strips", 5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("tier", choices=["S", "P"])
    g.add_argument("first", type=int)
    g.add_argument("count", type=int)
    g.add_argument("out")
    g.add_argument("--label", default=DEFAULT_LABEL)
    o = sub.add_parser("one")
    o.add_argument("cand")
    o.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        done = set()
        if os.path.exists(a.out):
            done = {json.loads(x)["k"] for x in open(a.out) if x.strip()}
        with open(a.out, "a") as fo:
            for k in range(a.first, a.first + a.count):
                if k in done:
                    continue
                r = evaluate(draw_candidate(a.tier, k, a.label))
                r["host"] = os.uname().nodename
                fo.write(json.dumps(r, default=float) + "\n")
                fo.flush()
                print(json.dumps(dict(k=k, tier=a.tier, fam=r["family"], theta=round(r["theta_deg"], 1), n=r["n_inlets"],
                                      hold=r.get("hold_measured_N"), acc=r["accepted"], rej=r["rejected_by"], cpu=r["cpu_s"])), flush=True)
    else:
        r = evaluate(json.load(open(a.cand)))
        r["host"] = os.uname().nodename
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
