"""One attempt through the generator's stages A-H (see the package doc), cost accounting, and the command line.

``run_attempt(label, kind, cell, k, j)`` -> the attempt record: dict(label, kind, cell, k, j, seed, seed_text,
passed, stage (the last stage reached), rejected {stage, cause, info} | None, stages {name: dict(cpu_s, ...)},
site_id, package_sha256, ...).  When the attempt passes, the site description (``site``) and the package bytes are
returned too (``run_attempt(..., keep=True)``).

Decisions of this package recorded here ([C]):
  * windows: the blocker's cover and hold windows (world.blocker.placement.covered_m2, world.blocker.hold_N) and the
    pouches' (world.pouch.placement) are judged on the scene's own sealed areas after the 600 s settle (the fixture
    bench differs from the scene by a few per cent); the fixture's record keeps the bench's
    pull outcomes and ``passed`` = every pull condition met (the bench's own cover / hold flags are kept in the
    attempt record as information);
  * no clutter (cargo bags) in trial and debug sites: the as-surveyed map does not know them and fly_to does not
    avoid unmapped obstacles; the site rule calls them picture clutter only;
  * the telemetry is drawn twice from the same sub-stream (sha256 of the attempt seed and 'telemetry'): a screen on the
    replica's sealed areas at stage A, the final draw on the in-scene areas at stage E.

Command line:
  python -I -B pipeline.py attempt <kind> <cell> <k> <j> <out_dir> [--schema site.schema.json]
  python -I -B pipeline.py candidate <kind> <cell> <k> <out_dir> [--max-attempts N] [--schema ...]
     (attempts j = 0, 1, ... until one passes; writes <out_dir>/<site_id>.site.tar.xz)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    __package__ = "gen"
    import gen  # noqa: F401

import numpy as np

from . import cells as CE
from . import checks as CK
from . import draw as DR
from . import freeze as FZ
from . import mission as MI
from . import settle as SE
from . import sitepack as SP
from . import VERSION

from cabincell.site import build as BU
from cabincell.site import spec as SPEC

STAGES = ("A_draw", "B_screen", "C_fixture", "D_settle", "E_telemetry", "F_check", "G_dry_run", "H_pack")


def _now():
    return time.process_time(), time.time()


def _cost(t0) -> dict:
    return dict(cpu_s=round(time.process_time() - t0[0], 3), wall_s=round(time.time() - t0[1], 3))


def telemetry_draw(site_t: dict, seed: int):
    """(telemetry, units patch, info) of the station service's draw on a site with truth.cover_m2 / telemetry.pouch_cover_m2 set; the
    sub-stream is fixed by the attempt seed."""
    import telemetry_synth as TS
    rng = np.random.default_rng(CE.derived_seed(seed, "telemetry"))
    return TS.draw_telemetry(site_t, rng)


def apply_telemetry(site_t: dict, tel: dict, patch: dict) -> dict:
    S = copy.deepcopy(site_t)
    S["telemetry"] = tel
    for uid, vv in patch.items():
        S["cabin_air_units"][uid].update(vv)
    S["truth"]["onset_min"] = tel["onset_min"]
    S["truth"]["advisory_points_elsewhere"] = bool(tel["advisory_expected"]["points_elsewhere"])
    S["truth"]["blocked_unit_over_limit"] = bool(tel["advisory_expected"]["blocked_unit_over_limit"])
    return S


def fixture_record(res: dict) -> dict:
    """station_site/2 fixture_records.blocker from the fixture's evaluation: the pull outcomes, passed = every pull
    condition met (the windows are the scene's, see the module doc)."""
    rec = copy.deepcopy(res["record"])
    rec["passed"] = all(bool(c["removed"]) == (c["want"] == "removed") for c in rec["conditions"])
    return rec


def default_fixture(site: dict) -> dict:
    from review.sitegen.fixture import fixture as FX
    return FX.evaluate(FX.candidate_from_site(site))


def provenance(spec_sha: str, gen_sha: str, label: str, j: int, seed_text: str) -> dict:
    return dict(generator_sha256=gen_sha, site_spec_sha256=spec_sha, spacekit="0.4.0", reference_build="v1.2",
                generator_version=VERSION, label=label, attempt=int(j), seed_text=seed_text)


def run_attempt(label: str, kind: str, cell: str, k: int, j: int, keep: bool = False, schema=None, fixture_fn=None,
                pre_s: float | None = None, dry_run_s: float | None = None, gen_sha: str | None = None) -> dict:
    """One attempt (module doc).  ``fixture_fn(site) -> evaluate result`` replaces the fixture (tests); ``pre_s`` /
    ``dry_run_s`` replace the site_spec durations (tests only: a site written so is not a valid site)."""
    v = SPEC.values()
    feats = DR.features()
    rec = dict(schema="caretaking.sitegen_attempt/1", label=label, kind=kind, cell=cell, k=int(k), j=int(j),
               seed_text=CE.seed_text(label, cell, k, j), seed=CE.attempt_seed(label, cell, k, j),
               candidate_seed=CE.candidate_seed(label, cell, k), host=os.uname().nodename, passed=False, stage=None,
               rejected=None, stages={}, generator_version=VERSION,
               durations=dict(pre_s=pre_s, dry_run_s=dry_run_s, test_mode=bool(pre_s is not None or dry_run_s is not None or fixture_fn is not None)))
    gen_sha = gen_sha or FZ.digest()
    rec["generator_sha256"] = gen_sha
    spec_sha = SPEC.sha256()
    rec["site_spec_sha256"] = spec_sha
    out = dict(record=rec)

    def reject(stage, cause, info=None):
        rec["rejected"] = dict(stage=stage, cause=cause, info=info or {})
        return out
    try:
        # ---------------------------------------------------------------- A: draw + telemetry screen
        rec["stage"] = "A_draw"
        t0 = _now()
        try:
            site, info = DR.draw(label, cell, k, j, kind, feats, v)
        except DR.Reject as R:
            rec["stages"]["A_draw"] = _cost(t0)
            return reject("A_draw", R.cause, R.info)
        import telemetry_synth as TS
        st = DR.with_truth(site, feats, info["replica_areas"], v)
        try:
            telemetry_draw(st, rec["seed"])
        except TS.Reject as R:
            rec["stages"]["A_draw"] = dict(_cost(t0), replica_holds=info["replica_holds"])
            return reject("A_draw", "telemetry_screen:" + R.cause, dict(R.info, causes=R.info.get("causes")))
        rec["stages"]["A_draw"] = dict(_cost(t0), replica_areas=info["replica_areas"], replica_holds=info["replica_holds"],
                                       chain_fold_deg=info["chain_fold_deg"], blocker_tries=info["blocker_tries"],
                                       cover_geometric_m2=info["cover_geometric_m2"])
        rec["site_id"] = site["site_id"]
        rec["draw"] = dict(blocked_inlet=site["blocker"]["inlet"], family=site["blocker"]["family"], theta_deg=site["blocker"]["theta_deg"],
                           offset_m=site["blocker"]["offset_m"], r_face=site["cabin_air_units"][f"cau_{site['cell']['module']}"]["r_face"],
                           pouches=[(P["id"], P["inlet"], P["columns"], P["end"]) for P in site["small_items"]],
                           stowage_units={mm: e["unit_site"] for mm, e in site["stowage_units"].items()})
        # ---------------------------------------------------------------- B: screen at t = 0
        rec["stage"] = "B_screen"
        t0 = _now()
        S0 = copy.deepcopy(st)
        S0["robot"] = dict(S0["robot"], present=False)
        m0, info0 = BU.build(S0)
        d0 = BU.initial_data(m0, info0)
        ck0 = CK.check_site(st, m0, d0, info0, feats, v, telemetry=False)
        fails = [f for f in ck0["failures"] if not f.startswith("arrangement:truth_") and not f.startswith("arrangement:advisory")
                 and not f.startswith("arrangement:blocked_unit_over")]
        rec["stages"]["B_screen"] = dict(_cost(t0), pregrasp_counts=ck0["pregrasp_counts"], station=ck0["station"])
        if fails:
            return reject("B_screen", fails[0], dict(failures=fails))
        # ---------------------------------------------------------------- C: fixture
        rec["stage"] = "C_fixture"
        t0 = _now()
        res = (fixture_fn or default_fixture)(st)
        frec = fixture_record(res)
        rec["stages"]["C_fixture"] = dict(_cost(t0), fixture_cpu_s=res.get("cpu_s"), bench_rejected_by=res.get("rejected_by"),
                                          bench_A_settled_m2=res.get("A_settled_m2"), bench_hold_N=res.get("hold_measured_N"),
                                          conditions=frec["conditions"], repeat_pull=frec["repeat_pull"],
                                          record_sha256=res.get("record_sha256"))
        if not frec["passed"]:
            bad = [f"{c['pull']}@{c['force_N']}" for c in frec["conditions"] if bool(c["removed"]) != (c["want"] == "removed")]
            return reject("C_fixture", "condition:" + bad[0], dict(failed=bad))
        # ---------------------------------------------------------------- D: settle 600 s
        rec["stage"] = "D_settle"
        t0 = _now()
        pre = SE.pre_settle(st, pre_s=pre_s)
        meas = pre["measures"]
        areas = {kk: float(vv["sealed_m2"]) for kk, vv in meas.items()}
        rec["stages"]["D_settle"] = dict(_cost(t0), measures=meas, settle_disp=pre["settle_disp"], sim_s=pre["sim_s"],
                                         state_sha256=pre["state_sha256"])
        probs = DR.window_problems(st, feats, areas, v)
        H = DR.holds(st, feats, areas, v)
        rec["stages"]["D_settle"]["network_holds_on_scene_areas"] = H
        if probs:
            return reject("D_settle", "scene_window:" + str(probs[0][0]), dict(problems=probs))
        # ---------------------------------------------------------------- E: telemetry on the in-scene areas
        rec["stage"] = "E_telemetry"
        t0 = _now()
        st2 = DR.with_truth(site, feats, areas, v)
        st2["truth"]["hold_N"] = float(meas["blocker"]["hold_N"])
        try:
            tel, patch, tinfo = telemetry_draw(st2, rec["seed"])
        except TS.Reject as R:
            rec["stages"]["E_telemetry"] = _cost(t0)
            return reject("E_telemetry", R.cause, R.info)
        S = apply_telemetry(st2, tel, patch)
        S["fixture_records"] = dict(blocker=frec)
        rec["stages"]["E_telemetry"] = dict(_cost(t0), tries=tinfo["tries"], causes=tinfo["causes"], side=tinfo["side"],
                                            advisory_log=tinfo["advisory_log"], csv_sha256=tinfo["csv_sha256"])
        # ---------------------------------------------------------------- F: check_site on the settled state
        rec["stage"] = "F_check"
        t0 = _now()
        st0 = pre["state0"]
        m1, d1, info1 = SE.robot_free_from_state0(S, st0)
        ck = CK.check_site(S, m1, d1, info1, feats, v, telemetry=True)
        S["pregrasp_records"] = ck["pregrasp_records"]
        rec["stages"]["F_check"] = dict(_cost(t0), pregrasp_counts=ck["pregrasp_counts"], station=ck["station"],
                                        telemetry=dict(side=ck["telemetry"]["side"], blocked_margin_sigma=ck["telemetry"]["blocked_margin_sigma"],
                                                       advisory_units=ck["telemetry"]["advisory_units"],
                                                       min_margin_sigma=_min_margin(ck["telemetry"]["units"])))
        if not ck["ok"]:
            return reject("F_check", ck["failures"][0], dict(failures=ck["failures"]))
        # ---------------------------------------------------------------- G: dry run 2400 s from state0
        rec["stage"] = "G_dry_run"
        t0 = _now()
        dry = SE.dry_run(S, st0, dry_run_s=dry_run_s)
        st0_bytes = SP.dumps(st0)
        st0_sha = hashlib.sha256(st0_bytes).hexdigest()
        S["settle"] = SE.settle_record(dict(sim_s=pre["sim_s"]), dry, st0_sha, v)
        rec["stages"]["G_dry_run"] = dict(_cost(t0), items=dry["items"], sim_s=dry["sim_s"], final_state_sha256=dry["final_state_sha256"],
                                          end=dry["end"])
        rec["dry_run"] = dict(passed=bool(dry["passed"]), items=dry["items"])
        if not dry["passed"]:
            bad = [kk for kk, x in dry["items"].items() if not x["passed"]]
            return reject("G_dry_run", "moved:" + bad[0], dict(items=dry["items"]))
        # ---------------------------------------------------------------- H: pack
        rec["stage"] = "H_pack"
        t0 = _now()
        S["truth"]["hold_N"] = float(meas["blocker"]["hold_N"])
        S["truth"]["cover_m2"] = float(meas["blocker"]["sealed_m2"])
        S["provenance"] = provenance(spec_sha, gen_sha, label, j, rec["seed_text"])
        if schema is not None:
            import schemalite as SL
            errs = SL.errors(S, schema)
            if errs:
                rec["stages"]["H_pack"] = _cost(t0)
                return reject("H_pack", "schema", dict(errors=errs[:5]))
        mem = MI.members(S, st0)
        data = SP.pack(mem, S["site_id"])
        probs = SP.verify(data, schema)
        import telemetry_synth as TS2
        probs += TS2.verify_members(S, {kk: vv for kk, vv in mem.items()})
        rec["stages"]["H_pack"] = dict(_cost(t0), bytes=len(data), members={kk: hashlib.sha256(vv).hexdigest() for kk, vv in sorted(mem.items())})
        if probs:
            return reject("H_pack", "package", dict(problems=probs))
        rec["package_sha256"] = hashlib.sha256(data).hexdigest()
        rec["passed"] = True
        rec["summary"] = summary(S, rec)
        if keep:
            out["site"] = S
            out["package"] = data
            out["state0"] = st0
        return out
    except Exception as exc:  # noqa: BLE001  (an attempt that crashes is recorded, never silently retried)
        rec["error"] = dict(type=type(exc).__name__, text=str(exc)[:500], trace=traceback.format_exc()[-3000:])
        return reject(rec["stage"] or "A_draw", "error:" + type(exc).__name__)
    finally:
        rec["cost_cpu_s"] = round(sum(float(x.get("cpu_s", 0.0)) for x in rec["stages"].values()), 3)


def _min_margin(units: dict) -> float | None:
    vals = []
    for d in units.values():
        for kk in ("max_margin_sigma", "pre_margin_sigma", "post_margin_sigma", "before_drift_margin_sigma", "end_margin_sigma"):
            x = d.get(kk)
            if x is not None:
                vals.append(abs(float(x)))
    return min(vals) if vals else None


def summary(S: dict, rec: dict) -> dict:
    """The attribute row of a site."""
    from cabincell.site import placement as PL
    feats = DR.features()
    m = S["cell"]["module"]
    B = S["blocker"]
    tel = S["telemetry"]
    blk_unit = f"cau_{m}"
    F = rec["stages"]["F_check"]
    pouch_hold = {}
    for P in S["small_items"]:
        pouch_hold[P["id"]] = dict(inlet=P["inlet"], columns=P["columns"], end=P["end"],
                                   hold_N=rec["stages"]["D_settle"]["measures"][P["id"]]["hold_N"],
                                   sealed_m2=rec["stages"]["D_settle"]["measures"][P["id"]]["sealed_m2"],
                                   registered=P["register_ref"] is not None)
    return dict(site_id=S["site_id"], cell=CE.cell_key(m, S["cell"]["tier"]), kind=S["kind"], k=S["cell"]["k"], attempt=rec["j"],
                tier=S["cell"]["tier"], blocked_inlet=B["inlet"], family=B["family"], colour=B["colour"],
                hold_N=S["truth"]["hold_N"], cover_m2=S["truth"]["cover_m2"], chain_fold_deg=PL.chain_angle_fold(B["theta_deg"]),
                theta_deg=B["theta_deg"], offset_m=B["offset_m"], short_axis_offset_m=B["offset_m"][1],
                r_face=S["cabin_air_units"][blk_unit]["r_face"], unit_build=S["cabin_air_units"][blk_unit]["build"],
                n_inlets=len(PL.unit_inlets(feats, blk_unit)), n_pouches=len(S["small_items"]), pouches=pouch_hold,
                advisory_points_elsewhere=S["truth"]["advisory_points_elsewhere"],
                advisory_units=F["telemetry"]["advisory_units"],
                blocked_unit_over_limit=S["truth"]["blocked_unit_over_limit"],
                blocked_unit_margin_sigma=F["telemetry"]["blocked_margin_sigma"],
                min_unit_margin_sigma=F["telemetry"]["min_margin_sigma"],
                lighting=S["lighting"], failed_light_module=S["truth"]["failed_light_module"],
                battery_pct=S["robot"]["battery_pct"], berth=S["robot"]["berth"],
                stowage_units={mm: e["unit_site"] for mm, e in S["stowage_units"].items()},
                moved_stowage_unit=S["truth"]["moved_stowage_unit"], onset_min=S["truth"]["onset_min"],
                drift=tel.get("drift"), register_entries=len(S["register"]["entries"]),
                strap_sheet_entries=sum(1 for e in S["register"]["entries"] if "carry strap" in e["features"]
                                        and e["class"] in ("soft stowage sack", "equipment cover", "crew towel")),
                unregistered=S["register"]["unregistered"],
                pregrasp_accepted={kk: vv["accepted"] for kk, vv in F["pregrasp_counts"].items()},
                fixture=[(c["pull"], c["force_N"], c["removed"], c["want"]) for c in S["fixture_records"]["blocker"]["conditions"]],
                repeat_pull_removed=(S["fixture_records"]["blocker"]["repeat_pull"] or {}).get("removed"),
                settle=S["settle"], package_sha256=rec["package_sha256"])


def run_candidate(label: str, kind: str, cell: str, k: int, out_dir: str, max_attempts: int = 60, schema=None) -> dict:
    """Attempts j = 0, 1, ... until one passes (sequential); every attempt record is written to
    <out_dir>/attempts/<site_id>_j<j>.json, the package to <out_dir>/<site_id>.site.tar.xz."""
    os.makedirs(os.path.join(out_dir, "attempts"), exist_ok=True)
    gen_sha = FZ.digest()
    sid = CE.site_id(kind, cell, k)
    for j in range(max_attempts):
        r = run_attempt(label, kind, cell, k, j, keep=True, schema=schema, gen_sha=gen_sha)
        with open(os.path.join(out_dir, "attempts", f"{sid}_j{j}.json"), "w") as fh:
            json.dump(r["record"], fh, indent=1, default=float)
        print(json.dumps(dict(site=sid, j=j, passed=r["record"]["passed"], rejected=r["record"]["rejected"] and
                              {kk: r["record"]["rejected"][kk] for kk in ("stage", "cause")}, cpu=r["record"]["cost_cpu_s"])), flush=True)
        if r["record"]["passed"]:
            with open(os.path.join(out_dir, f"{sid}{SP.SUFFIX}"), "wb") as fh:
                fh.write(r["package"])
            return r["record"]
    raise RuntimeError(f"{sid}: no attempt passed in {max_attempts}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("attempt")
    for x in ("kind", "cell"):
        a1.add_argument(x)
    a1.add_argument("k", type=int)
    a1.add_argument("j", type=int)
    a1.add_argument("out_dir")
    a1.add_argument("--schema")
    a2 = sub.add_parser("candidate")
    for x in ("kind", "cell"):
        a2.add_argument(x)
    a2.add_argument("k", type=int)
    a2.add_argument("out_dir")
    a2.add_argument("--max-attempts", type=int, default=60)
    a2.add_argument("--schema")
    a = ap.parse_args(argv)
    schema = json.load(open(a.schema)) if a.schema else None
    if a.schema:
        sys.path.insert(0, os.path.dirname(os.path.abspath(a.schema)))
    label = CE.LABELS[a.kind]
    if a.cmd == "attempt":
        r = run_attempt(label, a.kind, a.cell, a.k, a.j, keep=True, schema=schema)
        os.makedirs(a.out_dir, exist_ok=True)
        sid = CE.site_id(a.kind, a.cell, a.k)
        with open(os.path.join(a.out_dir, f"{sid}_j{a.j}.json"), "w") as fh:
            json.dump(r["record"], fh, indent=1, default=float)
        if r["record"]["passed"]:
            with open(os.path.join(a.out_dir, f"{sid}_j{a.j}{SP.SUFFIX}"), "wb") as fh:
                fh.write(r["package"])
        print(json.dumps(dict(site=sid, j=a.j, passed=r["record"]["passed"], rejected=r["record"]["rejected"] and
                              {kk: r["record"]["rejected"][kk] for kk in ("stage", "cause")}, cpu=r["record"]["cost_cpu_s"],
                              package_sha256=r["record"].get("package_sha256"))), flush=True)
        return 0
    run_candidate(label, a.kind, a.cell, a.k, a.out_dir, a.max_attempts, schema)
    return 0


if __name__ == "__main__":
    sys.exit(main())
