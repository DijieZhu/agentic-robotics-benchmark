"""Maintainers' mission-only harness: one bay mission run of a submission in the real bay, exactly as the grader runs it
(the probe account, the mission stage, the day of mission K of a set staged in the probe's own config and named by
`ARB_MISSION`, the mission's noise seed, the match lock from the mission's lock key, the agent's program behind the
`GuardedSolver` as the agent's account with no internet sockets), without the canary and the scene probes.

    # as root in the task image, the staged tests at /opt/ale/tests (read-only is fine):
    python3 /opt/ale/tests/review/run_mission.py --submission /home/user/submission --set FILE --index K [--time-limit 600]
        [--settle] [--no-token] [--out result.json] [--kit-dir DIR] [--program-env NAME=VALUE ...]
        [--sensors on|off] [--compare RECORD] [--frames-out DIR [--frames-at PLAN] [--frames-sensors IDS]]

FILE is the fixed hidden set (`tests/hidden/fixed_set.json`), a batch of candidates or the development days
(`make_fixed_set.py draw` / `corners`): anything with the set's `missions` list. The token state is the submission's
(`.oracle_token`), unless `--no-token`. With `--settle` the settle probe runs first, so that the real parts are named as
the agent's scene names them (as in a full verify); without it the real names are used. `--kit-dir` keeps Kit's caches
(the compiled shaders) between runs. `--program-env` adds a variable to the program's environment (e.g. the reference's
own record, ARB_REFERENCE_LOG=<a file the agent's account may write>): the grader passes none. Prints the mission's
outcome, the day as applied (`hidden`: how far each part moved in the pre-roll, ...), the per-check scores of this one
run and the timing; `--out` writes them with the run's whole record and the submission's file hashes (what
`make_fixed_set.py select` reads), with the sensors' frame statistics (`frames`: render, read, processing and write
times per frame set, bytes, the frame clock's anomalies, a digest per sensor).
`--sensors off` (development only) runs the mission without rendering (no frames; the wire otherwise as graded), to
tell the wire's filter from the renderer. `--compare RECORD` prints every outcome that differs from another run's
record (`compare_runs.py`; the timings and the frame statistics left out). `--frames-out DIR` keeps chosen frame sets
(`--frames-at PLAN`: a JSON object or file, {"every": N, "windows_s": [[t0, t1], ...], "events": {"grip" | "release"
| "lever_closed" | "liftoff" | "pour_start" | "fire_out": [before_s, after_s]}}; `--frames-sensors`: ids from the
manifest, comma-separated, or all): the probe copies them from the slots into its own folder with the true state at
each capture step (a token run), and they are moved to DIR after the run, compressed (one .npz per frame set and an
index.jsonl). Probe config keys only this tool sets; the grader never does.
Several of these may run at once in one container: the probe account is created under a lock (two concurrent
`useradd` lose one account), and the grader's stop of every process of the agent's account before and after a run is
left out here (it would stop the other runs' controllers; a full verify runs one mission at a time and does it).
Nothing here is part of the grade; the task's grader never runs it.
"""

import argparse
import fcntl
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import verify  # noqa: E402

verify.kill_agent = lambda: None  # (parallel runs in one container share the agent's account: see above)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def day_digest(day):
    """The sha256 of a day as the set holds it (canonical JSON): what ties a result to its candidate."""
    return hashlib.sha256(json.dumps(day, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/home/user/submission")
    ap.add_argument("--set", required=True, dest="set_file")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--no-token", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kit-dir", default=None)
    ap.add_argument("--program-env", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--sensors", choices=("on", "off"), default="on")
    ap.add_argument("--compare", default=None, metavar="RECORD")
    ap.add_argument("--frames-out", default=None, metavar="DIR")
    ap.add_argument("--frames-at", default=None, metavar="PLAN")
    ap.add_argument("--frames-sensors", default="all", metavar="IDS")
    ap.add_argument("--probe-set", action="append", default=[], metavar="DOTTED.KEY=JSON",
                    help="a maintainer's bench: set a key of the probe's config (e.g. sensors.bench.lag_check_every=10)")
    a = ap.parse_args()
    config = json.load(open(os.path.join(TESTS, "grader_config.json")))
    truth = json.load(open(os.path.join(TESTS, "truth.json")))
    with open(a.set_file) as f:
        spec = json.load(f)
    try:
        missions = verify.read_missions(spec.get("missions"), truth["groups"]["settle"]["items"])
    except ValueError as exc:
        sys.exit(f"{a.set_file}: {exc}")
    if not 0 <= a.index < len(missions):
        sys.exit(f"{a.set_file} holds {len(missions)} missions: no index {a.index}")
    if any("=" not in e or not e.split("=", 1)[0] for e in a.program_env):
        sys.exit("--program-env takes NAME=VALUE")
    mission = missions[a.index]
    sub = a.submission
    scene = next((os.path.join(sub, n) for n in ("scene.usda", "scene.usd", "scene.usdc") if os.path.isfile(os.path.join(sub, n))), None)
    if scene is None:
        sys.exit(f"no scene in {sub}")
    work = tempfile.mkdtemp(prefix="arb_mission_", dir="/var/tmp")
    os.chmod(work, 0o755)
    snap = os.path.join(work, "submission")
    verify.snapshot(sub, snap)
    scene = os.path.join(snap, os.path.basename(scene))
    hashes = {os.path.relpath(os.path.join(d, n), snap): sha256_file(os.path.join(d, n)) for d, _, files in os.walk(snap) for n in files}
    os.environ["PATH"] = "/usr/sbin:/usr/bin:/bin:" + os.environ.get("PATH", "")
    with open("/var/tmp/arb_probe_account.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        verify.probe_account(work)
    if a.kit_dir:
        import pwd
        pw = pwd.getpwnam(verify.PROBE_USER)
        os.makedirs(os.path.join(a.kit_dir, "optix"), exist_ok=True)  # (the probe's OptiX cache, as verify.probe_account makes it)
        for d, _, files in os.walk(a.kit_dir):
            os.chown(d, pw.pw_uid, pw.pw_gid)
            for f in files:
                os.chown(os.path.join(d, f), pw.pw_uid, pw.pw_gid)
        shutil.rmtree(os.path.join(work, "kit"))
        os.symlink(os.path.abspath(a.kit_dir), os.path.join(work, "kit"))
    for name in verify.PROBE_FILES:
        verify.stage_file(work, name)
    bay_layer = verify.stage_file(work, os.path.join("hidden", "bay_true.usda"))
    privileged, mismatch = verify.token_state(snap)
    privileged = privileged and not a.no_token
    cfg_probe = dict(config, bay=verify.bay_from_truth(truth), bay_facts=json.load(open(os.path.join(verify.HIDDEN, "bay_facts.json"))),
                     bay_layer=bay_layer, missions=[m["day"] for m in missions])
    if a.time_limit:
        cfg_probe["control_sim"] = dict(cfg_probe.get("control_sim") or {}, time_limit_s=float(a.time_limit))
    if privileged:
        cfg_probe["privileged"] = True
    if a.sensors == "off":
        cfg_probe["sensors_off"] = True
    if a.frames_out:
        plan = {"every": 10}
        if a.frames_at:
            plan = json.load(open(a.frames_at)) if os.path.isfile(a.frames_at) else json.loads(a.frames_at)
        ids = "all" if a.frames_sensors == "all" else [x for x in a.frames_sensors.split(",") if x]
        cfg_probe["frames_out"] = {"dir": os.path.join(work, "out", f"frames_{a.index}"), "plan": dict(plan, sensors=ids)}
    for kv in a.probe_set:
        key, val = kv.split("=", 1)
        node = cfg_probe
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = json.loads(val)
    verify.write_probe_config(work, cfg_probe)
    print(f"[mission] work {work}; mission {a.index} ({mission['id']}) of {a.set_file}; privileged={privileged} mismatch={mismatch}", flush=True)
    if a.settle:
        t0 = time.time()
        path = os.path.join(work, "out", "settle_0.json")
        verify.as_probe([os.path.join(verify.probe_dir(work), "probes.py"), scene, path, "settle"], work, 900, seed=0)
        items = (json.load(open(path))["groups"].get("settle") or {}).get("items") if os.path.isfile(path) else {}
        alias = verify.match_items(items or {}, truth["groups"]["settle"]["items"], config["tol"])
        cfg_probe["item_alias"] = {t: n for t, (n, _) in alias.items()}
        verify.write_probe_config(work, cfg_probe)
        print(f"[mission] settle probe {round(time.time() - t0)} s; parts by the scene's names: {cfg_probe['item_alias']}", flush=True)
    egress = verify.egress_mode()
    print(f"[mission] egress guard: {egress}", flush=True)
    t0 = time.time()
    data = verify.run_control(scene, work, config, mission, a.index, snap, egress, env_prefix=a.program_env)
    wall = time.time() - t0
    keys = ("end", "abort", "error", "timing", "done_t", "t_end", "steps", "steam_end", "steam_off_t", "lever_deg", "fire_end", "fire_out_t", "parts_in_crate",
            "n_parts", "parts", "parts_dropped", "liftoff_t", "liftoff_tilt_max", "swing_max", "spilt_l", "poured_l", "water0_l", "lost", "crash", "dropped",
            "n_collisions", "collisions", "min_sep", "gusts", "hidden", "real_bay", "lock", "relay_steps", "reply_s", "egress_mode", "egress_hits",
            "setup_stage", "render_error", "frames", "dress", "streamed", "not_streamed")
    summary = {k: data.get(k) for k in keys if k in data}
    summary["wall_s"] = round(wall, 1)
    summary["ms_per_step"] = round(1000.0 * wall / max(1, data.get("steps") or 1), 1)
    scan = dict(ok=True, facts=dict(robots={}))
    try:  # this one run's checks (the scene sections are not measured here; consistency is moot with one run)
        _, checks, sections, combos = verify.score(scan, dict(truth["groups"], control=[data, data]), truth, config)
        summary["checks"] = {c["name"]: c["score"] for c in checks if c["name"].startswith("control.")}
        summary["section_control"] = dict(no_progress_gate=combos.get("section.control.no_progress_gate"),
                                          progress_gate=combos.get("section.control.progress_gate"))
    except Exception as exc:  # noqa: BLE001
        summary["checks_error"] = repr(exc)
    if a.frames_out:  # the probe's exported frame sets, compressed, moved out of its folder
        src = os.path.join(work, "out", f"frames_{a.index}")
        os.makedirs(a.frames_out, exist_ok=True)
        n = 0
        if os.path.isdir(src):
            import numpy as np
            by_set = {}
            for f in sorted(glob.glob(os.path.join(src, "*.npy"))):
                m, sid = os.path.basename(f)[:-4].split("_", 1)
                by_set.setdefault(m, {})[sid] = f
            for m, files in by_set.items():
                np.savez_compressed(os.path.join(a.frames_out, f"{m}.npz"), **{sid: np.load(f) for sid, f in files.items()})
                n += 1
            for f in ("index.jsonl", "manifest.json"):
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copy(os.path.join(src, f), os.path.join(a.frames_out, f))
            shutil.rmtree(src, ignore_errors=True)
        summary["frames_exported"] = n
    if a.compare:
        sys.path.insert(0, HERE)
        import compare_runs
        d = compare_runs.compare(dict(data=data), json.load(open(a.compare)))
        summary["compare"] = dict(record=os.path.abspath(a.compare), identical=not d, differences=[[k, x, y] for k, x, y in d[:40]], n=len(d))
    print(json.dumps(summary, indent=1, default=str), flush=True)
    if a.out:
        record = dict(mission=dict(index=a.index, id=mission["id"], file=os.path.abspath(a.set_file), file_sha256=sha256_file(a.set_file),
                                   day_sha256=day_digest(spec["missions"][a.index]["day"])),
                      submission_sha256=hashes, privileged=privileged, options=dict(time_limit=a.time_limit, settle=a.settle, no_token=a.no_token,
                                                                                     program_env=a.program_env, sensors=a.sensors,
                                                                                     frames_out=a.frames_out, frames_at=a.frames_at, probe_set=a.probe_set),
                      summary=summary, data=data)
        with open(a.out, "w") as f:
            json.dump(record, f, indent=1, default=str)


if __name__ == "__main__":
    main()
