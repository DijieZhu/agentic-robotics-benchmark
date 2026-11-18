"""One graded station of rov_subsea_manifold_leak_isolation (tests/station.py): the cell process of one slot.

    python3 -s station.py JOB.json RESULT.json

verify.py (root) starts one of these per station, pinned to the slot's cell CPU (slots.pin_argv), and reads RESULT.json
when it exits.  This process, still root:
  1. builds the cell on the station's stored site package (subseacell.server.SubseaCellServer, mode hidden: the site,
     its work order, the hidden water, flow, plume and valve states come from the package; the program gets the
     vehicle's 17 public calls only, or, on a token-privileged run, also the reference program's two calls);
  2. lays out the program's run directory (cellkit.runner: a root-owned read-only copy of the submission, the client
     library from the root-owned givens, an empty private HOME / TMPDIR) and starts the program as the SLOT's runner
     account under the no-network filter, its CPU limit, and pinned to the slot's program CPU with its affinity locked
     (slots.pin_argv(..., lock=True): sched_setaffinity fails, so the program runs on one core as instruction.md says);
  3. serves the episode on the JSON-lines wire (cellkit.wire.serve_episode: program-time cap = wall clock minus the
     cell's own work, silence cap); anything the wire raises (a malformed line deep enough to overflow the decoder's
     recursion, ...) ends the episode as `protocol`: the program's doing, never a cell failure;
  4. ends the program and every process of the slot's account, sweeps what the account left in the shared temporary
     folders and the agent home, and removes the run directory;
  5. takes the terminal state BEFORE the world closes (subseacell.scoring.capture_terminal), the run's record
     (server.finalize) and the event log, and scores the station (subseacell.scoring.score_station: the eleven items
     from the hidden site, the event log and the terminal state only; nothing the program said is an input);
  6. writes RESULT.json: the items and points, the record, and `cell_failed` (what went wrong on the cell's side, not
     the program's: the build, an internal error, a physics fault, a lost frame, a scoring inconsistency) - verify.py
     re-runs one such station once, from scratch.

Results depend on the job only: time passes only inside step() and sonar_scan(), the station has its own process pair,
directories and random streams, so the slot, the CPU, the other stations running at the same time and the order of
completion change nothing (DESIGN section 13; the parallel / serial comparison of P24).
A maintainer's fault injection (job["inject_fault_at_s"], set by verify.py only from ARB_TEST_INJECT_CELL_FAULT) makes
the world raise the physics fault once its clock passes that time: the self-test of the re-run path.
"""

import json
import os
import resource
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True

import layout  # noqa: E402  (stdlib only; sets nothing until apply_cell_env)

layout.apply_cell_env(sys.path)

from cellkit import lockdown, netaudit, nonet, runner, wire  # noqa: E402
import slots  # noqa: E402

STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
RECORD_KEYS = ("cell_version", "site", "site_label", "site_sha256", "kind", "sensor_seed", "done_reason", "cell_fault",
               "sim_time_s", "physics_steps", "budget", "budget_limits", "calls", "illegal_call_count", "internal_errors",
               "render_failures", "frames_lost", "frames", "pings", "program_time_s", "cell_work_s", "wall_used_s",
               "steps_after_done", "build_wall_s", "events_sha256", "fingerprint", "reference")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def cell_failures(rec, render_enabled=True):
    """What went wrong on the cell's side of the run (never the program's doing): [] when nothing."""
    out = []
    if rec.get("cell_fault"):
        out.append(str(rec["cell_fault"]))
    if rec.get("done_reason") == "internal_error" or int(rec.get("internal_errors") or 0) > 0:
        out.append("internal_error")
    if render_enabled and (int(rec.get("frames_lost") or 0) > 0 or int(rec.get("render_failures") or 0) > 0):
        out.append("render_failure")
    return sorted(set(out))


def _ensure_judge_thresholds(server, values):
    """A run that took no camera frame leaves the event log's "imaged" thresholds unset (the cell's judge derives
    them from the camera's lamps at its first frame), and the scoring refuses a log without them.  With no frame
    no predicate was ever evaluated, so the thresholds only have to be the ones the judge WOULD have used: set them
    here with the cell's own function, from the same camera.  Returns True when it had to (a harness note)."""
    from subseacell import events as EV
    elog = server.event_log()
    judge = getattr(elog, "judge", None)
    if judge is None or getattr(judge, "P", None) is not None:
        return False
    cams = getattr(server.world, "cams", {}) or {}
    cam = cams.get((640, 480)) or (next(iter(cams.values())) if cams else None)
    if cam is None:
        return False
    half = min(float(L.field_angle_deg) for L in cam.lamps) / 2.0
    judge.P = EV.imaged_thresholds(values, half)
    return True


def _inject_fault(server, at_s):
    """Maintainers' self-test of the re-run path: the world's advance raises the physics fault once the simulated clock
    reaches `at_s` (the server then ends the run as physics_fault, exactly as a real one)."""
    W = server.world
    real = W.advance
    state = {"fired": False}

    def advance(dt):
        if not state["fired"] and float(W.sim.time) + float(dt) >= float(at_s):
            state["fired"] = True
            raise FloatingPointError("injected cell fault (maintainers' self-test)")
        return real(dt)
    W.advance = advance


def run(job):
    t0 = time.time()
    cfg = job["config"]
    res = {"tag": job["tag"], "station": job["station"], "attempt": job["attempt"], "site": job["site_name"],
           "cell_failed": [], "harness_notes": []}
    from subseacell import budget as BU
    from subseacell import client as CL
    from subseacell import scoring as SC
    from subseacell import server as SV
    values = BU.load_task_values(layout.VALUES_PATH)
    rules = SC.Rules.from_values(values)
    privileged = bool(job.get("privileged"))
    tb = time.time()
    try:
        server = SV.SubseaCellServer(values=values, config={"budgets": dict(cfg["budgets"])}, mode="hidden",
                                     privileged=privileged, render=True, log=_log,
                                     reference_dir=layout.REFERENCE_DIR if privileged else None)
        server.reset(job["site_path"], seed=None, kind="hidden", name=job["tag"])
    except Exception:  # noqa: BLE001 - the cell depends on the stored package only: a build failure is the cell's
        res["cell_failed"] = ["build"]
        res["error"] = traceback.format_exc(limit=12)[-3000:]
        res["wall_s"] = time.time() - t0
        return res
    res["build_s"] = time.time() - tb
    if job.get("inject_fault_at_s") is not None:
        _inject_fault(server, float(job["inject_fault_at_s"]))
        res["fault_injected_at_s"] = float(job["inject_fault_at_s"])

    acct = job.get("account")                    # [name, uid, gid] of the slot's runner account, or None (not root)
    name, uid, gid = (acct if acct else (None, None, None))
    root = os.path.join(job["runs_root"], job["tag"])
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    dirs = runner.prepare_run_dir(root, job["program_dir"], {"subseacell": job["lib_src"]},
                                  owner=(uid, gid) if acct else None)
    res["symlinks_skipped"] = int(dirs["skipped"])
    watch = STICKY_DIRS + [job["agent_home"]]
    before = {r: runner.snapshot_tree(r, (dirs["root"], job["runs_root"])) for r in watch} if acct else None
    # the program runs on the program's environment (the agent's venv, job["program_python"]), never on the cell's:
    # the launcher, the no-network filter's bootstrap and the program itself all use that interpreter
    py = job.get("program_python") or sys.executable
    argv = CL.launcher_argv(py, dirs["lib"], dirs["program"])
    argv = runner.wrap_limits(nonet.nonet_argv(argv, python=py), cpu_s=float(cfg["program_cpu_cap_s"]), user=name,
                              nonet=False)
    prog_cpu = job.get("prog_cpu")
    argv = slots.pin_argv(argv, [prog_cpu] if prog_cpu is not None else [], lock=bool(cfg.get("lock_program_affinity", True)))
    env = layout.program_env(runner.child_env(dirs["home"], dirs["tmp"], dirs["lib"], job["given_dir"], threads=1))
    program_cap = float(cfg["program_time_cap_s"])
    silence = float(cfg["silence_s"])
    sampler = netaudit.SocketSampler(uid=uid if acct else None, pids_fn=None if acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    reason, wire_error, proc = None, None, None
    old_umask = os.umask(0o077)                   # the program's files are private to its account by default
    t_ep = time.time()
    try:
        with sampler:
            try:
                proc = wire.AgentProcess(argv, cwd=dirs["program"], user=None, env=env,
                                         stderr_path=os.path.join(job["logs_dir"], "%s.log" % job["tag"]),
                                         cpu_s=None, nonet=False)
            except Exception:  # noqa: BLE001 - a launch failure is the grader's (the image or the stage), a task error
                res["harness_error"] = "the program could not be launched\n" + traceback.format_exc(limit=8)
                return res
            finally:
                os.umask(old_umask)
            try:
                reason = wire.serve_episode(server, proc, program_cap, silence, log=_log)
            except BaseException as e:  # noqa: BLE001 - whatever the wire raises is the program's line: protocol
                wire_error = ("%s: %s" % (type(e).__name__, e))[:300]
                reason = "protocol"
                server.wall_used = time.time() - t_ep
                server.program_used = max(0.0, server.wall_used - float(server.work_wall))
                _log("[station] %s: the wire raised %s; the episode ends as protocol" % (job["tag"], wire_error))
            finally:
                if proc is not None:
                    proc.terminate()
                if acct:
                    killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
                    res["runner_processes_killed"], res["runner_processes_left"] = int(killed), int(left)
                    stray = 0
                    for r in watch:
                        after = runner.snapshot_tree(r, (dirs["root"], job["runs_root"]))
                        new, modified = runner.tree_changes(before[r], after, uid)
                        stray += len(new) + len(modified)
                        runner.sweep_new_entries(r, before[r], after, uid, (dirs["root"], job["runs_root"]))
                    res["stray_files_removed"] = int(stray)
                shutil.rmtree(dirs["root"], ignore_errors=True)
    finally:
        os.umask(old_umask)
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    res["program_cpu_s"] = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
    res["episode_wall_s"] = time.time() - t_ep
    res["end_reason"] = reason
    res["wire_error"] = wire_error
    res["remote_sockets_seen"] = int(sampler.max_seen)
    res["socket_samples"] = int(sampler.samples)
    try:
        if not server.done:
            server._end(reason or "done")
        term = SC.capture_terminal(server.world)
        rec = server.finalize()
        if _ensure_judge_thresholds(server, values):
            res["harness_notes"].append("no camera frame in the run: the imaged thresholds set from the cell's camera")
        elog = server.event_log()
        doc = elog.export()
    except Exception:  # noqa: BLE001 - finalize / the terminal capture do not raise by design
        res["cell_failed"] = ["record"]
        res["error"] = traceback.format_exc(limit=12)[-3000:]
        server.close()
        return res
    res["record"] = {k: rec.get(k) for k in RECORD_KEYS}
    res["step_calls"] = int((rec.get("calls") or {}).get("step", 0)) + int((rec.get("calls") or {}).get("reference_run", 0))
    res["cell_failed"] = cell_failures(rec)
    try:
        sc = SC.score_station(job["site_path"], doc, term, rules)
        res["items"] = {k: bool(v) for k, v in sc["items"].items()}
        res["points"] = int(sc["points"])
        res["score"] = float(sc["score"])
        res["score_fraction"] = str(sc.get("score_fraction"))
        res["log_digest"] = sc.get("log_digest")
        res["details"] = sc.get("details")
    except SC.ScoringError as e:                  # the cell's record disagrees with the rules: the cell's failure
        res["cell_failed"] = sorted(set(res["cell_failed"]) | {"scoring"})
        res["error"] = "ScoringError: %s" % e
    except Exception:  # noqa: BLE001
        res["cell_failed"] = sorted(set(res["cell_failed"]) | {"scoring"})
        res["error"] = traceback.format_exc(limit=12)[-3000:]
    try:                                          # the maintainers' record of the run; never a scoring input
        d = os.path.join(job["out_dir"], "events_%s" % job["tag"])
        elog.save(d)
    except Exception:  # noqa: BLE001
        res["harness_notes"].append("event log not saved")
    try:
        server.close()
    except Exception:  # noqa: BLE001
        pass
    res["wall_s"] = time.time() - t0
    res["worker_cpu_s"] = time.process_time()
    try:
        res["worker_affinity"] = sorted(os.sched_getaffinity(0))
    except OSError:
        res["worker_affinity"] = []
    return res


def main(argv):
    job_path, out_path = argv[0], argv[1]
    with open(job_path) as f:
        job = json.load(f)
    try:
        res = run(job)
    except Exception:  # noqa: BLE001 - anything else is the harness's own defect: a task error in verify.py
        res = {"tag": job.get("tag"), "station": job.get("station"), "attempt": job.get("attempt"),
               "harness_error": traceback.format_exc(limit=12)[-3000:]}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, default=_jsonable, allow_nan=True)
    os.replace(tmp, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
