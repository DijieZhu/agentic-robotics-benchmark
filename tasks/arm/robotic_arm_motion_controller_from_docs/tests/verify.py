"""Grader of robotic_arm_motion_controller_from_docs (a motion controller written from the vendor's documents): runs as
ROOT in the verify stage.

The agent left /home/user/submission/controller/controller.py (class Controller: setup / plan / step, CONTRACT.md of
the grader package `arm.motion_ctl`). Verification is separate (task.toml): this runs in a fresh verifier sandbox from
the task image into which the engine restored only the submission (root-owned). This grader:
  0. FIRST, stdlib only (`_early_lockdown`, the six-servo pattern; tests/test.sh closed the stage folder already): freezes
     and kills every process of the agent account until the verdict is written, locks the stage, snapshots the
     submission root-owned, gates late writes, quarantines late entries elsewhere;
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (a mismatch is a task error);
  2. contract_ok: controller/controller.py is a regular file that parses and defines class Controller, the submission is
     at most max_submission_mb, the network audit passes (static: no networking / process / ctypes import, no computed
     import or code; runtime: the controller under the client's no-network seccomp filter and no remote socket of its
     account, else no outbound packet in the sandbox while it runs), the controller answers
     at least one call, no late write, no surviving process, >= on_time_min of the bus cycles answered within 10 ms CPU;
  3. the oracle token: controller/oracle_token whose sha256 equals hidden/oracle_token.sha256 adds the truth payloads
     (stat privileged_mode); a wrong token is ignored (stat oracle_token_mismatch);
  4. runs the fixed hidden set (tests/hidden/scenarios: three scenarios, a unit and its three cells each, kept by the
     pre-registered selection; sha256 and the grader code's digest checked; a missing set is a task error) through `arm.motion_ctl.grade.Acceptance` (shelf, bins,
     dispense, about 14 jobs each; a contact fails the cell), the controller as `nobody` under the client's no-network
     filter, from a read-only copy of the submission (`arm.motion_ctl.wire.Runner`);
  5. reward = contract_ok x the mean over the runs of the episode score (`arm.motion_ctl.metrics.episode`: 0.5 reach +
     0.4 path + 0.1 infeasibility J, a category without jobs left out and the weights renormalised; job = completed x
     no_contact x clearance x conditioning x accuracy x (0.6 + 0.2 time + 0.2 smoothness), the path accuracy gated by
     the path's progress).
Untouched sandbox: no controller.py -> contract_ok 0 -> 0. Oracle: the token -> the privileged reference controller
replays the truth trajectories -> 1.
"""

import ast
import hashlib
import json
import math
import os
import pwd
import shutil
import stat as statmod
import sys
import time
import traceback

sys.dont_write_bytecode = True

STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)

HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
ROOT_PAYLOAD = os.environ.get("ARB_ROOT_PAYLOAD", "/opt/arb")
RUNS_ROOT = os.path.join(ROOT_PAYLOAD, "runs")
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0

from arm.assembly_cell import runner  # noqa: E402  (stdlib only)
from arm.assembly_six import lockdown  # noqa: E402  (stdlib only)
from arm import verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0, "late_write_gate_active": 0.0,
           "submission_snapshot": 0.0, "saved": None, "parent": None, "t_ref": None, "late": [], "reasons": []}
    if os.getuid() != 0:
        return out
    t_ref = lockdown.stage_upload_time(STAGE_DIR)
    out["t_ref"] = t_ref
    try:
        uid = pwd.getpwnam(AGENT_USER).pw_uid
    except KeyError:
        uid = None
    if uid is not None and uid != 0:
        killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
        out["agent_processes_killed"], out["agent_processes_left"] = float(killed), float(left)
        if left:
            out["reasons"].append("%d process(es) of the agent account survived the kill at verify start" % left)
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
    out["saved"] = runner.lock_stage(STAGE_DIR)
    out["stage_locked"] = 1.0 if out["saved"] is not None else 0.0
    parent = os.path.dirname(os.path.abspath(STAGE_DIR))
    if parent and parent not in ("/", os.path.abspath(HOME)) and os.path.isdir(parent):
        try:
            mode = statmod.S_IMODE(os.lstat(parent).st_mode)
            os.chmod(parent, 0o700)
            saved_mode = os.environ.get("ARB_STAGE_PARENT_MODE")
            out["parent"] = (parent, int(saved_mode, 8) if saved_mode else mode)
            out["stage_parent_locked"] = 1.0
        except (OSError, ValueError):
            pass
    gate_active = t_ref is not None
    cfg_json = os.path.join(STAGE_DIR, "config.json")
    if gate_active and os.path.isfile(cfg_json):
        try:
            gate_active = (os.lstat(cfg_json).st_ctime - t_ref) <= LATE_GATE_MAX_GAP_S
        except OSError:
            pass
    out["late_write_gate_active"] = 1.0 if gate_active else 0.0
    if os.path.isdir(SUBMISSION):
        if gate_active:
            out["late"] = lockdown.late_entries(SUBMISSION, t_ref)
            out["submission_late_writes"] = float(len(out["late"]))
            if out["late"]:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(out["late"]), "y" if len(out["late"]) == 1 else "ies", ", ".join(e[0] for e in out["late"][:5])))
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)
        lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    if gate_active:
        q = 0
        for root in WATCH_ROOTS:
            late = lockdown.late_entries(root, t_ref, exclude=(SUBMISSION, RUNS_ROOT, STAGE_DIR))
            if late:
                log("[grader] %d late entr%s under %s quarantined" % (len(late), "y" if len(late) == 1 else "ies", root))
                q += lockdown.quarantine(root, late, QUARANTINE)
        out["late_writes_quarantined"] = float(q)
    out["stage_lock_latency_ms"] = lockdown.process_age_ms()
    return out


EARLY = _early_lockdown() if __name__ == "__main__" else {"saved": None, "parent": None, "reasons": [], "late": [], "t_ref": None}

import numpy as np  # noqa: E402

from arm.assembly_info import netaudit  # noqa: E402
from arm.motion_ctl import grade as G  # noqa: E402
from arm.motion_ctl import scenarios as SC  # noqa: E402

SCENARIO_DIR = os.path.join(STAGE_DIR, "hidden", "scenarios")


def hidden_scenarios():
    """The fixed hidden set (tests/hidden/scenarios/manifest.json: each file's sha256 checked). A missing or empty
    set, a file that does not match, or a scenario made with other grader code (scenarios.ScenarioStale) is a task
    error: the exception propagates and no verdict is written."""
    man = os.path.join(SCENARIO_DIR, "manifest.json")
    with open(man) as fh:
        manifest = json.load(fh)
    if not manifest.get("kept"):
        raise RuntimeError("the hidden set is empty")
    out = []
    for ent in manifest["kept"]:
        path = os.path.join(SCENARIO_DIR, ent["file"])
        if sha256_file(path) != ent["sha256"]:
            raise RuntimeError("hidden scenario {} does not match its manifest".format(ent["file"]))
        out.append(SC.load(path))
    return out
from arm.motion_ctl import wire as W  # noqa: E402

DESIGN = os.path.join(STAGE_DIR, "design")
HIDDEN = os.path.join(STAGE_DIR, "hidden")
CLIENT = os.path.join(STAGE_DIR, "arm", "motion_ctl", "client.py")
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
CRITERIA = ["acceptance_score"]
GATES = ["contract_ok"]


def record_dir(path):
    os.makedirs(RECORD_ROOT, exist_ok=True)
    os.chmod(RECORD_ROOT, 0o700)
    os.makedirs(path, exist_ok=True)
    return path


def program_source():
    src = SNAPSHOT if os.getuid() == 0 and os.path.isdir(SNAPSHOT) else SUBMISSION
    return os.path.join(src, "controller")


def fail(reason):
    log("[grader] TASK ERROR: " + reason)
    raise SystemExit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def root_payload_ok(expected):
    problems = []
    owner = 0 if os.getuid() == 0 else os.getuid()
    if not os.path.isdir(ROOT_PAYLOAD):
        return False, ["%s missing (tests/test.sh builds it)" % ROOT_PAYLOAD]
    for rel, digest in expected.items():
        p = os.path.join(ROOT_PAYLOAD, rel)
        try:
            st = os.lstat(p)
        except OSError:
            problems.append("missing " + rel)
            continue
        if os.path.islink(p) or st.st_uid != owner or (st.st_mode & 0o022):
            problems.append("not a root-owned, group/world-unwritable regular file: " + rel)
            continue
        if sha256_file(p) != digest:
            problems.append("differs " + rel)
    return not problems, problems


def tree_size(root):
    total = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            try:
                total += os.lstat(os.path.join(dp, fn)).st_size
            except OSError:
                pass
    return total


def contract_check(cfg):
    src = program_source()
    path = os.path.join(src, "controller.py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no controller/controller.py (regular file) in the submission", []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        return False, "controller.py does not parse: %s" % e, []
    if not any(isinstance(n, ast.ClassDef) and n.name == "Controller" for n in ast.walk(tree)) and \
            not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "Controller" for a in n.names) for n in ast.walk(tree)):
        return False, "controller.py defines no class Controller", []
    if tree_size(src) > float(cfg.get("max_submission_mb", 80)) * 1024 * 1024:
        return False, "the controller folder is larger than %s MB" % cfg.get("max_submission_mb", 80), []
    net = netaudit.scan_imports(src)
    if net:
        return False, "network audit: %s" % ", ".join("%s: %s" % x for x in net[:5]), net
    return True, "ok", []


def token_state():
    p = os.path.join(program_source(), "oracle_token")
    want_p = os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(want_p):
        return False, False
    try:
        tok = open(p, "rb").read().strip()
        want = open(want_p).read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return (got == want), (got != want)


class Dead(object):
    """A controller that could not start: every call fails."""
    dead = True

    def __init__(self, why):
        self.why = why
        self.stats = {"start_error": str(why)[:300]}

    def call(self, op, payload=None, cpu_s=1.0, wall_s=10.0):
        return "dead", None, 0.0, 0.0

    def close(self):
        return dict(self.stats)


def make_factory(cfg, tag, counters):
    def make():
        counters["starts"] += 1
        root = os.path.join(RUNS_ROOT, "%s_%d" % (tag, counters["starts"]))
        record_dir(LOGS)
        r = W.Runner(program_source(), root, CLIENT, user=cfg.get("runner_account", "nobody"),
                     stderr_path=os.path.join(LOGS, "%s_%d.stderr.log" % (tag, counters["starts"])))
        ok, info = r.start()       # a RuntimeError here (no runner account, no runuser, the uid check) is the task's
        if not ok:                 # failure, not the controller's: it propagates and no verdict is written
            counters["start_failures"] += 1
            r.close()
            return Dead(info)
        counters["uid"] = r.uid
        counters["runners"].append(r)
        return r
    return make


def write_verdict(rewards, metrics):
    """One gated `reward` (robotics_grader.ratio_rewards against anchor.json: value 1.0 = the oracle, full_at 1.0)."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["acceptance_score"])
    anchor, meta = read_anchor()
    extra = {k: float(v) for k, v in metrics.items()}
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "acceptance_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (controller.py defines Controller, size cap, network audit, the calls answered, no late write, "
            "no surviving process, >= 90 %% of the bus cycles on time); acceptance_score %.4f (mean over the acceptance runs of "
            "0.5 reach + 0.4 path + 0.1 infeasibility J on the true unit in the true cells)") % (rewards["contract_ok"], rewards["acceptance_score"])
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {}
        for k, v in sorted(verdict["metrics"].items()):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                recorded[k.replace(".", "_").replace("/", "_")] = v
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, diagnostic=text)
        return overall
    kit_write_verdict(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"))
    return overall


def _restore_modes():
    runner.unlock_stage(EARLY.get("saved"))
    mode = os.environ.get("ARB_STAGE_MODE")
    if mode:
        try:
            os.chmod(STAGE_DIR, int(mode, 8))
        except (OSError, ValueError):
            pass
    parent = EARLY.get("parent")
    if parent:
        try:
            os.chmod(parent[0], parent[1])
        except OSError:
            pass


def main():
    cfg = load_json(os.path.join(STAGE_DIR, "grader_config.json"))
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "acceptances": float(cfg.get("acceptances", 2)), "root_payload_ok": 0.0,
               "privileged_mode": 0.0, "oracle_token_mismatch": 0.0, "network_imports": 0.0, "remote_sockets_seen": 0.0,
               "net_outbound_events": 0.0, "controller_no_network": 0.0, "runner_processes_killed": 0.0, "stray_processes": 0.0, "controller_starts": 0.0,
               "controller_start_failures": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        _main_locked(cfg, rewards, metrics)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        log("[grader] grader exception\n" + traceback.format_exc())
        raise
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()


def _main_locked(cfg, rewards, metrics):
    t_start = time.time()
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, net = contract_check(cfg)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["network_imports"] = float(len(net))
    log("[grader] contract check: %s (%s); privileged %s" % (ok, why, privileged))
    if not ok:
        _finish(metrics, t_start)
        write_verdict(rewards, metrics)
        return
    scen = hidden_scenarios()
    n = len(scen)
    metrics["hidden_set"] = float(n)
    runs = []
    counters = {"starts": 0, "start_failures": 0, "uid": None, "runners": []}
    for i in range(n):
        log("[grader] acceptance %d/%d" % (i + 1, n))
        acc = G.Acceptance(DESIGN, scenario=scen[i], log=log, privileged=privileged)
        net_c = netaudit.NetCounters()
        acct = runner.runner_account(cfg.get("runner_account", "nobody")) if os.getuid() == 0 else None
        if acct:
            sampler = netaudit.SocketSampler(uid=acct[1], period_s=float(cfg.get("socket_sample_s", 0.5)))
        else:
            sampler = netaudit.SocketSampler(uid=None, pids_fn=lambda: [p for r in counters["runners"] for p in (r._pids_full() if r.proc else [])],
                                             period_s=float(cfg.get("socket_sample_s", 0.5)))
        with net_c, sampler:
            out = acc.run(make_factory(cfg, "acc%d" % (i + 1), counters))
        out["remote_sockets_seen"] = int(sampler.max_seen)
        out["net_outbound"] = int(net_c.total)
        out["net_delta"] = {k: int(v) for k, v in net_c.delta.items() if v}
        runs.append(out)
        s = out["scores"]
        log("[grader]   unit %s: overall %.4f (reach %.3f, path %.3f, J %.3f); contacts %d; on time %.3f" % (
            out["serial"], s["overall"], s["reach"], s["path"], s["infeasibility"], out["stats"]["contacts"], out["stats"]["on_time_fraction"]))
    for r in counters["runners"]:
        st = r.close()
        metrics["runner_processes_killed"] += float(st.get("runner_processes_killed", 0))
        metrics["stray_processes"] += float(st.get("stray_processes", 0))
    metrics["controller_starts"] = float(counters["starts"])
    metrics["controller_start_failures"] = float(counters["start_failures"])
    metrics["remote_sockets_seen"] = float(max(r["remote_sockets_seen"] for r in runs))
    metrics["net_outbound_events"] = float(sum(r["net_outbound"] for r in runs))
    for r in runs:                                   # which namespace counters moved (diagnostics)
        for k, v in r.get("net_delta", {}).items():
            metrics["net_" + k] = metrics.get("net_" + k, 0.0) + float(v)
    on_time = min(r["stats"]["on_time_fraction"] for r in runs)
    answered = sum(r["stats"]["cycles"] for r in runs) > 0 or any(rec.get("refused") for r in runs for rec in r["records"])
    # the controller runs under the client's no-network seccomp filter (inet sockets refused for it and everything it
    # starts): then the namespace counters, which also count the sandbox's own loopback connections, are a statistic;
    # without the filter they gate (in the verifier sandbox only, as root: a host-side run shares the host's namespace)
    no_net = bool(counters["runners"]) and all(r.stats.get("no_network") for r in counters["runners"])
    metrics["controller_no_network"] = 1.0 if no_net else 0.0
    ns_gate = True if (no_net or os.getuid() != 0) else metrics["net_outbound_events"] <= 0
    gates = {"on_time": on_time >= float(cfg.get("on_time_min", 0.9)), "network": metrics["remote_sockets_seen"] <= 0 and ns_gate,
             "answered": bool(answered), "clean": metrics["stray_processes"] <= 0}
    for k, v in gates.items():
        metrics["gate_" + k] = 1.0 if v else 0.0
    gate = 1.0 if all(gates.values()) else 0.0
    rewards["contract_ok"] = gate
    vals = [float(r["scores"]["overall"]) for r in runs]
    mean = float(np.mean(vals))
    if not math.isfinite(mean):                # never through the clamp: min(1.0, nan) is 1.0
        raise RuntimeError("a run's score is not finite: {}".format(vals))
    rewards["acceptance_score"] = float(max(0.0, min(1.0, gate * mean)))
    for i, r in enumerate(runs):
        for k in ("overall", "reach", "path", "infeasibility"):
            metrics["run%d_%s" % (i + 1, k)] = float(r["scores"][k])
        for k in ("contacts", "rehomes", "restarts", "controller_errors", "cycles", "cycles_late", "truth_payloads"):
            metrics["run%d_%s" % (i + 1, k)] = float(r["stats"].get(k, 0))
        metrics["run%d_on_time" % (i + 1)] = float(r["stats"]["on_time_fraction"])
        recs = r["records"]
        feas = [x for x in recs if not x.get("infeasible")]
        metrics["run%d_completed_rate" % (i + 1)] = float(np.mean([1.0 if x.get("completed") else 0.0 for x in feas])) if feas else 0.0
        metrics["run%d_refused_feasible" % (i + 1)] = float(sum(1 for x in feas if x.get("refused")))
    metrics["score_mean_ungated"] = float(np.mean(vals))
    record_dir(OUT)
    with open(os.path.join(OUT, "acceptances.json"), "w") as f:
        json.dump([{k: v for k, v in r.items()} for r in runs], f, indent=1, default=float)
    _finish(metrics, t_start)
    overall = write_verdict(rewards, metrics)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall, metrics["verify_wall_s"]))


def _finish(metrics, t_start):
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics["verify_wall_s"] = time.time() - t_start
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
