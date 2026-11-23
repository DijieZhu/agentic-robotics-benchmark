"""dexterous_hand_full_assembly_16dof_panda: the grader writes a verdict in every case that depends on the submission.

tests/verify.py splits its work at one line: the task's own inputs (config, givens, kit, mating graph, runner account,
seeds; and, per episode, the cell's build before the program starts) fail loudly (a task error: exit 1, no verdict);
everything after that depends on the submission (its token, the contract check, the episodes from the program's launch
on, the scoring) and any exception there gives a verdict with grader_exception 1, contract_ok 0 and reward 0, its
traceback in the maintainers' record only. The checks run verify.py's own functions in a child interpreter that can
import the grader (numpy and MuJoCo: this one or ARB_NUMPY_PYTHON; skipped otherwise), as the non-root authoring path,
in temporary folders (nothing downloaded, no kit needed):
  * fault injection: the contract check raises RuntimeError -> the verdict above, the grader's log names the type only,
    the traceback (with the message) is in <record>/logs/grader_exception.log;
  * the same for an exception in an episode after the contract passed;
  * a task error before that line (the kit's fetch) and the cell's build at an episode's start still end with exit 1
    and no verdict;
  * the task's own faults stay loud although they would only bite after the line: a configuration value the episodes
    read (here `privileged_wall_cap_s`, missing) and a record root that cannot be made each end with exit 1, a task
    error and no verdict (the values are read and the record folders made before the line);
  * the contract check on assemble.py: a 100 000-deep expression gives `assemble.py cannot be parsed: RecursionError`, a
    null byte and bytes that are not UTF-8 `assemble.py does not parse: ...`, the review's deep helper file a network
    audit hit, a helper that does not fit within the parse limits (grader_config.json parse_limits, lowered here to
    1 GiB, then to 1 s) its own reason `<file>: cannot be parsed within the grader's limits (...)` and no network-audit
    hit, and a file above the size cap the size reason without being opened; where the route gate did not run, the
    suspicious-lines statistic is -1;
  * the grader never parses a source of the submission itself: with `ast.parse` broken in the grader's process the
    contract check of a plain submission still passes, and a 2 MB data module (a list literal of a million small
    integers, about 1 GB for Python's parser) passes without raising the grader's own peak memory (VmHWM) by more than
    300 MB;
  * the oracle token is checked by lstat before it is opened: a 5 GiB sparse `.oracle_token`, a 4097-byte one, a FIFO,
    a folder and a symbolic link are each a mismatch without being opened (an open of the token path is refused in
    these cases, so a grader that reads first fails here instead of reading gigabytes), within seconds and with the
    grader's peak memory at its baseline; the reference's own token, as solution/solve.sh lays it out, switches the
    privileged mode;
  * the walk at verify start (`_early_lockdown`, run here with the root check answered yes and every process-, account-
    and stage-touching call stubbed, so nothing outside the temporary folders is touched) checks the caps before the
    snapshot copies anything: a submission with an 8 GiB sparse data file, one with 12 000 empty files and one nested
    3000 folders deep each get the size, the entry or the path-length reason, the snapshot and the late-write walk are
    never called (the stubs record calls and copy nothing), and the verdict that follows carries that reason with
    contract_ok 0 and grader_exception 0 while the contract check and the episodes never run; a plain submission is
    snapshotted as before;
  * the contract check on a tree nested 3000 folders deep (the authoring path, on the submission itself) gives the
    path-length reason instead of an error in the copy for the import check;
  * the reasons of the episodes' gates: a run without a step call and a run that leaves a process the kill did not
    end are each a zero logged with its rule (`contract_ok 0: ...`), and the log never prints the contract as passed
    before the episodes have decided it; when several of them fail, their reasons are logged in the instruction's
    order (a step call, no network activity, no process left); a helper that declares a disallowed text encoding is a
    source-audit reason.
Every case runs in its own temporary folder with a short path (the system's /tmp where it is writable), whatever
TMPDIR is: the walk at verify start opens folders by their absolute paths.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SLUG = "dexterous_hand_full_assembly_16dof_panda"


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "hand" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / "tests" / "verify.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


def _short_tmp():
    """a temporary folder with a short absolute path, whatever TMPDIR is: the walk at verify start opens every folder
    by its absolute path, so a tree 3000 folders deep (3999 bytes below the submission) is seen whole only while the
    submission's own path is short, as the verifier's is (/home/user/submission); the shortest writable of the
    system's temporary folders, the environment's last"""
    for d in ("/tmp", "/var/tmp", "/dev/shm", tempfile.gettempdir()):
        if os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK) and len(os.path.realpath(d)) <= 40:
            return tempfile.mkdtemp(prefix="hgx_", dir=os.path.realpath(d))
    raise unittest.SkipTest("no temporary folder with a path of at most 40 bytes for the path-length tests")


TASK = _task_dir()
TESTS = TASK / "tests"


def _grader_python():
    for py in (sys.executable, os.environ.get("ARB_NUMPY_PYTHON")):
        if py and os.path.isfile(py):
            r = subprocess.run([py, "-I", "-B", "-c", "import numpy, mujoco"], capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return py
    return None


CHILD = r'''
import builtins, json, os, shutil, sys
sys.dont_write_bytecode = True
case, tests, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tests)
import verify as V

def fake_gt():
    gt = os.path.join(tmp, "gt")
    os.makedirs(gt, exist_ok=True)
    with open(os.path.join(gt, "mating_graph.json"), "w") as fh:
        json.dump({"mates": [{"id": "m1", "weight": 1.0}, {"id": "m2", "weight": 1.0}]}, fh)
    V.GT = gt

def run_main():
    cfg = V.load_json(os.path.join(tests, "grader_config.json"))
    cfg["episodes"] = 1
    rewards = dict((k, 0.0) for k in V.GATES + V.CRITERIA)
    metrics = {"grader_exception": 0.0, "stage_lock_latency_ms": -1.0, "network_imports": 0.0, "suspicious_lines": 0.0,
               "remote_sockets_seen": 0.0, "net_outbound_events": 0.0, "agent_wire_errors": 0.0}
    V.root_payload_ok = lambda expected: (True, [])
    for k in drop_keys:
        cfg.pop(k, None)
    V._main_locked(cfg, rewards, metrics)

drop_keys = []

class DummyServer(object):
    done = False
    def __init__(self, *a, **kw):
        pass
    def reset(self, seed):
        pass

def vm_kb(field):
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return -1

def token_cases(tok, sub, refuse, case_now, states, real_open):
    import time as _t
    hwm0 = vm_kb("VmHWM")
    t0 = _t.time()
    with real_open(tok, "wb") as fh:
        fh.truncate(5 * 1024 ** 3)              # 5 GiB, sparse: no disk used
    case_now[0] = "sparse_5g"; states["sparse_5g"] = list(V.token_state())
    os.remove(tok)
    with real_open(tok, "wb") as fh:
        fh.write(b"a" * 4097)
    case_now[0] = "4097"; states["4097"] = list(V.token_state())
    os.remove(tok); os.mkfifo(tok)
    case_now[0] = "fifo"; states["fifo"] = list(V.token_state())
    os.remove(tok); os.mkdir(tok)
    case_now[0] = "folder"; states["folder"] = list(V.token_state())
    os.rmdir(tok)
    ref = os.path.join(os.path.dirname(tests), "solution", "reference", "oracle_token")
    if os.path.isfile(ref):
        os.symlink(ref, tok)
        case_now[0] = "link"; states["link"] = list(V.token_state())
        os.remove(tok)
        shutil.copyfile(ref, tok)
        refuse[0] = False
        case_now[0] = "reference"; states["reference"] = list(V.token_state())
        os.remove(tok)
    secs = round(_t.time() - t0, 2)
    hwm = vm_kb("VmHWM") - hwm0
    states["_stats"] = {"seconds": secs, "hwm_delta_kb": hwm}

def deep_make(base, depth):
    # `depth` nested folders "d" under base, made relative to an open folder (paths beyond PATH_MAX are fine)
    fd = os.open(base, os.O_RDONLY)
    for _ in range(depth):
        os.mkdir("d", dir_fd=fd)
        nfd = os.open("d", os.O_RDONLY, dir_fd=fd)
        os.close(fd)
        fd = nfd
    os.close(fd)

def deep_remove(base, depth):
    # removes what deep_make made, bottom up, by relative steps (shutil.rmtree stops at the open-file limit)
    here = os.getcwd()
    os.chdir(base)
    n = 0
    while n < depth and os.path.isdir("d"):
        os.chdir("d")
        n += 1
    for _ in range(n):
        os.chdir("..")
        os.rmdir("d")
    os.chdir(here)

out = {"case": case}
try:
    if case == "contract_raises":
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        def boom(cfg, rn):
            raise RuntimeError("injected fault 7f3a")
        V.contract_check = boom
        run_main()
    elif case == "episode_raises":
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        V.contract_check = lambda cfg, rn: (True, "ok", 0, [])
        def boom(*a, **kw):
            raise RuntimeError("injected fault 7f3a")
        V.run_episode = boom
        run_main()
    elif case in ("gate_nostep", "gate_left", "gate_net_left"):
        # the episodes' gates with their reasons: the contract check passes, each episode returns a summary without
        # a mate; "gate_nostep" makes no step call, "gate_left" steps but leaves two processes the kill did not end,
        # "gate_net_left" does that and also reports network activity (the episode's summary as the grader reads it)
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        V.contract_check = lambda cfg, rn: (True, "ok", 0, [])
        def fake_episode(cfg, seed, rn, privileged, kit_info, plan, trace_tag=None):
            if case in ("gate_left", "gate_net_left"):
                rn.left += 2
                rn.unclean_runs += 1
            dmg = {"peak": {}, "sustained_0.2s": {}}
            return {"mates": [{"id": "m1", "achieved": False}, {"id": "m2", "achieved": False}], "weights": {"m1": 1.0, "m2": 1.0},
                    "damage": dmg, "n_achieved": 0, "n_fastened": 0, "all_mates": False, "stage_score": 0.0, "safety_ok": True,
                    "sim_time_s": 0.0, "episode_wall_s": 0.0, "calls": {}, "camera_frames": {}, "camera_frames_total": 0,
                    "render_wall_s": 0.0, "renderer": None, "layout": {}, "end_reason": "finish", "violations": [],
                    "step_calls": 0 if case == "gate_nostep" else 3, "illegal_call_count": 0,
                    "remote_sockets_seen": 1 if case == "gate_net_left" else 0, "net_outbound": 4 if case == "gate_net_left" else 0,
                    "dropped_parts": [], "lost_parts": [], "off_design": []}
        V.run_episode = fake_episode
        out["gates"] = [list(V.episode_gates(0, 0, 0, 0, 0)), list(V.episode_gates(5, 1, 2, 0, 0)), list(V.episode_gates(5, 0, 0, 0, 0)),
                        list(V.episode_gates(5, 1, 2, 4, 1)), list(V.episode_gates(0, 1, 2, 4, 1))]
        run_main()
    elif case == "kit_task_error":
        def bad_kit(cfg):
            V.fail("the cell's kit: injected task error")
        V.fetch_hidden_gt = bad_kit
        run_main()
    elif case == "cell_build_task_error":
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        V.contract_check = lambda cfg, rn: (True, "ok", 0, [])
        class Broken(object):
            def __init__(self, *a, **kw):
                raise ValueError("injected cell fault")
        V.HandInfoServer = Broken
        run_main()
    elif case == "missing_config_key":
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        V.token_state = lambda: (True, False)
        V.contract_check = lambda cfg, rn: (True, "ok", 0, [])
        V.HandInfoServer = DummyServer
        drop_keys.append("privileged_wall_cap_s")
        run_main()
    elif case == "record_root_unwritable":
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        V.contract_check = lambda cfg, rn: (True, "ok", 0, [])
        V.HandInfoServer = DummyServer
        blocker = os.path.join(tmp, "blocker")
        open(blocker, "w").close()
        V.RECORD_ROOT = os.path.join(blocker, "rec")
        V.LOGS = os.path.join(V.RECORD_ROOT, "logs")
        V.OUT = os.path.join(V.RECORD_ROOT, "out")
        run_main()
    elif case == "grader_never_parses":
        import ast
        import io, tokenize      # noqa: F401 - imported before ast.parse is broken
        cfg = V.load_json(os.path.join(tests, "grader_config.json"))
        rn = V.Runner(cfg)
        sub = V.SUBMISSION
        shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(sub)
        with open(os.path.join(sub, "assemble.py"), "w") as fh:
            fh.write("import data\n\n\ndef solve(cell):\n    cell.step(1)\n")
        with open(os.path.join(sub, "data.py"), "w") as fh:
            fh.write("DATA = [" + ",".join(str(i % 10) for i in range(1000000)) + "]\n")
        # a 2 MB .sh declaring punycode: if the grader decoded it in-process it would stall (quadratic); the child reads
        # it as UTF-8 with replacement, streaming
        with open(os.path.join(sub, "notes.sh"), "wb") as fh:
            fh.write(b"# coding: punycode\necho hi\nx-" + b"a" * (2 * 1024 * 1024) + b"\n")
        import time as _t
        real_parse = ast.parse
        def broken_parse(*a, **kw):
            raise RuntimeError("the grader parsed a source in its own process")
        ast.parse = broken_parse
        hwm0 = vm_kb("VmHWM")
        t0 = _t.time()
        try:
            r = V.contract_check(cfg, rn)
        except RuntimeError as e:
            r = (False, "raised: %s" % e, None, None)
        out["seconds"] = round(_t.time() - t0, 2)
        ast.parse = real_parse
        out["result"] = [r[0], r[1], r[2]]
        out["hwm_delta_kb"] = vm_kb("VmHWM") - hwm0
    elif case == "contract_reasons":
        cfg = V.load_json(os.path.join(tests, "grader_config.json"))
        sub = V.SUBMISSION
        reasons = {}
        susp = {}
        rn = V.Runner(cfg)
        def put(files):
            shutil.rmtree(sub, ignore_errors=True)
            os.makedirs(sub)
            for name, data in files.items():
                with open(os.path.join(sub, name), "wb") as fh:
                    fh.write(data)
        steps = b"def solve(cell):\n    cell.step(1)\n"
        put({"helper.py": steps})
        r = V.contract_check(cfg, rn)
        reasons["no_assemble"], susp["no_assemble"] = r[1], r[2]
        put({"assemble.py": steps + b"x = 1" + b"+1" * 100000 + b"\n"})
        r = V.contract_check(cfg, rn)
        reasons["deep_assemble"], susp["deep_assemble"] = r[1], r[2]
        put({"assemble.py": steps, "deep.py": b"x = 1" + b"+1" * 100000 + b"\n"})
        reasons["deep_helper"] = V.contract_check(cfg, rn)[1]
        cfg_as = json.loads(json.dumps(cfg))
        cfg_as["parse_limits"]["address_space_gib"] = 1          # 1 GiB: about 2.5 MB of dense code
        put({"assemble.py": steps, "dense.py": b"x = 1\n" * 700000})
        r = V.contract_check(cfg_as, rn)
        reasons["dense_helper"], susp["dense_helper"], out["dense_net"] = r[1], r[2], r[3]
        cfg_t = json.loads(json.dumps(cfg))
        cfg_t["parse_limits"]["per_file_s"] = 1
        put({"assemble.py": steps, "slow.py": b"x = 1\n" * 400000})
        r = V.contract_check(cfg_t, rn)
        reasons["slow_helper"], out["slow_net"] = r[1], r[3]
        put({"assemble.py": steps + b"# \x00\n"})
        reasons["nul_assemble"] = V.contract_check(cfg, rn)[1]
        put({"assemble.py": steps + b"s = '\xe9t\xe9'\n"})
        reasons["latin_assemble"] = V.contract_check(cfg, rn)[1]
        cfg_small = json.loads(json.dumps(cfg))
        cfg_small.setdefault("route_gate", {})["max_submission_mb"] = 1
        put({"assemble.py": steps + b"#" * (1024 * 1024 + 10) + b"\n"})
        opened = []
        real_open = builtins.open
        def tracking_open(path, *a, **kw):
            opened.append(os.path.basename(str(path)))
            return real_open(path, *a, **kw)
        V.open = tracking_open
        r = V.contract_check(cfg_small, rn)
        reasons["big_assemble"], susp["big_assemble"] = r[1], r[2]
        out["opened_big"] = list(opened)
        del opened[:]
        put({"assemble.py": steps, "part1.py": b"#" * (600 * 1024) + b"\n", "part2.py": b"#" * (600 * 1024) + b"\n"})
        reasons["sum_over"] = V.contract_check(cfg_small, rn)[1]
        out["opened_sum"] = list(opened)
        del V.open
        # a 2 MB .sh that declares punycode must not stall the grader (it decodes nothing in-process): passes
        # the route gate, contract reaches the import check, within a few seconds
        import time as _t
        put({"assemble.py": steps, "notes.sh": b"# coding: punycode\necho hi\nx-" + b"a" * (2 * 1024 * 1024) + b"\n"})
        _t0 = _t.time(); r = V.contract_check(cfg, rn); out["puny_sh"] = [r[0], r[1], round(_t.time() - _t0, 2)]
        # a .py declaring a text encoding other than UTF-8 / ASCII / Latin-1 (assemble.py, a helper)
        put({"assemble.py": b"# coding: punycode\n" + steps})
        r = V.contract_check(cfg, rn); reasons["enc_assemble"], susp["enc_assemble"] = r[1], r[2]
        put({"assemble.py": steps, "h_utf7.py": b"# coding: utf-7\n+AGk-mport socket\n"})
        r = V.contract_check(cfg, rn); reasons["enc_helper"], susp["enc_helper"], out["enc_net"] = r[1], r[2], r[3]
        put({"assemble.py": steps, "h_cp.py": b"# -*- coding: cp1252 -*-\nx = 1\n"})
        reasons["enc_cp1252"] = V.contract_check(cfg, rn)[1]
        # a Latin-1 source with its declaration and a non-ASCII byte is accepted (reaches the import check)
        put({"assemble.py": b"# coding: latin-1\n# a comment with \xe9\n" + steps})
        r = V.contract_check(cfg, rn); out["latin_decl_ok"], reasons["latin_decl"] = r[0], r[1]
        # the entry cap (max_entries lowered) refuses a tree with too many entries, before any read
        cfg_e = json.loads(json.dumps(cfg)); cfg_e.setdefault("route_gate", {})["max_entries"] = 50
        put(dict({"assemble.py": steps}, **{"f%03d.txt" % i: b"" for i in range(80)}))
        reasons["too_many_entries"] = V.contract_check(cfg_e, rn)[1]
        # the whole-audit time limit (total_s lowered) refuses a slow audit with the total's reason
        cfg_tot = json.loads(json.dumps(cfg)); cfg_tot["parse_limits"]["total_s"] = 2
        put({"assemble.py": steps, "slow.py": b"x = 1\n" * 900000})
        r = V.contract_check(cfg_tot, rn); reasons["audit_total"], susp["audit_total"] = r[1], r[2]
        out["reasons"] = reasons
        out["suspicious"] = susp
    elif case == "token_caps":
        sub = os.path.join(tmp, "tok")
        os.makedirs(sub)
        tok = os.path.join(sub, ".oracle_token")
        V.SUBMISSION = sub                      # token_state() reads the submission (the authoring path: no snapshot)
        import time as _t
        real_open, real_os_open = builtins.open, os.open
        opened = []
        def guard(path):
            if os.path.basename(str(path)) == ".oracle_token" and refuse[0]:
                opened.append(case_now[0])
                raise OSError("the token was opened")
        def g_open(path, *a, **kw):
            guard(path)
            return real_open(path, *a, **kw)
        def g_os_open(path, *a, **kw):
            guard(path)
            return real_os_open(path, *a, **kw)
        V.open = g_open
        os.open = g_os_open
        refuse, case_now, states = [True], [""], {}
        try:
            token_cases(tok, sub, refuse, case_now, states, real_open)
        finally:
            os.open = real_os_open
            del V.open
        out["states"], out["opened"] = states, opened
        out.update(states.pop("_stats"))
    elif case.startswith("early_"):
        import time as _t
        sub = V.SUBMISSION
        shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(sub)
        with builtins.open(os.path.join(sub, "assemble.py"), "w") as fh:
            fh.write("def solve(cell):\n    cell.step(1)\n")
        if case == "early_sparse":
            with builtins.open(os.path.join(sub, "data.bin"), "wb") as fh:
                fh.truncate(8 * 1024 ** 3)          # 8 GiB, sparse: no disk used
        elif case == "early_entries":
            for i in range(12000):
                builtins.open(os.path.join(sub, "e%05d" % i), "wb").close()
        elif case == "early_deep":
            deep_make(sub, 3000)
        # the verify-start lockdown with the root check answered yes; everything that touches processes, accounts or
        # the stage is stubbed, the snapshot and the late-write walk only record their calls (they copy nothing)
        calls = {"snapshot_dir": 0, "late_entries": 0}
        def rec_snapshot(src, dst):
            calls["snapshot_dir"] += 1
            return 0
        def rec_late(root, t_ref, exclude=(), skip_uids=(0,)):
            calls["late_entries"] += 1
            return []
        class NoFreezer(object):
            frozen = ()
            uid = None
            @classmethod
            def for_user(cls, *a, **kw):
                return cls()
            def start(self, *a, **kw):
                return self
            def stop(self):
                pass
            def freeze_now(self):
                return 0
        stage = os.path.join(tmp, "stage")
        os.makedirs(stage)
        shutil.copyfile(os.path.join(tests, "grader_config.json"), os.path.join(stage, "grader_config.json"))
        V.lockdown.snapshot_dir, V.lockdown.late_entries = rec_snapshot, rec_late
        V.lockdown.stage_upload_time = lambda s: None        # the late-write gate off: no walk of shared folders
        V.lockdown.stop_and_kill_uid = lambda uid, grace_s=8.0: (0, 0)
        V.verify_freeze.AgentFreezer = NoFreezer
        V.runner.lock_stage = lambda root: None
        V.AGENT_USER = "hand-test-no-such-account"
        real_getuid, real_stage = os.getuid, V.STAGE_DIR
        V.STAGE_DIR = stage
        hwm0 = vm_kb("VmHWM")
        t0 = _t.time()
        os.getuid = lambda: 0
        try:
            early = V._early_lockdown()
        finally:
            os.getuid = real_getuid
            V.STAGE_DIR = real_stage
        out["seconds"] = round(_t.time() - t0, 2)
        out["hwm_delta_kb"] = vm_kb("VmHWM") - hwm0
        out["calls"] = dict(calls)
        out["cap_reason"] = early.get("cap_reason")
        out["exception"] = (early.get("exception") or "")[-300:]
        out["snapshot_exists"] = os.path.exists(V.SNAPSHOT)
        # the verdict that follows: the contract check and the episodes must not run
        V.EARLY = early
        fake_gt()
        V.fetch_hidden_gt = lambda cfg: (2, 0.0)
        ran = []
        def no_contract(cfg, rn):
            ran.append("contract_check")
            raise RuntimeError("the contract check ran on a submission beyond a cap")
        def no_episode(*a, **kw):
            ran.append("run_episode")
            raise RuntimeError("an episode ran on a submission beyond a cap")
        V.contract_check, V.run_episode = no_contract, no_episode
        try:
            run_main()
        finally:
            out["ran"] = ran
            if case == "early_deep":
                deep_remove(sub, 3000)
    elif case == "deep_contract":
        cfg = V.load_json(os.path.join(tests, "grader_config.json"))
        rn = V.Runner(cfg)
        sub = V.SUBMISSION
        shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(sub)
        with builtins.open(os.path.join(sub, "assemble.py"), "w") as fh:
            fh.write("def solve(cell):\n    cell.step(1)\n")
        deep_make(sub, 3000)
        try:
            r = V.contract_check(cfg, rn)
            out["result"] = [r[0], r[1], r[2]]
        except Exception as e:  # noqa: BLE001 - the state before the walk: an error in the copy for the import check
            out["result"] = [False, "raised %s" % type(e).__name__, None]
        finally:
            deep_remove(sub, 3000)
except SystemExit as e:
    out["system_exit"] = e.code
out["verdict_exists"] = os.path.isfile(os.environ["ROBOTICS_GRADER_VERDICT_PATH"])
print("CHILD_RESULT " + json.dumps(out))
'''


class TestGraderVerdictInEveryCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.py = _grader_python()
        if cls.py is None:
            raise unittest.SkipTest("no interpreter that imports the grader (numpy, mujoco): set ARB_NUMPY_PYTHON")

    def _run(self, case):
        tmp = _short_tmp()
        try:
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "LANG": "C.UTF-8",
                   "ALE_STAGE_DIR": str(TESTS), "ALE_HOME": os.path.join(tmp, "home"),
                   "ARB_ROOT_PAYLOAD": os.path.join(tmp, "arb"), "ARB_RECORD_ROOT": os.path.join(tmp, "rec"),
                   "ROBOTICS_GRADER_VERDICT_PATH": os.path.join(tmp, "verdict.json"), "MUJOCO_GL": "osmesa",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            os.makedirs(os.path.join(tmp, "home", "submission"))
            r = subprocess.run([self.py, "-I", "-B", "-c", CHILD, case, str(TESTS), tmp], env=env, capture_output=True,
                               text=True, timeout=600)
            line = [l for l in r.stdout.splitlines() if l.startswith("CHILD_RESULT ")]
            self.assertTrue(line, "child gave no result (rc %s)\n%s" % (r.returncode, r.stderr[-3000:]))
            res = json.loads(line[-1][len("CHILD_RESULT "):])
            vp = os.path.join(tmp, "verdict.json")
            res["verdict"] = json.load(open(vp)) if os.path.isfile(vp) else None
            lp = os.path.join(tmp, "rec", "logs", "grader_exception.log")
            res["record"] = open(lp).read() if os.path.isfile(lp) else None
            res["stderr"] = r.stderr
            return res
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _assert_exception_verdict(self, res):
        v = res["verdict"]
        self.assertIsNotNone(v, res["stderr"][-3000:])
        self.assertNotIn("system_exit", res)
        self.assertEqual(v["metrics"]["grader_exception"], 1.0)
        self.assertEqual(v["metrics"]["contract_ok"], 0.0)
        self.assertEqual(v["rewards"]["reward"], 0.0)
        self.assertIn("RuntimeError", res["stderr"])
        self.assertNotIn("injected fault 7f3a", res["stderr"])          # the message stays in the maintainers' record
        self.assertIn("injected fault 7f3a", res["record"] or "")

    def test_fault_in_the_contract_check_gives_the_exception_verdict(self):
        self._assert_exception_verdict(self._run("contract_raises"))

    def test_fault_in_an_episode_gives_the_exception_verdict(self):
        self._assert_exception_verdict(self._run("episode_raises"))

    def _assert_gate_reason(self, case, reason):
        res = self._run(case)
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        v = res["verdict"]
        self.assertIsNotNone(v, res["stderr"][-2000:])
        self.assertEqual(v["metrics"]["contract_ok"], 0.0)
        self.assertEqual(v["metrics"]["grader_exception"], 0.0)
        self.assertEqual(v["rewards"]["reward"], 0.0)
        # before the episodes the log says that the check passed and that the episodes decide; it never prints the
        # contract as passed (contract_ok=True / contract_ok 1) before they have decided it
        err = res["stderr"]
        self.assertIn("contract check passed before the episodes (contract_ok follows from the episodes' gates)", err)
        self.assertNotIn("contract_ok=True", err)
        self.assertNotIn("contract_ok 1", err)
        self.assertIn("[grader] contract_ok 0: " + reason, err)
        return res

    def test_a_run_without_a_step_call_is_logged_with_the_step_rule(self):
        res = self._assert_gate_reason("gate_nostep", "no step call over the episodes (the program must make at least one step call)")
        g = res["gates"]
        self.assertEqual(g[0][:3], [0.0, 1.0, 1.0])
        self.assertEqual(g[0][3], ["no step call over the episodes (the program must make at least one step call)"])
        self.assertEqual(g[2], [1.0, 1.0, 1.0, []])

    def test_a_process_that_survives_the_kill_is_logged_with_its_rule(self):
        res = self._assert_gate_reason(
            "gate_left", "2 process(es) of the run account still running 8 s after the kill at an episode's end (1 episode(s))")
        g = res["gates"]
        self.assertEqual(g[1][:3], [1.0, 0.0, 1.0])
        self.assertEqual(g[1][3], ["2 process(es) of the run account still running 8 s after the kill at an episode's end (1 episode(s))"])

    def test_the_episode_gates_reasons_follow_the_instructions_order(self):
        # the instruction: "then the rules about the episodes (a `step` call, no network activity, no process left)"
        net = "network activity during an episode (4 outbound events, 1 remote sockets seen)"
        left = "2 process(es) of the run account still running 8 s after the kill at an episode's end (1 episode(s))"
        res = self._assert_gate_reason("gate_net_left", net)
        err = res["stderr"]
        self.assertIn("[grader] contract_ok 0: " + left, err)
        self.assertLess(err.index("[grader] contract_ok 0: " + net), err.index("[grader] contract_ok 0: " + left))
        g = res["gates"]
        self.assertEqual(g[3][:3], [1.0, 0.0, 0.0])
        self.assertEqual(g[3][3], [net, left])
        self.assertEqual(g[4][:3], [0.0, 0.0, 0.0])
        self.assertEqual(g[4][3], ["no step call over the episodes (the program must make at least one step call)", net, left])

    def test_a_task_error_before_the_line_stays_loud(self):
        res = self._run("kit_task_error")
        self.assertEqual(res.get("system_exit"), 1, res["stderr"][-2000:])
        self.assertIsNone(res["verdict"])
        self.assertIn("TASK ERROR", res["stderr"])

    def test_the_cells_build_stays_a_task_error(self):
        res = self._run("cell_build_task_error")
        self.assertEqual(res.get("system_exit"), 1, res["stderr"][-2000:])
        self.assertIsNone(res["verdict"])
        self.assertIn("could not be built: ValueError", res["stderr"])

    def test_a_missing_config_value_stays_a_task_error(self):
        res = self._run("missing_config_key")
        self.assertEqual(res.get("system_exit"), 1, res["stderr"][-2000:])
        self.assertIsNone(res["verdict"], "a verdict was written for a fault of the task's own configuration")
        self.assertIn("TASK ERROR", res["stderr"])
        self.assertIn("privileged_wall_cap_s", res["stderr"])

    def test_an_unwritable_record_root_stays_a_task_error(self):
        res = self._run("record_root_unwritable")
        self.assertEqual(res.get("system_exit"), 1, res["stderr"][-2000:])
        self.assertIsNone(res["verdict"], "a verdict was written although the record root cannot be made")
        self.assertIn("TASK ERROR", res["stderr"])

    def test_the_grader_never_parses_a_source_itself(self):
        res = self._run("grader_never_parses")
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        self.assertEqual(res["result"][:2], [True, "ok"], res["stderr"][-2000:])
        self.assertLess(res["hwm_delta_kb"], 300 * 1024, "the grader's own peak memory rose by %d kB" % res["hwm_delta_kb"])
        self.assertLess(res["seconds"], 60, "the contract check took %s s (a 2 MB punycode .sh decoded in-process?)" % res["seconds"])

    def test_contract_reasons_for_unparsable_and_oversized_sources(self):
        res = self._run("contract_reasons")
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        rs = res["reasons"]
        self.assertEqual(rs["deep_assemble"], "assemble.py cannot be parsed: RecursionError")
        self.assertEqual(rs["deep_helper"], "network audit: networking import(s) deep.py: cannot be parsed (RecursionError)")
        self.assertEqual(rs["dense_helper"], "dense.py: cannot be parsed within the grader's limits (1 GiB, 300 s)")
        self.assertEqual(res["dense_net"], [])
        self.assertEqual(rs["slow_helper"], "slow.py: cannot be parsed within the grader's limits (12 GiB, 1 s)")
        self.assertEqual(res["slow_net"], [])
        self.assertEqual(rs["no_assemble"], "no assemble.py (regular file) in the submission")
        for k in ("no_assemble", "deep_assemble", "dense_helper", "big_assemble"):
            self.assertEqual(res["suspicious"][k], -1, k)
        self.assertTrue(rs["nul_assemble"].startswith("assemble.py does not parse: "), rs)
        self.assertTrue(rs["latin_assemble"].startswith("assemble.py does not parse: "), rs)
        self.assertEqual(rs["big_assemble"], "route gate: submission larger than 1 MB")
        self.assertEqual(rs["sum_over"], "route gate: submission larger than 1 MB")
        self.assertEqual(res["opened_big"], [], "a file was opened although assemble.py is above the cap")
        self.assertFalse([n for n in res["opened_sum"] if n.startswith("part")], res["opened_sum"])
        # a 2 MB punycode .sh passes the route gate quickly (no in-process quadratic decode)
        self.assertEqual(res["puny_sh"][0], True, res["puny_sh"])
        self.assertLess(res["puny_sh"][2], 60, "a punycode .sh stalled the route gate: %s s" % res["puny_sh"][2])
        # a .py that declares a disallowed text encoding is refused with the encoding reason
        self.assertEqual(rs["enc_assemble"], "assemble.py: declares the text encoding punycode; sources must be UTF-8, ASCII or Latin-1")
        self.assertEqual(res["suspicious"]["enc_assemble"], -1)
        self.assertEqual(rs["enc_helper"], "source audit: h_utf7.py declares the text encoding utf-7; sources must be UTF-8, ASCII or Latin-1")
        self.assertEqual(res["enc_net"], [["h_utf7.py", "declares the text encoding utf-7; sources must be UTF-8, ASCII or Latin-1"]])
        self.assertIn("cp1252", rs["enc_cp1252"])
        # a Latin-1 source with its declaration and a non-ASCII byte is accepted
        self.assertEqual([res["latin_decl_ok"], rs["latin_decl"]], [True, "ok"])
        # the entry cap and the whole-audit time limit, each with its own reason
        self.assertEqual(rs["too_many_entries"], "route gate: submission has more than 50 files and folders")
        self.assertEqual(rs["audit_total"], "the source audit did not finish within 2 s")
        self.assertEqual(res["suspicious"]["audit_total"], -1)

    def test_the_oracle_token_is_lstat_checked_before_it_is_opened(self):
        res = self._run("token_caps")
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        st = res["states"]
        for k in ("sparse_5g", "4097", "fifo", "folder"):
            self.assertEqual(st[k], [False, True], "%s: %s" % (k, st))
        self.assertEqual(res["opened"], [], "the token was opened in: %s" % res["opened"])
        if "reference" in st:
            self.assertEqual(st["link"], [False, True], st)
            self.assertEqual(st["reference"], [True, False], "the reference's own token no longer switches the mode")
        self.assertLess(res["seconds"], 60, res)
        self.assertLess(res["hwm_delta_kb"], 60 * 1024, "the grader's peak memory rose by %d kB" % res["hwm_delta_kb"])

    def _assert_early_cap(self, case, reason):
        res = self._run(case)
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        self.assertEqual(res["exception"], "", res["exception"])
        self.assertEqual(res["cap_reason"], reason, res)
        self.assertEqual(res["calls"], {"snapshot_dir": 0, "late_entries": 0}, "the snapshot or the late walk ran")
        self.assertFalse(res["snapshot_exists"])
        self.assertEqual(res["ran"], [], "ran on a submission beyond a cap: %s" % res["ran"])
        v = res["verdict"]
        self.assertIsNotNone(v, res["stderr"][-2000:])
        self.assertEqual(v["metrics"]["contract_ok"], 0.0)
        self.assertEqual(v["metrics"]["grader_exception"], 0.0)
        self.assertEqual(v["rewards"]["reward"], 0.0)
        self.assertIn("contract_ok 0: %s" % reason, res["stderr"])
        self.assertLess(res["seconds"], 60, res)
        self.assertLess(res["hwm_delta_kb"], 60 * 1024, "the grader's peak memory rose by %d kB" % res["hwm_delta_kb"])

    def test_an_8_gib_sparse_file_is_refused_before_the_snapshot(self):
        self._assert_early_cap("early_sparse", "route gate: submission larger than 60 MB")

    def test_12000_files_are_refused_before_the_snapshot(self):
        self._assert_early_cap("early_entries", "route gate: submission has more than 10000 files and folders")

    def test_a_tree_3000_deep_is_refused_before_the_snapshot(self):
        self._assert_early_cap("early_deep", "route gate: submission has a path longer than 4000 bytes")

    def test_a_plain_submission_is_snapshotted_as_before(self):
        res = self._run("early_plain")
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        self.assertIsNone(res["cap_reason"])
        self.assertEqual(res["calls"]["snapshot_dir"], 1, res)

    def test_the_contract_check_refuses_a_tree_3000_deep(self):
        res = self._run("deep_contract")
        self.assertNotIn("system_exit", res, res["stderr"][-2000:])
        self.assertEqual(res["result"], [False, "route gate: submission has a path longer than 4000 bytes", -1],
                         res["result"])


if __name__ == "__main__":
    unittest.main()
