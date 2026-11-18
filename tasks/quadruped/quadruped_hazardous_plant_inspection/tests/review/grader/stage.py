"""The amendment 1 tests' verify run (tests/review/grader/): one `bash test.sh` the way the engine's verify stage runs it,
as root in a fresh container of the task image, on a one-site set made from a practice package (never a scored
building), with the submission a test writes. For the maintainers; nothing runs these in the verify stage.

    stage = Stage(tests_dir=None, site=None)        # tests_dir: the tests/ folder to verify with (default: this one's);
    out = stage.run({"patrol.py": "..."})           #   site: a practice package (default: the image's training site t1)
    out["rc"], out["verdict"], out["grader"], out["program_log"], out["contract_reason"]

The stage is a copy of `tests_dir` without hidden/: hidden/ holds the one package as `sites/p01.tar.xz` with its sha256
in seeds.json, and grader_config.json asks for one episode. The submission is written before the stage is copied
(the late-write gate compares with the stage's upload time), the verdict goes to the stage's verdict.local.json
(ALE_VERIFY_PYTHON unset), the grader's records under the run's own ARB_RECORD_ROOT, its root-owned copy of the
givens under the run's own ARB_ROOT_PAYLOAD. stdlib only.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
TASK = os.path.dirname(TESTS)
SITES = ("/opt/plantcell-practice/sites/t1.tar.xz", os.path.join(TASK, "environment", "practice_ground", "sites", "t1.tar.xz"))
SEED = 12345


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def default_site():
    for p in SITES:
        if os.path.isfile(p):
            return p
    raise SystemExit("no practice package found (%s): pass --site" % ", ".join(SITES))


def grader_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("[grader]")]


class Stage(object):
    def __init__(self, tests_dir=None, site=None, cfg_extra=None, keep=False):
        if os.getuid() != 0:
            raise SystemExit("run as root in a container of the task image (test.sh chowns the givens' copy to root)")
        self.tests_dir = os.path.abspath(tests_dir or TESTS)
        self.site = os.path.abspath(site or default_site())
        self.cfg_extra = dict(cfg_extra or {})
        self.keep = keep
        self.n = 0

    def run(self, files, timeout=3600, name=None):
        """files: {relative path: text} of the submission. Returns a dict with rc, wall_s, verdict (or None), grader
        (its [grader] lines), err (test.sh's stderr), program_log (the mission's program stderr, '' if none),
        contract_ok, contract_reason (from the grader's contract line) and the run folder."""
        self.n += 1
        work = tempfile.mkdtemp(prefix="amend1_%s_" % (name or self.n))
        os.chmod(work, 0o755)               # the program's account (`nobody`) traverses to its run folder below
        home = os.path.join(work, "home", "user")
        sub = os.path.join(home, "submission")
        os.makedirs(sub)
        for rel, text in files.items():
            p = os.path.join(sub, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        time.sleep(1.1)                      # the stage is uploaded after the submission was written
        st = os.path.join(work, "stage", "tests")
        shutil.copytree(self.tests_dir, st, ignore=shutil.ignore_patterns("__pycache__", "hidden"))
        hid = os.path.join(st, "hidden")
        os.makedirs(os.path.join(hid, "sites"))
        shutil.copyfile(self.site, os.path.join(hid, "sites", "p01.tar.xz"))
        with open(os.path.join(hid, "seeds.json"), "w") as f:
            json.dump({"seeds": [SEED], "sites": [{"archive": "sites/p01.tar.xz", "seed": SEED,
                                                    "sha256": sha256_file(os.path.join(hid, "sites", "p01.tar.xz"))}],
                       "note": "amendment 1 test: one practice package, never a scored building"}, f)
        cfg_p = os.path.join(st, "grader_config.json")
        cfg = json.load(open(cfg_p))
        cfg["episodes"] = 1
        cfg.update(self.cfg_extra)
        with open(cfg_p, "w") as f:
            json.dump(cfg, f, indent=1)
        os.chmod(hid, 0o700)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ALE_", "ARB_", "ROBOTICS_GRADER"))}
        env.update({"ALE_HOME": home, "ALE_STAGE_DIR": st, "ARB_ROOT_PAYLOAD": os.path.join(work, "arb"),
                    "ARB_RECORD_ROOT": os.path.join(work, "rec")})
        t0 = time.time()
        try:
            r = subprocess.run(["bash", "test.sh"], cwd=st, env=env, capture_output=True, text=True, timeout=timeout)
            rc, err = r.returncode, r.stderr
        except subprocess.TimeoutExpired as e:
            rc, err = "timeout", (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        out = {"rc": rc, "wall_s": round(time.time() - t0, 1), "err": err, "grader": grader_lines(err), "work": work}
        vp = os.path.join(st, "verdict.local.json")
        out["verdict"] = json.load(open(vp)) if os.path.isfile(vp) else None
        logs = os.path.join(work, "rec", "logs")
        out["program_log"] = ""
        if os.path.isdir(logs):
            for fn in sorted(os.listdir(logs)):
                out["program_log"] += open(os.path.join(logs, fn), errors="replace").read()
        m = (out["verdict"] or {}).get("metrics") or {}
        out["contract_ok"] = m.get("contract_ok")
        line = next((ln for ln in out["grader"] if ln.startswith("[grader] contract_ok=")), "")
        out["contract_reason"] = line[len("[grader] contract_ok="):] if line else ""
        if not self.keep:
            shutil.rmtree(work, ignore_errors=True)
        return out


def report(cases):
    """cases: [(name, ok, detail)]. Prints one line each and the total; returns the exit status."""
    bad = 0
    for name, ok, detail in cases:
        print("%s  %s: %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
        bad += 0 if ok else 1
    print("%d of %d passed" % (len(cases) - bad, len(cases)), flush=True)
    return 1 if bad else 0
