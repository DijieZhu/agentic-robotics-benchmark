"""Hidden grader for cloth_material_id_from_video (plans in cphys, executes on the real fabric, creal).

Reads /home/user/submission/params.json (numbers only) and scores fabrics A-J on the fixed hidden tests
(tests/ref/fabrics.json, four per family and fabric): every test is planned by ctasks.plan in a world made of the
submitted material and then executed once on the real fabric (creal, its real material). A fabric's score is the
product of its two family means; the metric `material_transfer_score` is the mean over the ten fabrics (higher is
better). Separate verifier sandbox; agent processes are killed first and nothing else the agent left is read.

Nothing here depends on the clock: there is no deadline inside the grader (the verifier's timeout bounds the stage,
and a stage that runs out of time has no score). Every test runs in its own process; a test that ends abnormally is
run once more, and if it ends abnormally again the stage fails without a score (an environment error, never a zero).
The numerics fingerprint (tests/numerics_check.py) is checked before anything is graded.
"""
import json
import os
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cgrading as CG  # noqa: E402
from robotics_grader import fail, ratio_rewards, read_anchor, stage_config, write_verdict, zero_verdict  # noqa: E402

HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT = os.path.basename(HOME.rstrip("/")) or "user"
REF = os.path.join(HERE, "ref", "fabrics.json")
ONE = os.path.join(HERE, "one_test.py")
NUMERICS = os.path.join(HERE, "numerics_check.py")
NUMERICS_REF = os.path.join(HERE, "numerics.json")
TEST_TIMEOUT_S = 1800


def stop_agent(max_rounds=20):
    for _ in range(max_rounds):
        r = subprocess.run(["pgrep", "-u", AGENT], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            return True
        subprocess.run(["runuser", "-u", AGENT, "--", "kill", "-KILL", "-1"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(0.2)
    return False


def read_submission():
    """Bytes of params.json, never following a symlink; None if absent or not a regular file."""
    sub = os.path.join(HOME, "submission")
    try:
        dfd = os.open(sub, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None, "no submission directory"
    try:
        try:
            fd = os.open("params.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError:
            return None, "no regular params.json"
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None, "params.json is not a regular file"
            chunks, total = [], 0
            while total <= CG.MAX_BYTES:
                b = os.read(fd, 65536)
                if not b:
                    break
                chunks.append(b)
                total += len(b)
            return b"".join(chunks), None
        finally:
            os.close(fd)
    finally:
        os.close(dfd)


class GraderError(RuntimeError):
    """The grading machinery failed (not the submission): the verify stage ends without a score."""


def check_numerics():
    """Run numerics_check.py in a fresh interpreter and compare with tests/numerics.json: a machine that does not
    reproduce the recorded fingerprint would grade the same submission differently, so nothing is graded there."""
    try:
        r = subprocess.run([sys.executable, "-I", "-B", NUMERICS], capture_output=True, text=True, timeout=900, cwd=HERE)
        got = json.loads(r.stdout.strip().splitlines()[-1])
        want = json.load(open(NUMERICS_REF))
    except Exception as exc:
        fail("the numerics check did not run: {}: {}".format(type(exc).__name__, str(exc)[:200]))
    if got.get("fingerprint") != want.get("fingerprint"):
        diff = [k for k in want.get("parts", {}) if got.get("parts", {}).get(k) != want["parts"][k]]
        fail("this machine does not reproduce the recorded numerics (tests/numerics.json): parts that differ: {}; "
             "numpy {} mujoco {}".format(diff, got.get("numpy"), got.get("mujoco")))


def run_one(job):
    """One test in its own process. An abnormal end (a crash, a timeout, no result line) is not the submission's:
    the same test is run once more, and a second abnormal end is a grader error (no score is written)."""
    key, i, material = job
    payload = json.dumps({"material": material, "fabric": key, "index": i})
    last = ""
    for attempt in range(2):
        try:
            r = subprocess.run([sys.executable, "-I", "-B", ONE], input=payload, capture_output=True, text=True,
                               timeout=TEST_TIMEOUT_S, cwd=HERE)
            if r.returncode == 0:
                out = json.loads(r.stdout.strip().splitlines()[-1])
                q = float(out["q"])
                if 0.0 <= q <= 1.0:
                    return key, i, q, attempt
            last = "exit %s %s" % (r.returncode, (r.stderr or "")[-200:])
        except subprocess.TimeoutExpired:
            last = "timeout after %d s" % TEST_TIMEOUT_S
        except Exception as exc:
            last = "%s: %s" % (type(exc).__name__, str(exc)[:200])
    raise GraderError("test %s/%d ended abnormally twice: %s" % (key, i, last))


def main():
    t0 = time.time()
    config = stage_config()
    anchor, meta = read_anchor()
    anchored = anchor is not None
    cleanup_ok = stop_agent()   # best effort in a fresh verifier sandbox; recorded, never a reason to score 0
    check_numerics()
    data, why = read_submission()
    if data is None:
        write_verdict(zero_verdict(reason=why, anchored=anchored))
        return
    try:
        entries, invalid = CG.parse_submission_bytes(data)
    except CG.InvalidSubmission as exc:
        write_verdict(zero_verdict(reason="invalid params.json: %s" % exc, anchored=anchored))
        return
    ref = CG.load_ref(REF)
    order = {"overhang": 0, "slide": 1}
    jobs = [(k, i, entries[k]) for k in CG.FABRICS if k in entries for i in range(len(ref[k]["tests"]))]
    jobs.sort(key=lambda j: order[ref[j[0]]["tests"][j[1]]["fam"]])
    qs = {k: [0.0] * len(ref[k]["tests"]) for k in CG.FABRICS}
    reruns = 0
    if jobs:
        try:
            with ThreadPoolExecutor(max_workers=int(config.get("procs", 4))) as ex:
                for k, i, q, attempt in ex.map(run_one, jobs):
                    qs[k][i] = q
                    reruns += attempt
        except GraderError as exc:
            fail("the grader could not finish: %s" % exc)
    per = {}
    for k in CG.FABRICS:
        per[k] = CG.aggregate(qs[k], ref[k]["tests"]) if k in entries else (0.0, {})
    measured = float(sum(per[k][0] for k in CG.FABRICS) / len(CG.FABRICS))
    extra = {"grading_seconds": round(time.time() - t0, 1), "agent_cleanup_incomplete": 0.0 if cleanup_ok else 1.0,
             "tests_rerun": float(reruns), "numerics_ok": 1.0}
    lines = []
    for k in CG.FABRICS:
        s, fm = per[k]
        extra["score_" + k] = round(s, 4)
        for f, v in fm.items():
            extra["q_%s_%s" % (k, f)] = round(v, 4)
        if k in invalid:            # metrics are numbers only: the flag here, the reason in the diagnostic
            extra["invalid_" + k] = 1.0
            lines.append("fabric %s scores 0: %s" % (k, str(invalid[k])[:200]))
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="material_transfer_score", extra=extra, full_at=meta.get("full_at", 1.0))
    lines.append("material_transfer_score=%.4f anchor=%s reward=%.3f %.0fs reruns=%d" % (
        measured, anchor, verdict["rewards"]["reward"], time.time() - t0, reruns))
    write_verdict(verdict, diagnostic="\n".join(lines))
    print("[grader] " + lines[-1], file=sys.stderr)


if __name__ == "__main__":
    main()
