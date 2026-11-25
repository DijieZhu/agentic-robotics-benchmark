"""Run the task's real tests/verify.py on the FROZEN hidden set for the grader's own QC (maintainer tool).

    qc_verify.py --work DIR --tag NAME --submission {oracle | oracle-no-token | oracle-wrong-token | <program dir>}
                 [--variant NAME | --variant-file PLAN.json] [--episodes N]

The stage is a byte copy of the task's tests/ as it is (hidden/seeds.json, hidden/sites/, limits.json,
oracle_token.sha256, reference/, anchor.json, given_hashes.json, verify.py, the cell and the scoring). The
root-owned givens copy (ARB_ROOT_PAYLOAD) is what tests/test.sh builds: a byte copy of the stage's setup_payload/
(modes 0755 / 0644, owned by the running account), so the stage's own given_hashes.json is checked file by file.
The submission:
  oracle               solution/solve.sh (operate.py and the real token)
  oracle-no-token      the oracle's operate.py without any token (its privileged call must be refused)
  oracle-wrong-token   the oracle's operate.py with a token that is not the task's (must be ignored)
  <dir>                a program folder (the degenerate submissions of tests/review/degenerate/)
--variant NAME / --variant-file PLAN.json (with --submission oracle): the maintainers' switch of reference_run: the
stage's reference/reference.py is replaced by variants/qc_reference.py (next to this file) and the plan (NAME of
variants/plans.json, or the plan file itself: the degenerate submissions that need the true pose keep their plan in
their own folder of tests/review/degenerate/) is written next to it as reference/qc_variant.json (the reference's library
_lib/ stays the task's bytes).
Runs as the current account (no root: no lockdown, no account switch; the no-network filter and every other gate as in
the image). The verify process gets the numerics of tests/test.sh (Haswell, L2 256, the NPY cap; BLAS 2, OMP 1,
LP 8). As tests/test.sh does, the hidden sites' plate bundles come from the stage's local copy
assets/plates.tar.xz when it is there (ARB_HIDDEN_PLATES_TARBALL is set to the stage's copy; verify.py only reads it
and checks it against the sha256 and the file count that grader_config.json pins). Writes DIR/summary_<tag>.json: the verdict, every episode's score details (c, F, per-report rows with the
distance and the dwell reasons, retrieval), reports, the reference summary (with the variant's log), final state
sha256 and its comparison with the admission run pinned in seeds.json, cell failures, and every file under the stage,
the home and the givens copy whose name says 'debrief' other than code.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
T_TESTS = os.path.join(TASK, "tests")
SOLVE = os.path.join(TASK, "solution", "solve.sh")
ORACLE_OPERATE = os.path.join(TASK, "solution", "reference", "operate.py")
VARIANTS = os.path.join(HERE, "variants")
WRONG_TOKEN = b"qc-a-token-that-is-not-the-task-token-000000000000000000000000000000000000000000"


def sha_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ignore(_d, names):
    return [n for n in names if n in ("__pycache__", ".pytest_cache") or n.endswith(".pyc")]


def load_plan(a):
    if a.variant_file:
        with open(a.variant_file, encoding="utf-8") as fh:
            return json.load(fh), "file " + os.path.relpath(a.variant_file, DEV)
    plans = json.load(open(os.path.join(VARIANTS, "plans.json"), encoding="utf-8"))["variants"]
    if a.variant not in plans:
        raise SystemExit("no variant %r in plans.json" % a.variant)
    return plans[a.variant], "plans.json " + a.variant


def build(a):
    stage, arb, home = (os.path.join(a.work, p + "_" + a.tag) for p in ("stage", "arb", "home"))
    for d in (stage, arb, home):
        if os.path.exists(d):
            shutil.rmtree(d)
    shutil.copytree(T_TESTS, stage, ignore=ignore)
    added = []
    plan = None
    if a.variant or a.variant_file:
        if a.submission != "oracle":
            raise SystemExit("--variant / --variant-file need --submission oracle")
        plan, src = load_plan(a)
        shutil.copyfile(os.path.join(VARIANTS, "qc_reference.py"), os.path.join(stage, "reference", "reference.py"))
        with open(os.path.join(stage, "reference", "qc_variant.json"), "w") as fh:
            json.dump(plan, fh, indent=1)
        added.append("reference/reference.py = variants/qc_reference.py; reference/qc_variant.json = %s" % src)
    for need in ("anchor.json", "given_hashes.json", "setup_payload"):
        if not os.path.exists(os.path.join(stage, need)):
            raise SystemExit("tests/%s is missing" % need)
    if a.episodes:
        cp = os.path.join(stage, "grader_config.json")
        cfg = json.load(open(cp))
        cfg["episodes"] = int(a.episodes)
        with open(cp, "w") as fh:
            json.dump(cfg, fh, indent=1)
        added.append("grader_config.json episodes = %d (development check)" % a.episodes)
    shutil.copytree(os.path.join(stage, "setup_payload"), arb)
    for dp, _dns, fns in os.walk(arb):
        os.chmod(dp, 0o755)
        for f in fns:
            os.chmod(os.path.join(dp, f), 0o644)
    os.makedirs(home)
    sub = os.path.join(home, "submission")
    if a.submission == "oracle":
        rc = subprocess.call(["bash", SOLVE], env=dict(os.environ, ALE_HOME=home))
        if rc != 0:
            raise SystemExit("solve.sh failed: %d" % rc)
    elif a.submission in ("oracle-no-token", "oracle-wrong-token"):
        os.makedirs(sub)
        shutil.copyfile(ORACLE_OPERATE, os.path.join(sub, "operate.py"))
        if a.submission == "oracle-wrong-token":
            with open(os.path.join(sub, ".oracle_token"), "wb") as fh:
                fh.write(WRONG_TOKEN + b"\n")
    else:
        shutil.copytree(a.submission, sub, ignore=ignore)
    return stage, arb, home, added, plan


def episode_row(s, pin):
    sc = s.get("score") or {}
    ref = s.get("reference") if isinstance(s.get("reference"), dict) else {}
    return {"tag": pin.get("tag"), "site": s.get("site"), "candidate": pin.get("candidate"),
            "episode_score": s.get("episode_score"), "S": sc.get("S"), "c": sc.get("c"), "F": sc.get("F"),
            "points": sc.get("points"), "retrieval": sc.get("retrieval"),
            "done_reason": s.get("done_reason"), "end_reason": s.get("end_reason"), "end": s.get("end"),
            "sim_time_s": s.get("sim_time_s"), "episode_wall_s": round(s.get("episode_wall_s", 0.0), 1),
            "program_time_s": round(s.get("program_time_s", 0.0), 2),
            "frames": (s.get("budgets") or {}).get("frames_used"), "budgets": s.get("budgets"),
            "reports": s.get("reports"), "calls": s.get("calls"),
            "illegal_call_count": s.get("illegal_call_count"), "internal_errors": s.get("internal_errors"),
            "internal_errors_live": s.get("internal_errors_live"), "render_failures": s.get("render_failures"),
            "retried_after": s.get("retried_after"),
            "n_commands": len(s.get("commands") or []), "n_history": len(s.get("executor_history") or []),
            "final_state_sha256": s.get("final_state_sha256"),
            "pinned_reference_final_state_sha256": pin.get("reference_final_state_sha256"),
            "same_as_admission_run": s.get("final_state_sha256") == pin.get("reference_final_state_sha256"),
            "reference_summary": {k: ref.get(k) for k in ("version", "n_prims", "n_under_slab", "n_blind_in",
                                                          "stuck_events", "end_triangles", "aborted", "sim_time_s",
                                                          "inline_score", "done_reason")},
            "qc_variant": ref.get("qc_variant")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--variant-file", default=None)
    ap.add_argument("--episodes", type=int, default=None, help="development check only: the first N pinned sites")
    a = ap.parse_args()
    if a.variant and a.variant_file:
        raise SystemExit("--variant or --variant-file, not both")
    os.makedirs(a.work, exist_ok=True)
    stage, arb, home, added, plan = build(a)
    env = dict(os.environ)
    env.update({"ALE_STAGE_DIR": stage, "ALE_HOME": home, "ARB_ROOT_PAYLOAD": arb, "PYTHONNOUSERSITE": "1",
                "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "MESA_SHADER_CACHE_DISABLE": "true",
                "LP_NUM_THREADS": "8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
                "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"})
    env.pop("ALE_VERIFY_PYTHON", None)
    plates = os.path.join(stage, "assets", "plates.tar.xz")
    if os.path.isfile(plates):                   # as tests/test.sh: the local copy replaces the pinned download
        env["ARB_HIDDEN_PLATES_TARBALL"] = plates
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_path = os.path.join(a.work, "verify_%s.log" % a.tag)
    with open(log_path, "w") as lf:
        rc = subprocess.call([sys.executable, "-B", "-s", os.path.join(stage, "verify.py")], cwd=stage, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdict = None
    vp = os.path.join(stage, "verdict.local.json")
    if os.path.isfile(vp):
        verdict = json.load(open(vp))
    seeds = json.load(open(os.path.join(stage, "hidden", "seeds.json")))
    eps = []
    ep_path = os.path.join(stage, "out", "episodes.json")
    if os.path.isfile(ep_path):
        for s, pin in zip(json.load(open(ep_path)), seeds["sites"]):
            eps.append(episode_row(s, pin))
    m = (verdict or {}).get("metrics") or {}
    debriefs = []
    for r in (stage, home, arb):
        for dp, _dns, fns in os.walk(r):
            debriefs += [os.path.join(dp, f) for f in fns if "debrief" in f.lower() and not f.endswith(".py")]
    out = {"tag": a.tag, "submission": a.submission, "variant": a.variant,
           "variant_file": os.path.abspath(a.variant_file) if a.variant_file else None, "plan": plan,
           "rc": rc, "started_utc": started, "ended_utc": ended, "wall_s": round(wall, 1),
           "reward": ((verdict or {}).get("rewards") or {}).get("reward"),
           "void_search_score": m.get("void_search_score"), "contract_ok": m.get("contract_ok"),
           "privileged_mode": m.get("privileged_mode"), "oracle_token_mismatch": m.get("oracle_token_mismatch"),
           "illegal_call_count": m.get("illegal_call_count"), "network_imports": m.get("network_imports"),
           "root_payload_ok": m.get("root_payload_ok"), "time_limit_s": m.get("time_limit_s"),
           "acceptance_version": m.get("acceptance_version"),
           "episode_retries": m.get("episode_retries"), "grader_exception": m.get("grader_exception"),
           "diagnostic": (verdict or {}).get("diagnostic"), "metrics": m,
           "episodes": eps, "episodes_json": ep_path if os.path.isfile(ep_path) else None,
           "stage_added": added, "debrief_files": debriefs, "log": log_path, "stage": stage,
           "files_sha256": {p: sha_file(os.path.join(stage, p)) for p in
                            ("hidden/seeds.json", "hidden/limits.json", "hidden/oracle_token.sha256", "anchor.json",
                             "verify.py", "grader_config.json", "given_hashes.json", "test.sh", "voidcell/client.py",
                             "voidcell/server.py", "voidscore/scoring.py", "voidscore/acceptance.yaml",
                             "reference/reference.py")},
           "operate_py_sha256": sha_file(os.path.join(home, "submission", "operate.py"))
           if os.path.isfile(os.path.join(home, "submission", "operate.py")) else None}
    with open(os.path.join(a.work, "summary_%s.json" % a.tag), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("tag", "rc", "wall_s", "reward", "void_search_score", "contract_ok",
                                          "privileged_mode", "oracle_token_mismatch", "root_payload_ok",
                                          "debrief_files")}))
    for e in eps:
        print(json.dumps({k: e[k] for k in ("tag", "episode_score", "c", "F", "sim_time_s", "episode_wall_s",
                                            "same_as_admission_run")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
