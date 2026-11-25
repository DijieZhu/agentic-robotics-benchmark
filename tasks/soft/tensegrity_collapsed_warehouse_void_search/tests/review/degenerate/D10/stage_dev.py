#!/usr/bin/env python3
"""D10 on a DEVELOPMENT site through the task's real tests/verify.py (task version 2.0; maintainer tool, the
development check of degenerate submission D10 before it runs on the frozen hidden set through
tests/review/qc/qc_verify.py --submission oracle --variant-file D10/plan.json).

    stage_dev.py --work DIR --tag NAME --site PKG [--seed N] [--time-limit 1820] [--plan PLAN.json] [--keep]

Builds under DIR (keep it mode 700) what the verify stage would hold, as run_verify.py's build() does
(run_verify.py itself is not changed: it has no switch for the reference and copies the whole of tests/), and runs
verify.py there as the current account (no root: no lockdown, no account switch):
  stage_<tag>/   a copy of the task's tests/ WITHOUT tests/hidden (the frozen archives, seeds.json, selection.json and
                 limits.json never enter the stage), then, written by this script:
                   hidden/sites/d01.tar.xz   the development package (run_verify.archive_of: deterministic tar.xz)
                   hidden/seeds.json         {"seeds": [N], "sites": [{archive, seed, sha256, ...}]}
                   hidden/limits.json        {"time_limit_s": --time-limit} (always written: verify.py reads it)
                   hidden/oracle_token.sha256  run_verify's development token
                   anchor.json               run_verify's development anchor (value 1.0)
                   grader_config.json        episodes = 1
                   given_hashes.json         ONLY the files the givens copy really holds (the client library); the
                                             task's own given_hashes.json may be older than the givens
                   reference/reference.py    = tests/review/qc/variants/qc_reference.py; reference/qc_variant.json = PLAN
  arb_<tag>/     the root-owned copy of the givens verify.py checks: given/lib/voidcell/{__init__.py, client.py}
  home_<tag>/submission   the oracle's program (solution/reference/operate.py) + the development token
and writes DIR/summary_<tag>.json: the verdict, the episode (score, c, F, end reason, sim time, the cell's internal
errors), the variant's record (the heap it pressed, the pushes, the cell-side monitor's contact samples with every
heap__ geom, the reference rule's stuck rows) and the stage's file hashes. Without --keep the three folders are
removed afterwards (the summary stays).
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
DEG = os.path.dirname(HERE)
DEV = os.path.normpath(os.path.join(DEG, "..", ".."))
sys.path.insert(0, DEG)
import run_verify as rv  # noqa: E402  (archive_of, the development token and the client __init__; read-only import)

TASK = rv.TASK
T_TESTS = rv.T_TESTS
QC_REF = os.path.join(DEV, "tools", "qc", "variants", "qc_reference.py")
OPERATE = os.path.join(TASK, "solution", "reference", "operate.py")
OURS = ("limits.json", "oracle_token.sha256", "seeds.json", os.path.join("sites", "d01.tar.xz"))


def sha_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ignore_top(src, names):
    out = [n for n in names if n in ("__pycache__", ".pytest_cache") or n.endswith(".pyc")]
    if os.path.abspath(src) == os.path.abspath(T_TESTS) and "hidden" in names:
        out.append("hidden")
    return out


def build(a):
    stage, arb, home = (os.path.join(a.work, p + "_" + a.tag) for p in ("stage", "arb", "home"))
    for d in (stage, arb, home):
        if os.path.exists(d):
            shutil.rmtree(d)
    shutil.copytree(T_TESTS, stage, ignore=ignore_top)
    if os.path.exists(os.path.join(stage, "hidden")):
        raise SystemExit("stage_dev: tests/hidden entered the stage")
    hid = os.path.join(stage, "hidden")
    os.makedirs(os.path.join(hid, "sites"))
    data = rv.archive_of(a.site, "d01")
    with open(os.path.join(hid, "sites", "d01.tar.xz"), "wb") as fh:
        fh.write(data)
    with open(os.path.join(hid, "seeds.json"), "w") as fh:
        json.dump({"seeds": [a.seed], "sites": [{"archive": "sites/d01.tar.xz", "seed": a.seed,
                                                  "sha256": rv.sha(data), "source": os.path.abspath(a.site)}],
                   "note": "DEVELOPMENT hidden set of D10/stage_dev.py: not the task's"}, fh, indent=1)
    with open(os.path.join(hid, "limits.json"), "w") as fh:
        json.dump({"time_limit_s": float(a.time_limit), "note": "DEVELOPMENT copy (the freeze writes the task's)"}, fh,
                  indent=1)
    with open(os.path.join(hid, "oracle_token.sha256"), "w") as fh:
        fh.write(rv.sha(rv.DEV_TOKEN) + "\n")
    with open(os.path.join(stage, "anchor.json"), "w") as fh:
        json.dump({"direction": "higher", "full_at": 0.95, "metric": "void_search_score", "value": 1.0,
                   "note": "DEVELOPMENT anchor of D10/stage_dev.py (the freeze writes the task's)"}, fh, indent=1)
    cfgp = os.path.join(stage, "grader_config.json")
    cfg = json.load(open(cfgp))
    cfg["episodes"] = 1
    with open(cfgp, "w") as fh:
        json.dump(cfg, fh, indent=1)
    shutil.copyfile(QC_REF, os.path.join(stage, "reference", "reference.py"))
    plan = json.load(open(a.plan))
    with open(os.path.join(stage, "reference", "qc_variant.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    lib = os.path.join(arb, "given", "lib", "voidcell")
    os.makedirs(lib)
    shutil.copyfile(os.path.join(T_TESTS, "voidcell", "client.py"), os.path.join(lib, "client.py"))
    with open(os.path.join(lib, "__init__.py"), "w") as fh:
        fh.write(rv.CLIENT_INIT)
    with open(os.path.join(stage, "given_hashes.json"), "w") as fh:
        json.dump({"given/lib/voidcell/__init__.py": sha_file(os.path.join(lib, "__init__.py")),
                   "given/lib/voidcell/client.py": sha_file(os.path.join(lib, "client.py"))}, fh, indent=1)
    for dp, dns, fns in os.walk(arb):
        for d in dns:
            os.chmod(os.path.join(dp, d), 0o755)
        for f in fns:
            os.chmod(os.path.join(dp, f), 0o644)
    sub = os.path.join(home, "submission")
    os.makedirs(sub)
    shutil.copyfile(OPERATE, os.path.join(sub, "operate.py"))
    with open(os.path.join(sub, ".oracle_token"), "wb") as fh:
        fh.write(rv.DEV_TOKEN + b"\n")
    return stage, arb, home, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--time-limit", type=float, default=1820.0)
    ap.add_argument("--plan", default=os.path.join(HERE, "plan.json"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    stage, arb, home, plan = build(a)
    hidden_files = sorted(os.path.relpath(os.path.join(dp, f), os.path.join(stage, "hidden"))
                          for dp, _d, fs in os.walk(os.path.join(stage, "hidden")) for f in fs)
    env = dict(os.environ)
    env.update({"ALE_STAGE_DIR": stage, "ALE_HOME": home, "ARB_ROOT_PAYLOAD": arb, "PYTHONNOUSERSITE": "1",
                "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "MESA_SHADER_CACHE_DISABLE": "true",
                "LP_NUM_THREADS": "8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
                "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"})
    env.pop("ALE_VERIFY_PYTHON", None)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_path = os.path.join(a.work, "verify_%s.log" % a.tag)
    with open(log_path, "w") as lf:
        rc = subprocess.call([sys.executable, "-B", "-s", os.path.join(stage, "verify.py")], cwd=stage, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    verdict = None
    vp = os.path.join(stage, "verdict.local.json")
    if os.path.isfile(vp):
        verdict = json.load(open(vp))
    eps = []
    ep_path = os.path.join(stage, "out", "episodes.json")
    if os.path.isfile(ep_path):
        for s in json.load(open(ep_path)):
            sc = s.get("score") or {}
            ref = s.get("reference") if isinstance(s.get("reference"), dict) else {}
            qv = ref.get("qc_variant") or {}
            log = qv.get("log") or []
            eps.append(dict(site=s.get("site"), episode_score=s.get("episode_score"), S=sc.get("S"), c=sc.get("c"),
                            F=sc.get("F"), done_reason=s.get("done_reason"), end_reason=s.get("end_reason"),
                            sim_time_s=s.get("sim_time_s"), episode_wall_s=round(s.get("episode_wall_s", 0.0), 1),
                            reports=s.get("reports"), internal_errors=s.get("internal_errors"),
                            internal_errors_live=s.get("internal_errors_live"), render_failures=s.get("render_failures"),
                            retried_after=s.get("retried_after"), final_state_sha256=s.get("final_state_sha256"),
                            reference_summary={k: ref.get(k) for k in ("n_prims", "n_under_slab", "aborted",
                                                                        "sim_time_s", "inline_score", "done_reason",
                                                                        "stuck_rows_reference_rule", "monitor",
                                                                        "heap_contact_samples")},
                            variant_log=[{k: v for k, v in e.items() if k != "rows"} for e in log]))
    m = (verdict or {}).get("metrics") or {}
    out = dict(tag=a.tag, plan=plan, rc=rc, started_utc=started, wall_s=round(wall, 1), time_limit_s=a.time_limit,
               reward=((verdict or {}).get("rewards") or {}).get("reward"),
               void_search_score=m.get("void_search_score"), contract_ok=m.get("contract_ok"),
               privileged_mode=m.get("privileged_mode"), root_payload_ok=m.get("root_payload_ok"),
               grader_exception=m.get("grader_exception"), diagnostic=(verdict or {}).get("diagnostic"), metrics=m,
               episodes=eps, log=log_path,
               stage_hidden_files=hidden_files,
               stage_hidden_only_ours=sorted(hidden_files) == sorted(OURS),
               files_sha256={p: sha_file(os.path.join(stage, p)) for p in
                             ("hidden/seeds.json", "hidden/limits.json", "hidden/oracle_token.sha256", "anchor.json",
                              "verify.py", "grader_config.json", "given_hashes.json", "voidcell/server.py",
                              "voidscore/scoring.py", "reference/reference.py", "reference/qc_variant.json")})
    with open(os.path.join(a.work, "summary_%s.json" % a.tag), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    if not a.keep:
        for d in (stage, arb, home):
            shutil.rmtree(d, ignore_errors=True)
    print(json.dumps({k: out[k] for k in ("tag", "rc", "wall_s", "reward", "void_search_score", "contract_ok",
                                          "privileged_mode", "root_payload_ok")}))
    for e in eps:
        print(json.dumps({k: e[k] for k in ("episode_score", "c", "F", "done_reason", "sim_time_s", "internal_errors")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
