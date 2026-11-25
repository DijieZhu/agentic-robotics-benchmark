"""Run the task's real tests/verify.py on a development hidden set (maintainer tool; used again by the freeze and the
grader's QC).

    run_verify.py --work DIR --tag NAME --program DIR [--oracle] --site PKG[:SEED] [--site ...]
                  [--time-limit 2400] [--reference-dir DIR] [--anchor 1.0]

Builds, under DIR, what the verify stage would hold and runs verify.py there as the current account (no root: no
lockdown, no account switch, the no-network filter and every gate as in the image):
  stage_<tag>/        a copy of the task's tests/ (cellkit, voidcell, voidscore, robotics_grader, verify.py,
                      grader_config.json with `episodes` = the number of sites) plus, where tests/ does not have them
                      yet (the reference pinning, the freeze and the image build write them): hidden/{sites/<name>.tar.xz, seeds.json, limits.json, oracle_token.sha256},
                      reference/ (from --reference-dir), anchor.json (value --anchor), given_hashes.json
  arb_<tag>/          the root-owned copy of the givens verify.py checks (here: given/lib/voidcell = the client)
  home_<tag>/submission   a copy of --program (+ .oracle_token with --oracle)
and prints / writes summary_<tag>.json: the verdict (reward, void_search_score, contract_ok), every episode's score,
c, F, end reason, sim time and wall time, and every file under the stage, the home and the givens copy whose name
says 'debrief' other than the code (voidscore/debrief.py): a scored episode must leave none.
"""
import argparse
import hashlib
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
T_TESTS = os.path.join(TASK, "tests")
DEV_TOKEN = b"development-oracle-token-of-run_verify-not-the-task-token-00000000000"
CLIENT_INIT = '"""voidcell (the team\'s copy): only the station\'s client, voidcell.client."""\n'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    with open(p, "rb") as fh:
        return sha(fh.read())


def archive_of(pkg, name):
    """bytes of a deterministic .tar.xz of a site package (an archive is taken as it is)."""
    if os.path.isfile(pkg):
        with open(pkg, "rb") as fh:
            return fh.read()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for m in ("site.json", "site.xml"):
            data = open(os.path.join(pkg, m), "rb").read()
            ti = tarfile.TarInfo("%s/%s" % (name, m))
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
    return lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)


def ignore(_d, names):
    return [n for n in names if n in ("__pycache__", ".pytest_cache") or n.endswith(".pyc")]


def build(a):
    stage = os.path.join(a.work, "stage_" + a.tag)
    arb = os.path.join(a.work, "arb_" + a.tag)
    home = os.path.join(a.work, "home_" + a.tag)
    for d in (stage, arb, home):
        if os.path.exists(d):
            shutil.rmtree(d)
    shutil.copytree(T_TESTS, stage, ignore=ignore)
    made = []
    hid = os.path.join(stage, "hidden")
    os.makedirs(os.path.join(hid, "sites"), exist_ok=True)
    seeds, rows = [], []
    for k, spec in enumerate(a.site):
        pkg, _, seed = spec.partition(":")
        seed = int(seed) if seed else 1000 + k
        name = "d%02d" % (k + 1)
        data = archive_of(pkg, name)
        with open(os.path.join(hid, "sites", name + ".tar.xz"), "wb") as fh:
            fh.write(data)
        seeds.append(seed)
        rows.append({"archive": "sites/%s.tar.xz" % name, "seed": seed, "sha256": sha(data), "source": os.path.abspath(pkg)})
    with open(os.path.join(hid, "seeds.json"), "w") as fh:
        json.dump({"seeds": seeds, "sites": rows, "note": "DEVELOPMENT hidden set of run_verify.py: not the task's"}, fh,
                  indent=1)
    if not os.path.isfile(os.path.join(hid, "limits.json")) or a.force_dev_hidden:
        with open(os.path.join(hid, "limits.json"), "w") as fh:
            json.dump({"time_limit_s": a.time_limit, "note": "DEVELOPMENT copy (the freeze writes the task's)"}, fh, indent=1)
        made.append("hidden/limits.json")
    with open(os.path.join(hid, "oracle_token.sha256"), "w") as fh:
        fh.write(sha(DEV_TOKEN) + "\n")
    made += ["hidden/sites", "hidden/seeds.json", "hidden/oracle_token.sha256 (development token)"]
    ref = os.path.join(stage, "reference")
    if not os.path.isfile(os.path.join(ref, "reference.py")):
        if not a.reference_dir:
            raise SystemExit("tests/reference/reference.py is not there yet: give --reference-dir")
        shutil.copytree(a.reference_dir, ref, ignore=ignore, dirs_exist_ok=True)
        made.append("reference/ (from %s)" % a.reference_dir)
    anc = os.path.join(stage, "anchor.json")
    if not os.path.isfile(anc) or a.force_dev_hidden:
        with open(anc, "w") as fh:
            json.dump({"direction": "higher", "full_at": 0.95, "metric": "void_search_score", "value": a.anchor,
                       "note": "DEVELOPMENT anchor of run_verify.py (the freeze writes the task's)"}, fh, indent=1)
        made.append("anchor.json (value %s)" % a.anchor)
    cfgp = os.path.join(stage, "grader_config.json")
    cfg = json.load(open(cfgp))
    cfg["episodes"] = len(a.site)
    if a.pose_source:
        cfg["pose_source"] = a.pose_source
    with open(cfgp, "w") as fh:
        json.dump(cfg, fh, indent=1)
    lib = os.path.join(arb, "given", "lib", "voidcell")
    os.makedirs(lib)
    shutil.copyfile(os.path.join(T_TESTS, "voidcell", "client.py"), os.path.join(lib, "client.py"))
    with open(os.path.join(lib, "__init__.py"), "w") as fh:
        fh.write(CLIENT_INIT)
    gh = os.path.join(stage, "given_hashes.json")
    if not os.path.isfile(gh):
        with open(gh, "w") as fh:
            json.dump({"given/lib/voidcell/__init__.py": sha_file(os.path.join(lib, "__init__.py")),
                       "given/lib/voidcell/client.py": sha_file(os.path.join(lib, "client.py"))}, fh, indent=1)
        made.append("given_hashes.json (the client only)")
    for dp, dns, fns in os.walk(arb):
        for d in dns:
            os.chmod(os.path.join(dp, d), 0o755)
        for f in fns:
            os.chmod(os.path.join(dp, f), 0o644)
    sub = os.path.join(home, "submission")
    shutil.copytree(a.program, sub, ignore=ignore)
    if a.oracle:
        with open(os.path.join(sub, ".oracle_token"), "wb") as fh:
            fh.write(DEV_TOKEN + b"\n")
    return stage, arb, home, made


def debrief_files(*roots):
    out = []
    for r in roots:
        for dp, _dns, fns in os.walk(r):
            out += [os.path.join(dp, f) for f in fns if "debrief" in f.lower() and not f.endswith(".py")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--site", action="append", required=True)
    ap.add_argument("--time-limit", type=float, default=2400.0)
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--pose-source", default=None)
    ap.add_argument("--force-dev-hidden", action="store_true", help="write the development limits and anchor even when "
                                                                   "the task's tests/ has its own")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    stage, arb, home, made = build(a)
    env = dict(os.environ)
    env.update({"ALE_STAGE_DIR": stage, "ALE_HOME": home, "ARB_ROOT_PAYLOAD": arb, "PYTHONNOUSERSITE": "1",
                "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "MESA_SHADER_CACHE_DISABLE": "true",
                "LP_NUM_THREADS": env.get("LP_NUM_THREADS", "8"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
                "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"})
    env.pop("ALE_VERIFY_PYTHON", None)
    t0 = time.time()
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
            eps.append({"site": s.get("site"), "source": None, "episode_score": s.get("episode_score"), "c": sc.get("c"),
                        "F": sc.get("F"), "done_reason": s.get("done_reason"), "end_reason": s.get("end_reason"),
                        "sim_time_s": s.get("sim_time_s"), "episode_wall_s": round(s.get("episode_wall_s", 0.0), 1),
                        "program_time_s": round(s.get("program_time_s", 0.0), 2), "frames": (s.get("budgets") or {}).get("frames_used"),
                        "reports": [(r["i"], r["accepted"]) for r in s.get("reports", [])],
                        "final_state_sha256": s.get("final_state_sha256"), "reference": s.get("reference")})
    seeds = json.load(open(os.path.join(stage, "hidden", "seeds.json")))
    for e, r in zip(eps, seeds["sites"]):
        e["source"] = r["source"]
    m = (verdict or {}).get("metrics") or {}
    out = {"tag": a.tag, "program": os.path.abspath(a.program), "oracle": a.oracle, "rc": rc, "wall_s": round(wall, 1),
           "reward": ((verdict or {}).get("rewards") or {}).get("reward"),
           "void_search_score": m.get("void_search_score"), "contract_ok": m.get("contract_ok"),
           "privileged_mode": m.get("privileged_mode"), "diagnostic": (verdict or {}).get("diagnostic"),
           "episodes": eps, "dev_made": made,
           "debrief_files": debrief_files(stage, home, arb), "log": log_path, "stage": stage}
    with open(os.path.join(a.work, "summary_%s.json" % a.tag), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("tag", "rc", "wall_s", "reward", "void_search_score", "contract_ok",
                                          "debrief_files")}))
    print(json.dumps([{k: e[k] for k in ("site", "episode_score", "c", "F", "done_reason", "sim_time_s", "episode_wall_s")}
                      for e in eps]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
