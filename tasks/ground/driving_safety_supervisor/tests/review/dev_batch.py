"""dev_batch - the reference expert on development episodes (a maintainer's tool; the grader never runs it).

Development seeds lie below 2**32, outside the range the hidden set's seeds are drawn from ([2**32, 2**62), by
os.urandom), so nothing tuned here can be a hidden episode.

    python3 -B dev_batch.py make --out SPECS.json [--per-variant 8] [--rng 20261106] [--variants ...]
    python3 -B dev_batch.py run --specs SPECS.json --out DIR [--gpus 0,1] [--cpus 16-23,24-31] [--image IMG]
    python3 -B dev_batch.py report DIR [DIR ...] [--md]

`make`: per graded variant `--per-variant` episodes, towns cycled as the grader cycles them (graded_cells; a variant
limited to some towns cycles over those), each with a seed below 2**32, a distractor count and a lock key from
random.Random(--rng); grouped by town.
`run` (host; `--solver FILE` swaps the submission's solver.py, e.g. for the expert before this work): one
container of the task image per GPU (named pr24t2-dev-<n>, no network, 32 GB, the given cores,
NVIDIA_DRIVER_CAPABILITIES with graphics), each started only once its GPU has 8 GB free, running a share of the
episodes (whole towns) through review/run_episode.py exactly as the grader would: tests/ staged root-only at
/opt/ale/tests, solution/solve.sh run as the agent account, the submission then left root-owned as the engine
restores it, the expert's debug log in DIR. The containers are removed when they end.
`report`: per variant the mean score, failures (not completed, any collision, red light or abort), the smallest
clearance, steps, harsh brakes, pace; the gate of the design (section 1.5) per variant; the episodes that failed.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
sys.path.insert(0, TESTS)

IMAGE = "ale-framework:c803df660b128dc54c63d9d48cca95b5"
GPU_UUIDS = ("GPU-d19373da-d7df-4311-0891-9f572961c478", "GPU-18963cff-56a1-bce7-9532-6af2a14ebe90")
TOWNS = ("Town04", "Town05", "Town10HD")
GATE = {"debris": 0.96, "walking_in_lane": 0.96, "shoulder_pass": 0.92, "spinning_car": 0.82,
        "overtaking_oncoming": 0.95, "reversing_car": 0.87, "dropped_cargo": 0.90, "crowd_against_light": 0.92}


def make(args):
    from drive_variants import GRADED_MIX, TOWNS_FOR, VARIANTS
    rng = random.Random(args.rng)
    kind_of = {v: kind for kind, variants in VARIANTS.items() for v in variants}
    variants = args.variants or [v for v, _ in GRADED_MIX]
    specs = []
    for variant in variants:
        towns = [t for t in TOWNS if t in TOWNS_FOR.get(variant, TOWNS)] or list(TOWNS)
        for k in range(args.per_variant):
            seed = rng.randrange(1, 1 << 32)
            specs.append({"id": "d{}-{}-{}".format(args.rng % 1000, variant[:6], k), "town": towns[k % len(towns)],
                          "kind": kind_of[variant], "variant": variant, "seed": seed,
                          "distractors": rng.randrange(0, 3), "lock_key": "%064x" % rng.getrandbits(256)})
    specs.sort(key=lambda s: TOWNS.index(s["town"]))
    with open(args.out, "w") as handle:
        json.dump(specs, handle, indent=1)
    print("{} specs -> {}".format(len(specs), args.out))


def gpu_free_mib(index):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i",
                          str(index)], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return int(out.strip().splitlines()[0])


CONTAINER_SCRIPT = r"""
set -e
mkdir -p /opt/ale && rm -rf /opt/ale/tests && cp -a /task/tests /opt/ale/tests
find /opt/ale/tests -name __pycache__ -prune -exec rm -rf {{}} +
chown -R root:root /opt/ale/tests && chmod -R go-rwx /opt/ale/tests && chmod 700 /opt/ale
rm -rf /home/user/submission && mkdir -p /home/user/submission && chown user:user /home/user/submission
runuser -u user -- env ALE_HOME=/home/user bash /task/solution/solve.sh
{swap}chown -R root:root /home/user/submission
mkdir -p /out/logs && chmod 777 /out/logs
cd /opt/ale/tests
ALE_STAGE_DIR=/opt/ale/tests ALE_HOME=/home/user SUP_DEBUG=1 SUP_LOG=/out/logs/expert_{shard}.log \
  python3 -B /opt/ale/tests/review/run_episode.py --specs /out/specs_{shard}.json --out /out \
  --footprints /task/solution/footprints.json --steps 2> /out/run_{shard}.stderr
"""


def run(args):
    with open(args.specs) as handle:
        specs = json.load(handle)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o777)
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.solver:      # another expert in the same submission (e.g. the one before this work)
        import shutil
        shutil.copy(args.solver, os.path.join(out, "solver_override.py"))
    cpus = args.cpus.split(",")
    # contiguous chunks of the town-ordered list, one per container (each loads a town or two)
    specs = sorted(specs, key=lambda s: TOWNS.index(s["town"]))
    size = -(-len(specs) // len(gpus))
    shards = [specs[i * size:(i + 1) * size] for i in range(len(gpus))]
    procs = []
    for n, (gpu, cpu, shard) in enumerate(zip(gpus, cpus, shards)):
        if not shard:
            continue
        shard.sort(key=lambda s: TOWNS.index(s["town"]))
        with open(os.path.join(out, "specs_{}.json".format(n)), "w") as handle:
            json.dump(shard, handle, indent=1)
        while gpu_free_mib(gpu) < 8192:
            sys.stderr.write("[dev] GPU {} has {} MiB free; waiting\n".format(gpu, gpu_free_mib(gpu)))
            time.sleep(60)
        name = "pr24t2-dev-{}-{}".format(os.path.basename(out.rstrip("/"))[-12:], n)
        cmd = ["docker", "run", "--rm", "--name", name, "--gpus", "device=" + GPU_UUIDS[gpu],
               "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics", "--cpuset-cpus", cpu,
               "--memory", "32g", "--shm-size", "8g", "--network", "none", "--user", "root",
               "-v", "{}:/task:ro".format(TASK), "-v", "{}:/out".format(out), args.image,
               "bash", "-c", CONTAINER_SCRIPT.format(
                   shard=n, swap="cp /out/solver_override.py /home/user/submission/solver.py\n" if args.solver else "")]
        sys.stderr.write("[dev] {} on GPU {} ({} MiB free), cpus {}: {} episodes\n".format(
            name, gpu, gpu_free_mib(gpu), cpu, len(shard)))
        procs.append((name, subprocess.Popen(cmd)))
        time.sleep(5)
    rc = 0
    for name, proc in procs:
        rc |= proc.wait()
        sys.stderr.write("[dev] {} exit {}\n".format(name, proc.returncode))
    return rc


def failed(r):
    t = r["truth"]
    c = t.get("collisions", {})
    return (r["aborted"] or t.get("termination") != "completed" or t.get("hazard_collisions", 0) > 0 or
            any(c.get(k, 0) for k in ("walker", "vehicle", "static")) or t.get("red_lights", 0) > 0)


def report(args):
    rows = []
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "results.jsonl"))):
            with open(path) as handle:
                rows += [json.loads(line) for line in handle if line.strip()]
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["spec"]["variant"], []).append(r)
    lines = ["| variant | n | mean | gate | fails | min clearance | mean pace | harsh/ep | steps/ep | probes moved |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    total, n_all, fails_all, moved, pairs = 0.0, 0, 0, 0, 0
    for variant in GATE:
        rs = by_variant.get(variant, [])
        if not rs:
            continue
        mean = sum(r["score"] for r in rs) / len(rs)
        fails = sum(1 for r in rs if failed(r))
        clear = [r["clearance"]["all"][0] for r in rs if r["clearance"].get("all")]
        pace = sum(r["truth"].get("pace", 0.0) for r in rs) / len(rs)
        harsh = sum(r["truth"].get("harsh_brakes", 0) for r in rs) / len(rs)
        steps = sum(r["truth"].get("steps", 0) for r in rs) / len(rs)
        pm = sum(r["probes"][1] for r in rs)
        pt = sum(r["probes"][0] for r in rs)
        moved += pm
        pairs += pt
        total += sum(r["score"] for r in rs)
        n_all += len(rs)
        fails_all += fails
        lines.append("| {} | {} | {:.3f} | {} {} | {} | {} | {:.3f} | {:.1f} | {:.0f} | {}/{} |".format(
            variant, len(rs), mean, GATE[variant], "ok" if mean >= GATE[variant] else "BELOW", fails,
            "{:.2f}".format(min(clear)) if clear else "-", pace, harsh, steps, pm, pt))
    if n_all:
        lines.append("| **all** | {} | {:.4f} | 0.92 | {} | | | | | {}/{} |".format(
            n_all, total / n_all, fails_all, moved, pairs))
    print("\n".join(lines))
    print()
    for r in rows:
        if failed(r) or r["score"] < GATE.get(r["spec"]["variant"], 0.9) - 0.1:
            t = r["truth"]
            print("- {} {} {} seed={} d={}: score {:.3f}, {} rc={} col={} hz={} red={} harsh={} pace={} steps={} "
                  "clear={} note={}".format(r["id"], r["spec"]["town"], r["spec"]["variant"], r["spec"]["seed"],
                                            r["spec"]["distractors"], r["score"], t.get("termination"),
                                            t.get("route_completion"), t.get("collisions"),
                                            t.get("hazard_collisions"), t.get("red_lights"), t.get("harsh_brakes"),
                                            t.get("pace"), t.get("steps"),
                                            r["clearance"]["all"][0] if r["clearance"].get("all") else None,
                                            r.get("note")))
    for r in rows:
        if r.get("walker_launch"):
            print("- {} {} {}: a walker near the car over 4 m/s {} (fastest {})".format(
                r["id"], r["spec"]["town"], r["spec"]["variant"], r["walker_launch"], r.get("walker_fast")))
    replies = [r["max_reply_s"] for r in rows]
    if replies:
        print("\nslowest reply {:.2f} s; {} episodes".format(max(replies), len(rows)))
    errors = 0
    for d in args.dirs:
        for path in glob.glob(os.path.join(d, "logs", "expert_*.log")):
            with open(path) as handle:
                errors += sum(1 for line in handle if "act error" in line)
    print("expert act errors (each answered [0]): {}".format(errors))
    fidelity(args.dirs)


def fidelity(dirs):
    """The swept path's prediction against what happened, on passes: at each step with a plan, the expert's debug
    line `pred t= odo= d0= plan= path=` (real requests only) holds the rollout's (metres travelled, offset) every
    0.5 s; 10 and 20 steps later, under the same plan, the car has travelled odo' - odo and stands at offset d0'.
    The error is |d0' - the predicted offset at that distance|; p95 over all such pairs, at +1 s and +2 s.

    "Same plan" is checked twice: by the plan's start and offset (`plan=`), and strictly, with its way back too
    (the `t=` line's pass=a/b/c/e, metres from the car, made absolute with odo): a way back that moved on in
    between (the car waits until its rear is past someone walking on) changes where the car goes without the
    rollout being wrong, so only the strict pairs measure the vehicle model alone."""
    import re
    e1, e2, s1, s2 = [], [], [], []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "logs", "expert_*.log"))):
            episodes, cur = [], None
            with open(path) as handle:
                for line in handle:
                    if line.startswith("[expert] reset"):
                        cur = {"pred": {}, "plan": {}}
                        episodes.append(cur)
                    elif line.startswith("[expert] pred ") and cur is not None:
                        f = dict(kv.split("=", 1) for kv in line.split()[2:] if "=" in kv)
                        if "path" not in f:      # an expert older than the rollout's log line
                            continue
                        samples = [tuple(float(x) for x in pt.split(",")) for pt in f["path"].split(";") if pt]
                        cur["pred"][int(f["t"])] = (float(f["odo"]), float(f["d0"]), f["plan"], samples)
                    elif line.startswith("[expert] t=") and cur is not None:
                        m = re.match(r"\[expert\] t=(\d+) .* pass=(\S+)", line)
                        if m and m.group(2) != "None":
                            cur["plan"][int(m.group(1))] = [float(x) for x in m.group(2).split("/")]
            for ep in episodes:
                pred, plans = ep["pred"], ep["plan"]

                def geometry(t):
                    rel = plans.get(t)
                    return None if rel is None or t not in pred else [r + pred[t][0] for r in rel]
                for t, (odo, _d0, plan, samples) in pred.items():
                    if plan == "-" or not samples:
                        continue
                    for lag, sink, strict in ((10, e1, s1), (20, e2, s2)):
                        later = pred.get(t + lag)
                        if later is None or later[2] != plan:
                            continue
                        run = later[0] - odo
                        pts = [(0.0, pred[t][1])] + samples
                        for (r0, y0), (r1, y1) in zip(pts, pts[1:]):
                            if r0 <= run <= r1 and r1 > r0:
                                err = abs(later[1] - (y0 + (y1 - y0) * (run - r0) / (r1 - r0)))
                                sink.append(err)
                                g0, g1 = geometry(t), geometry(t + lag)
                                if g0 and g1 and all(abs(a - b) <= 0.15 for a, b in zip(g0, g1)):
                                    strict.append(err)
                                break

    def p95(xs):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else float("nan")
    print("swept-path fidelity on passes: +1 s p95 {:.3f} m (n={}), +2 s p95 {:.3f} m (n={}); gate 0.15 m".format(
        p95(e1), len(e1), p95(e2), len(e2)))
    print("  the same plan with its way back unchanged: +1 s p95 {:.3f} m (n={}), +2 s p95 {:.3f} m (n={})".format(
        p95(s1), len(s1), p95(s2), len(s2)))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("make")
    p.add_argument("--out", required=True)
    p.add_argument("--per-variant", type=int, default=8)
    p.add_argument("--rng", type=int, default=20261106)
    p.add_argument("--variants", nargs="*")
    p = sub.add_parser("run")
    p.add_argument("--specs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--cpus", default="16-23,24-31")
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--solver", help="run this solver.py in place of solution/solver.py")
    p = sub.add_parser("report")
    p.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    if args.cmd == "make":
        make(args)
    elif args.cmd == "run":
        sys.exit(run(args))
    elif args.cmd == "report":
        report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
