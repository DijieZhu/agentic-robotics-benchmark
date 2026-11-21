"""Is the grader's probe schedule a function of the episode alone? (verify.probe_stream; a review tool, task image)

    python3 -B test_probe_schedule.py [--tests DIR] [--work DIR]

DIR defaults: the task's tests/ (the parent of this file's directory when it sits in tests/review/, else /task/tests);
--work: a scratch directory for the stub controllers (default: a new temporary directory, removed at the end).
An episode's schedule is what the grader's lock draws from probe_stream(n, seed) through verify.ask: the steps that
carry a repeated t, the order of each pair and the seed of each perturbation. Checks:
  (a) the schedules of the set's 12 episodes over 15000 steps (an episode's longest run), drawn through verify.ask
      with a recording wire and no simulator, are identical in two child processes with different hash seeds;
  (b) the 12 schedules differ from one another;
  (c) at the reference's episode lengths (the first wire run's steps of the set's episodes in
      tests/hidden/episode_selection.json) the probed steps are about probe_rate (1 %) of all steps: within three
      binomial standard deviations;
  (d) verify.run_episodes over the set's first three episodes (the real environment, cameras and wire) in three
      child processes, one per stub controller: "mover" moves on a small circle and ignores its observation;
      "reader" moves the same way and its gripper number, which the environment clips to the same closed gripper,
      reads q and the wrench; "still" commands no motion and its gripper number reads the observation. Each ends its
      episodes at other steps (a malformed reply), so earlier episodes take different numbers of steps. Every pair
      each child met is the schedule of (a) at that step (same steps, order and perturbation seed), in the steps all
      three ran they meet the same probes; mover and reader (different replies, the same motion) receive
      byte-identical requests, copies included; still (another trajectory) gets the same offsets on q and the wrench
      (to float precision);
  (e) mover fails the lock (none of its pairs differ), reader passes (all differ), and still's pairs, whose replies
      differ in the gripper number but command no motion, are not counted and leave the lock ok.
Prints counts and digests only (no probe step, no seed). Exit 1 when a check fails, 2 when a child fails.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T_MAX = 15000                    # an episode's longest run: episode.EPISODE_STEPS
LOOP_EPISODES = 3
STOPS = {"mover": (900, 1500, 1200), "reader": (400, 1200, 1500), "still": (1300, 1500, 700)}
STEP_M, PERIOD = 0.0004, 400     # the movers' circle: 0.4 mm per step, 400 steps per turn (2.5 cm radius)
Q_TOL, W_TOL = 1e-9, 1e-9       # the wire carries q and the wrench unrounded

STUB = r'''import json, math, sys
MODE, STOPS = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
STEP_M, PERIOD = float(sys.argv[3]), float(sys.argv[4])
episode = -1


def reply(t, obs):
    if episode < len(STOPS) and t >= STOPS[episode]:
        return [0.0] * 7                       # malformed (7 numbers): the grader ends the episode here
    reads = 3.0 + 100.0 * abs(obs["wrench"][0]) + 1000.0 * abs(sum(obs["q"]))   # >= 1: clipped to a closed gripper
    if MODE == "still":
        return [0.0] * 6 + [reads, 0, 0]
    a = 2.0 * math.pi * t / PERIOD
    return [STEP_M * math.cos(a), STEP_M * math.sin(a), 0.0, 0.0, 0.0, 0.0, reads if MODE == "reader" else 2.0, 0, 0]


print("READY", flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if msg["type"] == "reset":
        episode += 1
        out = {"ok": True}
    elif msg["type"] == "act":
        out = {"action": reply(msg["t"], msg["obs"])}
    else:
        print(json.dumps({"ok": True}), flush=True)
        break
    print(json.dumps(out), flush=True)
'''


def default_tests():
    up = os.path.dirname(HERE)
    if os.path.basename(HERE) == "review" and os.path.isfile(os.path.join(up, "verify.py")):
        return up
    return "/task/tests"


def load(path):
    with open(path) as f:
        return json.load(f)


def the_set(tests):
    return [(str(e["unit"]), int(e["seed"])) for e in load(os.path.join(tests, "hidden", "episodes.json"))["episodes"]]


def modules(tests):
    sys.dont_write_bytecode = True
    sys.path.insert(0, tests)
    import shelf_env as E  # noqa: E402
    import verify as V  # noqa: E402
    return V, E


def new_lock(V, config):
    return V.IdleAwareProbeLock(None, rate=config.get("probe_rate", 0.01),
                                pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-4)


def pairs_of(log):
    """[(episode, t, copy_first, seed)] from a request log [(episode, t, seed or None)] in the order written."""
    out, k = [], 0
    while k < len(log):
        e, t, s = log[k]
        if k + 1 < len(log) and log[k + 1][:2] == (e, t):
            s2 = log[k + 1][2]
            out.append((e, t, 1 if s is not None else 0, s if s is not None else s2))
            k += 2
        else:
            k += 1
    return out


# ------------------------------------------------------------------------------------------------ children
def child_schedule(args):
    """The 12 schedules over T_MAX steps: verify.ask with the lock as run_episodes keeps it (one lock, the episode's
    stream from verify.probe_stream), a wire that records what it is asked and a stand-in environment whose copy
    carries the perturbation generator's seed."""
    V, _E = modules(args.tests)
    config = load(os.path.join(args.tests, "grader_config.json"))

    class Wire(object):
        def __init__(self):
            self.log = []

        def act(self, t, obs):
            self.log.append((self.episode, int(t), obs.get("_copy_seed")))
            return [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]

    class Env(object):
        @staticmethod
        def perturbed_observation(obs, rng):
            return {"_copy_seed": int(rng.bit_generator.seed_seq.entropy)}

    class Ep(object):
        env = Env()

    lock, wire = new_lock(V, config), Wire()
    for n, (unit, seed) in enumerate(the_set(args.tests)):
        lock.rng = V.probe_stream(n, seed)
        wire.episode = n
        for t in range(T_MAX):
            V.ask(lock, wire, Ep(), t, {"t": t}, unit, False)
    print("RESULT " + json.dumps({"pairs": pairs_of(wire.log), "requests": len(wire.log)}), flush=True)


def child_loop(args):
    """verify.run_episodes over the set's first LOOP_EPISODES episodes with one stub controller. The copy is marked
    in the grader's process (the marker is removed before the request is written) to log the pair's order and the
    perturbation's seed; every request is logged with the sha256 of the line written."""
    import subprocess as sp
    V, E = modules(args.tests)
    config = load(os.path.join(args.tests, "grader_config.json"))
    units = load(os.path.join(args.tests, "units.json"))["units"]
    episodes = the_set(args.tests)[:LOOP_EPISODES]
    specs = {u: units[u] for u in sorted({u for u, _s in episodes})}
    sub = os.path.join(args.work, args.child_stub)
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "stub.py"), "w") as f:
        f.write(STUB)
    with open(os.path.join(sub, "run.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\"\nexec %s -I -B stub.py %s %s %r %r\n" % (
            sys.executable, args.child_stub, ",".join(str(x) for x in STOPS[args.child_stub]), STEP_M, PERIOD))
    log, state = [], {"episode": -1}
    original = E.ShelfEnv.perturbed_observation

    def marked(self, obs, rng):
        seed = int(rng.bit_generator.seed_seq.entropy)
        out = original(self, obs, rng)
        out["_copy_seed"] = seed
        return out
    E.ShelfEnv.perturbed_observation = marked

    base = V.GuardedSolver          # verify's class: V.GuardedSolver becomes Recording below

    class Recording(base):
        def _spawn(self):
            return sp.Popen(["bash", self.entry], cwd=self.submission, stdin=sp.PIPE, stdout=sp.PIPE,
                            stderr=sp.DEVNULL, universal_newlines=True, bufsize=1, start_new_session=True)

        def reset(self):
            state["episode"] += 1
            return base.reset(self)

        def act(self, t, obs):
            seed = obs.pop("_copy_seed", None)
            line = json.dumps({"type": "act", "t": int(t), "obs": obs})       # what Solver.request writes
            action = base.act(self, t, obs)
            log.append({"e": state["episode"], "t": int(t), "seed": seed,
                        "sha": hashlib.sha256(line.encode()).hexdigest(), "q": obs.get("q"),
                        "wrench": obs.get("wrench"), "action": action})
            return action
    V.GuardedSolver = Recording
    (scores, aborted, _skipped, faults, _pu, _pr, _eg, _be, lock, records) = V.run_episodes(
        config, os.getuid(), "none", time.monotonic(), specs, sub, False, episodes)
    out = {"records": [{k: r[k] for k in ("index", "unit", "score", "reason", "steps")} for r in records],
           "aborted": aborted, "faults": faults,
           "lock": {"total": lock.total, "passed": lock.passed, "idle_skipped": lock.idle_skipped, "ok": lock.ok},
           "log": log}
    print("RESULT " + json.dumps(out), flush=True)


def run_child(args, extra, env=None):
    cmd = [sys.executable, "-B", os.path.abspath(__file__), "--tests", args.tests, "--work", args.work] + extra
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {})))
    lines = [line for line in p.stdout.splitlines() if line.startswith("RESULT ")]
    if p.returncode != 0 or not lines:
        print("child %s failed (exit %d):\n%s" % (" ".join(extra), p.returncode, p.stderr[-3000:]))
        return None
    return json.loads(lines[-1][7:])


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ checks
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=default_tests())
    ap.add_argument("--work", default=None)
    ap.add_argument("--child", default=None, choices=("schedule", "loop"))
    ap.add_argument("--stub", dest="child_stub", default=None, choices=sorted(STOPS))
    ap.add_argument("--json", default=None, help="also write the summary (counts and digests) to this file")
    args = ap.parse_args()
    args.tests = os.path.abspath(args.tests)
    if args.child == "schedule":
        return child_schedule(args)
    if args.child == "loop":
        return child_loop(args)
    own_work = args.work is None
    args.work = tempfile.mkdtemp(prefix="probe_schedule_") if own_work else os.path.abspath(args.work)
    try:
        return checks(args)
    finally:
        if own_work:
            shutil.rmtree(args.work, ignore_errors=True)


def checks(args):
    config = load(os.path.join(args.tests, "grader_config.json"))
    rate = float(config.get("probe_rate", 0.01))
    eps = the_set(args.tests)
    summary, ok_all = {}, True

    def report(name, ok, text):
        nonlocal ok_all
        ok_all = ok_all and ok
        summary[name] = {"pass": bool(ok), "detail": text}
        print("(%s) %s: %s" % (name, "PASS" if ok else "FAIL", text), flush=True)

    # (a) two processes
    s1 = run_child(args, ["--child", "schedule"], {"PYTHONHASHSEED": "1"})
    s2 = run_child(args, ["--child", "schedule"], {"PYTHONHASHSEED": "2"})
    if s1 is None or s2 is None:
        return 2
    sched = {}
    for e, t, first, seed in s1["pairs"]:
        sched.setdefault(e, {})[t] = (first, seed)
    report("a", s1 == s2, "%d episodes x %d steps, %d requests, %d probed steps; process 1 digest %s, process 2 %s"
           % (len(eps), T_MAX, s1["requests"], len(s1["pairs"]), digest(s1), digest(s2)))

    # (b) different episodes, different schedules
    steps = [tuple(sorted(sched.get(n, {}))) for n in range(len(eps))]
    counts = [len(s) for s in steps]
    overlap = max(len(set(steps[i]) & set(steps[j])) / float(max(1, len(set(steps[i]) | set(steps[j]))))
                  for i in range(len(eps)) for j in range(i))
    report("b", len(set(steps)) == len(eps) and min(counts) > 0,
           "%d distinct schedules of %d; probed steps per episode %d-%d (of %d); largest overlap of two episodes' "
           "probed steps %.3f (Jaccard)" % (len(set(steps)), len(eps), min(counts), max(counts), T_MAX, overlap))

    # (c) the realised rate at the reference's lengths
    rec = load(os.path.join(args.tests, "hidden", "episode_selection.json"))
    made = [d for d in rec["draws"] if d["status"] == "made"][-1]
    by_name = {"%s#%d" % (c["unit"], c["index"]): c for c in rec["candidates"] if c["draw"] == made["draw"]}
    kept = [by_name[name] for name in made["graded_order"]]
    same_set = [(c["unit"], int(c["seed"])) for c in kept] == eps
    lengths = [int(c["wire"][0]["steps"]) for c in kept]
    probed = sum(sum(1 for t in sched.get(n, {}) if t < lengths[n]) for n in range(len(eps)))
    total = sum(lengths)
    sd = math.sqrt(rate * (1 - rate) / total)
    full = len(s1["pairs"]) / float(len(eps) * T_MAX)
    report("c", same_set and abs(probed / float(total) - rate) <= 3 * sd,
           "at the reference's lengths (%d steps over %d episodes) %d probed steps = %.3f %% (probe_rate "
           "%.1f %%, 3 sd = %.3f %%); over the full %d steps per episode %.3f %%; record matches the set: %s"
           % (total, len(eps), probed, 100.0 * probed / total, 100 * rate, 300 * sd, T_MAX, 100 * full, same_set))

    # (d) the real loop with three stub controllers
    runs = {}
    for stub in sorted(STOPS):
        runs[stub] = run_child(args, ["--child", "loop", "--stub", stub])
        if runs[stub] is None:
            return 2
    seen = {}
    for stub, r in runs.items():
        seen[stub] = pairs_of([(q["e"], q["t"], q["seed"]) for q in r["log"]])
    match_schedule, met = True, {}
    for stub, r in runs.items():
        last = {}
        for q in r["log"]:
            last[q["e"]] = max(last.get(q["e"], -1), q["t"])
        want = [(e, t, f, s) for e in range(LOOP_EPISODES) for t, (f, s) in sorted(sched.get(e, {}).items())
                if t <= last.get(e, -1)]
        match_schedule = match_schedule and [tuple(p) for p in seen[stub]] == want
        met[stub] = len(want)
    # the steps every stub ran, from the requests themselves (the environment's own step count starts after its
    # settling steps, so an episode's `steps` is not the last `t` asked)
    common = [min(max([q["t"] for q in runs[s]["log"] if q["e"] == e] or [0]) for s in runs)
              for e in range(LOOP_EPISODES)]
    within = {s: [p for p in seen[s] if p[1] < common[p[0]]] for s in runs}
    same_probes = within["mover"] == within["reader"] == within["still"]

    def requests(stub, e):
        return [q["sha"] for q in runs[stub]["log"] if q["e"] == e and q["t"] < common[e]]
    bytes_same = all(requests("mover", e) == requests("reader", e) for e in range(LOOP_EPISODES))
    n_req = sum(len(requests("mover", e)) for e in range(LOOP_EPISODES))

    def offsets(stub):
        out, log = {}, runs[stub]["log"]
        for k in range(len(log) - 1):
            a, b = log[k], log[k + 1]
            if (a["e"], a["t"]) == (b["e"], b["t"]) and a["t"] < common[a["e"]]:
                cp, real = (a, b) if a["seed"] is not None else (b, a)
                out[(a["e"], a["t"])] = ([x - y for x, y in zip(cp["q"], real["q"])],
                                         [x - y for x, y in zip(cp["wrench"], real["wrench"])])
        return out
    om, ost = offsets("mover"), offsets("still")
    dq = max([abs(x - y) for k in om for x, y in zip(om[k][0], ost[k][0])] or [0.0])
    dw = max([abs(x - y) for k in om for x, y in zip(om[k][1][:3], ost[k][1][:3])] or [0.0])
    torque_same = all(abs(x) == 0.0 for k in om for x in om[k][1][3:] + ost[k][1][3:])
    bounds = all(abs(x) <= 0.001 + Q_TOL for k in om for x in om[k][0]) and \
        all(abs(x) <= 0.2 + W_TOL for k in om for x in om[k][1][:3])
    same_offsets = set(om) == set(ost) and dq <= Q_TOL and dw <= W_TOL and torque_same and bounds
    lengths_d = {s: [r["steps"] for r in runs[s]["records"]] for s in sorted(runs)}
    report("d", match_schedule and same_probes and bytes_same and same_offsets and len(within["mover"]) > 0,
           "episode steps per stub %s; every pair met is the schedule's (pairs met: %s): %s; in the %s steps all "
           "ran, the same %d probes (steps, order, perturbation seed): %s; mover and reader: %d requests "
           "byte-identical, copies included: %s; still vs mover: q offsets equal within %.1e (max %.1e), force "
           "offsets within %.1e (max %.1e), torques unperturbed, offsets within 0.001 rad / 0.2 N: %s"
           % (json.dumps(lengths_d), json.dumps(met), match_schedule, common, len(within["mover"]), same_probes,
              n_req, bytes_same, Q_TOL, dq, W_TOL, dw, same_offsets))

    # (e) the lock
    lk = {s: runs[s]["lock"] for s in runs}
    still_pairs = [(a, b) for a, b in zip(runs["still"]["log"], runs["still"]["log"][1:])
                   if (a["e"], a["t"]) == (b["e"], b["t"])]
    replied = [(a, b) for a, b in still_pairs if a["t"] < STOPS["still"][a["e"]]]   # not the malformed last step
    still_differ = sum(1 for a, b in replied if max(abs(x - y) for x, y in zip(a["action"], b["action"])) > 1e-4)
    ok_e = (lk["mover"]["total"] > 0 and lk["mover"]["passed"] == 0 and not lk["mover"]["ok"]
            and lk["reader"]["total"] > 0 and lk["reader"]["passed"] == lk["reader"]["total"] and lk["reader"]["ok"]
            and lk["still"]["total"] == 0 and lk["still"]["idle_skipped"] == len(still_pairs) > 0
            and still_differ == len(replied) > 0 and lk["still"]["ok"])
    report("e", ok_e, "mover %d of %d pairs differ, lock ok %s; reader %d of %d, ok %s; still: %d pairs, replies "
           "differ in %d of the %d answered with an action, counted %d, not counted %d, ok %s"
           % (lk["mover"]["passed"], lk["mover"]["total"], lk["mover"]["ok"], lk["reader"]["passed"],
              lk["reader"]["total"], lk["reader"]["ok"], len(still_pairs), still_differ, len(replied),
              lk["still"]["total"], lk["still"]["idle_skipped"], lk["still"]["ok"]))
    print("PASS" if ok_all else "FAIL")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(summary, result="PASS" if ok_all else "FAIL"), f, indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
