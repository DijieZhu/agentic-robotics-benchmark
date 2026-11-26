"""Self-test of a controller against the toy chain (toy_chain.py): the calls, the frames, the feedback, the 4 ms
exchange with replies one cycle old, the late-step rule and the supervisor's signal with a two-slot toy shift.

    python3 check_controller.py [path/to/controller.py] [--cycles N]

It checks the protocol and the timing and prints the budgets used. The toy is not the robot and holds no value of
any unit: a pass says nothing about a score.

What it applies, as the acceptance runtime does:
  - setup(unit_record, site) once, budget 60 s of CPU and 120 s of wall; it must return
    {"gripper_mode": {"left": m, "right": m}}, m "mit" or "posforce";
  - step(feedback) every cycle; a step that uses more than 4 ms of CPU is late by floor(cpu / 4 ms) whole cycles:
    its frames are not sent, for those cycles the modules receive nothing and keep their last command, and the next
    feedback carries age > 1; a reply that is an object but not a valid reply counts as a late cycle (counted once,
    as malformed); a step above 1 s of CPU or 5 s of wall, an exception or a reply that is not an object ends the
    controller.
The toy's delivery record and site are in the formats of the unit's record and of the site (made-up values); the toy
reads its modules' ranges from its record's register dumps.
CPU is measured here with the process's CPU clock around each call (in the acceptance runtime the controller runs in
a process of its own and every thread and process it starts is charged).
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import frames as FR                                    # the frame codec, beside this file
import toy_chain as TOY                                # noqa: E402

CYCLE_S = 0.004
SETUP_BUDGET = (60.0, 120.0)                           # s of CPU, s of wall
STEP_FATAL = (1.0, 5.0)                                # s of CPU, s of wall
ON_TIME_SHARE_MIN = 0.90
MODES = ("mit", "posforce")


def load(path):
    spec = importlib.util.spec_from_file_location("controller", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod.Controller()


def timed(fn, *a):
    c0, w0 = time.process_time(), time.perf_counter()
    r = fn(*a)
    return r, time.process_time() - c0, time.perf_counter() - w0


def run(ctl, cycles=None):
    """The self-test of the controller object `ctl`: a report (a dict). report['ok'] is True when setup returned a
    valid mode choice inside its budget, the controller answered to the end, and at least 90 % of the cycles were on
    time."""
    rep = {"ok": False, "problems": []}
    record = TOY.toy_record()
    site = TOY.toy_site()
    try:
        res, cpu, wall = timed(ctl.setup, record, site)
    except Exception as e:                             # noqa: BLE001
        rep["problems"].append(f"setup raised {type(e).__name__}: {e}")
        return rep
    rep["setup_cpu_s"], rep["setup_wall_s"] = cpu, wall
    if cpu > SETUP_BUDGET[0] or wall > SETUP_BUDGET[1]:
        rep["problems"].append("setup over its budget")
    gm = res.get("gripper_mode") if isinstance(res, dict) else None
    if not isinstance(gm, dict) or set(gm) != {"left", "right"} or any(gm[s] not in MODES for s in gm):
        rep["problems"].append('setup must return {"gripper_mode": {"left": m, "right": m}}, m "mit" or "posforce"')
        return rep
    toy = TOY.ToyChain(gm, FR.record_ranges(record))      # the toy's ranges, read from its record's register dumps
    sig = TOY.toy_schedule(site)
    n = TOY.toy_cycles(site) if cycles is None else int(cycles)
    last = toy.start()
    last_cycle = [-1] * TOY.N
    hold = 0
    counts = {"on_time": 0, "late": 0, "malformed": 0}
    step_cpu = []
    ages_seen = set()
    ended = None
    for k in range(n):
        frames = [None] * TOY.N
        if ended is None and hold > 0:
            hold -= 1
            counts["late"] += 1
        elif ended is None:
            tray, sc = sig(k)
            mods = [FR.module_feedback(last[i], k - last_cycle[i]) for i in range(TOY.N)]
            ages_seen.update(m["age"] for m in mods)
            fb = FR.feedback(k, tray, sc, mods)
            try:
                r, cpu, wall = timed(ctl.step, fb)
            except Exception as e:                     # noqa: BLE001
                ended = f"cycle {k}: step raised {type(e).__name__}: {e}"
                r = None
            if ended is None:
                step_cpu.append(cpu)
                if cpu > STEP_FATAL[0] or wall > STEP_FATAL[1]:
                    ended = f"cycle {k}: a step used {cpu:.3f} s of CPU, {wall:.3f} s of wall"
            if ended is None:
                late = int(cpu // CYCLE_S) if cpu > CYCLE_S else 0
                try:
                    checked, _decline = FR.parse_step_reply(r, TOY.N)
                except FR.ReplyNotObject as e:
                    ended = f"cycle {k}: {e}"
                    checked = None
                except FR.ReplyMalformed as e:
                    counts["malformed"] += 1
                    rep.setdefault("first_malformed", f"cycle {k}: {e}")
                    late = max(late, 1)
                    checked = None
                if ended is None:
                    if late >= 1:
                        if checked is not None:      # a malformed reply is counted once, as malformed
                            counts["late"] += 1
                        hold = late - 1
                    else:
                        counts["on_time"] += 1
                        frames = [f for _kind, f in checked]
        replies = toy.exchange(frames)
        for i, x in enumerate(replies):
            if x is not None:
                last[i] = x
                last_cycle[i] = k
    rep["cycles"] = n
    rep["counts"] = counts
    rep["ended"] = ended
    alive = counts["on_time"] + counts["late"] + counts["malformed"]
    rep["on_time_share"] = counts["on_time"] / alive if alive else 0.0
    if step_cpu:
        s = sorted(step_cpu)
        rep["step_cpu_ms"] = {"mean": 1e3 * sum(s) / len(s), "p50": 1e3 * s[len(s) // 2],
                              "p99": 1e3 * s[min(len(s) - 1, int(0.99 * len(s)))], "max": 1e3 * s[-1]}
    rep["ages_seen"] = sorted(ages_seen)
    if ended is not None:
        rep["problems"].append("the controller ended: " + ended)
    if rep["on_time_share"] < ON_TIME_SHARE_MIN:
        rep["problems"].append(f"{100 * rep['on_time_share']:.1f} % of the cycles on time (at least 90 % needed)")
    rep["ok"] = not rep["problems"]
    return rep


def main(argv):
    path = os.path.join(HERE, "example_controller.py")
    cycles = None
    a = list(argv)
    if "--cycles" in a:
        i = a.index("--cycles")
        cycles = int(a[i + 1])
        del a[i:i + 2]
    if a:
        path = a[0]
    rep = run(load(path), cycles)
    print(f"controller: {path}")
    if "setup_cpu_s" in rep:
        print(f"setup: {rep['setup_cpu_s']:.3f} s CPU, {rep['setup_wall_s']:.3f} s wall (budget 60 s CPU, 120 s wall)")
    if "counts" in rep:
        c = rep["counts"]
        print(f"cycles: {rep['cycles']}; on time {c['on_time']}, late {c['late']}, malformed {c['malformed']}; "
              f"on time {100 * rep['on_time_share']:.1f} %")
    if "step_cpu_ms" in rep:
        s = rep["step_cpu_ms"]
        print(f"step CPU: mean {s['mean']:.3f} ms, median {s['p50']:.3f} ms, 99th percentile {s['p99']:.3f} ms, "
              f"largest {s['max']:.3f} ms (budget 4 ms per step)")
    for p in rep["problems"]:
        print("PROBLEM: " + p)
    print("PASS (protocol and timing)" if rep["ok"] else "FAIL")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
