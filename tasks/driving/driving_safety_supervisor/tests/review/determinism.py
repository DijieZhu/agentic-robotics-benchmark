"""determinism - does a CARLA episode repeat itself? (a maintainer's tool; the grader never runs it)

Runs inside the task image under the bridge's interpreter (/opt/carla/venv/bin/python), as a non-root account (Unreal
refuses root), with the bridge imported in process from `--bridge-dir` (a folder holding drive_bridge.py and
drive_variants.py: this tests/ folder, or an older copy of it). Every listed episode runs through ONE bridge backend,
in order; a fresh process per episode is one invocation per episode.

Per episode it logs, at every simulator tick, the transform of every actor the bridge spawned (in spawn order: ego,
staging, distractors, traffic, walkers, and pieces spawned later), as repr floats, and at the end the bridge's
`truth`. Actors are compared by spawn index, never by CARLA id (ids keep growing in a server's life).

Who drives:
  * `--driver planner` (D1-D5): `[0]` at every step, the nominal planner alone;
  * `--driver expert --expert <solver.py>`: the reference expert in process, with the privileged block, asked
    twice at a fixed schedule of steps (every `--probe-every` steps from `--probe-from`), the perturbed copy first
    (`--order probe-first`) or second (`--order real-first`); only the real reply is applied;
  * `--driver expert-ab --expert <solver.py>` (D6): two instances of the expert see the same observations every
    step, one asked probe-first and one real-first at the probe schedule; the first one drives. Any difference in
    their real replies (recorded per episode as `ab_first_diff`) is the probe order leaking into the expert's
    memory, independent of the simulator's own noise.

    /opt/carla/venv/bin/python -B determinism.py --bridge-dir DIR --out FILE.jsonl.gz --specs SPECS.json
        [--driver planner|expert] [--expert solver.py] [--order probe-first|real-first] [--background 20]
        [--walkers 20] [--image-size 512]

SPECS.json is a JSON list of [name, spec]. `compare` reads two or more outputs and reports, per episode name (or per
`--pair i:j` of positions in one file), whether they are identical at every tick and, if not, the first differing tick and which
actors differ there:

    python3 -B determinism.py compare RUN1.jsonl.gz RUN2.jsonl.gz [...]
    python3 -B determinism.py compare RUN.jsonl.gz --pair 0:2      (two episodes of one process, by position)
"""

import argparse
import gzip
import json
import math
import os
import sys
import time


def _transform(actor):
    tf = actor.get_transform()
    loc, rot = tf.location, tf.rotation
    return [repr(loc.x), repr(loc.y), repr(loc.z), repr(rot.pitch), repr(rot.yaw), repr(rot.roll)]


def snapshot(backend):
    out = []
    for actor in backend.actors:
        try:
            out.append([actor.type_id] + _transform(actor))
        except RuntimeError:
            out.append([getattr(actor, "type_id", "?"), "gone"])
    return out


def run(args):
    sys.path.insert(0, os.path.abspath(args.bridge_dir))
    import drive_bridge as DB  # noqa: E402
    from drive_protocol import agent_observation, parse_action, probe_observation  # noqa: E402
    expert = twin = None
    if args.driver in ("expert", "expert-ab"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.expert)))
        import importlib.util
        spec = importlib.util.spec_from_file_location("expert_solver", args.expert)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expert = module.Expert()
        if args.driver == "expert-ab":
            twin = module.Expert()
    with open(args.specs) as handle:
        jobs = json.load(handle)
    ns = argparse.Namespace(carla_root="/opt/carla/server", port=0, image_size=args.image_size,
                            quality="Epic", background=args.background, walkers=args.walkers)
    t0 = time.time()
    backend = DB.CarlaBackend(ns)
    sys.stderr.write("[det] CARLA up in {:.1f}s\n".format(time.time() - t0))
    out = gzip.open(args.out, "wt")
    try:
        for name, spec in jobs:
            if args.driver in ("expert", "expert-ab"):
                spec = dict(spec, privileged=True)
            t1 = time.time()
            state = backend.reset(spec)
            reset_s = time.time() - t1
            ticks = [snapshot(backend)]

            def hook():
                ticks.append(snapshot(backend))
            backend.scripted.append(hook)
            if expert is not None:
                expert.reset()
            if twin is not None:
                twin.reset()
            t, replies, ab_first_diff = 0, [], None
            while not state["truth"]["done"]:
                obs = json.loads(json.dumps(agent_observation(state, t)))
                if expert is None:
                    raw = [0]
                else:
                    probed = t >= args.probe_from and (t - args.probe_from) % args.probe_every == 0
                    if probed:
                        fake = json.loads(json.dumps(probe_observation(obs, state)))
                        if args.order == "probe-first":
                            expert.act(t, fake)
                            raw = expert.act(t, obs)
                        else:
                            raw = expert.act(t, obs)
                            expert.act(t, fake)
                        if twin is not None:            # the other order, same observations
                            other = twin.act(t, obs)
                            twin.act(t, fake)
                    else:
                        raw = expert.act(t, obs)
                        if twin is not None:
                            other = twin.act(t, obs)
                    if twin is not None and ab_first_diff is None and \
                            [float(v) for v in other] != [float(v) for v in raw]:
                        ab_first_diff = t
                    raw = [float(v) for v in raw]
                    if raw[0] == 0 or not all(math.isfinite(v) for v in raw):
                        raw = [0]
                    else:
                        raw = [1, min(1.0, max(0.0, raw[1])), min(1.0, max(-1.0, raw[2])),
                               min(1.0, max(0.0, raw[3]))]
                replies.append(raw)
                mode, control = parse_action(raw)
                state = backend.step(mode, control)
                t += 1
                if t > 3000:
                    raise RuntimeError("episode did not terminate")
            ticks.append(snapshot(backend))
            record = {"name": name, "spec": spec, "truth": state["truth"], "steps": t, "ab_first_diff": ab_first_diff,
                      "ego_index": backend.actors.index(backend.ego),
                      "reset_s": round(reset_s, 2), "wall_s": round(time.time() - t1, 1),
                      "ticks": ticks, "replies": replies}
            out.write(json.dumps(record) + "\n")
            out.flush()
            sys.stderr.write("[det] {} steps={} termination={} completion={} layout={} wall={:.1f}s\n".format(
                name, t, state["truth"]["termination"], state["truth"]["route_completion"],
                state["truth"].get("layout", "-")[:12], time.time() - t1))
    finally:
        out.close()
        backend.close()


def load(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def diff(a, b):
    """Whether two episode records are identical at every tick; else (first tick, differing actor indexes, the
    ego's largest position gap over the run, both truths' termination/steps/pace)."""
    ta, tb = a["ticks"], b["ticks"]
    first, actors = None, []
    for k in range(min(len(ta), len(tb))):
        if ta[k] != tb[k]:
            first = k
            n = max(len(ta[k]), len(tb[k]))
            actors = [(i, (ta[k][i] if i < len(ta[k]) else tb[k][i])[0]) for i in range(n)
                      if i >= len(ta[k]) or i >= len(tb[k]) or ta[k][i] != tb[k][i]]
            break
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    ego = a.get("ego_index")
    if ego is None:      # (older records: the first car of the ego's blueprint after the staging)
        ego = next((i for i, e in enumerate(ta[0]) if e[0] == "vehicle.lincoln.mkz_2020"), None)
    gap = 0.0
    if ego is not None:
        for k in range(min(len(ta), len(tb))):
            pa, pb = ta[k][ego], tb[k][ego]
            if len(pa) > 2 and len(pb) > 2:
                gap = max(gap, math.hypot(float(pa[1]) - float(pb[1]), float(pa[2]) - float(pb[2])))
    keys = ("termination", "route_completion", "time_s", "pace", "harsh_brakes", "collisions", "layout",
            "hazard_collisions")
    same_truth = all(a["truth"].get(k) == b["truth"].get(k) for k in keys)
    return {"identical": first is None and same_truth, "first_tick": first, "actors": actors[:8],
            "n_actors_differing": len(actors), "ego_max_gap_m": round(gap, 6), "same_truth": same_truth,
            "a": {k: a["truth"].get(k) for k in keys if k != "collisions"} | {"steps": a["steps"]},
            "b": {k: b["truth"].get(k) for k in keys if k != "collisions"} | {"steps": b["steps"]},
            "replies_identical": a.get("replies") == b.get("replies")}


def compare(args):
    runs = [load(p) for p in args.paths]
    if args.pair:
        # records of one file by position: "0:2" compares its first and third episode
        i, j = (int(x) for x in args.pair.split(":"))
        for path, records in zip(args.paths, runs):
            if max(i, j) < len(records):
                print(json.dumps({"file": path, "pair": args.pair, "names": [records[i]["name"], records[j]["name"]],
                                  **diff(records[i], records[j])}))
        return
    base = runs[0]
    for other_path, other in zip(args.paths[1:], runs[1:]):
        for a, b in zip(base, other):
            print(json.dumps({"a": args.paths[0], "b": other_path, "name": a["name"], **diff(a, b)}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = argparse.ArgumentParser()
        parser.add_argument("cmd")
        parser.add_argument("paths", nargs="+")
        parser.add_argument("--pair")
        compare(parser.parse_args())
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--driver", choices=("planner", "expert", "expert-ab"), default="planner")
    parser.add_argument("--expert")
    parser.add_argument("--order", choices=("probe-first", "real-first"), default="probe-first")
    parser.add_argument("--probe-from", type=int, default=20)
    parser.add_argument("--probe-every", type=int, default=13)
    parser.add_argument("--background", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
