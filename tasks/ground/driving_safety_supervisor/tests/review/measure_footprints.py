"""measure_footprints - the collision footprint of every blueprint the stagings spawn (a maintainer's tool).

Runs inside the task image under /opt/carla/venv/bin/python, as a non-root account, with its own CARLA server. For
each blueprint it spawns one actor (physics off, yaw along the lane) on an open stretch of road, then sweeps it with
horizontal rays (`World.cast_ray`, which traces the collision geometry): from each of its four sides, on a grid of
positions along that side (every 3 cm, over the span a coarse sweep found it in) and heights (0.06 m to 1.62 m above
the road, every 6 cm: up to the car's roof), a ray toward the actor's centre line; the first hit on the actor (not the
road) is the actor's surface there. The spot is one where the same coarse sweep with no actor hits nothing. The
footprint is the axis-aligned rectangle, in the actor's own frame (x forward, y right, from the actor's origin), that
holds every hit, with the height of the highest hit; it is compared with the bounding box CARLA reports.

    /opt/carla/venv/bin/python -B measure_footprints.py --out footprints.json [--town Town05] [--blueprints ...]

The output is the expert's `solution/footprints.json`: {"format": ..., "source": ..., "footprints": {blueprint:
{"x": [min, max], "y": [min, max], "height": h, "box": {...}}}}.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

PROPS = (
    # drive_variants: PROPS_DEBRIS, PROPS_KERB, PROPS_CARGO, the closure and its warning sign
    "static.prop.trashcan01", "static.prop.trashcan03", "static.prop.box01", "static.prop.box02",
    "static.prop.box03", "static.prop.shoppingcart", "static.prop.garbage01", "static.prop.plasticchair",
    "static.prop.haybale", "static.prop.clothcontainer",
    "static.prop.mailbox", "static.prop.trashcan02", "static.prop.bench01", "static.prop.vendingmachine",
    "static.prop.plantpot01",
    "static.prop.streetbarrier", "static.prop.warningaccident", "static.prop.trafficwarning",
    # drive_bridge's standard construction zone and the distractors' cones
    "static.prop.constructioncone",
)
EGO = "vehicle.lincoln.mkz_2020"
GROUND_LABELS = ("Roads", "RoadLines", "Sidewalks", "Ground", "Terrain")


def start_carla(root, port):
    argv = [os.path.join(root, "CarlaUE4.sh"), "-RenderOffScreen", "-nosound", "-quality-level=Epic",
            "-carla-rpc-port={}".format(port), "-carla-streaming-port=0"]
    return subprocess.Popen(argv, cwd=root, stdout=sys.stderr, stderr=sys.stderr, stdin=subprocess.DEVNULL,
                            start_new_session=True)


def open_spot(world, carla, lane_wp):
    """The centre of a straight stretch at least 30 m long: (location on the road, yaw)."""
    tf = lane_wp.transform
    return tf.location, tf.rotation.yaw


def label_name(point):
    return str(point.label).split(".")[-1]


def sweep(world, carla, centre, yaw, reach, heights, spans, step):
    """Rays from four sides toward the centre line, at `heights` and every `step` over `spans[side]` (a range of
    positions along that side); returns [(side, x_local, y_local, z_rel), ...] of every hit that is not the road."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

    def world_of(lx, ly, lz):
        return carla.Location(centre.x + lx * c - ly * s, centre.y + lx * s + ly * c, centre.z + lz)

    def local_of(loc):
        dx, dy = loc.x - centre.x, loc.y - centre.y
        return dx * c + dy * s, -dx * s + dy * c, loc.z - centre.z

    out = []
    for side in ("+y", "-y", "+x", "-x"):
        lo, hi = spans[side]
        n = int(math.ceil((hi - lo) / step))
        for i in range(n + 1):
            along = lo + i * step
            for h in heights:
                if side in ("+y", "-y"):
                    sign = 1.0 if side == "+y" else -1.0
                    a, b = world_of(along, sign * reach, h), world_of(along, 0.0, h)
                else:
                    sign = 1.0 if side == "+x" else -1.0
                    a, b = world_of(sign * reach, along, h), world_of(0.0, along, h)
                for p in world.cast_ray(a, b):
                    if label_name(p) not in GROUND_LABELS:
                        out.append((side,) + local_of(p.location))
    return out


def extreme(points, side):
    """The surface the rays from `side` reached first: per ray the hit nearest the ray's start."""
    pick = {"+y": max, "-y": min, "+x": max, "-x": min}[side]
    axis = 1 if side in ("+y", "-y") else 0
    return pick(p[axis] for p in points)


def measure(world, carla, bp_name, spot, yaw, ground_z, args):
    lib = world.get_blueprint_library()
    bp = lib.find(bp_name)
    tf = carla.Transform(carla.Location(spot.x, spot.y, ground_z + 0.05), carla.Rotation(yaw=yaw))
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        return {"error": "spawn failed"}
    try:
        actor.set_simulate_physics(False)
    except RuntimeError:
        pass
    world.tick()
    world.tick()
    atf = actor.get_transform()
    box = actor.bounding_box
    centre = carla.Location(atf.location.x, atf.location.y, ground_z)
    full = (-args.reach + 0.3, args.reach - 0.3)
    coarse = sweep(world, carla, centre, atf.rotation.yaw, args.reach, (0.15, 0.5, 1.0, 1.4),
                   {k: full for k in ("+y", "-y", "+x", "-x")}, 0.16)
    if not coarse:
        actor.destroy()
        world.tick()
        return {"blueprint": bp_name, "error": "no ray hit it"}
    xs = [p[1] for p in coarse]
    ys = [p[2] for p in coarse]
    spans = {"+y": (min(xs) - 0.2, max(xs) + 0.2), "-y": (min(xs) - 0.2, max(xs) + 0.2),
             "+x": (min(ys) - 0.2, max(ys) + 0.2), "-x": (min(ys) - 0.2, max(ys) + 0.2)}
    heights = [args.h0 + k * args.dh for k in range(int(round((args.h1 - args.h0) / args.dh)) + 1)]
    fine = sweep(world, carla, centre, atf.rotation.yaw, args.reach, heights, spans, args.step)
    bounds = {side: extreme([p[1:] for p in fine if p[0] == side], side) for side in ("+y", "-y", "+x", "-x")
              if any(p[0] == side for p in fine)}
    result = {
        "blueprint": bp_name,
        "rays_hit": len(fine),
        "box": {"extent": [round(box.extent.x, 3), round(box.extent.y, 3), round(box.extent.z, 3)],
                "location": [round(box.location.x, 3), round(box.location.y, 3), round(box.location.z, 3)]},
        "origin_above_road": round(atf.location.z - ground_z, 3),
        "x": [round(bounds.get("-x", 0.0), 3), round(bounds.get("+x", 0.0), 3)],
        "y": [round(bounds.get("-y", 0.0), 3), round(bounds.get("+y", 0.0), 3)],
        "height": round(max(p[3] for p in fine), 3) if fine else None,
    }
    actor.destroy()
    world.tick()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--town", default="Town05")
    parser.add_argument("--carla-root", default="/opt/carla/server")
    parser.add_argument("--port", type=int, default=2400)
    parser.add_argument("--blueprints", nargs="*")
    parser.add_argument("--reach", type=float, default=4.5)
    parser.add_argument("--step", type=float, default=0.03)
    parser.add_argument("--h0", type=float, default=0.06)
    parser.add_argument("--h1", type=float, default=1.62)
    parser.add_argument("--dh", type=float, default=0.06)
    parser.add_argument("--spot", type=int, default=-1, help="index of the spawn point to use (default: searched)")
    args = parser.parse_args()
    sys.path.insert(0, os.path.join(args.carla_root, "PythonAPI", "carla"))
    import carla

    server = start_carla(args.carla_root, args.port)
    try:
        deadline = time.time() + 300
        while True:
            client = carla.Client("127.0.0.1", args.port)
            client.set_timeout(10.0)
            try:
                client.get_server_version()
                break
            except RuntimeError:
                if time.time() > deadline:
                    raise
                time.sleep(2.0)
        client.set_timeout(120.0)
        world = client.load_world(args.town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        world.tick()
        cmap = world.get_map()
        # an open spot: a straight driving lane whose neighbourhood the empty sweep finds clear
        candidates = []
        for i, sp in enumerate(cmap.get_spawn_points()):
            wp = cmap.get_waypoint(sp.location)
            ahead = wp.next(15.0)
            back = wp.previous(15.0)
            if wp.is_junction or not ahead or not back:
                continue
            turn = abs((ahead[0].transform.rotation.yaw - back[0].transform.rotation.yaw + 180) % 360 - 180)
            if turn < 2.0:
                candidates.append((i, wp))
        if args.spot >= 0:
            candidates = [c for c in candidates if c[0] == args.spot] or candidates
        chosen = None
        for i, wp in candidates[:60]:
            loc, yaw = open_spot(world, carla, wp)
            full = (-args.reach + 0.3, args.reach - 0.3)
            empty = sweep(world, carla, carla.Location(loc.x, loc.y, loc.z), yaw, args.reach, (0.15, 0.5, 1.0, 1.4),
                          {k: full for k in ("+y", "-y", "+x", "-x")}, 0.16)
            if not empty:
                chosen = (i, wp, loc, yaw)
                break
        if chosen is None:
            raise RuntimeError("no open spot")
        i, wp, loc, yaw = chosen
        sys.stderr.write("[fp] spot {} at ({:.1f}, {:.1f}) yaw {:.1f}: no ray hits anything there\n".format(
            i, loc.x, loc.y, yaw))
        names = args.blueprints or list(PROPS) + [EGO]
        results = {}
        for name in names:
            t0 = time.time()
            results[name] = measure(world, carla, name, loc, yaw, loc.z, args)
            sys.stderr.write("[fp] {} {} ({:.1f}s)\n".format(name, json.dumps(results[name]), time.time() - t0))
        out = {"format": "driving_safety_supervisor footprints 1",
               "source": "tests/review/measure_footprints.py: horizontal ray sweeps of each blueprint's collision "
                         "geometry (World.cast_ray), CARLA 0.9.15 in the task image, {}, step {} m, heights "
                         "{}-{} m".format(args.town, args.step, args.h0, args.h1),
               "footprints": results}
        with open(args.out, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
            handle.write("\n")
    finally:
        try:
            os.killpg(server.pid, 9)
        except OSError:
            pass


if __name__ == "__main__":
    main()
