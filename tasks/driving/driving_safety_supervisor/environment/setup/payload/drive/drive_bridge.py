"""drive_bridge - the simulator side of sim_client's pipe, for supervised driving.

Runs as a child of the grader (or of the practice grader / the agent's own tools) under the
CARLA client interpreter, `/opt/carla/venv/bin/python` (Python 3.10). JSON lines:

    <- {"ready": true, "backend": ...}
    {"cmd": "reset", "spec": {"town", "kind", "seed"}}                 -> state
    {"cmd": "step", "mode": 0|1, "control": [throttle, steer, brake]}  -> state
    {"cmd": "close"}                                                    -> {"ok": true}

`state` holds what the supervisor may see (the front camera frame, ego speed, navigation,
the nominal control), a pre-rendered probe frame, and a privileged `truth` block (route
completion, infractions, pace, termination, a digest of the episode's layout) that the
grading side keeps to itself (drive_protocol.agent_observation filters it out).

A spec with `"privileged": true` adds a `privileged` block to every state: the true position,
velocity and size of every road user and prop near the car, and the route's lane geometry
and traffic lights ahead, all in the car's frame. The bridge adds it only when the spec asks
for it, as a practice aid; graded runs do not give it to your supervisor. The probe then also
reports the road user it pasted into the probe frame.

Repeatability: the world is reloaded for every episode, in synchronous mode with fixed
physics substeps, so an episode's layout (route, staging, traffic and their draws) follows
from its spec alone; `truth["layout"]` is a digest of it. Collisions and camera frames are
taken tick by tick on the main thread, one camera frame per simulator tick. If the CARLA
server dies while an episode is being built, the reply says so (`server_exited`) and the
bridge exits; sim_client then starts a new bridge, which builds the episode again from its
spec.

A spec may name a `variant`: a different staging of its kind, defined in `drive_variants`
next to this file. The practice copy ships without that module and stages every kind in
its standard form.

Backends:

* `carla` - CARLA 0.9.15 in synchronous mode at 20 Hz, one control period = 0.1 s. Routes
  and scenarios are generated from the spec's seed; the nominal planner is the stock
  `agents.navigation.behavior_agent.BehaviorAgent` ('normal' profile) following that route.
* `mock` - a stdlib kinematic world (a straight two-lane road) with the same state
  machine and kinds (standard forms only), for testing the grader, the wire and a
  supervisor's control flow on a machine without a GPU or CARLA.

Shared byte-identically by the grading stage and the agent payload.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib

CAM_X, CAM_Z = 1.5, 1.6   # front camera: 1.5 m ahead of the centre, 1.6 m up, level, 90 deg
DT = 0.05                 # simulator step
SUBSTEP_DT, MAX_SUBSTEPS = 0.01, 10   # physics substeps inside one simulator step
TICKS_PER_STEP = 2        # control period 0.1 s
HARSH_DECEL = 7.0         # m/s^2
BLOCKED_S = 25.0
MAX_EXCUSED_S = 90.0      # waiting excused per episode (held at a red or yellow light)
OFF_ROUTE_M = 15.0
TIME_PER_M, TIME_EXTRA_S = 1 / 5.0, 15.0   # time limit: route length / 5 m/s + 15 s
REF_FRACTION = 0.85       # reference pace: this fraction of the speed limit ...
REF_CAP_KMH = 50.0        # ... counting limits above the nominal planner's own cap as the cap
NAV_SPACING_M = 30.0      # navigation targets: about every 30 m and where the command changes
VEHICLE_RADIUS_M, NEAR_RADIUS_M = 250.0, 80.0   # privileged block: vehicles, walkers and props
FRAME_TIMEOUT_S = 30.0    # a camera frame not arriving this long after its tick is a bridge error
LOAD_TIMEOUT_S = 120.0    # a town (re)load (Town10HD's takes 9-12 s)
HERE = os.path.dirname(os.path.abspath(__file__))


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(message):
    sys.stderr.write("[drive_bridge] {}\n".format(message))
    sys.stderr.flush()


class ServerExited(RuntimeError):
    """The CARLA server died: this bridge cannot go on (sim_client replaces it)."""


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def layout_digest(town, route, actors, rng):
    """sha256 of what an episode's reset laid out: the town, the route's lane keys, the actors
    spawned (blueprint ids in spawn order) and the scenario generator's state after every draw
    of the reset. The same spec gives the same digest on the same build."""
    payload = {"town": town, "route": route, "actors": actors, "rng": repr(rng.getstate())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode("ascii")).hexdigest()


class Tracker(object):
    """Route progress, pace and infractions, identical for both backends."""

    def __init__(self, route_xy, kind, variant=None, hazard_ids=()):
        self.route = route_xy
        self.hazard_ids = set(hazard_ids)    # the staged scenario's own actors
        self.hazard_hits = 0
        self.length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route_xy, route_xy[1:]))
        self.index = 0
        self.time = 0.0
        self.time_limit = self.length * TIME_PER_M + TIME_EXTRA_S
        self.kind = kind
        self.variant = variant
        self.collisions = {"walker": 0, "vehicle": 0, "static": 0}
        self.red_lights = 0
        self.harsh = 0
        self._harsh_active = False
        self._still = 0.0
        self.excused = 0.0       # time held at a red or yellow light: no blame
        self.reference_m = 0.0   # distance at the reference pace over the unexcused time
        self.travelled = 0.0
        self._last_xy = None
        self._last_hit = {}
        self.termination = ""
        self.layout = ""         # layout_digest of the reset that built this episode
        self.staging = {}        # what the staging drew (grading side only, like all of truth)

    def collision(self, category, other_id, when=None):
        """One contact with `other_id`; contacts with the same actor within 1 s (episode time,
        `when` if given) count once."""
        now = self.time if when is None else when
        if now - self._last_hit.get(other_id, -10.0) < 1.0:
            self._last_hit[other_id] = now
            return
        self._last_hit[other_id] = now
        self.collisions[category] += 1
        if other_id in self.hazard_ids:
            self.hazard_hits += 1
        if category == "walker":
            self.termination = "pedestrian_collision"

    def update(self, x, y, speed, prev_speed, dt, limit_kmh, excused=False):
        self.time += dt
        if self._last_xy is not None:
            self.travelled += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        best, best_d = self.index, float("inf")
        for i in range(self.index, min(len(self.route), self.index + 25)):
            d = math.hypot(self.route[i][0] - x, self.route[i][1] - y)
            if d < best_d:
                best, best_d = i, d
        if best_d < 4.0:
            self.index = best
        decel = (prev_speed - speed) / dt
        if decel > HARSH_DECEL and not self._harsh_active:
            self.harsh += 1
        self._harsh_active = decel > HARSH_DECEL
        if excused and speed < 0.1 and self.excused < MAX_EXCUSED_S:
            self.excused += dt       # paused: counts towards neither "blocked", timeout nor pace
        else:
            self._still = self._still + dt if speed < 0.1 else 0.0
            self.reference_m += REF_FRACTION * min(limit_kmh, REF_CAP_KMH) / 3.6 * dt
        nearest = min(math.hypot(px - x, py - y) for px, py in self.route[max(0, self.index - 5):
                                                                         self.index + 30])
        if self.termination:
            pass
        elif self.completion >= 0.97:
            self.termination = "completed"
        elif self._still >= BLOCKED_S:
            self.termination = "blocked"
        elif nearest > OFF_ROUTE_M:
            self.termination = "off_route"
        elif self.time - self.excused >= self.time_limit:
            self.termination = "timeout"

    @property
    def completion(self):
        return self.index / float(max(1, len(self.route) - 1))

    @property
    def pace(self):
        if self.reference_m <= 0.0:
            return 0.0
        return min(1.5, self.completion * self.length / self.reference_m)

    def truth(self):
        return {"done": bool(self.termination), "termination": self.termination,
                "kind": self.kind, "variant": self.variant,
                "route_completion": round(self.completion, 4),
                "route_length_m": round(self.length, 1),
                "collisions": dict(self.collisions), "red_lights": self.red_lights,
                "hazard_collisions": self.hazard_hits,
                "harsh_brakes": self.harsh, "time_s": round(self.time, 2),
                "excused_s": round(self.excused, 2), "pace": round(self.pace, 4),
                "mean_speed": round(self.travelled / max(self.time, 1e-6), 2),
                "layout": self.layout, "staging": dict(self.staging)}


def junction_commands(route_xy, in_junction):
    """Per route point: `lanefollow`, or for the points inside a junction what the route
    does through it (`left` / `right` / `straight`, from its heading change)."""
    n = len(route_xy)
    commands = ["lanefollow"] * n
    i = 0
    while i < n:
        if not in_junction[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and in_junction[j + 1]:
            j += 1
        a0, a1 = route_xy[max(0, i - 3)], route_xy[max(0, i - 1)]
        b0, b1 = route_xy[min(n - 1, j + 1)], route_xy[min(n - 1, j + 3)]
        turn = math.degrees(_wrap(math.atan2(b1[1] - b0[1], b1[0] - b0[0]) -
                                  math.atan2(a1[1] - a0[1], a1[0] - a0[0])))
        command = "left" if turn < -30 else "right" if turn > 30 else "straight"
        for k in range(i, j + 1):
            commands[k] = command
        i = j + 1
    return commands


def navigation(route_xy, options, index, x, y, yaw):
    """The next sparse navigation target ahead of route index `index`, in the car's frame
    ([forward, right] metres), and its command. Targets sit about every NAV_SPACING_M and
    wherever the command changes (a junction's entry and exit), like a leaderboard route."""
    marks, run = [0], 0.0
    for i in range(1, len(route_xy)):
        run += math.hypot(route_xy[i][0] - route_xy[i - 1][0], route_xy[i][1] - route_xy[i - 1][1])
        if options[i] != options[i - 1] or run >= NAV_SPACING_M:
            marks.append(i)
            run = 0.0
    marks.append(len(route_xy) - 1)
    target = next((m for m in marks if m > index + 2), len(route_xy) - 1)
    tx, ty = route_xy[target]
    dx, dy = tx - x, ty - y
    c, s = math.cos(yaw), math.sin(yaw)
    return {"target": [c * dx + s * dy, -s * dx + c * dy], "command": options[target]}


# =========================================================================== PNG (mock)
def _png(width, height, rows, bit16=False):
    raw = bytearray()
    for row in rows:
        raw.append(0)
        if bit16:
            for value in row:
                raw += struct.pack(">H", max(0, min(65535, int(value))))
        else:
            for pixel in row:
                raw += bytes(pixel)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 16 if bit16 else 8, 0 if bit16 else 2,
                         0, 0, 0)
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                            chunk(b"IDAT", zlib.compress(bytes(raw), 6)) +
                            chunk(b"IEND", b"")).decode("ascii")


# =========================================================================== mock backend
class MockBackend(object):
    """A straight two-lane road along +x (ego lane centred on y = 0, the opposite lane on
    y = -3.5), 200 m long. Obstacles are boxes; the pedestrian a 0.6 m box. The nominal
    planner has the same blind spots as the CARLA one: it ignores static props and
    crossing traffic, reacts to pedestrians only within 3 m, and waits forever behind a
    stopped vehicle."""

    W, H = 64, 32
    LANE = 3.5

    def __init__(self, args):
        self.args = args

    def reset(self, spec):
        rng = random.Random(spec["seed"])
        self.kind = spec["kind"]
        self.probe_rng = random.Random(spec["seed"] ^ 0x5EED)
        self.x, self.y, self.yaw, self.v = 0.0, 0.0, 0.0, 0.0
        self.applied = [0.0, 0.0, 0.0]
        self.boxes = []        # dicts: x, y, hx, hy, vx, vy, cat, id, trigger
        self.junction = None
        h = rng.uniform(55, 80)
        if self.kind == "construction_zone":
            for dy in (-1.2, 0.0, 1.2):
                self._box(h, dy, 0.3, 0.3, "static")
            self._box(h + 2.0, 0.0, 0.3, 1.5, "static")
        elif self.kind == "broken_down_vehicle":
            self._box(h, 0.0, 2.3, 1.0, "vehicle")
        elif self.kind == "occluded_pedestrian":
            self._box(h, 3.2, 2.5, 1.0, "vehicle", hazard=False)      # parked van at the kerb
            self._box(h + 3.5, 3.4, 0.3, 0.3, "walker", trigger=("ped", rng.uniform(2.5, 3.5)))
        elif self.kind in ("red_light_runner", "oncoming_left_turn"):
            side = 1 if self.kind == "red_light_runner" else -1
            self._box(h, 30.0 * side, 2.3, 1.0, "vehicle", trigger=("cross", 9.0 * side))
            self.junction = h
        self.route = [(float(i), 0.0) for i in range(0, 201)]
        self.options = ["lanefollow"] * len(self.route)
        if self.kind == "oncoming_left_turn":       # the mock road stays straight
            self.options[int(h):int(h) + 12] = ["left"] * 12
        elif self.kind == "red_light_runner":
            self.options[int(h):int(h) + 12] = ["straight"] * 12
        self.tracker = Tracker(self.route, self.kind, spec.get("variant"),
                               [b["id"] for b in self.boxes if b["hazard"]])
        self.tracker.layout = layout_digest(
            spec.get("town", ""), [len(self.route), self.options.count("lanefollow")],
            [[b["cat"], b["x"], b["y"], b["hx"], b["hy"], b["trigger"]] for b in self.boxes], rng)
        self._still_nominal = 0.0
        return self._state()

    def _box(self, x, y, hx, hy, cat, trigger=None, hazard=True):
        self.boxes.append({"x": x, "y": y, "hx": hx, "hy": hy, "vx": 0.0, "vy": 0.0,
                           "cat": cat, "id": len(self.boxes) + 1, "trigger": trigger,
                           "active": False, "hazard": hazard})

    # -- nominal planner (with its blind spots) -----------------------------
    def _nominal(self):
        target = 8.0
        for b in self.boxes:
            ahead = b["x"] - b["hx"] - (self.x + 2.3)
            in_lane = abs(b["y"] - self.y) < b["hy"] + 1.2
            if b["cat"] == "vehicle" and in_lane and 0 < ahead < 25 and b["vx"] == 0 and b["vy"] == 0:
                target = min(target, max(0.0, (ahead - 6.0) * 0.8))
            if b["cat"] == "walker" and in_lane and 0 < ahead < 3.0:
                return [0.0, self._steer_to(0.0), 1.0]
        err = target - self.v
        throttle = max(0.0, min(1.0, 0.4 * err)) if err > 0 else 0.0
        brake = max(0.0, min(1.0, -0.3 * err)) if err < 0 else 0.0
        if target == 0.0 and self.v < 0.5:
            brake = 1.0
        return [throttle, self._steer_to(0.0), brake]

    def _check_hit(self, b):
        if abs(b["x"] - self.x) < b["hx"] + 2.3 and abs(b["y"] - self.y) < b["hy"] + 1.0:
            self.tracker.collision(b["cat"], b["id"])
            if b["cat"] != "walker":
                self.v = 0.0

    def _steer_to(self, lane_y):
        lookahead = max(4.0, self.v * 1.0)
        heading = math.atan2(lane_y - self.y, lookahead)
        return max(-1.0, min(1.0, 2.0 * _wrap(heading - self.yaw)))

    # -- stepping ------------------------------------------------------------
    def step(self, mode, control):
        nominal = self._nominal()
        throttle, steer, brake = nominal if mode == 0 else control
        self.applied = [throttle, steer, brake]
        for _ in range(TICKS_PER_STEP):
            prev = self.v
            accel = 3.0 * throttle - 8.0 * brake - 0.3
            self.v = max(0.0, self.v + accel * DT)
            self.yaw += self.v / 2.9 * math.tan(1.22 * steer) * DT   # 70 degree max wheel angle
            self.x += self.v * math.cos(self.yaw) * DT
            self.y += self.v * math.sin(self.yaw) * DT
            for b in self.boxes:
                trig = b["trigger"]
                if trig and not b["active"]:
                    if trig[0] == "ped" and b["x"] - self.x < self.v * 1.8 + 5:
                        b["active"], b["vy"] = True, -trig[1]
                    if trig[0] == "cross" and self.junction - self.x < self.v * 3.3 + 1:
                        b["active"], b["vy"] = True, -trig[1]
                if trig and trig[0] == "ped" and b["active"]:
                    b.setdefault("pause", 3.0)
                    if b["y"] <= 0.0 and b["pause"] > 0.0:
                        b["pause"] -= DT
                        continue_moving = False
                    else:
                        continue_moving = True
                    if not continue_moving:
                        self._check_hit(b)
                        continue
                b["x"] += b["vx"] * DT
                b["y"] += b["vy"] * DT
                self._check_hit(b)
            self.tracker.update(self.x, self.y, self.v, prev, DT, limit_kmh=30.0)
            if self.tracker.termination:
                break
        return self._state()

    # -- rendering -------------------------------------------------------------
    def _render(self):
        W, H = self.W, self.H
        f = W / 2.0
        rgb = [[(140, 180, 230) if r < H // 2 else (90, 90, 90) for _ in range(W)] for r in range(H)]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        colours = {"static": (240, 120, 20), "vehicle": (30, 60, 200), "walker": (200, 30, 160)}
        cx = self.x + CAM_X * math.cos(self.yaw)
        cy = self.y + CAM_X * math.sin(self.yaw)
        for b in sorted(self.boxes, key=lambda b: -math.hypot(b["x"] - cx, b["y"] - cy)):
            dx, dy = b["x"] - cx, b["y"] - cy
            fwd, right = c * dx + s * dy, -s * dx + c * dy
            if fwd < 1.0 or fwd > 60.0:
                continue
            # visible half width: the box's extent across the line of sight
            sight = math.atan2(right, fwd) + self.yaw
            half_w = abs(b["hy"] * math.cos(sight)) + abs(b["hx"] * math.sin(sight))
            u0, u1 = int(f + f * (right - half_w) / fwd), int(f + f * (right + half_w) / fwd)
            v0, v1 = int(H / 2 + f * (-0.2) / fwd), int(H / 2 + f * CAM_Z / fwd)
            for r in range(max(0, v0), min(H, v1 + 1)):
                for col in range(max(0, u0), min(W, u1 + 1)):
                    rgb[r][col] = colours[b["cat"]]
        return rgb

    def _state(self):
        rgb = self._render()
        W, H, f = self.W, self.H, self.W / 2.0
        probe = [row[:] for row in rgb]
        dist = self.probe_rng.uniform(8.0, 25.0)       # a pedestrian-sized blob on the road
        u = int(f + f * self.probe_rng.uniform(-1.0, 1.0) / dist)
        v1, v0 = int(H / 2 + f * CAM_Z / dist), int(H / 2 + f * (CAM_Z - 1.7) / dist)
        half = max(1, int(f * 0.3 / dist))
        for r in range(max(0, v0), min(H, v1 + 1)):
            for col in range(max(0, u - half), min(W, u + half + 1)):
                probe[r][col] = (200, 30, 160)
        nav = navigation(self.route, self.options, self.tracker.index, self.x, self.y, self.yaw)
        return {
            "frame": {"image": _png(W, H, rgb)},
            "probe": {"image": _png(W, H, probe)},
            "ego": {"speed": round(self.v, 3), "applied": [round(v, 3) for v in self.applied]},
            "navigation": nav,
            "nominal": {"control": [round(v, 3) for v in self._nominal()]},
            "truth": self.tracker.truth(),
        }

    def close(self):
        pass


# =========================================================================== CARLA backend
class CarlaBackend(object):
    def __init__(self, args):
        self.args = args
        root = args.carla_root
        for path in (os.path.join(root, "PythonAPI", "carla"),):
            if path not in sys.path:
                sys.path.insert(0, path)
        import carla  # noqa: F401
        import numpy  # noqa: F401
        import cv2  # noqa: F401
        self.carla, self.np, self.cv2 = carla, numpy, cv2
        self.port = args.port or _free_port()
        argv = [os.path.join(root, "CarlaUE4.sh"), "-RenderOffScreen", "-nosound",
                "-quality-level={}".format(args.quality), "-carla-rpc-port={}".format(self.port),
                "-carla-streaming-port=0"]
        self.server = subprocess.Popen(argv, cwd=root, stdout=sys.stderr, stderr=sys.stderr,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            self._connect(carla)
        except Exception:
            self._kill_server()       # never leave a server running behind a bridge that failed
            raise
        threading.Thread(target=self._watch_server, daemon=True).start()
        self.world = None
        self.town = None
        self.map = None
        self.actors = []
        self.sensors = []
        _log("CARLA {} up on port {}".format(self.client.get_server_version(), self.port))

    def _connect(self, carla):
        deadline = time.time() + 300
        while True:
            if self.server.poll() is not None:
                raise RuntimeError("CARLA exited with code {}".format(self.server.returncode))
            # A client made before the server listens never recovers: make one per attempt.
            self.client = carla.Client("127.0.0.1", self.port)
            self.client.set_timeout(10.0)
            try:
                self.client.get_server_version()
                break
            except RuntimeError:
                if time.time() > deadline:
                    raise RuntimeError("CARLA did not answer within 300 s")
                time.sleep(2.0)
        self.client.set_timeout(60.0)
        for attempt in range(3):      # a server still warming up can miss the first request
            try:
                self.tm = self.client.get_trafficmanager(self.port + 50)
                return
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(5.0)

    def _watch_server(self):
        """Once the server has died, every later RPC fails within a second instead of waiting
        out its time-out (a staging retries its spawns, each one an RPC)."""
        self.server.wait()
        try:
            self.client.set_timeout(1.0)
        except Exception:  # noqa: BLE001
            pass

    def _kill_server(self):
        if self.server.poll() is None:
            try:
                os.killpg(self.server.pid, 9)
            except OSError:
                pass

    # -- world setup ----------------------------------------------------------
    def _sync(self, world):
        """Synchronous mode at DT with fixed physics substeps, rendering on."""
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.substepping = True
        settings.max_substep_delta_time = SUBSTEP_DT
        settings.max_substeps = MAX_SUBSTEPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)

    def _load(self, town):
        """A fresh world for every episode: the town loaded (or the current one reloaded) with
        synchronous mode already on, so nothing runs between episodes and no actor, physics
        state or traffic light phase carries over from the previous one."""
        carla = self.carla
        self._sync(self.world if self.world is not None else self.client.get_world())
        # a (re)load of a large town can take minutes on a busy host: wait for it
        self.client.set_timeout(LOAD_TIMEOUT_S)
        try:
            if self.town != town:
                self.world = self.client.load_world(town, reset_settings=False)
                self.town = town
                self.map = None
            else:
                self.world = self.client.reload_world(reset_settings=False)
        finally:
            self.client.set_timeout(60.0)
        self._sync(self.world)
        if self.map is None:
            self.map = self.world.get_map()     # the same town's road network: kept across reloads
        self.world.reset_all_traffic_lights()
        self.world.tick()
        return carla

    def _cleanup(self):
        for sensor in self.sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        # Release the Traffic Manager's vehicles before they are destroyed (the reload that
        # follows destroys whatever is left).
        for actor in self.actors:
            if self.server.poll() is not None:      # not one RPC time-out per car
                raise RuntimeError("CARLA exited with code {}".format(self.server.returncode))
            if actor.type_id.startswith("vehicle."):
                try:
                    actor.set_autopilot(False, self.tm.get_port())
                except RuntimeError:
                    pass
        ids = [a.id for a in self.sensors + self.actors]
        if ids:
            self.client.apply_batch_sync([self.carla.command.DestroyActor(i) for i in ids], False)
        self.sensors, self.actors = [], []
        self.walker_triggers, self.scripted = [], []

    def reset(self, spec):
        # CARLA 0.9.15 can die (signal 11) in a reset after many episodes in one server life
        # (seen at the 31st). This client's threads then belong to a dead server, so the
        # bridge does not carry on: it reports the exit, and sim_client starts a new bridge.
        if self.server.poll() is None:
            try:
                return self._reset(spec)
            except Exception as exc:  # noqa: BLE001
                try:
                    self.server.wait(timeout=10.0)   # a crashed server's wrapper exits at once
                except subprocess.TimeoutExpired:
                    raise exc                        # the server is alive: a real error
        raise ServerExited("CARLA exited with code {} while the episode was built".format(
            self.server.returncode))

    def _reset(self, spec):
        from agents.navigation.behavior_agent import BehaviorAgent
        from agents.navigation.local_planner import RoadOption
        self.RoadOption = RoadOption
        if self.world is not None:
            self._cleanup()
        carla = self._load(spec["town"])
        rng = random.Random(spec["seed"])
        self.probe_rng = random.Random(spec["seed"] ^ 0x5EED)
        self.tm.set_random_device_seed(spec["seed"] & 0x7FFFFFFF)
        self.walker_triggers, self.scripted = [], []
        weathers = ["ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset", "WetCloudyNoon",
                    "SoftRainNoon"]
        weather = rng.choice(weathers)
        self.world.set_weather(getattr(carla.WeatherParameters, weather))
        builder = ScenarioBuilder(self, rng, spec["kind"], spec.get("variant"),
                                  distractors=int(spec.get("distractors", 0)))
        route, ego_tf = builder.build()
        hazard_ids = [a.id for a in builder.hazards]
        self.privileged = bool(spec.get("privileged"))
        self._seen = {}              # actor id -> (x, y, time): velocity of moved-by-script actors
        self.route_wps = [wp for wp, _ in route]
        self.probe_actor = None
        self.ego = self._spawn("vehicle.lincoln.mkz_2020", ego_tf, role="hero")
        self._attach_sensors()
        self.nominal = BehaviorAgent(self.ego, behavior="normal")
        # the planner gets the road on past the route's end, so it never runs out of plan
        beyond, last = [], route[-1][0]
        for _ in range(20):
            nxt = last.next(2.0)
            if not nxt:
                break
            last = nxt[0]
            beyond.append((last, RoadOption.LANEFOLLOW))
        self.nominal.set_global_plan(route + beyond, stop_waypoint_creation=True, clean_queue=True)
        self._background_traffic(rng, route)
        if not getattr(builder, "quiet", False):
            self._background_walkers(rng, route)
        self.route_xy = [(wp.transform.location.x, wp.transform.location.y) for wp, _ in route]
        self.options = junction_commands(self.route_xy, [wp.is_junction for wp, _ in route])
        self.tracker = Tracker(self.route_xy, spec["kind"], spec.get("variant"), hazard_ids)
        builder.staging["weather"] = weather
        self.tracker.staging = builder.staging      # the variant's hooks may add to it later
        self.tracker.layout = layout_digest(
            spec["town"], [[wp.road_id, wp.section_id, wp.lane_id, round(wp.s, 1)] for wp, _ in route],
            [a.type_id for a in self.actors], rng)
        # the traffic lights whose stop line lies on the route (listed in the privileged block)
        self.route_lights = self._route_lights() if self.privileged else []
        self.applied = [0.0, 0.0, 0.0]
        self.prev_speed = 0.0
        self.red_candidate = None
        self.red_counted = None      # the light this crossing has already counted as run
        for _ in range(10):          # settle the vehicle and fill the sensor queues
            self.ego.apply_control(carla.VehicleControl(brake=1.0))
            self._frame_time[self._tick()] = 0.0
        image = self._frames()
        self._apply_hits()
        self._nominal_control = self._nominal_step()
        return self._state(image)

    def _tick(self):
        """One simulator tick; the front camera owes one frame for every tick."""
        frame = self.world.tick()
        self._owed_frames += 1
        return frame

    def _apply_hits(self):
        """Contacts the collision sensor queued so far, in the order of the frames they happened
        in, applied here on the main thread."""
        hits = []
        while True:
            try:
                hits.append(self._hits.get_nowait())
            except Exception:  # noqa: BLE001 - queue.Empty
                break
        for frame, other_id, category in sorted(hits, key=lambda h: (h[0], h[1], h[2])):
            when = self._frame_time.get(frame)
            if when is None:
                earlier = [t for f, t in self._frame_time.items() if f <= frame]
                when = max(earlier) if earlier else 0.0
            self.tracker.collision(category, other_id, when=when)

    def _spawn(self, blueprint, transform, role=None, physics=True):
        bp = self.world.get_blueprint_library().find(blueprint)
        if role and bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role)
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            transform.location.z += 0.5
            actor = self.world.spawn_actor(bp, transform)
        if not physics:
            actor.set_simulate_physics(False)
        self.actors.append(actor)
        return actor

    def _attach_sensors(self):
        import queue
        carla = self.carla
        lib = self.world.get_blueprint_library()
        self.queues = {}
        size = self.args.image_size
        bp = lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(size))
        bp.set_attribute("image_size_y", str(size // 2))
        bp.set_attribute("fov", "90")
        tf = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z), carla.Rotation())
        sensor = self.world.spawn_actor(bp, tf, attach_to=self.ego)
        q = queue.Queue()
        sensor.listen(q.put)
        self.sensors.append(sensor)
        self.queues["front"] = q
        self._owed_frames = 0
        self._frame_time = {}
        # Contacts arrive on the sensor's own thread: queue them (with the frame they happened
        # in) for the main thread. Each episode has its own queue, so a late report from the
        # previous episode's sensor cannot land in this one.
        hits = queue.Queue()
        self._hits = hits
        bp = lib.find("sensor.other.collision")
        collision = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        collision.listen(lambda event: hits.put(self._contact(event)))
        self.sensors.append(collision)

    @staticmethod
    def _contact(event):
        other = event.other_actor
        type_id = getattr(other, "type_id", "") or ""
        category = ("walker" if type_id.startswith("walker") else
                    "vehicle" if type_id.startswith("vehicle") else "static")
        return int(event.frame), int(getattr(other, "id", 0) or 0), category

    def _background_walkers(self, rng, route):
        """People walking along the pavements on both sides of the route. They are scripted
        to follow the pavement and never step onto the road (CARLA's own pedestrian AI
        stops on crossings)."""
        if not self.args.walkers:
            return
        carla = self.carla
        Driving, Sidewalk = carla.LaneType.Driving, carla.LaneType.Sidewalk

        def pavement(wp, side):
            """The pavement lane on one side of the road at wp, or None."""
            w = wp
            for _ in range(6):
                same = (w.lane_id > 0) == (wp.lane_id > 0)
                nxt = (w.get_right_lane() if same else w.get_left_lane()) if side > 0 else \
                    (w.get_left_lane() if same else w.get_right_lane())
                if nxt is None:
                    return None
                if nxt.lane_type == Sidewalk:
                    return nxt
                if nxt.lane_type not in (Driving, carla.LaneType.Parking, carla.LaneType.Shoulder,
                                         carla.LaneType.Biking, carla.LaneType.Border):
                    return None
                w = nxt
            return None

        lib = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        spots = [wp for wp, _ in route[2::3] if not wp.is_junction]
        rng.shuffle(spots)
        people = []
        for wp in spots:
            if len(people) >= self.args.walkers:
                break
            side = rng.choice((1, -1))
            pave = pavement(wp, side)
            if pave is None:
                continue
            r = pave.transform.get_right_vector()
            shift = rng.uniform(-0.4, 0.4)
            loc = pave.transform.location + carla.Location(r.x * shift, r.y * shift, 1.0)
            bp = rng.choice(lib)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc, pave.transform.rotation))
            if walker is None:
                continue
            self.actors.append(walker)
            people.append((walker, rng.choice((1.0, -1.0)), rng.uniform(0.9, 1.5)))

        def hook():
            for walker, sense, speed in people:
                here = self.map.get_waypoint(walker.get_location(), project_to_road=True,
                                             lane_type=Sidewalk)
                if here is None:
                    walker.apply_control(carla.WalkerControl(speed=0.0))
                    continue
                f = here.transform.get_forward_vector()
                walker.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(f.x * sense, f.y * sense, 0.0), speed=speed))
        self.scripted.append(hook)

    def _background_traffic(self, rng, route):
        # spawn points near the route (but clear of the car's start), so the traffic is
        # traffic the car meets
        start = route[0][0].transform.location
        near = [wp.transform.location for wp, _ in route[::5]]
        points = [p for p in self.map.get_spawn_points() if p.location.distance(start) > 30]
        rng.shuffle(points)
        points.sort(key=lambda p: min(p.location.distance(q) for q in near) > 120.0)
        lib = self.world.get_blueprint_library().filter("vehicle.*")
        cars = [bp for bp in lib if int(bp.get_attribute("number_of_wheels")) == 4]
        for tf in points[:self.args.background]:
            actor = self.world.try_spawn_actor(rng.choice(cars), tf)
            if actor is not None:
                actor.set_autopilot(True, self.tm.get_port())
                self.actors.append(actor)

    # -- stepping ----------------------------------------------------------------
    def step(self, mode, control):
        carla = self.carla
        throttle, steer, brake = self._control_tuple(self._nominal_control) if mode == 0 else control
        self.applied = [throttle, steer, brake]
        vc = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        for _ in range(TICKS_PER_STEP):
            for hook in self.scripted:
                hook()
            self.ego.apply_control(vc)
            frame = self._tick()
            speed = self._speed()
            loc = self.ego.get_location()
            at_red = self._red_light_check(loc, speed)
            self.tracker.update(loc.x, loc.y, speed, self.prev_speed, DT,
                                limit_kmh=self.ego.get_speed_limit() or 30.0, excused=at_red)
            self._frame_time[frame] = self.tracker.time
            self.prev_speed = speed
            if self.tracker.termination:
                break
        image = self._frames()       # this step's camera frames: by then its contacts are in too
        self._apply_hits()
        self._nominal_control = self._nominal_step()
        return self._state(image)

    def _nominal_step(self):
        try:
            return self.nominal.run_step()
        except AttributeError:
            # BehaviorAgent dereferences its next waypoint unchecked once its plan is used up
            # (or the car has left it): the planner then asks to stop
            return self.carla.VehicleControl(throttle=0.0, brake=0.5)

    @staticmethod
    def _control_tuple(vc):
        return [float(vc.throttle), float(vc.steer), float(vc.brake)]

    def _speed(self):
        v = self.ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _red_light_check(self, loc, speed):
        """Count red lights run; return whether the car is held at a red (or yellow) light.

        A light counts as run when the car enters the junction while that light is red, so
        pulling away the moment it turns green is not an infraction. It counts once for the
        crossing, however many ticks the car then moves through the junction."""
        carla = self.carla
        at_red = False
        at_light = self.ego.is_at_traffic_light()
        if at_light:
            light = self.ego.get_traffic_light()
            if light is not None:
                self.red_candidate = (light, self.tracker.time)
                at_red = light.get_state() in (carla.TrafficLightState.Red,
                                               carla.TrafficLightState.Yellow)
        if not at_red:
            # The planner looks ahead and stops for a red before the car reaches the light's
            # trigger volume; that wait is just as legitimate.
            at_red = bool(self.nominal.traffic_light_manager())
        here = None
        if self.red_candidate:
            light, seen = self.red_candidate
            here = self.map.get_waypoint(loc)
            if speed > 0.5 and here is not None and here.is_junction:
                if light.get_state() == carla.TrafficLightState.Red and light.id != self.red_counted:
                    self.tracker.red_lights += 1
                    self.red_counted = light.id
                self.red_candidate = None
            elif self.tracker.time - seen >= 3.0:
                self.red_candidate = None
        if self.red_counted is not None and not at_light:
            here = here if here is not None else self.map.get_waypoint(loc)
            if here is None or not here.is_junction:
                self.red_counted = None      # out of the junction: that crossing is over
        return at_red

    # -- observation -------------------------------------------------------------
    def _frames(self):
        """The front camera's frames owed for the ticks since the last call, one per tick, in
        arrival order: the last one is the current view. A frame missing for FRAME_TIMEOUT_S
        is a bridge error."""
        import queue
        q, item = self.queues["front"], None
        owed, self._owed_frames = self._owed_frames, 0
        for _ in range(owed):
            try:
                item = q.get(timeout=FRAME_TIMEOUT_S)
            except queue.Empty:
                raise RuntimeError("the front camera sent no frame within {:.0f} s of its "
                                   "tick".format(FRAME_TIMEOUT_S))
        if item is None:
            raise RuntimeError("no simulator tick since the last camera frame")
        return item

    def _probe_frame(self, bgr):
        """The frame with a road user pasted on the road ahead (8-25 m, within about a lane
        width of the centre line), sized for its distance by the ground-plane geometry."""
        np, cv2, rng = self.np, self.cv2, self.probe_rng
        if not hasattr(self, "_sprites"):
            folder = os.path.join(HERE, "probe_sprites")
            names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
            self._sprites = [(n, cv2.imread(os.path.join(folder, n), cv2.IMREAD_UNCHANGED))
                             for n in names if n.endswith(".png")]
            self._sprites = [(n, img) for n, img in self._sprites if img is not None and img.shape[2] == 4]
        h, w = bgr.shape[:2]
        f = w / 2.0
        dist = rng.uniform(8.0, 25.0)
        lateral = rng.uniform(-1.5, 1.5)
        out = bgr.copy()
        self.probe_actor = {"type": "walker", "x": round(CAM_X + dist, 2), "y": round(lateral, 2),
                            "yaw": 0.0, "vx": 0.0, "vy": 0.0, "length": 0.6, "width": 0.6}
        if self._sprites:
            name, sprite = rng.choice(self._sprites)
            metres = 1.5 if name.startswith("vehicle") else 1.75     # sprite height in metres
            if name.startswith("vehicle"):
                self.probe_actor.update(type="vehicle", length=4.5, width=1.9)
            ph = max(4, int(f * metres / dist))
            pw = max(2, int(sprite.shape[1] * ph / float(sprite.shape[0])))
            sprite = cv2.resize(sprite, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            ph, pw = max(4, int(f * 1.75 / dist)), max(2, int(f * 0.5 / dist))
            sprite = np.zeros((ph, pw, 4), np.uint8)
            sprite[:, :, :3], sprite[:, :, 3] = 40, 255
        bottom = int(h / 2.0 + f * CAM_Z / dist)
        left = int(f + f * lateral / dist - pw / 2.0)
        y0, x0 = max(0, bottom - ph), max(0, left)
        y1, x1 = min(h, bottom), min(w, left + pw)
        if y1 <= y0 or x1 <= x0:
            return out
        part = sprite[y0 - (bottom - ph):y1 - (bottom - ph), x0 - left:x1 - left].astype(np.float64)
        alpha = part[:, :, 3:4] / 255.0
        region = out[y0:y1, x0:x1].astype(np.float64)
        out[y0:y1, x0:x1] = (alpha * part[:, :, :3] + (1.0 - alpha) * region).astype(np.uint8)
        return out

    def _state(self, image):
        np, cv2 = self.np, self.cv2
        bgr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        encode = lambda img: base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode("ascii")  # noqa: E731
        tf = self.ego.get_transform()
        nav = navigation(self.route_xy, self.options, self.tracker.index, tf.location.x,
                         tf.location.y, math.radians(tf.rotation.yaw))
        state = {
            "frame": {"image": encode(bgr)},
            "probe": {"image": encode(self._probe_frame(bgr))},
            "ego": {"speed": round(self._speed(), 3), "applied": [round(v, 3) for v in self.applied]},
            "navigation": nav,
            "nominal": {"control": [round(v, 3) for v in self._control_tuple(self._nominal_control)]},
            "truth": self.tracker.truth(),
        }
        if self.privileged:
            state["privileged"] = self._privileged(tf)
            state["probe"]["actor"] = self.probe_actor
        return state

    def _privileged(self, tf):
        """Ground truth in the car's frame (x forward, y right, metres, from the car's centre):
        every vehicle within VEHICLE_RADIUS_M (with `reverse`: its reversing lights are on)
        and every walker and prop within NEAR_RADIUS_M, and the route ahead with its lane
        width, the driving lanes to its left (nearest first, until one that is not a driving
        lane) with their centre and direction, and the lanes to its right as far as the kerb
        (driving, shoulder, parking or bike lanes) with their centre and type; `lights`: the
        traffic lights whose stop line lies on that stretch of the route, each with its stop
        point and state (red, yellow, green or off)."""
        yaw = math.radians(tf.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = tf.location.x, tf.location.y
        reverse_light = int(self.carla.VehicleLightState.Reverse)

        def local(x, y):
            dx, dy = x - ox, y - oy
            return round(dx * c + dy * s, 2), round(-dx * s + dy * c, 2)

        actors = []
        for actor in self.world.get_actors():
            type_id = actor.type_id
            kind = ("vehicle" if type_id.startswith("vehicle.") else
                    "walker" if type_id.startswith("walker.") else
                    "prop" if type_id.startswith("static.prop.") else None)
            if kind is None or actor.id == self.ego.id:
                continue
            atf = actor.get_transform()
            loc = atf.location
            if loc.distance(tf.location) > (VEHICLE_RADIUS_M if kind == "vehicle" else NEAR_RADIUS_M):
                continue
            box = actor.bounding_box
            centre = atf.transform(box.location)
            x, y = local(centre.x, centre.y)
            v = actor.get_velocity()
            now, last = self.tracker.time, self._seen.get(actor.id)
            self._seen[actor.id] = (loc.x, loc.y, now)
            if last is not None and now > last[2] and abs(v.x) + abs(v.y) < 0.05:
                # staged actors moved by set_transform report no velocity: difference it
                v = self.carla.Vector3D((loc.x - last[0]) / (now - last[2]),
                                        (loc.y - last[1]) / (now - last[2]), 0.0)
            ext = box.extent
            entry = {"id": actor.id, "type": kind, "blueprint": type_id, "x": x, "y": y,
                     "z": round(centre.z - tf.location.z, 2),
                     "yaw": round(math.degrees(_wrap(math.radians(atf.rotation.yaw) - yaw)), 1),
                     "vx": round(v.x * c + v.y * s, 2), "vy": round(-v.x * s + v.y * c, 2),
                     "length": round(2 * ext.x, 2), "width": round(2 * ext.y, 2),
                     "height": round(2 * ext.z, 2)}
            if kind == "vehicle":
                try:
                    entry["reverse"] = bool(int(actor.get_light_state()) & reverse_light)
                except (RuntimeError, AttributeError, TypeError):
                    entry["reverse"] = False
            actors.append(entry)
        lanes = []
        LaneType = self.carla.LaneType
        passable = (LaneType.Driving, LaneType.Shoulder, LaneType.Parking, LaneType.Biking)
        for wp in self.route_wps[max(0, self.tracker.index - 5):self.tracker.index + 45]:
            loc = wp.transform.location
            x, y = local(loc.x, loc.y)
            lefts, w = [], wp
            for _ in range(4):
                # beyond the centre line, CARLA's "left" of an oncoming lane points back
                # across the road: walk outward from the car's side instead
                nxt = w.get_left_lane() if (w.lane_id > 0) == (wp.lane_id > 0) else w.get_right_lane()
                if nxt is None or nxt.lane_type != self.carla.LaneType.Driving:
                    break
                lx, ly = local(nxt.transform.location.x, nxt.transform.location.y)
                lefts.append({"x": lx, "y": ly, "width": round(nxt.lane_width, 2),
                              "same_direction": (nxt.lane_id > 0) == (wp.lane_id > 0)})
                w = nxt
            rights, w = [], wp
            for _ in range(3):
                # to the kerb: driving lanes, then shoulder / parking / bike lane, not pavement
                nxt = w.get_right_lane()
                if nxt is None or nxt.lane_type not in passable:
                    break
                rx, ry = local(nxt.transform.location.x, nxt.transform.location.y)
                rights.append({"x": rx, "y": ry, "width": round(nxt.lane_width, 2),
                               "type": str(nxt.lane_type)})
                w = nxt
            lanes.append({"x": x, "y": y, "width": round(wp.lane_width, 2),
                          "junction": wp.is_junction, "left": lefts[0] if lefts else None,
                          "lefts": lefts, "rights": rights})
        TLS = self.carla.TrafficLightState
        lights = []
        for index, signal, stop in self.route_lights:
            if max(0, self.tracker.index - 5) <= index < self.tracker.index + 45:
                x, y = local(stop.x, stop.y)
                state = signal.get_state()
                lights.append({"id": signal.id, "x": x, "y": y,
                               "state": "red" if state == TLS.Red else "yellow" if state == TLS.Yellow
                               else "green" if state == TLS.Green else "off"})
        light = self.ego.get_traffic_light() if self.ego.is_at_traffic_light() else None
        return {"actors": actors, "route": lanes, "lights": lights,
                "red_light": bool(light is not None and light.get_state() in (TLS.Red, TLS.Yellow)),
                "speed_limit": round((self.ego.get_speed_limit() or 30.0) / 3.6, 2)}

    def _route_lights(self):
        """[(route index, light, stop location)]: every traffic light with a stop waypoint in a
        lane of the route (within 3 m of one of its points), at the route point nearest that
        stop, in route order."""
        by_lane = {}
        for i, wp in enumerate(self.route_wps):
            by_lane.setdefault((wp.road_id, wp.lane_id), []).append(i)
        found = {}
        for light in self.world.get_actors().filter("traffic.traffic_light*"):
            try:
                stops = light.get_stop_waypoints()
            except RuntimeError:
                continue
            for stop in stops:
                loc, best = stop.transform.location, None
                for i in by_lane.get((stop.road_id, stop.lane_id), ()):
                    gap = self.route_wps[i].transform.location.distance(loc)
                    if gap < 3.0 and (best is None or gap < best[0]):
                        best = (gap, i)
                if best is not None and (light.id not in found or best[1] < found[light.id][0]):
                    found[light.id] = (best[1], light, loc)
        return sorted(found.values(), key=lambda item: (item[0], item[1].id))

    def close(self):
        try:
            if self.world is not None and self.server.poll() is None:
                self._cleanup()
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except Exception:  # noqa: BLE001
            pass
        self._kill_server()


class ScenarioBuilder(object):
    """Route + staged hazard for one kind, from the episode's rng. Everything is placed
    relative to map waypoints, so it works in any town with suitable roads; a draw that
    does not fit is redrawn (bounded). A variant (see the module docstring) restages the
    kind through `drive_variants.stage(builder, route, junction)`."""

    ROUTE_M = 120.0           # laid out this long, then cut TAIL_M past the hazard
    HAZARD_M = (35.0, 55.0)
    TAIL_M = (25.0, 35.0)     # past a junction hazard, counted from where the junction ends

    def __init__(self, backend, rng, kind, variant=None, distractors=0):
        self.b, self.rng, self.kind, self.variant = backend, rng, kind, variant
        self.distractors = distractors
        self.carla, self.map, self.world = backend.carla, backend.map, backend.world
        self.RoadOption = backend.RoadOption
        self.hazards = []       # the staged actors; colliding with one zeroes the episode
        self.staging = {}       # what the staging drew, for the grading side's records
        self.variants = None
        if variant:
            try:
                sys.path.insert(0, HERE)
                import drive_variants
            except ImportError:
                raise RuntimeError("variant {!r} needs drive_variants, which this copy of the "
                                   "bridge does not have".format(variant))
            self.variants = drive_variants

    def build(self):
        for attempt in range(80):
            self.hazards = []
            self.staging = {}
            spawned = len(self.b.actors)
            try:
                result = self._attempt()
            except (_Redraw, RuntimeError):   # RuntimeError: a spawn collided; draw again
                if self.b.server.poll() is not None:
                    raise                     # no collision: the server is gone
                self._discard(self.b.actors[spawned:])
                del self.b.actors[spawned:]
                self.b.scripted = []
                continue
            if result is not None:
                return self._trim(*result)
        raise RuntimeError("no placement for {} in {} after 80 draws".format(
            self.kind, self.b.town))

    def _trim(self, route, ego_tf):
        """End the route a little past the hazard (past its junction, for a junction hazard):
        once the car is through the situation the episode is over."""
        index = getattr(self, "hazard_index", None)
        if index is None:
            return route, ego_tf
        while index < len(route) - 1 and route[index][0].is_junction:
            index += 1
        tail, run = self.rng.uniform(*self.TAIL_M), 0.0
        for i in range(index + 1, len(route)):
            run += route[i][0].transform.location.distance(route[i - 1][0].transform.location)
            if run >= tail:
                return route[:i + 1], ego_tf
        return route, ego_tf

    def _discard(self, actors):
        if actors:
            self.b.client.apply_batch_sync([self.carla.command.DestroyActor(a.id) for a in actors], False)

    # -- route --------------------------------------------------------------------
    def _turn(self, entry):
        """Degrees turned through the junction starting at `entry` (left < 0 < right)."""
        wp, guard = entry, 0
        while wp.is_junction and guard < 40:
            nxt = wp.next(2.0)
            if not nxt:
                break
            wp, guard = nxt[0], guard + 1
        return math.degrees(_wrap(math.radians(wp.transform.rotation.yaw - entry.transform.rotation.yaw)))

    def _route(self, want_turn):
        rng = self.rng
        start = self.map.get_waypoint(rng.choice(self.map.get_spawn_points()).location)
        wps, opts, total, wp = [start], [self.RoadOption.LANEFOLLOW], 0.0, start
        hazard_junction = None
        while total < self.ROUTE_M:
            nexts = wp.next(2.0)
            if not nexts:
                raise _Redraw()
            if len(nexts) > 1:
                turns = [(self._turn(n), n) for n in nexts]
                if want_turn and hazard_junction is None and total >= self.HAZARD_M[0] - 15:
                    if want_turn == "left":
                        fit = [n for a, n in turns if a < -45]
                    else:
                        fit = [n for a, n in turns if abs(a) < 30]
                    if not fit:
                        raise _Redraw()
                    choice = rng.choice(fit)
                    hazard_junction = (len(wps), wp)
                else:
                    straight = [n for a, n in turns if abs(a) < 30]
                    choice = rng.choice(straight or [n for _, n in turns])
                angle = self._turn(choice)
                option = (self.RoadOption.LEFT if angle < -45 else
                          self.RoadOption.RIGHT if angle > 45 else self.RoadOption.STRAIGHT)
            else:
                choice = nexts[0]
                option = self.RoadOption.LANEFOLLOW if not choice.is_junction else opts[-1]
            total += choice.transform.location.distance(wp.transform.location)
            wps.append(choice)
            opts.append(option)
            wp = choice
        if want_turn and hazard_junction is None:
            raise _Redraw()
        return list(zip(wps, opts)), hazard_junction

    def _at_distance(self, route, metres):
        total = 0.0
        for i in range(1, len(route)):
            total += route[i][0].transform.location.distance(route[i - 1][0].transform.location)
            if total >= metres:
                return i, route[i][0]
        raise _Redraw()

    # -- kinds ---------------------------------------------------------------------
    def _attempt(self):
        carla, rng = self.carla, self.rng
        want = {"red_light_runner": "straight", "oncoming_left_turn": "left"}.get(self.kind)
        if self.variants is not None:
            want = self.variants.junction_wanted(self.variant, want)
        # a variant that needs a particular kind of road may lay out the route itself
        planned = self.variants.route(self, want) if self.variants is not None else None
        route, junction = planned if planned is not None else self._route(want)
        ego_tf = route[0][0].transform
        ego_tf = carla.Transform(ego_tf.location + carla.Location(z=0.5), ego_tf.rotation)
        self.hazard_index = None
        if self.variants is not None:
            self.hazard_index = self.variants.stage(self, route, junction)
            return route, ego_tf
        if self.kind in ("construction_zone", "broken_down_vehicle", "occluded_pedestrian"):
            self.hazard_index, hwp = self._at_distance(route, rng.uniform(*self.HAZARD_M))
            if hwp.is_junction or any(w.is_junction for w in hwp.previous(12.0) + hwp.next(12.0)):
                raise _Redraw()
            if self.kind != "occluded_pedestrian" and not self._passable(hwp):
                raise _Redraw()
            getattr(self, "_stage_" + self.kind)(hwp)
            return route, ego_tf
        index, entry = junction
        self.hazard_index = index
        getattr(self, "_stage_" + self.kind)(route, index, entry)
        return route, ego_tf

    def _passable(self, hwp, back_m=30.0):
        """A drivable lane on the left at the hazard and over the approach to it, so a car
        stopped in front of the hazard can always pull out around it."""
        wps = [hwp] + [w for d in range(2, int(back_m) + 1, 2) for w in hwp.previous(float(d))[:1]]
        if len(wps) < int(back_m) // 2 + 1:
            return False
        for w in wps:
            left = w.get_left_lane()
            if w.is_junction or left is None or left.lane_type != self.carla.LaneType.Driving:
                return False
        return True

    def _offset(self, wp, forward, right, z=0.0):
        tf = wp.transform
        f, r = tf.get_forward_vector(), tf.get_right_vector()
        loc = tf.location + self.carla.Location(f.x * forward + r.x * right,
                                                f.y * forward + r.y * right, z)
        return self.carla.Transform(loc, tf.rotation)

    def _stage_construction_zone(self, wp):
        for lateral in (-1.2, 0.0, 1.2):
            self.hazards.append(self.b._spawn("static.prop.constructioncone",
                                              self._offset(wp, 0.0, lateral, 0.1)))
        self.hazards.append(self.b._spawn("static.prop.streetbarrier", self._offset(wp, 2.5, 0.0, 0.1)))
        self.hazards.append(self.b._spawn("static.prop.trafficwarning", self._offset(wp, 6.0, 0.0, 0.1)))

    def _stage_broken_down_vehicle(self, wp):
        lib = self.world.get_blueprint_library().filter("vehicle.*")
        cars = [bp.id for bp in lib if int(bp.get_attribute("number_of_wheels")) == 4]
        car = self.b._spawn(self.rng.choice(cars), self._offset(wp, 0.0, 0.0, 0.3))
        self.hazards.append(car)
        car.apply_control(self.carla.VehicleControl(hand_brake=True, brake=1.0))
        lights = self.carla.VehicleLightState
        car.set_light_state(lights(lights.LeftBlinker | lights.RightBlinker))

    def _stage_occluded_pedestrian(self, wp):
        carla, rng = self.carla, self.rng
        kerb = wp.lane_width / 2.0
        van = self.b._spawn("vehicle.volkswagen.t2", self._offset(wp, 0.0, kerb + 1.2, 0.3),
                            physics=False)
        van.set_simulate_physics(False)
        walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        bp = rng.choice(walkers)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        spot = self._offset(wp, 3.5, kerb + 1.6, 1.0)
        walker = self.world.spawn_actor(bp, spot)
        self.b.actors.append(walker)
        self.hazards.append(walker)
        speed = rng.uniform(2.5, 3.5)
        right = wp.transform.get_right_vector()
        direction = carla.Vector3D(-right.x, -right.y, 0.0)
        to_centre = (kerb + 1.6) / speed          # walking time to the ego lane centre
        pause = 3.0                                # hesitates there, then walks on
        state = {"started": False, "t0": 0.0}

        def hook():
            ego = self.b.ego.get_location()
            ahead = walker.get_location().distance(ego)
            v = self.b._speed()
            if not state["started"] and ahead < v * 1.8 + 5.0:
                state["started"], state["t0"] = True, self.b.tracker.time
            if state["started"]:
                elapsed = self.b.tracker.time - state["t0"]
                moving = elapsed < to_centre or to_centre + pause <= elapsed < to_centre + pause + 4.0
                walker.apply_control(carla.WalkerControl(direction=direction,
                                                         speed=speed if moving else 0.0))
        self.b.scripted.append(hook)

    def _conflict_vehicle(self, entry, angle_range, straight_through):
        """A vehicle approaching the same junction from another arm."""
        carla, rng = self.carla, self.rng
        junction = None
        probe = entry
        for _ in range(20):
            nxt = probe.next(2.0)
            if not nxt:
                break
            probe = nxt[0]
            if probe.is_junction:
                junction = probe.get_junction()
                break
        if junction is None:
            raise _Redraw()
        ego_yaw = entry.transform.rotation.yaw
        pairs = []
        for start, end in junction.get_waypoints(carla.LaneType.Driving):
            rel = abs(math.degrees(_wrap(math.radians(start.transform.rotation.yaw - ego_yaw))))
            turn = abs(math.degrees(_wrap(math.radians(end.transform.rotation.yaw - start.transform.rotation.yaw))))
            if angle_range[0] <= rel <= angle_range[1] and (turn < 30 if straight_through else True):
                pairs.append((start, end))
        if not pairs:
            raise _Redraw()
        start, end = rng.choice(pairs)
        back = start.previous(25.0)
        if not back:
            raise _Redraw()
        tf = back[0].transform
        lib = self.world.get_blueprint_library().filter("vehicle.*")
        cars = [bp.id for bp in lib if int(bp.get_attribute("number_of_wheels")) == 4]
        car = self.b._spawn(rng.choice(cars), carla.Transform(tf.location + carla.Location(z=0.5), tf.rotation))
        self.hazards.append(car)
        exit_wp = end.next(20.0)
        return car, start, (exit_wp[0] if exit_wp else end), junction

    def _stage_red_light_runner(self, route, index, entry):
        from agents.navigation.basic_agent import BasicAgent
        carla = self.carla
        car, start, goal, junction = self._conflict_vehicle(entry, (60, 120), True)
        lights = self.world.get_traffic_lights_from_waypoint(entry, 40.0)
        for light in lights:
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(120.0)
        for light in self.world.get_traffic_lights_from_waypoint(start, 40.0):
            light.set_state(carla.TrafficLightState.Red)
            light.set_red_time(120.0)
        self._drive_when(car, goal, entry, lead_s=2.6, speed_kmh=40.0,
                         opts={"ignore_traffic_lights": True, "ignore_vehicles": True},
                         agent_cls=BasicAgent)

    def _stage_oncoming_left_turn(self, route, index, entry):
        from agents.navigation.basic_agent import BasicAgent
        carla = self.carla
        car, start, goal, junction = self._conflict_vehicle(entry, (150, 180), True)
        for light in (self.world.get_traffic_lights_from_waypoint(entry, 40.0) +
                      self.world.get_traffic_lights_from_waypoint(start, 40.0)):
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(120.0)
        self._drive_when(car, goal, entry, lead_s=3.0, speed_kmh=35.0,
                         opts={"ignore_traffic_lights": True, "ignore_vehicles": True},
                         agent_cls=BasicAgent)

    def _drive_when(self, car, goal, ego_entry, lead_s, speed_kmh, opts, agent_cls):
        carla = self.carla
        agent = agent_cls(car, target_speed=speed_kmh, opt_dict=opts)
        agent.set_destination(goal.transform.location)
        car.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        state = {"go": False}

        def hook():
            ego = self.b.ego.get_location()
            distance = ego.distance(ego_entry.transform.location)
            if not state["go"] and distance < self.b._speed() * lead_s + 4.0:
                state["go"] = True
            if state["go"]:
                car.apply_control(agent.run_step())
        self.b.scripted.append(hook)


class _Redraw(Exception):
    pass


def _free_port():
    for _ in range(50):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        if 2000 <= port < 59000 and port % 2 == 0:
            return port
    return 2000


# =========================================================================== main loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("carla", "mock"), default="carla")
    parser.add_argument("--carla-root", default="/opt/carla/server")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512,
                        help="camera width in pixels; height is half of it")
    # Low crashes CARLA 0.9.15 (signal 11) right after a town load on recent drivers.
    parser.add_argument("--quality", default="Epic", choices=("Low", "Epic"))
    parser.add_argument("--background", type=int, default=20,
                        help="background traffic vehicles (CARLA backend)")
    parser.add_argument("--walkers", type=int, default=20,
                        help="pedestrians on the pavements near the route (CARLA backend)")
    args = parser.parse_args()

    reply_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(reply_fd, "w", buffering=1)

    try:
        backend = (MockBackend if args.backend == "mock" else CarlaBackend)(args)
    except Exception as exc:  # noqa: BLE001
        _emit({"ready": False, "error": "{}: {}".format(type(exc).__name__, exc)})
        return 1
    _emit({"ready": True, "backend": args.backend})
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                cmd = message.get("cmd")
                if cmd == "reset":
                    try:
                        _emit(backend.reset(message["spec"]))
                    except ServerExited as exc:
                        _log("error: {}".format(exc))
                        _emit({"error": "ServerExited: {}".format(exc), "server_exited": True})
                        os._exit(3)      # nothing to close: the server is gone, its client threads too
                elif cmd == "step":
                    _emit(backend.step(int(message["mode"]), message["control"]))
                elif cmd == "close":
                    _emit({"ok": True})
                    break
                else:
                    _emit({"error": "unknown command {!r}".format(cmd)})
            except Exception as exc:  # noqa: BLE001
                _log("error: {}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc()))
                _emit({"error": "{}: {}".format(type(exc).__name__, exc)})
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
