"""Scenes for the sensor model's tests and gates: the example venue, synthetic duct fields for the timing gate, a round room
for the distortion gate, the stub-car model from the contract's greybox, exact placement of the car and kinematic motion.

Nothing here is part of the grader: the cell builds its models through the venue generator, the duct model and the drive model and drives the suite itself.
"""

import copy
import math
import os

import mujoco
import numpy as np

from sketch.examples import constants as K
from sketch.examples import greybox

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def min_world():
    import json
    with open(os.path.join(EXAMPLES, "min_world.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def bare_world(world_id, ducts, clutter=(), patches=(), floor_kind="polished_concrete", seed=1):
    """A world dict with just what greybox and the sensors read (not a checker-valid world file)."""
    return {"schema": K.WORLD_SCHEMA, "world_id": world_id, "venue_sheet": {"floor": {"kind": floor_kind}},
            "truth": {"ducts": list(ducts), "clutter": list(clutter), "open_walls": [],
                      "floor": {"kind": floor_kind, "base_grip": 0.7, "patches": list(patches)},
                      "timing_line": {"a": [0.0, -0.05], "b": [0.0, 0.05]}, "checkpoints": [],
                      "avoidance": {"boxes": [], "slow_cars": []},
                      "seeds": {"relocation": seed, "putback": seed + 1, "sensors": seed + 2, "others": seed + 3}}}


def chain_ducts(path, prefix, wall_class="orange_ribbed", piece=K.PIECE_LENGTH_NOMINAL_M, per_duct=K.DUCT_PIECES_FULL,
                closed=True, limit=None):
    """Lay ducts along a polyline by walking chords of `piece` length; 15 pieces per duct. Returns the duct list."""
    p = np.asarray(path, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(s[-1] // piece)
    if limit is not None:
        n = min(n, limit)
    pts = np.stack([np.interp(np.arange(n + 1) * piece, s, p[:, 0]), np.interp(np.arange(n + 1) * piece, s, p[:, 1])], axis=1)
    pieces = []
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        m = 0.5 * (a + b)
        pieces.append([float(m[0]), float(m[1]), math.atan2(b[1] - a[1], b[0] - a[0])])
    ducts = []
    for j in range(0, n, per_duct):
        chunk = pieces[j:j + per_duct]
        # a chord walk on a bend gives chords slightly shorter than the arc; the piece length is the chord
        c = float(np.linalg.norm(pts[j + 1] - pts[j]))
        ducts.append({"id": "%s%d" % (prefix, j // per_duct), "wall": prefix, "wall_class": wall_class, "diameter_m": 0.33,
                      "piece_length_m": c, "length_m": c * len(chunk), "pieces": chunk})
    return ducts


def rounded_rect(a, b, rc, step=0.02):
    """Closed polyline of a rounded rectangle with half sizes a, b and corner radius rc, counter-clockwise."""
    pts = []
    for cx, cy, th0 in ((a - rc, b - rc, 0.0), (-(a - rc), b - rc, 0.5 * math.pi), (-(a - rc), -(b - rc), math.pi),
                        (a - rc, -(b - rc), 1.5 * math.pi)):
        n = max(2, int(0.5 * math.pi * rc / step))
        for i in range(n):
            th = th0 + 0.5 * math.pi * i / n
            pts.append((cx + rc * math.cos(th), cy + rc * math.sin(th)))
        nxt = th0 + 0.5 * math.pi
        ex, ey = cx + rc * math.cos(nxt), cy + rc * math.sin(nxt)
        pts.append((ex, ey))
    return np.array(pts)


def duct_field_world(n_pieces=902, layout="rings"):
    """A synthetic venue with exactly `n_pieces` duct pieces for the timing gate.

    rings  concentric rounded rectangles 1.93 m apart (a dense hall: every wall within about 15 m of the car;
           the worst case for the ray cost because the 10 m cutoff removes few pieces). Car pose: `car_pose`.
    rows   the example oval (108 pieces) plus straight 15-piece ducts in rows 0.6 m apart beside it, the layout of
           an early 902-segment probe (5.9 ms with a 30 m cutoff)."""
    ducts = []
    if layout == "rings":
        m, total = 0, 0
        while total < n_pieces:
            a, b = 3.0 + 1.93 * m, 2.0 + 1.93 * m
            d = chain_ducts(rounded_rect(a, b, 1.5 + 0.5 * m), "R%d_" % m, wall_class=("orange_ribbed", "yellow_ribbed", "black_plain")[m % 3],
                            limit=n_pieces - total)
            total += sum(len(x["pieces"]) for x in d)
            ducts += d
            m += 1
        w = bare_world("duct_field_rings_%d" % n_pieces, ducts)
        w["car_pose"] = [0.5 * (3.0 + 1.93 * 2 + 3.0 + 1.93 * 3), 0.0, 0.5 * math.pi]
        return w
    if layout == "rows":
        base = min_world()
        ducts = copy.deepcopy(base["truth"]["ducts"])
        total = sum(len(d["pieces"]) for d in ducts)
        row = 0
        while total < n_pieces:
            k = min(K.DUCT_PIECES_FULL, n_pieces - total)
            y0 = 4.8 + 0.6 * row
            pcs = [[-3.81 + K.PIECE_LENGTH_NOMINAL_M * (i + 0.5), y0, 0.0] for i in range(k)]
            ducts.append({"id": "X%d" % row, "wall": "X", "wall_class": "orange_ribbed", "diameter_m": 0.33,
                          "piece_length_m": K.PIECE_LENGTH_NOMINAL_M, "length_m": K.PIECE_LENGTH_NOMINAL_M * k, "pieces": pcs})
            total += k
            row += 1
        w = bare_world("duct_field_rows_%d" % n_pieces, ducts)
        w["car_pose"] = base["truth"]["start_pose"]["pose"]
        return w
    raise ValueError("layout must be 'rings' or 'rows'")


def round_room_world(radius=3.0, pillars=True):
    """A closed ring of ducts (axis radius `radius`) with a few pillars inside: every beam meets a surface at an oblique
    angle, so each sub-scan pins the sensor position in x and y (the distortion gate)."""
    n = 400
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ducts = chain_ducts(np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1), "W", wall_class="orange_ribbed")
    clutter = []
    if pillars:
        for i, (x, y) in enumerate(((1.6, 1.2), (-1.4, 1.5), (-1.7, -1.1), (1.2, -1.6), (2.2, 0.1))):
            clutter.append({"id": "Q%d" % i, "kind": "other", "shape": "cylinder", "pose": [x, y, 0.0],
                            "size_m": [0.2, 0.2, 1.0], "reflectivity": "normal"})
    return bare_world("round_room", ducts, clutter)


def _q(yaw):
    return "%.9g 0 0 %.9g" % (math.cos(0.5 * yaw), math.sin(0.5 * yaw))


def bodies_xml(world, xml, grouping):
    """Rewrite the static ducts of a greybox model text as bodies, for the timing gate only:
    'piece' one body per piece with planar joints (the body structure of the duct model's pushable ducts: 902 bodies of 2 geoms),
    'duct'  one free body per duct holding all its pieces (what a merged LiDAR proxy per duct would cost).
    The geometry, names, groups and bits are those of greybox; only the bodies differ."""
    lines = [ln for ln in xml.split("\n") if not ln.startswith('<geom name="duct/')]
    end = lines.index("</worldbody>")
    rot = 'quat="0.70710678 0 0.70710678 0"'
    out = []
    for d in world["truth"]["ducts"]:
        c = d["piece_length_m"]
        if grouping == "duct":
            out.append('<body name="duct/%s"><freejoint name="duct/%s/root"/>' % (d["id"], d["id"]))
        for k, (x, y, yaw) in enumerate(d["pieces"]):
            g = ('<geom name="%s" type="cylinder" size="%.9g %.9g" %s group="%d" contype="0" conaffinity="0" mass="0"/>'
                 '<geom name="%s" type="capsule" size="%.9g %.9g" %s group="%d" contype="%d" conaffinity="%d" mass="0.254"/>'
                 % (K.duct_geom(d["id"], k, "lidar"), K.DUCT_RADIUS_M, 0.5 * c, rot, K.GROUP_LIDAR,
                    K.duct_geom(d["id"], k, "col"), K.DUCT_RADIUS_M, max(1e-3, 0.5 * c - K.DUCT_RADIUS_M), rot, K.GROUP_COLLISION,
                    K.CONTACT["duct"][0], K.CONTACT["duct"][1]))
            if grouping == "duct":
                out.append('<frame pos="%.9g %.9g %.9g" quat="%s">%s</frame>' % (x, y, K.DUCT_RADIUS_M, _q(yaw), g))
            else:
                out.append('<body name="%s" pos="%.9g %.9g %.9g" quat="%s"><joint name="duct/%s/%d/x" type="slide" axis="1 0 0"/>'
                           '<joint name="duct/%s/%d/y" type="slide" axis="0 1 0"/><joint name="duct/%s/%d/yaw" type="hinge" axis="0 0 1"/>'
                           '%s</body>' % (K.duct_body(d["id"], k), x, y, K.DUCT_RADIUS_M, _q(yaw), d["id"], k, d["id"], k, d["id"], k, g))
        if grouping == "duct":
            out.append("</body>")
    return "\n".join(lines[:end] + out + lines[end:])


def model_of(world, variant="track", cars=None, timestep=K.PHYSICS_DT_S, ego_pose=None, duct_bodies=None):
    """The contract's static greybox model of `world` with stub cars; mj_forward done. Returns (model, data).
    duct_bodies: None (static ducts, greybox), 'piece' or 'duct' (see bodies_xml; timing only)."""
    if cars is None:
        pose = ego_pose if ego_pose is not None else world.get("car_pose") or world["truth"].get("start_pose", {}).get("pose", [0.0, 0.0, 0.0])
        cars = {K.EGO_PREFIX: pose}
    xml = greybox.build_xml(world, variant=variant, cars=cars, timestep=timestep)
    if duct_bodies:
        xml = bodies_xml(world, xml, duct_bodies)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def root_adr(model, prefix=K.EGO_PREFIX):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def quat_rpy(roll, pitch, yaw):
    """Quaternion of the rotation R = Rz(yaw) Ry(pitch) Rx(roll) (C's x forward, y left, z up)."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw], dtype=float), "XYZ")
    return q


def rot_rpy(roll, pitch, yaw):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, quat_rpy(roll, pitch, yaw))
    return r.reshape(3, 3)


def set_state(model, data, pos, quat, vel=(0.0, 0.0, 0.0), omega_local=(0.0, 0.0, 0.0), prefix=K.EGO_PREFIX):
    """Write the car's free joint: C's origin `pos`, orientation `quat`, world velocity `vel`, body rates `omega_local`."""
    qa, va = root_adr(model, prefix)
    data.qpos[qa:qa + 3] = pos
    data.qpos[qa + 3:qa + 7] = quat
    data.qvel[va:va + 3] = vel
    data.qvel[va + 3:va + 6] = omega_local


def set_pose(model, data, x, y, yaw, z=0.0, roll=0.0, pitch=0.0, prefix=K.EGO_PREFIX):
    set_state(model, data, [x, y, z], quat_rpy(roll, pitch, yaw), prefix=prefix)


class Kinematic(object):
    """Moves the ego car exactly along `motion(t) -> (pos, quat, vel_world, omega_local)` and runs the suite's per-step
    schedule without physics (the world is static). `tick()` returns the observation of one tick."""

    def __init__(self, model, data, suite, motion, t0=0.0, prefix=K.EGO_PREFIX):
        self.model, self.data, self.suite, self.motion, self.t, self.prefix = model, data, suite, motion, t0, prefix
        self.h = float(model.opt.timestep)
        self.apply()

    def apply(self):
        pos, quat, vel, om = self.motion(self.t)
        set_state(self.model, self.data, pos, quat, vel, om, prefix=self.prefix)

    def tick(self):
        self.suite.begin_tick()
        for i in range(1, self.suite.n_steps + 1):
            self.t += self.h
            self.apply()
            self.suite.after_step(i)
        return self.suite.end_tick()


def straight(x0, y0, yaw, v, z=0.0, roll=0.0, pitch=0.0, accel=0.0):
    """Motion along the heading `yaw` at speed v + accel t, with a fixed attitude."""
    q = quat_rpy(roll, pitch, yaw)
    c, s = math.cos(yaw), math.sin(yaw)

    def motion(t):
        dist = v * t + 0.5 * accel * t * t
        sp = v + accel * t
        return [x0 + c * dist, y0 + s * dist, z], q, [c * sp, s * sp, 0.0], [0.0, 0.0, 0.0]
    return motion


def spin_about_origin(x0, y0, omega, yaw0=0.0):
    """C turns about its own origin at a constant yaw rate (the IMU, off the origin, sees a centripetal force)."""
    def motion(t):
        yaw = yaw0 + omega * t
        return [x0, y0, 0.0], quat_rpy(0.0, 0.0, yaw), [0.0, 0.0, 0.0], [0.0, 0.0, omega]
    return motion


def circle(cx, cy, radius, speed):
    """C's origin moves counter-clockwise on a circle at constant speed, heading along the tangent."""
    om = speed / radius

    def motion(t):
        th = om * t
        yaw = th + 0.5 * math.pi
        return ([cx + radius * math.cos(th), cy + radius * math.sin(th), 0.0], quat_rpy(0.0, 0.0, yaw),
                [-speed * math.sin(th), speed * math.cos(th), 0.0], [0.0, 0.0, om])
    return motion
