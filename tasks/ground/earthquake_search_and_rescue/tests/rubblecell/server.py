"""RubbleCellServer: the grader-owned simulation of one search mission and the robot's API (cell internals).

One episode = one site: the crawler deploys at the street, the program searches the collapsed building, reports the
survivor's position, delivers the survival kit and brings the robot back. The server owns the MuJoCo model, steps it
only inside `step`, renders the sensors on request, applies the grip rule, watches safety and scores the mission
from its own state. Nothing the program sends is a scoring input except its survivor report (checked against the
truth) and what it physically does.

API (metres / radians; map frame = the robot's pose at deployment; `robot_api.md` is the agent-facing contract):
  describe()                       static facts: robot, sensors, budgets, rules
  get_state()                      sim time, odometry pose (map frame), IMU, joints, gripper, link, kit, done
  drive(v, w)                      forward speed (m/s) and yaw rate (rad/s), held until changed
  set_flippers(front, rear)        flipper angles (rad; 0 = straight out, + = tip raised)
  set_gauge(width)                 track gauge, centre to centre (0.30 .. 0.50 m)
  set_mast(height)                 mast extension (0 .. 0.45 m)
  set_head(pan, tilt)              pan-tilt head (rad)
  set_arm(q) | arm_to(p, pitch)    arm joint targets | a TCP point in the chassis frame (on-board IK)
  gripper(width)                   jaw opening target (0 .. 0.16 m)
  step(n)                          advance n x 10 ms; {sim_time, done, events}
  get_camera(name, depth)          mast | front | rear | gripper: PNG colour (+ 16-bit depth, mm)
  get_lidar()                      the lidar's latest sweep (float32 x, y, z, channel; lidar frame)
  get_thermal()                    radiometric thermal frame (uint16 centi-kelvin)
  report_survivor(x, y, z)         the survivor's position in the map frame (sent when the link is up)
  finish()                         ends the mission
"""

import base64
import importlib.util
import io
import math
import os
import time
import traceback

import mujoco
import numpy as np

from . import CELL_VERSION
from . import grasp
from .plan import draw_plan
from .robot import CRAWLER, Crawler, arm_fk, arm_ik
from .scene import build_scene
from .look import SceneLook, look_mode
from .sensors import CAMERAS, LIDAR, THERMAL, Odometry, Sensors

__all__ = ["RubbleCellServer", "DEFAULT_CONFIG", "PUBLIC_METHODS"]

PUBLIC_METHODS = ("describe", "get_state", "drive", "set_flippers", "set_gauge", "set_mast", "set_head", "set_arm", "arm_to",
                  "gripper", "step", "get_camera", "get_lidar", "get_thermal", "report_survivor", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "reference_run")
PRACTICE_METHODS = ("reset", "truth")
STATE_METHODS = ("drive", "set_flippers", "set_gauge", "set_mast", "set_head", "set_arm", "arm_to", "gripper", "step",
                 "reference_run", "reset")
COMMAND_METHODS = ("drive", "set_flippers", "set_gauge", "set_mast", "set_head", "set_arm", "arm_to", "gripper",
                   "report_survivor")
CTRL_DT = 0.01
PHYS_DT = 0.002

DEFAULT_CONFIG = {
    "sim_cap_s": 1800.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 700,
    "lidar_budget_sweeps": 1800,
    "thermal_budget_frames": 1800,
    "score": {"located": 0.45, "kit": 0.25, "returned": 0.30, "report_full_m": 1.0, "report_zero_m": 3.0,
              "seen_px": 40, "seen_range_m": 6.0, "kit_reach_m": 0.8, "home_radius_m": 3.0},
    "safety": {"survivor_force_N": 60.0, "survivor_ticks": 3, "stop_score_cap": 0.45, "prop_move_m": 0.05,
               "overturn_deg": 70.0, "overturn_s": 2.0, "fall_z_m": -0.8},
    "grip": {"min_force_N": 4.0, "payload_kg": 8.0, "slip_force_N": 260.0, "slip_ticks": 20},
    "link": {"tx_dbm": -28.0, "floor_dbm": -85.0, "ext_wall_db": 11.0, "int_wall_db": 7.0, "void_db": 14.0,
             "slab_db": 9.0},
    "temps": {"spread_c": 0.6},
    "trajectory_hz": 4.0,
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _finite(x, depth=0):
    """True when every number of a request's parameters is finite (JSON lets Infinity, NaN and 1e999 through) and the
    structure is small: a request cannot put a non-finite value into the cell."""
    if x is None or isinstance(x, (bool, str)):
        return True
    if isinstance(x, int):
        return abs(x) < (1 << 62)
    if isinstance(x, float):
        return math.isfinite(x)
    if depth >= 4:
        return False
    if isinstance(x, (list, tuple)):
        return len(x) <= 64 and all(_finite(v, depth + 1) for v in x)
    if isinstance(x, dict):
        return len(x) <= 32 and all(isinstance(k, str) and _finite(v, depth + 1) for k, v in x.items())
    return False


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _b64png(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RubbleCellServer(object):
    def __init__(self, config=None, mode="hidden", privileged=False, allow_reset=False, render=True, log=None,
                 reference_dir=None):
        self.cfg = _deep_update(_deep_update({}, DEFAULT_CONFIG), config or {})
        self.mode = mode
        self.privileged = bool(privileged)
        self.allow_reset = bool(allow_reset)
        self.render_enabled = bool(render)
        self.log = log or (lambda m: None)
        self.reference_dir = reference_dir
        self.done = True
        self.done_reason = None
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.work_wall = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.sc = None

    # ------------------------------------------------------------------ episode
    def reset(self, seed, kind="hidden"):
        t0 = time.time()
        if self.sc is not None:
            try:
                self.sensors.close()
            except Exception:  # noqa: BLE001
                pass
        self.seed = int(seed)
        self.kind = kind
        self.plan = draw_plan(self.seed, kind)
        self.sc = build_scene(self.plan)
        m, d = self.sc.model, self.sc.data
        self.m, self.d = m, d
        self.rng = np.random.default_rng((self.seed * 104729 + 7) & ((1 << 63) - 1))
        # get_state's own noise stream: reading the state never changes the odometry's or the sensors' noise
        self.rng_state = np.random.default_rng((self.seed * 7919 + 13) & ((1 << 63) - 1))
        self.robot = Crawler(m, d)
        self.robot.set_initial()
        mujoco.mj_forward(m, d)
        look = None
        if self.render_enabled and look_mode() == "filament":
            # the colour images' look (look.py): its own twins of the scene; the physics model is not touched
            look = SceneLook(self.sc)
            self.look_build_s = look.build_s
        self.sensors = Sensors(self.sc, self.rng, render=self.render_enabled, look=look)
        self._index()
        env = self.plan["env"]
        self.sensors.set_temperatures(self._temperatures(), self.survivor_geoms,
                                      outdoor_c=env["ambient_c"] + 7.0 * env["sun"])
        st = self.plan["start"]
        self.odom = Odometry(self.rng, (st["x"], st["y"]), st["yaw"])
        # settle 1 s (the robot set down on the street)
        for _ in range(500):
            self.robot.tick(PHYS_DT)
            mujoco.mj_step(m, d)
        d.time = 0.0
        for pr in self.props.values():
            pr["p0"] = {b: d.xpos[b].copy() for b in pr["bodies"]}    # where the prop and its slab lie once settled
        p, q = self.robot.pose()
        self.start_xy = np.array(p[:2])
        self._z_ground0 = float(self.sc.site.height_at(st["x"], st["y"]))
        self.odom.update(p[0], p[1], _quat_rpy(q)[2], 0.0)
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.calls = {}
        self.step_calls = 0
        self.illegal_calls = 0
        self.internal_errors = 0
        self.frames_used = 0
        self.frames_by_camera = {k: 0 for k in CAMERAS}
        self.lidar_used = 0
        self.thermal_used = 0
        self.render_wall_s = 0.0
        self.render_failures = 0
        self.frames_lost = 0
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.work_wall = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.held = -1
        self._arm_lin = None
        self.kit_latched = True
        self.slip_count = 0
        self._slip_ticks = 0
        self.violations = []
        self.voids = []
        self.stops = []
        self.lost = None
        self._surv_press = 0
        self._tilt_t = 0.0
        self.seen = []
        self.reports = []
        self.link_up = True
        self.link_dbm = 0.0
        self.collapsed = []
        self.traj = []
        self._traj_next = 0.0
        self.max_survivor_force = 0.0
        self.build_wall_s = time.time() - t0
        self._update_link()
        self._log_traj(force=True)
        return {"ok": True}

    def _index(self):
        m = self.m
        self.chassis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.gripper_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.kit_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "kit")
        self.pad = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_%s" % s) for s in ("fl", "fr")}
        # the robot's bodies: the chassis subtree
        robot_bodies = set()
        for b in range(m.nbody):
            r = b
            while r > 0 and r != self.chassis:
                r = int(m.body_parentid[r])
            if r == self.chassis:
                robot_bodies.add(b)
        self.robot_bodies = robot_bodies
        self.robot_geoms = set(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot_bodies)
        self.survivor_geoms = set(g for g in range(m.ngeom) if m.geom(g).name.startswith("survivor_"))
        site = self.sc.site
        self.graspable = {}
        self.props = {}
        self.debris_bodies = []
        for bd in site.bodies:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bd["name"])
            self.debris_bodies.append(bid)
            if bd.get("graspable"):
                self.graspable[bid] = bd
            if bd["kind"] == "prop":
                self.props[bid] = {"name": bd["name"], "bodies": [bid]}
        for bd in site.bodies:
            if bd["kind"] == "propslab":
                pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bd["held_by"])
                if pid in self.props:
                    self.props[pid]["bodies"].append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bd["name"]))
        self.graspable[self.kit_body] = {"name": "kit", "kind": "kit"}
        movers = set(self.robot_geoms)
        mover_bodies = set(self.debris_bodies + [self.kit_body])
        movers |= set(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in mover_bodies)
        self._movers = np.array(sorted(movers), np.int32)
        for pr in self.props.values():
            # a prop and the slab that rests on it touch each other all the time: what counts as a touch is anything
            # else that moves (the robot, the kit, other debris)
            own = set(pr["bodies"])
            pr["grp"] = np.array(sorted(own), np.int32)
            pr["ext"] = np.array(sorted(g for g in movers if int(m.geom_bodyid[g]) not in own), np.int32)
            pr["p0"] = {b: self.d.xpos[b].copy() for b in pr["bodies"]}
        self._surv_arr = np.array(sorted(self.survivor_geoms), np.int32)
        self._pad_arr = np.array([self.pad["fl"], self.pad["fr"]], np.int32)
        self._kit_geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.kit_body], np.int32)
        self.free_bodies = [b for b in range(m.nbody) if m.body_jntnum[b] > 0 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
        sp = site.survivor_points
        self.surv_torso = np.array(sp["torso"])
        self.surv_heads = np.array(sp["heads"])
        self.surv_hands = np.array(sp["hands"])
        # wall pieces for the link model (2D segments) and void footprints
        self._link_walls = []
        for st in site.statics:
            if st["type"] == "box" and st["name"].startswith("wall") and st["collide"]:
                c = np.array(st["pos"][:2])
                hx, hy, hz = st["size"]
                if st["pos"][2] + hz < 1.0:
                    continue
                if hx > hy:
                    a, b = c + [-hx, 0], c + [hx, 0]
                    ext = hy * 2 > 0.25
                else:
                    a, b = c + [0, -hy], c + [0, hy]
                    ext = hx * 2 > 0.25
                self._link_walls.append((a, b, ext))
        self._voids = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in self.plan["bays"].values() if b["state"] == "void"]

    def _temperatures(self):
        m = self.m
        env = self.plan["env"]
        Ta = env["ambient_c"]
        sun = env["sun"]
        rng = np.random.default_rng(self.seed & 0xFFFFFFFF)
        temps = np.full(m.ngeom, Ta, np.float32) + rng.normal(0.0, self.cfg["temps"]["spread_c"], m.ngeom).astype(np.float32)
        th = self.sc.site.thermal
        sv = self.plan["survivor"]
        for g in range(m.ngeom):
            nm = m.geom(g).name
            b = int(m.geom_bodyid[g])
            cls = th.get(nm)
            if g in self.robot_geoms or b == self.kit_body:
                temps[g] = Ta + 4.0
                continue
            if cls == "skin":
                temps[g] = sv["skin_c"] + float(rng.normal(0, 0.2))
            elif cls == "hair":
                temps[g] = sv["skin_c"] - 3.0
            elif cls == "cloth":
                temps[g] = sv["cloth_c"] + float(rng.normal(0, 0.4))
            elif cls == "boiler":
                d0 = [d for d in self.plan["decoys"] if d["type"] == "boiler"]
                temps[g] = d0[0]["temp_c"] if d0 else Ta
            elif cls == "sunlit_metal":
                temps[g] = Ta + 18.0 * sun
            elif nm == "ground":
                temps[g] = Ta + 1.0
            else:
                # surfaces out in the sun (street, facade outside, roofs) are warmer than the shaded interior
                pos = self.d.geom_xpos[g]
                if pos[1] < -0.2 or pos[2] > 3.4:
                    temps[g] = Ta + 7.0 * sun + float(rng.normal(0, 0.8))
        return temps

    # ------------------------------------------------------------------ the public calls
    def describe(self):
        C = CRAWLER
        return _jsonable({
            "cell_version": CELL_VERSION, "robot": "SR-2 rescue crawler",
            "control_dt_s": CTRL_DT, "sim_cap_s": self.cfg["sim_cap_s"], "max_ticks_per_step": self.cfg["max_ticks_per_step"],
            "budgets": {"camera_frames": self.cfg["camera_budget_frames"], "lidar_sweeps": self.cfg["lidar_budget_sweeps"],
                        "thermal_frames": self.cfg["thermal_budget_frames"]},
            "limits": {"speed_max": C["speed_max"], "yaw_rate_max": C["yaw_rate_max"], "gauge": [C["gauge_min"], C["gauge_max"]],
                       "flipper_rad": list(C["flipper_range"]), "mast_m": [0.0, C["mast_travel"]],
                       "head_pan_rad": list(C["head_pan_range"]), "head_tilt_rad": list(C["head_tilt_range"]),
                       "arm_rad": [list(r) for r in C["arm_range"]], "gripper_m": [0.0, C["grip_open_max"]],
                       "rates": {"flipper_rad_s": C["flipper_rate"], "gauge_m_s": C["gauge_rate"], "mast_m_s": C["mast_rate"],
                                 "head_rad_s": C["head_rate"], "arm_rad_s": C["arm_rate"], "gripper_m_s": C["grip_rate"]}},
            "cameras": {k: {"width": c["w"], "height": c["h"], "depth": bool(c.get("depth")),
                            "depth_range_m": list(c.get("range", (0, 0))), "intrinsics": self.sensors.intrinsics(k)}
                        for k, c in CAMERAS.items()},
            "thermal": {"width": THERMAL["w"], "height": THERMAL["h"], "intrinsics": self.sensors.intrinsics("thermal"),
                        "units": "centi-kelvin (uint16)", "netd_k": THERMAL["netd_k"]},
            "lidar": {"channels": LIDAR["channels"], "elevation_deg": [LIDAR["el_min"], LIDAR["el_max"]],
                      "azimuth_step_deg": LIDAR["az_step"], "range_m": list(LIDAR["range"])},
            "map_frame": "origin at the robot's position at deployment (on the ground under the chassis centre), x along its "
                         "initial heading, y to its left, z up",
        })

    def get_state(self):
        d = self.d
        p, q = self.robot.pose()
        roll, pitch, yaw = _quat_rpy(q)
        v = d.qvel[self.robot.vadr:self.robot.vadr + 6]
        R = d.xmat[self.chassis].reshape(3, 3)
        vb = R.T @ v[:3]
        wb = np.asarray(v[3:6], float)          # a free joint's angular velocity is already in the body frame
        ns = self.rng_state
        vb = vb + ns.normal(0.0, 0.01, 3)       # the velocity estimate's and the gyro's noise (robot_manual.md section 6)
        wb = wb + ns.normal(0.0, 0.005, 3)
        est = self.odom.est
        r = self.robot
        hold = self.held >= 0
        load = grasp.weld_load(self.m, self.d, "grip") if hold else None
        return _jsonable({
            "sim_time": float(d.time), "done": bool(self.done), "done_reason": self.done_reason,
            "pose": {"x": float(est[0]), "y": float(est[1]), "z": float(p[2] - self.start_z_ground() + ns.normal(0, 0.01)),
                     "yaw": float(est[2]),
                     "roll": float(roll + ns.normal(0, 0.002)), "pitch": float(pitch + ns.normal(0, 0.002))},
            "velocity": {"forward": float(vb[0]), "lateral": float(vb[1]), "yaw_rate": float(wb[2])},
            "imu": {"gyro": [float(x) for x in wb], "roll": float(roll + ns.normal(0, 0.002)),
                    "pitch": float(pitch + ns.normal(0, 0.002))},
            "joints": {"gauge": float(r.gauge()), "flipper_front": float((r.q("flip_FL") + r.q("flip_FR")) / 2),
                       "flipper_rear": float((r.q("flip_BL") + r.q("flip_BR")) / 2), "mast": float(r.q("mast")),
                       "head_pan": float(r.q("head_pan")), "head_tilt": float(r.q("head_tilt")),
                       "arm": [float(x) for x in r.arm_q()], "gripper": float(r.grip_width())},
            "commands": {"v": r.cmd["v"], "w": r.cmd["w"]},
            "tracks": {"left": float(r.belt["L"]), "right": float(r.belt["R"])},
            "gripper": {"holding": bool(hold), "load_N": float(load[0]) if load else 0.0},
            "arm_moving": bool(self._arm_lin is not None),
            "kit_in_cradle": bool(self.kit_latched),
            "link": {"up": bool(self.link_up), "rssi_dbm": float(round(self.link_dbm, 1))},
            "reports_pending": int(sum(1 for rr in self.reports if not rr["sent"])),
        })

    def start_z_ground(self):
        return float(getattr(self, "_z_ground0", 0.0))

    def _refuse(self):
        if self.done:
            return {"error": "episode_over"}
        if self.lost:
            return {"error": "robot_lost"}
        return None

    def drive(self, v=0.0, w=0.0):
        e = self._refuse()
        if e:
            return e
        C = CRAWLER
        self.robot.cmd["v"] = float(np.clip(float(v), -C["speed_max"], C["speed_max"]))
        self.robot.cmd["w"] = float(np.clip(float(w), -C["yaw_rate_max"], C["yaw_rate_max"]))
        return {"ok": True}

    def set_flippers(self, front=None, rear=None):
        e = self._refuse()
        if e:
            return e
        lo, hi = CRAWLER["flipper_range"]
        if front is not None:
            self.robot.cmd["flip_F"] = float(np.clip(float(front), lo, hi))
        if rear is not None:
            self.robot.cmd["flip_B"] = float(np.clip(float(rear), lo, hi))
        return {"ok": True}

    def set_gauge(self, width):
        e = self._refuse()
        if e:
            return e
        self.robot.cmd["gauge"] = float(np.clip(float(width), CRAWLER["gauge_min"], CRAWLER["gauge_max"]))
        return {"ok": True}

    def set_mast(self, height):
        e = self._refuse()
        if e:
            return e
        self.robot.cmd["mast"] = float(np.clip(float(height), 0.0, CRAWLER["mast_travel"]))
        return {"ok": True}

    def set_head(self, pan=None, tilt=None):
        e = self._refuse()
        if e:
            return e
        if pan is not None:
            self.robot.cmd["pan"] = float(np.clip(float(pan), *CRAWLER["head_pan_range"]))
        if tilt is not None:
            self.robot.cmd["tilt"] = float(np.clip(float(tilt), *CRAWLER["head_tilt_range"]))
        return {"ok": True}

    def set_arm(self, q):
        e = self._refuse()
        if e:
            return e
        q = [float(x) for x in q]
        if len(q) != 5:
            return {"error": "bad_params", "detail": "q must have 5 joint angles"}
        out = []
        for i, (lo, hi) in enumerate(CRAWLER["arm_range"]):
            out.append(float(np.clip(q[i], lo, hi)))
        self._arm_lin = None
        self.robot.cmd["arm"] = out
        return {"ok": True, "q": out}

    def arm_to(self, p, pitch=-math.pi / 2, roll=None, elbow_up=True, speed=0.15):
        """A straight-line move of the tool centre point to chassis-frame point p with the tool pitched `pitch` from
        the horizontal and the wrist rolled `roll` (the on-board Cartesian mode: the controller interpolates the TCP
        along the line at `speed` m/s and solves the arm's inverse kinematics every control tick)."""
        e = self._refuse()
        if e:
            return e
        p = [float(v) for v in p]
        if len(p) != 3:
            return {"error": "bad_params", "detail": "p must have 3 coordinates"}
        q_now = self.robot.arm_q()
        roll = float(q_now[4] if roll is None else roll)
        q_ref = list(q_now)
        q_ref[4] = roll
        q = arm_ik([float(v) for v in p], float(pitch), q_ref=q_ref, elbow_up=bool(elbow_up))
        if q is None:
            return {"ok": False, "error": "unreachable"}
        p0, _dir = arm_fk(q_now)
        pitch0 = float(q_now[1] + q_now[2] + q_now[3])
        dist = float(np.linalg.norm(np.asarray(p, float) - p0))
        ang = max(abs(_wrap(float(pitch) - pitch0)), abs(_wrap(roll - float(q_now[4]))))
        dur = max(dist / max(0.02, min(0.3, float(speed))), ang / CRAWLER["arm_rate"], 0.05)
        self._arm_lin = {"p0": p0, "p1": np.asarray(p, float), "pitch0": pitch0, "pitch1": float(pitch), "roll0": float(q_now[4]),
                         "roll1": roll, "t0": float(self.d.time), "dur": dur, "elbow_up": bool(elbow_up), "q1": q}
        return {"ok": True, "q": [float(x) for x in q], "duration_s": dur}

    def gripper(self, width):
        e = self._refuse()
        if e:
            return e
        self.robot.cmd["grip"] = float(np.clip(float(width), 0.0, CRAWLER["grip_open_max"]))
        return {"ok": True}

    def step(self, n_ticks=1):
        if isinstance(n_ticks, bool) or not isinstance(n_ticks, (int, float)) or not math.isfinite(n_ticks):
            return {"error": "bad_params", "detail": "n_ticks must be a finite number"}
        n = int(n_ticks)
        if n < 1:
            return {"error": "bad_params", "detail": "n_ticks must be >= 1"}
        n = min(n, int(self.cfg["max_ticks_per_step"]))
        self.step_calls += 1
        t = time.time()
        events = []
        try:
            for _ in range(n):
                if self.done:
                    break
                events += self.physics_tick()
                if self.d.time >= self.cfg["sim_cap_s"] - 1e-9:
                    self._end("sim_cap")
        finally:
            self.work_wall += time.time() - t
        ev = [e for e in events if not e.startswith("_")]
        if self.done:
            ev.append("episode_over")
        if os.environ.get("RUBBLE_PROGRESS") and self.d.time >= getattr(self, "_prog_next", 0.0):
            self._prog_next = float(self.d.time) + 30.0
            p, _q = self.robot.pose()
            self.log("[progress] sim %.0f s wall %.0f s ncon %d robot (%.2f, %.2f, %.2f)" % (
                self.d.time, time.time() - self.wall_start, self.d.ncon, p[0], p[1], p[2]))
        return {"sim_time": float(self.d.time), "done": bool(self.done), "events": ev}

    def _budget(self, kind):
        used = {"camera": self.frames_used, "lidar": self.lidar_used, "thermal": self.thermal_used}[kind]
        cap = {"camera": self.cfg["camera_budget_frames"], "lidar": self.cfg["lidar_budget_sweeps"],
               "thermal": self.cfg["thermal_budget_frames"]}[kind]
        return used < int(cap)

    def get_camera(self, name="mast", depth=True):
        name = str(name)
        if name not in CAMERAS:
            return {"error": "bad_params", "detail": "unknown camera (cameras: %s)" % ", ".join(CAMERAS)}
        if self.lost:
            return {"error": "robot_lost"}
        if not self._budget("camera"):
            return {"error": "camera_budget"}
        t = time.time()
        try:
            try:
                rgb, dep, K, T = self.sensors.camera(name, depth=bool(depth))
                npx = self.sensors.survivor_pixels(name)
            except Exception as e:  # noqa: BLE001
                self.render_failures += 1
                self.frames_lost += 1
                self.log("[cell] %s frame not rendered: %s\n%s" % (name, type(e).__name__, traceback.format_exc()))
                return {"error": "camera_unavailable"}
            self.frames_used += 1
            self.frames_by_camera[name] += 1
            self._record_seen(name, npx)
            out = {"ok": True, "camera": name, "sim_time": float(self.d.time), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                   "rgb_png_b64": _b64png(rgb), "intrinsics": K, "T_chassis_cam": T.tolist()}
            if dep is not None:
                mm = np.where(np.isfinite(dep), np.clip(np.round(dep * 1000.0), 1, 65535), 0).astype(np.uint16)
                out["depth_u16_mm_b64"] = _b64png(mm)
                out["depth_unit_m"] = 0.001
                out["depth_invalid_value"] = 0
            return out
        finally:
            dt = time.time() - t
            self.render_wall_s += dt
            self.work_wall += dt

    def get_lidar(self):
        if self.lost:
            return {"error": "robot_lost"}
        if not self._budget("lidar"):
            return {"error": "lidar_budget"}
        t = time.time()
        try:
            try:
                pts, T = self.sensors.lidar()
            except Exception as e:  # noqa: BLE001
                self.render_failures += 1
                self.frames_lost += 1
                self.log("[cell] lidar sweep not rendered: %s" % type(e).__name__)
                return {"error": "lidar_unavailable"}
            self.lidar_used += 1
            return {"ok": True, "sim_time": float(self.d.time), "n": int(len(pts)),
                    "points_f32_b64": base64.b64encode(pts.astype("<f4").tobytes()).decode("ascii"),
                    "fields": ["x", "y", "z", "channel"], "T_chassis_lidar": T.tolist()}
        finally:
            dt = time.time() - t
            self.render_wall_s += dt
            self.work_wall += dt

    def get_thermal(self):
        if self.lost:
            return {"error": "robot_lost"}
        if not self._budget("thermal"):
            return {"error": "thermal_budget"}
        t = time.time()
        try:
            try:
                ck, npx = self.sensors.thermal()
            except Exception as e:  # noqa: BLE001
                self.render_failures += 1
                self.frames_lost += 1
                self.log("[cell] thermal frame not rendered: %s\n%s" % (type(e).__name__, traceback.format_exc()))
                return {"error": "thermal_unavailable"}
            self.thermal_used += 1
            self._record_seen("thermal", npx)
            cid = self.sensors.thermal_cam
            return {"ok": True, "sim_time": float(self.d.time), "width": int(ck.shape[1]), "height": int(ck.shape[0]),
                    "centikelvin_u16_png_b64": _b64png(ck), "intrinsics": self.sensors.intrinsics("thermal"),
                    "T_chassis_cam": self.sensors.cam_extrinsics(cid).tolist()}
        finally:
            dt = time.time() - t
            self.render_wall_s += dt
            self.work_wall += dt

    def report_survivor(self, x, y, z=0.0):
        if self.done:
            return {"error": "episode_over"}
        rep = {"t": float(self.d.time), "x": float(x), "y": float(y), "z": float(z), "sent": bool(self.link_up),
               "sent_t": float(self.d.time) if self.link_up else None}
        self.reports.append(rep)
        self._event("report:%s" % ("sent" if rep["sent"] else "queued"))
        return {"ok": True, "sent": rep["sent"], "note": "the team holds your last report that reached it over the link"}

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.d.time)}

    # ------------------------------------------------------------------ privileged / practice
    def get_state_privileged(self):
        return self.truth()

    def truth(self):
        p, q = self.robot.pose()
        return _jsonable({"sim_time": float(self.d.time), "robot": {"pos": p, "quat": q, "rpy": _quat_rpy(q)},
                          "start": self.plan["start"], "survivor": {"torso": self.surv_torso, "heads": self.surv_heads,
                                                                   "hands": self.surv_hands},
                          "route": self.sc.site.route, "odom": self.odom.est, "score_so_far": self.score()})

    def reference_run(self):
        if not self.reference_dir:
            return {"error": "no reference program"}
        path = os.path.join(self.reference_dir, "reference.py")
        spec = importlib.util.spec_from_file_location("rubblecell_reference", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        result = mod.run(self)
        return {"ok": True, "wall_s": time.time() - t0, "sim_time": float(self.d.time), "result": _jsonable(result)}

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        self.calls[method] = self.calls.get(method, 0) + 1
        try:
            return self._dispatch(method, params)
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
            self.internal_errors += 1
            self.log("[cell] internal error in %s\n%s" % (method, traceback.format_exc()))
            if method in STATE_METHODS:
                self._end("internal_error")
            return {"error": "internal_error"}

    def _dispatch(self, method, params):
        if method in PRACTICE_METHODS:
            if not (self.mode == "practice" and (self.allow_reset or method == "truth")):
                self.illegal_calls += 1
                return {"error": "forbidden"}
            if method == "reset":
                self.reset(int((params or {}).get("seed", 0)), (params or {}).get("kind", "training"))
                return {"ok": True}
            return self.truth()
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        if params is not None and not isinstance(params, dict):
            return {"error": "bad_params", "detail": "params must be an object"}
        if not _finite(params):
            return {"error": "bad_params", "detail": "every number must be finite"}
        try:
            return getattr(self, method)(**(params or {}))
        except (TypeError, ValueError, KeyError, IndexError, OverflowError, ArithmeticError) as e:
            return {"error": "bad_params", "detail": str(e)[:200]}
        except Exception as e:  # noqa: BLE001
            # a command that only sets a target does nothing but read its parameters: whatever it raises is the
            # request's doing (never a cell failure, which the grader answers by running the site again)
            if method in COMMAND_METHODS:
                return {"error": "bad_params", "detail": type(e).__name__}
            raise

    # ------------------------------------------------------------------ the tick
    def _event(self, e):
        self.events_log.append((float(self.d.time), e))

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = reason
            self._log_traj(force=True)

    def physics_tick(self):
        m, d = self.m, self.d
        ev = []
        if self.lost:
            self.robot.cmd["v"] = 0.0
            self.robot.cmd["w"] = 0.0
        if self._arm_lin is not None:
            self._advance_arm()
        n = int(round(CTRL_DT / PHYS_DT))
        for _ in range(n):
            self.robot.tick(PHYS_DT)
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or np.abs(d.qvel).max() > 1e3:
            j = int(np.argmax(np.abs(np.nan_to_num(d.qvel, nan=1e9))))
            b = int(m.dof_bodyid[j])
            self.log("[cell] physics fault at t=%.2f: dof %d of body %s, |qvel| %.1f" % (d.time, j, m.body(b).name, abs(d.qvel[j])))
            self._event("physics_fault")
            self._end("physics_fault")
            return ["physics_fault"]
        self.tick_count += 1
        ev += self._grip_rule()
        ev += self._watch()
        p, q = self.robot.pose()
        self.odom.update(p[0], p[1], _quat_rpy(q)[2], CTRL_DT)
        if self.tick_count % 10 == 0:
            ev += self._update_link()
        self._log_traj()
        for e in ev:
            self._event(e)
        return ev

    def _advance_arm(self):
        L = self._arm_lin
        u = min(1.0, (float(self.d.time) + CTRL_DT - L["t0"]) / L["dur"])
        u = u * u * (3.0 - 2.0 * u)
        p = L["p0"] + u * (L["p1"] - L["p0"])
        pitch = L["pitch0"] + u * _wrap(L["pitch1"] - L["pitch0"])
        roll = L["roll0"] + u * _wrap(L["roll1"] - L["roll0"])
        q_ref = list(self.robot.cmd["arm"])
        q_ref[4] = roll
        q = arm_ik(list(p), pitch, q_ref=q_ref, elbow_up=L["elbow_up"])
        if q is None:
            q = L["q1"] if u >= 1.0 else None
        if q is not None:
            self.robot.cmd["arm"] = [float(x) for x in q]
            self.robot.tgt["arm"] = [float(x) for x in q]
        if u >= 1.0:
            self.robot.cmd["arm"] = [float(x) for x in L["q1"]]
            self._arm_lin = None

    def _contacts_between(self, set_a, body_b=None, geom_b=None):
        """Total normal force of the contacts between geoms of set_a and body body_b / geom geom_b."""
        m, d = self.m, self.d
        f = np.zeros(6)
        tot = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for ga, gb in ((g1, g2), (g2, g1)):
                if ga in set_a and ((body_b is not None and int(m.geom_bodyid[gb]) == body_b) or (geom_b is not None and gb == geom_b)):
                    mujoco.mj_contactForce(m, d, i, f)
                    tot += abs(f[0])
        return tot

    def _grip_rule(self):
        m, d = self.m, self.d
        r = self.robot
        G = self.cfg["grip"]
        ev = []
        width = r.grip_width()
        closing = r.cmd["grip"] < width - 0.002
        if self.held < 0:
            if not closing:
                return ev
            touch = {}
            if not d.ncon:
                return ev
            g1 = d.contact.geom1[:d.ncon]
            g2 = d.contact.geom2[:d.ncon]
            idx = np.nonzero(np.isin(g1, self._pad_arr) | np.isin(g2, self._pad_arr))[0]
            f = np.zeros(6)
            for i in idx:
                a, b2 = int(g1[i]), int(g2[i])
                for ga, gb in ((a, b2), (b2, a)):
                    for side, pg in self.pad.items():
                        if ga == pg:
                            b = int(m.geom_bodyid[gb])
                            mujoco.mj_contactForce(m, d, int(i), f)
                            touch.setdefault(b, {}).setdefault(side, 0.0)
                            touch[b][side] += abs(f[0])
            for b, sides in touch.items():
                if len(sides) == 2 and min(sides.values()) >= G["min_force_N"] and b in self.graspable:
                    mass = float(m.body_subtreemass[b])
                    if mass > G["payload_kg"]:
                        ev.append("grip_slip")
                        continue
                    if b == self.kit_body and self.kit_latched:
                        grasp.release(m, d, "kit_latch")
                        self.kit_latched = False
                        ev.append("kit_unlatched")
                    grasp.attach(m, d, "grip", b)
                    self.held = b
                    self._slip_ticks = 0
                    ev.append("grasped")
                    break
        else:
            if r.cmd["grip"] > width + 0.008:
                grasp.release(m, d, "grip")
                self.held = -1
                ev.append("released")
                return ev
            load = grasp.weld_load(m, d, "grip")
            if load and load[0] > G["slip_force_N"]:
                self._slip_ticks += 1
                if self._slip_ticks >= G["slip_ticks"]:
                    grasp.release(m, d, "grip")
                    self.held = -1
                    self.slip_count += 1
                    ev.append("slipped")
            else:
                self._slip_ticks = 0
        return ev

    def _watch(self):
        m, d = self.m, self.d
        S = self.cfg["safety"]
        ev = []
        # the survivor: the robot, the kit or debris pressing on them
        press = 0.0
        if d.ncon:
            g1 = d.contact.geom1[:d.ncon]
            g2 = d.contact.geom2[:d.ncon]
            hit = (np.isin(g1, self._surv_arr) & np.isin(g2, self._movers)) | (np.isin(g2, self._surv_arr) & np.isin(g1, self._movers))
            if hit.any():
                f = np.zeros(6)
                for i in np.nonzero(hit)[0]:
                    mujoco.mj_contactForce(m, d, int(i), f)
                    press += abs(f[0])
        self.max_survivor_force = max(self.max_survivor_force, press)
        if press > S["survivor_force_N"]:
            self._surv_press += 1
            if self._surv_press >= S["survivor_ticks"] and not self.stops:
                # the robot's protective stop: contact with a person above the limit stops every drive for good. The
                # mission ends here; it keeps what the team already has, without the return and under a cap (score)
                self.stops.append({"t": float(d.time), "what": "protective_stop", "force_N": float(press)})
                self.robot.cmd["v"] = 0.0
                self.robot.cmd["w"] = 0.0
                ev.append("protective_stop")
                self._end("protective_stop")
        else:
            self._surv_press = 0
        # load-bearing props: a crate that carries one end of a slab (free bodies: the load is real). Once the robot,
        # what it holds or other debris has touched the crate or its slab, moving either by more than prop_move_m is
        # a secondary collapse; before that, the group settling on its own is not the robot's doing.
        for pid, pr in self.props.items():
            if pr.get("fallen"):
                continue
            if not pr.get("touched"):
                if self.held in pr["bodies"]:
                    pr["touched"] = True
                elif d.ncon:
                    g1 = d.contact.geom1[:d.ncon]
                    g2 = d.contact.geom2[:d.ncon]
                    b1 = m.geom_bodyid[g1]
                    b2 = m.geom_bodyid[g2]
                    hit = (np.isin(b1, pr["grp"]) & np.isin(g2, pr["ext"])) | (np.isin(b2, pr["grp"]) & np.isin(g1, pr["ext"]))
                    if hit.any():
                        pr["touched"] = True
                if not pr.get("touched"):
                    pr["p0"] = {b: d.xpos[b].copy() for b in pr["bodies"]}
                    continue
            if max(float(np.linalg.norm(d.xpos[b] - pr["p0"][b])) for b in pr["bodies"]) > S["prop_move_m"]:
                pr["fallen"] = True
                self.collapsed.append({"t": float(d.time), "prop": pr["name"]})
                self.voids.append({"t": float(d.time), "what": "secondary_collapse", "prop": pr["name"]})
                ev.append("collapse")
                # the team aborts the mission after a secondary collapse
                self._end("secondary_collapse")
        # the robot: overturned or fallen into a void
        if not self.lost:
            R = d.xmat[self.chassis].reshape(3, 3)
            if R[2, 2] < math.cos(math.radians(S["overturn_deg"])):
                self._tilt_t += CTRL_DT
            else:
                self._tilt_t = 0.0
            if self._tilt_t >= S["overturn_s"]:
                self.lost = "overturned"
                ev.append("robot_overturned")
                self._end("robot_overturned")
            elif d.xpos[self.chassis][2] < S["fall_z_m"]:
                self.lost = "fell"
                ev.append("robot_fell")
                self._end("robot_fell")
        return ev

    def _update_link(self):
        """Path loss from the base station (1.5 m over the deployment point) to the mast's antenna."""
        L = self.cfg["link"]
        p, _ = self.robot.pose()
        a = np.array(self.start_xy if hasattr(self, "start_xy") else p[:2])
        b = np.array(p[:2])
        dist = max(1.0, float(np.linalg.norm(b - a)))
        loss = 20.0 * math.log10(dist)
        for (w0, w1, ext) in self._link_walls:
            if _seg_x(a, b, w0, w1):
                loss += L["ext_wall_db"] if ext else L["int_wall_db"]
        for (x0, y0, x1, y1) in self._voids:
            if x0 <= b[0] <= x1 and y0 <= b[1] <= y1:
                loss += L["void_db"]
            elif _seg_box(a, b, x0, y0, x1, y1):
                loss += L["slab_db"]
        dbm = L["tx_dbm"] - loss
        was = self.link_up
        self.link_dbm = dbm
        self.link_up = dbm >= L["floor_dbm"]
        ev = []
        if self.link_up != was:
            ev.append("link_up" if self.link_up else "link_down")
        if self.link_up:
            for rep in self.reports:
                if not rep["sent"]:
                    rep["sent"] = True
                    rep["sent_t"] = float(self.d.time)
                    ev.append("report_sent")
        return ev

    def _record_seen(self, cam, npx):
        if npx <= 0:
            return
        cid = self.sensors.cam_ids.get(cam, self.sensors.thermal_cam)
        dist = float(np.linalg.norm(self.d.cam_xpos[cid] - self.surv_torso))
        self.seen.append({"t": float(self.d.time), "camera": cam, "px": int(npx), "dist_m": dist})

    def _log_traj(self, force=False):
        hz = float(self.cfg.get("trajectory_hz", 4.0))
        if hz <= 0:
            return
        if not force and self.d.time < self._traj_next:
            return
        self._traj_next = float(self.d.time) + 1.0 / hz
        self.traj.append(np.concatenate([[self.d.time], self.d.qpos.copy(), self.d.eq_active.astype(np.float64)]))

    # ------------------------------------------------------------------ scoring
    def score(self):
        S = self.cfg["score"]
        sent = [r for r in self.reports if r["sent"]]
        located = 0.0
        err = None
        seen_before = False
        if sent:
            rep = sent[-1]
            xw, yw = self.odom.to_world(rep["x"], rep["y"])
            err = float(math.hypot(xw - self.surv_torso[0], yw - self.surv_torso[1]))
            seen_before = any(s["t"] <= rep["t"] + 1e-6 and s["px"] >= S["seen_px"] and s["dist_m"] <= S["seen_range_m"]
                              for s in self.seen)
            if seen_before:
                if err <= S["report_full_m"]:
                    located = 1.0
                elif err < S["report_zero_m"]:
                    located = (S["report_zero_m"] - err) / (S["report_zero_m"] - S["report_full_m"])
        # the kit: at rest near the survivor's head or a hand, not held, not lying on them
        kit_credit = 0.0
        kp = self.d.xpos[self.kit_body]
        kit_dist = None
        if not self.kit_latched and self.held != self.kit_body:
            pts = np.vstack([self.surv_heads, self.surv_hands])
            dd = np.hypot(pts[:, 0] - kp[0], pts[:, 1] - kp[1])
            kit_dist = float(dd.min())
            on_body = self._contacts_between(self.survivor_geoms, body_b=self.kit_body) > 2.0
            vel = float(np.linalg.norm(self.d.cvel[self.kit_body]))
            if kit_dist <= S["kit_reach_m"] and not on_body and abs(kp[2] - 0.15) < 0.6 and vel < 0.5:
                kit_credit = 1.0
        p, q = self.robot.pose()
        home = float(np.linalg.norm(np.array(p[:2]) - self.start_xy))
        upright = self.d.xmat[self.chassis].reshape(3, 3)[2, 2] > 0.5
        stopped = len(self.stops) > 0
        returned = 1.0 if (not self.lost and not stopped and upright and home <= S["home_radius_m"] and located > 0.0) else 0.0
        raw = S["located"] * located + S["kit"] * kit_credit + S["returned"] * returned
        if stopped:
            # a protective stop on the survivor: what the team already had, at most stop_score_cap
            raw = min(raw, float(self.cfg["safety"].get("stop_score_cap", 0.45)))
        voided = len(self.voids) > 0
        return {"score": 0.0 if voided else float(raw), "raw_score": float(raw), "voided": voided, "voids": list(self.voids),
                "parts": {"located": located, "kit": kit_credit, "returned": returned}, "stops": list(self.stops),
                "report_error_m": err,
                "seen_before_report": bool(seen_before), "kit_dist_m": kit_dist, "home_dist_m": home,
                "reports": len(self.reports), "reports_sent": len(sent)}

    def finalize(self):
        if not self.done:
            self._end("solve_returned")
        s = self.score()
        seen_best = max((x["px"] for x in self.seen), default=0)
        summary = {
            "cell_version": CELL_VERSION, "seed": self.seed, "kind": self.kind, "done_reason": self.done_reason,
            "sim_time_s": float(self.d.time), "ticks": int(self.tick_count), "episode_score": s["score"], "raw_score": s["raw_score"],
            "voided": s["voided"], "voids": s["voids"], "stops": s["stops"], "parts": s["parts"],
            "report_error_m": s["report_error_m"],
            "seen_before_report": s["seen_before_report"], "seen_frames": len(self.seen), "seen_best_px": int(seen_best),
            "kit_dist_m": s["kit_dist_m"], "home_dist_m": s["home_dist_m"], "reports": s["reports"], "reports_sent": s["reports_sent"],
            "lost": self.lost, "collapses": list(self.collapsed), "max_survivor_force_N": float(self.max_survivor_force),
            "slips": int(self.slip_count), "illegal_call_count": int(self.illegal_calls), "internal_errors": int(self.internal_errors),
            "calls": dict(self.calls), "step_calls": int(self.step_calls), "program_time_s": float(self.program_used),
            "cell_work_s": float(self.work_wall), "camera_frames_total": int(self.frames_used), "camera_frames": dict(self.frames_by_camera),
            "lidar_sweeps": int(self.lidar_used), "thermal_frames": int(self.thermal_used), "render_wall_s": float(self.render_wall_s),
            "render_failures": int(self.render_failures), "frames_lost": int(self.frames_lost), "build_wall_s": float(self.build_wall_s),
            "events_log": [[round(t, 2), e] for t, e in self.events_log][:3000],
            "route_bays": [list(k) for k in self.plan["route_bays"]], "target": list(self.plan["target"]),
        }
        return _jsonable(summary)

    def trajectory(self):
        """The episode's state samples (time, qpos, eq_active) for the replay renderer, float32 (n, k)."""
        if not self.traj:
            return np.zeros((0, 0), np.float32)
        return np.vstack(self.traj).astype(np.float32)


def _seg_x(a, b, c, d):
    """True when segments ab and cd cross."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _seg_box(a, b, x0, y0, x1, y1):
    """True when segment ab passes through the box."""
    for t in np.linspace(0.0, 1.0, 9):
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False
