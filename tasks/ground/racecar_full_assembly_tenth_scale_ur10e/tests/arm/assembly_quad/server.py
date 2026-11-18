"""The quadruped assembly cell behind the camera-only API (`QuadInfoServer`).

The physics is `cell.QuadCell`: the robot's body bolted on its stand, the 16 loose units (four shoulders, four hip AA
supports, four upper and four lower legs) resting on the bench where the UR10e reaches them, the coupling state machine
(`coupling.py`: a unit on its feature is held by a soft coupling and pulled home, then welded; fastened joints stay free
against the unpowered actuators' friction). The UR10e is driven by `ur_controller.URController` (computed torque before
every physics step), the 2F-85 by its width command. The observation side is the information tier's
(`arm.assembly_info.cell_info.InfoCellServer`, inherited): uncalibrated RGB-D cameras (top, wrist) through the render
layer, a noisy flange F/T sensor, the joint state; no part poses, no mate events.

`QuadInfoServer.__init__` calls no parent constructor (all three above it load the six-servo arm's kit); the inherited
methods read the attributes set here and in `reset` / `_init_state`. The bottom layer of the safety watch is
`_QuadBaseWatch._watch` (the MRO puts it under the six-servo damage layer and the information tier's sustained-force
layer): robot-neutral names and every unit's speed / height at its centre of mass (the supports' body origins lie
0.15-0.17 m below the bench while they rest on it).

Per episode (`layout.episode_layout`): the stand +-10 mm, the 16 units around the hidden bench plan (turned and moved,
footprints from their collision vertices), the output pulleys at their index +-2 deg; the built cell must pass the
reach check (every shoulder / support / upper leg keeps a grasp that passes the reference's pick checks there, every
lower leg the knee's place check of one of its upper leg's grasps) or the layout is drawn again. With the oracle token
(privileged mode) and `oracle_nominal_layout` the layout is the plan itself and `reference_assemble()` runs the
reference program (`reference.program`) inside the cell, through the same routed ticks and safety watch.

Score of an episode (`finalize`): `stage_score` = the locked slots' weight / the total weight (22.0): a shoulder in the
design key orientation, a hip FE or a knee (either orientation) 1.5 each, a support 1.0 (it needs its shoulder).
"""

import json
import math
import time

import numpy as np

from ..assembly_cell.server import CellServer as _BaseCellServer
from ..assembly_cell.server import _deep_update
from ..assembly_info import layout_info as LI
from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG, InfoCellServer
from . import layout as LY
from .cell import CELL_CONFIG, QuadCell, load_kit
from .coupling import weld_name

CELL_VERSION = "quadruped-assembly-cell/1.0.0"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged", "reference_assemble")

_MOUNT_TEXT = {
    "top": "overhead, roughly 1.6 m above the bench near the middle of the working area, looking straight down; the image covers "
           "the bench, the stand with the body and the robot's base; image width roughly along the cell's x",
    "wrist": "on the gripper's coupling, about 6 cm off the tool axis on the side across from the pads (the tool's +x), a few cm out "
             "from the flange, tilted about 16 deg towards the fingertips; the fingers and a held unit are in the lower part of the image",
}

# the given kit.json's unit ids per kit unit type (the ids its parts/*.stl files carry): describe()'s key heights use them
GIVEN_UNIT_IDS = {"base": "body", "S_plain": "shoulder_plain", "S_adapter": "shoulder_adapter", "A": "support", "U": "upper_leg",
                  "L": "lower_leg"}
# coupling.CouplingStateMachine._inside: a captured key stays captured until its root is the key's height + 2 mm out of the
# socket, i.e. until its tip is 2 mm clear of the pulley face
KEY_RELEASE_CLEAR_MM = 2.0
# the sockets' entry window with the lead-in (kit core "socket_lead_in_mm": the drawing's R1.2 lead-in as a 45 deg chamfer
# on both pulley faces), measured in this cell: the arm pushing the held unit in at stiffness 0.5 and 10 mm/s along the
# insertion, backing off 15 mm and retrying after a stall (4 mm behind its command for 0.3 s), the pulley turned up to 4 deg
# off its index with the hand aiming at the index, the key off across the socket's short side; the misalignment the key had
# as it reached the pulley face, captured and fastened (slots haa_FL, haa_FR, hfe_S1, kfe_U1 of the nominal cell: every key)
ENTRY_WINDOW_TEXT = ("pushed in by the arm at stiffness 0.5 and 10 mm/s (backing off to retry after a stall), with the pulley up to 4 deg "
                     "off its index and the hand aiming at the index, a key went in when it reached the pulley face at most this far "
                     "off across the socket's short side: a plain shoulder's 0.4 mm within 1 deg of turn (0.1 mm at 2-3 deg, none at "
                     "4 deg), a hip adapter shoulder's only about the 0.3 mm play within 1 deg (a quarter of its 5 mm key lies deeper "
                     "than the lead-in), an upper leg's 1.5 mm within 3 deg (1 mm at 4 deg), and a lower leg lying loose on the bench "
                     "took a knee socket lowered onto it from 1.5 mm and 4 deg, the most tried; farther off the key stops on the "
                     "chamfer or the pulley face")

# the documented datums (the cell manual's survey; exact by construction)
BENCH_BOARD = {"cols": 9, "rows": 6, "square_mm": 20.0, "pos": [1.20, -0.50, 0.0], "yaw_deg": 0.0}
STAND_BOARD = {"cols": 5, "rows": 3, "square_mm": 10.0, "margin_mm": 4.0, "thickness_mm": 2.0}
TOOL_PLATE = {"cols": 5, "rows": 4, "square_mm": 6.0, "margin_mm": 3.0, "thickness_mm": 1.5}

DEFAULT_QUAD_CONFIG = {
    "sim_cap_s": 1200.0,
    "wall_cap_s": 2400.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 600,
    "settle_s": 1.0,
    "safety": {"robot_force_N": 30.0, "part_speed_m_s": 3.0},
    "held": {"force_N": 1.0, "ticks": 5},
    "drop": {"speed_m_s": 0.25, "watch_ticks": 50},
    "lost_z_m": -0.05,
    "damage": json.loads(json.dumps(DEFAULT_INFO_CONFIG["damage"])),
    "photo": json.loads(json.dumps(DEFAULT_INFO_CONFIG["photo"])),
    "info_layout": {},
    "agent_cameras": ["top", "wrist"],
    # the agent's cameras: "filament" (agent_render.py: physically based, the baked factory hall) or "mujoco"
    # (render_twin.py: MuJoCo's classic renderer)
    "agent_renderer": "mujoco",
    "robot": {"model": "ur10e", "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
    "layout": json.loads(json.dumps(LY.DEFAULT_LAYOUT)),
    "reach_check": {"enabled": True, "max_draws": 8},
    "oracle_nominal_layout": True,
    "privileged_wall_cap_s": 7200.0,
}


class _QuadBaseWatch(_BaseCellServer):
    """The bottom layer of the safety watch (replaces assembly_cell.server.CellServer._watch): a robot link other than
    the four pad boxes pressing on a unit or the environment above robot_force_N; both pads on one unit (held); every
    loose unit's speed and height at its centre of mass (part speed, lost), the drop after a release."""

    def _watch(self, d):
        m = self.model
        mj = self.mujoco
        events = []
        thr_f = float(self.cfg["safety"]["robot_force_N"])
        left, right = {}, {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if c1 == 0 and c2 == 0:
                continue
            mj.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            if (c1 == 1 and c2 in (4, 5)) or (c2 == 1 and c1 in (4, 5)):
                if fn > thr_f:
                    other = con.geom2 if c1 == 1 else con.geom1
                    link = con.geom1 if c1 == 1 else con.geom2
                    what = "robot_contact:%s:%s:%.0fN" % (m.body(m.geom_bodyid[link]).name, m.body(m.geom_bodyid[other]).name, fn)
                    if not self.violations or self.violations[-1]["t"] < d.time - 0.5 or self.violations[-1]["what"].split(":")[1:3] != what.split(":")[1:3]:
                        self.violations.append({"t": float(d.time), "what": what, "force_N": fn})
                        events.append("safety:" + what)
            for cls, g in ((c1, con.geom2), (c2, con.geom1)):
                if cls in (2, 3) and self.geom_class[g] == 4:
                    part = self.geom_part[g]
                    (left if cls == 2 else right)[part] = (left if cls == 2 else right).get(part, 0.0) + fn
        thr_h = self.cfg["held"]["force_N"]
        cand, best = None, 0.0
        for part in left:
            if part in right and left[part] >= thr_h and right[part] >= thr_h:
                s = min(left[part], right[part])
                if s > best:
                    cand, best = part, s
        if cand is not None and cand == self._held_cand:
            self._held_ticks += 1
        else:
            self._held_cand = cand
            self._held_ticks = 1 if cand is not None else 0
        new_held = self._held_cand if self._held_ticks >= self.cfg["held"]["ticks"] else None
        if new_held != self.held:
            if new_held is not None:
                self.regrasps += 1
                events.append("grasped:" + new_held)
            else:
                events.append("released:" + self.held)
                self._drop_watch[self.held] = {"t0": self.tick_count, "vmax": 0.0}
            self.held = new_held
        thr_v = float(self.cfg["safety"]["part_speed_m_s"])
        v6 = self._v6
        for n in self.meta["loose"]:
            b = self.part_body[n]
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, v6, 0)        # at the body's centre of mass (xipos)
            v = float(np.linalg.norm(v6[3:]))
            z = float(d.xipos[b][2])
            if v > self._v_peak.get(n, 0.0):
                self._v_peak[n] = v
            if v > thr_v:
                what = "part_speed:%s:%.1fm/s" % (n, v)
                if not any(vi["what"].startswith("part_speed:" + n) for vi in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": what, "speed": v})
                    events.append("safety:" + what)
            if z < self.cfg["lost_z_m"] and n not in self._lost:
                self._lost.add(n)
                events.append("lost:" + n)
            w = self._drop_watch.get(n)
            if w is not None:
                w["vmax"] = max(w["vmax"], v)
                if self.tick_count - w["t0"] >= self.cfg["drop"]["watch_ticks"]:
                    del self._drop_watch[n]
                    if w["vmax"] > self.cfg["drop"]["speed_m_s"] and not self._welded(n) and n not in self._dropped:
                        self._dropped.add(n)
                        events.append("dropped:" + n)
        return events


class QuadInfoServer(InfoCellServer, _QuadBaseWatch):
    """`kit_dir` = the carved kit (tests/gt: kit.json + meshes, downloaded at verify start), `robot_dir` = the given robot models (the UR10e and the
    2F-85 MJCF folders), `plan` = the hidden bench plan ({"rest": {unit: {pos, quat, grasp, step, ...}}, "oracle":
    {stand, rotor_rad}})."""

    def __init__(self, kit_dir, robot_dir, plan, config=None, mode="hidden", privileged=False, render=True, log=None, kit_info=None):
        import mujoco
        if mode != "hidden":
            raise ValueError("this cell has no practice mode")
        self.mujoco = mujoco
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_QUAD_CONFIG)), config or {})
        self.kit_dir, self.robot_dir = kit_dir, robot_dir
        self.kit = load_kit(kit_dir)
        _check_kit(self.kit)
        self.bench_plan = dict(plan["rest"])
        self.oracle = dict(plan.get("oracle") or {})
        self.unit_vertices = LY.unit_vertices(self.kit)
        self.mode, self.allow_reset = mode, False
        self.privileged, self.render_enabled = bool(privileged), bool(render)
        self.log = log or (lambda s: None)
        self.kit_info = kit_info or {}
        self.camera_names = tuple(self.cfg.get("agent_cameras") or ("top", "wrist"))
        self.model = self.data = self.seed = self.cell = None
        self.done = False
        self.session = None
        self.renderer = None
        self.token_mismatch = False
        self._grasps = None
        self.calls = {}
        self.illegal_calls = 0

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        self.episode = LI.draw_episode(seed, self.cfg.get("info_layout"), strict=False)
        missing = [n for n in self.camera_names if self.episode["cameras"].get(n) is None]
        if missing and self.render_enabled:
            raise LI.MissingMount("camera(s) %s have no mount in the cell's config" % ", ".join(missing))
        nominal = bool(self.privileged and self.cfg.get("oracle_nominal_layout", True))
        rc = self.cfg.get("reach_check") or {}
        t0 = time.time()
        draws, reach = [], None
        for k in range(1 if nominal else max(1, int(rc.get("max_draws", 8)))):
            sub = int(seed) if k == 0 else (int(seed) ^ (0x9E3779B97F4A7C15 * k)) & ((1 << 63) - 1)
            lay = LY.episode_layout(sub, self.bench_plan, self.kit, self.unit_vertices, self.cfg["layout"], nominal=nominal,
                                    oracle=self.oracle)
            cell = self._build_cell(lay)
            if nominal or not rc.get("enabled", True):
                break
            reach = self._reach_check(cell)
            draws.append({"draw": k, "ok": reach["ok"], "failed": reach["failed"]})
            if reach["ok"]:
                break
            self.log("[cell] layout draw %d failed the reach check (%s): drawn again" % (k, ", ".join(reach["failed"])))
        else:
            # every draw failed: the plan itself (nominal, always reachable) with this episode's stand and pulleys
            lay = LY.episode_layout(seed, self.bench_plan, self.kit, self.unit_vertices, self.cfg["layout"], nominal=True, oracle=None)
            lay["stand"] = LY.stand_pose(seed, self.cfg["layout"])
            cell = self._build_cell(lay)
            draws.append({"draw": "plan", "ok": True, "failed": []})
        self.log("[cell] quadruped cell for seed %d built (%d draw%s) in %.0f s" % (int(seed) & 0xFFFFFFFF, max(1, len(draws)),
                                                                                     "" if len(draws) <= 1 else "s", time.time() - t0))
        cell.recording = False
        self.cell = cell
        self.model, self.data = cell.m, cell.d
        self.ctrl, self.sm = cell.ctrl, cell.sm
        self.tracker = None
        welds = {}
        for s in self.kit["slots"]:
            cu = self.kit["bodies"][s["carrier"]]["unit"]
            for c in s["candidates"]:
                welds[(cu, c["unit"])] = weld_name(s["id"], c["unit"])
        self.meta = {"loose": list(cell.loose), "pad_geoms": list(cell.robot["pad_geoms"]), "root": "base", "welds": welds,
                     "config": {"tick_steps": cell.tick_steps, "bench": cell.cfg["bench"], "stand": cell.cfg["stand"]}}
        moved = {u: round(float(np.linalg.norm(cell.body_T(u)[:3, 3] - np.asarray(lay["parts"][u][0], float))) * 1e3, 2) for u in lay["parts"]}
        self.layout = {"stand": lay["stand"], "parts": {u: {"pos": p, "quat": q} for u, (p, q) in lay["parts"].items()},
                       "rotor_rad": lay["rotor_rad"], "nominal": nominal, "jitter": lay["jitter"], "draws": draws,
                       "settle_moves_mm": moved}
        self._init_state(seed)
        self._close_session()
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in self.camera_names}
        self.render_wall_s = 0.0
        self.ep_seed = int(seed)
        hi = (int(seed) >> 32) & 0xFFFFFFFF
        self._wrench_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x3E17C4] + ([hi] if hi else []))
        # every control tick goes through the cell's watch, whoever drives it (an agent's step() or the reference)
        self._raw_tick = cell.tick_one

        def routed():
            self.physics_tick(track=True)
        cell.tick_one = routed
        return {"ok": True}

    def _build_cell(self, lay):
        rgba = (self.kit_info or {}).get("rgba")
        config = {"robot": dict(self.cfg["robot"], models_dir=self.robot_dir)}
        if rgba:
            config["materials"] = dict(CELL_CONFIG["materials"], printed=list(rgba))
        return QuadCell(self.kit_dir, {"stand": lay["stand"], "parts": lay["parts"], "rotor_rad": lay["rotor_rad"]}, config=config,
                        log=self.log, settle_s=float(self.cfg["settle_s"]))

    def _reach_check(self, cell):
        """every shoulder, support and upper leg keeps a grasp that passes the reference's pick checks where it rests
        (approach from above, IK in the natural branches, no contact on the approach and while closing); every lower
        leg the knee's place check (the upper leg lowered onto it at the bench's edge) for one of its upper leg's grasps
        that also passes the pick checks"""
        from .reference import Assembler, unit_grasps
        if self._grasps is None:
            self._grasps = unit_grasps(self.kit)
        asm = Assembler(cell, log=lambda *a: None, plan=self.bench_plan, grasps=self._grasps, monitor=False)
        failed = []
        for u in self.kit["loose"]:
            if u.startswith("L"):
                continue
            T_u = asm.unit_T(u)
            hint = (self.bench_plan.get(u) or {}).get("grasp")
            cands = ([(np.array(hint["G"], float), float(hint["w"]), "plan")] if hint else []) + list(asm.grasps[u])
            if u.startswith("U"):
                k = u[1:]
                T_k, ins_k = asm.target("kfe_" + u, u, part="L" + k)
                feas, _rej = asm.feasible_grasps(u, T_u, [("kfe_" + u, T_k, ins_k, {})], limit=1)
                if not feas:
                    failed.append("knee U%s/L%s" % (k, k))
                continue
            if not any(asm.pick_checks(u, T_u, G, w)[0] is not None for G, w, _d in cands):
                failed.append("pick " + u)
        return {"ok": not failed, "failed": failed}

    def _init_state(self, seed):
        m = self.model
        cell = self.cell
        self.seed = int(seed)
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.violations = []
        self.illegal_calls = 0
        self.calls = {}
        self.step_calls = 0
        self.regrasps = 0
        self.held = None
        self._held_cand = None
        self._held_ticks = 0
        self._drop_watch = {}
        self._lost = set()
        self._dropped = set()
        self._v_peak = {}
        self._v6 = np.zeros(6)
        self.wall_start = None
        self.wall_used = 0.0
        self.renderer = None
        self.tick_s = cell.tick_s
        self.tick_steps = cell.tick_steps
        self._obs_noise = (0.0, 0.0)
        self._obs_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x0B5E])
        self.geom_class = np.zeros(m.ngeom, dtype=int)
        self.geom_part = {}
        prefix = cell.robot["prefix"]
        robot_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith(prefix))
        loose = set(cell.loose)
        unit_of = cell.unit_of_body
        self.loose_body_ids = {b: u for b, u in unit_of.items() if u in loose}
        self.static_env_bodies = set([0, m.body("stand").id] + [b for b, u in unit_of.items() if u == "base"])
        pads = set(self.meta["pad_geoms"])
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            name = m.geom(g).name
            if name in pads:
                self.geom_class[g] = 2 if "left" in name else 3
            elif b in robot_bodies:
                self.geom_class[g] = 1
            elif b in self.loose_body_ids:
                self.geom_class[g] = 4
                self.geom_part[g] = self.loose_body_ids[b]
            elif b in self.static_env_bodies:
                self.geom_class[g] = 5
        self.part_body = {n: m.body(n).id for n in cell.loose}
        self.part_qadr = {n: m.jnt_qposadr[m.joint("free_" + n).id] for n in cell.loose}
        self.part_vadr = {n: m.jnt_dofadr[m.joint("free_" + n).id] for n in cell.loose}
        self.sensor_force = m.sensor(cell.robot["force_sensor"]).id
        self.sensor_torque = m.sensor(cell.robot["torque_sensor"]).id
        self.wrist_parent = cell.robot["gripper_prefix"] + "base"
        self._f6 = np.zeros(6)
        self._dmg_peak = {"pad": 0.0, "part": 0.0}
        self._dmg_over = {"pad": 0, "part": 0}
        self._dmg_sustained = {"part": 0.0}
        self._part_hist = []

    # ------------------------------------------------------------------ physics
    def physics_tick(self, track=True):
        if self.done:
            return []
        n0 = len(self.sm.events)
        self._raw_tick()
        self.tick_count += 1
        events = []
        if track:
            events += ["%s" % e for _t, e in self.sm.events[n0:]]
            events += self._watch(self.data)
            for e in events:
                self.events_log.append((float(self.data.time), e))
        if self.data.time >= self.cfg["sim_cap_s"] - 1e-9:
            self._end("sim_cap")
        return events

    # ------------------------------------------------------------------ the restricted API
    def describe(self):
        m = self.model
        cams = {}
        for n in self.camera_names:
            c = self.episode["config"]["cameras"][n]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)),
                       "nominal_hfov_deg": round(float(c["hfov_deg"])), "mount": _MOUNT_TEXT.get(n, "")}
        rb, ct = self.cell.robot, self.ctrl
        from .gripper_2f85 import FORCE_MODEL_TEXT
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
            "task": ("build the quadruped on its stand from its body: the four shoulder modules onto the body's hip ab/ad outputs, the "
                     "four hip ab/ad supports over the shoulders onto the body, the four upper legs onto the shoulders' hip flexion "
                     "outputs and the four lower legs onto the upper legs' knee outputs, as the picture manual shows"),
            "robot": {"model": "6-DoF UR10e-class arm (given model: robot/universal_robots_ur10e/) with a 2F-85-class gripper "
                               "(robot/robotiq_2f85/)",
                      "base_pos": [float(v) for v in self.cfg["robot"]["pos"]],
                      "base_axes": "the arm's base frame is the cell frame turned 180 deg about z: q = 0 on the first joint points the arm "
                                   "along the cell's -x",
                      "joints": list(rb["joints"]),
                      "joint_range": [[float(m.jnt_range[m.joint(j).id][0]), float(m.jnt_range[m.joint(j).id][1])] for j in rb["joints"]],
                      "home_q": [float(v) for v in ct.q_cmd],
                      "torque_limits_Nm": [float(v) for v in rb["torque_limits_Nm"]],
                      "tcp_site": rb["tcp_site"], "tcp_offset_from_flange_m": [0.0, 0.0, round(float(rb["tcp_from_flange_m"]), 4)],
                      "tcp_axes": "z = the approach (the flange axis, out of the flange), y = the closing axis, x across the pads",
                      "gripper": {"max_width_m": float(ct.max_width), "rate_m_s": 0.10,
                                  "width": "the gap between the pad faces at the pads' middle",
                                  "pad_size_m": [0.022, 0.0375], "force_model": FORCE_MODEL_TEXT,
                                  "kinematics": "parallel pinch only: the finger linkage is rigid; along the approach the pads' face centre "
                                                "lies about 14 mm short of the tcp fully open and 1-4 mm short for 10-45 mm grasps, the "
                                                "fingertips 5-18 mm beyond it, depending on the width; the finger linkage spans +-62-74 mm "
                                                "along the closing axis"},
                      "wrist_housing": "a 36 x 24 x 35 mm camera housing on the gripper's coupling, on the tool's +x side (part of the robot)",
                      "gravity_compensated": True,
                      "ft_sensor": ("6-axis at the tool flange's centre in the flange frame (z along the tool axis outwards, x along the "
                                    "closing axis = the tcp's y, y along the tcp's -x); what the wrist applies to the tool, the gripper's "
                                    "own weight included (about 10.3 N with the tool pointing down): tare it; noise ~0.3 N / 0.02 N m rms "
                                    "per reading, an unknown bias up to 1 N / 0.05 N m per episode")},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": float(min(1.5, min(rb["speed_limits_rad_s"]))),
                        "cartesian_speed_limit_m_s": 0.5, "cartesian_angular_limit_rad_s": 1.5, "stiffness_scale_range": [0.2, 1.0],
                        "modes": ["joint", "cartesian"]},
            "fastening": self._fastening_text(),
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "datums": quad_datums(),
            "items": ["the robot's body with its four hip ab/ad modules bolted on the stand at about (%.2f, %.2f) m (+-10 mm), its long "
                      "axis along the cell's y, its front towards +y" % tuple(self.cfg["layout"]["stand"]["nominal_xy"]),
                      "the 16 loose units (4 shoulder modules: 2 plain, 2 with the hip adapter; 4 hip ab/ad supports, 4 upper legs, "
                      "4 lower legs), each lying on the bench in a stable orientation on one of its long sides within the robot's reach "
                      "(the lower legs across the bench's long edges, key up); their places and their turns about the vertical vary "
                      "between episodes, the face each rests on does not; identical units are interchangeable"],
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "wall_clock_cap_s": self.cfg["wall_cap_s"], "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"], "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": dict(self.cfg["safety"], part_force_N=(self.cfg.get("damage") or {}).get("part_force_N"),
                           part_force_window_s=0.01 * int((self.cfg.get("damage") or {}).get("ticks", 20)),
                           rules=["a robot link other than the 2F-85's four pad boxes pressing on a unit, the stand, the body or the bench "
                                  "above robot_force_N", "a unit's centre of mass faster than part_speed_m_s",
                                  "a unit pressed on the environment or another unit above part_force_N for part_force_window_s (the "
                                  "normal forces summed per unit and body it presses on; the gripper's pads on a unit do not count)"]),
        }

    def _fastening_text(self):
        sm = self.sm
        keyed = [s for s in self.kit["slots"] if s["kind"] == "keyed"]
        seat = [s for s in self.kit["slots"] if s["kind"] == "seat"]
        k0, s0 = keyed[0], seat[0]
        keys = (self.kit.get("carve") or {}).get("keys", {})
        # the key heights per unit type, by the given kit.json's unit ids (one height per type in this kit)
        heights = {}
        for v in keys.values():
            n = GIVEN_UNIT_IDS.get(self.kit["units"][v["child"]]["type"], v["child"])
            heights.setdefault(n, set()).add(round(float(v["len_mm"]), 1))
        order = list(GIVEN_UNIT_IDS.values())
        key_mm = {n: (min(heights[n]) if len(heights[n]) == 1 else sorted(heights[n]))
                  for n in sorted(heights, key=lambda n: order.index(n) if n in order else len(order))}
        h = self.kit["hinge"]
        f = sm.release_factor
        play = float(self.kit["core"]["socket_clearance_mm"])
        lead_in = float(self.kit["core"].get("socket_lead_in_mm", 0.0))
        if lead_in > 0.0:
            entry = ("the socket has %.1f mm of play per side and the drawing's lead-in as a 45 deg x %.1f mm chamfer on both pulley faces "
                     "(the lead-in counts towards the height inside); a key gets in only from close to its socket: %s" % (
                         play, lead_in, ENTRY_WINDOW_TEXT))
        else:
            entry = "the socket's walls let it in only within their play (%.1f mm per side)" % play
        return {
            "keyed": ("a unit's key in an output pulley's socket is captured once at least %d %% of the key's height is inside the socket, "
                      "its axis within %.0f mm and %.0f deg of the socket's and its turn about the axis within %.0f deg of a key orientation "
                      "(the key fits the socket in two orientations, 180 deg apart); %s; each output pulley starts up to about %.0f deg off "
                      "its index, so measure or search the pulley's turn; the cell then holds it on the feature and pulls it home at <= %.0f "
                      "mm/s and %.0f deg/s, and fastens it "
                      "when it rests there (within %.1f mm / %.0f deg of home, %.1f mm axially) for %d consecutive 10 ms ticks moving slower "
                      "than %.0f mm/s and turning slower than %.0f deg/s relative to the pulley; a captured key is released once its tip is "
                      "pulled more than %.0f mm clear of the socket, its axis is more than %.0f mm / %.0f deg off the socket's or its turn "
                      "more than %.0f deg off the key orientation" % (
                          round(100 * sm.key_frac), sm.cap_lat * 1e3, math.degrees(sm.cap_ang), math.degrees(sm.cap_ang),
                          entry, float(self.cfg["layout"]["rotor_jitter_deg"]),
                          sm.pull_v * 1e3, math.degrees(sm.pull_w),
                          float(k0["tol_pos_mm"]), float(k0["tol_ang_deg"]), float(k0.get("axial_tol_mm", 0.0)), sm.hold_ticks,
                          sm.rel_speed_max * 1e3, math.degrees(sm.rest_ang),
                          KEY_RELEASE_CLEAR_MM, sm.cap_lat * f * 1e3, math.degrees(sm.cap_ang * f), math.degrees(sm.cap_ang * f))),
            "key_heights_mm": key_mm,
            "seat": ("a hip ab/ad support is captured on the body over its shoulder's journal up to %.0f mm short of home along the hip "
                     "ab/ad axis (within %.0f mm across, %.0f deg; its two screws reach their threads) and is pulled home and fastened as a "
                     "key (within %.1f mm / %.0f deg); it needs its shoulder fastened first; a captured support is released when it is "
                     "more than %.0f mm short of home, %.0f mm across or %.0f deg off" % (
                         sm.seat_depth * 1e3, sm.seat_lat * 1e3, math.degrees(sm.cap_ang), float(s0["tol_pos_mm"]), float(s0["tol_ang_deg"]),
                         sm.seat_depth * f * 1e3, sm.seat_lat * f * 1e3, math.degrees(sm.cap_ang * f))),
            "counts": ("a shoulder counts only in its design key orientation (fastened the other way round it is bolted on and counts 0); a "
                       "hip flexion or knee key counts in either orientation; a slot counts once it is fastened; nothing is reported"),
            "joints": ("every joint of the robot being built is unpowered and back-drives against its friction (%.2f N m; damping %.2g N m s/rad); "
                       "a fastened shoulder limits its hip ab/ad joint to its range; each output pulley starts at its index within about "
                       "%.0f deg and stays wherever it is turned" % (float(h["frictionloss"]), float(h["damping"]),
                                                                       float(self.cfg["layout"]["rotor_jitter_deg"]))),
            "weights": {"shoulder": 1.5, "support": 1.0, "hip_flexion": 1.5, "knee": 1.5, "total": float(sm.total_w)},
        }

    def _session(self):
        if self.session is None:
            from . import render_twin
            t = time.time()
            want = str(self.cfg.get("agent_renderer", "mujoco"))
            if want == "filament":
                from . import agent_render
                ok, why = agent_render.available()
                if not ok:
                    raise RuntimeError("agent renderer filament unavailable: %s" % why)
                self.session = agent_render.open_session(self)
                self.renderer = "filament"
            else:
                self.session = render_twin.open_session(self)
                self.renderer = "mujoco"
            self.render_wall_s += time.time() - t
            self.log("[cell] %s render session opened in %.1f s" % (self.renderer, time.time() - t))
        return self.session

    def camera_spec(self, name):
        from render.spec import CameraSpec
        c = self.episode["cameras"][name]
        intr = c["intrinsics"]
        if name == "wrist":
            b = self.model.body(self.wrist_parent).id
            T = np.eye(4)
            T[:3, :3] = self.data.xmat[b].reshape(3, 3)
            T[:3, 3] = self.data.xpos[b]
            W = T @ np.asarray(c["hand_T_cam"], float)
        else:
            W = np.asarray(c["world_T_cam"], float)
        return CameraSpec(name=name, width=intr["width"], height=intr["height"], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                          world_T_cam=W.tolist(), k1=intr["k1"], k2=intr["k2"], p1=intr["p1"], p2=intr["p2"])

    def photo_spec(self, name):
        """The information tier's photo pipeline with the colour response drawn once per camera unit and episode (a live
        camera's gain / tint does not change from frame to frame); the sensor noise stays per frame."""
        import dataclasses
        from render.spec import derive_seed
        p = super().photo_spec(name)
        return dataclasses.replace(p, colour_jitter=dataclasses.replace(p.colour_jitter, seed=derive_seed(self.ep_seed, "colour_unit", name)))

    def get_camera(self, name="top", depth=True):
        if str(name) not in self.camera_names:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(self.camera_names)}
        return super().get_camera(name, depth=depth)

    def describe_privileged(self):
        out = {"layout": self.layout, "slots": self.kit["slots"], "states": dict(self.sm.state), "score": self.sm.score()}
        return json.loads(json.dumps(out, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))

    def get_state_privileged(self):
        d, m = self.data, self.model
        parts = {n: {"pos": [float(v) for v in d.xpos[b]], "quat": [float(v) for v in d.xquat[b]]} for n, b in self.part_body.items()}
        rotors = {bn: float(d.qpos[m.jnt_qposadr[m.joint("hinge_" + bn).id]]) for bn, bb in self.kit["bodies"].items() if bb.get("joint")}
        return {"sim_time": float(d.time), "parts": parts, "rotors": rotors, "states": dict(self.sm.state), "records": self.sm.record(),
                "score": self.sm.score()}

    def reference_assemble(self):
        """The privileged reference program on this episode (oracle token only)."""
        from .reference import program
        t0 = time.time()
        rep = program(self.cell, plan=self.bench_plan, log=self.log)
        return {"ok": True, "score": float(self.sm.score()), "placed_by_ur": int(sum(1 for s in rep["steps"] if s.get("placed_by") == "ur")),
                "unplanned_contacts": int(len(rep["unplanned_contacts"])), "sim_s": float(self.data.time), "wall_s": float(time.time() - t0)}

    def handle(self, method, params):
        if method in PRIVILEGED_METHODS:
            self.calls[method] = self.calls.get(method, 0) + 1
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method == "truth":
            self.calls[method] = self.calls.get(method, 0) + 1
            self.illegal_calls += 1
            return {"error": "forbidden"}
        return super().handle(method, params)

    # ------------------------------------------------------------------ end of episode
    def lowest_part_z(self):
        """the lowest point of any loose unit's collision geometry (m): below the bench top = a unit on the floor"""
        m, d = self.model, self.data
        low = float("inf")
        for g in range(m.ngeom):
            if self.geom_class[g] != 4 or m.geom_dataid[g] < 0:
                continue
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            z = v @ d.geom_xmat[g].reshape(3, 3)[2] + d.geom_xpos[g][2]
            low = min(low, float(z.min()))
        return low

    def finalize(self, functional_rng=None, run_functional_check=False):
        self._close_session()
        if not self.done:
            self._end("solve_returned")
        rec = self.sm.record()
        weights = {r["id"]: float(r["weight"]) for r in rec}
        W = sum(weights.values())
        achieved_w = sum(weights[r["id"]] for r in rec if r["achieved"])
        m, d = self.model, self.data
        rotors = {bn: float(d.qpos[m.jnt_qposadr[m.joint("hinge_" + bn).id]]) for bn, bb in self.kit["bodies"].items() if bb.get("joint")}
        return {
            "cell_version": CELL_VERSION, "seed": self.seed, "done_reason": self.done_reason,
            "sim_time_s": float(d.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "mates": rec, "states": dict(self.sm.state), "weights": weights, "stage_score": achieved_w / W if W else 0.0,
            "n_achieved": int(sum(1 for r in rec if r["achieved"])), "n_fastened": int(sum(1 for r in rec if r.get("fastened"))),
            "all_mates": bool(all(r["achieved"] for r in rec)),
            "off_design": [r["id"] for r in rec if r.get("fastened") and not r["achieved"]],
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "lost_parts": sorted(self._lost), "dropped_parts": sorted(self._dropped),
            "events": [[t, e] for t, e in self.events_log][-400:], "coupling_events": [list(e) for e in self.sm.events],
            "rotor_angles_rad": rotors, "haa_limits": {k: [float(x) for x in v] for k, v in self.sm.limits_set.items()},
            "max_part_speed_m_s": {k: round(v, 3) for k, v in self._v_peak.items()},
            "layout": self.layout, "damage": self.damage_summary(),
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_wall_s": float(self.render_wall_s), "renderer": getattr(self, "renderer", None), "privileged_mode": bool(self.privileged),
            "lowest_part_z_m": float(self.lowest_part_z()),
        }


def _check_kit(kit):
    slots = kit["slots"]
    if len(slots) != 16:
        raise ValueError("kit: %d slots, the cell expects 16" % len(slots))
    w = sum(float(s["weight"]) for s in slots)
    if abs(w - 22.0) > 1e-9:
        raise ValueError("kit: total slot weight %.2f, the cell expects 22" % w)
    if len(kit["loose"]) != 16:
        raise ValueError("kit: %d loose units, the cell expects 16" % len(kit["loose"]))
    keys = (kit.get("carve") or {}).get("keys") or {}
    if len(keys) < 12:
        raise ValueError("kit: no carve.keys block for the keyed slots (the coupling would use 4 mm for every key)")
    rotors = [bn for bn, bb in kit["bodies"].items() if bb.get("joint")]
    if len(rotors) != 12:
        raise ValueError("kit: %d output pulleys, the cell expects 12" % len(rotors))


def quad_datums():
    """The documented datums (the cell manual's survey table; exact, public). Inner corner (i, j) of a board of C x R
    squares of side s at ((i - C / 2) s, (R / 2 - j) s) in the board frame (render/scale_refs.py)."""
    b, s, t = BENCH_BOARD, STAND_BOARD, TOOL_PLATE
    return {
        "frame": "cell frame: origin on the bench top at the robot arm's base axis, z up (the bench top is z = 0, 0.994 m above the floor), "
                 "x towards the stand; the bench top spans x -0.55..1.45 m, y -0.70..0.70 m",
        "bench_board": {"squares": [b["cols"], b["rows"]], "square_mm": b["square_mm"], "margin_mm": b["square_mm"], "thickness_mm": 3.0,
                        "inner_corner_cell_m": "x = %.3f + (i - 4.5) x 0.020, y = %.3f + (3 - j) x 0.020, z = 0.00305 (i = 1..8, j = 1..5)" % (
                            b["pos"][0], b["pos"][1]),
                        "black_corner_square": "at the board's -x, +y corner", "lies_flat_on_bench": True},
        "stand_board": {"squares": [s["cols"], s["rows"]], "square_mm": s["square_mm"], "margin_mm": s["margin_mm"], "thickness_mm": s["thickness_mm"],
                        "mounting": "vertical on the stand rail's face towards the robot arm (the cell's -x), centred on the rail",
                        "note": "moves with the stand (+-10 mm)"},
        "tool_plate": {"squares": [t["cols"], t["rows"]], "square_mm": t["square_mm"], "margin_mm": t["margin_mm"], "thickness_mm": t["thickness_mm"],
                       "mounting": "on the camera housing's outer face (the tool's +x side), facing +x", "mounting_tolerance": "about 1 mm / 0.5 deg"},
        "stand": {"nominal_centre_m": list(DEFAULT_QUAD_CONFIG["layout"]["stand"]["nominal_xy"]), "tolerance_mm": 10.0,
                  "body_centre_height_m": float(CELL_CONFIG["stand"]["body_z"]), "long_axis": "the cell's y"},
    }


__all__ = ["QuadInfoServer", "CELL_VERSION", "DEFAULT_QUAD_CONFIG", "quad_datums", "PRIVILEGED_METHODS"]
