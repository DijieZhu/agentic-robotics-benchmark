"""The final-assembly-and-trim cell (grader side): the base vehicle on the two-position turnover positioner over three
load cells, the five stations with the kit's nineteen loose units (twelve ballast blocks in the ballast box, four
vertical thrusters in the tray, the electronics enclosure on its rack, a guard on each side's rack), the arm with the
wide gripper, the coupling state machine and the cell's watch (the hard rules, the dome rule, the scrap rules, the
metrics, the budgets).

Sources (the structure copied, then rewritten for this cell; none of them is imported):
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/cell.py
      sha256 62fc787aec22cf4b5d9b0efb2eaf96567984cf2668b1e7079557f4548f317676 (CarCell: the kit-driven build, one
      inactive weld per (slot, candidate), the contact clusters of the fastened groups, the per-tick watch, the
      recording, live_target / design_world)
  -dev/proto/gripper_g0b/g0b_lib.py sha256 0ac3b64257b38ac94e9bc15f7642f77e00ce7789fc3ada132a8b57cd2e78617a (the
      physics as a configuration, `configure`; the variable step `VarStep` and its 'cmd' rule)
  -dev/proto/bigparts/bp_cell.py sha256 72923f67b67fba18c211b8d303abb9d0244c4b798db40cd96220ba5d98a8c74a and
  -dev/proto/ballast/bal_cell.py sha256 73cbb15cc0e464fdd91239976e5df20ec0423665fda76a7c357f3fa0e3883693 (the
      Monitor: the dome's contacts, the weld forces apart from the contact forces)
  -dev/kit/kit_contacts.py sha256 1eac7055d23e255864d67f63d0d3f6383b9515fbab12f9dbe7405ea59250f3f6 (reading the kit:
      geoms, explicit inertia, the weld data)
The modules this cell is made of (all in this package): `gripper_wide` (the arm + the wide gripper, its controller and
force register), `positioner` (the prescribed turn, the load cells, the column interlock, the turn-start scrap rule),
`fixtures` / `layout` (the stations, the per-episode draws), `coupling` (capture, rest gate, lock, scoring records).

PHYSICS COMES ONLY FROM THE CONFIGURATION (`CELL_CONFIG["physics"]`, overridden by `config["physics"]`): the step,
the steps per control tick, the pads' contact time constant, the integrator, the cone, impratio, noslip, multi-CCD,
gravity, the geom defaults, the weld impedance, the arena, MuJoCo's sleeping, and the variable-step fallback. The
cell passes it to the positioner (build and tare) and to the gripper (its pad solref, impratio); nothing in this file
types a physics number outside CELL_CONFIG. Ruling M8: 0.5 ms implicitfast, 20 steps per 10 ms tick (the arm's
computed torque recomputed before every step), pad solref 0.001, elliptic cones, impratio 100, noslip 5, multi-CCD
off, g 9.80665; with the supplement of rulings P08-1 / P14-1 (group Q): MuJoCo's sleeping of resting loose units on
(`physics.sleep`, tolerance 1e-5 m/s; the positioner and the robot never sleep). The fallback order of M8 is two
switches: `physics.timestep` 0.00025 (fixed 0.25 ms; tick_steps 40) and `physics.varstep.enabled` (0.25 ms only while
the gripper is commanded closed, rule 'cmd' of G0b: a function of the commands, switched between ticks, so a run stays
bit-for-bit reproducible).

`RovCell(kit_dir, layout, config=None, log=print, robot_hook=None, spec_hook=None, settle_s=None, robot_init=None,
kit=None, fx=None, tare=None)` compiles from any kit directory in the hidden kit's format (`rov-asm-kit/1`, the dev
folder's kit/KIT_FORMAT.md) and an episode layout (`layout.episode_layout`):
  * the bench and the floor; the five stations as static bodies at the layout's poses (`fixtures.add_station`); the
    datum boards as non-colliding plates (`board_<name>`, group 2) at the layout's board poses;
  * the positioner (`positioner.build_positioner`: open base on the three tendon load cells, the hinge, the carrier)
    with the kit's `vehicle` body on the carrier at the layout's mounting offset (its collision geoms group 3, its
    visual meshes group 2, its explicit inertia);
  * every loose unit a free body at the layout's pose, its collision geoms as the kit gives them (mesh / box /
    sphere, mass 0, the configuration's contact defaults), its explicit inertia; visual meshes as non-colliding geoms
    (a thruster's propeller turned by the layout's `prop_deg` about the thruster axis); the station fasteners
    (`visual_fasteners`) hidden until the unit is fastened;
  * one inactive weld per (slot, candidate) (`weld_<slot>__<unit>`, data [0]*10 + [1]: anchor at body2's origin);
  * the robot through `robot_hook(spec, cfg) -> dict` or cfg["robot"] (`gripper_wide.add_ur_wide`'s cfg; None: no
    robot, e.g. the floating check).
Then the positioner's driver (`positioner.Positioner`, the turn-start scrap rule wired to the coupling), the coupling
(`coupling.CouplingStateMachine`), the robot's controller (`gripper_wide.WideController` at `robot_home_q`, the gripper
open), the contact clusters and the watch; the cell settles for `settle_s` (physics only) and resets the clock to 0.

Every control tick (`tick_one`): [variable step: choose the step for this tick from the commands] the controllers'
tick (the commanded configuration, the gripper's ctrl and force register) and the mocap jigs; tick_steps x (the
positioner's prescribed hinge angle, mj_step1, the arm's computed torque, mj_step2); mj_forward; the coupling's tick
(`sm.tick`); the contact clusters (no contacts inside a rigidly fastened group); the watch (`watch_tick`); the budget's
tick; the hooks; the recording (every 4 ticks).

The watch (DESIGN v1.2 section 1.3; rulings L12, L19, N5; group Q's P15-1 / P16-1 for H2):
  * H1: a robot link other than the four pads pressing on any body (but the robot) above `watch.robot_force_N`, the
    NORMAL forces of the link's contacts on that body summed per (link, body) -> violation
    "robot_contact:<link>:<body>:<N>N" (the episode scores 0);
  * H2: a loose unit pressed on the environment or another unit with a CONTACT force above `watch.part_force_N` on
    `watch.part_force_ticks` consecutive ticks -> violation "part_force:<unit>:<body>:<N>N". The force of a
    (unit, body it presses on) pair is the MAGNITUDE OF THE VECTOR SUM of the contact forces of that pair on the unit
    (normal and tangential components together, in the world frame: the force the body exerts on the unit), the
    largest pair of the tick; not the sum of the normal forces, which a unit wedged in a V-block or between two faces
    inflates (the enclosure resting in its two V-blocks: normal forces 46-135 N, their resultant its weight). The pads
    on a unit and a robot link on a unit do not count (H1 has the latter); the welds' constraint forces (capture pull,
    lock) are NOT contact forces and never count: they are reported apart (`metrics()["weld"]`, the coupling's soft /
    hard weld maxima and their 0.2 s sustained maximum);
  * H3: anything pressing on the enclosure's dome (the kit's `role: dome` geoms) above `watch.dome_force_N`, the sum
    of the tick, on `watch.dome_ticks` consecutive ticks -> the enclosure is scrapped (its place counts nothing, VOID
    if it was fastened; the episode goes on);
  * the scrap rules (each scraps only that unit; the coupling releases it if captured; it can no longer be fastened):
    S1 a loose unit whose centre of mass moves faster than `watch.scrap_speed_m_s`; S2 a loose unit whose centre of mass
    is below the bench top by more than `watch.scrap_below_bench_m` (it fell off the bench); S3 at the start of a turn
    of the positioner, every unit on the vehicle that is not fastened (touching the vehicle, the carrier or a fastened
    unit, or its centre of mass in the vehicle-with-all-units box + 30 mm: `positioner.unlocked_on_vehicle`); S4 a
    unit let go without being captured that hangs or lies on the vehicle is NOT scrapped at once (none of S1-S3 holds):
    it counts nothing, can be picked up again and falls under S3 at the next turn;
  * the column interlock is the positioner's (`fixture_index` refused); while a turn runs the cell counts the ticks
    with a robot geom inside the column (a metric, not a rule).
`watch_summary()` has the maxima, the violations, `hard_ok`; `metrics()` the summary + the coupling's metrics (the
weld forces on their own) + the positioner's report + the budget.

The budget (`Budget`, DESIGN section 1.3): simulated seconds, ticks per step call, step calls, camera frames per camera,
turns of the positioner (counted by the positioner), illegal calls, wall seconds between `budget.start_wall()` and now.
The cell charges the ticks; the server (P15) charges the step calls, frames and illegal calls. `budget.over()` names
the first exhausted budget ('sim_cap', 'frames', ...); `step(n)` refuses n above the per-call cap.

`state_hash()` is the sha256 of everything the run's future depends on (time, qpos, qvel, act, ctrl, warm start,
weld activity / data / impedance, collision bits, the gripper's force range, the step, the coupling's, positioner's,
watch's and budget's state): the determinism checks compare it at checkpoints.
"""
import copy
import gc
import hashlib
import json
import math
import os
import time

import numpy as np

from . import fixtures as FX
from . import layout as LY
from . import positioner as PZ
from .coupling import CouplingStateMachine, _inv, _rotz, body_pose, quat_wxyz, weld_name

KIT_VERSION = "rov-asm-kit/1"
GEOM_TYPES = ("box", "sphere", "mesh")
STATE_GEOM = {"col": 0, "seat": 0, "dome": 0}

CELL_CONFIG = {
    # ---- the cell's physics: the ONLY place the cell reads it from (ruling M8; DESIGN v1.2 section 1.3)
    "physics": {
        "timestep": 0.0005, "tick_steps": 20, "integrator": "implicitfast", "cone": "elliptic",
        "impratio": 100.0, "noslip_iterations": 5, "multiccd": False, "gravity": 9.80665,
        "pad_solref": [0.001, 1.0],
        "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0],
                          "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
        "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
        "arena_memory_bytes": 256 << 20,   # the measured peak of the scripted run: < 2 MB (record P2_P14)
        # MuJoCo's sleeping of resting trees: ON, part of the designed physics (rulings P08-1 / P14-1 of group Q, a
        # supplement to M8; records P2_P14, P2_P18a): a loose unit whose every dof moves slower than sleep_tolerance
        # (m/s; rotations scaled by the body's size) for 10 steps is frozen until something awake touches it or its
        # state is set; the positioner's tree and the robot's tree never sleep (sleep_never)
        "sleep": True, "sleep_tolerance": 1e-5, "sleep_never": ["positioner", "robot"],
        # the second fallback of M8: a fine step while the gripper is commanded closed (G0b's 'cmd' rule: commanded
        # width goal below the open width - open_margin_m, and holdover_s after it was raised again)
        "varstep": {"enabled": False, "rule": "cmd", "fine_timestep": 0.00025, "open_margin_m": 0.001,
                    "holdover_s": 0.3, "pad_solref_steps": 2.0},
    },
    "lmag": False,
    "cell_layout": None,               # None: fixtures.load_cell_layout() ($ROV_CELL_LAYOUT or <tests>/cell_layout.json)
    "fixture_facts": None,             # None: fixtures.load_facts() ($ROV_FIXTURE_FACTS or <tests>/fixture_facts.json)
    "settle_s": 1.0,
    "start_index": "keel_up",
    # the arm (None: no robot): gripper_wide.add_ur_wide's cfg (model, pos, yaw_deg, models_dir, ...); the cell adds
    # the physics' pad solref and impratio
    "robot": None,
    # the arm's ready posture (bench_plan.READY_Q, checked there: clear of every station, the positioner and its
    # column, the vehicle in both index positions)
    "robot_home_q": None,
    "grip_force_N": 100.0,
    "positioner": {},                  # overrides of positioner.DEFAULT_CFG
    "coupling": {},                    # CouplingStateMachine keyword arguments
    "clusters": True,
    "visual": True,                    # the kit's visual meshes as non-colliding geoms
    "show_collision": False,
    "watch": {"enabled": True, "robot_force_N": 30.0, "part_force_N": 150.0, "part_force_ticks": 20,
              "dome_force_N": 20.0, "dome_ticks": 5, "scrap_speed_m_s": 3.0, "scrap_below_bench_m": 0.05,
              "scrap": True, "column_while_turning": True},
    "budget": {"sim_s": 1200.0, "wall_s": 2400.0, "frames": 600, "step_ticks": 500, "turns": 4},
    "materials": {"bench": [0.80, 0.74, 0.62, 1.0], "floor": [0.30, 0.30, 0.30, 1.0], "board": [0.92, 0.92, 0.90, 1.0]},
    "floor_z": -0.994,
    "record_every": 4,
}
# colours where the kit names no material or the collision is shown
UNIT_RGBA = {"base_vehicle": [0.45, 0.47, 0.50, 1.0], "electronics_enclosure": [0.75, 0.85, 0.95, 0.9],
             "thruster_a": [0.15, 0.15, 0.17, 1.0], "thruster_b": [0.18, 0.15, 0.15, 1.0],
             "thruster_guard": [0.10, 0.10, 0.10, 1.0], "ballast_block": [0.55, 0.55, 0.58, 1.0]}
MATERIAL_RGBA = {"acrylic_clear": [0.85, 0.92, 0.98, 0.35], "dome_clear": [0.85, 0.92, 0.98, 0.35],
                 "al_anodized_black": [0.08, 0.08, 0.09, 1.0], "al_anodized_blue": [0.10, 0.25, 0.65, 1.0],
                 "al_bare": [0.75, 0.76, 0.78, 1.0], "steel": [0.62, 0.63, 0.65, 1.0], "ballast_steel": [0.50, 0.50, 0.52, 1.0],
                 "hdpe_black": [0.07, 0.07, 0.07, 1.0], "rubber_black": [0.05, 0.05, 0.05, 1.0],
                 "plastic_white": [0.90, 0.90, 0.88, 1.0], "plastic_grey": [0.55, 0.56, 0.58, 1.0],
                 "pcb_green": [0.10, 0.45, 0.20, 1.0], "pcb_black": [0.08, 0.08, 0.08, 1.0], "cable_black": [0.05, 0.05, 0.05, 1.0],
                 "battery_wrap": [0.15, 0.30, 0.70, 1.0], "dvl_black": [0.08, 0.08, 0.09, 1.0],
                 "thruster_black": [0.10, 0.10, 0.11, 1.0], "propeller": [0.12, 0.12, 0.13, 1.0], "housing": [0.10, 0.10, 0.11, 1.0]}
INTEGRATORS = {"euler": "mjINT_EULER", "implicit": "mjINT_IMPLICIT", "implicitfast": "mjINT_IMPLICITFAST", "rk4": "mjINT_RK4"}
CONES = {"elliptic": "mjCONE_ELLIPTIC", "pyramidal": "mjCONE_PYRAMIDAL"}
CAPTURED = ("ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")


# ====================================================================================================== config, kit
def merge_cfg(config=None, base=None):
    """deep merge of `config` over CELL_CONFIG (dicts merged key by key at every level)"""
    out = copy.deepcopy(CELL_CONFIG if base is None else base)

    def upd(a, b):
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                upd(a[k], v)
            else:
                a[k] = copy.deepcopy(v)
    upd(out, config)
    return out


def tick_s_of(physics):
    """the control tick (s) of a physics config: timestep x tick_steps (must be the cell's 10 ms tick)"""
    return float(physics["timestep"]) * int(physics["tick_steps"])


def check_physics(physics):
    """ValueError for a physics config the cell cannot run"""
    P = physics
    dt, n = float(P["timestep"]), int(P["tick_steps"])
    if not (dt > 0 and n >= 1):
        raise ValueError("physics: timestep %r, tick_steps %r" % (dt, n))
    if abs(dt * n - 0.01) > 1e-12:
        raise ValueError("physics: timestep x tick_steps = %.6f s, the cell's control tick is 0.01 s" % (dt * n))
    if P["integrator"] not in INTEGRATORS or P["cone"] not in CONES:
        raise ValueError("physics: integrator %r / cone %r" % (P["integrator"], P["cone"]))
    vs = P.get("varstep") or {}
    if vs.get("enabled"):
        f = float(vs["fine_timestep"])
        if f <= 0 or abs(round(0.01 / f) * f - 0.01) > 1e-12 or f >= dt:
            raise ValueError("physics.varstep: fine_timestep %r must divide the 10 ms tick and be below the step %r" % (f, dt))
        if vs.get("rule", "cmd") != "cmd":
            raise ValueError("physics.varstep: rule %r (only the command rule 'cmd' is bit-reproducible)" % vs.get("rule"))


def positioner_physics(physics):
    """the positioner's physics dict (positioner.PHYSICS keys) from the cell's physics config"""
    P = physics
    return {"timestep": float(P["timestep"]), "tick_steps": int(P["tick_steps"]), "impratio": float(P["impratio"]),
            "noslip_iterations": int(P["noslip_iterations"]), "multiccd": bool(P["multiccd"]),
            "geom_defaults": copy.deepcopy(P["geom_defaults"]), "weld_solref": list(P["weld_solref"]),
            "weld_solimp": list(P["weld_solimp"]), "gravity": float(P["gravity"]),
            "arena_memory_bytes": int(P["arena_memory_bytes"])}


def apply_physics(spec, physics):
    """the cell's physics on an MjSpec (options and geom defaults; called before and again after the robot's
    attachment, which brings options of its own)"""
    import mujoco
    P = physics
    spec.memory = int(P["arena_memory_bytes"])
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    o = spec.option
    o.timestep = float(P["timestep"])
    o.integrator = getattr(mujoco.mjtIntegrator, INTEGRATORS[P["integrator"]])
    o.cone = getattr(mujoco.mjtCone, CONES[P["cone"]])
    o.noslip_iterations = int(P["noslip_iterations"])
    o.impratio = float(P["impratio"])
    o.gravity = [0.0, 0.0, -float(P["gravity"])]
    if P["multiccd"]:
        o.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    else:
        o.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    if P.get("sleep"):
        o.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        o.sleep_tolerance = float(P["sleep_tolerance"])
    else:
        o.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)


def apply_sleep_policy(spec, physics, roots):
    """the trees that never sleep (physics.sleep_never: 'positioner' and / or 'robot' -> their root bodies in `roots`);
    every loose unit may sleep (MuJoCo's default policy for an unactuated tree)"""
    import mujoco
    if not physics.get("sleep"):
        return []
    out = []
    for key in physics.get("sleep_never") or []:
        b = roots.get(key)
        if b is None:
            continue
        spec.body(b).sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
        out.append(b)
    return out


def set_geom_defaults(spec, physics):
    gd = spec.default.geom
    G = physics["geom_defaults"]
    gd.condim = int(G["condim"])
    gd.friction = [float(v) for v in G["friction"]]
    gd.solref = [float(v) for v in G["solref"]]
    gd.solimp = [float(v) for v in G["solimp"]]


def physics_report(m, physics, tick_s):
    """what the compiled model runs (for the record and the tests): the options, the pads' solref"""
    import mujoco
    return {"timestep": float(m.opt.timestep), "tick_steps": int(round(tick_s / m.opt.timestep)),
            "integrator": int(m.opt.integrator), "cone": int(m.opt.cone), "impratio": float(m.opt.impratio),
            "noslip_iterations": int(m.opt.noslip_iterations), "gravity": float(-m.opt.gravity[2]),
            "multiccd": not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
            "sleep": bool(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)),
            "sleep_tolerance": float(m.opt.sleep_tolerance),
            "varstep": bool((physics.get("varstep") or {}).get("enabled"))}


def load_kit(kit_dir):
    with open(os.path.join(kit_dir, "kit.json"), encoding="utf-8") as f:
        kit = json.load(f)
    kit["dir"] = os.path.abspath(kit_dir)
    return kit


def check_kit(kit):
    """the fields this cell reads (KIT_FORMAT.md), or ValueError"""
    if kit.get("kit_version") != KIT_VERSION:
        raise ValueError("kit: kit_version %r, this cell reads %s" % (kit.get("kit_version"), KIT_VERSION))
    B = kit["bodies"]
    base = kit.get("base_body")
    if base not in B or not B[base].get("static"):
        raise ValueError("kit: no static base body %r" % base)
    for u in kit["loose"]:
        if u not in B or B[u].get("joint") != "free" or B[u].get("parent"):
            raise ValueError("kit: loose unit %s is not a free root body" % u)
    for bn, bb in B.items():
        if bb.get("parent"):
            raise ValueError("kit: body %s has a parent (every body of this kit is a root body)" % bn)
        for g in bb.get("collision", []):
            if g.get("type") not in GEOM_TYPES:
                raise ValueError("kit: geom %s of %s has type %r" % (g.get("name"), bn, g.get("type")))
            if g.get("role", "col") not in STATE_GEOM:
                raise ValueError("kit: geom %s of %s has role %r" % (g.get("name"), bn, g.get("role")))
    for s in kit["slots"]:
        if s["carrier"] != base:
            raise ValueError("kit: slot %s carrier %s is not the base body" % (s["id"], s["carrier"]))
        for c in s["candidates"]:
            if c["part"] not in B:
                raise ValueError("kit: slot %s candidate %s has no body" % (s["id"], c["part"]))
    if not any(g.get("role") == "dome" for u in kit["loose"] for g in B[u].get("collision", [])):
        raise ValueError("kit: no dome geom (role 'dome') on any loose unit")
    return True


def _rgba(kit, bn, key=None):
    mats = kit.get("materials") or {}
    if key is not None and key in mats:
        m = mats[key]
        return list(m["rgba"] if isinstance(m, dict) else m)
    if key is not None and key.split("__")[0] in MATERIAL_RGBA:
        return list(MATERIAL_RGBA[key.split("__")[0]])
    return list(UNIT_RGBA.get(kit["bodies"][bn].get("type"), [0.5, 0.5, 0.5, 1.0]))


# ====================================================================================================== the budget
class Budget(object):
    """the episode's budgets (module doc). caps: CELL_CONFIG["budget"]; the turns are counted by the positioner"""

    KEYS = ("sim_s", "wall_s", "frames", "step_ticks", "turns")

    def __init__(self, caps, tick_s):
        self.caps = {k: caps[k] for k in self.KEYS}
        self.tick_s = float(tick_s)
        self.ticks = 0
        self.step_calls = 0
        self.max_step_ticks = 0
        self.refused_steps = 0
        self.frames = {}
        self.illegal = {}
        self.turns = 0
        self.wall0 = None
        self.wall_extra = 0.0

    def tick(self):
        self.ticks += 1

    def sim_s(self):
        return self.ticks * self.tick_s

    def charge_step(self, n):
        """a step(n) call: True if n is inside the per-call cap (the ticks are charged as they run)"""
        n = int(n)
        if n < 0 or n > int(self.caps["step_ticks"]):
            self.refused_steps += 1
            return False
        self.step_calls += 1
        self.max_step_ticks = max(self.max_step_ticks, n)
        return True

    def charge_frames(self, camera, k=1):
        self.frames[camera] = self.frames.get(camera, 0) + int(k)
        return self.frames_used() <= int(self.caps["frames"])

    def frames_used(self):
        return int(sum(self.frames.values()))

    def charge_illegal(self, what):
        self.illegal[what] = self.illegal.get(what, 0) + 1

    def start_wall(self):
        if self.wall0 is None:
            self.wall0 = time.time()

    def wall_s(self):
        return 0.0 if self.wall0 is None else time.time() - self.wall0 + self.wall_extra

    def over(self):
        """the first exhausted budget ('sim_cap', 'wall_cap', 'frames'), or None"""
        if self.sim_s() >= float(self.caps["sim_s"]) - 1e-9:
            return "sim_cap"
        if self.wall0 is not None and self.wall_s() >= float(self.caps["wall_s"]):
            return "wall_cap"
        if self.frames_used() > int(self.caps["frames"]):
            return "frames"
        return None

    def state(self):
        return {"ticks": self.ticks, "sim_s": round(self.sim_s(), 6), "step_calls": self.step_calls,
                "max_step_ticks": self.max_step_ticks, "refused_steps": self.refused_steps,
                "frames": dict(sorted(self.frames.items())), "frames_used": self.frames_used(),
                "illegal_calls": dict(sorted(self.illegal.items())), "illegal_total": int(sum(self.illegal.values())),
                "turns": self.turns, "caps": dict(self.caps), "over": self.over()}


# ====================================================================================================== the spec
def _add_body_geoms(spec, body, kit, bn, cfg, meshes, prefix=""):
    """the kit body's collision geoms (mass 0, the config's contact defaults through the spec's default class) and,
    with cfg["visual"], its visual meshes; returns {"collision": [names], "dome": [names], "visual": {key: name},
    "fasteners": [names]}"""
    import mujoco
    kd = kit["dir"]
    bb = kit["bodies"][bn]
    show = bool(cfg.get("show_collision"))
    out = {"collision": [], "dome": [], "visual": {}, "fasteners": []}
    gt = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "mesh": mujoco.mjtGeom.mjGEOM_MESH}
    for g in bb.get("collision", []):
        kw = {"name": prefix + g["name"], "type": gt[g["type"]], "pos": [float(v) for v in g.get("pos", [0, 0, 0])],
              "quat": [float(v) for v in g.get("quat", [1, 0, 0, 0])], "mass": 0.0, "group": 1 if show else 3}
        if g["type"] == "mesh":
            mn = "col_" + os.path.splitext(os.path.basename(g["file"]))[0]
            if mn not in meshes:
                spec.add_mesh(name=mn, file=os.path.join(kd, g["file"]))
                meshes.add(mn)
            kw["meshname"] = mn
        elif g["type"] == "sphere":
            kw["size"] = [float(g["size"][0]), 0.0, 0.0]
        else:
            kw["size"] = [float(v) for v in g["size"]]
        rgba = _rgba(kit, bn)
        kw["rgba"] = rgba if show else [rgba[0], rgba[1], rgba[2], 0.4]
        body.add_geom(**kw)
        out["collision"].append(kw["name"])
        if g.get("role") == "dome":
            out["dome"].append(kw["name"])
    if not cfg.get("visual", True):
        return out
    for key, f in sorted((bb.get("visual") or {}).items()):
        mn = "vis_" + os.path.splitext(os.path.basename(f))[0]
        if mn not in meshes:
            spec.add_mesh(name=mn, file=os.path.join(kd, f))
            meshes.add(mn)
        nm = "%svis_%s_%s" % (prefix, bn, key)
        mat = "mat_vis_%s_%s" % (bb.get("type"), key)
        if mat not in meshes:
            spec.add_material(name=mat, rgba=_rgba(kit, bn, key))
            meshes.add(mat)
        body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, contype=0, conaffinity=0, group=2,
                      material=mat, mass=0.0)
        out["visual"][key] = nm
    for k, f in enumerate(bb.get("visual_fasteners") or []):
        mn = "vis_" + os.path.splitext(os.path.basename(f))[0]
        if mn not in meshes:
            spec.add_mesh(name=mn, file=os.path.join(kd, f))
            meshes.add(mn)
        nm = "%svisf_%s_%d" % (prefix, bn, k)
        rgba = list(MATERIAL_RGBA["steel"])
        rgba[3] = 0.0                                   # hidden until fastened (update_fastener_visuals)
        body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, contype=0, conaffinity=0, group=2,
                      rgba=rgba, mass=0.0)
        out["fasteners"].append(nm)
    return out


def _set_inertia(body, bb):
    body.mass = float(bb["mass_kg"])
    body.ipos = [float(v) for v in bb["com_m"]]
    body.iquat = [float(v) for v in bb["inertia_quat"]]
    body.inertia = [max(float(v), 1e-10) for v in bb["inertia_diag"]]
    body.explicitinertial = True


def _T_pose(p):
    pos, quat = p
    T = np.eye(4)
    T[:3, :3] = LY._quat_R(quat)
    T[:3, 3] = np.asarray(pos, float)
    return T


def build_spec(kit, layout, cfg, fx, robot_hook=None):
    """(MjSpec, robot dict, positioner Handles, names) of the cell (module docstring)"""
    import mujoco
    P = cfg["physics"]
    spec = mujoco.MjSpec()
    spec.modelname = "rov_final_assembly_cell"
    apply_physics(spec, P)
    set_geom_defaults(spec, P)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.stat.extent, spec.stat.center = 1.8, [0.45, 0.0, 0.3]
    w = spec.worldbody
    w.add_light(name="sun", pos=[0.4, -0.4, 2.5], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.6, 0.8, 1.6], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    for name, rgba in cfg["materials"].items():
        spec.add_material(name="mat_" + name, rgba=list(rgba))
    tb = FX.TABLE
    (x0, x1), (y0, y1), th = tb["x_m"], tb["y_m"], float(tb["thickness_m"])
    w.add_geom(name="bench_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, th / 2],
               pos=[(x0 + x1) / 2, (y0 + y1) / 2, -th / 2], material="mat_bench")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1], pos=[0.0, 0.0, float(cfg["floor_z"])],
               material="mat_floor")
    names = {"stations": {}, "units": {}, "vehicle": None, "boards": []}
    for st in FX.STATIONS:
        pose = (layout.get("stations") or {}).get(st)
        if pose is None:
            raise ValueError("layout: no pose for the station %s" % st)
        names["stations"][st] = fx.add_station(spec, st, LY.T_world_station(pose))
    for bn, Tb in sorted((layout.get("boards") or {}).items()):
        b = fx.boards()[bn]
        Tw = np.asarray(Tb, float)
        bx, by, bt = (v / 1000.0 for v in b["plate_mm"])
        w.add_geom(name="board_" + bn, type=mujoco.mjtGeom.mjGEOM_BOX, size=[bx / 2, by / 2, bt / 2],
                   pos=(Tw[:3, 3] + Tw[:3, :3] @ np.array([0.0, 0.0, bt / 2])).tolist(), quat=quat_wxyz(Tw[:3, :3]).tolist(),
                   contype=0, conaffinity=0, group=2, material="mat_board")
        names["boards"].append("board_" + bn)
    meshes = set()
    base = kit["base_body"]
    vb = kit["bodies"][base]

    def add_vehicle_geoms(sp, body):
        names["vehicle"] = _add_body_geoms(sp, body, kit, base, cfg, meshes)

    pcfg = PZ.merge_cfg(dict(cfg["positioner"], start_index=cfg["start_index"]))
    H = PZ.build_positioner(spec, fx, vehicle={"mass_kg": vb["mass_kg"], "cg_B_m": vb["com_m"],
                                               "inertia_about_com_B_kgm2": _inertia_matrix(vb)},
                            mount_T=LY.mount_T(layout), cfg=pcfg, add_vehicle_geoms=add_vehicle_geoms,
                            physics=positioner_physics(P))
    if H.vehicle != base:
        raise ValueError("the positioner's vehicle body %r is not the kit's base body %r" % (H.vehicle, base))
    parts = layout.get("parts") or {}
    prop = {n["unit"]: float(n.get("prop_deg", 0.0)) for n in (layout.get("nests") or [])}
    for u in kit["loose"]:
        if u not in parts:
            raise ValueError("layout: no pose for the loose unit %s" % u)
        T = _T_pose(parts[u])
        body = w.add_body(name=u, pos=T[:3, 3].tolist(), quat=quat_wxyz(T[:3, :3]).tolist())
        body.add_freejoint(name="free_" + u)
        names["units"][u] = _add_body_geoms(spec, body, kit, u, cfg, meshes)
        _set_inertia(body, kit["bodies"][u])
        pg = names["units"][u]["visual"].get("propeller")
        if pg is not None and u in prop:
            g = spec.geom(pg)
            g.quat = quat_wxyz(_rotz(math.radians(prop[u]))[:3, :3]).tolist()
    robot = {}
    if robot_hook is None and cfg.get("robot"):
        from . import gripper_wide as GW
        rc = dict(cfg["robot"])
        rc["design"] = dict(rc.get("design") or {}, solref=list(P["pad_solref"]))
        rc["impratio"] = float(P["impratio"])
        robot_hook = lambda sp, c: GW.add_ur_wide(sp, rc)       # noqa: E731
    if robot_hook is not None:
        robot = robot_hook(spec, cfg) or {}
    for s in kit["slots"]:
        for c in s["candidates"]:
            eq = spec.add_equality(name=weld_name(s["id"], c["unit"]), type=mujoco.mjtEq.mjEQ_WELD,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, name1=s["carrier"], name2=c["part"])
            eq.active = False
            eq.solref = [float(v) for v in P["weld_solref"]]
            eq.solimp = [float(v) for v in P["weld_solimp"]]
            eq.data = [0.0] * 10 + [1.0]            # anchor at body2's origin (MjSpec's default puts it 1 m off)
    for b1, b2 in [tuple(p) for p in kit.get("excludes", [])]:
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = b1, b2
    apply_physics(spec, P)                          # again: the robot's attachment brings options of its own
    arm = robot.get("arm_bodies") or []
    names["sleep_never"] = apply_sleep_policy(spec, P, {"positioner": H.names["base"],
                                                        "robot": arm[1] if len(arm) > 1 else None})   # the first moving link
    return spec, robot, H, names


def _inertia_matrix(bb):
    """the 3x3 inertia about the COM in the body frame from the kit's principal values and their frame"""
    q = bb["inertia_quat"]
    R = LY._quat_R(q)
    return (R @ np.diag([float(v) for v in bb["inertia_diag"]]) @ R.T).tolist()


# ====================================================================================================== the cell
class RovCell(object):
    def __init__(self, kit_dir=None, layout=None, config=None, log=print, robot_hook=None, spec_hook=None, settle_s=None,
                 robot_init=None, kit=None, fx=None, tare=None):
        import mujoco
        gc.collect()                                # one cell after another under ulimit -v (U3 section 3.6)
        self.mj, self.np, self.log = mujoco, np, log
        self.kit = kit if kit is not None else load_kit(kit_dir)
        check_kit(self.kit)
        self.kit_dir = self.kit["dir"]
        self.cfg = merge_cfg(config)
        P = self.cfg["physics"]
        check_physics(P)
        self.physics = P
        if layout is None:
            raise ValueError("RovCell needs an episode layout (layout.episode_layout)")
        self.layout = layout
        self.fx = fx if fx is not None else FX.Fixtures(FX.load_cell_layout(self.cfg["cell_layout"]),
                                                        FX.load_facts(self.cfg["fixture_facts"]), lmag=bool(self.cfg["lmag"]))
        if bool(self.fx.lmag) != bool(layout.get("lmag", False)):
            raise ValueError("the fixtures' L-mag (%s) is not the layout's (%s)" % (self.fx.lmag, layout.get("lmag")))
        t0 = time.time()
        spec, robot, H, names = build_spec(self.kit, layout, self.cfg, self.fx, robot_hook=robot_hook)
        if spec_hook is not None:
            spec_hook(spec)
        m = spec.compile()
        self.spec, self.m = spec, m
        d = mujoco.MjData(m)
        self.d = d
        self.robot, self.H, self.names = robot, H, names
        self.loose = list(self.kit["loose"])
        self.base = self.kit["base_body"]
        self.tick_steps = int(P["tick_steps"])
        self.tick_s = tick_s_of(P)
        self.dt_coarse = float(P["timestep"])
        # ---- the positioner
        pcfg = PZ.merge_cfg(dict(self.cfg["positioner"], start_index=self.cfg["start_index"]))
        self.robot_geoms = self._robot_geoms()
        if tare is None:
            tare = PZ.compute_tare(self.fx, pcfg, positioner_physics(P))
        self.P = PZ.Positioner(m, d, H, self.fx, pcfg, units=self.loose, is_fastened=self.is_fastened,
                               on_scrap=self._on_turn_scrap, robot_geoms=self.robot_geoms, tare=tare)
        self.P.place(self.cfg["start_index"])
        mujoco.mj_forward(m, d)
        # ---- the robot
        if robot_init is None and self.robot:
            robot_init = _ur_init
        self.ctrls = robot_init(self) if robot_init is not None else {}
        self.ctrl = self.ctrls.get("ur") if self.ctrls else None
        self.physics_mismatch = None
        if self.robot:
            from . import gripper_wide as GW
            self.physics_mismatch = GW.physics_mismatch(m, self.robot["prefix"], self.tick_s)
        # ---- the coupling
        ckw = dict(self.cfg.get("coupling") or {})
        ckw.setdefault("tick_s", self.tick_s)
        self.sm = CouplingStateMachine(m, d, self.kit, **ckw)
        self.slots = {s["id"]: s for s in self.kit["slots"]}
        unit_of = {bn: bb["unit"] for bn, bb in self.kit["bodies"].items()}
        self.unit_of_body = {b: unit_of[m.body(b).name] for b in range(m.nbody) if m.body(b).name in unit_of}
        self.base_unit = unit_of[self.base]
        self.geoms_of = {u: [] for u in self.loose + [self.base_unit]}
        for g in range(m.ngeom):
            u = self.unit_of_body.get(int(m.geom_bodyid[g]))
            if u in self.geoms_of and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geoms_of[u].append(g)
        self.col_bits = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom)}
        self.sm.collision_bits = self.col_bits
        self.n_fastened = -1
        self.clusters = []
        self.fastener_geoms = {u: [m.geom(n).id for n in names["units"][u]["fasteners"]] for u in self.loose}
        self.fastener_alpha = {}
        # ---- the variable step (M8's second fallback)
        self.n_tick = 0
        vs = P.get("varstep") or {}
        self.varstep = bool(vs.get("enabled"))
        self.vs_since = 10 ** 9
        self.vs_mode = None
        self.vs_log = []
        self.vs_fine_ticks = 0
        if self.varstep:
            self.vs_hold_ticks = int(round(float(vs["holdover_s"]) / self.tick_s))
            self._set_step("coarse")
        # ---- recording, hooks, jigs, watch, budget
        self.rec_t, self.rec_q, self.rec_s = [], [], []
        self.label = {"step": 0, "text": "start", "part": None, "slot": None}
        self.hooks = []
        self.recording = True
        self.jig = {}
        self.budget = Budget(self.cfg["budget"], self.tick_s)
        self.done = None
        self._init_watch()
        self.update_clusters()
        rec, self.recording = self.recording, False
        settle = float(self.cfg["settle_s"] if settle_s is None else settle_s)
        for _ in range(int(round(settle / self.tick_s))):
            self.tick_one_physics()
        self.recording = rec
        mujoco.mj_forward(m, d)
        d.time = 0.0
        self.initial = {u: self.body_T(u) for u in self.loose}
        self.build_s = time.time() - t0
        log("[cell] rov cell built and settled in %.1f s (%d bodies, %d geoms, %d welds)" % (
            self.build_s, m.nbody, m.ngeom, m.neq))

    # ------------------------------------------------------------------ helpers
    def _robot_geoms(self):
        if not self.robot:
            return []
        m = self.m
        rb = m.body(self.robot["base_body"]).id
        sub = set([rb])
        for b in range(m.nbody):
            p = b
            while p > 0 and p not in sub:
                p = int(m.body_parentid[p])
            if p in sub:
                sub.add(b)
        return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub]

    def body_T(self, name):
        return body_pose(self.d, self.m.body(name).id)

    def vehicle_T(self):
        return self.body_T(self.base)

    def is_fastened(self, u):
        for sid, st in self.sm.rec.items():
            if st["fastened"] and st["unit"] == u and not st["void"]:
                return True
        return False

    def fastened_slot(self, u):
        for sid, st in self.sm.rec.items():
            if st["fastened"] and st["unit"] == u:
                return sid
        return None

    def place(self, unit, T, forward=True):
        """put a loose unit at the world pose T at rest (a test's or a check's start)"""
        m, d = self.m, self.d
        j = m.joint("free_" + unit).id
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[a:a + 3] = np.asarray(T, float)[:3, 3]
        d.qpos[a + 3:a + 7] = quat_wxyz(np.asarray(T, float)[:3, :3])
        d.qvel[v:v + 6] = 0.0
        if forward:
            self.mj.mj_forward(m, d)

    def save_model(self, out_dir):
        with open(os.path.join(out_dir, "cell.xml"), "w") as f:
            f.write(self.spec.to_xml())

    def save_trajectory(self, out_dir):
        labels = np.array([r["label"].get("text", "") for r in self.rec_s])
        score = np.array([r["score"] for r in self.rec_s], np.float32)
        np.savez_compressed(os.path.join(out_dir, "trajectory.npz"), t=np.array(self.rec_t), q=np.array(self.rec_q),
                            label=labels, score=score)

    def record(self):
        self.rec_t.append(float(self.d.time))
        self.rec_q.append(np.array(self.d.qpos, np.float32))
        self.rec_s.append({"states": dict(self.sm.state), "score": self.sm.score(), "label": dict(self.label)})

    # ------------------------------------------------------------------ the variable step
    def _set_step(self, mode):
        if mode == self.vs_mode:
            return
        P = self.physics
        vs = P["varstep"]
        dt = float(vs["fine_timestep"]) if mode == "fine" else self.dt_coarse
        m = self.m
        m.opt.timestep = dt
        n = int(round(self.tick_s / dt))
        if self.robot:
            from . import gripper_wide as GW
            sr = float(vs["pad_solref_steps"]) * dt if mode == "fine" else float(P["pad_solref"][0])
            GW.set_pad_solref(m, [sr, float(P["pad_solref"][1])], self.robot["prefix"])
        for ct in (self.ctrls or {}).values():
            ct.n_sub = n
        self.n_sub = n
        self.vs_mode = mode
        self.vs_log.append((self.n_tick, float(self.d.time), mode))

    def _varstep_choose(self):
        """G0b's 'cmd' rule: fine while the commanded width goal is below the open width - open_margin_m and for
        holdover_s after it was raised again (the command for this tick is set before the tick runs)"""
        ct = self.ctrl
        vs = self.physics["varstep"]
        holding = ct is not None and float(ct.width_goal) < float(ct.max_width) - float(vs["open_margin_m"])
        self.vs_since = 0 if holding else self.vs_since + 1
        self._set_step("fine" if self.vs_since <= self.vs_hold_ticks else "coarse")
        self.vs_fine_ticks += self.vs_mode == "fine"

    # ------------------------------------------------------------------ the tick
    def tick_one_physics(self):
        """one control tick of the controllers and physics only (the settling before t = 0 uses it alone)"""
        m, d, mj = self.m, self.d, self.mj
        if self.varstep:
            self._varstep_choose()
        for ct in self.ctrls.values():
            ct.tick()
        for _name, (k, T) in self.jig.items():
            d.mocap_pos[k] = T[:3, 3]
            d.mocap_quat[k] = quat_wxyz(T[:3, :3])
        n = int(round(self.tick_s / m.opt.timestep))
        ct = self.ctrl
        P = self.P
        for _s in range(n):
            P.before_step()
            mj.mj_step1(m, d)
            if ct is not None:
                ct.torque()
            mj.mj_step2(m, d)

    def tick(self, n=1):
        for _ in range(int(n)):
            if self.done:
                return
            self.tick_one()

    def tick_one(self):
        m, d = self.m, self.d
        self.tick_one_physics()
        self.mj.mj_forward(m, d)
        self.sm.tick(float(d.time))
        if self.cfg.get("clusters", True):
            nf = sum(1 for x in self.sm.rec.values() if x["fastened"])
            if nf != self.n_fastened:
                self.update_clusters()
        if self.cfg["watch"].get("enabled", True):
            self.watch_tick()
        self.n_tick += 1
        self.budget.tick()
        self.budget.turns = self.P.turns_used
        if self.done is None:
            ov = self.budget.over()
            if ov == "sim_cap":
                self.done = ov
        for h in self.hooks:
            h()
        if self.recording and self.n_tick % int(self.cfg["record_every"]) == 0:
            self.record()

    def step(self, n):
        """the API's step(n): refused (False) above the per-call cap; otherwise n ticks (fewer if the sim cap ends
        the episode); returns True"""
        if not self.budget.charge_step(n):
            return False
        self.tick(int(n))
        return True

    def fixture_index(self, target):
        """the API's fixture_index(target): the positioner's answer; the units it scraps are logged by the watch"""
        ans = self.P.request_index(target)
        self.budget.turns = self.P.turns_used
        return ans

    def _on_turn_scrap(self, u, why):
        t = float(self.d.time)
        if self.sm.scrap(u, t, why):
            self.watch_events.append((t, "scrapped:%s:%s" % (u, why)))

    # ------------------------------------------------------------------ contact clusters, fastener visuals
    def update_clusters(self):
        """every rigidly fastened group (the vehicle and the units fastened on it) gets a collision bit of its own:
        nothing inside it collides, it still collides with everything else (the precedent cell's rule)"""
        m = self.m
        parent = {u: u for u in self.geoms_of}

        def root(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        if self.cfg.get("clusters", True):
            for sid, st in self.sm.rec.items():
                if st["fastened"] and st["unit"] is not None:
                    cu = self.kit["bodies"][self.slots[sid]["carrier"]]["unit"]
                    if cu in parent and st["unit"] in parent:
                        parent[root(st["unit"])] = root(cu)
        groups = {}
        for u in self.geoms_of:
            groups.setdefault(root(u), []).append(u)
        clusters = sorted([sorted(v) for v in groups.values() if len(v) > 1],
                          key=lambda v: (0 if self.base_unit in v else 1, v))
        allbits = (1 << 31) - 1
        bits = {}
        for i, cl in enumerate(clusters[:30]):
            b = 1 << (i + 1)
            for u in cl:
                for g in self.geoms_of[u]:
                    bits[g] = (b, allbits & ~b)
        for u, gs in self.geoms_of.items():
            for g in gs:
                ct, ca = bits.get(g, (1, allbits))
                self.col_bits[g] = (ct, ca)
                if m.geom_contype[g] or m.geom_conaffinity[g]:
                    m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        self.clusters = clusters
        self.n_fastened = sum(1 for x in self.sm.rec.values() if x["fastened"])
        self.update_fastener_visuals()

    def update_fastener_visuals(self):
        """a unit's station fasteners are drawn once it is fastened (and not scrapped)"""
        m = self.m
        for u, gs in self.fastener_geoms.items():
            a = 1.0 if (self.is_fastened(u) and u not in self.sm.scrapped) else 0.0
            if self.fastener_alpha.get(u) == a:
                continue
            for g in gs:
                m.geom_rgba[g, 3] = a
            self.fastener_alpha[u] = a

    # ------------------------------------------------------------------ the watch
    def _init_watch(self):
        m = self.m
        prefix = (self.robot or {}).get("prefix")
        pads = set(m.geom(n).id for n in (self.robot or {}).get("pad_geoms", []))
        loose = set(self.loose)
        # geom classes: 0 environment (world, bench, stations, positioner, vehicle), 1 robot link, 2 pad, 3 loose unit
        self.gclass = np.zeros(m.ngeom, np.int64)
        self.gunit_idx = np.full(m.ngeom, -1, np.int64)
        self.unit_index = {u: i for i, u in enumerate(self.loose)}
        rgeoms = set(self.robot_geoms)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if g in pads:
                self.gclass[g] = 2
            elif g in rgeoms or (prefix and m.body(b).name.startswith(prefix)):
                self.gclass[g] = 1
            elif self.unit_of_body.get(b) in loose:
                self.gclass[g] = 3
                self.gunit_idx[g] = self.unit_index[self.unit_of_body[b]]
        self.dome = np.zeros(m.ngeom, bool)
        self.dome_unit = None
        for u in self.loose:
            for nm in self.names["units"][u]["dome"]:
                self.dome[m.geom(nm).id] = True
                self.dome_unit = u
        self.unit_body = {u: m.body(u).id for u in self.loose}
        self.unit_body_ids = np.array([self.unit_body[u] for u in self.loose], np.int64)
        self.geom_body = np.asarray(m.geom_bodyid, np.int64)
        self.elliptic = int(m.opt.cone) == int(self.mj.mjtCone.mjCONE_ELLIPTIC)
        self._f6 = np.zeros(6)
        self.flange_sensor = None
        if self.robot and self.robot.get("force_sensor"):
            sid = m.sensor(self.robot["force_sensor"]).id
            self.flange_sensor = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))
        self.reset_watch()

    def reset_watch(self):
        n = int(self.cfg["watch"]["part_force_ticks"])
        self.violations = []
        self.watch_events = []
        self.max_robot_force = 0.0
        self.max_robot_force_what = None
        self.max_robot_contact = 0.0
        self.max_part_force = 0.0
        self.max_part_force_what = None
        self.max_part_sustained = 0.0
        self.max_part_sustained_what = None
        self.max_pad_force = 0.0
        self.max_dome_force = 0.0
        self.max_dome_sustained = 0.0
        self.max_flange_force = 0.0
        self.max_flange_sustained = 0.0
        self.max_speed = np.zeros(len(self.loose))
        self.column_ticks_turning = 0
        self._part_hist = [0.0] * n
        self._flange_hist = [0.0] * n
        self._hist_k = 0
        self._hist_n = 0
        self._part_over = 0
        self._dome_over = 0
        self._dome_hist = [0.0] * max(1, int(self.cfg["watch"]["dome_ticks"]))
        self._dome_k = 0
        self._dome_n = 0

    def _contact_normals(self, nc):
        """(geom1, geom2, |normal force|) of the tick's contacts (excluded contacts left out)"""
        d = self.d
        con = d.contact
        g1 = np.asarray(con.geom1[:nc], np.int64)
        g2 = np.asarray(con.geom2[:nc], np.int64)
        adr = np.asarray(con.efc_address[:nc], np.int64)
        ok = adr >= 0
        fn = np.zeros(nc)
        if self.elliptic:
            fn[ok] = np.abs(d.efc_force[adr[ok]])
        else:
            for i in np.nonzero(ok)[0]:
                self.mj.mj_contactForce(self.m, d, int(i), self._f6)
                fn[i] = abs(float(self._f6[0]))
        return g1[ok], g2[ok], fn[ok]

    def _contact_forces(self, nc):
        """(geom1, geom2, |normal force|, F) of the tick's contacts (excluded contacts left out): F (k x 3) the contact
        force in the world frame that geom1 exerts on geom2 (normal and tangential components; the contact frame's
        first row is the normal from geom1 to geom2, its rows two and three the tangents; a torsional or rolling
        component is a torque and not part of F)"""
        d = self.d
        con = d.contact
        g1 = np.asarray(con.geom1[:nc], np.int64)
        g2 = np.asarray(con.geom2[:nc], np.int64)
        adr = np.asarray(con.efc_address[:nc], np.int64)
        dim = np.asarray(con.dim[:nc], np.int64)
        fr = np.asarray(con.frame[:nc], float).reshape(-1, 3, 3)
        ok = adr >= 0
        f3 = np.zeros((nc, 3))
        if self.elliptic:
            ef = d.efc_force
            f3[ok, 0] = ef[adr[ok]]
            tan = ok & (dim >= 3)
            f3[tan, 1] = ef[adr[tan] + 1]
            f3[tan, 2] = ef[adr[tan] + 2]
        else:
            for i in np.nonzero(ok)[0]:
                self.mj.mj_contactForce(self.m, d, int(i), self._f6)
                f3[i] = self._f6[:3]
        F = np.einsum("ki,kij->kj", f3, fr)
        return g1[ok], g2[ok], np.abs(f3[ok, 0]), F[ok]

    def watch_tick(self):
        """the hard rules, the dome rule and the scrap rules S1 / S2 for this tick (module doc); returns its events"""
        m, d = self.m, self.d
        W = self.cfg["watch"]
        t = float(d.time)
        events = []
        nc = int(d.ncon)
        pm, pk = 0.0, None
        dome_sum = 0.0
        if nc:
            g1, g2, fn, F = self._contact_forces(nc)
            c1, c2 = self.gclass[g1], self.gclass[g2]
            rob1, rob2 = (c1 == 1), (c2 == 1)
            robany1, robany2 = (c1 == 1) | (c1 == 2), (c2 == 1) | (c2 == 2)
            # H1: a robot link (not a pad) on a body that is not the robot
            mask = (rob1 & ~robany2) | (rob2 & ~robany1)
            if mask.any():
                rob_sum = {}
                for i in np.nonzero(mask)[0]:
                    a, b = (g1[i], g2[i]) if rob1[i] else (g2[i], g1[i])
                    key = (int(self.geom_body[a]), int(self.geom_body[b]))
                    rob_sum[key] = rob_sum.get(key, 0.0) + float(fn[i])
                    if fn[i] > self.max_robot_contact:
                        self.max_robot_contact = float(fn[i])
                thr = float(W["robot_force_N"])
                for (lb, ob), f in rob_sum.items():
                    what = "%s:%s" % (m.body(lb).name, m.body(ob).name)
                    if f > self.max_robot_force:
                        self.max_robot_force, self.max_robot_force_what = f, what
                    if f > thr:
                        v = "robot_contact:%s:%.0fN" % (what, f)
                        last = [x for x in self.violations[-5:] if x["what"].startswith("robot_contact:" + what + ":")]
                        if not last or last[-1]["t"] < t - 0.5:
                            self.violations.append({"t": t, "what": v, "force_N": f})
                            events.append("safety:" + v)
            # the pads' largest single contact (a metric)
            pad = (c1 == 2) | (c2 == 2)
            if pad.any():
                self.max_pad_force = max(self.max_pad_force, float(fn[pad].max()))
            # H2: a unit pressed on anything but the robot (contacts only): per (unit, body it presses on) the
            # magnitude of the vector sum of that pair's contact forces on the unit (F acts on geom2: + on a unit that
            # is geom2, - on a unit that is geom1; a unit-unit contact counts for both units)
            pmask = ~(robany1 | robany2) & ((c1 == 3) | (c2 == 3))
            if pmask.any():
                nb = int(m.nbody)
                keys, vecs = [], []
                for gp, go, cp, sgn in ((g2, g1, c2, 1.0), (g1, g2, c1, -1.0)):
                    sel = pmask & (cp == 3)
                    if sel.any():
                        keys.append(self.gunit_idx[gp[sel]] * nb + self.geom_body[go[sel]])
                        vecs.append(sgn * F[sel])
                keys = np.concatenate(keys)
                vecs = np.concatenate(vecs)
                uk, inv = np.unique(keys, return_inverse=True)
                tot = np.zeros((len(uk), 3))
                np.add.at(tot, inv, vecs)
                mag = np.sqrt((tot * tot).sum(axis=1))
                j = int(np.argmax(mag))
                if mag[j] > pm:
                    pm, pk = float(mag[j]), (int(uk[j] // nb), int(uk[j] % nb))
            # H3: everything on the dome
            dm = self.dome[g1] ^ self.dome[g2]
            if dm.any():
                dome_sum = float(fn[dm].sum())
        pkn = None if pk is None else (self.loose[pk[0]], m.body(pk[1]).name)
        if pm > self.max_part_force:
            self.max_part_force, self.max_part_force_what = pm, pkn
        n = int(W["part_force_ticks"])
        flange = 0.0
        if self.flange_sensor is not None:
            a, k = self.flange_sensor
            flange = float(np.linalg.norm(d.sensordata[a:a + k]))
            self.max_flange_force = max(self.max_flange_force, flange)
        self._part_hist[self._hist_k] = pm
        self._flange_hist[self._hist_k] = flange
        self._hist_k = (self._hist_k + 1) % n
        self._hist_n += 1
        if self._hist_n >= n:
            s = min(self._part_hist)
            if s > self.max_part_sustained:
                self.max_part_sustained, self.max_part_sustained_what = s, pkn
            self.max_flange_sustained = max(self.max_flange_sustained, min(self._flange_hist))
        if pm > float(W["part_force_N"]):
            self._part_over += 1
            if self._part_over >= n:
                v = "part_force:%s:%s:%.0fN" % (pkn[0], pkn[1], pm)
                last = [x for x in self.violations[-5:] if x["what"].startswith("part_force:%s:" % pkn[0])]
                if not last or last[-1]["t"] < t - 0.5:
                    self.violations.append({"t": t, "what": v, "force_N": pm})
                    events.append("safety:" + v)
        else:
            self._part_over = 0
        # H3: the dome
        nd = len(self._dome_hist)
        self._dome_hist[self._dome_k] = dome_sum
        self._dome_k = (self._dome_k + 1) % nd
        self._dome_n += 1
        self.max_dome_force = max(self.max_dome_force, dome_sum)
        if self._dome_n >= nd:
            self.max_dome_sustained = max(self.max_dome_sustained, min(self._dome_hist))
        if dome_sum > float(W["dome_force_N"]):
            self._dome_over += 1
            if self._dome_over >= nd and self.dome_unit is not None and self.dome_unit not in self.sm.scrapped and W.get("scrap", True):
                why = "dome pressed with %.1f N for %.2f s" % (dome_sum, nd * self.tick_s)
                if self.sm.scrap(self.dome_unit, t, why):
                    events.append("scrapped:%s:%s" % (self.dome_unit, why))
                    self.update_fastener_visuals()
        else:
            self._dome_over = 0
        # S1 / S2
        sp = np.linalg.norm(d.cvel[self.unit_body_ids, 3:], axis=1)
        np.maximum(self.max_speed, sp, out=self.max_speed)
        if W.get("scrap", True):
            z_off = -float(W["scrap_below_bench_m"])
            vmax = float(W["scrap_speed_m_s"])
            zc = d.xipos[self.unit_body_ids, 2]
            bad = np.nonzero((zc < z_off) | (sp > vmax))[0]
            for i in bad:
                u = self.loose[int(i)]
                if u in self.sm.scrapped:
                    continue
                why = "off the bench" if zc[i] < z_off else "faster than %.1f m/s (%.2f m/s)" % (vmax, sp[i])
                if self.sm.scrap(u, t, why):
                    events.append("scrapped:%s:%s" % (u, why))
                    self.update_fastener_visuals()
        # the column while turning (a metric)
        if W.get("column_while_turning", True) and self.P.moving() and self.robot_geoms:
            if self.P.robot_in_column()[0]:
                self.column_ticks_turning += 1
        for e in events:
            self.watch_events.append((t, e))
        return events

    def watch_summary(self):
        hard = [v for v in self.violations if v["what"].split(":")[0] in ("robot_contact", "part_force")]
        return {"max_robot_link_force_N": round(self.max_robot_force, 3), "max_robot_link_force_pair": self.max_robot_force_what,
                "max_robot_link_contact_N": round(self.max_robot_contact, 3),
                "max_part_force_instant_N": round(self.max_part_force, 3),
                "max_part_force_instant_pair": None if self.max_part_force_what is None else list(self.max_part_force_what),
                "max_part_force_sustained_N": round(self.max_part_sustained, 3),
                "max_part_force_sustained_pair": None if self.max_part_sustained_what is None else list(self.max_part_sustained_what),
                "max_pad_contact_N": round(self.max_pad_force, 3),
                "max_dome_force_N": round(self.max_dome_force, 3), "max_dome_force_sustained_N": round(self.max_dome_sustained, 3),
                "max_flange_force_N": round(self.max_flange_force, 3),
                "max_flange_force_sustained_N": round(self.max_flange_sustained, 3),
                "max_part_speed_m_s": {u: round(float(v), 4) for u, v in zip(self.loose, self.max_speed)},
                "robot_in_column_while_turning_ticks": int(self.column_ticks_turning),
                "scrapped": dict(self.sm.scrapped), "violations": list(self.violations), "hard_ok": not hard}

    def metrics(self):
        """the episode's metrics (DESIGN section 3.4): the watch, the coupling's (welds apart), the positioner, the budget"""
        cm = self.sm.metrics()
        weld = {k: cm.pop(k) for k in list(cm) if k.startswith("weld_")}
        return {"watch": self.watch_summary(), "coupling": cm, "weld": weld,
                "positioner": self.P.report(), "budget": self.budget.state(),
                "varstep": {"enabled": self.varstep, "fine_ticks": int(self.vs_fine_ticks), "switches": len(self.vs_log)}}

    # ------------------------------------------------------------------ determinism
    def state_hash(self):
        """sha256 of the cell's whole state (module doc)"""
        m, d = self.m, self.d
        h = hashlib.sha256()
        for a in (np.float64(d.time), d.qpos, d.qvel, d.act, d.ctrl, d.qacc_warmstart, d.eq_active, m.eq_data, m.eq_solref,
                  m.geom_contype, m.geom_conaffinity, m.actuator_forcerange, np.float64(m.opt.timestep), d.mocap_pos,
                  d.mocap_quat, d.qfrc_applied, d.xfrc_applied):
            h.update(np.ascontiguousarray(a).tobytes())
        st = {"states": self.sm.state, "rec": self.sm.rec, "scrapped": self.sm.scrapped, "bound": self.sm.bound,
              "used": self.sm.used, "hold": self.sm.hold, "pos": [self.P.state, self.P.target, self.P.traj, self.P.turns_used,
                                                               self.P.scrapped], "viol": self.violations,
              "watch": [self._part_over, self._dome_over, round(self.max_part_sustained, 9)], "n": self.n_tick,
              "vs": [self.vs_mode, self.vs_since if self.varstep else None]}
        h.update(json.dumps(st, sort_keys=True, default=str).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ targets
    def candidate(self, sid, unit):
        return [c for c in self.slots[sid]["candidates"] if c["unit"] == unit][0]

    def live_target(self, sid, unit, k=None):
        """{"T": the unit's body pose (world) home on the slot from the vehicle as it is now, "F": the slot's frame on
        the vehicle (world; z along the insertion), "insertion", "k"}; keyed: k (default: the design orientation;
        any_key slots: the nearest one)"""
        s = self.slots[sid]
        c = self.candidate(sid, unit)
        Tc = self.body_T(s["carrier"])
        Fc = np.asarray(s["frame_in_carrier"], float)
        Fp = np.asarray(c["frame_in_part"], float)
        pitch = math.radians(float(s.get("pitch_deg", 360.0)))
        F = Tc @ Fc
        if k is None:
            if s.get("any_key"):
                x = self.sm.by_id[sid]
                cc = [q for q in x["cands"] if q["unit"] == unit][0]
                k = int(self.sm.evaluate(x, cc)["k"])
            else:
                k = int(s.get("design_k", 0))
        T = F @ _rotz(k * pitch) @ _inv(Fp)
        return {"T": T, "carrier": s["carrier"], "F": F, "insertion": F[:3, 2] / np.linalg.norm(F[:3, 2]), "k": int(k)}

    def design_world(self, assignment=None):
        """{body: world pose} of the units at home on the vehicle as it is now; assignment {body: slot} (default: the
        candidates' T_part_in_carrier of the first slot that lists each body, the design part type first)"""
        Tv = self.vehicle_T()
        out = {self.base: Tv}
        if assignment is None:
            assignment = {}
            for s in self.kit["slots"]:
                want = s.get("design_part")
                for c in sorted(s["candidates"], key=lambda c: (want is not None and c.get("type") != want)):
                    if c["part"] not in assignment and s["id"] not in assignment.values():
                        assignment[c["part"]] = s["id"]
                        break
        for b, sid in assignment.items():
            out[b] = Tv @ np.asarray(self.candidate(sid, b)["T_part_in_carrier"], float)
        return out


def _ur_init(cell):
    """the arm's controller (the wide gripper's: force register) at the ready posture (joint mode), the gripper open"""
    from . import gripper_wide as GW
    from .bench_plan import READY_Q
    m, d = cell.m, cell.d
    rb = cell.robot
    q0 = np.array(cell.cfg["robot_home_q"] if cell.cfg.get("robot_home_q") is not None else READY_Q, float)
    for jn, q in zip(rb["joints"], q0):
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q
    GW.set_opening(m, d, GW.MAX_WIDTH_M, prefix=rb["prefix"])
    cell.mj.mj_forward(m, d)
    ct = GW.WideController(m, d, tick_s=cell.tick_s, prefix=rb["prefix"], tcp_site=rb["tcp_site"],
                           force_N=float(cell.cfg["grip_force_N"]))
    ct.q_cmd = q0.copy()
    ct.q_goal = q0.copy()
    ct.command_width(ct.max_width)
    ct.width_cmd = ct.width_goal = ct.max_width
    GW.set_opening(m, d, ct.max_width, prefix=rb["prefix"])
    cell.mj.mj_forward(m, d)
    ct.write_ctrl()
    return {"ur": ct}
