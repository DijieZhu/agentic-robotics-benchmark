"""The scene of the finishing cell: bench, the two Pandas, the cradle, the infeed tray, two shoes, two laces, the box
blank and the lid tray, composed with MjSpec from the building blocks (cell, shoe, lace, carton), plus every physical
query the server and the work order need (PUBLIC CODE).

One batch per episode: every pair of the work order has the same shoe model (size, rows), box and lid; the colours,
the arrival poses and the faults vary per pair. The two shoe bodies, the two lace chains, the blank and the lid are
re-used from pair to pair: a shipped pair is taken away (parked below the bench) and the next one is set down on the
infeed tray at once.

World frame = bench frame: the bench top is the plane z = 0, x along the bench (1.6 m), y across it (1.0 m); both
robot pedestals stand ON the bench (LEFT, the gripper arm, at (-0.05, +0.30); RIGHT, the hook arm, at (-0.55, -0.30)).
"""

import math
import os

import mujoco
import numpy as np

from . import carton as carton_mod
from . import cell as cell_mod
from . import grasp as grasp_mod
from . import lace as lace_mod
from . import shoe as shoe_mod
from .cameras import CAMERA_SPECS, CameraRig, build_cameras
from .control import ArmController, mat_to_quat, quat_mul, quat_to_mat

__all__ = ["LAYOUT", "Scene"]

LAYOUT = {
    "bench_half": (0.80, 0.50, 0.02),
    "arm_bases": {"left": {"pos": (-0.05, 0.30, 0.0), "yaw": -math.pi / 2},
                  "right": {"pos": (-0.55, -0.30, 0.0), "yaw": math.pi / 2}},
    "home_q": {"left": [0.0, -0.40, 0.0, -2.43, 0.0, 2.03, -0.785],
               "right": [0.0, -0.40, 0.0, -2.43, 0.0, 2.03, 2.356]},
    "infeed": {"centre": (0.40, 0.26), "half": (0.24, 0.24), "top_z": 0.006},
    "cradle": {"centre": (-0.36, 0.05), "yaw": -math.pi / 2},
    "blank": {"centre": (0.30, -0.22)},
    "outfeed_x": 0.55,
    "park": {"z": -0.40},
    "timestep": 0.002,
    "tick_s": 0.01,
    "weld_solref": (0.004, 1.0),
}
TICK_S = LAYOUT["tick_s"]
# The cord: 10 mm links on two-hinge joints (bending only; a round cord needs no twist). A lace threaded IN through an
# eyelet and fetched back up through the throat slot turns round the 4.4 mm flap strip between the hole and the slot;
# 15 mm rigid links could not make that U-turn and wedged there (8-11 N on the strip edge), and 10 mm ball joints cost
# 2.8x the run time. Two hinges keep the DOF count of the 15 mm ball chain (2026-11-04: reference lacing of shoe 0
# 13/32 -> 25/32 seeds, physics faults 2 -> 1, at ~0.7x instead of ~0.95x real time).
LACE_SEGMENTS = 63
LACE_SEG_LEN = 0.010
LACE_JOINT = "hinge2"
# the cord's continuum bending stiffness, the same as the earlier 15 mm ball chain's (2e-4 N m/rad per joint), and its
# bending damping, 10x that chain's 4e-6 N m s/rad (a bend relaxes in ~0.2 s): with the lighter damping a tail pulled
# round the clamped sole or against a finger drove the chain unstable within 0.1 s (2 N -> 1300 N, NaN) in about one
# reference episode of eight (3x: one in 32); with 10x none in 80 (2026-11-05). The price: a tail dragged fast round
# an edge resists (a few N at 0.08 m/s), so drag slowly.
LACE_BEND_EI = 2e-4 * 0.015
LACE_BEND_DAMP = 4e-5 * 0.015
LACE_AGLET_LEN = 0.020
LID_WALL_H = 0.025
LID_T = 0.003
LID_CLEAR = 0.004
CATCH_DIST = 0.012
AGLET_OFFSET_GRIPPER = 0.014      # the aglet's centre beyond the tcp when the gripper holds it (tip 24 mm out)
AGLET_OFFSET_HOOK = 0.008         # beyond the hook tip's end
SNUG_RATIO = 1.5                  # a lace pulled snug springs back 0.5-2.5 cm through the ring when let go; a loose one
                                  # lies at 2.5-5x the straight distance
SNUG_THROAT_M = 0.03              # extra lace allowed for a segment that dips under a flap and returns through the throat
LACE_DRAG = 0.004                 # N s/m per lace body (see Scene.step_physics)
# The lace's inertia: a 0.14 g, 1.5 cm capsule is so light that MuJoCo's soft contacts (whose stiffness scales with
# the bodies' mass) let a pull of a few newtons press it 1-4 mm into a 1.2 mm eyelet wire and through it, and the
# chain against the 1-3 kg arm links is badly conditioned; twice the mass and 20x the joint armature fixed both
# (2026-11-04: reference lacing 1/16 -> 12/32 seeds, physics faults 4/16 -> 2/32).
LACE_MASS = 0.012                 # kg, the whole lace with its aglets
LACE_ARMATURE = 1e-7              # kg m^2 on every ball-joint DOF of the lace
AGLET_SEAT_S = 0.3                # a caught aglet is drawn into its seat on the gripper / hook over this time
AGLET_SLIP_N = 6.0                # the pinch and the clip hold an aglet against at most this pull ...
AGLET_SLIP_TICKS = 3              # ... for this many control ticks (30 ms); then it slips out


def _f(v):
    return [float(x) for x in v]


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_rotate(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def resample_exact(path, lengths):
    """Points along the polyline `path` such that consecutive points are exactly `lengths[i]` apart (chord lengths);
    corners are cut by the chords. The path must be long enough."""
    path = np.asarray(path, float)
    pts = [path[0].copy()]
    cur = path[0].copy()
    seg, s_cur = 0, 0.0          # cur lies on segment `seg` at parameter s_cur
    for ell in lengths:
        placed = False
        j = seg
        while j < len(path) - 1:
            a, b = path[j], path[j + 1]
            d = b - a
            A = float(d @ d)
            if A < 1e-14:
                j += 1
                continue
            f = a - cur
            B = 2.0 * float(f @ d)
            C = float(f @ f) - ell * ell
            disc = B * B - 4 * A * C
            lo = s_cur if j == seg else 0.0
            if disc >= 0:
                r = math.sqrt(disc)
                cands = [s for s in ((-B - r) / (2 * A), (-B + r) / (2 * A)) if lo - 1e-9 <= s <= 1.0 + 1e-9]
                if cands:
                    s = min(1.0, max(0.0, min(cands)))
                    cur = a + s * d
                    pts.append(cur.copy())
                    seg, s_cur = j, s
                    placed = True
                    break
            j += 1
        if not placed:
            raise ValueError("path too short for the lace")
    return np.asarray(pts)


class Arm(ArmController):
    """The cell's arm controller with the few extras the server needs."""

    def __init__(self, model, data, prefix, tcp_site, gripper, home_q, tick_s):
        ArmController.__init__(self, model, data, prefix, tcp_site, gripper=gripper, tick_s=tick_s)
        self.home_q = np.asarray(home_q, float)
        self.name = prefix.rstrip("/")

    @staticmethod
    def mat_to_quat(R):
        return mat_to_quat(R)

    def ranges(self):
        return [[float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1])] for j in self.joints]

    def rebase(self):
        self.q_cmd = self.q().copy()
        self.q_goal = self.q_cmd.copy()
        if self.has_gripper:
            self.width_cmd = self.width()
            self.width_goal = self.width_cmd
        self.write_ctrl()

    def width(self):
        return max(0.0, ArmController.width(self))      # the finger slides can read slightly negative after a collision

    def hold_width(self, w):
        """While the gripper holds something (w = the handle's pinch width) the finger goal never goes below w, so
        the fingers stop on the object instead of closing through it; None lifts the hold."""
        self.hold_w = None if w is None else float(w)
        if self.hold_w is not None and self.width_goal < self.hold_w:
            self.width_goal = self.hold_w

    def command_width(self, width):
        ArmController.command_width(self, width)
        hw = getattr(self, "hold_w", None)
        if hw is not None and self.width_goal < hw:
            self.width_goal = hw


class Scene(object):
    def __init__(self, given_dir, order, cfg, layout=None):
        self.given_dir = given_dir
        self.order = order
        self.cfg = cfg
        self.layout = dict(LAYOUT)
        if layout:
            self.layout.update(layout)
        z = cfg.get("zones") or {}
        if "cradle" in z:
            self.layout["cradle"] = {"centre": tuple(z["cradle"]["centre"]), "yaw": float(z["cradle"].get("yaw", 0.0))}
        if "infeed" in z:
            self.layout["infeed"] = dict(self.layout["infeed"], centre=tuple(z["infeed"]["centre"]), half=tuple(z["infeed"]["half"]))
        if "blank" in z:
            self.layout["blank"] = {"centre": tuple(z["blank"]["centre"])}
        if "outfeed_x" in z:
            self.layout["outfeed_x"] = float(z["outfeed_x"])
        self.tick_s = float(self.layout["tick_s"])
        self.substeps = int(round(self.tick_s / float(self.layout["timestep"])))
        self.batch = order["batch"]
        self._build()
        self._index()

    # ------------------------------------------------------------------ build
    def _build(self):
        lay = self.layout
        panda_dir = os.path.join(self.given_dir, "panda")
        spec = mujoco.MjSpec()
        spec.modelname = "shoe_cell"
        spec.compiler.degree = False
        spec.compiler.autolimits = True
        cell_mod.apply_physics(spec, timestep=float(lay["timestep"]), iterations=20)
        spec.visual.global_.offwidth = 640
        spec.visual.global_.offheight = 480
        spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]
        spec.visual.map.znear = 0.005
        spec.visual.map.zfar = 30.0
        spec.stat.extent = 1.0
        spec.stat.center = [0.0, 0.0, 0.2]
        gd = spec.default.geom
        gd.condim = 3
        gd.friction = [1.0, 0.005, 0.0001]
        w = spec.worldbody
        info = {}
        info["bench"] = cell_mod.build_bench(spec, w, "bench", half=lay["bench_half"])
        info["arms"] = {}
        self.arm_base = {}
        for side in ("left", "right"):
            base = lay["arm_bases"][side]
            cell_mod.build_pedestal(spec, w, "pedestal_" + side, base["pos"][:2], half_xy=0.10, top_z=base["pos"][2], floor_z=-0.75)
            info["arms"][side] = cell_mod.add_panda(spec, w, side, base["pos"], base["yaw"], hand=(side == "left"), panda_dir=panda_dir)
            self.arm_base[side] = {"pos": base["pos"], "yaw": base["yaw"], "quat": _f(yaw_quat(base["yaw"]))}
        info["welds"] = cell_mod.build_welds(spec, info["arms"]["left"]["tool_body"], info["arms"]["right"]["tool_body"],
                                             solref=tuple(lay["weld_solref"]))
        # the row-1 tack: a weld (inactive until delivery) between each shoe and the lace segment that crosses its cavity
        # between the two row-1 rings, so that pulling a tail can never draw the lace through row 1 (the stitch a
        # children's-shoe factory puts in); body2 = world is a placeholder retargeted at delivery
        tack_data = [0.0] * 11
        tack_data[10] = 1.0
        for i in range(2):
            spec.add_equality(name="tack%d" % i, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1="shoe%d/body" % i, name2="world", active=False, solref=list(lay["weld_solref"]), data=tack_data)
        info["lights"] = cell_mod.add_lights(spec, w)
        info["cameras"] = build_cameras(spec, info)
        self.info = info
        # infeed tray: a thin plate with a low rim
        cx, cy = lay["infeed"]["centre"]
        hx, hy = lay["infeed"]["half"]
        tz = float(lay["infeed"]["top_z"])
        tray = w.add_body(name="infeed/base", pos=[cx, cy, 0.0])
        tray.add_geom(name="infeed/plate", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, tz / 2], size=[hx, hy, tz / 2],
                      rgba=[0.45, 0.47, 0.50, 1.0], condim=3, friction=[0.9, 0.005, 0.0001])
        for i, (px, py, sx, sy) in enumerate(((hx, 0, 0.004, hy), (-hx, 0, 0.004, hy), (0, hy, hx, 0.004), (0, -hy, hx, 0.004))):
            tray.add_geom(name="infeed/rim%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[px, py, tz + 0.006], size=[sx, sy, 0.006],
                          rgba=[0.35, 0.37, 0.40, 1.0])
        self.infeed_top_z = tz
        # outfeed line (visual only)
        w.add_geom(name="outfeed/line", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[lay["outfeed_x"], 0.0, 0.0005],
                   size=[0.004, lay["bench_half"][1] - 0.02, 0.0005], rgba=[0.95, 0.85, 0.1, 1.0], contype=0, conaffinity=0)
        # shoes (two, re-used by every pair), parked below the bench
        b = self.batch
        self.shoe_params = dict(L=b["length"], W=b["width"], n_rows=b["n_rows"], s=b["row_spacing"], g=b["throat_gap"],
                                upper_rgba=[0.5, 0.5, 0.5, 1.0], sole_rgba=[0.9, 0.9, 0.9, 1.0], mass=0.22, dc=b.get("cavity_depth", 0.045),
                                toe_cap_len=b["toe_cap_len"], heel_len=b["heel_len"], first_row_offset=b["first_row_offset"])
        park_z = float(lay["park"]["z"])
        self.shoes = []
        for i in range(2):
            d = shoe_mod.add_shoe(spec, w, "shoe%d" % i, params=self.shoe_params, pos=(0.3 * i - 0.6, 0.0, park_z + 0.3), free=True,
                                  solref=(0.005, 1.0), ring_condim=3, ring_friction=(0.3, 0.005, 0.0001))
            # (a nearly hard shell (solimp 0.98) was tried against the lace popping out of a ring under a sideways pull:
            #  it made the lace blow up after hook catches and raised every hand/shoe contact; the pop-out comes from
            #  the other tail being drawn through row 1 when a pass runs out of lace, which the batch length avoids)
            self.shoes.append(d)
        # cradle sized for the batch
        cc = lay["cradle"]["centre"]
        jit = self.order.get("jitter", {}).get("cradle", [0.0, 0.0, 0.0])
        self.cradle_pos = (cc[0] + jit[0], cc[1] + jit[1], 0.0)
        self.cradle_yaw = float(lay["cradle"]["yaw"]) + jit[2]
        self.cradle = shoe_mod.add_cradle(spec, w, "cradle", self.cradle_pos, self.cradle_yaw, self.shoes[0])
        # laces
        self.laces = []
        for i in range(2):
            # condim 1 (the lace has priority 1): frictionless against the bench, the tray, the shoe shell and the fingers;
            # the eyelet rings (priority 2, condim 3) keep their friction, which is what holds and brakes a lace.
            # LACE_MASS / LACE_ARMATURE: see their definitions
            d = lace_mod.add_lace(spec, w, "lace%d" % i, n_segments=LACE_SEGMENTS, seg_len=LACE_SEG_LEN, aglet_len=LACE_AGLET_LEN,
                                  start=(0.0, 0.0, park_z + 0.5 + 0.1 * i), direction=(1.0, 0.0, 0.0), rgba=(0.9, 0.9, 0.9, 1.0),
                                  condim=1, mass_total=LACE_MASS, armature=LACE_ARMATURE, joint_type=LACE_JOINT,
                                  bend_stiffness=LACE_BEND_EI / LACE_SEG_LEN, bend_damping=LACE_BEND_DAMP / LACE_SEG_LEN)
            self.laces.append(d)
        # the box blank (no attached lid) and the lid tray
        cb = self.order["carton"]
        self.carton = carton_mod.add_carton(spec, w, "box", L=cb["L"], W=cb["W"], H=cb["H"], pos=(0.0, 0.0, park_z + 0.6), yaw=0.0,
                                            with_lid=False, hinge_range=(0.0, 1.62))
        self.lid = self._add_lid_tray(spec, w, "lid", cb, pos=(0.0, 0.0, park_z + 0.9))
        # a parking shelf below the bench for the objects between pairs
        w.add_geom(name="park/shelf", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, park_z - 0.01], size=[1.2, 0.8, 0.01],
                   rgba=[0.2, 0.2, 0.2, 1.0])
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer_sizes = {k: v["size"] for k, v in CAMERA_SPECS.items()}

    def _add_lid_tray(self, spec, parent, name, cb, pos):
        """A rigid two-piece-box lid: a top plate with four shallow walls hanging down, sized to slide over the erected
        base (clearance LID_CLEAR per side). Grasp handles at the middle of each wall's top edge."""
        t = float(self.carton["params"]["thickness"])
        Li = cb["L"] + 2 * t + 2 * LID_CLEAR
        Wi = cb["W"] + 2 * t + 2 * LID_CLEAR
        Lo, Wo = Li + 2 * LID_T, Wi + 2 * LID_T
        body = parent.add_body(name=name + "/body", pos=_f(pos))
        body.add_freejoint(name=name + "/free")
        rgba = [0.72, 0.55, 0.35, 1.0]
        kw = dict(condim=3, friction=[0.6, 0.005, 0.0001], solref=[0.004, 1.0])
        # local frame: the top plate's underside at z = 0 (the walls hang below), the plate at [0, LID_T]
        body.add_geom(name=name + "/top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, LID_T / 2], size=[Lo / 2, Wo / 2, LID_T / 2],
                      rgba=rgba, mass=0.035, **kw)
        walls = {"px": ([Li / 2 + LID_T / 2, 0, -LID_WALL_H / 2], [LID_T / 2, Wo / 2, LID_WALL_H / 2]),
                 "mx": ([-Li / 2 - LID_T / 2, 0, -LID_WALL_H / 2], [LID_T / 2, Wo / 2, LID_WALL_H / 2]),
                 "py": ([0, Wi / 2 + LID_T / 2, -LID_WALL_H / 2], [Li / 2, LID_T / 2, LID_WALL_H / 2]),
                 "my": ([0, -Wi / 2 - LID_T / 2, -LID_WALL_H / 2], [Li / 2, LID_T / 2, LID_WALL_H / 2])}
        for k, (p, s) in walls.items():
            body.add_geom(name="%s/wall_%s" % (name, k), type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=s, rgba=rgba, mass=0.008, **kw)
        body.add_geom(name=name + "/logo", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, LID_T + 0.0001], size=[0.25 * Lo, 0.25 * Wo, 0.0001],
                      rgba=[0.12, 0.32, 0.78, 1.0], contype=0, conaffinity=0, density=0, group=1)
        # handles: pinch a wall across its thickness from above; site z up, site y = the pinch axis (the wall normal)
        sites = {}
        for k, (p, s) in walls.items():
            top = [p[0], p[1], LID_T]
            if k in ("px", "mx"):
                q = [1.0, 0.0, 0.0, 0.0]            # site y = local y? the wall normal is x: rotate so site y = x
                q = yaw_quat(-math.pi / 2)
            else:
                q = [1.0, 0.0, 0.0, 0.0]
            body.add_site(name="%s/grip_%s" % (name, k), pos=top, quat=_f(q), size=[0.003, 0.003, 0.003], group=4)
            sites[k] = "%s/grip_%s" % (name, k)
        body.add_site(name=name + "/centre", pos=[0, 0, LID_T / 2], size=[0.003, 0.003, 0.003], group=4)
        return {"name": name, "body": name + "/body", "joint": name + "/free", "sites": sites, "centre_site": name + "/centre",
                "dims": {"Lo": Lo, "Wo": Wo, "Li": Li, "Wi": Wi, "wall_h": LID_WALL_H, "t": LID_T}, "pinch_width": LID_T}

    def _index(self):
        m, d = self.model, self.data
        self.arms = {}
        for side in ("left", "right"):
            self.arms[side] = Arm(m, d, side + "/", side + "/tcp", gripper=(side == "left"), home_q=self.layout["home_q"][side], tick_s=self.tick_s)
        self.reset_arms_home()
        self.cameras = CameraRig(m, self.renderer_sizes, scene=self)
        self.tip_site = m.site("right/hook_tip").id
        self.left_tcp = m.site("left/tcp").id
        self.hand_body = m.body("left/hand").id
        # ids
        self.shoe_body = [m.body(s["body"]).id for s in self.shoes]
        self.shoe_qadr = [m.jnt_qposadr[m.joint(s["joint"]).id] for s in self.shoes]
        self.shoe_vadr = [m.jnt_dofadr[m.joint(s["joint"]).id] for s in self.shoes]
        self.lace_bodies = [lace_mod.lace_body_ids(m, l) for l in self.laces]
        self.lace_lengths = [np.asarray(l["lengths"], float) for l in self.laces]
        self.lid_body = m.body(self.lid["body"]).id
        self.lid_qadr = m.jnt_qposadr[m.joint(self.lid["joint"]).id]
        self.lid_vadr = m.jnt_dofadr[m.joint(self.lid["joint"]).id]
        self.carton_floor_body = m.body(self.carton["bodies"]["floor"]).id
        self.carton_qadr = m.jnt_qposadr[m.joint(self.carton["joints"]["free"]).id]
        self.carton_vadr = m.jnt_dofadr[m.joint(self.carton["joints"]["free"]).id]
        self.carton_hinge_qadr = {h: m.jnt_qposadr[m.joint(self.carton["joints"][h]).id] for h in self.carton["walls"]}
        self.carton_hinge_vadr = {h: m.jnt_dofadr[m.joint(self.carton["joints"][h]).id] for h in self.carton["walls"]}
        # geom classes for the safety watch: 1 arm link, 2/3 pads, 4 loose object, 5 static environment, 6 the hook tool
        gc = np.zeros(m.ngeom, dtype=int)
        pads = set(self.info["arms"]["left"]["pad_geoms"])
        hook_geoms = set(self.info["arms"]["right"]["hook"]["geoms"]) if isinstance(self.info["arms"]["right"]["hook"].get("geoms"), (list, tuple)) else set()
        loose_bodies = set(self.shoe_body) | {self.lid_body} | set(int(b) for ids in self.lace_bodies for b in ids)
        for bn in self.carton["bodies"].values():
            loose_bodies.add(m.body(bn).id)
        self.loose_bodies = loose_bodies
        for g in range(m.ngeom):
            name = m.geom(g).name
            b = m.geom_bodyid[g]
            bname = m.body(b).name
            if name in pads:
                gc[g] = 2 if "left_" in name or "pad_left" in name else 3
            elif name in hook_geoms or bname == "right/hook":
                gc[g] = 6
            elif bname.startswith("left/") or bname.startswith("right/"):
                gc[g] = 1
            elif b in loose_bodies:
                gc[g] = 4
            elif b == 0 or bname.startswith(("cradle/", "infeed/", "pedestal")):
                gc[g] = 5
        self.geom_class = gc
        ga = np.zeros(m.ngeom, dtype=int)
        for g in range(m.ngeom):
            bname = m.body(m.geom_bodyid[g]).name
            ga[g] = 1 if bname.startswith("left/") else (2 if bname.startswith("right/") else 0)
        self.geom_arm = ga
        # contact bit scheme (contact iff (contype_a & conaffinity_b) or (contype_b & conaffinity_a)):
        #   ordinary geoms (1, 31); the gripper hand + fingers (1, 11); the hook tool (8, 0); lace segments (2, 5);
        #   while the gripper holds a body its geoms become (4, 0) (shoe, panel, lid) or (16, 0) (an aglet).
        # So: a lace never touches itself, another lace or the hook; a held shoe keeps colliding with everything but
        # the hand that holds it; a held aglet collides with nothing of its own lace nor the hand.
        gripper_bodies = {m.body(n).id for n in ("left/hand", "left/left_finger", "left/right_finger")}
        lace_body_set = set(int(b) for ids in self.lace_bodies for b in ids)
        for g in range(m.ngeom):
            if int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
                continue
            b = int(m.geom_bodyid[g])
            if gc[g] == 6:
                ct, ca = 8, 0
            elif b in lace_body_set:
                ct, ca = 2, 5
            elif b in gripper_bodies:
                ct, ca = 1, 11
            else:
                ct, ca = 1, 31
            m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        self.physics_fault = False
        self._lace_body_ids = sorted(lace_body_set)
        self._vel6 = np.zeros(6)
        # the shoe's upper geoms (recoloured per pair) and the lace cord geoms
        self.shoe_upper_geoms = []
        self.shoe_sole_geoms = []
        for s in self.shoes:
            gs = s["geoms"]
            upper = []
            for key in ("heel_counter", "toe_cap", "walls", "flap_L", "flap_R"):
                v = gs.get(key)
                if isinstance(v, (list, tuple)):
                    upper += list(v)
                elif isinstance(v, dict):
                    upper += [x for x in v.values() if isinstance(x, str)]
                elif isinstance(v, str):
                    upper.append(v)
            self.shoe_upper_geoms.append([m.geom(n).id for n in upper if n in [m.geom(i).name for i in range(m.ngeom)]])
            sole = gs.get("sole")
            sole = list(sole) if isinstance(sole, (list, tuple)) else ([sole] if isinstance(sole, str) else list(sole.values()))
            self.shoe_sole_geoms.append([m.geom(n).id for n in sole if isinstance(n, str)])
        self.lace_cord_geoms = [[m.geom(n).id for n in l["geoms"][1:-1]] for l in self.laces]
        self.lid_logo_geom = m.geom("lid/logo").id
        self._eyelet_cache = {}
        self.current = None

    # ------------------------------------------------------------------ arms and stepping
    def reset_arms_home(self):
        m, d = self.model, self.data
        for side, arm in self.arms.items():
            info = self.info["arms"][side]
            for i, jn in enumerate(info["joints"]):
                d.qpos[m.jnt_qposadr[m.joint(jn).id]] = arm.home_q[i]
                d.qvel[m.jnt_dofadr[m.joint(jn).id]] = 0.0
            if info["hand"]:
                for jn in info["finger_joints"]:
                    d.qpos[m.jnt_qposadr[m.joint(jn).id]] = 0.04
        mujoco.mj_forward(m, d)
        for arm in self.arms.values():
            arm.rebase()
            arm.set_mode("joint")
            arm.q_goal = arm.home_q.copy()
            if arm.has_gripper:
                arm.command_width(0.08)

    def step_physics(self):
        m, d = self.model, self.data
        t0 = float(d.time)
        # a light linear drag on every lace body (a few mN at 0.1 m/s): the condim-1 lace comes to rest on the
        # tray instead of sliding for ever, and a hanging tail stops swinging; held constant over the tick
        v = self._vel6
        for b in self._lace_body_ids:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            d.xfrc_applied[b, 0:3] = -LACE_DRAG * v[3:6]
        grasp_mod.advance_seats(m, d)                       # caught aglets glide into their seats (AGLET_SEAT_S)
        for _ in range(self.substeps):
            mujoco.mj_step(m, d)
        if self.data.time < t0 or int(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0:
            self.physics_fault = True

    def settle(self, seconds):
        n = int(round(seconds / self.tick_s))
        for _ in range(n):
            for arm in self.arms.values():
                arm.tick()
            self.step_physics()
            self.update_locks(self.tick_s)

    def update_locks(self, dt):
        carton_mod.update_locks(self.model, self.data, self.carton, dt)

    # ------------------------------------------------------------------ descriptions
    def tool_description(self, name):
        if name == "left":
            return {"kind": "parallel-jaw gripper (Franka hand)", "max_width_m": 0.08, "tcp": "left/tcp: midpoint between the finger pads, +z = approach (out of the fingers), +y = finger closing axis, +x across the pads",
                    "attach": "closing on a documented grasp handle inside the pinch box attaches it (cell_manual.md)"}
        hk = self.info["arms"]["right"]["hook"]
        return {"kind": "lacing hook: a shaft, an L-foot and an upturned tip with a clip", "shaft_len_m": hk["shaft_len"], "foot_len_m": hk["foot_len"],
                "tip_len_m": hk["tip_len"], "shaft_radius_m": hk["shaft_r"], "foot_radius_m": hk["foot_r"],
                "tcp": "right/tcp: the shaft end, +z = the shaft direction (approach); the foot runs along tool +x from the shaft end; the tip runs back along -z from the foot end",
                "tip_site": "right/hook_tip: at the tip end, +z along the tip (= tool -z)",
                "attach": "hook(True) catches an aglet within %.0f mm of the tip and seats it along the tip, protruding beyond it; hook(False) releases" % (CATCH_DIST * 1000)}

    def attach_rules(self):
        return {"gripper": {"pinch_box_m": {"x_across": 0.012, "y_along_closing": "half the current opening", "z_along_approach": [-0.010, 0.012]},
                            "handles": ["an aglet (anywhere on it; it is seated along the approach axis, its centre 14 mm beyond the tcp, its tip 24 mm)",
                                        "a shoe at the heel counter (pinch %.0f mm) or the toe cap (pinch %.0f mm)" % (self.batch["heel_len"] * 1000, self.batch["toe_cap_len"] * 1000),
                                        "a box-blank wall at its free edge: the centre and the two points at 30 % of the edge from it, 12 mm in from the edge (pinch 3 mm)",
                                        "the lid tray at the middle of any of its four walls' top edges (pinch 3 mm)"],
                            "rule": "the fingers must close with the handle inside the pinch box; the object is then attached to the hand at its current pose (an aglet is drawn into its seat over aglet_seat_s) until the fingers open again; closing on an aglet that sits on the hook takes it off the hook"},
                "hook": {"catch_distance_m": CATCH_DIST, "rule": "hook(True) catches the nearest aglet whose centre is within the catch distance of the tip segment and draws it into its seat along the tip over aglet_seat_s; hook(False) releases"},
                "aglet": {"aglet_seat_s": AGLET_SEAT_S, "slip_force_N": AGLET_SLIP_N, "slip_after_s": AGLET_SLIP_TICKS * TICK_S,
                          "rule": "the pinch and the clip hold an aglet against at most slip_force_N of pull: pulled harder for slip_after_s it slips out (step event aglet_slipped:left / aglet_slipped:right)"},
                "cradle": {"rule": "a shoe resting sole-down, heel-back in the seat (within %.0f mm, %.0f deg) for %.1f s is clamped; release_clamp() frees it" % (
                    self.cradle["tolerances"]["xy"] * 1000, self.cradle["tolerances"]["yaw_deg"], float(self.cfg.get("clamp_rest_s", 0.3)))},
                "carton": {"rule": "a wall held at >= 87 deg for 0.3 s snap-locks at 90 deg; locks never release",
                           "lid": "the separate lid tray is closed when it sits over the erected walls: centred within 12 mm, aligned within 6 deg, resting"}}

    def cradle_description(self):
        c = self.cradle
        return {"plate_size_m": c["dims"]["plate_size"] if "plate_size" in c["dims"] else None, "plate_top_z": float(c["plate_top"]),
                "clearance_m": float(c["clearance"]), "seat": "sole-down, heel toward the heel stop; corner bumpers %.0f mm tall" % (c["dims"]["bumper_h"] * 1000),
                "tolerances": c["tolerances"]}

    def object_description(self):
        b, cb, ld = self.batch, self.order["carton"], self.lid["dims"]
        s0 = self.shoes[0]
        return {"shoe": {"length_m": b["length"], "outer_width_m": b["width"], "n_rows": b["n_rows"], "row_spacing_m": b["row_spacing"], "throat_gap_m": b["throat_gap"],
                         "eyelet_hole_radius_m": 0.0045, "eyelet_ring_outer_radius_m": float(s0["dims"]["ring_outer_radius"]), "flap_top_z_m": float(s0["dims"]["z_f"]),
                         "cavity_floor_z_m": float(s0["dims"]["z_floor"]), "heel_top_z_m": float(s0["dims"]["z_hc_top"]), "toe_top_z_m": float(s0["dims"]["z_tc_top"]),
                         "note": "one model and size per batch; colours per pair; a left and a right shoe are mirror images (identical here)",
                         "rows": "row 1 is nearest the toe; rings sit on the flap top surface, 9 mm in from the flap's inner (throat) edge"},
                "lace": {"length_m": float(sum(self.lace_lengths[0])), "radius_m": LACE_SEG_LEN * 0 + 0.0015, "aglet_length_m": LACE_AGLET_LEN, "aglet_radius_m": 0.0018,
                         "delivery": "threaded through row 1 and tacked there (the cord is stitched to the insole between the two row-1 rings, so pulling an end never slides the lace through row 1): both ends leave the shoe outward through the row-1 rings with about equal tails; the tails hang over the sides"},
                "box": {"L_m": cb["L"], "W_m": cb["W"], "H_m": cb["H"], "thickness_m": 0.003, "blank": "floor + four walls on creases, flat on the bench, inner (light) face up",
                        "lid": {"outer_m": [ld["Lo"], ld["Wo"]], "wall_height_m": ld["wall_h"], "note": "a rigid lid tray, delivered printed side up on the infeed tray"}}}

    # ------------------------------------------------------------------ pair delivery
    def deliver_pair(self, spec):
        """Set the pair down: colours, the two shoes on the infeed tray (one may lie on its side), the laces threaded
        through row 1, the flat blank at the blank spot, the lid tray on the infeed tray."""
        m, d = self.model, self.data
        self.release_all()
        for i in range(2):
            for g in self.shoe_upper_geoms[i]:
                m.geom_rgba[g] = spec["upper_rgba"]
            for g in self.shoe_sole_geoms[i]:
                m.geom_rgba[g] = spec["sole_rgba"]
            for g in self.lace_cord_geoms[i]:
                m.geom_rgba[g] = spec["lace_rgba"]
        m.geom_rgba[self.lid_logo_geom] = spec["logo_rgba"]
        for i, sh in enumerate(spec["shoes"]):
            pos = [sh["pos"][0], sh["pos"][1], self.infeed_top_z + (self.batch["width"] / 2 + 0.003 if sh["on_side"] else 0.002)]
            q = yaw_quat(sh["yaw"])
            if sh["on_side"]:
                roll = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4) * (1 if sh.get("side_sign", 1) > 0 else -1), 0.0, 0.0])
                q = quat_mul(q, roll)
            shoe_mod.set_shoe_pose(m, d, self.shoes[i], pos, q, forward=True)
            self._thread_lace(i, i, spec["lid"], tails=sh.get("tails", (0.5, 0.5)))
        # blank: flat, inner face up, at the blank spot
        bl = spec["blank"]
        self._place_carton([bl["pos"][0], bl["pos"][1], 0.0005], bl["yaw"])
        ld = spec["lid"]
        self._set_free(self.lid_qadr, self.lid_vadr, [ld["pos"][0], ld["pos"][1], self.infeed_top_z + LID_WALL_H + 0.002], yaw_quat(ld["yaw"]))
        mujoco.mj_forward(m, d)
        self.current = {"shoe_ids": [0, 1], "lace_ids": [0, 1], "carton_id": 0, "lid_id": 0}
        return dict(self.current)

    def take_away_pair(self, rec):
        """Park the pair's objects below the bench (the cell took the box away)."""
        m, d = self.model, self.data
        self.release_all()
        pz = float(self.layout["park"]["z"])
        # parking spots on the shelf below the bench, clear of the two pedestal columns (x [-0.15, 0.05] x y [0.2, 0.4]
        # and x [-0.65, -0.45] x y [-0.4, -0.2]) and of the bench legs at (+-0.75, +-0.45): a lace laid through a
        # pedestal column used to blow the physics up right after a pair shipped
        for i in range(2):
            shoe_mod.set_shoe_pose(m, d, self.shoes[i], [-0.65 + 0.35 * i, 0.3, pz + 0.02], [1, 0, 0, 0], forward=False)
            poly = lace_mod.straight_polyline(self.laces[i], (-0.3, 0.45 + 0.07 * i, pz + 0.01), (1.0, 0.0, 0.0))
            lace_mod.set_lace_polyline(m, d, self.laces[i], poly, forward=False)
        self._place_carton([0.4, 0.0, pz + 0.01], 0.0)
        self._set_free(self.lid_qadr, self.lid_vadr, [0.4, -0.45, pz + LID_WALL_H + 0.01], [1, 0, 0, 0])
        mujoco.mj_forward(m, d)
        self.current = None

    def release_all(self):
        grasp_mod.release(self.model, self.data, "left/grasp")
        grasp_mod.release(self.model, self.data, "right/hook")
        for i in range(2):
            grasp_mod.release(self.model, self.data, "tack%d" % i)
        if shoe_mod.clamp_active(self.model, self.data, self.cradle):
            for s in self.shoes:
                shoe_mod.set_clamp(self.model, self.data, self.cradle, s, False)

    def _set_free(self, qadr, vadr, pos, quat):
        d = self.data
        d.qpos[qadr:qadr + 3] = pos
        d.qpos[qadr + 3:qadr + 7] = quat
        d.qvel[vadr:vadr + 6] = 0.0

    def _place_carton(self, pos, yaw):
        m, d = self.model, self.data
        carton_mod.reset_locks(m, d, self.carton)
        for h, qa in self.carton_hinge_qadr.items():
            d.qpos[qa] = 0.0
            d.qvel[self.carton_hinge_vadr[h]] = 0.0
        self._set_free(self.carton_qadr, self.carton_vadr, pos, yaw_quat(yaw))

    def _thread_lace(self, lace_i, shoe_i, lid, tails=(0.5, 0.5)):
        """Pose lace `lace_i` threaded through row 1 of shoe `shoe_i` (shoe already placed, kinematics current; `lid`:
        the pair's lid tray spot, where it is set down next).

        World path: tail A on the tray -> up beside the shoe -> 3 cm along the left row-1 ring's axis -> through the
        ring into the cavity -> across -> up through the right ring -> 3 cm along its axis -> down beside the shoe
        -> tail B. Each tail runs on the tray plate from below its exit point toward the heel, alongside the shoe,
        and turns away from the shoe at the heel. Every tail point lies on the plate whatever the shoe's pose
        (upright or on its side) so nothing is ever laid inside the tray; the two tails get exactly their share of
        the slack (`tails`)."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        sd = self.shoes[shoe_i]
        dims = sd["dims"]
        x0 = float(dims["x_rows"][0])
        y_e = float(dims["eyelet_y"])
        z_f = float(dims["z_f"])
        z_lo = max(float(dims["z_floor"]) + 0.006, z_f - 0.03)
        W, L = float(dims["W"]), float(dims["L"])
        b = self.shoe_body[shoe_i]
        pos, quat = d.xpos[b].copy(), d.xquat[b].copy()

        def tw(p):
            return pos + quat_rotate(quat, np.asarray(p, float))
        up = 0.03
        z_s = self.infeed_top_z + 0.0035                     # the lace's centre line resting on the tray plate
        heading = quat_rotate(quat, np.array([1.0, 0.0, 0.0]))
        heading[2] = 0.0
        hn = np.linalg.norm(heading)
        heading = heading / hn if hn > 1e-6 else np.array([1.0, 0.0, 0.0])
        left = np.array([-heading[1], heading[0], 0.0])
        mid = [tw((x0, y_e, z_f + up)), tw((x0, y_e, z_lo)), tw((x0, -y_e, z_lo)), tw((x0, -y_e, z_f + up))]

        def exit_point(sign, ring_up):
            e = tw((x0, sign * (W / 2 + 0.014), z_f + up))
            e[2] = max(float(e[2]), z_s + 0.012)
            if np.linalg.norm(e - ring_up) < 0.012:          # (a shoe on its side: the low exit point got clamped)
                out = left if float((ring_up - pos) @ left) >= 0.0 else -left
                e = ring_up + out * 0.015
                e[2] = max(float(e[2]), z_s + 0.012)
            return e
        edge_L, edge_R = exit_point(1.0, mid[0]), exit_point(-1.0, mid[3])
        cx, cy = self.layout["infeed"]["centre"]
        hx, hy = self.layout["infeed"]["half"]
        # keep-out boxes on the plate (xy AABBs, with clearance): the lid tray's footprint at its spot (1 cm) and the
        # other shoe's footprint
        lid_x, lid_y, lid_yaw = float(lid["pos"][0]), float(lid["pos"][1]), float(lid["yaw"])
        lid_c, lid_s = abs(math.cos(lid_yaw)), abs(math.sin(lid_yaw))
        lid_hl, lid_hw = float(self.lid["dims"]["Lo"]) / 2, float(self.lid["dims"]["Wo"]) / 2
        lid_ex, lid_ey = lid_hl * lid_c + lid_hw * lid_s + 0.01, lid_hl * lid_s + lid_hw * lid_c + 0.01
        boxes = [((lid_x - lid_ex, lid_y - lid_ey), (lid_x + lid_ex, lid_y + lid_ey))]
        other = self.shoe_centre(1 - shoe_i)
        if other[2] > -0.1:
            boxes.append(((other[0] - L / 2 - 0.03, other[1] - 0.07), (other[0] + L / 2 + 0.03, other[1] + 0.07)))

        def room(q, direction):
            """Distance from q along `direction` before leaving the plate (3 cm inside the rim) or entering a keep-out box."""
            best = 1.0
            for k, (c, h) in enumerate(((cx, hx), (cy, hy))):
                if abs(direction[k]) > 1e-6:
                    lim = ((c + math.copysign(h - 0.03, direction[k])) - q[k]) / direction[k]
                    best = min(best, max(0.0, lim))
            for lo, hi in boxes:
                t_in, t_out = -1e9, 1e9
                for k in range(2):
                    if abs(direction[k]) < 1e-9:
                        if not (lo[k] <= q[k] <= hi[k]):
                            t_in, t_out = 1e9, -1e9
                        continue
                    t1, t2 = (lo[k] - q[k]) / direction[k], (hi[k] - q[k]) / direction[k]
                    t_in, t_out = max(t_in, min(t1, t2)), min(t_out, max(t1, t2))
                if t_in <= t_out and t_out > 0.0:
                    best = min(best, max(0.0, t_in))
            return best
        run = (x0 + L / 2) + 0.03                            # along the shoe to just past the heel

        def tail(edge, length, margin=0.0):
            """The tail on the tray plate below its exit point: a spiral of up to three legs with 90-degree corners
            (first the free direction with the most room, never toward the shoe), inside the plate and clear of the
            lid tray and the other shoe."""
            q1 = np.array([edge[0], edge[1], z_s])
            out = left if float((edge - pos) @ left) >= 0.0 else -left
            cands = [(-heading, min(length, run)), (out, length), (heading, length)]
            scored = sorted(cands, key=lambda c: -min(c[1], room(q1, c[0])))
            d1 = scored[0][0]
            pts = [q1]
            rest = length
            legs = [d1]
            d2 = out if abs(float(d1 @ out)) < 0.5 else -heading
            legs = [d1, d2, -d1, -d2]
            for i, dd in enumerate(legs):
                if rest <= 1e-6:
                    break
                r = room(pts[-1], dd)
                step = min(rest, r if i < len(legs) - 1 else rest)
                if step > 0.005:
                    pts.append(pts[-1] + dd * step)
                    rest -= step
            if rest > 1e-6:
                pts.append(pts[-1] + legs[-1] * rest)        # (should not happen on this plate; keeps the path long enough)
            if margin > 0.0:
                pts.append(pts[-1] + (pts[-1] - pts[-2]) / max(1e-9, np.linalg.norm(pts[-1] - pts[-2])) * margin)
            return pts
        total = float(np.sum(self.lace_lengths[lace_i]))
        fixed = (np.linalg.norm(mid[0] - edge_L) + np.linalg.norm(mid[1] - mid[0]) + np.linalg.norm(mid[2] - mid[1])
                 + np.linalg.norm(mid[3] - mid[2]) + np.linalg.norm(edge_R - mid[3])
                 + (edge_L[2] - z_s) + (edge_R[2] - z_s))
        rest = max(0.06, total - fixed)
        ta, tb = rest * float(tails[0]), rest * float(tails[1])
        # the chords cut the path's corners, so the path runs on past tail B's end (`margin`). A tail that doubles back
        # where a leg had no room (a shoe on its side) costs more than the usual 8 cm, so a longer run-out is tried then;
        # where 8 cm suffices, the path and the lace are exactly as before.
        for margin in (0.08, 0.16, 0.32):
            path = list(reversed(tail(edge_L, ta))) + [edge_L] + mid + [edge_R] + tail(edge_R, tb, margin=margin)
            path = [p for j, p in enumerate(path) if j == 0 or np.linalg.norm(p - path[j - 1]) > 1e-6]
            try:
                pts = resample_exact(path, self.lace_lengths[lace_i])
                break
            except ValueError:
                if margin >= 0.32:
                    raise
        lace_mod.set_lace_polyline(m, d, self.laces[lace_i], pts, check=False, forward=True)
        # tack: weld the segment nearest the middle of the cavity crossing to the shoe (current relative pose)
        centre = 0.5 * (mid[1] + mid[2])
        ids = [int(b) for b in self.lace_bodies[lace_i]]
        seg = min(ids, key=lambda b: float(np.linalg.norm(d.xpos[b] - centre)))
        grasp_mod.attach(m, d, "tack%d" % shoe_i, m.body(seg).name)

    # ------------------------------------------------------------------ queries: shoes
    def shoe_pose(self, i):
        p, q = shoe_mod.shoe_pose(self.model, self.data, self.shoes[i])
        return [_f(p), _f(q)]

    def shoe_centre(self, i):
        b = self.shoe_body[i]
        return self.data.xpos[b] + quat_rotate(self.data.xquat[b], np.asarray(self.shoes[i]["centre_local"], float))

    def shoe_heading(self, i):
        R = self.data.xmat[self.shoe_body[i]].reshape(3, 3)
        return R[:, 0].copy()

    def upright(self, i, max_deg=10.0):
        return shoe_mod.upright(self.model, self.data, self.shoes[i], max_tilt_deg=max_deg)

    def inside_cavity(self, i, p):
        return shoe_mod.inside_cavity(self.model, self.data, self.shoes[i], p)

    def aglet_ends_out(self, lace_i, shoe_i):
        """Per lace end (a, b): True when that tail has passed a ring beyond row 1 and its aglet is clearly out of the
        shoe (outside the cavity and either 2 cm above the flap tops or off the shoe's footprint), i.e. the end was
        brought back out after its last pass; None when the tail has no pass beyond row 1."""
        passes, _ = self.lace_passes(lace_i, shoe_i)
        sd = self.shoes[shoe_i]
        dims = sd["dims"]
        out = []
        for k, (c, edge) in enumerate(zip(self.aglet_centres(lace_i), (passes[:1], passes[-1:]))):
            if not edge or edge[0][0] == 1:
                out.append(None)
                continue
            q = shoe_mod.world_to_shoe(self.model, self.data, sd, c)
            off_shoe = abs(float(q[0])) > float(dims["L"]) / 2 or abs(float(q[1])) > float(dims["W"]) / 2
            out.append(bool(not self.inside_cavity(shoe_i, c) and (off_shoe or float(q[2]) > float(dims["z_f"]) + 0.02)))
        return out

    def aglet_left_inside(self, lace_i, shoe_i, brought_out=(False, False)):
        """True when an aglet of lace `lace_i` lies inside shoe `shoe_i`'s cavity and was never brought back out after
        its tail's last ring pass (`brought_out`: per end, latched by the work order whenever aglet_ends_out saw it
        out): an end passed in and left there. An end that came out (through the throat slot) and later slid or fell
        back in does not count: a finished lace tucked into the shoe is still laced."""
        for k, c in enumerate(self.aglet_centres(lace_i)):
            if self.inside_cavity(shoe_i, c) and not (brought_out and brought_out[k]):
                return True
        return False

    def eyelets_world(self, i):
        out = []
        for e in shoe_mod.eyelets_world(self.model, self.data, self.shoes[i]):
            out.append({"row": int(e["row"]) + 1, "side": e["side"], "center": _f(e["center"]), "axis": _f(e["axis"])})
        return out

    def shoe_handles(self, i):
        m, d = self.model, self.data
        out = {}
        for key in ("heel_grip", "toe_grip"):
            sid = m.site(self.shoes[i]["sites"][key]).id
            out[key] = {"pos": _f(d.site_xpos[sid]), "quat": _f(mat_to_quat(d.site_xmat[sid].reshape(3, 3))),
                        "pinch_width": float(self.shoes[i]["grip"][key]["pinch_width"])}
        return out

    def seated(self, i):
        return shoe_mod.seated(self.model, self.data, self.shoes[i], self.cradle)

    def at_rest(self, i, v_max=0.02, w_max=0.15):
        v, w = shoe_mod.shoe_velocity(self.model, self.data, self.shoes[i])
        return float(np.linalg.norm(v)) < v_max and float(np.linalg.norm(w)) < w_max

    def set_clamp(self, i, on):
        shoe_mod.set_clamp(self.model, self.data, self.cradle, self.shoes[i], bool(on))

    def cradle_truth(self):
        p, q = shoe_mod.cradle_pose(self.cradle, self.model, self.data)
        return {"seat_pos": _f(p), "seat_quat": _f(q), "pos": _f(self.cradle_pos), "yaw": float(self.cradle_yaw),
                "clamp_active": bool(shoe_mod.clamp_active(self.model, self.data, self.cradle))}

    # ------------------------------------------------------------------ queries: laces
    def lace_polyline(self, i):
        return lace_mod.lace_polyline(self.model, self.data, self.laces[i], self.lace_bodies[i])

    def aglet_centres(self, i):
        poly = self.lace_polyline(i)
        return [0.5 * (poly[0] + poly[1]), 0.5 * (poly[-2] + poly[-1])]

    def aglet_info(self, i):
        """[(which, body name, centre, axis pointing out of the lace)] for aglet A and B of lace i."""
        poly = self.lace_polyline(i)
        a_axis = poly[0] - poly[1]
        b_axis = poly[-1] - poly[-2]
        a_axis /= max(1e-9, np.linalg.norm(a_axis))
        b_axis /= max(1e-9, np.linalg.norm(b_axis))
        ld = self.laces[i]
        return [("a", ld["bodies"][0], 0.5 * (poly[0] + poly[1]), a_axis), ("b", ld["bodies"][-1], 0.5 * (poly[-2] + poly[-1]), b_axis)]

    def lace_passes(self, lace_i, shoe_i):
        """(passes, snug): passes = [(row, side, direction, t)] along the lace from aglet A to B with direction +1 =
        the lace runs from inside to outside (along the ring axis) when followed from A to B; snug = {row: bool} for
        rows 2.. whose lace from the previous row is at most SNUG_RATIO x the straight distance (plus SNUG_THROAT_M when
        the previous row's pass went in: that segment dips under the flap and returns through the throat)."""
        rings = []
        centres = {}
        hole_r = float(self.shoes[shoe_i]["dims"]["hole_radius"])
        for e in shoe_mod.eyelets_world(self.model, self.data, self.shoes[shoe_i]):
            key = (int(e["row"]) + 1, e["side"])
            rings.append((key, np.asarray(e["center"], float), np.asarray(e["axis"], float), hole_r + 0.0015))
            centres[key] = np.asarray(e["center"], float)
        poly = self.lace_polyline(lace_i)
        seq = lace_mod.pass_sequence(poly, rings)
        total = float(np.sum(self.lace_lengths[lace_i]))
        passes = [(int(k[0]), k[1], -int(dd), float(t)) for (k, dd, t) in seq]
        snug = {}
        r1 = [j for j, p in enumerate(passes) if p[0] == 1]
        if len(r1) == 2:
            chains = [list(reversed(passes[:r1[0] + 1])), passes[r1[1]:]]
            for chain in chains:
                for a, b in zip(chain[:-1], chain[1:]):
                    if b[0] == a[0] + 1 and b[0] >= 2:
                        arc = abs(b[3] - a[3]) * total
                        straight = float(np.linalg.norm(centres[(b[0], b[1])] - centres[(a[0], a[1])]))
                        ok = arc <= SNUG_RATIO * max(straight, 1e-6) + (SNUG_THROAT_M if a[0] >= 2 else 0.0)
                        snug[b[0]] = snug.get(b[0], True) and ok
        return passes, snug

    # ------------------------------------------------------------------ queries: carton and lid
    def carton_state(self, k=0):
        st = carton_mod.carton_state(self.model, self.data, self.carton)
        return {"erected": bool(st["erected"]), "wall_angles_deg": {h: float(v) for h, v in st["angles_deg"].items()},
                "locked": {h: bool(v) for h, v in st["locked"].items()}, "floor_pos": _f(st["floor_pos"]), "floor_quat": _f(st["floor_quat"]),
                "closed": bool(self.lid_closed()), "interior_aabb_floor": st["interior_aabb_floor"], "_inside": st["inside_box"], "_to_floor": st["to_floor_frame"]}

    def inside_box(self, k, p):
        st = carton_mod.carton_state(self.model, self.data, self.carton)
        return bool(st["inside_box"](p))

    def resting_on_wall(self, k, shoe_i):
        """The shoe's sole is not down on the box floor (it sits on a wall edge or is tilted up)."""
        st = carton_mod.carton_state(self.model, self.data, self.carton)
        b = self.shoe_body[shoe_i]
        sole_pt = self.data.xpos[b] + quat_rotate(self.data.xquat[b], np.array([0.0, 0.0, 0.0]))
        pf = st["to_floor_frame"](sole_pt)
        t = float(self.carton["params"]["thickness"])
        return bool(pf[2] > t + 0.02)

    def lid_pose(self):
        b = self.lid_body
        return self.data.xpos[b].copy(), self.data.xquat[b].copy()

    def lid_closed(self):
        """The lid tray sits over the erected box: centred, aligned (mod 180 deg), at wall-top height, resting."""
        st = carton_mod.carton_state(self.model, self.data, self.carton)
        if not st["erected"]:
            return False
        lp, lq = self.lid_pose()
        fp, fq = st["floor_pos"], st["floor_quat"]
        rel = st["to_floor_frame"](lp)
        L, W, H, t = (self.order["carton"][k] for k in ("L", "W", "H")), None, None, None
        cp = self.carton["params"]
        H, t = float(cp["H"]), float(cp["thickness"])
        if abs(rel[0]) > 0.012 or abs(rel[1]) > 0.012:
            return False
        if abs(rel[2] - (t + H + 0.0015)) > 0.01:
            return False
        Rf, Rl = quat_to_mat(fq), quat_to_mat(lq)
        if float(Rl[2, 2]) < math.cos(math.radians(6.0)):
            return False
        xl = Rf.T @ Rl[:, 0]
        yaw = math.atan2(xl[1], xl[0])
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        if abs(yaw) > math.radians(6.0):
            return False
        v = self.data.qvel[self.lid_vadr:self.lid_vadr + 3]
        return bool(np.linalg.norm(v) < 0.03)

    def carton_handles(self, k=0):
        m, d = self.model, self.data
        out = {}
        for w in self.carton["walls"]:
            for sn in self.carton["sites"][w]:
                sid = m.site(sn).id
                out[sn] = {"pos": _f(d.site_xpos[sid]), "quat": _f(mat_to_quat(d.site_xmat[sid].reshape(3, 3))), "pinch_width": float(self.carton["params"]["thickness"])}
        for key, sn in self.lid["sites"].items():
            sid = m.site(sn).id
            out[sn] = {"pos": _f(d.site_xpos[sid]), "quat": _f(mat_to_quat(d.site_xmat[sid].reshape(3, 3))), "pinch_width": LID_T}
        return out

    def infeed_empty(self, rec):
        cx, cy = self.layout["infeed"]["centre"]
        hx, hy = self.layout["infeed"]["half"]
        for i in rec["shoe_ids"]:
            c = self.shoe_centre(i)
            if abs(c[0] - cx) <= hx + 0.05 and abs(c[1] - cy) <= hy + 0.05 and c[2] > -0.1:
                return False
        lp, _ = self.lid_pose()
        if abs(lp[0] - cx) <= hx + 0.05 and abs(lp[1] - cy) <= hy + 0.05 and lp[2] > -0.1:
            return False
        return True

    # ------------------------------------------------------------------ grasping
    def _handles(self):
        """Every grasp handle: dicts with name, body, pos, mat, pinch_width, kind, aglet."""
        m, d = self.model, self.data
        hs = []
        hook_held = grasp_mod.is_attached(m, d, "right/hook")
        for i, sd in enumerate(self.shoes):
            for key in ("heel_grip", "toe_grip"):
                sid = m.site(sd["sites"][key]).id
                hs.append({"name": sd["sites"][key], "body": sd["body"], "pos": d.site_xpos[sid].copy(), "mat": d.site_xmat[sid].reshape(3, 3).copy(),
                           "pinch_width": float(sd["grip"][key]["pinch_width"]), "kind": "shoe", "aglet": None})
        for i in range(2):
            for which, bname, centre, axis in self.aglet_info(i):
                hs.append({"name": "%s/aglet_%s" % (self.laces[i]["name"], which), "body": bname, "pos": centre, "mat": np.eye(3),
                           "pinch_width": 0.0036, "kind": "aglet", "aglet": (i, which), "on_hook": bname == hook_held})
        for w in self.carton["walls"]:
            for sn in self.carton["sites"][w]:
                sid = m.site(sn).id
                hs.append({"name": sn, "body": self.carton["bodies"][w], "pos": d.site_xpos[sid].copy(), "mat": d.site_xmat[sid].reshape(3, 3).copy(),
                           "pinch_width": float(self.carton["params"]["thickness"]), "kind": "carton", "aglet": None})
        for key, sn in self.lid["sites"].items():
            sid = m.site(sn).id
            hs.append({"name": sn, "body": self.lid["body"], "pos": d.site_xpos[sid].copy(), "mat": d.site_xmat[sid].reshape(3, 3).copy(),
                       "pinch_width": LID_T, "kind": "lid", "aglet": None})
        return hs

    def find_left_handle(self):
        d = self.data
        p = d.site_xpos[self.left_tcp]
        R = d.site_xmat[self.left_tcp].reshape(3, 3)
        hs = self._handles()
        width = self.arms["left"].width()
        names = grasp_mod.find_handle(p, R, [(h["name"], h["pos"], h["mat"], h["pinch_width"]) for h in hs], opening=max(width, 0.004))
        if not names:
            return None
        return next(h for h in hs if h["name"] == names[0])

    def _aglet_canonical(self, eq_name, which):
        """(rel_pos, rel_quat) of the aglet BODY (origin at the chain's joint point) in the weld's body1 frame: the aglet
        protrudes tip-first along the site's +z, its centre 8 mm beyond the site."""
        m = self.model
        eq = m.equality(eq_name).id
        site = m.site(grasp_mod.WELD_SITE[eq_name]).id
        sp = np.array(m.site_pos[site], float)
        sq = np.array(m.site_quat[site], float)
        Rs = quat_to_mat(sq)
        half = LACE_AGLET_LEN / 2
        off = AGLET_OFFSET_GRIPPER if eq_name == "left/grasp" else AGLET_OFFSET_HOOK
        if which == "b":
            # body +z runs toward the tip: origin half an aglet before the centre
            return sp + Rs @ np.array([0.0, 0.0, off - half]), sq
        # aglet A: the body origin IS the tip, body +z runs back into the lace: flip the frame 180 deg about x
        flip = np.array([0.0, 1.0, 0.0, 0.0])
        return sp + Rs @ np.array([0.0, 0.0, off + half]), quat_mul(sq, flip)

    def attach_left(self, hit):
        if hit["aglet"] is not None:
            rp, rq = self._aglet_canonical("left/grasp", hit["aglet"][1])
            grasp_mod.attach(self.model, self.data, "left/grasp", hit["body"], rel_pos=rp, rel_quat=rq, bits=(16, 0),
                             seat_s=AGLET_SEAT_S)
        else:
            grasp_mod.attach(self.model, self.data, "left/grasp", hit["body"], bits=(4, 0))

    def release_left(self):
        grasp_mod.release(self.model, self.data, "left/grasp")

    def left_holding(self):
        return grasp_mod.is_attached(self.model, self.data, "left/grasp")

    def hook_catch(self):
        """Catch the nearest free aglet whose centre is within CATCH_DIST of the tip segment; returns the lace index or None."""
        m, d = self.model, self.data
        tip = d.site_xpos[self.tip_site]
        zt = d.site_xmat[self.tip_site].reshape(3, 3)[:, 2]
        hk = self.info["arms"]["right"]["hook"]
        a, b = tip - zt * float(hk["tip_len"]), tip
        held_left = grasp_mod.is_attached(m, d, "left/grasp")
        best = None
        for i in range(2):
            for which, bname, centre, axis in self.aglet_info(i):
                if bname == held_left:
                    continue
                ab = b - a
                s = float(np.clip(((centre - a) @ ab) / max(1e-9, ab @ ab), 0.0, 1.0))
                dist = float(np.linalg.norm(centre - (a + s * ab)))
                if dist <= CATCH_DIST and (best is None or dist < best[0]):
                    best = (dist, i, which, bname)
        if best is None:
            return None
        rp, rq = self._aglet_canonical("right/hook", best[2])
        grasp_mod.attach(m, d, "right/hook", best[3], rel_pos=rp, rel_quat=rq, seat_s=AGLET_SEAT_S)
        return best[1]

    def release_hook(self):
        grasp_mod.release(self.model, self.data, "right/hook")

    def aglet_pull(self, eq_name):
        """The pull (N) on the aglet weld eq_name holds: the norm of its translational constraint force after the last
        step; None when it holds no lace body."""
        m, d = self.model, self.data
        eq = m.equality(eq_name).id
        if not d.eq_active[eq] or not m.body(int(m.eq_obj2id[eq])).name.startswith("lace"):
            return None
        rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == eq]
        return float(np.linalg.norm(d.efc_force[rows[:3]])) if rows else 0.0

    def hook_loaded(self):
        return grasp_mod.is_attached(self.model, self.data, "right/hook")

    # ------------------------------------------------------------------ safety
    def object_speeds(self):
        d = self.data
        out = []
        for i in range(2):
            v = d.qvel[self.shoe_vadr[i]:self.shoe_vadr[i] + 3]
            out.append(("shoe%d" % i, float(np.linalg.norm(v))))
        out.append(("lid", float(np.linalg.norm(d.qvel[self.lid_vadr:self.lid_vadr + 3]))))
        out.append(("box", float(np.linalg.norm(d.qvel[self.carton_vadr:self.carton_vadr + 3]))))
        return out
