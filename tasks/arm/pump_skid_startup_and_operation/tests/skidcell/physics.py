"""The skid cell's scene (cell internals): the floor, the skid built from the episode's layout (tanks, pipes, the pump, the
duplex filter, the valves, the instruments and the local control station), the rail with its carriage and the Panda,
lights and cameras, composed with MjSpec; plus every physical query the server needs: the devices' positions as
valve openings, the grip rule, button presses, the selector's detents, the instruments' visuals and the contact
classes of the safety watch.

World frame: the floor is z = 0, the skid's front plane y = 0 (everything operable faces -y), x along the skid
(0 .. 4 m); the carriage runs along x with the Panda's base on y = -0.70 at the lift's height, facing +y (its floor
track and mast stand behind the base, clear of the arm).
MuJoCo 3.14, timestep 2 ms, implicitfast, Newton, elliptic cones; a 10 ms control tick = 5 steps.
"""

import math
import os

import mujoco
import numpy as np

from . import devices as dv
from . import grasp as grasp_mod
from . import textures as tx
from .cameras import CAMERA_SPECS, CameraRig, build_cameras
from .control import ArmController, mat_to_quat
from .layout import RAIL

__all__ = ["Scene", "Arm", "PHYS", "HOME_Q", "GRIP", "WELD_ROT_ROWS", "TCP_FORWARD_QUAT", "OPERABLE", "BUTTONS", "LAMPS"]

PHYS = {"timestep": 0.002, "tick_s": 0.01, "weld_solref": (0.004, 1.0), "iterations": 30}
# the Panda's ready posture: the tool points into the skid (+y) at mid height, the fingers' closing axis vertical
HOME_Q = [0.0, -0.85, 0.0, -2.35, 0.0, 3.05, -0.785]
# the tool pointing into the skid (+y world), closing axis +z world: columns x = -x_w, y = +z_w, z = +y_w
TCP_FORWARD_QUAT = [0.0, 0.0, 0.70710678, 0.70710678]
# the grip rule; slip_torque_Nm is the moment about the held part's pivot (v0.6: in physical units, 3 N m; v0.5 wrote 6 N m,
# the grip weld's rotational constraint rows, which carry twice the moment: see WELD_ROT_ROWS)
GRIP = {"box_x": 0.012, "box_z_lo": -0.012, "box_z_hi": 0.014, "angle_deg": 30.0, "slip_force_N": 60.0, "slip_torque_Nm": 3.0,
        "slip_ticks": 3}
# MuJoCo's weld writes its orientation error as a quaternion's vector part, whose rate is half the relative angular
# velocity: its rotational rows' force is twice the moment the weld transmits (measured on development sites: a torque
# T applied about a held valve's stem raises the rows' norm by 2T). The slip rule compares the rows' norm with
# WELD_ROT_ROWS * slip_torque_Nm (= 6.0, the v0.5 cell's number), so it slips exactly where it did.
WELD_ROT_ROWS = 2.0
PAD_FRICTION = (1.5, 0.005, 0.0001)
GRIPPER = {"kp": 2500.0, "kv": 50.0, "max_force": 60.0}
RAIL_DRIVE = {"kp": 40000.0, "kv": 3500.0, "max_force": 3000.0, "carriage_mass": 30.0}
VALVES = ("V-101", "V-102", "V-103", "V-104", "V-105", "V-106", "V-107", "V-111", "V-112", "V-301", "V-302", "V-303")
OPERABLE = VALVES + ("HS-101",)
ANGLE_VALVES = ("V-111", "V-303")      # opening() gives the lever angle (0 = A .. pi = B), not a fraction
BUTTONS = ("START-101", "STOP-101", "START-301", "STOP-301", "RESET", "ESTOP")
LAMPS = ("RUN-101", "FAULT-101", "RUN-301", "ALM")
SELECTOR_DETENT = {"k_Nm_per_rad": 1.5, "max_Nm": 0.45, "snap_rad": 0.12}
# contact bits: robot (2, 1) touches the world (1, 2); the world never touches itself; a held body (4, 0) touches nothing
BITS_ROBOT = (2, 1)
BITS_WORLD = (1, 2)
BITS_HELD = (4, 0)


class Arm(ArmController):
    """The cell's arm controller with the few extras the server needs (the shoe cell's, same benchmark)."""

    def __init__(self, model, data, prefix, tcp_site, home_q, tick_s):
        ArmController.__init__(self, model, data, prefix, tcp_site, gripper=True, tick_s=tick_s)
        self.home_q = np.asarray(home_q, float)
        self.hold_w = None

    @staticmethod
    def mat_to_quat(R):
        return mat_to_quat(R)

    def ranges(self):
        return [[float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1])] for j in self.joints]

    def rebase(self):
        self.q_cmd = self.q().copy()
        self.q_goal = self.q_cmd.copy()
        self.width_cmd = self.width()
        self.width_goal = self.width_cmd
        self.write_ctrl()

    def width(self):
        return max(0.0, ArmController.width(self))

    def hold_width(self, w):
        """While the gripper holds a handle its finger goal never goes below the handle's width."""
        self.hold_w = None if w is None else float(w)
        if self.hold_w is not None and self.width_goal < self.hold_w:
            self.width_goal = self.hold_w

    def command_width(self, width):
        ArmController.command_width(self, width)
        if self.hold_w is not None and self.width_goal < self.hold_w:
            self.width_goal = self.hold_w


def _add_panda(spec, parent, prefix, pos, quat, panda_dir):
    """Attach MuJoCo Menagerie's Panda (with its hand) to `parent` at (pos, quat); a tcp site between the finger pads."""
    child = mujoco.MjSpec.from_file(os.path.join(panda_dir, "panda.xml"))
    for k in list(child.keys):
        child.delete(k)
    for l in list(child.lights):
        child.delete(l)
    frame = parent.add_frame(name=prefix + "base_frame", pos=list(pos), quat=list(quat))
    spec.attach(child, prefix=prefix, frame=frame)
    for b in spec.bodies:
        if b.name.startswith(prefix) and b.name != prefix + "link0":
            b.gravcomp = 1.0
    hand = spec.body(prefix + "hand")
    hand.add_site(name=prefix + "tcp", pos=[0.0, 0.0, 0.1034], size=[0.003, 0.003, 0.003], rgba=[0.2, 0.9, 0.2, 0.5], group=4)
    pads = []
    for g in spec.geoms:
        b = g.parent
        if b is not None and b.name in (prefix + "left_finger", prefix + "right_finger") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
            side = "left" if "left" in b.name else "right"
            g.name = "%spad_%s_%d" % (prefix, side, len([p for p in pads if side in p]))
            g.friction = list(PAD_FRICTION)
            g.condim = 4
            pads.append(g.name)
    for a in spec.actuators:
        if a.name == prefix + "actuator8":
            scale = 0.04 / 255.0
            a.gainprm = [GRIPPER["kp"] * scale] + [0.0] * 9
            a.biasprm = [0.0, -GRIPPER["kp"], -GRIPPER["kv"]] + [0.0] * 7
            a.forcerange = [-GRIPPER["max_force"], GRIPPER["max_force"]]
    return {"prefix": prefix, "tcp_site": prefix + "tcp", "tcp_local": (0.0, 0.0, 0.1034), "pads": pads,
            "finger_bodies": [prefix + "left_finger", prefix + "right_finger"], "hand_body": prefix + "hand"}


class Scene(object):
    def __init__(self, given_dir, layout, cfg=None):
        self.given_dir = given_dir
        self.L = layout
        self.cfg = cfg or {}
        self.tick_s = float(PHYS["tick_s"])
        self.substeps = int(round(self.tick_s / float(PHYS["timestep"])))
        self._build()
        self._index()

    # ------------------------------------------------------------------ build
    def _build(self):
        L = self.L
        spec = mujoco.MjSpec()
        spec.modelname = "pump_skid_cell"
        spec.compiler.degree = False
        spec.compiler.autolimits = True
        o = spec.option
        o.timestep = PHYS["timestep"]
        o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        o.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        o.solver = mujoco.mjtSolver.mjSOL_NEWTON
        o.iterations = int(PHYS["iterations"])
        o.ls_iterations = 30
        o.impratio = 10.0
        o.gravity = [0.0, 0.0, -9.81]
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
        spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
        spec.visual.headlight.ambient = [0.32, 0.32, 0.32]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]
        spec.visual.map.znear = 0.01
        spec.visual.map.zfar = 40.0
        spec.visual.quality.shadowsize = 0
        xt = float(L.get("x_total", 5.0))
        spec.stat.extent = 3.5
        spec.stat.center = [xt / 2.0, -0.3, 0.8]
        gd = spec.default.geom
        gd.condim = 3
        gd.friction = [1.0, 0.005, 0.0001]
        b = dv.Builder(spec)
        self.b = b
        w = spec.worldbody
        # floor, base frame, back wall
        spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.42, 0.43, 0.45], rgb2=[0.36, 0.37, 0.39], width=256, height=256)
        fm = spec.add_material(name="floor_mat", texrepeat=[12, 12], texuniform=True, reflectance=0.0)
        fm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[7.0, 6.0, 0.1], pos=[xt / 2.0, 0.0, 0.0], material="floor_mat")
        w.add_geom(name="skid/base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[xt / 2.0 - 0.02, 0.50, 0.025], pos=[xt / 2.0, 0.45, 0.025],
                   material=b.mats["frame"])
        w.add_geom(name="skid/wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[xt / 2.0 + 0.2, 0.02, 1.15], pos=[xt / 2.0, 1.02, 1.15],
                   rgba=[0.70, 0.72, 0.70, 1], contype=0, conaffinity=0, group=1)
        # lights (no shadows: speed); the episode scales their strength
        g = float(L.get("light", 1.0))
        for i, (pos, d) in enumerate((((xt / 2.0, -2.8, 3.2), (0.0, 0.6, -0.8)), ((0.5, -1.8, 2.4), (0.35, 0.7, -0.6)),
                                      ((xt - 0.5, -1.8, 2.4), (-0.35, 0.7, -0.6)))):
            w.add_light(name="light%d" % i, pos=list(pos), dir=list(d), diffuse=[0.42 * g] * 3, specular=[0.05] * 3, castshadow=False)
        # the skid
        self.dev = {}
        self._build_skid(b, L)
        # the rail, the carriage with its lift, and the Panda (its base rides the lift, facing +y). v0.5.2: the floor track,
        # the sled and the mast stand 0.32-0.46 m behind the base, clear of the arm's elbow (which folds up to about
        # 0.25 m behind the base), and the platform is cantilevered off a guide plate on the mast's front face (before,
        # the mast passed through the platform, link0 and the elbow, and a low platform sank into the sled and the
        # track). These geoms are visual only (contype 0, as before); the two bodies keep the inertials v0.5 compiled
        # from the old boxes, written out here, so the compiled dynamics are bit-identical to v0.5.
        ry = float(RAIL["y"])
        w.add_geom(name="rail/track", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5 * (RAIL["x_hi"] - RAIL["x_lo"]) + 0.15, 0.06, 0.03],
                   pos=[0.5 * (RAIL["x_lo"] + RAIL["x_hi"]), ry - 0.39, 0.03], material=b.mats["steel"], contype=0, conaffinity=0,
                   group=1)
        x0 = float(np.clip(L.get("rail_start", xt / 2.0), RAIL["x_lo"], RAIL["x_hi"]))
        z0 = float(np.clip(L.get("lift_start", RAIL["lift_start"]), RAIL["lift_lo"], RAIL["lift_hi"]))
        car = w.add_body(name="rail/carriage", pos=[0.0, ry, 0.0])
        car.explicitinertial = True
        car.mass = RAIL_DRIVE["carriage_mass"]
        car.ipos = [0.0, -0.04200000000000001, 0.22549999999999998]
        car.iquat = [0.703571518318439, 0.07061953420327198, -0.07061953420327198, 0.703571518318439]
        car.inertia = [3.0109824986275027, 2.8923724999999987, 0.5883900013724965]
        car.add_joint(name="rail/x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1.0, 0.0, 0.0], range=[RAIL["x_lo"], RAIL["x_hi"]],
                      damping=50.0, armature=1.0)
        car.add_geom(name="rail/sled", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.22, 0.07, 0.03], pos=[0.0, -0.39, 0.09],
                     material=b.mats["frame"], contype=0, conaffinity=0)
        car.add_geom(name="rail/mast", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.07, 0.04, 0.39], pos=[0.0, -0.36, 0.51],
                     material=b.mats["steel"], contype=0, conaffinity=0)
        lift = car.add_body(name="rail/lift", pos=[0.0, 0.0, 0.0], gravcomp=1.0)
        lift.explicitinertial = True
        lift.mass = 8.0
        lift.ipos = [0.0, -0.02, -0.02]
        lift.iquat = [1.0, 0.0, 0.0, 0.0]
        lift.inertia = [0.061066666666666665, 0.04613333333333334, 0.10506666666666668]
        lift.add_joint(name="rail/z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 0.0, 1.0], range=[RAIL["lift_lo"], RAIL["lift_hi"]],
                       damping=50.0, armature=1.0)
        lift.add_geom(name="rail/platform", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.13, 0.15, 0.02], pos=[0.0, -0.02, -0.02],
                      material=b.mats["frame"], contype=0, conaffinity=0)
        lift.add_geom(name="rail/cantilever", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.075, 0.02], pos=[0.0, -0.2425, -0.02],
                      material=b.mats["frame"], contype=0, conaffinity=0)
        lift.add_geom(name="rail/guide", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.007, 0.07], pos=[0.0, -0.3125, 0.03],
                      material=b.mats["frame"], contype=0, conaffinity=0)
        for jname, aname, lo, hi in (("rail/x", "rail/drive", RAIL["x_lo"], RAIL["x_hi"]), ("rail/z", "rail/lift_drive", RAIL["lift_lo"], RAIL["lift_hi"])):
            act = spec.add_actuator(name=aname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm = [RAIL_DRIVE["kp"]] + [0.0] * 9
            act.biasprm = [0.0, -RAIL_DRIVE["kp"], -RAIL_DRIVE["kv"]] + [0.0] * 7
            act.ctrlrange = [lo, hi]
            act.forcerange = [-RAIL_DRIVE["max_force"], RAIL_DRIVE["max_force"]]
        self.rail_x0, self.lift_z0 = x0, z0
        panda_dir = os.path.join(self.given_dir, "panda")
        yaw = math.pi / 2                                         # base x -> world +y (facing the skid)
        self.panda = _add_panda(spec, lift, "panda/", (0.0, 0.0, 0.0), [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], panda_dir)
        spec.body("panda/link0").gravcomp = 1.0
        # the grip weld (inactive; body2 = world placeholder)
        data = [0.0] * 11
        data[10] = 1.0
        spec.add_equality(name="grip", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="panda/hand", name2="world",
                          active=False, solref=list(PHYS["weld_solref"]), data=data)
        self.cam_info = build_cameras(spec, {"tcp_local": self.panda["tcp_local"]})
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer_sizes = {k: v["size"] for k, v in CAMERA_SPECS.items()}

    def _pipe(self, b, name, p0, p1, r=0.024, visual=False):
        dv.add_pipe(b, b.w, "pipe/" + name, p0, p1, r=r, visual=visual)

    def _gap(self, tag, axis):
        """Half the length of a valve's body with its flanges along `axis` (where the pipe ends)."""
        t = self.dev[tag]["type"]
        small = bool(self.L["devices"][tag].get("small", False))
        if t == "handwheel":
            return 0.057 if axis == "x" else 0.062
        if t == "lever":
            if axis == "x":
                return 0.042 if small else 0.052
            return 0.044 if small else 0.054
        return 0.037

    def _build_skid(self, b, L):
        D = L["devices"]
        ST = L["site"]["tags"]
        s = float(L["side"])
        zs, zd = float(L["z_suc"]), float(L["z_dis"])
        X = lambda tag: float(D[tag]["pos"][0])  # noqa: E731
        Z = lambda tag: float(D[tag]["pos"][2])  # noqa: E731
        dev = self.dev
        # vessels and machines
        for tag in ("T-101", "T-201"):
            p = D[tag]
            dev[tag] = dv.add_tank(b, tag, p["centre_xy"], p["radius"], p["z0"], p["height"], label=ST[tag])
        for tag, letter in (("T-301", "A"), ("T-302", "B")):
            p = D[tag]
            dev[tag] = dv.add_drum(b, tag, p["centre_xy"], p["radius"], p["z0"], p["height"], label=ST[tag], letter=letter)
        dev["P-101"] = dv.add_pump(b, "P-101", D["P-101"]["casing"], direction=D["P-101"]["direction"], label=ST["P-101"])
        p301 = D["P-301"]["pos"]
        dev["P-301"] = dv.add_dosing_pump(b, "P-301", p301, direction=D["P-301"].get("direction", 1), label=ST["P-301"])
        f = D["F-101"]
        for k, (xc, yc) in (("A", f["a"]), ("B", f["b"])):
            b.w.add_geom(name="F-101/%s" % k, type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[xc, yc, f["z0"], xc, yc, f["z1"]],
                         size=[0.075, 0, 0], material=b.mats["valve"])
            b.w.add_geom(name="F-101/%s_cover" % k, type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[xc, yc, f["z1"], xc, yc, f["z1"] + 0.025],
                         size=[0.085, 0, 0], material=b.mats["steel"])
            dv.add_plate(b, b.w, "F-101/label%s" % k, (xc, yc - 0.077, f["z0"] + 0.12), 0.03, 0.03,
                         tx.tag_plate(k, w_px=96, h_px=96), thick=0.0015)
        dv.add_plate(b, b.w, "F-101/name", (0.5 * (f["a"][0] + f["b"][0]), 0.0, f["z0"] + 0.03), 0.04, 0.015,
                     tx.tag_plate(ST["F-101"], w_px=256, h_px=96))
        xa, xb = f["a"][0], f["b"][0]
        # valves, each built the way the site built it
        for tag in VALVES:
            p = D[tag]
            if p["builder"] == "handwheel":
                dev[tag] = dv.add_handwheel_valve(b, tag, p["pos"], wheel_d=p["wheel_d"], spokes=p["spokes"], turns=p["turns"],
                                                  stem_len=p["stem_len"], friction=p["friction"], damping=p["damping"],
                                                  wheel_rgba=p["wheel_rgba"], pipe_axis=p.get("pipe_axis", "x"),
                                                  tag_side=p.get("tag_side", 1), wheel_phase=p["wheel_phase"], label=ST[tag])
                dev[tag]["kind"] = p["kind"]
            elif p["builder"] == "lever":
                dev[tag] = dv.add_lever_valve(b, tag, p["pos"], lever_len=p["lever_len"], closed_dir=p["closed_dir"],
                                              range_rad=p["range_rad"], stem_len=p["stem_len"], friction=p["friction"],
                                              grip_rgba=p["grip_rgba"], pipe_axis=p.get("pipe_axis", "x"),
                                              tag_offset=p.get("tag_offset", (0.0, -0.075)), small=p.get("small", False),
                                              label=ST[tag], end_labels=p.get("labels"))
            else:
                dev[tag] = dv.add_knob_valve(b, tag, p["pos"], turns=p["turns"], stem_len=p["stem_len"], friction=p["friction"],
                                             knob_r=p.get("knob_r", 0.019), tag_offset=p.get("tag_offset", (0.0, -0.06)),
                                             pipe_axis=p.get("pipe_axis", "x"), label=ST[tag])
        gx = lambda tag: self._gap(tag, "x")  # noqa: E731
        gz = lambda tag: self._gap(tag, "z")  # noqa: E731
        # instruments
        for tag in ("PI-101", "PI-103", "PDI-111"):
            p = D[tag]
            dev[tag] = dv.add_dial_gauge(b, tag, p["pos"], p["kind"], dial_d=p["dial_d"], riser_to=p.get("riser_to"),
                                         title=p.get("title", ""), label=ST[tag])
        p = D["FI-105"]
        dev["FI-105"] = dv.add_rotameter(b, "FI-105", p["pos"], p["kind"], height=p["height"], label=ST["FI-105"])
        for tag in ("LG-101", "LG-201"):
            p = D[tag]
            dev[tag] = dv.add_level_gauge(b, tag, p["pos"], height=p["height"], label=ST[tag])
        for tag in ("VS-102", "VS-112"):
            dev[tag] = dv.add_sight_glass(b, tag, D[tag]["pos"])
        dev["AI-201"] = dv.add_display(b, "AI-201", D["AI-201"]["pos"], unit="mg/L Cl2", title=ST["AI-201"])
        self._build_panel(b, D["LCS"]["centre"], bool(D["LCS"].get("p101_left", True)), ST)
        # pipes
        pipe = lambda *a, **k: self._pipe(b, *a, **k)  # noqa: E731
        t101 = D["T-101"]["centre_xy"]
        t201 = D["T-201"]["centre_xy"]
        xp = D["P-101"]["casing"][0]
        x101, x103, x104, x105, x107 = X("V-101"), X("V-103"), X("V-104"), X("V-105"), X("V-107")
        xfi = float(D["FI-105"]["pos"][0])
        x111, z111 = X("V-111"), Z("V-111")
        # the suction: T-101 -> the strainer's chambers -> the changeover valve -> V-101 -> P-101
        pipe("feed1", (t101[0] + s * 0.18, 0.30, 0.30), (t101[0] + s * 0.18, 0.04, 0.30))
        pipe("feed2", (t101[0] + s * 0.18, 0.04, 0.30), (xa, 0.04, 0.30))
        pipe("manifold", (xa, 0.04, 0.30), (xb, 0.04, 0.30))
        pipe("fout1", (x111 + s * gx("V-111"), 0.0, z111), (xb + s * 0.11, 0.0, z111))
        pipe("fout2", (xb + s * 0.11, 0.0, z111), (xb + s * 0.11, 0.0, zs))
        pipe("suc1", (xb + s * 0.11, 0.0, zs), (x101 - s * gx("V-101"), 0.0, zs))
        pipe("suc2", (x101 + s * gx("V-101"), 0.0, zs), (xp - s * 0.17, 0.0, zs))
        pipe("suc3", (xp - s * 0.17, 0.0, zs), (xp - s * 0.17, 0.0, 0.44))
        pipe("suc4", (xp - s * 0.17, 0.0, 0.44), (xp - s * 0.10, 0.0, 0.44))
        # the discharge header: P-101 -> V-103 -> the minimum-flow tee, the flow meter's loop and V-105 (with its bypass
        # through V-107 below the header) in the site's order -> T-201
        pipe("dis1", (xp, 0.0, 0.55), (xp, 0.0, zd))
        pipe("dis2", (xp, 0.0, zd), (x103 - s * gx("V-103"), 0.0, zd))
        pipe("pi103", (xp, 0.0, zd), (xp, 0.0, zd + 0.21), r=0.008)
        items = sorted([("tee104", x104), ("fi", xfi), ("v105", x105)], key=lambda it: s * it[1])
        cur = x103 + s * gx("V-103")
        for name, xx in items:
            if name == "tee104":
                entry = exit_ = xx
            elif name == "fi":
                entry, exit_ = xx - s * 0.10, xx + s * 0.10
            else:
                entry, exit_ = xx - s * gx("V-105"), xx + s * gx("V-105")
            if abs(entry - cur) > 1e-3:
                pipe("hdr_" + name, (cur, 0.0, zd), (entry, 0.0, zd))
            cur = exit_
        pipe("hdr_end", (cur, 0.0, zd), (t201[0] - s * 0.16, 0.0, zd))
        pipe("dis6", (t201[0] - s * 0.16, 0.0, zd), (t201[0] - s * 0.16, 0.30, zd))
        pipe("fi1", (xfi - s * 0.10, 0.0, zd), (xfi - s * 0.10, 0.0, zd - 0.24))
        pipe("fi2", (xfi - s * 0.10, 0.0, zd - 0.24), (xfi, 0.0, zd - 0.24))
        pipe("fi3", (xfi, 0.0, zd - 0.24), (xfi, 0.0, zd - 0.19))
        pipe("fi4", (xfi, 0.0, zd + 0.235), (xfi, 0.0, zd + 0.28))
        pipe("fi5", (xfi, 0.0, zd + 0.28), (xfi + s * 0.10, 0.0, zd + 0.28))
        pipe("fi6", (xfi + s * 0.10, 0.0, zd + 0.28), (xfi + s * 0.10, 0.0, zd))
        zb = Z("V-107")
        bl, br = x105 - s * 0.13, x105 + s * 0.13
        pipe("byp1", (bl, 0.0, zd), (bl, 0.0, zb))
        pipe("byp2", (bl, 0.0, zb), (x107 - s * gx("V-107"), 0.0, zb))
        pipe("byp3", (x107 + s * gx("V-107"), 0.0, zb), (br, 0.0, zb))
        pipe("byp4", (br, 0.0, zb), (br, 0.0, zd))
        z104 = Z("V-104")
        pipe("mf1", (x104, 0.0, zd), (x104, 0.0, z104 + gz("V-104")))
        pipe("mf2", (x104, 0.0, z104 - gz("V-104")), (x104, 0.0, 0.20), r=0.016)
        pipe("mf3", (x104, 0.0, 0.20), (x104, 0.12, 0.20), r=0.016, visual=True)
        pipe("mf4", (x104, 0.12, 0.20), (t101[0] + s * 0.25, 0.12, 0.20), r=0.016, visual=True)
        # T-101's make-up line: from the plant's supply down to V-106 and into the tank (the float valve is inside it)
        x106, z106 = X("V-106"), Z("V-106")
        xr = x106 + s * 0.14
        x_in = t101[0] + s * 0.22
        pipe("mk1", (xr, 0.0, 2.05), (xr, 0.0, z106))
        pipe("mk2", (xr, 0.0, z106), (x106 + s * gx("V-106"), 0.0, z106))
        pipe("mk3", (x106 - s * gx("V-106"), 0.0, z106), (x_in, 0.0, z106))
        pipe("mk4", (x_in, 0.0, z106), (x_in, 0.0, 1.97), visual=True)
        pipe("mk5", (x_in, 0.0, 1.97), (x_in, t101[1] - 0.08, 1.97), visual=True)
        pipe("mk6", (xr, 0.0, 2.05), (xr, 0.95, 2.05), visual=True)
        pipe("sup1", (t101[0], t101[1], 1.95), (t101[0], 0.95, 1.95), r=0.020, visual=True)
        # vents, dosing
        vs102, vs112 = D["VS-102"]["pos"], D["VS-112"]["pos"]
        x102, x112, x302, x301 = X("V-102"), X("V-112"), X("V-302"), X("V-301")
        z102, z112, z302, z301 = Z("V-102"), Z("V-112"), Z("V-302"), Z("V-301")
        pipe("vent102a", (x102, 0.0, 0.52), (x102, 0.0, z102 - gz("V-102")), r=0.012)
        pipe("vent102b", (x102, 0.0, z102 + gz("V-102")), (x102, 0.0, vs102[2]), r=0.010)
        pipe("vent102c", (x102, 0.0, vs102[2]), (vs102[0], vs102[1], vs102[2]), r=0.008, visual=True)
        pipe("vent112a", (x112, 0.0, 0.925), (x112, 0.0, z112 - gz("V-112")), r=0.012)
        pipe("vent112b", (x112, 0.0, z112 + gz("V-112")), (x112, 0.0, vs112[2]), r=0.010)
        pipe("vent112c", (x112, 0.0, vs112[2]), (vs112[0], vs112[1], vs112[2]), r=0.008, visual=True)
        pipe("vent302", (x302, 0.0, 0.37), (x302, 0.0, z302 - gz("V-302")), r=0.010)
        pipe("dose1", (p301[0] - s * 0.11, 0.04, 0.30), (x301, 0.04, 0.30), r=0.008)
        pipe("dose2", (x301, 0.04, 0.30), (x301, 0.0, 0.30), r=0.008)
        pipe("dose3", (x301, 0.0, 0.30), (x301, 0.0, z301 - gz("V-301")), r=0.008)
        pipe("dose4", (x301, 0.0, z301 + gz("V-301")), (x301, 0.0, zd - 0.024), r=0.008)
        # the dosing pump's suction: the drums -> the selector V-303 -> P-301
        x303, z303 = X("V-303"), Z("V-303")
        for k, drum in (("a", "T-301"), ("b", "T-302")):
            dd = D[drum]
            dx, dy = float(dd["centre_xy"][0]), float(dd["centre_xy"][1])
            sx = 1.0 if dx > x303 else -1.0
            p0 = (x303 + sx * gx("V-303"), 0.0, z303)
            p1 = (x303 + sx * (gx("V-303") + 0.05), 0.0, z303)
            pipe("drum%s1" % k, p0, p1, r=0.008)
            pipe("drum%s2" % k, p1, (p1[0], dy, z303), r=0.008, visual=True)
            pipe("drum%s3" % k, (p1[0], dy, z303), (dx, dy, z303), r=0.008, visual=True)
            pipe("drum%s4" % k, (dx, dy, z303), (dx, dy, float(dd["z0"]) + float(dd["height"])), r=0.008, visual=True)
        pipe("dsuc1", (x303, 0.0, z303 - 0.032), (x303, 0.0, 0.20), r=0.008, visual=True)
        pipe("dsuc2", (x303, 0.0, 0.20), (p301[0] - s * 0.09, 0.04, 0.20), r=0.008, visual=True)
        # the level gauges' connections to their tanks (visual)
        for tag, tc in (("LG-101", t101), ("LG-201", t201)):
            g = D[tag]["pos"]
            for zz in (g[2] - 0.04, g[2] + D[tag]["height"] + 0.04):
                nm = "%s_%d" % (tag, int(zz * 100))
                pipe(nm + "a", (g[0], g[1] + 0.035, zz), (g[0], 0.20, zz), r=0.012, visual=True)
                pipe(nm + "b", (g[0], 0.20, zz), (tc[0], 0.20, zz), r=0.012, visual=True)

    def _build_panel(self, b, centre, p101_left, ST):
        """The local control station: a grey box on a stand; P-101's selector and buttons on one half, P-301's on the
        other (the site's choice), the lamps in the middle, RESET under P-101's group and the emergency stop under
        P-301's. Its labels carry the site's tags."""
        cx, cy, cz = [float(v) for v in centre]
        w = b.w
        w.add_geom(name="LCS/housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.20, 0.055, 0.235], pos=[cx, cy + 0.055, cz],
                   material=b.mats["panel"])
        w.add_geom(name="LCS/stand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.04, (cz - 0.235) / 2], pos=[cx, cy + 0.06, (cz - 0.235) / 2],
                   material=b.mats["frame"])
        dv.add_plate(b, w, "LCS/title", (cx, cy - 0.0005, cz + 0.195), 0.17, 0.022,
                     tx.label_plate(["%s  LOCAL CONTROL STATION" % ST["LCS"]], w_px=760, h_px=96))
        dev = self.dev
        g1 = -1.0 if p101_left else 1.0
        x1, x2 = cx + g1 * 0.125, cx - g1 * 0.125
        dv.add_plate(b, w, "LCS/p101", (x1, cy - 0.0005, cz + 0.150), 0.045, 0.012, tx.label_plate(ST["P-101"], w_px=240, h_px=64))
        dv.add_plate(b, w, "LCS/p301", (x2, cy - 0.0005, cz + 0.150), 0.045, 0.012, tx.label_plate(ST["P-301"], w_px=240, h_px=64))
        dev["HS-101"] = dv.add_selector(b, w, "HS-101", (x1, cy, cz + 0.06))
        dev["START-101"] = dv.add_button(b, w, "START-101", (x1, cy, cz - 0.035), (0.10, 0.62, 0.16, 1), "START")
        dev["STOP-101"] = dv.add_button(b, w, "STOP-101", (x1, cy, cz - 0.115), (0.80, 0.10, 0.08, 1), "STOP")
        dev["START-301"] = dv.add_button(b, w, "START-301", (x2, cy, cz + 0.010), (0.10, 0.62, 0.16, 1), "START")
        dev["STOP-301"] = dv.add_button(b, w, "STOP-301", (x2, cy, cz - 0.075), (0.80, 0.10, 0.08, 1), "STOP")
        dev["RUN-101"] = dv.add_lamp(b, w, "RUN-101", (cx, cy, cz + 0.12), (0.25, 1.0, 0.30, 1), "%s RUN" % ST["P-101"])
        dev["FAULT-101"] = dv.add_lamp(b, w, "FAULT-101", (cx, cy, cz + 0.05), (1.0, 0.15, 0.10, 1), "%s FAULT" % ST["P-101"])
        dev["ALM"] = dv.add_lamp(b, w, "ALM", (cx, cy, cz - 0.02), (1.0, 0.62, 0.05, 1), "COMMON ALARM")
        dev["RUN-301"] = dv.add_lamp(b, w, "RUN-301", (x2, cy, cz + 0.085), (0.25, 1.0, 0.30, 1), "%s RUN" % ST["P-301"])
        dev["RESET"] = dv.add_button(b, w, "RESET", (cx + g1 * 0.07, cy, cz - 0.175), (0.12, 0.18, 0.55, 1), "RESET / ACK")
        xe = cx - g1 * 0.10
        dv.add_plate(b, w, "LCS/estop_disc", (xe, cy - 0.0004, cz - 0.17), 0.036, 0.036,
                     tx.label_plate(" ", w_px=64, h_px=64, bg=(240, 200, 20)), thick=0.0008)
        dev["ESTOP"] = dv.add_button(b, w, "ESTOP", (xe, cy, cz - 0.17), (0.85, 0.05, 0.05, 1), "EMERGENCY STOP", mushroom=True)
        self.panel = {"centre": [cx, cy, cz], "static_geoms": ["LCS/housing", "LCS/stand"], "p101_left": bool(p101_left)}

    # ------------------------------------------------------------------ index
    def _index(self):
        m, d = self.model, self.data
        self.arm = Arm(m, d, "panda/", "panda/tcp", HOME_Q, self.tick_s)
        self.rail_qadr = m.jnt_qposadr[m.joint("rail/x").id]
        self.lift_qadr = m.jnt_qposadr[m.joint("rail/z").id]
        self.rail_act = m.actuator("rail/drive").id
        self.lift_act = m.actuator("rail/lift_drive").id
        self.tcp = m.site("panda/tcp").id
        self.hand_body = m.body("panda/hand").id
        self.cameras = CameraRig(m, self.renderer_sizes, scene=self)
        # operable joints
        self.jq, self.jv, self.jrange = {}, {}, {}
        for tag in OPERABLE:
            jn = self.dev[tag]["joint"]
            j = m.joint(jn).id
            self.jq[tag] = m.jnt_qposadr[j]
            self.jv[tag] = m.jnt_dofadr[j]
            self.jrange[tag] = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1]))
        self.btn_q = {t: m.jnt_qposadr[m.joint(self.dev[t]["joint"]).id] for t in BUTTONS}
        self.btn_body = {t: m.body(self.dev[t]["body"]).id for t in BUTTONS}
        self.moving_body = {t: m.body(self.dev[t]["body"]).id for t in OPERABLE}
        self.body_tag = {v: k for k, v in self.moving_body.items()}
        for t in BUTTONS:
            self.body_tag[self.btn_body[t]] = t
        # visual (kinematic) joints and geoms
        self.vis_q = {}
        for tag in ("PI-101", "PI-103", "PDI-111", "FI-105"):
            self.vis_q[tag] = m.jnt_qposadr[m.joint(self.dev[tag]["joint"]).id]
        self.handwheels = [t for t in VALVES if self.dev[t]["type"] == "handwheel"]
        for tag in self.handwheels:
            self.vis_q[tag + "/rise"] = m.jnt_qposadr[m.joint(self.dev[tag]["rise_joint"]).id]
        kin_bodies = ["PI-101/needle", "PI-103/needle", "PDI-111/needle", "FI-105/float"] + [t + "/stemrod" for t in self.handwheels]
        for bn in kin_bodies:
            m.body_gravcomp[m.body(bn).id] = 1.0
        # an operable part stays where it was left: MuJoCo's friction loss is a soft constraint that lets a constant
        # load (a lever's own weight) creep, so the moving parts carry no weight
        for t in OPERABLE:
            m.body_gravcomp[self.moving_body[t]] = 1.0
        self.lens = {t: m.geom(self.dev[t]["lens"]).id for t in LAMPS}
        self.sight = {t: m.geom(self.dev[t]["glass"]).id for t in ("VS-102", "VS-112")}
        self.flaps = {t: m.geom(self.dev[t]["flaps"]).id for t in ("LG-101", "LG-201")}
        self.drum_liquid = {t: m.geom(self.dev[t]["liquid"]).id for t in ("T-301", "T-302")}
        self.segs = [{k: m.geom(g).id for k, g in dig.items()} for dig in self.dev["AI-201"]["segments"]]
        self.dp_geom = m.geom(self.dev["AI-201"]["dp"]).id
        # geom classes for the safety watch: 1 robot link, 2 finger, 5 static skid / floor, 7 operable moving part, 0 other
        gc = np.zeros(m.ngeom, dtype=int)
        finger_bodies = {m.body(n).id for n in self.panda["finger_bodies"]}
        moving = set(self.moving_body.values()) | set(self.btn_body.values())
        for gi in range(m.ngeom):
            bid = int(m.geom_bodyid[gi])
            bname = m.body(bid).name
            visual = int(m.geom_contype[gi]) == 0 and int(m.geom_conaffinity[gi]) == 0
            if bname.startswith("panda/"):
                gc[gi] = 2 if bid in finger_bodies else 1
                if not visual:
                    m.geom_contype[gi], m.geom_conaffinity[gi] = BITS_ROBOT
            elif bname.startswith("rail/"):
                gc[gi] = 0
            elif visual:
                gc[gi] = 0
            elif bid in moving:
                gc[gi] = 7
                m.geom_contype[gi], m.geom_conaffinity[gi] = BITS_WORLD
            elif bid == 0:
                gc[gi] = 5
                m.geom_contype[gi], m.geom_conaffinity[gi] = BITS_WORLD
            else:
                gc[gi] = 0
                m.geom_contype[gi], m.geom_conaffinity[gi] = 0, 0
        self.geom_class = gc
        # the external wrench estimate (v0.6): what the robot's joint torques feel of the robot's contacts with the
        # world and of the grip, mapped to the tcp as a Franka does (tau_ext = J_c^T f; F = (J J^T)^-1 J tau_ext)
        self.arm_dofs = np.asarray(self.arm.vadr, dtype=int)
        self.robot_geom = np.isin(gc, (1, 2))
        self.grip_eq = m.equality("grip").id
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self.physics_fault = False
        self._f6 = np.zeros(6)
        self.reset_state()

    # ------------------------------------------------------------------ initial state
    def reset_state(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.rail_qadr] = self.rail_x0
        d.ctrl[self.rail_act] = self.rail_x0
        d.qpos[self.lift_qadr] = self.lift_z0
        d.ctrl[self.lift_act] = self.lift_z0
        for i, jn in enumerate(self.arm.joints):
            d.qpos[self.arm.qadr[i]] = HOME_Q[i]
        for jn in ("panda/finger_joint1", "panda/finger_joint2"):
            d.qpos[m.jnt_qposadr[m.joint(jn).id]] = 0.04
        init = self.L["initial"]
        for tag in OPERABLE:
            if tag == "HS-101":
                d.qpos[self.jq[tag]] = self.dev["HS-101"]["positions"][init.get("HS-101", "OFF")]
                continue
            lo, hi = self.jrange[tag]
            if tag in ANGLE_VALVES:
                d.qpos[self.jq[tag]] = float(np.clip(float(init.get(tag, 0.0)), lo, hi))
            else:
                d.qpos[self.jq[tag]] = lo + float(init.get(tag, 0.0)) * (hi - lo)
        mujoco.mj_forward(m, d)
        self.arm.rebase()
        self.arm.set_mode("joint")
        self.arm.q_goal = np.asarray(HOME_Q, float).copy()
        self.arm.command_width(0.08)
        self.rail_cmd = self.rail_x0
        self.rail_goal = self.rail_x0
        self.lift_cmd = self.lift_z0
        self.lift_goal = self.lift_z0
        self.held = None

    # ------------------------------------------------------------------ stepping
    def tick_rail(self):
        dx = float(RAIL["speed_m_s"]) * self.tick_s
        self.rail_cmd += float(np.clip(self.rail_goal - self.rail_cmd, -dx, dx))
        self.data.ctrl[self.rail_act] = self.rail_cmd
        dz = float(RAIL["lift_speed_m_s"]) * self.tick_s
        self.lift_cmd += float(np.clip(self.lift_goal - self.lift_cmd, -dz, dz))
        self.data.ctrl[self.lift_act] = self.lift_cmd

    def command_rail(self, x=None, lift=None):
        """The carriage's goal along the rail (x, m) and / or the lift's base height (m); ramped at the drives' speeds."""
        if x is not None:
            x = float(x)
            if not math.isfinite(x):
                raise ValueError("non-finite rail target")
            self.rail_goal = float(np.clip(x, RAIL["x_lo"], RAIL["x_hi"]))
        if lift is not None:
            z = float(lift)
            if not math.isfinite(z):
                raise ValueError("non-finite lift target")
            self.lift_goal = float(np.clip(z, RAIL["lift_lo"], RAIL["lift_hi"]))

    def rail_x(self):
        return float(self.data.qpos[self.rail_qadr])

    def lift_z(self):
        return float(self.data.qpos[self.lift_qadr])

    def apply_detents(self):
        """The selector's detent spring (a torque towards the nearest of its three positions)."""
        d = self.data
        q = float(d.qpos[self.jq["HS-101"]])
        pos = sorted(self.dev["HS-101"]["positions"].values())
        nearest = min(pos, key=lambda p: abs(p - q))
        tau = float(np.clip(SELECTOR_DETENT["k_Nm_per_rad"] * (nearest - q), -SELECTOR_DETENT["max_Nm"], SELECTOR_DETENT["max_Nm"]))
        d.qfrc_applied[self.jv["HS-101"]] = tau

    def step_physics(self):
        m, d = self.model, self.data
        t0 = float(d.time)
        for _ in range(self.substeps):
            mujoco.mj_step(m, d)
        if d.time < t0 or int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0:
            self.physics_fault = True

    # ------------------------------------------------------------------ device readings
    def opening(self, tag):
        """A valve's opening as the fraction of its travel (0 closed .. 1 fully open); V-111 and V-303: the lever angle
        in radians (0 = A .. pi = B)."""
        q = float(self.data.qpos[self.jq[tag]])
        lo, hi = self.jrange[tag]
        if tag in ANGLE_VALVES:
            return float(np.clip(q, lo, hi))
        return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))

    def openings(self):
        return {t: self.opening(t) for t in OPERABLE if t != "HS-101"}

    def selector(self, last="OFF"):
        q = float(self.data.qpos[self.jq["HS-101"]])
        for name, p in self.dev["HS-101"]["positions"].items():
            if abs(q - p) < SELECTOR_DETENT["snap_rad"]:
                return name
        return last

    def button_depth(self, tag):
        return float(self.data.qpos[self.btn_q[tag]])

    # ------------------------------------------------------------------ the grip rule
    def handles_world(self, tag):
        """The handles of a grippable device in the world frame: (name, pos, pinch axis or None, approach, width)."""
        d = self.data
        bid = self.moving_body[tag]
        R = d.xmat[bid].reshape(3, 3)
        p0 = d.xpos[bid]
        out = []
        for h in self.dev[tag]["handles"]:
            pinch = None if h["pinch"] is None else R @ np.asarray(h["pinch"], float)
            out.append((h["name"], p0 + R @ np.asarray(h["pos"], float), pinch, R @ np.asarray(h["approach"], float), float(h["width"])))
        return out

    def find_handle(self, opening):
        """The nearest handle inside the gripper's pinch box, its approach and closing axes within GRIP['angle_deg'] of the
        handle's, and narrower than the opening; None when there is none."""
        d = self.data
        p = d.site_xpos[self.tcp]
        R = d.site_xmat[self.tcp].reshape(3, 3)
        ca = math.cos(math.radians(GRIP["angle_deg"]))
        best = None
        for tag in OPERABLE:
            for name, hp, pinch, appr, width in self.handles_world(tag):
                v = R.T @ (hp - p)
                if abs(v[0]) > GRIP["box_x"] or abs(v[1]) > 0.5 * opening or not (GRIP["box_z_lo"] <= v[2] <= GRIP["box_z_hi"]):
                    continue
                if width > opening + 1e-9:
                    continue
                if float(R[:, 2] @ appr) < ca:
                    continue
                if pinch is not None and abs(float(R[:, 1] @ pinch)) < ca:
                    continue
                dist = float(np.linalg.norm(v))
                if best is None or dist < best[0]:
                    best = (dist, tag, name, width)
        if best is None:
            return None
        return {"tag": best[1], "handle": best[2], "width": best[3]}

    def attach(self, tag):
        grasp_mod.attach(self.model, self.data, "grip", self.dev[tag]["body"], bits=BITS_HELD)
        self.held = tag

    def release(self):
        grasp_mod.release(self.model, self.data, "grip")
        self.held = None

    def grip_load(self):
        """The grip weld's constraint load (force N, rotational rows): the slip rule's input. The rotational rows carry
        twice the moment the grip transmits (WELD_ROT_ROWS); the reference program paces its turns on these norms."""
        return grasp_mod.weld_load(self.model, self.data, "grip")

    def external_wrench(self):
        """The external wrench on the robot at the tcp, world axes: (force N (3), moment N m about the tcp (3)), as a
        Franka's joint-torque estimate gives it (O_F_ext_hat_K): the generalized forces of the robot's contacts with the
        skid and the floor and of the grip on the arm's seven joints (tau_ext = J_c^T f), mapped to the tcp through the
        tcp's Jacobian (F = (J J^T)^-1 J tau_ext). Read after a step (the constraint forces of its last substep); the
        server adds the sensor's noise. Read-only: nothing of the simulation changes."""
        m, d = self.model, self.data
        nefc = int(d.nefc)
        if nefc == 0:
            return np.zeros(3), np.zeros(3)
        typ = np.asarray(d.efc_type[:nefc])
        eid = np.asarray(d.efc_id[:nefc])
        sel = (typ == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (eid == self.grip_eq)
        con = (typ == mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS) | (typ == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL) | \
              (typ == mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
        if np.any(con) and d.ncon > 0:
            g = np.asarray(d.contact.geom[:d.ncon])
            ext = self.robot_geom[g[:, 0]] != self.robot_geom[g[:, 1]]
            sel |= con & ext[np.clip(eid, 0, d.ncon - 1)]
        if not np.any(sel):
            return np.zeros(3), np.zeros(3)
        J = np.asarray(d.efc_J[:nefc * m.nv]).reshape(nefc, m.nv)
        tau = J[sel][:, self.arm_dofs].T @ np.asarray(d.efc_force[:nefc])[sel]
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.tcp)
        Jt = np.vstack([self._jp[:, self.arm_dofs], self._jr[:, self.arm_dofs]])
        w = np.linalg.solve(Jt @ Jt.T + 1e-9 * np.eye(6), Jt @ tau)
        return w[:3], w[3:]

    # ------------------------------------------------------------------ visuals
    def update_visuals(self, proc, t):
        """Set the instruments, lamps, sight glasses and the rising stems from the process (before a frame is rendered)."""
        m, d = self.model, self.data
        vals = proc.values()
        kinds = self.L["gauge_kinds"]
        fl = proc.flutter()
        for tag in ("PI-101", "PI-103", "PDI-111"):
            k = tx.GAUGE_KINDS[kinds[tag]]
            v = vals[tag] / k["scale"]
            if tag != "PDI-111" and fl > 0.0:
                span = k["hi"] - k["lo"]
                v += fl * span * (0.035 * math.sin(2 * math.pi * 7.3 * t) + 0.02 * math.sin(2 * math.pi * 3.1 * t + 1.0))
            d.qpos[self.vis_q[tag]] = tx.dial_angle(v, k["lo"], k["hi"])
        k = tx.GAUGE_KINDS[kinds["FI-105"]]
        dev = self.dev["FI-105"]
        u = (vals["FI-105"] / k["scale"] - k["lo"]) / (k["hi"] - k["lo"])
        if proc.airbound > 0.0 and proc.run101:
            u += 0.04 * math.sin(2 * math.pi * 1.7 * t)
        z_top = dev["z_lo"] + float(np.clip(u, -0.02, 1.03)) * (dev["z_hi"] - dev["z_lo"])
        d.qpos[self.vis_q["FI-105"]] = float(np.clip(z_top - dev["z_bottom"], 0.0, dev["z_hi"] - dev["z_bottom"] + 0.02))
        for tag in self.handwheels:
            d.qpos[self.vis_q[tag + "/rise"]] = self.opening(tag) * self.dev[tag]["rise_max"]
        for tag, frac in proc.drum_levels().items():
            g = self.drum_liquid[tag]
            dd = self.dev[tag]
            h = max(0.004, float(np.clip(frac, 0.0, 1.0)) * (dd["height"] - 0.03))
            m.geom_size[g][1] = h / 2.0
            m.geom_pos[g][2] = dd["z0"] + 0.01 + h / 2.0
        for tag in ("LG-101", "LG-201"):
            g = self.flaps[tag]
            lev = float(np.clip(vals[tag] / 100.0, 0.0, 1.02))
            h = max(0.004, lev * self.dev[tag]["height"])
            m.geom_size[g][2] = h / 2.0
            m.geom_pos[g][2] = self.dev[tag]["z0"] + h / 2.0
        for tag, on in proc.lamps().items():
            m.geom_rgba[self.lens[tag]] = self.dev[tag]["rgba_on"] if on else self.dev[tag]["rgba_off"]
        for tag, liquid in proc.sights().items():
            m.geom_rgba[self.sight[tag]] = self.dev[tag]["rgba_liquid"] if liquid else self.dev[tag]["rgba_air"]
        self.set_display(vals["AI-201"])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    def set_display(self, value):
        m = self.model
        ds = self.dev["AI-201"]
        txt = "%4.2f" % float(np.clip(value, 0.0, 9.99))
        digits = txt.replace(".", "")[:ds["n_digits"]]
        for k, ch in enumerate(digits):
            lit = dv.DIGIT_SEGMENTS.get(ch, "")
            for sname, gid in self.segs[k].items():
                m.geom_rgba[gid] = ds["rgba_on"] if sname in lit else ds["rgba_off"]
        m.geom_rgba[self.dp_geom] = ds["rgba_on"]

    # ------------------------------------------------------------------ truth (development)
    def device_truth(self):
        """Every device's pose, joint and handles (the practice cell's development call and the reference program's eyes)."""
        d = self.data
        out = {}
        ST = self.L["site"]["tags"]
        for tag, ds in self.dev.items():
            e = {"type": ds["type"], "site_tag": ST.get(tag)}
            for key in ("centre", "tag_centre", "kind", "turns", "range", "spokes", "wheel_d", "lever_len", "closed_dir"):
                if key in ds:
                    e[key] = ds[key]
            if tag in OPERABLE:
                e["q"] = float(d.qpos[self.jq[tag]])
                e["opening"] = self.opening(tag) if tag != "HS-101" else None
                e["handles"] = [{"name": n, "pos": p.tolist(), "pinch": None if pc is None else pc.tolist(), "approach": a.tolist(), "width": wd}
                                for n, p, pc, a, wd in self.handles_world(tag)]
                bid = self.moving_body[tag]
                e["body_pos"] = d.xpos[bid].tolist()
                e["body_quat"] = d.xquat[bid].tolist()
            if tag in BUTTONS:
                e["depth"] = self.button_depth(tag)
            out[tag] = e
        return out
