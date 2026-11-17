"""Lab sample handling: the simulation of the bench (the environment's world, nothing of the reference).

The world: box transport's table, Panda and operational-space controller
(`box_transport_env.py`); a centrifuge with a fixed-angle (45 deg) rotor of eight holes on a damped hinge, a
knob bar on its hub (on its own hinge when the handle has play), a hinged lid; the incoming rack (one row, A1-A10),
the cold rack (3 x 3, A1-C3) and the rack camera; capped 15 mL tubes from primitives with a visual liquid column.
`build_bench` builds one episode's bench, `world` its `World` (the controller's step, the joint step, the
sensors); `draw_unit` draws an episode from a seed, `draw_episode` the same with a unit's hidden properties.

The reference's modules (`tube_insert_probe.py`, `centrifuge_probe.py`, `localize_probe.py`, `worklist_scene.py`,
`worklist_probe.py`) import it: one copy of the world.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "box_env"))
import box_transport_env as B
import mujoco
import numpy as np


def rotvec(R):
    """Rotation vector of a rotation matrix."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th > math.pi - 1e-6:                     # near pi: from the symmetric part
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, np.argmax(w)] * th
    return v / (2 * math.sin(th)) * th


# the tubes, the racks' rings, the arm (tube_insert_probe.py)
TABLE_Z = B.TABLE_Z
WALL = 0.010                 # m, hole wall thickness (its top face catches a tube that misses)
NSEG = 16                    # box segments per hole ring
SOURCE_CLEARANCE = 0.002     # m, radial


def rot_y(b):
    c, s = math.cos(b), math.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def base_world():
    """MjSpec with box transport's physics options (the stiff contact model), floor, table and Panda."""
    w = mujoco.MjSpec()
    w.compiler.degree = False
    w.compiler.autolimits = True
    w.option.timestep = B.TIMESTEP
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = B.IMPRATIO
    w.option.noslip_iterations = B.STIFF_NOSLIP_ITERATIONS
    w.memory = B.STIFF_ARENA_BYTES
    wb = w.worldbody
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [3, 3, 0.1]
    g = wb.add_geom()
    g.name = "table"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [B.TABLE_HALF[0], B.TABLE_HALF[1], TABLE_Z / 2]
    g.pos = [B.TABLE_CENTRE_X, 0.0, TABLE_Z / 2]
    ra = B.robosuite_assets()
    robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml"))
    for cam in list(robot.cameras):
        if cam.name == "robotview":
            robot.delete(cam)
    robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
    wb.add_frame(pos=[0.0, 0.0, TABLE_Z]).attach_body(robot.body("base"), "r_", "")
    return w, wb


def add_ring(parent, name, top, R, radius, depth, rgba=(0.75, 0.78, 0.85, 1)):
    """A hole of `radius` and `depth` on `parent` (a body; positions in its frame): NSEG wall segments
    around the axis R[:, 2] (pointing out of the hole) from the entrance centre `top`, and a floor."""
    half_t = (radius + WALL) * math.tan(math.pi / NSEG) + 0.0005
    for k in range(NSEG):
        th = 2 * math.pi * k / NSEG
        gg = parent.add_geom(name=f"{name}_w{k}")
        gg.type = mujoco.mjtGeom.mjGEOM_BOX
        gg.size = [WALL / 2, half_t, depth / 2]
        rr = radius + WALL / 2
        local = np.array([rr * math.cos(th), rr * math.sin(th), -depth / 2])
        gg.pos = list(top + R @ local)
        gg.quat = list(B.quat_from_mat(R @ B.quat_to_mat(B.yaw_quat(th))))
        gg.rgba = list(rgba)
    fl = parent.add_geom(name=f"{name}_floor")
    fl.type = mujoco.mjtGeom.mjGEOM_BOX
    fl.size = [radius + WALL, radius + WALL, 0.002]
    fl.pos = list(top + R @ np.array([0, 0, -depth - 0.002]))
    fl.quat = list(B.quat_from_mat(R))
    return {"top": np.asarray(top, float), "axis": R[:, 2], "R": R, "radius": radius, "depth": depth}


def add_tube(wb, name, pos, a, quat=(1, 0, 0, 0)):
    """A capped conical tube from primitives: local z along its axis, origin at the tip."""
    r_t = a.tube_d / 2
    L_body = a.tube_len - 0.012
    tb = wb.add_body(name=name)
    tb.pos = list(pos)
    tb.quat = list(quat)
    tb.add_freejoint().name = name + "_free"
    cap = tb.add_geom(name=name + "_body")
    cap.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    cap.size = [r_t, (L_body - 2 * r_t) / 2, 0]
    cap.pos = [0, 0, L_body / 2]
    c2 = tb.add_geom(name=name + "_cap")
    c2.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    c2.size = [r_t + 0.001, 0.006, 0]
    c2.pos = [0, 0, L_body + 0.006]
    for geom in (cap, c2):
        geom.condim = 4
        geom.priority = 1
        geom.friction = [a.friction, B.STIFF_TORSION, 0.0001]
        geom.rgba = [0.9, 0.9, 0.95, 1] if geom is cap else [0.2, 0.4, 0.9, 1]
    return tb


def set_tube_mass(m, name, a, mass=None, com_z=None):
    r_t = a.tube_d / 2
    mass = a.mass if mass is None else mass
    b = m.body(name).id
    m.body_mass[b] = mass
    m.body_inertia[b] = [mass * (3 * r_t ** 2 + a.tube_len ** 2) / 12] * 2 + [mass * r_t ** 2 / 2]
    m.body_iquat[b] = [1, 0, 0, 0]
    m.body_ipos[b] = [0, 0, a.com_z if com_z is None else com_z]


def stiff_pads(m):
    for name in ("r_g_finger1_pad_collision", "r_g_finger2_pad_collision"):
        gid = m.geom(name).id
        m.geom_condim[gid] = 4
        m.geom_friction[gid, 1] = B.STIFF_TORSION


class Probe:
    """Box transport's controller (operational space, 10 Hz) on this world."""

    _osc_torque = B.BoxTransportEnv._osc_torque

    def __init__(self, m):
        self.model = m
        self.data = d = mujoco.MjData(m)
        self.jq = np.array([m.joint(f"r_joint{i}").qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint(f"r_joint{i}").dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator(f"r_torq_j{i}").id for i in range(1, 8)])
        self.gact = [m.actuator("r_g_gripper_finger_joint1").id, m.actuator("r_g_gripper_finger_joint2").id]
        self.gq = [m.joint("r_g_finger_joint1").qposadr[0], m.joint("r_g_finger_joint2").qposadr[0]]
        self.site = m.site("r_g_grip_site").id
        self.ft_site = m.site("r_g_ft_frame").id
        self.ft_force = m.sensor("r_g_force_ee").adr[0]
        self.pads = (m.geom("r_g_finger1_pad_collision").id, m.geom("r_g_finger2_pad_collision").id)
        self.tube = m.body("tube").id
        self.tube_dof = m.joint("tube_free").dofadr[0]
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        mujoco.mj_resetData(m, d)
        d.qpos[self.jq] = B.HOME_Q
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        for _ in range(250):
            d.ctrl[self.act] = 200.0 * (B.HOME_Q - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
        self.q_null = B.HOME_Q.copy()
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()
        R = d.site_xmat[self.site].reshape(3, 3)
        f = R.T @ (d.geom_xpos[self.pads[0]] - d.geom_xpos[self.pads[1]])
        f[2] = 0.0
        self.f_local = f / np.linalg.norm(f)
        self.t = 0
        self.force_ref = None
        self.max_force = 0.0
        self.track_force = False

    def ee(self):
        return self.data.site_xpos[self.site].copy()

    def ee_R(self):
        return self.data.site_xmat[self.site].reshape(3, 3).copy()

    def tube_pos(self):
        return self.data.xpos[self.tube].copy()

    def tube_R(self):
        return self.data.xmat[self.tube].reshape(3, 3).copy()

    def wrench_force(self):
        d = self.data
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return R @ d.sensordata[self.ft_force:self.ft_force + 3]

    def step(self, dp, dr, grip):
        d = self.data
        dp = np.asarray(dp, float)
        n = np.linalg.norm(dp)
        if n > B.MAX_DPOS:
            dp = dp * (B.MAX_DPOS / n)
        dr = np.asarray(dr, float)
        n = np.linalg.norm(dr)
        if n > B.MAX_DROT:
            dr = dr * (B.MAX_DROT / n)
        self.target_pos = d.site_xpos[self.site] + dp
        self.target_R = B.axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
        gt = 0.04 * (1.0 - float(np.clip(grip, -1, 1))) / 2.0
        for _ in range(B.SUBSTEPS):
            d.ctrl[self.act] = self._osc_torque()
            d.ctrl[self.gact] = [gt, -gt]
            mujoco.mj_step(self.model, d)
        self.t += 1
        if not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qvel[self.jv])) > B.MAX_ARM_SPEED:
            raise FloatingPointError("unstable")
        if self.track_force:
            f = float(np.linalg.norm(self.wrench_force() - self.force_ref))
            self.max_force = max(self.max_force, f)

    def grasped(self):
        m, d = self.model, self.data
        touching = set()
        for c in d.contact[:d.ncon]:
            for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if m.geom_bodyid[g2] == self.tube:
                    touching.add(int(g1))
        return all(p in touching for p in self.pads)

    def down(self, yaw):
        z = np.array([0.0, 0.0, -1.0])
        fw = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        fl = self.f_local
        A = np.column_stack([fl, np.cross([0.0, 0.0, 1.0], fl), [0.0, 0.0, 1.0]])
        Bm = np.column_stack([fw, np.cross(z, fw), z])
        return Bm @ A.T

    def servo(self, target, R_goal, grip, speed, tol=0.002, max_steps=300):
        self.integral = np.zeros(3)
        for _ in range(max_steps):
            err = np.asarray(target, float) - self.ee()
            if grip > 0:                         # holding: integrate away the steady sag (box transport's rule)
                self.integral = np.clip(self.integral + 0.15 * err, -0.02, 0.02)
            dp = err + (self.integral if grip > 0 else 0.0)
            n = np.linalg.norm(dp)
            if n > speed:
                dp *= speed / n
            dr = rotvec(R_goal @ self.ee_R().T)
            n = np.linalg.norm(dr)
            if n > 0.15:
                dr *= 0.15 / n
            self.step(dp, dr, grip)
            if np.linalg.norm(err) < tol and np.linalg.norm(dr) < 0.02:
                return True
        return False


# the centrifuge (centrifuge_probe.py)
HERE = os.path.dirname(os.path.abspath(__file__))
CENTRE = np.array([0.45, -0.10])      # rotor axis on the table (robot frame)
ROTOR_TOP = TABLE_Z + 0.11            # z of the rotor's top face (the hole entrances)
HOLE_RADIUS_AT = 0.07                 # m, hole entrances from the rotor axis
TILT = math.radians(45)
HOLE_DEPTH = 0.10
NHOLES = 8
CHAMBER = 0.18                        # m, half size of the chamber (square)
LID_GAP = 0.03                        # m, the closed lid's underside above the rotor top
LID_LEN = 2 * CHAMBER
HANDLE = np.array([0.0, LID_LEN + 0.045, 0.006])   # handle bar centre in the lid frame, beyond the free edge
LID_OPEN = 1.75                       # rad, the open lid's stop (100 deg: past vertical, away from the arm). A stop at
                                      # 80 deg (leaning over the chamber) put the lid in the way of the loads; the
                                      # handle's reach is kept by `bench_ok` instead
SRC_XY = [np.array([0.36, 0.18]), np.array([0.36, 0.28]),   # 0.10 apart: open fingers (along y) clear the neighbour
          np.array([0.46, 0.18]), np.array([0.46, 0.28]),
          np.array([0.26, 0.18]), np.array([0.26, 0.28]), np.array([0.36, 0.38]), np.array([0.46, 0.38])]   # 8: a full rotor
SRC_DEPTH = 0.05
ROTOR_BOWL_R = 0.035                  # m, the bowl's inner edge (the hub)
ROTOR_RIM_R = 0.095                   # m, the bowl's rim (its top 25 mm above the entrances: under the closed lid)
ROTOR_BASE_R = 0.178                  # m, the side's bottom edge (it covers the bores' lower ends)
ROTOR_BOTTOM = -0.088                 # m, the rotor's underside (rotor frame)
CONE_LIFT = 0.0005                    # m, the bowl above the hole entrances (it hides the rings' tops)
HUB_TOP = -0.010                      # m, the solid rotor's hub top (rotor frame): the knob sits low, under the
                                      # fingers of a hand pushing a tube into the hole across
KNOB = np.array([0.035, 0.006, 0.009])  # half sizes of the knob bar (along the rotor's x), on the hub's top
DTH_MESH = math.radians(0.5)          # the rotor mesh's angular step
MAX_DQ = 0.1                          # rad, a joint step's largest target change from the current angle
HANDLE_LIMIT_TC = 0.004               # s, the handle play's stop: the constraint's time constant
KNOB_YAW = math.pi / NHOLES           # the bar lies between two holes' planes: a tube going in passes over the hub
KNOB_Z = HUB_TOP + KNOB[2]            # its centre, rotor frame
ROTOR_RGBA = (0.76, 0.77, 0.79, 1)
MARK_R = 0.088                        # m, the position-1 mark (hole 0) on the bowl, from the axis
MARK_RADIUS = 0.006                   # m
MARK_RGBA = (0.15, 0.65, 0.25, 1)     # green, a colour nothing else in the scene has
BORE_RGBA = (0.10, 0.10, 0.11, 1)
# a shield over the hole ring with one loading port, and a one-way rotor (a ratchet)
PORT_AZ = math.radians(160.0)         # the port's azimuth about the rotor's axis, cabinet frame (reachable from the
                                      # scale's holder on every bench)
SHIELD_Z = 0.023                      # m, the shield's underside above the rotor's top (a seated tube's cap clears it
                                      # by 2 mm)
SHIELD_T = 0.002                      # m, its thickness
SHIELD_RIN = 0.048                    # m, its inner edge from the axis (the hub and the knob stay open: a hand
                                      # grips the knob's bar up to about 18 mm from the axis)
SHIELD_RIM = 0.006                    # m, its outer edge inside the hole circle (the entrances' outer halves show)
PORT_W = 0.024                        # m, the port's width (a 19 mm cap passes; a hole within about 3 deg of it)
SHIELD_RGBA = (0.85, 0.78, 0.15, 1)    # yellow (a guard's colour; no other part of the scene is yellow)
RATCHET_TOOTH = math.radians(2.0)     # the pawl's tooth: the rotor turns counter-clockwise only


def rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def centre_xy(a):
    """The rotor's axis on the table this episode (the bench draw; the nominal centre if none)."""
    return np.asarray(getattr(a, "centre", CENTRE), float)


def rotor_top(a):
    """z of the rotor's top face (the hole entrances) this episode."""
    return float(getattr(a, "rotor_top", ROTOR_TOP))


def hole_r(a):
    """The hole entrances' distance from the rotor's axis this episode."""
    return float(getattr(a, "hole_r", HOLE_RADIUS_AT))


def cab_yaw(a):
    """The centrifuge cabinet's yaw (rad) this episode: the chamber, the lid and its hinge turn with it."""
    return float(getattr(a, "cab_yaw", 0.0))


def port_az(a):
    """The loading port's world azimuth about the rotor's axis this episode (it turns with the cabinet)."""
    return cab_yaw(a) + PORT_AZ


def add_shield(wb, a):
    """The shield: a 2 mm plate SHIELD_Z above the rotor's top over the hole ring, from SHIELD_RIN to SHIELD_RIM
    inside the hole circle, cut through by one port PORT_W wide on azimuth `port_az`. Every tube reaches its hole
    through the port, so a hole is loaded only when the rotor brings it there. Boxes: 15 deg segments, the port's
    two sides parallel to its centre line."""
    cxy, top_z, r_out = centre_xy(a), rotor_top(a), hole_r(a) - SHIELD_RIM
    z = top_z + SHIELD_Z + SHIELD_T / 2
    az = port_az(a)
    seg = math.radians(15.0)
    half_t = r_out * math.tan(seg / 2) + 0.001
    boxes = []
    for i in range(24):
        phi = (i + 0.5) * seg                         # segment centres at 7.5 + 15 i deg from the port
        if abs((phi + math.pi) % (2 * math.pi) - math.pi) < math.radians(25.0):
            continue                                  # the four segments nearest the port: the sides below
        boxes.append(((SHIELD_RIN + r_out) / 2, 0.0, (r_out - SHIELD_RIN) / 2, half_t, phi))
    y1 = r_out * math.sin(math.radians(30.0)) + 0.002
    for side in (1.0, -1.0):                          # the port's sides, from its edge out to the 30 deg segments
        boxes.append(((SHIELD_RIN + r_out) / 2, side * (PORT_W / 2 + y1) / 2, (r_out - SHIELD_RIN) / 2,
                      (y1 - PORT_W / 2) / 2, 0.0))
    for i, (x, y, sx, sy, phi) in enumerate(boxes):
        R = rot_z(az + phi)
        c = R[:2, :2] @ np.array([x, y])
        g = wb.add_geom(name=f"shield{i}")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [sx, sy, SHIELD_T / 2]
        g.pos = [cxy[0] + c[0], cxy[1] + c[1], z]
        g.quat = list(B.yaw_quat(az + phi))
        g.rgba = list(SHIELD_RGBA)


def hole_local(k, r=HOLE_RADIUS_AT):
    """Entrance (rotor frame) and rotation of rotor hole k on a hole circle of radius r: axis leaning toward the
    rotor's axis."""
    th = 2 * math.pi * k / NHOLES
    R = rot_z(th) @ rot_y(-TILT)
    top = np.array([r * math.cos(th), r * math.sin(th), 0.0])
    return top, R


def rotor_mesh(bore_r, dr=0.001, dth=DTH_MESH, r_holes=HOLE_RADIUS_AT):
    """Vertices and faces of the solid rotor's visible surface (rotor frame): a 45 deg bowl through the hole
    entrances (each entrance plane is tangent to it) with a round cut-out at every hole, and a side widening
    downward that covers the bores. Visual only: MuJoCo collides a mesh as its convex hull, so the holes'
    collision stays the rings."""
    def z_bowl(r):
        return r - r_holes + CONE_LIFT

    holes = [hole_local(k, r_holes) for k in range(NHOLES)]
    nth = round(2 * math.pi / dth)
    th = np.arange(nth) * dth
    rs = np.arange(ROTOR_BOWL_R, ROTOR_RIM_R + 1e-9, dr)
    verts, faces = [], []

    def ring(r, z):
        base = len(verts)
        verts.extend(np.column_stack([r * np.cos(th), r * np.sin(th), np.full(nth, z)]))
        return base

    rows = [ring(r, z_bowl(r)) for r in rs]
    for i in range(len(rs) - 1):
        for j in range(nth):
            jn = (j + 1) % nth
            a, b, c, d = rows[i] + j, rows[i + 1] + j, rows[i + 1] + jn, rows[i] + jn
            ctr = (verts[a] + verts[c]) / 2
            cut = False
            for top, R in holes:
                v = ctr - top
                t = float(np.dot(v, R[:, 2]))
                if abs(t) < 0.03 and np.linalg.norm(v - t * R[:, 2]) < bore_r:
                    cut = True
                    break
            if not cut:
                faces += [(a, b, c), (a, c, d)]
    bot = ring(ROTOR_BASE_R, ROTOR_BOTTOM)
    top_row = rows[-1]
    for j in range(nth):
        jn = (j + 1) % nth
        faces += [(top_row + j, bot + j, bot + jn), (top_row + j, bot + jn, top_row + jn)]
    return np.array(verts), np.array(faces, dtype=np.int32)


def add_lid(wb, wall_top, a):
    """The lid: hinged at one edge of the chamber (`--lid-side`: back = +x, right = -y), the plate over the
    chamber, a handle bar across the free edge. Lid frame: x along the hinge, the plate toward +y, so a positive
    angle (about +x) lifts the free edge."""
    hinge_dir = {"back": (np.array([CHAMBER, 0.0]), math.pi / 2), "right": (np.array([0.0, -CHAMBER]), 0.0)}
    off_xy, lid_yaw = hinge_dir[a.lid_side]
    yaw = cab_yaw(a)
    off_xy = rot_z(yaw)[:2, :2] @ off_xy
    lid_yaw += yaw
    c = centre_xy(a)
    hinge = np.array([c[0] + off_xy[0], c[1] + off_xy[1], wall_top])
    lb = wb.add_body(name="lid", pos=list(hinge), quat=list(B.yaw_quat(lid_yaw)))
    j = lb.add_joint(name="lid_hinge")
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = [1, 0, 0]                      # positive angle lifts the free edge
    j.range = [0.0, a.lid_open]
    j.damping = [a.lid_damping, 0, 0]
    j.frictionloss = a.lid_friction
    plate = lb.add_geom(name="lid_plate")
    plate.type = mujoco.mjtGeom.mjGEOM_BOX
    plate.size = [CHAMBER, CHAMBER, 0.006]
    plate.pos = [0, CHAMBER, 0.006]
    plate.rgba = [0.3, 0.55, 0.8, 0.6]
    bar = lb.add_geom(name="lid_handle")
    bar.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bar.size = [0.0075, 0.05, 0]
    bar.pos = list(HANDLE)
    bar.quat = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]     # axis along the lid frame's x
    bar.rgba = [0.2, 0.2, 0.2, 1]
    for s in (-1, 1):                       # arms from the plate's free edge out to the bar
        post = lb.add_geom(name=f"lid_post{s}")
        post.type = mujoco.mjtGeom.mjGEOM_BOX
        post.size = [0.006, (HANDLE[1] - LID_LEN) / 2, 0.004]
        post.pos = [s * 0.056, (HANDLE[1] + LID_LEN) / 2, HANDLE[2]]
        post.rgba = [0.2, 0.2, 0.2, 1]
    for geom in [bar, plate]:
        geom.condim = 4
        geom.priority = 1
        geom.friction = [0.8, B.STIFF_TORSION, 0.0001]


TUBE_OPAQUE_RGBA = (0.90, 0.89, 0.84, 1)


def build(a, fills=None, extra=None):
    w, wb = base_world()
    light = wb.add_light()                 # box transport's light (rendering only)
    light.pos = [0.4, 0.0, 2.4]
    light.dir = [0.1, 0.0, -1.0]
    light.diffuse = [0.8, 0.8, 0.8]
    light.castshadow = False
    src_depth = SRC_DEPTH if fills is None else 0.015     # render: a low collar, so the tubes show
    if getattr(a, "src_depth", None):
        src_depth = a.src_depth                           # a deep rack, its liquid shown
    r_t = a.tube_d / 2
    radius = r_t + a.clearance
    # cabinet and chamber (static), at this episode's centre and yaw
    cxy, top_z, r_holes, yaw = centre_xy(a), rotor_top(a), hole_r(a), cab_yaw(a)
    Rc = rot_z(yaw)[:2, :2]
    yq = list(B.yaw_quat(yaw))
    base_top = top_z - HOLE_DEPTH * math.cos(TILT) - 0.02
    g = wb.add_geom(name="cabinet")
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [CHAMBER, CHAMBER, (base_top - TABLE_Z) / 2]
    g.pos = [cxy[0], cxy[1], (base_top + TABLE_Z) / 2]
    g.quat = yq
    g.rgba = [0.85, 0.86, 0.88, 1]
    wall_top = top_z + LID_GAP
    for i, (dx, dy, sx, sy) in enumerate(((CHAMBER, 0, 0.005, CHAMBER), (-CHAMBER, 0, 0.005, CHAMBER),
                                          (0, CHAMBER, CHAMBER, 0.005), (0, -CHAMBER, CHAMBER, 0.005))):
        g = wb.add_geom(name=f"wall{i}")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [sx, sy, (wall_top - base_top) / 2]
        off = Rc @ np.array([dx, dy])
        g.pos = [cxy[0] + off[0], cxy[1] + off[1], (wall_top + base_top) / 2]
        g.quat = yq
        g.rgba = [0.85, 0.86, 0.88, 1]
    # rotor: a hub and eight tilted holes on a damped vertical hinge (its angle is the world angle of hole 0)
    rb = wb.add_body(name="rotor", pos=[cxy[0], cxy[1], top_z])
    j = rb.add_joint(name="rotor_hinge")
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = [0, 0, 1]
    j.damping = [a.rotor_damping, 0, 0]
    j.frictionloss = a.rotor_friction
    hub = rb.add_geom(name="hub")
    hub.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    hub_top = HUB_TOP if getattr(a, "rotor_model", "rings") == "solid" else 0.0
    hub.size = [0.035, (hub_top + 0.04) / 2, 0]
    hub.pos = [0, 0, (hub_top - 0.04) / 2]
    hub.rgba = [0.5, 0.5, 0.55, 1] if getattr(a, "rotor_model", "rings") != "solid" else list(ROTOR_RGBA)
    solid = getattr(a, "rotor_model", "rings") == "solid"
    for k in range(NHOLES):
        top, R = hole_local(k, r_holes)
        add_ring(rb, f"h{k}", top, R, radius, HOLE_DEPTH, rgba=BORE_RGBA if solid else (0.55, 0.57, 0.62, 1))
    if solid and not getattr(a, "no_shell", False):
        verts, faces = rotor_mesh(radius, r_holes=r_holes)
        mesh = w.add_mesh(name="rotor_shell")
        mesh.uservert = verts.flatten().tolist()
        mesh.userface = faces.flatten().tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        g = rb.add_geom(name="rotor_shell")
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "rotor_shell"
        g.contype = 0
        g.conaffinity = 0
        g.rgba = list(ROTOR_RGBA)
        # the position-1 mark: a green dot on the bowl just outside hole 0 (real rotors number their positions),
        # without which the eight-fold rotor's angle is known only to 45 deg
        mark = rb.add_geom(name="mark")
        mark.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        mark.size = [MARK_RADIUS, 0.0004, 0]
        mark_r = r_holes + MARK_R - HOLE_RADIUS_AT
        mark.pos = [mark_r - 0.0004 * math.sin(TILT), 0, MARK_R - HOLE_RADIUS_AT + CONE_LIFT + 0.0004 * math.cos(TILT)]
        mark.quat = [math.cos(-TILT / 2), 0, math.sin(-TILT / 2), 0]    # its axis along the bowl's normal
        mark.contype = 0
        mark.conaffinity = 0
        mark.rgba = list(MARK_RGBA)
    if solid and not getattr(a, "no_knob", False):
        kb = rb
        play = math.radians(getattr(a, "handle_play", 0.0))
        if play > 0:
            # the handle turns freely within +-play/2 before it drives the rotor (a loose coupling)
            kb = rb.add_body(name="handle")
            hj = kb.add_joint(name="handle_play")
            hj.type = mujoco.mjtJoint.mjJNT_HINGE
            hj.axis = [0, 0, 1]
            hj.range = [-play / 2, play / 2]
            hj.damping = [0.002, 0, 0]
            hj.frictionloss = 0.01            # it stays where it was left
            hj.solref_limit = [HANDLE_LIMIT_TC, 1.0]   # a hard stop (the default limit gives way under the turn)
            hj.armature = 1e-4
        knob = kb.add_geom(name="knob")
        knob.type = mujoco.mjtGeom.mjGEOM_BOX
        knob.size = list(KNOB)
        knob.pos = [0, 0, KNOB_Z]
        knob.quat = list(B.yaw_quat(KNOB_YAW))
        knob.rgba = [0.2, 0.2, 0.22, 1]
        knob.condim = 4
        knob.priority = 1
        knob.friction = [0.8, B.STIFF_TORSION, 0.0001]
    if getattr(a, "port", False):
        add_shield(wb, a)
    if not a.no_lid:
        add_lid(wb, wall_top, a)
    # source rack and tubes
    src = []
    src_xy = getattr(a, "src_xy", None) or SRC_XY
    src_rgba = getattr(a, "src_rgba", None) or (0.75, 0.78, 0.85, 1)
    for i, xy in enumerate(src_xy[:max(2, a.tubes)]):
        top = np.array([xy[0], xy[1], TABLE_Z + src_depth + 0.015])
        src.append(add_ring(wb, f"src{i}", top, np.eye(3), r_t + getattr(a, "src_clearance", SOURCE_CLEARANCE),
                               src_depth, rgba=src_rgba))
    occupied = list(getattr(a, "tube_slots", None) or range(a.tubes))
    # one tube per occupied slot (some slots are empty, their name None); the first is "tube" (the probes'
    # default body)
    names = [None] * max(a.tubes, len(src))
    for n, i in enumerate(occupied):
        names[i] = "tube" if n == 0 else f"tube{i}"
    names = names[:max(a.tubes, max(occupied) + 1)]
    for i, name in enumerate(names):
        if name is None:
            continue
        tb = add_tube(wb, name, src[i]["top"] - src[i]["axis"] * (src_depth - 0.0005), a)
        if fills is not None and getattr(a, "opaque", False):
            # opaque polypropylene tubes; the liquid does not show (its mass does: `set_tube_mass`)
            for gg in tb.geoms:
                if gg.name.endswith("_body"):
                    gg.rgba = list(TUBE_OPAQUE_RGBA)
        elif fills is not None:
            # render mode: a translucent tube with a visible liquid column (visual only)
            for gg in tb.geoms:
                if gg.name.endswith("_body"):
                    gg.rgba = [0.92, 0.94, 0.97, 0.35]
            liq = tb.add_geom(name=name + "_liquid")
            liq.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            h = fill_height(fills[i], a)
            liq.size = [a.tube_d / 2 - 0.0012, h / 2, 0]
            liq.pos = [0, 0, h / 2 + 0.002]
            liq.rgba = [0.85, 0.25, 0.2, 1]
            liq.contype = 0
            liq.conaffinity = 0
            liq.mass = 1e-6
    for name, (p, tgt, fovy, _w, _h) in B.CAMERAS.items():
        c = wb.add_camera()
        c.name = name
        c.pos = list(p)
        c.quat = list(B.quat_from_mat(B.look_at(p, tgt)))
        c.fovy = fovy
    w.visual.global_.offwidth = 640
    w.visual.global_.offheight = 480
    if extra is not None:                  # a caller's additions (localisation probe: an empty rack)
        extra(wb)
    if any(st.name == "scale_site" for st in w.sites):
        sen = w.add_sensor()
        sen.name = "scale_force"
        sen.type = mujoco.mjtSensor.mjSENS_FORCE
        sen.objtype = mujoco.mjtObj.mjOBJ_SITE
        sen.objname = "scale_site"
    for name, pos in getattr(a, "extra_tubes", {}).items():
        tb = add_tube(wb, name, pos, a)            # tubes already in the rotor, other people's in the cold rack
        for gg in tb.geoms:
            if gg.name.endswith("_body"):
                gg.rgba = list(TUBE_OPAQUE_RGBA)
    m = w.compile()
    for name in [n for n in names if n is not None] + list(getattr(a, "extra_tubes", {})):
        set_tube_mass(m, name, a)
    b = m.body("rotor").id
    m.body_mass[b] = 1.0
    m.body_inertia[b] = [0.006, 0.006, getattr(a, "rotor_inertia", 0.008)]
    m.body_ipos[b] = [0, 0, -0.03]
    m.body_iquat[b] = [1, 0, 0, 0]
    if not a.no_lid:
        set_lid_mass(m, a)
    if getattr(a, "rotor_model", "rings") == "solid":
        for k in range(NHOLES):
            m.geom_rgba[m.geom(f"h{k}_floor").id] = BORE_RGBA
    stiff_pads(m)
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m, src, names


def set_lid_mass(m, a):
    b = m.body("lid").id
    m.body_mass[b] = a.lid_mass
    m.body_inertia[b] = [a.lid_mass * LID_LEN ** 2 / 12] * 2 + [a.lid_mass * LID_LEN ** 2 / 6]
    m.body_iquat[b] = [1, 0, 0, 0]
    m.body_ipos[b] = [0, CHAMBER, 0.006]


def fill_height(ml, a):
    r_in = a.tube_d / 2 - 0.0012
    return ml * 1e-6 / (math.pi * r_in ** 2)


class World(Probe):
    """Probe on the centrifuge world; `use(name)` switches the tube it reads."""

    def __init__(self, m, a):
        super().__init__(m)
        self.a = a
        self.rotor_q = m.joint("rotor_hinge").qposadr[0]
        self.lid_q = None if a.no_lid else m.joint("lid_hinge").qposadr[0]
        if self.lid_q is not None:
            self.data.qpos[self.lid_q] = a.lid_start
        q0 = math.radians(getattr(a, "rotor_start", 0.0))
        if a.rotor_jitter:
            rng = np.random.default_rng(a.seed)
            q0 += math.radians(rng.uniform(-a.rotor_jitter, a.rotor_jitter))
        self.data.qpos[self.rotor_q] = q0
        mujoco.mj_forward(m, self.data)
        self.handle_q = None
        if getattr(a, "handle_play", 0.0) > 0 and not getattr(a, "no_knob", False):
            # where the handle sits in its play at the start: drawn per seed
            self.handle_q = m.joint("handle_play").qposadr[0]
            half = math.radians(a.handle_play) / 2
            self.data.qpos[self.handle_q] = np.random.default_rng([a.seed, 5]).uniform(-half, half)
            mujoco.mj_forward(m, self.data)
        self.ratchet = None
        if getattr(a, "ratchet", False):
            # the pawl: the hinge's lower limit follows the last tooth passed, so the rotor turns one way
            j = m.joint("rotor_hinge").id
            m.jnt_range[j] = [math.floor(q0 / RATCHET_TOOTH) * RATCHET_TOOTH, 1e3]
            m.jnt_limited[j] = 1
            self.ratchet = j
        self.home_pos, self.home_R = self.ee(), self.ee_R()
        self.scale_adr = None
        if any(m.sensor(i).name == "scale_force" for i in range(m.nsensor)):
            self.scale_adr = m.sensor("scale_force").adr[0]
            self.scale_tare = float(m.body_subtreemass[m.body("scale").id])
            self.scale_rng = np.random.default_rng([a.seed % 2**32, a.seed >> 32, 53])
            self.scale_f = self.scale_raw()

    def scale_raw(self):
        """The load on the scale's pan now (g, from its force sensor less its own weight)."""
        f = self.data.sensordata[self.scale_adr + 2]
        return (float(f) / -self.model.opt.gravity[2] - self.scale_tare) * 1000.0

    def scale_tick(self):
        """One control step of the display's lag (first order, SCALE_TAU)."""
        if self.scale_adr is not None:
            self.scale_f += (1.0 - math.exp(-0.1 / SCALE_TAU)) * (self.scale_raw() - self.scale_f)

    def scale_display(self):
        """What the scale's display shows now (g): the lagged load with noise, in SCALE_STEP steps."""
        if self.scale_adr is None:
            return None
        v = self.scale_f + self.scale_rng.normal(0.0, SCALE_NOISE)
        return round(round(v / SCALE_STEP) * SCALE_STEP, 1)

    def ratchet_tick(self):
        """The pawl drops into the next tooth: the hinge's lower limit rises to the last tooth the rotor passed."""
        if self.ratchet is not None:
            lo = math.floor(float(self.data.qpos[self.rotor_q]) / RATCHET_TOOTH) * RATCHET_TOOTH
            rng = self.model.jnt_range[self.ratchet]
            rng[0] = max(rng[0], lo)

    def step(self, dp, dr, grip):
        super().step(dp, dr, grip)
        self.scale_tick()
        self.ratchet_tick()

    def joint_step(self, q_target, grip):
        """One control step in joint space: a gravity-compensated joint PD toward q_target, each joint's target
        at most MAX_DQ from its current angle (the environment's joint action); the gripper
        as in `step`. The target also becomes the null-space posture of later Cartesian steps."""
        m, d = self.model, self.data
        q = d.qpos[self.jq].copy()
        qd = q + np.clip(np.asarray(q_target, float) - q, -MAX_DQ, MAX_DQ)
        self.q_null = np.asarray(q_target, float).copy()
        gt = 0.04 * (1.0 - float(np.clip(grip, -1, 1))) / 2.0
        for _ in range(B.SUBSTEPS):
            d.ctrl[self.act] = 200.0 * (qd - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [gt, -gt]
            mujoco.mj_step(m, d)
        self.t += 1
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError("unstable")
        self.scale_tick()
        self.ratchet_tick()

    def home_joints(self, steps=15, max_steps=40):
        """Back to the start posture in joint space (`joint_step`, the gripper open): recovers from a posture the
        operational-space controller cannot leave (a joint at its limit). At least `steps` steps, until every
        joint is within 0.01 rad (MAX_DQ per step), at most `max_steps`."""
        for i in range(max_steps):
            self.joint_step(B.HOME_Q, -1)
            if i + 1 >= steps and np.max(np.abs(self.data.qpos[self.jq] - B.HOME_Q)) < 0.01:
                break
        self.servo(self.home_pos, self.home_R, -1, 0.03, tol=0.01, max_steps=50)

    def home(self):
        """Back to the start pose, gripper open (the controller's null space pulls the joints home too)."""
        self.servo(self.ee() + [0, 0, 0.05], self.ee_R(), -1, 0.02, tol=0.005, max_steps=40)
        return self.servo(self.home_pos, self.home_R, -1, 0.03, tol=0.01, max_steps=200)

    def use(self, name):
        self.tube = self.model.body(name).id

    def hole(self, k):
        """Rotor hole k in the world frame, now."""
        d = self.data
        b = self.model.body("rotor").id
        Rr, pr = d.xmat[b].reshape(3, 3), d.xpos[b]
        top, R = hole_local(k, hole_r(self.a))
        RR = Rr @ R
        return {"top": pr + Rr @ top, "axis": RR[:, 2], "R": RR, "radius": self.a.tube_d / 2 + self.a.clearance,
                "depth": HOLE_DEPTH}

    def lid_angle(self):
        return float(self.data.qpos[self.lid_q])

    def handle_world(self):
        b = self.model.body("lid").id
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ HANDLE


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def target_hole(p, k):
    """Hole k: a rotor hole by its index (its pose from the simulation, followed as the rotor moves), or a hole
    given as a pose dict (an estimate, or another rack's hole)."""
    return k if isinstance(k, dict) else p.hole(k)


def seated(p, name, k):
    """Tube `name` at the floor of hole k (a rotor hole's index, or a hole's true pose dict), from the simulation."""
    p.use(name)
    h = target_hole(p, k)
    rel = p.tube_pos() - h["top"]
    depth_in = float(-np.dot(rel, h["axis"]))
    return bool(depth_in > h.get("depth", HOLE_DEPTH) - 0.006
                and np.linalg.norm(rel + depth_in * h["axis"]) < h["radius"])


# cameras (localize_probe.py)
RING_RGBA = (0.95, 0.55, 0.10, 1)


def intrinsics(fovy, w, h):
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def cam_pose(d, cid):
    """World pose of a MuJoCo camera in the OpenCV convention (x right, y down, z forward)."""
    R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return R, d.cam_xpos[cid].copy()


def project(K, R, p, X):
    v = R.T @ (np.asarray(X) - p)
    return np.array([K[0, 0] * v[0] / v[2] + K[0, 2], K[1, 1] * v[1] / v[2] + K[1, 2]])


class RenderError(RuntimeError):
    """The renderer failed (no OpenGL context, a broken driver, an error while drawing): the grader's own error,
    never the controller's."""


def render(m, d, cam, w, h, rng):
    try:
        r = mujoco.Renderer(m, h, w)
        try:
            r.update_scene(d, camera=cam)
            img = r.render().astype(np.float32)
        finally:
            r.close()
    except Exception as exc:  # noqa: BLE001 - whatever the renderer raises is reported as one error
        raise RenderError("{} camera ({}x{}): {}: {}".format(cam, w, h, type(exc).__name__, str(exc)[:200])) from exc
    img += rng.normal(0, B.RGB_NOISE, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# the bench (worklist_scene.py)
N_SLOTS = 10
RACK_X, RACK_Y0, RACK_PITCH, RACK_SHIFT = 0.36, 0.16, 0.045, 0.01     # A1 at y 0.12 put the hand on the chamber wall
RACK_RGBA = RING_RGBA            # opaque, as the cold rack's (a near-transparent rack would show the fills; the
                                # tubes are opaque, and an empty slot has to show)
RACK_CLEARANCE = 0.00075        # m, radial: a 15 mL tube rack's hole (about 1.5 mm over the tube); at 2 mm a
                                # tube leaned up to 2.3 deg, which the hand takes over at the grasp: 3-4 mm at the tip
COLD_XY, COLD_PITCH, COLD_SHIFT, COLD_DEPTH = np.array([0.20, 0.64]), 0.045, 0.02, 0.05
RACK_CAM = {"back": 0.45, "up": 0.08, "fovy": 50.0, "width": 640, "height": 480}


def slot_names():
    return [f"A{j + 1}" for j in range(N_SLOTS)]


def cold_names():
    return [f"{r}{c}" for r in "ABC" for c in (1, 2, 3)]


# the bench's per-episode draw: each value uniform in +-range around the nominal, scaled by
# `a.bench_scale` (1: the task's ranges; 0: the nominal bench)
BENCH_RANGES = {"centre": 0.040,          # m, the centrifuge's centre, in x and in y
                "cab_yaw": 20.0,          # deg, the cabinet's yaw (the chamber, lid and hinge)
                "hole_r": 0.005,          # m, the hole circle's radius (around HOLE_RADIUS_AT)
                "rotor_top": 0.010,       # m, the rotor top's height
                "rack": 0.030,            # m, the incoming rack's centre, in x and in y
                "rack_yaw": 10.0,         # deg
                "cold": 0.040,            # m, the cold rack's centre, in x and in y
                "cold_yaw": 20.0,         # deg
                "scale": 0.030}           # m, the scale's centre, in x and in y


BENCH_CLEAR = {"slot": 0.06, "cold": 0.06, "scale": 0.10, "scale_slot": 0.10,   # m, see `bench_ok`
               "cold_slot": 0.07}
HANDLE_REACH = 0.60             # m, the open lid's handle at most this far from the robot's base on the table (inverse
                                # kinematics over the drawn poses: every handle within it was reachable from above with
                                # 0.4 rad or more to the joint limits, every one beyond it was not)


def open_handle_xy(b):
    """Where the open lid's handle is on the table for a drawn bench (its hinge on the cabinet's -y wall)."""
    yaw = b["cab_yaw"]
    hinge = b["centre"] + rot_z(yaw)[:2, :2] @ np.array([0.0, -CHAMBER])
    y_l = HANDLE[1] * math.cos(LID_OPEN) - HANDLE[2] * math.sin(LID_OPEN)
    return hinge + rot_z(yaw)[:2, :2] @ np.array([0.0, y_l])


def _square_dist(pt, c, yaw, half):
    """Distance from a point to a square (centre c, yaw, half size) on the table; 0 inside."""
    lp = rot_z(-yaw)[:2, :2] @ (np.asarray(pt, float) - c)
    return float(np.linalg.norm(np.maximum(np.abs(lp) - half, 0.0)))


def bench_ok(b):
    """Whether a drawn bench keeps the instruments apart: every incoming slot and cold hole BENCH_CLEAR from the
    centrifuge's cabinet (a hand grasping there stays off its walls), the scale's centre from the cabinet and from
    every slot, and every cold hole from every incoming slot (a cold rack turned onto the incoming rack's end
    left a tube lifted from A10 caught on a cold ring)."""
    half = CHAMBER + 0.005
    if float(np.linalg.norm(open_handle_xy(b))) > HANDLE_REACH:
        return False
    slots, cold = _rack_points(b)
    if min(_square_dist(s, b["centre"], b["cab_yaw"], half) for s in slots) < BENCH_CLEAR["slot"]:
        return False
    if min(_square_dist(h, b["centre"], b["cab_yaw"], half) for h in cold) < BENCH_CLEAR["cold"]:
        return False
    if _square_dist(b["scale"], b["centre"], b["cab_yaw"], half) < BENCH_CLEAR["scale"]:
        return False
    if min(float(np.linalg.norm(np.asarray(h) - np.asarray(s))) for h in cold for s in slots) < BENCH_CLEAR["cold_slot"]:
        return False
    return min(float(np.linalg.norm(b["scale"] - s)) for s in slots) >= BENCH_CLEAR["scale_slot"]


def _rack_points(b):
    mid = (N_SLOTS - 1) / 2
    c_in = np.array([RACK_X, RACK_Y0 + RACK_PITCH * mid]) + b["rack_shift"]
    R_in = rot_z(b["rack_yaw"])[:2, :2]
    slots = [c_in + R_in @ np.array([0.0, RACK_PITCH * (j - mid)]) for j in range(N_SLOTS)]
    c_cold = COLD_XY + b["cold_shift"]
    R_cold = rot_z(b["cold_yaw"])[:2, :2]
    cold = [c_cold + R_cold @ (COLD_PITCH * np.array([i - 1, j - 1])) for i in range(3) for j in range(3)]
    return slots, cold


def bench_draw(seed, scale=1.0):
    """This episode's bench: the centrifuge's pose and rotor dimensions, the racks' and the scale's poses (true);
    drawn again (the same generator, so still one bench per seed) until `bench_ok`."""
    rng = np.random.default_rng([seed, 43])
    while True:
        b = _bench_once(rng, scale)
        if bench_ok(b):
            return b


def _bench_once(rng, scale):
    r = {k: v * scale for k, v in BENCH_RANGES.items()}
    return {"centre": CENTRE + rng.uniform(-r["centre"], r["centre"], 2),
            "cab_yaw": math.radians(rng.uniform(-r["cab_yaw"], r["cab_yaw"])),
            "hole_r": HOLE_RADIUS_AT + rng.uniform(-r["hole_r"], r["hole_r"]),
            "rotor_top": ROTOR_TOP + rng.uniform(-r["rotor_top"], r["rotor_top"]),
            "rack_shift": rng.uniform(-r["rack"], r["rack"], 2),
            "rack_yaw": math.radians(rng.uniform(-r["rack_yaw"], r["rack_yaw"])),
            "cold_shift": rng.uniform(-r["cold"], r["cold"], 2),
            "cold_yaw": math.radians(rng.uniform(-r["cold_yaw"], r["cold_yaw"])),
            "scale": SCALE_XY + rng.uniform(-r["scale"], r["scale"], 2)}


def set_bench(a):
    """Put this episode's centrifuge pose and rotor dimensions on `a` (`bench_draw` of `a.seed`), once."""
    if getattr(a, "bench", None) is None:
        a.bench = bench_draw(a.seed, getattr(a, "bench_scale", 1.0))
    for k in ("centre", "cab_yaw", "hole_r", "rotor_top"):
        setattr(a, k, a.bench[k])
    return a.bench


def layout(a):
    """Where the racks are in this episode: the incoming slots' (x, y) and the cold rack's holes, true. Each rack
    turns about its own centre; the cold rack's positions are named in its own frame (A-C along its x)."""
    bench = set_bench(a)
    slots, pts = _rack_points(bench)
    top_z = B.TABLE_Z + COLD_DEPTH + 0.015
    cold = {}
    for n, xy in zip([f"{r}{j + 1}" for r in "ABC" for j in range(3)], pts):
        cold[n] = {"top": np.array([xy[0], xy[1], top_z]), "R": np.eye(3), "axis": np.array([0.0, 0.0, 1.0])}
    return slots, cold


def rack_mid_z(a):
    """Height of the tubes' middle in the incoming rack (the rack's geometry, known)."""
    return B.TABLE_Z + 0.015 + 0.0005 + a.tube_len / 2


def build_bench(a, seed, fills):
    """The centrifuge world with this bench (`a.seed`'s draw). Returns (model, sources, tube names, cold holes)."""
    slots, cold = layout(a)
    a.src_xy = slots
    a.src_depth = SRC_DEPTH
    a.src_rgba = RACK_RGBA
    a.src_clearance = RACK_CLEARANCE
    a.tubes = N_SLOTS
    mid = rack_mid_z(a)
    yc = RACK_Y0 + RACK_PITCH * (N_SLOTS - 1) / 2

    def extra(wb):
        if getattr(a, "scale", True):
            a.scale_hole = add_scale(wb, a.bench["scale"], a)
        for name, h in cold.items():
            h.update(add_ring(wb, f"cold_{name}", h["top"], h["R"], a.tube_d / 2 + SOURCE_CLEARANCE,
                                 COLD_DEPTH, rgba=RING_RGBA))
        cam = wb.add_camera()
        cam.name = "rack_cam"
        pos = np.array([RACK_X - RACK_CAM["back"], yc, mid + RACK_CAM["up"]])
        cam.pos = list(pos)
        cam.quat = list(B.quat_from_mat(B.look_at(pos, np.array([RACK_X, yc, mid]))))
        cam.fovy = RACK_CAM["fovy"]

    m, src, names = build(a, fills=fills, extra=extra)
    m.site_rgba[:, 3] = 0.0                          # sites are not part of the scene
    for h in src:
        h["rack_yaw"] = float(a.bench["rack_yaw"])   # an incoming-rack slot: its neighbours lie along the rack
    return m, src, names, cold


# the bench scale: a lab balance with one upright tube holder on its pan; the force between its
# body and the table is the load; the display (`World.scale_display`) settles with SCALE_TAU, shows SCALE_STEP
# steps and has SCALE_NOISE of noise
SCALE_XY = np.array([0.54, 0.32])
SCALE_HALF = np.array([0.045, 0.045, 0.015])     # m, its base's half sizes
SCALE_TAU, SCALE_NOISE, SCALE_STEP = 0.4, 0.03, 0.1   # s, g, g
SCALE_RGBA = (0.25, 0.27, 0.30, 1)


def add_scale(wb, xy, a):
    """The scale at (x, y): a base box on the table and a holder (a rack hole) on it, all one body; a force
    sensor at its base. Returns the holder's hole (a rack hole's dict)."""
    sb = wb.add_body(name="scale", pos=[xy[0], xy[1], B.TABLE_Z + SCALE_HALF[2]])
    g = sb.add_geom(name="scale_base")
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(SCALE_HALF)
    g.rgba = list(SCALE_RGBA)
    g.mass = 0.3
    top_local = np.array([0.0, 0.0, SCALE_HALF[2] + SRC_DEPTH + 0.004])
    h = add_ring(sb, "scale_holder", top_local, np.eye(3), a.tube_d / 2 + RACK_CLEARANCE, SRC_DEPTH,
                 rgba=(0.55, 0.57, 0.62, 1))
    sb.add_site(name="scale_site", pos=[0, 0, -SCALE_HALF[2]])
    return dict(h, top=np.array([xy[0], xy[1], B.TABLE_Z + SCALE_HALF[2]]) + top_local)


def rack_cam_K():
    f = (RACK_CAM["height"] / 2) / math.tan(math.radians(RACK_CAM["fovy"]) / 2)
    return np.array([[f, 0, RACK_CAM["width"] / 2], [0, f, RACK_CAM["height"] / 2], [0, 0, 1.0]])


# an episode's worklist: the spun tubes' fills drawn so that one balanced pairing exists: each pair
# PAIR_DIFF apart, consecutive pairs PAIR_GAP apart (so two tubes within 1 g are a pair, and weighing to a few
# tenths of a gram finds the pairing); 1 mL weighs 1 g
PAIR_DIFF = (0.2, 0.8)          # g
PAIR_GAP = (1.2, 2.0)           # g
FILL_RANGE = (3.0, 13.0)        # mL
FILL_GAP = 2.0                  # mL, an earlier worklist's levels (used by the reference's fill reader only)
FILL_PAIR = 0.3
N_COLD = 2                      # tubes to the cold rack


SIM_DEFAULTS = {"opaque": True, "rotor_model": "solid", "no_lid": False, "lid_side": "right", "lid_open": LID_OPEN,
                "lid_start": LID_OPEN, "lid_mass": 0.4, "lid_damping": 0.3, "lid_friction": 0.5,
                "rotor_damping": 0.05, "rotor_jitter": 0.0, "clearance": 0.001, "tube_d": 0.017,
                "tube_len": 0.120, "mass": 0.015, "com_z": 0.035, "friction": 0.6, "tubes": 2,
                "port": True, "ratchet": True}


def sim_defaults(a):
    """The simulation's settings on `a` where not set (the probes' defaults)."""
    for k, v in SIM_DEFAULTS.items():
        if not hasattr(a, k):
            setattr(a, k, v)


def sim_args(seed):
    """A fresh settings object for an episode of `seed` (the environment's; the reference adds its own)."""
    import argparse
    a = argparse.Namespace(seed=seed, rotor_jitter=0.0)
    sim_defaults(a)
    return a


UNIT_PROPS = ("rotor_friction", "handle_play", "rotor_inertia")
# the scored units span these; the practice units sit at the low end (PRACTICE_RANGES). Friction
# is against a grip 18 mm from the axis (the shield's inner edge): 0.45 N m there loses about what 0.65 did at 26 mm
UNIT_RANGES = {"rotor_friction": (0.1, 0.45), "handle_play": (0.0, 20.0), "rotor_inertia": (0.01, 0.04)}
PRACTICE_RANGES = {"rotor_friction": (0.1, 0.15), "handle_play": (0.0, 4.0), "rotor_inertia": (0.01, 0.015)}
N_PAIRS = (2, 4)                # the rotor's opposite pairs to fill (with the tubes already in it)
N_PRE = (0, 2)                  # tubes already seated in the rotor (another job's; spun too, left where they are)
N_COLD_V2 = (2, 3)              # tubes to the cold rack
N_FOREIGN = (2, 4)              # cold-rack positions already holding other people's tubes
N_IN_MIN = 8                    # tubes in the incoming rack: N_IN_MIN to N_SLOTS
LID_CLOSED_P = 0.5              # the lid starts closed (it must be opened first)


def draw_unit(seed):
    """An episode's draw: the unit's turn properties (replaced by the unit's when graded,
    `draw_episode`), the rotor's start angle, which incoming slots hold tubes, which of them go to the cold rack
    (to which positions) and which are spun, the tubes already in the rotor (their holes and masses), the cold
    positions already taken, the lid's start, and the fills (`fills[slot]`, None for an empty slot)."""
    rng = np.random.default_rng([seed, 23])
    unit = {k: float(rng.uniform(*UNIT_RANGES[k])) for k in UNIT_PROPS}
    unit["rotor_start"] = float(rng.uniform(0, 360))
    n_pairs = int(rng.integers(N_PAIRS[0], N_PAIRS[1] + 1))
    n_pre = int(rng.integers(N_PRE[0], min(N_PRE[1], n_pairs) + 1))
    n_cold = int(rng.integers(N_COLD_V2[0], N_COLD_V2[1] + 1))
    n_spin = 2 * n_pairs - n_pre
    n_cold = min(n_cold, N_SLOTS - n_spin)
    n_in = int(rng.integers(max(N_IN_MIN, n_spin + n_cold), N_SLOTS + 1))
    while True:
        lo, pairs = rng.uniform(FILL_RANGE[0], FILL_RANGE[0] + 2.0), []
        for i in range(n_pairs):
            d = rng.uniform(*PAIR_DIFF)
            pairs.append((lo, lo + d))
            lo += d + rng.uniform(*PAIR_GAP)
        if pairs[-1][1] <= FILL_RANGE[1]:
            break
    pre_pairs = sorted(int(i) for i in rng.choice(n_pairs, n_pre, replace=False))
    pre_fills, spun = [], []
    for i, (f1, f2) in enumerate(pairs):
        if i in pre_pairs:
            keep = int(rng.integers(0, 2))
            pre_fills.append((f1, f2)[keep])
            spun.append((f1, f2)[1 - keep])
        else:
            spun += [f1, f2]
    # the tubes already in the rotor: holes no two of which are opposite
    holes = []
    for h in rng.permutation(NHOLES):
        if len(holes) < n_pre and (int(h) + NHOLES // 2) % NHOLES not in holes:
            holes.append(int(h))
    occupied = sorted(int(i) for i in rng.choice(N_SLOTS, n_in, replace=False))
    roles = rng.permutation(occupied)
    spin_slots = sorted(int(i) for i in roles[:n_spin])
    cold_slots = sorted(int(i) for i in roles[n_spin:n_spin + n_cold])
    names = cold_names()
    picked = [str(n) for n in rng.permutation(names)]
    cold_to, foreign = picked[:n_cold], sorted(picked[n_cold:n_cold + int(rng.integers(N_FOREIGN[0],
                                                                                         N_FOREIGN[1] + 1))])
    fills = [None] * N_SLOTS
    for i, f in zip(spin_slots, rng.permutation(spun)):
        fills[i] = round(float(f), 3)
    for i in occupied:
        if fills[i] is None:
            fills[i] = round(float(rng.uniform(*FILL_RANGE)), 3)
    unit.update(fills=fills, cold=dict(zip(cold_slots, cold_to)), spin=spin_slots,
                preloaded={h: round(float(f), 3) for h, f in zip(holes, pre_fills)},
                foreign={n: round(float(rng.uniform(*FILL_RANGE)), 3) for n in foreign},
                lid_closed=bool(rng.random() < LID_CLOSED_P))
    return unit




def draw_episode(seed, props):
    """An episode of a unit (the unit's hidden properties fixed, the rest from the seed): `draw_unit(seed)` with
    the unit's properties in place of the drawn ones."""
    ep = draw_unit(seed)
    ep.update({k: float(props[k]) for k in UNIT_PROPS})
    return ep


def world(a, unit, world_cls=World):
    """The episode's world: `(p, src, names, cold, masses)`; `names[slot]` and `masses[slot]` (kg) per incoming
    slot (None where empty). The tubes already in the rotor (`p.pre`: hole -> body) and other people's in the cold
    rack (`p.foreign`: position -> body) are extra bodies, their masses in `p.extra_mass`."""
    for k in ("rotor_friction", "handle_play", "rotor_inertia", "rotor_start"):
        setattr(a, k, unit[k])
    a.tube_slots = [i for i, f in enumerate(unit["fills"]) if f is not None]
    a.lid_start = 0.0 if unit.get("lid_closed") else a.lid_open
    set_bench(a)
    _slots, cold_xy = layout(a)
    extra = {f"pre{h}": np.zeros(3) for h in unit.get("preloaded", {})}
    for n in unit.get("foreign", {}):
        extra[f"fx_{n}"] = cold_xy[n]["top"] - np.array([0, 0, COLD_DEPTH - 0.0005])
    a.extra_tubes = extra
    m, src, names, cold = build_bench(a, a.seed, unit["fills"])
    masses = [None if f is None else a.mass + f * 1e-3 for f in unit["fills"]]
    for name, mass in zip(names, masses):
        if name is not None:
            set_tube_mass(m, name, a, mass=mass)
    extra_mass = {f"pre{h}": a.mass + f * 1e-3 for h, f in unit.get("preloaded", {}).items()}
    extra_mass.update({f"fx_{n}": a.mass + f * 1e-3 for n, f in unit.get("foreign", {}).items()})
    for name, mass in extra_mass.items():
        set_tube_mass(m, name, a, mass=mass)
    mujoco.mj_setConst(m, mujoco.MjData(m))
    p = world_cls(m, a)
    p.pre = {int(h): f"pre{h}" for h in unit.get("preloaded", {})}
    p.foreign = {n: f"fx_{n}" for n in unit.get("foreign", {})}
    p.extra_mass = extra_mass
    if p.pre:
        # the rotor's tubes seated at the floor of their holes at the rotor's start angle
        d = p.data
        for h, name in p.pre.items():
            hole = p.hole(h)
            j = m.joint(name + "_free").qposadr[0]
            d.qpos[j:j + 3] = hole["top"] - hole["axis"] * (HOLE_DEPTH - 0.0005)
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, hole["R"].flatten())
            d.qpos[j + 3:j + 7] = q
        mujoco.mj_forward(m, d)
    if world_cls is World:                    # a shadow (`wire_world.WireWorld`) takes its state from the wire
        for _ in range(10):
            p.step([0, 0, 0], [0, 0, 0], -1)
    return p, src, names, cold, masses
