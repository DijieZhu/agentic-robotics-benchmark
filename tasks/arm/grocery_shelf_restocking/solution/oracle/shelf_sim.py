"""Shelf restocking: the scene's building blocks (a gondola bay, the robot on a pedestal or a lift, primitive products).

Box transport's Panda, physics options and operational-space controller (`box_transport_env.py`, copied
unchanged from `../box_env/`), with the robot on a pedestal facing a gondola bay built from primitives:
three boards, a front lip on each, a back panel, two uprights and a top. Products are primitives here
(boxes and cylinders sized like grocery items); appearance comes later.

Frames: world z up; the robot base at (0, 0, BASE_Z) facing +x; the bay's front plane (the lips' outer
face) at x = bay["front_x"], centred on y = bay["y"]. Metres, radians.
"""

import itertools
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco
import numpy as np

import box_transport_env as B

BASE_Z = 0.60                  # m, the robot base on a pedestal (without the lift)
LIFT_RANGE = (0.05, 1.30)      # m, the robot base's height on the vertical lift
LIFT_STEP = 0.01               # m per control step at most (0.1 m/s)
LIFT_KP, LIFT_KD = 2.0e5, 1.5e4
BOARD_T = 0.02                 # m, board thickness
LIP_H = 0.02                   # m, front lip height above the board's top
LIP_T = 0.01                   # m, lip thickness (along x)
UPRIGHT_T = 0.03
BACK_T = 0.01
BAY_RGBA = (0.82, 0.83, 0.85, 1)
LIP_RGBA = (0.20, 0.22, 0.25, 1)


def default_bay(**kw):
    """One bay: `boards` = top-surface heights (world z) of the shelves, bottom first; `top_z` = the
    underside of the bay's top cover; `depth` = usable depth from the lip's inner face to the back panel."""
    bay = {"front_x": 0.45, "y": 0.0, "yaw": 0.0, "width": 0.90, "depth": 0.32,
           "boards": [0.55, 0.85, 1.15], "top_z": 1.45}
    bay.update(kw)
    return bay


def bay_frame(bay):
    c, s = math.cos(bay["yaw"]), math.sin(bay["yaw"])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return R, np.array([bay["front_x"], bay["y"], 0.0])


def add_box(body, name, centre, half, rgba, group=0):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(half)
    g.pos = list(centre)
    g.rgba = list(rgba)
    g.group = group
    return g


def add_bay(wb, bay):
    """The bay as one static body `bay` (bay frame: x into the bay from the lip's outer face, y left)."""
    R, p = bay_frame(bay)
    body = wb.add_body()
    body.name = "bay"
    body.pos = list(p)
    body.quat = list(B.quat_from_mat(R))
    D, W = bay["depth"], bay["width"]
    xb = LIP_T + D + BACK_T            # the back panel's outer face
    for i, z in enumerate(bay["boards"]):
        add_box(body, f"board{i}", (xb / 2, 0, z - BOARD_T / 2), (xb / 2, W / 2, BOARD_T / 2), BAY_RGBA)
        add_box(body, f"lip{i}", (LIP_T / 2, 0, z + LIP_H / 2), (LIP_T / 2, W / 2, LIP_H / 2), LIP_RGBA)
    z0 = bay["boards"][0] - BOARD_T
    zt = bay["top_z"]
    add_box(body, "top", (xb / 2, 0, zt + BOARD_T / 2), (xb / 2, W / 2 + UPRIGHT_T, BOARD_T / 2), BAY_RGBA)
    add_box(body, "back", (LIP_T + D + BACK_T / 2, 0, (z0 + zt) / 2), (BACK_T / 2, W / 2, (zt - z0) / 2), BAY_RGBA)
    for side in (-1, 1):
        add_box(body, f"upright{'LR'[side > 0]}", (xb / 2, side * (W / 2 + UPRIGHT_T / 2), zt / 2),
                (xb / 2, UPRIGHT_T / 2, zt / 2), BAY_RGBA)
    # the plinth under the bottom board (to the floor)
    add_box(body, "plinth", (xb / 2, 0, z0 / 2), (xb / 2, W / 2, z0 / 2), BAY_RGBA)
    return body


def base_world(bay, products=(), stiff=True, lift=False):
    """MjSpec: box transport's physics options, floor, pedestal, Panda, the bay and `products`
    (list of dicts: name, kind box|cyl, size (half extents or (r, half h)), pos, yaw, mass, com_z).
    `lift`: the Panda's base rides a vertical slide joint `lift` (its value = the base's height, LIFT_RANGE)
    on a column (visual only) instead of the fixed pedestal; a motor `lift_motor` drives it."""
    w = mujoco.MjSpec()
    w.compiler.degree = False
    w.compiler.autolimits = True
    w.option.timestep = B.TIMESTEP
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = B.IMPRATIO
    if stiff:
        w.option.noslip_iterations = B.STIFF_NOSLIP_ITERATIONS
        w.memory = B.STIFF_ARENA_BYTES
    w.visual.global_.offwidth = 1280
    w.visual.global_.offheight = 960
    wb = w.worldbody
    light = wb.add_light()
    light.pos = [0.0, 0.0, 2.6]
    light.dir = [0.3, 0.0, -1.0]
    light.diffuse = [0.8, 0.8, 0.8]
    light.castshadow = False
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [4, 4, 0.1]
    g.rgba = [0.55, 0.52, 0.48, 1]
    if lift:
        carriage = wb.add_body()
        carriage.name = "carriage"
        j = carriage.add_joint()
        j.name = "lift"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = [0, 0, 1]
        j.range = list(LIFT_RANGE)
        j.damping = [50.0, 0.0, 0.0]          # MjSpec 3.16: a 3-vector (the first entry is the damping)
        col = add_box(carriage, "column", (-0.05, 0, -0.65), (0.14, 0.14, 0.65), (0.3, 0.3, 0.32, 1))
        col.contype = 0
        col.conaffinity = 0
        col.mass = 8.0
        mot = w.add_actuator()
        mot.name = "lift_motor"
        mot.target = "lift"
        mot.trntype = mujoco.mjtTrn.mjTRN_JOINT
        mot.ctrlrange = [-5000, 5000]
        mount = carriage
    else:
        add_box(wb, "pedestal", (-0.05, 0, BASE_Z / 2), (0.16, 0.16, BASE_Z / 2), (0.3, 0.3, 0.32, 1))
        mount = None
    ra = B.robosuite_assets()
    robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml"))
    for cam in list(robot.cameras):
        if cam.name == "robotview":
            robot.delete(cam)
    robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
    if mount is None:
        wb.add_frame(pos=[0.0, 0.0, BASE_Z]).attach_body(robot.body("base"), "r_", "")
    else:
        mount.add_frame(pos=[0.0, 0.0, 0.0]).attach_body(robot.body("base"), "r_", "")
    add_bay(wb, bay)
    for p in products:
        add_product(wb, p)
    return w


def add_product(wb, p):
    body = wb.add_body()
    body.name = p["name"]
    body.pos = list(p["pos"])
    body.quat = list(B.yaw_quat(p.get("yaw", 0.0)))
    body.add_freejoint().name = p["name"] + "_free"
    g = body.add_geom()
    g.name = p["name"] + "_geom"
    if p["kind"] == "box":
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(p["size"])
    else:
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [p["size"][0], p["size"][1], 0]
    g.rgba = list(p.get("rgba", (0.8, 0.3, 0.2, 1)))
    g.friction = [p.get("friction", 0.6), 0.005, 0.0001]
    return body


def finish(m, products=()):
    """Masses, centres of mass and the stiff contact model on the pads and the products (after compile)."""
    for p in products:
        b = m.body(p["name"]).id
        B.set_mass(m, b, p["mass"])
        if p.get("com_z") is not None:
            ipos = m.body_ipos[b].copy()
            ipos[2] = p["com_z"]
            m.body_ipos[b] = ipos
    if m.opt.noslip_iterations:
        stiff = {m.geom("r_g_finger1_pad_collision").id, m.geom("r_g_finger2_pad_collision").id}
        stiff |= {m.geom(p["name"] + "_geom").id for p in products}
        for g in stiff:
            m.geom_condim[g] = 4
            m.geom_friction[g, 1] = B.STIFF_TORSION
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


# ----------------------------------------------------------------------------------------------
# Kinematics: tool orientations and damped-least-squares IK on the grip site
# ----------------------------------------------------------------------------------------------
# Grip-site frame (measured): z = approach (out of the fingertips), x = the fingers' closing axis;
# the hand's collision box spans x +-0.105, y +-0.032, z -0.124..+0.012 about the site.
def tool_R(name, yaw=0.0):
    """World rotation of the grip site for a named tool pose, the bay facing -x (yaw about z added):
    flat  - approach +x, fingers close along y (palm 21 cm wide horizontally, 6.4 cm tall);
    tall  - approach +x, fingers close along z (palm 21 cm tall, 6.4 cm wide): a pusher;
    down  - approach -z, fingers close along y;
    down45 - approach pitched 45 deg down from +x, fingers along y;
    pNN   - approach pitched NN deg down from +x, fingers along y."""
    if name == "flat":
        z, x = [1, 0, 0], [0, 1, 0]
    elif name == "tall":
        z, x = [1, 0, 0], [0, 0, 1]
    elif name == "down":
        z, x = [0, 0, -1], [0, 1, 0]
    elif name == "down45":
        z, x = [math.sqrt(0.5), 0, -math.sqrt(0.5)], [0, 1, 0]
    elif name.startswith("p") and name[1:].isdigit():      # approach pitched N deg below horizontal
        a = math.radians(int(name[1:]))
        z, x = [math.cos(a), 0, -math.sin(a)], [0, 1, 0]
    else:
        raise ValueError(name)
    z, x = np.array(z, float), np.array(x, float)
    R = np.column_stack([x, np.cross(z, x), z])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz @ R


class Kin:
    """Kinematics only (no dynamics): IK and the robot-vs-bay contacts at a joint posture."""

    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.jq = np.array([m.joint(f"r_joint{i}").qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint(f"r_joint{i}").dofadr[0] for i in range(1, 8)])
        self.site = m.site("r_g_grip_site").id
        rng = np.array([m.joint(f"r_joint{i}").range for i in range(1, 8)])
        self.lo, self.hi = rng[:, 0].copy(), rng[:, 1].copy()
        self.lift_adr = m.joint("lift").qposadr[0] if "lift" in [m.joint(i).name for i in range(m.njnt)] else None
        self.lift_z = BASE_Z
        bay = m.body("bay").id
        self.bay_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bay}
        self.robot_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("r_")
                            and (m.geom_contype[g] or m.geom_conaffinity[g])}
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def set_q(self, q):
        self.d.qpos[self.jq] = q
        self.d.qpos[self.jq[-1] + 1], self.d.qpos[self.jq[-1] + 2] = 0.0, 0.0
        if self.lift_adr is not None:
            self.d.qpos[self.lift_adr] = self.lift_z
        mujoco.mj_forward(self.m, self.d)

    def ik(self, pos, R, q0, iters=300, tol=(1e-3, 1e-2), q_pref=None):
        q = np.clip(np.asarray(q0, float).copy(), self.lo + 1e-3, self.hi - 1e-3)
        for _ in range(iters):
            self.set_q(q)
            ep = pos - self.d.site_xpos[self.site]
            Rc = self.d.site_xmat[self.site].reshape(3, 3)
            Re = R @ Rc.T
            eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(eo) < tol[1]:
                return q, True
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            J = np.vstack([self.jp, self.jr])[:, self.jv]
            e = np.concatenate([ep, eo])
            lam = 0.05
            JJ = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            if q_pref is not None:
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq += N @ (0.1 * (q_pref - q))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)
        return q, False

    def bay_hits(self, margin=0.005):
        """Robot geoms closer than `margin` to a bay geom: list of (robot geom, bay geom, distance)."""
        out = []
        fromto = np.zeros(6)
        for gr in self.robot_geoms:
            for gb in self.bay_geoms:
                dist = mujoco.mj_geomDistance(self.m, self.d, gr, gb, margin, fromto)
                if dist < margin:
                    out.append((self.m.geom(gr).name, self.m.geom(gb).name, float(dist)))
        return out

    def intrusion(self, bay, y0, z_lo, z_hi):
        """Exact (mesh vertices) extent of the robot inside one shelf's slot (bay frame: past the lip's inner
        face, between z_lo and z_hi): (largest |y - y0|, deepest x past the lip), or (0, 0) if outside."""
        Rb, pb = bay_frame(bay)
        m, d = self.m, self.d
        hw, deep = 0.0, 0.0
        for g in self.robot_geoms:
            Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == 7:                         # mesh
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                v = np.array([c + h * np.array(s) for s in itertools.product((-1, 1), repeat=3)])
            w = (v @ Rg.T + pg - pb) @ Rb
            inside = (w[:, 0] > LIP_T) & (w[:, 2] > z_lo) & (w[:, 2] < z_hi)
            if inside.any():
                hw = max(hw, float(np.max(np.abs(w[inside, 1] - y0))))
                deep = max(deep, float(np.max(w[inside, 0] - LIP_T)))
        return hw, deep

    def intrusion_halfwidth(self, bay, y0):
        """Largest |y - y0| (bay frame) of robot collision geoms whose box reaches past the lips into the bay:
        the half-width a lane must keep clear around y0 for this posture."""
        Rb, pb = bay_frame(bay)
        worst = 0.0
        for g in self.robot_geoms:
            c = self.d.geom_xpos[g] + self.d.geom_xmat[g].reshape(3, 3) @ self.m.geom_aabb[g, :3]
            h = np.abs(self.d.geom_xmat[g].reshape(3, 3)) @ self.m.geom_aabb[g, 3:]
            lc = Rb.T @ (c - pb)
            lh = np.abs(Rb.T) @ h
            if lc[0] + lh[0] > LIP_T:            # reaches into the bay
                worst = max(worst, abs(lc[1] - y0) + lh[1])
        return worst
