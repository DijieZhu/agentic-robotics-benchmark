"""The SR-2 rescue crawler (cell internals): the robot's MuJoCo model and its on-board controller.

A slim chassis spine carries two main track modules on telescopic cross beams (track gauge 0.30-0.50 m), a pair of
front and a pair of rear flippers on the track modules' axles, a telescopic sensor mast at the rear (lidar puck on
its top plate, a pan-tilt head with the RGB-D camera, the thermal camera and two LED lamps), a five-axis arm with a
parallel gripper at the front and a survival-kit cradle on the rear bumper.

Tracks are modelled as conveyor surfaces (MuJoCo geom `surfacevel`): every track run and every sprocket / idler
drum of a side moves at that side's belt speed, so the robot is propelled by its contacts like a real tracked
vehicle (skid steering, slip on loose debris, climbing with the flipper runs). Joints are servoed by position
actuators whose targets move at the axis' rated speed (`Crawler.tick`).

Frames: chassis frame x forward, y left, z up, origin at the spine's centre. Units: metres, radians, seconds.
"""

import math

import mujoco
import numpy as np

__all__ = ["CRAWLER", "build_crawler", "Crawler", "ARM_LINKS", "ARM_STOW", "arm_fk", "arm_ik"]

# ---------------------------------------------------------------------------------------------------- dimensions
CRAWLER = {
    "track_wheel_r": 0.090,          # sprocket / idler radius (track height 0.18 m)
    "track_half_len": 0.270,         # axle to chassis centre along x (axle spacing 0.54 m, track length 0.72 m)
    "track_half_w": 0.040,           # track width 0.08 m
    "axle_z": -0.035,                # axles below the chassis origin
    "gauge_min": 0.30, "gauge_max": 0.50,
    "spine_half": (0.300, 0.095, 0.065),
    "flipper_len": 0.300, "flipper_r0": 0.085, "flipper_r1": 0.045, "flipper_half_w": 0.030,
    "flipper_offset": 0.078,         # flipper centre line outboard of the track centre line
    "mast_x": -0.235, "mast_y": -0.05, "mast_travel": 0.45,
    "arm_x": 0.165, "arm_links": (0.075, 0.340, 0.340, 0.070, 0.150),  # base height, upper arm, forearm, wrist, tcp
    "kit_cradle": (-0.275, 0.055, 0.065),   # the kit's seat (bottom centre) on the deck
    "mass": {"spine": 26.0, "track": 5.0, "flipper": 1.2, "mast": 2.0, "head": 1.6, "arm": (2.2, 2.0, 1.6, 0.7, 0.6)},
    "speed_max": 0.6, "yaw_rate_max": 1.2,
    "flipper_rate": math.radians(60.0), "gauge_rate": 0.05, "mast_rate": 0.10, "head_rate": math.radians(90.0),
    "arm_rate": math.radians(45.0), "grip_rate": 0.08,
    "flipper_range": (math.radians(-80.0), math.radians(190.0)),
    "head_pan_range": (math.radians(-175.0), math.radians(175.0)),
    "head_tilt_range": (math.radians(-60.0), math.radians(45.0)),
    "arm_range": ((math.radians(-175.0), math.radians(175.0)), (math.radians(-20.0), math.radians(190.0)),
                  (math.radians(-186.0), math.radians(186.0)), (math.radians(-120.0), math.radians(120.0)),
                  (math.radians(-180.0), math.radians(180.0))),
    "grip_open_max": 0.16,
}
ARM_LINKS = CRAWLER["arm_links"]
# stowed: upper arm back along the deck, forearm folded flat over it, the gripper tucked down behind the tracks' noses
ARM_STOW = (0.0, math.radians(184.0), math.radians(-184.0), math.radians(-55.0), 0.0)
ROCKER = 0.012
ROCKER_HALF = 0.17
MAST_BASE_H = 0.22         # the stowed mast puts the sensor head above the stowed arm
ARC_FRICTION = 0.65
ROBOT_GROUP = 1            # collision geoms of the robot (drawn)
VISUAL_GROUP = 2
TRACK_RGBA = (0.10, 0.10, 0.10, 1.0)
BODY_RGBA = (0.93, 0.47, 0.08, 1.0)        # safety orange
DARK_RGBA = (0.18, 0.19, 0.21, 1.0)
METAL_RGBA = (0.55, 0.57, 0.60, 1.0)
CT_ROBOT, CA_ROBOT = 2, 1                 # the robot touches the world and debris, never itself


def _q_axis(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


Q_CYL_Y = _q_axis((1, 0, 0), math.pi / 2)   # a cylinder's axis (local z) along the body's y


def _geom(body, name, gtype, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=DARK_RGBA, mass=None, collide=True,
          friction=(0.9, 0.02, 0.001), surfacevel=None, group=None):
    g = body.add_geom()
    g.name = name
    g.type = gtype
    g.size = list(size) + [0.0] * (3 - len(size))
    g.pos = list(pos)
    g.quat = list(quat)
    g.rgba = list(rgba)
    if mass is not None:
        g.mass = mass
    else:
        g.density = 0.0
    if collide:
        g.contype, g.conaffinity = CT_ROBOT, CA_ROBOT
        g.condim = 3
        g.friction = list(friction)
        g.group = ROBOT_GROUP if group is None else group
    else:
        g.contype, g.conaffinity = 0, 0
        g.group = VISUAL_GROUP if group is None else group
    if surfacevel is not None:
        g.surfacevel = list(surfacevel)
    return g


def _hinge(body, name, axis, rng, damping=1.0, armature=0.01):
    j = body.add_joint()
    j.name = name
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = list(axis)
    j.range = list(rng)
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = [damping, 0.0, 0.0]
    j.armature = armature
    return j


def _slide(body, name, axis, rng, damping=50.0, armature=0.1):
    j = body.add_joint()
    j.name = name
    j.type = mujoco.mjtJoint.mjJNT_SLIDE
    j.axis = list(axis)
    j.range = list(rng)
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = [damping, 0.0, 0.0]
    j.armature = armature
    j.solref_limit = [0.004, 1.0]
    return j


def _pos_act(spec, name, joint, kp, kv, frange, ctrlrange=None):
    a = spec.add_actuator()
    a.name = name
    a.target = joint
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.set_to_position(kp=kp, kv=kv)
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    a.forcerange = [-frange, frange]
    if ctrlrange is not None:
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = list(ctrlrange)
    return a


def _mat_quat(xa, za):
    """wxyz quaternion of the frame whose x axis is xa and z axis is za (unit, orthogonal)."""
    xa = np.asarray(xa, float)
    za = np.asarray(za, float)
    ya = np.cross(za, xa)
    R = np.column_stack([xa, ya, za])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return list(q)


def belt_loop(body, prefix, c0, r0, c1, r1, half_w, arcs=("front", "rear"), runs=("low", "up"), facets=6,
              rocker=0.0, rocker_half=None, mass=1.0, rgba=None):
    """A track belt loop around two drums in the body's xz plane: c0 = rear / hub drum centre (x, z), c1 = front /
    tip drum centre. Every geom is a thin box whose local x is the direction its surface travels when the belt
    drives the vehicle forward (bottom run backward, front arc downward, top run forward, rear arc upward) and whose
    local z is the outward normal; at run time each gets surfacevel (v, 0, 0). Returns the geom names."""
    t = 0.008
    names = []
    (x0, z0), (x1, z1) = c0, c1
    L = math.hypot(x1 - x0, z1 - z0)
    ux = np.array([(x1 - x0) / L, 0.0, (z1 - z0) / L])     # from c0 to c1
    uz = np.array([-ux[2], 0.0, ux[0]])                       # "up", perpendicular in the plane
    sb = (r0 - r1) / L
    cb = math.sqrt(max(0.0, 1.0 - sb * sb))
    n_low = sb * ux - cb * uz
    n_up = sb * ux + cb * uz
    C0 = np.array([x0, 0.0, z0])
    C1 = np.array([x1, 0.0, z1])
    k = 0
    def add_box(center, xa, za, half_len, nm, arc=False):
        g = _geom(body, "%s_%s" % (prefix, nm), mujoco.mjtGeom.mjGEOM_BOX, (half_len, half_w, t),
                  pos=tuple(center), quat=_mat_quat(xa, za), surfacevel=[1e-9, 0, 0, 0, 0, 0], mass=mass,
                  rgba=rgba or TRACK_RGBA)
        if arc:
            # round a drum the belt meets an obstacle's face end-on: little grip there (a track does not climb a
            # wall with its sprocket), the runs keep the ground's grip (world priority)
            g.priority = 2
            g.friction = [ARC_FRICTION, 0.02, 0.001]
        names.append(g.name)
    def tdir(n):
        # travel direction at a point with outward normal n: (n_z, 0, -n_x)  [= y x n for forward driving]
        return np.array([n[2], 0.0, -n[0]])
    for run in runs:
        n = n_low if run == "low" else n_up
        a = C0 + r0 * n
        b = C1 + r1 * n
        mid = (a + b) / 2.0
        half = float(np.linalg.norm(b - a)) / 2.0
        sag = 0.0
        if run == "low" and rocker > 0.0:
            # the central part sags `rocker` below the drums (road-wheel suspension); inclined runs join it to the
            # drums' bottoms so the belt profile has no step
            sag = rocker
            hc = rocker_half
            add_box(mid - (t - sag) * n, tdir(n), n, hc, run)
            for tag, sgn in (("lf", 1.0), ("lr", -1.0)):
                p_end = (b if sgn > 0 else a)                       # drum tangent point
                p_mid = mid + sgn * hc * ux + sag * n               # end of the sagging part
                seg = p_end - p_mid
                ln = float(np.linalg.norm(seg))
                if ln < 1e-6:
                    continue
                xa = seg / ln
                if sgn > 0:
                    xa = -xa                                        # bottom runs travel backward (toward c0)
                nz = np.cross(xa, np.array([0.0, 1.0, 0.0]))
                if np.dot(nz, n) < 0:
                    nz = -nz
                add_box((p_end + p_mid) / 2.0 - t * nz, xa, nz, ln / 2.0 + 0.004, tag)
            continue
        add_box(mid - (t - sag) * n + 0.0, tdir(n), n, half, run)
    for arc in arcs:
        if arc == "front":
            cc, rr = C1, r1
            phi0 = math.atan2(n_low[2], n_low[0])
            phi1 = math.atan2(n_up[2], n_up[0])
            if phi0 > phi1:
                phi0 -= 2 * math.pi
        else:
            cc, rr = C0, r0
            phi0 = math.atan2(n_up[2], n_up[0])
            phi1 = math.atan2(n_low[2], n_low[0])
            if phi1 < phi0:
                phi1 += 2 * math.pi
        span = phi1 - phi0
        nf = max(2, int(round(facets * span / math.pi)))
        dphi = span / nf
        for i in range(nf):
            ph = phi0 + (i + 0.5) * dphi
            n = np.array([math.cos(ph), 0.0, math.sin(ph)])
            center = cc + (rr - t) * n
            half = rr * math.tan(dphi / 2.0) + 0.002
            add_box(center, tdir(n), n, half, "%s%d" % (arc[0], i), arc=True)
    return names


def build_crawler(spec, parent, pos=(0, 0, 0), yaw=0.0, name="crawler"):
    """Adds the crawler (free body `chassis`) to `spec` under `parent`. Returns the dict of element names."""
    C = CRAWLER
    spec.compiler.degree = False          # every angle here is in radians
    r = C["track_wheel_r"]
    hl = C["track_half_len"]
    hw = C["track_half_w"]
    names = {"tracks": {"L": [], "R": []}, "flipper_geoms": {}}
    ch = parent.add_body()
    ch.name = "chassis"
    ch.pos = list(pos)
    ch.quat = _q_axis((0, 0, 1), yaw)
    fj = ch.add_freejoint()
    fj.name = "chassis_free"
    sx, sy, sz = C["spine_half"]
    _geom(ch, "spine", mujoco.mjtGeom.mjGEOM_BOX, (sx, sy, sz), mass=C["mass"]["spine"], rgba=BODY_RGBA,
          friction=(0.5, 0.01, 0.001))
    # bumpers (front / rear skid plates, angled)
    for nm, x, ang in (("bumper_f", sx, -0.6), ("bumper_r", -sx, 0.6)):
        _geom(ch, nm, mujoco.mjtGeom.mjGEOM_BOX, (0.03, sy * 0.95, sz * 0.8), pos=(x, 0, -0.01),
              quat=_q_axis((0, 1, 0), ang), mass=0.5, rgba=DARK_RGBA, friction=(0.5, 0.01, 0.001))
    imu = ch.add_site()
    imu.name = "imu"
    imu.pos = [0, 0, 0]
    # ---- track modules on the telescopic cross beams
    gmin = C["gauge_min"]
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        tm = ch.add_body()
        tm.name = "track_%s" % side
        tm.pos = [0.0, sgn * gmin / 2.0, C["axle_z"]]
        _slide(tm, "gauge_%s" % side, (0, sgn, 0), (0.0, (C["gauge_max"] - gmin) / 2.0), damping=200.0, armature=1.0)
        names["tracks"][side] += [(g, 1.0) for g in belt_loop(tm, "belt_%s" % side, (-hl, 0.0), r, (hl, 0.0), r, hw,
                                                               rocker=ROCKER, rocker_half=ROCKER_HALF, mass=0.35)]
        # the cross beam (visual) and the side plate (visual)
        _geom(tm, "%s_plate" % side, mujoco.mjtGeom.mjGEOM_BOX, (hl, 0.004, r * 0.55), pos=(0, sgn * (hw + 0.004), 0),
              rgba=BODY_RGBA, collide=False, mass=0.8)
        _geom(tm, "%s_beam" % side, mujoco.mjtGeom.mjGEOM_BOX, (0.05, 0.12, 0.018), pos=(0.0, -sgn * 0.10, 0.02),
              rgba=METAL_RGBA, collide=False, mass=0.3)
        # flippers on the axles, outboard
        for end, x, fsgn in (("F", hl, 1.0), ("B", -hl, -1.0)):
            fb = tm.add_body()
            fb.name = "flipper_%s%s" % (end, side)
            fb.pos = [x, sgn * C["flipper_offset"], 0.0]
            # angle 0 = flipper pointing straight out (forward for the front pair, backward for the rear pair);
            # positive = tip raised. The rear pair's frame is turned by pi about z so both read the same.
            fb.quat = [1, 0, 0, 0] if fsgn > 0 else _q_axis((0, 0, 1), math.pi)
            _hinge(fb, "flip_%s%s" % (end, side), (0, -1, 0), C["flipper_range"], damping=5.0, armature=0.05)
            L, r0, r1, fw = C["flipper_len"], C["flipper_r0"], C["flipper_r1"], C["flipper_half_w"]
            gl = belt_loop(fb, "fl_%s%s" % (end, side), (0.0, 0.0), r0, (L, 0.0), r1, fw, arcs=("front",), facets=6,
                           mass=0.12)
            # the rear pair's frame is turned by pi about z: driving forward, its belt runs the other way round
            names["tracks"][side] += [(g, 1.0 if fsgn > 0 else -1.0) for g in gl]
            hub = _geom(fb, "fl_%s%s_hub" % (end, side), mujoco.mjtGeom.mjGEOM_CYLINDER, (r0 * 0.6, fw), pos=(0, 0, 0),
                        quat=Q_CYL_Y, mass=0.3, rgba=TRACK_RGBA, collide=False)
            _geom(fb, "fl_%s%s_arm" % (end, side), mujoco.mjtGeom.mjGEOM_BOX, (L / 2.0, 0.004, 0.03),
                  pos=(L / 2.0, sgn * (fw + 0.004), 0), rgba=BODY_RGBA, collide=False, mass=0.1)
    # ---- sensor mast (rear) with the lidar and the pan-tilt head
    mb = ch.add_body()
    mb.name = "mast"
    mb.pos = [C["mast_x"], C["mast_y"], sz]
    _geom(mb, "mast_base", mujoco.mjtGeom.mjGEOM_BOX, (0.045, 0.045, MAST_BASE_H / 2), pos=(0, 0, MAST_BASE_H / 2),
          rgba=DARK_RGBA, mass=0.8)
    mt = mb.add_body()
    mt.name = "mast_top"
    mt.pos = [0.0, 0.0, MAST_BASE_H]
    _slide(mt, "mast", (0, 0, 1), (0.0, C["mast_travel"]), damping=80.0, armature=0.5)
    _geom(mt, "mast_tube", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.022, 0.24), pos=(0, 0, -0.20), rgba=METAL_RGBA,
          collide=False, mass=0.5)
    _geom(mt, "mast_plate", mujoco.mjtGeom.mjGEOM_BOX, (0.06, 0.06, 0.012), pos=(0, 0, 0.05), rgba=DARK_RGBA,
          mass=0.4)
    _geom(mt, "lidar", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.045, 0.035), pos=(0, 0, 0.097), rgba=(0.12, 0.12, 0.13, 1),
          mass=0.8)
    ls = mt.add_site()
    ls.name = "lidar"
    ls.pos = [0, 0, 0.105]
    pan = mt.add_body()
    pan.name = "head_pan"
    pan.pos = [0.0, 0.0, 0.0]
    _hinge(pan, "head_pan", (0, 0, 1), C["head_pan_range"], damping=0.5)
    _geom(pan, "pan_yoke", mujoco.mjtGeom.mjGEOM_BOX, (0.035, 0.03, 0.02), pos=(0.03, 0, -0.035), rgba=DARK_RGBA,
          collide=False, mass=0.3)
    tilt = pan.add_body()
    tilt.name = "head"
    tilt.pos = [0.06, 0.0, 0.0]
    _hinge(tilt, "head_tilt", (0, -1, 0), C["head_tilt_range"], damping=0.5)
    _geom(tilt, "head_box", mujoco.mjtGeom.mjGEOM_BOX, (0.04, 0.075, 0.035), pos=(0.0, 0, 0), rgba=DARK_RGBA,
          mass=C["mass"]["head"])
    for nm, y in (("lamp_l", 0.065), ("lamp_r", -0.065)):
        _geom(tilt, nm, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.014, 0.006), pos=(0.041, y, 0.012),
              quat=_q_axis((0, 1, 0), math.pi / 2), rgba=(1.0, 0.98, 0.85, 1), collide=False, mass=0.01)
    _geom(tilt, "lens_rgbd", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.012, 0.004), pos=(0.041, 0.02, -0.01),
          quat=_q_axis((0, 1, 0), math.pi / 2), rgba=(0.02, 0.02, 0.05, 1), collide=False, mass=0.01)
    _geom(tilt, "lens_thermal", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.010, 0.004), pos=(0.041, -0.025, -0.01),
          quat=_q_axis((0, 1, 0), math.pi / 2), rgba=(0.05, 0.03, 0.01, 1), collide=False, mass=0.01)
    # cameras look along their -z with +y up: the head looks along +x
    q_cam = (0.5, 0.5, -0.5, -0.5)
    for cname, y in (("mast_rgbd", 0.02), ("thermal", -0.025)):
        cam = tilt.add_camera()
        cam.name = cname
        cam.pos = [0.046, y, -0.01]
        cam.quat = list(q_cam)
        cam.fovy = 55.0 if cname == "mast_rgbd" else 42.0
    for nm, y in (("lamp_l", 0.065), ("lamp_r", -0.065)):
        lt = tilt.add_light()
        lt.name = nm
        lt.pos = [0.05, y, 0.012]
        lt.dir = [1.0, 0.0, -0.05]
        lt.castshadow = False
        lt.diffuse = [0.55, 0.53, 0.48]
        lt.specular = [0.1, 0.1, 0.1]
        lt.ambient = [0.0, 0.0, 0.0]
        lt.cutoff = 40.0
        lt.exponent = 6.0
        lt.attenuation = [1.0, 0.0, 0.08]
        lt.active = True
    # fixed driving cameras on the spine
    for cname, x, qz in (("front", sx + 0.005, 0.0), ("rear", -sx - 0.035, math.pi)):
        cam = ch.add_camera()
        cam.name = cname
        cam.pos = [x, 0.0, sz - 0.01]
        # look along +x (front) / -x (rear), pitched 12 deg down
        qa = _q_axis((0, 0, 1), qz)
        qb = _q_axis((0, 1, 0), math.radians(12.0))
        q = _qmul(qa, _qmul(qb, q_cam))
        cam.quat = q
        cam.fovy = 80.0
    # ---- the arm (front)
    b0, l1, l2, l3, l4 = C["arm_links"]
    rng = C["arm_range"]
    a0 = ch.add_body()
    a0.name = "arm_base"
    a0.pos = [C["arm_x"], 0.0, sz]
    _geom(a0, "arm_plinth", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.055, 0.015), pos=(0, 0, 0.015), rgba=DARK_RGBA,
          collide=False, mass=0.5)
    t1 = a0.add_body()
    t1.name = "arm_turret"
    t1.pos = [0, 0, 0.03]
    t1.gravcomp = 1.0
    _hinge(t1, "arm_1", (0, 0, 1), rng[0], damping=2.0, armature=0.05)
    _geom(t1, "turret", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.05, 0.03), pos=(0, 0, 0.025), rgba=BODY_RGBA,
          mass=C["mass"]["arm"][0])
    u = t1.add_body()
    u.name = "arm_upper"
    u.pos = [0, 0, b0]
    u.gravcomp = 1.0
    # joint 2 about -y: 0 = upper arm pointing forward (+x), +pi/2 = straight up, pi = backward
    _hinge(u, "arm_2", (0, -1, 0), rng[1], damping=2.0, armature=0.05)
    _geom(u, "upper", mujoco.mjtGeom.mjGEOM_CAPSULE, (0.032, l1 / 2.0), pos=(l1 / 2.0, 0, 0),
          quat=_q_axis((0, 1, 0), math.pi / 2), rgba=BODY_RGBA, mass=C["mass"]["arm"][1])
    f = u.add_body()
    f.name = "arm_fore"
    f.pos = [l1, 0, 0]
    f.gravcomp = 1.0
    _hinge(f, "arm_3", (0, -1, 0), rng[2], damping=1.5, armature=0.04)
    _geom(f, "fore", mujoco.mjtGeom.mjGEOM_CAPSULE, (0.027, l2 / 2.0), pos=(l2 / 2.0, 0, 0),
          quat=_q_axis((0, 1, 0), math.pi / 2), rgba=BODY_RGBA, mass=C["mass"]["arm"][2])
    w = f.add_body()
    w.name = "arm_wrist"
    w.pos = [l2, 0, 0]
    w.gravcomp = 1.0
    _hinge(w, "arm_4", (0, -1, 0), rng[3], damping=1.0, armature=0.02)
    _geom(w, "wrist", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.03, 0.03), pos=(l3 / 2.0, 0, 0),
          quat=_q_axis((0, 1, 0), math.pi / 2), rgba=DARK_RGBA, mass=C["mass"]["arm"][3])
    h = w.add_body()
    h.name = "gripper"
    h.pos = [l3, 0, 0]
    h.gravcomp = 1.0
    _hinge(h, "arm_5", (1, 0, 0), rng[4], damping=0.5, armature=0.01)
    _geom(h, "palm", mujoco.mjtGeom.mjGEOM_BOX, (0.025, 0.06, 0.025), pos=(0.025, 0, 0), rgba=DARK_RGBA,
          mass=C["mass"]["arm"][4])
    tcp = h.add_site()
    tcp.name = "tcp"
    tcp.pos = [l4, 0, 0]
    gc = h.add_camera()
    gc.name = "gripper"
    gc.pos = [0.03, 0.0, 0.045]
    gc.quat = _qmul(_q_axis((0, 1, 0), math.radians(10.0)), list(q_cam))
    gc.fovy = 70.0
    for fs, sgn in (("fl", 1.0), ("fr", -1.0)):
        fb = h.add_body()
        fb.name = "finger_%s" % fs
        fb.pos = [0.05, 0.0, 0.0]
        fb.gravcomp = 1.0
        _slide(fb, "finger_%s" % fs, (0, sgn, 0), (0.0, C["grip_open_max"] / 2.0), damping=5.0, armature=0.01)
        _geom(fb, "pad_%s" % fs, mujoco.mjtGeom.mjGEOM_BOX, (0.055, 0.007, 0.018), pos=(0.06, sgn * 0.007, 0),
              rgba=METAL_RGBA, mass=0.1, friction=(1.2, 0.02, 0.001))
    # ---- survival kit cradle (rear bumper, visual)
    kx, ky, kz = C["kit_cradle"]
    _geom(ch, "cradle", mujoco.mjtGeom.mjGEOM_BOX, (0.09, 0.05, 0.006), pos=(kx, ky, kz + 0.003),
          rgba=DARK_RGBA, collide=False, mass=0.2)
    for nm, dx in (("cradle_f", 0.09), ("cradle_b", -0.09)):
        _geom(ch, nm, mujoco.mjtGeom.mjGEOM_BOX, (0.004, 0.05, 0.02), pos=(kx + dx, ky, kz + 0.02),
              rgba=DARK_RGBA, collide=False, mass=0.05)
    # ---- actuators
    # the gauge's lead screws and the flippers' worm drives are not back-drivable: stiff servos, stiff limits
    _pos_act(spec, "gauge_L", "gauge_L", kp=2.0e5, kv=1.5e4, frange=2.0e4)
    _pos_act(spec, "gauge_R", "gauge_R", kp=2.0e5, kv=1.5e4, frange=2.0e4)
    for end in "FB":
        for side in "LR":
            _pos_act(spec, "flip_%s%s" % (end, side), "flip_%s%s" % (end, side), kp=6000.0, kv=250.0, frange=600.0)
    _pos_act(spec, "mast", "mast", kp=4000.0, kv=300.0, frange=400.0)
    _pos_act(spec, "head_pan", "head_pan", kp=40.0, kv=2.0, frange=10.0)
    _pos_act(spec, "head_tilt", "head_tilt", kp=40.0, kv=2.0, frange=10.0)
    # arm: rated torques (a payload of ~5 kg at full reach stalls the shoulder)
    for i, (kp, kv, fr) in enumerate(((600, 40, 80), (1500, 90, 120), (900, 50, 70), (300, 15, 30), (100, 5, 12)), 1):
        _pos_act(spec, "arm_%d" % i, "arm_%d" % i, kp=kp, kv=kv, frange=fr)
    _pos_act(spec, "finger_fl", "finger_fl", kp=2500.0, kv=60.0, frange=150.0)
    _pos_act(spec, "finger_fr", "finger_fr", kp=2500.0, kv=60.0, frange=150.0)
    return names


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


# ---------------------------------------------------------------------------------------------------- arm kinematics
def arm_fk(q):
    """TCP position (chassis frame) and approach direction for arm joints q[0:5]; the base at (arm_x, 0, spine top)."""
    C = CRAWLER
    b0, l1, l2, l3, l4 = C["arm_links"]
    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    base = np.array([C["arm_x"], 0.0, C["spine_half"][2] + 0.03 + b0])
    # planar chain in the turret's vertical plane: angles from the horizontal, measured upward
    a2 = q2
    a3 = q2 + q3
    a4 = q2 + q3 + q4
    r = l1 * math.cos(a2) + l2 * math.cos(a3) + (l3 + l4) * math.cos(a4)
    z = l1 * math.sin(a2) + l2 * math.sin(a3) + (l3 + l4) * math.sin(a4)
    p = base + np.array([r * math.cos(q1), r * math.sin(q1), z])
    d = np.array([math.cos(a4) * math.cos(q1), math.cos(a4) * math.sin(q1), math.sin(a4)])
    return p, d


def arm_ik(p, pitch, q_ref=None, elbow_up=True):
    """Joint angles (5, wrist roll from q_ref or 0) that put the TCP at chassis-frame point p with the tool axis pitched
    `pitch` from the horizontal (negative = pointing down). None when out of reach or out of the joint ranges."""
    C = CRAWLER
    b0, l1, l2, l3, l4 = C["arm_links"]
    base = np.array([C["arm_x"], 0.0, C["spine_half"][2] + 0.03 + b0])
    d = np.asarray(p, float) - base
    q1 = math.atan2(d[1], d[0])
    r = math.hypot(d[0], d[1])
    z = d[2]
    # wrist centre
    rw = r - (l3 + l4) * math.cos(pitch)
    zw = z - (l3 + l4) * math.sin(pitch)
    D = (rw * rw + zw * zw - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(D) > 1.0:
        return None
    q3 = math.acos(D) * (-1.0 if elbow_up else 1.0)
    q2 = math.atan2(zw, rw) - math.atan2(l2 * math.sin(q3), l1 + l2 * math.cos(q3))
    q4 = pitch - q2 - q3
    q5 = 0.0 if q_ref is None else float(q_ref[4])
    q = [q1, q2, q3, q4, q5]
    for i, (lo, hi) in enumerate(C["arm_range"][:4]):
        v = q[i]
        # wrap into range if possible
        for k in (0, 1, -1):
            vv = v + k * 2 * math.pi
            if lo <= vv <= hi:
                q[i] = vv
                break
        else:
            return None
    return q


# ---------------------------------------------------------------------------------------------------- controller
class Crawler(object):
    """The on-board controller: belt speeds from (v, yaw rate), rate-limited servo targets, the grip rule's inputs.
    It reads and writes the physics model through the indices resolved at construction."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        C = CRAWLER
        n2id = lambda t, n: mujoco.mj_name2id(model, t, n)
        G = mujoco.mjtObj.mjOBJ_GEOM
        self.body = n2id(mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.qadr = model.jnt_qposadr[n2id(mujoco.mjtObj.mjOBJ_JOINT, "chassis_free")]
        self.vadr = model.jnt_dofadr[n2id(mujoco.mjtObj.mjOBJ_JOINT, "chassis_free")]
        # belt geoms of each side with their sign (the rear flippers' frames are turned by pi)
        self.belts = {"L": [], "R": []}
        for gi in range(model.ngeom):
            nm = model.geom(gi).name
            for side in "LR":
                if nm.startswith("belt_%s_" % side):
                    self.belts[side].append((gi, 1.0))
                elif nm.startswith("fl_F%s_" % side) and not nm.endswith(("_hub", "_arm")):
                    self.belts[side].append((gi, 1.0))
                elif nm.startswith("fl_B%s_" % side) and not nm.endswith(("_hub", "_arm")):
                    self.belts[side].append((gi, -1.0))
        self.act = {model.actuator(i).name: i for i in range(model.nu)}
        self.jq = {}
        for jn in ("gauge_L", "gauge_R", "flip_FL", "flip_FR", "flip_BL", "flip_BR", "mast", "head_pan", "head_tilt",
                   "arm_1", "arm_2", "arm_3", "arm_4", "arm_5", "finger_fl", "finger_fr"):
            j = n2id(mujoco.mjtObj.mjOBJ_JOINT, jn)
            self.jq[jn] = (model.jnt_qposadr[j], model.jnt_dofadr[j])
        self.r = C["track_wheel_r"]
        # commands (set by the API) and the servo targets that follow them at the rated speeds
        self.cmd = {"v": 0.0, "w": 0.0, "gauge": C["gauge_min"], "flip_F": 0.0, "flip_B": 0.0, "mast": 0.0,
                    "pan": 0.0, "tilt": 0.0, "arm": list(ARM_STOW),
                    "grip": 0.08}
        self.tgt = dict((k, (list(v) if isinstance(v, list) else v)) for k, v in self.cmd.items())
        self.belt = {"L": 0.0, "R": 0.0}
        self.belt_acc = 1.5          # m/s^2

    # -- state helpers
    def q(self, jn):
        return float(self.d.qpos[self.jq[jn][0]])

    def gauge(self):
        return CRAWLER["gauge_min"] + self.q("gauge_L") + self.q("gauge_R")

    def arm_q(self):
        return [self.q("arm_%d" % i) for i in range(1, 6)]

    def grip_width(self):
        return self.q("finger_fl") + self.q("finger_fr")

    def set_initial(self, data=None):
        """Put every joint at its commanded value (used at reset before settling)."""
        d = self.d if data is None else data
        C = CRAWLER
        half = (self.cmd["gauge"] - C["gauge_min"]) / 2.0
        vals = {"gauge_L": half, "gauge_R": half, "flip_FL": self.cmd["flip_F"], "flip_FR": self.cmd["flip_F"],
                "flip_BL": self.cmd["flip_B"], "flip_BR": self.cmd["flip_B"], "mast": self.cmd["mast"],
                "head_pan": self.cmd["pan"], "head_tilt": self.cmd["tilt"], "finger_fl": self.cmd["grip"] / 2.0,
                "finger_fr": self.cmd["grip"] / 2.0}
        for i in range(5):
            vals["arm_%d" % (i + 1)] = self.cmd["arm"][i]
        for jn, v in vals.items():
            d.qpos[self.jq[jn][0]] = v
        self.tgt = dict((k, (list(v) if isinstance(v, list) else v)) for k, v in self.cmd.items())
        self._write_ctrl()

    # -- the servo loop, once per physics tick
    def tick(self, dt):
        C = CRAWLER
        c, t = self.cmd, self.tgt

        def follow(key, rate):
            t[key] = t[key] + max(-rate * dt, min(rate * dt, c[key] - t[key]))

        follow("gauge", C["gauge_rate"])
        follow("flip_F", C["flipper_rate"])
        follow("flip_B", C["flipper_rate"])
        follow("mast", C["mast_rate"])
        follow("pan", C["head_rate"])
        follow("tilt", C["head_rate"])
        follow("grip", C["grip_rate"])
        for i in range(5):
            t["arm"][i] = t["arm"][i] + max(-C["arm_rate"] * dt, min(C["arm_rate"] * dt, c["arm"][i] - t["arm"][i]))
        # belts: the side speeds of (v, w) on the CURRENT gauge, ramped
        g = self.gauge()
        for side, sgn in (("L", -1.0), ("R", 1.0)):
            want = c["v"] + sgn * c["w"] * g / 2.0
            want = max(-C["speed_max"], min(C["speed_max"], want))
            b = self.belt[side]
            self.belt[side] = b + max(-self.belt_acc * dt, min(self.belt_acc * dt, want - b))
        self._write_ctrl()

    def _write_ctrl(self):
        m, d, t = self.m, self.d, self.tgt
        C = CRAWLER
        half = (t["gauge"] - C["gauge_min"]) / 2.0
        d.ctrl[self.act["gauge_L"]] = half
        d.ctrl[self.act["gauge_R"]] = half
        for side in "LR":
            d.ctrl[self.act["flip_F%s" % side]] = t["flip_F"]
            d.ctrl[self.act["flip_B%s" % side]] = t["flip_B"]
        d.ctrl[self.act["mast"]] = t["mast"]
        d.ctrl[self.act["head_pan"]] = t["pan"]
        d.ctrl[self.act["head_tilt"]] = t["tilt"]
        for i in range(5):
            d.ctrl[self.act["arm_%d" % (i + 1)]] = t["arm"][i]
        d.ctrl[self.act["finger_fl"]] = t["grip"] / 2.0
        d.ctrl[self.act["finger_fr"]] = t["grip"] / 2.0
        # belts: every belt geom's local x is its travel direction when driving forward (belt_loop)
        for side in "LR":
            v = self.belt[side]
            for gi, sg in self.belts[side]:
                m.geom_surfacevel[gi, 0] = sg * v

    # -- pose
    def pose(self):
        q = self.d.qpos[self.qadr:self.qadr + 7]
        return np.array(q[:3]), np.array(q[3:7])
