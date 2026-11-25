"""The Robotiq 2F-85 on the UR flange (`ur_robot.add_ur`): opening table, force model, grasp-planner geometry.

Everything here was measured on the attached model (UR10e + the patched given 2F-85, coupler locked, pad condim 4 /
solref 0.002, tendon force range +-9, cone elliptic, impratio 100, 1 ms implicitfast, noslip 5): see
OPENING_TABLE_SOURCE and FORCE_MODEL["source"].

Geometry (the given 2f85.xml): each finger carries two pad boxes (22 x 8 x 18.75 mm) end to end, so a pad face is
22 mm across (tcp x) by 37.5 mm along the finger (tcp z); the inner face lies at y = -6.6 mm of the pad body, its
middle at z = 18.75 mm. The tool frame is the `tcp` site (the pinch point, 155.8 mm from the flange face): z =
approach, y = closing axis (the right pad on +y), x across the pads. The fingers swing on the 4-bar arc while
closing: the pad centre moves 13 mm along the approach between open and closed, and the faces converge 0-1.8 deg
towards the tips (the pads are parallel only to that).

* `width` (the API's gripper width) = the gap between the two inner pad faces at the pads' middle, along the closing
  axis (`PadGeometry.gap`); 0.0009-0.0851 m over ctrl 255-0 (measured: OPENING_TABLE). Along the approach the pads'
  face centre sits 14.2 mm short of the tcp when fully open, 3.6 mm at 45 mm, 2.5 mm at 37 mm, 1.1-1.2 mm below
  20 mm (`pad_centre_z`); the fingertips reach 4.6 mm (open) to 17.9 mm beyond the tcp (`fingertip_z`: 160.4-173.7 mm
  from the flange); the fingers' outer envelope along the closing axis (the 4-bar's drivers and couplers, not the
  pads) is +-62 mm closed to +-74 mm open (`finger_outer_y`): a 124-148 mm wide hand that a grasp next to a wall or a
  built unit must clear.
* The controller maps a commanded width to the servo ctrl through the free-air opening table (`ctrl_of_width`): with
  nothing between the pads the fingers settle at the commanded width.
* Force: the 2F-85's tendon servo pulls 100 N per rad of driver-angle error (kp 100), capped at the force range 9;
  the linkage turns that into a pad normal force of 8.7-9.0 N per N of tendon force for 10-45 mm parts (10.6 at
  75 mm). Closing on a part, the pad force grows with the SQUEEZE (the part's width minus the commanded width):
  5.7-6.3 N/mm for 10-45 mm parts, capped at 77-81 N (86 N at 60 mm, 95 N at 75 mm), reached ~12 mm below the
  part's width (`FORCE_MODEL`, `pad_force`, `squeeze_for_force`; the text for the API docs: FORCE_MODEL_TEXT).
"""
import math

import numpy as np

__all__ = ["PAD_ACROSS_M", "PAD_ALONG_M", "PAD_FACE_Y", "PAD_MID_Z", "OPENING_TABLE", "OPENING_COLUMNS",
           "FORCE_MODEL", "FORCE_MODEL_TEXT", "PadGeometry", "ctrl_of_width", "width_of_ctrl", "driver_of_width",
           "pad_centre_z", "pad_tip_z", "fingertip_z", "finger_outer_y", "linkage_qpos", "set_opening",
           "pad_force", "squeeze_for_force", "measure_opening_table", "MAX_WIDTH_M"]

PAD_ACROSS_M = 0.022          # pad face across the finger (tcp x): 2 x 0.011 (pad box size)
PAD_ALONG_M = 0.0375          # pad face along the finger (tcp z): 2 boxes x 2 x 0.009375
PAD_FACE_Y = -0.0066          # inner face in the pad body frame: box centre y -0.0026, half thickness 0.004
PAD_MID_Z = 0.01875           # middle of the pad face along the pad body's z (towards the fingertip)
MAX_WIDTH_M = 0.085           # the API's width range 0..0.085 m (the table reaches 0.0851 at ctrl 0)
LINKAGE = ["right_driver_joint", "right_spring_link_joint", "right_follower_joint", "left_driver_joint",
           "left_spring_link_joint", "left_follower_joint"]

# columns of OPENING_TABLE; lengths in m in the tcp frame (z along the approach, from the tcp; negative = towards the
# flange), angles in rad / deg as named. pad_tip_z: the pad face's tip end; fingertip_z: the farthest point of the
# finger's collision geoms (follower + pads) along the approach; finger_outer_y: the largest |y| (closing axis) of
# any finger collision geom (the fingers' outer envelope at that opening).
OPENING_COLUMNS = ["ctrl", "driver_rad", "gap_mid_m", "gap_root_m", "gap_tip_m", "pad_centre_z_m", "pad_tip_z_m",
                   "fingertip_z_m", "finger_outer_y_m", "tilt_deg"] + ["q_" + j for j in LINKAGE]
# OPENING_TABLE, OPENING_TABLE_SOURCE, FORCE_MODEL and FORCE_MODEL_TEXT: the measured data at the end of this file


class PadGeometry(object):
    """The pads of one attached 2F-85: the gap, the faces, the pad-midpoint frame, the pad forces on a geom set."""

    def __init__(self, model, prefix="ur/", gripper_prefix=None):
        G = gripper_prefix or (prefix + "g/")
        self.m = model
        self.rp = model.body(G + "right_pad").id
        self.lp = model.body(G + "left_pad").id
        self.tcp = model.site(prefix + "tcp").id
        self.pad_geoms = {"right": [model.geom(G + n).id for n in ("right_pad1", "right_pad2")],
                          "left": [model.geom(G + n).id for n in ("left_pad1", "left_pad2")]}
        self.finger_bodies = {s: [model.body(G + s + "_" + n).id for n in ("driver", "coupler", "spring_link",
                                                                           "follower", "pad")]
                              for s in ("right", "left")}
        self.linkage_qadr = [int(model.jnt_qposadr[model.joint(G + j).id]) for j in LINKAGE]
        self.driver_qadr = [self.linkage_qadr[0], self.linkage_qadr[3]]

    def face_point(self, data, side, z):
        b = self.rp if side == "right" else self.lp
        return data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array([0.0, PAD_FACE_Y, z])

    def gap(self, data, z=PAD_MID_Z):
        """inner-face gap along the closing axis (tcp y) at pad height z (default: the pads' middle)"""
        y = data.site_xmat[self.tcp].reshape(3, 3)[:, 1]
        return float((self.face_point(data, "right", z) - self.face_point(data, "left", z)) @ y)

    def pad_frame(self, data):
        """(origin, R): origin midway between the two face centres, y from the left to the right face, z along the
        pads' mean length axis (towards the tips): the frame a held part's slip is measured in"""
        cr, cl = self.face_point(data, "right", PAD_MID_Z), self.face_point(data, "left", PAD_MID_Z)
        o = 0.5 * (cr + cl)
        y = cr - cl
        y /= np.linalg.norm(y)
        z = data.xmat[self.rp].reshape(3, 3)[:, 2] + data.xmat[self.lp].reshape(3, 3)[:, 2]
        z -= (z @ y) * y
        z /= np.linalg.norm(z)
        return o, np.column_stack([np.cross(y, z), y, z])

    def tilt_deg(self, data):
        """mean signed tilt of the two pad faces from parallel to the tool axis, deg (+: converging towards the tips)"""
        root = self.gap(data, 0.0)
        tip = self.gap(data, 2 * PAD_MID_Z)
        return math.degrees(math.atan2(0.5 * (root - tip), 2 * PAD_MID_Z))

    def forces(self, data, other_geoms):
        """(right, left) normal force (N) of the pad geoms on any geom in `other_geoms`, and the contact count"""
        import mujoco
        other = set(int(g) for g in other_geoms)
        f6 = np.zeros(6)
        out = {"right": 0.0, "left": 0.0}
        n = 0
        rs, ls = set(self.pad_geoms["right"]), set(self.pad_geoms["left"])
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if b in other and (a in rs or a in ls):
                    mujoco.mj_contactForce(self.m, data, i, f6)
                    out["right" if a in rs else "left"] += float(f6[0])
                    n += 1
        return out["right"], out["left"], n


def _col(name):
    i = OPENING_COLUMNS.index(name)
    rows = sorted(OPENING_TABLE, key=lambda r: r[2])               # ascending gap
    return np.array([r[2] for r in rows]), np.array([r[i] for r in rows])


def _interp_by_width(width, name):
    w, v = _col(name)
    return float(np.interp(min(max(float(width), w[0]), w[-1]), w, v))


def ctrl_of_width(width):
    """servo ctrl (0-255) at which the free fingers settle at `width` (m); below the smallest gap: 255"""
    return _interp_by_width(width, "ctrl")


def width_of_ctrl(ctrl):
    c = np.array([r[0] for r in OPENING_TABLE])
    w = np.array([r[2] for r in OPENING_TABLE])
    o = np.argsort(c)
    return float(np.interp(float(ctrl), c[o], w[o]))


def driver_of_width(width):
    return _interp_by_width(width, "driver_rad")


def pad_centre_z(width):
    """the pads' face centre along the approach, from the tcp (m; negative = towards the flange)"""
    return _interp_by_width(width, "pad_centre_z_m")


def pad_tip_z(width):
    return _interp_by_width(width, "pad_tip_z_m")


def fingertip_z(width):
    """the fingers' farthest collision point along the approach, from the tcp (m)"""
    return _interp_by_width(width, "fingertip_z_m")


def finger_outer_y(width):
    """half the fingers' outer envelope along the closing axis (m)"""
    return _interp_by_width(width, "finger_outer_y_m")


def linkage_qpos(width):
    """{joint suffix: angle} of the six linkage joints at an opening (a planner copy has no solver to close the
    4-bar: set every joint from the table)"""
    return {j: _interp_by_width(width, "q_" + j) for j in LINKAGE}


def set_opening(model, data, width, prefix="ur/"):
    """pose the gripper of a model copy at an opening (kinematic: qpos only; call mj_kinematics after)"""
    G = prefix + "g/"
    for j, v in linkage_qpos(width).items():
        data.qpos[model.jnt_qposadr[model.joint(G + j).id]] = v


def _force_params(width_m):
    fm = FORCE_MODEL
    return [float(np.interp(width_m, fm["width_m"], fm[k])) for k in ("k_N_per_m", "deadband_m", "cap_N")]


def pad_force(squeeze_m, width_m=0.037):
    """steady normal force per pad (N) when the commanded width is `squeeze_m` below the part's width `width_m`
    (FORCE_MODEL: linear in the squeeze, capped). A command at or below the smallest free-air gap (0.9 mm: ctrl 255,
    the servo's target 0.8 rad lies past the closed fingers) always grips at the cap."""
    k, db, cap = _force_params(width_m)
    if width_m - squeeze_m <= _col("ctrl")[0][0] + 1e-6:
        return float(cap)
    return float(min(cap, max(0.0, k * (squeeze_m - db))))


def squeeze_for_force(force_N, width_m=0.037):
    """the squeeze (part width - commanded width, m) that gives `force_N` per pad; inf at or above the cap"""
    k, db, cap = _force_params(width_m)
    if force_N >= cap:
        return float("inf")
    return db + max(0.0, force_N) / k


def measure_opening_table(model, data, q_arm, arm_qadr, arm_vadr, prefix="ur/", ctrls=None, settle_s=1.5,
                          grip_act=None):
    """Measure the opening table on an attached model: the arm frozen at q_arm (qpos reset after every step; the
    fingers see a still base), the servo held at each ctrl (closing sweep) until it settles in free air. Returns
    rows in OPENING_COLUMNS order."""
    import mujoco
    pg = PadGeometry(model, prefix)
    G = prefix + "g/"
    act = model.actuator(grip_act or (G + "fingers_actuator")).id
    ctrls = list(np.linspace(0.0, 255.0, 52)) if ctrls is None else list(ctrls)
    arm_qadr, arm_vadr = np.asarray(arm_qadr, int), np.asarray(arm_vadr, int)
    q_arm = np.asarray(q_arm, float)
    fingers = []
    for s in ("right", "left"):
        for b in pg.finger_bodies[s]:
            fingers += [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_contype[g]]
    rows = []
    n = int(round(settle_s / model.opt.timestep))
    for c in ctrls:
        data.ctrl[act] = c
        for _ in range(n):
            data.qpos[arm_qadr] = q_arm
            data.qvel[arm_vadr] = 0.0
            mujoco.mj_step(model, data)
        data.qpos[arm_qadr] = q_arm
        data.qvel[arm_vadr] = 0.0
        mujoco.mj_forward(model, data)
        pt, Rt = data.site_xpos[pg.tcp].copy(), data.site_xmat[pg.tcp].reshape(3, 3).copy()
        zc = 0.5 * ((pg.face_point(data, "right", PAD_MID_Z) - pt) @ Rt[:, 2] +
                    (pg.face_point(data, "left", PAD_MID_Z) - pt) @ Rt[:, 2])
        ztip = 0.5 * ((pg.face_point(data, "right", 2 * PAD_MID_Z) - pt) @ Rt[:, 2] +
                      (pg.face_point(data, "left", 2 * PAD_MID_Z) - pt) @ Rt[:, 2])
        zmax, ymax = -1.0, 0.0
        for g in fingers:
            P = _geom_points(model, data, g)
            if P is None:
                continue
            loc = (P - pt) @ Rt
            zmax = max(zmax, float(loc[:, 2].max()))
            ymax = max(ymax, float(np.abs(loc[:, 1]).max()))
        drv = 0.5 * (data.qpos[pg.driver_qadr[0]] + data.qpos[pg.driver_qadr[1]])
        rows.append([float(c), float(drv), pg.gap(data), pg.gap(data, 0.0), pg.gap(data, 2 * PAD_MID_Z), float(zc),
                     float(ztip), zmax, ymax, pg.tilt_deg(data)] + [float(data.qpos[a]) for a in pg.linkage_qadr])
    return rows


def _geom_points(model, data, g):
    import mujoco
    p = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k] + model.mesh_vertnum[k]]
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        s = model.geom_size[g]
        v = np.array([[a * s[0], b * s[1], c * s[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    else:
        return None
    return p + v @ R.T


# ---------------------------------------------------------------------------------------------- measured data
OPENING_TABLE_SOURCE = ("2026-10-28, tools/robot_smoke.py 'table' (measure_opening_table): UR10e + 2F-85 of "
                        "ur_robot.add_ur at the origin of the empty cell, the arm frozen tool-down at the ready "
                        "posture, gravity on (along the approach), 1 ms implicitfast, noslip 5; ctrl 0..255 in 52 "
                        "steps, a closing sweep, each held 1.5 s in free air (settled to < 0.01 mm)")
# ctrl, driver_rad, gap_mid_m, gap_root_m, gap_tip_m, pad_centre_z_m, pad_tip_z_m, fingertip_z_m, finger_outer_y_m,
# tilt_deg, then the six linkage joints (right driver, spring link, follower, left driver, spring link, follower; rad)
OPENING_TABLE = [
    [0.0, 0.00293, 0.085132, 0.085132, 0.085131, -0.014218, 0.004532, 0.004533, 0.074089, 0.00,
     0.00293, 0.00304, -0.00304, 0.00293, 0.00305, -0.00304],
    [5.0, 0.01860, 0.083712, 0.083738, 0.083686, -0.013653, 0.005097, 0.005102, 0.073409, 0.04,
     0.01860, 0.01863, -0.01794, 0.01860, 0.01864, -0.01795],
    [10.0, 0.03427, 0.082275, 0.082326, 0.082224, -0.013100, 0.005650, 0.005661, 0.072719, 0.08,
     0.03427, 0.03423, -0.03286, 0.03427, 0.03423, -0.03286],
    [15.0, 0.04993, 0.080821, 0.080897, 0.080744, -0.012557, 0.006193, 0.006209, 0.072018, 0.12,
     0.04993, 0.04982, -0.04779, 0.04993, 0.04982, -0.04779],
    [20.0, 0.06560, 0.079349, 0.079451, 0.079248, -0.012026, 0.006724, 0.006746, 0.071307, 0.15,
     0.06560, 0.06542, -0.06272, 0.06560, 0.06542, -0.06273],
    [25.0, 0.08127, 0.077862, 0.077987, 0.077736, -0.011505, 0.007245, 0.007272, 0.070592, 0.19,
     0.08127, 0.08103, -0.07767, 0.08127, 0.08103, -0.07767],
    [30.0, 0.09693, 0.076357, 0.076508, 0.076207, -0.010996, 0.007754, 0.007786, 0.069872, 0.23,
     0.09693, 0.09663, -0.09262, 0.09693, 0.09664, -0.09263],
    [35.0, 0.11260, 0.074837, 0.075012, 0.074663, -0.010498, 0.008252, 0.008289, 0.069143, 0.27,
     0.11260, 0.11225, -0.10759, 0.11260, 0.11225, -0.10759],
    [40.0, 0.12827, 0.073302, 0.073501, 0.073103, -0.010011, 0.008738, 0.008781, 0.068403, 0.30,
     0.12827, 0.12786, -0.12256, 0.12827, 0.12786, -0.12256],
    [45.0, 0.14393, 0.071751, 0.071974, 0.071528, -0.009537, 0.009213, 0.009261, 0.068363, 0.34,
     0.14393, 0.14348, -0.13754, 0.14393, 0.14348, -0.13755],
    [50.0, 0.15960, 0.070185, 0.070432, 0.069939, -0.009074, 0.009676, 0.009729, 0.068350, 0.38,
     0.15960, 0.15910, -0.15253, 0.15960, 0.15910, -0.15253],
    [55.0, 0.17526, 0.068605, 0.068875, 0.068335, -0.008623, 0.010127, 0.010185, 0.068327, 0.41,
     0.17526, 0.17472, -0.16753, 0.17526, 0.17473, -0.16753],
    [60.0, 0.19093, 0.067011, 0.067304, 0.066717, -0.008184, 0.010566, 0.010628, 0.068299, 0.45,
     0.19093, 0.19035, -0.18253, 0.19093, 0.19035, -0.18253],
    [65.0, 0.20659, 0.065402, 0.065719, 0.065086, -0.007757, 0.010993, 0.011060, 0.068269, 0.48,
     0.20659, 0.20598, -0.19754, 0.20659, 0.20599, -0.19754],
    [70.0, 0.22226, 0.063781, 0.064120, 0.063441, -0.007342, 0.011407, 0.011480, 0.068229, 0.52,
     0.22226, 0.22162, -0.21256, 0.22226, 0.22162, -0.21256],
    [75.0, 0.23793, 0.062146, 0.062509, 0.061783, -0.006940, 0.011809, 0.011887, 0.068180, 0.55,
     0.23793, 0.23726, -0.22758, 0.23793, 0.23726, -0.22758],
    [80.0, 0.25359, 0.060498, 0.060884, 0.060112, -0.006550, 0.012199, 0.012281, 0.068122, 0.59,
     0.25359, 0.25290, -0.24261, 0.25359, 0.25290, -0.24261],
    [85.0, 0.26926, 0.058838, 0.059247, 0.058430, -0.006173, 0.012576, 0.012663, 0.068054, 0.62,
     0.26926, 0.26854, -0.25764, 0.26926, 0.26854, -0.25764],
    [90.0, 0.28492, 0.057166, 0.057598, 0.056735, -0.005809, 0.012940, 0.013032, 0.067978, 0.66,
     0.28492, 0.28419, -0.27268, 0.28492, 0.28419, -0.27268],
    [95.0, 0.30059, 0.055483, 0.055937, 0.055029, -0.005457, 0.013291, 0.013388, 0.067894, 0.69,
     0.30059, 0.29984, -0.28772, 0.30059, 0.29984, -0.28772],
    [100.0, 0.31625, 0.053788, 0.054265, 0.053311, -0.005119, 0.013630, 0.013732, 0.067810, 0.73,
     0.31625, 0.31549, -0.30277, 0.31625, 0.31549, -0.30277],
    [105.0, 0.33192, 0.052083, 0.052583, 0.051583, -0.004793, 0.013955, 0.014062, 0.067716, 0.76,
     0.33192, 0.33114, -0.31782, 0.33192, 0.33115, -0.31782],
    [110.0, 0.34758, 0.050367, 0.050889, 0.049845, -0.004480, 0.014268, 0.014379, 0.067614, 0.80,
     0.34758, 0.34680, -0.33287, 0.34758, 0.34681, -0.33287],
    [115.0, 0.36325, 0.048641, 0.049186, 0.048096, -0.004181, 0.014567, 0.014683, 0.067502, 0.83,
     0.36325, 0.36247, -0.34793, 0.36325, 0.36247, -0.34793],
    [120.0, 0.37891, 0.046906, 0.047473, 0.046338, -0.003895, 0.014853, 0.014974, 0.067381, 0.87,
     0.37891, 0.37813, -0.36299, 0.37891, 0.37813, -0.36299],
    [125.0, 0.39458, 0.045161, 0.045751, 0.044570, -0.003622, 0.015125, 0.015251, 0.067252, 0.90,
     0.39458, 0.39380, -0.37805, 0.39458, 0.39380, -0.37806],
    [130.0, 0.41024, 0.043407, 0.044020, 0.042794, -0.003363, 0.015385, 0.015515, 0.067113, 0.94,
     0.41024, 0.40947, -0.39312, 0.41024, 0.40947, -0.39312],
    [135.0, 0.42591, 0.041645, 0.042281, 0.041010, -0.003117, 0.015630, 0.015766, 0.066976, 0.97,
     0.42591, 0.42514, -0.40819, 0.42591, 0.42514, -0.40819],
    [140.0, 0.44157, 0.039875, 0.040534, 0.039217, -0.002885, 0.015862, 0.016003, 0.066830, 1.01,
     0.44157, 0.44082, -0.42326, 0.44157, 0.44082, -0.42326],
    [145.0, 0.45724, 0.038098, 0.038779, 0.037416, -0.002666, 0.016081, 0.016226, 0.066675, 1.04,
     0.45724, 0.45650, -0.43833, 0.45724, 0.45650, -0.43833],
    [150.0, 0.47290, 0.036313, 0.037017, 0.035609, -0.002461, 0.016286, 0.016436, 0.066511, 1.08,
     0.47290, 0.47218, -0.45341, 0.47290, 0.47218, -0.45341],
    [155.0, 0.48856, 0.034521, 0.035248, 0.033794, -0.002270, 0.016476, 0.016632, 0.066339, 1.11,
     0.48856, 0.48787, -0.46848, 0.48856, 0.48787, -0.46848],
    [160.0, 0.50423, 0.032723, 0.033473, 0.031973, -0.002093, 0.016654, 0.016814, 0.066157, 1.15,
     0.50423, 0.50356, -0.48356, 0.50423, 0.50356, -0.48356],
    [165.0, 0.51989, 0.030919, 0.031692, 0.030146, -0.001929, 0.016817, 0.016982, 0.065967, 1.18,
     0.51989, 0.51925, -0.49864, 0.51989, 0.51925, -0.49864],
    [170.0, 0.53556, 0.029109, 0.029905, 0.028313, -0.001780, 0.016966, 0.017136, 0.065778, 1.22,
     0.53556, 0.53494, -0.51371, 0.53556, 0.53495, -0.51371],
    [175.0, 0.55122, 0.027295, 0.028114, 0.026475, -0.001644, 0.017101, 0.017276, 0.065581, 1.25,
     0.55122, 0.55064, -0.52879, 0.55122, 0.55064, -0.52879],
    [180.0, 0.56689, 0.025475, 0.026318, 0.024632, -0.001522, 0.017223, 0.017403, 0.065375, 1.29,
     0.56689, 0.56634, -0.54387, 0.56689, 0.56635, -0.54387],
    [185.0, 0.58255, 0.023651, 0.024518, 0.022785, -0.001415, 0.017330, 0.017515, 0.065161, 1.32,
     0.58255, 0.58205, -0.55894, 0.58255, 0.58205, -0.55894],
    [190.0, 0.59821, 0.021824, 0.022714, 0.020934, -0.001322, 0.017423, 0.017613, 0.064939, 1.36,
     0.59821, 0.59776, -0.57402, 0.59821, 0.59776, -0.57402],
    [195.0, 0.61388, 0.019993, 0.020906, 0.019079, -0.001242, 0.017502, 0.017697, 0.064708, 1.40,
     0.61388, 0.61347, -0.58910, 0.61388, 0.61347, -0.58910],
    [200.0, 0.62954, 0.018158, 0.019096, 0.017220, -0.001177, 0.017567, 0.017767, 0.064468, 1.43,
     0.62954, 0.62918, -0.60417, 0.62954, 0.62919, -0.60417],
    [205.0, 0.64521, 0.016322, 0.017284, 0.015359, -0.001126, 0.017617, 0.017823, 0.064228, 1.47,
     0.64521, 0.64490, -0.61924, 0.64521, 0.64490, -0.61924],
    [210.0, 0.66087, 0.014482, 0.015469, 0.013496, -0.001090, 0.017654, 0.017864, 0.063983, 1.51,
     0.66087, 0.66063, -0.63431, 0.66087, 0.66063, -0.63431],
    [215.0, 0.67653, 0.012642, 0.013653, 0.011630, -0.001067, 0.017676, 0.017891, 0.063730, 1.54,
     0.67653, 0.67635, -0.64938, 0.67653, 0.67635, -0.64938],
    [220.0, 0.69220, 0.010799, 0.011835, 0.009763, -0.001059, 0.017683, 0.017904, 0.063468, 1.58,
     0.69220, 0.69208, -0.66445, 0.69220, 0.69208, -0.66445],
    [225.0, 0.70786, 0.008956, 0.010017, 0.007895, -0.001066, 0.017677, 0.017903, 0.063198, 1.62,
     0.70786, 0.70781, -0.67951, 0.70786, 0.70782, -0.67951],
    [230.0, 0.72353, 0.007112, 0.008199, 0.006026, -0.001086, 0.017656, 0.017888, 0.062921, 1.66,
     0.72353, 0.72355, -0.69457, 0.72353, 0.72355, -0.69457],
    [235.0, 0.73919, 0.005268, 0.006381, 0.004156, -0.001121, 0.017621, 0.017858, 0.062635, 1.70,
     0.73919, 0.73929, -0.70963, 0.73919, 0.73930, -0.70963],
    [240.0, 0.75485, 0.003425, 0.004563, 0.002286, -0.001170, 0.017571, 0.017814, 0.062347, 1.74,
     0.75485, 0.75504, -0.72468, 0.75485, 0.75504, -0.72468],
    [245.0, 0.77052, 0.001582, 0.002746, 0.000417, -0.001233, 0.017507, 0.017756, 0.062057, 1.78,
     0.77052, 0.77079, -0.73973, 0.77052, 0.77079, -0.73973],
    [250.0, 0.77719, 0.001050, 0.002104, -0.000004, -0.001235, 0.017508, 0.017733, 0.061930, 1.61,
     0.77719, 0.77675, -0.74863, 0.77719, 0.77674, -0.74865],
    [255.0, 0.78121, 0.000898, 0.001805, -0.000008, -0.001216, 0.017528, 0.017722, 0.061854, 1.38,
     0.78121, 0.77983, -0.75559, 0.78121, 0.77981, -0.75569],
]
FORCE_MODEL = {
    "source": ("2026-10-28, tools/robot_smoke.py 'table': a fixed 40 x width x 40 mm block centred on the pad "
               "faces (ready pose, tool down), the fingers opened to width + 10 mm, then commanded width - squeeze "
               "in increasing steps of 0.8 s (force = the pad geoms' normal force on the block, mean of the last "
               "0.2 s); the cap at commanded width 0; k fitted for 0.5 N < F < 0.8 cap"),
    "width_m": [0.010, 0.020, 0.030, 0.037, 0.045, 0.060, 0.075],
    "k_N_per_m": [5668.7, 5778.7, 5940.4, 6089.1, 6302.8, 7276.6, 9368.1],
    # negative: the force starts before the commanded width reaches the part (the faces converge 0.5-1.8 deg towards
    # the tips, so the tips touch first): F = k (squeeze - deadband)
    "deadband_m": [-0.000936, -0.000827, -0.000715, -0.000636, -0.000542, -0.000238, 0.000041],
    "cap_N": [76.7, 78.3, 78.8, 79.8, 81.4, 86.3, 95.0],
    "cap_over_tendon_force": [8.68, 8.69, 8.76, 8.87, 9.04, 9.59, 10.55],
    # largest per-pad force at any 1 ms step closing from 85 mm open onto fixed 20 / 37 / 45 mm blocks (command 0)
    "closing_peak_N": {"width_m": [0.020, 0.037, 0.045],
                       "0.06": [79.3, 80.4, 82.1], "0.08": [79.8, 82.2, 82.9], "0.10": [80.4, 88.7, 89.6],
                       "0.12": [81.9, 99.2, 108.1], "0.14": [83.6, 102.5, 113.4], "0.20": [98.3, 144.0, 166.3],
                       "ctrl step": [111.2, 190.4, 221.9]},
}
FORCE_MODEL_TEXT = (
    "Grip force: the fingers are position-controlled; closing on a part, each pad presses with about 6 N per mm "
    "that the commanded width is below the part's width (5.7 N/mm for 10 mm parts, 6.1 at 37 mm, 6.3 at 45 mm, "
    "9.4 at 75 mm; about 4 N already when the command equals the width, the pad tips touch first), up to a cap of "
    "about 80 N per pad (77-81 N for 10-45 mm parts, 86 N at 60 mm, 95 N at 75 mm), reached some 12 mm below the "
    "part's width; commanding width 0 always grips at the cap. The commanded width moves at 0.10 m/s (a full close "
    "takes 0.85 s); closing onto a part from fully open the contact peak is about 10 % above the steady grip "
    "(80-90 N). The pads are 22 mm across and 37.5 mm long, parallel to within 1.8 deg (pinch grasps only: the "
    "fingers do not wrap around a part).")
