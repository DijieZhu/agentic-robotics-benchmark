"""carm: the robot's visual model (Franka Panda from the robosuite 1.5.2 package, closed fingers, carrying the
fabric clamp), posed by the per-frame joint angles in clipNN/arm_q.npy. Useful e.g. to render robot masks with the
cameras of data/cameras.json. The arm never touches anything; only the clamp's jaws hold the fabric (cphys).

The clamp frame (cphys) is x: into the fabric, y: along the jaws, z: up. The tool approaches it from above at
TOOL_TILT, and the clamp centre is TOOL_TIP metres beyond the flange-side `eef` frame along the tool axis.
"""
import importlib.util
import json
import os
import re

import mujoco
import numpy as np

RS = os.path.join(importlib.util.find_spec("robosuite").submodule_search_locations[0], "models", "assets")
ROBOT_BASE = np.array([0.30, -0.62, 0.0])
TOOL_TILT = np.radians(50)
TOOL_TIP = 0.05
ARM_JOINTS = ["r_joint%d" % i for i in range(1, 8)]
Q_HOME = np.array([0.0, 0.2, 0.0, -2.2, 0.0, 2.4, 0.785])

_s, _c = np.sin(TOOL_TILT), np.cos(TOOL_TILT)
# eef axes expressed in the clamp frame (columns): z_eef points forward-down; the clamp is mounted a quarter turn
# about the tool axis, which keeps the arm's last joint inside its range over the whole working area
R_CLAMP_EEF = np.array([[-_s, 0.0, _c], [0.0, 1.0, 0.0], [-_c, 0.0, -_s]]) @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _abs_files(xml, base):
    return re.sub(r'file="([^"]+)"', lambda m: 'file="%s"' % (m.group(1) if os.path.isabs(m.group(1))
                                                              else os.path.join(base, m.group(1))), xml)


def arm_xml_parts(prefix="r_"):
    """(asset_xml, body_xml) for the Panda arm + closed gripper + fabric clamp, base at ROBOT_BASE."""
    rx = _abs_files(open(os.path.join(RS, "robots/panda/robot.xml")).read(), os.path.join(RS, "robots/panda"))
    gx = _abs_files(open(os.path.join(RS, "grippers/panda_gripper.xml")).read(), os.path.join(RS, "grippers"))
    ra = re.search(r"<asset>(.*?)</asset>", rx, re.S).group(1)
    ga = re.search(r"<asset>(.*?)</asset>", gx, re.S).group(1)
    rb = re.search(r"<worldbody>(.*)</worldbody>", rx, re.S).group(1)
    gb = re.search(r"<worldbody>(.*)</worldbody>", gx, re.S).group(1)

    def clean(s):
        s = re.sub(r'<geom[^>]*group="0"[^>]*/>', "", s)
        s = re.sub(r"<camera[^>]*/>", "", s)
        s = re.sub(r"<site[^>]*/>", "", s)
        return s
    rb, gb = clean(rb), clean(gb)
    gb = re.sub(r"<joint[^>]*/>", "", gb)
    # the clamp: a bracket along the tool axis ending in two horizontal jaws (in the clamp frame) at TOOL_TIP
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_CLAMP_EEF.T.copy().ravel())       # clamp frame expressed in the eef frame
    clamp = ('<body name="%sclamp_vis" pos="0 0 %g" quat="%.9g %.9g %.9g %.9g">'
             '<geom type="box" size="0.012 0.052 0.0025" pos="0 0 0.006" rgba="0.2 0.2 0.22 1" contype="0" conaffinity="0" group="1"/>'
             '<geom type="box" size="0.012 0.052 0.0025" pos="0 0 -0.006" rgba="0.2 0.2 0.22 1" contype="0" conaffinity="0" group="1"/>'
             '<geom type="box" size="0.006 0.052 0.009" pos="-0.016 0 0" rgba="0.25 0.25 0.27 1" contype="0" conaffinity="0" group="1"/>'
             '</body>'
             '<geom type="cylinder" size="0.008 %g" pos="0 0 %g" rgba="0.3 0.3 0.32 1" contype="0" conaffinity="0" group="1"/>'
             ) % (prefix, TOOL_TIP, *q, TOOL_TIP / 2 - 0.01, TOOL_TIP / 2 - 0.01)
    gb = gb.replace('<body name="eef" pos="0 0 0.097" quat="1 0 0 0">',
                    '<body name="eef" pos="0 0 0.097" quat="1 0 0 0">' + clamp + '<site name="eef_site" size="0.002"/>', 1)
    rb = rb.replace("<!-- to add gripper -->", gb)

    def pre(s):
        s = re.sub(r'name="([^"]+)"', lambda m: 'name="%s%s"' % ("" if m.group(1).startswith(prefix) else prefix, m.group(1)), s)
        s = re.sub(r'(mesh|material|joint)="([^"]+)"',
                   lambda m: '%s="%s%s"' % (m.group(1), "" if m.group(2).startswith(prefix) else prefix, m.group(2)), s)
        return s
    body = pre(rb).replace('<body name="%sbase" pos="0 0 0">' % prefix,
                           '<body name="%sbase" pos="%g %g %g" quat="0.7071068 0 0 0.7071068">' % (prefix, *ROBOT_BASE), 1)
    return pre(ra + ga), body


def cameras_xml(cameras_json):
    """MuJoCo cameras from cameras.json (OpenCV pinhole: pixel ~ K (R_w2c X + t_w2c))."""
    out = ""
    for c in json.load(open(cameras_json))["cameras"]:
        R_w2c = np.array(c["R_w2c"])
        pos = -R_w2c.T @ np.array(c["t_w2c"])
        R_mj = R_w2c.T @ np.diag([1, -1, -1])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_mj.ravel())
        fovy = np.degrees(2 * np.arctan(0.5 * c["height"] / c["K"][1][1]))
        out += '<camera name="%s" pos="%.9g %.9g %.9g" quat="%.9g %.9g %.9g %.9g" fovy="%.9g"/>' % (c["name"], *pos, *q, fovy)
    return out


def arm_model(cameras_json=None):
    """MjModel of the arm alone (+ the cameras of cameras.json)."""
    a, b = arm_xml_parts()
    cams = cameras_xml(cameras_json) if cameras_json else ""
    xml = ('<mujoco><compiler angle="radian"/><visual><global offwidth="1280" offheight="960"/></visual>'
           '<asset>%s</asset><worldbody>%s%s</worldbody></mujoco>' % (a, cams, b))
    return mujoco.MjModel.from_xml_string(xml)


def set_arm(m, d, q):
    for i, n in enumerate(ARM_JOINTS):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = q[i]
    mujoco.mj_kinematics(m, d)


def eef_target(clamp_row):
    """Flange-side eef position and rotation that put the clamp centre at the commanded clamp pose."""
    R_c = np.zeros(9)
    mujoco.mju_quat2Mat(R_c, np.asarray(clamp_row[3:7], float))
    R_e = R_c.reshape(3, 3) @ R_CLAMP_EEF
    return np.asarray(clamp_row[:3], float) - R_e[:, 2] * TOOL_TIP, R_e


def first_pose(m, d, clamp_row, starts=6, iters=300):
    """Joint angles for one clamp pose: the best of several starts (the arm has more than one way to reach a pose)."""
    sid = m.site("r_eef_site").id
    best = None
    p_t, R_t = eef_target(clamp_row)
    for s in range(starts):
        q0 = Q_HOME + np.array([0.9 * (s - (starts - 1) / 2), 0, 0, 0, 0, 0, 0.8 * ((s % 3) - 1)])
        q = solve_ik(m, d, [clamp_row], q0=q0, iters=iters)[0]
        set_arm(m, d, q)
        R = d.site_xmat[sid].reshape(3, 3)
        e = float(np.linalg.norm(d.site_xpos[sid] - p_t)) + 0.5 * float(np.arccos(np.clip((np.trace(R.T @ R_t) - 1) / 2, -1, 1)))
        if best is None or e < best[0]:
            best = (e, q)
    return best[1]


def solve_ik(m, d, clamp_rows, q0=Q_HOME, iters=60, damping=1e-3):
    """Joint angles (N, 7) following clamp poses (N, 8 rows) by damped least squares, warm-started frame to frame."""
    sid = m.site("r_eef_site").id
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
    dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
    lo = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 0]
    hi = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 1]
    q = np.array(q0, float)
    out = []
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for row in clamp_rows:
        p_t, R_t = eef_target(row)
        for _ in range(iters):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[sid]
            R = d.site_xmat[sid].reshape(3, 3)
            ep = p_t - p
            er = 0.5 * (np.cross(R[:, 0], R_t[:, 0]) + np.cross(R[:, 1], R_t[:, 1]) + np.cross(R[:, 2], R_t[:, 2]))
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            J = np.vstack([jp[:, dadr], jr[:, dadr]])
            e = np.r_[ep, er]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            dq += 0.05 * (Q_HOME - q) - 0.05 * J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), J @ (Q_HOME - q))
            q = np.clip(q + dq, lo, hi)
        out.append(q.copy())
    return np.array(out)
