"""The gripper's hold on a piece of debris or the survival kit: one weld equality retargeted at run time (cell
internals; the same machinery as the benchmark's arm cells' grasp module).

The model carries one inactive weld `grip` (body1 = the gripper) and the kit's cradle latch `kit_latch` (chassis ->
kit, active at the start). `attach` points `grip` at the held body with the CURRENT relative pose (the object stays
where it is) and activates it; `release` deactivates it. eq_data layout of a MuJoCo weld (3.x): [anchor(3),
relpose pos(3), relpose quat wxyz(4), torquescale(1)].
"""

import mujoco
import numpy as np

__all__ = ["attach", "release", "held_body", "rel_pose", "weld_load"]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))


def rel_pose(m, d, b1, b2):
    R1 = d.xmat[b1].reshape(3, 3)
    p = R1.T @ (d.xpos[b2] - d.xpos[b1])
    q = _qmul(_qinv(d.xquat[b1]), d.xquat[b2])
    return p, q / np.linalg.norm(q)


def attach(m, d, eq_name, body):
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    b1 = int(m.eq_obj1id[eq])
    b2 = int(body)
    p, q = rel_pose(m, d, b1, b2)
    m.eq_obj2id[eq] = b2
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = p
    m.eq_data[eq, 6:10] = q
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1
    return eq


def release(m, d, eq_name):
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    was = int(m.eq_obj2id[eq]) if d.eq_active[eq] else -1
    d.eq_active[eq] = 0
    return was


def held_body(m, d, eq_name):
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    return int(m.eq_obj2id[eq]) if d.eq_active[eq] else -1


def weld_load(m, d, eq_name):
    """(force N, torque N m) carried by the weld after the last step; None when inactive."""
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if not d.eq_active[eq]:
        return None
    rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == eq]
    if not rows:
        return 0.0, 0.0
    f = np.asarray(d.efc_force[rows], float)
    return float(np.linalg.norm(f[:3])), (float(np.linalg.norm(f[3:6])) if len(f) >= 6 else 0.0)
