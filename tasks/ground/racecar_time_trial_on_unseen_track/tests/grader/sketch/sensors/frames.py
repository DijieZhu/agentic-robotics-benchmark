"""Exact pose and velocity of a car's chassis frame C, read from its free joint `<prefix>root`.

Why not `data.site_xpos`: after `mj_step` MuJoCo's derived quantities (xpos, xmat, site_xpos, cvel) belong to the state
at the START of that step, because the forward pass runs before the integration. The contract casts sub-scan k
"from the pose of site lidar at that state", i.e. after step i. The free joint's qpos / qvel ARE that state, so the
sensors compute the chassis pose and the velocity of a site from them directly, without calling `mj_kinematics`
(which would also change what other packages read from `data` between steps).

`ChassisState.verify(model)` compares this against MuJoCo's own kinematics once, so a car whose chassis is not the
root body of its free joint is rejected at construction rather than giving a silently wrong sensor pose.
"""

import mujoco
import numpy as np


class ChassisState(object):
    """The chassis of the car with `prefix`: body `<prefix>chassis`, free joint `<prefix>root`."""

    def __init__(self, model, prefix):
        self.prefix = prefix
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "chassis")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root")
        if self.body < 0 or jid < 0:
            raise ValueError("car %r: body 'chassis' or joint 'root' is missing" % prefix)
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(model.jnt_bodyid[jid]) != self.body:
            raise ValueError("car %r: joint 'root' must be the free joint of 'chassis'" % prefix)
        self.qa = int(model.jnt_qposadr[jid])
        self.va = int(model.jnt_dofadr[jid])
        self._r9 = np.zeros(9)
        self._q = np.zeros(4)

    def rotation(self, data):
        """3 x 3 rotation of C in W (columns = C's axes in world coordinates)."""
        q = data.qpos[self.qa + 3:self.qa + 7]
        self._q[:] = q / np.sqrt(np.einsum("i,i->", q, q))        # BLAS-free
        mujoco.mju_quat2Mat(self._r9, self._q)
        return self._r9.reshape(3, 3).copy()

    def position(self, data):
        return np.array(data.qpos[self.qa:self.qa + 3])

    def linear_velocity(self, data):
        """World-frame velocity of C's origin (a free joint's translational velocity is in world coordinates)."""
        return np.array(data.qvel[self.va:self.va + 3])

    def angular_velocity_local(self, data):
        """Angular velocity of the chassis in C (a free joint's rotational velocity is in the body frame)."""
        return np.array(data.qvel[self.va + 3:self.va + 6])

    def site_offset(self, model, name):
        """(position in C, rotation of the site against C) of the site `<prefix><name>`; the site must hang on chassis."""
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.prefix + name)
        if sid < 0:
            raise ValueError("car %r: site %r is missing" % (self.prefix, name))
        if int(model.site_bodyid[sid]) != self.body:
            raise ValueError("car %r: site %r is not on the chassis body" % (self.prefix, name))
        r9 = np.zeros(9)
        mujoco.mju_quat2Mat(r9, np.array(model.site_quat[sid], dtype=float))
        return sid, np.array(model.site_pos[sid], dtype=float), r9.reshape(3, 3)

    def site_pose(self, data, pos_c, rot_c):
        """World position and rotation of a site given its offset in C."""
        rot = self.rotation(data)
        return self.position(data) + np.einsum("ij,j->i", rot, pos_c), np.einsum("ij,jk->ik", rot, rot_c)

    def site_velocity(self, data, pos_c, rot=None):
        """World-frame linear velocity of the point `pos_c` of C."""
        if rot is None:
            rot = self.rotation(data)
        w = np.einsum("ij,j->i", rot, self.angular_velocity_local(data))
        r = np.einsum("ij,j->i", rot, pos_c)
        return self.linear_velocity(data) + np.array([w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2],
                                                      w[0] * r[1] - w[1] * r[0]])

    def verify(self, model, sites=("lidar", "imu"), tol=1e-9):
        """Compare the joint-based pose and site velocity with MuJoCo's kinematics on a scratch MjData in a tilted,
        moving test state. Raises ValueError on a mismatch (for example a free joint with a position offset)."""
        d = mujoco.MjData(model)
        d.qpos[self.qa:self.qa + 3] = [0.3, -0.2, 0.05]
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0.05, -0.08, 0.7]), "xyz")
        d.qpos[self.qa + 3:self.qa + 7] = quat
        d.qvel[self.va:self.va + 6] = [1.5, -0.4, 0.1, 0.3, -0.2, 2.0]
        mujoco.mj_forward(model, d)
        res = np.zeros(6)
        for name in sites:
            sid, pos_c, rot_c = self.site_offset(model, name)
            p, r = self.site_pose(d, pos_c, rot_c)
            if np.abs(p - d.site_xpos[sid]).max() > tol or np.abs(r - d.site_xmat[sid].reshape(3, 3)).max() > tol:
                raise ValueError("car %r: the pose of site %r from joint 'root' disagrees with MuJoCo" % (self.prefix, name))
            mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
            if np.abs(self.site_velocity(d, pos_c) - res[3:]).max() > 1e-7:
                raise ValueError("car %r: the velocity of site %r from joint 'root' disagrees with MuJoCo" % (self.prefix, name))
        return True
