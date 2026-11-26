"""creal: the REAL fabric of cloth_material_id_from_video (hidden: verify stage and the build-time renderer only).

The agent's simulator (cphys) is the robot's planning model. The real fabric the videos show and the scored tests
execute on is a different, finer physical model of the same piece of fabric, so no cphys material reproduces it
exactly (a sim-to-real gap):

  - the sheet is resolved four times finer: a 41 x 21 vertex grid (7.5 mm; every cphys vertex coincides with a real
    vertex at the flat pose, ix_real = 2 ix, iy_real = 2 iy),
  - it is integrated at 1 ms (cphys: 2 ms),
  - its contacts are softer (contact time constant 8 ms; cphys: 4 ms) and its contact skin thinner (1.5 mm),
  - the air acts on it: every vertex gets the quadratic drag of the air flowing against the sheet's local normal,
    F = -1/2 rho_air C_d a |v.n| (v.n) n (rho_air 1.2 kg/m^3, C_d 1.2, a = the vertex's share of the sheet area),
    which cphys does not model at all (light fabrics fall, swing and slide visibly differently in air),
  - its friction depends on sliding speed (static friction above kinetic, a Stribeck curve): the coefficient in use is
    mu (1 + STATIC_EXCESS exp(-v / V_STRIBECK)), v the mean horizontal speed of the vertices touching the table,
    where cphys has one constant mu; `mu` of a real material is its kinetic coefficient,
  - the clamp holds it with the same jaws and the same equality constraints.

A real fabric is described by the same five MuJoCo numbers (young, poisson, thickness, damping, mu) of THIS model; they
are not the cphys material that best reproduces it (see tests/review/make_set.py: the reference).

The interface mirrors cphys.Sim: run() takes the commanded clamp rows at cphys.DT (2 ms) and record_every / start in
those rows, so ctasks.outcome(..., world=creal) and the renderer use it unchanged.
"""
import numpy as np
import mujoco

import cphys as W

NX, NY = 2 * W.NX - 1, 2 * W.NY - 1
SPACING = W.SPACING / 2
DT = W.DT / 2
SUB = int(round(W.DT / DT))          # real steps per commanded clamp row
RADIUS = 0.0015
CONTACT_SOLREF = 0.008
G = W.G
RHO_AIR, CD = 1.2, 1.2
STATIC_EXCESS, V_STRIBECK = 0.6, 0.12


def _vertex_area():
    w = np.ones((NX, NY))
    w[0, :] *= 0.5
    w[-1, :] *= 0.5
    w[:, 0] *= 0.5
    w[:, -1] *= 0.5
    return (w * SPACING * SPACING).ravel()


AREA_V = _vertex_area()


def scene_xml(material, mass_kg, texture=None):
    m = W.clamp_material(material)
    tx, ty = W.TABLE_HALF
    mat_attr = ' material="fabric"' if texture else ""
    mat_asset = ('<material name="fabric" texture="%s" texrepeat="1 1" specular="0.05" shininess="0.1"/>' % texture
                 if texture else "")
    eqs = "\n".join('    <connect name="hold%d" body1="clamp" body2="fabric_%d" ' % (i, i) + "anc" "hor"
                    + '="0 0 0" active="false" solref="%g 1"/>' % W.HOLD_SOLREF for i in range(NX * NY))
    return f"""<mujoco model="cloth_world_real">
  <compiler angle="radian"/>
  <option timestep="{DT}" integrator="discrete" solver="CG" cone="elliptic" gravity="0 0 {-G}"/>
  <size memory="256M"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="4096"/></visual>
  <asset>{mat_asset}</asset>
  <worldbody>
    <geom name="table" type="box" size="{tx} {ty} 0.02" pos="{W.EDGE_X - tx} 0 -0.02" friction="0 0 0" condim="3"
          rgba="0.55 0.45 0.35 1" group="1"/>
    <flexcomp name="fabric" type="grid" count="{NX} {NY} 1" spacing="{SPACING} {SPACING} {SPACING}" pos="0 0 {RADIUS}"
              dim="2" radius="{RADIUS}" mass="{mass_kg:.9g}" dof="full" rgba="0.8 0.8 0.8 1"{mat_attr}>
      <elasticity young="{m['young']:.9g}" poisson="{m['poisson']:.9g}" thickness="{m['thickness']:.9g}"
                  damping="{m['damping']:.9g}" elastic2d="both"/>
      <contact condim="3" friction="{m['mu']:.9g} 0.002 0.0001" solref="{CONTACT_SOLREF} 1" selfcollide="none"/>
      <edge equality="false"/>
    </flexcomp>
    <body name="clamp" mocap="true" pos="0 0 1"/>
  </worldbody>
  <equality>
{eqs}
  </equality>
</mujoco>"""


def flat_vertices(pose_xy_yaw):
    x, y, yaw = pose_xy_yaw
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    lx = ((ix - (NX - 1) / 2) * SPACING).ravel()
    ly = ((iy - (NY - 1) / 2) * SPACING).ravel()
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * lx - s * ly, y + s * lx + c * ly, np.full(lx.shape, RADIUS)], 1)


COARSE_INDEX = np.array([(2 * ix) * NY + 2 * iy for ix in range(W.NX) for iy in range(W.NY)])


class Sim(W.Sim):
    """The real fabric. Same interface as cphys.Sim; vertices() returns all NX * NY real vertices."""

    def __init__(self, material, mass_kg, xml=None, assets=None):
        self.material = W.clamp_material(material)
        self.mass = float(mass_kg)
        self.m = mujoco.MjModel.from_xml_string(xml or scene_xml(self.material, self.mass), assets or {})
        self.d = mujoco.MjData(self.m)
        a, n = self.m.flex_vertadr[0], self.m.flex_vertnum[0]
        self.vbody = self.m.flex_vertbodyid[a:a + n].copy()
        assert n == NX * NY
        self.vq = np.array([self.m.jnt_qposadr[self.m.body_jntadr[b]] for b in self.vbody])
        self.vd = np.array([self.m.jnt_dofadr[self.m.body_jntadr[b]] for b in self.vbody])
        self.body0 = self.m.body_pos[self.vbody].copy()
        self.held = None
        self.eq0 = self.m.equality("hold0").id
        self.prev_closed = 0.0
        self.unstable = False

    def place_flat(self, pose_xy_yaw):
        mujoco.mj_resetData(self.m, self.d)
        self.set_vertices(flat_vertices(pose_xy_yaw))
        self.held, self.prev_closed, self.unstable = None, 0.0, False
        self.d.eq_active[self.eq0:self.eq0 + NX * NY] = 0
        self.d.xfrc_applied[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def _release(self):
        self.d.eq_active[self.eq0:self.eq0 + NX * NY] = 0
        self.held = None

    def _air(self):
        P = self.vertices().reshape(NX, NY, 3)
        qv = self.d.qvel
        V = np.stack([qv[self.vd], qv[self.vd + 1], qv[self.vd + 2]], 1)
        gx = np.zeros_like(P)
        gy = np.zeros_like(P)
        gx[1:-1] = P[2:] - P[:-2]
        gx[0], gx[-1] = P[1] - P[0], P[-1] - P[-2]
        gy[:, 1:-1] = P[:, 2:] - P[:, :-2]
        gy[:, 0], gy[:, -1] = P[:, 1] - P[:, 0], P[:, -1] - P[:, -2]
        n = np.cross(gx, gy).reshape(-1, 3)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        vn = (V * n).sum(1)
        F = (-0.5 * RHO_AIR * CD * AREA_V * np.abs(vn) * vn)[:, None] * n
        self.d.xfrc_applied[self.vbody, :3] = F
        low = P.reshape(-1, 3)[:, 2] < 3 * RADIUS
        v = float(np.linalg.norm(V[low, :2], axis=1).mean()) if low.any() else 0.0
        self.m.flex_friction[0, 0] = self.material["mu"] * (1.0 + STATIC_EXCESS * np.exp(-v / V_STRIBECK))

    def run(self, clamp_traj, record_every=W.STEPS_PER_FRAME, start=0):
        """clamp_traj rows at cphys.DT; each row is held for SUB real steps with the clamp pose interpolated linearly
        to the next row. record_every / start count clamp rows, as in cphys.Sim.run."""
        C = np.asarray(clamp_traj, float)
        n = len(C)
        out = []
        for k in range(n):
            closed = C[k, 7]
            nxt = C[min(k + 1, n - 1)]
            for j in range(SUB):
                f = j / SUB
                p = (1 - f) * C[k, :3] + f * nxt[:3]
                q = (1 - f) * C[k, 3:7] + f * nxt[3:7]
                q = q / np.linalg.norm(q)
                self.d.mocap_pos[0] = p
                self.d.mocap_quat[0] = q
                if j == 0:
                    if closed > 0.5 and self.prev_closed <= 0.5:
                        self._grasp(p, W.quat_to_mat(q))
                    elif closed <= 0.5 and self.prev_closed > 0.5:
                        self._release()
                    self.prev_closed = closed
                    if k >= start and (k - start) % record_every == 0:
                        out.append(self.vertices())
                self._air()
                mujoco.mj_step(self.m, self.d)
            if not np.isfinite(self.d.qpos[self.vq[:1]]).all() or np.abs(self.d.qpos[self.vq]).max() > 10:
                self.unstable = True
                break
        return np.array(out)
