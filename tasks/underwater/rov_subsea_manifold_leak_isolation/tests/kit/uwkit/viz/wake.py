"""Thruster wake as decorative particles (render-only, demo videos for people; never in an agent-visible image).

Each thruster emits particles at the duct exit on its outflow side: the rear lip for forward thrust (water leaves through
the back of the nozzle), the front lip around the nose cone for reverse thrust.  Emission rate, jet speed and opacity grow
with |thrust|; particles start with the local vehicle velocity plus the jet velocity (actuator-disc estimate
v_jet = jet_scale * sqrt(2 |T| / (rho A_duct))), then relax towards the ambient current with time constant tau_s (so the
wake stays behind in the water and drifts with the current), grow and fade over their life.  Deterministic for a given
seed and row sequence (no wall clock, its own Generator).
Drawn by appending ellipsoid geoms (stretched along the particle's water-relative velocity) to an mjvScene.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class WakeStyle:
    rate_max: float = 150.0          # particles / s per thruster at |T| >= thrust_ref
    thrust_ref: float = 35.0         # N
    jet_scale: float = 0.55          # fraction of the actuator-disc jet speed given to the particles
    tau_s: float = 0.32              # relaxation time towards the ambient water
    life_s: float = 1.0
    life_jitter: float = 0.25
    spread: float = 0.10             # transverse velocity jitter / jet speed
    r0: float = 0.0025               # particle radius at birth / at the end [m]
    r1: float = 0.010
    streak_s: float = 0.010          # ellipsoid half-length = r + streak_s * |v_rel|
    alpha0: float = 0.55
    rgb: tuple = (0.82, 0.92, 1.0)
    emission: float = 0.35
    min_thrust: float = 0.15         # N: no emission below
    rho: float = 998.0


class Wake:
    def __init__(self, rotors, seed: int = 0, style: WakeStyle | None = None, floor_z: float | None = None,
                 max_particles: int = 4000):
        self.rotors = rotors
        self.st = style or WakeStyle()
        self.seed = int(seed)
        self.floor_z = floor_z
        self.max_particles = int(max_particles)
        self.reset()

    def reset(self):
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.acc = np.zeros(self.rotors.n)
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.life = np.zeros(0)
        self.amp = np.zeros(0)              # |T| / thrust_ref at birth (opacity / size scale)

    def advance(self, dt, hull_pos, hull_R, v_lin_w, omega_b, thrust, water_w):
        """One row: emit at the duct exits, then integrate all particles over dt."""
        st = self.st
        R = np.asarray(hull_R, float)
        w_w = R @ np.asarray(omega_b, float)
        water_w = np.asarray(water_w, float)
        new_p, new_v, new_l, new_a = [], [], [], []
        T = np.asarray(thrust, float)
        for k, r in enumerate(self.rotors.rotors):
            Tk = float(T[k])
            if abs(Tk) < st.min_thrust:
                self.acc[k] = 0.0
                continue
            x = min(1.0, abs(Tk) / st.thrust_ref)
            self.acc[k] += st.rate_max * x * dt
            n = int(self.acc[k])
            if n <= 0:
                continue
            self.acc[k] -= n
            axis_w = R @ r.axis_B
            out = -axis_w if Tk > 0 else axis_w                      # water leaves opposite to the force on the body
            lip = r.rear_lip_B if Tk > 0 else r.front_lip_B
            c = hull_pos + R @ lip
            A = math.pi * r.duct_inner_radius ** 2
            vj = st.jet_scale * math.sqrt(2.0 * abs(Tk) / (st.rho * A))
            # random points in the exit annulus (outside the hub / cone)
            e1 = np.cross(out, [0.0, 0.0, 1.0] if abs(out[2]) < 0.9 else [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(out, e1)
            rr = np.sqrt(self.rng.uniform((r.hub_radius / r.duct_inner_radius) ** 2, 0.85, n)) * r.duct_inner_radius
            th = self.rng.uniform(0, 2 * math.pi, n)
            pts = c + (rr * np.cos(th))[:, None] * e1 + (rr * np.sin(th))[:, None] * e2
            pts += out * self.rng.uniform(0.0, vj * dt, n)[:, None]
            v_body = v_lin_w + np.cross(w_w, pts - hull_pos)
            jit = self.rng.normal(0.0, st.spread * vj, (n, 3))
            jit -= (jit @ out)[:, None] * out
            new_p.append(pts)
            new_v.append(v_body + out * vj + jit)
            new_l.append(st.life_s * (1.0 + st.life_jitter * self.rng.uniform(-1, 1, n)))
            new_a.append(np.full(n, x))
        if new_p:
            self.p = np.vstack([self.p] + new_p)
            self.v = np.vstack([self.v] + new_v)
            self.age = np.concatenate([self.age] + [np.zeros(len(q)) for q in new_p])
            self.life = np.concatenate([self.life] + new_l)
            self.amp = np.concatenate([self.amp] + new_a)
        if len(self.p):
            k = math.exp(-dt / st.tau_s)
            vr = self.v - water_w
            # exact update of x' = v, v' = -(v - w) / tau over dt for constant w
            self.p = self.p + water_w * dt + vr * st.tau_s * (1.0 - k)
            self.v = water_w + vr * k
            self.age = self.age + dt
            keep = self.age < self.life
            if self.floor_z is not None:
                keep &= self.p[:, 2] > self.floor_z
            if keep.sum() > self.max_particles:
                keep &= np.cumsum(keep[::-1])[::-1] <= self.max_particles      # drop the oldest
            self.p, self.v, self.age, self.life, self.amp = (a[keep] for a in (self.p, self.v, self.age, self.life,
                                                                                self.amp))

    @property
    def n(self):
        return len(self.p)

    def draw(self, scn, water_w=(0, 0, 0), budget=None, size_scale=1.0, alpha_scale=1.0, rgb=None, emission=None):
        """Append the particles to an mjvScene (after update_scene).  Returns the number of geoms added.
        size_scale / alpha_scale / rgb / emission change only the look (e.g. bigger, bluer particles for a far camera);
        the particle state is the same for every view."""
        st = self.st
        room = scn.maxgeom - scn.ngeom
        n = min(self.n, room if budget is None else min(room, budget))
        if n <= 0:
            return 0
        sl = slice(self.n - n, self.n)                   # the newest n if the scene has no room for all
        P = self.p[sl]
        f = self.age[sl] / self.life[sl]
        rad = size_scale * (st.r0 + (st.r1 - st.r0) * np.sqrt(f))
        alpha = np.minimum(1.0, alpha_scale * st.alpha0 * (0.35 + 0.65 * self.amp[sl]) * (1.0 - f) ** 1.3)
        col = st.rgb if rgb is None else rgb
        vr = self.v[sl] - np.asarray(water_w, float)
        sp = np.linalg.norm(vr, axis=1)
        for i in range(n):
            if alpha[i] < 0.01:
                continue
            z = vr[i] / sp[i] if sp[i] > 1e-6 else np.array([0.0, 0.0, 1.0])
            x = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            M = np.column_stack([x, y, z])
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                                np.array([rad[i], rad[i], rad[i] + st.streak_s * sp[i]]), P[i], M.reshape(-1),
                                np.array([*col, alpha[i]], np.float32))
            g.emission = st.emission if emission is None else emission
            g.specular = 0.2
            g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            g.transparent = 1
            scn.ngeom += 1
        return n
