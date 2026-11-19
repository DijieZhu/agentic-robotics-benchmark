"""Calibration gates of the drive model. Procedures and lines are fixed here,
before the runs, and are not changed after a result.

    python -m sketch.car.gates --all --write          # every gate, both tyre models -> calibration_table.json
    python -m sketch.car.gates --gate brake           # one gate

Gates and their pre-registered procedures (privileged state is used to drive and to measure; the car is commanded
only through `CarRuntime.set_targets`, once per tick of 25 ms, as the referee does, without the command delay):

static      at rest after 2 s on grip 1.0: total mass 1.8359 kg (|d| <= 0.1 g); centre of mass within 0.1 mm of the
            owner's model; front axle share of the contact normal forces 48.46 % within 1 point; ride height (lowest
            point of the chassis collision hull above the floor) 14.69 mm within 1 mm; origin of C at z = 0 within 1 mm.
radius      full lock, speed target 0.5 m/s, grip 1.0 and 0.45, left and right, 14 s; circle fitted to the path of the
            origin of C over 7..14 s; line |R / 0.5791 - 1| <= 3 %. Since v1.1 the line is judged on the
            geometric radius, with the centre coupling off ("unlocked"); the coupled value ("locked", the 1.0 line,
            verdict kept as `pass_v1_0`) is reported: driveline wind-up, not geometry, makes it larger.
top_speed   grip 1.0, speed target 8.0 m/s (the command clip), 8 s; top speed = mean ground speed over the last 2 s;
            line 5.5 .. 6.5 m/s.
brake       grip 0.45, 0.6, 0.8, 1.0: speed target 4.0 m/s for 4 s, then 0 (full brake); deceleration = slope of a
            least-squares line of ground speed against time while 0.9 v0 >= v >= 0.1 v0; line |a / (grip g) - 1| <= 10 %.
lateral     circle of radius 2.0 m (left), privileged path controller, speed target ramped from 1.0 m/s at
            0.2 m/s per s until the car leaves the circle; a_y = ground speed x yaw rate; maximum over 0.5 s moving means
            whose lateral error stays within 0.10 m; ratio to grip x g for grip 0.45, 0.6, 0.8, 1.0; line 0.85 .. 1.0
            for every grip and the maximum strictly increasing with grip.
yaw         grip 1.0, 3.0 m/s, steering step 0 -> 0.10 rad at a tick boundary; steady yaw rate = mean over 1.0 .. 1.5 s
            after the step; rise time = time from 10 % to 90 % of it; line 0.10 .. 0.25 s. Also reported: t63, and the
            same at 5.0 m/s and at grip 0.6.
cornering   (a) car level, steady circles at the constant steering angle atan(L / 3.0 m) at 1.5 and 3.0 m/s on grip 1.0
            (first run used the path controller, which oscillated with the slip model at 3 m/s):
            per-axle slip angles of the single-track model and C_S = (a_y / g) / alpha (normalised by load, like the
            platform's C_S at mu = 1);
            (b) tyre rig: the chassis welded to a mocap sled moving at 3.0 m/s at a fixed slip angle alpha, wheels
            rolling under the speed loop; F_y / N of the four tyres for alpha 0.005 .. 0.8 rad, grip 1.0 and 0.6.
            Compared (not a pass line) with the public 4.718 (front) / 5.4562 (rear) per rad; the difference is stated.
stability   at 2.5 ms and 1.25 ms: (S0, added after a launch instability found by the brake gate) full throttle from
            rest on grip 0.45 and 1.0, 3 s; (S1) 7.0 m/s with a 0.08 rad, 0.8 Hz steering sine, 4 s; (S2) 3.0 m/s into a static
            duct-shaped capsule wall, head-on and at 30 deg, 2 s; (S3) spin: grip 0.6, 5.0 m/s, full lock and full
            throttle, 4 s; (S4) a 7 mm slab edge at 1, 3, 5, 7 m/s square and at 3 m/s at 30 deg, 2 s.
            A run is stable when: no MuJoCo warning; all of qpos / qvel finite; chassis speed never above
            max(v0, 6.5) + 0.5 m/s; tyre x floor penetration <= 2 mm and car x wall penetration <= 10 mm at every step;
            chassis tilt <= 1.0 rad. Agreement of 2.5 ms with 1.25 ms: S1 final speed within 1 % and final position
            within 0.05 m; S0 final speed within 1 %; S2 rebound speed within 25 % (or both below 0.1 m/s) and peak deceleration within 25 %;
            S3 peak yaw rate within 10 %; S4 final speed within 2 % and peak chassis vertical acceleration within 30 %.
            2.5 ms is kept when every run is stable and agrees; otherwise the fallback is 1.25 ms.
            Since v1.1 (the verdict by the lines above is kept as `pass_v1_0`): 2.5 ms is final and the gate
            is: every run with no MuJoCo warning, all finite and chassis tilt <= 0.1 rad; against the 1.25 ms run, the
            end speed within 1 % for the runs that end moving (S0, S1, S3, S4) and the peak deceleration within 10 %
            for the impacts (S2, whose end speeds are rebounds below 0.1 m/s). Penetration and the S1 trajectory
            divergence under limit sliding are reported only (S4's impact table carries penetration).
artifact    grip 0.45, full lock both ways, from rest, speed target 0.5 m/s and 8.0 m/s, 8 s: no contact between two
            geoms of the car at any step; at 0.5 m/s the mean ground speed over the last 4 s >= 0.4 m/s (the car is not
            held). Also reported: the smallest distance between a hull box and a front tyre over steering
            -lock .. +lock and suspension droop .. bump (mj_geomDistance; self-contacts are excluded anyway).
determinism the same 12 s script twice in one process and once in another process: sha256 of qpos and qvel equal.
perf        microseconds per physics step (pre_step + mj_step, and pre_step alone): the car on a flat floor and the
            car in the contract's example venue (static ducts), median and p25 over 20 000 steps; machine load recorded.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

from sketch.examples import constants as K

from .build_car import add_car, _bits, _f
from .params import HERE, load_params
from .runtime import CarRuntime

TABLE_PATH = os.path.join(HERE, "calibration_table.json")
TICK = K.TICK_S
G = 9.81
R_KIN = K.MIN_TURN_RADIUS_M
TYRES = ("cone", "slip_angle")


# ------------------------------------------------------------------------------------------------------------ bench
def floor_spec(timestep, size=60.0, extra=""):
    return mujoco.MjSpec.from_string(
        '<mujoco model="s1_bench"><compiler angle="radian" autolimits="true"/><option timestep="%s" cone="elliptic"/>'
        '<worldbody><geom name="floor" type="plane" size="%s %s 0.05" group="%d" %s/>%s</worldbody></mujoco>'
        % (_f(timestep), _f(size), _f(size), K.GROUP_FLOOR, _bits("floor"), extra))


class Bench(object):
    """One car on a flat floor (plus optional static extras), stepped like the referee does."""

    def __init__(self, tyre_model="cone", grip=1.0, timestep=None, extra="", pose=(0.0, 0.0, 0.0), v0=0.0,
                 params=None, centre_lock=True, spec_hook=None):
        p = params or load_params()
        self.dt = float(timestep or p["physics_dt_s"])
        spec = floor_spec(self.dt, extra=extra)
        if spec_hook is not None:
            spec_hook(spec)
        add_car(spec, prefix=K.EGO_PREFIX, pose=pose, params=p, tyre_model=tyre_model)
        self.spec = spec
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.car = CarRuntime(self.m, self.d, prefix=K.EGO_PREFIX, params=p, grip=grip, centre_coupling=centre_lock)
        self.car.place(pose)
        self.n_tick = int(round(TICK / self.dt))
        self.cid = self.car.chassis
        if v0:
            self.set_speed(v0)
        self.warn0 = [w.number for w in self.d.warning]

    def set_speed(self, v):
        """Give the resting car the forward speed v (chassis and rolling wheels), loops preset to hold it."""
        d, car = self.d, self.car
        yaw = car.state()["pose"][2]
        d.qvel[car.v_root:car.v_root + 3] = [v * math.cos(yaw), v * math.sin(yaw), 0.0]
        d.qvel[car.spin_dof] = v / car.r
        car.speed_target = v
        mujoco.mj_forward(self.m, d)

    def warnings(self):
        return int(sum(w.number for w in self.d.warning) - sum(self.warn0))

    def tick(self, steer=None, speed=None, each=None):
        if steer is not None:
            self.car.set_targets(steer, speed)
        for i in range(self.n_tick):
            self.car.pre_step()
            mujoco.mj_step(self.m, self.d)
            if each is not None:
                each(self)

    def run(self, seconds, control=None, each=None):
        """control(bench) -> (steer, speed) at every tick; each(bench) after every physics step."""
        for _ in range(int(round(seconds / TICK))):
            if control is not None:
                s, v = control(self)
                self.car.set_targets(s, v)
            self.tick(each=each)

    # privileged readings
    def pose(self):
        return self.car.state()["pose"]

    def speed(self):
        v = self.d.qvel[self.car.v_root:self.car.v_root + 2]
        return float(math.hypot(v[0], v[1]))

    def yaw_rate(self):
        """World z angular velocity of the chassis."""
        w = self.d.qvel[self.car.v_root + 3:self.car.v_root + 6]
        return float((self.d.xmat[self.cid].reshape(3, 3) @ w)[2])

    def tilt(self):
        return float(math.acos(max(-1.0, min(1.0, self.d.xmat[self.cid][8]))))


def circle_fit(xy):
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    c, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    return float(math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)), (float(c[0]), float(c[1]))


def car_geom_ids(m, prefix=K.EGO_PREFIX):
    return {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)}


def contact_normals(m, d, names):
    """Sum of contact normal forces on the named geoms."""
    f6 = np.zeros(6)
    out = {n: 0.0 for n in names}
    for i in range(d.ncon):
        c = d.contact[i]
        for gid in (c.geom1, c.geom2):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if n in out:
                mujoco.mj_contactForce(m, d, i, f6)
                out[n] += float(f6[0])
    return out


# ------------------------------------------------------------------------------------------------------------ gates
def gate_static(tyre_model="cone"):
    b = Bench(tyre_model, 1.0)
    b.run(2.0, control=lambda bb: (0.0, 0.0))
    m, d = b.m, b.d
    tyres = ["car/%s/tyre" % w for w in K.WHEELS]
    n = contact_normals(m, d, tyres)
    tot = sum(n.values())
    front = n["car/fl/tyre"] + n["car/fr/tyre"]
    mass = float(m.body_subtreemass[b.cid])
    com = d.subtree_com[b.cid].copy()
    # world position of the owner's centre of mass point in C = C origin + R (com_c)
    rot = d.xmat[b.cid].reshape(3, 3)
    com_c = rot.T @ (com - d.xpos[b.cid])
    # lowest point of the collision hull (the boxes' bottom corners)
    zmin = 1e9
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("car/chassis/col"):
            hs = m.geom_size[g]
            r = d.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        zmin = min(zmin, float((d.geom_xpos[g] + r @ (hs * [sx, sy, sz]))[2]))
    out = {"mass_kg": round(mass, 7), "com_c_m": [round(float(x), 6) for x in com_c],
           "normal_n": {w: round(n["car/%s/tyre" % w], 4) for w in K.WHEELS}, "normal_total_n": round(tot, 4),
           "weight_n": round(mass * G, 4), "front_share": round(front / tot, 5),
           "ride_height_m": round(zmin, 6), "origin_z_m": round(float(d.xpos[b.cid][2]), 7),
           "susp_m": [round(float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "car/%s/susp" % w)]]), 7)
                      for w in K.WHEELS]}
    out["pass"] = {"mass": abs(mass - K.MASS_KG) <= 1e-4,
                   "com": float(np.abs(com_c - np.asarray(K.COM_C)).max()) <= 1e-4,
                   "front_share": abs(out["front_share"] - K.COM_C[0] / K.WHEELBASE_M) <= 0.01,
                   "ride_height": abs(zmin - K.GROUND_CLEARANCE_M) <= 1e-3,
                   "origin_z": abs(out["origin_z_m"]) <= 1e-3}
    return out


def gate_radius(tyre_model="cone", grips=(1.0, 0.45), speed=0.5, centre_lock=True, seconds=14.0):
    rows = []
    for grip in grips:
        for side in (1, -1):
            b = Bench(tyre_model, grip, centre_lock=centre_lock)
            xy = []
            b.run(seconds, control=lambda bb: (side * K.STEER_MAX_RAD, speed),
                  each=lambda bb: xy.append(bb.d.qpos[bb.car.q_root:bb.car.q_root + 2].copy()) if bb.d.time > 7.0 else None)
            r, _ = circle_fit(xy)
            rows.append({"grip": grip, "side": "left" if side > 0 else "right", "radius_m": round(r, 4),
                         "error": round(r / R_KIN - 1.0, 4), "speed_mps": round(b.speed(), 3),
                         "pass": abs(r / R_KIN - 1.0) <= 0.03, "warnings": b.warnings()})
    return {"rows": rows, "centre_lock": centre_lock, "pass": all(r["pass"] for r in rows)}


def gate_top_speed(tyre_model="cone"):
    b = Bench(tyre_model, 1.0)
    vs = []
    b.run(8.0, control=lambda bb: (0.0, 8.0), each=lambda bb: vs.append(bb.speed()) if bb.d.time > 6.0 else None)
    v = float(np.mean(vs))
    return {"top_speed_mps": round(v, 4), "wheel_speed_mps": round(b.car.wheel_speed(), 4),
            "current_a": round(b.car.current, 3), "pass": 5.5 <= v <= 6.5}


def gate_brake(tyre_model="cone", grips=(0.45, 0.6, 0.8, 1.0), v_cmd=4.0):
    rows = []
    for grip in grips:
        b = Bench(tyre_model, grip)
        b.run(4.0, control=lambda bb: (0.0, v_cmd))
        v0 = b.speed()
        ts, vs = [], []
        b.run(2.5, control=lambda bb: (0.0, 0.0), each=lambda bb: (ts.append(bb.d.time), vs.append(bb.speed())))
        ts, vs = np.array(ts), np.array(vs)
        sel = (vs <= 0.9 * v0) & (vs >= 0.1 * v0)
        a = -float(np.polyfit(ts[sel], vs[sel], 1)[0])
        rows.append({"grip": grip, "v0_mps": round(v0, 3), "decel_mps2": round(a, 4), "ratio": round(a / (grip * G), 4),
                     "pass": abs(a / (grip * G) - 1.0) <= 0.10, "warnings": b.warnings()})
    return {"rows": rows, "pass": all(r["pass"] for r in rows)}


class CircleDriver(object):
    """Privileged path controller on a circle of radius R about (0, R) (left turn from the origin heading +x):
    feed-forward atan(L / R), plus proportional and integral action on the lateral error of the origin of C, minus
    the course error (direction of the velocity of the origin of C against the circle's tangent; the chassis heading
    would count the body slip angle as an error). Called once per tick."""

    def __init__(self, radius, k_e=0.6, k_i=0.6, k_chi=1.0, i_max=0.5):
        self.R, self.k_e, self.k_i, self.k_chi, self.i_max = radius, k_e, k_i, k_chi, i_max
        self.integ = 0.0

    def errors(self, bench):
        x, y, yaw = bench.pose()
        dx, dy = x, y - self.R
        rho = math.hypot(dx, dy)
        e = rho - self.R                                       # positive: outside
        tang = math.atan2(dy, dx) + 0.5 * math.pi              # direction of the circle (counter-clockwise)
        v = bench.d.qvel[bench.car.v_root:bench.car.v_root + 2]
        chi = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 0.2 else yaw
        psi = math.atan2(math.sin(chi - tang), math.cos(chi - tang))
        return e, psi

    def steer(self, bench):
        e, psi = self.errors(bench)
        self.integ = max(-self.i_max, min(self.i_max, self.integ + e * TICK))
        d = math.atan(K.WHEELBASE_M / self.R) + self.k_e * e + self.k_i * self.integ - self.k_chi * psi
        return max(-K.STEER_MAX_RAD, min(K.STEER_MAX_RAD, d))


def gate_lateral(tyre_model="cone", grips=(0.45, 0.6, 0.8, 1.0), radius=2.0, ramp=0.2, v_start=1.0):
    rows = []
    for grip in grips:
        b = Bench(tyre_model, grip, pose=(0.0, 0.0, 0.0), v0=v_start)
        drv = CircleDriver(radius)
        t0 = b.d.time
        log = []
        state = {"left": False}

        def control(bb):
            v = v_start + ramp * (bb.d.time - t0)
            e, _ = drv.errors(bb)
            if abs(e) > 0.5:
                state["left"] = True
            return drv.steer(bb), v

        def each(bb):
            e, _ = drv.errors(bb)
            v = bb.d.qvel[bb.car.v_root:bb.car.v_root + 2]
            log.append((bb.d.time, bb.speed(), bb.yaw_rate(), math.atan2(v[1], v[0]), e))
        t_max = (math.sqrt(1.2 * grip * G * radius) + 1.0 - v_start) / ramp
        for _ in range(int(t_max / 0.5)):
            b.run(0.5, control=control, each=each)
            if state["left"]:
                break
        arr = np.array(log)
        # tick means: speed, yaw rate, course rate (centred difference of the course angle over one tick), error
        n = b.n_tick
        k = len(arr) // n
        a = arr[:k * n].reshape(k, n, 5)
        t_k = a[:, -1, 0]
        v_k = a[:, :, 1].mean(axis=1)
        r_k = a[:, :, 2].mean(axis=1)
        chi = np.unwrap(arr[:, 3])
        chi_k = chi[:k * n].reshape(k, n)[:, -1]
        chidot = np.gradient(chi_k, t_k)
        e_k = np.abs(a[:, :, 4]).max(axis=1)
        w = int(round(0.5 / TICK))

        def best_window(values, ok):
            best, t_best = 0.0, None
            cs = np.concatenate([[0.0], np.cumsum(values)])
            bad = np.concatenate([[0], np.cumsum(~ok)])
            for i in range(0, len(values) - w + 1):
                if bad[i + w] - bad[i] == 0:
                    mean = (cs[i + w] - cs[i]) / w
                    if mean > best:
                        best, t_best = float(mean), float(t_k[i + w - 1])
            return best, t_best
        # pre-registered measure: v x yaw rate with |e| <= 0.10 m
        vr, t_vr = best_window(v_k * np.abs(r_k), e_k <= 0.10)
        # steady-state measure: v x course rate, and the yaw rate equal to the course rate within 10 %
        steady = (e_k <= 0.10) & (np.abs(r_k - chidot) <= 0.10 * np.maximum(np.abs(r_k), 0.5))
        best, t_best = best_window(v_k * np.abs(chidot), steady)
        rows.append({"grip": grip, "ay_max_mps2": round(best, 4), "ratio": round(best / (grip * G), 4), "t_s": t_best,
                     "ay_max_vr_mps2": round(vr, 4), "ratio_vr": round(vr / (grip * G), 4), "t_vr_s": t_vr,
                     "left_circle": state["left"], "warnings": b.warnings(),
                     "pass": 0.85 <= best / (grip * G) <= 1.0})
    mono = all(rows[i + 1]["ay_max_mps2"] > rows[i]["ay_max_mps2"] for i in range(len(rows) - 1))
    return {"rows": rows, "monotonic": mono, "radius_m": radius, "pass": mono and all(r["pass"] for r in rows)}


def gate_yaw(tyre_model="cone", grip=1.0, v=3.0, step=0.10, params=None):
    b = Bench(tyre_model, grip, v0=v, params=params)
    b.run(2.0, control=lambda bb: (0.0, v))
    t0 = b.d.time
    ts, ys = [], []
    b.run(1.5, control=lambda bb: (step, v), each=lambda bb: (ts.append(bb.d.time - t0), ys.append(bb.yaw_rate())))
    ts, ys = np.array(ts), np.array(ys)
    ss = float(ys[(ts >= 1.0) & (ts <= 1.5)].mean())

    def cross(frac):
        i = int(np.argmax(ys >= frac * ss))
        if i == 0:
            return float(ts[0])
        return float(np.interp(frac * ss, [ys[i - 1], ys[i]], [ts[i - 1], ts[i]]))
    t10, t90, t63 = cross(0.1), cross(0.9), cross(0.632)
    kin = v * math.tan(step) / K.WHEELBASE_M
    return {"grip": grip, "v_mps": v, "step_rad": step, "yaw_ss_radps": round(ss, 4), "yaw_kinematic_radps": round(kin, 4),
            "rise_10_90_s": round(t90 - t10, 4), "t10_s": round(t10, 4), "t90_s": round(t90, 4), "t63_s": round(t63, 4),
            "overshoot": round(float(ys.max()) / ss - 1.0, 4), "pass": 0.10 <= t90 - t10 <= 0.25}


def gate_cornering_car(tyre_model="cone", radius=3.0, speeds=(1.5, 3.0), grip=1.0):
    """Steady circle at a constant steering angle atan(L / radius) (no path controller: the measurement needs only a
    steady state; ISO 4138's constant-steer method): single-track slip angles from privileged state over 5..10 s;
    C_S = (a_y / g) / alpha per axle; also the radius actually driven and the yaw-rate spread (steadiness)."""
    rows = []
    lr = K.COM_C[0]                       # COM to rear axle
    lf = K.WHEELBASE_M - lr
    delta_cmd = math.atan(K.WHEELBASE_M / radius)
    for v in speeds:
        b = Bench(tyre_model, grip, v0=v)
        log = []
        steer_q = int(b.m.jnt_qposadr[mujoco.mj_name2id(b.m, mujoco.mjtObj.mjOBJ_JOINT, "car/steer")])

        def each(bb):
            if bb.d.time < 5.0:
                return
            st = bb.car.state()
            vx, vy = st["vel_c"]                       # origin of C, car frame
            r = bb.yaw_rate()
            log.append((vx, vy + r * lr, r, float(bb.d.qpos[steer_q]), bb.speed()))
        b.run(10.0, control=lambda bb: (delta_cmd, v), each=each)
        a = np.array(log)
        vx, vy, r, delta, sp = a.mean(axis=0)
        ay = sp * r
        a_f = delta - math.atan2(vy + lf * r, vx)
        a_r = -math.atan2(vy - lr * r, vx)
        rows.append({"v_mps": v, "ay_mps2": round(ay, 4), "delta_rad": round(delta, 5),
                     "radius_driven_m": round(sp / r, 4) if abs(r) > 1e-9 else None,
                     "yaw_rate_std_radps": float("%.3g" % a[:, 2].std()),
                     "alpha_front_rad": float("%.4g" % a_f), "alpha_rear_rad": float("%.4g" % a_r),
                     "cs_front_per_rad": float("%.4g" % ((ay / G) / a_f)) if abs(a_f) > 1e-6 else None,
                     "cs_rear_per_rad": float("%.4g" % ((ay / G) / a_r)) if abs(a_r) > 1e-6 else None,
                     "understeer_rad": round(delta - K.WHEELBASE_M * r / sp, 5) if sp > 0 else None})
    return {"rows": rows, "radius_m": radius, "grip": grip,
            "public_cs_per_rad": {"front": 4.718, "rear": 5.4562}}


def tyre_rig(tyre_model="cone", grip=1.0, v=3.0, alphas=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                                          0.4, 0.5, 0.6, 0.8),
             settle=1.0):
    """The car welded to a mocap sled moving at speed v along a direction at -alpha to its heading (the car's
    velocity then has the slip angle alpha at every wheel); F_y / N from the tyre contact forces in C."""
    rows = []
    p = load_params()
    dt = p["physics_dt_s"]
    for alpha in alphas:
        spec = floor_spec(dt)
        spec.worldbody.add_body(name="sled", mocap=True, pos=[0.0, 0.0, 0.0])
        add_car(spec, prefix=K.EGO_PREFIX, pose=(0.0, 0.0, 0.0), params=p, tyre_model=tyre_model)
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="car/chassis", name2="sled",
                          objtype=mujoco.mjtObj.mjOBJ_BODY, solref=[0.005, 1.0])
        m = spec.compile()
        d = mujoco.MjData(m)
        car = CarRuntime(m, d, params=p, grip=grip)
        car.place((0.0, 0.0, 0.0))
        z = float(d.qpos[car.q_root + 2])
        mid = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sled")])
        d.mocap_pos[mid] = [0.0, 0.0, z]
        d.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        ux, uy = math.cos(alpha), -math.sin(alpha)
        n_tick = int(round(TICK / dt))
        tyre_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "car/%s/tyre" % w): w for w in K.WHEELS}
        f6 = np.zeros(6)
        sums = {"fy": 0.0, "fx": 0.0, "fz": 0.0, "n": 0}
        steps = int(round((settle + 1.0) / dt))
        for k in range(steps):
            if k % n_tick == 0:
                car.set_targets(0.0, v * math.cos(alpha))
            t = (k + 1) * dt
            d.mocap_pos[mid] = [v * ux * t, v * uy * t, z]
            car.pre_step()
            mujoco.mj_step(m, d)
            if t > settle:
                fy = fx = fz = 0.0
                for i in range(d.ncon):
                    c = d.contact[i]
                    if c.geom1 in tyre_ids or c.geom2 in tyre_ids:
                        mujoco.mj_contactForce(m, d, i, f6)
                        fr = c.frame.reshape(3, 3)
                        fw = fr.T @ f6[:3]                    # force of geom1 on geom2, world frame
                        if c.geom1 in tyre_ids:
                            fw = -fw
                        fx += fw[0]
                        fy += fw[1]
                        fz += fw[2]
                sums["fx"] += fx
                sums["fy"] += fy
                sums["fz"] += fz
                sums["n"] += 1
        n = sums["n"]
        fy, fx, fz = sums["fy"] / n, sums["fx"] / n, sums["fz"] / n
        rows.append({"alpha_rad": alpha, "fy_over_n": round(fy / fz, 5), "fx_over_n": round(fx / fz, 5),
                     "normal_n": round(fz, 4), "cs_secant_per_rad": round(fy / fz / alpha, 4)})
    peak = max(rows, key=lambda r: r["fy_over_n"])
    return {"tyre_model": tyre_model, "grip": grip, "v_mps": v, "rows": rows,
            "peak_fy_over_n": peak["fy_over_n"], "alpha_at_peak_rad": peak["alpha_rad"],
            "fy_over_n_at_max_alpha": rows[-1]["fy_over_n"]}


# ------------------------------------------------------------------------------------------------------------ stability
WALL = ('<geom name="duct/W/0/col" type="capsule" fromto="%s" size="0.165" group="1" contype="4" conaffinity="10" '
        'rgba="0.95 0.45 0.1 1"/>')
SLAB = ('<geom name="zone/E/slab" type="box" size="2 1.5 0.0035" pos="3 0 0.0035" group="3" contype="1" conaffinity="10" '
        'rgba="0.15 0.3 0.7 1"/>')


def _run_monitored(b, seconds, control, wall=False):
    m, d = b.m, b.d
    tyre = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "car/%s/tyre" % w) for w in K.WHEELS}
    cg = car_geom_ids(m)
    mon = {"max_speed": 0.0, "pen_floor": 0.0, "pen_wall": 0.0, "max_tilt": 0.0, "finite": True, "max_yaw_rate": 0.0,
           "max_az": 0.0, "max_decel": 0.0}
    prev = {"vz": float(d.qvel[b.car.v_root + 2]), "v": b.speed()}

    def each(bb):
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            mon["finite"] = False
        mon["max_speed"] = max(mon["max_speed"], bb.speed())
        mon["max_tilt"] = max(mon["max_tilt"], bb.tilt())
        mon["max_yaw_rate"] = max(mon["max_yaw_rate"], abs(bb.yaw_rate()))
        vz = float(d.qvel[b.car.v_root + 2])
        mon["max_az"] = max(mon["max_az"], abs(vz - prev["vz"]) / bb.dt)
        prev["vz"] = vz
        v = bb.speed()
        mon["max_decel"] = max(mon["max_decel"], (prev["v"] - v) / bb.dt)
        prev["v"] = v
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in cg or c.geom2 in cg:
                pen = max(0.0, -float(c.dist))
                if c.geom1 in tyre or c.geom2 in tyre:
                    other = c.geom2 if c.geom1 in tyre else c.geom1
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                    if name == "floor" or name.startswith("zone/"):
                        mon["pen_floor"] = max(mon["pen_floor"], pen)
                        continue
                mon["pen_wall"] = max(mon["pen_wall"], pen)
    b.run(seconds, control=control, each=each)
    mon["warnings"] = b.warnings()
    mon["final_speed"] = b.speed()
    mon["final_pos"] = [float(x) for x in b.d.qpos[b.car.q_root:b.car.q_root + 2]]
    return mon


def _stable(mon, v0):
    return bool(mon["warnings"] == 0 and mon["finite"] and mon["max_speed"] <= max(v0, 6.5) + 0.5
                and mon["pen_floor"] <= 0.002 and mon["pen_wall"] <= 0.010 and mon["max_tilt"] <= 1.0)


def stability_runs(tyre_model, dt):
    runs = {}
    # S0 (added after the brake gate found a launch instability): full throttle from rest
    for grip in (0.45, 1.0):
        b = Bench(tyre_model, grip, timestep=dt)
        mon = _run_monitored(b, 3.0, lambda bb: (0.0, 8.0))
        mon["note"] = "launch"
        runs["S0_launch_grip%g" % grip] = (mon, 0.0)
    # S1: 7 m/s with a steering sine
    b = Bench(tyre_model, 1.0, timestep=dt, v0=7.0)
    t0 = b.d.time
    runs["S1_7mps_sine"] = (_run_monitored(b, 4.0, lambda bb: (0.08 * math.sin(2 * math.pi * 0.8 * (bb.d.time - t0)), 7.0)), 7.0)
    # S2: wall hits at 3 m/s, head-on and at 30 deg (wall 1.5 m ahead of the car's frame origin)
    for name, ang in (("S2_wall_0deg", 0.0), ("S2_wall_30deg", math.radians(30.0))):
        cx = 1.5
        a0 = (cx - 3.0 * math.sin(ang), -3.0 * math.cos(ang), 0.165)
        a1 = (cx + 3.0 * math.sin(ang), 3.0 * math.cos(ang), 0.165)
        b = Bench(tyre_model, 1.0, timestep=dt, v0=3.0, extra=WALL % _f(*a0, *a1))
        mon = _run_monitored(b, 2.0, lambda bb: (0.0, 3.0), wall=True)
        mon["rebound_speed"] = mon["final_speed"]
        runs[name] = (mon, 3.0)
    # S3: spin on grip 0.6 from 5 m/s
    b = Bench(tyre_model, 0.6, timestep=dt, v0=5.0)
    t0 = b.d.time
    runs["S3_spin"] = (_run_monitored(b, 4.0, lambda bb: ((K.STEER_MAX_RAD, 8.0) if bb.d.time - t0 > 0.5 else (0.0, 5.0))), 5.0)
    # S4: 7 mm slab edge
    for v in (1.0, 3.0, 5.0, 7.0):
        b = Bench(tyre_model, 1.0, timestep=dt, v0=v, extra=SLAB, pose=(0.0, 0.0, 0.0))
        runs["S4_edge_%gmps" % v] = (_run_monitored(b, 2.0, lambda bb, v=v: (0.0, v)), v)
    b = Bench(tyre_model, 1.0, timestep=dt, v0=3.0, extra=SLAB, pose=(0.0, -0.4, math.radians(30.0)))
    runs["S4_edge_3mps_30deg"] = (_run_monitored(b, 2.0, lambda bb: (0.0, 3.0)), 3.0)
    return {k: dict(mon, stable=_stable(mon, v0), v0=v0) for k, (mon, v0) in runs.items()}


def _agree(name, a, b):
    if name.startswith("S0"):
        return abs(a["final_speed"] / b["final_speed"] - 1.0) <= 0.01
    if name.startswith("S1"):
        return (abs(a["final_speed"] / b["final_speed"] - 1.0) <= 0.01
                and math.dist(a["final_pos"], b["final_pos"]) <= 0.05)
    if name.startswith("S2"):
        reb = (a["rebound_speed"] < 0.1 and b["rebound_speed"] < 0.1) or \
              abs(a["rebound_speed"] / max(b["rebound_speed"], 1e-9) - 1.0) <= 0.25
        return reb and abs(a["max_decel"] / max(b["max_decel"], 1e-9) - 1.0) <= 0.25
    if name.startswith("S3"):
        return abs(a["max_yaw_rate"] / b["max_yaw_rate"] - 1.0) <= 0.10
    if name.startswith("S4"):
        return (abs(a["final_speed"] / b["final_speed"] - 1.0) <= 0.02
                and abs(a["max_az"] / max(b["max_az"], 1e-9) - 1.0) <= 0.30)
    raise KeyError(name)


MOVING_END_MPS = 0.1           # a run "ends moving" when both end speeds are at least this


def _stable_v11(mon):
    return bool(mon["warnings"] == 0 and mon["finite"] and mon["max_tilt"] <= 0.1)


def _agree_v11(name, a, b):
    """end speed within 1 % (runs that end moving), peak deceleration within 10 % (the impacts)."""
    if name.startswith("S2"):
        return abs(a["max_decel"] / max(b["max_decel"], 1e-9) - 1.0) <= 0.10
    if min(a["final_speed"], b["final_speed"]) < MOVING_END_MPS:
        return False
    return abs(a["final_speed"] / b["final_speed"] - 1.0) <= 0.01


def gate_stability(tyre_model="cone", dts=(0.0025, 0.00125)):
    res = {"%g" % dt: stability_runs(tyre_model, dt) for dt in dts}
    a, b = res["%g" % dts[0]], res["%g" % dts[1]]
    agree = {k: bool(_agree(k, a[k], b[k])) for k in a}
    stable = {k: bool(a[k]["stable"]) for k in a}
    stable_ref = {k: bool(b[k]["stable"]) for k in b}
    keep = all(stable.values()) and all(agree.values())
    stable11 = {k: _stable_v11(a[k]) and _stable_v11(b[k]) for k in a}
    agree11 = {k: bool(_agree_v11(k, a[k], b[k])) for k in a}
    reported = {k: {"pen_floor_mm": round(1e3 * a[k]["pen_floor"], 2), "pen_wall_mm": round(1e3 * a[k]["pen_wall"], 2),
                    "final_pos_diff_m": round(math.dist(a[k]["final_pos"], b[k]["final_pos"]), 4)} for k in a}
    ok11 = all(stable11.values()) and all(agree11.values())
    out = {"runs": {k: {"%g" % dt: _round(res["%g" % dt][k]) for dt in dts} for k in a}, "stable": stable,
           "stable_reference": stable_ref, "agree": agree, "keep_2p5ms": keep, "pass_v1_0": keep,
           "stable_v1_1": stable11, "agree_v1_1": agree11, "reported_v1_1": reported, "pass": ok11,
           "rule": "v1.1 (ruling S1-C3): no warning, finite, tilt <= 0.1 rad; end speed within 1 % of the 1.25 ms run "
                   "(runs ending moving), peak deceleration within 10 % (S2 impacts); penetration and divergence reported"}
    return out


def _round(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float("%.5g" % float(x))
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


# ------------------------------------------------------------------------------------------------------------ artifact
def hull_tyre_clearance(tyre_model="cone"):
    """Smallest distance between a chassis collision box and a front tyre over steering -lock..+lock and the
    suspension's travel (kinematic poses; mj_geomDistance ignores the excludes)."""
    p = load_params()
    b = Bench(tyre_model, 1.0)
    m, d = b.m, b.d
    hull = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("car/chassis/col")]
    front = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "car/%s/tyre" % w) for w in ("fl", "fr")]
    qa = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "car/" + n)])
          for n in ("steer", "fl/kingpin", "fr/kingpin", "fl/susp", "fr/susp")}
    ack = p["derived_steering"]
    best = (1e9, None)
    fromto = np.zeros(6)
    for delta in np.linspace(-K.STEER_MAX_RAD, K.STEER_MAX_RAD, 41):
        for zs in (-p["suspension"]["droop_m"], 0.0, p["suspension"]["bump_m"]):
            d.qpos[qa["steer"]] = delta
            for w in ("fl", "fr"):
                c = ack[w]["polycoef"]
                d.qpos[qa[w + "/kingpin"]] = sum(c[i] * delta ** i for i in range(5))
                d.qpos[qa[w + "/susp"]] = zs
            mujoco.mj_kinematics(m, d)
            for h in hull:
                for t in front:
                    dist = mujoco.mj_geomDistance(m, d, h, t, 0.05, fromto)
                    if dist < best[0]:
                        best = (dist, {"delta_rad": round(float(delta), 4), "susp_m": zs,
                                       "hull": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, h),
                                       "tyre": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, t)})
    return {"min_distance_m": round(float(best[0]), 5), "at": best[1]}


def gate_artifact(tyre_model="cone", grip=0.45):
    rows = []
    for speed in (0.5, 8.0):
        for side in (1, -1):
            b = Bench(tyre_model, grip)
            cg = car_geom_ids(b.m)
            self_contacts = {"n": 0}
            vs = []

            def each(bb):
                d = bb.d
                for i in range(d.ncon):
                    c = d.contact[i]
                    if c.geom1 in cg and c.geom2 in cg:
                        self_contacts["n"] += 1
                if bb.d.time > 4.0:
                    vs.append(bb.speed())
            b.run(8.0, control=lambda bb: (side * K.STEER_MAX_RAD, speed), each=each)
            mean_v = float(np.mean(vs))
            ok = self_contacts["n"] == 0 and (speed > 1.0 or mean_v >= 0.4)
            rows.append({"speed_target_mps": speed, "side": "left" if side > 0 else "right", "self_contacts": self_contacts["n"],
                         "mean_speed_mps": round(mean_v, 4), "pass": ok, "warnings": b.warnings()})
    clear = hull_tyre_clearance(tyre_model)
    return {"rows": rows, "clearance": clear, "pass": all(r["pass"] for r in rows)}


# ------------------------------------------------------------------------------------------------------------ determinism
def script_run(tyre_model="cone", seconds=12.0):
    """A fixed 12 s drive: accelerate, slalom, brake, full-lock spin on grip 0.6, recover. sha256 of qpos+qvel."""
    def grip_at(x, y):
        return 0.6 if x > 6.0 else 1.0
    p = load_params()
    spec = floor_spec(p["physics_dt_s"])
    add_car(spec, params=p, tyre_model=tyre_model)
    m = spec.compile()
    d = mujoco.MjData(m)
    car = CarRuntime(m, d, params=p, grip_at=grip_at)
    car.place((0.0, 0.0, 0.0))
    n_tick = int(round(TICK / m.opt.timestep))
    for k in range(int(round(seconds / TICK))):
        t = k * TICK
        if t < 2.0:
            cmd = (0.0, 4.0)
        elif t < 5.0:
            cmd = (0.3 * math.sin(2.0 * math.pi * 0.7 * (t - 2.0)), 5.0)
        elif t < 6.0:
            cmd = (0.0, 0.0)
        elif t < 9.0:
            cmd = (K.STEER_MAX_RAD, 8.0)
        else:
            cmd = (-0.2, 2.0)
        car.set_targets(*cmd)
        for _ in range(n_tick):
            car.pre_step()
            mujoco.mj_step(m, d)
    h = hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest()
    return h, [float(x) for x in d.qpos[:3]]


def gate_determinism(tyre_model="cone"):
    h1, pos = script_run(tyre_model)
    h2, _ = script_run(tyre_model)
    dev = os.path.dirname(os.path.dirname(HERE))
    code = "import sys; sys.path.insert(0, %r); from sketch.car.gates import script_run; print(script_run(%r)[0])" % (dev, tyre_model)
    env = dict(os.environ)
    h3 = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, check=True).stdout.strip()
    return {"sha256": [h1, h2, h3], "final_pos": [round(x, 6) for x in pos], "pass": h1 == h2 == h3}


# ------------------------------------------------------------------------------------------------------------ perf
def _timeit(m, d, car, n, control=None):
    full, pre = np.zeros(n), np.zeros(n)
    n_tick = int(round(TICK / m.opt.timestep))
    for i in range(n):
        if control is not None and i % n_tick == 0:
            car.set_targets(*control(d.time))
        t0 = time.perf_counter()
        car.pre_step()
        t1 = time.perf_counter()
        mujoco.mj_step(m, d)
        t2 = time.perf_counter()
        full[i], pre[i] = t2 - t0, t1 - t0
    return full, pre


def gate_perf(tyre_model="cone", n=20000, world=None):
    out = {"load1_before": round(os.getloadavg()[0], 1)}
    ctl = lambda t: (0.25 * math.sin(t), 3.0)
    b = Bench(tyre_model, 1.0)
    full, pre = _timeit(b.m, b.d, b.car, n, ctl)
    out["floor"] = {"us_step_median": round(1e6 * float(np.median(full)), 1), "us_step_p25": round(1e6 * float(np.percentile(full, 25)), 1),
                    "us_pre_step_median": round(1e6 * float(np.median(pre)), 1), "nv": int(b.m.nv)}
    if world is not None:
        from sketch.examples import greybox
        from sketch.examples.check import FloorMap
        spec = mujoco.MjSpec.from_string(greybox.build_xml(world, variant="track", cars=None))
        add_car(spec, pose=world["truth"]["start_pose"]["pose"], tyre_model=tyre_model)
        m = spec.compile()
        d = mujoco.MjData(m)
        from .floor import FloorGrip
        for label, lookup in (("example_venue", FloorMap(world).grip_at), ("example_venue_floorgrip", FloorGrip(world).grip_at)):
            d = mujoco.MjData(m)
            car = CarRuntime(m, d, grip_at=lookup)
            car.place(world["truth"]["start_pose"]["pose"])
            full, pre = _timeit(m, d, car, n, lambda t: (0.0, 1.0))
            out[label] = {"us_step_median": round(1e6 * float(np.median(full)), 1),
                          "us_step_p25": round(1e6 * float(np.percentile(full, 25)), 1),
                          "us_pre_step_median": round(1e6 * float(np.median(pre)), 1), "ngeom": int(m.ngeom)}
    out["load1_after"] = round(os.getloadavg()[0], 1)
    return out


# ------------------------------------------------------------------------------------------------------------ driver
GATES = {"static": gate_static, "radius": gate_radius, "top_speed": gate_top_speed, "brake": gate_brake,
         "lateral": gate_lateral, "yaw": gate_yaw, "cornering": None, "stability": gate_stability,
         "artifact": gate_artifact, "determinism": gate_determinism, "perf": gate_perf}


def run_gate(name, tyre_model, world=None):
    if name == "cornering":
        return {"car": gate_cornering_car(tyre_model),
                "rig_grip_1.0": tyre_rig(tyre_model, 1.0), "rig_grip_0.6": tyre_rig(tyre_model, 0.6)}
    if name == "radius":
        locked, unlocked = gate_radius(tyre_model), gate_radius(tyre_model, centre_lock=False)
        return {"locked": locked, "unlocked": unlocked, "pass": unlocked["pass"], "pass_v1_0": locked["pass"],
                "rule": "v1.1 (ruling S1-D1): the geometric radius (centre coupling off) within 3 % of 0.5791 m; the "
                        "coupled value is reported"}
    if name == "yaw":
        return {"gate": gate_yaw(tyre_model), "v5": gate_yaw(tyre_model, v=5.0), "grip0.6": gate_yaw(tyre_model, grip=0.6),
                "lag_0.158": gate_yaw_lag(tyre_model, 0.158)}
    if name == "perf":
        return gate_perf(tyre_model, world=world)
    return GATES[name](tyre_model)


def gate_yaw_lag(tyre_model, lag):
    """The yaw gate with another servo lag (a sensitivity row; not the car's parameter set)."""
    p = load_params()
    p["steering"]["lag_s"] = lag
    r = gate_yaw(tyre_model, params=p)
    return {"lag_s": lag, "rise_10_90_s": r["rise_10_90_s"], "t63_s": r["t63_s"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="S1 calibration gates")
    ap.add_argument("--gate", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tyre", choices=TYRES, action="append", default=[])
    ap.add_argument("--write", action="store_true", help="merge the results into calibration_table.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    names = list(GATES) if a.all else a.gate
    tyres = a.tyre or list(TYRES)
    world = None
    if "perf" in names:
        from sketch.examples import testing
        world = testing.load("min_world.json")
    table = {}
    if a.write and os.path.exists(TABLE_PATH):
        with open(TABLE_PATH, "r", encoding="utf-8") as f:
            table = json.load(f)
    for tm in tyres:
        for name in names:
            t0 = time.time()
            res = run_gate(name, tm, world=world)
            res = _round(res)
            res["wall_s"] = round(time.time() - t0, 1)
            table.setdefault(tm, {})[name] = res
            print(json.dumps({"tyre": tm, "gate": name, "result": res})[:3000])
            sys.stdout.flush()
    table["meta"] = {"mujoco": mujoco.__version__, "host": os.uname().nodename, "load1": round(os.getloadavg()[0], 1),
                     "params_physics_dt_s": load_params()["physics_dt_s"]}
    if a.write or a.out:
        path = a.out or TABLE_PATH
        with open(path, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=1, sort_keys=True)
            f.write("\n")
        print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
