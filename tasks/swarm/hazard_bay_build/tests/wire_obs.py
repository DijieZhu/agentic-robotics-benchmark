"""The bay mission's wire observation (control.md, What you observe): what a real platform has, each robot's own
state as its onboard systems estimate it, and a reference to the latest frame set its declared sensors delivered.
Nothing on it says where an object is.

    wo = WireObs(noise_seed, mission, ugv_dofs=..., uav_dofs=...)
    wire = wo.build(obs, k, ugv_truth, uav_joints, vacuum, frames)

`obs` is `ControlSim.observe()`'s legacy observation, computed in full and unchanged (its noise draws come from the
simulator's own stream in a fixed order: the reference's inputs `uav.*.load` and `ugv.joints` keep their numbers only
if nothing else draws from that stream). This module filters it: the object fields go, the values that stay are kept
as they are (the same objects), the vehicle's odometry is replaced, and the new robot-state fields are added. Every
new random number comes from a NumPy stream of its own, `Generator(PCG64(SeedSequence([noise_seed, k])))` with a
fixed k per quantity; a frame's noise is keyed by its frame set's index as well (`frame_rng`), so it never depends on
the frames before it.

The vehicle's dead reckoning (control.md: `ugv.pose`), from the true root pose at each observation:

- t = 0: heading 0 (as parked), position `mission.ugv.start`; per-run scale errors eps_s ~ U(-0.02, 0.02) (wheels)
  and eps_g ~ U(-0.01, 0.01) (gyro), drawn once from the odometry stream;
- each step k >= 1, psi the true heading (0 facing +y, counter-clockwise): d = R(-psi_mid) (p_k - p_k-1) with
  psi_mid = psi_k-1 + wrap(psi_k - psi_k-1) / 2; d~ = (1 + eps_s) d + N(0, 0.02 m sqrt(|d| / 1 m)) per component; the
  gyro w~ = (1 + eps_g) w + N(0, 0.387 deg/s), w the step's true heading change over 1/60 s; heading^ += w~ dt;
  position^ += R(heading^_k-1 + w~ dt / 2) d~;
- reported: [x^ + N(0, 5 mm), y^ + N(0, 5 mm), heading^ in degrees, wrapped to (-180, 180]].

NumPy only (the probe imports it under Kit; the maintainers' checks import it without).
"""

import math

import numpy as np

# the streams (SeedSequence([noise_seed, key])): one per quantity; never the simulator's own `random.Random`
K_ODO = 1  # the vehicle's wheel odometry: the per-run scale errors (wheels, gyro), then each step's displacement noise
K_GYRO = 2  # the gyro's white noise
K_VEL = 3  # the body-frame velocity's noise
K_POSE = 4  # the white noise on the reported position
K_UAV_JOINTS = 5  # the flying robots' encoders
K_FRAME = 100  # a frame's noise: [noise_seed, K_FRAME + the sensor's index in the manifest, the frame set's index]

# the legacy observation's object fields, and the vehicle's fields that were truth in disguise (control.md)
REMOVED_UAV = ("neighbors", "bucket", "bucket_tilt", "pan", "fire")
REMOVED_UGV = ("pose", "vel", "yaw_rate", "holding", "sees", "gauge")
REMOVED_MISSION = ("pan", "items")
UAV_KEYS = ("pos", "vel", "load", "joints", "disabled")
UGV_KEYS = ("pose", "vel_body", "yaw_rate", "joints", "grip_point", "vacuum", "disabled")
TOP_KEYS = ("t", "time", "mission", "uav", "ugv", "frames")

DEFAULT = dict(odo_scale=0.02, gyro_scale=0.01, odo_walk_m_sqrt_m=0.02, gyro_white_deg_s=0.387, vel_body_m_s=0.02,
               pose_white_m=0.005, uav_joint_rad=0.001)


def stream(noise_seed, key, *more):
    """A NumPy generator keyed by the run's noise seed and a fixed key (and, for a frame, its sensor and index)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(noise_seed), int(key)] + [int(m) for m in more])))


def frame_rng(noise_seed, sensor_index, n):
    """The generator of one frame's noise: its sensor (index in the manifest) and its frame set's index."""
    return stream(noise_seed, K_FRAME + int(sensor_index), int(n))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(a, v):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def heading_deg(psi):
    """A heading in degrees, rounded as the wire rounds it, in (-180, 180]."""
    d = round(math.degrees(wrap(psi)), 2)
    return d + 360.0 if d <= -180.0 else d


class DeadReckoning:
    """The vehicle's onboard odometry (wheels and gyro) in the parking bay's nominal frame, from the true root pose."""

    def __init__(self, noise_seed, start_xy, dt, cfg=None):
        c = dict(DEFAULT, **(cfg or {}))
        self.c, self.dt = c, float(dt)
        self.odo, self.gyro = stream(noise_seed, K_ODO), stream(noise_seed, K_GYRO)
        self.vel_rng, self.pose_rng = stream(noise_seed, K_VEL), stream(noise_seed, K_POSE)
        self.eps_s = float(self.odo.uniform(-c["odo_scale"], c["odo_scale"]))
        self.eps_g = float(self.odo.uniform(-c["gyro_scale"], c["gyro_scale"]))
        self.p_hat = np.array(start_xy[:2], dtype=float)
        self.psi_hat = 0.0
        self.last = None  # (the true position and heading at the previous observation)
        self.w = 0.0

    def update(self, p_xy, psi):
        """One observation's update from the true root position (x, y) and heading (rad); returns the gyro's rate."""
        p_xy = np.array(p_xy[:2], dtype=float)
        sig_w = math.radians(self.c["gyro_white_deg_s"])
        if self.last is None:  # (t = 0: the odometry's origin; the vehicle stands still)
            self.w = (1.0 + self.eps_g) * 0.0 + float(self.gyro.normal(0.0, sig_w))
        else:
            p0, psi0 = self.last
            dpsi = wrap(psi - psi0)
            psi_mid = psi0 + dpsi / 2
            d_body = rot2(-psi_mid, p_xy - p0)
            ds = float(np.hypot(d_body[0], d_body[1]))
            d_tilde = (1.0 + self.eps_s) * d_body + self.odo.normal(0.0, self.c["odo_walk_m_sqrt_m"] * math.sqrt(ds), 2)
            self.w = (1.0 + self.eps_g) * dpsi / self.dt + float(self.gyro.normal(0.0, sig_w))
            self.p_hat = self.p_hat + rot2(self.psi_hat + self.w * self.dt / 2, d_tilde)
            self.psi_hat = self.psi_hat + self.w * self.dt
        self.last = (p_xy, float(psi))
        return self.w

    def pose(self):
        n = self.pose_rng.normal(0.0, self.c["pose_white_m"], 2)
        return [round(float(self.p_hat[0] + n[0]), 4), round(float(self.p_hat[1] + n[1]), 4), heading_deg(self.psi_hat)]

    def vel_body(self, v_xy, psi):
        """[forward, left] in the vehicle's own frame (its parked +y is forward): the root's true velocity, the wheels'
        scale error, white noise."""
        f = np.array([-math.sin(psi), math.cos(psi)])
        lft = np.array([-math.cos(psi), -math.sin(psi)])
        v = np.array(v_xy[:2], dtype=float)
        n = self.vel_rng.normal(0.0, self.c["vel_body_m_s"], 2)
        return [round(float((1.0 + self.eps_s) * float(v @ f) + n[0]), 4), round(float((1.0 + self.eps_s) * float(v @ lft) + n[1]), 4)]


class WireObs:
    """The probe's builder of the wire observation, one per run. `mission_extra` is what the mission block gains
    (`sensors`: where the frames are); `uav_dofs` the flying robots' DOF names in PhysX's order (for `mission.uav.*.
    joints`); `cfg` the noise (`DEFAULT`)."""

    def __init__(self, noise_seed, dt, ugv_start=None, uav_dofs=None, mission_extra=None, cfg=None):
        self.seed = int(noise_seed)
        self.c = dict(DEFAULT, **(cfg or {}))
        self.dr = DeadReckoning(self.seed, ugv_start, dt, self.c) if ugv_start is not None else None
        self.uav_dofs = {n: list(v) for n, v in (uav_dofs or {}).items()}
        self.uav_rng = stream(self.seed, K_UAV_JOINTS)
        self.mission_extra = dict(mission_extra or {})

    def mission(self, m):
        """The mission block without the object facts (`pan`, `items`), with each flying robot's DOF names and the
        frames' place (the legacy block's other values as they are)."""
        out = {k: v for k, v in m.items() if k not in REMOVED_MISSION}
        if isinstance(out.get("uav"), dict):
            out["uav"] = {n: dict(u, joints=list(self.uav_dofs.get(n, []))) for n, u in out["uav"].items()}
        out.update(self.mission_extra)
        return out

    def build(self, obs, ugv_truth=None, uav_joints=None, vacuum=False, frames=None):
        """The wire observation of one step. `ugv_truth`: (true root x, y; heading rad; true root velocity x, y) or
        None; `uav_joints`: {robot: (positions, velocities)} in PhysX's DOF order; `vacuum`: the suction tool holds a
        part; `frames`: the latest frame set delivered ({n, t, time, slot}) or None."""
        out = {"t": obs["t"], "time": obs["time"], "mission": self.mission(obs["mission"]), "uav": {}}
        noise = None
        nd = sum(len(self.uav_dofs.get(n, [])) for n in obs.get("uav", {}))
        if nd:
            noise = self.uav_rng.normal(0.0, self.c["uav_joint_rad"], nd)
        i = 0
        for n, u in obs.get("uav", {}).items():
            names = self.uav_dofs.get(n, [])
            q, qd = (uav_joints or {}).get(n, ([], []))
            joints = {}
            for j, name in enumerate(names):
                joints[name] = [round(float(q[j]) + float(noise[i + j]), 5), round(float(qd[j]), 4)]
            i += len(names)
            out["uav"][n] = {"pos": u["pos"], "vel": u["vel"], "load": u["load"], "joints": joints, "disabled": u["disabled"]}
        if "ugv" in obs:
            g = obs["ugv"]
            if self.dr is not None and ugv_truth is not None:
                (px, py), psi, (vx, vy) = ugv_truth
                w = self.dr.update((px, py), psi)
                pose, vel_body, yaw_rate = self.dr.pose(), self.dr.vel_body((vx, vy), psi), round(math.degrees(w), 2)
            else:
                pose, vel_body, yaw_rate = g.get("pose"), [0.0, 0.0], 0.0
            out["ugv"] = {"pose": pose, "vel_body": vel_body, "yaw_rate": yaw_rate, "joints": g["joints"], "grip_point": g["grip_point"],
                          "vacuum": bool(vacuum), "disabled": g["disabled"]}
        out["frames"] = frames
        return out


def check_keys(wire):
    """The keys of a wire observation against control.md's schema: a list of what differs (empty when it is exact)."""
    bad = []
    if tuple(wire) != tuple(k for k in TOP_KEYS if k in wire) or not {"t", "time", "mission", "uav", "frames"} <= set(wire):
        bad.append(f"top: {list(wire)}")
    for n, u in (wire.get("uav") or {}).items():
        if tuple(u) != UAV_KEYS:
            bad.append(f"uav.{n}: {list(u)}")
    if "ugv" in wire and tuple(wire["ugv"]) != UGV_KEYS:
        bad.append(f"ugv: {list(wire['ugv'])}")
    m = wire.get("mission") or {}
    for k in REMOVED_MISSION:
        if k in m:
            bad.append(f"mission.{k} present")
    for n, u in (m.get("uav") or {}).items():
        if "joints" not in u:
            bad.append(f"mission.uav.{n}.joints missing")
    return bad
