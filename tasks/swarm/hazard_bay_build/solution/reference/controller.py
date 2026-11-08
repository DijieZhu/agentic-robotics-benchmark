"""The reference hoist controller: three flying robots lift the bucket together, carry it to the burning pan, pour,
and hold. Speaks the task's wire on stdin/stdout (one JSON object per line; `READY` first).

Design, in the world frame of the twin (the robots' headings are fixed):
- one position loop per robot (PID on the estimated position, the rope's measured load fed forward);
- the formation's SHAPE comes from the lidars (every pair's relative vector, measured from both ends, propagated with
  the odometry velocities and corrected by the returns), its PLACE from the mean of the three odometries: the
  odometries drift apart, and the bucket hangs level only if the hooks' heights agree to a centimetre;
- the hooks stand on a circle round the bucket (a symmetric rig: equal tensions mean a level bucket); the robots'
  centres are then 1.35 m apart;
- a take-off in TENSION mode: while the bucket is on the floor the rope is a stiff spring (a centimetre of hook
  height is tens of newtons), so the vertical thrust commands each rope's tension directly (the height is free):
  the three tensions are equal by construction, ramp together, and the bucket leaves the floor level; position
  control takes over once it hangs, with a height trim on the measured loads that keeps the three equal;
- minimum-jerk lift and carry profiles (no acceleration steps: little swing, little slosh), the bucket's swing
  damped from what the cameras see, and the pour driven by what the thermal cameras see.

With the true state in each observation (`privileged`, which the grader adds for the reference's token only) the same
phases and the same control run on it in place of the sensors' estimates: the robots' and the bucket's true positions,
the pan where it stands today, the steam and the fire as they are, and the wind (the force on each robot and on the
bucket now, the day's gusts on the mission's clock). Then:
- the drones wait for the steam to be off and lift as soon as the bucket is preloaded and no gust is due over the
  lift-off; the wind's force on each robot is fed forward whole (the measured load carries only the rope's then);
- the bucket is carried as soon as it hangs at the cruise height, round the vehicle when it stands near the line (a
  gust on the way is ridden out: a bucket held up waiting for one rocks water out, and its swing counts all the same);
- over the pan it waits until it hangs still and no gust is due while the pour lasts; the pouring robot is UAV_3 (its
  side faces away from the steam line), and its hook stops coming down while a gust blows. As its rope slackens the
  bucket comes to hang from the other two, further toward them: the formation moves toward UAV_3 by as much, in step
  with the hook (and the same for the second robot's slow drop), so the bucket stays over the pan instead of swinging
  out past the pour zone, and back again as the hooks come up;
- once the fire is out the bucket is flown away from the station (east of the pan, further east than any of the
  vehicle's stands and routes) and set down there, the ropes slack: it stands still on the floor for the rest of the run;
- the hooks' heights are not trimmed on the measured loads (the true heights are equal by command; the trim would
  only follow the ropes' noise and tilt the bucket);
- the vehicle's program is told where the bucket is about to go (it keeps out of those stretches) and when the hoist
  has settled.

A step may be asked twice with the same t (the grader's lock asks some steps again with shifted positions and applies
one answer): each query is answered from the state the last step left, and the state carried on is the one whose
observation this step's continues to within a step's motion; a query that continues none (a shifted copy) is
answered from the nearest, and the next query of the same t picks again (a stateful controller must never advance on
a shifted copy: a phase change or a path planned on one would be carried on into the run).
"""
import copy
import json
import math
import sys

import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # (run with -I: the script's folder is not on the path)
from vehicle import Vehicle, kinematics, log  # noqa: E402

G = 9.81
UAVS = ("UAV_1", "UAV_2", "UAV_3")
# the rigging the reference scene has: which ring each robot's rope goes to, as a bearing from the bucket's centre
RING_BEARING = {"UAV_1": 90.0, "UAV_2": 210.0, "UAV_3": 330.0}
RING_R, RING_Z = 0.14, 0.25
BUCKET_START = np.array([0.0, -2.60])
FORMATION_R = 0.78         # the hooks' distance from the bucket's centre, horizontally (the robots' centres then sit
                           # 1.35 m apart: the grader's separation rule wants 1.2 m)
KP, KD, KI = 3.0, 2.6, 1.0  # the position loop [1/s^2], [1/s], [1/s^3] (the integral absorbs what the feed-forward misses)
A_RISE, V_TAUT = 0.3, 0.08  # the take-off move's peak acceleration [m/s^2]; the approach to the taut rope [m/s]
A_LIFT, A_CARRY = 0.35, 0.2  # [m/s^2] the peak accelerations of the minimum-jerk lift and carry
V_CARRY_PRIV = 0.55        # [m/s] privileged: the carry to the pan peaks no faster (the bucket trails its hooks by about 0.15 m
                           # per m/s it is carried at, which the grader counts as its swing)
T_PRELOAD = 20.0           # [N] each rope's share of the bucket before the lift (3 x 20 N < the lightest bucket's weight)
KD_T, V_CATCH = 12.0, 0.06  # tension mode: the vertical damping [1/s]; the climb while the rope is still slack [m/s]
T_RAMP, T_LIFT_RAMP = 8.0, 10.0  # [N/s] the preload ramp; the lift-off ramp (all three ropes together)
T_LIFT_CAP = 45.0          # [N] over the rope's weight: the bucket is surely off the floor by then (11.2 kg / 3 = 36.6 N)
K_TRIM = 0.002             # [m/(N s)] the height trim that keeps the three loads equal while the bucket hangs still
POUR_DROP, T_POUR, T_RECOVER = 0.9, 4.0, 4.0  # how far the pouring robot lowers its hook [m]; how long down and back up take [s]
POUR_PRE = 0.10            # [m] when its rope slackens the bucket swings away from the pouring robot by about this much (it
                           # hangs from the other two): the level bucket is aligned that far towards the pouring robot


def minjerk(t, T):
    """The minimum-jerk profile 0 -> 1 over T seconds: (fraction, rate)."""
    if T <= 0 or t >= T:
        return 1.0, 0.0
    if t <= 0:
        return 0.0, 0.0
    x = t / T
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T


def t_minjerk(dist, a_max):
    """The duration that keeps a minimum-jerk move of `dist` under the peak acceleration a_max (peak = 5.77 d / T^2)."""
    return math.sqrt(5.77 * max(abs(dist), 1e-6) / a_max)


STEAM_WAIT = 4.0          # [s] after the lever has been pushed to its stop: the leak stops 3 s after it closes
POURERS_PRIV = ("UAV_3",)  # privileged: the robot that lowers its hook to pour (on the side away from the steam line)
SECOND_PRIV = "UAV_2"     # privileged: with one rope slack the bucket tips only about 27 degrees (it hangs from two rings),
                          # which pours nothing when it holds 8.5 L (the water then stands 6 cm below the rim); if it does
                          # not pour, this robot comes down slowly too, pausing while water leaves: the bucket tips further
V_SECOND = 0.06           # [m/s] how fast it comes down
VEHICLE_ROOM = 1.3        # [m] privileged: the carry keeps the bucket this far from the vehicle
BUCKET_ROOM = 1.0         # [m] privileged: the vehicle keeps its centre this far from where the bucket is about to pass
# privileged: the wind (the day's gusts, from the true state)
GUST_LIFT = 8.0           # [s] no gust may blow from the lift-off ramp's start until the bucket has hung a while (the grader takes
                          # the tilt over the 1.5 s after the bucket leaves the floor; the gust tied to lift-off comes 2 s after it at
                          # the earliest, so it is never in that window)
GUST_SETTLE = 2.0         # [s] after a gust, the swing it left damps this long before the lift-off, a carry, a pour or the set-down
GUST_SOON = 0.5           # [s] the pouring hook stops coming down this long before a gust (and until GUST_SETTLE after it)
# privileged: the pour (sized on an offline model of this rig: a rigid bucket on three taut ropes from fixed hooks)
POUR_DROP_PRIV, T_POUR_PRIV, T_RECOVER_PRIV = 0.55, 6.0, 6.0  # the pouring hook's drop [m] (its rope goes slack at about 0.35 m),
                                                              # how long it takes to come down and to go back up [s]
POUR_PRE_PRIV = 0.0       # (the formation's shift keeps the bucket where it hung level: no lead toward the pourer)
POUR_SHIFT, POUR_SLACK = 1.26, 0.35  # the bucket's rest moves 1.26 m per metre of the pouring hook's drop toward the other two hooks,
                                     # until that rope goes slack at a 0.35 m drop (0.44 m in all, the bucket tipped 26.6 degrees)
SECOND_SHIFT, SECOND_BEARING = 1.15, 60.0  # the second robot's drop then moves it on toward this bearing [m/m], [deg]
SECOND_MAX = 0.6          # [m] the second robot comes down at most this far (the bucket then tips 42 degrees)
POUR_WINDOW = 14.0        # [s] no gust may be due this long after the pour starts (the lowering and the water's first litre)
ALIGN_STILL, ALIGN_MAX = 0.8, 6.0  # [s] over its mark and hanging still this long before the pour; the pour starts anyway after
                                   # this long of calm (the bucket rocks a little on its ropes: the pour zone has room for that)
# privileged: after the fire
PARK_DX, PARK_X, PARK_Y = 1.5, 4.7, (1.0, 1.7)  # [m] the bucket goes at least 1.5 m east of where it poured and no nearer the
                                   # station than x = PARK_X, its y kept in this band (the pipe runs north of y = 3.27). Set down
                                   # there, its robots hover low with their centres east of x = 4.0 (their rotors reach 0.6 m): the
                                   # vehicle's stands lie west of x = 2.75 and its planner's grid ends at x = 4.0, and it keeps clear
                                   # of a robot that flies low besides
DOWN_LOW, V_DOWN = 0.15, 0.04  # set-down: a minimum-jerk descent until the bucket's bottom is this high [m], then this slow [m/s]
DOWN_SLACK, DOWN_MAX = 0.30, 0.6  # [m] the hooks go this far below where the bucket touched the floor (the ropes slack); the slow
                                  # descent's longest (the robots then still stand over 1.2 m high)
A_DOWN = 0.25             # [m/s^2] the descent's peak acceleration


def calm_for(pv, t, window, settle=GUST_SETTLE):
    """privileged: no gust blows now, none ended less than `settle` seconds ago and none starts within `window` seconds
    (the gust tied to lift-off counts once its times are known: until the bucket leaves the floor it cannot blow)."""
    for g in ((pv or {}).get("wind") or {}).get("gusts") or []:
        t0, t1 = g.get("t0_s"), g.get("t1_s")
        if t0 is not None and t1 is not None and float(t0) < t + window and float(t1) > t - settle:
            return False
    return True


def wind_now(pv):
    """privileged: the wind's force on each flying robot now [N] (zero without the wind in the true state)."""
    w = ((pv or {}).get("wind") or {}).get("uav")
    return np.array(w, dtype=float) if w else np.zeros(3)


def pour_shift(drop, rate, drop2=0.0, rate2=0.0):
    """privileged: the formation's horizontal offset (and its rate) that keeps the bucket's rest where it hung level,
    with the pouring hook `drop` metres down and the second robot's `drop2`: toward the pourer (bearing 330 degrees) as
    its rope slackens, then toward bearing 240 degrees as the second comes down."""
    d1 = np.array([math.cos(math.radians(RING_BEARING[POURERS_PRIV[0]])), math.sin(math.radians(RING_BEARING[POURERS_PRIV[0]]))])
    d2 = -np.array([math.cos(math.radians(SECOND_BEARING)), math.sin(math.radians(SECOND_BEARING))])
    if drop >= POUR_SLACK:
        s1, r1 = POUR_SHIFT * POUR_SLACK, 0.0
    elif drop <= 0.0:
        s1, r1 = 0.0, 0.0
    else:
        s1, r1 = POUR_SHIFT * drop, POUR_SHIFT * rate
    s2, r2 = SECOND_SHIFT * max(0.0, drop2), (SECOND_SHIFT * rate2 if drop2 > 0.0 else 0.0)
    return s1 * d1 + s2 * d2, r1 * d1 + r2 * d2


KEY_TOL = (0.06, 0.15)  # [m] the most any position may move between one step's observation and the next's (with the true
                        # state; on the odometry, its noise besides): a query further from every answer of the last step
                        # is a shifted copy (the lock shifts every robot's position by up to 0.4 m)


class Controller:
    def __init__(self):
        self.state = None
        self.base, self.sure = None, True
        self.last_t, self.cands, self.prev = None, [], []
        self.vehicle = Vehicle()

    def reset(self):
        self.state = None
        self.base, self.sure = None, True
        self.last_t, self.cands, self.prev = None, [], []

    # -- set-up ---------------------------------------------------------------------------------------------------
    def init(self, obs):
        mis = obs["mission"]
        hooks = {n: np.array(mis["uav"][n]["hook"]) for n in UAVS}
        priv = "privileged" in obs
        self.vehicle.setup(mis, privileged=priv)
        s = dict(phase="WAIT", t_phase=0.0, last_t=-1, mis=mis, hooks=hooks, ugv=self.vehicle.init_state(), priv=priv,
                 pourers=POURERS_PRIV if priv else ("UAV_1",), pour_pre=POUR_PRE_PRIV if priv else POUR_PRE,
                 load_f={n: np.zeros(3) for n in UAVS}, ie={n: np.zeros(3) for n in UAVS},
                 taut={n: False for n in UAVS}, z_ref=None, preload_ok=0.0, trim={n: 0.0 for n in UAVS},
                 rel={}, bucket_xy=None, bucket_f=None, off_f=None, off_rate=np.zeros(2), fire_seen_out=0.0, carry=None, align=None,
                 rise=None, tz={n: 0.0 for n in UAVS}, vz_ref={n: 0.0 for n in UAVS}, z_lo=None, wind_last=np.zeros(3), then=None)
        # formation: each robot's HOOK stands out from the bucket along its own ring's bearing (the hook sits behind
        # the airframe's centre; the headings are fixed, so the root is the hook less that offset)
        s["form"] = {}
        for n in UAVS:
            b = math.radians(RING_BEARING[n])
            s["form"][n] = FORMATION_R * np.array([math.cos(b), math.sin(b)]) - hooks[n][:2]
        L = mis["rope_length"]
        rh = FORMATION_R - RING_R  # hook to ring, horizontally (the same bearing)
        s["z_taut"] = {n: RING_Z + math.sqrt(max(0.0, L * L - rh * rh)) - hooks[n][2] for n in UAVS}  # the root with the rope just taut
        s["z_cruise"] = mis["cruise_z"] - float(np.mean([hooks[n][2] for n in UAVS]))  # the hooks at the cruise height
        return s

    # -- estimation -----------------------------------------------------------------------------------------------
    def estimates(self, obs, s, dt):
        """Positions: the formation's shape from the lidars, its place from the mean of the odometries."""
        pos = {n: np.array(obs["uav"][n]["pos"]) for n in UAVS}
        vel = {n: np.array(obs["uav"][n]["vel"]) for n in UAVS}
        rel = s["rel"]
        for (n, m) in list(rel.keys()):  # propagate every pair with the odometry velocities (no lag while moving)
            rel[(n, m)] = rel[(n, m)] + (vel[m] - vel[n]) * dt
        seen = {}  # (n, m) -> true_m - true_n, as the lidars measured it this step (from either end)
        for n in UAVS:
            for v in obs["uav"][n].get("neighbors") or []:
                v = np.array(v)
                guess = {m: (rel[(n, m)] if (n, m) in rel else pos[m] - pos[n]) for m in UAVS if m != n}
                d, m = min((float(np.linalg.norm(guess[m] - v)), m) for m in guess)
                if d < 1.0:
                    seen.setdefault((n, m), []).append(v)
                    seen.setdefault((m, n), []).append(-v)
        a = min(1.0, dt / 0.4)  # the returns are noisy (a few cm); the propagation carries the motion
        for key, vs in seen.items():
            v = np.mean(vs, axis=0)
            rel[key] = v if key not in rel else (1 - a) * rel[key] + a * v
        c = np.mean([pos[n] for n in UAVS], axis=0)
        est = {}
        for n in UAVS:
            others = [rel.get((n, m)) for m in UAVS if m != n]
            if all(o is not None for o in others):
                est[n] = c - np.sum(others, axis=0) / 3.0  # true_n - centroid = -(1/3) sum_m (true_m - true_n)
            else:
                est[n] = pos[n]  # no lidar picture yet: the odometry alone
        return est, vel

    # -- the robot's loop -----------------------------------------------------------------------------------------
    def thrust(self, n, p, v, p_ref, v_ref, s, obs, dt, tz=None, vz_ref=0.0, wind=None):
        """The robot's thrust: a PID position loop with the rope's load fed forward; in tension mode (`tz`, newtons)
        the vertical thrust commands the rope's tension instead and the height is left free (damped). With the true
        wind (`wind`, the force on the robot now) the load measured over the last step less the wind of that step is
        the rope's, filtered as before, and the wind now is fed forward whole (no lag at a gust's edges)."""
        m = s["mis"]["uav"][n]["mass"]
        load = np.array(obs["uav"][n]["load"])
        if wind is not None:
            load = load - s["wind_last"]
        s["load_f"][n] = 0.7 * s["load_f"][n] + 0.3 * load
        ff = -s["load_f"][n]
        ff[2] = max(0.0, ff[2])  # the rope only pulls down; never feed a ground reaction forward
        zmax = 120.0 if s.get("priv") else 60.0  # (with the true state one robot may carry most of the bucket while it pours)
        ff = np.clip(ff, [-60.0, -60.0, -60.0], [60.0, 60.0, zmax])
        if wind is not None:
            ff = ff - wind
        e = p_ref - p
        ie = np.clip(s["ie"][n] + e * dt, -0.5, 0.5)
        acc = KP * e + KD * (v_ref - v) + KI * ie
        f = m * (np.array([0.0, 0.0, G]) + acc) + ff
        if tz is not None:
            f[2] = m * G + tz - m * KD_T * (v[2] - vz_ref)
            ie[2] = 0.0
        tmax = s["mis"]["uav"][n]["thrust_max"]
        mag = float(np.linalg.norm(f))
        if mag > tmax:
            f = f * tmax / mag
        else:
            s["ie"][n] = ie  # (no wind-up while saturated)
        return [float(x) for x in f]

    def trim(self, s, dt):
        """Keep the three measured loads equal: a robot whose rope pulls more than the mean comes down a little. (Not with
        the true state: the true heights are equal by command, and a trim on the loads would only follow their noise.)"""
        if s.get("priv"):
            return
        lz = {n: float(s["load_f"][n][2]) for n in UAVS}
        mean = sum(lz.values()) / 3.0
        for n in UAVS:
            s["trim"][n] = float(np.clip(s["trim"][n] + K_TRIM * (lz[n] - mean) * dt, -0.06, 0.06))

    # -- the carry's legs ---------------------------------------------------------------------------------------------
    @staticmethod
    def leg(p0, p1, vmax=None):
        """A minimum-jerk move along a straight leg: under the peak acceleration A_CARRY and, with `vmax`, under that
        peak speed (a minimum-jerk move peaks at 1.875 d / T)."""
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        dist = max(1e-6, float(np.linalg.norm(p1 - p0)))
        T = t_minjerk(dist, A_CARRY)
        if vmax:
            T = max(T, 1.875 * dist / vmax)
        return dict(p0=p0, p1=p1, dist=dist, dirn=(p1 - p0) / dist, T=T)

    @staticmethod
    def legs(p0, p1, pv):
        """The straight line p0 -> p1, or two legs by a point beside the vehicle when the line passes within
        VEHICLE_ROOM of it (the bucket hangs about a metre above the floor; the vehicle's arm reaches higher)."""
        u = pv.get("ugv") or {}
        if not u.get("pos"):
            return [(p0, p1)]
        v = np.array(u["pos"][:2])
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return [(p0, p1)]
        f = float(np.clip((v - p0) @ d / (L * L), 0.0, 1.0))
        near = p0 + f * d
        gap = float(np.linalg.norm(v - near))
        if gap >= VEHICLE_ROOM or f <= 0.0 or f >= 1.0:
            return [(p0, p1)]
        side = near - v if gap > 1e-6 else np.array([-d[1], d[0]]) / L
        via = v + side / (np.linalg.norm(side) + 1e-9) * (VEHICLE_ROOM + 0.2)
        return [(p0, via), (via, p1)]

    @staticmethod
    def legs_time(legs):
        """How long a carry along these legs lasts [s]: the minimum-jerk moves, the pauses between them and after."""
        return (sum(t_minjerk(float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))), A_CARRY) for a, b in legs)
                + 0.8 * (len(legs) - 1) + 1.5)

    @staticmethod
    def hoist_info(s, pv):
        """What the vehicle's program is told of the hoist (with the true state): its phase; the stretches of the floor
        the bucket is about to cross (while it is carried: from where it is to the end of each leg still to fly, with
        the room the vehicle keeps from it); and whether it has settled for good (set down, or no true state)."""
        keep = []
        if pv is not None and s["phase"] == "CARRY" and s.get("carry") is not None:
            pts = [np.array(pv["bucket"]["pos"][:2], dtype=float), np.asarray(s["carry"]["p1"], dtype=float)]
            pts += [np.asarray(b, dtype=float) for _, b in s.get("carry_legs") or []]
            keep = [[float(a[0]), float(a[1]), float(b[0]), float(b[1]), BUCKET_ROOM] for a, b in zip(pts[:-1], pts[1:])]
        return dict(phase=s["phase"], keep=keep, settled=pv is None or s["phase"] == "HOLD")

    # -- the mission ----------------------------------------------------------------------------------------------
    @staticmethod
    def obs_key(obs):
        """The positions a step's observation reports: the true ones with the true state (exact), else the odometry's;
        and which of the two."""
        pv = obs.get("privileged") or {}
        v = []
        if pv.get("uav"):
            for n in UAVS:
                v += list(((pv.get("uav") or {}).get(n) or {}).get("pos") or [0.0, 0.0, 0.0])
            if pv.get("ugv"):
                v += list(pv["ugv"]["pos"][:2])
            return np.array(v, dtype=float), 0
        for n in UAVS:
            v += list((obs.get("uav") or {}).get(n, {}).get("pos") or [0.0, 0.0, 0.0])
        if obs.get("ugv"):
            v += list(obs["ugv"]["pose"][:2])
        return np.array(v, dtype=float), 1

    def pick(self, key):
        """The last step's answer this query continues: the state that step's query left whose observation lies
        nearest this one's, and whether it lies near enough to be the one (else this query is a shifted copy)."""
        if not self.prev:
            return self.state, True
        dist = [float(np.max(np.abs(k[0] - key[0]))) if len(k[0]) == len(key[0]) and k[1] == key[1] else 1e9 for k, _ in self.prev]
        i = int(np.argmin(dist))
        return self.prev[i][1], dist[i] < KEY_TOL[key[1]]

    def act(self, t, obs):
        """One step. A step may be asked more than once with the same t (the grader's lock asks some steps again
        with shifted positions, and applies only one answer): each query is answered from the state the last step
        left, and the state carried on is the one whose observation this step's continues. A query that continues
        none of the last step's answers (a shifted copy) is answered from the nearest; the next query with the same t
        (the true one) then picks again: a state computed on a shifted copy is never carried on."""
        if self.state is None or t == 0 and self.last_t != 0:
            self.state = self.init(obs)
            self.base, self.sure = None, True
            self.last_t, self.cands, self.prev = None, [], []
        key = self.obs_key(obs)
        if t != self.last_t:
            if self.cands:
                self.prev = self.cands
            self.cands, self.last_t = [], t
            self.base, self.sure = self.pick(key)
        elif not self.sure:
            base, sure = self.pick(key)
            if sure:
                self.base, self.sure = base, True
        s = copy.deepcopy(self.base)
        out = self.step(t, obs, s)
        self.cands.append((key, s))
        return out

    def step(self, t, obs, s):
        mis = s["mis"]
        dt = mis["dt"]
        s["t_phase"] += dt
        pv = obs.get("privileged") if s["priv"] else None
        if pv is not None:  # the true state, in place of what the odometry, the lidars and the cameras give
            pos = {n: np.array(pv["uav"][n]["pos"]) for n in UAVS}
            vel = {n: np.array(pv["uav"][n]["vel"]) for n in UAVS}
            bucket_est, pan_est, fire_seen = [np.array(pv["bucket"]["pos"])], [np.array(pv["pan"])], [bool(pv["fire"])]
        else:
            pos, vel = self.estimates(obs, s, dt)
            # the bucket and the pan as the cameras see them (world estimates through the position estimates)
            bucket_est, pan_est, fire_seen = [], [], []
            for n in UAVS:
                u = obs["uav"][n]
                if u.get("bucket") is not None:
                    bucket_est.append(pos[n] + np.array(u["bucket"]))
                if u.get("pan") is not None:
                    pan_est.append(pos[n] + np.array(u["pan"]))
                    fire_seen.append(bool(u.get("fire")))
        cen_est = np.mean([pos[n] for n in UAVS], axis=0)
        if bucket_est:
            s["bucket_xy"] = np.mean(bucket_est, axis=0)[:2]
            # a slow filter of it: where the formation centres itself for the take-off (the cameras, not the prior)
            s["bucket_f"] = s["bucket_xy"].copy() if s["bucket_f"] is None else s["bucket_f"] + min(1.0, dt / 1.0) * (s["bucket_xy"] - s["bucket_f"])
        bucket_xy = s["bucket_xy"] if s["bucket_xy"] is not None else BUCKET_START
        cxy = s["bucket_f"] if s["bucket_f"] is not None else BUCKET_START  # the formation's centre until the carry
        # the bucket's horizontal offset from the robots' centroid, filtered twice; its rate damps the swing
        off = bucket_xy - cen_est[:2]
        a = min(1.0, dt / 0.25)
        if s["off_f"] is None:
            s["off_f"] = off.copy()
        prev = s["off_f"]
        s["off_f"] = prev + a * (off - prev)
        s["off_rate"] = s["off_rate"] + a * ((s["off_f"] - prev) / dt - s["off_rate"])
        damp = np.clip(0.6 * s["off_rate"], -0.25, 0.25)
        if pv is not None and os.environ.get("ARB_REFERENCE_LOG") and s["phase"] != "WAIT" and int(round(obs["time"] / dt)) % 12 == 0:
            hooks_c = np.mean([np.array(pv["hooks"][n]) for n in UAVS], axis=0)
            log(f"trace t={obs['time']:.2f} {s['phase']} bucket={np.round(pv['bucket']['pos'], 3).tolist()} tilt={pv['bucket']['tilt_deg']:.1f} "
                f"off={np.round(bucket_xy - hooks_c[:2], 3).tolist()} z={[round(float(pos[n][2]), 3) for n in UAVS]} "
                f"load={[round(float(obs['uav'][n]['load'][2]), 1) for n in UAVS]} water={pv['bucket']['water_l']:.2f}"
                f" wind={[round(float(v), 2) for v in ((pv.get('wind') or {}).get('bucket') or [0.0, 0.0])[:2]]}")
        pourers = s["pourers"]
        pour_dir = np.sum([[math.cos(math.radians(RING_BEARING[n])), math.sin(math.radians(RING_BEARING[n]))] for n in pourers], axis=0)
        pour_dir = pour_dir / (np.linalg.norm(pour_dir) + 1e-9)  # (toward the lowered robots' side)

        def form(n, cxy, z):
            return np.array([cxy[0] + s["form"][n][0], cxy[1] + s["form"][n][1], z])

        def waiting(why):  # (a maintainer's record of what the hoist waits for, once per wait)
            if s.get("waiting") != [phase, why]:
                s["waiting"] = [phase, why]
                log(f"t={obs['time']:.1f} hoist {phase}: waiting ({why}; gusts "
                    f"{[[g.get('t0_s'), g.get('t1_s')] for g in ((pv or {}).get('wind') or {}).get('gusts') or []]})")

        refs, vrefs = {}, {}
        ugv_cmd = self.vehicle.act(s["ugv"], obs, self.hoist_info(s, pv)) if "ugv" in obs else {}
        phase = s["phase"]
        if phase == "WAIT":  # on the pads until the steam is off: the pour hover is inside the plume
            done_t = s["ugv"].get("lever_done_t")
            gauge = (obs.get("ugv") or {}).get("gauge")
            if pv is not None:
                go = not pv["steam"]
            else:
                go = (done_t is not None and obs["time"] > done_t + STEAM_WAIT) or (gauge is not None and gauge < 0.05)
            if go:
                s["phase"], s["t_phase"] = "RISE", 0.0
            return {"thrust": {n: [0.0, 0.0, 0.0] for n in UAVS}, "ugv": ugv_cmd}
        if phase == "RISE":  # a minimum-jerk move from the pad to the formation point just below the taut height
            if s["rise"] is None:
                starts = {n: np.array(mis["uav"][n]["start"], dtype=float) for n in UAVS}
                goals = {n: form(n, cxy, s["z_taut"][n] - 0.10) for n in UAVS}
                T = max([2.5] + [t_minjerk(float(np.linalg.norm(goals[n] - starts[n])), A_RISE) for n in UAVS])
                s["rise"] = dict(starts=starts, T=T)
            frac, rate = minjerk(s["t_phase"], s["rise"]["T"])
            for n in UAVS:
                goal = form(n, cxy, s["z_taut"][n] - 0.10)  # (the goal follows the cameras' bucket as it comes into view)
                d = goal - s["rise"]["starts"][n]
                refs[n] = s["rise"]["starts"][n] + frac * d
                vrefs[n] = rate * d
            if s["t_phase"] > s["rise"]["T"] + 0.5 and all(abs(pos[n][2] - (s["z_taut"][n] - 0.10)) < 0.08 for n in UAVS):
                s["phase"], s["t_phase"] = "TENSION", 0.0
        elif phase == "TENSION":  # tension mode: climb gently until each rope catches, then every rope to the same preload
            ok = all(s["taut"].values())
            for n in UAVS:
                rope_w = mis["uav"][n]["rope_mass"] * G
                t_meas = -float(s["load_f"][n][2])
                target = rope_w + T_PRELOAD
                if not s["taut"][n]:
                    s["tz"][n], s["vz_ref"][n] = rope_w + 2.0, V_CATCH  # (the slack rope hangs from the hook: its weight is a load)
                    if t_meas > rope_w + 4.0:
                        s["taut"][n] = True
                        s["tz"][n] = max(rope_w + 2.0, t_meas)
                else:
                    s["vz_ref"][n] = 0.0
                    s["tz"][n] = min(target, s["tz"][n] + T_RAMP * dt)
                    ok = ok and abs(t_meas - target) < 4.0  # (the tensions are equal by command; this only waits for the ramp)
                refs[n] = form(n, cxy, pos[n][2])  # (the height is not controlled here)
                vrefs[n] = np.zeros(3)
            s["preload_ok"] = s["preload_ok"] + dt if ok else 0.0
            # (with the true state: not into a gust; the bucket stays preloaded on the floor while one is due)
            ready = s["preload_ok"] > 0.3 or s["t_phase"] > 12.0
            if ready and pv is not None and not calm_for(pv, obs["time"], GUST_LIFT):
                waiting("a gust due over the lift-off")
                ready = False
            if ready:
                s["phase"], s["t_phase"] = "LIFTOFF", 0.0
                if pv is not None:
                    s["ie"] = {n: np.zeros(3) for n in UAVS}
                s["z_lo"] = float(np.mean([pos[n][2] for n in UAVS]))
        elif phase == "LIFTOFF":  # the three tensions ramp together until the bucket leaves the floor (level: equal pulls)
            for n in UAVS:
                s["tz"][n] += T_LIFT_RAMP * dt
                s["vz_ref"][n] = 0.0
                refs[n] = form(n, cxy, pos[n][2])
                vrefs[n] = np.zeros(3)
            rope_w = mis["uav"]["UAV_1"]["rope_mass"] * G
            if float(np.mean([pos[n][2] for n in UAVS])) - s["z_lo"] > 0.04 or s["tz"]["UAV_1"] - rope_w > T_LIFT_CAP:
                s["phase"], s["t_phase"] = "LIFT", 0.0
                s["lift_xy"] = np.array(cxy, dtype=float)  # (frozen: once the bucket hangs, the formation must not chase it)
                s["lift_z0"] = {n: float(pos[n][2]) for n in UAVS}
                s["T_lift"] = t_minjerk(s["z_cruise"] - float(np.mean(list(s["lift_z0"].values()))), A_LIFT)
        elif phase == "LIFT":  # all three up together, a minimum-jerk climb to the cruise height, then a short hold
            if s["t_phase"] > s["T_lift"]:
                self.trim(s, dt)  # (only once the climb is over: the loads are dynamic while it accelerates)
            frac, rate = minjerk(s["t_phase"], s["T_lift"])
            for n in UAVS:
                h = s["z_cruise"] - s["lift_z0"][n]
                refs[n] = form(n, s["lift_xy"] + damp, s["lift_z0"][n] + frac * h + s["trim"][n])
                vrefs[n] = np.array([0.0, 0.0, rate * h])
            go = s["t_phase"] > s["T_lift"] + 1.5
            if go and pv is not None:  # straight to above today's pan, round the vehicle if need be (a gust on the way is
                # ridden out, the wind fed forward: a bucket held up for it only rocks water out, and its swing counts the same)
                legs = self.legs(s["lift_xy"].copy(), np.array(pv["pan"][:2]) + s["pour_pre"] * pour_dir, pv)
            if go:
                s["phase"], s["t_phase"], s["then"] = "CARRY", 0.0, "ALIGN"
                if pv is not None:
                    s["carry_legs"] = legs
                else:
                    s["carry_legs"] = [(s["lift_xy"].copy(), np.array(mis["pan"], dtype=float))]
                s["carry_vmax"] = V_CARRY_PRIV if pv is not None else None
                s["carry"] = self.leg(*s["carry_legs"].pop(0), vmax=s["carry_vmax"])
        elif phase == "CARRY":  # the formation's centre rides minimum-jerk moves along straight legs
            self.trim(s, dt)
            c = s["carry"]
            frac, rate = minjerk(s["t_phase"], c["T"])
            cen = c["p0"] + frac * c["dist"] * c["dirn"]
            v = rate * c["dist"] * c["dirn"]
            for n in UAVS:
                refs[n] = form(n, cen + damp, s["z_cruise"] + s["trim"][n])
                vrefs[n] = np.array([v[0], v[1], 0.0])
            if s["t_phase"] > c["T"] + (0.8 if s.get("carry_legs") else 1.5):
                if s.get("carry_legs"):
                    s["carry"], s["t_phase"] = self.leg(*s["carry_legs"].pop(0), vmax=s.get("carry_vmax")), 0.0
                else:
                    s["phase"], s["t_phase"] = s.get("then") or "ALIGN", 0.0
                    s["align"] = c["p1"].copy()
                    s["carry"] = None
        elif phase == "ALIGN" and pv is not None:  # over its mark, hanging still, and no gust due while the pour lasts
            target = np.array(pv["pan"][:2]) + s["pour_pre"] * pour_dir
            err = target - s["bucket_f"]  # (the slow filter: the formation does not chase the swing)
            s["align"] = s["align"] + np.clip(0.8 * err * dt, -0.06 * dt, 0.06 * dt)
            hooks_c = np.mean([np.array(pv["hooks"][n][:2]) for n in UAVS], axis=0)
            still = (float(np.linalg.norm(target - bucket_xy)) < 0.06 and float(np.linalg.norm(bucket_xy - hooks_c)) < 0.05
                     and float(np.linalg.norm(np.array(pv["bucket"]["vel"][:2]))) < 0.12)
            s["still_t"] = s.get("still_t", 0.0) + dt if still else 0.0
            calm = calm_for(pv, obs["time"], POUR_WINDOW)
            if not calm:  # (the wait for the wind is not the alignment's: its clock starts again once the gust has gone)
                s["t_phase"] = 0.0
                waiting("a gust due while the pour lasts")
            for n in UAVS:
                refs[n] = form(n, s["align"] + damp, s["z_cruise"])
                vrefs[n] = np.zeros(3)
            if calm and (s["still_t"] >= ALIGN_STILL or s["t_phase"] > ALIGN_MAX):
                s["phase"], s["t_phase"] = "POUR", 0.0
                s["pour_clock"], s["drop"], s["drop2"], s["still_t"] = 0.0, 0.0, 0.0, 0.0
        elif phase == "ALIGN":  # servo the bucket over the pan on what the cameras see (drift-free), gently
            self.trim(s, dt)
            aligned = False
            if pan_est and bucket_est:
                err = np.mean(pan_est, axis=0)[:2] + s["pour_pre"] * pour_dir - bucket_xy
                s["align"] = s["align"] + np.clip(0.8 * err * dt, -0.06 * dt, 0.06 * dt)
                aligned = float(np.linalg.norm(err)) < 0.08
            for n in UAVS:
                refs[n] = form(n, s["align"] + damp, s["z_cruise"] + s["trim"][n])
                vrefs[n] = np.zeros(3)
            if (aligned and s["t_phase"] > 1.5) or s["t_phase"] > 8.0:
                s["phase"], s["t_phase"] = "POUR", 0.0
        elif phase == "POUR" and pv is not None:
            # UAV_3's hook comes down (its clock stops while a gust blows or is about to), the formation moving toward it
            # by as much as the bucket would swing away; if the bucket tips too little to pour, UAV_2 comes down slowly
            # too (the formation moving on); after the lowering, a slow correction on the bucket's filtered place
            gust = not calm_for(pv, obs["time"], GUST_SOON)  # (and on again once the swing it left has had GUST_SETTLE)
            if not gust:
                s["pour_clock"] += dt
            else:
                waiting("a gust: the pouring hook stops")
            frac, rate = minjerk(s["pour_clock"], T_POUR_PRIV)
            if gust:
                rate = 0.0
            s["drop"] = frac * POUR_DROP_PRIV
            hist = (s.get("water_hist") or []) + [[obs["time"], float(pv["bucket"]["water_l"])]]
            s["water_hist"] = hist = [h for h in hist if obs["time"] - h[0] <= 0.5]
            pouring = hist[0][1] - hist[-1][1] > 0.01  # (more than 0.02 L/s leaving)
            v2 = 0.0
            if s["pour_clock"] > T_POUR_PRIV + 2.0 and pv["fire"] and not pouring and not gust and s["drop2"] < SECOND_MAX:
                s["drop2"] = min(SECOND_MAX, s["drop2"] + V_SECOND * dt)
                v2 = V_SECOND
            if s["pour_clock"] > T_POUR_PRIV:
                err = np.array(pv["pan"][:2]) - s["bucket_f"]
                s["align"] = s["align"] + np.clip(0.8 * err * dt, -0.03 * dt, 0.03 * dt)
            shift, shift_v = pour_shift(s["drop"], rate * POUR_DROP_PRIV, s["drop2"], v2)
            for n in UAVS:
                z, vz = s["z_cruise"], 0.0
                if n in pourers:
                    z, vz = z - s["drop"], -rate * POUR_DROP_PRIV
                elif n == SECOND_PRIV:
                    z, vz = z - s["drop2"], -v2
                refs[n] = form(n, s["align"] + shift + damp, z)
                vrefs[n] = np.array([shift_v[0], shift_v[1], vz])
            s["fire_seen_out"] = s["fire_seen_out"] + dt if not pv["fire"] else 0.0
            if s["fire_seen_out"] > 0.5 or s["pour_clock"] > 40.0:
                s["phase"], s["t_phase"] = "RECOVER", 0.0
                s["recover_drop"], s["recover_drop2"], s["water_hist"] = s["drop"], s["drop2"], []
        elif phase == "POUR":  # one robot lowers its hook: the bucket tips; the thermal cameras say when the fire is out
            if pan_est and bucket_est:  # (a gentle correction only: the bucket swings towards the pan as its rope slackens)
                err = np.mean(pan_est, axis=0)[:2] - bucket_xy
                s["align"] = s["align"] + np.clip(0.8 * err * dt, -0.04 * dt, 0.04 * dt)
            frac, rate = minjerk(s["t_phase"], T_POUR)
            for n in UAVS:
                z = s["z_cruise"] + s["trim"][n] - (frac * POUR_DROP if n in pourers else 0.0)
                refs[n] = form(n, s["align"] + damp, z)
                vrefs[n] = np.array([0.0, 0.0, -rate * POUR_DROP if n in pourers else 0.0])
            if fire_seen and not any(fire_seen):
                s["fire_seen_out"] += dt
            else:
                s["fire_seen_out"] = 0.0
            if s["fire_seen_out"] > 0.5 or s["t_phase"] > 15.0:
                s["phase"], s["t_phase"] = "RECOVER", 0.0
                s["recover_drop"] = frac * POUR_DROP
                s["recover_drop2"], s["drop2"], s["water_hist"] = 0.0, 0.0, []
        elif phase == "RECOVER" and pv is not None:
            # both hooks back up, the formation back with them (the bucket levels where it poured); then, the fire out,
            # away from the station; still burning: another pour, at most three in all
            frac, rate = minjerk(s["t_phase"], T_RECOVER_PRIV)
            d1, d2 = (1.0 - frac) * s["recover_drop"], (1.0 - frac) * s["recover_drop2"]
            shift, shift_v = pour_shift(d1, -rate * s["recover_drop"], d2, -rate * s["recover_drop2"])
            for n in UAVS:
                z, vz = s["z_cruise"], 0.0
                if n in pourers:
                    z, vz = z - d1, rate * s["recover_drop"]
                elif n == SECOND_PRIV:
                    z, vz = z - d2, rate * s["recover_drop2"]
                refs[n] = form(n, s["align"] + shift + damp, z)
                vrefs[n] = np.array([shift_v[0], shift_v[1], vz])
            if s["t_phase"] > T_RECOVER_PRIV + 1.0:
                if pv["fire"] and s.get("pours", 1) < 3:
                    s["pours"] = s.get("pours", 1) + 1
                    s["phase"], s["t_phase"] = "ALIGN", 0.0
                elif not pv["fire"]:
                    s["phase"], s["t_phase"] = "PARK", 0.0
        elif phase in ("PARK", "DOWN", "HOLD") and pv is None:  # (reached with the true state only: should it go, hold here)
            for n in UAVS:
                refs[n] = form(n, s["align"], s.get("hold_z") or s["z_cruise"])
                vrefs[n] = np.zeros(3)
        elif phase == "PARK":  # (the true state only) the fire out: east, away from the station, once no gust will blow on the way
            p0 = s["align"].copy()
            p1 = np.array([max(p0[0] + PARK_DX, PARK_X), float(np.clip(p0[1], *PARK_Y))])
            legs = self.legs(p0, p1, pv)
            for n in UAVS:
                refs[n] = form(n, s["align"] + damp, s["z_cruise"])
                vrefs[n] = np.zeros(3)
            if not calm_for(pv, obs["time"], self.legs_time(legs)):
                waiting("a gust due on the way to the park")
            else:
                s["phase"], s["t_phase"], s["then"] = "CARRY", 0.0, "DOWN"
                s["carry_legs"], s["carry_vmax"] = legs, None
                s["carry"] = self.leg(*s["carry_legs"].pop(0))
        elif phase == "DOWN":  # (the true state only) the bucket set down: quickly to just above the floor, then slowly
            zb = float(pv["bucket"]["pos"][2])
            if s.get("down") is None:
                dz = max(0.0, zb - DOWN_LOW)
                s["down"] = dict(dz=dz, T=t_minjerk(dz, A_DOWN), slow=0.0, touch=None, go=False)
            d = s["down"]
            if not d["go"]:  # (not into a gust: the bucket would land swinging)
                d["go"] = calm_for(pv, obs["time"], d["T"] + DOWN_LOW / V_DOWN + 1.0)
                s["t_phase"] = 0.0
                if not d["go"]:
                    waiting("a gust due over the set-down")
            frac, rate = minjerk(s["t_phase"], d["T"]) if d["go"] else (0.0, 0.0)
            z, vz = s["z_cruise"] - frac * d["dz"], -rate * d["dz"]
            if d["go"] and s["t_phase"] > d["T"]:
                d["slow"] += V_DOWN * dt
                z, vz = z - d["slow"], -V_DOWN
                if d["touch"] is None and zb < 0.02:
                    d["touch"] = d["slow"]
                if (d["touch"] is not None and d["slow"] - d["touch"] >= DOWN_SLACK) or d["slow"] >= DOWN_MAX:
                    s["phase"], s["t_phase"] = "HOLD", 0.0
                    s["hold_z"] = z
            for n in UAVS:
                refs[n] = form(n, s["align"] + damp, z)
                vrefs[n] = np.array([0.0, 0.0, vz])
        elif phase == "HOLD":  # (the true state only) over the bucket on the floor, the ropes slack, for the rest of the run
            for n in UAVS:
                refs[n] = form(n, s["align"], s["hold_z"])
                vrefs[n] = np.zeros(3)
        else:  # RECOVER without the true state: level again and hover, for the rest of the run
            frac, rate = minjerk(s["t_phase"], T_RECOVER)
            if s["t_phase"] > T_RECOVER + 0.5:
                self.trim(s, dt)
            # still burning (the thermal cameras say so) and water left: another pour, at most three in all
            if s["t_phase"] > T_RECOVER + 3.0 and fire_seen and any(fire_seen) and s.get("pours", 1) < 3:
                s["pours"] = s.get("pours", 1) + 1
                s["phase"], s["t_phase"] = "ALIGN", 0.0
            for n in UAVS:
                z = s["z_cruise"] + s["trim"][n]
                vz = 0.0
                if n in pourers:
                    z -= (1.0 - frac) * s["recover_drop"]
                    vz = rate * s["recover_drop"]
                refs[n] = form(n, s["align"] + damp, z)
                vrefs[n] = np.array([0.0, 0.0, vz])
        if s["phase"] != phase:
            log(f"t={obs['time']:.1f} hoist {phase} -> {s['phase']}" + (f" (bucket at {np.round(bucket_xy, 2).tolist()})" if bucket_xy is not None else ""))
        tension_mode = s["phase"] in ("TENSION", "LIFTOFF")
        wind = wind_now(pv) if pv is not None else None
        out = {"thrust": {n: self.thrust(n, pos[n], vel[n], refs[n], vrefs[n], s, obs, dt,
                                         tz=(s["tz"][n] if tension_mode else None), vz_ref=s["vz_ref"][n], wind=wind) for n in UAVS}}
        if wind is not None:
            s["wind_last"] = wind
        out["ugv"] = ugv_cmd
        return out


def main():
    ctl = Controller()
    try:  # (the vehicle's kinematics from the scene beside the program, read before READY: the first step stays short)
        kinematics(os.path.join(os.getcwd(), "scene.usda"), "base_link", "end_effector_link")
    except Exception as exc:  # noqa: BLE001
        log(f"no kinematics before READY: {type(exc).__name__}: {exc}")
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("type")
        if kind == "reset":
            ctl.reset()
            reply = {"ok": True}
        elif kind == "act":
            try:
                reply = {"action": ctl.act(int(msg["t"]), msg["obs"])}
            except Exception as exc:  # noqa: BLE001
                reply = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        elif kind == "close":
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
            sys.stdout.flush()
            return
        else:
            reply = {"error": "unknown request"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
