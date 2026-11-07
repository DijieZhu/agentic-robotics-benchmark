"""The privileged reference controller of fault_adaptive_spacecraft_docking.

solution/solve.sh installs it with the reference token (`.oracle_token`). With that token the grader adds to every
observation the simulator's current truth (`privileged`: every quantity env.step reads that is in effect now, named as
env.py names it; never a value drawn for the future), and this controller reads nothing else: not the 16-number vector,
not the clock, no random numbers, no threads, no numpy. It shows that the hidden episodes can be done with perfect
present knowledge; an agent never gets the block.

How it works, per request:
  * Plant replica. `Replica` is env.py's own FaultAdaptiveDocking (the copy beside this file) built from the block, with
    no observation and no event to come. Its `step` is the simulator's step code, so from a block it predicts the next
    block exactly, except across the two bus events, which the block shows one step after they fire.
  * Which request is real. A probed step is asked twice: the real observation and a copy whose true x, y and yaw are
    shifted. A request is verified when its block equals the replica's prediction from one of the previous step's
    (block, reply) pairs; the real one always is (after the re-key every transition is exact), the copy never is. Every
    request is answered from the state committed at the previous step plus its own block, and only a verified request
    commits, so the copy gets the same law applied to its shifted pose (it differs by genuine feedback) and never
    touches the plan.
  * Law. A reference point on the port's radial line, at a gap above the port, with the co-rotation and the boom's
    radial motion fed forward; PD tracking with damping compensation and acceleration caps; yaw tracking of the port's
    bearing and slew. Far field: staging at r = 1.60 m, outside the plume field and inside the re-key's proximity, with
    a waypoint that keeps the path off the target body. The relay trigger is pinned at full duty once per bus event.
    Allocation: bounded least squares in realized-thrust space (radial 2, tangential 1, yaw 3, in the craft's own frame)
    on the next step's true plant (rotor map, degradations, exponents, valve states, plume columns at the true pose with
    the current swirl signs), by 80 projected Gauss-Seidel sweeps from w = lo.
  * Plan. At t_ready (the first verified request after the re-key's relay reversal) it rolls the staging law
    PLAN_OFFSET steps ahead on the replica, giving the state at t_p, then rolls each policy of a fixed family of 266
    (descent delay, hold gap, descent speed, a load penalty on one nozzle, a relative-speed cap, a yaw offset; the first
    80 are the prototypes' two families, the other 186 are tried only when none of those docks) from there to the
    episode's end, in order, at most QUOTA simulated steps per verified request; at t_p it commits the first that docks
    (none docks: the best final quality). The planned rollout and the executed episode are the same computation, so a
    planned dock is an executed dock.

Run as `python3 -B oracle.py` in the submission folder (run.sh); `Oracle.handle` serves one wire line, for in-process
use by tests/review.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spacecraft_docking_env import DT, PORT_RADIUS, FaultAdaptiveDocking  # noqa: E402

PRIV_FORMAT = "fault_adaptive_spacecraft_docking privileged 1"
STATE = ("steps", "x", "y", "vx", "vy", "theta", "omega", "phase", "phase_rate", "port_extension_phase", "fuel",
         "routing_active", "rekey_active", "routing_relay_armed", "nozzle_exponents", "valve_memory", "slot_valves",
         "plume_swirl", "plume_energy", "first_relay_reversals", "rekey_relay_reversals", "hold_steps", "best_error")
ROUTING = ("routing_order", "routing_phase", "routing_direction", "failure_nozzle", "secondary_failure_nozzle",
           "failure_scale", "secondary_failure_scale")

# guidance
STAGE_R = 1.60              # staging radius on the port's radial line (outside the plume field, which starts at 1.47)
NEAR_R = 1.52               # inside it the near-field caps apply
KP, KD, KTH, KW = 0.25, 1.0, 1.0, 2.0
KP_KD = KP / KD
AMAX_FAR, AMAX_NEAR = 0.15, 0.08
ALPHA_FAR, ALPHA_NEAR = 0.08, 0.05
VREL_MAX = 0.20             # the relative-speed cap of the "v4" policies while they descend
DAMP_V = (1.0 / 0.998 - 1.0) / DT
DAMP_W = (1.0 / 0.996 - 1.0) / DT
CHORD_R = 1.45              # a straight path to the reference that dips below this radius goes via a waypoint
WAY_STEP, WAY_R_MIN, WAY_R_MAX = 1.0, 1.75, 2.4   # (at most 1 rad ahead: a 0.6 rad carrot chased a fast port
                                                 # too slowly and let the re-key fire at its deadline)
# allocation
W_R, W_T, W_Y = 2.0, 1.0, 3.0
RIDGE = 1e-6
PENALTY = 0.05
SWEEPS = 80
# planning
PLAN_OFFSET = 8
QUOTA = 10000               # simulated steps per verified request before t_p: 8 x 10000 covers the family
UNLIMITED = 10 ** 12
HOLD = 30
STEPS = FaultAdaptiveDocking.max_steps


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Replica(FaultAdaptiveDocking):
    """env.py's plant built from a block: its own step code, no observation, no pose history, no event to come."""

    def _observe(self, force_valid=False):
        return None

    def _record_pose(self):
        pass


_NO_EVENT = {"failure_step": 10 ** 9, "rekey_earliest": 10 ** 9, "rekey_deadline": 10 ** 9, "rekey_proximity": 0.0,
             "success": False, "failure_reason": "", "rng": None}
_IDENTITY_ROUTING = {"routing_order": [0, 1, 2, 3, 4, 5], "routing_phase": 0, "routing_direction": 1,
                     "failure_nozzle": 0, "secondary_failure_nozzle": 1, "failure_scale": 1.0,
                     "secondary_failure_scale": 1.0}  # (never read before the first event: the plant is unrouted)


def replica(st):
    """A Replica of a block's state (its own copies of every list)."""
    r = Replica.__new__(Replica)
    d = r.__dict__
    d.update(_NO_EVENT)
    for k, v in st.items():
        if v is None:
            v = _IDENTITY_ROUTING[k]
        d[k] = list(v) if isinstance(v, list) else v
    return r


def clone(r):
    """A copy of a Replica that steps independently (step mutates these three lists in place)."""
    c = Replica.__new__(Replica)
    d = c.__dict__
    d.update(r.__dict__)
    d["slot_valves"] = list(r.slot_valves)
    d["plume_energy"] = list(r.plume_energy)
    d["plume_swirl"] = list(r.plume_swirl)
    return c


def same_state(r, st):
    """Whether a Replica's state is exactly a block's state (every float to the last bit)."""
    for k in STATE:
        if getattr(r, k) != st[k]:
            return False
    if st["routing_active"]:
        for k in ROUTING:
            if getattr(r, k) != st[k]:
                return False
    return True


class Variant(object):
    """A policy after t_p: when the descent starts, where it holds, how fast it descends, which nozzle it spares,
    whether its relative speed is capped, its yaw offset during the descent."""

    __slots__ = ("index", "family", "delay", "hold_gap", "speed", "penalty", "yaw_off", "vcap")

    def __init__(self, index, family, delay=0, hold_gap=0.0, speed=0.035, penalty=-1, yaw_off=0.0):
        self.index, self.family, self.delay, self.hold_gap = index, family, delay, hold_gap
        self.speed, self.penalty, self.yaw_off, self.vcap = speed, penalty, yaw_off, family == "v4"

    def describe(self):
        return {"index": self.index, "family": self.family, "delay": self.delay, "hold_gap": self.hold_gap,
                "speed": self.speed, "penalty_nozzle": self.penalty, "yaw_off": self.yaw_off}


def _family():
    out = []

    def add(family, **kw):
        out.append(Variant(len(out), family, **kw))
    for family in ("v2", "v4"):
        for delay in (0, 30, 60, 90):
            for hold_gap in (0.0, 0.045):
                for speed in (0.035, 0.022, 0.05):
                    add(family, delay=delay, hold_gap=hold_gap, speed=speed)
        for delay in (0, 45):
            for nozzle in range(6):
                add(family, delay=delay, penalty=nozzle)
    for delay in (0, 45):
        for yaw_off in (0.25, -0.25):
            for speed in (0.035, 0.022):
                add("v4", delay=delay, speed=speed, yaw_off=yaw_off)
    # the wider family, tried only when none of the first 80 docks: other descent delays, hold gaps and speeds, the
    # penalties at later delays, more yaw offsets, a yaw offset with a penalty
    for family in ("v2", "v4"):
        for delay in (15, 45, 75, 105, 120, 150):
            for hold_gap in (0.0, 0.045):
                for speed in (0.035, 0.022, 0.05):
                    add(family, delay=delay, hold_gap=hold_gap, speed=speed)
        for delay in (0, 30, 60, 90):
            for hold_gap in (0.02, 0.07):
                for speed in (0.035, 0.015):
                    add(family, delay=delay, hold_gap=hold_gap, speed=speed)
        for delay in (30, 60, 90):
            for nozzle in range(6):
                add(family, delay=delay, penalty=nozzle)
    for delay in (0, 30, 60, 90):
        for yaw_off in (0.12, -0.12, 0.4, -0.4):
            add("v4", delay=delay, yaw_off=yaw_off)
    for delay in (30, 60, 90):
        for yaw_off in (0.25, -0.25):
            add("v4", delay=delay, yaw_off=yaw_off)
    for delay in (0, 45):
        for yaw_off in (0.25, -0.25):
            for nozzle in range(6):
                add("v2", delay=delay, yaw_off=yaw_off, penalty=nozzle)
    return tuple(out)


FAMILY = _family()          # 36 "v2" policies, 44 "v4", then 186 wider ones: 266
STAGING = Variant(-1, "staging")


def _seg_min_radius(px, py, qx, qy):
    dx, dy = qx - px, qy - py
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px, py)
    s = -(px * dx + py * dy) / l2
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    return math.hypot(px + s * dx, py + s * dy)


def allocate(sim, bx, by, ba, aa, penalty, pin):
    """Commands for the wrench (bx, by, ba) in the body frame on the next step's true plant: bounded least squares in
    realized-thrust space, the rows weighted radial / tangential / yaw in the craft's own frame (aa: the bearing of its
    position in the body frame). `penalty`: a physical nozzle whose load is penalised (-1: none); `pin`: a slot held at
    full duty (-1: none)."""
    if sim.routing_active:
        order, ph = sim.routing_order, sim.routing_phase
        nz = [order[(i + ph) % 6] for i in range(6)]
        fn, sn = sim.failure_nozzle, sim.secondary_failure_nozzle
        fs, ss = sim.failure_scale, sim.secondary_failure_scale
    else:
        nz = [0, 1, 2, 3, 4, 5]
        fn = sn = -1
        fs = ss = 1.0
    expo, mem, z = sim.nozzle_exponents, sim.valve_memory, sim.slot_valves
    ca, sa = math.cos(aa), math.sin(aa)
    g0, g1, g2 = [0.0] * 6, [0.0] * 6, [0.0] * 6
    lo, hi, ex, zl = [0.0] * 6, [0.0] * 6, [1.0] * 6, [0.0] * 6
    columns = sim.columns
    for i in range(6):
        n = nz[i]
        col = sim._plume_column(columns[n], n)[0]
        s = fs if n == fn else (ss if n == sn else 1.0)
        cx, cy, ct = col[0] * s, col[1] * s, col[2] * s
        g0[i] = W_R * (ca * cx + sa * cy)
        g1[i] = W_T * (ca * cy - sa * cx)
        g2[i] = W_Y * ct
        e = expo[n]
        m = mem[i]
        zi = m * z[i]
        ex[i], zl[i] = e, zi
        lo[i] = zi ** e
        hi[i] = (zi + 1.0 - m) ** e
    if pin >= 0:
        lo[pin] = hi[pin] = (zl[pin] + (1.0 - mem[pin])) ** ex[pin]
    bias = sim.bias
    dx, dy, dt = bx - bias[0], by - bias[1], ba - bias[2]
    h0, h1, h2 = W_R * (ca * dx + sa * dy), W_T * (ca * dy - sa * dx), W_Y * dt
    Q = [[g0[i] * g0[j] + g1[i] * g1[j] + g2[i] * g2[j] for j in range(6)] for i in range(6)]
    for i in range(6):
        Q[i][i] += RIDGE
        if nz[i] == penalty:
            Q[i][i] += PENALTY
    q = [g0[i] * h0 + g1[i] * h1 + g2[i] * h2 for i in range(6)]
    w = list(lo)
    for _ in range(SWEEPS):
        for i in range(6):
            Qi = Q[i]
            v = w[i] + (q[i] - (Qi[0] * w[0] + Qi[1] * w[1] + Qi[2] * w[2] + Qi[3] * w[3] + Qi[4] * w[4]
                                + Qi[5] * w[5])) / Qi[i]
            w[i] = hi[i] if v > hi[i] else (lo[i] if v < lo[i] else v)
    u = [0.0] * 6
    for i in range(6):
        if i == pin:
            u[i] = 1.0
            continue
        wi = w[i] if w[i] > 0.0 else 0.0
        val = (wi ** (1.0 / ex[i]) - zl[i]) / (1.0 - mem[i])
        u[i] = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
    return u


def law(sim, v, descending, gap_prev):
    """(commands, gap): the policy `v` on the state `sim`. Staging (not descending): the reference sits at STAGE_R on
    the port's radial line. Descending: its gap above the port shrinks at v.speed from gap_prev down to v.hold_gap."""
    x, y, vx, vy, th = sim.x, sim.y, sim.vx, sim.vy, sim.theta
    amp, wb, psi = sim.port_amplitude, sim.port_omega, sim.port_extension_phase
    s_psi, c_psi = math.sin(psi), math.cos(psi)
    R = PORT_RADIUS + amp * s_psi
    Rd = amp * wb * c_psi
    Rdd = -amp * wb * wb * s_psi
    r = math.hypot(x, y)
    if descending:
        hold = v.hold_gap
        gap = max(hold, gap_prev - v.speed * DT) if gap_prev > hold else hold
        gap_rate = (gap - gap_prev) / DT
    else:
        gap = STAGE_R - R
        gap_rate = -Rd
    prate, phi = sim.phase_rate, sim.phase
    c_phi, s_phi = math.cos(phi), math.sin(phi)
    rr = R + gap
    rd = Rd + gap_rate
    px, py = rr * c_phi, rr * s_phi
    vt = rr * prate
    rvx, rvy = rd * c_phi - vt * s_phi, rd * s_phi + vt * c_phi
    ar, at = Rdd - rr * prate * prate, 2.0 * rd * prate
    rax, ray = ar * c_phi - at * s_phi, ar * s_phi + at * c_phi
    if not descending and _seg_min_radius(x, y, px, py) < CHORD_R:
        ang = math.atan2(y, x)
        dphi = _wrap(phi - ang)
        wa = ang + math.copysign(min(abs(dphi), WAY_STEP), dphi)
        rw = max(WAY_R_MIN, min(r, WAY_R_MAX))
        px, py = rw * math.cos(wa), rw * math.sin(wa)
        rvx = rvy = rax = ray = 0.0
    dvx, dvy = KP_KD * (px - x), KP_KD * (py - y)
    if descending and v.vcap:
        nv = math.hypot(dvx, dvy)
        if nv > VREL_MAX:
            f = VREL_MAX / nv
            dvx, dvy = dvx * f, dvy * f
    ax = rax + KD * (rvx + dvx - vx) + vx * DAMP_V
    ay = ray + KD * (rvy + dvy - vy) + vy * DAMP_V
    amax = AMAX_FAR if r > NEAR_R else AMAX_NEAR
    n = math.hypot(ax, ay)
    if n > amax:
        f = amax / n
        ax, ay = ax * f, ay * f
    yoff = v.yaw_off * min(1.0, max(0.0, (gap - 0.03) / 0.12)) if descending else 0.0
    alpha = KTH * _wrap(phi + yoff - th) + KW * (prate - sim.omega) + prate * DAMP_W
    am = ALPHA_FAR if r > NEAR_R else ALPHA_NEAR
    alpha = am if alpha > am else (-am if alpha < -am else alpha)
    c, s = math.cos(th), math.sin(th)
    bx, by = c * ax + s * ay, c * ay - s * ax
    pin = -1
    if sim.routing_active and sim.routing_relay_armed and (
            sim.rekey_relay_reversals == 0 if sim.rekey_active else sim.first_relay_reversals == 0):
        order, ph, fn = sim.routing_order, sim.routing_phase, sim.failure_nozzle
        for i in range(6):
            if order[(i + ph) % 6] == fn:
                pin = i
                break
    return allocate(sim, bx, by, alpha, math.atan2(y, x) - th, v.penalty, pin), gap


class Planner(object):
    """From the verified request at t_ready: the staging law rolled to t_p on the replica, then each policy of FAMILY
    from there to the episode's end, in order, under a quota of simulated steps per call. The first policy that docks
    is the plan; if none docks, the one with the best final quality (the lowest index on ties)."""

    def __init__(self, sim, t_p):
        self.t_p = t_p
        self.simulated = 0
        self.results = []           # (docked, steps, quality) of each policy rolled out, in order
        self.chosen = None
        self.k = 0
        self.cur = None
        r, gap, q, done = clone(sim), None, 0.0, False
        while r.steps < t_p:
            u, gap = law(r, STAGING, False, gap)
            _, q, done, _ = r.step(u)
            self.simulated += 1
            if done:
                break
        self.base, self.base_gap = r, gap
        if done:                    # the episode ends before t_p: nothing left to plan
            self.results.append((bool(r.success), r.steps, q))
            self.chosen = 0

    def work(self, quota):
        while quota > 0 and self.chosen is None:
            v = FAMILY[self.k]
            if self.cur is None:
                self.cur = (clone(self.base), self.base_gap)
            r, gap = self.cur
            start = self.t_p + v.delay
            end = None
            while quota > 0:
                u, gap = law(r, v, r.steps >= start, gap)
                _, q, done, _ = r.step(u)
                quota -= 1
                self.simulated += 1
                if done or r.steps + HOLD - r.hold_steps > STEPS:   # (ended, or it can no longer dock in time)
                    end = (bool(r.success), r.steps, q)
                    break
            if end is None:
                self.cur = (r, gap)
                return
            self.cur = None
            self.results.append(end)
            if end[0]:
                self.chosen = self.k
            elif self.k + 1 == len(FAMILY):
                self.chosen = max(range(len(self.results)), key=lambda i: (self.results[i][2], -i))
            else:
                self.k += 1


class Committed(object):
    """What a verified request commits: the reference gap of its step and the plan's progress."""

    __slots__ = ("gap", "t_ready", "t_p", "planner", "variant")

    def __init__(self, gap=None, t_ready=None, t_p=None, planner=None, variant=None):
        self.gap, self.t_ready, self.t_p, self.planner, self.variant = gap, t_ready, t_p, planner, variant


class Episode(object):
    """One episode's requests. Each is answered from the state committed at the previous step plus its own block."""

    def __init__(self):
        self.t = None
        self.pairs = []             # (block state, reply) answered at step self.t
        self.prev = []              # the same for step self.t - 1
        self.preds = None           # replicas predicted from self.prev
        self.base = Committed()     # committed by the last verified step before self.t
        self.next = None            # committed by the verified request at self.t
        self.stats = {"requests": 0, "verified": 0, "steps_unverified_after_ready": 0, "first_event_seen": None,
                      "rekey_seen": None}

    def _close_step(self):
        if self.next is not None:
            self.base = self.next
        elif self.t is not None and self.base.t_ready is not None and self.t > self.base.t_ready:
            self.stats["steps_unverified_after_ready"] += 1
        self.next = None

    def finish(self):
        self._close_step()
        self.t = None
        self.pairs, self.prev, self.preds = [], [], None

    def _verified(self, st):
        if not self.prev:
            return False
        if self.preds is None:
            self.preds = []
            for pst, pu in self.prev:
                r = replica(pst)
                r.step(pu)
                self.preds.append(r)
        for r in self.preds:
            if same_state(r, st):
                return True
        return False

    def _commit(self, c, sim, gap):
        n = Committed(gap, c.t_ready, c.t_p, c.planner, c.variant)
        if n.t_ready is None and sim.rekey_active and sim.rekey_relay_reversals >= 1:
            n.t_ready = sim.steps
            n.t_p = sim.steps + PLAN_OFFSET
            n.planner = Planner(sim, n.t_p)
        if n.planner is not None and n.variant is None:
            n.planner.work(QUOTA)
            if sim.steps >= n.t_p - 1:
                n.planner.work(UNLIMITED)   # (never needed: PLAN_OFFSET * QUOTA covers the whole family)
                n.variant = FAMILY[n.planner.chosen]
        return n

    def act(self, t, st):
        if t != self.t:
            self._close_step()
            self.prev = self.pairs if (self.t is not None and t == self.t + 1) else []
            self.pairs, self.preds, self.t = [], None, t
        self.stats["requests"] += 1
        if st["routing_active"] and self.stats["first_event_seen"] is None:
            self.stats["first_event_seen"] = st["steps"]
        if st["rekey_active"] and self.stats["rekey_seen"] is None:
            self.stats["rekey_seen"] = st["steps"]
        verified = self._verified(st)
        sim = replica(st)
        c = self.base
        if c.variant is None or sim.steps < c.t_p:
            v, descending = STAGING, False
        else:
            v = c.variant
            descending = sim.steps >= c.t_p + v.delay
        u, gap = law(sim, v, descending, c.gap)
        if verified and self.next is None:
            self.stats["verified"] += 1
            self.next = self._commit(c, sim, gap)
        self.pairs.append((st, u))
        return u

    def committed(self):
        """The latest committed state (for the review tools)."""
        return self.next if self.next is not None else self.base


def _block_state(obs):
    if isinstance(obs, dict):
        p = obs.get("privileged")
        if isinstance(p, dict) and p.get("format") == PRIV_FORMAT and isinstance(p.get("state"), dict):
            return p["state"]
    return None


class Oracle(object):
    """The wire: reset, act, close (one JSON object per line)."""

    def __init__(self):
        self.episode = Episode()
        self.closed = False
        self.warned = False

    def handle(self, line):
        msg = json.loads(line)
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "act":
            st = _block_state(msg.get("obs"))
            if st is None:
                if not self.warned:
                    print("[oracle] no privileged block in the observation: this program reads nothing else, so it "
                          "answers zeros", file=sys.stderr, flush=True)
                    self.warned = True
                return json.dumps({"action": [0.0] * 6})
            return json.dumps({"action": self.episode.act(int(msg["t"]), st)})
        if kind == "reset":
            self.episode = Episode()
            return '{"ok": true}'
        if kind == "close":
            self.closed = True
            return '{"ok": true}'
        return json.dumps({"error": "unknown request"})


def main():
    oracle = Oracle()
    out = sys.stdout
    out.write("READY\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = oracle.handle(line)
        except Exception as exc:  # noqa: BLE001 - a defect costs its episode loudly instead of hanging the wire
            reply = json.dumps({"error": "{}: {}".format(type(exc).__name__, exc)[:300]})
        out.write(reply + "\n")
        out.flush()
        if oracle.closed:
            break


if __name__ == "__main__":
    main()
