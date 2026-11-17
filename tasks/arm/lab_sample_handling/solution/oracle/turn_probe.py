"""Lab sample handling, feasibility probe 5: turning the rotor by its handle when how it turns is hidden.

Owner (2026-11-15): turning the rotor by gripping its handle stays in the task, and how the rotor answers a turn
is a hidden property of the unit that the agent learns from its own trials. Three hidden properties:

- the handle's play (`--handle-play` deg): the handle turns freely within it before it drives the rotor (a loose
  coupling); where the handle sits in its play at the start is drawn per seed;
- the rotor hinge's friction (`--rotor-friction` N m): more friction, more of the turn is lost in the fingers'
  give; little friction, a rotor let go while moving coasts on;
- the rotor's inertia about its axis (`--rotor-inertia` kg m^2; a heavier rotor, or one with tubes in it).

Two modes, on centrifuge_probe's world with the solid rotor (the turn is centrifuge_probe.turn_once: grip the bar
26 mm out, carry the hand along the arc, let go):

- `--mode response`: one grip turning by `--cmd` deg, at `--turn-speed` rad per control step, released after
  holding still (`--release hold`) or while moving (`--release moving`); reports the rotor's actual turn.
- `--mode policy`: hole 0 starts at `--phi0` deg (out of reach) and must end within `--window` deg of `--goal`;
  a policy turns, reads the rotor's angle (the truth plus `--read-noise` deg, standing for the wrist camera's
  reading, about 1 deg at 70 mm per mm), and turns again, at most `--max-grips` times:
  `naive` turns once by the needed angle; `feedback` turns by what is left each time, taking the command
  for the result; `learner` also keeps what a grip lost (the same direction: the fingers' give; a reversal:
  the play on top) and adds it to the next command.

    python -I turn_probe.py --mode response --cmd 45 --handle-play 10 --rotor-friction 0.3
    python -I turn_probe.py --mode policy --policy learner --handle-play 10 --rotor-friction 0.05
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco
import numpy as np

import centrifuge_probe as C
import lab_sim as S
import rotor_angle as RA


def defaults(a):
    """The simulation's settings (`lab_sim.sim_defaults`)."""
    S.sim_defaults(a)


def world(a, start_deg):
    a.rotor_start = start_deg
    m, _src, _names = C.build(a)
    p = C.World(m, a)
    for _ in range(10):
        p.step([0, 0, 0], [0, 0, 0], -1)
    return p


def rotor_deg(p):
    return math.degrees(float(p.data.qpos[p.rotor_q]))


def response_trial(a):
    t0 = time.time()
    defaults(a)
    p = world(a, a.phi0)
    q0 = rotor_deg(p)
    h0 = None if p.handle_q is None else math.degrees(float(p.data.qpos[p.handle_q]))
    got = C.turn_once(p, math.radians(a.cmd), a)
    out = {k: getattr(a, k) for k in ("cmd", "handle_play", "rotor_friction", "rotor_inertia", "turn_speed",
                                      "release", "seed")}
    out.update(handle_start_deg=None if h0 is None else round(h0, 2), status=got["turn_status"],
               actual_deg=round(rotor_deg(p) - q0, 2), steps=p.t, wall_s=round(time.time() - t0, 1))
    if got["turn_status"] != "ok":
        out["q"] = got.get("q")
    return out


def seat(p, a, holes):
    """Put tubes straight into rotor holes `holes` (set-up only: the tubes of the source rack, moved), settle."""
    m, d = p.model, p.data
    names = ["tube"] + [f"tube{i}" for i in range(1, a.tubes)]
    for name, k in zip(names, holes):
        h = p.hole(int(k))
        adr = m.joint(name + "_free").qposadr[0]
        d.qpos[adr:adr + 3] = h["top"] - h["axis"] * (C.HOLE_DEPTH - 0.0005)
        d.qpos[adr + 3:adr + 7] = C.B.quat_from_mat(h["R"])
    mujoco.mj_forward(m, d)
    for _ in range(10):
        p.step([0, 0, 0], [0, 0, 0], -1)


def read_trial(a):
    """The rotor at a random angle, the handle somewhere in its play: the wrist camera's reading against the
    truth."""
    t0 = time.time()
    defaults(a)
    rng = np.random.default_rng([a.seed, 19])
    a.tubes = max(2, a.loaded)
    p = world(a, float(rng.uniform(0, 360)))
    seat(p, a, rng.choice(C.NHOLES, a.loaded, replace=False))
    prior = C.CENTRE + rng.uniform(-a.centre_err, a.centre_err, 2)
    save = os.path.join(a.save_dir, f"read_{a.seed}.png") if a.save_dir and a.seed < 3 else None
    if save:
        os.makedirs(a.save_dir, exist_ok=True)
    est, diag = RA.read_angle(p, rng, prior, dist=a.view_dist, save=save)
    diag["looks"] = 1
    if est is None:                              # a second look with the camera turned (the gripper hides part)
        est, diag = RA.read_angle(p, rng, prior, dist=a.view_dist, heading=math.pi / 2)
        diag["looks"] = 2
    truth = RA.true_angle(p)
    out = {"seed": a.seed, "handle_play": a.handle_play, "loaded": a.loaded, "true_deg": round(math.degrees(truth), 2),
           "view_dist": a.view_dist, "centre_err": a.centre_err, **diag}
    if est is not None:
        out["err_deg"] = round(math.degrees(C.wrap(est - truth)), 3)
    out["steps"] = p.t
    out["wall_s"] = round(time.time() - t0, 1)
    return out


class Reader:
    """The rotor's angle (deg, unwrapped against the last reading): the truth plus noise, or the wrist camera above
    the rotor (its axis known to centre_err; a second look with the camera turned when the gripper hides the
    mark). `fallback` is used only if the very first reading fails (privileged; counted in `failed`)."""

    def __init__(self, p, a, rng, fallback):
        self.p, self.a, self.rng, self.fallback = p, a, rng, fallback
        self.last, self.failed = None, 0
        self.centre = S.centre_xy(a).copy()   # the rotor's axis: the fitted circle's centre (camera)

    def __call__(self):
        p, a, rng = self.p, self.a, self.rng
        if a.read == "truth":
            return rotor_deg(p) + rng.normal(0, a.read_noise)
        prior = S.centre_xy(a) + rng.uniform(-a.centre_err, a.centre_err, 2)
        est = None
        for heading in (0.0, math.pi / 2):
            est, diag = RA.read_angle(p, rng, prior, dist=a.view_dist, heading=heading,
                                     px_offset=getattr(a, "px_offset", 0.0))
            if est is not None:
                self.centre = np.asarray(diag["centre"], float)
                break
            self.failed += 1
        p.home()
        if est is None:
            return self.last if self.last is not None else self.fallback
        deg = math.degrees(est)
        if self.last is not None:
            deg = self.last + ((deg - self.last + 180) % 360 - 180)
        self.last = deg
        return deg


class Turner:
    """Turns the rotor to a goal angle by its handle, grip after grip, reading the rotor after each: `naive` turns
    once by the needed angle; `feedback` turns by what is left, taking the command for the result; `learner` also
    keeps what a grip lost (the same direction: the fingers' give; a reversal: the play on top) and adds it to the
    next command. What it learns stays for the next `turn_to` (a worklist turns the same rotor again)."""

    def __init__(self, policy):
        self.policy = policy
        self.est_loss, self.est_rev, self.last_dir = None, None, None

    def turn_to(self, p, a, goal, read, meas, max_grips, stop_within):
        grips, status = [], "ok"
        for _ in range(max_grips):
            resid = (goal - meas + 180) % 360 - 180
            if abs(resid) <= stop_within:
                break
            d = math.copysign(1.0, resid)
            cmd = resid
            if self.policy == "learner":
                if self.est_loss is not None:
                    cmd += d * self.est_loss
                if self.last_dir is not None and d != self.last_dir and self.est_rev is not None:
                    cmd += d * self.est_rev
            got = C.turn_once(p, math.radians(cmd), a)
            if got["turn_status"] != "ok":
                status = got["turn_status"]
                break
            p.home()
            new = read()
            actual = new - meas
            loss = d * cmd - d * actual                  # how much of the command the grip lost
            if self.policy == "learner":
                if self.last_dir is None or d == self.last_dir:
                    self.est_loss = loss if self.est_loss is None else min(self.est_loss, loss)
                else:
                    self.est_rev = max(0.0, loss - (self.est_loss or 0.0))
            grips.append({"cmd": round(cmd, 1), "actual": round(actual, 1)})
            self.last_dir = d
            meas = new
            if self.policy == "naive":
                break
        return status, grips, meas


# --- probe 12 : a one-way rotor (a ratchet) turned to a loading port, goal after goal -------------

READ_STEPS = 35            # control steps a wrist-camera reading of the rotor's angle costs (measured, `Reader`)


def install_ratchet(p, tooth_deg):
    """Make the rotor turn one way only (counter-clockwise, the hinge angle increasing): after every control step
    the hinge's lower limit is raised to the last tooth passed (a pawl on `tooth_deg` teeth). MuJoCo's own limit
    constraint holds it."""
    m = p.model
    j = m.joint("rotor_hinge").id
    tooth = math.radians(tooth_deg)
    q0 = float(p.data.qpos[p.rotor_q])
    m.jnt_range[j] = [math.floor(q0 / tooth) * tooth, 1e3]
    m.jnt_limited[j] = 1

    def update():
        lo = math.floor(float(p.data.qpos[p.rotor_q]) / tooth) * tooth
        m.jnt_range[j, 0] = max(m.jnt_range[j, 0], lo)

    step, joint_step = p.step, p.joint_step

    def step_r(*args, **kw):
        step(*args, **kw)
        update()

    def joint_step_r(*args, **kw):
        joint_step(*args, **kw)
        update()

    p.step, p.joint_step = step_r, joint_step_r


def set_friction(p, f):
    p.model.dof_frictionloss[p.model.joint("rotor_hinge").dofadr[0]] = f


class OneWay:
    """Turn policies for a one-way rotor: every grip turns it forward; a goal passed costs a turn round.

    - `feedback`: turn by what is left, taking the command for the result.
    - `learner`: probe 5's learner (`Turner`): adds the smallest loss seen so far (the first grip's loss includes
      the handle's play).
    - `careful`: adds the last grip's loss and aims `margin` short of the goal, then a final grip by what is left.
    - `identifier`: tells the play from the give: the first grip of an episode takes up the play, so its extra
      loss is the play and later grips add only the give (the mean of the later grips' losses), aims `margin`
      short while its estimate rests on one grip.
    `prior`: a starting loss estimate (deg) taken from the practice units, else none.
    """

    def __init__(self, policy, margin, prior=None):
        self.policy, self.margin = policy, margin
        self.losses = []               # (grip index in the episode, loss)
        self.moves = []                # (grip index, loss, command, actual)
        self.prior = prior
        self.n = 0

    def give(self):
        if self.policy == "learner":
            ls = [x for _, x in self.losses]
            return min(ls) if ls else (self.prior or 0.0)
        if self.policy == "careful":
            return self.losses[-1][1] if self.losses else (self.prior or 0.0)
        if self.policy == "identifier":
            d = self.dead_band()
            return d if d is not None else (self.prior or 0.0)
        return 0.0

    def dead_band(self, need=None):
        """The identifier's estimate of the give (deg lost per grip): the median loss of the grips after the first
        (which takes up the play) that moved the rotor by more than 1 deg and asked for at most 60 deg (a long grip
        at high friction loses an uneven amount). v3 : for the last grip (`need` up to twice the margin)
        the median loss of middling grips (12-40 deg asked); for a short final grip (`need` under 8 deg) the smallest
        loss of the short grips (at most 25 deg asked), when there are any."""
        ls = [x for i, x, cmd, got in self.moves if i > 0 and got > 1.0 and cmd <= 60.0]
        if need is not None and need < 8.0:
            # a short final grip: the smallest loss seen on a short grip (near the dead band the rotor sticks, then
            # slips 6-8 deg at once; erring short costs a grip, erring past a turn round)
            short = [x for i, x, cmd, got in self.moves if i > 0 and got > 1.0 and cmd <= 25.0]
            ls = short or ls
            return float(min(ls)) if ls else None
        if need is not None and need <= 2 * self.margin:
            # the last grip, a middling one (its loss is the steadiest): the losses of middling grips if any
            mid = [x for i, x, cmd, got in self.moves if i > 0 and got > 1.0 and 12.0 <= cmd <= 40.0]
            ls = mid or ls
        return float(np.median(ls)) if ls else None

    def command(self, need):
        g = self.give()
        if self.policy in ("feedback", "learner"):
            return need + g
        if self.policy == "identifier":
            # a grip past the port costs a turn round (about 8 grips), one short costs one more grip: always err
            # short. A grip that did not move the rotor shows the give is larger than its command: ask 1.5 deg more
            if self.moves and self.moves[-1][3] < 1.0:
                return self.moves[-1][2] + 1.5
            d = self.dead_band(need)
            if d is None:
                # the play and the give not told apart yet (the first grip takes up the play): aim well short
                return need - self.margin
            return need - (self.margin if need > 2 * self.margin else 1.0) + d
        short = 0.0 if need <= 2 * self.margin else self.margin
        return need - short + g

    def turn_to(self, p, a, goal, read, meas, max_grips, budget=None, home=True):
        """Turn the one-way rotor until `goal` (deg, the rotor angle) is within `a.stop_within` of the reading,
        grip after grip (at most `a.max_cmd` each; a longer way goes in full grips first). Returns (status, grips,
        overshoots, [command, got] per grip, the last reading). `home`: back to the start pose after each grip
        (probe 12); else the next grip starts from where the last one let go (the v3 reference)."""
        grips, over, glog = 0, 0, []
        need = (goal - meas) % 360
        status = "ok"
        while True:
            err = (goal - meas + 180) % 360 - 180
            if -max(a.stop_within, getattr(a, "past_ok", 0.0)) <= err <= a.stop_within:
                break                                       # there (a little past it is still inside the port)
            if err < 0 and need < 180:
                over += 1                                   # it went past the port: a turn round
            need = err if err > 0 else err + 360
            if (budget is not None and budget()) or grips >= max_grips:
                status = "budget" if budget is not None and budget() else "grips"
                break
            cmd = a.max_cmd if need > a.max_cmd + 20 else self.command(need)
            got = C.turn_once(p, math.radians(cmd), a)
            if got["turn_status"] != "ok":
                status = got["turn_status"]
                break
            if home:
                p.home()
            new = read()
            self.learn(cmd, new - meas)
            glog.append([round(cmd, 1), round(new - meas, 1)])
            meas = new
            grips += 1
        return status, grips, over, glog, meas

    def learn(self, cmd, actual):
        self.losses.append((self.n, cmd - actual))
        self.moves.append((self.n, cmd - actual, cmd, actual))
        self.n += 1


def sequence_trial(a):
    """Bring `goals` rotor holes to the loading port one after another on a one-way rotor (`install_ratchet`),
    each within +-window; reading the angle (truth + read noise) costs READ_STEPS. Stops at the step budget."""
    t0 = time.time()
    defaults(a)
    rng = np.random.default_rng([a.seed, 29])
    start = float(rng.uniform(0, 360))
    p = world(a, start)
    install_ratchet(p, a.tooth)
    f0 = a.rotor_friction
    pol = OneWay(a.policy, a.margin, a.prior)
    reads = [0]

    def read():
        reads[0] += 1
        return rotor_deg(p) + rng.normal(0, a.read_noise)

    def steps():
        return p.t + reads[0] * READ_STEPS

    holes = [int(h) for h in rng.choice(C.NHOLES, a.goals, replace=False)]
    meas = read()
    done, log, status = 0, [], "ok"
    while holes and status == "ok":
        # the next hole: the one the rotor reaches first going forward (a planned order on a one-way rotor)
        h = min(holes, key=lambda k: (a.port - 45 * k - meas) % 360)
        holes.remove(h)
        goal = a.port - 45 * h
        status, grips, over, glog, meas = pol.turn_to(p, a, goal, read, meas, a.max_grips,
                                                      budget=lambda: steps() > a.budget)
        true_err = (goal - rotor_deg(p) + 180) % 360 - 180
        ok = status == "ok" and abs(true_err) <= a.window
        log.append({"hole": h, "grips": grips, "over": over, "err": round(true_err, 2), "ok": ok, "steps": steps(),
                    "cmd_got": glog})
        if not ok:
            if status == "ok":
                status = "window"
            break
        done += 1
        if a.load_friction:
            # variant 2b: a tube went in; the bearing's friction rises with the load (hidden slope per unit)
            set_friction(p, f0 + a.load_friction * (done / a.goals))
    out = {k: getattr(a, k) for k in ("policy", "handle_play", "rotor_friction", "rotor_inertia", "load_friction",
                                      "prior", "seed", "goals", "budget", "window")}
    out.update(status=status, done=done, all_done=done == a.goals, grips=sum(g["grips"] for g in log),
               overshoots=sum(g["over"] for g in log), steps=steps(), log=log, wall_s=round(time.time() - t0, 1))
    return out


def policy_trial(a):
    t0 = time.time()
    defaults(a)
    p = world(a, a.phi0)
    rng = np.random.default_rng([a.seed, 17])
    read = Reader(p, a, rng, fallback=a.phi0)
    status, grips, _meas = Turner(a.policy).turn_to(p, a, a.goal, read, read(), a.max_grips, a.stop_within)
    err = (a.goal - rotor_deg(p) + 180) % 360 - 180
    out = {k: getattr(a, k) for k in ("policy", "handle_play", "rotor_friction", "rotor_inertia", "phi0", "goal",
                                      "seed", "read")}
    out.update(status=status, grips=len(grips), log=grips, read_failures=read.failed, final_err_deg=round(err, 2),
               within_5=bool(abs(err) <= 5), within_10=bool(abs(err) <= 10), steps=p.t,
               wall_s=round(time.time() - t0, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("response", "policy", "read", "sequence"), default="response")
    ap.add_argument("--tooth", type=float, default=2.0, help="sequence: deg, the ratchet's tooth")
    ap.add_argument("--port", type=float, default=200.0,
                    help="sequence: deg, the rotor angle that brings hole 0 to the port")
    ap.add_argument("--max-cmd", type=float, default=60.0, help="sequence: deg, the largest turn one grip makes")
    ap.add_argument("--goals", type=int, default=6, help="sequence: holes brought to the port in turn")
    ap.add_argument("--window", type=float, default=2.5, help="sequence: deg, the port's window (+-)")
    ap.add_argument("--budget", type=int, default=8000, help="sequence: control steps (reads included)")
    ap.add_argument("--margin", type=float, default=4.0, help="sequence: deg a careful policy aims short")
    ap.add_argument("--prior", type=float, default=None, help="sequence: a starting loss estimate (deg)")
    ap.add_argument("--load-friction", type=float, default=0.0,
                    help="sequence, variant 2b: N m the friction rises over the sequence (the rotor filling)")
    ap.add_argument("--read", choices=("truth", "camera"), default="truth",
                    help="policy mode: the rotor's angle as the truth plus --read-noise, or from the wrist camera")
    ap.add_argument("--centre-err", type=float, default=0.005, help="m, the rotor axis' prior error (each axis)")
    ap.add_argument("--view-dist", type=float, default=0.20, help="m, the wrist camera above the rotor")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--loaded", type=int, default=0, help="read mode: tubes seated in random rotor holes (0-4)")
    ap.add_argument("--handle-play", type=float, default=0.0, help="deg, the handle's free play")
    ap.add_argument("--rotor-friction", type=float, default=0.3, help="N m")
    ap.add_argument("--rotor-inertia", type=float, default=0.008, help="kg m^2 about the rotor's axis")
    ap.add_argument("--turn-speed", type=float, default=0.04, help="rad per control step along the arc")
    ap.add_argument("--release", choices=("hold", "moving"), default="hold")
    ap.add_argument("--cmd", type=float, default=45.0, help="deg, response mode")
    ap.add_argument("--phi0", type=float, default=90.0, help="deg, hole 0's start angle")
    ap.add_argument("--goal", type=float, default=135.0, help="deg, policy mode: hole 0's goal angle")
    ap.add_argument("--policy", choices=("naive", "feedback", "learner", "careful", "identifier"), default="learner")
    ap.add_argument("--read-noise", type=float, default=0.8, help="deg, the rotor angle reading's noise")
    ap.add_argument("--stop-within", type=float, default=3.0, help="deg, a policy stops when this close (sequence: 1.5)")
    ap.add_argument("--max-grips", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    fn = {"response": response_trial, "policy": policy_trial, "read": read_trial, "sequence": sequence_trial}[a.mode]
    if not a.grid:
        print(json.dumps(fn(a)))
        return
    import itertools
    from concurrent.futures import ProcessPoolExecutor
    grid = json.loads(a.grid)
    keys = list(grid)
    jobs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        b = argparse.Namespace(**vars(a))
        for k, val in zip(keys, combo):
            setattr(b, k.replace("-", "_"), val)
        jobs.append(b)
    with ProcessPoolExecutor(a.jobs) as ex, open(a.out, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in ex.map(fn, jobs))
    print(f"{len(jobs)} trials -> {a.out}")


if __name__ == "__main__":
    main()
