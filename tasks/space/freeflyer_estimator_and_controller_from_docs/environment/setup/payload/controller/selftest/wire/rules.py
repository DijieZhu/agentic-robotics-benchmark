"""The rules the runner applies to the controller's replies (pure functions; numbers from config.Config):

  * the forms of the setup, plan and step replies (a reply read whole whose form is wrong, or whose numbers are not
    finite or beyond the float range, is refused: rule reply.malformed);
  * the plan validity rules (plan.max_setpoints: 1 to plan.max_setpoints setpoints; plan.limit_rule with the mode's
    four hard limits; plan.consistency; plan.start_tolerance; plan.end_tolerance), checked in this order, the first
    broken one named by its table key with its setpoint index (rule plan.validity);
  * the event rule (interface.event_rule): which done / ready is accepted, and the reason another is ignored.
"""
import math

import numpy as np

# ---------------------------------------------------------------------------------------------- quaternions (x y z w)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qnormalize(q):
    q = np.asarray(q, float)
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return q / n


def qexp(rv):
    """rotation vector -> unit quaternion"""
    th = math.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
    if th < 1e-12:
        return qnormalize(np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0]))
    s = math.sin(0.5 * th) / th
    return np.array([rv[0] * s, rv[1] * s, rv[2] * s, math.cos(0.5 * th)])


def qangle(a, b):
    """the angle between two unit quaternions (the runner normalises a plan's attitudes first)"""
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * math.acos(min(1.0, d))


# ---------------------------------------------------------------------------------------------- reply forms


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("not a number")
    f = float(x)                                   # an integer beyond the float range raises OverflowError
    if not math.isfinite(f):
        raise ValueError("not finite")
    return f


def _vec(x, n):
    if not isinstance(x, list) or len(x) != n:
        raise ValueError("not a list of %d numbers" % n)
    return np.array([_num(v) for v in x])


def parse_step(rep):
    """(force, torque, event) of a step reply; force and torque None for 'no new command'. Raises ValueError."""
    if not isinstance(rep, dict):
        raise ValueError("a step reply is an object")
    if "force" not in rep or "torque" not in rep:
        raise ValueError("a step reply has force and torque")
    ev = rep.get("event", "none")
    if ev not in ("none", "done", "ready"):
        raise ValueError("event is none, done or ready")
    if rep["force"] is None and rep["torque"] is None:
        return None, None, ev
    try:
        return _vec(rep["force"], 3), _vec(rep["torque"], 3), ev
    except OverflowError:
        raise ValueError("a number beyond the float range")


def parse_plan(rep, refusal_reasons, max_setpoints, zero_eps):
    """('refuse', reason) or ('accept', [setpoint tuples]). Raises ValueError for a malformed reply. A list longer than
    plan.max_setpoints is well-formed (the validity rule names it); its entries are not parsed. An attitude of norm
    zero (at most zero_eps, the printed zero) is a malformed reply (plan.check_order)."""
    if not isinstance(rep, dict) or (("refuse" in rep) == ("setpoints" in rep)):
        raise ValueError("a plan reply is {'setpoints': [...]} or {'refuse': reason}")
    if "refuse" in rep:
        if rep["refuse"] not in refusal_reasons:
            raise ValueError("a refusal names one reason of the printed list")
        return "refuse", rep["refuse"]
    sps = rep["setpoints"]
    if not isinstance(sps, list):
        raise ValueError("setpoints is a list")
    if len(sps) > max_setpoints:
        return "accept", None
    out = []
    try:
        for s in sps:
            if not isinstance(s, list) or len(s) != 7:
                raise ValueError("a setpoint is [t, position, velocity, acceleration, attitude, body rate, angular "
                                 "acceleration]")
            out.append((_num(s[0]), _vec(s[1], 3), _vec(s[2], 3), _vec(s[3], 3), _vec(s[4], 4), _vec(s[5], 3),
                        _vec(s[6], 3)))
    except OverflowError:
        raise ValueError("a number beyond the float range")
    norm = []
    for s in out:
        n = math.sqrt(float(np.dot(s[4], s[4])))
        if not n > zero_eps:
            raise ValueError("an attitude of norm zero")
        # a plan's attitudes are unit quaternions: the runner normalises each one before it checks the plan and
        # passes the normalised plan on (a quaternion scaled up would pass an angle test on the raw dot product)
        norm.append(s[:4] + (qnormalize(s[4]),) + s[5:])
    return "accept", norm


def parse_setup(rep):
    """None, or the mass profile {'mass_kg', 'com_m', 'inertia'}. Raises ValueError."""
    if rep is None:
        return None
    if not isinstance(rep, dict) or set(rep) - {"mass_profile"}:
        raise ValueError("a setup reply is null or {'mass_profile': {...}}")
    mp = rep.get("mass_profile")
    if mp is None:
        return None
    if not isinstance(mp, dict):
        raise ValueError("mass_profile is an object")
    m = _num(mp.get("mass_kg"))
    com = _vec(mp.get("com_m"), 3)
    J = mp.get("inertia")
    if not isinstance(J, list) or len(J) != 3:
        raise ValueError("inertia is 3 x 3")
    J = np.array([_vec(r, 3) for r in J])
    if m <= 0.0:
        raise ValueError("mass_kg is positive")
    return {"mass_kg": m, "com_m": com.tolist(), "inertia": J.tolist()}


# ---------------------------------------------------------------------------------------------- plan validity

RULE_KEYS = ("plan.max_setpoints", "plan.limit_rule", "plan.consistency", "plan.start_tolerance",
             "plan.end_tolerance")


def check_plan(sps, lim, start_pose, goal_pose, kind, R):
    """None if the plan is valid, else (table key of the first rule broken, setpoint index or None, detail). The rules
    are checked in their printed order (rule plan.validity); "zero" means at most the printed zero (zero_eps). lim = (v, a,
    w, al) of the job's mode; start_pose and goal_pose = (position, quaternion x y z w); R =
    config.Config.plan_rules()."""
    v_l, a_l, w_l, al_l = lim
    rt = R["rel_tol"]
    if not sps or len(sps) > R["max_setpoints"]:
        return "plan.max_setpoints", None, "no setpoint" if not sps else "more than the cap"
    for i, s in enumerate(sps):
        if (np.linalg.norm(s[2]) > v_l * (1 + rt) or np.linalg.norm(s[3]) > a_l * (1 + rt) or
                np.linalg.norm(s[5]) > w_l * (1 + rt) or np.linalg.norm(s[6]) > al_l * (1 + rt)):
            return "plan.limit_rule", i, "a speed or acceleration above the mode's hard limit"
    for i, (a, b) in enumerate(zip(sps[:-1], sps[1:])):
        dt = b[0] - a[0]
        if dt <= 0 or dt > R["max_gap_s"] + R["gap_eps"]:
            return "plan.consistency", i + 1, "setpoint times do not increase or are too far apart"
        p = a[1] + a[2] * dt + 0.5 * a[3] * dt * dt
        v = a[2] + a[3] * dt
        q = qmul(a[4], qexp(a[5] * dt + 0.5 * a[6] * dt * dt))
        w = a[5] + a[6] * dt
        if (np.linalg.norm(p - b[1]) > R["cons_pos"] or qangle(q, b[4]) > R["cons_att"] or
                np.linalg.norm(v - b[2]) > R["cons_vel"] or np.linalg.norm(w - b[5]) > R["cons_rate"]):
            return "plan.consistency", i + 1, "a setpoint integrated forward does not meet the next one"
    if (np.linalg.norm(sps[0][1] - np.asarray(start_pose[0], float)) > R["start_pos"] or
            qangle(sps[0][4], start_pose[1]) > R["start_att"]):
        return "plan.start_tolerance", 0, "the first setpoint is not at the pose that meets the job before"
    e = sps[-1]
    z = R["zero_eps"]
    if np.any(np.abs(e[2]) > z) or np.any(np.abs(e[3]) > z) or np.any(np.abs(e[5]) > z) or np.any(np.abs(e[6]) > z):
        return "plan.end_tolerance", len(sps) - 1, "the last setpoint's rates are not zero"
    if kind in ("leg", "slew"):
        if (np.linalg.norm(e[1] - np.asarray(goal_pose[0], float)) > R["end_pos"] or
                qangle(e[4], goal_pose[1]) > R["end_att"]):
            return "plan.end_tolerance", len(sps) - 1, "the last setpoint is not at the goal"
    return None


# ---------------------------------------------------------------------------------------------- the event rule


def event_verdict(ev, kind, t, t_last, accepted_before, applied, eps):
    """(accepted?, reason if ignored) of an event in a reply (interface.event_rule). An event is accepted when all
    five conditions hold, checked in this order: in an applied reply (else step_not_applied); for the job being flown
    (else no_job); once per job (else already_accepted); of the job's kind, done for a leg or slew and ready for a view
    (else wrong_kind); at or after the time of the last setpoint of the job's accepted plan (else
    before_last_setpoint). kind None = no job is flown at this frame."""
    if ev not in ("done", "ready"):
        return False, None
    if not applied:
        return False, "step_not_applied"
    if kind is None:
        return False, "no_job"
    if accepted_before:
        return False, "already_accepted"
    if not ((ev == "done" and kind in ("leg", "slew")) or (ev == "ready" and kind == "view")):
        return False, "wrong_kind"
    if t < t_last - eps:
        return False, "before_last_setpoint"
    return True, None
