"""Mutants of every sentence of the credit rule (a maintainer's certificate of the grader; nothing runs it at verify).

    python3 -B tests/review/mutants.py --out DIR

Each mutant is a constructed episode in the cell (a privileged construction: parcels placed, turned, pushed or
lifted by writing the world's state, the robot posed directly, a damage or a lock-out entered into the safety
watch), with the verdict the rule's sentence demands written next to it BEFORE it runs; the measured verdict is the
cell's own record (``ParcelCellServer.finalize``). "Benign" mutants are variants that must score exactly like the
plain truth. The sentences (``cell_manual.md`` section 8):

  class     W / U / X fixed at arrival (spawn), by the kind and the label position against the station's duty;
            nothing the robot does changes it
  judge     readable at the first downstream crossing of the scan line when (1) label within the tilt, (2) on the belt,
            calm and moving with it, (3) no robot touch, (4) no parcel touch, (5) not damaged
  lost      fell, damaged at any time, or not across by the shift's end (whatever ended it)
  credit    W readable - U not readable - lost (any class); score = max(0, credit) / #W (0 without W)

The line setting's sentences (the station's job; ``cell_manual.md`` sections 3, 7 and 8), mutants N*:

  class     every parcel in the duty: W = label not up (any kind, any label position), U = label up; no X
  feed      one parcel at a time: the next set down at the tick after the one before it is settled (scan line,
            fall, damage)
  indexing  the belt stands while the station's parcel has its centre in the zone and runs again once it is out
  not done  a parcel not done when the shift ends (never set down, in the station and not across) costs nothing
  credit    W readable - U done and not readable - lost (fell or damaged); score = max(0, credit) / #W of the list
  end       the shift ends at the end of the tick its list's last parcel is settled
  due       a parcel crossing after its due time (set-down time + free ride + due_margin_s) is not readable

Output: ``DIR/mutants.json`` and a table (expected against measured, pass / FAIL).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402

V = 0.2            # belt speed of the constructed shifts (m/s)


# ------------------------------------------------------------------------------------------------ construction kit
def shift(*parcels, v=V, duration=30.0, t0=0.1, dt=6.0):
    from parcelcell.stream import Arrival, Parcel, Shift
    arr = []
    for k, kw in enumerate(parcels):
        p = dict(kind="rigid", size=(0.24, 0.18, 0.12), mass=0.8, friction=0.55, label_face=0, label_uv=(0.0, 0.0),
                 yaw=0.0, lateral=0.0)
        p.update(kw)
        arr.append(Arrival(t=round(t0 + k * dt, 2), parcel=Parcel(**p)))
    return Shift(belt_speed=((0.0, v),), arrivals=tuple(arr), duration=duration, seed=4242)


def server(sh, over=None):
    srv = common.make_server(common.load_cfg(over=over), render=False)
    srv.reset(sh)
    return srv


def steps(srv, n):
    r = None
    while n > 0:
        r = srv.handle("step", {"n": min(50, n)})
        n -= min(50, n)
    return r


def until(srv, cond, limit=3000):
    for _ in range(limit):
        if cond():
            return True
        srv.handle("step", {"n": 1})
    return cond()


def spawned(srv, i):
    return until(srv, lambda: srv.world.slots[i].active)


def y_of(srv, i):
    w = srv.world
    return float(w.data.qpos[w._slot_qadr[i] + 1])


def quat_axis(axis, deg):
    a = math.radians(deg) / 2
    ax = [float(x) for x in axis]
    n = math.sqrt(sum(x * x for x in ax))
    return (math.cos(a),) + tuple(math.sin(a) * x / n for x in ax)


def rest_z(srv, i, quat=(1, 0, 0, 0)):
    """Height of parcel i's centre resting on the belt in orientation quat (box extent along z)."""
    import mujoco
    import numpy as np
    w = srv.world
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    R = R.reshape(3, 3)
    half = np.asarray(w.slots[i].parcel.size, float) / 2
    return w.station.belt_height + float(np.sum(np.abs(R[2, :]) * half)) - 0.0002


def place(srv, i, y, quat=(1, 0, 0, 0), dz=0.0, v=None, wv=None, x=None):
    import numpy as np
    w = srv.world
    vel = np.zeros(6)
    vel[1] = V if v is None else v
    if wv is not None:
        vel[3:] = wv
    x = w.station.belt_center_x if x is None else x
    w.set_pose(i, [x, y, rest_z(srv, i, quat) + dz], quat, vel)


def kick(srv, i, v=None, wv=None):
    """Change parcel i's velocity only (its pose and its contact with the belt kept)."""
    import mujoco
    w = srv.world
    a = w._slot_dadr[i]
    if v is not None:
        w.data.qvel[a + 1] = v
    if wv is not None:
        import numpy as np
        R = np.zeros(9)
        q = w._slot_qadr[i]
        mujoco.mju_quat2Mat(R, w.data.qpos[q + 3:q + 7])
        w.data.qvel[a + 3:a + 6] = R.reshape(3, 3).T @ np.asarray(wv, float)   # free joint: local angular velocity
    mujoco.mj_forward(w.model, w.data)


def to_end(srv):
    while not srv.handle("step", {"n": 50})["shift_over"]:
        pass
    return srv.finalize()


def pose_robot(srv, q):
    """Write the robot's joint state (a construction) and hold it as the target."""
    import mujoco
    import numpy as np
    w = srv.world
    q = np.asarray(q, float)
    w.data.qpos[w._arm_qadr] = q
    w.data.qvel[w._arm_dadr] = 0.0
    w._q_applied = q.copy()
    w.data.ctrl[w._arm_aid] = q
    srv._q_hold = q.copy()
    mujoco.mj_forward(w.model, w.data)


def verdict(rec, i):
    p = rec["parcels"][i]
    return {"cls": p["cls"], "readable": bool(p["readable"]), "failed": sorted(p["failed"]),
            "lost_reason": p["lost_reason"], "credit": rec["credit"], "score": rec["shift_score"]}


# ------------------------------------------------------------------------------------------------ the mutants
M = []


def mutant(mid, sentence, what, expect, benign=False):
    def deco(fn):
        M.append({"id": mid, "sentence": sentence, "what": what, "expect": expect, "benign": benign, "fn": fn})
        return fn
    return deco


@mutant("J1a", "judge (1) label within the tilt", "a W parcel (label on the downstream face) turned towards label up "
        "but 30 deg short of it (on an edge, tipping) as it crosses", {"cls": "W", "readable": False,
                                                                        "failed_includes": ["label_not_up"],
                                                                        "lost_reason": None, "credit": 0})
def j1a(_):
    srv = server(shift(dict(label_face=4)))
    spawned(srv, 0)
    until(srv, lambda: srv.record.parcels[0]["cls"] is not None)
    steps(srv, 20)
    place(srv, 0, 0.95 - 0.012, quat=quat_axis((1, 0, 0), 60.0), dz=0.0002)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J1b", "judge (1)-(2) the thresholds", "constructed crossing records at each threshold: label 20 / 25 / 25.01 / "
        "30 deg, turning 1.99 / 2.0 rad/s, sliding 0.099 / 0.1 m/s", {"verdicts": [True, True, False, False, True, False,
                                                                        True, False]})
def j1b(_):
    srv = server(shift({}))
    rec = srv.record
    out = []
    for change in ({"label_angle_deg": 20.0}, {"label_angle_deg": 25.0}, {"label_angle_deg": 25.01},
                   {"label_angle_deg": 30.0}, {"angular_speed": 1.99}, {"angular_speed": 2.0},
                   {"speed_rel_belt": 0.099}, {"speed_rel_belt": 0.1}):
        rec.reset()
        rec._on_event(_crossing(**change))
        out.append(bool(rec.parcels[0]["readable"]))
    return {"verdicts": out}, None


@mutant("J1c", "judge (1) label within the tilt", "an untouched parcel with its label on the downstream face (W)",
        {"cls": "W", "readable": False, "failed": ["label_not_up"], "lost_reason": None, "credit": 0})
def j1c(_):
    rec = to_end(server(shift(dict(label_face=4))))
    return verdict(rec, 0), rec


@mutant("J1d", "judge (1) label within the tilt", "a W parcel (label down) turned exactly label up after its arrival",
        {"cls": "W", "readable": True, "failed": [], "lost_reason": None, "credit": 1, "score": 1.0}, benign=True)
def j1d(_):
    srv = server(shift(dict(label_face=1)))
    spawned(srv, 0)
    until(srv, lambda: srv.record.parcels[0]["cls"] is not None)
    steps(srv, 10)
    place(srv, 0, y_of(srv, 0), quat=quat_axis((1, 0, 0), 180.0), dz=0.002)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J2a", "judge (2) on the belt", "the parcel is 6 cm above the belt (airborne) as it crosses",
        {"cls": "U", "readable": False, "failed": ["not_on_belt"], "lost_reason": None, "credit": -1})
def j2a(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.95 - 0.002, dz=0.06)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J2b", "judge (2) moving with the belt", "the parcel is kicked to 0.3 m/s faster than the belt just before it crosses (sliding)",
        {"cls": "U", "readable": False, "failed_includes": ["sliding"], "lost_reason": None, "credit": -1})
def j2b(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.80)
    until(srv, lambda: y_of(srv, 0) > 0.95 - 0.006)
    kick(srv, 0, v=V + 0.3)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J2c", "judge (2) calm", "the parcel is set spinning at 8 rad/s about the vertical just before it crosses",
        {"cls": "U", "readable": False, "failed_includes": ["turning"], "lost_reason": None, "credit": -1})
def j2c(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.80)
    until(srv, lambda: y_of(srv, 0) > 0.95 - 0.006)
    kick(srv, 0, wv=(0.0, 0.0, 8.0))
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J2d", "judge (2) calm", "the parcel turned 90 deg about the vertical, at rest (yaw is free)",
        {"cls": "U", "readable": True, "failed": [], "lost_reason": None, "credit": 0}, benign=True)
def j2d(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.7, quat=quat_axis((0, 0, 1), 90.0))
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J2e", "judge (2) on the belt", "the parcel crosses at the belt's far edge, at rest",
        {"cls": "U", "readable": True, "failed": [], "lost_reason": None, "credit": 0}, benign=True)
def j2e(_):
    srv = server(shift(dict(size=(0.18, 0.16, 0.10))))
    spawned(srv, 0)
    steps(srv, 20)
    w = srv.world
    place(srv, 0, 0.6, x=w.station.belt_center_x + w.station.half_width - 0.10)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J3a", "judge (3) no robot touch", "a parcel crosses while it rubs under the left gripper's closed fingers "
        "(robot posed over the belt; the scan line moved under the hand for this construction)",
        {"cls": "U", "readable": False, "failed_includes": ["robot_touching"], "lost_reason": None, "credit": -1})
def j3a(ctx):
    import numpy as np
    over = {"world": {"scan_y": 0.15}}

    def posed(sh):
        srv = server(sh, over=over)
        q = np.asarray(srv.describe()["robot"]["start_posture"], float)
        q[:7] = np.asarray(ctx["teach_left_in"][:7])
        pose_robot(srv, q)
        srv._grip_hold = np.array([1.0, 0.0])
        steps(srv, 40)                                  # the fingers close
        return srv

    srv0 = posed(shift({}, duration=60.0, t0=50.0))
    gap = _gap_under(srv0, 0.15)
    srv0.close()
    h = round(gap + 0.001, 3)                           # 1 mm taller than the gap under the fingers
    srv = posed(shift(dict(size=(0.20, 0.16, h)), duration=60.0, t0=1.0))
    spawned(srv, 0)
    steps(srv, 5)
    place(srv, 0, -0.10)
    rec = to_end(srv)
    v = verdict(rec, 0)
    v["note"] = "gap under the closed fingers %.4f m, parcel %.3f m tall" % (gap, h)
    return v, rec


def _gap_under(srv, y0):
    """Lowest robot point above the belt around (belt centre, y0): rays cast up from the belt surface."""
    import mujoco
    import numpy as np
    w = srv.world
    grp = np.zeros(6, np.uint8)
    grp[3] = 1
    gid = np.zeros(1, np.int32)
    robot = set(int(g) for g in w.robot_geoms)
    low = 9.0
    for x in np.linspace(w.station.belt_center_x - 0.08, w.station.belt_center_x + 0.08, 33):
        for y in np.linspace(y0 - 0.1, y0 + 0.1, 41):
            dd = mujoco.mj_ray(w.model, w.data, np.array([x, y, w.station.belt_height + 0.0005]),
                               np.array([0.0, 0.0, 1.0]), grp, 1, -1, gid)
            if dd >= 0 and int(gid[0]) in robot:
                low = min(low, dd + 0.0005)
    return low


@mutant("J3b", "judge (3) no robot touch", "constructed crossing record: robot touching (the judge's own input)",
        {"cls": "U", "readable": False, "failed": ["robot_touching"], "lost_reason": None, "credit": -1})
def j3b(_):
    srv = server(shift({}))
    rec = srv.record
    rec.reset()
    rec._on_event(_crossing(touching_robot=True))
    f = rec.finalize()
    return verdict(f, 0), f


def _crossing(**over):
    e = {"type": "line", "line": "scan", "direction": 1, "t": 5.0, "t_cross": 4.99, "arrival": 0,
         "label_angle_deg": 3.0, "on_belt": True, "angular_speed": 0.1, "speed_rel_belt": 0.01,
         "touching_robot": False, "touching_parcel": False, "x_rel": 0.0}
    e.update(over)
    return e


@mutant("J4a", "judge (4) no parcel touch", "a second parcel stacked on the first as it crosses",
        {"cls": "U", "readable": False, "failed": ["parcel_touching"], "lost_reason": None, "credit": -2})
def j4a(_):
    srv = server(shift({}, dict(size=(0.20, 0.16, 0.10)), dt=0.1))
    spawned(srv, 0)
    steps(srv, 2)
    place(srv, 0, 0.0)
    spawned(srv, 1)
    steps(srv, 20)
    w = srv.world
    place(srv, 0, 0.95 - 0.012)
    y0 = y_of(srv, 0)
    w.set_pose(1, [w.station.belt_center_x, y0, w.station.belt_height + 0.12 + 0.05 - 0.0002], (1, 0, 0, 0),
               [0, V, 0, 0, 0, 0])
    rec = to_end(srv)
    v = verdict(rec, 0)
    v["second_parcel"] = verdict(rec, 1)
    return v, rec


@mutant("J4b", "judge (4) no parcel touch", "two parcels crossing side by side with a 3 cm gap (no touch)",
        {"cls": "U", "readable": True, "failed": [], "lost_reason": None, "credit": 0}, benign=True)
def j4b(_):
    srv = server(shift(dict(size=(0.18, 0.16, 0.10)), dict(size=(0.18, 0.16, 0.10)), dt=0.1))
    spawned(srv, 0)
    steps(srv, 2)
    place(srv, 0, 0.0)
    spawned(srv, 1)
    w = srv.world
    cx = w.station.belt_center_x
    place(srv, 0, 0.6, x=cx - 0.095)
    place(srv, 1, 0.6, x=cx + 0.095)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J5a", "judge (5) not damaged", "a parcel damaged (squeeze entered into the watch) before its crossing",
        {"cls": "U", "readable": False, "failed": ["damaged"], "lost_reason": "damaged", "credit": -2})
def j5a(_):
    srv = server(shift({}))
    spawned(srv, 0)
    srv.watch.damaged[0] = {"t": 1.0, "peak_N": 200.0, "reason": "squeeze"}
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("J5b", "lost: damaged at any time", "a W parcel readable at its crossing, damaged after it",
        {"cls": "W", "readable": True, "failed": [], "lost_reason": "damaged", "credit": 0})
def j5b(_):
    srv = server(shift(dict(label_face=1)))
    spawned(srv, 0)
    until(srv, lambda: srv.record.parcels[0]["cls"] is not None)
    steps(srv, 5)
    place(srv, 0, y_of(srv, 0), quat=quat_axis((1, 0, 0), 180.0), dz=0.002)
    until(srv, lambda: srv.record.parcels[0]["scan"] is not None)
    steps(srv, 5)
    srv.watch.damaged[0] = {"t": round(srv.world.time, 4), "peak_N": 200.0, "reason": "squeeze"}
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("T1", "judge: the first downstream crossing", "crosses 6 cm in the air first, then (moved back) at rest",
        {"cls": "U", "readable": False, "failed": ["not_on_belt"], "lost_reason": None, "credit": -1})
def t1(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.95 - 0.002, dz=0.06)
    until(srv, lambda: srv.record.parcels[0]["scan"] is not None)
    place(srv, 0, 0.80)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("T2", "judge: downstream crossings only", "the parcel crosses the scan line backwards (thrown upstream), "
        "then the belt brings it across downstream: judged at that later crossing",
        {"cls": "U", "backward_crossings": ">= 1", "judged_after_backward": True, "lost_reason": None})
def t2(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.952, v=-0.6)
    rec = to_end(srv)
    ev = [e for e in srv.world.truth.events if e["type"] == "line" and e["line"] == "scan"] if srv.world else []
    v = verdict(rec, 0)
    back = [e for e in ev if e["direction"] < 0]
    v["backward_crossings"] = len(back)
    v["judged_after_backward"] = bool(back) and rec["parcels"][0]["scan"] is not None and \
        rec["parcels"][0]["scan"]["t"] > back[0]["t"]
    return v, rec


@mutant("T3", "judge: a crossing, not a position", "the parcel is moved past the scan line by a state write "
        "(no crossing): never judged, lost as not across",
        {"cls": "U", "readable": False, "failed": [], "lost_reason": "not_across", "credit": -2})
def t3(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 1.05)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("L1a", "lost: not across by the shift's end", "the shift ends (its duration) before a W parcel reaches the "
        "scan line", {"cls": "W", "readable": False, "failed": [], "lost_reason": "not_across", "credit": -1})
def l1a(_):
    rec = to_end(server(shift(dict(label_face=1), duration=6.0)))
    return verdict(rec, 0), rec


@mutant("L1b", "lost: not across by the shift's end", "finish() called while a U parcel is still upstream",
        {"cls": "U", "readable": False, "failed": [], "lost_reason": "not_across", "credit": -2})
def l1b(_):
    srv = server(shift({}))
    steps(srv, 150)
    srv.handle("finish", {})
    rec = srv.finalize()
    return verdict(rec, 0), rec


@mutant("L1c", "lost: not across by the shift's end", "solve returns at once (the launcher's finish): every parcel "
        "lost", {"cls": "W", "readable": False, "failed": [], "lost_reason": "not_across", "credit": -1})
def l1c(_):
    srv = server(shift(dict(label_face=1)))
    srv.handle("finish", {})
    rec = srv.finalize()
    return verdict(rec, 0), rec


@mutant("L2a", "lost: fell", "a label-up (U) parcel pushed off the open near edge in the work window (it drops)",
        {"cls": "U", "readable": False, "failed": [], "lost_reason": "fall", "credit": -2})
def l2a(_):
    srv = server(shift(dict(size=(0.20, 0.16, 0.10))))
    spawned(srv, 0)
    steps(srv, 20)
    w = srv.world
    place(srv, 0, 0.2, x=w.station.belt_center_x - w.station.half_width - 0.12, dz=0.0, v=0.0)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("L2b", "lost: fell", "a W parcel dropped to the floor beside the belt",
        {"cls": "W", "readable": False, "failed": [], "lost_reason": "fall", "credit": -1})
def l2b(_):
    srv = server(shift(dict(label_face=1)))
    spawned(srv, 0)
    until(srv, lambda: srv.record.parcels[0]["cls"] is not None)
    w = srv.world
    w.set_pose(0, [w.station.belt_center_x + 0.45, 0.3, 0.2], (1, 0, 0, 0), [0] * 6)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("L2c", "lost: fell", "a label-up (U) parcel knocked over in the work window (turned onto its side, it stays "
        "on the belt): crosses unreadable", {"cls": "U", "readable": False, "failed": ["label_not_up"],
                                             "lost_reason": None, "credit": -1})
def l2c(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.0, quat=quat_axis((0, 1, 0), 90.0), dz=0.002)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("L2d", "lost: the belt end is not a loss", "an untouched U parcel crosses and leaves at the belt end",
        {"cls": "U", "readable": True, "failed": [], "lost_reason": None, "credit": 0}, benign=True)
def l2d(_):
    srv = server(shift({}))
    rec = to_end(srv)
    v = verdict(rec, 0)
    v["exit_t"] = rec["parcels"][0]["exit_t"]
    return v, rec


@mutant("C1a", "class at arrival", "a label-up parcel turned onto its side right after its arrival: still U, now "
        "unreadable (-1)", {"cls": "U", "readable": False, "failed": ["label_not_up"], "lost_reason": None,
                            "credit": -1})
def c1a(_):
    srv = server(shift({}))
    spawned(srv, 0)
    steps(srv, 3)
    place(srv, 0, -1.05, quat=quat_axis((1, 0, 0), 90.0), dz=0.002)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("C1b", "class at arrival", "a label-down parcel turned label up right after its arrival: still W, readable "
        "(+1)", {"cls": "W", "readable": True, "failed": [], "lost_reason": None, "credit": 1, "score": 1.0})
def c1b(_):
    srv = server(shift(dict(label_face=1)))
    spawned(srv, 0)
    steps(srv, 3)
    place(srv, 0, -1.05, quat=quat_axis((1, 0, 0), 180.0), dz=0.002)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("C1c", "class at arrival", "a parcel that never gets far from the entry: classed by its landing face",
        {"cls": "W", "readable": False, "failed": [], "lost_reason": "not_across", "credit": -1})
def c1c(_):
    rec = to_end(server(shift(dict(label_face=4), duration=1.0)))
    return verdict(rec, 0), rec


@mutant("S1", "credit arithmetic", "constructed rows: 5 W (4 readable, one of them lost), 2 U (1 not readable)",
        {"credit": 2, "score": 0.4})
def s1(_):
    srv = server(shift({}))
    rec = srv.record
    rec.reset()
    tmpl = rec.parcels[0]
    rows = []
    for cls, readable, lost in [("W", True, False), ("W", True, False), ("W", True, False), ("W", False, False),
                                ("U", True, False), ("U", False, False), ("W", True, True)]:
        r = copy.deepcopy(tmpl)
        r["cls"], r["readable"] = cls, readable
        r["scan"] = {"t": 1.0}
        if lost:
            r["fall_t"] = 2.0
        rows.append(r)
    rec.parcels = rows
    f = rec.finalize()
    return {"credit": f["credit"], "score": f["shift_score"]}, f


@mutant("S2", "credit arithmetic: the floor", "constructed rows: credit negative -> score 0",
        {"credit": -3, "score": 0.0})
def s2(_):
    srv = server(shift({}))
    rec = srv.record
    rec.reset()
    tmpl = rec.parcels[0]
    rows = []
    for cls, readable in [("W", False), ("U", False), ("U", False), ("W", False)]:
        r = copy.deepcopy(tmpl)
        r["cls"], r["readable"] = cls, readable
        r["scan"] = {"t": 1.0}
        rows.append(r)
    rows[-1]["scan"] = None
    rec.parcels = rows
    f = rec.finalize()
    return {"credit": f["credit"], "score": f["shift_score"]}, f


@mutant("S3", "credit arithmetic: no W parcel", "a shift of U parcels only, all readable: score 0 (no division)",
        {"credit": 0, "score": 0.0})
def s3(_):
    rec = to_end(server(shift({}, {}, dt=8.0)))
    return {"credit": rec["credit"], "score": rec["shift_score"]}, rec


@mutant("K1", "lock-out: parcels still judged", "the robot locked out at t = 0 (entered into the watch): an "
        "untouched U parcel still crosses readable", {"cls": "U", "readable": True, "failed": [], "lost_reason": None,
                                                    "credit": 0, "locked_out": True})
def k1(_):
    srv = server(shift({}))
    srv.watch.locked_out = True
    srv.watch.lockout_reason = "constructed"
    srv.watch._halt()
    rec = to_end(srv)
    v = verdict(rec, 0)
    v["locked_out"] = rec["safety"]["locked_out"]
    return v, rec


@mutant("K2", "lock-out: commands refused", "after the lock-out a step with a target is refused and nothing of it "
        "is applied", {"refused": "locked_out", "target_applied": False})
def k2(_):
    import numpy as np
    srv = server(shift({}))
    srv.watch.locked_out = True
    srv.watch._halt()
    q = np.asarray(srv.describe()["robot"]["start_posture"], float)
    q2 = q.copy()
    q2[3] -= 0.5
    r = srv.handle("step", {"q_target": q2.tolist(), "n": 5})
    rec = srv.finalize()
    return {"refused": r["refused"], "target_applied": bool(abs(r["q_target"][3] - q2[3]) < 1e-6)}, rec


# ------------------------------------------------------------------------------------------------ the duty's classes
@mutant("X1", "class: outside the duty earns nothing", "an X parcel (a carton with its label on the upstream face, "
        "outside the default duty) turned exactly label up after its arrival, beside an untouched W parcel: readable, "
        "credit 0", {"cls": "X", "readable": True, "failed": [], "lost_reason": None, "credit": 0, "score": 0.0})
def x1(_):
    srv = server(shift(dict(label_face=5), dict(label_face=1), dt=8.0))
    spawned(srv, 0)
    steps(srv, 5)
    place(srv, 0, y_of(srv, 0), quat=quat_axis((1, 0, 0), -90.0), dz=0.002)
    rec = to_end(srv)
    v = verdict(rec, 0)
    v["other"] = rec["parcels"][1]["cls"]
    return v, rec


@mutant("X2", "class: an untouched X parcel costs nothing", "a carton with its label on the upstream face left alone "
        "(it crosses label not up)", {"cls": "X", "readable": False, "failed": ["label_not_up"], "lost_reason": None,
                                      "credit": 0})
def x2(_):
    rec = to_end(server(shift(dict(label_face=5))))
    return verdict(rec, 0), rec


@mutant("X3", "lost: any class", "a soft mailer with its label underneath (X: mailers are outside the duty) dropped "
        "to the floor beside the belt: lost, -1", {"cls": "X", "readable": False, "failed": [], "lost_reason": "fall",
                                                   "credit": -1})
def x3(_):
    srv = server(shift(dict(kind="soft", size=(0.26, 0.20, 0.07), label_face=1)))
    spawned(srv, 0)
    w = srv.world
    w.set_pose(0, [w.station.belt_center_x + 0.45, 0.3, 0.2], (1, 0, 0, 0), [0] * 6)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("X4", "judge (1) for U: a mailer knocked over", "a label-up soft mailer (U) turned onto its side in the work "
        "window: crosses unreadable, -1", {"cls": "U", "readable": False, "failed_includes": ["label_not_up"],
                                          "lost_reason": None, "credit": -1})
def x4(_):
    srv = server(shift(dict(kind="soft", size=(0.26, 0.20, 0.07), label_face=0)))
    spawned(srv, 0)
    steps(srv, 20)
    place(srv, 0, 0.0, quat=quat_axis((0, 1, 0), 90.0), dz=0.004)
    rec = to_end(srv)
    return verdict(rec, 0), rec


@mutant("X5", "class: the duty is one config entry", "the same upstream-face carton is X under the default duty and "
        "W when the duty lists the upstream face (score.duty)", {"default": "X", "widened": "W"})
def x5(_):
    a = server(shift(dict(label_face=5)))
    b = server(shift(dict(label_face=5)), over={"score": {"duty": {"kinds": ["rigid"], "label_positions": [
        "underneath", "downstream", "upstream", "near_side", "far_side"]}}})
    out = {"default": a.record.parcels[0]["cls"], "widened": b.record.parcels[0]["cls"]}
    a.close()
    b.close()
    return out, None


@mutant("X6", "class: the label position is the world's", "a label on the parcel's +x face is on the far side at "
        "yaw 0, on the downstream face at yaw 90 deg, on the near side at yaw 180 deg, upstream at yaw 270 deg: "
        "only the downstream one is in the default duty",
        {"positions": ["far_side", "downstream", "near_side", "upstream"], "classes": ["X", "W", "X", "X"]})
def x6(_):
    out = {"positions": [], "classes": []}
    for yaw in (0.0, math.pi / 2, math.pi, 1.5 * math.pi):
        srv = server(shift(dict(label_face=2, yaw=yaw)))
        out["positions"].append(srv.record.parcels[0]["label_position"])
        out["classes"].append(srv.record.parcels[0]["cls"])
        srv.close()
    return out, None


# ------------------------------------------------------------------------------------------------ the line setting
def line_shift(*parcels, v=V, indexing=False, duration=60.0, y=-1.1):
    """A line shift (fed on settle) of the given parcels, each set down at y on the centre line."""
    from parcelcell.shiftgen import line_setting
    from parcelcell.stream import Arrival, Parcel, Shift
    arr = []
    for kw in parcels:
        p = dict(kind="rigid", size=(0.24, 0.18, 0.12), mass=0.8, friction=0.55, label_face=0, label_uv=(0.0, 0.0),
                 yaw=0.0, lateral=0.0)
        p.update(kw)
        arr.append(Arrival(t=0.0, parcel=Parcel(**p), y=y))
    return Shift(belt_speed=((0.0, v),), arrivals=tuple(arr), duration=duration, seed=4343,
                 setting=line_setting(common.load_cfg(), indexing))


def line_verdict(rec, i):
    p = rec["parcels"][i]
    return {"cls": p["cls"], "readable": bool(p["readable"]), "failed": sorted(p["failed"]),
            "lost_reason": p["lost_reason"], "not_done_reason": p.get("not_done_reason"), "credit": rec["credit"],
            "score": rec["shift_score"], "n_W": rec["n_W"]}


@mutant("N1a", "line: class", "on a line shift every parcel is in the duty: a soft parcel with its label underneath, a "
        "rigid parcel with its label on the upstream face and one on a side face are W, a label-up one U, none X",
        {"classes": ["W", "W", "W", "U"]})
def n1a(_):
    srv = server(line_shift(dict(kind="soft", size=(0.24, 0.18, 0.08), label_face=1), dict(label_face=5),
                            dict(label_face=2), dict()))
    out = {"classes": [r["cls"] for r in srv.record.parcels]}
    srv.close()
    return out, None


@mutant("N1b", "line: not done", "the shift ends (its duration) before a W parcel reaches the scan line: not done, "
        "costs nothing", {"cls": "W", "readable": False, "lost_reason": None, "not_done_reason": "in_station",
                          "credit": 0, "score": 0.0})
def n1b(_):
    rec = to_end(server(line_shift(dict(label_face=1), duration=6.0)))
    return line_verdict(rec, 0), rec


@mutant("N1c", "line: not done", "finish() while a U parcel is still upstream: not done, costs nothing (on the rated "
        "setting -2)", {"cls": "U", "readable": False, "lost_reason": None, "not_done_reason": "in_station",
                         "credit": 0})
def n1c(_):
    srv = server(line_shift({}))
    steps(srv, 150)
    srv.handle("finish", {})
    rec = srv.finalize()
    return line_verdict(rec, 0), rec


@mutant("N1d", "line: not done", "the first parcel is held in the station (kept upstream of the scan line) to the "
        "shift's end: the second is never set down; both W count in the divisor",
        {"not_done": {"not_set_down": 1, "in_station": 1}, "credit": 0, "n_W": 2, "spawned": 1})
def n1d(_):
    srv = server(line_shift(dict(label_face=1), dict(label_face=1), duration=20.0))
    for _ in range(int(20.0 / 0.02)):
        if srv.handle("step", {"n": 1})["shift_over"]:
            break
        if y_of(srv, 0) > 0.0:
            place(srv, 0, -0.6, quat=quat_axis((1, 0, 0), 0.0), dz=0.0)
    rec = srv.finalize()
    return {"not_done": rec["not_done"], "credit": rec["credit"], "n_W": rec["n_W"], "spawned": rec["spawned"]}, rec


@mutant("N2a", "line: feed", "the second parcel is set down at the tick after the first crossed the scan line",
        {"next_after_scan_s": 0.0})
def n2a(_):
    srv = server(line_shift({}, {}, v=0.3))
    rec = to_end(srv)
    t_scan = rec["parcels"][0]["scan"]["t"]
    return {"next_after_scan_s": round(rec["parcels"][1]["t_spawn"] - t_scan, 6)}, rec


@mutant("N2b", "line: feed", "the first parcel falls off the belt: the next one is set down at the following tick",
        {"lost_reason": "fall", "next_after_fall_s": 0.0})
def n2b(_):
    srv = server(line_shift(dict(label_face=1), {}))
    spawned(srv, 0)
    steps(srv, 10)
    w = srv.world
    w.set_pose(0, [w.station.belt_center_x + 0.45, -0.5, 0.2], (1, 0, 0, 0), [0] * 6)
    rec = to_end(srv)
    return {"lost_reason": rec["parcels"][0]["lost_reason"],
            "next_after_fall_s": round(rec["parcels"][1]["t_spawn"] - rec["parcels"][0]["fall_t"], 6)}, rec


@mutant("N2c", "line: feed", "the first parcel is damaged (entered into the safety watch) in the station: it is settled, "
        "the next is set down at the following tick; the damaged one is lost",
        {"lost_reason": "damaged", "next_after_damage_s": 0.0})
def n2c(_):
    srv = server(line_shift(dict(label_face=1), {}))
    spawned(srv, 0)
    steps(srv, 200)                      # 4 s: well clear of the set-down place (a parcel there makes the next wait)
    t_d = round(float(srv.world.time), 4)
    srv.watch.damaged[0] = {"t": t_d, "peak_N": 999.0, "reason": "constructed"}
    rec = to_end(srv)
    return {"lost_reason": rec["parcels"][0]["lost_reason"],
            "next_after_damage_s": round(rec["parcels"][1]["t_spawn"] - t_d, 6)}, rec


@mutant("N3a", "line: indexing", "on an indexing line the belt stands with the parcel's centre in the zone and runs "
        "again once it has been moved past the zone; the parcel then rides on and is judged",
        {"halted": True, "belt_stood": True, "runs_again": True, "judged": True})
def n3a(_):
    srv = server(line_shift({}, v=0.25, indexing=True))
    w = srv.world
    until(srv, lambda: w._halted, limit=1000)
    halted = bool(w._halted)
    steps(srv, 50)
    stood = abs(float(w.data.qvel[w._slab_dadr])) < 1e-3
    zone_hi = float(w.build_shift.setting["zone_y"][1])
    place(srv, 0, zone_hi + 0.05, v=0.0)
    steps(srv, 25)
    runs = abs(float(w.data.qvel[w._slab_dadr]) - 0.25) < 0.01
    rec = to_end(srv)
    return {"halted": halted, "belt_stood": stood, "runs_again": runs,
            "judged": rec["parcels"][0]["scan"] is not None}, rec


@mutant("N3b", "line: indexing", "a parcel left in the zone (nobody moves it) stalls the list: the shift runs to its "
        "duration, the rest of the list is never set down, credit 0",
        {"credit": 0, "not_done": {"not_set_down": 2, "in_station": 1}, "end": "shift_end"})
def n3b(_):
    srv = server(line_shift(dict(label_face=1), {}, dict(label_face=1), v=0.3, indexing=True, duration=20.0))
    rec = to_end(srv)
    return {"credit": rec["credit"], "not_done": rec["not_done"], "end": rec["done_reason"]}, rec


@mutant("N4a", "line: credit", "of four W parcels one is turned label up after its set-down: score exactly 1/4",
        {"credit": 1, "score": 0.25})
def n4a(_):
    srv = server(line_shift(dict(label_face=1), dict(label_face=1), dict(label_face=1), dict(label_face=1), v=0.3))
    spawned(srv, 0)
    steps(srv, 3)
    place(srv, 0, -1.05, quat=quat_axis((1, 0, 0), 180.0), dz=0.002, v=0.3)
    rec = to_end(srv)
    return {"credit": rec["credit"], "score": rec["shift_score"]}, rec


@mutant("N4b", "line: credit", "two W parcels turned label up, one U parcel turned label down: the spoiled U costs 1 "
        "(credit 1, score 1/3)", {"credit": 1, "score": 1.0 / 3.0})
def n4b(_):
    srv = server(line_shift(dict(label_face=1), {}, dict(label_face=1), dict(label_face=1), v=0.3))
    for i, deg in ((0, 180.0), (1, 180.0), (2, 180.0)):
        spawned(srv, i)
        steps(srv, 3)
        place(srv, i, -1.05, quat=quat_axis((1, 0, 0), deg), dz=0.002, v=0.3)
    rec = to_end(srv)
    return {"credit": rec["credit"], "score": rec["shift_score"]}, rec


@mutant("N4c", "line: end", "the shift ends at the end of the tick in which its list's last parcel is settled",
        {"end": "shift_end", "t_end_is_list_done": True, "before_duration": True})
def n4c(_):
    srv = server(line_shift({}, {}, v=0.3, duration=60.0))
    rec = to_end(srv)
    return {"end": rec["done_reason"], "t_end_is_list_done": rec["t_end"] == rec["list_done_t"],
            "before_duration": rec["t_end"] < 60.0}, rec


def _held(srv, i, s_hold):
    """Hold parcel i still in place for s_hold (a state write every tick), then let it ride."""
    import numpy as np
    w = srv.world
    q = w._slot_qadr[i]
    pos, quat = w.data.qpos[q:q + 3].copy(), w.data.qpos[q + 3:q + 7].copy()
    for _ in range(int(round(s_hold / 0.02))):
        srv.handle("step", {"n": 1})
        w.set_pose(i, pos, quat, np.zeros(6))


@mutant("N5a", "line: due", "a W parcel turned label up right after its set-down, then held 6 s (1 s past the rated "
        "cycle): late, not readable, credit 0", {"cls": "W", "readable": False, "failed": ["late"], "credit": 0})
def n5a(_):
    srv = server(line_shift(dict(label_face=1), v=0.3))
    spawned(srv, 0)
    steps(srv, 3)
    place(srv, 0, -1.05, quat=quat_axis((1, 0, 0), 180.0), dz=0.002, v=0.3)
    steps(srv, 2)
    _held(srv, 0, 6.0)
    rec = to_end(srv)
    return line_verdict(rec, 0), rec


@mutant("N5b", "line: due", "the same W parcel held 4 s: in time, readable (+1)",
        {"cls": "W", "readable": True, "failed": [], "credit": 1, "score": 1.0})
def n5b(_):
    srv = server(line_shift(dict(label_face=1), v=0.3))
    spawned(srv, 0)
    steps(srv, 3)
    place(srv, 0, -1.05, quat=quat_axis((1, 0, 0), 180.0), dz=0.002, v=0.3)
    steps(srv, 2)
    _held(srv, 0, 4.0)
    rec = to_end(srv)
    return line_verdict(rec, 0), rec


@mutant("N5c", "line: due", "a U parcel held 6 s: late, it counts as made unreadable (-1)",
        {"cls": "U", "readable": False, "failed": ["late"], "credit": -1})
def n5c(_):
    srv = server(line_shift({}, v=0.3))
    spawned(srv, 0)
    steps(srv, 5)
    _held(srv, 0, 6.0)
    rec = to_end(srv)
    return line_verdict(rec, 0), rec


@mutant("N5d", "line: due", "on an indexing line the halt counts against the rated cycle: a U parcel left in the zone "
        "for 8 s, then moved past it, is late (-1)", {"cls": "U", "failed": ["late"], "credit": -1})
def n5d(_):
    srv = server(line_shift({}, v=0.3, indexing=True))
    w = srv.world
    until(srv, lambda: w._halted, limit=1000)
    steps(srv, 400)
    place(srv, 0, float(w.build_shift.setting["zone_y"][1]) + 0.05, v=0.0)
    rec = to_end(srv)
    v = line_verdict(rec, 0)
    return {k: v[k] for k in ("cls", "failed", "credit")}, rec


@mutant("N5e", "line: due", "a parcel riding free crosses well inside its due time (free ride = due time - 5 s)",
        {"late": False, "margin_ok": True})
def n5e(_):
    srv = server(line_shift({}, dict(label_face=1), v=0.05, duration=120.0))
    rec = to_end(srv)
    p = rec["parcels"][0]
    return {"late": p["late"], "margin_ok": 4.5 < p["due_t"] - p["scan"]["t_cross"] < 5.0 + 1e-9}, rec


def _expect_ok(exp, got):
    for k, v in exp.items():
        if k == "failed_includes":
            if not set(v) <= set(got.get("failed") or []):
                return False
            continue
        if v == ">= 1":
            if not (isinstance(got.get(k), int) and got[k] >= 1):
                return False
        elif isinstance(v, float):
            if got.get(k) is None or abs(float(got[k]) - v) > 1e-9:
                return False
        elif got.get(k) != v:
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ctx = {"teach_left_in": TEACH_LEFT_IN}
    rows = []
    for m in M:
        if a.only and m["id"] not in a.only:
            continue
        try:
            got, rec = m["fn"](ctx)
            err = None
        except Exception:  # noqa: BLE001
            got, err = {}, traceback.format_exc(limit=3)
        ok = err is None and _expect_ok(m["expect"], got)
        rows.append({"id": m["id"], "sentence": m["sentence"], "what": m["what"], "benign": m["benign"],
                     "expected": m["expect"], "measured": got, "pass": ok, "error": err})
        print("%-4s %-4s %s\n      expected %s\n      measured %s" % (m["id"], "ok" if ok else "FAIL", m["what"],
                                                                   m["expect"], got), flush=True)
        if err:
            print(err)
    common.dump_json(os.path.join(a.out, "mutants.json"), {"rows": rows, "notes": ctx.get("notes", {}),
                                                            "all_pass": all(r["pass"] for r in rows)})
    print("\n%d of %d mutants give the verdict their sentence demands" % (sum(r["pass"] for r in rows), len(rows)))
    return 0 if all(r["pass"] for r in rows) else 1


# The left arm's posture over the belt (tool down, the tool centre point 6 cm above the belt at y = +0.15), taught
# offline on the given models (the degenerate program p2's second posture).
TEACH_LEFT_IN = None


def _teach():
    global TEACH_LEFT_IN
    import importlib.util
    p = os.path.join(HERE, "ladder", "programs", "p2_arms_out.py")
    spec = importlib.util.spec_from_file_location("review_p2_teach", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(p))
    spec.loader.exec_module(mod)
    TEACH_LEFT_IN = list(mod.BOTH_IN)


_teach()

if __name__ == "__main__":
    sys.exit(main())
