"""Grader controls of the degenerate ladder that WRITE the world's state (in process only; never a submission, never
on the wire). They certify that the score can move both ways on the same shifts the degenerate programs run on.

* ``positive``: every W parcel, once 0.15 m past the entry and before the work window, is turned in place so that its
  label faces straight up, set down on the belt at belt speed and left alone. Expected shift score 1 (every W parcel
  readable, no U parcel disturbed, nothing lost).
* ``turn_all_down``: every parcel 0.15 m past the entry is turned in place so that its label faces straight down.
  Expected shift score 0 (every U parcel becomes not readable; no W parcel readable).
* ``turn_x_only``: every X parcel (outside the station's duty, label not up) is turned label up, nothing else.
  Expected shift score 0: turning a parcel outside the duty earns nothing.
* ``turn_w_and_x``: every W and every X parcel is turned label up. Expected shift score 1 (the X parcels add
  nothing and cost nothing).

* ``turn_one_w``: the first W parcel only is turned label up. Expected credit exactly 1 (score 1 / #W).
* ``turn_two_w_spoil_one_u``: the first two W parcels turned label up and the first U parcel turned label down.
  Expected credit exactly 1 (+2 - 1: a spoiled U parcel costs 1).
* ``turn_one_w_late`` (the line setting's due time): the first W parcel turned label up, then held still in place for
  6 s (1 s more than the rated cycle allows). Expected credit exactly 0: a late parcel earns nothing.
* ``two_w_one_u_late``: the first two W parcels turned label up, the first U parcel held still for 6 s. Expected
  credit exactly 1 (+2 - 1: a late U parcel counts as made unreadable).

Both drive the world through ``server.step`` with the arms at the start posture (the speed limit, the safety watch
and the record apply as to any program). On an indexing line (the line setting) a parcel the belt holds in the zone
is moved past the zone once it has stood for half a second (a state write), so that the list goes on.
"""

from __future__ import annotations

import numpy as np


def _rot_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b (minimal)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _mat2quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def _turn(srv, i, up=True):
    """Turn parcel i in place so its label normal points up (or down); rest it on the belt at belt speed."""
    import mujoco
    w = srv.world
    st = w.truth.parcel_state(i)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, st.quat)
    R = R.reshape(3, 3)
    Rfix = _rot_between(st.label_normal, [0, 0, 1.0] if up else [0, 0, -1.0])
    Rn = Rfix @ R
    half = np.asarray(w.slots[i].parcel.size, float) / 2
    hz = float(np.sum(np.abs(Rn[2, :]) * half))
    pos = np.array([st.pos[0], st.pos[1], w.station.belt_height + hz + 0.002])
    v = float(w.data.qvel[w._slab_dadr])
    w.set_pose(i, pos, _mat2quat(Rn), np.r_[0.0, v, 0.0, 0.0, 0.0, 0.0])


def _drive(srv, pick, up, budget=None, hold=None, hold_s=6.0):
    """pick(record row) -> turn it? ``budget``: {class: how many of that class at most} (None: no limit); ``up``: a
    bool, or a function of the record row (None: not turned). ``hold``: {class: how many} of the picked parcels that
    are then held still in place for ``hold_s`` (a state write every tick)."""
    info = srv.describe()
    eye = info["station"]["lines"]["entry_y"] + 0.15            # the turning line: 0.15 m past the entry
    window = -info["station"]["belt"]["work_window_half_length"]
    q0 = info["robot"]["start_posture"]
    st = srv.handle("step", {"q_target": q0, "grip": [0.0, 0.0], "n": 1})
    done, turned, held = set(), [], 0
    left = dict(budget or {})
    hold_left = dict(hold or {})
    holding = {}                                   # slot -> (pose, ticks left)
    while not st["shift_over"]:
        st = srv.handle("step", {"n": 1})
        w = srv.world
        for i in list(holding):
            pos, quat, n_left = holding[i]
            if n_left <= 0 or not w.slots[i].active:
                del holding[i]
                continue
            w.set_pose(i, pos, quat, np.zeros(6))
            holding[i] = (pos, quat, n_left - 1)
        line = w.build_shift.setting or {}
        if line.get("indexing"):
            held = held + 1 if w._halted else 0
            i = w._station_parcel()
            if held >= 25 and i is not None:          # the belt stood half a second: move the parcel past the zone
                q = w._slot_qadr[i]
                pos = w.data.qpos[q:q + 3].copy()
                pos[1] = float(line["zone_y"][1]) + 0.05
                w.set_pose(i, pos, vel=np.zeros(6))
                held = 0
        for i, s in enumerate(w.slots[:w.n_parcels]):
            if not s.active or i in done:
                continue
            y = float(w.data.qpos[w._slot_qadr[i] + 1])
            r = srv.record.parcels[i]
            if r["cls"] is None or not (eye + 0.05 < y < window - 0.05):
                continue
            done.add(i)
            if pick(r) and (budget is None or left.get(r["cls"], 0) > 0):
                if budget is not None:
                    left[r["cls"]] -= 1
                u = up(r) if callable(up) else up
                if u is not None:
                    _turn(srv, i, up=u)
                    turned.append(i)
                if hold_left.get(r["cls"], 0) > 0:
                    hold_left[r["cls"]] -= 1
                    q = w._slot_qadr[i]
                    holding[i] = (w.data.qpos[q:q + 3].copy(), w.data.qpos[q + 3:q + 7].copy(),
                                  int(round(hold_s / 0.02)))
    return {"seen": sorted(done), "turned": turned}


def positive(srv):
    return _drive(srv, lambda r: r["cls"] == "W", up=True)


def turn_all_down(srv):
    return _drive(srv, lambda r: True, up=False)


def turn_x_only(srv):
    return _drive(srv, lambda r: r["cls"] == "X", up=True)


def turn_w_and_x(srv):
    return _drive(srv, lambda r: r["cls"] in ("W", "X"), up=True)


def turn_one_w(srv):
    return _drive(srv, lambda r: r["cls"] == "W", up=True, budget={"W": 1})


def turn_two_w_spoil_one_u(srv):
    return _drive(srv, lambda r: r["cls"] in ("W", "U"), up=lambda r: r["cls"] == "W", budget={"W": 2, "U": 1})


def turn_one_w_late(srv):
    return _drive(srv, lambda r: r["cls"] == "W", up=True, budget={"W": 1}, hold={"W": 1})


def two_w_one_u_late(srv):
    return _drive(srv, lambda r: r["cls"] in ("W", "U"), up=lambda r: True if r["cls"] == "W" else None,
                  budget={"W": 2, "U": 1}, hold={"U": 1})


CONTROLS = {"c1_positive_control": (positive, 1.0, "state write: every W parcel turned label up before the window"),
            "c2_turn_all_down": (turn_all_down, 0.0, "state write: every parcel turned label down 0.15 m past the entry"),
            "c3_turn_x_only": (turn_x_only, 0.0, "state write: every X parcel (outside the duty) turned label up, "
                                                 "nothing else"),
            "c4_turn_w_and_x": (turn_w_and_x, 1.0, "state write: every W and every X parcel turned label up"),
            "c5_turn_one_w": (turn_one_w, "credit 1", "state write: the first W parcel turned label up, nothing "
                                                      "else: credit exactly 1, score 1 / #W"),
            "c6_two_w_spoil_one_u": (turn_two_w_spoil_one_u, "credit 1", "state write: two W parcels turned label "
                                                                         "up, one U parcel turned label down: credit "
                                                                         "exactly 1 (a spoiled U costs 1)"),
            "c7_turn_one_w_late": (turn_one_w_late, "credit 0", "state write: the first W parcel turned label up, "
                                                                "then held 6 s: late on the line setting (credit 0)"),
            "c8_two_w_one_u_late": (two_w_one_u_late, "credit 1", "state write: two W parcels turned label up, one "
                                                                  "U parcel held 6 s: credit exactly 1 on the line "
                                                                  "setting (a late U costs 1)")}
