"""Reference cable router for cable_routing_through_clips (serves the closed-loop wire).

Routing one fixture, the way a technician does it by hand:

1. Pick the cable up at the arc length (from the connector) it needs to reach a point a few
   centimetres past the fixture, measured along the route from where the cable is already held.
2. Carry it there high enough that the cable hangs clear, so the stretch from the last fixed
   point runs straight over the fixture.
3. Clip: lower it along that line and pull a little, so the cable drops between the pins.
   Post: lower it on the outer side of the post, then swing the grip around the post so the
   cable wraps it, and pull.
4. Let go, lift, go on with the next fixture.

All decisions (which fixture, which arc length) are kept in relative terms and every target is
recomputed from the observation in hand, so a step asked twice with the same `t` is answered
from each request's own observation and never advances anything.
"""

import copy
import json
import math
import sys

import numpy as np

REACH = 0.06            # grasp this far past the fixture
SLACK = 0.010           # extra cable length when picking
PULL = 0.007            # extra pull while wrapping a post
TAUT_N = 1.5            # connector load that means the cable is taut
TENSION_N = 5.0         # back off once the connector load cell reads this
LOW_N = 1.2             # pull out while the load is below this (slack)
CARRY_Z = 0.035
HIGH_Z = 0.06            # carry height on the taut sphere
GRASP_Z = 0.0015
LAY_Z = 0.004
HOVER_Z = 0.06


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _arc(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])


def _at_arc(P, s):
    a = _arc(P)
    s = min(max(s, 0.0), a[-1])
    i = int(np.clip(np.searchsorted(a, s) - 1, 0, len(P) - 2))
    f = (s - a[i]) / max(a[i + 1] - a[i], 1e-9)
    p = P[i] + f * (P[i + 1] - P[i])
    tang = _unit(P[i + 1, :2] - P[i, :2])
    return p, tang


def _turn(d, side, angle=1.0):
    """Direction d turned toward a post's side (a final post still needs a bend to wrap)."""
    a = angle if side == "left" else -angle
    c, s = math.cos(a), math.sin(a)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def route_points(obs):
    """Route geometry from the observation: connector exit, then per fixture its passage point
    and the directions in and out."""
    R = obs["cable_radius"]
    q = np.array(obs["connector"]["pos"], float)
    pts = []
    fx = obs["fixtures"]
    # first pass: passage points (clip centre; post: tangent point on the route side)
    for i, f in enumerate(fx):
        c = np.array(f["pos"], float)
        if f["type"] == "clip":
            pts.append(c)
        else:
            pts.append(None)
    # posts need the neighbouring points; resolve iteratively using clip/connector neighbours
    for _ in range(3):
        for i, f in enumerate(fx):
            if f["type"] != "post":
                continue
            prev = q if i == 0 else (pts[i - 1] if pts[i - 1] is not None else np.array(fx[i - 1]["pos"], float))
            nxt = pts[i + 1] if i + 1 < len(fx) and pts[i + 1] is not None else (
                np.array(fx[i + 1]["pos"], float) if i + 1 < len(fx) else None)
            c = np.array(f["pos"], float)
            r = f["radius"] + R
            d_in = _unit(c - prev)
            d_out = _unit(nxt - c) if nxt is not None else _turn(d_in, f["side"])
            # corner point: on the bisector away from the post
            bis = _unit(d_in - d_out) if np.linalg.norm(d_in - d_out) > 1e-6 else np.array([-d_in[1], d_in[0]])
            pts[i] = c + r * bis
    return q, pts


PIN_R = 0.0025
SEATED_Z = 0.013


def _crosses(P, a, b, zmax):
    s = 0.0
    for j in range(len(P) - 1):
        p, q = P[j, :2], P[j + 1, :2]
        r, w = q - p, b - a
        den = r[0] * w[1] - r[1] * w[0]
        ln = float(np.linalg.norm(r))
        if abs(den) > 1e-12:
            t = ((a[0] - p[0]) * w[1] - (a[1] - p[1]) * w[0]) / den
            u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                z = P[j, 2] + t * (P[j + 1, 2] - P[j, 2])
                if z < zmax:
                    return s + t * ln
        s += ln
    return None


def fixture_status(obs):
    """Which fixtures the observed cable satisfies, in route order (the cell's own check)."""
    P = np.array(obs["cable"], float)
    R = obs["cable_radius"]
    arc = _arc(P)
    ok = []
    last = -1.0
    for f in obs["fixtures"]:
        c = np.array(f["pos"], float)
        if f["type"] == "clip":
            a = f["axis"]
            n = np.array([-math.sin(a), math.cos(a)]) * (f["gap"] / 2 + PIN_R)
            sx = _crosses(P, c - n, c + n, SEATED_Z + 0.002)
        else:
            dist = np.linalg.norm(P[:, :2] - c, axis=1)
            j = int(np.argmin(dist))
            jj = min(max(j, 1), len(P) - 2)
            tang = P[jj + 1, :2] - P[jj - 1, :2]
            cross = tang[0] * (c[1] - P[j, 1]) - tang[1] * (c[0] - P[j, 0])
            side_ok = (cross > 0) if f["side"] == "left" else (cross < 0)
            sx = float(arc[j]) if (dist[j] <= f["radius"] + R + 0.008 and side_ok) else None
        good = sx is not None and sx > last
        ok.append(good)
        if good:
            last = sx
    return ok


class Router(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = {"k": 0, "phase": "plan", "n": 0, "s": 0.0, "yaw": 0.0, "pa": 0.0}
        self.last_t = None
        self.saved = None

    def act(self, t, obs):
        if t == self.last_t:
            st = copy.deepcopy(self.saved)
        else:
            self.saved = copy.deepcopy(self.state)
            st = copy.deepcopy(self.state)
            self.last_t = t
        action = self._policy(st, obs)
        self.state = st
        return action

    # ----------------------------------------------------------------------------------
    def _geometry(self, st, obs):
        k = st["k"]
        fx = obs["fixtures"]
        q, pts = route_points(obs)
        prev = q if k == 0 else pts[k - 1]
        T = pts[k]
        d_in = _unit(T - prev)
        if k + 1 < len(fx):
            d_out = _unit(pts[k + 1] - T)
        elif fx[k]["type"] == "post":
            d_out = _turn(d_in, fx[k]["side"])
        else:
            d_out = d_in
        return fx[k], prev, T, d_in, d_out

    def _held_arc(self, st, obs, P):
        """Arc length at which the cable passes the last fixed point."""
        k = st["k"]
        if k == 0:
            return 0.0
        q, pts = route_points(obs)
        a = _arc(P)
        j = int(np.argmin(np.linalg.norm(P[:, :2] - pts[k - 1], axis=1)))
        return float(a[j])

    def _policy(self, st, obs):
        g = obs["gripper"]
        gp = np.array(g["pos"], float)
        P = np.array(obs["cable"], float)
        fx = obs["fixtures"]
        R = obs["cable_radius"]
        load = obs["connector_force"]
        if st["k"] >= len(fx):
            return {"pos": [gp[0], gp[1], HOVER_Z], "yaw": st["yaw"], "grip": 0.0, "done": True}
        f, prev, T, d_in, d_out = self._geometry(st, obs)
        ph = st["phase"]

        def align(direction):
            a = math.atan2(direction[1], direction[0])
            cur = st["yaw"]
            return cur + ((a - cur + math.pi / 2) % math.pi - math.pi / 2)

        def out(pos, grip):
            return {"pos": [float(pos[0]), float(pos[1]), float(pos[2])], "yaw": st["yaw"], "grip": grip}

        if f["type"] == "clip":
            land = T + REACH * d_in                     # where the grip lands: straight on past the clip
        else:
            # straight on past the corner, a little outward so the cable clears the post
            side = 1.0 if f["side"] == "left" else -1.0
            outward = -side * np.array([-d_in[1], d_in[0]])
            land = T + REACH * d_in + 0.004 * outward

        if ph == "plan":
            s0 = self._held_arc(st, obs, P)
            need = float(np.linalg.norm(land - prev))
            st["s0"] = s0
            st["s"] = s0 + need + SLACK
            st["rad"] = need + SLACK - 0.003          # taut radius around the last fixed point
            st["phase"], st["n"] = "approach", 0
            ph = "approach"

        gpt, tang = _at_arc(P, st["s"])
        prev3 = np.array([prev[0], prev[1], R])

        def on_sphere(direction_xy, z):
            """Point at height z on the sphere of the taut radius around the last fixed point."""
            h = math.sqrt(max(st["rad"] ** 2 - (z - R) ** 2, 1e-6))
            dvec = _unit(direction_xy)
            return np.array([prev[0] + h * dvec[0], prev[1] + h * dvec[1], z])

        if ph == "approach":
            st["yaw"] = align(tang)
            if np.linalg.norm(gp[:2] - gpt[:2]) < 0.004 and abs(gp[2] - HOVER_Z) < 0.01:
                st["phase"], st["n"] = "descend", 0
            return out([gpt[0], gpt[1], HOVER_Z], 0.0)
        if ph == "descend":
            st["yaw"] = align(tang)
            st["n"] += 1
            if gp[2] < GRASP_Z + 0.002 or st["n"] > 25:
                st["phase"], st["n"] = "close", 0
            return out([gpt[0], gpt[1], GRASP_Z], 0.0)
        if ph == "close":
            st["n"] += 1
            if st["n"] >= 6:
                st["phase"], st["n"] = "lift", 0
            return out([gp[0], gp[1], GRASP_Z], 1.0)
        if ph == "lift":
            # straight up, but never further from the last fixed point than the free cable allows
            st["n"] += 1
            z = min(HIGH_Z, gp[2] + 0.02)
            v = gp[:2] - prev
            h = math.sqrt(max(st["rad"] ** 2 - (z - R) ** 2, 1e-6))
            xy = gp[:2] if np.linalg.norm(v) <= h else prev + _unit(v) * h
            st["yaw"] = align(gp[:2] - prev)
            if gp[2] > HIGH_Z - 0.005 or st["n"] > 25:
                st["phase"], st["n"] = "swing", 0
            return out([xy[0], xy[1], z], 1.0)
        if ph == "swing":
            # around the last fixed point at the taut radius, toward the landing direction
            st["n"] += 1
            a_cur = math.atan2(gp[1] - prev[1], gp[0] - prev[0])
            a_tgt = math.atan2(land[1] - prev[1], land[0] - prev[0])
            da = (a_tgt - a_cur + math.pi) % (2 * math.pi) - math.pi
            a_next = a_cur + max(-0.12, min(0.12, da))
            if load > TENSION_N:
                st["rad"] -= 0.004
            elif load < LOW_N:
                st["rad"] += 0.001
            tgt = on_sphere(np.array([math.cos(a_next), math.sin(a_next)]), HIGH_Z)
            st["yaw"] = align(tgt[:2] - prev)
            if abs(da) < 0.01 or st["n"] > 60:
                st["phase"], st["n"] = "lower", 0
            return out(tgt, 1.0)
        if ph == "lower":
            # down the sphere: the taut cable comes down straight onto the fixture
            st["n"] += 1
            if load > TENSION_N:
                st["rad"] -= 0.004
            elif load < LOW_N:
                st["rad"] += 0.002
            a = min(1.0, st["n"] / 14.0)
            z = HIGH_Z + a * (LAY_Z - HIGH_Z)
            tgt = on_sphere(land - prev, z)
            st["yaw"] = align(land - prev)
            if st["n"] >= 18:
                st["phase"], st["n"] = ("release" if f["type"] == "clip" else "sweep"), 0
            return out(tgt, 1.0)
        if ph == "sweep":
            # swing around the post corner from d_in to d_out at REACH, keeping the cable taut
            a0 = math.atan2(land[1] - T[1], land[0] - T[0])
            a1 = math.atan2(d_out[1], d_out[0])
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            st["n"] += 1
            frac = min(1.0, st["n"] / 20.0)
            ang = a0 + frac * da
            if load > TENSION_N:
                st["pr"] = st.get("pr", 0.0) - 0.004
            elif load < 0.6 * TAUT_N:
                st["pr"] = min(st.get("pr", 0.0) + 0.001, 0.03)
            rad = REACH + st.get("pr", 0.0)
            xy = T + rad * np.array([math.cos(ang), math.sin(ang)])
            st["yaw"] = align(np.array([math.cos(ang), math.sin(ang)]))
            if st["n"] >= 26:
                st["phase"], st["n"], st["pr"] = "release", 0, 0.0
            return out([xy[0], xy[1], LAY_Z + 0.002], 1.0)
        if ph == "release":
            st["n"] += 1
            if st["n"] >= 6:
                st["phase"], st["n"] = "retreat", 0
            return out([gp[0], gp[1], gp[2]], 0.0)
        if ph == "retreat":
            st["n"] += 1
            if gp[2] > HOVER_Z - 0.01 or st["n"] > 15:
                st["phase"], st["n"] = "check", 0
            return out([gp[0], gp[1], HOVER_Z], 0.0)
        if ph == "check":
            # let the cable settle, then verify every fixture so far and redo the first one missed
            st["n"] += 1
            if st["n"] < 5:
                return out([gp[0], gp[1], HOVER_Z], 0.0)
            status = fixture_status(obs)
            first_bad = next((i for i in range(st["k"] + 1) if not status[i]), None)
            tries = st.setdefault("tries", {})
            if first_bad is None:
                st["k"] += 1
            else:
                key = str(first_bad)
                tries[key] = tries.get(key, 0) + 1
                if tries[key] >= 3:
                    st["k"] += 1                      # give up on it and go on
                else:
                    st["k"] = first_bad
            st["phase"], st["n"] = "plan", 0
            return out([gp[0], gp[1], HOVER_Z], 0.0)
        return out(gp, 0.0)


def main():
    router = Router()
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            print(json.dumps({"error": "bad json"}), flush=True)
            continue
        kind = req.get("type")
        if kind == "reset":
            router.reset()
            print(json.dumps({"ok": True}), flush=True)
        elif kind == "act":
            try:
                print(json.dumps({"action": router.act(req["t"], req["obs"])}), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"error": str(exc)[:200]}), flush=True)
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return
        else:
            print(json.dumps({"error": "unknown request"}), flush=True)


if __name__ == "__main__":
    main()
