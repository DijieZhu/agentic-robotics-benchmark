import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shift as SH

K = SH.K
SIDES = SH.SIDES
DT = SH.DT
SPC = SH.SPC
CYCLE = SH.CYCLE
G_ABS = 9.81
LOAD_RAMP_END_S = 5.5
INTERNAL_AFTER_CAPTURE_S = 0.5
P90 = 0.9
RELEASED_CLAMP_N = 2.0
S_FOOT_ON_TOP_TOL_M = 0.0010


def cy(t):
    return int(round(t / CYCLE))


VISIT_TOL_M = 0.020
FOOT_ON_PLATE_TOL_M = 0.0010
CUP_INSIDE_TOL_M = 0.00025
Q_WEIGHTS = {"H": 0.30, "R": 0.25, "A": 0.25, "P": 0.20}
TERM_QUANTITIES = {"H": ("e_z_mm", "tilt_deg"), "R": ("rho_mm",), "A": ("e_xy_mm",), "P": ("d_m_mm",)}
PLACE_QUANTITY = "P_r_N"
BOX_GRID_M = 0.005


class TraceStore:
    VEC = {"tray.pose": 7, "tray.twist": 6, "g_left.wrench": 6, "g_right.wrench": 6}
    SCALAR = ("g_left.weld_state", "g_right.weld_state", "g_left.clamp_force", "g_right.clamp_force")

    def __init__(self):
        self._rows = {k: [] for k in ("step", "S_tray_N", *self.VEC, *self.SCALAR)}
        self._a = None

    def add(self, rec):
        r = self._rows
        r["step"].append(int(rec["step"]))
        for k in self.VEC:
            r[k].append(np.asarray(rec[k], dtype=np.float64).copy())
        for k in self.SCALAR:
            r[k].append(float(rec[k]))
        r["S_tray_N"].append(SH.Monitor.shelf_normal(rec))
        self._a = None

    def arrays(self):
        if self._a is None:
            a = {}
            for k, v in self._rows.items():
                if k == "step":
                    a[k] = np.asarray(v, dtype=np.int64)
                elif k in self.VEC:
                    a[k] = np.asarray(v, dtype=np.float64).reshape(-1, self.VEC[k])
                else:
                    a[k] = np.asarray(v, dtype=np.float64)
            self._a = a
        return self._a

    def save(self, path):
        a = self.arrays()
        np.savez_compressed(path, **{k.replace(".", "__"): v for k, v in a.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        t = cls()
        t._a = {k.replace("__", "."): z[k] for k in z.files}
        t._rows = None
        return t

    def index_of(self, step):
        s = self.arrays()["step"]
        i = int(np.searchsorted(s, step))
        return i if i < len(s) and int(s[i]) == int(step) else None


def _p90(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(math.ceil(P90 * len(v))) - 1)] if v else None


def stats(v):
    v = sorted(v)
    if not v:
        return None
    return {"n": len(v), "median": v[len(v) // 2], "p90": _p90(v), "max": v[-1]}


def provisional_full_end(reference_values, floor):
    p = _p90(list(reference_values))
    return max(float(floor), 1.25 * float(p)) if p is not None else float(floor)


def term_value(values, F):
    v = [min(1.0, max(0.0, (3.0 * F - x) / (2.0 * F))) for x in values]
    return sum(v) / len(v) if v else None


def _R(q):
    return SH.matrix_of_quat(np.asarray(q, float))


def tilt_deg(quat):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(_R(quat)[2, 2])))))


def diameter_mm(P):
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return 0.0
    best = 0.0
    for j in range(len(P) - 1):
        d = np.sqrt(np.max(np.sum((P[j + 1:] - P[j]) ** 2, axis=1)))
        if d > best:
            best = float(d)
    return 1e3 * best


class Quantity(dict):
    ""


class Run:

    def __init__(self, result, trace, slots, jobs, site):
        self.res = result
        self.tr = trace
        self.a = trace.arrays()
        self.slots = slots
        self.jobs = jobs
        self.site = site
        self.step0 = int(result["step0"])
        self.pres = {int(p["index"]): p for p in result["presentations"]}
        self.ev = result.get("unit_events", [])

    def t_shift(self, step):
        return (int(step) - self.step0) * DT

    def slot_of_step(self, step):
        cur = None
        for i, p in sorted(self.pres.items(), key=lambda kv: kv[1]["step"]):
            if step >= p["step"]:
                cur = i
        return cur

    def first_event(self):
        evs = self.res.get("events") or []
        st = self.res.get("stopped") or {}
        if evs:
            e = dict(evs[0])
            step = int(e["step"])
        elif st.get("error"):
            step = self.step0 + int(st["cycle"]) * SPC
            e = {"kind": st["error"]["name"], "message": st["error"]["message"], "step": step}
        elif st.get("plant_invalid") or st.get("integration") or st.get("no_frame"):
            key = "plant_invalid" if st.get("plant_invalid") else ("integration" if st.get("integration")
                                                                   else "no_frame")
            info = st[key] if isinstance(st[key], dict) else {}
            step = int(info["step"]) if info.get("step") is not None else self.step0 + int(st["cycle"]) * SPC
            e = {"kind": key, "message": str(st[key]), "step": step}
        else:
            return None
        i = self.slot_of_step(step)
        e["t_shift_s"] = self.t_shift(step)
        e["slot_index"] = i
        e["t_after_presentation_s"] = None if i is None else (step - self.pres[i]["step"]) * DT
        e["load_over_capacity_before"] = self.load_over_capacity(until_step=step)
        return e

    def end_step(self):
        e = self.first_event()
        last = int(self.a["step"][-1]) + 1 if len(self.a["step"]) else self.step0
        return min(last, int(e["step"])) if e is not None else last

    def _end_text(self):
        e = self.first_event()
        if e is None:
            return "the run's trace ends"
        side = e.get("side")
        who = f" ({side} hand)" if side else ""
        tp = "" if e["t_after_presentation_s"] is None else \
            f", {e['t_after_presentation_s']:.3f} s after the presentation of slot {self.slots[e['slot_index']]['slot']}"
        return f"{e['kind']}{who} at {e['t_shift_s']:.3f} s of the shift{tp}"

    def shelf_exception(self):
        se = self.res.get("shelf_exception")
        if se in ("step", "latch"):
            return se
        return "step" if any("te_step" in p for p in self.res.get("presentations", [])) else "latch"

    def _capacity_arrays(self, shelf_exception=None):
        a = self.a
        n = len(a["step"])
        held = {s: a[f"g_{s}.weld_state"] > 0 for s in SIDES}
        anyh = held["left"] | held["right"]
        on_shelf = np.zeros(n, dtype=bool)
        S = a["S_tray_N"]
        mode = self.shelf_exception() if shelf_exception is None else shelf_exception
        if mode == "latch":
            latch = False
            for k in range(n):
                if not latch and anyh[k] and S[k] >= SH.Monitor.SHELF_CARRIES_N:
                    latch = True
                if latch and not anyh[k]:
                    latch = False
                on_shelf[k] = latch
        elif mode == "step":
            te = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
            for p in sorted(self.res.get("presentations", []), key=lambda x: x["step"]):
                k0 = int(np.searchsorted(a["step"], int(p["step"])))
                te[k0:] = int(p["te_step"]) if p.get("te_step") is not None else np.iinfo(np.int64).max
            on_shelf = (a["step"] >= te) & anyh & (S >= SH.Monitor.SHELF_CARRIES_N)
        else:
            raise ValueError(f"shelf_exception {mode!r}")
        out = {"mode": mode, "ev": {}, "share": {}}
        arm_n = int(round(K.CAPACITY_RULE["armed_after_capture_s"] / DT))
        mu = self.site["pad_friction"]
        for s in SIDES:
            h = held[s]
            armed = np.zeros(n, dtype=bool)
            k = 0
            while k < n:
                if h[k] and (k == 0 or not h[k - 1]):
                    j = k
                    while j < n and h[j]:
                        j += 1
                    cap_step = int(a["step"][k])
                    for m in range(k, j):
                        armed[m] = int(a["step"][m]) >= cap_step + arm_n
                    k = j
                else:
                    k += 1
            w = a[f"g_{s}.wrench"]
            Fc = a[f"g_{s}.clamp_force"]
            Fs = np.hypot(w[:, 0], w[:, 2])
            Fn = np.abs(w[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                sh = np.where(Fc > 0, np.maximum(Fs / (mu[s] * np.where(Fc > 0, Fc, 1.0)),
                                                 Fn / np.where(Fc > 0, Fc, 1.0)), np.inf)
            out["ev"][s] = armed & ~on_shelf
            out["share"][s] = sh
        return out

    def load_over_capacity(self, until_step=None, shelf_exception=None):
        a = self.a
        n = len(a["step"])
        stop = n if until_step is None else int(np.searchsorted(a["step"], until_step))
        ca = self._capacity_arrays(shelf_exception)
        out = {}
        for s in SIDES:
            ev = ca["ev"][s].copy()
            sh = ca["share"][s]
            ev[stop:] = False
            out[s] = float(np.max(sh[ev])) if ev.any() else 0.0
        out["max"] = max(out["left"], out["right"])
        return out

    def grasp_load_largest(self, from_step=None, until_step=None, shelf_exception=None):
        a = self.a
        st = a["step"]
        n = len(st)
        k0 = 0 if from_step is None else int(np.searchsorted(st, int(from_step)))
        k1 = n if until_step is None else int(np.searchsorted(st, int(until_step)))
        W = int(round(K.CAPACITY_RULE["for_s"] / DT))
        ca = self._capacity_arrays(shelf_exception)
        per = {}
        for s in SIDES:
            ev = np.zeros(n, dtype=bool)
            ev[k0:k1] = ca["ev"][s][k0:k1]
            sh = ca["share"][s]
            single = {"value": 0.0, "step": None}
            if ev.any():
                idx = np.flatnonzero(ev)
                j = int(idx[int(np.argmax(sh[idx]))])
                single = {"value": float(sh[j]), "step": int(st[j])}
            held = {"value": 0.0, "first_step": None, "last_step": None}
            k = 0
            while k < n:
                if not ev[k]:
                    k += 1
                    continue
                j = k + 1
                while j < n and ev[j] and int(st[j]) == int(st[j - 1]) + 1:
                    j += 1
                if j - k >= W:
                    seg = sh[k:j]
                    mins = np.lib.stride_tricks.sliding_window_view(seg, W).min(axis=1)
                    m = int(np.argmax(mins))
                    if float(mins[m]) > held["value"] or held["first_step"] is None:
                        held = {"value": float(mins[m]), "first_step": int(st[k + m]),
                                "last_step": int(st[k + m + W - 1])}
                k = j
            per[s] = {"held": held, "single": single}

        def best(kind, step_key):
            c = [(per[s][kind]["value"], s) for s in SIDES if per[s][kind][step_key] is not None]
            if not c:
                return dict({"value": 0.0, "side": None}, **{k: None for k in per["left"][kind] if k != "value"})
            v = max(x[0] for x in c)
            tied = [s for x, s in c if x == v]
            s = min(tied, key=lambda q: (per[q][kind][step_key], SIDES.index(q)))
            return dict(per[s][kind], side=s)
        return {"held": best("held", "first_step"), "single": best("single", "step"), "per_side": per,
                "window_steps": W, "from_step": from_step, "until_step": until_step, "shelf_exception": ca["mode"]}

    def _events_of(self, kind, i):
        p0 = self.pres[i]["step"]
        nxt = [p["step"] for j, p in self.pres.items() if p["step"] > p0]
        p1 = min(nxt) if nxt else 1 << 62
        return [e for e in self.ev if e["kind"] == kind and p0 <= e["step"] < p1]

    def capture_mismatch(self, i):
        out = {}
        for e in self._events_of("grasp_closed", i):
            f = e["fields"]
            side = f.get("arm")
            if side in out:
                continue
            m = np.asarray(f["mismatch_pos_m"], float) * 1e3
            q = np.asarray(f["mismatch_quat"], float)
            ang = math.degrees(2.0 * math.acos(min(1.0, abs(float(q[0])))))
            out[side] = {"step": int(e["step"]), "mm": m.tolist(), "norm_mm": float(np.linalg.norm(m)),
                         "rot_deg": ang}
        return out

    def internal_force_at(self, step):
        k = self.tr.index_of(step)
        if k is None:
            return None
        a = self.a
        if a["g_left.weld_state"][k] <= 0 or a["g_right.weld_state"][k] <= 0:
            return None
        Rt = _R(a["tray.pose"][k][3:7])
        Rg = SH.matrix_of_quat(SH.Cell.grip_quat())
        F = {s: Rt @ (Rg @ a[f"g_{s}.wrench"][k][:3]) for s in SIDES}
        return float(0.5 * (F["left"] - F["right"]) @ Rt[:, 1])

    def internal_force(self, i):
        caps = self.capture_mismatch(i)
        out = {"after_capture_step": None, "after_capture_N": None, "load_ramp_end_step": None, "load_ramp_end_N": None}
        if len(caps) == 2:
            st = max(c["step"] for c in caps.values()) + int(round(INTERNAL_AFTER_CAPTURE_S / DT))
            out["after_capture_step"] = st
            out["after_capture_N"] = self.internal_force_at(st)
        st = self.pres[i]["step"] + cy(LOAD_RAMP_END_S) * SPC
        out["load_ramp_end_step"] = st
        out["load_ramp_end_N"] = self.internal_force_at(st)
        return out

    def landing(self, i):
        a = self.a
        p0 = self.pres[i]["step"]
        k0 = int(np.searchsorted(a["step"], p0))
        S = a["S_tray_N"]
        idx = np.nonzero(S[k0:] > 0.0)[0]
        if len(idx) == 0:
            return None
        k = k0 + int(idx[0])
        if k == 0:
            return None
        return {"first_contact_step": int(a["step"][k]), "v_c_m_s": float(-a["tray.twist"][k - 1][2]),
                "step_read": int(a["step"][k - 1])}

    def te_step(self, i):
        if i not in self.pres:
            return None
        p = self.pres[i]
        if p.get("te_step") is not None:
            return int(p["te_step"])
        if p.get("missed_capture") or p.get("decline_cycle") is not None:
            return None
        if self.slots[i]["label"] == "carry" or p.get("carried"):
            return int(p["step"]) + int(round(K.carry_end_s(len(self.jobs[i]["targets"])) / DT))
        return None

    def release_in_place_window(self, i):
        te = self.te_step(i)
        if te is None:
            return None
        hi = te + int(round(K.SLOT["set_down_s"] / DT))
        rel = [e for e in self._events_of("grasp_released", i) if te <= int(e["step"]) < hi]
        return min(int(e["step"]) for e in rel) if rel else None

    def press(self, i):
        s_rel = self.release_in_place_window(i)
        if s_rel is None:
            return None
        c = (s_rel - self.step0) // SPC
        steps = [self.step0 + c * SPC + j for j in range(SPC)]
        ks = [self.tr.index_of(s) for s in steps]
        if any(k is None for k in ks):
            return None
        mg = float(self.slots[i]["mass_kg"]) * G_ABS
        Sv = [float(self.a["S_tray_N"][k]) for k in ks]
        return {"release_step": int(s_rel), "cycle": int(c), "P_r_N": abs(sum(Sv) / len(Sv) - mg),
                "mean_abs_N": sum(abs(x - mg) for x in Sv) / len(Sv), "S_N": Sv, "weight_N": mg}

    def hold_windows(self, i):
        if i not in self.pres:
            return []
        p = self.pres[i]["step"]
        tg = [tuple(float(x) for x in t) for t in self.jobs[i]["targets"]]
        L = self.lifted_point()
        out = []
        for k, t in enumerate(tg):
            c_end = cy(K.CARRY_START_S) + (k + 1) * cy(K.TARGET_S)
            c_lo = c_end - cy(K.HOLD_WINDOW_S)
            out.append((p + c_lo * SPC, p + c_end * SPC, k, t, tg[k - 1] if k else L))
        return out

    def holds_of(self, i):
        return len(self.jobs[i]["targets"])

    def lifted_point(self):
        tF = SH.Cell.tray_in(SH.pose_of(self.site["F"]["true"]))
        return (tF[0][0], tF[0][1], tF[0][2] + K.LIFTED_POINT_M)

    def hold_limit(self, i):
        lim = self.end_step()
        if i not in self.pres:
            return lim
        p = self.pres[i]
        nxt = [int(q["step"]) for q in self.pres.values() if int(q["step"]) > int(p["step"])]
        if nxt:
            lim = min(lim, min(nxt))
        rem = p.get("removal")
        if rem is not None and (p.get("missed_capture") or p.get("decline_cycle") is not None):
            lim = min(lim, self.step0 + int(rem["cycle"]) * SPC)
        return lim

    def hold_rows(self, i, until_step=None):
        a = self.a
        rows = []
        end = self.hold_limit(i) if until_step is None else until_step
        for lo, hi, k, t, fr in self.hold_windows(i):
            if hi > end:
                break
            k0, k1 = int(np.searchsorted(a["step"], lo)), int(np.searchsorted(a["step"], hi))
            if k1 - k0 != hi - lo:
                break
            P = a["tray.pose"][k0:k1]
            m = P[:, :3].mean(axis=0)
            tilt = sum(tilt_deg(q) for q in P[:, 3:7]) / (k1 - k0)
            rows.append({"slot": self.slots[i]["slot"], "k": k, "target": t, "from": fr, "pos": m.tolist(),
                         "e_z_mm": 1e3 * abs(m[2] - t[2]), "e_xy_mm": 1e3 * math.hypot(m[0] - t[0], m[1] - t[1]),
                         "tilt_deg": tilt})
        return rows

    def repeatability_rows(self, i, rows=None):
        rows = self.hold_rows(i) if rows is None else rows
        by = {}
        for r in rows:
            by.setdefault(r["target"], []).append(r)
        out = []
        for t, rs in by.items():
            if len(rs) < 2:
                continue
            dirs = [np.array(t) - np.array(r["from"]) for r in rs]
            if any(SH._angle_deg(d1, d2) >= K.REPEAT_DIRECTION_MIN_DEG for d1 in dirs for d2 in dirs):
                d = max(float(np.linalg.norm(np.array(r1["pos"]) - np.array(r2["pos"]))) for r1 in rs for r2 in rs)
                out.append({"slot": self.slots[i]["slot"], "target": t, "rho_mm": 1e3 * d})
        return out

    def path_rows(self, i, until_step=None):
        if i not in self.pres:
            return []
        a = self.a
        p = self.pres[i]["step"]
        tg = [tuple(float(x) for x in t) for t in self.jobs[i]["targets"]]
        L = self.lifted_point()
        prof = SH.trapezoid(K.MOVE_S)
        n = cy(K.MOVE_S) * SPC
        end = self.hold_limit(i) if until_step is None else until_step
        rows = []
        for k, t in enumerate(tg):
            s0 = p + (cy(K.CARRY_START_S) + k * cy(K.TARGET_S)) * SPC
            if s0 + n > end:
                break
            k0 = int(np.searchsorted(a["step"], s0))
            if k0 + n > len(a["step"]) or int(a["step"][k0 + n - 1]) != s0 + n - 1:
                break
            p0 = np.array(tg[k - 1] if k else L)
            p1 = np.array(t)
            X = a["tray.pose"][k0:k0 + n, :3]
            ref = np.array([p0 + prof((j + 1) * DT)[0] * (p1 - p0) for j in range(n)])
            d = math.sqrt(float(np.mean(np.sum((X - ref) ** 2, axis=1))))
            dp = p1 - p0
            prog = float(np.clip((X[-1] - p0) @ dp / float(dp @ dp), 0.0, 1.0))
            rows.append({"slot": self.slots[i]["slot"], "k": k, "d_m_mm": 1e3 * d, "progress": prog})
        return rows

    def holds_quantity(self, slot_indices, key, reduce="median"):
        rows, N = [], 0
        for i in slot_indices:
            N += self.holds_of(i)
            rows += self.hold_rows(i)
        vals = [r[key] for r in rows]
        part = stats(vals)
        q = Quantity(read=len(rows), of=N, partial=part)
        if len(rows) == N and self._no_event_before(self._last_hold_end(slot_indices)):
            q.update(exists=True, value=part[reduce] if part else None, text=None)
        else:
            q.update(exists=False, value=None,
                     text=f"does not exist: ended by {self._end_text()}; holds read {len(rows)} of {N}")
        return q

    def repeatability_quantity(self, slot_indices, F=None):
        rows, holds, N = [], 0, 0
        for i in slot_indices:
            N += self.holds_of(i)
            hr = self.hold_rows(i)
            holds += len(hr)
            rows += self.repeatability_rows(i, hr)
        vals = [r["rho_mm"] for r in rows]
        part = {"rows": len(vals), "stats": stats(vals), "R": None if F is None else term_value(vals, F)}
        q = Quantity(read=holds, of=N, partial=part)
        if holds == N and self._no_event_before(self._last_hold_end(slot_indices)):
            q.update(exists=True, value=part["R"] if F is not None else part["stats"], text=None)
        else:
            q.update(exists=False, value=None,
                     text=f"does not exist: ended by {self._end_text()}; holds read {holds} of {N}")
        return q

    def path_quantity(self, slot_indices, F=None):
        rows, N = [], 0
        for i in slot_indices:
            N += len(self.jobs[i]["targets"])
            rows += self.path_rows(i)
        vals = [r["d_m_mm"] for r in rows]
        part = {"rows": len(vals), "stats": stats(vals), "term": None if F is None else term_value(vals, F),
                "progress_min": min((r["progress"] for r in rows), default=None)}
        q = Quantity(read=len(rows), of=N, partial=part)
        if len(rows) == N and self._no_event_before(self._last_hold_end(slot_indices)):
            q.update(exists=True, value=part["term"] if F is not None else part["stats"], text=None)
        else:
            q.update(exists=False, value=None,
                     text=f"does not exist: ended by {self._end_text()}; moves read {len(rows)} of {N}")
        return q

    def setdown_quantity(self, i, which):
        N = self.holds_of(i)
        holds = len(self.hold_rows(i))
        if i not in self.pres:
            q = Quantity(read=holds, of=N, partial={"value": None, "detail": None})
            q.update(exists=False, value=None,
                     text=f"does not exist: ended by {self._end_text()}; holds read {holds} of {N}")
            return q
        f = self.landing(i) if which == "v_c" else self.press(i)
        rel = self._events_of("grasp_released", i)
        s_rel = min(e["step"] for e in rel) if rel else None
        ok = f is not None and s_rel is not None and self._no_event_before(s_rel + 1)
        val = None if f is None else (f["v_c_m_s"] if which == "v_c" else f["P_r_N"])
        q = Quantity(read=holds, of=N, partial={"value": val, "detail": f})
        if ok:
            q.update(exists=True, value=val, text=None)
        else:
            q.update(exists=False, value=None,
                     text=f"does not exist: ended by {self._end_text()}; holds read {holds} of {N}")
        return q

    def _last_hold_end(self, slot_indices):
        if any(i not in self.pres for i in slot_indices):
            return 1 << 62
        return max(self.hold_windows(i)[-1][1] for i in slot_indices)

    def stillness_rows(self, i, until_step=None):
        a = self.a
        rows = []
        end = self.hold_limit(i) if until_step is None else until_step
        for lo, hi, k, t, fr in self.hold_windows(i):
            if hi > end:
                break
            k0, k1 = int(np.searchsorted(a["step"], lo)), int(np.searchsorted(a["step"], hi))
            if k1 - k0 != hi - lo:
                break
            rows.append({"slot": self.slots[i]["slot"], "k": k, "stillness_mm": diameter_mm(a["tray.pose"][k0:k1, :3])})
        return rows

    def stillness_quantity(self, slot_indices, F=None):
        rows, N = [], 0
        for i in slot_indices:
            N += self.holds_of(i)
            rows += self.stillness_rows(i)
        vals = [r["stillness_mm"] for r in rows]
        part = {"rows": len(vals), "stats": stats(vals), "term": None if F is None else term_value(vals, F)}
        q = Quantity(read=len(rows), of=N, partial=part)
        if len(rows) == N and self._no_event_before(self._last_hold_end(slot_indices)):
            q.update(exists=True, value=part["term"] if F is not None else part["stats"], text=None)
        else:
            q.update(exists=False, value=None,
                     text=f"does not exist: ended by {self._end_text()}; holds read {len(rows)} of {N}")
        return q

    def _no_event_before(self, step):
        e = self.first_event()
        return e is None or int(e["step"]) >= int(step)

    def carry_start_position(self, i):
        if i not in self.pres:
            return None
        k = self.tr.index_of(self.pres[i]["step"] + cy(K.CARRY_START_S) * SPC - 1)
        return None if k is None else self.a["tray.pose"][k][:3].copy()

    def visited_rows(self, i):
        hr = self.hold_rows(i)
        p_start = self.carry_start_position(i)
        if not hr or p_start is None:
            return []
        tg = [r["target"] for r in hr]
        pos = [r["pos"] for r in hr]
        rows = visited_rows(tg, pos, p_start, self.lifted_point())
        for r, h in zip(rows, hr):
            r["slot"] = h["slot"]
        return rows

    def visited(self, i):
        rows = self.visited_rows(i)
        n = self.holds_of(i)
        return {"V": (sum(1 for r in rows if r["visited"]) / n) if n else 0.0, "visited": sum(1 for r in rows if
                                                                                        r["visited"]),
                "read": len(rows), "of": n, "rows": rows}

    def place_row(self, i):
        pr = self.press(i)
        if pr is None:
            return None
        s_mean = sum(pr["S_N"]) / len(pr["S_N"])
        return {"slot": self.slots[i]["slot"], "P_r_N": pr["P_r_N"], "S_release_cycle_N": s_mean,
                "in_air": s_mean < SH.Monitor.SHELF_CARRIES_N, "release_step": pr["release_step"]}

    def decline_row(self, i):
        sl = self.slots[i]
        row = {"slot": sl["slot"], "presented": i in self.pres, "decline_cycle": None, "captured_before": {},
               "put_back": None, "declined": False}
        if i not in self.pres:
            return row
        p = self.pres[i]
        if p.get("missed_capture"):
            row["missed_capture"] = True
            return row
        dc = p.get("decline_cycle")
        row["decline_cycle"] = dc
        if dc is None:
            return row
        dstep = self.step0 + int(dc) * SPC
        caps = {}
        for e in self._events_of("grasp_closed", i):
            arm = e["fields"].get("arm")
            if arm in SIDES and int(e["step"]) < dstep and arm not in caps:
                caps[arm] = int(e["step"])
        row["captured_before"] = caps
        step_pb = self.step0 + (int(dc) + cy(K.SLOT["decline_put_back_s"])) * SPC - 1
        k = self.tr.index_of(step_pb)
        if k is None:
            row["put_back"] = {"step": step_pb, "in_trace": False}
            return row
        a = self.a
        released = all(float(a[f"g_{s}.weld_state"][k]) <= 0 for s in SIDES)
        fc = feet_in_cups(a["tray.pose"][k], SH.pose_of(self.site["F"]["true"]))
        row["put_back"] = {"step": step_pb, "in_trace": True, "released": released, **fc}
        decided = self.end_step() > step_pb
        row["put_back"]["decided_before_the_end"] = decided
        row["declined"] = bool(len(caps) == 2 and released and fc["feet_in_cups"] and decided)
        return row

    def seating_displacement(self, i):
        if i not in self.pres:
            return None
        seat = {}
        for e in self._events_of("grasp_seated", i):
            arm = e["fields"].get("arm")
            if arm in SIDES and arm not in seat:
                seat[arm] = int(e["step"])
        if len(seat) != 2:
            return None
        k0, k1 = self.tr.index_of(int(self.pres[i]["step"])), self.tr.index_of(max(seat.values()))
        if k0 is None or k1 is None:
            return None
        P_0, P1 = self.a["tray.pose"][k0], self.a["tray.pose"][k1]
        d = 1e3 * (P1[:3] - P_0[:3])
        q = SH.W.qmul(SH.W.qconj(tuple(float(x) for x in P_0[3:7])), tuple(float(x) for x in P1[3:7]))
        return {"slot": self.slots[i]["slot"], "from_step": int(self.pres[i]["step"]), "to_step": max(seat.values()),
                "seated_steps": seat, "d_mm": d.tolist(), "norm_mm": float(np.linalg.norm(d)),
                "yaw_deg": math.degrees(2.0 * math.atan2(q[3], q[0]))}

    def completed_row(self, i):
        row = {"slot": self.slots[i]["slot"], "completed": False, "presented": i in self.pres}
        te = self.te_step(i)
        if te is None:
            row["reason"] = "not presented" if i not in self.pres else "no set-down window (declined or missed)"
            return row
        caps = {e["fields"].get("arm") for e in self._events_of("grasp_closed", i) if int(e["step"]) < te}
        row["captured"] = sorted(c for c in caps if c in SIDES)
        st_ = te + int(round(K.SLOT["set_down_s"] / DT)) - 1
        row["decided_at_step"] = st_
        k = self.tr.index_of(st_)
        if not self._no_event_before(st_ + 1) or k is None:
            row["reason"] = "the shift ended before T_e + 6.0 s"
            return row
        a = self.a
        opened = all(float(a[f"g_{s}.weld_state"][k]) <= 0 for s in SIDES)
        ok_s, feet = SH.Cell.stands_on_S((tuple(float(x) for x in a["tray.pose"][k][:3]),
                                          tuple(float(x) for x in a["tray.pose"][k][3:7])),
                                         SH.pose_of(self.site["S"]["true"]),
                                         grow=K.S_SEAT_FRAME_GROW_M, tol_z=S_FOOT_ON_TOP_TOL_M)
        row.update(both_open=opened, stands_on_S=bool(ok_s), feet=feet)
        row["completed"] = bool(len(row["captured"]) == 2 and opened and ok_s)
        return row

    def lap_row(self, i):
        rows = self.visited_rows(i)
        first = bool(rows and rows[0]["k"] == 0 and rows[0]["visited"])
        return {"slot": self.slots[i]["slot"], "first_hold_visited": first, "counts_against": not first}

    def J(self):
        dec = [self.decline_row(i) for i, sl in enumerate(self.slots) if sl["label"] == "decline"]
        lap = [self.lap_row(i) for i, sl in enumerate(self.slots) if sl["label"] == "carry" and sl["job"] == "lap"]
        return j_from_rows(dec, lap)

    def quantity_rows(self, i):
        hr = self.hold_rows(i)
        out = {k: [r[k] for r in hr] for k in ("e_z_mm", "tilt_deg", "e_xy_mm")}
        out["hold_rows"] = hr
        out["rho_mm"] = [r["rho_mm"] for r in self.repeatability_rows(i, hr)]
        pr = self.path_rows(i)
        out["d_m_mm"] = [r["d_m_mm"] for r in pr]
        out["path_rows"] = pr
        pl = self.place_row(i)
        out["P_r_N"] = [] if pl is None else [pl["P_r_N"]]
        out["place_row"] = pl
        return out

    def chain_rows(self):
        out = {"slots": [], "J": None, "tray_in_work": self.res.get("tray_in_work"), "first_event": self.first_event()}
        for i, sl in enumerate(self.slots):
            r = {"slot": sl["slot"], "label": sl["label"], "job": sl["job"], "presented": i in self.pres}
            if sl["label"] == "carry" or (i in self.pres and self.pres[i].get("carried")):
                v = self.visited(i)
                r.update(completed=self.completed_row(i), V=v["V"], visited=v["visited"], holds=v["of"],
                         place=self.place_row(i))
            if sl["label"] == "decline":
                r["decline"] = self.decline_row(i)
            r["seating"] = self.seating_displacement(i)
            out["slots"].append(r)
        j = self.J()
        out["J"] = {k: j[k] for k in ("TPR", "FPR", "J", "n_decline", "n_lap_carry", "note")}
        return out

    def summary(self):
        out = {"first_event": self.first_event(), "load_over_capacity": self.load_over_capacity(), "slots": []}
        for i, sl in enumerate(self.slots):
            if i not in self.pres or (sl["label"] == "decline" and not self.pres[i].get("carried")):
                continue
            hr = self.hold_rows(i)
            out["slots"].append({
                "slot": sl["slot"], "capture_mismatch": self.capture_mismatch(i),
                "internal_force": self.internal_force(i), "landing": self.landing(i), "press": self.press(i),
                "holds": {k: stats([r[k] for r in hr]) for k in ("e_z_mm", "e_xy_mm", "tilt_deg")},
                "holds_read": len(hr), "holds_of": len(self.hold_windows(i)),
                "repeatability": stats([r["rho_mm"] for r in self.repeatability_rows(i, hr)]),
                "path": stats([r["d_m_mm"] for r in self.path_rows(i)])})
        return out


def visited_rows(targets, positions, p_start, L, tol=VISIT_TOL_M):
    rows = []
    pp, tp = np.asarray(p_start, float), np.asarray(L, float)
    for k, (t, p) in enumerate(zip(targets, positions)):
        t, p = np.asarray(t, float), np.asarray(p, float)
        d = float(np.linalg.norm(p - t))
        e = float(np.linalg.norm((p - pp) - (t - tp)))
        rows.append({"k": k, "target": tuple(float(x) for x in t), "pos": [float(x) for x in p], "dist_m": d,
                     "disp_err_m": e, "visited": bool(d <= tol and e <= tol)})
        pp, tp = p, t
    return rows


def _box_grid(step=BOX_GRID_M):
    def ax(lo, hi):
        n = int(round((hi - lo) / step))
        return np.array([lo + step * j for j in range(n + 1)])
    return ax(*K.BOX["x"]), ax(*K.BOX["y"]), ax(*K.BOX["z"])


def v_strategies(targets, L, tol=VISIT_TOL_M):
    T = np.asarray(targets, float)
    n = len(T)
    L = np.asarray(L, float)
    xs, ys, zs = _box_grid()
    out = {}
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    cnt = (np.linalg.norm(P[:, None, :] - T[None, :, :], axis=2) <= tol).sum(axis=1)
    j = int(np.argmax(cnt))
    Pb = P[j]
    rows = visited_rows(T, [Pb] * n, L, L, tol)
    out["parked"] = {"point": Pb.tolist(), "count_by_position": int(cnt[j]), "rows": rows}
    C = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    cc = (np.linalg.norm(C[:, None, :] - T[None, :, :2], axis=2) <= tol).sum(axis=1)
    j = int(np.argmax(cc))
    X, Y = C[j]
    rows = visited_rows(T, [np.array([X, Y, t[2]]) for t in T], L, L, tol)
    out["heights"] = {"point": [float(X), float(Y)], "count_by_position": int(cc[j]), "rows": rows}
    rows = visited_rows(T, [T[-1]] * n, L, L, tol)
    out["straight"] = {"point": T[-1].tolist(), "rows": rows}
    rows = visited_rows(T, [L] * (n - 2) + [np.array([L[0], L[1], T[-1][2]]), T[-1]], L, L, tol)
    out["skip"] = {"rows": rows}
    for v in out.values():
        v["visited"] = sum(1 for r in v["rows"] if r["visited"])
        v["n"] = n
        v["V"] = v["visited"] / n if n else 0.0
    return out


def feet_in_cups(tray_pose, F_pose):
    C = SH.Cell
    if hasattr(C, "feet_in_cups"):
        ok, rows = C.feet_in_cups((tuple(float(x) for x in tray_pose[:3]), tuple(float(x) for x in tray_pose[3:7])),
                                  SH.pose_of(F_pose))
        return {"feet_in_cups": bool(ok), "feet": rows, "definition": "shift.Cell.feet_in_cups"}
    Rt = SH.matrix_of_quat(np.asarray(tray_pose[3:7], float))
    pt = np.asarray(tray_pose[:3], float)
    RF = SH.matrix_of_quat(np.asarray(F_pose[1], float))
    pF = np.asarray(F_pose[0], float)
    fx, fy, _ = C.FOOT_HALF
    half = (fx + C.CUP_CLEAR, fy + C.CUP_CLEAR)
    zb = -C.GRIP_ABOVE_FEET
    feet, ok = [], True
    for (x, y) in C.FOOT_XY:
        hz, ov = -1e9, -1e9
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                cw = pt + Rt @ np.array([x + sx * fx, y + sy * fy, zb])
                cf = RF.T @ (cw - pF)
                hz = max(hz, abs(float(cf[2])))
                ov = max(ov, abs(float(cf[0] - x)) - half[0], abs(float(cf[1] - y)) - half[1])
        f_ok = hz <= FOOT_ON_PLATE_TOL_M and ov <= CUP_INSIDE_TOL_M
        ok = ok and f_ok
        feet.append({"on_plate_mm": 1e3 * hz, "inside_mm": 1e3 * ov, "ok": bool(f_ok)})
    return {"feet_in_cups": bool(ok), "feet": feet, "definition": "quantities"}


def j_from_rows(decline_rows, lap_rows):
    nd = len(decline_rows)
    tpr = (sum(1 for r in decline_rows if r["declined"]) / nd) if nd else 0.0
    nl = len(lap_rows)
    fpr = (sum(1 for r in lap_rows if r["counts_against"]) / nl) if nl else 0.0
    return {"TPR": tpr, "FPR": fpr, "J": max(0.0, tpr - fpr), "decline_rows": decline_rows, "lap_rows": lap_rows,
            "n_decline": nd, "n_lap_carry": nl, "note": None if nl else "no carry tray on a lap job: FPR = 0"}


def j_enumeration_hold_still(slots, jobs, L):
    import itertools
    lap = [i for i, sl in enumerate(slots) if sl["job"] == "lap"]
    seated = np.asarray(L, float) - np.array([0.0, 0.0, K.LIFTED_POINT_M])
    out = []
    for r in range(len(lap) + 1):
        for sub in itertools.combinations(lap, r):
            dec_rows, lap_rows = [], []
            for i in lap:
                sl = slots[i]
                if sl["label"] == "decline":
                    caps = {}
                    declined = bool(i in sub and len(caps) == 2)
                    dec_rows.append({"slot": sl["slot"], "declined": declined, "decline_sent": i in sub})
                else:
                    t0 = jobs[i]["targets"][0]
                    vr = visited_rows([t0], [seated], seated, L)
                    first = vr[0]["visited"]
                    lap_rows.append({"slot": sl["slot"], "first_hold_visited": first, "counts_against": not first,
                                     "decline_sent": i in sub})
            dec_rows += [{"slot": sl["slot"], "declined": False, "decline_sent": False}
                         for j, sl in enumerate(slots) if sl["label"] == "decline" and j not in lap]
            jj = j_from_rows(dec_rows, lap_rows)
            out.append({"declined_slots": [slots[i]["slot"] for i in sub], "J": jj["J"], "TPR": jj["TPR"],
                        "FPR": jj["FPR"]})
    return out


def L_credit(x, F, Z):
    if x <= F:
        return 1.0
    if x >= Z:
        return 0.0
    return (Z - x) / (Z - F)


def zero_end(mutant_rows_by_unit, reference_rows_by_unit):
    pooled = [float(x) for u in sorted(mutant_rows_by_unit) for x in mutant_rows_by_unit[u]]
    n_ref = sum(len(v) for v in reference_rows_by_unit.values())
    per = {}
    k = 0
    for u in sorted(reference_rows_by_unit):
        m = len(mutant_rows_by_unit.get(u, []))
        r = len(reference_rows_by_unit[u])
        tenth = r > 0 and m >= 0.1 * r
        per[u] = {"rows": m, "reference_rows": r, "a_tenth": bool(tenth)}
        k += int(tenth)
    ok_half = len(pooled) >= 0.5 * n_ref and n_ref > 0
    ok_units = k >= 2
    out = {"rows": len(pooled), "reference_rows": n_ref, "units_with_a_tenth": k, "per_unit": per}
    if ok_half and ok_units:
        v = sorted(pooled)
        out.update(exists=True, Z=v[len(v) // 2], reason=None)
    else:
        why = []
        if not ok_half:
            why.append(f"the mutant's rows number {len(pooled)} of the reference's {n_ref} (at least half needed)")
        if not ok_units:
            why.append(f"{k} unit(s) give at least one tenth of their reference rows (two needed)")
        out.update(exists=False, Z=None, reason="does not exist: " + "; ".join(why))
    return out


def decide(floor_table, reference_rows_by_unit, mutant_rows_by_unit):
    qs = {}
    for qn in list(TERM_QUANTITIES["H"]) + ["rho_mm", "e_xy_mm", "d_m_mm", PLACE_QUANTITY]:
        if qn not in floor_table:
            raise KeyError(f"the floor table has no floor for {qn}")
        ref = reference_rows_by_unit.get(qn, {})
        pooled = [float(x) for u in sorted(ref) for x in ref[u]]
        F = provisional_full_end(pooled, floor_table[qn])
        z = zero_end(mutant_rows_by_unit.get(qn, {}), ref)
        passes = bool(z["exists"] and z["Z"] >= 3.0 * F)
        qs[qn] = {"F": F, "floor": float(floor_table[qn]), "reference_p90": _p90(pooled), "zero_end": z,
                  "passes": passes}
    terms = {t: [q for q in TERM_QUANTITIES[t] if qs[q]["passes"]] for t in TERM_QUANTITIES}
    kept = {t: Q_WEIGHTS[t] for t in TERM_QUANTITIES if terms[t]}
    tot = sum(kept.values())
    weights = {t: w / tot for t, w in kept.items()} if tot > 0 else {}
    return {"quantities": qs, "terms": terms, "weights": weights, "place_passes": qs[PLACE_QUANTITY]["passes"]}


def q_value(rows, decision):
    if not decision["weights"]:
        return 1.0
    q = decision["quantities"]

    def L(qn, x):
        return L_credit(x, q[qn]["F"], q[qn]["zero_end"]["Z"])
    val = 0.0
    for t, w in decision["weights"].items():
        kept = decision["terms"][t]
        if t == "H":
            hs = [math.prod(L(qn, h[qn]) for qn in kept) for h in rows["hold_rows"]]
        elif t == "P":
            hs = [L("d_m_mm", r["d_m_mm"]) * r["progress"] for r in rows["path_rows"]]
        else:
            hs = [L(kept[0], x) for x in rows[kept[0]]]
        val += w * (sum(hs) / len(hs) if hs else 0.0)
    return val


def place_value(place_row, decision):
    if not decision["place_passes"]:
        return 1.0
    if place_row is None:
        return 0.5
    q = decision["quantities"][PLACE_QUANTITY]
    lv = 0.0 if place_row["in_air"] else L_credit(place_row["P_r_N"], q["F"], q["zero_end"]["Z"])
    return 0.5 + 0.5 * lv


def tray_value(completed, V, Q, place):
    return float(bool(completed)) * float(V) * float(Q) * float(place)
