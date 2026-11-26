from typing import NamedTuple

import numpy as np

from .state import public_params

N_MODULES = 16
FORM_KEYS = ("copper_alpha_per_K", "alpha_reference_C", "fault_hysteresis_K")


class ThermalError(ValueError):
    ""


def _timestep():
    return float(public_params()["numerics"]["option"]["value"]["timestep"])


DT = _timestep()


class ThermalForm(NamedTuple):
    alpha: float
    ref: float
    hysteresis: float


def _form_of_node(node):
    if isinstance(node, dict) and all(k in node for k in FORM_KEYS):
        v = [node[k]["value"] if isinstance(node[k], dict) else node[k] for k in FORM_KEYS]
        return ThermalForm(*(float(x) for x in v))
    return None


def thermal_form(source=None):
    if isinstance(source, ThermalForm):
        return source
    if isinstance(source, dict):
        f = _form_of_node(source.get("thermal_form")) or _form_of_node(source)
        if f is None and isinstance(source.get("thermal_constants"), dict):
            f = _form_of_node(source["thermal_constants"])
        if f is not None:
            return f
    f = _form_of_node(public_params().get("thermal_form"))
    if f is None:
        raise ThermalError("the thermal law's form (copper coefficient, reference, hysteresis) is in no given source")
    return f


def _constants_node(thermal_constants):
    tc = thermal_constants
    if isinstance(tc, dict) and "thermal_constants" in tc and "a_K_per_s_per_Nm2" not in tc:
        return tc["thermal_constants"]
    return tc


def edition_constants(thermal_constants, edition):
    tc = _constants_node(thermal_constants)
    try:
        a = float(tc["a_K_per_s_per_Nm2"][edition]["value"])
        k1, k2, k3 = (float(v) for v in tc["k1_k2_k3_per_s"][edition]["value"])
    except (KeyError, TypeError) as exc:
        raise ThermalError(f"no thermal constants for edition {edition!r}") from exc
    if not (a > 0.0 and k1 > 0.0 and k2 > 0.0 and k3 > 0.0):
        raise ThermalError("thermal constants are positive")
    return a, k1, k2, k3


class ThermalConstants:

    def __init__(self, a, h, k1, k2, k3, form=None):
        arr = lambda v: np.array([float(x) for x in v], dtype=np.float64)
        self.a, self.h, self.k1, self.k2, self.k3 = arr(a), arr(h), arr(k1), arr(k2), arr(k3)
        n = self.a.shape[0]
        if any(x.shape[0] != n for x in (self.h, self.k1, self.k2, self.k3)):
            raise ThermalError("one value per module in every array")
        self.ah = self.a * self.h
        self.form = thermal_form(form)
        self.alpha, self.ref, self.hysteresis = self.form

    @property
    def n(self):
        return self.a.shape[0]


def constants_for_modules(thermal_constants, edition_per_module, h=None, form=None):
    rows = [edition_constants(thermal_constants, e) for e in edition_per_module]
    hh = [1.0] * len(rows) if h is None else list(h)
    if len(hh) != len(rows):
        raise ThermalError("one heat factor per module")
    src = form if form is not None else thermal_constants
    return ThermalConstants([r[0] for r in rows], hh, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                            form=thermal_form(src))


def step(t_w, t_h, t_del, consts, t_amb, dt=DT):
    t_w = np.asarray(t_w, dtype=np.float64)
    t_h = np.asarray(t_h, dtype=np.float64)
    t_del = np.asarray(t_del, dtype=np.float64)
    heat = consts.ah * (t_del * t_del) * (1.0 + consts.alpha * (t_w - consts.ref))
    gap = t_w - t_h
    dw = heat - consts.k1 * gap
    dh = consts.k2 * gap - consts.k3 * (t_h - t_amb)
    return t_w + dt * dw, t_h + dt * dh


def coil_fault(t_w, ot_value):
    return np.asarray(t_w, dtype=np.float64) >= np.asarray(ot_value, dtype=np.float64)


def coil_latch(latched, t_w, ot_value):
    latched = np.asarray(latched, dtype=bool)
    hit = coil_fault(t_w, ot_value)
    return latched | hit, hit & ~latched


def coil_clear_allowed(t_w, ot_value, hysteresis):
    return np.asarray(t_w, dtype=np.float64) < (np.asarray(ot_value, dtype=np.float64) - float(hysteresis))


def reported_temperature(t_h):
    return np.floor(np.asarray(t_h, dtype=np.float64))


def _steps_of(seconds, dt, what):
    n = int(round(float(seconds) / dt))
    if n < 0 or abs(n * dt - float(seconds)) > 1e-9:
        raise ThermalError(f"{what} must be a whole, non-negative number of {dt} s steps, not {seconds!r}")
    return n


def hold_observations(thermal_constants, cases, duration_s, cooldown_s, form=None, dt=DT, trace_every=1):
    cases = list(cases)
    n = len(cases)
    if n == 0:
        raise ThermalError("at least one case")
    n_hold = _steps_of(duration_s, dt, "duration_s")
    n_cool = _steps_of(cooldown_s, dt, "cooldown_s")
    every = int(trace_every)
    if every < 1:
        raise ThermalError("trace_every is a positive number of steps")
    src = form if form is not None else thermal_constants
    consts = constants_for_modules(thermal_constants, [c["edition"] for c in cases], [float(c["h"]) for c in cases],
                                   form=src)
    vec = lambda k: np.array([float(c[k]) for c in cases], dtype=np.float64)
    t_amb, tw, th, ot = vec("T_amb"), vec("T_w0"), vec("T_h0"), vec("ot_value")
    const_t = np.zeros(n, dtype=np.float64)
    series = np.zeros((n_hold, n), dtype=np.float64) if any(np.ndim(c["drive_torque"]) for c in cases) else None
    for k, c in enumerate(cases):
        d = c["drive_torque"]
        if np.ndim(d) == 0:
            if series is None:
                const_t[k] = float(d)
            else:
                series[:, k] = float(d)
        else:
            d = np.asarray(d, dtype=np.float64)
            if d.shape != (n_hold,):
                raise ThermalError(f"case {k}: {d.shape[0] if d.ndim == 1 else d.shape} torque values for {n_hold} "
                                   "steps of the hold")
            series[:, k] = d
    for arr, what in ((const_t, "drive_torque"), (t_amb, "T_amb"), (tw, "T_w0"), (th, "T_h0"), (ot, "ot_value")):
        if not np.all(np.isfinite(arr)):
            raise ThermalError(f"{what} must be finite")
    if series is not None and not np.all(np.isfinite(series)):
        raise ThermalError("drive_torque must be finite")

    total = n_hold + n_cool
    latched = np.zeros(n, dtype=bool)
    trip = np.full(n, -1, dtype=np.int64)
    rep_trip = np.full(n, np.nan)
    tw_trip = np.full(n, np.nan)
    th_trip = np.full(n, np.nan)
    clear = np.full(n, -1, dtype=np.int64)
    zero = np.zeros(n, dtype=np.float64)
    t_steps, w_tr, h_tr = [], [], []
    for i in range(total + 1):
        if i % every == 0 or i == total:
            t_steps.append(i)
            w_tr.append(tw.copy())
            h_tr.append(th.copy())
        latched, new = coil_latch(latched, tw, ot)
        if new.any():
            trip[new] = i
            rep_trip[new] = reported_temperature(th)[new]
            tw_trip[new] = tw[new]
            th_trip[new] = th[new]
        ok = latched & (clear < 0) & coil_clear_allowed(tw, ot, consts.hysteresis)
        clear[ok] = i
        if i == total:
            break
        if i < n_hold:
            td = const_t if series is None else series[i]
        else:
            td = zero
        td = np.where(latched, 0.0, td)
        tw, th = step(tw, th, td, consts, t_amb, dt)
    h_arr = np.array(h_tr)
    return {"trip_step": trip, "reported_at_trip": rep_trip, "winding_at_trip": tw_trip, "housing_at_trip": th_trip,
            "clear_allowed_step": clear, "trace_steps": np.array(t_steps, dtype=np.int64),
            "winding_trace": np.array(w_tr), "housing_trace": h_arr, "reported_trace": reported_temperature(h_arr),
            "n_hold_steps": n_hold, "n_cool_steps": n_cool, "dt": dt}


def hold_observation(thermal_constants, edition, h, T_amb, T_w0, T_h0, drive_torque, ot_value, duration_s, cooldown_s,
                     form=None, dt=DT, trace_every=1):
    r = hold_observations(thermal_constants, [dict(edition=edition, h=h, T_amb=T_amb, T_w0=T_w0, T_h0=T_h0,
                                                   drive_torque=drive_torque, ot_value=ot_value)],
                          duration_s, cooldown_s, form=form, dt=dt, trace_every=trace_every)
    tr = int(r["trip_step"][0])
    cl = int(r["clear_allowed_step"][0])
    rep = None if tr < 0 else int(r["reported_at_trip"][0])
    return {"trip_step": None if tr < 0 else tr, "trip_time_s": None if tr < 0 else tr * dt,
            "reported_rotor_at_trip": rep, "reported_mos_at_trip": rep,
            "winding_at_trip": None if tr < 0 else float(r["winding_at_trip"][0]),
            "housing_at_trip": None if tr < 0 else float(r["housing_at_trip"][0]),
            "clear_allowed_step": None if cl < 0 else cl, "trace_steps": r["trace_steps"],
            "reported_trace": r["reported_trace"][:, 0].astype(np.int64), "winding_trace": r["winding_trace"][:, 0],
            "housing_trace": r["housing_trace"][:, 0]}


def simulate_constant(consts, t_del, t_w0, t_h0, t_amb, seconds, dt=DT, ot_value=None, record_every=0):
    n = consts.n
    tw = np.broadcast_to(np.asarray(t_w0, dtype=np.float64), (n,)).copy()
    th = np.broadcast_to(np.asarray(t_h0, dtype=np.float64), (n,)).copy()
    td = np.broadcast_to(np.asarray(t_del, dtype=np.float64), (n,)).copy()
    steps = int(round(seconds / dt))
    trip = np.full(n, -1, dtype=np.int64)
    th_trip = np.full(n, np.nan)
    series = []
    for i in range(steps):
        if ot_value is not None:
            new = coil_fault(tw, ot_value) & (trip < 0)
            if new.any():
                trip[new] = i
                th_trip[new] = th[new]
        if record_every and i % record_every == 0:
            series.append((i, tw.copy(), th.copy()))
        tw, th = step(tw, th, td, consts, t_amb, dt)
    if ot_value is not None:
        new = coil_fault(tw, ot_value) & (trip < 0)
        trip[new] = steps
        th_trip[new] = th[new]
    if record_every:
        series.append((steps, tw.copy(), th.copy()))
    return {"t_w": tw, "t_h": th, "trip_step": trip, "t_h_at_trip": th_trip, "series": series}


def steady_state(a, h, k1, k2, k3, t_del, t_amb, form=None):
    alpha, ref, _ = thermal_form(form)
    p = a * h * t_del * t_del
    r = k3 / k2
    den = k1 * r - p * alpha * (1.0 + r)
    if den <= 0.0:
        return None
    x = p * (1.0 + alpha * (t_amb - ref)) / den
    return t_amb + x + r * x, t_amb + x
