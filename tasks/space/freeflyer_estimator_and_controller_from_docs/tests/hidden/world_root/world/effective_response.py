"""The response the reference reads, as a function of the session (world.effective_response): per session, signed axis
and fan level, the least-squares gain of the delivered over the commanded wrench on the registered check-out
sequence (cert.checkout_sequence) in that session's world, with the session's steady wrench acting and without
sensor noise: a calibration run of the plant with the session's parameters. No job is flown and no controller is in
the loop: the commanded wrench is a fixed time series, the plant is stepped open loop, and no sensor model is built.






























"""
import math
import re

import numpy as np

from .config import V, SIGNED, torque_gains
from .plant import static_response

LAG_GRID_S = tuple(0.004 * k for k in range(41))


def _sequence_numbers():
    txt = V("cert.checkout_sequence")
    m_rest = re.search(r"([0-9.]+) s at rest", txt)
    m_pulse = re.search(r"one torque pulse of ([0-9.]+) s on each of the six signed axes, each followed by its return", txt)
    m_burn = re.search(r"one out-and-back burn of about ([0-9.]+) s on each of the six signed force axes", txt)
    m_tot = re.search(r"([0-9.]+) s in all", txt)
    # the assert below accepts either wording of the limits in cert.checkout_sequence: each fan level at its own
    # mode's limits, or the nominal mode's limits
    assert m_rest and m_pulse and m_burn and m_tot and ("at each fan level's own mode's limits (the lowest level by the quiet mode's)" in txt
                                                         or "at the nominal mode's limits" in txt), "cert.checkout_sequence changed"
    rest, pulse, burn, total = (float(m.group(1)) for m in (m_rest, m_pulse, m_burn, m_tot))
    assert abs(rest + 6 * 2 * pulse + 6 * burn - total) < 1e-9, "the sequence's parts do not add up to its length"
    return rest, pulse, burn, total


def checkout_profile(mass, inertia, static, dt, mode="nominal"):
    """the commanded wrench (n, 6) of the check-out sequence, one row per step of dt (reading (a)).
    static: {"force": [6 signed], "torque": [6 signed]} the session's static response at the level flown."""
    n_rest, n_pulse, n_quarter = step_counts(dt)
    ac = V("plan.acceleration_limit")[mode]
    sp = V("plan.speed_limit." + mode)
    alpha = min(float(ac["angular_rad_s2"]), float(sp["angular_rad_s"]) / (n_pulse * dt))
    acc = min(float(ac["linear_m_s2"]), float(sp["linear_m_s"]) / (n_quarter * dt))
    J = np.asarray(inertia, float)
    rows = [np.zeros((n_rest, 6))]
    for i in range(3):
        for sg in (1.0, -1.0):
            g_out = static["torque"][2 * i + (0 if sg > 0 else 1)]
            g_back = static["torque"][2 * i + (1 if sg > 0 else 0)]
            w = np.zeros(6); w[3 + i] = sg * J[i, i] * alpha / g_out
            r = np.zeros(6); r[3 + i] = -sg * J[i, i] * alpha / g_back
            rows += [np.repeat(w[None], n_pulse, 0), np.repeat(r[None], n_pulse, 0)]
    for i in range(3):
        for sg in (1.0, -1.0):
            g_out = static["force"][2 * i + (0 if sg > 0 else 1)]
            g_back = static["force"][2 * i + (1 if sg > 0 else 0)]
            w = np.zeros(6); w[i] = sg * mass * acc / g_out
            r = np.zeros(6); r[i] = -sg * mass * acc / g_back
            rows += [np.repeat(w[None], n_quarter, 0), np.repeat(r[None], 2 * n_quarter, 0), np.repeat(w[None], n_quarter, 0)]
    return np.vstack(rows)


def step_counts(dt):
    """the sequence's parts on the world's step grid: (rest, one pulse (its return is as long), a quarter of a burn),
    each the registered duration rounded to whole steps (a pulse of 1.5 s is 94 steps of 16 ms: 1.504 s)."""
    rest, pulse, burn, total = _sequence_numbers()
    return int(round(rest / dt)), int(round(pulse / dt)), int(round(0.25 * burn / dt))


def static_of(plant, level):
    """the static response per signed axis at a level (as World.reference_reads computes it)."""
    return {"force": [static_response(plant, level, i // 2, 0.05 if i % 2 == 0 else -0.05) for i in range(6)],
            "torque": [static_response(plant, level, 3 + i // 2, 0.005 if i % 2 == 0 else -0.005) for i in range(6)]}


def steady_wrench(world, level):
    """reading (b): (the body-fixed air wrench acting, the hold command that balances it at the level's gains)."""
    air = np.asarray(world.air_body, float)
    gt = torque_gains(level)
    gf = world.force_gains(level)
    hold = np.zeros(6)
    for i in range(3):
        f = -air[i]
        hold[i] = f / (gf[2 * i] if f >= 0 else gf[2 * i + 1])
        t = -air[3 + i]
        hold[3 + i] = t / (gt[2 * i] if t >= 0 else gt[2 * i + 1])
    return air, hold


def calibration_run(world, level, commands, steady=True, plant=None):
    """the plant alone, open loop: (commanded (n, 6), delivered (n, 6)) over the given commands (rows of dt).
    plant: a stand-in with the same step / initial_state interface (the cause analysis switches parts of it)."""
    P = plant or world.plant
    air, hold = steady_wrench(world, level) if steady else (np.zeros(6), np.zeros(6))
    cmd = np.asarray(commands, float) + hold[None, :]
    sp = world.sess["start_pose"]
    s = P.initial_state(level, sp["position"], sp["attitude"], 0.0)
    out = np.empty_like(cmd)
    for k in range(cmd.shape[0]):
        P.step(s, cmd[k], level, air)
        out[k] = s["wrench"]
    return cmd, out


def _shifted(c, lag, dt):
    """the mean over each step of the piecewise-constant series c (rows of dt) delayed by lag."""
    n = int(math.floor(lag / dt + 1e-12))
    f = max(0.0, lag / dt - n)
    # step j covers [t_j - lag, t_j - lag + dt]: a fraction f of step j - n - 1 and 1 - f of step j - n
    a0 = np.vstack([np.repeat(c[:1], n + 1, 0), c])      # a0[j + 1] = c[j - n], a0[j] = c[j - n - 1]
    cur = a0[1: c.shape[0] + 1]
    prev = a0[0: c.shape[0]]
    return (1.0 - f) * cur + f * prev


def ls_gains(cmd, deliv, dt, lag=None, lags=LAG_GRID_S, skip_s=0.0):
    """reading (c): per kind {"lag_s", "gain": [6 signed], "const": [3], "rss": float}."""
    out = {}
    k0 = int(round(skip_s / dt))
    for kind, sl in (("force", slice(0, 3)), ("torque", slice(3, 6))):
        c = cmd[:, sl]
        y = deliv[:, sl]
        best = None
        for L in ([lag] if lag is not None else lags):
            cs = _shifted(c, L, dt)[k0:]
            yy = y[k0:]
            gains, consts, rss = [], [], 0.0
            for i in range(3):
                X = np.stack([np.maximum(cs[:, i], 0.0), np.minimum(cs[:, i], 0.0), np.ones(cs.shape[0])], 1)
                coef, *_ = np.linalg.lstsq(X, yy[:, i], rcond=None)
                rss += float(((X @ coef - yy[:, i]) ** 2).sum())
                gains += [float(coef[0]), float(coef[1])]
                consts.append(float(coef[2]))
            if best is None or rss < best["rss"] - 1e-30:
                best = {"lag_s": float(L), "gain": gains, "const": consts, "rss": rss}
        out[kind] = best
    return out


def effective_response(world, levels=(1, 2, 3), steady=True, lag=None, plant=None, commands=None, limits="own"):
    """world.effective_response of the session the World was built for: {level: {"force": {signed: gain},
    "torque": {signed: gain}, "lag_s": {...}, "static": {...}}}. commands: an optional {level: (n, 6)} of the
    reference package's own check-out commands (rows of the world's step) in place of reading (a)'s profile.
    limits: "own" or "nominal" (the first run's reading, the nominal mode's limits at every level: the comparison
    only).
    """
    if limits not in ("own", "nominal"):
        raise ValueError(limits)
    P = plant or world.plant
    dt = world.dt
    own = {int(v): k for k, v in V("mode.fan_level").items()}
    res = {}
    for lv in levels:
        st = static_of(world.plant, lv)
        mode = "nominal" if limits == "nominal" else own[int(lv)]
        cmds = commands[lv] if commands is not None else checkout_profile(world.mass, world.J, st, dt, mode=mode)
        cmd, dl = calibration_run(world, lv, cmds, steady=steady, plant=P)
        g = ls_gains(cmd, dl, dt, lag=lag)
        res[lv] = {"force": dict(zip(SIGNED, g["force"]["gain"])), "torque": dict(zip(SIGNED, g["torque"]["gain"])),
                   "lag_s": {"force": g["force"]["lag_s"], "torque": g["torque"]["lag_s"]},
                   "static": {"force": dict(zip(SIGNED, st["force"])), "torque": dict(zip(SIGNED, st["torque"]))}}
    return res
