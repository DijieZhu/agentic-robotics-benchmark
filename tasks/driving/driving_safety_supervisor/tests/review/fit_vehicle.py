"""fit_vehicle - the ego car's steering response, fitted from recorded episodes (a maintainer's tool; CPU only).

The reference expert (solution/solver.py) checks its own swept path by rolling a kinematic bicycle forward under its
own steering law. This script fits that bicycle's steering response to recorded episodes of the task's car
(vehicle.lincoln.mkz_2020 in CARLA 0.9.15), whose every control step holds the applied [throttle, steer, brake], the
car's speed and its pose (x, y, yaw) at the step's start:

    s <- s + (u - s) * (1 - exp(-dt / tau))       the steer the wheels actually have, lagging the command u
    w = g * MAX_STEER * s * curve(v)              the wheel angle (curve: CARLA's default steering curve, or 1)
    beta = atan(slip * tan(w))                    the side-slip angle at the car's centre (slip = l_r / L)
    heading rate = v * cos(beta) * tan(w) / WHEELBASE,  position rate = v along heading + beta

The filter runs from each episode's first step (the car stands settled, s = 0). From every step with v >= 2 m/s the
model is rolled 1 s and 2 s ahead on the recorded commands and speeds (0.05 s substeps, as the simulator ticks), and
the predicted position is compared with the recorded one in the car's frame at the start: the lateral error at +1 s
and +2 s is what the expert's swept-path check relies on.

    python3 -B fit_vehicle.py <每步记录_steps_meta.tar.gz | a folder of */*/steps.jsonl> [--out fit.json]

Prints the best (tau, g, slip, curve) by the RMS lateral error at +1 s and +2 s, its percentiles, and the same for
the model the expert used before (no lag, g = 1, no slip, no curve). stdlib only.
"""

import argparse
import glob
import io
import json
import math
import os
import tarfile

DT_STEP, SUBSTEPS = 0.1, 2
WHEELBASE, MAX_STEER = 2.9, 1.22
CURVE = ((0.0, 1.0), (20.0, 0.9), (60.0, 0.8), (120.0, 0.7))   # km/h -> factor (CARLA 0.9.15's default)


def curve_factor(v):
    kmh = v * 3.6
    for (a, fa), (b, fb) in zip(CURVE, CURVE[1:]):
        if kmh <= b:
            return fa + (fb - fa) * (kmh - a) / (b - a)
    return CURVE[-1][1]


def read_episodes(source):
    """[(name, [(x, y, yaw_rad, v, u), ...]), ...] from the tarball or an extracted folder."""
    texts = []
    if os.path.isdir(source):
        for path in sorted(glob.glob(os.path.join(source, "**", "steps.jsonl"), recursive=True)):
            with open(path) as handle:
                texts.append((os.path.relpath(os.path.dirname(path), source), handle.read()))
    else:
        with tarfile.open(source) as tar:
            for member in sorted(tar.getmembers(), key=lambda m: m.name):
                if member.name.endswith("steps.jsonl"):
                    raw = tar.extractfile(member).read()
                    texts.append((os.path.dirname(member.name), io.TextIOWrapper(io.BytesIO(raw)).read()))
    episodes = []
    for name, text in texts:
        rows = []
        for line in text.splitlines():
            rec = json.loads(line)
            if rec.get("final"):
                break
            x, y, _z, yaw = rec["ego"]
            rows.append((x, y, math.radians(yaw), float(rec["speed"]), float(rec["applied"][1])))
        if len(rows) > 30:
            episodes.append((name, rows))
    return episodes


def lag_states(rows, tau):
    """The filter's steer at each step's start (the car stands settled at step 0)."""
    alpha = 1.0 - math.exp(-DT_STEP / SUBSTEPS / tau) if tau > 0 else 1.0
    s, out = 0.0, []
    for row in rows:
        out.append(s)
        for _ in range(SUBSTEPS):
            s += (row[4] - s) * alpha
    return out


def rollout(rows, k, s0, horizon, tau, g, slip, curve):
    """Predicted (x, y) after `horizon` steps from step k, in the frame of the car at step k."""
    alpha = 1.0 - math.exp(-DT_STEP / SUBSTEPS / tau) if tau > 0 else 1.0
    x = y = psi = 0.0
    s = s0
    h = DT_STEP / SUBSTEPS
    for j in range(k, k + horizon):
        v0, v1 = rows[j][3], rows[j + 1][3]
        u = rows[j][4]
        for i in range(SUBSTEPS):
            s += (u - s) * alpha
            v = v0 + (v1 - v0) * (i + 0.5) / SUBSTEPS
            w = g * MAX_STEER * s * (curve_factor(v) if curve else 1.0)
            tw = math.tan(w)
            beta = math.atan(slip * tw)
            x += v * math.cos(psi + beta) * h
            y += v * math.sin(psi + beta) * h
            psi += v * math.cos(beta) * tw / WHEELBASE * h
    return x, y


def errors(episodes, tau, g, slip, curve, stride=1):
    """Lateral errors at +1 s and +2 s (metres, signed) over every start with v >= 2 m/s."""
    e1, e2 = [], []
    for _name, rows in episodes:
        states = lag_states(rows, tau)
        for k in range(1, len(rows) - 21, stride):
            if rows[k][3] < 2.0:
                continue
            x0, y0, yaw0 = rows[k][0], rows[k][1], rows[k][2]
            c, s = math.cos(yaw0), math.sin(yaw0)
            for horizon, sink in ((10, e1), (20, e2)):
                px, py = rollout(rows, k, states[k], horizon, tau, g, slip, curve)
                ax, ay = rows[k + horizon][0] - x0, rows[k + horizon][1] - y0
                # CARLA's frame is left-handed (y to the right): rotate by -yaw, keep "right" positive
                lat = -s * ax + c * ay
                sink.append(py - lat)
    return e1, e2


def summary(values):
    a = sorted(abs(v) for v in values)
    if not a:
        return {}
    pick = lambda q: a[min(len(a) - 1, int(q * len(a)))]  # noqa: E731
    return {"n": len(a), "rms": round(math.sqrt(sum(v * v for v in a) / len(a)), 4),
            "p50": round(pick(0.5), 4), "p95": round(pick(0.95), 4), "max": round(a[-1], 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--out")
    args = parser.parse_args()
    episodes = read_episodes(args.source)
    print("[fit] {} episodes, {} steps".format(len(episodes), sum(len(r) for _n, r in episodes)))

    def cost(params, stride=3):
        e1, e2 = errors(episodes, *params, stride=stride)
        return math.sqrt((sum(v * v for v in e1) / len(e1) + sum(v * v for v in e2) / len(e2)) / 2.0)

    best = None
    for curve in (False, True):
        for slip in (0.0, 0.5):
            for tau in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7):
                for gi in range(0, 25):
                    g = 0.5 + 0.025 * gi
                    c = cost((tau, g, slip, curve))
                    if best is None or c < best[0]:
                        best = (c, tau, g, slip, curve)
    _c, tau, g, slip, curve = best
    # refine around the coarse optimum
    for tau_r in [max(0.005, tau + d) for d in (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04)]:
        for g_r in [g + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]:
            c = cost((tau_r, g_r, slip, curve))
            if c < best[0]:
                best = (c, tau_r, g_r, slip, curve)
    _c, tau, g, slip, curve = best
    e1, e2 = errors(episodes, tau, g, slip, curve)
    b1, b2 = errors(episodes, 1e-9, 1.0, 0.0, False)
    result = {
        "model": "lagged kinematic bicycle (fit_vehicle.py)",
        "tau_s": round(tau, 4), "gain": round(g, 4), "slip": slip, "steering_curve": curve,
        "wheelbase_m": WHEELBASE, "max_steer_rad": MAX_STEER,
        "episodes": len(episodes), "steps": sum(len(r) for _n, r in episodes),
        "lateral_error_1s": summary(e1), "lateral_error_2s": summary(e2),
        "before": {"model": "no lag, gain 1, no slip, no curve",
                   "lateral_error_1s": summary(b1), "lateral_error_2s": summary(b2)},
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
