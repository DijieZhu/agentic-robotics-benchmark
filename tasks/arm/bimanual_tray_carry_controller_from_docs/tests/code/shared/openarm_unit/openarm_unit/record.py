import copy
import dataclasses
import hashlib
import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from . import hostcodec as hc
from . import rng
from .frames import Frame
from .state import public_params

NS_PER_MS = 1_000_000
LAYOUT_VERSION = "0.4.0"
SIDES = ("left", "right")
SIDE_BUS = {"left": 0, "right": 1}
MAX_STEPS = 50
SIGNED_PHASE = "signed_phase"
ACTION_GRID_OFFSETS = "action_grid_offsets"
OBSERVATION_MODELS = (SIGNED_PHASE, ACTION_GRID_OFFSETS)
STAMP_PATHS = ("float", "int")


class RecordError(RuntimeError):
    ""


def _law():
    from .law import law_params
    return law_params()


def _quantile_points(d):
    pts = []
    for k, v in d.items():
        p = 1.0 if k == "max" else float(Fraction(k[1:]) / 100)
        pts.append((p, float(v)))
    pts.sort()
    return tuple(pts)


def quantile(pts, u):
    if u <= pts[0][0]:
        return pts[0][1]
    for (p0, x0), (p1, x1) in zip(pts[:-1], pts[1:]):
        if u <= p1:
            return x0 + (x1 - x0) * ((u - p0) / (p1 - p0))
    return pts[-1][1]


def cdf(pts, x):
    if x <= pts[0][1]:
        return pts[0][0]
    for (p0, x0), (p1, x1) in zip(pts[:-1], pts[1:]):
        if x <= x1:
            return p0 + (p1 - p0) * ((x - x0) / (x1 - x0))
    return pts[-1][0]


def float_path_ns(t_ns):
    us = int(t_ns) // 1000
    secs = us / 1_000_000
    x = secs * 1_000_000_000
    return -math.floor(-x)


@dataclass(frozen=True)
class RecorderProfile:
    name: str
    interval_q: tuple
    offset_q: tuple
    count_hist: tuple
    lead_ms: float = None
    dispatch_ms: tuple = None
    obs_stamp_path: str = "float"
    action_stamp_path: str = "int"
    observation_model: str = SIGNED_PHASE
    neg_fraction: float = None
    neg_q: tuple = None
    phase_q: tuple = None

    def __post_init__(self):
        if self.observation_model not in OBSERVATION_MODELS:
            raise RecordError(f"observation_model: one of {OBSERVATION_MODELS}")
        if self.observation_model == SIGNED_PHASE:
            if self.neg_fraction is None or self.neg_q is None or self.phase_q is None:
                raise RecordError("the signed-phase model needs neg_fraction, neg_q and phase_q")
            if not 0.0 <= float(self.neg_fraction) <= 1.0:
                raise RecordError("neg_fraction: a fraction in [0, 1]")
        if self.obs_stamp_path not in STAMP_PATHS or self.action_stamp_path not in STAMP_PATHS:
            raise RecordError(f"stamp paths: one of {STAMP_PATHS}")
        if (self.lead_ms is None) == (self.dispatch_ms is None):
            raise RecordError("a profile has either a lead or a dispatch window")

    @classmethod
    def from_law(cls, name, law=None):
        law = _law() if law is None else law
        rp = law["recorder_profiles"]
        ru = rp["real_unit"]
        base = dict(interval_q=_quantile_points(ru["tick_interval_quantiles_ms"]["value"]),
                    offset_q=_quantile_points(ru["offset_quantiles_ms"]["value"]),
                    count_hist=tuple(sorted((int(k), int(v)) for k, v in
                                            ru["count_difference_histogram"]["value"].items())),
                    observation_model=str(ru["observation_model"]["value"]),
                    neg_fraction=float(ru["phase_negative_fraction"]["value"]),
                    neg_q=_quantile_points(ru["phase_negative_quantiles_ms"]["value"]),
                    phase_q=_quantile_points(ru["phase_positive_quantiles_ms"]["value"]))
        if int(ru["stamp_grid_ns"]["value"]) != 256:
            raise RecordError("the law's observation stamp grid is not the float path's 256 ns")
        if name == "real_unit":
            return cls("real_unit", lead_ms=float(ru["lead_ms"]["value"]),
                       obs_stamp_path=str(ru["observation_stamp_path"]["value"]), **base)
        if name == "stage04_day":
            sd = rp["stage04_day"]
            if not str(sd["timing"]["value"]).startswith("as real_unit except the lead"):
                raise RecordError("stage04_day's timing is not 'as real_unit except the lead'")
            lo, hi = (float(x) for x in sd["dispatch_after_stamp_ms"]["value"])
            return cls("stage04_day", dispatch_ms=(lo, hi),
                       obs_stamp_path=str(sd["observation_stamp_path"]["value"]), **base)
        raise RecordError(f"unknown recorder profile {name!r}")

    def with_observation_model(self, model):
        if model not in OBSERVATION_MODELS:
            raise RecordError(f"observation_model: one of {OBSERVATION_MODELS}")
        if model == self.observation_model:
            return self
        return dataclasses.replace(self, name=f"{self.name}+{model}", observation_model=model)

    def lead_text(self):
        return f"lead {self.lead_ms} ms" if self.lead_ms is not None else f"dispatch {self.dispatch_ms} ms after"


class Tick(NamedTuple):
    side: str
    index: int
    stamp_ns: int
    step: int
    unit: object


@dataclass
class ArmSchedule:
    side: str
    stamps: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    obs_stamps: list = field(default_factory=list)
    obs_steps: list = field(default_factory=list)
    next_step: int = 0


class SessionRecorder:

    def __init__(self, unit, host, profile, out_dir, key, *, epoch_ns, metadata=None, sides=SIDES, name="session"):
        if not isinstance(key, (bytes, bytearray)) or len(key) != rng.KEY_BYTES:
            raise RecordError("key: 32 bytes")
        if isinstance(epoch_ns, bool) or int(epoch_ns) != epoch_ns or epoch_ns < 0:
            raise RecordError("epoch_ns: a non-negative integer of nanoseconds")
        if isinstance(profile, str):
            profile = RecorderProfile.from_law(profile)
        if not isinstance(profile, RecorderProfile):
            raise RecordError("profile: a RecorderProfile or a profile name")
        sides = tuple(sides)
        if not sides or any(s not in SIDES for s in sides) or len(set(sides)) != len(sides):
            raise RecordError(f"sides: some of {SIDES}")
        for s in sides:
            d = host.get(s) if hasattr(host, "get") else None
            if d is None or not all(hasattr(d, a) for a in ("send_position", "receive", "elapse")):
                raise RecordError(f"host[{s!r}]: a driver with send_position, receive and elapse")
        if not name or "/" in name:
            raise RecordError("name: a non-empty name without '/'")
        self.unit, self.host, self.profile = unit, host, profile
        self.out_dir = os.path.abspath(out_dir)
        self.key = bytes(key)
        self.epoch_ns = int(epoch_ns)
        self.metadata = copy.deepcopy(metadata) if metadata is not None else {}
        self.sides = sides
        self.name = name
        self.episodes = []
        self._next_id = 0
        p = public_params()
        self.step_ns = int(round(float(p["conventions"]["time"]["value"]["step_s"]) * 1e9))
        if self.step_ns != NS_PER_MS:
            raise RecordError("the recorder's grid is the unit's 1 ms step")
        os.makedirs(self.out_dir, exist_ok=True)

    def _stream(self, ep, side, what):
        return f"rec/{self.name}/e{int(ep)}/{side}/{what}"

    def _u(self, ep, side, what, start, count):
        return rng.uniforms(self.key, self._stream(ep, side, what), start, count)

    def t_ns(self, step):
        return self.epoch_ns + int(step) * NS_PER_MS

    def _effect_step(self, s, d_u):
        pr = self.profile
        if pr.lead_ms is not None:
            target = s - int(math.floor(pr.lead_ms * NS_PER_MS + 0.5))
            return (target - self.epoch_ns + NS_PER_MS // 2) // NS_PER_MS
        lo_ms, hi_ms = pr.dispatch_ms
        lo_ns = int(math.floor(lo_ms * NS_PER_MS + 0.5))
        hi_ns = int(math.floor(hi_ms * NS_PER_MS + 0.5))
        d = lo_ns + int(math.floor((hi_ns - lo_ns) * d_u))
        first = -((-(s + lo_ns - self.epoch_ns)) // NS_PER_MS)
        last = (s + hi_ns - self.epoch_ns) // NS_PER_MS
        if last < first:
            raise RecordError("the dispatch window holds no step")
        want = s + d - self.epoch_ns
        best = min(range(first, last + 1), key=lambda n: (abs(n * NS_PER_MS - want), n))
        return best

    def plan(self, duration_ms, episode=None, start_step=None):
        if isinstance(duration_ms, bool) or int(duration_ms) != duration_ms or duration_ms < 1:
            raise RecordError("duration_ms: a positive whole number of milliseconds")
        ep = self._next_id if episode is None else int(episode)
        n0 = int(self.unit.bus.step_count) if start_step is None else int(start_step)
        pr = self.profile
        out = {}
        for side in self.sides:
            sch = ArmSchedule(side)
            if pr.lead_ms is not None:
                s = self.t_ns(n0) + int(math.floor(pr.lead_ms * NS_PER_MS + 0.5))
            else:
                lo_ms, hi_ms = pr.dispatch_ms
                lo_ns = int(math.floor(lo_ms * NS_PER_MS + 0.5))
                hi_ns = int(math.floor(hi_ms * NS_PER_MS + 0.5))
                d0 = lo_ns + int(math.floor((hi_ns - lo_ns) * float(self._u(ep, side, "dispatch", 0, 1)[0])))
                s = self.t_ns(n0) - d0
            end = s + int(duration_ms) * NS_PER_MS
            k = 0
            draw = _Chunked(self, ep, side)
            prev = None
            phase = pr.observation_model == SIGNED_PHASE
            o_grid = None
            while True:
                if phase and k > 0:
                    s = o_grid - self._phase_ns(draw, k)
                n = n0 if k == 0 else self._effect_step(s, draw("dispatch", k))
                if phase and prev is not None and n <= prev:
                    n = prev + 1
                if prev is not None and n <= prev:
                    raise RecordError("two ticks of one arm act at one step")
                if s >= end:
                    sch.next_step = n
                    break
                I_ms = quantile(pr.interval_q, draw("interval", k))
                I_ns = int(math.floor(I_ms * NS_PER_MS + 0.5))
                if phase:
                    o = s + self._phase_ns(draw, k) if k == 0 else o_grid
                    o_grid = o + I_ns
                else:
                    u = draw("offset", k)
                    x_ms = quantile(pr.offset_q, u)
                    if x_ms >= I_ms:
                        x_ms = quantile(pr.offset_q, u * cdf(pr.offset_q, I_ms))
                    x_ns = min(int(math.floor(x_ms * NS_PER_MS + 0.5)), I_ns - 1)
                    o = s + x_ns
                m = max(n0, (o - self.epoch_ns + NS_PER_MS // 2) // NS_PER_MS)
                sch.stamps.append(s)
                sch.steps.append(n)
                sch.obs_stamps.append(o)
                sch.obs_steps.append(m)
                prev = n
                s = s + I_ns
                k += 1
            out[side] = sch
        return out

    def _phase_ns(self, draw, k):
        pr = self.profile
        u = draw("offset", k)
        if draw("phase_sign", k) < pr.neg_fraction:
            return -int(math.floor(quantile(pr.neg_q, u) * NS_PER_MS + 0.5))
        return int(math.floor(quantile(pr.phase_q, u) * NS_PER_MS + 0.5))

    def record_episode(self, script, duration_ms, episode_id=None, success=True, task_index=0, admit="all"):
        ep = self._next_id if episode_id is None else int(episode_id)
        if any(e["id"] == str(ep) for e in self.episodes):
            raise RecordError(f"episode {ep} already exists in this dataset")
        ep_dir = os.path.join(self.out_dir, "episodes", str(ep))
        if os.path.exists(ep_dir):
            raise RecordError(f"episode directory already exists: {ep_dir}")
        unit = self.unit
        n0 = int(unit.bus.step_count)
        sched = self.plan(duration_ms, ep, n0)
        dispatch = {}
        obs_at = {}
        end_step = n0 + 1
        for side, sch in sched.items():
            for k, (n, m) in enumerate(zip(sch.steps, sch.obs_steps)):
                dispatch.setdefault(n, []).append((side, k))
                obs_at.setdefault(m, []).append((side, k))
                end_step = max(end_step, m + 1)
            end_step = max(end_step, sch.next_step)
        bounds = sorted(set(dispatch) | {end_step})
        actions = {s: [] for s in self.sides}
        rows_at = {}
        events = []
        unit.tapped()
        for b, e in zip(bounds[:-1], bounds[1:]):
            frames = []
            for side, k in sorted(dispatch.get(b, ()), key=lambda t: SIDES.index(t[0])):
                sch = sched[side]
                req = script(Tick(side, k, sch.stamps[k], b, unit))
                if req is None:
                    continue
                req = np.asarray(req, dtype=np.float64)
                if req.shape != (8,) or not np.all(np.isfinite(req)):
                    raise RecordError(f"the script's request for {side} at tick {k}: eight finite positions")
                actions[side].append((k, sch.stamps[k], req.copy()))
                frames.extend(self.host[side].send_position(req))
            pos = b
            first = True
            while pos < e:
                L = min(MAX_STEPS, e - pos)
                taps = [m - pos for m in sorted(obs_at) if pos <= m < pos + L]
                replies = unit.exchange((frames or None) if first else None, steps=L, taps=taps, admit=admit)
                for side in self.sides:
                    bus = SIDE_BUS[side]
                    drv = self.host[side]
                    drv.receive([r for r in replies if r.bus == bus])
                    drv.elapse(L)
                for step, _j, rows in unit.tapped():
                    rows_at[int(step)] = rows
                first = False
                pos += L
            for side in self.sides:
                de = getattr(self.host[side], "drain_events", None)
                if de is not None:
                    for ev in de():
                        events.append(dict(ev, side=side, step=b))
        files = {}
        summary = {"id": str(ep), "steps": [n0, end_step], "ticks": {}, "n_obs": {}, "n_act": {},
                   "count_difference": {}, "removed": {}, "events": events, "files": files}
        mods = public_params()["conventions"]["modules"]["value"]
        for side in self.sides:
            sch = sched[side]
            drv = self.host[side]
            idx = [int(m["index"]) for m in sorted((m for m in mods if m["side"] == side), key=lambda m: m["joint"])]
            limits = drv.motors.limits
            offs = np.asarray(getattr(drv, "joint_offsets", np.zeros(8)), dtype=np.float64)
            obs = []
            for k, m in enumerate(sch.obs_steps):
                rows = rows_at.get(m)
                if rows is None:
                    raise RecordError(f"no tap at step {m}")
                q, dq, tau = [], [], []
                for j, i in enumerate(idx):
                    st = hc.unpack_state(rows[i], limits[j])
                    q.append(st["q"])
                    dq.append(st["dq"])
                    tau.append(st["tau"])
                obs.append((k, self._obs_stamp(sch.obs_stamps[k]), np.asarray(q) - offs, np.asarray(dq),
                            np.asarray(tau)))
            K = len(sch.stamps)
            c = self._count_difference(ep, side)
            rm_obs, rm_act = [], []
            if c < 0:
                rm_obs = self._removal(ep, side, -c, len(obs), [o[1] for o in obs])
                drop = set(rm_obs)
                obs = [o for n, o in enumerate(obs) if n not in drop]
            elif c > 0:
                rm_act = self._removal(ep, side, c, len(actions[side]), [a[1] for a in actions[side]])
                drop = set(rm_act)
                actions[side] = [a for n, a in enumerate(actions[side]) if n not in drop]
            summary["ticks"][side] = K
            summary["count_difference"][side] = c
            summary["removed"][side] = {"obs": rm_obs, "action": rm_act}
            summary["n_obs"][side] = len(obs)
            summary["n_act"][side] = len(actions[side])
            if obs:
                p = os.path.join(ep_dir, "obs", "arms", side, "state.parquet")
                files[os.path.relpath(p, self.out_dir)] = _write_state(
                    p, [o[1] for o in obs], {"qpos": [o[2] for o in obs], "qvel": [o[3] for o in obs],
                                             "qtorque": [o[4] for o in obs]})
            if actions[side]:
                p = os.path.join(ep_dir, "action", "arms", side, "state.parquet")
                files[os.path.relpath(p, self.out_dir)] = _write_state(
                    p, [self._action_stamp(a[1]) for a in actions[side]], {"qpos": [a[2] for a in actions[side]]})
        self.episodes.append({"id": str(ep), "success": bool(success), "task_index": int(task_index)})
        files["metadata.yaml"] = self._write_metadata()
        self._next_id = max(self._next_id, ep + 1)
        return summary

    def _obs_stamp(self, t):
        return float_path_ns(t) if self.profile.obs_stamp_path == "float" else int(t)

    def _action_stamp(self, t):
        return float_path_ns(t) if self.profile.action_stamp_path == "float" else int(t)

    def _count_difference(self, ep, side):
        hist = self.profile.count_hist
        total = sum(n for _, n in hist)
        u = float(self._u(ep, side, "count", 0, 1)[0])
        x = u * total
        acc = 0
        for c, n in hist:
            acc += n
            if x < acc:
                return int(c)
        return int(hist[-1][0])

    def _removal(self, ep, side, n_remove, n_rows, stamps=None):
        n_remove = min(int(n_remove), max(0, n_rows - 1))
        us = self._u(ep, side, "remove", 0, n_remove)
        kept = list(range(n_rows))
        gap_max = int(math.floor(self.profile.interval_q[-1][1] * NS_PER_MS + 0.5))
        out = []
        for j in range(n_remove):
            pool = kept
            if stamps is not None:
                ok = []
                for p, r in enumerate(kept):
                    if 0 < p < len(kept) - 1 and stamps[kept[p + 1]] - stamps[kept[p - 1]] > gap_max:
                        continue
                    ok.append(r)
                pool = ok or kept
            r = pool[int(math.floor(float(us[j]) * len(pool)))]
            kept.remove(r)
            out.append(r)
        return sorted(out)

    def _write_metadata(self):
        import yaml
        md = copy.deepcopy(self.metadata)
        md["version"] = LAYOUT_VERSION
        md["episodes"] = copy.deepcopy(self.episodes)
        p = os.path.join(self.out_dir, "metadata.yaml")
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(md, f, default_flow_style=False, allow_unicode=True)
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


class _Chunked:
    CHUNK = 256

    def __init__(self, rec, ep, side):
        self.rec, self.ep, self.side = rec, ep, side
        self.cache = {}

    def __call__(self, what, k):
        c = k // self.CHUNK
        arr = self.cache.get((what, c))
        if arr is None:
            arr = self.rec._u(self.ep, self.side, what, c * self.CHUNK, self.CHUNK)
            self.cache[(what, c)] = arr
        return float(arr[k - c * self.CHUNK])


def _write_state(path, timestamps, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    list_type = pa.list_(pa.float32())
    fields = {"timestamp": pa.array([int(t) for t in timestamps], type=pa.timestamp("ns"))}
    for name, vals in columns.items():
        fields[name] = pa.array([np.asarray(v, dtype=np.float32) for v in vals], type=list_type)
    pq.write_table(pa.table(fields), path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_state(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    ts = t.column("timestamp").cast("int64").to_numpy()
    cols = {}
    for name in t.column_names:
        if name == "timestamp":
            continue
        cols[name] = np.array(t.column(name).to_pylist(), dtype=np.float32)
    return ts, cols
