"""The house cell's wire surface (grader side): CellServer wraps one Cell episode and answers the JSON-lines requests
of the program under test (cellkit.wire.serve_episode drives it).

Public methods (all a program may call). Every parameter is validated strictly (allowed keys, types, lengths, finite
numbers within +-1000); a malformed request changes nothing and is answered {"error": "bad_params", "detail": ...}:
  describe()                       the cell's published facts (robot, joints, control limits, grip, cameras, room,
                                   budgets, safety)
  get_state()                      sim time and the robot's own state (localised base pose, joints, hands)
  command(**targets)               the one robot's targets, any subset of
                                     base_velocity [vx, vy, wz]  |  base_goal [x, y, yaw] or null   (not both)
                                     stand_height h
                                     waist [yaw, roll, pitch]
                                     left_arm / right_arm  {"q": [7 joint targets]}  |
                                                           {"pose": [x, y, z, qw, qx, qy, qz], "frame": "base" | "room",
                                                            "path": "line" | "joint"}  |
                                                           {"pos": [x, y, z], "frame": ..., "path": ...}
                                                           (exactly one of q, pose, pos; frame and path optional, with
                                                           pose or pos only; pos leaves the wrist's orientation free)
                                     left_hand / right_hand "open" | "close"
  step(n_ticks)                    advance n_ticks x 10 ms, 1 <= n_ticks <= max_ticks_per_step
  get_camera(name, depth=True)     one RGB-D frame of ceiling_kitchen | ceiling_dining | ceiling_living | head (frame
                                   budget)
  finish()                         end the episode
Privileged methods (only with privileged=True, i.e. the oracle token; otherwise {"error": "forbidden"}, counted as an
illegal call like any unknown method):
  reference_clean()                the privileged reference cleaner (reference.Reference(cell).run()) inside the call
  truth()                          the true poses of the loose objects, the fixtures' states (lid angles, faucet lever,
                                   soap pump), the sponge's state, the grease and crumb states, the robot's true base
                                   pose and what its hands hold
`handle()` never raises: an unexpected exception is logged and answered {"error": "internal_error"} (inside step /
command / reference_clean it also ends the episode, whose state is then scored as it stands).

The attributes cellkit.wire.serve_episode relies on: done, done_reason, _end(reason), wall_start, wall_used,
work_wall (the cell's own time inside step / get_camera, not charged to the program), program_used,
steps_after_done. finalize() returns the episode's record: Cell.score() plus the call counts.
"""

from __future__ import annotations

import math
import time
import traceback

from . import objects as O
from .cell import PUBLIC_CAMERAS, Cell
from .robot import PELVIS_HEIGHT

__all__ = ["CellServer", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "BadParams", "validate_command", "LIMIT"]

PUBLIC_METHODS = ("describe", "get_state", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("reference_clean", "truth")
KNOWN_METHODS = PUBLIC_METHODS + PRIVILEGED_METHODS
SIDES = ("left", "right")
LIMIT = 1000.0                          # |x| bound of every number on the wire (metres, radians, m/s, ...)
COMMAND_KEYS = ("base_velocity", "base_goal", "stand_height", "waist", "left_arm", "right_arm", "left_hand",
                "right_hand")
ARM_KEYS = ("q", "pose", "pos", "frame", "path")
ARM_TARGETS = ("q", "pose", "pos")
FRAMES = ("base", "room")
PATHS = ("line", "joint")
HAND = ("open", "close")
MAX_LOGGED_TRACEBACKS = 5


class BadParams(ValueError):
    """A malformed request: answered {"error": "bad_params", "detail": str(e)}."""


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BadParams("%s must be a number" % what)
    try:
        f = float(v)
    except (OverflowError, ValueError):
        raise BadParams("%s must be a finite number within +-%g" % (what, LIMIT))
    if not math.isfinite(f) or abs(f) > LIMIT:
        raise BadParams("%s must be a finite number within +-%g" % (what, LIMIT))
    return f


def _vec(v, n, what):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise BadParams("%s must be a list of %d numbers" % (what, n))
    return [_num(x, "%s[%d]" % (what, i)) for i, x in enumerate(v)]


def _keys(params, allowed, required=(), what="parameters"):
    extra = sorted(str(k) for k in params if k not in allowed)
    if extra:
        raise BadParams("unknown %s %s (allowed: %s)" % (what, ", ".join(repr(k) for k in extra[:5]),
                                                         ", ".join(allowed) or "none"))
    missing = [k for k in required if k not in params]
    if missing:
        raise BadParams("missing %s %s" % (what, ", ".join(repr(k) for k in missing)))


def _choice(v, options, what):
    if not isinstance(v, str) or v not in options:
        raise BadParams("%s must be one of %s" % (what, ", ".join(repr(o) for o in options)))
    return v


def validate_command(params: dict) -> dict:
    """The command dict for Cell.command from a command request's params, or BadParams."""
    _keys(params, COMMAND_KEYS)
    cmd = {}
    if "base_velocity" in params and "base_goal" in params:
        raise BadParams("base_velocity and base_goal exclude each other (send base_goal null to stop a goal)")
    if "base_velocity" in params:
        cmd["base_velocity"] = _vec(params["base_velocity"], 3, "base_velocity")
    if "base_goal" in params:
        g = params["base_goal"]
        cmd["base_goal"] = None if g is None else _vec(g, 3, "base_goal")
    if "stand_height" in params:
        cmd["stand_height"] = _num(params["stand_height"], "stand_height")
    if "waist" in params:
        cmd["waist"] = _vec(params["waist"], 3, "waist")
    for side in SIDES:
        k = side + "_arm"
        if k in params:
            a = params[k]
            if not isinstance(a, dict):
                raise BadParams('%s must be an object: {"q": [7]}, {"pose": [7], "frame": ..., "path": ...} or '
                                '{"pos": [3], "frame": ..., "path": ...}' % k)
            _keys(a, ARM_KEYS, what="keys of " + k)
            given = [t for t in ARM_TARGETS if t in a]
            if len(given) != 1:
                raise BadParams("%s needs exactly one of q, pose, pos (got %s)" % (k, ", ".join(given) or "none"))
            if "q" in a:
                if "frame" in a or "path" in a:
                    raise BadParams("%s: frame and path go with pose or pos only" % k)
                cmd[k] = {"q": _vec(a["q"], 7, k + ".q")}
            else:
                if "pose" in a:
                    p = _vec(a["pose"], 7, k + ".pose")
                    if math.sqrt(sum(x * x for x in p[3:])) < 1e-6:
                        raise BadParams("%s.pose: the quaternion (qw, qx, qy, qz) has zero length" % k)
                    out = {"pose": p}
                else:
                    out = {"pos": _vec(a["pos"], 3, k + ".pos")}
                if "frame" in a:
                    out["frame"] = _choice(a["frame"], FRAMES, k + ".frame")
                if "path" in a:
                    out["path"] = _choice(a["path"], PATHS, k + ".path")
                cmd[k] = out
        k = side + "_hand"
        if k in params:
            cmd[k] = _choice(params[k], HAND, k)
    return cmd


def _plain(x):
    """A JSON-safe copy (numpy scalars / arrays and tuples as plain lists and numbers; dict keys as str)."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "tolist"):
        return _plain(x.tolist())
    return str(x)


class CellServer:
    """One episode of the house cell behind the wire."""

    def __init__(self, seed: int, *, privileged: bool = False, nominal: bool = False, config: dict | None = None,
                 log=None, render: bool = True):
        self.cfg = dict(config or {})
        self.seed = int(seed)
        self.privileged = bool(privileged)
        self.nominal = bool(nominal)
        self.log = log or (lambda s: None)
        self.max_ticks = int(self.cfg.get("max_ticks_per_step", 500))
        self.cell = Cell(self.seed, nominal=self.nominal, render=render, config=self.cfg)
        self.calls = {}
        self.illegal_calls = 0
        self.forbidden_names = []          # the first unknown method names a program called (the record)
        self.bad_param_calls = 0
        self.step_calls = 0
        self.internal_errors = 0
        self.render_errors = 0
        self.wall_start = None             # set by cellkit.wire.serve_episode
        self.wall_used = 0.0               # the episode's wall clock (set by the wire)
        self.program_used = 0.0            # its program time: the wall clock minus the cell's work_wall (set by the wire)
        self.steps_after_done = 0
        self.reference_report = None
        self._closed = False

    # ------------------------------------------------------------------------------------------- episode state
    @property
    def done(self) -> bool:
        return bool(self.cell.done)

    @property
    def done_reason(self):
        return self.cell.done_reason

    @property
    def work_wall(self) -> float:
        """Seconds the cell has spent simulating (step) and producing camera frames (get_camera) so far."""
        return float(self.cell.work_wall)

    def _end(self, reason: str) -> None:
        if not self.cell.done:
            self.cell._end(str(reason))

    # ------------------------------------------------------------------------------------------- dispatch
    def handle(self, method, params) -> dict:
        """Answer one request: the method's result, or {"error": code[, "detail": text]}. Never raises."""
        try:
            return _plain(self._dispatch(method, params))
        except Exception:  # noqa: BLE001 - the server never raises into the wire
            self.internal_errors += 1
            if self.internal_errors <= MAX_LOGGED_TRACEBACKS:
                self.log("[cell] internal error in %r:\n%s" % (method, traceback.format_exc()))
            if method in ("step", "command", "reference_clean"):
                self._end("internal_error")
            return {"error": "internal_error"}

    def _dispatch(self, method, params):
        if not isinstance(method, str):
            method = str(method)
        key = method if method in KNOWN_METHODS else "other"
        self.calls[key] = self.calls.get(key, 0) + 1
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
        elif method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            if len(self.forbidden_names) < 20:
                self.forbidden_names.append(method[:80])
            return {"error": "forbidden"}
        if params is None:
            params = {}
        try:
            if not isinstance(params, dict):
                raise BadParams("params must be an object")
            return getattr(self, "_m_" + method)(params)
        except BadParams as e:
            self.bad_param_calls += 1
            return {"error": "bad_params", "detail": str(e)}

    # ------------------------------------------------------------------------------------------- public methods
    def _m_describe(self, params):
        _keys(params, ())
        return self.cell.describe()

    def _m_get_state(self, params):
        _keys(params, ())
        return self.cell.get_state()

    def _m_command(self, params):
        return self.cell.command(validate_command(params))

    def _m_step(self, params):
        _keys(params, ("n_ticks",), required=("n_ticks",))
        n = params["n_ticks"]
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= self.max_ticks:
            raise BadParams("n_ticks must be an integer in [1, %d]" % self.max_ticks)
        self.step_calls += 1
        return self.cell.step(n)

    def _m_get_camera(self, params):
        _keys(params, ("name", "depth"), required=("name",))
        name = _choice(params["name"], PUBLIC_CAMERAS, "name")
        depth = params.get("depth", True)
        if not isinstance(depth, bool):
            raise BadParams("depth must be true or false")
        if self.cell.cams is None:
            return {"error": "camera_unavailable"}
        try:
            return self.cell.get_camera(name, depth=depth)
        except Exception:  # noqa: BLE001 - a renderer failure is the cell's, not the program's
            self.render_errors += 1
            if self.render_errors <= MAX_LOGGED_TRACEBACKS:
                self.log("[cell] camera %s failed:\n%s" % (name, traceback.format_exc()))
            return {"error": "camera_unavailable"}

    def _m_finish(self, params):
        _keys(params, ())
        return self.cell.finish()

    # ------------------------------------------------------------------------------------------- privileged methods
    def _m_reference_clean(self, params):
        _keys(params, ())
        if self.done:
            return {"ok": False, "reason": "the episode is over", "sim_time": self.cell.sim_time}
        t0 = time.time()
        err = None
        try:
            from .reference import Reference          # grader-side only
            ok = Reference(self.cell, log=self.log).run()
        except Exception:  # noqa: BLE001 - the reference's failure is reported, the episode is scored as it stands
            ok, err = False, traceback.format_exc()
            self.log("[cell] reference_clean raised:\n" + err)
        out = {"ok": bool(ok), "sim_time": self.cell.sim_time, "done": self.done, "wall_s": time.time() - t0}
        try:
            sc = self.cell.score()
            out["stage_score"] = float(sc["stage_score"])
            out["credit_by_part"] = dict(sc.get("credit_by_part") or {})
        except Exception:  # noqa: BLE001
            pass
        if err:
            out["exception"] = err.strip().splitlines()[-1][:300]
        self.reference_report = dict(out)
        return out

    def _m_truth(self, params):
        _keys(params, ())
        c = self.cell
        d = c.d
        objects = {}
        for name in O.LOOSE:                      # every loose object (trash, dishes, books, sponge, broom, dustpan)
            b = c.body[name]
            objects[name] = {"pos": [float(v) for v in d.xpos[b]], "quat": [float(v) for v in d.xquat[b]]}
        r = c.robot
        q = d.qpos
        z_off = float(q[r.base_q[3]])
        robot = {"base_pose": [float(q[r.base_q[0]]), float(q[r.base_q[1]]), float(q[r.base_q[2]])],
                 "base_z_offset": z_off, "stand_height": PELVIS_HEIGHT + z_off,
                 "holding": {s: c.grips.holding(s) for s in SIDES}}
        mech, pr = c.mech, c.process
        out = {
            "sim_time": c.sim_time,
            "objects": objects,
            "bins": dict(c.meta["bins"]),
            "lid_angles": {slot: mech.lid_angle(slot) for slot in c.meta["bins"]},
            "faucet": {"lever_angle": mech.lever_angle(), "water_on": bool(mech.water_on()), "flow": mech.flow(),
                       "water_s": float(mech.water_s)},
            "soap_pump": {"strokes": int(mech.strokes), "doses_missed": int(pr.doses_missed)},
            "sponge": pr.sponge.record(),
            "grease": {"removed_fraction": pr.grease.removed_fraction(), "residual": pr.grease.residual(),
                       "total0": float(pr.grease.total0)},
            "crumbs": pr.crumbs.counts(),
            "tools_clipped": dict(c.clips.engaged),
            "broken": dict(c.fragile.broken),
            "robot": robot,
            "held": sorted(c.grips.held()),
            "safety_ok": bool(c.safety.ok),
        }
        try:
            out["stage_score"] = float(c.score()["stage_score"])
        except Exception:  # noqa: BLE001
            pass
        return out

    # ------------------------------------------------------------------------------------------- end of episode
    def close(self) -> None:
        """Free the cameras' render contexts (once)."""
        if self._closed:
            return
        self._closed = True
        cams = getattr(self.cell, "cams", None)
        if cams is not None:
            try:
                cams.close()
            except Exception:  # noqa: BLE001
                pass

    def finalize(self) -> dict:
        """The episode's record: Cell.score() (stage_score in [0, 1], credit_by_part, safety_ok, violations, ...) plus
        the call counts. Ends the episode first when nothing else did."""
        if not self.done:
            self._end("finalized")
        try:
            rec = dict(self.cell.score())
            rec["score_error"] = None
        except Exception:  # noqa: BLE001 - an unscoreable episode scores 0, never a grader crash
            self.internal_errors += 1
            err = traceback.format_exc()
            self.log("[cell] score() failed:\n" + err)
            rec = {"stage_score": 0.0, "credit": 0.0, "credit_by_part": {}, "safety_ok": True, "violations": [],
                   "score_error": err.strip().splitlines()[-1][:300]}
        rec.update({
            "seed": self.seed,
            "privileged_mode": self.privileged,
            "nominal_layout": self.nominal,
            "done_reason": self.cell.done_reason,
            "calls": dict(self.calls),
            "illegal_call_count": int(self.illegal_calls),
            "forbidden_methods": list(self.forbidden_names),
            "bad_param_calls": int(self.bad_param_calls),
            "step_calls": int(self.step_calls),
            "internal_errors": int(self.internal_errors),
            "render_errors": int(self.render_errors),
            "steps_after_done": int(self.steps_after_done),
            "wall_time_s": float(self.wall_used or 0.0),
            "program_time_s": float(self.program_used or 0.0),
            "cell_work_s": float(self.cell.work_wall),
            "reference": self.reference_report,
        })
        rec.setdefault("sim_time_s", float(self.cell.sim_time))
        rec.setdefault("camera_frames", dict(self.cell.frames))
        rec.setdefault("camera_frames_total", int(self.cell.frames_total))
        self.close()
        return _plain(rec)
