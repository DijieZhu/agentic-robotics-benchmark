"""api.py -- the program's calls of the caretaking cell (the robot API, version 1.1.0).

``CellAPI(cell).call(method, params)`` carries out one whitelisted call and returns its JSON-able answer: action calls
``{code, t, ...}``, query calls a data object.  Arguments are named as in the robot API; a missing, unknown or malformed
argument answers INVALID_ARGS and changes nothing (no budget used).  On top of the task firmware (``firmware``:
motion, arm, gripper, capture, dock, lamps, status, pose) this module adds what the cell owns:

* time: step(dt) clamps dt to (0, cell.step_max_dt_s]; when the site's simulated time is used up during a call, that
  call and every later action call answer TIME_UP (query calls still answer) and the run ends; 'used up' = less than
  half a physics step left (a smaller remainder cannot be stepped: step() would answer OK without time passing);
* budgets (budgets.py): camera frames by class, capture calls (the firmware's hook), the first valid report;
* camera(name): a frame id unique on the site, the delivered pinhole model (spacekit's configured K at the published
  image size, as spacekit.cameras.RobotCameras delivers), the renderer's pixels when a renderer is plugged in
  (``NullRenderer`` delivers none: sensors_render.py plugs the real one), and the evidence predicate's inputs of every
  return inlet near the camera at the TRUE pose of the shot (the evidence rule D): the inlet centre in the image, distance,
  incidence, the opening's long side in pixels, the occluded share of 20 rays to the inlet frame (5 per side; the
  occluders are the station, the robot as the frames show it -- the maker's mesh, nothing nearer than the frames' near
  clip along the camera's optical axis (firmware ``RobotOccluder``) -- and every item that is not on that
  inlet; the inlet's own part and the items on it are not), the blocker's covered fraction of the inlet at the shot;
* report(): validation (inlet of the map, register reference of this site, finite onset, a frame id issued on this
  site before the call; anything else INVALID_ARGS, not counted), only the first valid one is recorded;
* station_telemetry() / station_request(): the station service;
* mission(): the site package's mission members (work order, the 72 h log, the configuration record, the advisory
  log, the register), the as-surveyed map, the start charge and berth.

The firmware calls are the task firmware's, the codes are the robot API's (robot_api.md, return codes), the evidence
geometry is ``evidence_ok``'s, the occlusion rays are the firmware's RayTester (whose robot occluder is the maker's
mesh since firmware 1.1.0).
"""
from __future__ import annotations

import base64
import math

import mujoco
import numpy as np

WHITELIST = ("arm", "arm_deploy", "arm_stow", "camera", "capture", "dock", "fly_rel", "fly_to", "gripper", "hold", "lamp",
             "mission", "pose", "report", "set_mode", "station_request", "station_telemetry", "status", "step", "stop",
             "undock")
QUERY_CALLS = ("status", "pose", "mission", "station_telemetry")
ACTION_CALLS = tuple(c for c in WHITELIST if c not in QUERY_CALLS)
LOGGED_EVERY_CALL = ("step", "status", "pose", "station_telemetry", "mission")    # counted in the metrics, not listed
CAMERAS = {"FIC": ("forward_imaging", "cam_fwd_img"), "FWC": ("forward_wide", "cam_fwd_wide"),
           "AWC": ("aft_wide", "cam_aft_wide"), "FDC": ("forward_depth", "cam_fwd_depth"),
           "ADC": ("aft_depth", "cam_aft_depth")}
EVIDENCE_CAMERAS = ("FIC", "FWC")
EVIDENCE_RANGE_M = 3.0        # inlets farther than this from the camera get no evidence entry (the rule's line is 2 m)
SIDE_POINTS = (-0.8, -0.4, 0.0, 0.4, 0.8)     # 5 ray targets per side of the inlet frame

SIG = {   # call -> [(argument, kind, default)]; kind: num, num3, num4, num3opt, bool, str, int; default REQ = required
    "step": [("dt", "num", "REQ")],
    "status": [], "pose": [], "mission": [], "station_telemetry": [], "stop": [], "hold": [], "undock": [],
    "set_mode": [("mode", "str", "REQ")],
    "fly_to": [("pos", "num3", "REQ"), ("quat", "num4", "REQ"), ("face_forward", "bool", True), ("max_speed", "numopt", None)],
    "fly_rel": [("fiducial_id", "str", "REQ"), ("pos", "num3", "REQ"), ("quat", "num4", "REQ"), ("max_speed", "numopt", None)],
    "arm_deploy": [("wait", "bool", False)], "arm_stow": [("wait", "bool", False)],
    "arm": [("tilt_deg", "num", "REQ"), ("pan_deg", "num", "REQ"), ("wait", "bool", False)],
    "gripper": [("open", "bool", "REQ")],
    "capture": [("bar_pos", "num3", "REQ"), ("bar_axis", "num3", "REQ"), ("palm_dir", "num3opt", None)],
    "dock": [("berth", "int", "REQ")],
    "camera": [("name", "str", "REQ")],
    "lamp": [("which", "str", "REQ"), ("on", "bool", "REQ")],
    "station_request": [("kind", "str", "REQ"), ("target", "str", "REQ"), ("value", "str", "REQ")],
    "report": [("inlet_id", "any", "REQ"), ("item_ref", "any", "REQ"), ("onset_time", "any", "REQ"), ("frame_id", "any", "REQ")],
}
NUM_MAX = 1.0e6


class BadArgs(Exception):
    """An argument fails its check (the text is the reply's detail)."""


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and abs(float(v)) <= NUM_MAX


def _vec(v, n):
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(_is_num(x) for x in v):
        return None
    return [float(x) for x in v]


def check_args(method: str, params) -> dict:
    """Named arguments of a call after their check (raises BadArgs)."""
    if not isinstance(params, dict):
        raise BadArgs("params must be an object")
    sig = SIG[method]
    names = {a for a, _, _ in sig}
    for k in params:
        if k not in names:
            raise BadArgs(f"{method} has no argument {str(k)[:64]!r}")
    out = {}
    for name, kind, default in sig:
        if name not in params:
            if default == "REQ":
                raise BadArgs(f"{method} needs {name}")
            out[name] = default
            continue
        v = params[name]
        if kind == "num":
            if not _is_num(v):
                raise BadArgs(f"{name} must be a finite number")
            out[name] = float(v)
        elif kind == "numopt":
            if v is not None and not _is_num(v):
                raise BadArgs(f"{name} must be a finite number or null")
            out[name] = None if v is None else float(v)
        elif kind in ("num3", "num4", "num3opt"):
            if kind == "num3opt" and v is None:
                out[name] = None
                continue
            vv = _vec(v, 4 if kind == "num4" else 3)
            if vv is None:
                raise BadArgs(f"{name} must be a list of {4 if kind == 'num4' else 3} finite numbers")
            out[name] = vv
        elif kind == "bool":
            if not isinstance(v, bool):
                raise BadArgs(f"{name} must be true or false")
            out[name] = v
        elif kind == "str":
            if not isinstance(v, str) or len(v) > 64:
                raise BadArgs(f"{name} must be a string of at most 64 characters")
            out[name] = v
        elif kind == "int":
            if isinstance(v, bool) or not isinstance(v, int):
                raise BadArgs(f"{name} must be an integer")
            out[name] = int(v)
        else:
            out[name] = v
    return out


# ------------------------------------------------------------------------------------------------ frames
def frame_model(camera: str) -> dict:
    """The delivered pinhole model of a camera: spacekit's configured K at the published image size
    (spacekit.cameras.RobotCameras: ``set_intrinsics(model, cam, K, IMAGE_SIZE[role])``)."""
    from spacekit.cameras import IMAGE_SIZE
    from spacekit.sensors import camera_model
    role, cam = CAMERAS[camera]
    K = np.asarray(camera_model(role)["K"], float)
    W, H = IMAGE_SIZE[role]
    return dict(role=role, cam=cam, K=K, width=int(W), height=int(H), depth=camera in ("FDC", "ADC"))


class NullRenderer:
    """The renderer stub: no pixels (frame ids and the evidence inputs only).  The real renderer (the camera
    frames, tests/cabincell/sensors_render.py) has the same four methods; ``render`` returns {'image': uint8 H x W x 3} for a
    colour camera or {'depth': float32 H x W metres (0 = no return), 'amplitude': uint16 H x W} for a depth camera."""

    name = "null"

    def render(self, camera: str, model, data) -> dict | None:
        return None

    def set_lighting(self, module: str, state: str) -> None:
        return None

    def set_lamp(self, which: str, on: bool) -> None:
        return None

    def close(self) -> None:
        return None


def _png_b64(arr) -> str:
    import cv2
    a = np.ascontiguousarray(arr)
    if a.ndim == 3:
        a = np.ascontiguousarray(a[..., ::-1])                  # cv2 writes BGR
    ok, buf = cv2.imencode(".png", a, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_pixels(px: dict | None) -> dict:
    """Wire encoding of a renderer's frame (cabinkit decodes it): colour -> 8-bit RGB PNG; depth -> 16-bit PNG in
    millimetres (0 = no return); amplitude -> 16-bit PNG."""
    out = {}
    if not px:
        return out
    if px.get("image") is not None:
        out["image_png_b64"] = _png_b64(np.asarray(px["image"], np.uint8))
    if px.get("depth") is not None:
        z = np.asarray(px["depth"], np.float64)
        ok = np.isfinite(z) & (z > 0)
        mm = np.where(ok, np.clip(np.round(np.where(ok, z, 0.0) * 1000.0), 1, 65535), 0).astype(np.uint16)
        out["depth_png_b64"] = _png_b64(mm)
        out["depth_unit_m"] = 0.001
    if px.get("amplitude") is not None:
        out["amplitude_png_b64"] = _png_b64(np.asarray(px["amplitude"], np.uint16))
    return out


# ------------------------------------------------------------------------------------------------ the calls
class CellAPI:
    """The calls of one run (module doc).  ``cell`` is the server's Cell (scene, firmware, service, recorder)."""

    def __init__(self, cell):
        self.c = cell
        self.tf = cell.tf
        self.frames = []                   # the record's frames
        self._frame_ids = set()
        self._next_frame = 1
        self.report = None
        self.capture_hints = []            # per capture call: hint error to the nearest bar (true body frame)
        self._mission = None
        self.frame_models = {k: frame_model(k) for k in CAMERAS}
        m = cell.model
        self.cam_ids = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, v["cam"]) for k, v in self.frame_models.items()}
        feats = cell.tf.features_scene
        self.inlet_labels = {i["label"]: i["id"] for i in feats["v02"]["inlets"]}
        self.inlet_geo = {}
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        for i in feats["v02"]["inlets"]:
            F = cell.world.faces[i["id"]]
            n = np.asarray(F["n"], float)
            a = np.asarray(F["t1"], float)
            b = np.cross(n, a)
            fh = (0.5 * float(i["frame_m"][0]), 0.5 * float(i["frame_m"][1]))
            pts = []
            for s in SIDE_POINTS:
                for sb in (-1.0, 1.0):
                    pts.append(F["c"] + a * s * fh[0] + b * sb * fh[1])
                    pts.append(F["c"] + b * s * fh[1] * 0.999 + a * sb * fh[0])
            self.inlet_geo[i["id"]] = dict(c=np.asarray(F["c"], float), n=n, a=a, long_m=2.0 * float(F["half"][0]),
                                           pts=np.array(pts), own=frozenset(g for g, nm in enumerate(names) if nm.startswith(i["id"] + "_")))
        gb = np.asarray(m.geom_bodyid)
        self.item_geoms = {k: frozenset(np.flatnonzero(np.isin(gb, it["bodies"])).tolist()) for k, it in cell.world.items.items()}
        self.register_refs = {e.get("ref") for e in ((cell.site.get("register") or {}).get("entries") or []) if isinstance(e, dict)}

    # -- dispatch -----------------------------------------------------------------------------------------------
    def call(self, method: str, params) -> dict:
        try:
            a = check_args(method, params)
        except BadArgs as e:
            return dict(code="INVALID_ARGS", detail=str(e)[:200], t=self.t())
        return getattr(self, "c_" + method)(**a)

    def t(self) -> float:
        return round(float(self.tf.t()), 4)

    def _act(self, code, **kw) -> dict:
        """An action call's answer; a call that used up the site's time answers TIME_UP and ends the run."""
        if self.c.out_of_time() and code != "INVALID_ARGS":
            code = "TIME_UP"
            self.c.mark_time_up()
        out = dict(code=code, t=self.t())
        out.update(kw)
        return out

    def _over(self) -> dict | None:
        if self.c.out_of_time():
            self.c.mark_time_up()
            return dict(code="TIME_UP", t=self.t())
        return None

    # -- time ---------------------------------------------------------------------------------------------------
    def c_step(self, dt):
        if not (dt > 0):
            return dict(code="INVALID_ARGS", detail="dt must be > 0", t=self.t())
        o = self._over()
        if o:
            return o
        dt = min(float(dt), float(self.c.budgets.step_max_dt_s))
        return self._act(self.tf.step(dt))

    # -- queries ------------------------------------------------------------------------------------------------
    def c_status(self):
        return self.tf.status()

    def c_pose(self):
        return self.tf.pose()

    def c_station_telemetry(self):
        return self.c.svc.telemetry()

    def c_mission(self):
        if self._mission is None:
            self._mission = self.c.mission_document()
        return self._mission

    # -- motion and arm (the firmware's calls) ------------------------------------------------------------------
    def _fw(self, fn, *args, **kw) -> dict:
        o = self._over()
        if o:
            return o
        r = fn(*args, **kw)
        if isinstance(r, dict):
            r = dict(r)
            code = r.pop("code")
            return self._act(code, **r)
        return self._act(r)

    def c_set_mode(self, mode):
        return self._fw(self.tf.set_mode, mode)

    def c_fly_to(self, pos, quat, face_forward, max_speed):
        return self._fw(self.tf.fly_to, pos, quat, face_forward=face_forward, max_speed=max_speed)

    def c_fly_rel(self, fiducial_id, pos, quat, max_speed):
        return self._fw(self.tf.fly_rel, fiducial_id, pos, quat, max_speed=max_speed)

    def c_stop(self):
        return self._fw(self.tf.stop)

    def c_hold(self):
        return self._fw(self.tf.hold)

    def c_arm_deploy(self, wait):
        return self._fw(self.tf.arm_deploy, wait=wait)

    def c_arm_stow(self, wait):
        return self._fw(self.tf.arm_stow, wait=wait)

    def c_arm(self, tilt_deg, pan_deg, wait):
        return self._fw(self.tf.arm, tilt_deg, pan_deg, wait=wait)

    def c_gripper(self, open):  # noqa: A002 (the robot API's name)
        return self._fw(self.tf.gripper, open)

    def c_capture(self, bar_pos, bar_axis, palm_dir):
        o = self._over()
        if o:
            return o
        hint = self._hint_error(bar_pos, bar_axis)
        n0 = self.c.budgets.n_capture
        r = self.tf.capture(bar_pos, bar_axis, palm_dir)
        if self.c.budgets.n_capture > n0 and hint is not None:
            hint["code"] = r
            self.capture_hints.append(hint)
        return self._act(r)

    def _hint_error(self, bar_pos, bar_axis) -> dict | None:
        """The hint against the nearest bar of the scene (true body frame at the call): position error [m] and axis
        error [deg] (a diagnostic metric; never an input of the firmware)."""
        d, b = self.c.data, self.c.body
        R = d.xmat[b].reshape(3, 3)
        p = d.xpos[b] + R @ np.asarray(bar_pos, float)
        ax = R @ np.asarray(bar_axis, float)
        nax = float(np.linalg.norm(ax))
        if nax <= 0:
            return None
        ax /= nax
        best = None
        for bar in self.tf.detector.bars:
            c, a, _ = bar.pose(d)
            e = float(np.linalg.norm(np.asarray(c) - p))
            if best is None or e < best[0]:
                ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(np.asarray(a, float), ax))))))
                best = (e, ang, bar.name)
        if best is None:
            return None
        return dict(t=self.t(), err_m=best[0], err_deg=best[1], bar=best[2])

    def c_dock(self, berth):
        o = self._over()
        if o:
            return o
        r = self.tf.dock(berth)
        if r == "DOCKED":
            self.c.on_docked()
        return self._act(r)

    def c_undock(self):
        return self._fw(self.tf.undock)

    def c_lamp(self, which, on):
        o = self._over()
        if o:
            return o
        r = self.tf.lamp(which, on)
        if r == "OK":
            self.c.renderer.set_lamp(which, on)
        return self._act(r)

    # -- station ------------------------------------------------------------------------------------------------
    def c_station_request(self, kind, target, value):
        o = self._over()
        if o:
            return o
        rep = self.c.svc.request(kind, target, value)
        r = dict(rep)
        code = r.pop("code")
        return self._act(code, **r)

    # -- camera -------------------------------------------------------------------------------------------------
    def c_camera(self, name):
        if name not in CAMERAS:
            return dict(code="INVALID_ARGS", detail=f"camera must be one of {sorted(CAMERAS)}", t=self.t())
        o = self._over()
        if o:
            return o
        if not self.c.budgets.take_frame(name):
            return self._act("REFUSED_BUDGET")
        d = self.c.data
        mujoco.mj_kinematics(self.c.model, d)
        mujoco.mj_camlight(self.c.model, d)
        fid = self._next_frame
        self._next_frame += 1
        fm = self.frame_models[name]
        ev = self.evidence(name) if name in EVIDENCE_CAMERAS else {}
        self.frames.append(dict(frame_id=fid, camera=name, t=self.t(), evidence=ev))
        self._frame_ids.add(fid)
        K = fm["K"]
        out = dict(frame_id=fid, camera=name, width=fm["width"], height=fm["height"],
                   intrinsics=dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2])))
        px = self.c.renderer.render(name, self.c.model, d)
        out.update(encode_pixels(px))
        return self._act("OK", **out)

    def evidence(self, camera: str) -> dict:
        """The evidence predicate's inputs of a frame at this instant, true pose (module doc)."""
        c, d = self.c, self.c.data
        fm = self.frame_models[camera]
        cid = self.cam_ids[camera]
        from spacekit.sensors import mujoco_camera_to_optical
        cp = d.cam_xpos[cid].copy()
        Ro = mujoco_camera_to_optical(d.cam_xmat[cid].reshape(3, 3))
        K, W, H = fm["K"], fm["width"], fm["height"]
        seal = c.world.seal_by_face(d, c.world.mp["cover_band"])
        tester = self.tf.fid_loc.gate["tester"]
        axis = Ro[:, 2]                                   # the optical axis: the frames' near clip is along it
        area = float(c.world.mp["opening_area_m2"])
        out = {}
        for iid, G in self.inlet_geo.items():
            v = G["c"] - cp
            dist = float(np.linalg.norm(v))
            if dist > EVIDENCE_RANGE_M:
                continue
            P = v @ Ro
            if P[2] <= 1e-6:
                continue
            u = K[0, 0] * P[0] / P[2] + K[0, 2]
            w = K[1, 1] * P[1] / P[2] + K[1, 2]
            inc = math.degrees(math.acos(max(-1.0, min(1.0, float(G["n"] @ (-v / dist))))))
            ends = []
            for s in (-0.5, 0.5):
                Q = (G["c"] + G["a"] * s * G["long_m"] - cp) @ Ro
                if Q[2] <= 1e-6:
                    ends = None
                    break
                ends.append(np.array([K[0, 0] * Q[0] / Q[2] + K[0, 2], K[1, 1] * Q[1] / Q[2] + K[1, 2]]))
            long_px = float(np.linalg.norm(ends[1] - ends[0])) if ends else 0.0
            on_it = seal.get(iid) or {}
            skip = set(G["own"])
            for k, a in on_it.items():
                if a > 0.0:
                    skip |= self.item_geoms[k]
            blocked = sum(0 if tester.clear(d, cp, p, skip_geoms=skip, axis=axis) else 1 for p in G["pts"])
            out[iid] = dict(in_frame=bool(0.0 <= u < W and 0.0 <= w < H), dist_m=round(dist, 6), incidence_deg=round(inc, 4),
                            long_px=round(long_px, 3), occluded=round(blocked / float(len(G["pts"])), 4),
                            blocker_cover=round(min(1.0, float(on_it.get("blocker", 0.0)) / area), 6))
        return out

    # -- report -------------------------------------------------------------------------------------------------
    def c_report(self, inlet_id, item_ref, onset_time, frame_id):
        o = self._over()
        if o:
            return o
        from cabincell import predicates as P
        bad = None
        if not isinstance(inlet_id, str) or P.inlet_from_report(inlet_id) not in set(self.inlet_labels.values()):
            bad = "inlet_id is not an inlet of the map (RI-<module>-<n>)"
        elif not isinstance(item_ref, str) or item_ref not in self.register_refs:
            bad = "item_ref is not a reference of this site's register (LI-nn)"
        elif not _is_num(onset_time):
            bad = "onset_time must be a finite number (minutes on the log's time axis)"
        elif isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id not in self._frame_ids:
            bad = "frame_id is not a frame issued on this site"
        if bad is not None:
            return self._act("INVALID_ARGS", detail=bad)
        if self.report is not None:
            return self._act("ALREADY_REPORTED")
        self.report = dict(t=self.t(), inlet=inlet_id, item_ref=item_ref, onset_min=float(onset_time), frame_id=int(frame_id))
        return self._act("RECORDED")

    # -- metrics ------------------------------------------------------------------------------------------------
    def hint_metrics(self) -> dict:
        out = {}
        ok = [h for h in self.capture_hints if h.get("code") == "DONE"]
        for tag, rows in (("", ok), ("_all", self.capture_hints)):
            if rows:
                out[f"capture_hint_err_m_max{tag}"] = max(h["err_m"] for h in rows)
                out[f"capture_hint_err_deg_max{tag}"] = max(h["err_deg"] for h in rows)
        return out
