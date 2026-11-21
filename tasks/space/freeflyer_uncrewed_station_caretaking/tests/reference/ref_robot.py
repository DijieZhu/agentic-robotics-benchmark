"""ref_robot.py -- the reference program's robot: the public calls of the cell (the robot API) through the cell's
own call boundary ``server.handle(method, params)`` with the wire's JSON round trip, exactly what an agent's program
gets through cabinkit (arguments by name, action calls {code, t, ...}, query calls a data object, frames decoded).

Nothing here reads the cell's state: every call is counted, budgeted, logged and recorded by the cell like a
program's.  The proxy keeps its own call log (method, code, mission time) for the reference's summary.
"""
from __future__ import annotations

import base64
import json

ERROR_KEYS = frozenset(("error", "detail"))


class CellError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        RuntimeError.__init__(self, code + ((": " + detail) if detail else ""))
        self.code, self.detail = code, detail


def _plain(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, float):
        return float(v)
    return v


def decode_frame(reply: dict) -> dict:
    """A camera reply with its pixels as numpy arrays when it carries any (cabinkit.client.decode_frame's encoding:
    8-bit RGB PNG, 16-bit depth PNG in millimetres, 16-bit amplitude PNG)."""
    out = {k: v for k, v in reply.items() if not str(k).endswith("_b64")}
    if not any(str(k).endswith("_b64") for k in reply):
        return out
    import cv2
    import numpy as np

    def png(b64):
        raw = base64.b64decode(b64)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        return img[..., ::-1].copy() if (img is not None and img.ndim == 3) else img
    if reply.get("image_png_b64"):
        out["image"] = png(reply["image_png_b64"])[..., :3]
    if reply.get("depth_png_b64"):
        out["depth"] = png(reply["depth_png_b64"]).astype(np.float32) * np.float32(reply.get("depth_unit_m", 0.001))
    if reply.get("amplitude_png_b64"):
        out["amplitude"] = png(reply["amplitude_png_b64"]).astype(np.uint16)
    return out


class Robot:
    """The public calls (the robot API) on ``handle`` (module doc)."""

    def __init__(self, handle):
        self._handle = handle
        self.log = []
        self.n_calls = 0

    def call(self, method: str, **params):
        req = json.loads(json.dumps(_plain(params)))
        res = self._handle(method, req)
        res = json.loads(json.dumps(res, default=_plain))
        self.n_calls += 1
        if isinstance(res, dict) and "error" in res and set(res) <= ERROR_KEYS:
            self.log.append(dict(call=method, error=res.get("error")))
            raise CellError(str(res["error"]), str(res.get("detail") or ""))
        if isinstance(res, dict) and "code" in res:
            self.log.append(dict(call=method, code=res["code"], t=res.get("t")))
        return res

    # -- time and queries
    def step(self, dt):
        return self.call("step", dt=float(dt))

    def status(self):
        return self.call("status")

    def pose(self):
        return self.call("pose")

    def mission(self):
        return self.call("mission")

    def station_telemetry(self):
        return self.call("station_telemetry")

    # -- motion
    def set_mode(self, mode):
        return self.call("set_mode", mode=mode)

    def fly_to(self, pos, quat, face_forward=False, max_speed=None):
        return self.call("fly_to", pos=list(map(float, pos)), quat=list(map(float, quat)), face_forward=bool(face_forward),
                         max_speed=(None if max_speed is None else float(max_speed)))

    def fly_rel(self, fiducial_id, pos, quat, max_speed=None):
        return self.call("fly_rel", fiducial_id=fiducial_id, pos=list(map(float, pos)), quat=list(map(float, quat)),
                         max_speed=(None if max_speed is None else float(max_speed)))

    def stop(self):
        return self.call("stop")

    def hold(self):
        return self.call("hold")

    def dock(self, berth):
        return self.call("dock", berth=int(berth))

    def undock(self):
        return self.call("undock")

    # -- arm and gripper
    def arm_deploy(self, wait=False):
        return self.call("arm_deploy", wait=bool(wait))

    def arm_stow(self, wait=False):
        return self.call("arm_stow", wait=bool(wait))

    def arm(self, tilt_deg, pan_deg, wait=False):
        return self.call("arm", tilt_deg=float(tilt_deg), pan_deg=float(pan_deg), wait=bool(wait))

    def gripper(self, open):  # noqa: A002 (the API's argument name)
        return self.call("gripper", open=bool(open))

    def capture(self, bar_pos, bar_axis, palm_dir=None):
        return self.call("capture", bar_pos=list(map(float, bar_pos)), bar_axis=list(map(float, bar_axis)),
                         palm_dir=(None if palm_dir is None else list(map(float, palm_dir))))

    # -- sensors, lamps, station, report
    def camera(self, name):
        return decode_frame(self.call("camera", name=name))

    def lamp(self, which, on):
        return self.call("lamp", which=which, on=bool(on))

    def station_request(self, kind, target, value):
        return self.call("station_request", kind=kind, target=target, value=value)

    def report(self, inlet_id, item_ref, onset_time, frame_id):
        return self.call("report", inlet_id=inlet_id, item_ref=item_ref, onset_time=float(onset_time), frame_id=int(frame_id))
