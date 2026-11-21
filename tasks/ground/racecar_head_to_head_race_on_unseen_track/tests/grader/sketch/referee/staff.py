"""Where the staff put a stopped car.

The car goes beside the track at the station of its last clean position, along the track, with its side
`roadside_clearance_m` from the nearer wall face (the side of the last clean position), and with three independent
uniform errors: along the track and across it (+-`relocate_pos_err_m`) and in heading (+-`relocate_yaw_err_rad`).
The errors come from one numpy generator seeded with `truth.seeds.relocation` when the venue is built; every
relocation draws exactly three numbers in that order, whatever happens next, so the stream does not depend on the
search below.

Placement rules (the cell's decisions): the target station is pulled back to `line_clearance_m` short of the timing line or
a checkpoint that the along-track error would carry it across (a relocation never crosses a line); a candidate pose is
valid when all four wheel ground points are inside the corridor by `wheel_margin_m`, the car's envelope corners are
inside the corridor, and the envelope keeps `object_clearance_m` from every obstacle polygon given (boxes, slow cars).
Candidates: the nearer side, then the other side, then the same two `retreat_step_m`, 2 x `retreat_step_m`, ...
further back, up to `max_tries`; if none is valid, the centreline at the last clean station without errors.
"""

import math

import numpy as np

from sketch.examples import constants as K

from .geometry import car_corners, polygons_overlap


def _expand(poly, margin):
    """A convex polygon grown by `margin` (corners pushed out along the bisectors; conservative for rectangles)."""
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    out = []
    for x, y in poly:
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy) or 1.0
        out.append((x + dx / r * margin * math.sqrt(2.0), y + dy / r * margin * math.sqrt(2.0)))
    return out


class Relocator(object):
    def __init__(self, track, lines, config, params, seed):
        st = config["staff"]
        self.track = track
        self.line_s = [ln.s_m for ln in lines]
        self.pos_err = float(st["relocate_pos_err_m"])
        self.yaw_err = float(st["relocate_yaw_err_rad"])
        self.clearance = float(st["roadside_clearance_m"])
        p = params["relocation"]
        self.wheel_margin = float(p["wheel_margin_m"])
        self.obj_clear = float(p["object_clearance_m"])
        self.line_clear = float(p["line_clearance_m"])
        self.retreat = float(p["retreat_step_m"])
        self.max_tries = int(p["max_tries"])
        self.rng = np.random.default_rng(seed)
        self.half_w = 0.5 * K.ENVELOPE_M[1]
        self.log = []

    def _clamp(self, s_clean, s_target):
        """Pull s_target back so that no line lies between s_clean and it."""
        L = self.track.length
        delta = self.track.signed_gap(s_target, s_clean)
        for s_line in self.line_s:
            rel = self.track.signed_gap(s_line, s_clean)
            if delta > 0.0 and 0.0 < rel <= delta:
                delta = max(0.0, rel - self.line_clear)
            elif delta < 0.0 and delta <= rel < 0.0:
                delta = min(0.0, rel + self.line_clear)
        return (s_clean + delta) % L

    def _pose_at(self, s, side, e_d, e_yaw):
        x, y, h, wl, wr = self.track.at(s)
        if side == "left":
            d = wl - self.clearance - self.half_w
        else:
            d = -(wr - self.clearance - self.half_w)
        d += e_d
        rx, ry = x - d * math.sin(h), y + d * math.cos(h)            # where the reference point goes
        yaw = math.atan2(math.sin(h + e_yaw), math.cos(h + e_yaw))
        cx, cy = rx - K.REF_POINT_C[0] * math.cos(yaw), ry - K.REF_POINT_C[0] * math.sin(yaw)
        return [cx, cy, yaw]

    def valid(self, pose, obstacles=()):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        for wx, wy, _ in K.WHEEL_GROUND_C.values():
            if not self.track.inside1(x + c * wx - s * wy, y + s * wx + c * wy, self.wheel_margin)[0]:
                return False
        env = car_corners(pose)
        for px, py in env:
            if not self.track.inside1(px, py)[0]:
                return False
        for poly in obstacles:
            if polygons_overlap(env, _expand(poly, self.obj_clear)):
                return False
        return True

    def place(self, s_clean, d_clean, obstacles=()):
        """(pose, info): the relocated pose of the car frame C and how it was found."""
        e_s = float(self.rng.uniform(-self.pos_err, self.pos_err))
        e_d = float(self.rng.uniform(-self.pos_err, self.pos_err))
        e_yaw = float(self.rng.uniform(-self.yaw_err, self.yaw_err))
        s0 = self._clamp(s_clean, (s_clean + e_s) % self.track.length)
        near = "left" if d_clean >= 0.0 else "right"
        far = "right" if near == "left" else "left"
        tries = 0
        for k in range(self.max_tries):
            s = (s0 - k * self.retreat) % self.track.length
            for side in (near, far):
                tries += 1
                pose = self._pose_at(s, side, e_d, e_yaw)
                if self.valid(pose, obstacles):
                    info = {"s": s, "side": side, "errors": [e_s, e_d, e_yaw], "tries": tries, "fallback": False}
                    self.log.append(info)
                    return pose, info
        x, y, h, _, _ = self.track.at(s_clean)
        pose = [x - K.REF_POINT_C[0] * math.cos(h), y - K.REF_POINT_C[0] * math.sin(h), h]
        info = {"s": s_clean, "side": "centre", "errors": [e_s, e_d, e_yaw], "tries": tries, "fallback": True}
        self.log.append(info)
        return pose, info
