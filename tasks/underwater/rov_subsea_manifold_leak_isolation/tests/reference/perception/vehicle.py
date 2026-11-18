"""Vehicle geometry a perception program may use: the datasheet-level numbers of the vehicle manual (numpy only).

Camera mount (entrance pupil, axes) and its tilt hinge, field of view, gripper TCP / jaw tip / gripper axis / opening
direction, and the gripper drawing as data (convex hulls of the visible gripper parts in the body frame).  Nothing
here is a site-specific (grader-side) quantity; no vehicle number is typed in this file: everything comes from a dict (the
program's copy of the manual) or from the two JSON files the cell side keeps (`from_spec_values`, `load_outline`).

Frames: body B = x forward, y left, z up (origin = the vehicle body origin of the manual's drawings); camera C =
OpenCV axes (x right, y down, z forward).  Tilt is up-positive (decision U8): rotation about the hinge axis given in
the manual (about -y_B).

    man = Manual.from_spec_values(path_spec_values, path_gripper_outline)     # cell / reference / checks
    man = Manual(dict(cam_pos_B=..., cam_quat_B=..., tilt_pivot_B=..., tilt_axis_B=..., hfov_deg=..., tcp_B=...,
                      tip_B=..., axis_B=..., open_dir_B=...), outline=parts)  # a program that typed the manual in
    p_B, R_Bc = man.camera_in_body(tilt_rad);  K = man.intrinsics(640, 480)
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

# OpenCV-style camera axes (x right, y down, z forward) expressed in the camera mount frame (x fwd, y left, z up)
M_SITE_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

REQUIRED = ("cam_pos_B", "cam_quat_B", "tilt_pivot_B", "tilt_axis_B", "hfov_deg", "tcp_B", "tip_B", "axis_B",
            "open_dir_B")


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_axis(axis, ang: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def euler_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll) (body -> nav, the AHRS convention of the vehicle manual)."""
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Dome port pinhole: f = (W/2) / tan(hfov/2), square pixels, principal point at the image centre (continuous
    pixel coordinates: pixel (i, j) centre at (i + 0.5, j + 0.5)) -- the panel standard's convention (panel_std)."""
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def load_outline(path: str, closed: bool = False, with_marker: bool = False) -> list:
    """The manual's gripper drawing as data (gripper_outline.json, schema gripper_outline/1): a list of (N, 3) convex
    hull point sets in the body frame (jaws open, or closed with closed=True); with_marker adds the marker block held
    in the jaws (box / capsule parts converted to corner point sets)."""
    with open(path, encoding="utf-8") as f:
        js = json.load(f)
    key = "parts_closed" if closed and "parts_closed" in js else "parts"
    parts = [np.asarray(p["points_B_m"], float) for p in js[key]]
    if with_marker and js.get("marker_held"):
        for p in js["marker_held"]["parts"]:
            if p.get("kind") == "box":
                parts.append(np.asarray(p["corners_B_m"], float))
            elif p.get("kind") == "capsule":
                a, b, r = np.asarray(p["a_B_m"], float), np.asarray(p["b_B_m"], float), float(p["radius_m"])
                pts = [c + np.array(o) * r for c in (a, b) for o in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                                                      (0, 0, 1), (0, 0, -1))]
                parts.append(np.asarray(pts))
    return parts


class Manual:
    """The vehicle manual's perception numbers (see the module docstring)."""

    def __init__(self, values: dict, outline: Optional[list] = None):
        miss = [k for k in REQUIRED if k not in values]
        if miss:
            raise ValueError(f"manual values missing: {miss}")
        self.v = {k: (np.asarray(v, float) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in values.items()}
        self.outline = outline or []
        a = self.axis_B
        self.yaw_offset_rad = math.atan2(a[1], a[0])          # the gripper axis is turned this far left of body +x
        self.tip_ahead_m = float((self.tip_B - self.tcp_B) @ a)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_spec_values(cls, spec_path: str, outline_path: Optional[str] = None, closed: bool = False,
                         with_marker: bool = False) -> "Manual":
        """From the digest's spec_values.json (cell / maintainer side) + the gripper drawing JSON."""
        with open(spec_path, encoding="utf-8") as f:
            S = json.load(f)["values"]
        sites = S["mission.sites"]["value"]
        art = S["uwkit.mission.articulation"]["value"]
        cam = S["sensor_specs"]["value"]["cam_front"]
        g = art["gripper"]
        vals = dict(cam_pos_B=sites["cam_front"]["pos_m"], cam_quat_B=sites["cam_front"]["quat_wxyz"],
                    tilt_pivot_B=art["camera_tilt"]["pivot_B_m"], tilt_axis_B=art["camera_tilt"]["axis_B"],
                    hfov_deg=float(cam["hfov_deg_air"]), tcp_B=g["tcp_B_m"], tip_B=g["jaw_tip_B_m"],
                    axis_B=g["gripper_axis_B"], open_dir_B=g["opening_dir_B"],
                    sonar_pos_B=sites["sonar_head"]["pos_m"], sonar_quat_B=sites["sonar_head"]["quat_wxyz"],
                    lights_B=[sites[f"light_{i}"]["pos_m"] for i in (1, 2, 3, 4) if f"light_{i}" in sites])
        outline = load_outline(outline_path, closed=closed, with_marker=with_marker) if outline_path else None
        return cls(vals, outline)

    # ------------------------------------------------------------------ attributes
    def __getattr__(self, k):
        try:
            return self.__dict__["v"][k]
        except KeyError as e:
            raise AttributeError(k) from e

    def as_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.v.items()}

    # ------------------------------------------------------------------ camera
    def camera_in_body(self, tilt_rad: float):
        """(p_B, R_B_cv): camera entrance pupil and OpenCV-axes rotation in the body frame at a tilt (up positive)."""
        Rt = rot_axis(self.tilt_axis_B, float(tilt_rad))
        piv = self.tilt_pivot_B
        p = piv + Rt @ (self.cam_pos_B - piv)
        return p, Rt @ quat_to_mat(self.cam_quat_B) @ M_SITE_CV

    def intrinsics(self, width: int = 640, height: int = 480) -> np.ndarray:
        return intrinsics(width, height, float(self.hfov_deg))

    def sonar_in_body(self):
        """(p_B, R_B_sonar): scanning sonar head position and frame (x = zero bearing, z up) in the body frame."""
        return self.sonar_pos_B, quat_to_mat(self.sonar_quat_B)
