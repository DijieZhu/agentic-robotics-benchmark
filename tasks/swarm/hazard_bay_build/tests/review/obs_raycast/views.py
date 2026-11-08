# Moved in from the round-4 observability study (arb-eval/pr31r4-sensors/obs_map/views.py, map-observability.md section 7) for
# check_observability.py --raycast: the pxr / numpy cross-check of the renderer's measure. Unchanged but this header.
"""Specific views for map-observability.md: what the reference fleet's own cameras see at the start, at the valve and
over the pan, on given days; the dial, the lever, the pan and the bucket in pixels.

    python -I -B views.py <geom.json> <days.json ...> --out <views.json>
"""
import argparse
import json
import math

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visibility as V  # noqa: E402

# the reference fleet's cameras, relative to the root body (truth.json fleet group, pos_in_root / look_in_root)
UGV_MAST = dict(model="CAM-RGB-M", pos=(-0.16, 0.14, 0.8534), look=(0.4623, -0.8695, 0.1736))  # Jackal base_link: x fwd, y left
UGV_ROOT = np.array([-2.8, 2.2, 0.0635])
UAV_CAMS = {"Thermal": dict(model="THERM-C", pos=(0.0, 0.14, -0.0265), look=(0.0, 0.6428, -0.766)),  # hex root: y fwd (heading +y)
            "Camera": dict(model="CAM-RGB-M", pos=(0.0, 0.14, 0.0235), look=(0.0, 0.9063, -0.4226)),
            "ThermalDown": dict(model="THERM-A", pos=(0.06, 0.0, -0.0615), look=(0.0, 0.0, -1.0)),
            "CameraDown": dict(model="CAM-RGB-W", pos=(-0.06, 0.0, -0.0615), look=(0.0, 0.0, -1.0))}


def ugv_frame(root_xy, heading_deg):
    """World axes of the Jackal's base_link at a heading (0 = facing +y, counter-clockwise positive): columns x fwd,
    y left, z up."""
    p = math.radians(heading_deg)
    fwd = np.array([-math.sin(p), math.cos(p), 0.0])
    left = np.array([-math.cos(p), -math.sin(p), 0.0])
    return np.stack([fwd, left, np.array([0, 0, 1.0])], axis=1)


def cam_world(root, R, cam):
    C = root + R @ np.array(cam["pos"])
    d = R @ np.array(cam["look"])
    d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) - d[2] * d
    if np.linalg.norm(up) < 1e-6:  # (straight down: image up = the robot's forward)
        up = R @ np.array([0, 1.0, 0]) if abs(R[2, 2]) > 0.5 and abs(d[2]) > 0.99 else up
    up /= np.linalg.norm(up)
    r = np.cross(d, up)
    return C, d, up, r


def find(shapes, key):
    return [s for s in shapes if key in s.name]


def landmarks(occ):
    keys = {"dial": "Gauge/Dial", "lever_bar": "Handle/bar", "lever_grip": "Handle/grip", "valve_block": "ValveBody/block",
            "support_0": "support_0", "support_1": "support_1", "support_2": "support_2", "support_3": "support_3",
            "crate_wall_s": "Crate/wall_s", "tableA_top": "Table_A/top", "tableB_top": "Table_B/top", "tank_wall_n": "Tank/wall_n",
            "bucket": "Bucket/Body/shell", "pan": "FirePan/pan", "pipe": "Pipe/main"}
    out = {}
    for k, v in keys.items():
        f = find(occ, v)
        if f:
            out[k] = f[0]
    return out


def start_views(day, occ):
    dx, dy, dpsi = day["ugv"]
    root = UGV_ROOT + np.array([dx, dy, 0.0])
    R = ugv_frame(root[:2], dpsi)
    C, d, u, r = cam_world(root, R, UGV_MAST)
    res = {}
    for k, s in landmarks(occ).items():
        v = V.view(C, d, u, r, "CAM-RGB-M", s, occ, n=6)
        res[k] = {kk: (round(vv, 3) if isinstance(vv, float) and math.isfinite(vv) else vv) for kk, vv in v.items()}
    return dict(camera=[round(float(x), 3) for x in C], look=[round(float(x), 3) for x in d], views=res)


def dial_needle_px(C, model="CAM-RGB-M"):
    """The dial (0.10 m) and the needle (4 mm wide, 70 mm long) across, in px, from C; the angle between the line of
    sight and the dial's normal (-y)."""
    dial = np.array([-0.6, 3.22, 1.2])
    v = dial - C
    dist = float(np.linalg.norm(v))
    f = V.fpx(model)
    ang = math.degrees(math.acos(abs(v[1]) / dist))
    return dict(dist=round(dist, 3), dial_px=round(0.10 / dist * f, 1), dial_px_foreshortened=round(0.10 * math.cos(math.radians(ang)) / dist * f, 1),
                needle_w_px=round(0.004 / dist * f, 2), needle_len_px=round(0.07 / dist * f, 1), off_normal_deg=round(ang, 1))


def valve_views(day, bay):
    out = {}
    for deg in (float(day["lever_deg"]), 85.0, 90.0):
        occ, _ = V.day_scene(bay, dict(day, lever_deg=deg))
        bar, grip = find(occ, "Handle/bar")[0], find(occ, "Handle/grip")[0]
        for label, root_xy, head in (("valve_stand_h-8", (-1.24, 2.61), -8.4), ("valve_stand_h60", (-1.24, 2.61), 60.0),
                                     ("parking_bay_start", None, None)):
            if root_xy is None:
                dx, dy, dpsi = day["ugv"]
                root = UGV_ROOT + np.array([dx, dy, 0.0])
                head = dpsi
            else:
                root = np.array([root_xy[0], root_xy[1], 0.0635])
            R = ugv_frame(root[:2], head)
            C, d, u, r = cam_world(root, R, UGV_MAST)
            vb = V.view(C, d, u, r, "CAM-RGB-M", bar, occ)
            vg = V.view(C, d, u, r, "CAM-RGB-M", grip, occ)
            # the best a camera at the mast's height on the white square can do, aimed at the lever (look-at)
            Cl = np.array([root[0], root[1], 0.92])
            dl, ul, rl = V.look_cam(Cl, bar.c)
            vl = V.view(Cl, dl, ul, rl, "CAM-RGB-M", bar, occ)
            out[f"{label}@{deg:.1f}"] = dict(bar=_r(vb), grip=_r(vg), bar_lookat=_r(vl))
        dxy = None
    return out


def _r(v):
    return {k: (round(x, 3) if isinstance(x, float) and math.isfinite(x) else x) for k, x in v.items()}


def pan_views(day, occ):
    pan = find(occ, "FirePan/pan")[0]
    out = {}
    hovers = {"approach_hover": np.array([2.60, -0.40, 2.50]), "over_pan_2.5": np.array([pan.c[0], pan.c[1] - 0.8, 2.5])}
    R = np.eye(3)  # (the flying robots are level, heading +y: root frame = world axes)
    for hk, H in hovers.items():
        for ck in ("Thermal", "ThermalDown"):
            C, d, u, r = cam_world(H, R, UAV_CAMS[ck])
            v = V.view(C, d, u, r, UAV_CAMS[ck]["model"], pan, occ, n=8)
            out[f"{hk}/{ck}"] = _r(v)
    # from the three pads at rest (the robots' cameras 0.45-0.5 m above the floor), the forward thermal camera
    for pk, (px, py) in {"Pad_1": (0.0, -1.5), "Pad_2": (-0.95, -3.15), "Pad_3": (0.95, -3.15)}.items():
        H = np.array([px, py, 0.479])
        C, d, u, r = cam_world(H, R, UAV_CAMS["Thermal"])
        out[f"{pk}_rest/Thermal"] = _r(V.view(C, d, u, r, "THERM-C", pan, occ, n=8))
        C, d, u, r = cam_world(H, R, UAV_CAMS["Camera"])
        out[f"{pk}_rest/Camera"] = _r(V.view(C, d, u, r, "CAM-RGB-M", pan, occ, n=8))
    return out


def bucket_views(occ):
    """The bucket hanging in the reference formation (hooks on a 0.78 m circle, ropes 1.5 m to rings 0.14 m from the
    axis): each robot's wide down camera; the bucket lifted so its bottom is at 1.0 m."""
    shell = find(occ, "Bucket/Body/shell")[0]
    bc = np.array([0.0, -1.0, 1.0 + 0.125])
    b = V.Cyl("bucket", bc, (0, 0, 1), 0.12, 0.125)
    out = {}
    vert = math.sqrt(1.5 ** 2 - (0.78 - 0.14) ** 2)
    for name, bear in (("UAV_1", 90.0), ("UAV_2", 210.0), ("UAV_3", 330.0)):
        hook = bc + np.array([0.78 * math.cos(math.radians(bear)), 0.78 * math.sin(math.radians(bear)), 0.125 + vert])
        root = hook - np.array([0.0, -0.104, -0.08])
        C, d, u, r = cam_world(root, np.eye(3), UAV_CAMS["CameraDown"])
        v = V.view(C, d, u, r, "CAM-RGB-W", b, [b], n=8)
        out[name] = _r(v)
        out[name]["root"] = [round(float(x), 3) for x in root]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("geom")
    ap.add_argument("days", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bay = V.load_bay(a.geom)
    res = {}
    seen = set()
    for f in a.days:
        for m in json.load(open(f))["missions"]:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            day = m["day"]
            occ, _ = V.day_scene(bay, day)
            dx, dy, dpsi = day["ugv"]
            root = UGV_ROOT + np.array([dx, dy, 0.0])
            C, d, u, r = cam_world(root, ugv_frame(root[:2], dpsi), UGV_MAST)
            res[m["id"]] = dict(start=start_views(day, occ), dial_from_start=dial_needle_px(C),
                                dial_from_valve_stand=dial_needle_px(np.array([-1.24, 2.61, 0.92])),
                                valve=valve_views(day, bay), pan=pan_views(day, occ))
    res["bucket_formation"] = bucket_views(V.load_bay(a.geom))
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
