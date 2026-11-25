"""A minimal operator program that runs and scores nothing. It makes every call of the station once: it reads the
mission, takes one camera frame and turns its depth image into site-frame points, reads the telemetry, starts one
primitive with a guessed set of ground nodes, sends one target shape, stops, reports a search point from where the
robot happens to stand (no dwell, far from the point: no credit) and finishes. It does not look for the robot in the
frame, does not work out which nodes are on the ground and goes nowhere.

Copy it to ~/submission/operate.py to try the wire:

    /opt/venv/bin/python3 ~/training/run_mission.py --list
    /opt/venv/bin/python3 ~/training/run_mission.py --site cal

print() goes to stderr (the launcher keeps stdout for the wire).
"""

import numpy as np

from voidcell.client import StationError


def site_points(frame, camera):
    """The frame's depth image as site-frame points (H x W x 3, NaN where the camera has no reading), with the
    intrinsics and the calibrated pose of mission()["camera"] (camera_datasheet.md, back-projection)."""
    fx, fy, cx, cy = camera["K"]
    T = np.asarray(camera["T_world_camera"], float)
    z = frame["depth"].astype(float) * frame["depth_unit_m"]
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    pts = cam @ T[:3, :3].T + T[:3, 3]
    pts[z <= 0] = np.nan
    return pts


def wait(station, dt=0.5):
    """Step the simulated time until no command runs."""
    while station.status()["busy"]:
        if station.step(dt)["done"]:
            break


def solve(station):
    m = station.mission()
    print("site", m["site"]["size_m"], "| pad", m["pad"]["xy"], "| search points",
          [(p["id"], p["xy"], p["weight"]) for p in m["search_points"]])
    print("budgets", m["limits"], "| pose source", m["pose_source"])

    f = station.get_frame()                                   # one frame of the budget
    pts = site_points(f, m["camera"])
    seen = np.isfinite(pts[..., 2])
    print("frame %d at t = %.2f s: %d pixels with depth, highest point %.2f m, %d frames left"
          % (f["index"], f["t"], int(seen.sum()), float(np.nanmax(pts[..., 2])), f["frames_left"]))

    station.step(1.0)                                         # simulated time passes only here
    tl = station.telemetry()                                  # the records of that second
    print("telemetry: %d records | short sensors (mm) %s | accelerometer x of struts 0, 1, 2 (g) %s"
          % (tl["n"], np.round(tl["sensor_length_mm"][-1, :6], 1), np.round(tl["imu_accel_g"][-1, :, 0], 2)))

    # The three nodes on the ground are a guess here; the station does not check them (robot_handbook.md, section 6):
    # with a wrong triple the robot executes the primitive of another face.
    guess = [0, 3, 5]
    print("ccw:", station.run_primitive("ccw", guess))
    wait(station)
    print("after the primitive:", station.status())

    lo, hi = m["limits"]["cable_length_mm"]
    rest = [200.0] * 6                                        # the rest shape (robot_handbook.md, section 5)
    print("set_targets:", station.set_targets(rest), "| limits", lo, hi)
    wait(station)
    station.stop()                                            # nothing runs: it does nothing

    # A report without a dwell and far from the point: accepted, and worth nothing.
    print("report_point(1):", station.report_point(1))
    try:
        print("pose:", station.pose())
    except StationError as e:                                 # pose_source "estimated": not_available
        print("pose:", e.code)
    print("finish:", station.finish())
