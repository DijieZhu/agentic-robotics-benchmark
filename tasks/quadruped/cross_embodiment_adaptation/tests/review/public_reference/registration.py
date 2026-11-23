"""Align unlabeled safety-post caps seen by two calibrated RGB cameras."""
import itertools
import math

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment

from client import camera_image


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def cap_rays(rgb, fov):
    """Silhouette tangent angles: center slopes (right, up) and forward depth."""
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    h, w = g.shape
    f = h/(2*math.tan(math.radians(fov)/2))
    labels, _ = ndimage.label((g > 55) & (g > 1.55*r) & (g > 1.70*b))
    for box in ndimage.find_objects(labels):
        v, u = box
        if min(u.start, v.start, w-u.stop, h-v.stop) < 2 or min(u.stop-u.start, v.stop-v.start) < 5:
            continue
        if not .55 < (u.stop-u.start)/(v.stop-v.start) < 1.8:
            continue
        left, right = np.arctan((np.array([u.start, u.stop])-w/2)/f)
        bottom, top = np.arctan((h/2-np.array([v.stop, v.start]))/f)
        horizontal, vertical = (right+left)/2, (top+bottom)/2
        depth = .105*math.cos(horizontal)/math.sin((right-left)/2)
        yield np.tan(horizontal), np.tan(vertical), depth


def clusters(points, tolerance):
    groups = []
    for pt in points:
        for group in groups:
            if np.linalg.norm(pt-np.median(group, axis=0)) < tolerance:
                group.append(pt)
                break
        else:
            groups.append([pt])
    return np.array([np.median(group, axis=0) for group in groups])


def map_caps(rgb, xy):
    for right, up, depth in cap_rays(rgb, 78):
        if abs(depth-(1.65-.72)) < .09:
            yield xy+(1.65-.72)*np.array([right, up])


def mast_scan(robot, obs):
    up = -np.array(obs["gravity_body_unit"])
    up /= np.linalg.norm(up)
    left = np.cross(up, [1., 0., 0.])
    left /= np.linalg.norm(left)
    # Level the rigid camera using gravity, not a world compass.
    level = np.array([np.cross(left, up), left, up])
    offset = (level @ [0, 0, .20])[:2]
    points = []
    pitch = math.radians(36)
    for detent in range(8):
        pan = detent*math.pi/4
        rgb = camera_image(robot.look(detent))
        for right, above, depth in cap_rays(rgb, 70):
            forward_left = rot(pan) @ [math.cos(pitch)-above*math.sin(pitch), -right]
            ray = level @ [*forward_left, math.sin(pitch)+above*math.cos(pitch)]
            points.append(offset+depth*ray[:2])
    return clusters(points, .20), offset


def register(local, chart, offset):
    """Determine correspondences geometrically, then refine using accurate bearings."""
    if min(len(local), len(chart)) < 3:
        raise ValueError("fewer than three usable caps in a view")
    best = None
    for i, j in itertools.combinations(range(len(local)), 2):
        line = local[j]-local[i]
        for a, b in itertools.permutations(range(len(chart)), 2):
            match = chart[b]-chart[a]
            if abs(np.linalg.norm(line)-np.linalg.norm(match)) > .30:
                continue
            yaw = math.atan2(*match[::-1])-math.atan2(*line[::-1])
            R = rot(yaw)
            t = .5*(chart[a]+chart[b]-R @ (local[i]+local[j]))
            delta = (local @ R.T+t)[:, None, :]-chart
            distances = np.linalg.norm(delta, axis=2)
            rows, cols = linear_sum_assignment(distances)
            ds = distances[rows, cols]
            hits = ds < .19
            count = int(hits.sum())
            if count < 3:
                continue
            score = (-count, float(np.mean(ds[hits]**2)))
            if best is None or score < best[0]:
                best = score, [*t, yaw], rows[hits], cols[hits]
    if best is None:
        raise ValueError("no consistent rigid alignment between cap layouts")
    _, initial, rows, cols = best
    rays = local[rows]-offset
    ranges = np.linalg.norm(rays, axis=1)
    rays /= ranges[:, None]
    normals = np.column_stack((-rays[:, 1], rays[:, 0]))

    def residual(pose):
        diff = (chart[cols]-pose[:2]) @ rot(pose[2])-local[rows]
        lateral = (diff*normals).sum(axis=1)
        radial = (diff*rays).sum(axis=1)
        return np.r_[lateral/.005, radial/(.018+.012*ranges**2)]

    pose = least_squares(residual, initial).x
    pose[2] = math.atan2(math.sin(pose[2]), math.cos(pose[2]))
    return pose, {"matched_caps": len(rows), "ground_caps": len(local), "uav_caps": len(chart),
                  "cross_ray_rms_m": float(np.sqrt(np.mean((residual(pose)[:len(rows)]*.005)**2)))}
