"""A compiled MuJoCo camera as an exact OpenCV pinhole: resolution, fx, fy, cx, cy (pixels) and pose.

OpenCV convention: x right, y down, z forward, and pixel (0, 0) is the centre of the top-left pixel. MuJoCo cameras
look down their -z axis with +y up, so world_R_opencv = cam_xmat @ diag(1, -1, -1).

MuJoCo 3.16 decoding (measured with sphere centroids rendered by MuJoCo itself, error below 0.4 px):
  with sensorsize:  fx = intrinsic[0] / sensorsize[0] * W        fy = intrinsic[1] / sensorsize[1] * H
                    px = intrinsic[2] / sensorsize[0] * W        py = intrinsic[3] / sensorsize[1] * H
                    cx = (W - 1) / 2 - px                         cy = (H - 1) / 2 - py
                    (px, py are the MJCF principalpixel values; focalpixel gives fx, fy directly)
  without:          fx = fy = (H / 2) / tan(fovy / 2),  cx = (W - 1) / 2,  cy = (H - 1) / 2
So to put the principal point at OpenCV (cx, cy), write principalpixel = ((W - 1) / 2 - cx, (H - 1) / 2 - cy).
"""
import math

import numpy as np

CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])
RULE = ("fx = intrinsic[0]/sensorsize[0]*W, fy = intrinsic[1]/sensorsize[1]*H, "
        "cx = (W-1)/2 - intrinsic[2]/sensorsize[0]*W, cy = (H-1)/2 - intrinsic[3]/sensorsize[1]*H; "
        "without sensorsize fx = fy = (H/2)/tan(fovy/2), cx = (W-1)/2, cy = (H-1)/2")


class CameraError(ValueError):
    pass


def intrinsics(m, cid, size=None):
    """(W, H, fx, fy, cx, cy) of camera `cid`. `size` = (W, H) replaces the camera's own resolution (the focal
    lengths and principal point then follow MuJoCo's own rule at that resolution)."""
    if int(m.cam_projection[cid]) != 0:
        raise CameraError("the camera is orthographic; only perspective cameras can be rendered")
    W, H = (int(v) for v in m.cam_resolution[cid])
    if size is not None:
        W, H = int(size[0]), int(size[1])
    if W < 2 or H < 2:
        raise CameraError("the camera has no resolution (set resolution=\"W H\" on it, or pass --size WxH)")
    sw, sh = (float(v) for v in m.cam_sensorsize[cid])
    if sw > 0 and sh > 0:
        f0, f1, p0, p1 = (float(v) for v in m.cam_intrinsic[cid])
        fx, fy = f0 / sw * W, f1 / sh * H
        cx, cy = (W - 1) / 2.0 - p0 / sw * W, (H - 1) / 2.0 - p1 / sh * H
    else:
        fovy = float(m.cam_fovy[cid])
        if not 0.0 < fovy < 180.0:
            raise CameraError("the camera's fovy must lie between 0 and 180 degrees (it is %g)" % fovy)
        fy = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)
        fx = fy
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise CameraError("the camera intrinsics are not finite and positive")
    return W, H, fx, fy, cx, cy


def opencv_camera(m, d, cid, size=None):
    """(K, world_R_cv, centre) of camera `cid` in the state `d` (after mj_forward)."""
    _, _, fx, fy, cx, cy = intrinsics(m, cid, size)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    R = np.asarray(d.cam_xmat[cid], float).reshape(3, 3) @ CV_FROM_MJ
    return K, R, np.asarray(d.cam_xpos[cid], float).copy()


def principalpixel_for(W, H, cx, cy):
    """The MJCF principalpixel that puts the principal point at OpenCV (cx, cy)."""
    return (W - 1) / 2.0 - cx, (H - 1) / 2.0 - cy
