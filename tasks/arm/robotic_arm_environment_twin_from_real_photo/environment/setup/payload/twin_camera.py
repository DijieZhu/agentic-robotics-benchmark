"""Camera conventions of the twin: MuJoCo <camera> intrinsics <-> an OpenCV pinhole camera, and the OVRTX camera.

Only numpy is needed. decode() and camera_frame() read a compiled mujoco.MjModel / mujoco.MjData; nothing here
creates a rendering context.

OpenCV pinhole convention used everywhere in the twin (no distortion):
  - pixel (u, v): u counts columns to the right, v counts rows downward;
  - the centre of the top-left pixel is (0, 0), so a W x H image covers [-0.5, W - 0.5] x [-0.5, H - 0.5] and a
    centred principal point is ((W - 1) / 2, (H - 1) / 2);
  - K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]];
  - camera frame: x right, y down, z forward along the optical axis; "depth" is z, not the length of the ray.
  - A world point X projects to (u, v) = (fx x / z + cx, fy y / z + cy) with (x, y, z) = R_cv^T (X - p), where the
    columns of R_cv are the camera axes in world coordinates and p is the camera centre.

MuJoCo cameras look along their local -z axis with +y up, so R_cv = cam_xmat @ diag(1, -1, -1).

How MuJoCo 3.16 turns a <camera> into K (measured by rendering small spheres at known points and comparing their
image centroids with this projection: within 0.25 px, and within 0.07 px of the exact image of each sphere;
for a render of W x H pixels, W x H being
the camera's resolution attribute):
  sensorsize form (sensorsize plus focal or focalpixel, and optionally principal or principalpixel):
      fx = focalpixel_x                     fy = focalpixel_y
      cx = (W - 1) / 2 - principalpixel_x   cy = (H - 1) / 2 - principalpixel_y
    so principalpixel = (image centre) - (cx, cy), both axes in OpenCV pixel axes. The compiler stores the
    length form, cam_intrinsic = [focal_x, focal_y, principal_x, principal_y] in sensor length units, with
    focal = focalpixel * sensorsize / resolution (the same for principal), so the sensor size itself does not
    matter. focal / principal in length units are converted with the same ratio.
  fovy form (no sensorsize): fy = (H / 2) / tan(fovy / 2), fx = fy, (cx, cy) = the image centre.
A render at another size w x h stretches the same frustum over the new image (sensorsize form), or keeps fovy
and square pixels (fovy form); decode(model, cam, size=(w, h)) returns the K of such a render.
"""
import math

import numpy as np

CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])     # MuJoCo camera axes (x right, y up, z back) -> OpenCV axes
DEFAULT_PITCH_M = 3e-6                      # pixel pitch used to write a sensorsize (any positive value works)


class CameraError(ValueError):
    """The camera cannot be read as a perspective pinhole camera."""


def K_matrix(fx, fy, cx, cy):
    return np.array([[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]])


def _cam_id(model, cam):
    if isinstance(cam, str):
        try:
            return int(model.camera(cam).id)
        except Exception as e:  # noqa: BLE001 - the bindings raise KeyError or ValueError depending on version
            raise CameraError("no camera named %r" % cam) from e
    cid = int(cam)
    if not 0 <= cid < int(model.ncam):
        raise CameraError("camera id %d out of range (ncam = %d)" % (cid, int(model.ncam)))
    return cid


def _fmt(v):
    return "%.10g" % float(v)


def encode(K, W, H, form="pixel", pitch=DEFAULT_PITCH_M, tol=1e-9):
    """MJCF <camera> attributes (a dict of strings) that reproduce K at resolution W x H.

    form "pixel": resolution, sensorsize, focalpixel, principalpixel (the recommended form);
    form "length": resolution, sensorsize, focal, principal (sensor length units, pixel pitch `pitch`);
    form "fovy": resolution and fovy; only for fx == fy with a centred principal point (within `tol` px)."""
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = int(W), int(H)
    if W < 2 or H < 2 or not (fx > 0 and fy > 0) or not all(math.isfinite(v) for v in (fx, fy, cx, cy)):
        raise CameraError("need W, H >= 2 and finite fx, fy > 0")
    if abs(K[0, 1]) > tol:
        raise CameraError("MuJoCo cameras have no skew")
    px, py = (W - 1) / 2.0 - cx, (H - 1) / 2.0 - cy
    out = {"resolution": "%d %d" % (W, H)}
    if form == "fovy":
        if abs(fx - fy) > tol or abs(px) > tol or abs(py) > tol:
            raise CameraError("the fovy form needs fx == fy and a centred principal point; use form='pixel'")
        out["fovy"] = _fmt(math.degrees(2.0 * math.atan((H / 2.0) / fy)))
        return out
    out["sensorsize"] = "%s %s" % (_fmt(W * pitch), _fmt(H * pitch))
    if form == "pixel":
        out["focalpixel"] = "%s %s" % (_fmt(fx), _fmt(fy))
        out["principalpixel"] = "%s %s" % (_fmt(px), _fmt(py))
    elif form == "length":
        out["focal"] = "%s %s" % (_fmt(fx * pitch), _fmt(fy * pitch))
        out["principal"] = "%s %s" % (_fmt(px * pitch), _fmt(py * pitch))
    else:
        raise CameraError("unknown form %r" % form)
    return out


def describe(model, cam, size=None):
    """Everything decode() reads, plus the decoded K: a dict for printing in the check tools."""
    cid = _cam_id(model, cam)
    proj = int(model.cam_projection[cid]) if hasattr(model, "cam_projection") else 0
    res = [int(v) for v in model.cam_resolution[cid]]
    ss = [float(v) for v in model.cam_sensorsize[cid]]
    intr = [float(v) for v in model.cam_intrinsic[cid]]
    fovy = float(model.cam_fovy[cid])
    form = "sensorsize" if (ss[0] > 0 and ss[1] > 0) else "fovy"
    out = {"camera_id": cid, "projection": "perspective" if proj == 0 else "orthographic", "resolution": res,
           "form": form, "sensorsize": ss, "intrinsic": intr, "fovy_deg": fovy}
    try:
        K, W, H = decode(model, cid, size=size)
        out.update({"W": W, "H": H, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]})
    except CameraError as e:
        out["error"] = str(e)
    return out


def decode(model, cam, size=None):
    """(K, W, H) of a compiled MuJoCo camera (id or name), for a render of the camera's own resolution, or of
    size = (w, h) pixels. Raises CameraError for orthographic cameras, a missing resolution, or a focal length
    that is not positive."""
    cid = _cam_id(model, cam)
    if hasattr(model, "cam_projection") and int(model.cam_projection[cid]) != 0:
        raise CameraError("the camera is orthographic; a pinhole camera must be perspective")
    W0, H0 = (int(v) for v in model.cam_resolution[cid])
    if W0 < 2 or H0 < 2:
        raise CameraError("the camera has no resolution attribute (resolution=\"W H\")")
    W, H = (W0, H0) if size is None else (int(size[0]), int(size[1]))
    if W < 2 or H < 2:
        raise CameraError("render size must be at least 2 x 2")
    sw, sh = (float(v) for v in model.cam_sensorsize[cid])
    if sw > 0 and sh > 0:
        f0, f1, p0, p1 = (float(v) for v in model.cam_intrinsic[cid])
        # focal and principal in pixels of the camera's own resolution
        fx0, fy0 = f0 / sw * W0, f1 / sh * H0
        px0, py0 = p0 / sw * W0, p1 / sh * H0
        sx, sy = W / W0, H / H0                    # the frustum is stretched over the render
        fx, fy = fx0 * sx, fy0 * sy
        cx, cy = (W - 1) / 2.0 - px0 * sx, (H - 1) / 2.0 - py0 * sy
    else:
        fovy = float(model.cam_fovy[cid])
        if not 0.0 < fovy < 180.0:
            raise CameraError("fovy must be in (0, 180) degrees, got %g" % fovy)
        fy = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)
        fx = fy
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise CameraError("the focal lengths must be finite and positive (fx %g, fy %g)" % (fx, fy))
    return K_matrix(fx, fy, cx, cy), W, H


def scale_K(K, sx, sy=None):
    """K of the same camera after resizing the image by sx (and sy): pixel edges, not centres, scale."""
    sy = sx if sy is None else sy
    K = np.asarray(K, float)
    return K_matrix(K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5)


def camera_frame(model, data, cam):
    """(R_cv, p): the OpenCV camera axes (columns of R_cv: x right, y down, z forward) and the camera centre, in
    world coordinates, in the state `data`. `data` must have been through mj_forward (or mj_kinematics followed
    by mj_camlight); with data=None the model's qpos0 is used."""
    cid = _cam_id(model, cam)
    if data is None:
        import mujoco  # only needed for this convenience path
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
    R = np.asarray(data.cam_xmat[cid], float).reshape(3, 3) @ CV_FROM_MJ
    return R, np.asarray(data.cam_xpos[cid], float).copy()


def to_camera(R_cv, p, X):
    """World points (N, 3) -> camera-frame points (N, 3)."""
    return (np.asarray(X, float).reshape(-1, 3) - np.asarray(p, float)) @ np.asarray(R_cv, float)


def project(K, R_cv, p, X):
    """World points (N, 3) -> (uv (N, 2), z (N,)); uv is NaN for points with z <= 0."""
    Xc = to_camera(R_cv, p, X)
    z = Xc[:, 2]
    K = np.asarray(K, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.c_[K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]]
    uv[z <= 0] = np.nan
    return uv, z


def pixel_rays(K, R_cv, uv):
    """Unit world directions of the rays through pixel positions uv (N, 2), and the z component of each unit ray
    in the camera frame (depth = ray length * that component)."""
    K = np.asarray(K, float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    d = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    n = np.linalg.norm(d, axis=1)
    d /= n[:, None]
    return d @ np.asarray(R_cv, float).T, d[:, 2]


def backproject(K, R_cv, p, uv, depth):
    """Pixel positions uv (N, 2) with depth z (N,) -> world points (N, 3)."""
    K = np.asarray(K, float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    z = np.asarray(depth, float).reshape(-1)
    Xc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z]
    return Xc @ np.asarray(R_cv, float).T + np.asarray(p, float)


def pixel_grid(W, H, stride=1):
    """Pixel centres (N, 2) of a W x H image, row by row, every `stride` pixels."""
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    return np.c_[u.ravel(), v.ravel()].astype(float)


def quat_wxyz(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix (MuJoCo's order)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-300))
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def camera_xml(name, K, W, H, R_cv, p, form="pixel", pitch=DEFAULT_PITCH_M):
    """A complete <camera .../> element (fixed, on the body it is written in) for an OpenCV camera."""
    attrs = encode(K, W, H, form=form, pitch=pitch)
    q = quat_wxyz(np.asarray(R_cv, float) @ CV_FROM_MJ)
    head = 'name="%s" pos="%s" quat="%s"' % (name, " ".join("%.9f" % v for v in np.asarray(p, float)),
                                             " ".join("%.12f" % v for v in q))
    return "<camera %s %s/>" % (head, " ".join('%s="%s"' % kv for kv in attrs.items()))


# ---------------------------------------------------------------------------------------------------------------
# OVRTX (USD) camera, measured the same way with ovrtx 0.6 path tracing:
#   route A, OmniLensDistortionOpenCvPinholeAPI: OVRTX reads cx, cy with the pixel centres at +0.5 (the corner
#     convention), so write cx + 0.5 and cy + 0.5; fx, fy as they are; imageSize = the render resolution; zero
#     distortion. Exact (0.01 px) with fx != fy. This is the route to use.
#   route B, plain UsdGeomCamera: square pixels only (OVRTX ignores verticalAperture and takes the vertical scale
#     from the horizontal one), with pixel pitch q = focalLength / fx:
#       horizontalAperture = W q, verticalAperture = H q,
#       horizontalApertureOffset = ((W - 1) / 2 - cx) q, verticalApertureOffset = (cy - (H - 1) / 2) q.
CV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes -> USD camera axes (USD looks along -z, y up)
USD_PINHOLE_DC = 0.5                          # added to cx, cy for the opencvPinhole schema
USD_LENS_UNIT = 0.1                           # USD lens attributes are in tenths of a scene unit (metres here)
_DIST_KEYS = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4")


def usd_matrix_rows(R_cv, p):
    """USD xformOp:transform of the camera prim (row-vector convention: translation in the last row)."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R_cv, float), np.asarray(p, float)
    M = (T @ CV_TO_USD).T
    return "(" + ", ".join("(" + ", ".join("%.12g" % v for v in row) + ")" for row in M) + ")"


def usd_aperture_attrs(K, W, H, focal_m=0.02):
    """Route B lens attributes (USD units on a metre stage) for K at W x H, with square pixels of pitch
    focal / fx. Exact only when fx == fy; route A uses them as the nominal lens (depth of field)."""
    K = np.asarray(K, float)
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    q = focal_m / fx / USD_LENS_UNIT
    return {"focalLength": focal_m / USD_LENS_UNIT, "horizontalAperture": W * q, "verticalAperture": H * q,
            "horizontalApertureOffset": ((W - 1) / 2.0 - cx) * q, "verticalApertureOffset": (cy - (H - 1) / 2.0) * q}


def usd_pinhole_attrs(K, W, H):
    """Route A: OmniLensDistortionOpenCvPinholeAPI attributes (every one authored, distortion zero)."""
    K = np.asarray(K, float)
    out = {"omni:lensdistortion:model": "opencvPinhole",
           "omni:lensdistortion:opencvPinhole:imageSize": (int(W), int(H)),
           "omni:lensdistortion:opencvPinhole:fx": float(K[0, 0]),
           "omni:lensdistortion:opencvPinhole:fy": float(K[1, 1]),
           "omni:lensdistortion:opencvPinhole:cx": float(K[0, 2]) + USD_PINHOLE_DC,
           "omni:lensdistortion:opencvPinhole:cy": float(K[1, 2]) + USD_PINHOLE_DC}
    for k in _DIST_KEYS:
        out["omni:lensdistortion:opencvPinhole:" + k] = 0.0
    return out


def usd_camera_prim(name, K, W, H, R_cv, p, route="A", near=0.01, far=100.0, indent="    ", extra_lines=()):
    """USDA text of a Camera prim that reproduces the OpenCV camera (K, W x H, R_cv, p) in OVRTX.
    Route A also writes the aperture attributes (OVRTX uses them for depth of field only). extra_lines are
    appended inside the prim as written (for example the exposure attributes)."""
    body = ['token projection = "perspective"', "float2 clippingRange = (%r, %r)" % (near, far), "float fStop = 0"]
    body += ["float %s = %.9g" % kv for kv in usd_aperture_attrs(K, W, H).items()]
    if route == "A":
        for k, v in usd_pinhole_attrs(K, W, H).items():
            if k.endswith(":model"):
                body.append('token %s = "%s"' % (k, v))
            elif k.endswith(":imageSize"):
                body.append("int2 %s = (%d, %d)" % (k, v[0], v[1]))
            else:
                body.append("float %s = %.9g" % (k, v))
    elif route == "B":
        if abs(K[0][0] - K[1][1]) > 1e-6 * K[0][0]:
            raise CameraError("route B (apertures) cannot give fx != fy in OVRTX; use route A")
    else:
        raise ValueError("route must be 'A' or 'B'")
    body += list(extra_lines)
    body += ["matrix4d xformOp:transform = %s" % usd_matrix_rows(R_cv, p), 'uniform token[] xformOpOrder = ["xformOp:transform"]']
    head = ['def Camera "%s"' % name]
    if route == "A":
        head = ['def Camera "%s" (' % name, '    prepend apiSchemas = ["OmniLensDistortionOpenCvPinholeAPI"]', ")"]
    lines = head + ["{"] + ["    " + b for b in body] + ["}"]
    return "".join(indent + ln + "\n" for ln in lines)
