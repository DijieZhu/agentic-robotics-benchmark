"""Tag plate of an inspection point: version 1's marker plate (an AprilTag / ArUco fiducial next to the printed tag
text; still the grid marker plates of version 2) or version 2's text plate (`TagSpec(text, style="text")`).

Text plate (version 2): the same outline, thickness, body, geom and material names as a 60 mm
marker plate (0.1908 x 0.0792 x 0.002 m, `<name>`, `<name>/plate`), signal yellow with a black border line and the
tag in the plant lettering (`bricks.lettering.tag_plate_image`); no marker and no `<name>/marker` site; its
`TargetNames.state` is `{"text", "style": "text", "plate_m": (w, h)}`. `ground_truth` gives its centre, face
rotation and corners. `plate_rgb`, `text_rgb`, `marker_*` and `quiet_frac` / `text_w_frac` do not apply to it.

Plate layout (face frame: +x out of the plate, +y right, +z up): a white quiet zone, the marker on the left, the tag
text on the right. The marker's printed side `marker_mm` is the black square including its black border (the corners
OpenCV reports). A site '<name>/marker' (group 5, not drawn by default) sits at the marker centre with the marker's
frame: x right, y up, z out of the plate (OpenCV's marker convention).

Reader: `read(rgb, hint)` detects markers with cv2.aruco, picks the requested id (or the one nearest the image centre)
and solves its pose with solvePnP (IPPE_SQUARE) from the camera intrinsics in the hint. Output pose is in the camera's
optical frame (x right, y down, z forward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .. import lettering
from ..common import Pose, TargetNames, fit_font, quat_from_mat, save_png, texture_material

__all__ = ["TagSpec", "plate_image", "add_to_spec", "set_state", "read", "ground_truth", "R_FACE_MARKER",
           "DEFAULT_DICT", "detector", "text_plate", "STYLES"]

STYLES = ("marker", "text")

DEFAULT_DICT = "DICT_APRILTAG_36h11"
# marker frame (x right, y up, z out) expressed in the face frame (x out, y right, z up)
R_FACE_MARKER = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class TagSpec:
    text: str = "PI-2041"
    marker_id: int = 0
    marker_mm: float = 60.0
    dictionary: str = DEFAULT_DICT
    quiet_frac: float = 0.16          # white margin around the marker, in units of the marker side
    text_w_frac: float = 1.7          # text area width in units of the marker side
    plate_rgb: tuple = (238, 238, 230)
    text_rgb: tuple = (15, 15, 15)
    thickness_mm: float = 2.0
    px_per_mm: float = 4.0
    style: str = "marker"             # "marker" (version 1, grid markers) or "text" (version 2 instrument plates)

    @property
    def marker_m(self) -> float:
        return self.marker_mm / 1000.0

    @property
    def is_text(self) -> bool:
        if self.style not in STYLES:
            raise ValueError("TagSpec.style must be one of %s, not %r" % (STYLES, self.style))
        return self.style == "text"

    def plate_size_m(self):
        if self.is_text:
            return lettering.TAG_PLATE_W_M, lettering.TAG_PLATE_H_M
        m = self.marker_mm
        q = self.quiet_frac * m
        w = q + m + q + self.text_w_frac * m + q
        h = q + m + q
        return w / 1000.0, h / 1000.0

    def marker_offset_m(self):
        """(y, z) of the marker centre from the plate centre in the face frame."""
        if self.is_text:
            raise ValueError("a text plate has no marker")
        w, _ = self.plate_size_m()
        q = self.quiet_frac * self.marker_m
        return -w / 2.0 + q + self.marker_m / 2.0, 0.0


def _dict(name):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def text_plate(text: str, **kw) -> TagSpec:
    """A version 2 text plate for an instrument tag (the tag grammar is checked)."""
    return TagSpec(text=lettering.check_tag(text), style="text", **kw)


def plate_image(ts: TagSpec) -> np.ndarray:
    """RGB uint8 image of the plate: a marker plate (the marker, the text, a thin border) or a text plate
    (`lettering.tag_plate_image`, 763 x 317 px at the default 4 px/mm)."""
    if ts.is_text:
        return lettering.tag_plate_image(ts.text, px_per_mm=ts.px_per_mm)
    w_m, h_m = ts.plate_size_m()
    ppm = ts.px_per_mm
    bits = _dict(ts.dictionary).markerSize + 2
    side = int(round(ts.marker_mm * ppm / bits)) * bits
    ppm = side / ts.marker_mm
    W, H = int(round(w_m * 1000 * ppm)), int(round(h_m * 1000 * ppm))
    img = Image.new("RGB", (W, H), tuple(ts.plate_rgb))
    q = int(round(ts.quiet_frac * ts.marker_mm * ppm))
    mk = cv2.aruco.generateImageMarker(_dict(ts.dictionary), int(ts.marker_id), side, borderBits=1)
    img.paste(Image.fromarray(mk).convert("RGB"), (q, (H - side) // 2))
    d = ImageDraw.Draw(img)
    bw = max(2, H // 60)
    d.rectangle([bw // 2, bw // 2, W - 1 - bw // 2, H - 1 - bw // 2], outline=(60, 60, 60), width=bw)
    tx0 = q + side + q
    tw = W - tx0 - q
    f = fit_font(d, ts.text, tw, H * 0.42, H * 0.40)
    d.text((tx0 + tw / 2.0, H * 0.45), ts.text, font=f, fill=tuple(ts.text_rgb), anchor="mm")
    f2 = fit_font(d, "ID %d" % ts.marker_id, tw, H * 0.16, H * 0.14)
    d.text((tx0 + tw / 2.0, H * 0.80), "ID %d" % ts.marker_id, font=f2, fill=(70, 70, 70), anchor="mm")
    return np.asarray(img, dtype=np.uint8)


def add_to_spec(spec, parent, pose: Pose, ts: TagSpec, assets_dir: str, name: str | None = None) -> TargetNames:
    """Add the plate as a thin box facing +x of `pose` (its front face at x = 0 of the pose frame)."""
    name = name or "tag_%s" % ts.text
    w, h = ts.plate_size_m()
    t = ts.thickness_mm / 1000.0
    body = parent.add_body(name=name, pos=list(pose.pos), quat=list(pose.quat))
    prefix = ("tagtext_" if ts.is_text else "tag_") + ts.text
    mat = texture_material(spec, name + "/plate", save_png(plate_image(ts), assets_dir, prefix),
                           specular=0.2, shininess=0.3)
    body.add_geom(name=name + "/plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t / 2, w / 2, h / 2],
                  pos=[-t / 2, 0, 0], material=mat)
    if ts.is_text:
        return TargetNames(name=name, kind="tag", body=name, geoms=[name + "/plate"], face_geoms=[name + "/plate"],
                           bodies=[name], materials=[mat], state={"text": ts.text, "style": "text",
                                                                 "plate_m": (float(w), float(h))})
    my, mz = ts.marker_offset_m()
    body.add_site(name=name + "/marker", pos=[0.0005, my, mz], quat=list(quat_from_mat(R_FACE_MARKER)),
                  size=[0.005, 0.005, 0.005], group=5)
    return TargetNames(name=name, kind="tag", body=name, geoms=[name + "/plate"], face_geoms=[name + "/plate"],
                       bodies=[name], materials=[mat], state={"site": name + "/marker", "id": int(ts.marker_id),
                                                             "text": ts.text, "marker_m": ts.marker_m})


def set_state(model, data, names: TargetNames, value=None):
    """A tag has no state."""
    return None


def ground_truth(model, data, names: TargetNames) -> dict:
    """After mj_forward / mj_kinematics. Marker plate: {'id', 'text', 'pos_world' (marker centre), 'R_world_marker'}.
    Text plate: {'text', 'pos_world' (centre of the plate's front face), 'R_world_face' (columns: x out of the plate,
    y the viewer's right, z up), 'corners_world' (4 x 3: the outline's TL, TR, BR, BL as the viewer sees them),
    'border_world' (the same for the border line's centre line)}."""
    if names.state.get("style") == "text":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.body)
        p = data.xpos[bid].copy()
        R = data.xmat[bid].reshape(3, 3).copy()
        w, h = names.state["plate_m"]
        bw, bh = lettering.tag_layout(names.state["text"])["border_centre"]

        def quad(hw, hh):
            return np.array([p + R @ np.array([0.0, sy * hw, sz * hh]) for sy, sz in ((-1, 1), (1, 1), (1, -1),
                                                                                    (-1, -1))])
        return {"text": names.state["text"], "pos_world": p, "R_world_face": R, "corners_world": quad(w / 2, h / 2),
                "border_world": quad(bw, bh)}
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, names.state["site"])
    return {"id": names.state["id"], "text": names.state["text"], "pos_world": data.site_xpos[sid].copy(),
            "R_world_marker": data.site_xmat[sid].reshape(3, 3).copy()}


_DETECTORS = {}


def detector(dictionary: str = DEFAULT_DICT):
    d = _DETECTORS.get(dictionary)
    if d is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.minMarkerPerimeterRate = 0.008
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 43
        p.adaptiveThreshWinSizeStep = 8
        d = cv2.aruco.ArucoDetector(_dict(dictionary), p)
        _DETECTORS[dictionary] = d
    return d


def read(rgb: np.ndarray, hint: dict):
    """Detect the tag and its pose.

    hint: {'K': 3x3 intrinsics, 'marker_size_m': 0.06, optional 'dist' (distortion), 'dictionary', 'id' (expected id),
    'id_to_text' (work-order table id -> tag text)}.
    Returns (value, confidence, debug); value = {'id', 'text', 'pos_cam' (marker centre, optical frame, m),
    'R_cam_marker', 'corners' (4x2 px), 'reproj_px'} or None when nothing is detected."""
    K = np.asarray(hint["K"], np.float64)
    dist = np.asarray(hint.get("dist", np.zeros(5)), np.float64)
    s = float(hint.get("marker_size_m", 0.06))
    det = detector(hint.get("dictionary", DEFAULT_DICT))
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = det.detectMarkers(gray)
    dbg = {"n_detected": 0 if ids is None else int(len(ids)), "n_rejected": len(rejected), "all": []}
    if ids is None or len(ids) == 0:
        return None, 0.0, dbg
    obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float64)
    H, W = gray.shape
    best, best_key = None, None
    for c, i in zip(corners, ids.ravel()):
        c = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - c) ** 2, 1))))
        R, _ = cv2.Rodrigues(rvec)
        side_px = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
        rec = {"id": int(i), "pos_cam": tvec.ravel(), "R_cam_marker": R, "corners": c, "reproj_px": err,
               "side_px": side_px}
        if "id_to_text" in hint:
            rec["text"] = hint["id_to_text"].get(int(i))
        dbg["all"].append({"id": int(i), "side_px": side_px, "reproj_px": err})
        ctr = c.mean(0)
        key = (0 if (hint.get("id") is not None and int(i) == int(hint["id"])) else 1,
               math.hypot(ctr[0] - W / 2, ctr[1] - H / 2))
        if best is None or key < best_key:
            best, best_key = rec, key
    if best is None:
        return None, 0.0, dbg
    conf = float(np.clip(1.0 - best["reproj_px"] / 3.0, 0.0, 1.0) * np.clip(best["side_px"] / 20.0, 0.0, 1.0))
    return best, conf, dbg
