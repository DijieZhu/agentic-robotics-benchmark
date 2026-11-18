"""The FIXED readers (numpy + scipy only): the exact functions and parameters the reference program, the unprivileged
pilot and the grader-side observability check use (DESIGN v1.2 section 2.4; G4 section 5 items 3, 4, 7).

  decode(rgb)                        square plate codes, correction limit 2 bits (codes.decode, max_errors = 2)
  read_tags(rgb, exclude)            painted tags (tagtext.read_tags_v3 = G4's v2 -- content registration of the
                                     plate quad -- over extra plate candidates for the light panel face of the
                                     finished skid; own-gripper pixels blanked first when the mask is given)
  read_handle(rgb, local, uv, hfov,  handle open / closed = the AGREEMENT of two independent readers, else None
              exclude)               ("don't know", never a guess): handles.read_handle_v4 (G4: blade shape of the
                                     brightest blob; answers only when both blade hypotheses project fully inside
                                     the frame) and handles.read_handle_v5 (P14: direction white -> orange paint in
                                     the blade plane); plus P14's own-gripper guard (own_gripper_mask: no answer when
                                     either hypothesis is > 15 % behind the gripper; gripper pixels blanked).  On the
                                     finished world v4 alone reversed 31 / 400 frames (lit jaw, light panel face
                                     taken for the blade), v4 + guard 2, v5 alone 7, the agreement 0 (records P2_P14)

Their sha256 are pinned in `perception.READER_SHA256` (written into records/P2_P14.md; P19 checks them).  The tag and
handle readers assume the camera roll relative to the panel is within +-45 deg (G4 section 8): with the vehicle rolled
90 deg about the gripper axis (turning a valve) only the code decoder is valid -- `read_panel` refuses the other two
when `roll_deg` says so.

Panel standard numbering (the as-designed tag list; the code of a plate is its index in this list):
    BV1..BV6, PV1..PV6, TV1..TV6, TP1..TP6, PARK-A, PARK-B, CAGE (0..26); codewords 27..47 are spares.
"Code first; the painted tag only when the code is covered, cross-checked against the neighbours" (panel standard):
`read_panel` returns codes, and painted tags only for plates whose code was not decoded.
"""
from __future__ import annotations

import math

import numpy as np

from . import codes as CD
from . import handles as HD
from . import imgproc as IP
from . import panel_std as PS
from . import tagtext as TT

MAX_ERRORS = 2
TAGS = [f"{k}V{i}" for k in "BPT" for i in range(1, 7)] + [f"TP{i}" for i in range(1, 7)] + ["PARK-A", "PARK-B", "CAGE"]
TAG_OF_CODE = {i: t for i, t in enumerate(TAGS)}
CODE_OF_TAG = {t: i for i, t in enumerate(TAGS)}
MAX_ROLL_DEG_TEXT = 45.0
MAX_BLADE_HIDDEN = 0.15


def decode(rgb):
    """[(code_id, corners_px (4, 2) TL TR BR BL)] with the fixed correction limit."""
    return CD.decode(rgb, max_errors=MAX_ERRORS)


def decode_verbose(rgb):
    return CD.decode_verbose(rgb, max_errors=MAX_ERRORS)


def read_tags(rgb, exclude=None):
    """[(tag, plate_corners_px (4, 2))] -- the painted-tag reader (v3 = v2 + light-face plate candidates).  `exclude` (own_gripper_mask): those pixels
    are blanked first (P14: a lit jaw just below a plate merged with it into one bright component and spoiled its
    outline)."""
    if exclude is not None and exclude.any():
        rgb = rgb.copy()
        rgb[exclude] = 0
    return TT.read_tags_v3(rgb)


def own_gripper_mask(man, tilt_rad: float, shape, pad_px: float = 3.0):
    """Pixels covered by the vehicle's own gripper (the manual's drawing) for a camera tilt: independent of the
    vehicle attitude (body-fixed gripper, body-fixed tilt hinge)."""
    from .blade import jaw_mask
    from .pose import Cam
    H, W = shape
    return jaw_mask(Cam(man, tilt_rad, np.eye(3), W, H), (H, W), man.outline, pad_px=pad_px)


def blade_hidden_fraction(shape, local_xy, image_uv, hfov_deg: float, exclude) -> dict:
    """Fraction of each blade hypothesis' projected box (open / closed) that falls in `exclude`."""
    H, W = shape
    K = PS.intrinsics(W, H, hfov_deg)
    Hm = IP.homography(np.asarray(local_xy, float), np.asarray(image_uv, float))
    R, t = HD.pose_from_homography(Hm, K)
    out = {}
    for st in ("open", "closed"):
        uv, z = HD.project(K, R, t, HD.blade_box(st))
        m, org = IP.convex_poly_mask((H, W), uv)
        if m is None or not m.any():
            out[st] = 0.0
            continue
        r0, c0 = org
        sub = exclude[r0:r0 + m.shape[0], c0:c0 + m.shape[1]]
        out[st] = float((m & sub).sum()) / float(m.sum())
    return out


def read_handle(rgb, local_xy, image_uv, hfov_deg: float, exclude=None, max_hidden: float = MAX_BLADE_HIDDEN):
    """('open' | 'closed' | None, info) from panel-local anchor points (code corners and / or plate corners in the
    plate's own stem frame, panel_std conventions) and their image positions: v4 and v5 must agree; the own-gripper
    guard when `exclude` (own_gripper_mask) is given."""
    local_xy, image_uv = np.asarray(local_xy, float), np.asarray(image_uv, float)
    if exclude is not None:
        hid = blade_hidden_fraction(rgb.shape[:2], local_xy, image_uv, hfov_deg, exclude)
        if max(hid.values()) > max_hidden:
            return None, {"reason": "blade_behind_own_gripper", "hidden": hid}
        if exclude.any():
            rgb = rgb.copy()
            rgb[exclude] = 0
    a4, i4 = HD.read_handle_v4(rgb, local_xy, image_uv, float(hfov_deg))
    if a4 is None:
        return None, dict(i4, by="v4")
    a5, i5 = HD.read_handle_v5(rgb, local_xy, image_uv, float(hfov_deg), exclude=exclude)
    info = dict(v4=i4, v5=i5, v4_state=a4, v5_state=a5)
    if a5 != a4:
        return None, dict(info, reason="v4_v5_disagree" if a5 is not None else "v5_no_answer")
    return a4, info


def read_handle_from_code(rgb, corners, hfov_deg: float, exclude=None):
    return read_handle(rgb, PS.marker_corners_local(), corners, hfov_deg, exclude=exclude)


def read_handle_from_plate(rgb, plate_corners, hfov_deg: float, exclude=None):
    return read_handle(rgb, PS.plate_corners_local(), plate_corners, hfov_deg, exclude=exclude)


def _quad_contains(quad, pt) -> bool:
    q = np.asarray(quad, float)
    s = None
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        c = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(c) < 1e-12:
            continue
        if s is None:
            s = c > 0
        elif (c > 0) != s:
            return False
    return True


def read_panel(rgb, hfov_deg: float, roll_deg: float = 0.0, handles: bool = True, exclude=None) -> list:
    """Everything the fixed readers see in one frame: one dict per plate --
    tag, code_id (None for a painted-tag-only read), source ('code' | 'text'), corners (code corners, or the plate
    corners for a text read), handle ('open' | 'closed' | None) and handle_info.  Painted tags are only reported for
    plates whose code did not decode (a text read whose plate quad contains a decoded code's centre is dropped), and
    only when |roll_deg| <= 45 (so are handle states).  `exclude` = own_gripper_mask for the frame's tilt."""
    out = []
    upright = abs(math.remainder(float(roll_deg), 360.0)) <= MAX_ROLL_DEG_TEXT
    for cid, c in decode(rgb):
        r = dict(tag=TAG_OF_CODE.get(int(cid)), code_id=int(cid), source="code", corners=np.asarray(c, float),
                 handle=None, handle_info=None)
        if handles and upright and r["tag"] is not None and r["tag"][1:2] == "V":
            r["handle"], r["handle_info"] = read_handle_from_code(rgb, c, hfov_deg, exclude=exclude)
        out.append(r)
    if upright:
        centres = [r["corners"].mean(0) for r in out]
        for tag, q in read_tags(rgb, exclude=exclude):
            if any(_quad_contains(q, cc) for cc in centres):
                continue
            r = dict(tag=tag, code_id=None, source="text", corners=np.asarray(q, float), handle=None, handle_info=None)
            if handles:
                r["handle"], r["handle_info"] = read_handle_from_plate(rgb, q, hfov_deg, exclude=exclude)
            out.append(r)
    return out
