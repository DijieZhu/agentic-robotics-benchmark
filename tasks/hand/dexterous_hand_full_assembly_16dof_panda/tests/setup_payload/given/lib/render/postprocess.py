"""Post-processing applied IDENTICALLY after any backend: the "camera + photo" half of the pipeline.

A backend only renders ideal pinhole images (display-referred sRGB uint8, no distortion, no noise) of the RAW cameras
that `plan_view` derives from the requested cameras; everything that makes a render look like a photograph happens
here, in numpy, deterministically from (request seed, view index):

    plan_view(camera, photo, native_distortion=False) -> ViewPlan
        raw camera = same pose and focal lengths, distortion removed, canvas enlarged by the margins lens distortion
        pulls into view, scaled by PhotoSpec.supersample
    finish_view(raw_rgb, plan, photo, seed, view_index, raw_depth=None, raw_seg=None) -> FinishedView
        1. box-filter the supersampled canvas down (depth / segmentation: centre sample)
        2. lens distortion: inverse-map every output pixel through CameraSpec.undistort_normalised, bilinear sample
           (depth / segmentation: nearest)                                   [skipped if the backend distorts natively]
        3. sRGB -> linear
        4. exposure 2**EV and white-balance gains (CameraSpec)
        5. vignetting 1 - v (r / r_corner)^2 about the principal point
        6. blur: one kernel = uniform disk (defocus radius) convolved with a line (motion length / angle), FFT with
           reflect padding
        7. sensor noise N(0, shot * I + read^2) per channel and pixel (linear light), clip to [0, 1]
        8. linear -> sRGB
        9. colour jitter (brightness, contrast, saturation, hue, gamma, tint), per view from the seed (or one fixed draw:
           ColourJitterSpec.seed)
       10. quantise (round half to even) to uint8
       11. encode PNG (zlib level PhotoSpec.png_compress_level, default 6, no metadata) or JPEG (quality / subsampling,
           baseline, no EXIF) and DECODE again:
           FinishedView.rgb is exactly what the agent's file contains
    apply_photometric(rgb_uint8, camera, photo, seed, view_index)   steps 3-10 on an already-geometric image

    Depth sensor (RA1; `sensor_depth` output, AGENT-VISIBLE): after step 2 the exact depth of the view (GRADER-ONLY,
    centre sample of the supersampled canvas, warped with the SAME distortion table as rgb, nearest) goes through
    apply_depth_sensor(depth, camera, PhotoSpec.depth_sensor, seed, view_index) -> float32 metres, 0 = invalid
    (the DepthSensorSpec docstring lists the seven steps), and is encoded as a 16-bit PNG of depth / quantisation_m
    (encode_depth_png / decode_depth_png; no metadata).  FinishedView.sensor_depth equals the decoded file exactly.

Random streams: spec.rng(seed, "view", view_index, <stage>) with stages "noise" and "jitter" (rgb) and "depth_noise",
"depth_dropout", "depth_frame" (+ "depth_edge" when edge_ragged_px > 0) (sensor depth): independent of the backend, of
the other views and of each other, and every value is drawn even when its range is 0, so changing one range never
re-draws another and requesting sensor depth never changes an rgb byte.  The depth unit's FIXED errors come from
spec.rng(DepthSensorSpec.unit_seed, "depth_unit", camera name, "wiggle" | "pattern" | "pixel"): constant over frames.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .spec import CameraSpec, DepthSensorSpec, PhotoSpec, rng

POSTPROCESS_VERSION = "arb-postprocess/1.0.0"
DEPTH_SENSOR_VERSION = "arb-depth-sensor/1.0.0"     # recorded in Provenance.sensor_depth_model when sensor_depth is produced
DEPTH_PNG_COMPRESS_LEVEL = 1                         # noisy 16-bit depth: level 1 = 3.4 ms / 80 kB vs level 6 = 20 ms / 74 kB (320 x 240)
_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


@dataclass
class ViewPlan:
    camera: CameraSpec                 # the requested (final) camera
    raw_camera: CameraSpec             # what the backend renders
    supersample: int
    margins: Tuple[int, int, int, int]  # left, top, right, bottom in final-resolution pixels
    warp: bool                         # True if step 2 (lens distortion) runs here


@dataclass
class FinishedView:
    rgb: np.ndarray
    encoded: bytes
    depth: Optional[np.ndarray] = None
    segmentation: Optional[np.ndarray] = None
    sensor_depth: Optional[np.ndarray] = None          # float32 metres, 0 = invalid (agent-visible)
    sensor_depth_encoded: Optional[bytes] = None       # 16-bit PNG (b"" when encoding was skipped)


# ----------------------------------------------------------------------------- geometry
def distortion_margins(camera: CameraSpec, pad: int = 2) -> Tuple[int, int, int, int]:
    """Pixels an undistorted pinhole canvas (same fx fy cx cy) needs beyond the image on each side so that every
    distorted output pixel samples inside it."""
    if not camera.has_distortion:
        return (0, 0, 0, 0)
    W, H = camera.width, camera.height
    n = 4 * max(W, H)
    us = np.concatenate([np.linspace(-0.5, W - 0.5, n), np.full(n, W - 0.5), np.linspace(-0.5, W - 0.5, n), np.full(n, -0.5)])
    vs = np.concatenate([np.full(n, -0.5), np.linspace(-0.5, H - 0.5, n), np.full(n, H - 0.5), np.linspace(-0.5, H - 0.5, n)])
    x, y = camera.undistort_normalised((us - camera.cx) / camera.fx, (vs - camera.cy) / camera.fy)
    uu, vv = camera.fx * x + camera.cx, camera.fy * y + camera.cy
    left = max(0, int(math.ceil(-0.5 - uu.min())) + pad)
    right = max(0, int(math.ceil(uu.max() - (W - 0.5))) + pad)
    top = max(0, int(math.ceil(-0.5 - vv.min())) + pad)
    bottom = max(0, int(math.ceil(vv.max() - (H - 0.5))) + pad)
    return (left, top, right, bottom)


def plan_view(camera: CameraSpec, photo: PhotoSpec, native_distortion: bool = False) -> ViewPlan:
    s = int(photo.supersample)
    warp = camera.has_distortion and not native_distortion
    ml, mt, mr, mb = distortion_margins(camera) if warp else (0, 0, 0, 0)
    W, H = camera.width + ml + mr, camera.height + mt + mb
    raw = camera.copy(name=camera.name, width=s * W, height=s * H, fx=s * camera.fx, fy=s * camera.fy,
                      cx=s * (camera.cx + ml + 0.5) - 0.5, cy=s * (camera.cy + mt + 0.5) - 0.5,
                      exposure_ev=0.0, white_balance=[1.0, 1.0, 1.0])
    if warp:
        raw = raw.copy(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)
    return ViewPlan(camera=camera, raw_camera=raw, supersample=s, margins=(ml, mt, mr, mb), warp=warp)


def _box_down(img, s):
    if s == 1:
        return img
    H, W = img.shape[0] // s, img.shape[1] // s
    x = img[:H * s, :W * s].reshape(H, s, W, s, *img.shape[2:]).astype(np.float64)
    return x.mean(axis=(1, 3))


def _centre_down(img, s):
    return img if s == 1 else img[s // 2::s, s // 2::s]


class _WarpTable:
    """Precomputed inverse lens-distortion map of one camera: gather indices + bilinear weights (+ nearest index)."""

    def __init__(self, camera: CameraSpec, margins, canvas_hw):
        ch, cw = canvas_hw
        mu, mv = _warp_maps(camera, margins)
        u = np.clip(mu, 0, cw - 1)
        v = np.clip(mv, 0, ch - 1)
        u0 = np.minimum(np.floor(u).astype(np.int64), max(cw - 2, 0))
        v0 = np.minimum(np.floor(v).astype(np.int64), max(ch - 2, 0))
        fu, fv = (u - u0)[..., None], (v - v0)[..., None]
        self.i00 = (v0 * cw + u0).reshape(-1)
        self.cw = cw
        self.w = [((1 - fu) * (1 - fv)).reshape(-1, 1), (fu * (1 - fv)).reshape(-1, 1), ((1 - fu) * fv).reshape(-1, 1), (fu * fv).reshape(-1, 1)]
        self.nearest = (np.clip(np.rint(mv).astype(np.int64), 0, ch - 1) * cw + np.clip(np.rint(mu).astype(np.int64), 0, cw - 1)).reshape(-1)
        self.shape = (camera.height, camera.width)

    def bilinear(self, img):
        C = img.shape[2]
        flat = img.reshape(-1, C)
        i = self.i00
        out = flat[i] * self.w[0] + flat[i + 1] * self.w[1] + flat[i + self.cw] * self.w[2] + flat[i + self.cw + 1] * self.w[3]
        return out.reshape(self.shape + (C,))

    def nearest_of(self, img):
        return img.reshape(-1)[self.nearest].reshape(self.shape)


_WARP_CACHE: "dict" = {}
_WARP_CACHE_MAX = 4


def warp_table(camera: CameraSpec, margins, canvas_hw) -> _WarpTable:
    key = (camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy, tuple(camera.distortion), tuple(margins), tuple(canvas_hw))
    t = _WARP_CACHE.get(key)
    if t is None:
        if len(_WARP_CACHE) >= _WARP_CACHE_MAX:
            _WARP_CACHE.pop(next(iter(_WARP_CACHE)))
        t = _WARP_CACHE[key] = _WarpTable(camera, margins, canvas_hw)
    return t


def _warp_maps(camera: CameraSpec, margins):
    ml, mt = margins[0], margins[1]
    vv, uu = np.mgrid[0:camera.height, 0:camera.width].astype(np.float64)
    x, y = camera.undistort_normalised((uu - camera.cx) / camera.fx, (vv - camera.cy) / camera.fy)
    return camera.fx * x + camera.cx + ml, camera.fy * y + camera.cy + mt


# ----------------------------------------------------------------------------- photometric
def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def disk_kernel(radius, oversample=8):
    if radius <= 0:
        return np.ones((1, 1))
    R = int(math.ceil(radius))
    n = 2 * R + 1
    o = (np.arange(n * oversample) + 0.5) / oversample - (R + 0.5)
    yy, xx = np.meshgrid(o, o, indexing="ij")
    inside = (xx * xx + yy * yy <= radius * radius).astype(float)
    k = inside.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
    if k.sum() <= 0:                      # a disk smaller than one sub-sample: no blur
        return np.ones((1, 1))
    return k / k.sum()


def line_kernel(length, angle_deg, samples=None):
    if length <= 0:
        return np.ones((1, 1))
    R = int(math.ceil(length / 2)) + 1
    n = 2 * R + 1
    k = np.zeros((n, n))
    a = math.radians(angle_deg)
    m = samples or max(16, int(length * 16))
    t = np.linspace(-length / 2, length / 2, m)
    xs, ys = R + t * math.cos(a), R - t * math.sin(a)          # screen: +u right, +v down; angle counter-clockwise
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (np.clip(y0 + dy, 0, n - 1), np.clip(x0 + dx, 0, n - 1)), w)
    return k / k.sum()


def _conv2(a, b):
    ha, wa = a.shape
    hb, wb = b.shape
    out = np.zeros((ha + hb - 1, wa + wb - 1))
    for i in range(hb):
        for j in range(wb):
            if b[i, j]:
                out[i:i + ha, j:j + wa] += a * b[i, j]
    return out


def blur_kernel(photo: PhotoSpec) -> np.ndarray:
    k = disk_kernel(photo.blur.defocus_px)
    if photo.blur.motion_px > 0:
        k = _conv2(k, line_kernel(photo.blur.motion_px, photo.blur.motion_angle_deg))
    if k.shape[0] % 2 == 0:                                   # keep odd, centred
        k = np.pad(k, ((0, 1), (0, 1)))
    return k / k.sum()


def convolve(img, kern):
    if kern.shape == (1, 1):
        return img
    if kern.size <= 81:                                        # small kernel: direct shifted sum (exact, fast)
        kh, kw = kern.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
        H, W = img.shape[:2]
        out = np.zeros_like(img)
        for i in range(kh):
            for j in range(kw):
                w = kern[kh - 1 - i, kw - 1 - j]
                if w:
                    out += w * padded[i:i + H, j:j + W]
        return out
    kh, kw = kern.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    H, W = padded.shape[:2]
    K = np.zeros((H, W))
    K[:kh, :kw] = kern
    K = np.roll(K, (-ph, -pw), axis=(0, 1))
    out = np.fft.irfft2(np.fft.rfft2(padded, axes=(0, 1)) * np.fft.rfft2(K)[..., None], s=(H, W), axes=(0, 1))
    return out[ph:ph + img.shape[0], pw:pw + img.shape[1]]


_YIQ = np.array([[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.311135]])
_YIQ_INV = np.linalg.inv(_YIQ)


def jitter_values(photo: PhotoSpec, seed: int, view_index: int):
    if photo.colour_jitter.seed is not None:           # a camera unit's fixed colour response (ColourJitterSpec.seed)
        r = rng(int(photo.colour_jitter.seed), "unit", "jitter")
    else:
        r = rng(seed, "view", view_index, "jitter")
    u = r.uniform(-1.0, 1.0, size=8)
    cj = photo.colour_jitter
    return {"brightness": 1 + cj.brightness * u[0], "contrast": 1 + cj.contrast * u[1], "saturation": 1 + cj.saturation * u[2],
            "hue_deg": cj.hue_deg * u[3], "gamma": 1 + cj.gamma * u[4], "tint": [1 + cj.tint * u[5], 1 + cj.tint * u[6], 1 + cj.tint * u[7]]}


def _colour_jitter(img, jv):
    f = np.float32
    if (jv["brightness"], jv["contrast"], jv["saturation"], jv["hue_deg"], jv["gamma"]) == (1.0, 1.0, 1.0, 0.0, 1.0) and jv["tint"] == [1.0, 1.0, 1.0]:
        return img
    img = img * f(jv["brightness"])
    mean = f(img.mean(dtype=np.float64))
    img = (img - mean) * f(jv["contrast"]) + mean
    luma = (img @ _YIQ[0].astype(np.float32))[..., None]
    img = luma + (img - luma) * f(jv["saturation"])
    if jv["hue_deg"]:
        a = math.radians(jv["hue_deg"])
        rot = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        img = img @ (_YIQ_INV @ rot @ _YIQ).T.astype(np.float32)
    img = np.clip(img, 0, 1) ** f(jv["gamma"])
    img = img * np.asarray(jv["tint"], np.float32)[None, None, :]
    return np.clip(img, 0, 1)


def _is_identity(camera: CameraSpec, photo: PhotoSpec) -> bool:
    cj = photo.colour_jitter
    return (camera.exposure_ev == 0 and list(camera.white_balance) == [1.0, 1.0, 1.0] and photo.vignetting == 0
            and photo.blur.defocus_px == 0 and photo.blur.motion_px == 0 and photo.noise.shot == 0 and photo.noise.read == 0
            and not any((cj.brightness, cj.contrast, cj.saturation, cj.hue_deg, cj.gamma, cj.tint)))


_SRGB8_TO_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)   # exact per code value


def _gain_only(photo: PhotoSpec) -> bool:
    cj = photo.colour_jitter
    return (photo.vignetting == 0 and photo.blur.defocus_px == 0 and photo.blur.motion_px == 0 and photo.noise.shot == 0
            and photo.noise.read == 0 and not any((cj.brightness, cj.contrast, cj.saturation, cj.hue_deg, cj.gamma, cj.tint)))


def apply_photometric(rgb, camera: CameraSpec, photo: PhotoSpec, seed: int, view_index: int, fast: bool = True) -> np.ndarray:
    """Steps 3-10 on a geometric (final-size) image: float or uint8 display-referred sRGB in, uint8 out.  The identity and
    gain-only fast paths give the same bytes as the general chain (`fast=False` forces it; tested)."""
    a = np.asarray(rgb)
    if fast and _is_identity(camera, photo):
        if a.dtype == np.uint8:
            return a.copy()
        return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    g = np.float32(2.0 ** camera.exposure_ev) * np.asarray(camera.white_balance, np.float32)    # exposure x white balance
    if fast and a.dtype == np.uint8 and _gain_only(photo):
        # exposure / white balance only: a per-channel 256-entry table, bit-identical to the general path per code value
        out = np.empty_like(a)
        for c in range(3):
            lut = np.clip(np.rint(linear_to_srgb(np.clip(_SRGB8_TO_LINEAR * g[c], 0.0, 1.0)).astype(np.float32) * np.float32(255.0)), 0, 255)
            out[..., c] = lut.astype(np.uint8)[a[..., c]]
        return out
    H, W = a.shape[:2]
    lin = _SRGB8_TO_LINEAR[a] if a.dtype == np.uint8 else srgb_to_linear(a.astype(np.float32))    # float input: 0..1
    lin = lin * g[None, None, :]
    if photo.vignetting:
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = (uu - np.float32(camera.cx)) ** 2 + (vv - np.float32(camera.cy)) ** 2
        corners = [(-0.5, -0.5), (W - 0.5, -0.5), (-0.5, H - 0.5), (W - 0.5, H - 0.5)]
        rc2 = max((cu - camera.cx) ** 2 + (cv - camera.cy) ** 2 for cu, cv in corners)
        lin = lin * (np.float32(1.0) - np.float32(photo.vignetting) * (r2 / np.float32(rc2)))[..., None]
    lin = convolve(lin, blur_kernel(photo)).astype(np.float32)
    if photo.noise.shot or photo.noise.read:
        r = rng(seed, "view", view_index, "noise")
        sigma = np.sqrt(np.clip(lin, 0, None) * np.float32(photo.noise.shot) + np.float32(photo.noise.read ** 2))
        lin = lin + r.standard_normal(lin.shape, dtype=np.float32) * sigma
    out = linear_to_srgb(np.clip(lin, 0.0, 1.0)).astype(np.float32)
    out = _colour_jitter(out, jitter_values(photo, seed, view_index))
    return np.clip(np.rint(out * np.float32(255.0)), 0, 255).astype(np.uint8)


def encode(rgb: np.ndarray, photo: PhotoSpec) -> bytes:
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(rgb, np.uint8), "RGB")
    buf = io.BytesIO()
    if photo.image_format == "png":
        im.save(buf, format="PNG", compress_level=int(photo.png_compress_level), optimize=False)
    else:
        im.save(buf, format="JPEG", quality=int(photo.jpeg_quality), subsampling=_SUBSAMPLING[photo.jpeg_subsampling],
                optimize=False, progressive=False)
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), np.uint8).copy()


# ----------------------------------------------------------------------------- depth sensor (RA1)
def _disk_offsets(r: int):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a disk of `radius` pixels (shifted ORs; exact, no wrap-around)."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    H, W = mask.shape
    out = mask.copy()
    for dy, dx in _disk_offsets(int(radius)):
        if dy == 0 and dx == 0:
            continue
        ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
        xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
        out[yd, xd] |= mask[ys, xs]
    return out


def depth_discontinuities(depth: np.ndarray, rel: float, abs_m: float = 0.0) -> np.ndarray:
    """Pixels whose exact depth differs from a 4-neighbour by more than rel * min(z) + abs_m, or that border a pixel with
    no surface (depth 0): the flying pixels / occlusion boundaries a real RGB-D sensor cannot measure."""
    z = np.asarray(depth, np.float64)
    hit = z > 0
    edge = np.zeros(z.shape, bool)
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))), ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        za, zb, ha, hb = z[a], z[b], hit[a], hit[b]
        jump = (ha != hb) | (ha & hb & (np.abs(za - zb) > rel * np.minimum(za, zb) + abs_m))
        edge[a] |= jump & ha
        edge[b] |= jump & hb
    return edge


_FIELD_CACHE: "dict" = {}
_FIELD_CACHE_MAX = 8


def unit_field(unit_seed: int, camera_name: str, height: int, width: int, cell_px: int) -> np.ndarray:
    """The smooth per-unit N(0, 1) field of the fixed-pattern error: i.i.d. N(0, 1) nodes on a grid of cell_px pixels
    (stream rng(unit_seed, "depth_unit", camera_name, "pattern")), bilinear between the nodes and divided by
    sqrt(sum w^2), so every pixel has unit variance.  Constant over frames; cached."""
    key = (int(unit_seed), str(camera_name), int(height), int(width), int(cell_px))
    f = _FIELD_CACHE.get(key)
    if f is None:
        c = float(cell_px)
        gh, gw = int(math.ceil((height - 1) / c)) + 2, int(math.ceil((width - 1) / c)) + 2
        g = rng(unit_seed, "depth_unit", camera_name, "pattern").standard_normal((gh, gw))
        v, u = np.arange(height) / c, np.arange(width) / c
        v0, u0 = np.floor(v).astype(np.int64), np.floor(u).astype(np.int64)
        fv, fu = (v - v0)[:, None], (u - u0)[None, :]
        w00, w01, w10, w11 = (1 - fv) * (1 - fu), (1 - fv) * fu, fv * (1 - fu), fv * fu
        num = (w00 * g[v0][:, u0] + w01 * g[v0][:, u0 + 1] + w10 * g[v0 + 1][:, u0] + w11 * g[v0 + 1][:, u0 + 1])
        f = num / np.sqrt(w00 ** 2 + w01 ** 2 + w10 ** 2 + w11 ** 2)
        if len(_FIELD_CACHE) >= _FIELD_CACHE_MAX:
            _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
        _FIELD_CACHE[key] = f
    return f


def depth_unit_error(z, camera: CameraSpec, sensor: DepthSensorSpec) -> Optional[np.ndarray]:
    """Step 1's per-unit FIXED errors in metres (None when off): wiggle + smooth field + per-pixel offsets, seeded by
    (sensor.unit_seed, camera.name) and never by the frame seed, so they survive frame averaging.  A camera keeps its
    unit's pattern as long as its name is stable (live cameras: name them by mount)."""
    if not sensor.has_fixed_errors:
        return None
    z = np.asarray(z, np.float64)
    H, W = z.shape
    e = np.zeros(z.shape)
    if sensor.wiggle_amp_m > 0:
        phi = 2.0 * math.pi * float(rng(sensor.unit_seed, "depth_unit", camera.name, "wiggle").random())
        e += sensor.wiggle_amp_m * np.sin(2.0 * math.pi * z / sensor.wiggle_period_m + phi)
    if sensor.fixed_pattern_a_m > 0 or sensor.fixed_pattern_b_per_m > 0:
        e += unit_field(sensor.unit_seed, camera.name, H, W, sensor.fixed_pattern_cell_px) * (
            sensor.fixed_pattern_a_m + sensor.fixed_pattern_b_per_m * z * z)
    if sensor.pixel_offset_sigma_m > 0:
        e += _unit_pixels(sensor.unit_seed, camera.name, H, W) * sensor.pixel_offset_sigma_m
    return e


def _unit_pixels(unit_seed: int, camera_name: str, height: int, width: int) -> np.ndarray:
    """Per-pixel N(0, 1) offsets of a depth unit (stream rng(unit_seed, "depth_unit", camera_name, "pixel")); cached."""
    key = ("pixel", int(unit_seed), str(camera_name), int(height), int(width))
    f = _FIELD_CACHE.get(key)
    if f is None:
        f = rng(unit_seed, "depth_unit", camera_name, "pixel").standard_normal((int(height), int(width)))
        if len(_FIELD_CACHE) >= _FIELD_CACHE_MAX:
            _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
        _FIELD_CACHE[key] = f
    return f


def depth_sensor_counts(depth, camera: CameraSpec, sensor: DepthSensorSpec, seed: int, view_index: int) -> np.ndarray:
    """The sensor model (DepthSensorSpec docstring) on an exact depth map of the final camera: uint16 counts of
    sensor.quantisation_m, 0 = invalid.  Pure function of (depth, camera, sensor, seed, view_index); camera enters only
    through its name (the unit's fixed errors) - never through its intrinsics."""
    z = np.asarray(depth, np.float64)
    hit = z > 0
    noise = rng(seed, "view", view_index, "depth_noise").standard_normal(z.shape)
    u_drop = rng(seed, "view", view_index, "depth_dropout").random(z.shape)
    u_frame = float(rng(seed, "view", view_index, "depth_frame").random())
    zm = sensor.scale * z + sensor.offset_m
    fixed = depth_unit_error(z, camera, sensor)
    if fixed is not None:
        zm = zm + np.where(hit, fixed, 0.0)
    zm = zm + noise * (sensor.sigma_a_m + sensor.sigma_b_per_m * z * z)
    if sensor.disparity_step_px > 0:
        fb = float(sensor.disparity_focal_px) * sensor.baseline_m      # the depth imager's geometry, not the colour camera's
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.where(zm > 0, fb / np.where(zm > 0, zm, 1.0), 0.0)
            disp = np.round(disp / sensor.disparity_step_px) * sensor.disparity_step_px
            zm = np.where(disp > 0, fb / np.where(disp > 0, disp, 1.0), 0.0)
    valid = hit & (zm >= sensor.min_range_m) & (zm <= sensor.max_range_m)
    counts = np.rint(np.where(valid, zm, 0.0) / sensor.quantisation_m)
    valid &= counts > 0
    edge = depth_discontinuities(z, sensor.edge_threshold_rel, sensor.edge_threshold_m)
    band = dilate(edge, sensor.edge_invalid_px)
    if sensor.edge_ragged_px > 0 and edge.any():
        u_edge = rng(seed, "view", view_index, "depth_edge").random(z.shape)
        R = int(sensor.edge_ragged_px)
        inner = band
        for r in range(1, R + 1):                  # ring r beyond the band is lost with probability (R + 1 - r) / (R + 1)
            outer = dilate(edge, sensor.edge_invalid_px + r)
            band = band | (outer & ~inner & (u_edge < (R + 1 - r) / (R + 1.0)))
            inner = outer
    valid &= ~band
    drop = u_drop < sensor.dropout_rate
    valid &= ~dilate(drop, sensor.dropout_blob_px)
    if u_frame < sensor.frame_dropout_rate:
        valid[:] = False
    return np.where(valid, np.clip(counts, 0, 65535), 0).astype(np.uint16)


def apply_depth_sensor(depth, camera: CameraSpec, sensor: DepthSensorSpec, seed: int, view_index: int) -> np.ndarray:
    """float32 metres (0 = invalid) = depth_sensor_counts(...) * quantisation_m (what decode_depth_png returns)."""
    return counts_to_metres(depth_sensor_counts(depth, camera, sensor, seed, view_index), sensor.quantisation_m)


def counts_to_metres(counts: np.ndarray, unit_m: float) -> np.ndarray:
    return (np.asarray(counts, np.float64) * float(unit_m)).astype(np.float32)


def encode_depth_png(counts: np.ndarray) -> bytes:
    """uint16 counts -> 16-bit greyscale PNG (zlib level DEPTH_PNG_COMPRESS_LEVEL, no text chunks): the RGB-D
    convention (e.g. millimetres)."""
    from PIL import Image
    a = np.ascontiguousarray(counts, np.uint16)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", compress_level=DEPTH_PNG_COMPRESS_LEVEL, optimize=False)
    return buf.getvalue()


def decode_depth_png(data: bytes, unit_m: Optional[float] = None) -> np.ndarray:
    """16-bit PNG -> uint16 counts, or float32 metres when `unit_m` is given (the agent's reader does exactly this)."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        a = np.asarray(im, dtype=np.uint16).copy()
    return a if unit_m is None else counts_to_metres(a, unit_m)


def finish_view(raw_rgb, plan: ViewPlan, photo: PhotoSpec, seed: int, view_index: int, raw_depth=None, raw_seg=None,
                encode_png: bool = True, sensor_depth: bool = False) -> FinishedView:
    """Steps 1-11.  encode_png=False (sessions at policy rate) skips the PNG file only; JPEG is always encoded because its
    artefacts are part of the photo.  sensor_depth=True also runs the depth sensor (needs raw_depth and
    photo.depth_sensor); its PNG is skipped with encode_png=False as well."""
    cam = plan.camera
    s = plan.supersample
    exp_shape = (plan.raw_camera.height, plan.raw_camera.width)
    if tuple(np.asarray(raw_rgb).shape[:2]) != exp_shape:
        raise ValueError("backend returned %s for raw camera %s (expected %s)" % (np.asarray(raw_rgb).shape[:2], cam.name, exp_shape))
    if s == 1 and not plan.warp:
        canvas = np.asarray(raw_rgb, np.uint8)       # uint8 straight into the photometric chain (identity = exact copy)
    else:
        canvas = (_box_down(np.asarray(raw_rgb, np.float64), s) / 255.0).astype(np.float32)
    depth = None if raw_depth is None else _centre_down(np.asarray(raw_depth, np.float32), s)
    seg = None if raw_seg is None else _centre_down(np.asarray(raw_seg, np.int32), s)
    if plan.warp:
        table = warp_table(cam, plan.margins, canvas.shape[:2])
        canvas = table.bilinear(canvas)
        if depth is not None:
            depth = table.nearest_of(depth)
        if seg is not None:
            seg = table.nearest_of(seg)
    elif canvas.shape[:2] != (cam.height, cam.width):
        raise ValueError("canvas / camera size mismatch")
    rgb8 = apply_photometric(canvas, cam, photo, seed, view_index)
    data = encode(rgb8, photo) if (encode_png or photo.image_format == "jpeg") else b""
    final = decode(data) if photo.image_format == "jpeg" else rgb8
    sd = sd_bytes = None
    if sensor_depth:
        if depth is None or photo.depth_sensor is None:
            raise ValueError("sensor depth needs the backend's exact depth and PhotoSpec.depth_sensor")
        counts = depth_sensor_counts(depth, cam, photo.depth_sensor, seed, view_index)
        sd = counts_to_metres(counts, photo.depth_sensor.quantisation_m)
        sd_bytes = encode_depth_png(counts) if encode_png else b""
    return FinishedView(rgb=final, encoded=data, depth=None if depth is None else depth.astype(np.float32),
                        segmentation=None if seg is None else seg.astype(np.int32), sensor_depth=sd, sensor_depth_encoded=sd_bytes)
