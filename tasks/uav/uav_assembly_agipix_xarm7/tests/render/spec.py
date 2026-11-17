"""The render layer's request / result contract: JSON-serialisable dataclasses shared by every backend.

    SceneSpec      what to render: an MJCF path or an MjSpec builder reference + material / texture / lighting overrides,
                   the physical scale references and the state (keyframe / qpos / named joints)
    CameraSpec     intrinsics fx fy cx cy width height (pixels), extrinsics world_T_cam (4 x 4), lens distortion
                   k1 k2 p1 p2 k3, exposure (EV) and white balance (linear per-channel gains)
    PhotoSpec      post-processing applied after ANY backend (postprocess.py): sensor noise, motion / defocus blur,
                   vignetting, colour jitter, JPEG / PNG encoding, supersampling
    ScaleRef       a scale reference placed as a PHYSICAL object in the scene (scale_refs.py builds its geometry and texture)
    RenderRequest  scene + cameras + photo spec + seed + backend name (+ requested outputs)
    RenderResult   rgb (+ optional depth, segmentation: GRADER-ONLY; + optional sensor_depth: AGENT-VISIBLE) + provenance
                   (backend name / version, request digest, LOD record)
    DepthSensorSpec  (inside PhotoSpec.depth_sensor) the RGB-D sensor model that turns the grader-only exact depth into the
                   agent-visible `sensor_depth` output (postprocess.apply_depth_sensor)

Conventions (normative for every backend):

* Units: metres, radians inside matrices; `*_deg` / `*_mm` / `*_px` fields say their unit in the name.
* World frame: z up (MuJoCo's).
* Camera frame = OpenCV: +x image right, +y image DOWN, +z forward (optical axis).  `world_T_cam` maps camera-frame
  points to world (columns 0..2 of its rotation are the camera x, y, z axes in world; column 3 is the camera centre).
  The MuJoCo backend converts to MuJoCo's camera frame (looks along -z, +y up) itself; nothing else should.
* Pixel coordinates = OpenCV: origin at the CENTRE of the top-left pixel, u right, v down; the image spans
  [-0.5, W - 0.5] x [-0.5, H - 0.5]; the ideal principal point is ((W - 1) / 2, (H - 1) / 2).
* Distortion = OpenCV radial-tangential (Brown-Conrady) on normalised coordinates (x, y) = (X / Z, Y / Z):
  x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2),  y_d = y (1 + ...) + p1 (r^2 + 2 y^2) + 2 p2 x y;
  u = fx x_d + cx, v = fy y_d + cy.  (`CameraSpec.project` implements exactly this.)
* Depth = distance along the optical axis (camera z) in metres, 0.0 where no surface was hit.
* Segmentation = int32 per pixel: the render model's geom id, -1 for background; `RenderResult.segmentation_labels`
  maps ids to {geom, body, scale_ref}.

Everything here is plain data: `to_dict()` / `from_dict()` round-trip exactly, `canonical_json()` is the byte form the
digests hash (sorted keys, no whitespace, shortest-repr floats).  Fields may be ADDED in later versions (with defaults
that reproduce the old behaviour); fields are never removed or re-interpreted (a change of meaning is a new
RENDER_LAYER_VERSION major).  A field added after arb-render/1.0.0 is listed in its class's `_OMIT_IF_DEFAULT` and is
left out of `to_dict()` (hence of every digest, photos_gt.json and request JSON) while it holds its default, so every
request, digest and photo set written before the addition stays byte-identical (pinned in
tests/test_render_determinism.py and tests/test_render_lens.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

RENDER_LAYER_VERSION = "arb-render/1.0.0"

OUTPUTS = ("rgb", "depth", "segmentation", "sensor_depth")
GRADER_ONLY_OUTPUTS = ("depth", "segmentation")          # exact: never into setup/ or image/
AGENT_VISIBLE_OUTPUTS = ("rgb", "sensor_depth")          # what a camera / an RGB-D sensor delivers
LOD_LEVELS = ("full", "policy")


# ----------------------------------------------------------------------------- (de)serialisation
def _is_default(f, value) -> bool:
    if f.default is not dataclasses.MISSING:
        return value == f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return value == f.default_factory()  # type: ignore[misc]
    return False


def _encode(v):
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        omit = getattr(type(v), "_OMIT_IF_DEFAULT", ())
        out = {}
        for f in dataclasses.fields(v):
            val = getattr(v, f.name)
            if f.name in omit and _is_default(f, val):
                continue                      # added after 1.0.0 and unused: old bytes / digests unchanged
            out[f.name] = _encode(val)
        return out
    if isinstance(v, np.ndarray):
        return _encode(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _encode(x) for k, x in v.items()}
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("non-finite number in a render spec: %r" % (v,))
        return f
    return v


def _decode(tp, v):
    if v is None:
        return None
    origin = typing.get_origin(tp)
    if origin is Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _decode(args[0], v)
        for a in args:  # first that fits
            try:
                return _decode(a, v)
            except Exception:  # noqa: BLE001
                continue
        return v
    if origin in (list, List):
        (a,) = typing.get_args(tp) or (Any,)
        return [_decode(a, x) for x in v]
    if origin in (dict, Dict):
        args = typing.get_args(tp) or (str, Any)
        return {k: _decode(args[1], x) for k, x in v.items()}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return tp.from_dict(v)
    if tp is float:
        return float(v)
    if tp is int and not isinstance(v, bool):
        return int(v)
    return v


class _Spec:
    """Mixin: to_dict / from_dict / to_json / digest for the spec dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        return _encode(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        if isinstance(d, cls):
            return d
        hints = typing.get_type_hints(cls)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError("%s: unknown field(s) %s" % (cls.__name__, sorted(unknown)))
        return cls(**{k: _decode(hints[k], v) for k, v in d.items()})

    def to_json(self, indent: Optional[int] = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def copy(self, **changes):
        return dataclasses.replace(self, **changes)


def _canon_numbers(v):
    """Integral floats as ints (1.0 -> 1), so a value typed 1 or 1.0 hashes the same."""
    if isinstance(v, dict):
        return {k: _canon_numbers(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_canon_numbers(x) for x in v]
    if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53:
        return int(v)
    return v


def canonical_json(obj) -> str:
    """The byte form every digest hashes: sorted keys, no whitespace, ASCII, shortest-repr floats, integral floats as
    ints, no NaN."""
    return json.dumps(_canon_numbers(_encode(obj)), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_hex(data: Union[bytes, str]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def derive_seed(seed: int, *labels) -> int:
    """A 63-bit child seed of `seed` for a label path (e.g. derive_seed(s, "view", 3, "noise")); stable across
    processes, platforms and Python hash randomisation."""
    h = hashlib.sha256(canonical_json([int(seed)] + [str(x) for x in labels]).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def rng(seed: int, *labels) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(seed, *labels)))


def new_seed() -> int:
    """A fresh 63-bit seed from OS entropy (secrets): the seed of a hidden photo set, episode or depth unit.  Seeds are
    the only secret of the noise fields (the render code may be public, T3): never a date, a small integer or 0."""
    import secrets
    return secrets.randbits(63) or 1


# ----------------------------------------------------------------------------- scene
@dataclass
class MaterialOverride(_Spec):
    """Change every material whose name matches `match` (fnmatch pattern) and none of `exclude`."""
    match: str
    exclude: List[str] = field(default_factory=list)
    rgba: Optional[List[float]] = None
    specular: Optional[float] = None
    shininess: Optional[float] = None
    reflectance: Optional[float] = None
    emission: Optional[float] = None
    roughness: Optional[float] = None
    metallic: Optional[float] = None
    texture: Optional[str] = None          # name of a texture of the model or of SceneSpec.textures
    texrepeat: Optional[List[float]] = None
    texuniform: Optional[bool] = None


@dataclass
class GeomOverride(_Spec):
    """Change geoms by name pattern and / or by body subtree (all geoms under `body`)."""
    match: str = "*"
    body: Optional[str] = None
    rgba: Optional[List[float]] = None
    material: Optional[str] = None
    hide: bool = False


@dataclass
class TextureSpec(_Spec):
    """A texture ADDED to the scene (or replacing the data of an existing texture of the same name).

    kind: "flat" | "checker" | "speckle" | "gradient" | "wood" | "brushed" | "grid" | "file" (textures.py).
    `file` is resolved against SceneSpec.base_dir.  Procedural kinds are deterministic in `seed`."""
    name: str
    kind: str = "flat"
    width: int = 256
    height: int = 256
    rgb1: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    rgb2: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])
    params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    file: Optional[str] = None


@dataclass
class LightSpec(_Spec):
    name: Optional[str] = None
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 3.0])
    dir: List[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    diffuse: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7])
    specular: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])
    ambient: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    directional: bool = False
    castshadow: bool = True
    attenuation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    cutoff_deg: float = 60.0
    exponent: float = 1.0


@dataclass
class HeadlightSpec(_Spec):
    ambient: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    diffuse: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.4])
    specular: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    active: bool = True


@dataclass
class BackgroundSpec(_Spec):
    """Skybox gradient (rgb1 at the zenith, rgb2 at the nadir); `None` in SceneSpec keeps the model's own."""
    rgb1: List[float] = field(default_factory=lambda: [0.85, 0.86, 0.88])
    rgb2: List[float] = field(default_factory=lambda: [0.55, 0.56, 0.58])


@dataclass
class StateSpec(_Spec):
    """The configuration rendered: keyframe first, then the full qpos, then named joints; optional settling."""
    keyframe: Optional[str] = None
    qpos: Optional[List[float]] = None
    joints: Dict[str, float] = field(default_factory=dict)
    settle_s: float = 0.0


@dataclass
class ScaleRef(_Spec):
    """A physical scale reference IN the scene (never a 2D overlay): every backend renders its geometry.

    kind: "ruler" | "checkerboard" | "can" | "card" | "a4" (scale_refs.KINDS).  Frame: origin at the centre of the
    face the object rests on, +z up out of that face; `pos` (m) / `quat` (w x y z) place it in the world.  `params`
    are the kind's nominal parameters (scale_refs.py documents each; the defaults are the nominal everyday sizes)."""
    kind: str
    name: str
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    params: Dict[str, Any] = field(default_factory=dict)
    collide: bool = False
    seed: int = 0          # procedural texture detail (brushing, print noise); never changes a nominal size


@dataclass
class SceneSpec(_Spec):
    """What to render.  Exactly one of `mjcf` (a path; relative paths against `base_dir`) or `builder`.

    `builder` = "package.module:function" or "path/to/file.py:function" (relative to `base_dir`); the function is
    called as function(**builder_args) and returns a `mujoco.MjSpec`.  It is author / grader code (never agent code).
    `content_files`: extra files or directories whose bytes enter the request digest (e.g. the meshes a builder
    reads); an MJCF's own includes / meshes / textures are found automatically."""
    mjcf: Optional[str] = None
    builder: Optional[str] = None
    builder_args: Dict[str, Any] = field(default_factory=dict)
    base_dir: Optional[str] = None
    content_files: List[str] = field(default_factory=list)
    state: StateSpec = field(default_factory=StateSpec)
    materials: List[MaterialOverride] = field(default_factory=list)
    geoms: List[GeomOverride] = field(default_factory=list)
    textures: List[TextureSpec] = field(default_factory=list)
    lights: List[LightSpec] = field(default_factory=list)
    replace_lights: bool = False
    headlight: Optional[HeadlightSpec] = None
    background: Optional[BackgroundSpec] = None
    scale_refs: List[ScaleRef] = field(default_factory=list)
    hide_bodies: List[str] = field(default_factory=list)
    visible_groups: List[int] = field(default_factory=lambda: [0, 1, 2])
    shadows: bool = True
    offsamples: int = 0          # GL MSAA samples.  0 by default: MSAA + shadows is NOT bitwise deterministic on the
                                 # NVIDIA EGL driver (measured 2026-09-27); anti-alias with PhotoSpec.supersample instead
    shadowsize: int = 4096
    near_m: float = 0.02
    far_m: float = 50.0
    extra: Dict[str, Any] = field(default_factory=dict)
    # ---- added after 1.0.0 (omitted from to_dict while default; RA3)
    lod: str = "full"            # render level of detail: "full" = the scene's meshes as they are; "policy" = visible mesh
                                 # geoms decimated by lod.py (recipe = lod.DEFAULT_POLICY_RECIPE updated by lod_params);
                                 # the recipe and a digest of the decimated meshes go into Provenance.lod
    lod_params: Dict[str, Any] = field(default_factory=dict)

    _OMIT_IF_DEFAULT = frozenset({"lod", "lod_params"})

    def __post_init__(self):
        if (self.mjcf is None) == (self.builder is None):
            raise ValueError("SceneSpec needs exactly one of mjcf / builder")
        if self.lod not in LOD_LEVELS:
            raise ValueError("SceneSpec.lod must be one of %s (got %r)" % (LOD_LEVELS, self.lod))
        if self.lod_params:
            if self.lod == "full":
                raise ValueError("SceneSpec.lod_params given but lod is 'full'")
            from .lod import check_recipe
            check_recipe(self.lod_params)


# ----------------------------------------------------------------------------- camera
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


@dataclass
class CameraSpec(_Spec):
    """A pinhole camera with OpenCV conventions (module docstring).  Exact values are GRADER-ONLY (photo_set.py)."""
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_T_cam: List[List[float]]
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    exposure_ev: float = 0.0                     # linear gain 2**EV applied in postprocess
    white_balance: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])   # linear R G B gains

    # ---- constructors
    @classmethod
    def look_at(cls, name, eye, target, width=640, height=480, hfov_deg=60.0, fx=None, fy=None, cx=None, cy=None,
                up=(0.0, 0.0, 1.0), roll_deg=0.0, **kw) -> "CameraSpec":
        eye = np.asarray(eye, float)
        z = _unit(np.asarray(target, float) - eye)
        up = np.asarray(up, float)
        if abs(float(np.dot(z, _unit(up)))) > 0.999:      # looking straight up / down: pick another up
            up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = _unit(np.cross(z, up))
        y = np.cross(z, x)
        if roll_deg:
            a = math.radians(roll_deg)
            x, y = math.cos(a) * x + math.sin(a) * y, -math.sin(a) * x + math.cos(a) * y
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
        if fx is None:
            fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        fy = fx if fy is None else fy
        cx = (width - 1) / 2.0 if cx is None else cx
        cy = (height - 1) / 2.0 if cy is None else cy
        return cls(name=name, width=int(width), height=int(height), fx=float(fx), fy=float(fy), cx=float(cx),
                   cy=float(cy), world_T_cam=T.tolist(), **kw)

    # ---- derived quantities
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def T(self) -> np.ndarray:
        return np.asarray(self.world_T_cam, float)

    @property
    def R(self) -> np.ndarray:            # world_R_cam
        return self.T[:3, :3]

    @property
    def position(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def distortion(self) -> List[float]:
        return [self.k1, self.k2, self.p1, self.p2, self.k3]

    @property
    def has_distortion(self) -> bool:
        return any(abs(v) > 0 for v in self.distortion)

    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2.0 * self.fx)))

    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2.0 * self.fy)))

    def opencv_extrinsics(self):
        """(R_cam_world, t_cam_world) = the OpenCV rvec / tvec pose (x_cam = R x_world + t)."""
        R = self.R.T
        return R, -R @ self.position

    def validate(self):
        T = self.T
        if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError("%s: world_T_cam must be a 4 x 4 rigid transform" % self.name)
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-6) or np.linalg.det(self.R) < 0:
            raise ValueError("%s: world_T_cam rotation is not orthonormal / right-handed" % self.name)
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("%s: bad intrinsics" % self.name)

    def distort_normalised(self, x, y):
        k1, k2, p1, p2, k3 = self.distortion
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def undistort_normalised(self, xd, yd, iterations: int = 40):
        """Inverse of distort_normalised by fixed-point iteration (OpenCV's undistortPoints scheme)."""
        k1, k2, p1, p2, k3 = self.distortion
        x, y = np.array(xd, float, copy=True), np.array(yd, float, copy=True)
        if not self.has_distortion:
            return x, y
        for it in range(iterations):
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            xn = (xd - dx) / radial
            yn = (yd - dy) / radial
            done = it % 4 == 3 and max(float(np.max(np.abs(xn - x), initial=0.0)), float(np.max(np.abs(yn - y), initial=0.0))) < 1e-13
            x, y = xn, yn
            if done:
                break
        return x, y

    def project(self, points_world, distort: bool = True):
        """World points (N, 3) -> (uv (N, 2) pixels, z (N,) depth along the optical axis)."""
        P = np.atleast_2d(np.asarray(points_world, float))
        pc = (P - self.position) @ self.R
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = pc[:, 0] / z, pc[:, 1] / z
        if distort:
            x, y = self.distort_normalised(x, y)
        return np.stack([self.fx * x + self.cx, self.fy * y + self.cy], axis=1), z


# ----------------------------------------------------------------------------- photo
@dataclass
class NoiseSpec(_Spec):
    """Sensor noise in LINEAR light (0..1): variance = shot * I + read**2 per channel, i.i.d. per pixel."""
    shot: float = 0.0
    read: float = 0.0


@dataclass
class BlurSpec(_Spec):
    motion_px: float = 0.0          # length of a linear motion-blur kernel (pixels of the final image)
    motion_angle_deg: float = 0.0   # 0 = along +u (image right), counter-clockwise on screen
    defocus_px: float = 0.0         # radius of a uniform disk (circle of confusion), pixels


@dataclass
class ColourJitterSpec(_Spec):
    """Uniform +- ranges drawn PER VIEW from the request seed (display-referred, after the sensor stage)."""
    brightness: float = 0.0   # gain in [1 - b, 1 + b]
    contrast: float = 0.0     # (x - mean) * [1 - c, 1 + c] + mean
    saturation: float = 0.0   # lerp towards luma by [1 - s, 1 + s]
    hue_deg: float = 0.0      # rotation of the chroma plane (YIQ) in [-h, h]
    gamma: float = 0.0        # x ** [1 - g, 1 + g]
    tint: float = 0.0         # per-channel gains in [1 - t, 1 + t]
    # a fixed seed: the same draw for every request and view (one camera unit's colour response, constant over a run of
    # live frames); None = drawn per view from the request seed (independent photos)
    seed: Optional[int] = None


@dataclass
class DepthSensorSpec(_Spec):
    """An RGB-D sensor's depth stream, registered to the colour camera (the 'aligned depth' of a stereo / ToF camera).

    postprocess.apply_depth_sensor turns the GRADER-ONLY exact depth (metres along the optical axis, already warped with
    the colour camera's distortion table) into the AGENT-VISIBLE `sensor_depth`, in this order:
      1. systematic error      z1 = scale * z + offset_m                       (a per-unit calibration error; hidden)
                               + per-unit FIXED errors, constant over frames (all off by default; hidden; seeded by
                               unit_seed and the camera name, never by the frame seed):
                               wiggle_amp_m * sin(2 pi z / wiggle_period_m + phi)   (range-dependent bias, e.g. ToF)
                               + field(u, v) * (fixed_pattern_a_m + fixed_pattern_b_per_m * z^2)   (a smooth N(0, 1) field:
                                 bilinear on a grid of fixed_pattern_cell_px cells; a stereo / ToF unit's spatial bias)
                               + pixel_offset_sigma_m * N(0, 1) per pixel                            (fixed per-pixel offsets)
      2. axial noise           z2 = z1 + N(0, 1) * (sigma_a_m + sigma_b_per_m * z^2)       stream "depth_noise" (per frame)
      3. disparity quantisation (if disparity_step_px > 0): d = f_d * baseline_m / z2 rounded to the step, z3 = f_d B / d,
                               with f_d = disparity_focal_px, the DEPTH IMAGER's own focal length (never the colour
                               camera's fx: the lattice would publish it)
      4. range                 invalid where z3 < min_range_m or z3 > max_range_m, or where nothing was hit (z = 0)
      5. unit quantisation     z4 = round(z3 / quantisation_m) * quantisation_m  (the file stores the integer, uint16)
      6. discontinuities       invalid where the exact depth jumps by more than edge_threshold_rel * z (+ edge_threshold_m)
                               to a 4-neighbour, dilated by edge_invalid_px pixels (flying pixels / occlusion shadows);
                               edge_ragged_px > 0 extends the band by up to that many pixels, per pixel at random
                               (stream "depth_edge"), so the band is ragged instead of an exact-width strip
      7. dropout               invalid pixels i.i.d. with dropout_rate, grown to disks of radius dropout_blob_px
                               (stream "depth_dropout"); the whole frame invalid with frame_dropout_rate ("depth_frame")
    Invalid pixels read 0.  Every random draw of the per-frame streams is made even when its rate / sigma is 0 (changing
    one knob never re-draws another).  The rgb streams are separate: requesting sensor_depth never changes a colour byte.
    Averaging frames of a static scene removes steps 2 and 7 (sigma / sqrt(N)) but not step 1: a task whose difficulty
    rests on the single-frame sigma enables the fixed errors or budgets frames (render/README.md, RA1).
    A near-ideal sensor (sigma_m(0.5 m) < 0.2 mm and no fixed errors) would publish the grader-only exact depth under the
    agent-visible name and is refused unless allow_ideal=True (tests, deliberate task choices)."""
    sigma_a_m: float = 0.001
    sigma_b_per_m: float = 0.0
    scale: float = 1.0
    offset_m: float = 0.0
    disparity_step_px: float = 0.0       # 0 = off; e.g. 1/8 px for an 8-bit subpixel stereo matcher (needs disparity_focal_px)
    baseline_m: float = 0.05             # stereo baseline used by the disparity quantisation only
    min_range_m: float = 0.1
    max_range_m: float = 4.0
    quantisation_m: float = 0.001        # depth unit of the stored value (uint16 PNG: value * quantisation_m metres)
    edge_threshold_rel: float = 0.03
    edge_threshold_m: float = 0.0
    edge_invalid_px: int = 1
    dropout_rate: float = 0.0
    dropout_blob_px: int = 0
    frame_dropout_rate: float = 0.0
    # ---- added after the first RA1 version (omitted from to_dict while default)
    disparity_focal_px: float = 0.0      # the depth imager's focal length (px) for step 3; required when disparity_step_px > 0
    unit_seed: int = 0                   # the physical unit (hidden): seeds the fixed errors of step 1; != 0 when they are on
    wiggle_amp_m: float = 0.0
    wiggle_period_m: float = 0.5
    fixed_pattern_a_m: float = 0.0
    fixed_pattern_b_per_m: float = 0.0
    fixed_pattern_cell_px: int = 32
    pixel_offset_sigma_m: float = 0.0
    edge_ragged_px: int = 0
    allow_ideal: bool = False

    _OMIT_IF_DEFAULT = frozenset({"disparity_focal_px", "unit_seed", "wiggle_amp_m", "wiggle_period_m", "fixed_pattern_a_m",
                                  "fixed_pattern_b_per_m", "fixed_pattern_cell_px", "pixel_offset_sigma_m", "edge_ragged_px",
                                  "allow_ideal"})

    def __post_init__(self):
        if min(self.sigma_a_m, self.sigma_b_per_m, self.min_range_m, self.edge_threshold_rel, self.edge_threshold_m,
               self.disparity_step_px, self.disparity_focal_px, self.wiggle_amp_m, self.fixed_pattern_a_m, self.fixed_pattern_b_per_m,
               self.pixel_offset_sigma_m) < 0:
            raise ValueError("DepthSensorSpec: negative sigma / range / threshold / amplitude")
        if not self.max_range_m > self.min_range_m:
            raise ValueError("DepthSensorSpec: max_range_m must exceed min_range_m")
        if not self.quantisation_m > 0 or self.max_range_m / self.quantisation_m > 65535:
            raise ValueError("DepthSensorSpec: quantisation_m must be > 0 and max_range_m / quantisation_m <= 65535 (uint16)")
        if self.disparity_step_px > 0 and not self.baseline_m > 0:
            raise ValueError("DepthSensorSpec: disparity quantisation needs baseline_m > 0")
        if self.disparity_step_px > 0 and not self.disparity_focal_px > 0:
            raise ValueError("DepthSensorSpec: disparity quantisation needs disparity_focal_px > 0, the depth imager's own focal "
                             "length (a lattice built on the colour camera's fx would publish that fx)")
        for k in ("dropout_rate", "frame_dropout_rate"):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError("DepthSensorSpec.%s must be in [0, 1]" % k)
        if self.edge_invalid_px < 0 or self.dropout_blob_px < 0 or self.edge_ragged_px < 0 or not self.scale > 0:
            raise ValueError("DepthSensorSpec: edge_invalid_px / edge_ragged_px / dropout_blob_px >= 0 and scale > 0")
        if not self.wiggle_period_m > 0 or int(self.fixed_pattern_cell_px) < 1:
            raise ValueError("DepthSensorSpec: wiggle_period_m > 0 and fixed_pattern_cell_px >= 1")
        if self.has_fixed_errors and int(self.unit_seed) == 0:
            raise ValueError("DepthSensorSpec: fixed (per-unit) errors need a unit_seed != 0, e.g. derive_seed(hidden episode or "
                             "instance seed, 'depth_unit'); seed 0 would make the pattern a public function of the code")
        if not self.allow_ideal and float(self.sigma_m(0.5)) < 2e-4 and not self.has_fixed_errors:
            raise ValueError("DepthSensorSpec: sigma_m(0.5 m) = %.3g m < 0.2 mm without fixed errors publishes the grader-only exact "
                             "depth as sensor_depth; pass allow_ideal=True if that is intended" % float(self.sigma_m(0.5)))

    @property
    def has_fixed_errors(self) -> bool:
        """True if any per-unit (frame-constant) error term of step 1 is on."""
        return bool(self.wiggle_amp_m > 0 or self.fixed_pattern_a_m > 0 or self.fixed_pattern_b_per_m > 0 or self.pixel_offset_sigma_m > 0)

    @property
    def is_stochastic(self) -> bool:
        """True if a frame of this sensor depends on the frame seed (noise, dropout, ragged edges)."""
        return bool(self.sigma_a_m > 0 or self.sigma_b_per_m > 0 or self.dropout_rate > 0 or self.frame_dropout_rate > 0
                    or self.edge_ragged_px > 0)

    def sigma_m(self, z):
        return self.sigma_a_m + self.sigma_b_per_m * np.asarray(z, float) ** 2


@dataclass
class PhotoSpec(_Spec):
    """Post-processing applied IDENTICALLY after any backend (postprocess.py, fixed order documented there)."""
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    blur: BlurSpec = field(default_factory=BlurSpec)
    vignetting: float = 0.0              # fractional brightness loss at the corner farthest from the principal point
    colour_jitter: ColourJitterSpec = field(default_factory=ColourJitterSpec)
    image_format: str = "png"            # "png" | "jpeg"
    jpeg_quality: int = 92
    jpeg_subsampling: str = "4:2:0"      # "4:4:4" | "4:2:2" | "4:2:0"
    supersample: int = 1                 # render at k x resolution, box-filter down (backend-independent anti-aliasing)
    # ---- added after 1.0.0 (omitted from to_dict while default; RA1)
    depth_sensor: Optional[DepthSensorSpec] = None   # required when a request asks for the "sensor_depth" output
    png_compress_level: int = 6          # zlib level of PNG photos (6 = the 1.0.0 bytes; 1 saves ~25 % time on noisy frames)

    _OMIT_IF_DEFAULT = frozenset({"depth_sensor", "png_compress_level"})

    def __post_init__(self):
        if self.image_format not in ("png", "jpeg"):
            raise ValueError("image_format must be png or jpeg")
        if self.supersample < 1 or self.supersample > 4:
            raise ValueError("supersample in 1..4")
        if not 0 <= int(self.png_compress_level) <= 9:
            raise ValueError("png_compress_level in 0..9")

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format == "jpeg" else "png"

    @property
    def is_stochastic(self) -> bool:
        """True if the photo depends on the seed (sensor noise, colour jitter, a stochastic depth sensor)."""
        cj = self.colour_jitter
        return bool(self.noise.shot or self.noise.read or any((cj.brightness, cj.contrast, cj.saturation, cj.hue_deg, cj.gamma, cj.tint))
                    or (self.depth_sensor is not None and self.depth_sensor.is_stochastic))


# ----------------------------------------------------------------------------- request / result
@dataclass
class RenderRequest(_Spec):
    scene: SceneSpec
    cameras: List[CameraSpec]
    photo: PhotoSpec = field(default_factory=PhotoSpec)
    seed: int = 0
    backend: str = "mujoco"
    outputs: List[str] = field(default_factory=lambda: ["rgb"])
    label: str = ""

    def __post_init__(self):
        bad = [o for o in self.outputs if o not in OUTPUTS]
        if bad or "rgb" not in self.outputs:
            raise ValueError("outputs must contain rgb and only %s (got %s)" % (OUTPUTS, self.outputs))
        if "sensor_depth" in self.outputs and self.photo.depth_sensor is None:
            raise ValueError("the sensor_depth output needs PhotoSpec.depth_sensor (a DepthSensorSpec)")
        if "sensor_depth" in self.outputs and self.photo.depth_sensor.is_stochastic and int(self.seed) == 0:
            raise ValueError("sensor_depth with seed 0: the noise field would be a public function of the render code; pass a "
                             "secret seed (live frames: derive_seed(<OS-entropy episode seed>, frame); fixed sets: build_photo_set "
                             "derives them from PhotoSetSpec.seed = spec.new_seed())")
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError("camera names must be unique")
        for c in self.cameras:
            c.validate()

    def digest(self) -> str:
        """sha256 of the canonical request JSON + the bytes of every file the scene reads (provenance.py)."""
        from .provenance import request_digest
        return request_digest(self)


@dataclass
class Provenance(_Spec):
    backend: str
    backend_version: str
    render_layer_version: str
    request_digest: str
    image_sha256: List[str]
    resolved_scene_digest: Optional[str] = None     # backend-side digest of the compiled scene (MuJoCo: the MJB bytes)
    environment: Dict[str, Any] = field(default_factory=dict)  # library / GL versions that can change pixels
    postprocess_version: str = ""
    # ---- added after 1.0.0 (omitted from to_dict while default)
    lod: Optional[Dict[str, Any]] = None              # SceneSpec.lod != "full": level, recipe, digest, face counts (RA3)
    sensor_depth_sha256: Optional[List[str]] = None   # per camera, when the sensor_depth output was requested (RA1)
    sensor_depth_model: Optional[str] = None          # postprocess.DEPTH_SENSOR_VERSION when sensor_depth was produced

    _OMIT_IF_DEFAULT = frozenset({"lod", "sensor_depth_sha256", "sensor_depth_model"})


@dataclass
class RenderResult:
    """Per camera (same order as RenderRequest.cameras).  depth / segmentation / raw_rgb are GRADER-ONLY."""
    rgb: List[np.ndarray]                                  # uint8 H x W x 3: exactly what the agent sees (decoded file)
    encoded: List[bytes]                                   # the image file bytes (PNG / JPEG)
    image_format: str
    provenance: Provenance
    depth: Optional[List[np.ndarray]] = None               # float32 H x W metres along the optical axis, 0 = none
    segmentation: Optional[List[np.ndarray]] = None        # int32 H x W geom id, -1 = background
    segmentation_labels: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    raw_rgb: Optional[List[np.ndarray]] = None             # the backend's pinhole image before postprocess (debug)
    facts: Dict[str, Any] = field(default_factory=dict)    # timings and backend notes (not part of any digest)
    sensor_depth: Optional[List[np.ndarray]] = None        # AGENT-VISIBLE float32 H x W metres, 0 = invalid (RA1)
    sensor_depth_encoded: Optional[List[bytes]] = None     # 16-bit PNG of sensor_depth / quantisation_m (b"" if not encoded)

    AGENT_FIELDS = ("rgb", "encoded", "image_format", "sensor_depth", "sensor_depth_encoded")

    def agent_view(self) -> Dict[str, Any]:
        """The fields that may reach an agent (a live frame store, a camera API): the images and the sensor depth.
        provenance (renderer / GL build, digests, LOD record), facts, segmentation_labels (geom / body names), depth,
        segmentation and raw_rgb are grader-side (render/README.md, public / hidden rule)."""
        return {k: getattr(self, k) for k in self.AGENT_FIELDS}
