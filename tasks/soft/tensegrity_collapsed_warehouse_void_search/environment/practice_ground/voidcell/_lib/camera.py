"""The search camera of the cell (given/camera_datasheet.md): exposures at 7 Hz, a renderer behind a renderer-agnostic
interface, the depth sensor model, the uint16 depth encoding.

Time model. The camera exposes at the nominal times k / rate_hz of the episode clock (k = 0, 1, ...): at the first
control tick (10 ms grid) at or after k / rate_hz the joint positions are copied (no rendering; the physics is not
touched). Exposure 0 is the state at the episode's start. get_frame() returns the newest exposure; it is rendered on
demand and cached by its index, so frames the program never asks for cost nothing. A frame's `t` is the time of the
control tick at which it was exposed (at most 10 ms after k / rate_hz, never after the current simulated time).

Camera. The site package's search camera: its true extrinsic (site.json camera.true.T_world_camera, OpenCV axes:
x right = +u, y down = +v, z forward) and the given intrinsics K (camera.model). The classic backend puts the GL
frustum on the pixel grid from K directly (pixel centres at integer coordinates; the frustum arithmetic of the
vendored base, ysim.camera.ClassicSession), so a world point projects with the given K and the true extrinsic; the
agent is given K and a calibrated extrinsic with the site's small calibration error (mission()). The site's MJCF
camera `search` (principalpixel form) is the same camera; the QC of the generator (G3) measured it.

Backends. A backend is any object with `name`, `version()` and `open(model, spec) -> session`; a session has
`render(data) -> (rgb uint8 H x W x 3, depth float32 H x W metres along the optical axis, 0 = nothing hit)` and
`close()`. `register_backend(name, factory)`; "classic" = MuJoCo's classic OpenGL renderer (OSMesa / EGL via
MUJOCO_GL). "plate_filament" (plate_filament.py) = the site's offline plate (the search camera's environment
rendered once per site, its bundle picked by the site package's sha256) with the robot rendered per frame; it takes
the episode's site package identity and the plate root as keyword arguments (backend_kw) and raises when the
bundle is missing or foreign: there is no fall-back to the classic image.

Sensor model (rated values of this simulated unit, given/camera_datasheet.md): depth along the optical axis, range 0.25-9.0 m
(outside: 0); at every jump of more than 0.05 m between 4-neighbours the farther pixel reads 0 (one-pixel dropout at
depth edges); Gaussian noise sigma = 2.5 mm + 1.6 mm x max(depth - 1 m, 0), seeded per (episode noise seed, frame
index) so a frame does not depend on when or how often it is fetched; quantised to 0.25 mm counts (uint16, 4000 per
metre, 0 = no reading). The colour image is the renderer's, without added noise.
"""
import base64
import hashlib
import math
import time
import zlib
from dataclasses import dataclass

import numpy as np

SENSOR = {"range_m": (0.25, 9.0), "edge_jump_m": 0.05, "sigma_mm_at_1m": 2.5, "sigma_mm_per_m": 1.6,
          "unit_m": 0.00025}
ENCODING = ("zlib-raw-v1: rgb_b64 = base64(zlib(uint8 H x W x 3 row-major)); depth_b64 = base64(zlib(little-endian "
            "uint16 H x W row-major)), depth counts x depth_unit_m = metres along the optical axis, 0 = no reading")
NOISE_STREAM = 0x0F


# ---------------------------------------------------------------------------------------------- backends
_BACKENDS = {}


def register_backend(name, factory):
    _BACKENDS[name] = factory


def get_backend(name="classic", **kw):
    if name not in _BACKENDS:
        raise KeyError("render backend %r not registered (have %s)" % (name, sorted(_BACKENDS)))
    return _BACKENDS[name](**kw)


class ClassicBackend:
    """MuJoCo's classic renderer through the vendored base's session (ysim.camera.ClassicSession): RGB from the
    model's multisampled context (msaa_rgb=True) or a single-sample one, depth from a single-sample context."""
    name = "classic"

    def __init__(self, msaa_rgb=True):
        self.msaa_rgb = bool(msaa_rgb)

    def version(self):
        import mujoco
        return "mujoco-classic/%s msaa_rgb=%s" % (mujoco.__version__, self.msaa_rgb)

    def open(self, model, spec):
        from ysim import camera as yc
        return _ClassicSession(yc.ClassicSession(model, spec.width, spec.height, (0, 1, 2), False, self.msaa_rgb),
                               model, spec)


class _ClassicSession:
    def __init__(self, inner, model, spec):
        self.inner, self.model, self.spec = inner, model, spec

    def render(self, data):
        return self.inner.render(self.model, data, self.spec, depth=True)

    def close(self):
        self.inner.close()


register_backend("classic", lambda **kw: ClassicBackend(**kw))


def _plate_filament(**kw):
    """The plate backend (plate_filament.py); kw: msaa_rgb (ignored), site_sha256, site_json_sha256,
    site_xml_sha256, plate_root (the server passes them, server.py reset)."""
    from . import plate_filament
    return plate_filament.PlateFilamentBackend(**kw)


register_backend("plate_filament", _plate_filament)


# ---------------------------------------------------------------------------------------------- the camera
def camera_spec(site, which="true"):
    """The base's pinhole CameraSpec of the site's search camera (true extrinsic by default; 'given' = the calibrated
    one handed to the agent)."""
    from ysim import camera as yc
    cm = site["camera"]["model"]
    T = site["camera"]["true" if which == "true" else "given"]["T_world_camera"]
    return yc.CameraSpec(name="search", width=int(cm["width"]), height=int(cm["height"]), fx=float(cm["fx"]),
                         fy=float(cm["fy"]), cx=float(cm["cx"]), cy=float(cm["cy"]),
                         world_T_cam=[list(map(float, r)) for r in T], rate_hz=float(cm["rate_hz"]),
                         depth_scale=1.0 / float(cm["depth_unit_m"]),
                         provenance={"pose": "site package camera.%s" % which, "intrinsics": "site package camera.model"})


def apply_site_lighting(m, site):
    """The site's own lighting on a model built with ModelConfig(lights=False): the headlight carries the site's
    ambient only (as in site.xml: ambient a, no diffuse, no specular); the site's key light is in its worldbody."""
    a = float(site["lighting"]["ambient"])
    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = [a, a, a]
    m.vis.headlight.diffuse[:] = [0.0, 0.0, 0.0]
    m.vis.headlight.specular[:] = [0.0, 0.0, 0.0]


def render_model(site, site_xml, mesh_dir, width=640, height=480):
    """The visual model of the site and the robot (decimated author mesh), lights of the base off, the site's
    lighting on. Separate from the physics model (visual=False): rendering never touches the physics; the joint
    layout is the same, so a physics qpos renders as it is."""
    import mujoco
    from cellcore import world as wmod
    ys = wmod.ysim()
    cfg = wmod.model_config(site_xml, visual=True, mesh_res="dec", lights=False, offscreen=(width, height),
                            assets=mesh_dir)
    xml, _info = ys.model.build_xml(cfg)
    m = mujoco.MjModel.from_xml_string(xml)
    apply_site_lighting(m, site)
    return m


@dataclass
class Frame:
    index: int
    t: float
    rgb: np.ndarray          # uint8 H x W x 3
    depth: np.ndarray        # uint16 H x W, counts of SENSOR unit_m, 0 = no reading

    def sha256(self):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.rgb).tobytes())
        h.update(np.ascontiguousarray(self.depth).astype("<u2").tobytes())
        return h.hexdigest()

    def wire(self):
        H, W = self.depth.shape
        return {"index": int(self.index), "t": round(float(self.t), 6), "width": int(W), "height": int(H),
                "encoding": ENCODING,
                "rgb_b64": base64.b64encode(zlib.compress(np.ascontiguousarray(self.rgb, np.uint8).tobytes(), 1)
                                            ).decode("ascii"),
                "depth_b64": base64.b64encode(zlib.compress(np.ascontiguousarray(self.depth).astype("<u2")
                                                            .tobytes(), 1)).decode("ascii"),
                "depth_unit_m": SENSOR["unit_m"]}


def depth_sigma_m(d):
    """Rated depth noise (m, one sigma) at depth d (m)."""
    return (SENSOR["sigma_mm_at_1m"] + SENSOR["sigma_mm_per_m"] * np.maximum(np.asarray(d, float) - 1.0, 0.0)) * 1e-3


def edge_dropout(d):
    """Mask of the pixels dropped at depth edges: of every 4-neighbour pair whose depths differ by more than the
    jump, the farther pixel (pixels without a hit take no part)."""
    drop = np.zeros(d.shape, bool)
    jump = SENSOR["edge_jump_m"]
    for a, b in ((d[:, :-1], d[:, 1:]), (d[:-1, :], d[1:, :])):
        ok = (a > 0) & (b > 0)
        far_a = ok & (a - b > jump)
        far_b = ok & (b - a > jump)
        if a.shape[1] != d.shape[1]:        # horizontal pairs
            drop[:, :-1] |= far_a
            drop[:, 1:] |= far_b
        else:                               # vertical pairs
            drop[:-1, :] |= far_a
            drop[1:, :] |= far_b
    return drop


def sensor_depth(depth_true, rng):
    """uint16 counts of the simulated depth sensor from the renderer's depth (float32 m, 0 = no hit)."""
    d = np.asarray(depth_true, np.float64)
    lo, hi = SENSOR["range_m"]
    valid = (d >= lo) & (d <= hi) & ~edge_dropout(d)
    z = d + rng.standard_normal(d.shape) * depth_sigma_m(d)
    counts = np.clip(np.round(z / SENSOR["unit_m"]), 1, 65535)
    return np.where(valid, counts, 0).astype(np.uint16)


class ExposureClock:
    """Exposure k at the first control tick at or after k / rate_hz of the episode clock (k = 0 at the start):
    calls every listener f(k, t, qpos). The camera and the pose source both listen, so the pose updates at the
    camera's exposures whether or not the cell renders."""

    def __init__(self, rate_hz):
        self.rate = float(rate_hz)
        self.k_next = 0
        self.listeners = []
        self.count = 0

    def on_tick(self, t, qpos):
        if t + 1e-9 >= self.k_next / self.rate:
            k = self.k_next
            self.k_next = int(math.floor(t * self.rate + 1e-9)) + 1
            self.count += 1
            for f in self.listeners:
                f(k, float(t), qpos)


class SearchCamera:
    """The fixed RGB-D search camera of one episode."""

    def __init__(self, site, site_xml, mesh_dir, noise_seed, backend="classic", backend_kw=None):
        import mujoco
        self.mj = mujoco
        self.site = site
        self.spec = camera_spec(site, "true")
        self.rate = float(self.spec.rate_hz)
        self.noise_seed = int(noise_seed) & ((1 << 63) - 1)
        self.model = render_model(site, site_xml, mesh_dir, self.spec.width, self.spec.height)
        self.data = mujoco.MjData(self.model)
        self.backend = get_backend(backend, **(backend_kw or {})) if isinstance(backend, str) else backend
        self.session = self.backend.open(self.model, self.spec)
        self.snap = None              # (index, t, qpos) of the newest exposure
        self.cache = None
        self.n_rendered = 0
        self.render_s = 0.0
        self.sensor_s = 0.0
        self.exposures = 0

    # ---- exposures (an ExposureClock calls this at every exposure, episode clock)
    def on_exposure(self, k, t, qpos):
        self.snap = (int(k), float(t), np.array(qpos, float, copy=True))
        self.exposures += 1

    def latest_index(self):
        return None if self.snap is None else int(self.snap[0])

    # ---- rendering
    def render_qpos(self, qpos):
        """(rgb, depth float32 m) of the model at qpos, without the sensor model (QC and the frame)."""
        self.data.qpos[:] = qpos
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        self.mj.mj_camlight(self.model, self.data)       # light directions (mj_kinematics alone leaves them zero)
        t0 = time.perf_counter()
        rgb, depth = self.session.render(self.data)
        self.render_s += time.perf_counter() - t0
        self.n_rendered += 1
        return rgb, depth

    def frame(self):
        """The newest exposure as a Frame (rendered and noised once per index)."""
        if self.snap is None:
            return None
        k, t, q = self.snap
        if self.cache is None or self.cache.index != k:
            rgb, depth = self.render_qpos(q)
            t0 = time.perf_counter()
            rng = np.random.default_rng([self.noise_seed, NOISE_STREAM, int(k)])
            du16 = sensor_depth(depth, rng)
            self.sensor_s += time.perf_counter() - t0
            self.cache = Frame(int(k), float(t), rgb, du16)
        return self.cache

    def close(self):
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            pass
