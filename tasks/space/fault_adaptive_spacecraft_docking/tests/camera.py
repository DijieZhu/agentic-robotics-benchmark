"""The chaser's docking camera (DOCK) and star tracker, as the grader renders and reads them.

The grader owns this module (tests/, root-only at verify time). verify.py builds the observation from env.py's state:
env.py is unchanged, so its pose history (`env.pose_history`: stamp, x, y, yaw, port x, port y, port yaw) is what a
frame shows, at the stamp the round-1 observation's delayed pose had. Nothing here draws from env.py's generator.

DOCK camera (the agent's given/camera.md states the same numbers, and a repository test compares that document's JSON
block with CAMERA_SPEC):
  * mount: the lens centre at (-0.082, 0, -0.115) m in the body frame (x forward, the docking face at -x, z up): on the
    -x face, 0.115 m below the probe axis; boresight -x, image right = body +y, image down = body -z;
  * a pinhole of 640 x 480 pixels, fx = fy = 320 px, principal point (319.5, 239.5) (pixel centres at integer
    coordinates, u right, v down from the top-left pixel): a 90.0 x 73.74 degree field, no distortion, global shutter;
  * monochrome, linear, 8-bit, at a fixed exposure (Filament's default camera, EV100 15): the linear luma
    (0.2126 R + 0.7152 G + 0.0722 B) of the rendered linear RGB, plus Gaussian read and shot noise
    sigma = sqrt(0.8**2 + DN / 39) DN, rounded and clipped to 0..255; a pixel saturated in all three colours reads 255.
The scene (Filament, MuJoCo 3.16's mjrf, on Mesa lavapipe: CPU only, as the verifier has no GPU) is the render bundle
(build_camera_bundle.py: the NASA Wind drum with its docking well, the boom and drogue, the craft for its shadow) plus
the target's fiducials (target.md): the docking target DT on the drum below the port, at the camera's height, with a
standoff LED; three LEDs on the boom tip. LEDs are emissive and saturate. No plumes (cold gas is close to invisible,
and a drawn plume would show which nozzle fires). The craft's six bells sit at the nominal places of the six nozzle
geometries, never at an episode's installed ones. Post-processing is off: Filament's output is then linear and has no
temporal dither, so the same state renders the same bytes.

Noise streams: a frame's per-pixel noise is keyed by the noise key (the set's hidden key), the episode's seed and the
frame's stamp alone (numpy PCG64), so a frame never depends on history and a probe copy has the same noise field; the
star tracker's noise is keyed by the key, the seed and the step (BLAKE2b, stdlib), so it costs no numpy.

Process rules for Filament (memory notes of the benchmark's maintainers): one context and one model per process,
created and used by one thread (the engine is bound to it); every Filament object freed before the context; glibc's
malloc arenas capped at import (Filament's job threads each get one otherwise); Vulkan pinned to lavapipe with
LP_NUM_THREADS threads before the module is imported.

The module imports only the standard library at import time (numpy, mujoco and Pillow are imported by `Camera` and by
the noise functions), so the grader's observation logic can be tested without the renderer (verify.BlankCamera).
"""

import base64
import ctypes
import hashlib
import json
import math
import os
import struct

try:  # cap glibc's malloc arenas before any thread allocates (Filament's JobSystem threads)
    ctypes.CDLL("libc.so.6").mallopt(-8, 2)  # M_ARENA_MAX
except (OSError, AttributeError):
    pass

BUNDLE_FORMAT = "fault_adaptive_spacecraft_docking camera bundle 1"
DEFAULT_BUNDLE = "/opt/fault-docking-render"
LAVAPIPE_ICDS = ("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json", "/usr/share/vulkan/icd.d/lvp_icd.json")

# ---------------------------------------------------------------------------------------------------- the camera
WIDTH, HEIGHT = 640, 480
FX = FY = 320.0
CX, CY = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
LENS_BODY = (-0.082, 0.0, -0.115)   # body frame: on the -x face (at -0.078), 0.115 m below the probe axis
NEAR, FAR = 0.004, 60.0             # m
SUN_RADIUS = math.radians(0.53 / 2.0)   # the sun's disc, saturated where it is in the field and not behind the drum
DRUM_OCCLUDER = (0.835, -1.27, 1.08)      # the drum's bounding cylinder (radius, z range) for the sun's occlusion
SHADOW_MAP = 2048                   # px, the sun's shadow map (PCF): no pixel differs by more than 8 DN from 4096's
CRAFT_PROXY = True                  # the craft's bus as one box: it is never in view, it is drawn for its shadow
LUMA = (0.2126, 0.7152, 0.0722)
READ_NOISE_DN = 0.8
E_PER_DN = 39.0
FULL_WELL_E = 10000.0
NOISE_DOMAIN_FRAME = 0xCA
NOISE_DOMAIN_STAR = 0x57
STAR_TRACKER_SIGMA = 2.0e-4         # rad
FRAME_BYTES = WIDTH * HEIGHT

# --------------------------------------------------------------------------------------------- the fiducials
# Port frame: x radially out along the port's bearing (the drum's local +x), y along increasing phase, z up.
# The docking target: a 50 mm square of flush LEDs centred at the camera's height and a standoff LED 30 mm in front
# (the spec's first layout, a 40 mm square and a 20 mm standoff, missed the pre-registered phase accuracy E2 and was
# enlarged as the spec prescribed; tests/README.md, the completability evidence). The backplate's top edge stays 3 mm
# clear of the docking well's collar (radius 0.075 m round the boom's axis).
DT_PLATE = {"x_back": 0.8295, "x_front": 0.8325, "y_half": 0.037, "z_center": -0.115, "z_half": 0.037}
LED_RADIUS = 0.010
LED_THICK = 0.002
DT_LEDS = (  # (name, face x, y, z): four flush on the plate, the standoff LED S on a post
    ("DT1", 0.8345, 0.025, -0.140),
    ("DT2", 0.8345, -0.025, -0.140),
    ("DT3", 0.8345, 0.025, -0.090),
    ("DT4", 0.8345, -0.025, -0.090),
    ("DTS", 0.8645, 0.000, -0.115),
)
DT_POST = {"radius": 0.004, "x0": 0.8325, "x1": 0.8625, "y": 0.0, "z": -0.115}
# Boom tip: in the boomtip frame (x radial outward, origin at the drogue apex, which sits at port_radius - BT_APEX_OFF)
BT_APEX_OFF = 0.12994285714285714   # craft half section 0.077943 + probe 0.040 + ball 0.012 (bundle parts.json)
BT_LEDS = (  # (name, face x above the apex, y, z)
    ("BT1", 0.039, 0.072, 0.0),
    ("BT2", 0.039, -0.072, 0.0),
    ("BT3", 0.039, 0.0, -0.072),
)
BT_TAB = {"x0": 0.033, "x1": 0.037, "r0": 0.056, "r1": 0.085, "half_width": 0.013}
NOMINAL_BELLS = (  # (exit position on the craft, exhaust direction): the six nominal geometries' places
    ((-0.07794, -0.055), (-1.0, 0.0)), ((-0.07794, 0.055), (-1.0, 0.0)),
    ((0.07794, -0.055), (1.0, 0.0)), ((0.07794, 0.055), (1.0, 0.0)),
    ((0.045, -0.08015), (0.0, -1.0)), ((0.045, 0.08015), (0.0, 1.0)),
)
NOZZLE_LEN, NOZZLE_INSET = 0.026, 0.004

CAMERA_SPEC = {
    "image": {"width": WIDTH, "height": HEIGHT, "channels": 1, "bits": 8, "response": "linear",
              "row_order": "top row first"},
    "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "distortion": None,
                   "pixel_centres": "integer (u right, v down from the top-left pixel)"},
    "mount_body_m": list(LENS_BODY),
    "boresight_body": [-1.0, 0.0, 0.0],
    "image_right_body": [0.0, 1.0, 0.0],
    "image_down_body": [0.0, 0.0, -1.0],
    "noise": {"read_dn": READ_NOISE_DN, "electrons_per_dn": E_PER_DN, "full_well_electrons": FULL_WELL_E,
              "sigma_dn": "sqrt(read_dn**2 + DN / electrons_per_dn)", "rounding": "nearest, clipped to 0..255"},
    "luma": list(LUMA),
    "star_tracker": {"sigma_rad": STAR_TRACKER_SIGMA, "output": "yaw in the inertial frame, wrapped to [-pi, pi)"},
}
TARGET_SPEC = {
    "frame": "port frame: x radially out along the port's bearing, y along increasing target phase, z up",
    "drum_radius_m": 0.82,
    "led_disc_diameter_m": 2 * LED_RADIUS,
    "docking_target": {
        "backplate_m": {"x_front": DT_PLATE["x_front"], "y": [-DT_PLATE["y_half"], DT_PLATE["y_half"]],
                        "z": [round(DT_PLATE["z_center"] - DT_PLATE["z_half"], 6),
                              round(DT_PLATE["z_center"] + DT_PLATE["z_half"], 6)]},
        "leds_m": {name: [x, y, z] for name, x, y, z in DT_LEDS},
        "standoff_post_m": {"radius": DT_POST["radius"], "x": [DT_POST["x0"], DT_POST["x1"]]},
    },
    "boom_tip_target": {
        "drogue_apex_radius_m": "port_radius - {:.4f}".format(BT_APEX_OFF),
        "leds_above_apex_m": {name: [x, y, z] for name, x, y, z in BT_LEDS},
    },
}


# ------------------------------------------------------------------------------------------------- noise streams
def _words(n, count):
    return [(int(n) >> (32 * i)) & 0xFFFFFFFF for i in range(count)]


def frame_noise_rng(key, seed, stamp):
    """numpy Generator of one frame's noise field: the key (256 bits), the seed (64), the frame domain, the stamp."""
    import numpy as np
    entropy = _words(key, 8) + _words(seed, 2) + [NOISE_DOMAIN_FRAME, int(stamp)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def star_tracker_noise(key, seed, k):
    """One N(0, 1) number for the star tracker at step count k: BLAKE2b keyed by the key, Box-Muller."""
    msg = struct.pack("<BQQ", NOISE_DOMAIN_STAR, int(seed) & ((1 << 64) - 1), int(k))
    raw = hashlib.blake2b(msg, key=int(key).to_bytes(32, "little"), digest_size=16).digest()
    a, b = struct.unpack("<QQ", raw)
    u1 = ((a >> 11) + 1) / float(1 << 53)      # (0, 1]
    u2 = (b >> 11) / float(1 << 53)            # [0, 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def star_tracker(theta, key, seed, k):
    return wrap(theta + STAR_TRACKER_SIGMA * star_tracker_noise(key, seed, k))


def expose(luma, saturated, rng):
    """Linear luma in DN (float, 0..255) -> the sensor's 8-bit frame with read and shot noise (uint8, H x W)."""
    import numpy as np
    n = rng.standard_normal(luma.shape, dtype=np.float32)
    sigma = np.sqrt(np.float32(READ_NOISE_DN * READ_NOISE_DN) + luma / np.float32(E_PER_DN))
    out = np.clip(np.rint(luma + sigma * n), 0, 255).astype(np.uint8)
    out[saturated] = 255
    return out


def encode(frame_u8):
    return base64.b64encode(frame_u8.tobytes()).decode("ascii")


def decode(data):
    """The agent's side (camera.md): the frame as a 480 x 640 uint8 array, row 0 = the top row."""
    import numpy as np
    return np.frombuffer(base64.b64decode(data), np.uint8).reshape(HEIGHT, WIDTH)


# ------------------------------------------------------------------------------------------------- the geometry
def camera_pose(x, y, theta):
    """(lens position, forward, up, right) in the inertial frame for a craft at (x, y) with yaw theta (z = 0 plane)."""
    c, s = math.cos(theta), math.sin(theta)
    lx, ly, lz = LENS_BODY
    pos = (x + c * lx - s * ly, y + s * lx + c * ly, lz)
    fwd = (-c, -s, 0.0)
    right = (-s, c, 0.0)
    return pos, fwd, (0.0, 0.0, 1.0), right


def project(point, x, y, theta):
    """Pinhole projection (u, v, depth) of an inertial point by the DOCK camera of a craft at (x, y, theta)."""
    pos, fwd, up, right = camera_pose(x, y, theta)
    d = [point[i] - pos[i] for i in range(3)]
    zc = sum(d[i] * fwd[i] for i in range(3))
    xc = sum(d[i] * right[i] for i in range(3))
    yc = -sum(d[i] * up[i] for i in range(3))
    if zc <= 0:
        return None
    return CX + FX * xc / zc, CY + FY * yc / zc, zc


def fiducials_world(phase, port_radius):
    """{name: (x, y, z)} of the DT and BT LED face centres in the inertial frame."""
    c, s = math.cos(phase), math.sin(phase)
    out = {}
    for name, px, py, pz in DT_LEDS:
        out[name] = (c * px - s * py, s * px + c * py, pz)
    apex = port_radius - BT_APEX_OFF
    for name, px, py, pz in BT_LEDS:
        r = apex + px
        out[name] = (c * r - s * py, s * r + c * py, pz)
    return out


# ------------------------------------------------------------------------------------------------- the renderer
class CameraError(Exception):
    """The camera cannot be built or rendered here (a task error, never a score)."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_bundle(bundle):
    """The bundle's manifest, every file checked against its sha256 (a defect is a CameraError)."""
    try:
        with open(os.path.join(bundle, "bundle.json")) as f:
            man = json.load(f)
    except (OSError, ValueError) as exc:
        raise CameraError("cannot read the camera bundle: {}".format(type(exc).__name__))
    if man.get("format") != BUNDLE_FORMAT:
        raise CameraError("the camera bundle is not of the format '{}'".format(BUNDLE_FORMAT))
    for rel, want in sorted(man.get("files", {}).items()):
        path = os.path.join(bundle, rel)
        if os.path.islink(path) or not os.path.isfile(path) or _sha256(path) != want:
            raise CameraError("the camera bundle's file {} is missing or changed".format(rel))
    return man


class Camera(object):
    """The DOCK camera's renderer: one Filament context and one model, used from the thread that made it."""

    def __init__(self, bundle=None, threads=2):
        bundle = bundle or os.environ.get("FAULT_DOCKING_RENDER", DEFAULT_BUNDLE)
        self.manifest = read_bundle(bundle)
        sun = [float(v) for v in self.manifest["sun_dir"]]
        n = math.sqrt(sum(v * v for v in sun))
        self.sun = [v / n for v in sun]
        icd = next((p for p in LAVAPIPE_ICDS if os.path.isfile(p)), None)
        if icd is None:
            raise CameraError("the camera cannot be rendered here: no lavapipe Vulkan driver ({})".format(
                ", ".join(LAVAPIPE_ICDS)))
        os.environ["VK_ICD_FILENAMES"] = icd             # lavapipe only: CPU, the same driver on every host
        os.environ["LP_NUM_THREADS"] = str(int(threads))
        for k in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(k, None)
        try:
            import mujoco
            import numpy as np
            self.mj, self.np = mujoco, np
            self.model, self.colorspaces = self._build(bundle)
            self.data = mujoco.MjData(self.model)
            self.mocap = {b: int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)])
                          for b in ("target", "boomtip", "craft")}
            self._filament(bundle)
        except CameraError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure to build the renderer is the task's, not the agent's
            self.close()
            raise CameraError("the camera cannot be rendered here: {}: {}".format(type(exc).__name__, exc)[:300])

    # -- the model -------------------------------------------------------------------------------------------- #
    def _png(self, path):
        from PIL import Image
        np = self.np
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), np.uint8)
        return np.ascontiguousarray(np.flipud(a))      # MuJoCo samples uv (0, 0) at the image's top-left texel

    def _build(self, bundle):
        mj, np = self.mj, self.np
        man = self.manifest
        meta = json.load(open(os.path.join(bundle, "meshes", "parts.json")))
        mats = json.load(open(os.path.join(bundle, "materials.json")))
        z = np.load(os.path.join(bundle, "meshes", "parts.npz"))
        spec = mj.MjSpec()
        spec.compiler.inertiafromgeom = mj.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
        spec.visual.headlight.active = 0
        for key, val in (("filament.bloom.enabled", 0), ("filament.ao.enabled", 0), ("filament.shadows.type", 0),
                         ("filament.shadows.map_size", SHADOW_MAP)):
            nm = spec.add_numeric()
            nm.name = key
            nm.data = [float(val)]
        cs = {}
        gains = man.get("gains", {})
        for name in man["materials_used"]:
            mt = mats[name]
            for kind in ("base", "normal", "orm"):
                a = self._png(os.path.join(bundle, "textures", mt["maps"][kind]))
                if kind == "base" and name in gains:   # the look's gain in linear light, capped at the brightest texel
                    c = a.astype(np.float64) / 255.0
                    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
                    g = min(float(gains[name]), 1.0 / max(float(np.percentile(lin.max(-1), 99.5)), 1e-6))
                    lin = np.clip(lin * g, 0, 1)
                    a = np.clip(np.round(255 * np.where(lin <= 0.0031308, 12.92 * lin,
                                                        1.055 * lin ** (1 / 2.4) - 0.055)), 0, 255).astype(np.uint8)
                tx = spec.add_texture()
                tx.name = "%s_%s" % (name, kind)
                tx.type = mj.mjtTexture.mjTEXTURE_2D
                tx.width, tx.height, tx.nchannel = a.shape[1], a.shape[0], 3
                tx.data = a.tobytes()
                cs[tx.name] = "srgb" if kind == "base" else "linear"
            m = spec.add_material()
            m.name = name
            m.rgba = [1.0, 1.0, 1.0, 1.0]
            m.metallic = 1.0
            m.roughness = 1.0
            m.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)] = name + "_base"
            m.textures[int(mj.mjtTextureRole.mjTEXROLE_NORMAL)] = name + "_normal"
            m.textures[int(mj.mjtTextureRole.mjTEXROLE_ORM)] = name + "_orm"

        def flat(name, base_lin, metallic, roughness, emission=0.0):
            m = spec.add_material()
            m.name = name
            c = np.clip(np.asarray(base_lin, float), 0, 1)
            srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
            m.rgba = [float(srgb[0]), float(srgb[1]), float(srgb[2]), 1.0]
            m.metallic = float(metallic)
            m.roughness = float(roughness)
            m.emission = float(emission)
        for name, fm in meta["flat_materials"].items():
            g = float(gains.get(name, 1.0))
            flat(name, np.minimum(np.asarray(fm["base"], float) * g, 1.0), fm["metallic"], fm["roughness"])
        flat("fiducial_black", [0.012, 0.012, 0.012], 0.0, 0.85)
        flat("led", [1.0, 1.0, 1.0], 0.0, 1.0, emission=LED_EMISSION)
        w = spec.worldbody
        sun = np.asarray(man["sun_dir"], float)
        sun /= np.linalg.norm(sun)
        lt = w.add_light()
        lt.name = "sun"
        lt.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
        lt.pos = (sun * 20.0).tolist()
        lt.dir = (-sun).tolist()
        lt.diffuse = [1.0, 0.97, 0.93]
        lt.castshadow = True
        lt.intensity = float(man["sun_lux"])
        bodies = {}
        for b in ("target", "boomtip", "craft"):
            body = w.add_body()
            body.name = b
            body.mocap = True
            body.mass = 1.0
            body.inertia = [1.0, 1.0, 1.0]
            bodies[b] = body

        def mesh_geom(body, name, V, N, UV, F, material):
            mesh = spec.add_mesh()
            mesh.name = name
            mesh.uservert = np.asarray(V, np.float32).reshape(-1).tolist()
            mesh.usernormal = np.asarray(N, np.float32).reshape(-1).tolist()
            if UV is not None and len(UV):
                mesh.usertexcoord = np.asarray(UV, np.float32).reshape(-1).tolist()
            mesh.userface = np.asarray(F, np.int32).reshape(-1).tolist()
            mesh.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
            g = body.add_geom()
            g.name = name
            g.type = mj.mjtGeom.mjGEOM_MESH
            g.meshname = name
            g.material = material
            g.contype = g.conaffinity = 0

        template = None
        for p in meta["parts"]:
            arrs = {k: z["%s|%s" % (p["name"], k)] for k in ("V", "N", "UV", "F")}
            if p["name"] == "nozzle_template":
                template = (p, arrs)
                continue
            if CRAFT_PROXY and p["name"].startswith("cubesat_"):
                continue
            mesh_geom(bodies[p["body"]], p["name"], arrs["V"], arrs["N"], arrs["UV"], arrs["F"], p["material"])
        if CRAFT_PROXY:
            g = bodies["craft"].add_geom()
            g.name = "craft_bus"
            g.type = mj.mjtGeom.mjGEOM_BOX
            g.size = [0.0779, 0.0801, 0.1895]
            g.material = "grey_paint"
            g.contype = g.conaffinity = 0
        p, arrs = template
        for n, ((ex, ey), (dx, dy)) in enumerate(NOMINAL_BELLS):
            e = np.array([dx, dy, 0.0])
            y_ax = np.cross([0.0, 0.0, 1.0], e)
            y_ax /= np.linalg.norm(y_ax)
            R = np.stack([e, y_ax, np.cross(e, y_ax)], 1)
            pos = np.array([ex, ey, 0.0]) + e * (NOZZLE_LEN - NOZZLE_INSET)
            V = np.asarray(arrs["V"], float) @ R.T + pos
            N = np.asarray(arrs["N"], float) @ R.T
            mesh_geom(bodies["craft"], "bell%d" % n, V, N, arrs["UV"], arrs["F"], p["material"])

        def prim(body, name, gtype, size, pos, quat, material):
            g = body.add_geom()
            g.name = name
            g.type = gtype
            g.size = list(size) + [0.0] * (3 - len(size))
            g.pos = list(pos)
            g.quat = list(quat)
            g.material = material
            g.contype = g.conaffinity = 0
        along_x = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]   # a cylinder's axis (z) turned onto +x
        box, cyl = mj.mjtGeom.mjGEOM_BOX, mj.mjtGeom.mjGEOM_CYLINDER
        pl = DT_PLATE
        prim(bodies["target"], "dt_plate", box, (0.5 * (pl["x_front"] - pl["x_back"]), pl["y_half"], pl["z_half"]),
             (0.5 * (pl["x_front"] + pl["x_back"]), 0.0, pl["z_center"]), (1, 0, 0, 0), "fiducial_black")
        po = DT_POST
        prim(bodies["target"], "dt_post", cyl, (po["radius"], 0.5 * (po["x1"] - po["x0"])),
             (0.5 * (po["x0"] + po["x1"]), po["y"], po["z"]), along_x, "fiducial_black")
        for name, fx, fy, fz in DT_LEDS:
            prim(bodies["target"], name.lower(), cyl, (LED_RADIUS, 0.5 * LED_THICK), (fx - 0.5 * LED_THICK, fy, fz),
                 along_x, "led")
        tb = BT_TAB
        for name, fx, fy, fz in BT_LEDS:
            r = math.hypot(fy, fz)
            uy, uz = fy / r, fz / r
            mid = 0.5 * (tb["r0"] + tb["r1"])
            ang = math.atan2(uz, uy)                       # the tab's long axis (local y) turned onto (uy, uz)
            q = [math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0]
            prim(bodies["boomtip"], name.lower() + "_tab", box,
                 (0.5 * (tb["x1"] - tb["x0"]), 0.5 * (tb["r1"] - tb["r0"]), tb["half_width"]),
                 (0.5 * (tb["x0"] + tb["x1"]), mid * uy, mid * uz), q, "fiducial_black")
            prim(bodies["boomtip"], name.lower(), cyl, (LED_RADIUS, 0.5 * LED_THICK), (fx - 0.5 * LED_THICK, fy, fz),
                 along_x, "led")
        model = spec.compile()
        for i in range(model.ntex):
            nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_TEXTURE, i)
            model.tex_colorspace[i] = int(mj.mjtColorSpace.mjCOLORSPACE_SRGB if cs.get(nm) == "srgb"
                                          else mj.mjtColorSpace.mjCOLORSPACE_LINEAR)
        return model, cs

    # -- Filament ------------------------------------------------------------------------------------------------ #
    def _ktx(self, path):
        F = self.F
        with open(path, "rb") as f:
            raw = f.read()
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
        tex = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(raw)
        tex.upload(td)
        return tex

    def _filament(self, bundle):
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  (registers Filament's resource providers)
        np = self.np
        self.F = F
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color(np.zeros(3, np.float32))
        self.res = F.ModelResources(self.ctx, self.model)
        self.scn = self.ctx.create_scene(F.SceneParams())
        self.scn.configure_from_model(self.model)
        self.lights = F.ModelLights(self.scn, self.res)
        self.rends = F.ModelRenderables(self.scn, self.res)
        self.ibl_tex = self._ktx(os.path.join(bundle, "ibl.ktx"))
        lp = F.LightParams()
        lp.type, lp.texture, lp.color = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0]
        lp.intensity = float(self.manifest["ibl_intensity"])
        self.ibl = self.ctx.create_light(lp)
        self.scn.add_light(self.ibl)
        self.sky_tex = self._ktx(os.path.join(bundle, "skybox.ktx"))
        self.scn.set_skybox(self.sky_tex)
        self.buf = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self.tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
        self.tgt.resize(WIDTH, HEIGHT)
        self.read = F.ReadRequest()
        self.read.target = self.tgt
        self.read.set_buffer(self.buf)

    def _pose(self, x, y, theta, phase, port_radius):
        mj, d = self.mj, self.data
        apex = port_radius - BT_APEX_OFF
        poses = {"target": ((0.0, 0.0, 0.0), phase),
                 "boomtip": ((apex * math.cos(phase), apex * math.sin(phase), 0.0), phase),
                 "craft": ((x, y, 0.0), theta)}
        for b, (p, yaw) in poses.items():
            i = self.mocap[b]
            d.mocap_pos[i] = p
            d.mocap_quat[i] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mj.mj_kinematics(self.model, d)
        mj.mj_camlight(self.model, d)

    def _draw_sun(self, img, x, y, theta):
        """The sun's disc, saturated (no glare model): the skybox carries no sun, the directional light does not draw
        one. Occluded by the drum's bounding cylinder."""
        np = self.np
        sun = self.sun
        pos, fwd, up, right = camera_pose(x, y, theta)
        zc = sum(sun[i] * fwd[i] for i in range(3))
        if zc <= 0.0:
            return
        u = CX + FX * sum(sun[i] * right[i] for i in range(3)) / zc
        v = CY - FY * sum(sun[i] * up[i] for i in range(3)) / zc
        r_px = FX * math.tan(SUN_RADIUS) / zc
        if not (-r_px - 1 <= u <= WIDTH + r_px and -r_px - 1 <= v <= HEIGHT + r_px):
            return
        R, z0, z1 = DRUM_OCCLUDER
        a = sun[0] ** 2 + sun[1] ** 2
        b = 2.0 * (pos[0] * sun[0] + pos[1] * sun[1])
        c = pos[0] ** 2 + pos[1] ** 2 - R * R
        disc = b * b - 4.0 * a * c
        if a > 0 and disc >= 0:
            for t in ((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)):
                if t > 0 and z0 <= pos[2] + t * sun[2] <= z1:
                    return
        vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH]
        img[(uu - u) ** 2 + (vv - v) ** 2 <= r_px * r_px] = 255

    def render_rgb(self, x, y, theta, phase, port_radius):
        """The linear RGB8 render (H x W x 3, row 0 = top) of the DOCK camera of a craft at (x, y, theta), the target
        at phase `phase` with its port at `port_radius`."""
        F, np = self.F, self.np
        self._pose(x, y, theta, phase, port_radius)
        self.lights.update(self.data)
        self.rends.update(self.data)
        pos, fwd, up, _ = camera_pose(x, y, theta)
        cam = F.Camera(pos=np.asarray(pos, np.float32), forward=np.asarray(fwd, np.float32),
                       up=np.asarray(up, np.float32), frustum_center=0.0,
                       frustum_width=NEAR * (WIDTH / 2.0) / FX, frustum_bottom=-NEAR * (HEIGHT / 2.0) / FY,
                       frustum_top=NEAR * (HEIGHT / 2.0) / FY, frustum_near=NEAR, frustum_far=FAR)
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT, post_processing=False)
        req.scene, req.target = self.scn, self.tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, WIDTH, HEIGHT
        self.ctx.wait_for_frame(self.ctx.render([req], [self.read]))
        img = self.buf.copy()
        self._draw_sun(img, x, y, theta)
        return img

    def luma(self, rgb):
        np = self.np
        lum = (rgb[..., 0].astype(np.float32) * np.float32(LUMA[0]) + rgb[..., 1].astype(np.float32) * np.float32(LUMA[1])
               + rgb[..., 2].astype(np.float32) * np.float32(LUMA[2]))
        return lum, (rgb == 255).all(-1)

    def frame(self, key, seed, stamp, x, y, theta, phase, port_radius):
        """The sensor's frame (uint8, 480 x 640): the render, its luma, the noise field of (key, seed, stamp)."""
        lum, sat = self.luma(self.render_rgb(x, y, theta, phase, port_radius))
        return expose(lum, sat, frame_noise_rng(key, seed, stamp))

    def close(self):
        """Free every Filament object, then the context (in the thread that made them)."""
        import gc
        for name in ("read", "tgt", "rends", "lights", "ibl", "scn", "res", "ibl_tex", "sky_tex"):
            if hasattr(self, name):
                setattr(self, name, None)
        gc.collect()
        self.ctx = None
        gc.collect()


LED_EMISSION = 50.0     # (MuJoCo material emission: the LEDs saturate at every range; checked by tests/review)
