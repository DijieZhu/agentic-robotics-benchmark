"""sensors_render.py -- the frames of the robot's five cameras (the frames of camera() and the
observability rules): the renderer the cell plugs in instead of ``api.NullRenderer`` (same four methods).

The render model ("twin").  The cell's physics model is never changed and never written: the renderer builds its
own model of the same site (``station.sites.build_site_spec`` with the station LOOK -- a look root, an input, the
transitional look v0.3 by default -- and the same robot and dock files), so the twin has the physics model's bodies
and joints (checked: nq, nv and ``server.model_signature`` equal).  Before each frame it copies qpos (and mocap) of
the physics data into its own data and runs kinematics there.  In the twin:
  * the robot that the cameras can see is the maker's public mesh, exactly as given (the robot the agent sees is the maker's own mesh): the asset
    ``robot_author`` (44 objects, 105 015 triangles; positions, triangle indices, texture coordinates and normals as
    written in the files, the node scale, the placement on the cell's bodies, the files' colours and textures) is
    added as visual-only geoms on the robot's bodies; every display geom of the reconstructed robot is hidden;
  * geom groups: 0-2 visible as built, 4 = every other visual-only geom (visible), 5 = the hidden reconstruction; the
    depth rays see groups 1-3 only, which then hold exactly the collidable geoms (the collision world);
  * the robot's two lamps are spot lights on the core (``spacekit.cameras.add_robot_lamps``) at the published lamp
    positions with the intensity of key ``dev.lamp.intensity_cd`` (452 cd) through the calibration below.

The render recipe (each item keyed in the runtime site_spec where it has a key):
  (a) near clip ``render.near_clip_m`` (1.5 cm) on every camera (MuJoCo: model-wide znear x extent);
  (b) the headlight of the module the camera is in (``station.scene.HEADLIGHT``: general = ambient 0.32 + a
      camera-attached diffuse 0.18; night = ambient x 0.35, no attached diffuse; failed circuit = ambient 0.016,
      none); the circuit's spot lights and luminaire emission follow ``set_lighting``; the classic renderer draws
      at most 8 lights: the headlight, the lamps that are on and the nearest cabin lights that are on;
  (c) the two lamps as 452 cd spots: MuJoCo diffuse at 1 m = (I / E_general) x E_model, where E_model is the
      general-lighting irradiance (ambient + the cabin spot lights, MuJoCo units) at the eight inlet faces and
      E_general = 150 lux [G] (the cabin's general illuminance); inverse-square fall-off, the 22.5 deg
      half beam with a cos^4.5 taper to the 30 deg cut-off (spacekit LAMP_DEFAULTS);
  (d) centre-weighted auto exposure in ``render.exposure_gain`` (1/16 - 8 x), a camera's AE loop: the frame is
      drawn at the camera's last gain, the gain that brings the Gaussian-weighted mean luminance (display values,
      in which the classic renderer is linear) to mid grey encoded (0.461) is computed from it and lowered while more than 2 % of the frame
      would clip over the frame or under the centre weighting (highlight protection); beyond a third of a stop the
      frame is drawn again at that gain; the gain scales every light, the
      headlight and the material emission (a longer or shorter exposure);
  (e) colour noise: per pixel and channel N(0, sqrt(read^2 + k x DN)), read 2 DN, k 0.25 DN per DN (nominal),
      rounded and clipped; depth: N(0, ``render.depth_noise_frac`` x z) on the ray-cast z, 0 = no return
      outside ``sens.depth.range``; amplitude [G] = 3000 x albedo / z^2 (albedo of the hit geom's colour).
Noise comes from the renderer's own generator (seed = the cell's seed, the frame count, the camera): rendering does
not draw from the cell's random streams, so a run's world-state hashes are the same with and without frames.

One OpenGL context per process (module-level, made at the first render, kept across sites); one MjrContext per twin.

Use:
    from cabincell import sensors_render as SR
    server = SR.make_server(**server_kwargs)       # a CabinCellServer with a bound CellRenderer
    SR.delivered_intrinsics()                      # the delivered pinhole model of every camera (= api.frame_model)
Lower level (check_observability, tests): ``FrameRenderer(site, look_root=None)`` with ``sync(model, data)``,
``color(cam)``, ``segmentation(cam)``, ``depth(cam)``, ``set_lighting``, ``set_lamp``, ``close``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from . import api as API

__all__ = ["CellRenderer", "FrameRenderer", "make_server", "delivered_intrinsics", "recipe", "author_dir",
           "load_author", "AUTHOR_INDEX_SHA256", "AUTHOR_ARCHIVE_SHA256", "GROUP_VISIBLE", "GROUP_HIDDEN"]

HERE = Path(__file__).resolve().parent
AUTHOR_COMPONENT = "robot_author"
AUTHOR_ARCHIVE_NAME = "robot_author_v1_1.tar.xz"
AUTHOR_ARCHIVE_SHA256 = "4cb81f46e3040ce80066114aba9a21e43bd647538d4f8eaed9f6e93c486c6e2e"
AUTHOR_INDEX_SHA256 = "03769c06f993b1e70dd07ad050f537788699e26d72b3b9e2127c162f5ff99d7c"
AUTHOR_SCHEMA = "cabincell-robot-author/1"
ENV_AUTHOR_ARCHIVE = "CABINCELL_ROBOT_AUTHOR_ARCHIVE"   # the maintainers' staged archive (unpacked next to it)

GROUP_OTHER_VISUAL = 4      # visual-only geoms of the station, items, dock and the author robot
GROUP_HIDDEN = 5            # the reconstructed robot's display geoms
GROUP_VISIBLE = (0, 1, 2, GROUP_OTHER_VISUAL)
DEPTH_GROUPS = (1, 2, 3)    # after the regrouping: exactly the collidable geoms
ROBOT_BODY_PATTERNS = ("core", "forward_sensor_module", "battery_*", "propulsion_module_*", "pm_s_*", "pm_p_*", "arm_*",
                       "grip_*", "gripper_*")
LAMP_WHICH = {"forward": "front", "aft": "aft"}          # the robot API's lamp names -> spacekit LAMPS keys

# [G] recipe constants without a contract key (choices of the renderer)
GENERAL_LUX = 150.0
NOISE_READ_DN = 2.0
NOISE_SHOT_K = 0.25
AE_TARGET_DISPLAY = 0.461     # mid grey (18 % linear) sRGB-encoded
AE_SIGMA_FRAC = 0.25
AE_CLIP_LEVEL = 254
AE_CLIP_MAX = 0.02
AMPLITUDE_K = 3000.0
SPECULAR_DIELECTRIC = 0.04


# ============================================================================================== recipe
def _runtime_values() -> dict:
    from . import budgets as BU
    return BU.load_runtime()["values"]


def _spec_values() -> dict:
    p = HERE / "data" / "robot" / "spec_values.json"
    doc = json.loads(p.read_text())
    return doc.get("values", doc)


def _val(values: dict, key: str):
    """A value of the runtime site_spec ({key: value}) or of a document with {key: {'value': ...}} entries."""
    x = values[key]
    return x["value"] if isinstance(x, dict) and "value" in x else x


def recipe(values: dict | None = None, robot_values: dict | None = None) -> dict:
    """The render recipe with its sources (contract keys read at run time, robot keys from data/robot)."""
    v = values if values is not None else _runtime_values()
    rv = robot_values if robot_values is not None else _spec_values()
    lamp = _val(rv, "dev.lamp")
    pos = _val(rv, "dev.lamp.positions")
    g = _val(v, "render.exposure_gain")
    return dict(
        near_clip_m=float(_val(v, "render.near_clip_m")),
        exposure_range=(float(g[0]), float(g[1])),
        depth_noise_frac=float(_val(v, "render.depth_noise_frac")),
        depth_range_m=tuple(float(x) for x in _val(rv, "sens.depth.range")),
        lamp_intensity_cd=float(_val(rv, "dev.lamp.intensity_cd")),
        lamp_view_deg=float(lamp["viewing_angle_deg"]),
        lamp_positions={k: tuple(float(x) for x in p) for k, p in pos.items()},
        general_lux=GENERAL_LUX, noise_read_dn=NOISE_READ_DN, noise_shot_k=NOISE_SHOT_K,
        ae_target_display=AE_TARGET_DISPLAY, ae_sigma_frac=AE_SIGMA_FRAC, ae_clip_level=AE_CLIP_LEVEL,
        ae_clip_max=AE_CLIP_MAX, amplitude_k=AMPLITUDE_K)


def delivered_intrinsics() -> dict:
    """The delivered pinhole model of every camera (the one function the cell's frames, the evidence predicate and the
    manuals use: ``api.frame_model``): {name: {role, width, height, fx, fy, cx, cy, hfov_deg, vfov_deg}}."""
    out = {}
    for name in API.CAMERAS:
        fm = API.frame_model(name)
        K, W, H = fm["K"], fm["width"], fm["height"]
        out[name] = dict(role=fm["role"], width=W, height=H, fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                         cy=float(K[1, 2]), depth=bool(fm["depth"]),
                         hfov_deg=math.degrees(math.atan(K[0, 2] / K[0, 0]) + math.atan((W - K[0, 2]) / K[0, 0])),
                         vfov_deg=math.degrees(math.atan(K[1, 2] / K[1, 1]) + math.atan((H - K[1, 2]) / K[1, 1])))
    return out


# ============================================================================================== the author asset
def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _verify_author_tree(d: Path) -> dict:
    """index.json of an unpacked robot_author directory after the chain check: its sha256 is AUTHOR_INDEX_SHA256 and
    every file it lists has its sha256; no other file."""
    raw = (d / "index.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUTHOR_INDEX_SHA256:
        raise RuntimeError(f"{d}/index.json is not the robot_author index this renderer was built with")
    idx = json.loads(raw.decode("ascii"))
    if idx.get("schema") != AUTHOR_SCHEMA:
        raise RuntimeError(f"{d}/index.json: schema {idx.get('schema')!r}")
    have = set()
    for dp, _dn, fn in os.walk(d):
        for f in fn:
            have.add(os.path.relpath(os.path.join(dp, f), d).replace(os.sep, "/"))
    want = set(idx["files"]) | {"index.json"}
    if have != want:
        raise RuntimeError(f"{d}: files differ from the index (missing {sorted(want - have)[:3]}, extra {sorted(have - want)[:3]})")
    for rel, sha in idx["files"].items():
        if _sha256_file(d / rel) != sha:
            raise RuntimeError(f"{d}/{rel}: sha256 differs from the index")
    return idx


def _unpack_author(archive: Path, cache: Path) -> Path:
    if _sha256_file(archive) != AUTHOR_ARCHIVE_SHA256:
        raise RuntimeError(f"{archive}: not the robot_author archive {AUTHOR_ARCHIVE_SHA256[:16]}...")
    final = cache / AUTHOR_ARCHIVE_SHA256[:16] / AUTHOR_COMPONENT
    if not final.is_dir():
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".robot_author_", dir=final.parent))
        try:
            with tarfile.open(archive, "r:xz") as tar:
                members = []
                for ti in tar.getmembers():
                    parts = Path(ti.name).parts
                    if ti.name.startswith("/") or ".." in parts or not parts or parts[0] != AUTHOR_COMPONENT \
                            or not (ti.isfile() or ti.isdir()):
                        raise RuntimeError(f"{archive}: bad member {ti.name!r}")
                    members.append(ti)
                try:
                    tar.extractall(tmp, members=members, filter="data")
                except TypeError:
                    tar.extractall(tmp, members=members)
            try:
                os.rename(tmp / AUTHOR_COMPONENT, final)
            except OSError:
                if not final.is_dir():
                    raise
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    return final


def author_dir() -> Path:
    """The verified robot_author directory.  Search: (1) the cell's asset root mechanism when the task carries the
    manifest data/robot_author/ASSETS.json (assetroot.find); (2) $CABINCELL_ASSET_ROOT/robot_author or the image's
    /opt/cabincell/assets/robot_author; (3) $CABINCELL_ROBOT_AUTHOR_ARCHIVE (the staged archive, unpacked into
    <its folder>/unpacked); (4) the maintainers' staging pointer (assetroot.dev_pointer)."""
    import assetroot
    if (HERE / "data" / AUTHOR_COMPONENT / "ASSETS.json").is_file():
        d = assetroot.find(AUTHOR_COMPONENT)
        _verify_author_tree(d)
        return d
    for root in (os.environ.get(assetroot.ENV), str(assetroot.IMAGE_ROOT)):
        if root and (Path(root) / AUTHOR_COMPONENT).is_dir():
            d = Path(root) / AUTHOR_COMPONENT
            _verify_author_tree(d)
            return d
    arc = os.environ.get(ENV_AUTHOR_ARCHIVE)
    if arc:
        d = _unpack_author(Path(arc), Path(arc).parent / "unpacked")
        _verify_author_tree(d)
        return d
    ptr = assetroot.dev_pointer(AUTHOR_COMPONENT)
    if ptr.is_file():
        st = json.loads(ptr.read_text())
        d = _unpack_author(Path(st["archive"]), Path(st["cache"]))
        _verify_author_tree(d)
        return d
    raise RuntimeError(f"robot_author asset not found ({AUTHOR_ARCHIVE_NAME}, sha256 {AUTHOR_ARCHIVE_SHA256}): set "
                       f"${assetroot.ENV} or ${ENV_AUTHOR_ARCHIVE}")


_AUTHOR_CACHE: dict = {}


def load_author(d: Path | None = None) -> dict:
    """{'index', 'objects': [{rec, arrays: {kind: ndarray}}], 'textures': {name: (H, W, C) uint8}} of the asset."""
    d = Path(d) if d is not None else author_dir()
    key = str(d)
    if key in _AUTHOR_CACHE:
        return _AUTHOR_CACHE[key]
    idx = _verify_author_tree(d)
    arrs = {}
    for h, a in idx["arrays"].items():
        x = np.fromfile(d / a["file"], dtype=np.dtype(a["dtype"])).reshape(a["shape"])
        arrs[h] = x
    objs = [dict(rec=o, arrays={k: arrs[h] for k, h in o["arrays"].items()}) for o in idx["objects"]]
    from PIL import Image
    tex = {}
    for name, t in idx["textures"].items():
        im = Image.open(d / t["file"])
        im.load()
        px = np.asarray(im).copy()
        if hashlib.sha256(np.ascontiguousarray(px).tobytes()).hexdigest() != t["pixels_sha256"]:
            raise RuntimeError(f"texture {name}: pixels differ from the index")
        tex[name] = px
    out = dict(index=idx, objects=objs, textures=tex, dir=str(d))
    _AUTHOR_CACHE[key] = out
    return out


# ============================================================================================== the twin
def _body_matches(name: str, pats=ROBOT_BODY_PATTERNS) -> bool:
    return any(name == p or (p.endswith("*") and name.startswith(p[:-1])) for p in pats)


def _shininess(roughness: float) -> float:
    """MuJoCo shininess (Blinn-Phong exponent / 128) of a GGX roughness: n = 2 / r^4 - 2 (the mapping the hero
    pipeline's table used the other way round: 0.44 <-> the files' exponent 50)."""
    n = 2.0 / max(roughness, 1e-3) ** 4 - 2.0
    return float(min(1.0, max(0.0, n / 128.0)))


def add_author_robot(spec, author: dict, robot_bodies: set) -> list:
    """Add the author objects as visual-only mesh geoms on their bodies (group GROUP_OTHER_VISUAL); returns the
    geom names."""
    import mujoco
    from spacekit.rotations import mat_to_quat
    names = []
    for tname, px in sorted(author["textures"].items()):
        t = spec.add_texture()
        t.name = "rr_tex_" + tname.split(".")[0]
        t.type = mujoco.mjtTexture.mjTEXTURE_2D
        rgb = np.ascontiguousarray(px[..., :3] if px.ndim == 3 else np.repeat(px[..., None], 3, -1), np.uint8)
        t.width, t.height, t.nchannel = int(rgb.shape[1]), int(rgb.shape[0]), 3
        t.data = rgb.tobytes()
        t.vflip = True          # COLLADA texture coordinates have their origin at the bottom left of the image
    pbr = author["index"]["pbr_table"]
    mats = {}
    for k, o in enumerate(author["objects"]):
        rec, arr = o["rec"], o["arrays"]
        if rec["body"] not in robot_bodies:
            raise ValueError(f"author object {rec['name']}: body {rec['body']} is not a robot body of the model")
        mk = (json.dumps(rec["material"], sort_keys=True), rec["pbr_row"])
        if mk not in mats:
            mt = spec.add_material()
            mt.name = f"rr_mat_{len(mats):02d}"
            mt.rgba = [float(x) for x in rec["material"]["rgba"]]
            row = pbr[rec["pbr_row"]]
            mt.specular = SPECULAR_DIELECTRIC
            mt.shininess = _shininess(float(row["roughness"]))
            mt.roughness = float(row["roughness"])
            mt.metallic = float(row["metallic"])
            mt.reflectance = 0.0
            if "texture" in rec["material"]:
                tx = list(mt.textures)
                tx[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "rr_tex_" + rec["material"]["texture"].split(".")[0]
                mt.textures = tx
            mats[mk] = mt.name
        m = spec.add_mesh()
        m.name = "rr_mesh_" + rec["name"]
        m.uservert = arr["vert"].reshape(-1).tolist()
        m.userface = arr["face"].reshape(-1).tolist()
        m.scale = [float(x) for x in rec["node_scale"]]
        if "nrm" in arr:
            m.usernormal = arr["nrm"].reshape(-1).tolist()
            m.userfacenormal = arr["nrmidx"].reshape(-1).tolist()
        if "tc" in arr:
            m.usertexcoord = arr["tc"].reshape(-1).tolist()
            m.userfacetexcoord = arr["tcidx"].reshape(-1).tolist()
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        T = np.asarray(rec["T_body_from_file"], float)
        b = spec.body(rec["body"])
        g = b.add_geom()
        g.name = "rr_geom_" + rec["name"]
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = m.name
        g.pos = T[:3, 3].tolist()
        g.quat = [float(x) for x in mat_to_quat(T[:3, :3])]
        g.contype, g.conaffinity = 0, 0
        g.group = GROUP_OTHER_VISUAL
        g.material = mats[mk]
        g.density = 0.0
        g.mass = 0.0
        names.append(g.name)
    return names


def lamp_calibration(model, data, feats, recipe_: dict) -> dict:
    """MuJoCo diffuse of a lamp at 1 m (attenuation 1/d^2) for the recipe's intensity: E_model at the eight inlet
    faces under general lighting (headlight ambient + every lum_* spot light, cos-weighted, no shadows) times
    I / E_general."""
    import mujoco
    from station import scene as SC
    amb = SC.HEADLIGHT["general"]["ambient"]
    vals = []
    for inl in feats["v02"]["inlets"]:
        c = np.asarray(inl["face_center"] if "face_center" in inl else inl.get("center"), float)
        n = np.asarray(inl["face_normal"] if "face_normal" in inl else inl.get("normal"), float)
        n = n / np.linalg.norm(n)
        s = 0.0
        for i in range(model.nlight):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or ""
            if not nm.startswith("lum_"):
                continue
            lp, ld = data.light_xpos[i], data.light_xdir[i] / np.linalg.norm(data.light_xdir[i])
            v = c - lp
            dist = float(np.linalg.norm(v))
            u = v / dist
            cosang = float(u @ ld)
            if cosang < math.cos(math.radians(float(model.light_cutoff[i]))):
                continue
            a0, a1, a2 = (float(x) for x in model.light_attenuation[i])
            s += SC.SPOT_DIFFUSE * cosang ** float(model.light_exponent[i]) / (a0 + a1 * dist + a2 * dist * dist) \
                * max(0.0, float(-u @ n))
        vals.append(amb + s)
    E = float(np.mean(vals))
    return dict(E_model_general=E, E_model_min=float(np.min(vals)), E_model_max=float(np.max(vals)),
                diffuse_at_1m=E * recipe_["lamp_intensity_cd"] / recipe_["general_lux"])


def build_twin(site: dict, look_root: str | None, author: dict, recipe_: dict):
    """(twin MjModel, info, report) of a site (module doc)."""
    import warnings

    import mujoco
    from spacekit import cameras as CA

    from .site import build as SB
    from station import data as SD
    from station import sites as SI

    S = SB.station_site(site)
    S["robot"] = dict(S.get("robot") or {}, present=True)
    paths = SB.robot_paths()
    side = SB.pad_side(site)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec, info = SI.build_site_spec(S, look_root, paths["robot_xml"], paths["dock_xml"], pad_fid_side=side)
    from .server import ARENA_BYTES
    spec.memory = ARENA_BYTES
    robot_bodies = {b.name for b in spec.bodies if _body_matches(b.name)}
    n_hidden, n_regroup = 0, 0
    for b in spec.bodies:
        robot = b.name in robot_bodies
        for g in b.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                if robot:
                    g.group = GROUP_HIDDEN
                    n_hidden += 1
                elif g.group != GROUP_OTHER_VISUAL:
                    g.group = GROUP_OTHER_VISUAL
                    n_regroup += 1
            elif robot and g.group in (0, 1, 2):
                raise ValueError(f"collidable robot geom {g.name} in a visible group")
    author_geoms = add_author_robot(spec, author, robot_bodies)
    pos = recipe_["lamp_positions"]
    lamps = {k: dict(pos=tuple(pos[k]), dir=CA.LAMPS[k]["dir"]) for k in ("front", "aft")}
    lamp_names = CA.add_robot_lamps(spec, body="core", lamps=lamps, active=False,
                                    intensity_cd=recipe_["lamp_intensity_cd"], diffuse=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec.compile()
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 1366)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 1024)
    CA.set_near_clip(model, recipe_["near_clip_m"])
    for name in API.CAMERAS:
        fm = API.frame_model(name)
        if not fm["depth"]:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, fm["cam"])
            CA.set_intrinsics(model, cid, fm["K"], (fm["width"], fm["height"]))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    feats = SD.features(info.get("scene", "mission"))
    cal = lamp_calibration(model, data, feats, recipe_)
    for nm in lamp_names:
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, nm)
        model.light_diffuse[lid] = [cal["diffuse_at_1m"]] * 3
        model.light_specular[lid] = [0.1, 0.1, 0.1]
    report = dict(n_hidden_reconstruction=n_hidden, n_regrouped_visual=n_regroup, n_author_geoms=len(author_geoms),
                  author_geoms=author_geoms, lamps=lamp_names, lamp_calibration=cal, robot_bodies=sorted(robot_bodies),
                  look_root=look_root, ngeom=int(model.ngeom), nmesh=int(model.nmesh))
    return model, info, report


# ============================================================================================== the GL context
_GL = {"ctx": None, "size": (0, 0)}


def _gl_context(w: int, h: int):
    import mujoco
    if _GL["ctx"] is None:
        _GL["ctx"] = mujoco.GLContext(int(w), int(h))
        _GL["size"] = (int(w), int(h))
    elif w > _GL["size"][0] or h > _GL["size"][1]:
        raise RuntimeError(f"the process's GL context is {_GL['size']}, a frame needs {(w, h)}")
    _GL["ctx"].make_current()
    return _GL["ctx"]


# ============================================================================================== frames of one site
class FrameRenderer:
    """The twin of one site and its frames (module doc).  ``site``: the site dict; ``look_root``: the unpacked station
    look (default the cell's station asset, the transitional look v0.3); ``seed``: the noise seed."""

    def __init__(self, site: dict, look_root: str | None = None, seed: int = 0, recipe_: dict | None = None,
                 author: dict | None = None, values: dict | None = None):
        import mujoco
        from spacekit.refbuild import has_correction
        from spacekit.sensors import DepthCamera
        from station import data as SD

        import assetroot
        self.recipe = recipe_ or recipe(values)
        self.look_root = str(look_root) if look_root is not None else str(assetroot.find("station"))
        self.author = author or load_author()
        self.model, self.info, self.report = build_twin(site, self.look_root, self.author, self.recipe)
        m = self.model
        self.data = mujoco.MjData(m)
        self.feats = SD.features(self.info.get("scene", "mission"))
        self.seed = int(seed)
        self.n_frames = 0
        self.lighting = {k: "general" for k in self.feats["modules"]}
        self.lighting.update({k: v for k, v in (self.info.get("lighting") or {}).items() if k in self.lighting})
        self.lamps_on = {"front": False, "aft": False}
        self.lid = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or ""): i for i in range(m.nlight)}
        self.mid = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""): i for i in range(m.nmat)}
        self.frames = {k: API.frame_model(k) for k in API.CAMERAS}
        self.cam_id = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, v["cam"]) for k, v in self.frames.items()}
        self.core = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "core")
        rigid = [b for b in range(m.nbody) if m.body_weldid[b] == m.body_weldid[self.core]]
        ray_excl = None if has_correction(m, "fwd_depth_window") else rigid
        gg = np.zeros(6, np.uint8)
        gg[list(DEPTH_GROUPS)] = 1
        self.depth_cams = {}
        for k, fm in self.frames.items():
            if fm["depth"]:
                K = fm["K"]
                self.depth_cams[k] = DepthCamera(m, fm["cam"], fm["width"], fm["height"], K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                                 kind="camera", range_m=self.recipe["depth_range_m"], noise=(0.0, 0.0),
                                                 exclude_body=self.core, geomgroup=gg, exclude_bodies=ray_excl)
        self.albedo = self._geom_albedo()
        self._base = None
        self._apply_lighting_all()
        self._con = self._scn = None
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 0
        for gid in GROUP_VISIBLE:
            self._opt.geomgroup[gid] = 1
        self._cam = mujoco.MjvCamera()
        self._pert = mujoco.MjvPerturb()
        self.last = {}
        self._gain = {}
        self._wcache = {}
        self._sig_lut = np.sqrt(self.recipe["noise_read_dn"] ** 2 + self.recipe["noise_shot_k"] * np.arange(256)).astype(np.float32)
        self._sig_lut_cv = self._sig_lut.reshape(1, 256).copy()
        self._nbuf = {}
        self._rbuf = {}

    # -- lighting -----------------------------------------------------------------------------------------------
    def _geom_albedo(self):
        m = self.model
        rgba = np.array(m.geom_rgba, float)
        mat = np.asarray(m.geom_matid)
        rgba[mat >= 0] = np.asarray(m.mat_rgba)[mat[mat >= 0]]
        return rgba[:, :3] @ np.array([0.2126, 0.7152, 0.0722])

    def _apply_lighting_all(self):
        from station import scene as SC
        m = self.model
        for nm, i in self.lid.items():
            parts = nm.split("_")
            if len(parts) >= 3 and parts[0] == "lum":
                st = self.lighting.get(parts[1], "general")
                sc = SC.LIGHT_SCALE[st]
                m.light_diffuse[i] = [SC.SPOT_DIFFUSE * sc, SC.SPOT_DIFFUSE * sc, SC.SPOT_DIFFUSE * sc * (0.97 if st != "night" else 0.80)]
                m.light_active[i] = sc > 0
            elif nm.startswith("lamp_"):
                m.light_active[i] = bool(self.lamps_on.get(nm[len("lamp_"):], False))
        self._active_base = np.array(m.light_active, bool)
        for nm, i in self.mid.items():
            for cls, em in SC.LUM_EMISSION.items():
                pre = f"m_{cls}_"
                if nm.startswith(pre) and nm[len(pre):] in self.lighting:
                    m.mat_emission[i] = em * SC.LIGHT_SCALE[self.lighting[nm[len(pre):]]]
        self._base = None

    def set_lighting(self, module: str, state: str) -> None:
        from station import scene as SC
        if state not in SC.LIGHT_SCALE:
            raise ValueError(f"lighting state {state!r}")
        self.lighting[module] = state
        self._apply_lighting_all()

    def set_lamp(self, which: str, on: bool) -> None:
        k = LAMP_WHICH.get(which, which)
        if k not in self.lamps_on:
            raise ValueError(f"lamp {which!r}")
        self.lamps_on[k] = bool(on)
        self._apply_lighting_all()

    MAX_GL_LIGHTS = 8          # the classic renderer's fixed-function pipeline draws at most 8 lights (headlight first)

    def _select_lights(self, cam: str) -> None:
        """The lights a frame draws: the classic renderer uses at most 8 (the headlight, then the active model lights
        in id order; any beyond are dropped, which would drop the robot's lamps first).  Keep the headlight and the
        lamps that are on, then the cabin lights that are on, nearest to the camera first."""
        m = self.model
        base = self._active_base
        lamps = [i for nm, i in self.lid.items() if nm.startswith("lamp_") and base[i]]
        lums = [i for nm, i in self.lid.items() if nm.startswith("lum_") and base[i]]
        room = self.MAX_GL_LIGHTS - (1 if m.vis.headlight.active else 0) - len(lamps)
        cpos = self.data.cam_xpos[self.cam_id[cam]]
        lums.sort(key=lambda i: (float(np.linalg.norm(self.data.light_xpos[i] - cpos)), i))
        keep = set(lamps) | set(lums[:max(0, room)])
        act = np.zeros(m.nlight, bool)
        act[list(keep)] = True
        m.light_active[:] = act
        self.last_lights = sorted(keep)

    def module_of_camera(self, cam: str) -> str | None:
        from station import scene as SC
        return SC.module_of_point(self.feats, self.data.cam_xpos[self.cam_id[cam]])

    # -- state --------------------------------------------------------------------------------------------------
    def sync(self, model, data) -> None:
        """Copy the physics state into the twin (qpos, mocap) and run kinematics there; ``model`` must have the
        twin's layout."""
        import mujoco
        if model.nq != self.model.nq or model.nmocap != self.model.nmocap:
            raise ValueError("the physics model does not have the twin's layout")
        self.data.qpos[:] = data.qpos
        if model.nmocap:
            self.data.mocap_pos[:] = data.mocap_pos
            self.data.mocap_quat[:] = data.mocap_quat
        self.data.time = data.time
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    # -- colour -------------------------------------------------------------------------------------------------
    def _context(self):
        import mujoco
        W, H = int(self.model.vis.global_.offwidth), int(self.model.vis.global_.offheight)
        _gl_context(max(W, 1366), max(H, 1024))
        if self._con is None:
            self._con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._con)
            self._scn = mujoco.MjvScene(self.model, maxgeom=12000)
        return self._con

    def _gain_state(self):
        m = self.model
        return (m.light_diffuse.copy(), m.light_ambient.copy(), m.light_specular.copy(), m.mat_emission.copy(),
                np.array(m.vis.headlight.ambient), np.array(m.vis.headlight.diffuse), np.array(m.vis.headlight.specular))

    def _draw(self, cam: str, size, gain: float, seg: bool = False) -> np.ndarray:
        import mujoco
        from station import scene as SC
        m = self.model
        con = self._context()
        mod = self.module_of_camera(cam)
        SC.apply_headlight(m, self.lighting.get(mod, "general") if mod else "general")
        self._select_lights(cam)
        saved = self._gain_state()
        try:
            if not seg:
                m.light_diffuse[:] = saved[0] * gain
                m.light_ambient[:] = saved[1] * gain
                m.light_specular[:] = saved[2] * gain
                m.mat_emission[:] = saved[3] * gain
                m.vis.headlight.ambient = saved[4] * gain
                m.vis.headlight.diffuse = saved[5] * gain
                m.vis.headlight.specular = saved[6] * gain
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED.value
            self._cam.fixedcamid = int(self.cam_id[cam])
            mujoco.mjv_updateScene(m, self.data, self._opt, self._pert, self._cam, mujoco.mjtCatBit.mjCAT_ALL.value, self._scn)
            self._scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = bool(seg)
            self._scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = bool(seg)
            W, H = int(size[0]), int(size[1])
            vp = mujoco.MjrRect(0, 0, W, H)
            mujoco.mjr_render(vp, self._scn, con)
            out = self._rbuf.get((H, W))
            if out is None:
                out = self._rbuf.setdefault((H, W), np.empty((H, W, 3), np.uint8))
            mujoco.mjr_readPixels(out, None, vp, con)
            self._scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
            self._scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False
            return np.ascontiguousarray(out[::-1])
        finally:
            m.light_diffuse[:], m.light_ambient[:], m.light_specular[:], m.mat_emission[:] = saved[:4]
            m.vis.headlight.ambient, m.vis.headlight.diffuse, m.vis.headlight.specular = saved[4], saved[5], saved[6]

    def _meter(self, img) -> tuple:
        """(centre-weighted mean luminance of the display values in [0, 1], brightest channel per pixel) of a frame,
        on every 4th pixel: Gaussian weight of sigma = ae_sigma_frac x the image size around the centre (the
        weighting of spacekit.cameras.exposure_gain).  The classic renderer's output is linear in the light scale
        in display values (no gamma in its lighting), so the exposure is metered there: the target is mid grey
        (18 % linear) encoded, 0.461."""
        sub = img[2::4, 2::4]
        key = sub.shape[:2]
        w = self._wcache.get(key)
        if w is None:
            H, W = img.shape[:2]
            v, u = np.mgrid[2:H:4, 2:W:4]
            sg = float(self.recipe["ae_sigma_frac"])
            w = np.exp(-0.5 * (((u - (W - 1) / 2) / (sg * W)) ** 2 + ((v - (H - 1) / 2) / (sg * H)) ** 2))
            w = (w / w.sum()).astype(np.float64)
            self._wcache[key] = w
        x = sub.astype(np.float32) * (1.0 / 255.0)
        Y = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
        mx = np.maximum(np.maximum(sub[..., 0], sub[..., 1]), sub[..., 2])
        return float(np.sum(w * Y)), mx

    def _clip_share(self, mx) -> float:
        """the larger of the frame's clipped share and its centre-weighted clipped share (metered pixels)"""
        c = mx >= float(self.recipe["ae_clip_level"])
        w = self._wcache[mx.shape[:2]]
        return max(float(np.count_nonzero(c)) / mx.size, float(np.sum(w[c])))

    def _clip_limited_gain(self, mx, g: float) -> float | None:
        """the largest gain at which at most ae_clip_max of the metered pixels clip, both over the frame and under
        the centre weighting (highlight protection of the centre-weighted meter), from a frame drawn at gain g (the
        classic renderer is linear in the lights below its clip): g x L / q with q the larger of the two
        (1 - ae_clip_max) quantiles of the pixels' brightest channel; None when more than that already clips."""
        L = float(self.recipe["ae_clip_level"])
        cm = float(self.recipe["ae_clip_max"])
        v = mx.reshape(-1).astype(np.float64)
        q_frame = float(np.quantile(v, 1.0 - cm))
        w = self._wcache[mx.shape[:2]].reshape(-1)
        o = np.argsort(v, kind="stable")
        cw = np.cumsum(w[o])
        k = int(np.searchsorted(cw, 1.0 - cm, side="left"))
        q_centre = float(v[o][min(k, len(v) - 1)])
        q = max(q_frame, q_centre)
        if q >= L:
            return None
        return g * L / max(q, 0.5)

    def reset_exposure(self, cam: str | None = None) -> None:
        """Forget the auto exposure state (of one camera or all): the next frame starts metering at gain 1."""
        if cam is None:
            self._gain.clear()
        else:
            self._gain.pop(cam, None)

    def color(self, cam: str, noise: bool = True, exposure: float | None = None) -> np.ndarray:
        """(H, W, 3) uint8 colour frame of a colour camera at the current twin state (module doc (b)-(e)).

        Auto exposure (a camera's AE loop): the frame is first drawn at the camera's last gain (1 for its first
        frame); the next gain = the one that brings the centre-weighted mean luminance to ae_target_display,
        capped by the largest gain at which at most ae_clip_max of the pixels clip (half the gain when more already
        clip), clamped to the contract range; within a third of a stop of the drawn gain (and not clipping) the
        drawn frame is the frame, else the frame is drawn again at the new gain (at most 6 more passes).  Exposure,
        clipped share and the passes are in ``self.last``."""
        fm = self.frames[cam]
        W, H = fm["width"], fm["height"]
        lo, hi = self.recipe["exposure_range"]
        cmax = float(self.recipe["ae_clip_max"])
        passes = []
        if exposure is not None:
            g = float(exposure)
            img = self._draw(cam, (W, H), g)
            mean, mx = self._meter(img)
            clip = self._clip_share(mx)
            passes.append(dict(gain=g, mean=mean, clip=clip))
        else:
            g = float(self._gain.get(cam, 1.0))
            img = self._draw(cam, (W, H), g)
            mean, mx = self._meter(img)
            clip = self._clip_share(mx)
            passes.append(dict(gain=g, mean=mean, clip=clip))
            for _ in range(6):
                gt = hi if mean <= 1e-9 else g * float(self.recipe["ae_target_display"]) / mean
                gc = self._clip_limited_gain(mx, g)
                gt = min(gt, gc) if gc is not None else min(gt, 0.5 * g)
                gt = min(max(gt, lo), hi)
                if abs(math.log2(gt / g)) <= 1.0 / 3.0 and clip <= cmax:
                    break
                if gc is None and g <= lo * 1.0001:
                    break
                g = gt
                img = self._draw(cam, (W, H), g)
                mean, mx = self._meter(img)
                clip = self._clip_share(mx)
                passes.append(dict(gain=g, mean=mean, clip=clip))
                if clip <= cmax and (gc is not None or len(passes) >= 3):
                    break
            self._gain[cam] = g
        self.n_frames += 1
        if noise:
            img = self._add_noise(img, [self.seed & 0xFFFFFFFF, 0x52454E44, self.n_frames, list(API.CAMERAS).index(cam)])
        mod = self.module_of_camera(cam)
        self.last = dict(camera=cam, exposure=g, clipped=clip, mean_display=mean, passes=passes, module=mod,
                         lighting=self.lighting.get(mod or "", "general"), lamps=dict(self.lamps_on), frame_no=self.n_frames)
        return img

    def _add_noise(self, img, seed) -> np.ndarray:
        """img + N(0, sigma(DN)) per pixel and channel, sigma = sqrt(read^2 + k x DN), rounded and clipped to 8 bit
        (persistent buffers: no large temporary arrays per frame)."""
        import cv2
        key = img.shape
        buf = self._nbuf.get(key)
        if buf is None:
            buf = (np.empty(key, np.float32), np.empty(key, np.float32))
            self._nbuf[key] = buf
        x, sig = buf
        rng = np.random.default_rng(seed)
        rng.standard_normal(dtype=np.float32, out=x)
        cv2.LUT(img, self._sig_lut_cv, dst=sig)
        cv2.multiply(x, sig, dst=x)
        cv2.add(x, img, dst=x, dtype=cv2.CV_32F)
        out = np.empty(key, np.uint8)
        x.round(out=x)
        np.copyto(out, np.clip(x, 0.0, 255.0, out=x), casting="unsafe")
        return out

    def segmentation(self, cam: str, size=None) -> np.ndarray:
        """(H, W) int32 geom id per pixel (-1 = background) of a colour camera (the visible geom groups)."""
        import mujoco
        fm = self.frames[cam]
        W, H = size or (fm["width"], fm["height"])
        px = self._draw(cam, (W, H), 1.0, seg=True).astype(np.int64)
        sid = px[..., 0] + px[..., 1] * 256 + px[..., 2] * 65536 - 1
        lut = np.full(int(self._scn.ngeom) + 1, -1, np.int64)
        for i in range(int(self._scn.ngeom)):
            gg = self._scn.geoms[i]
            if gg.objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
                lut[i] = int(gg.objid)
        ok = (sid >= 0) & (sid < len(lut))
        out = np.full(sid.shape, -1, np.int64)
        out[ok] = lut[sid[ok]]
        return out.astype(np.int32)

    # -- depth --------------------------------------------------------------------------------------------------
    def depth(self, cam: str, noise: bool = True):
        """(depth float32 H x W metres, 0 = no return; amplitude uint16; geom id int32 per pixel, -1 none)."""
        dc = self.depth_cams[cam]
        z = dc.read(self.model, self.data).astype(np.float64)
        gid = dc._gid.reshape(dc.shape).copy()
        ok = np.isfinite(z)
        self.n_frames += 1
        if noise:
            rng = np.random.default_rng([self.seed & 0xFFFFFFFF, 0x44455054, self.n_frames, list(API.CAMERAS).index(cam)])
            z = np.where(ok, z + rng.standard_normal(z.shape) * self.recipe["depth_noise_frac"] * np.where(ok, z, 0.0), z)
            ok = ok & (z >= self.recipe["depth_range_m"][0]) & (z <= self.recipe["depth_range_m"][1])
        zz = np.where(ok, z, 0.0).astype(np.float32)
        alb = np.where(gid >= 0, self.albedo[np.maximum(gid, 0)], 0.0)
        amp = np.where(ok, np.clip(np.rint(self.recipe["amplitude_k"] * alb / np.maximum(np.where(ok, z, 1.0), 0.05) ** 2), 1, 65535), 0)
        gid = np.where(ok, gid, -1).astype(np.int32)
        self.last = dict(camera=cam, frame_no=self.n_frames, valid=float(ok.mean()))
        return zz, amp.astype(np.uint16), gid

    def close(self) -> None:
        if self._con is not None:
            try:
                _GL["ctx"].make_current()
            except Exception:  # noqa: BLE001
                pass
            self._con.free()
            self._con = None
            self._scn = None


# ============================================================================================== the cell's renderer
class CellRenderer:
    """The renderer the cell plugs in (api.NullRenderer's surface).  Bind it to the server (``make_server`` does) so
    that it can build the twin of the run's site at the first frame."""

    name = "sensors_render"

    def __init__(self, look_root: str | None = None, values: dict | None = None):
        self.look_root = look_root
        self.values = values
        self._server = None
        self._cell = None
        self.fr = None
        self._fr_model = None
        self._lighting = {}
        self._lamps = {}
        self.frame_wall_s = []

    def bind(self, server_or_cell) -> "CellRenderer":
        if hasattr(server_or_cell, "handle"):
            self._server = server_or_cell
        else:
            self._cell = server_or_cell
        return self

    def _current_cell(self):
        return self._server.cell if self._server is not None else self._cell

    def _ensure(self, model):
        if self.fr is not None and self._fr_model is model:
            return self.fr
        cell = self._current_cell()
        if cell is None or cell.model is not model:
            raise RuntimeError("CellRenderer is not bound to the cell of this model (use sensors_render.make_server)")
        if self.fr is not None:
            self.fr.close()
        from .server import model_signature
        self.fr = FrameRenderer(cell.site, look_root=self.look_root, seed=int(cell.seed),
                                values=self.values if self.values is not None else cell.runtime["values"])
        if model_signature(self.fr.model) != model_signature(model) or self.fr.model.nv != model.nv:
            raise RuntimeError("the render twin does not have the physics model's layout")
        self._fr_model = model
        for mod, st in self._lighting.items():
            self.fr.set_lighting(mod, st)
        for w, on in self._lamps.items():
            self.fr.set_lamp(w, on)
        return self.fr

    def render(self, camera: str, model, data) -> dict | None:
        import time
        t0 = time.perf_counter()
        fr = self._ensure(model)
        fr.sync(model, data)
        if self.fr.frames[camera]["depth"]:
            z, amp, _gid = fr.depth(camera)
            out = dict(depth=z, amplitude=amp)
        else:
            out = dict(image=fr.color(camera))
        self.frame_wall_s.append((camera, time.perf_counter() - t0))
        return out

    def set_lighting(self, module: str, state: str) -> None:
        self._lighting[module] = state
        if self.fr is not None:
            self.fr.set_lighting(module, state)

    def set_lamp(self, which: str, on: bool) -> None:
        self._lamps[which] = bool(on)
        if self.fr is not None:
            self.fr.set_lamp(which, on)

    def close(self) -> None:
        if self.fr is not None:
            self.fr.close()
        self.fr = None
        self._fr_model = None
        self._lighting = {}
        self._lamps = {}


def make_server(**kwargs):
    """A CabinCellServer with a bound CellRenderer (kwargs: the server's; ``look_root`` goes to the renderer)."""
    from .server import CabinCellServer
    R = CellRenderer(look_root=kwargs.pop("look_root", None))
    srv = CabinCellServer(renderer=R, **kwargs)
    R.bind(srv)
    return srv
