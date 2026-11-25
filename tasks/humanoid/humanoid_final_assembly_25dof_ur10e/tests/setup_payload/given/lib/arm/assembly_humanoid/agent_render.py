"""The render layer of the cell's two cameras, for your own twin of the cell: the same render twin, room, look and
renderer that the cell renders its camera frames with (MuJoCo's Filament renderer on the CPU, Mesa's lavapipe, in the
baked integration bay; depth from MuJoCo's own renderer on the same geometry).

    from arm.assembly_humanoid import agent_render as AR
    twin = AR.load_twin()                       # render/twin/twin.json (render_dir() below)
    view = AR.open_view(twin)                   # one view per process; about 5.2 GB while it is open
    state = {"head/head_root/pose": [x, y, z, qw, qx, qy, qz], ...}     # the bodies you place (others stay put)
    rgb, depth = view.render(state, camera, depth=True)                  # uint8 H x W x 3, float32 H x W metres
    view.close()

THE TWIN (render/twin/twin.json, meshes in render/twin/meshes/): every body of the cell that has a mesh, as a free body
of its own, named
  "<unit>/<body>"   the 31 bodies of the base and the five modules; the body's frame is its unit's frame as the unit is
                    delivered (the frame of the unit's files under given/parts/<unit>/: module.json gives every body's
                    pose in it at the delivered joint state), carried rigidly by that body;
  "fixture/<id>"    every fixture and cell fixture; the body's frame is the cell frame at the fixture's nominal place,
                    so its pose is the rigid motion of the fixture away from its nominal place (identity = nominal);
  "ur/<body>"       the arm's and the gripper's bodies; their own frames (the body frames of the given models of the arm
                    and of the gripper).
A body's pose is its free joint "<name>/pose" = [x, y, z, qw, qx, qy, qz] (metres; quaternion w x y z) in the cell
frame. twin["bodies"] lists the names; nothing of the twin moves by itself, and a body you do not place keeps its last
pose (at the start: the cell origin, identity). The entry world of the twin holds what never moves: the lit floor
inside the tape of the cell zone and the lamps of the work-light bars. The camera layer's own settings are in
twin["camera_layer"].

CAMERAS: `camera` is a dict {"name", "width", "height", "fx", "fy", "cx", "cy", "world_T_cam"} (pixels; world_T_cam =
the camera's 4 x 4 pose in the cell frame, OpenCV axes: x right, y down, z forward), or a render.spec.CameraSpec. The
name of the camera chooses its exposure: a camera named wrist is exposed as the cell's wrist camera, which is brighter
than the overhead camera by the factor the twin's camera layer gives; a camera of any other name is exposed as the
overhead camera. The layer applies no lens distortion, no noise and no sensor model.

MuJoCo's own renderer (mujoco.Renderer) gives a different picture: no room, MuJoCo's lights, flat colours. This layer is
the way to see what the cell's cameras would see.
"""
import json
import os

import numpy as np

__all__ = ["render_dir", "twin_dir", "load_twin", "twin_spec", "open_view", "pose_joint", "camera_dict"]

TWIN_SCHEMA = "humanoid-asm-render-twin/1"
_LOADED = {}


def render_dir():
    """the cameras' scenery: $ARB_AGENT_RENDER_DIR, else the folder `render` next to $ARB_GIVEN_DIR (where the cell runs
    your program), else ~/render"""
    if os.environ.get("ARB_AGENT_RENDER_DIR"):
        return os.environ["ARB_AGENT_RENDER_DIR"]
    if os.environ.get("ARB_GIVEN_DIR"):
        return os.path.join(os.path.dirname(os.path.abspath(os.environ["ARB_GIVEN_DIR"])), "render")
    return os.path.join(os.path.expanduser("~"), "render")


def twin_dir(rdir=None):
    return os.path.join(rdir or render_dir(), "twin")


def load_twin(tdir=None):
    """the render twin's description (twin.json) with its folder under "_dir" """
    d = os.path.abspath(tdir or twin_dir())
    if d not in _LOADED:
        with open(os.path.join(d, "twin.json")) as f:
            t = json.load(f)
        if t.get("schema") != TWIN_SCHEMA:
            raise ValueError("%s: schema %r, this layer reads %s" % (d, t.get("schema"), TWIN_SCHEMA))
        t["_dir"] = d
        _LOADED[d] = t
    return _LOADED[d]


def pose_joint(body):
    """the free joint that places a twin body"""
    return body + "/pose"


def _room_of(twin, room=None):
    return os.path.abspath(room or os.path.join(os.path.dirname(twin["_dir"]), twin["camera_layer"]["room"]))


def twin_spec(twin=None, room=None):
    """the twin as a mujoco.MjSpec (every body free, every geom visible, group 2; the world-fixed geoms and lamps of
    twin["world"]; no cameras); a material's textures are files of the room folder `room` (default: the cell's)"""
    import mujoco
    twin = twin or load_twin()
    d = twin["_dir"]
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.gravity = [0.0, 0.0, 0.0]
    for name in sorted(twin["materials"]):
        mv = twin["materials"][name]
        m = spec.add_material(name=name)
        m.rgba = [float(v) for v in mv["rgba"]]
        for k in ("specular", "shininess", "reflectance", "metallic", "roughness"):
            if k in mv:
                setattr(m, k, float(mv[k]))
        if "emission" in mv:
            m.emission = float(mv["emission"])
        for role, tx in sorted((mv.get("textures") or {}).items()):
            t = spec.add_texture(name=tx["name"], type=mujoco.mjtTexture.mjTEXTURE_2D,
                                 file=os.path.join(_room_of(twin, room), tx["file"]))
            t.colorspace = (mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if tx["colorspace"] == "srgb"
                            else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)
            t.nchannel = int(tx["nchannel"])
            m.textures[int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_" + role))] = tx["name"]
        if "texrepeat" in mv:
            m.texrepeat = [float(v) for v in mv["texrepeat"]]
            m.texuniform = bool(mv["texuniform"])
    for key in sorted(twin["meshes"]):
        z = np.load(os.path.join(d, twin["meshes"][key]["file"]))
        ms = spec.add_mesh(name=key)
        ms.uservert = z["v"].astype(np.float64).ravel()
        ms.userface = z["f"].astype(np.int32).ravel()
        ms.usernormal = z["n"].astype(np.float64).ravel()
        ms.userfacenormal = z["fn"].astype(np.int32).ravel()
        ms.usertexcoord = z["tc"].astype(np.float64).ravel()
        ms.userfacetexcoord = z["ft"].astype(np.int32).ravel()
        ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    for b in twin["bodies"]:
        body = spec.worldbody.add_body(name=b["name"])
        body.add_freejoint(name=pose_joint(b["name"]))
        body.mass = 1.0
        body.inertia = [1e-3, 1e-3, 1e-3]
        body.explicitinertial = True
        for k, g in enumerate(b["geoms"]):
            gt = getattr(mujoco.mjtGeom, "mjGEOM_" + g["type"].upper())
            kw = {"name": "%s/g%03d" % (b["name"], k), "type": gt, "pos": [float(v) for v in g["pos"]],
                  "quat": [float(v) for v in g["quat"]], "contype": 0, "conaffinity": 0, "group": 2}
            if g["type"] == "mesh":
                kw["meshname"] = g["mesh"]
            else:
                kw["size"] = [float(v) for v in g["size"]]
            if g.get("material"):
                kw["material"] = g["material"]
            gg = body.add_geom(**kw)
            gg.rgba = [float(v) for v in g["rgba"]]
            gg.mass = 0.0
    world = twin.get("world") or {}
    for g in world.get("geoms", []):
        kw = {"name": g["name"], "type": getattr(mujoco.mjtGeom, "mjGEOM_" + g["type"].upper()),
              "pos": [float(v) for v in g["pos"]], "quat": [float(v) for v in g["quat"]], "contype": 0, "conaffinity": 0,
              "group": 2}
        if g["type"] == "mesh":
            kw["meshname"] = g["mesh"]
        else:
            kw["size"] = [float(v) for v in g["size"]]
        if g.get("material"):
            kw["material"] = g["material"]
        gg = spec.worldbody.add_geom(**kw)
        gg.rgba = [float(v) for v in g["rgba"]]
    for li in world.get("lights", []):
        lt = spec.worldbody.add_light(name=li["name"], type=mujoco.mjtLightType(int(li["type"])))
        for k in ("pos", "dir", "diffuse", "specular", "ambient", "attenuation"):
            setattr(lt, k, [float(v) for v in li[k]])
        for k in ("cutoff", "exponent", "bulbradius", "intensity", "range", "softness"):
            setattr(lt, k, float(li[k]))
        lt.castshadow, lt.active = bool(li["castshadow"]), bool(li["active"])
        lt.mode = mujoco.mjtCamLight(int(li["mode"]))
    return spec


class _LayerView(object):
    """the camera layer's view: a render.filament.view FilamentView that renders each camera at the cell's exposure for
    its name (twin["camera_layer"]["exposure_factor"])"""

    def __init__(self, view, exposure):
        self.view, self.exposure = view, dict(exposure or {})
        self.session, self.model, self.data, self.look = view.session, view.model, view.data, view.look
        self.t_open_s = view.t_open_s

    def exposure_factor(self, camera):
        name = camera.get("name") if isinstance(camera, dict) else (camera if isinstance(camera, str) else getattr(camera, "name", None))
        return float(self.exposure.get(str(name), self.exposure.get("default", 1.0)))

    def set_state(self, state=None):
        self.view.set_state(state)
        return self

    def camera(self, camera, width=None, height=None):
        return self.view.camera(camera, width, height)

    def render(self, state=None, camera=None, width=None, height=None, depth=False):
        """RGB uint8 (H, W, 3) of `camera` at `state`; with depth=True (rgb, depth float32 (H, W) metres, 0.0 = no hit)"""
        if camera is None:
            raise ValueError("render needs a camera")
        self.view.set_state(state)
        cam = self.view.camera(camera, width, height)
        v = self.session.render_raw_views([cam], ("rgb", "depth") if depth else ("rgb",),
                                          exposure_factor=self.exposure_factor(camera))[0]
        return (v.rgb, v.depth) if depth else v.rgb

    def close(self):
        self.view.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_view(twin=None, room=None, look=None, **kw):
    """The view of the twin in the room, with the cell's look, its live lamps and the exposure of each camera; the room
    and the look default to the cell's own."""
    from render.filament import view as V
    twin = twin or load_twin()
    cl = twin["camera_layer"]
    room = _room_of(twin, room)
    look = look or os.path.join(twin["_dir"], cl["look"])
    args = {"bench_xy": tuple(cl["bench_xy"]), "max_live_lamps": int(cl["max_live_lamps"])}
    args.update(kw)
    return _LayerView(V.open_view(twin_spec(twin, room=room), room=room, look=look, **args), cl.get("exposure_factor"))


def camera_dict(name, width, height, fx, fy, cx, cy, world_T_cam):
    """a camera for FilamentView.render (OpenCV axes)"""
    T = np.asarray(world_T_cam, float)
    if T.shape != (4, 4):
        raise ValueError("world_T_cam must be 4 x 4")
    return {"name": str(name), "width": int(width), "height": int(height), "fx": float(fx), "fy": float(fy),
            "cx": float(cx), "cy": float(cy), "world_T_cam": T.tolist()}
