"""The dressing pass (see the package docstring): settings, the bundle, the plan and its application to an MjSpec."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

import mujoco
import numpy as np

from . import scheme as S
from .checks import DressError, check_spec_unchanged, check_decor_clearance, check_skins_behind_protected, \
    check_tag_strings, signature
from .skins import face_mesh, wall_faces
from .xmath import dot3, mat_vec, norm

BUNDLE = "qinspect_look_v1"
DRESS_VERSION = "look-1.0"
DEFAULT_BUNDLE_DIRS = ("/opt/task-assets/" + BUNDLE,)
SKIN_OFF = 0.001                  # skins 1 mm in front of the face they cover (the box stays the sensors' surface)
FLOOR_TOP = 0.0015                # the generator's decorative floor slab: top at 1.5 mm (floor markings above it)


# ============================================================================================== settings / bundle
def settings() -> dict:
    mode = os.environ.get("SITE_GEN_DRESS", "auto").strip().lower() or "auto"
    if mode in ("0", "off", "false", "no"):
        mode = "off"
    elif mode in ("1", "on", "true", "yes", "required"):
        mode = "on"
    else:
        mode = "auto"
    return {"mode": mode, "bundle": bundle_dir()}


def bundle_dir():
    """The bundle directory: SITE_GEN_DRESS_BUNDLE when it is set (that folder only, no fallback), else
    the image's default. None when the chosen folder holds no manifest.json."""
    if os.environ.get("SITE_GEN_DRESS_BUNDLE"):
        cands = [os.environ["SITE_GEN_DRESS_BUNDLE"]]
    else:
        cands = list(DEFAULT_BUNDLE_DIRS)
    for c in cands:
        if os.path.isfile(os.path.join(c, "manifest.json")):
            return os.path.abspath(c)
    return None


def enabled() -> bool:
    s = settings()
    if s["mode"] == "off":
        return False
    if s["bundle"] is None:
        if s["mode"] == "on":
            raise DressError("SITE_GEN_DRESS=1 but no look bundle (SITE_GEN_DRESS_BUNDLE or %s)" % DEFAULT_BUNDLE_DIRS[0])
        return False
    return True


class Bundle:
    """The look bundle: manifest.json + tex/*.png + mesh/*.obj; every file read is checked against the manifest."""

    _cache: dict = {}

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        mb = open(os.path.join(self.path, "manifest.json"), "rb").read()
        self.manifest_sha256 = hashlib.sha256(mb).hexdigest()
        self.man = json.loads(mb)
        if self.man.get("format") != "qinspect-look-bundle/1":
            raise DressError("%s: not a look bundle" % self.path)
        self.items = self.man["items"]
        self._data = {}

    @classmethod
    def get(cls, path: str) -> "Bundle":
        key = os.path.abspath(path)
        b = cls._cache.get(key)
        if b is None:
            b = cls(key)
            cls._cache[key] = b
        return b

    def read(self, rel: str) -> bytes:
        b = self._data.get(rel)
        if b is None:
            rec = self.man["files"].get(rel)
            if rec is None:
                raise DressError("bundle %s has no file %s" % (self.path, rel))
            b = open(os.path.join(self.path, rel), "rb").read()
            if hashlib.sha256(b).hexdigest() != rec["sha256"]:
                raise DressError("bundle file %s does not match the manifest" % rel)
            self._data[rel] = b
        return b

    def obj_bbox(self, rel: str):
        """(lo, hi) of an OBJ's vertices (metres, file frame)."""
        key = ("bbox", rel)
        if key not in self._data:
            v = [list(map(float, ln.split()[1:4])) for ln in self.read(rel).decode().splitlines() if ln.startswith("v ")]
            v = np.asarray(v, float)
            self._data[key] = (v.min(0), v.max(0), v)
        return self._data[key]


# ============================================================================================== spec helpers
def _quat_z_to(v):
    """Quaternion turning +z onto the unit vector v."""
    v = np.asarray(v, float) / max(norm(v), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = dot3(z, v)
    if c < -0.999999:
        return [0.0, 1.0, 0.0, 0.0]
    ax = np.cross(z, v)
    q = np.array([1.0 + c, ax[0], ax[1], ax[2]])
    q /= norm(q)
    return [float(x) for x in q]



class Ctx:
    """What the pass adds to the spec: textures (VFS files), materials, meshes, bodies."""

    def __init__(self, spec, bundle: Bundle):
        self.spec = spec
        self.b = bundle
        self.assets = {}
        self.tex = set()
        self.mats = {}
        self.meshes = set()
        self.nbody = 0
        self.added_geoms = []          # (name, lo, hi, kind)
        self.tris = 0

    def texture(self, item: str, rel: str | None = None) -> str:
        if rel is None and item not in self.b.items:
            raise DressError("the look bundle %s has no item %r: the dressing code and the bundle are out of step "
                             "(rebuild the bundle with this tree's make_asset_bundle.py)" % (self.b.path, item))
        rel = rel or self.b.items[item]["files"]["albedo"]
        tn = "dress/tex/" + os.path.splitext(os.path.basename(rel))[0]
        if tn not in self.tex:
            fn = "dress_" + os.path.basename(rel)
            self.assets[fn] = self.b.read(rel)
            # the fake-light and contact-shadow ramps carry their alpha (MuJoCo reads a PNG as RGB unless told)
            nch = 4 if os.path.basename(rel).startswith("fx_") else 3
            self.spec.add_texture(name=tn, type=mujoco.mjtTexture.mjTEXTURE_2D, file=fn, nchannel=nch)
            self.tex.add(tn)
        return tn

    def material(self, key: str, tex: str | None, rgba=(1, 1, 1, 1), specular=0.2, shininess=0.2, emission=0.0,
                 repeat=None, uniform=False) -> str:
        name = "dress/" + key
        if name in self.mats:
            return name
        m = self.spec.add_material(name=name, rgba=[float(v) for v in rgba], specular=float(specular),
                                   shininess=float(shininess), reflectance=0.0, emission=float(emission))
        if tex:
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
            if repeat is not None:
                m.texrepeat = [float(repeat[0]), float(repeat[1])]
            m.texuniform = bool(uniform)
        self.mats[name] = m
        return name

    def body(self, name: str):
        b = self.spec.worldbody.add_body(name=name)
        b.explicitinertial = True
        b.mass = 1.0
        b.inertia = [1.0, 1.0, 1.0]
        b.ipos = [0.0, 0.0, 0.0]
        b.iquat = [1.0, 0.0, 0.0, 0.0]
        self.nbody += 1
        return b

    def user_mesh(self, name: str, V, F, UV):
        m = self.spec.add_mesh(name=name)
        m.uservert = np.asarray(V, float).ravel().tolist()
        m.userface = np.asarray(F, int).ravel().tolist()
        m.usertexcoord = np.asarray(UV, float).ravel().tolist()
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        self.meshes.add(name)
        self.tris += len(F)
        return name

    def file_mesh(self, rel: str, scale=(1.0, 1.0, 1.0)) -> str:
        key = "dress/mesh/%s@%.4f,%.4f,%.4f" % (os.path.splitext(os.path.basename(rel))[0], *scale)
        if key not in self.meshes:
            fn = "dress_" + os.path.basename(rel)
            self.assets[fn] = self.b.read(rel)
            m = self.spec.add_mesh(name=key, file=fn)
            # set explicitly: otherwise MuJoCo's compiler writes it only on whichever mesh of a shared file its (threaded)
            # decoder reached first, and the model's XML (the cell's model hash) changes from run to run
            m.content_type = "model/obj"
            m.scale = [float(s) for s in scale]
            m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            self.meshes.add(key)
        return key

    def finish(self):
        a = dict(self.spec.assets or {})
        for k, v in self.assets.items():
            if k in a and a[k] != v:
                raise DressError("asset name clash %s" % k)
            a[k] = v
        self.spec.assets = a


def _world_info(spec):
    """Compile the undressed spec once: every geom's name, group, material, world AABB, the model."""
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    mats = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(k)) if k >= 0 else "" for k in m.geom_matid]
    lo = np.zeros((m.ngeom, 3))
    hi = np.zeros((m.ngeom, 3))
    for i in range(m.ngeom):
        c = m.geom_aabb[i, :3]
        h = m.geom_aabb[i, 3:]
        R = d.geom_xmat[i].reshape(3, 3)
        p = d.geom_xpos[i] + mat_vec(R, c)
        e = mat_vec(np.abs(R), h)
        if int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            e = np.array([1e3, 1e3, 0.0])
        lo[i] = p - e
        hi[i] = p + e
    return {"model": m, "data": d, "names": names, "mats": mats, "lo": lo, "hi": hi,
            "group": m.geom_group.copy(), "contype": m.geom_contype.copy(), "type": m.geom_type.copy()}


# ============================================================================================== the pass
def maybe_dress(spec, site: dict):
    """The loader's hook: dress when the switch says so (see settings()); returns the report or None."""
    if not enabled():
        return None
    return dress_spec(spec, site)


def _sitegen_keepclear(site):
    try:
        from ..loader.keepclear import keepclear as kc
    except ImportError:
        return None
    r = kc(site)
    A, B = r["segments"]
    return {"geoms": list(r["geoms"]), "segments": (np.asarray(A, float).reshape(-1, 3), np.asarray(B, float).reshape(-1, 3)),
            "floor_rects": np.asarray(r["floor_rects"], float).reshape(-1, 4)}


def dress_spec(spec, site: dict, bundle: str | None = None, check: bool = True, decor: bool = True) -> dict:
    """Dress `spec` in place (see the package docstring). Returns the report."""
    t0 = time.perf_counter()
    bpath = bundle or bundle_dir()
    if bpath is None:
        raise DressError("no look bundle")
    B = Bundle.get(bpath)
    sig0 = signature(spec)
    W = _world_info(spec)
    ctx = Ctx(spec, B)
    seed = int(site.get("seed") or 0) % (2 ** 63)
    rng = np.random.default_rng([seed, 20261116])
    plan = S.site_scheme(site, rng)
    rooms = S.room_index(site)
    protected = np.array([S.is_protected(n, mt) for n, mt in zip(W["names"], W["mats"])], bool)
    keepclear = np.array([S.is_keepclear(n, mt) for n, mt in zip(W["names"], W["mats"])], bool)
    # the generator's keep-clear record (site_gen.loader.keepclear, from site.json alone) when this tree has it: its
    # geoms join the keep-clear set, its sight segments and floor rects join this module's own
    W["_extra"] = _sitegen_keepclear(site)
    if W["_extra"]:
        kg = set(W["_extra"]["geoms"])
        keepclear |= np.array([n in kg for n in W["names"]], bool)
    protected |= keepclear
    report = {"version": DRESS_VERSION, "bundle": B.path, "bundle_manifest_sha256": B.manifest_sha256,
              "scheme": plan["summary"], "counts": {}}
    # ---------------------------------------------------------------- re-material the generator's geoms
    by_name = {}
    for b in spec.bodies:
        for g in b.geoms:
            by_name[g.name] = g
    hidden = ctx.material("hidden", None, rgba=(1, 1, 1, 0))
    n_re = n_hide = 0
    skin_src = []                      # (geom index, kind)
    for i, (nm, mt) in enumerate(zip(W["names"], W["mats"])):
        if protected[i] or nm.startswith("robot/") or not nm:
            continue
        g = by_name.get(nm)
        if g is None:
            continue
        kind = S.structure_kind(nm, mt)
        if kind in ("wall", "partition", "shield"):
            skin_src.append((i, kind))
            new = S.box_material(ctx, kind)
            if new:
                g.material = new
                n_re += 1
            continue
        if kind in ("floor", "ceiling", "skirting"):
            if kind != "skirting":
                skin_src.append((i, kind))
            g.material = hidden
            n_hide += 1
            continue
        if kind == "lamp" and decor:
            skin_src.append((i, kind))
            g.material = hidden
            n_hide += 1
            continue
        new = S.equipment_material(ctx, mt, nm, plan)
        if new:
            g.material = new
            n_re += 1
    report["counts"].update({"rematerialed": n_re, "hidden": n_hide})
    # ---------------------------------------------------------------- skins
    nskin = 0
    sk_body = ctx.body("dress/skins")
    skin_planes = []
    skin_faces = []
    W["_struct_idx"] = np.array([i for i, k in skin_src if k in ("wall", "partition", "shield")], int)
    for i, kind in skin_src:
        lo, hi = W["lo"][i], W["hi"][i]
        if kind in ("wall", "partition", "shield"):
            faces = wall_faces(lo, hi, W, i)
            for f in faces:
                room = S.room_at(rooms, f["probe"])
                key, item, rgba = S.wall_finish(plan, kind, room)
                if f.get("top"):
                    key, item = key + "_top", ("shield_concrete" if kind != "partition" else "blockwork")
                V, F, UV = face_mesh(f, SKIN_OFF, cell=0.33, tile=tuple(B.items[item]["tile_m"]),
                                     wall=not f.get("top"))
                if len(F) == 0:
                    continue
                mn = ctx.user_mesh("dress/skin/%d" % nskin, V, F, UV)
                mat = ctx.material(key, ctx.texture(item), rgba=rgba, specular=0.12, shininess=0.15,
                                   emission=S.LOOK["wall_emission"])
                sk_body.add_geom(name="dress/skin/%d" % nskin, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn,
                                 material=mat, contype=0, conaffinity=0, group=1)
                skin_planes.append(f)
                skin_faces.append((f, room))
                nskin += 1
        elif kind == "floor":
            room = S.room_of_geom(rooms, W["names"][i])
            key, item, rgba, spec_ = S.floor_finish(plan, room)
            f = {"o": np.array([lo[0], lo[1], hi[2]]), "a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0]),
                 "la": hi[0] - lo[0], "lb": hi[1] - lo[1], "n": np.array([0, 0, 1.0])}
            V, F, UV = face_mesh(f, 0.0, cell=0.45, tile=tuple(B.items[item]["tile_m"]), wall=False)
            mn = ctx.user_mesh("dress/skin/%d" % nskin, V, F, UV)
            mat = ctx.material(key, ctx.texture(item), rgba=rgba, specular=spec_[0], shininess=spec_[1])
            sk_body.add_geom(name="dress/skin/%d" % nskin, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn,
                             material=mat, contype=0, conaffinity=0, group=1)
            nskin += 1
        elif kind == "ceiling":
            f = {"o": np.array([lo[0], hi[1], lo[2]]), "a": np.array([1.0, 0, 0]), "b": np.array([0, -1.0, 0]),
                 "la": hi[0] - lo[0], "lb": hi[1] - lo[1], "n": np.array([0, 0, -1.0])}
            item = "ceiling_plaster"
            V, F, UV = face_mesh(f, 0.0, cell=0.5, tile=tuple(B.items[item]["tile_m"]), wall=False)
            mn = ctx.user_mesh("dress/skin/%d" % nskin, V, F, UV)
            # a little emission stands in for the light the lamps throw up and the walls bounce (the classic renderer
            # has no indirect light): the ceiling is not a black slab
            mat = ctx.material("ceiling", ctx.texture(item), rgba=(0.92, 0.92, 0.9, 1), specular=0.0, shininess=0.0,
                               emission=0.24)
            sk_body.add_geom(name="dress/skin/%d" % nskin, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn,
                             material=mat, contype=0, conaffinity=0, group=2)
            nskin += 1
    report["counts"]["skins"] = nskin
    # ---------------------------------------------------------------- decor
    if decor:
        from .decor import place_decor
        dec = place_decor(ctx, site, W, protected, skin_src, skin_faces, rooms, plan, rng, by_name=by_name,
                          hidden=hidden, nohide=S.thermal_geoms(site), keepclear=keepclear)
        report["counts"].update(dec["counts"])
        report["decor"] = dec["items"]
    # ---------------------------------------------------------------- render settings (no physics)
    report["render"] = S.render_settings(spec, plan, site)
    ctx.finish()
    report["counts"]["textures"] = len(ctx.tex)
    report["counts"]["materials"] = len(ctx.mats)
    report["counts"]["meshes"] = len(ctx.meshes)
    report["counts"]["skin_tris"] = int(ctx.tris)
    report["counts"]["decor_tris"] = int(sum(x.get("tris", 0) for x in report.get("decor", [])))
    # ---------------------------------------------------------------- checks
    if check:
        chk = {}
        chk["unchanged"] = check_spec_unchanged(sig0, signature(spec), ctx.nbody)
        chk["skins_behind_protected"] = check_skins_behind_protected(skin_planes, W, protected, SKIN_OFF)
        if decor:
            chk["decor_clearance"] = check_decor_clearance(ctx.added_geoms, W, keepclear, site)
        chk["tag_strings"] = check_tag_strings(B)
        chk["sitegen_keepclear"] = ({"ok": True, "used": True, "geoms": len(W["_extra"]["geoms"]),
                                     "segments": int(len(W["_extra"]["segments"][0])),
                                     "floor_rects": int(len(W["_extra"]["floor_rects"]))} if W.get("_extra")
                                    else {"ok": True, "used": False})
        report["checks"] = chk
    report["seconds"] = round(time.perf_counter() - t0, 3)
    return report
