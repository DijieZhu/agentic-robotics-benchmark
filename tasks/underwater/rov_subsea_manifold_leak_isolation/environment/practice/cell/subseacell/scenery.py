"""Static scenery of a site world: rocks, the production flowline with its tie-in spool and pipe supports, the test
flowline, concrete mattresses (cell side; amendment 02, package P37; decisions AM2-1 to AM2-6 of
shared/DECISIONS_P2_BEH.md section 12).

A site package may carry the block site.json["scenery"] (schema scenery_layout/1, written by the site generator).
Every piece is world geometry, so it is in all three models of subseacell.world at the same pose:
  physics model   a collider (MuJoCo collides a mesh by its convex hull: the rock assets keep the hull within 3 cm of
                  the visible surface, tools/scenery_assets/rock_check.py) and a surface the DVL's rays meet;
  render twin     drawn by the camera (the twin copies the physics MjSpec, meshes / textures / materials included);
  acoustic model  a sonar target (subseacell.twin.AcousticModel copies the primitives and rebuilds every rock from the
                  same OBJ at the same scale and pose, so MuJoCo's mesh re-centring is identical in both models).
The pieces go into a few joint-free STATIC bodies, one per square tile of the sea bed (ZONE_M), each with a compact
bounding box: MuJoCo's body-level broad phase then drops every tile that is not near the vehicle (the world body holds
the sea-bed plane, whose bounding box covers everything, so pieces in the world body would be tested against the
vehicle wherever it is).  Static bodies have no degrees of freedom: qpos / qvel layout and the dynamics are those of
the world without scenery; contacts with the sea bed, the structure and other tiles cannot occur (same weld body).
Contact parameters (friction, condim, solref, solimp, margin, gap) are the sea-bed plane's.  A site package without
the block gives the world it gave before amendment 02, bit for bit.

scenery_layout/1 (the block; JSON types; world frame W, metres, quaternions w x y z):
    schema        "scenery_layout/1"
    rule_version  str: the placement rules that wrote the layout (generator side; carried, not interpreted)
    assets        {"rocks":    {<rock>: {"obj": rel. path, "obj_sha256": hex, "albedo": rel. path, "albedo_sha256": hex}},
                   "textures": {<texture>: {"file": rel. path, "sha256": hex}}}
                  paths relative to the scenery asset folder (asset_root()); the world refuses a file whose sha256
                  differs from the block's
    pieces        [piece], each:
        name      unique geom name, "scn_" + [a-z0-9_], at most 60 characters
        type      rock | flowline | spool | test_flowline | mattress | support
        kind      mesh (rock) | capsule (flowline, spool, test_flowline) | box (mattress, support)
        acls      acoustic class (uwkit.sensors.materials): rock -> rock; mattress, support -> concrete;
                  flowline, spool -> steel; test_flowline -> plastic (AM2-6; the world refuses any other)
        mesh      (rock) the rock asset's name;  scale  (rock) uniform scale, (0, 1.5]
        pos, quat (mesh, box) world position of the geom frame and its orientation
        half      (box) half sizes [m]
        a, b      (capsule) end points of the segment axis;  radius  (capsule) [m]
        tex       (capsule, box) tiling texture name (cube-mapped) ; rocks use their own baked albedo
        shade     albedo factor of the piece, [0.5, 1.5] (default 1.0)
        line      (optional) the line a piece belongs to ("flowline" | "test_flowline"), carried, not interpreted

    lay = parse(block)                                  # SceneryError on anything malformed
    info = add_to_spec(spec, lay, root, contact)        # before the physics compile
    check_compiled(model, info)                         # after it: bodies static, frames identity, geoms present
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Optional

import mujoco

from . import site as SITE

SCHEMA = "scenery_layout/1"
TYPES = {"rock": "mesh", "flowline": "capsule", "spool": "capsule", "test_flowline": "capsule", "mattress": "box",
         "support": "box"}
CLASS_OF_TYPE = {"rock": "rock", "mattress": "concrete", "support": "concrete", "flowline": "steel", "spool": "steel",
                 "test_flowline": "plastic"}
NAME_RX = re.compile(r"scn_[a-z0-9_]{1,56}")
ASSET_ENV = "SUBSEACELL_SCENERY_ASSETS"
ZONE_M = 10.0                       # tile edge of the static bodies [m] (P37 cost study: records/P2_P37.md section 5)
ZONE_PREFIX = "scn_zone_"
N_TEXROLE = 10
CONTACT_KEYS = ("friction", "condim", "solref", "solimp", "solmix", "priority", "margin", "gap", "contype",
                "conaffinity")
MAX_PIECES = 4000
SHADE_RANGE = (0.5, 1.5)
HERE = os.path.dirname(os.path.abspath(__file__))
_DEV = os.path.normpath(os.path.join(HERE, "..", ".."))


class SceneryError(SITE.SitePackageError):
    pass


# ==================================================================================================== assets
def asset_root(explicit: Optional[str] = None) -> str:
    """The scenery asset folder: `explicit`, else $SUBSEACELL_SCENERY_ASSETS, else <vehicle mesh folder>/scenery when it
    exists (the task's tests/meshes/scenery, the practice service's meshes/scenery: the vehicle's collision set lives
    next to it, subseacell.vehicle.COLLISION_MESH_DIR), else the development tree's assets/scenery."""
    if explicit:
        return explicit
    env = os.environ.get(ASSET_ENV)
    if env:
        return env
    from . import vehicle as VH
    p = os.path.join(VH.COLLISION_MESH_DIR, "scenery")
    if os.path.isdir(p):
        return p
    return os.path.join(_DEV, "assets", "scenery")


_SHA_CACHE = {}


def _sha256_file(p: str) -> str:
    st = os.stat(p)
    key = (os.path.abspath(p), st.st_size, st.st_mtime_ns)
    h = _SHA_CACHE.get(key)
    if h is None:
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                d.update(b)
        h = _SHA_CACHE[key] = d.hexdigest()
    return h


def _asset_file(root, rel, sha, what):
    if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/") or not rel:
        raise SceneryError(f"scenery: bad asset path {rel!r} ({what})")
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        raise SceneryError(f"scenery: asset {rel!r} ({what}) not found under {root}")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise SceneryError(f"scenery: asset {rel!r} ({what}) has no sha256 in the layout")
    have = _sha256_file(p)
    if have != sha:
        raise SceneryError(f"scenery: asset {rel!r} ({what}) is {have[:16]}..., the layout wants {sha[:16]}...")
    return os.path.abspath(p)


# ==================================================================================================== parse
def _vec(x, n, what):
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise SceneryError(f"scenery: {what} must be {n} numbers")
    out = []
    for v in x:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise SceneryError(f"scenery: {what} must be finite numbers")
        out.append(float(v))
    return out


def _pos(x, what, lo=0.0):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) <= lo:
        raise SceneryError(f"scenery: {what} must be a number > {lo}")
    return float(x)


def parse(block: dict) -> dict:
    """Validate a scenery_layout/1 block; returns a normalised copy {schema, rule_version, assets, pieces}."""
    if not isinstance(block, dict) or block.get("schema") != SCHEMA:
        raise SceneryError(f"scenery: schema {block.get('schema') if isinstance(block, dict) else block!r}, want {SCHEMA!r}")
    pieces = block.get("pieces")
    if not isinstance(pieces, list) or len(pieces) > MAX_PIECES:
        raise SceneryError("scenery: pieces must be a list (at most %d)" % MAX_PIECES)
    assets = block.get("assets") or {}
    rocks = dict(assets.get("rocks") or {})
    textures = dict(assets.get("textures") or {})
    names, out = set(), []
    for k, pc in enumerate(pieces):
        if not isinstance(pc, dict):
            raise SceneryError(f"scenery: piece {k} is not an object")
        n = pc.get("name")
        if not isinstance(n, str) or not NAME_RX.fullmatch(n) or n.startswith(ZONE_PREFIX):
            raise SceneryError(f"scenery: piece {k}: bad name {n!r}")
        if n in names:
            raise SceneryError(f"scenery: duplicate piece name {n!r}")
        names.add(n)
        t = pc.get("type")
        if t not in TYPES:
            raise SceneryError(f"scenery: {n}: unknown type {t!r}")
        kind = pc.get("kind")
        if kind != TYPES[t]:
            raise SceneryError(f"scenery: {n}: type {t} must be kind {TYPES[t]}, not {kind!r}")
        if pc.get("acls") != CLASS_OF_TYPE[t]:
            raise SceneryError(f"scenery: {n}: type {t} has the acoustic class {CLASS_OF_TYPE[t]}, not {pc.get('acls')!r}")
        shade = float(pc.get("shade", 1.0))
        if not (SHADE_RANGE[0] <= shade <= SHADE_RANGE[1]):
            raise SceneryError(f"scenery: {n}: shade {shade} outside {SHADE_RANGE}")
        q = dict(name=n, type=t, kind=kind, acls=pc["acls"], shade=shade)
        if pc.get("line") is not None:
            q["line"] = str(pc["line"])
        if kind == "mesh":
            r = pc.get("mesh")
            if r not in rocks:
                raise SceneryError(f"scenery: {n}: rock asset {r!r} not in the layout's assets")
            s = _pos(pc.get("scale"), f"{n}.scale")
            if s > 1.5:
                raise SceneryError(f"scenery: {n}: scale {s} > 1.5")
            q.update(mesh=r, scale=s, pos=_vec(pc.get("pos"), 3, f"{n}.pos"), quat=_vec(pc.get("quat"), 4, f"{n}.quat"))
        elif kind == "box":
            q.update(pos=_vec(pc.get("pos"), 3, f"{n}.pos"), quat=_vec(pc.get("quat"), 4, f"{n}.quat"),
                     half=[_pos(v, f"{n}.half") for v in _vec(pc.get("half"), 3, f"{n}.half")])
        else:
            a, b = _vec(pc.get("a"), 3, f"{n}.a"), _vec(pc.get("b"), 3, f"{n}.b")
            if math.dist(a, b) < 1e-6:
                raise SceneryError(f"scenery: {n}: zero-length capsule")
            q.update(a=a, b=b, radius=_pos(pc.get("radius"), f"{n}.radius"))
        if kind != "mesh":
            tx = pc.get("tex")
            if tx not in textures:
                raise SceneryError(f"scenery: {n}: texture {tx!r} not in the layout's assets")
            q["tex"] = tx
        if kind in ("mesh", "box"):
            nq = math.sqrt(sum(v * v for v in q["quat"]))
            if not (abs(nq - 1.0) < 1e-6):
                raise SceneryError(f"scenery: {n}: quaternion not unit ({nq})")
        out.append(q)
    for r, e in rocks.items():
        if not isinstance(e, dict) or not {"obj", "obj_sha256", "albedo", "albedo_sha256"} <= set(e):
            raise SceneryError(f"scenery: rock asset {r!r} needs obj, obj_sha256, albedo, albedo_sha256")
    for t, e in textures.items():
        if not isinstance(e, dict) or not {"file", "sha256"} <= set(e):
            raise SceneryError(f"scenery: texture asset {t!r} needs file, sha256")
    return dict(schema=SCHEMA, rule_version=block.get("rule_version"), assets=dict(rocks=rocks, textures=textures),
                pieces=out)


def piece_centre(pc) -> tuple:
    if pc["kind"] == "capsule":
        return tuple(0.5 * (a + b) for a, b in zip(pc["a"], pc["b"]))
    return tuple(pc["pos"])


def zones_of(lay: dict, zone_m: Optional[float] = None, origin_xy=(0.0, 0.0)) -> list:
    """Pieces per square tile of the sea bed (by the piece's centre); [(tile key (i, j), [piece indices])] in tile
    order.  Deterministic."""
    zone_m = float(ZONE_M if zone_m is None else zone_m)
    tiles = {}
    for k, pc in enumerate(lay["pieces"]):
        c = piece_centre(pc)
        key = (int(math.floor((c[0] - origin_xy[0]) / zone_m)), int(math.floor((c[1] - origin_xy[1]) / zone_m)))
        tiles.setdefault(key, []).append(k)
    return sorted(tiles.items())


# ==================================================================================================== physics spec
def contact_of(geom_spec) -> dict:
    """The contact parameters of a spec geom (the sea-bed plane), as add_geom keywords."""
    out = {}
    for k in CONTACT_KEYS:
        v = getattr(geom_spec, k)
        out[k] = [float(x) for x in v] if hasattr(v, "__len__") else (int(v) if k in ("condim", "priority", "contype",
                                                                                       "conaffinity") else float(v))
    return out


def add_to_spec(spec: mujoco.MjSpec, lay: dict, root: str, contact: dict, *, bodies: str = "zones",
                zone_m: Optional[float] = None, origin_xy=(0.0, 0.0), group: int = 0) -> dict:
    """Add the layout's pieces to the physics spec (before compile).  bodies = "zones" (static bodies, one per tile:
    the task world) or "world" (every piece in the world body: the L09 prototype's way, kept for the cost comparison
    only).  Returns the bookkeeping the world keeps (pieces by name, zone bodies, rock meshes for the acoustic model,
    acoustic classes)."""
    if bodies not in ("zones", "world"):
        raise ValueError(bodies)
    zone_m = float(ZONE_M if zone_m is None else zone_m)
    A = lay["assets"]
    pieces = lay["pieces"]
    used_rocks = sorted({pc["mesh"] for pc in pieces if pc["kind"] == "mesh"})
    used_tex = sorted({pc["tex"] for pc in pieces if pc["kind"] != "mesh"})
    mats = {}
    files = {}
    for t in used_tex:
        e = A["textures"][t]
        p = _asset_file(root, e["file"], e["sha256"], f"texture {t}")
        files[f"textures/{t}"] = p
        spec.add_texture(name=f"scn_tex_{t}", type=mujoco.mjtTexture.mjTEXTURE_CUBE, file=p)
        tex = [""] * N_TEXROLE
        tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = f"scn_tex_{t}"
        spec.add_material(name=f"scn_mat_{t}", textures=tex, rgba=[1.0, 1.0, 1.0, 1.0], specular=0.0, shininess=0.0)
        mats[t] = f"scn_mat_{t}"
    rock_obj = {}
    for r in used_rocks:
        e = A["rocks"][r]
        obj = _asset_file(root, e["obj"], e["obj_sha256"], f"rock {r} mesh")
        png = _asset_file(root, e["albedo"], e["albedo_sha256"], f"rock {r} albedo")
        rock_obj[r] = obj
        files[f"rocks/{r}"] = obj
        spec.add_texture(name=f"scn_tex_{r}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png)
        tex = [""] * N_TEXROLE
        tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = f"scn_tex_{r}"
        spec.add_material(name=f"scn_mat_{r}", textures=tex, rgba=[1.0, 1.0, 1.0, 1.0], specular=0.0, shininess=0.0)
        mats[r] = f"scn_mat_{r}"
    mesh_assets = {}
    for pc in pieces:
        if pc["kind"] == "mesh":
            key = (pc["mesh"], pc["scale"])
            if key not in mesh_assets:
                mn = f"scn_mesh_{pc['mesh']}_{len(mesh_assets):03d}"
                me = spec.add_mesh(name=mn, file=rock_obj[pc["mesh"]])
                me.scale = [pc["scale"]] * 3
                mesh_assets[key] = mn
    # bodies
    if bodies == "zones":
        tiles = zones_of(lay, zone_m, origin_xy)
        holder = {}
        zone_names = []
        for zi, (key, idx) in enumerate(tiles):
            zn = f"{ZONE_PREFIX}{zi:03d}"
            b = spec.worldbody.add_body(name=zn, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
            zone_names.append(zn)
            for k in idx:
                holder[k] = (b, zn)
    else:
        tiles, zone_names = [], []
        holder = {k: (spec.worldbody, "world") for k in range(len(pieces))}
    c = dict(contact)
    by_name, meshes, classes, zone_of = {}, [], {}, {}
    for k, pc in enumerate(pieces):
        b, zn = holder[k]
        n = pc["name"]
        sh = pc["shade"]
        kw = dict(name=n, rgba=[sh, sh, sh, 1.0], group=int(group), mass=0.0, **c)
        if pc["kind"] == "mesh":
            mn = mesh_assets[(pc["mesh"], pc["scale"])]
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, pos=list(pc["pos"]), quat=list(pc["quat"]),
                       material=mats[pc["mesh"]], **kw)
            meshes.append(dict(name=n, mesh=mn, file=rock_obj[pc["mesh"]], scale=pc["scale"], pos=list(pc["pos"]),
                               quat=list(pc["quat"])))
        elif pc["kind"] == "box":
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pc["half"]), pos=list(pc["pos"]), quat=list(pc["quat"]),
                       material=mats[pc["tex"]], **kw)
        else:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[pc["radius"], 0.0, 0.0],
                       fromto=list(pc["a"]) + list(pc["b"]), material=mats[pc["tex"]], **kw)
        by_name[n] = pc
        classes[n] = pc["acls"]
        zone_of[n] = zn
    return dict(bodies=bodies, zone_m=float(zone_m) if bodies == "zones" else None, zone_bodies=zone_names,
                pieces=by_name, meshes=meshes, classes=classes, zone_of=zone_of, files=files,
                rule_version=lay.get("rule_version"), n_tiles=len(tiles))


def check_compiled(m: mujoco.MjModel, info: dict) -> dict:
    """After the physics compile: every zone body static (welded to the world, child of the world, no joint), its frame
    the identity (geom poses in the body frame are world poses: the acoustic model copies them), every piece present
    and colliding.  Returns per-zone figures."""
    out = []
    for zn in info["zone_bodies"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, zn)
        if bid < 0:
            raise SceneryError(f"scenery: zone body {zn} missing after compile (fused?)")
        if int(m.body_weldid[bid]) != 0 or int(m.body_parentid[bid]) != 0 or int(m.body_jntnum[bid]) != 0:
            raise SceneryError(f"scenery: zone body {zn} is not a static child of the world body")
        if any(float(x) != 0.0 for x in m.body_pos[bid]) or [float(x) for x in m.body_quat[bid]] != [1.0, 0.0, 0.0, 0.0]:
            raise SceneryError(f"scenery: zone body {zn} frame is not the identity")
        out.append(dict(body=zn, ngeom=int(m.body_geomnum[bid])))
    for n in info["pieces"]:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        if g < 0:
            raise SceneryError(f"scenery: piece {n} missing after compile")
        if int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
            raise SceneryError(f"scenery: piece {n} does not collide")
    return dict(zones=out)


def acoustic_entries(m: mujoco.MjModel, info: dict) -> tuple:
    """(primitive piece names for AcousticModel.static_geoms, rock entries for AcousticModel.static_meshes)."""
    prim = [n for n, pc in info["pieces"].items() if pc["kind"] != "mesh"]
    return prim, [dict(e) for e in info["meshes"]]
