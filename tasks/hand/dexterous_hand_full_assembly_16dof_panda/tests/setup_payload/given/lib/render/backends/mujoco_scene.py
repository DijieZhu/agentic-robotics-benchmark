"""SceneSpec -> compiled MuJoCo render model (MjSpec edits), resolved-scene digest and a backend-neutral snapshot.

    build_spec(scene)                     -> (mujoco.MjSpec, info)   overrides, lights, background, scale refs applied
    compile_scene(scene)                  -> CompiledScene(model, data, spec, labels, info)   state applied, mj_forward;
                                             SceneSpec.lod = "policy": apply_lod decimates the visible meshes and recompiles
    apply_lod(spec, model, scene)         -> LOD record (level, recipe, digest, face counts) for Provenance.lod
    resolved_digest(model)                -> sha256 of the compiled model's MJB bytes (what MuJoCo actually renders)
    snapshot(compiled)                    -> JSON-able world-space list of every VISIBLE geom (type, size, pose, rgba,
                                             material, texture, mesh) + lights + headlight: the input of a USD export
                                             (isaac_backend.py) or of any other renderer
    resolve_point(scene, "body:NAME" | "site:NAME" | "geom:NAME" | [x, y, z]) -> world xyz (for camera rigs)

MuJoCo is also the ground-truth physics of every task of the tier, so every backend resolves the SceneSpec through
this module (the same MjSpec edits, the same state) and only the image formation differs.
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .. import lod as LOD
from .. import scale_refs as SR
from .. import textures as TX
from ..spec import SceneSpec, StateSpec

HIDDEN_GROUP = 5


@dataclass
class CompiledScene:
    model: Any
    data: Any
    spec: Any
    labels: Dict[int, Dict[str, Any]]
    info: Dict[str, Any] = field(default_factory=dict)


def load_builder(ref: str, base_dir=None):
    target, _, fn = ref.partition(":")
    if not fn:
        raise ValueError("builder must be 'module:function' or 'file.py:function', got %r" % ref)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True           # never leave __pycache__ next to task files
    try:
        if target.endswith(".py"):
            path = target if os.path.isabs(target) else os.path.join(base_dir or os.getcwd(), target)
            path = os.path.abspath(path)
            name = "_arb_render_builder_" + hashlib.sha256(path.encode()).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            d = os.path.dirname(path)
            added = d not in sys.path
            if added:
                sys.path.insert(0, d)
            try:
                spec.loader.exec_module(mod)
            finally:
                if added:
                    sys.path.remove(d)
        else:
            mod = importlib.import_module(target)
    finally:
        sys.dont_write_bytecode = old
    return getattr(mod, fn)


def _subtree_geoms(spec, body_name):
    import mujoco
    b = spec.body(body_name)
    if b is None:
        raise ValueError("no body %r" % body_name)
    return list(b.find_all(mujoco.mjtObj.mjOBJ_GEOM))


def _set_if(obj, attr, value):
    if value is not None:
        setattr(obj, attr, value)


def build_spec(scene: SceneSpec) -> Tuple[Any, Dict[str, Any]]:
    import mujoco
    base = scene.base_dir or os.getcwd()
    if scene.mjcf is not None:
        path = scene.mjcf if os.path.isabs(scene.mjcf) else os.path.join(base, scene.mjcf)
        spec = mujoco.MjSpec.from_file(os.path.abspath(path))
    else:
        spec = load_builder(scene.builder, base)(**scene.builder_args)
        if not isinstance(spec, mujoco.MjSpec):
            raise TypeError("builder %s returned %r, not a mujoco.MjSpec" % (scene.builder, type(spec)))
    if HIDDEN_GROUP in scene.visible_groups:
        raise ValueError("geom group %d is the render layer's hidden group" % HIDDEN_GROUP)
    info: Dict[str, Any] = {"scale_refs": {}, "textures_added": []}

    for t in scene.textures:                                   # added / replaced textures
        arr = TX.make(t, base)
        h, w = arr.shape[:2]
        existing = spec.texture(t.name)
        tex = existing if existing is not None else spec.add_texture(name=t.name, type=mujoco.mjtTexture.mjTEXTURE_2D)
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.file = ""
        tex.width, tex.height, tex.nchannel = w, h, 3
        tex.data = np.ascontiguousarray(arr).tobytes()
        info["textures_added"].append(t.name)

    for mo in scene.materials:
        hit = 0
        for m in spec.materials:
            if fnmatch.fnmatchcase(m.name, mo.match) and not any(fnmatch.fnmatchcase(m.name, e) for e in mo.exclude):
                hit += 1
                _set_if(m, "rgba", mo.rgba)
                for a in ("specular", "shininess", "reflectance", "emission", "roughness", "metallic"):
                    _set_if(m, a, getattr(mo, a))
                _set_if(m, "texrepeat", mo.texrepeat)
                _set_if(m, "texuniform", mo.texuniform)
                if mo.texture is not None:
                    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = mo.texture
        if hit == 0:
            raise ValueError("material override %r matched no material" % mo.match)

    for go in scene.geoms:
        cands = _subtree_geoms(spec, go.body) if go.body else list(spec.geoms)
        sel = [g for g in cands if fnmatch.fnmatchcase(g.name or "", go.match)]
        if not sel:
            raise ValueError("geom override (%r, body %r) matched no geom" % (go.match, go.body))
        for g in sel:
            if go.hide:
                g.group = HIDDEN_GROUP
                continue
            if go.material is not None:
                g.material = go.material
            if go.rgba is not None:
                g.material = "" if go.material is None else g.material   # geom rgba only shows without a material
                g.rgba = go.rgba

    for bname in scene.hide_bodies:
        for g in _subtree_geoms(spec, bname):
            g.group = HIDDEN_GROUP

    if scene.replace_lights:
        for l in spec.lights:
            l.active = False
    for k, ls in enumerate(scene.lights):
        typ = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if ls.directional else mujoco.mjtLightType.mjLIGHT_SPOT
        spec.worldbody.add_light(name=ls.name or "render_light_%d" % k, pos=ls.pos, dir=ls.dir, diffuse=ls.diffuse,
                                 specular=ls.specular, ambient=ls.ambient, castshadow=bool(ls.castshadow and scene.shadows),
                                 attenuation=ls.attenuation, cutoff=ls.cutoff_deg, exponent=ls.exponent, type=typ)
    if not scene.shadows:
        for l in spec.lights:
            l.castshadow = False
    if scene.headlight is not None:
        hl = spec.visual.headlight
        hl.ambient, hl.diffuse, hl.specular = scene.headlight.ambient, scene.headlight.diffuse, scene.headlight.specular
        hl.active = int(scene.headlight.active)
    if scene.background is not None:
        for t in list(spec.textures):
            if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(t)
        spec.add_texture(name="render_skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=scene.background.rgb1, rgb2=scene.background.rgb2, width=128, height=768)

    for ref in scene.scale_refs:
        if spec.body(ref.name) is not None:
            raise ValueError("scale reference name %r collides with a body of the scene" % ref.name)
        info["scale_refs"][ref.name] = SR.add_to_spec(spec, ref)

    q = spec.visual.quality
    q.offsamples = int(scene.offsamples)
    q.shadowsize = int(scene.shadowsize)
    return spec, info


def apply_state(model, data, state: StateSpec):
    import mujoco
    if state.keyframe is not None:
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, state.keyframe)
        if k < 0:
            raise ValueError("no keyframe %r" % state.keyframe)
        mujoco.mj_resetDataKeyframe(model, data, k)
    else:
        mujoco.mj_resetData(model, data)
    if state.qpos is not None:
        if len(state.qpos) != model.nq:
            raise ValueError("qpos has %d values, model nq = %d" % (len(state.qpos), model.nq))
        data.qpos[:] = state.qpos
    for name, val in state.joints.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError("no joint %r" % name)
        if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("joint %r is not a hinge / slide joint" % name)
        data.qpos[model.jnt_qposadr[j]] = float(val)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(state.settle_s / model.opt.timestep))):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def _labels(model, info) -> Dict[int, Dict[str, Any]]:
    import mujoco
    ref_bodies = {v["body"]: k for k, v in info["scale_refs"].items()}
    out = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        out[g] = {"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "", "body": bname,
                  "scale_ref": ref_bodies.get(bname)}
    return out


_LOD_CACHE: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
_LOD_CACHE_MAX = 256


def _decimate_cached(key_bytes: bytes, V, F, cell_m, representative):
    key = (hashlib.sha256(key_bytes).hexdigest(), "%r/%s" % (float(cell_m), representative))
    hit = _LOD_CACHE.get(key)
    if hit is None:
        if len(_LOD_CACHE) >= _LOD_CACHE_MAX:
            _LOD_CACHE.pop(next(iter(_LOD_CACHE)))
        hit = _LOD_CACHE[key] = LOD.vertex_cluster(V, F, cell_m, representative)
    return hit


def apply_lod(spec, model, scene: SceneSpec) -> Dict[str, Any]:
    """Replace every mesh drawn by a visible geom (group in scene.visible_groups) that has >= recipe min_faces faces by
    its vertex-clustered version (lod.py), IN the MjSpec, so the caller recompiles.  Decimation runs in the compiled
    mesh frame; the result goes back to the (scaled) asset frame by v_asset = R(mesh_quat) v + mesh_pos, so every
    geom stays where it was (MuJoCo re-centres the new mesh itself).  Decimated meshes get shell inertia (a render model
    only needs kinematics; clustering may open the surface).  Returns the record for Provenance.lod."""
    import mujoco
    recipe = LOD.resolve_recipe(scene.lod_params)
    visible = set(int(g) for g in scene.visible_groups)
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    geoms_per_mesh: Dict[int, int] = {}
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == mesh_type and int(model.geom_group[g]) in visible and int(model.geom_dataid[g]) >= 0:
            mid = int(model.geom_dataid[g])
            geoms_per_mesh[mid] = geoms_per_mesh.get(mid, 0) + 1
    meshes = list(spec.meshes)
    cand, kept = [], {}
    for mid in sorted(geoms_per_mesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        nf = int(model.mesh_facenum[mid])
        if nf < recipe["min_faces"]:
            kept[name] = "min_faces"
        elif recipe["skip_textured"] and int(model.mesh_texcoordnum[mid]) > 0:
            kept[name] = "textured"
        else:
            cand.append((mid, name))

    def decimate(cell):
        out = []
        for mid, name in cand:
            va, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fa, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            V = model.mesh_vert[va:va + nv].astype(np.float64)
            F = model.mesh_face[fa:fa + nf].astype(np.int64)
            key = np.ascontiguousarray(model.mesh_vert[va:va + nv]).tobytes() + np.ascontiguousarray(model.mesh_face[fa:fa + nf]).tobytes()
            V2, F2 = _decimate_cached(key, V, F, cell, recipe["representative"])
            out.append((mid, name, V2, F2))
        return out

    cell = recipe["cell_m"]
    for it in range(recipe["max_iterations"]):
        res = decimate(cell)
        total = sum(len(F2) if LOD.usable(V2, F2) else int(model.mesh_facenum[mid]) for mid, _, V2, F2 in res)
        if recipe["max_faces"] is None or total <= recipe["max_faces"] or it == recipe["max_iterations"] - 1:
            break
        cell *= recipe["grow"]
    applied = []
    faces_full = faces_lod = geom_faces_full = geom_faces_lod = 0
    for mid in sorted(geoms_per_mesh):
        geom_faces_full += geoms_per_mesh[mid] * int(model.mesh_facenum[mid])
        faces_full += int(model.mesh_facenum[mid])
    decimated_faces = {}
    for mid, name, V2, F2 in res:
        if not LOD.usable(V2, F2):
            kept[name] = "degenerate"
            continue
        m = spec.mesh(name) if name else None                 # by name (robust to attached sub-specs) ...
        if m is None and mid < len(meshes) and (meshes[mid].name or "") == name:
            m = meshes[mid]                                    # ... else by compile order, name-checked
        if m is None:
            raise RuntimeError("LOD: no spec mesh for model mesh %d %r" % (mid, name))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(model.mesh_quat[mid], float))
        Va = V2 @ R.reshape(3, 3).T + np.asarray(model.mesh_pos[mid], float)
        # a fresh mesh of the same name (geoms refer to meshes by name): editing the file mesh in place keeps MuJoCo's
        # loaded per-face normal data of the old size ("face data must have the same size as face normal data")
        keep = {"smoothnormal": m.smoothnormal, "maxhullvert": m.maxhullvert, "material": m.material}
        spec.delete(m)
        nm = spec.add_mesh(name=name, uservert=Va.reshape(-1).tolist(), userface=F2.reshape(-1).tolist())
        nm.scale = [1.0, 1.0, 1.0]                     # Va is already scaled; a <default><mesh scale> must not apply twice
        nm.refpos = [0.0, 0.0, 0.0]
        nm.refquat = [1.0, 0.0, 0.0, 0.0]
        nm.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        for k, v in keep.items():
            setattr(nm, k, v)
        applied.append((name, V2, F2))
        decimated_faces[mid] = len(F2)
    for mid in sorted(geoms_per_mesh):
        nf = decimated_faces.get(mid, int(model.mesh_facenum[mid]))
        faces_lod += nf
        geom_faces_lod += geoms_per_mesh[mid] * nf
    resolved = dict(recipe, cell_m=float(cell))
    return {"level": scene.lod, "lod_version": LOD.LOD_VERSION, "recipe": resolved,
            "digest": LOD.recipe_digest(resolved, applied),
            "visible_mesh_faces_full": faces_full, "visible_mesh_faces_lod": faces_lod,
            "visible_geom_faces_full": geom_faces_full, "visible_geom_faces_lod": geom_faces_lod,
            "meshes_decimated": len(applied), "meshes_kept": {k: kept[k] for k in sorted(kept)}}


def compile_scene(scene: SceneSpec) -> CompiledScene:
    import mujoco
    spec, info = build_spec(scene)
    model = spec.compile()
    if scene.lod != "full":
        info["lod"] = apply_lod(spec, model, scene)
        model = spec.compile()
    ext = float(model.stat.extent)
    model.vis.map.znear = scene.near_m / ext
    model.vis.map.zfar = scene.far_m / ext
    data = mujoco.MjData(model)
    apply_state(model, data, scene.state)
    return CompiledScene(model=model, data=data, spec=spec, labels=_labels(model, info), info=info)


def resolved_digest(model) -> str:
    import mujoco
    buf = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def _mat2quat(xmat):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, float).reshape(9))
    return q


def snapshot(cs: CompiledScene, visible_groups=(0, 1, 2)) -> Dict[str, Any]:
    """World-space description of what is rendered (visible geoms only), rounded to 1e-9 for byte stability."""
    import mujoco
    m, d = cs.model, cs.data
    r9 = lambda a: [round(float(v), 9) for v in np.asarray(a).reshape(-1)]  # noqa: E731
    name = lambda t, i: mujoco.mj_id2name(m, t, i) if i >= 0 else None  # noqa: E731
    geoms = []
    default_rgba = np.array([0.5, 0.5, 0.5, 1.0], np.float32)
    for g in range(m.ngeom):
        if int(m.geom_group[g]) not in visible_groups:
            continue
        mat = int(m.geom_matid[g])
        rgba = m.geom_rgba[g]
        if mat >= 0 and np.allclose(rgba, default_rgba):
            rgba = m.mat_rgba[mat]
        tex = -1
        if mat >= 0:
            tex = int(m.mat_texid[mat, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        geoms.append({"id": g, "name": name(mujoco.mjtObj.mjOBJ_GEOM, g), "body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
                      "type": mujoco.mjtGeom(int(m.geom_type[g])).name, "size": r9(m.geom_size[g]),
                      "pos": r9(d.geom_xpos[g]), "quat_wxyz": r9(_mat2quat(d.geom_xmat[g])), "rgba": r9(rgba),
                      "material": name(mujoco.mjtObj.mjOBJ_MATERIAL, mat),
                      "material_props": None if mat < 0 else {"specular": round(float(m.mat_specular[mat]), 9),
                                                              "shininess": round(float(m.mat_shininess[mat]), 9),
                                                              "reflectance": round(float(m.mat_reflectance[mat]), 9),
                                                              "emission": round(float(m.mat_emission[mat]), 9)},
                      "texture": name(mujoco.mjtObj.mjOBJ_TEXTURE, tex),
                      "mesh": name(mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None})
    lights = []
    for i in range(m.nlight):
        lights.append({"name": name(mujoco.mjtObj.mjOBJ_LIGHT, i), "type": mujoco.mjtLightType(int(m.light_type[i])).name,
                       "active": bool(m.light_active[i]), "pos": r9(d.light_xpos[i]), "dir": r9(d.light_xdir[i]),
                       "diffuse": r9(m.light_diffuse[i]), "specular": r9(m.light_specular[i]), "ambient": r9(m.light_ambient[i]),
                       "castshadow": bool(m.light_castshadow[i]), "cutoff_deg": round(float(m.light_cutoff[i]), 9)})
    hl = m.vis.headlight
    return {"format": "arb-render-snapshot/1", "frame": "world, z up, metres; quat w x y z; mesh vertices: model.mesh_vert "
            "rows of the geom's mesh are in the GEOM frame", "geoms": geoms, "lights": lights,
            "headlight": {"active": bool(hl.active), "ambient": r9(hl.ambient), "diffuse": r9(hl.diffuse), "specular": r9(hl.specular)},
            "scale_refs": {k: v["body"] for k, v in cs.info.get("scale_refs", {}).items()}}


def resolve_point(scene: SceneSpec, what, compiled: CompiledScene = None) -> np.ndarray:
    import mujoco
    if not isinstance(what, str):
        return np.asarray(what, float)
    kind, _, nm = what.partition(":")
    cs = compiled or compile_scene(scene)
    obj = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE, "geom": mujoco.mjtObj.mjOBJ_GEOM}[kind]
    i = mujoco.mj_name2id(cs.model, obj, nm)
    if i < 0:
        raise ValueError("no %s %r" % (kind, nm))
    arr = {"body": cs.data.xpos, "site": cs.data.site_xpos, "geom": cs.data.geom_xpos}[kind]
    return np.array(arr[i], float)
