"""make_parcel: one parcel's visual mesh, maps and record from its specification and a seed."""
from __future__ import annotations

import json
import math
import os
import zlib

import numpy as np

from . import export, geometry, maps, waybill
from .geometry import Carton, Mailer, face_axes, place_waybill, waybill_points
from .identity import WaybillIdentity, identity_for_seed

QUALITIES = {
    "full": {"tex": 2048, "mesh": "full", "use": "path-traced look"},
    "bg": {"tex": 1024, "mesh": "agent", "use": "background dressing"},
    "agent": {"tex": 512, "mesh": "agent", "use": "Filament agent view (MuJoCo)"},
}
LABEL_MAX_OUT = 0.0009          # m: the waybill sheet's largest vertex distance outside the collision shape
KIND_ALIASES = {"rigid": "carton", "carton": "carton", "soft": "mailer", "mailer": "mailer"}


def _spec_dict(spec) -> dict:
    if isinstance(spec, dict):
        d = dict(spec)
    else:                                            # a parcelcell.stream.Parcel or any object with these fields
        d = {k: getattr(spec, k) for k in ("kind", "size", "label_face", "label_uv", "mass") if hasattr(spec, k)}
    d["kind"] = KIND_ALIASES[str(d.get("kind", "rigid"))]
    d["size"] = [float(v) for v in d["size"]]
    if d.get("label_uv") is not None:
        d["label_uv"] = [float(v) for v in d["label_uv"]]
    return d


def _grid_mesh(shape, isl, f, quality):
    a, b = shape.face_grid(f, quality)
    A, B = np.meshgrid(a, b)                         # rows = b, cols = a
    S, Nrm = shape.project(f, A, B)
    UV = isl.uv(A, B)
    nb, na = A.shape
    idx = np.arange(na * nb).reshape(nb, na)
    p00, p10, p11, p01 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([p00, p10, p11], -1).reshape(-1, 3), np.stack([p00, p11, p01], -1).reshape(-1, 3)])
    return S.reshape(-1, 3), Nrm.reshape(-1, 3), UV.reshape(-1, 2), F


def _label_mesh(shape, pl, wb_isl, quality, rng, corner_lift):
    W, H = pl["size_m"]
    nu, nv = (14, 24) if quality == "full" else (8, 14)
    xl = np.linspace(-W / 2, W / 2, nu + 1)
    yl = np.linspace(-H / 2, H / 2, nv + 1)
    X, Y = np.meshgrid(xl, yl)
    a, b = waybill_points(pl, X, Y)
    lift = geometry.LABEL_LIFT + np.zeros_like(X)
    if corner_lift is not None:
        k, amt = corner_lift                         # corner index 0..3, lift (m)
        cx, cy = (W / 2 if k in (1, 2) else -W / 2), (H / 2 if k in (2, 3) else -H / 2)
        d = np.hypot(X - cx, Y - cy)
        lift = lift + amt * np.clip(1 - d / 0.03, 0, 1) ** 2
    S, Nrm = shape.project(pl["face"], a, b, lift=0.0)
    # the sheet is stiff: it rides on the highest film / board point within one grid step (no film poking through)
    n = np.asarray(pl["normal"], float)
    rr = 1.2 * max(W / nu, H / nv)
    h0 = S @ n
    hmax = h0.copy()
    for k in range(12):
        ang = 2 * np.pi * k / 12
        for frac in (0.5, 1.0):
            Sk, _ = shape.project(pl["face"], a + frac * rr * np.cos(ang), b + frac * rr * np.sin(ang), lift=0.0)
            hmax = np.maximum(hmax, Sk @ n)
    S = S + n * (hmax - h0)[..., None]
    S = S + Nrm * lift[..., None]
    # never more than LABEL_MAX_OUT outside the collision shape (the 2 mm rule with a margin)
    from .checks import outside_distance
    for _ in range(4):                 # push back along the surface normal (works where the sheet wraps a flank)
        out = outside_distance(shape, S.reshape(-1, 3)).reshape(S.shape[:-1])
        S = S - Nrm * np.maximum(out - LABEL_MAX_OUT, 0)[..., None]
    UV = maps.waybill_uv(wb_isl, pl, X, Y)
    nb, na = X.shape
    idx = np.arange(na * nb).reshape(nb, na)
    p00, p10, p11, p01 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([p00, p10, p11], -1).reshape(-1, 3), np.stack([p00, p11, p01], -1).reshape(-1, 3)])
    # the label's (right, up) is right-handed about the face normal, so this winding faces outward
    return S.reshape(-1, 3), Nrm.reshape(-1, 3), UV.reshape(-1, 2), F, S


def build(spec, seed, identity=None, paper=None, look=None):
    """Everything quality-independent: shape, placement, atlas, full-size maps, identity. Returns a dict."""
    d = _spec_dict(spec)
    kind = d["kind"]
    seed = int(seed)
    rng = np.random.default_rng([seed & 0xFFFFFFFF, zlib.crc32(kind.encode())])
    look = dict(look or {})
    if kind == "carton":
        shape = Carton(d["size"], radius=float(rng.uniform(0.003, 0.0045)), sag=float(rng.uniform(0.0, 0.0018)),
                       sag_bottom=float(rng.uniform(0.0, 0.0008)))
        params = maps.carton_params(rng, look.get("board"), look.get("tape"), look.get("tape_pattern"))
    else:
        shape = Mailer(d["size"], rng)
        params = maps.mailer_params(rng, look.get("film"))
        shape.flap = (int(params["flap_face"]), int(params["flap_end"]), float(params["flap_len"]))   #
    face = d.get("label_face")
    pl = None
    if face is not None and int(face) >= 0:
        pl = place_waybill(shape, int(face), rng, d.get("label_uv"), paper=paper)
    if identity is None:
        identity = identity_for_seed(seed, d.get("mass") if look.get("weight_from_mass") else None)
    elif isinstance(identity, dict):
        identity = WaybillIdentity.from_dict(identity)
    rects = [("face%d" % f, 2 * ha, 2 * hb, 1.0) for f, ha, hb in shape.island_faces()]
    if pl:
        rects.append(("waybill", pl["size_m"][0], pl["size_m"][1], maps.WAYBILL_DENSITY))
    packed, dens = maps.pack(rects)
    isls = [maps.Island("face%d" % f, packed["face%d" % f], f, ha, hb) for f, ha, hb in shape.island_faces()]
    wb_isl, wb_render = None, None
    if pl:
        wb_isl = maps.Island("waybill", packed["waybill"], None, pl["size_m"][0] / 2, pl["size_m"][1] / 2)
        x0, y0, w, h = wb_isl.rect
        wb_render = waybill.render(identity, pl["paper"], None, size_px=(w, h))
    marks_out = []
    fn = maps.carton_maps if kind == "carton" else maps.mailer_maps
    buf = fn(shape, isls, wb_isl, wb_render, pl, params, rng, marks_out)
    corner_lift = None
    if pl and rng.uniform() < 0.35:
        corner_lift = (int(rng.integers(4)), float(rng.uniform(0.0003, 0.0009)))
    return {"spec": d, "seed": seed, "kind": kind, "shape": shape, "params": params, "placement": pl,
            "identity": identity, "islands": isls, "wb_isl": wb_isl, "wb_render": wb_render, "buffers": buf,
            "density_px_per_m": dens, "marks": marks_out, "corner_lift": corner_lift, "rng": rng}


def mesh(b, quality):
    shape = b["shape"]
    parts = []
    for isl in b["islands"]:
        parts.append(_grid_mesh(shape, isl, isl.face, quality))
    label_pts = None
    if b["placement"]:
        V, N, UV, F, S = _label_mesh(shape, b["placement"], b["wb_isl"], quality, b["rng"], b["corner_lift"])
        parts.append((V, N, UV, F))
        label_pts = S
    off = np.cumsum([0] + [len(p[0]) for p in parts])[:-1]
    V = np.concatenate([p[0] for p in parts])
    N = np.concatenate([p[1] for p in parts])
    UV = np.concatenate([p[2] for p in parts])
    F = np.concatenate([p[3] + o for p, o in zip(parts, off)])
    return V, N, UV, F, label_pts


def placement_record(b, label_pts=None):
    pl = b["placement"]
    if not pl:
        return None
    shape = b["shape"]
    W, H = pl["size_m"]
    corners_l = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]])
    a, b_ = waybill_points(pl, corners_l[:, 0], corners_l[:, 1])
    Sc, _ = shape.project(pl["face"], a, b_, lift=geometry.LABEL_LIFT)
    ca, cb = pl["offset_ab_m"]
    S0, N0 = shape.project(pl["face"], np.array(ca), np.array(cb), lift=geometry.LABEL_LIFT)
    eu, ev, n = face_axes(pl["face"])
    return {
        "face": pl["face"], "face_name": pl["face_name"], "paper": pl["paper"], "size_mm": [W * 1000, H * 1000],
        "direction": pl["direction"],
        "direction_meaning": "text-up vector = the face's v axis turned by direction x 90 deg about the normal "
                             "(v axis: +z on side faces, +y on +z, -y on -z)",
        "skew_deg": round(pl["skew_deg"], 3), "fits": pl["fits"], "fit_note": pl.get("note"),
        "margin_mm": round(pl["margin_m"] * 1000, 2),
        "centre": [float(v) for v in np.asarray(S0).reshape(3)],
        "centre_on_face_plane": [float(v) for v in pl["centre_plane"]],
        "normal": [float(v) for v in n], "surface_normal_at_centre": [float(v) for v in np.asarray(N0).reshape(3)],
        "right": pl["right"], "up": pl["up"], "corners": np.asarray(Sc).round(6).tolist(),
        "frame_quat_wxyz": geometry.label_frame_quat(pl),
        "site": {"pos": [float(v) for v in pl["centre_plane"]], "zaxis": [float(v) for v in n],
                 "quat_wxyz": geometry.label_frame_quat(pl),
                 "note": "pos on the face plane at the nominal half size; z axis = face normal, x = reading "
                         "direction, y = text up"},
    }


def make_parcel(spec, seed, quality=("full", "agent"), out_dir=None, name=None, identity=None, paper=None,
                look=None, check=True):
    """Build one parcel.

    spec: parcelcell Parcel or dict with kind ('rigid' / 'soft', or 'carton' / 'mailer'), size (m, full sizes),
        label_face (0..5, or None / -1 for an unlabelled parcel), optional label_uv (the simulator's placement
        fractions), optional mass.
    seed: int; the same spec + seed gives the same parcel.
    quality: 'full' (2048 px maps, fine mesh), 'agent' (512 px, coarse mesh), 'bg' (1024 px, coarse mesh), or a
        tuple of them (content is made once and downsampled, so every quality shows the same thing).
    out_dir: where the files go (None: nothing written; the arrays are in the returned record under '_arrays').
    identity: a WaybillIdentity (see identity.identities for a shift's unique set), else one from the seed.
    look: optional overrides: board, tape, tape_pattern (carton); film (mailer); weight_from_mass.
    Returns the record (dict, JSON-serialisable apart from '_arrays')."""
    qs = (quality,) if isinstance(quality, str) else tuple(quality)
    b = build(spec, seed, identity, paper, look)
    name = name or "parcel_%s_%d" % (b["kind"], int(seed))
    rec = {
        "format": "parcelkit.parcel/1", "name": name, "kind": b["kind"], "seed": b["seed"],
        "spec": b["spec"], "collision": b["shape"].collision(),
        "look": {k: v for k, v in b["params"].items() if not k.startswith("_")},
        "marks": b["marks"],
        "waybill": placement_record(b),
        "identity": b["identity"].to_dict() if b["placement"] else None,
        "texel_density_px_per_mm_at_2048": {"surface": round(b["density_px_per_m"] / 1000, 3),
                                            "waybill": round(b["density_px_per_m"] * maps.WAYBILL_DENSITY / 1000, 3)
                                            if b["placement"] else None},
        "atlas": {isl.name: list(isl.rect) for isl in b["islands"] + ([b["wb_isl"]] if b["wb_isl"] else [])},
        "outputs": {},
    }
    arrays = {}
    for q in qs:
        cfg = QUALITIES[q]
        V, N, UV, F, label_pts = mesh(b, cfg["mesh"])
        tex = maps.finish(b["buffers"], cfg["tex"])
        arrays[q] = {"V": V, "N": N, "UV": UV, "F": F, "tex": tex}
        out = {"tex_px": cfg["tex"], "vertices": int(len(V)), "triangles": int(len(F))}
        if check:
            from .checks import mesh_vs_collision
            out["mesh_vs_collision"] = mesh_vs_collision(b["shape"], V, F, label_vertices=label_pts)
        if out_dir:
            tdir = os.path.join(out_dir, "tex")
            os.makedirs(tdir, exist_ok=True)
            files = {}
            for key in [k for k in ("basecolor", "normal", "orm", "cc") if k in tex]:
                fn = "%s_%s_%s.png" % (name, q, key)
                export.write_png(os.path.join(tdir, fn), tex[key])
                files[key] = "tex/" + fn
            obj = os.path.join(out_dir, "%s_%s.obj" % (name, q))
            export.write_obj(obj, V, N, UV, F, "%s_%s.mtl" % (name, q), "parcel_look", files)
            usda = os.path.join(out_dir, "%s_%s.usda" % (name, q))
            custom = {"parcelkit_kind": b["kind"], "parcelkit_seed": int(b["seed"]), "parcelkit_quality": q}
            if b["placement"]:
                wr = rec["waybill"]
                custom.update({"waybill_face": int(wr["face"]), "waybill_paper": wr["paper"],
                               "waybill_centre": wr["centre"], "waybill_frame_quat_wxyz": wr["frame_quat_wxyz"],
                               "waybill_tracking": b["identity"].tracking})
            export.write_usda(usda, V, N, UV, F, {k: "./" + v for k, v in files.items()}, prim="Parcel",
                              doc="parcelkit %s %s seed %d (%s quality)" % (b["kind"], name, b["seed"], q),
                              custom=custom)
            out["files"] = {"obj": os.path.basename(obj), "mtl": "%s_%s.mtl" % (name, q),
                            "usda": os.path.basename(usda), **files}
        rec["outputs"][q] = out
    if out_dir:
        with open(os.path.join(out_dir, "%s.json" % name), "w") as f:
            json.dump(rec, f, indent=1)
    rec["_arrays"] = arrays
    rec["_build"] = b
    return rec


def make_shift_parcels(parcels, shift_seed, quality=("full", "agent"), out_dir=None, prefix="p", look=None):
    """All parcels of one shift: identities differ pairwise (tracking number, routing code, address lines).
    ``parcels``: list of specs (e.g. [a.parcel for a in shift.arrivals]). Returns the list of records."""
    from .identity import identities
    ids = identities(len(parcels), shift_seed)
    recs = []
    for i, (p, ident) in enumerate(zip(parcels, ids)):
        seed = (int(shift_seed) * 1000003 + i) & 0x7FFFFFFF
        r = make_parcel(p, seed, quality, out_dir, name="%s%03d" % (prefix, i), identity=ident, look=look)
        recs.append(r)
    return recs


def strip(rec):
    """The JSON part of a record (drops the arrays and the build state)."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


__all__ = ["make_parcel", "make_shift_parcels", "QUALITIES", "strip", "build", "mesh"]
del math
