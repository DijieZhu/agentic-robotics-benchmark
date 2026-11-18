"""STEP solids, tessellation, mass properties and convex decomposition for the adapters.

Tessellation is 0.05 mm / 0.3 rad exactly as the CAD grader (`arm_design_check.py:31`
`TESS_TOL, TESS_ANG`), so the visual meshes A2 / A3 emit are the meshes the CAD grader scored.
Heavy imports (cadquery / OCP, trimesh, coacd) are deferred so JSON-only callers pay nothing.
"""

import hashlib
import math
import os

import numpy as np

__all__ = [
    "TESS_TOL", "TESS_ANG", "PLA_DENSITY_KG_M3", "sha256_file", "sha256_bytes", "load_step_solid",
    "tessellate", "mesh_mass_properties", "convex_pieces", "export_stl", "servo_collision_primitives",
    "box_inertia", "shift_inertia", "transform_mesh",
]

TESS_TOL, TESS_ANG = 0.05, 0.3
PLA_DENSITY_KG_M3 = 1240.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_step_solid(path):
    """The single solid of a STEP file (raises ValueError when the file is not one solid)."""
    import cadquery as cq
    wp = cq.importers.importStep(path)
    solids = wp.solids().vals()
    if len(solids) != 1:
        raise ValueError("{}: expected exactly one solid, found {}".format(path, len(solids)))
    return solids[0]


def tessellate(solid, tol=TESS_TOL, ang=TESS_ANG):
    """A processed, watertight trimesh of a cadquery Solid (mm, part frame)."""
    import trimesh
    vs, tris = solid.tessellate(tol, ang)
    verts = np.array([v.toTuple() for v in vs], dtype=float)
    faces = np.array(tris, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("tessellation produced no faces")
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if not mesh.is_watertight:
        raise ValueError("tessellated mesh is not watertight")
    if mesh.volume <= 0:
        raise ValueError("tessellated mesh has non-positive volume")
    return mesh


def transform_mesh(mesh, T):
    out = mesh.copy()
    out.apply_transform(np.asarray(T, float))
    return out


def mesh_mass_properties(mesh, density_kg_m3=PLA_DENSITY_KG_M3, mass_override_kg=None):
    """(mass_kg, com_mm, inertia_kg_mm2 about the CoM, part axes) of a mm mesh.

    Same numbers as `arm_design_check.mass_model` / `_inertia_of_link` for the printed
    solid: mass = volume x density (or the larger `mass_override_kg`), inertia scaled with it.
    """
    volume_mm3 = float(mesh.volume)
    m = volume_mm3 * 1e-9 * density_kg_m3
    inertia = np.asarray(mesh.moment_inertia, float) * 1e-9 * density_kg_m3   # mm^5 * kg/m^3 -> kg mm^2
    if mass_override_kg is not None and float(mass_override_kg) > m and m > 0:
        inertia = inertia * (float(mass_override_kg) / m)
        m = float(mass_override_kg)
    return m, np.asarray(mesh.center_mass, float), inertia


def box_inertia(mass, size):
    s = np.asarray(size, float)
    return mass / 12.0 * np.diag([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])


def shift_inertia(I_com, mass, d):
    """Parallel-axis shift of an inertia about the CoM by the vector d."""
    d = np.asarray(d, float)
    return I_com + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def convex_pieces(mesh, max_pieces=16, threshold=0.05, seed=0):
    """CoACD convex decomposition (<= `max_pieces` hulls) of a mm mesh; deterministic per seed.

    Falls back to the single convex hull when CoACD is unavailable or returns nothing.
    """
    import trimesh
    try:
        import coacd
    except ImportError:  # pragma: no cover - the image ships coacd
        return [mesh.convex_hull]
    try:
        coacd.set_log_level("error")
    except Exception:  # noqa: BLE001
        pass
    cm = coacd.Mesh(np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                    np.ascontiguousarray(mesh.faces, dtype=np.int32))
    parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=int(max_pieces), seed=int(seed))
    out = []
    for verts, faces in parts:
        piece = trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces, np.int64), process=True)
        if len(piece.faces) >= 4 and piece.volume > 0:
            out.append(piece.convex_hull)
    if not out:
        out = [mesh.convex_hull]
    if len(out) > max_pieces:
        # CoACD honours max_convex_hull, but never trust a bound you can check.
        out = sorted(out, key=lambda p: -p.volume)[:max_pieces]
    return out


def export_stl(mesh, path, scale=1.0):
    """Binary STL with a fixed header (byte-deterministic for identical geometry)."""
    import trimesh
    m = mesh.copy()
    if scale != 1.0:
        m.apply_scale(scale)
    data = trimesh.exchange.stl.export_stl(m)
    data = b"arb-chain".ljust(80, b"\0") + data[80:]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def servo_collision_primitives(servo_meta, shrink=(1.0, 0.3, 0.5)):
    """The assembly cell's collision model of the stand-in, servo frame, mm (CHAIN_PLAN 4.3).

    Body / ears / cable block as boxes with the published shrink per class; horn boss and rear
    boss are visual-only locators and are NOT part of the collision model.
    """
    body_s, ear_s, cable_s = shrink
    prims = [{"name": "body", "kind": "box", "min": list(servo_meta["body"]["min"]),
              "max": list(servo_meta["body"]["max"]), "shrink_mm": body_s}]
    for ear in servo_meta["ears"]:
        prims.append({"name": ear["id"], "kind": "box", "min": list(ear["box_min"]),
                      "max": list(ear["box_max"]), "shrink_mm": ear_s})
    cb = servo_meta["cable_exit_block"]
    prims.append({"name": "cable_exit_block", "kind": "box", "min": list(cb["box_min"]),
                  "max": list(cb["box_max"]), "shrink_mm": cable_s})
    return prims
