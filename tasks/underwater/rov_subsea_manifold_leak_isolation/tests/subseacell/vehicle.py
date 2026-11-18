"""The physics vehicle of task 07 (cell side; DESIGN v1.2 sections 2, 2.3.1 and 7 (1); decisions K2, L4, M1; package P05).

The robot is the nominal `mission` configuration of the frozen reference digest (v3) in its sea-water trim: six ballast
units, battery centred (`mission.state_sea_SP35_trim`).  It is not a hidden variant; what is hidden about it is the
water-interaction coefficient draw (`hydro=`, one draw per task world, decision U1).  This module is the product
version of the gate-G0 prototype (`proto/g0_vehicle/digest_vehicle.py` + `pin_meshes.py` + `mass_props.py`).

Fixed choices (G0; DESIGN section 7 (1)):
  config "mission", ballast "sea", collision "mesh" (every convex piece of the digest; 501 with the sea ballast) and
  jaw_collision "shell" (two pads + four throat / back pieces cut from the jaws' own hulls, passed explicitly even though
  the kit's default is "shell" since 0.3.0), time step 0.002 s (decision K2: the valve-hinge friction time constant is
  2 dt and the end stops 4 dt, and they are only valid at this step).
  No appearance geometry in the physics model (visual=False): the render twin adds the look (package P08).  The
  collision meshes are a pinned copy (`DEV/assets/vehicle_collision/`, SHA256SUMS) of the files the kit loads for
  this build, so the cell never reads the maintainer's mesh library.

No vehicle number is written here: mass, centres, sites and the gripper come from the digest at run time.

    from subseacell import vehicle as VH
    vb = VH.build_vehicle(hydro, world=None, pos=(x, y, z), quat=q, floor_depth=None)  # kit VehicleBuild
    # add the task's structures to vb.spec.worldbody, then uwkit.UnderwaterSim(vb, water=..., current=..., seed=...)
    props = VH.mass_properties(vb)       # rigid totals of the compiled model (hull frame)
    VH.compare_to_digest(props)          # relative differences to mission.state_sea_SP35_trim
    VH.gripper_geometry(vb)              # TCP, tool axis, jaw-tip distance, open angle, ... (from the build)

Maintainer CLI (pin / check the collision meshes; the live library path comes from the digest manifest):
    python -I -B vehicle.py --pin        # copy the build's collision meshes into DEV/assets/vehicle_collision/
    python -I -B vehicle.py --check      # every pinned file = SHA256SUMS = the live library; set = the build's set
    python -I -B vehicle.py --mass       # mass properties of the built model vs the digest
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DEV = os.path.normpath(os.path.join(HERE, "..", ".."))

if __name__ == "__main__":                                  # maintainer CLI from the dev tree: find the kit
    _sh = os.environ.get("SUBSEACELL_UWKIT_ROOT") or os.path.normpath(os.path.join(_DEV, "..", "..", "shared"))
    if os.path.isdir(os.path.join(_sh, "uwkit")) and _sh not in sys.path:
        sys.path.insert(0, _sh)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

import uwkit  # noqa: E402
from uwkit import hydrostatics as HS  # noqa: E402
from uwkit import refdata  # noqa: E402

# ---------------------------------------------------------------------------------------------------- fixed choices
BUILD = dict(config="mission", ballast="sea", collision="mesh", jaw_collision="shell")
TIMESTEP_S = 0.002                                          # world rule K2 (task_values world_rules.timestep_s)
SEA_STATE_KEY = "mission.state_sea_SP35_trim"
COLLISION_MESH_DIR = os.environ.get("SUBSEACELL_VEHICLE_MESHES") or os.path.join(_DEV, "assets", "vehicle_collision")
SUMS_NAME = "SHA256SUMS"
PIN_NAME = "PIN.json"


class VehicleBuildError(RuntimeError):
    pass


def build_vehicle(hydro=None, world=None, *, mesh_dir=None, visual=False, timestep=TIMESTEP_S, **kw):
    """The task-07 vehicle as a kit VehicleBuild.

    hydro    : None (nominal coefficient set), a set name, or a HydroCoeffs (the task world's draw)
    world    : an MjSpec to build into (the task's structures are added to vb.spec.worldbody before the simulator)
    mesh_dir : collision mesh root (default: the pinned copy COLLISION_MESH_DIR).  visual=True needs a root that also
               holds the visual meshes (maintainer library or the render twin's set) - the physics model has none.
    kw       : pos, quat, floor_depth, rho_build, battery_offset ... (passed to uwkit.build_vehicle)
    """
    for k in ("config", "ballast", "collision", "jaw_collision", "vehicle", "with_gripper", "with_tilt"):
        if k in kw:
            raise TypeError(f"{k} is fixed by the task-07 vehicle")
    if abs(float(timestep) - TIMESTEP_S) > 1e-15:
        raise ValueError(f"task 07 runs at dt = {TIMESTEP_S} s (world rule K2)")
    root = mesh_dir or COLLISION_MESH_DIR
    if visual and os.path.abspath(root) == os.path.abspath(COLLISION_MESH_DIR):
        raise VehicleBuildError("the pinned set holds collision meshes only; pass mesh_dir= with the visual meshes")
    if not os.path.isdir(root):
        raise VehicleBuildError(f"collision mesh directory {root} missing (run vehicle.py --pin in the dev tree)")
    vb = uwkit.build_vehicle(hydro=hydro, world=world, visual=visual, mesh_dir=root, timestep=timestep, **BUILD, **kw)
    g = vb.meta["gripper"]
    if g.get("type") != "hinge_pair" or abs(float(g["opening_dir_B"][2])) < 0.999:
        raise VehicleBuildError("the gripper must be the digest's hinged jaw pair opening vertically (R3)")
    ref = vb.meta["reference"]
    if ref.get("jaw_collision") != "shell" or ref.get("collision") != "mesh":
        raise VehicleBuildError("collision proxies differ from the task choice")
    st = refdata.spec(SEA_STATE_KEY)
    if abs(float(ref["totals"]["mass_kg"]) - float(st["mass_kg"])) > 1e-9 * float(st["mass_kg"]):
        raise VehicleBuildError("the kit's sea ballast is not the digest's sea state")
    vb.meta["task07"] = dict(build=dict(BUILD), visual=bool(visual), timestep=float(timestep), package="P05",
                             pinned_meshes=os.path.abspath(root) == os.path.abspath(COLLISION_MESH_DIR))
    vb.write_meta()
    return vb


# ---------------------------------------------------------------------------------------------------- mass properties
def model_parts(vb):
    """One hydrostatics Part per body of the compiled model (rigid values in the hull frame, model at qpos0)."""
    m = vb.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    hull = m.body(vb.meta["vehicle_body"]).id
    Rh, ph = d.xmat[hull].reshape(3, 3).copy(), d.xpos[hull].copy()
    parts = []
    for b in range(1, m.nbody):
        if b != hull and m.body_rootid[b] != hull:
            continue
        name = m.body(b).name
        rec = vb.meta["hydro_bodies"].get(name)
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]

        def to_h(p, Rb=Rb, pb=pb):
            return Rh.T @ (pb + Rb @ np.asarray(p, float) - ph)
        if rec is not None:
            I_b = np.asarray(rec["rigid_inertia"], float)
            parts.append(HS.Part(name, rec["rigid_mass"], to_h(rec["rigid_com"]), rec["volume"], to_h(rec["cob"]),
                                 inertia_com_kgm2=(Rh.T @ Rb) @ I_b @ (Rh.T @ Rb).T))
        elif m.body_mass[b] > 0:
            I = np.diag(m.body_inertia[b])
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, m.body_iquat[b])
            Ri = (Rh.T @ Rb) @ Rq.reshape(3, 3)
            parts.append(HS.Part(name, m.body_mass[b], to_h(m.body_ipos[b]), 0.0, None, inertia_com_kgm2=Ri @ I @ Ri.T))
    return m, parts


def mass_properties(vb, rho=None):
    """Rigid totals of the compiled model in the hull frame: mass, CG, displaced volume, CB, BG, net buoyancy, inertia
    about the CG.  rho = water density (default: the digest's sea state)."""
    st = refdata.spec(SEA_STATE_KEY)
    rho = float(st["rho_kg_m3"] if rho is None else rho)
    _, parts = model_parts(vb)
    h = HS.compute(parts, rho, uwkit.G)
    return dict(rho=rho, mass_kg=h.mass_kg, cg_B_m=h.cg_m.tolist(), displacement_m3=h.displacement_m3,
                cb_B_m=h.cb_m.tolist(), BG_m=h.BG_m, net_buoyancy_kg=h.net_buoyancy_kg,
                inertia_about_cg_B_kgm2=h.inertia_cg_kgm2.tolist(), pitch_trim_deg=h.pitch_trim_deg,
                roll_trim_deg=h.roll_trim_deg, n_parts=len(parts))


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def compare_to_digest(props):
    """Relative differences (|model - digest| / |digest|, vectors by norm) to mission.state_sea_SP35_trim, plus the
    absolute differences in metres for the centres."""
    st = refdata.spec(SEA_STATE_KEY)
    out = dict(mass=_rel(props["mass_kg"], st["mass_kg"]), cg=_rel(props["cg_B_m"], st["cg_B_m"]),
               cb=_rel(props["cb_B_m"], st["cb_B_m"]), displacement=_rel(props["displacement_m3"], st["displacement_m3"]),
               inertia=_rel(props["inertia_about_cg_B_kgm2"], st["inertia_about_cg_B_kgm2"]),
               net_buoyancy=_rel(props["net_buoyancy_kg"], st["net_buoyancy_kg"]),
               BG=_rel(props["BG_m"], st["BG_m"]))
    out["cg_abs_m"] = float(np.abs(np.subtract(props["cg_B_m"], st["cg_B_m"])).max())
    out["cb_abs_m"] = float(np.abs(np.subtract(props["cb_B_m"], st["cb_B_m"])).max())
    return out


# ---------------------------------------------------------------------------------------------------- gripper
def gripper_geometry(vb_or_meta):
    """The gripper as the build has it (body frame B): TCP, tool axis, opening direction, jaw-tip distance ahead of
    the TCP, full-open angle per jaw, stroke time, TCP clamp force, pad half width."""
    meta = vb_or_meta.meta if hasattr(vb_or_meta, "meta") else vb_or_meta
    g = meta["gripper"]
    return dict(tcp_B_m=list(map(float, g["tcp_B_m"])), axis_B=list(map(float, g["axis_B"])),
                opening_dir_B=list(map(float, g["opening_dir_B"])), tip_ahead_m=float(g["pad"]["x_to"]),
                theta_open_rad=float(g["theta_open_rad"]), close_open_s=float(g["close_open_s"]),
                lever_tcp_m=float(g["lever_tcp_m"]), grip_force_tcp_N=float(g["grip_force_tcp_N"]),
                pad_half_width_m=float(g["pad"]["half_width"]), pad_face_m=float(g["pad"]["face"]))


# ---------------------------------------------------------------------------------------------------- pinned meshes
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def collision_mesh_set(manifest=None):
    """Relative paths (mesh library layout) of every collision mesh the task build loads: the collision pieces of the
    `mission` items (hull pieces, ballast units used as templates, and the jaws, whose pieces the shell builder cuts)."""
    man = manifest if isinstance(manifest, dict) else refdata.load_manifest(manifest)
    items = [i for i in man["items"] if BUILD["config"] in i["configs"] and i.get("mesh")]
    return sorted({c for i in items for c in (i["mesh"].get("collision") or [])})


def pin_meshes(dest=COLLISION_MESH_DIR, library=None):
    """Copy the build's collision meshes from the live library into `dest` (library layout kept) and write
    SHA256SUMS + PIN.json (digest version, manifest sha256, file count and bytes)."""
    man = refdata.load_manifest()
    lib = library or refdata.mesh_root(man)
    files = collision_mesh_set(man)
    lines, total = [], 0
    for rel in files:
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(lib, rel), dst)
        lines.append(f"{_sha(dst)}  {rel}")
        total += os.path.getsize(dst)
    with open(os.path.join(dest, SUMS_NAME), "w") as f:
        f.write("\n".join(lines) + "\n")
    sv = refdata.load_spec_values()
    pin = dict(what="collision meshes of the task-07 physics vehicle (mission, sea ballast, collision='mesh', "
                    "jaw_collision='shell'); maintainer side - the file names carry part names",
               digest_version=sv.get("digest", {}).get("version"),
               manifest_sha256=sv.get("digest", {}).get("manifest_sha256"),
               uwkit_version=getattr(uwkit, "__version__", None), n_files=len(files), bytes=total,
               sums_sha256=_sha(os.path.join(dest, SUMS_NAME)))
    with open(os.path.join(dest, PIN_NAME), "w") as f:
        json.dump(pin, f, indent=1)
        f.write("\n")
    return pin


def check_meshes(dest=COLLISION_MESH_DIR, library=None, live=True):
    """Verify the pinned copy.  Returns dict(n_files, mismatches=[...]) - a mismatch is: a file of the build's set
    missing from the copy or from SHA256SUMS, a copy whose hash differs from SHA256SUMS, a live-library file whose hash
    differs (live=True), or an extra file in SHA256SUMS that the build does not load."""
    sums_path = os.path.join(dest, SUMS_NAME)
    want = {}
    if os.path.isfile(sums_path):
        for line in open(sums_path).read().splitlines():
            if line.strip():
                h, rel = line.split("  ", 1)
                want[rel] = h
    man = refdata.load_manifest()
    lib = library or refdata.mesh_root(man)
    files = collision_mesh_set(man)
    bad = []
    for rel in files:
        p = os.path.join(dest, rel)
        if rel not in want:
            bad.append(("not_in_sums", rel))
        elif not os.path.isfile(p):
            bad.append(("missing_copy", rel))
        elif _sha(p) != want[rel]:
            bad.append(("copy_hash", rel))
        elif live and _sha(os.path.join(lib, rel)) != want[rel]:
            bad.append(("live_hash", rel))
    for rel in sorted(set(want) - set(files)):
        bad.append(("extra", rel))
    on_disk = set()
    for dp, _, fn in os.walk(dest):
        for f in fn:
            r = os.path.relpath(os.path.join(dp, f), dest).replace(os.sep, "/")
            if r not in (SUMS_NAME, PIN_NAME):
                on_disk.add(r)
    for rel in sorted(on_disk - set(files)):
        bad.append(("stray_file", rel))
    return dict(n_files=len(files), n_sums=len(want), mismatches=bad)


def _main(argv):
    if "--pin" in argv:
        print(json.dumps(pin_meshes(), indent=1))
        return 0
    if "--check" in argv:
        r = check_meshes(live="--offline" not in argv)
        print(f"{r['n_files']} files, {len(r['mismatches'])} mismatches", r["mismatches"][:5])
        return 1 if r["mismatches"] else 0
    if "--mass" in argv:
        p = mass_properties(build_vehicle())
        print(json.dumps(dict(props={k: p[k] for k in ("mass_kg", "cg_B_m", "cb_B_m", "displacement_m3", "BG_m")},
                              rel=compare_to_digest(p)), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
