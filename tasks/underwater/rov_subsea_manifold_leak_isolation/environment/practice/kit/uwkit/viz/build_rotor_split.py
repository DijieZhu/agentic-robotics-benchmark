"""Maintainer tool: split every thruster prototype of the reference vehicle into rotor and stator meshes.

    python -I -B build_rotor_split.py [--out-json PATH] [--mesh-dir DIR] [--check]

Inputs (maintainer side, read only): the parts manifest (thruster items, their poses and visual mesh), the spec values
(per-site thruster table: position, force axis, propeller hand), and the per-solid tessellation cache that the
reference-vehicle digest generator wrote next to the mesh library (<mesh_root>/../cache/<src>/meshes/<proto>.npz, arrays
V [mm], F, solid).  The prototype keys are derived from the manifest's visual mesh file names at run time.

Outputs (only new files):
  <mesh_root>/viz/thruster_<hand>_stator.stl        every non-rotating solid (nozzle with its tail cone and rear vanes,
                                                    stator base with front struts, nose cone, inserts, screws)
  <mesh_root>/viz/thruster_<hand>_rotor_hub.stl     propeller hub (axisymmetric part, r <= hub radius)
  <mesh_root>/viz/thruster_<hand>_rotor_blades.stl  blades (minus the marker patch)
  <mesh_root>/viz/thruster_<hand>_rotor_marker.stl  outer part of one blade (visual aid colour only; same geometry)
  uwkit/viz/rotor_split.json                        solid roles, axis / centre (prototype frame and B frame per site),
                                                    blade pitch and hand measured from the B-rep tessellation, the spin
                                                    direction for forward thrust (two independent geometric checks),
                                                    the duct lips for the wake, and the checks against the digest table.
All meshes stay in the prototype's native frame (metres), so the manifest pose T_B places them unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from uwkit import refdata  # noqa: E402

CONFIGS = ("mission", "heavy", "std")
TARGET_FACES = dict(stator=9000, rotor=4000)
HUB_TOL_M = 0.0004          # hub = faces whose three vertices lie within the hub cylinder radius + this
MARKER_R_FRAC = 0.62        # marker patch: blade 0, centroid radius > this x tip radius


def _cache_npz(mesh_root, visual_lo):
    stem = os.path.splitext(os.path.basename(visual_lo))[0]
    src, proto = stem.split("__", 1)
    return os.path.join(os.path.dirname(os.path.abspath(mesh_root)), "cache", src, "meshes", proto + ".npz"), stem


def _face_geom(V, F):
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    N = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(N, axis=1)
    n = N / np.maximum(2 * area, 1e-30)[:, None]
    return (A + B + C) / 3.0, n, area


def _cyl(c, axis_z_sign=1.0):
    r = np.hypot(c[:, 0], c[:, 1])
    et = np.c_[-c[:, 1], c[:, 0], np.zeros(len(c))] / np.maximum(r, 1e-12)[:, None]
    return r, et


def helical_area(V, F):
    """Blade-likeness of a solid: area of the faces in the blade annulus (r > 20 mm) whose local helicoid pitch
    p = -2 pi r n_theta / n_z is within +-30 % of the area-weighted median pitch.  Blades are helicoids (one consistent
    pitch over both faces); struts, vanes, threads and surfaces of revolution give scattered or no pitch."""
    c, n, a = _face_geom(V, F)
    r, et = _cyl(c)
    nt, nz = (n * et).sum(1), n[:, 2]
    m = (r > 0.020) & (np.abs(nz) > 0.2) & (np.abs(nt) > 0.2)
    if m.sum() < 10:
        return 0.0
    p = -2 * math.pi * r[m] * nt[m] / nz[m]
    w = a[m]
    o = np.argsort(p)
    med = p[o][min(int(np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)), len(o) - 1)]
    return float(w[np.abs(p - med) < 0.3 * abs(med)].sum())


def blade_pitch(V, F, r_tip):
    """Area-weighted pitch of the blade faces from the face normals: on a helicoid z = p theta / 2 pi + f(r) the normal
    is orthogonal to r e_theta + p / (2 pi) e_z, so p = -2 pi r n_theta / n_z.  theta is right-handed about +z_proto, so
    p > 0 = right-handed screw.  Returns (median p [m], 10 / 90 % quantiles, area fraction with the median's sign)."""
    c, n, a = _face_geom(V, F)
    r, et = _cyl(c)
    nt, nz = (n * et).sum(1), n[:, 2]
    m = (r > 0.58 * r_tip) & (r < 0.92 * r_tip) & (np.abs(nz) > 0.2)
    p = -2 * math.pi * r[m] * nt[m] / nz[m]
    w = a[m]
    o = np.argsort(p)
    cw = np.cumsum(w[o]) / w.sum()
    q = [float(p[o][min(np.searchsorted(cw, x), len(o) - 1)]) for x in (0.1, 0.5, 0.9)]
    frac = float(w[np.sign(p) == np.sign(q[1])].sum() / w.sum())
    return q[1], (q[0], q[2]), frac


def panel_axial_force(V, F, centre, axis, omega_sign):
    """Sign check independent of the pitch formula: Newtonian impact model of the spinning rotor in still water.
    Faces that move into the water along their outward normal (v . n > 0) get a pressure force -k (v . n)^2 n.
    Returns the axial force on the rotor along +axis (arbitrary units) for omega = omega_sign * axis."""
    c, n, a = _face_geom(V, F)
    w = omega_sign * np.asarray(axis, float)
    v = np.cross(w, c - centre)
    vn = (v * n).sum(1)
    m = vn > 0
    f = -(vn[m] ** 2)[:, None] * n[m] * a[m][:, None]
    return float(f.sum(0) @ axis)


def decimate(V, F, target):
    import trimesh
    import fast_simplification
    m = trimesh.Trimesh(V, F, process=True)
    if len(m.faces) > target:
        Vd, Fd = fast_simplification.simplify(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64),
                                              target_reduction=1.0 - target / len(m.faces))
        m = trimesh.Trimesh(Vd, Fd, process=True)
    return np.asarray(m.vertices), np.asarray(m.faces)


def write_stl(path, V, F):
    import trimesh
    trimesh.Trimesh(V, F, process=False).export(path)


def _brep_solid(npz_path, solid):
    """Exact B-rep properties of one solid from the cache's protos.json (volume mm^3, centroid mm), or None."""
    pj = os.path.join(os.path.dirname(os.path.dirname(npz_path)), "protos.json")
    key = os.path.splitext(os.path.basename(npz_path))[0]
    if not os.path.exists(pj):
        return None
    with open(pj) as f:
        for p in json.load(f):
            if p.get("key") == key and len(p.get("solids", [])) > solid:
                return p["solids"][solid]
    return None


def analyse_prototype(npz_path, hand_label):
    d = np.load(npz_path)
    V = d["V"] / 1000.0
    F, S = d["F"], d["solid"]
    solids = sorted(int(s) for s in np.unique(S))
    info = {}
    for s in solids:
        f = F[S == s]
        vv = V[np.unique(f)]
        r = np.hypot(vv[:, 0], vv[:, 1])
        c, n, a = _face_geom(V, f)
        tet = np.einsum("ij,ij->i", V[f[:, 0]], np.cross(V[f[:, 1]], V[f[:, 2]])) / 6.0      # signed tetra volumes
        vol = float(tet.sum())
        com = ((V[f[:, 0]] + V[f[:, 1]] + V[f[:, 2]]) / 4.0 * tet[:, None]).sum(0) / vol          # volume centroid
        info[s] = dict(faces=int(len(f)), volume_m3=vol, r_min=float(r.min()), r_max=float(r.max()),
                       z_min=float(vv[:, 2].min()), z_max=float(vv[:, 2].max()),
                       helical_area=helical_area(V, f), com=com)
    rotor = max(solids, key=lambda s: info[s]["helical_area"])
    # the axis: the rotor's mass-symmetry axis (principal axis of the surface second moment); it is +z_proto here
    fr = F[S == rotor]
    c, n, a = _face_geom(V, fr)
    com = info[rotor]["com"]
    X = (c - com) * np.sqrt(a)[:, None]
    w_, U = np.linalg.eigh(X.T @ X)
    ax = U[:, int(np.argmax(np.abs(U[2])))]          # the eigenvector closest to z_proto
    ax = ax * (1 if ax[2] > 0 else -1)
    if np.degrees(math.acos(min(1.0, abs(ax[2])))) > 0.5:
        raise RuntimeError(f"rotor axis is not the prototype z axis ({ax})")
    axis_proto = np.array([0.0, 0.0, 1.0])
    centre_xy = com[:2]
    r_tip = info[rotor]["r_max"]
    # nose cone = slim solid (r_max < 0.6 r_tip) furthest along the axis from the propeller: defines "front"
    slim = [s for s in solids if s != rotor and info[s]["r_max"] < 0.6 * r_tip and info[s]["volume_m3"] > 1e-6]
    nose = max(slim, key=lambda s: abs(0.5 * (info[s]["z_min"] + info[s]["z_max"]) - com[2]))
    front = 1.0 if 0.5 * (info[nose]["z_min"] + info[nose]["z_max"]) > com[2] else -1.0
    # nozzle = the largest solid (duct + mounting base + rear vanes + tail cone); its lips bound the jet
    nozzle = max((s for s in solids if s != rotor), key=lambda s: info[s]["volume_m3"])
    fn = F[S == nozzle]
    vn = V[np.unique(fn)]
    rn = np.hypot(vn[:, 0], vn[:, 1])
    duct = vn[rn > 1.05 * r_tip]
    lips = (float(duct[:, 2].min()), float(duct[:, 2].max()))
    band = vn[(np.abs(vn[:, 2] - com[2]) < 0.004) & (rn > 0.9 * r_tip)]
    duct_inner_r = float(np.hypot(band[:, 0], band[:, 1]).min()) if len(band) else 1.03 * r_tip
    tail = vn[rn < 0.5 * r_tip]
    # hub radius: the most populated vertex radius of the rotor (the hub cylinder)
    vr = V[np.unique(fr)]
    rr = np.hypot(vr[:, 0] - centre_xy[0], vr[:, 1] - centre_xy[1])
    h, e = np.histogram(rr, bins=np.arange(0.0, r_tip + 0.0005, 0.0005))
    hub_r = float(e[np.argmax(h) + 1])
    p_med, p_q, p_frac = blade_pitch(V, fr, r_tip)
    # centre on the axis at the propeller's centroid; the exact B-rep centroid of the cache (protos.json, mm) is used when
    # present (the 0.2 mm tessellation of the thin blades overstates the volume by ~3 % and moves the centroid ~0.1 mm)
    brep = _brep_solid(npz_path, rotor)
    zc = brep["com"][2] / 1000.0 if brep else com[2]
    centre = np.array([centre_xy[0], centre_xy[1], zc])
    fz = {s: panel_axial_force(V, fr, centre, axis_proto * front, s) for s in (+1, -1)}
    spin_panel = +1 if fz[+1] > 0 else -1
    hand = "right" if p_med > 0 else "left"
    # right-handed screw advances along omega; forward thrust = force along +front
    spin_pitch = +1 if hand == "right" else -1
    spin_official = -1 if hand_label == "CW" else +1     # CW = clockwise seen from the nose (looking along -front)
    roles = {}
    for s in solids:
        if s == rotor:
            roles[s] = "propeller (blades + hub) - ROTOR"
        elif s == nose:
            roles[s] = "nose cone (cable entry, screwed to the stator base) - static"
        elif s == nozzle:
            roles[s] = ("nozzle: duct + mounting base + rear vanes + tail cone (one solid) - static"
                        if len(tail) else "nozzle: duct + mounting base - static")
        elif info[s]["volume_m3"] > 2e-5:
            roles[s] = "stator base with front struts and potted stator - static"
        elif info[s]["r_min"] > 0.8 * r_tip and info[s]["z_max"] - info[s]["z_min"] < 0.012 \
                and info[s]["r_max"] - info[s]["r_min"] > 0.010:
            roles[s] = "radial screw nozzle -> strut - static"
        elif abs(0.5 * (info[s]["z_min"] + info[s]["z_max"]) - info[nose]["z_min" if front > 0 else "z_max"]) < 0.008:
            roles[s] = "axial screw nose cone -> stator base - static"
        else:
            roles[s] = "threaded insert in the mounting base - static"
    return dict(V=V, F=F, S=S, solids=solids, info=info, rotor=rotor, nose=nose, nozzle=nozzle, roles=roles,
                front=front, centre=centre, r_tip=r_tip, hub_r=hub_r, lips=lips, duct_inner_r=duct_inner_r,
                tail_tip=(float(tail[:, 2].min()) if front > 0 else float(tail[:, 2].max())) if len(tail) else None,
                pitch=(p_med, p_q, p_frac), hand=hand, spin_pitch=spin_pitch, spin_panel=spin_panel,
                spin_official=spin_official, panel_force=fz)


def split_rotor(V, F, centre, hub_r, r_tip):
    """Decimated rotor -> (hub, blades without marker, marker) face sets, split on the decimated mesh (no seams)."""
    Vd, Fd = decimate(V, F, TARGET_FACES["rotor"])
    rv = np.hypot(Vd[:, 0] - centre[0], Vd[:, 1] - centre[1])
    hub = (rv[Fd] <= hub_r + HUB_TOL_M).all(1)
    c = Vd[Fd].mean(1)
    rc = np.hypot(c[:, 0] - centre[0], c[:, 1] - centre[1])
    th = np.arctan2(c[:, 1] - centre[1], c[:, 0] - centre[0])
    # blade 0 = the blade whose tip sector contains the largest tip area; marker = outer part of that blade
    tip = (~hub) & (rc > 0.85 * r_tip)
    t0 = float(np.angle(np.exp(1j * th[tip]).mean())) if tip.any() else 0.0
    # cluster the tip faces into 3 sectors and take the one around angle t0's nearest tip face
    k0 = np.argmin(np.abs(np.angle(np.exp(1j * (th[tip] - t0)))))
    a0 = th[tip][k0]
    sector = np.abs(np.angle(np.exp(1j * (th - a0)))) < math.radians(50)
    marker = (~hub) & (rc > MARKER_R_FRAC * r_tip) & sector
    return Vd, Fd[hub], Fd[(~hub) & (~marker)], Fd[marker], float(a0)


def thruster_table(config, items_by_id, protos, spec_path=None):
    rows = []
    tab = refdata.spec(f"{config}.thrusters", spec_path)
    for t in tab:
        it = items_by_id[t["part"]]
        pr = protos[it["mesh"]["visual_lo"]]
        T = np.asarray(it["T_B"], float)
        R, p = T[:3, :3], T[:3, 3]
        axis_B = R @ (np.array([0.0, 0.0, 1.0]) * pr["front"])
        axis_B = axis_B / np.linalg.norm(axis_B)        # the manifest pose is rounded to 7 decimals
        centre_B = R @ pr["centre"] + p
        lip_rear = R @ np.array([pr["centre"][0], pr["centre"][1], pr["lips"][0 if pr["front"] > 0 else 1]]) + p
        lip_front = R @ np.array([pr["centre"][0], pr["centre"][1], pr["lips"][1 if pr["front"] > 0 else 0]]) + p
        ad = np.asarray(t["axis_unit"], float)
        ad = ad / np.linalg.norm(ad)                     # the table rounds to 5 decimals (norm 0.999996)
        d_axis = math.degrees(math.acos(max(-1.0, min(1.0, float(axis_B @ ad)))))
        d_pos = float(np.linalg.norm(centre_B - np.asarray(t["pos_m"], float)))
        rows.append(dict(site=t["site"], item=t["part"], propeller=t["propeller"], prototype=pr["key"],
                         axis_B=[round(float(x), 7) for x in axis_B], centre_B_m=[round(float(x), 7) for x in centre_B],
                         rear_lip_B_m=[round(float(x), 6) for x in lip_rear],
                         front_lip_B_m=[round(float(x), 6) for x in lip_front],
                         spin_sign_forward=pr["spin"],
                         check=dict(axis_vs_digest_deg=round(d_axis, 5), centre_vs_digest_mm=round(1e3 * d_pos, 4),
                                    hand_vs_digest=pr["hand_label"] == t["propeller"]),
                         T_B=[[round(float(x), 9) for x in row] for row in T.tolist()]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(HERE, "rotor_split.json"))
    ap.add_argument("--mesh-dir", default=None, help="default: <mesh_root>/viz")
    ap.add_argument("--check", action="store_true", help="analyse only, write nothing")
    a = ap.parse_args()
    man = refdata.load_manifest()
    root = refdata.mesh_root(man)
    out_dir = a.mesh_dir or os.path.join(root, "viz")
    items = [i for i in man["items"] if i.get("category") == "thruster" and i.get("mesh")]
    by_id = {i["id"]: i for i in items}
    hands = {}
    for c in CONFIGS:
        for t in refdata.spec(f"{c}.thrusters"):
            vl = by_id[t["part"]]["mesh"]["visual_lo"]
            if hands.setdefault(vl, t["propeller"]) != t["propeller"]:
                raise RuntimeError("one prototype mesh with two propeller hands")
    protos = {}
    for vl, hand_label in sorted(hands.items(), key=lambda kv: kv[1]):
        npz, stem = _cache_npz(root, vl)
        A = analyse_prototype(npz, hand_label)
        spins = (A["spin_pitch"], A["spin_panel"], A["spin_official"])
        key = hand_label.lower()
        print(f"[{key}] rotor solid {A['rotor']} ({A['info'][A['rotor']]['faces']} tris, "
              f"{A['info'][A['rotor']]['volume_m3'] * 1e6:.2f} cm3), nose solid {A['nose']}, nozzle solid {A['nozzle']}, "
              f"front {A['front']:+.0f} z, hub r {A['hub_r'] * 1e3:.2f} mm, tip r {A['r_tip'] * 1e3:.2f} mm, "
              f"pitch {A['pitch'][0] * 1e3:+.1f} mm ({A['hand']}-handed, {A['pitch'][2] * 100:.0f} % of blade area), "
              f"spin for forward thrust: pitch {spins[0]:+d} panel {spins[1]:+d} official-label {spins[2]:+d}", flush=True)
        if len(set(spins)) != 1:
            raise RuntimeError(f"spin-direction checks disagree for the {hand_label} prototype: {spins}")
        rot_f = A["F"][A["S"] == A["rotor"]]
        sta_f = A["F"][A["S"] != A["rotor"]]
        Vr, Fh, Fb, Fm, a0 = split_rotor(A["V"], rot_f, A["centre"], A["hub_r"], A["r_tip"])
        Vs, Fs = decimate(A["V"], sta_f, TARGET_FACES["stator"])
        files = dict(stator=f"thruster_{key}_stator.stl", rotor_hub=f"thruster_{key}_rotor_hub.stl",
                     rotor_blades=f"thruster_{key}_rotor_blades.stl", rotor_marker=f"thruster_{key}_rotor_marker.stl")
        if not a.check:
            os.makedirs(out_dir, exist_ok=True)
            write_stl(os.path.join(out_dir, files["stator"]), Vs, Fs)
            write_stl(os.path.join(out_dir, files["rotor_hub"]), Vr, Fh)
            write_stl(os.path.join(out_dir, files["rotor_blades"]), Vr, Fb)
            write_stl(os.path.join(out_dir, files["rotor_marker"]), Vr, Fm)
        protos[vl] = dict(key=key, hand_label=hand_label, front=A["front"], centre=A["centre"], lips=A["lips"],
                          spin=spins[0])
        protos[vl]["record"] = dict(
            hand_label=hand_label, items=sorted(i for i, it in by_id.items() if it["mesh"]["visual_lo"] == vl),
            mesh_files=files, faces=dict(stator=int(len(Fs)), rotor_hub=int(len(Fh)), rotor_blades=int(len(Fb)),
                                         rotor_marker=int(len(Fm)), source_rotor=int(len(rot_f)), source_stator=int(len(sta_f))),
            solid_roles={str(s): A["roles"][s] for s in A["solids"]},
            solids={str(s): dict(faces=A["info"][s]["faces"], volume_cm3=round(A["info"][s]["volume_m3"] * 1e6, 3),
                                 r_mm=[round(A["info"][s]["r_min"] * 1e3, 2), round(A["info"][s]["r_max"] * 1e3, 2)],
                                 z_mm=[round(A["info"][s]["z_min"] * 1e3, 2), round(A["info"][s]["z_max"] * 1e3, 2)])
                    for s in A["solids"]},
            rotor_solids=[A["rotor"]], stator_solids=[s for s in A["solids"] if s != A["rotor"]],
            axis_proto=[0.0, 0.0, A["front"]], centre_proto_m=[round(float(x), 7) for x in A["centre"]],
            hub_radius_m=round(A["hub_r"], 5), tip_radius_m=round(A["r_tip"], 5), blades=3,
            marker_blade_angle_proto_deg=round(math.degrees(a0), 2),
            duct_lips_proto_z_m=[round(x, 5) for x in A["lips"]], duct_inner_radius_m=round(A["duct_inner_r"], 5),
            tail_cone_tip_proto_z_m=None if A["tail_tip"] is None else round(A["tail_tip"], 5),
            blade_pitch_m=dict(median=round(A["pitch"][0], 5), q10_q90=[round(x, 5) for x in A["pitch"][1]],
                               area_fraction_same_sign=round(A["pitch"][2], 3), hand=A["hand"]),
            spin_checks=dict(from_pitch_hand=spins[0], newtonian_panel=spins[1], official_label=spins[2],
                             panel_axial_force_omega_plus_minus=[round(A["panel_force"][+1], 9),
                                                                 round(A["panel_force"][-1], 9)]),
            spin_sign_forward=spins[0])
    out = dict(
        schema="uwkit.viz.rotor_split/1",
        conventions=dict(
            frames="prototype = native mesh frame of the thruster prototype (metres; the manifest pose T_B maps it to B); "
                   "B = body FLU",
            axis="axis_B = unit vector along the site +x of the digest table = direction of the force on the body under "
                 "positive (forward) thrust = from the propeller towards the nose cone (the 'front' of the thruster)",
            spin_sign_forward="sign of omega . axis_B for forward thrust (positive command / positive rotor speed); "
                              "-1 = the rotor turns clockwise when seen from the nose cone (looking along -axis_B), "
                              "+1 = counter-clockwise.  Visible spin = spin_sign_forward x sign(rotor speed).",
            outflow="forward thrust: water leaves through the rear lip (rear_lip_B_m, behind the propeller); reverse "
                    "thrust: through the front lip around the nose cone (front_lip_B_m)",
            what_rotates="only the propeller solid (blades + hub; the motor rotor, shaft, shaft collar and the propeller "
                         "screw are inside / not in the vehicle CAD); nose cone, nozzle (with its tail cone and rear "
                         "vanes), stator base and screws are static.  See UWKIT_VIZ.md section 2.",
            marker="rotor_marker = the outer part of one blade, same geometry as the blade, drawn in a contrasting colour "
                   "only to make the rotation readable on video (not a real paint scheme)"),
        sources=dict(geometry="per-solid B-rep tessellation of the vehicle CAD (0.2 mm / 0.3 rad), decimated with quadric "
                              "simplification; poses = parts manifest T_B; table = spec values <config>.thrusters",
                     spin="three checks: blade pitch hand from face normals, Newtonian panel model of the spinning "
                          "rotor, and the vendor rule 'a CW propeller gives forward thrust spinning clockwise seen from "
                          "the nose cone'"),
        prototypes={p["key"]: p["record"] for p in protos.values()},
        thrusters={c: thruster_table(c, by_id, protos) for c in CONFIGS})
    bad = [(c, r["site"], r["check"]) for c in CONFIGS for r in out["thrusters"][c]
           if r["check"]["axis_vs_digest_deg"] > 0.01 or r["check"]["centre_vs_digest_mm"] > 0.02
           or not r["check"]["hand_vs_digest"]]
    for c in CONFIGS:
        for r in out["thrusters"][c]:
            print(f"   {c:8s} {r['site']} {r['propeller']:3s} spin {r['spin_sign_forward']:+d} axis {r['axis_B']} "
                  f"d_axis {r['check']['axis_vs_digest_deg']:.5f} deg d_centre {r['check']['centre_vs_digest_mm']:.4f} mm")
    if bad:
        raise RuntimeError(f"thruster table mismatch: {bad}")
    if not a.check:
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", a.out_json, "and meshes in", out_dir)


if __name__ == "__main__":
    main()
