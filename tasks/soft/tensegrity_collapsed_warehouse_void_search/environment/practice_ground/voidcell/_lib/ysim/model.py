"""MJCF generator: three rigid struts made of the authors' strut mesh, collision geometry fitted to that mesh,
the authors' tendon net mapped onto the real hooks and line guides, Earth gravity, SI units.

The tendon-net code (dead-band springs as tension-only tendons, sites from measured features) is new; the idea of
generating the whole model as one MJCF string with name->id tables follows an earlier tensegrity kit that is not
published (not imported).
"""
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from . import geometry as geom
from . import params as prm
from . import paths


@dataclass
class ModelConfig:
    assets: str = None                 # asset dir (None -> paths.assets_dir()); only read when visual / cap_hulls
    mesh_res: str = "dec"              # "dec" (decimated copy) or "full" (authors' full-resolution triangles)
    visual: bool = True                # include the visual meshes (False -> collision primitives only, no asset dir)
    cap_hulls: bool = False            # add the end-cap convex hulls as collision geoms (asset dir; diagnostic)
    timestep: float = None             # None -> params "timestep" (evidence: the 0.3.0 timestep study, health.py)
    integrator: str = "implicitfast"   # params "integrator"
    iterations: int = 50
    gravity: float = None              # None -> params gravity
    friction: float = None
    solref_k: float = None
    solref_b: float = None
    line_EA: float = None              # None -> params line_EA: line stiffness k_i = EA / (L0_i + line_inner_length)
    cable_k: float = None              # diagnostic only: a constant line stiffness [N/m] instead of the material rule
    short_k: float = None
    long_k: float = None
    damping: float = None
    show_sensors: bool = False         # sensors have no authors' 3D shape: hidden unless asked (debug hairlines)
    end_map: str = "pz_even"           # which mesh end carries the even node (params mesh_end_to_node, with evidence)
    hook_rule: str = None              # None -> geometry.DEFAULT_HOOK_RULE "authors_handedness" (ASSUMPTION hook_roles);
                                       # "guide_side_v030" = the 0.3.0 rule (diagnostic)
    hook_assignment: tuple = None      # diagnostic: (even, odd) index into geometry.allowed_end_assignments (one of the
                                       # 64 assignments the assembly guide allows); overrides hook_rule
    strut_collisions: bool = True      # strut-strut contacts (False only for diagnostics)
    attachments: str = "mesh"          # "mesh" = real hooks/guides (default); "authors" = the authors' idealised
                                       # MJCF sites (diagnostic only: shows what the attachment mapping changes)
    floor: bool = True
    floor_style: str = "checker"       # "checker" or "plain" (uniform matte floor for inspection media)
    lights: bool = True                # False: no ysim key / fill light and the headlight off (a site brings its own
                                       # lighting; added in ysim 0.2.0 for this task's cell)
    world_xml: str = ""                # extra worldbody XML (terrain, obstacles)
    asset_xml: str = ""
    offscreen: tuple = (1280, 720)

    def resolved(self):
        d = asdict(self)
        defaults = {"gravity": "gravity", "friction": "friction", "solref_k": "contact_solref_k",
                    "solref_b": "contact_solref_b", "line_EA": "line_EA", "short_k": "short_sensor_stiffness",
                    "long_k": "long_sensor_stiffness", "damping": "tendon_damping", "timestep": "timestep"}
        for k, p in defaults.items():
            if d[k] is None:
                d[k] = prm.value(p)
        d["assets"] = paths.assets_dir(d["assets"])
        return d


def line_stiffness(L0, EA, constant_k=None):
    """Material rule of the fishing line (params line_stiffness): k = EA / (L0 + line_inner_length) [N/m] for an
    unstretched guide-to-hook length L0 [m]; constant_k (diagnostic) overrides the rule."""
    if constant_k is not None:
        return float(constant_k)
    return float(EA) / (float(L0) + prm.value("line_inner_length"))


def _hull_mesh_xml(name):
    """Inline MuJoCo mesh of the electronics-enclosure collision hull from notes/sim_base_data (no asset file)."""
    import json
    with open(paths.data_path("collision_hull_elec.json")) as fh:
        h = json.load(fh)
    vs = " ".join("%r" % float(x) for v in h["vertices_m"] for x in v)
    fs = " ".join(str(int(i)) for f in h["faces"] for i in f)
    return '<mesh name="%s" vertex="%s" face="%s" maxhullvert="64"/>' % (name, vs, fs)


SOCK_RGBA = {"red": (0.80, 0.10, 0.10, 1), "green": (0.10, 0.62, 0.20, 1), "blue": (0.12, 0.25, 0.85, 1)}


def _quat_from_R(R):
    """MuJoCo quaternion (w, x, y, z) of a rotation matrix (columns = body axes in world)."""
    q = np.zeros(4)
    import mujoco
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def initial_pose(feat=None, end_map="pz_even", fit=None):
    """Body poses (pos, R) placing the three struts on the authors' MJCF initial node positions (centre = node
    midpoint, +z = from the odd to the even node); the roll about each axis is fitted to the authors' role sites.

    fit "sites3d" (default, ysim 0.3.1): the closed-form roll that minimises the summed squared 3D distance between
    the strut's ten attachment sites (S_out, S_in, I, L and the line guide at both ends) and the authors' sites of the
    same roles (the comparison behind params hook_roles). fit "v030": ysim 0.3.0's angular fit (both S_out / S_in,
    the even end's I, L and guide); used automatically with hook_rule "guide_side_v030", so that the diagnostic
    reproduces 0.3.0 exactly."""
    feat = feat or geom.strut_features()
    if fit is None:
        fit = "v030" if feat.get("hook_rule") == "guide_side_v030" else "sites3d"
    nodes, sites = geom.authors_initial_nodes()
    out = []
    # authors' site name of each role at node n
    cab_from = {m: l for m, l in geom.CABLES}          # motor node -> line node
    cab_to = {l: m for m, l in geom.CABLES}            # line node -> motor node
    longp = {}
    for a, b in geom.LONG:
        longp[a], longp[b] = b, a
    if fit == "sites3d":
        for k in range(3):
            na, nb = 2 * k, 2 * k + 1
            c = 0.5 * (nodes[na] + nodes[nb])
            u = nodes[na] - nodes[nb]
            u /= np.linalg.norm(u)
            if geom.node_sign(na, end_map) < 0:
                u = -u
            e1 = np.cross(u, [0.0, 0.0, 1.0])
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(u, [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            sxx = sxy = 0.0
            for n in (na, nb):
                s = geom.node_sign(n, end_map)
                hk = feat["hooks"][s]
                for p, qn in ((hk["S_out"], "s_%d_%d" % (n, cab_from[n])), (hk["S_in"], "s_%d_%d" % (n, cab_to[n])),
                              (hk["I"], "s_%d_b%d" % (n, cab_to[n])), (hk["L"], "s_%d_%d" % (n, longp[n])),
                              (feat["guide"][s], "s_b%d_%d" % (n, cab_from[n]))):
                    d = sites[qn] - c
                    qx, qy = float(d @ e1), float(d @ e2)
                    sxx += p[0] * qx + p[1] * qy
                    sxy += p[0] * qy - p[1] * qx
            psi = math.atan2(sxy, sxx)
            x = math.cos(psi) * e1 + math.sin(psi) * e2
            R = np.stack([x, np.cross(u, x), u], axis=1)
            res = []
            for n in (na, nb):
                s = geom.node_sign(n, end_map)
                hk = feat["hooks"][s]
                for p, qn in ((hk["S_out"], "s_%d_%d" % (n, cab_from[n])), (hk["S_in"], "s_%d_%d" % (n, cab_to[n])),
                              (hk["I"], "s_%d_b%d" % (n, cab_to[n])), (hk["L"], "s_%d_%d" % (n, longp[n])),
                              (feat["guide"][s], "s_b%d_%d" % (n, cab_from[n]))):
                    res.append(float(np.linalg.norm(c + R @ p - sites[qn])))
            out.append({"pos": c, "R": R, "roll_fit_rms_deg": float("nan"),
                        "roll_fit_rms_mm": 1000.0 * math.sqrt(sum(r * r for r in res) / len(res))})
        return out
    if fit != "v030":
        raise ValueError(fit)
    for k in range(3):
        na, nb = 2 * k, 2 * k + 1
        c = 0.5 * (nodes[na] + nodes[nb])
        u = nodes[na] - nodes[nb]
        u /= np.linalg.norm(u)
        if geom.node_sign(na, end_map) < 0:
            u = -u
        e1 = np.cross(u, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(u, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        pairs = []
        for n in (na, nb):
            s = geom.node_sign(n, end_map)
            hk = feat["hooks"][s]
            pairs += [(hk["S_out"], sites["s_%d_%d" % (n, cab_from[n])]),
                      (hk["S_in"], sites["s_%d_%d" % (n, cab_to[n])])]
            if n == na:
                # the authors' odd-end sites are mirror images of their even-end pattern (I and L swapped, motor
                # site on the other side): only S_out / S_in of the odd end enter the fit
                pairs += [(hk["I"], sites["s_%d_b%d" % (n, cab_to[n])]),
                          (hk["L"], sites["s_%d_%d" % (n, longp[n])]),
                          (feat["guide"][s], sites["s_b%d_%d" % (n, cab_from[n])])]
        best = None
        for psi in np.radians(np.arange(0.0, 360.0, 0.25)):
            x = math.cos(psi) * e1 + math.sin(psi) * e2
            y = np.cross(u, x)
            R = np.stack([x, y, u], axis=1)
            err = 0.0
            for pl, qa in pairs:
                pw = R @ pl
                d = qa - c
                a1 = math.atan2(pw @ e2, pw @ e1)
                a2 = math.atan2(d @ e2, d @ e1)
                err += ((a1 - a2 + math.pi) % (2 * math.pi) - math.pi) ** 2
            if best is None or err < best[0]:
                best = (err, R, psi)
        rms = math.degrees(math.sqrt(best[0] / len(pairs)))
        out.append({"pos": c, "R": best[1], "roll_fit_rms_deg": rms})
    return out


def poses_from_nodes(nodes, feat=None, end_map="pz_even"):
    """Place the struts on given node centres (6x3, m): centre = node midpoint, axis = node direction, roll about
    the axis chosen so that the hooks face their partner nodes (least total hook-to-partner distance)."""
    feat = feat or geom.strut_features()
    nodes = np.asarray(nodes, float)
    partner = {}
    for i, (mot, line) in enumerate(geom.CABLES):
        partner.setdefault(mot, {})["S_out"] = line
        partner.setdefault(line, {})["S_in"] = mot
        partner[line]["I"] = mot
    for a, b in geom.LONG:
        partner.setdefault(a, {})["L"] = b
        partner.setdefault(b, {})["L"] = a
    out = []
    for k in range(3):
        na, nb = 2 * k, 2 * k + 1
        c = 0.5 * (nodes[na] + nodes[nb])
        u = nodes[na] - nodes[nb]
        u /= np.linalg.norm(u)
        if geom.node_sign(na, end_map) < 0:
            u = -u
        e1 = np.cross(u, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        best = None
        for psi in np.radians(np.arange(0.0, 360.0, 0.5)):
            x = math.cos(psi) * e1 + math.sin(psi) * e2
            R = np.stack([x, np.cross(u, x), u], axis=1)
            err = 0.0
            for n in (na, nb):
                sg = geom.node_sign(n, end_map)
                for role, q in partner[n].items():
                    err += float(np.linalg.norm(c + R @ feat["hooks"][sg][role] - nodes[q]))
            if best is None or err < best[0]:
                best = (err, R)
        out.append({"pos": c, "R": best[1], "roll_fit_rms_deg": float("nan")})
    return out


def _authors_sites_local(poses):
    """The authors' MJCF sites (node centres, role sites, motor sites) expressed in our strut frames
    (diagnostic attachment mode)."""
    nodes, sites = geom.authors_initial_nodes()
    cab_from = {m_: l for m_, l in geom.CABLES}
    cab_to = {l: m_ for m_, l in geom.CABLES}
    longp = {}
    for a, b in geom.LONG:
        longp[a], longp[b] = b, a
    out = {}
    for n in range(6):
        P = poses[geom.node_strut(n)]
        loc = lambda q: P["R"].T @ (q - P["pos"])
        out[n] = {"node": loc(nodes[n]), "S_out": loc(sites["s_%d_%d" % (n, cab_from[n])]),
                  "I": loc(sites["s_%d_b%d" % (n, cab_to[n])]), "S_in": loc(sites["s_%d_%d" % (n, cab_to[n])]),
                  "L": loc(sites["s_%d_%d" % (n, longp[n])]), "guide": loc(sites["s_b%d_%d" % (n, cab_from[n])])}
    return out


def build_xml(cfg: ModelConfig = None, poses=None):
    cfg = cfg or ModelConfig()
    c = cfg.resolved()
    geo = geom.load_geometry()
    feat = geom.strut_features(geo, c["hook_rule"], c["hook_assignment"])
    poses = poses or initial_pose(feat, c["end_map"])
    auth_local = _authors_sites_local(poses) if c["attachments"] == "authors" else None
    A = c["assets"]
    sub = "full" if c["mesh_res"] == "full" else "dec"
    zA = 0.5 * (geo["collision"]["encA_pz"]["z_range_m"][0] + geo["collision"]["encA_pz"]["z_range_m"][1])
    hA = 0.5 * (geo["collision"]["encA_pz"]["z_range_m"][1] - geo["collision"]["encA_pz"]["z_range_m"][0])
    rA = geo["collision"]["encA_pz"]["radius_m"]
    rS = geo["collision"]["sock_pz"]["radius_m"]
    zS = geo["collision"]["sock_pz"]["centre_m"][2]
    m = prm.value("strut_mass")
    It, Ia = prm.value("strut_inertia_transverse"), prm.value("strut_inertia_axial")
    L = []
    w = L.append
    w('<mujoco model="ysim_tribar">')
    w('<compiler angle="radian" autolimits="true" inertiafromgeom="false"/>')
    w('<option timestep="%.6g" gravity="0 0 %.6g" cone="elliptic" solver="Newton" integrator="%s" iterations="%d" '
      'tolerance="1e-10" ls_iterations="20"/>' % (c["timestep"], -c["gravity"], c["integrator"], c["iterations"]))
    w('<size memory="64M"/>')
    hl = ('<headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>' if c["lights"] else
          '<headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>')
    w('<visual><global offwidth="%d" offheight="%d" fovy="40"/><quality shadowsize="4096" offsamples="8"/>%s'
      '<map znear="0.005" zfar="50"/></visual>' % (tuple(c["offscreen"]) + (hl,)))
    w('<default>')
    w('<geom contype="1" conaffinity="1" condim="%d" friction="%.6g 0.005 0.0001" solref="%.6g %.6g"/>'
      % (prm.value("contact_condim"), c["friction"], -c["solref_k"], -c["solref_b"]))
    w('<default class="vis"><geom contype="0" conaffinity="0" group="1" density="0"/></default>')
    w('<default class="col"><geom group="3" rgba="0.9 0.6 0.1 0.35"/></default>')
    w('<site group="4" size="0.0015" rgba="1 1 0 1"/>')
    w('</default>')
    w('<asset>')
    vis_meshes = {}
    if c["visual"]:
        for g in ("sock", "endcap", "housing", "rod", "other"):
            if sub == "full":
                files = [f["file"] for f in geo["files"]["full_chunks"][g]]
            else:
                files = ["dec/%s.stl" % g]
            vis_meshes[g] = []
            for j, f in enumerate(files):
                nm = "vis_%s_%d" % (g, j)
                vis_meshes[g].append(nm)
                w('<mesh name="%s" file="%s" inertia="shell"/>' % (nm, os.path.join(A, f)))
    w(_hull_mesh_xml("col_elec"))      # 0.3.0: from the data table (identical compiled hull, no asset file)
    if c["cap_hulls"]:
        for t in ("pz", "nz"):
            w('<mesh name="col_cap_%s" file="%s" maxhullvert="64"/>' % (t, os.path.join(A, "col", "cap_hull_%s.stl" % t)))
    for col, rgba in SOCK_RGBA.items():
        w('<material name="sock_%s" rgba="%.3f %.3f %.3f %.3f" specular="0.15" shininess="0.3"/>' % ((col,) + rgba))
    w('<material name="endcap" rgba="0.82 0.82 0.80 1" specular="0.3" shininess="0.4"/>')
    w('<material name="housing" rgba="0.06 0.06 0.065 1" specular="0.25" shininess="0.35"/>')
    w('<material name="rod" rgba="0.75 0.76 0.78 1" specular="0.8" shininess="0.8" reflectance="0.1"/>')
    w('<material name="other" rgba="0.22 0.24 0.22 1" specular="0.3" shininess="0.3"/>')
    w('<material name="line" rgba="0.92 0.92 0.88 1"/>')
    if c["floor_style"] == "plain":
        w('<material name="floor" rgba="0.66 0.66 0.64 1" reflectance="0" specular="0.05" shininess="0.1"/>')
    else:
        w('<texture name="floortex" type="2d" builtin="checker" rgb1="0.58 0.58 0.56" rgb2="0.52 0.52 0.50" width="512" height="512"/>')
        w('<material name="floor" texture="floortex" texrepeat="10 10" texuniform="true" reflectance="0"/>')
    w(c["asset_xml"])
    w('</asset>')
    w('<worldbody>')
    if c["lights"]:
        w('<light name="key" pos="0.6 -0.8 2.5" dir="-0.25 0.35 -1" directional="true" diffuse="0.75 0.75 0.72" castshadow="true"/>')
        w('<light name="fill" pos="-1 1 2" dir="0.4 -0.4 -1" directional="true" diffuse="0.25 0.25 0.28" castshadow="false"/>')
    if c["floor"]:
        w('<geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="15" conaffinity="15"/>')
    w(c["world_xml"])
    for k, sname in enumerate(geom.STRUTS):
        P = poses[k]
        q = _quat_from_R(P["R"])
        col = geom.STRUT_COLOUR[sname]
        w('<body name="%s" pos="%.9g %.9g %.9g" quat="%.12g %.12g %.12g %.12g">' % ((sname,) + tuple(P["pos"]) + tuple(q)))
        w('<freejoint name="%s"/>' % sname)
        cbit = 1 if c["strut_collisions"] else (2 << k)
        w('<inertial pos="0 0 %.6g" mass="%.6g" diaginertia="%.9g %.9g %.9g"/>' % (prm.value("strut_com_offset"), m, It, It, Ia))
        if c["visual"]:
            for nm in vis_meshes["sock"]:
                w('<geom class="vis" type="mesh" mesh="%s" material="sock_%s"/>' % (nm, col))
            for g in ("endcap", "housing", "rod", "other"):
                for nm in vis_meshes[g]:
                    w('<geom class="vis" type="mesh" mesh="%s" material="%s"/>' % (nm, g))
        for n in (2 * k, 2 * k + 1):
            s = geom.node_sign(n, c["end_map"])
            w('<geom name="sock%d" class="col" type="sphere" size="%.6g" pos="0 0 %.9g" contype="%d" conaffinity="%d"/>' % (n, rS, s * abs(zS), cbit, cbit if c["strut_collisions"] else 1))
            w('<geom name="motor%d" class="col" type="cylinder" size="%.6g %.6g" pos="0 0 %.9g" contype="%d" conaffinity="%d"/>' % (n, rA, hA, s * abs(zA), cbit, cbit if c["strut_collisions"] else 1))
            if c["cap_hulls"]:
                w('<geom name="cap%d" class="col" type="mesh" mesh="col_cap_%s" contype="%d" conaffinity="%d"/>' % (n, "pz" if s > 0 else "nz", cbit, cbit if c["strut_collisions"] else 1))
            if auth_local is None:
                w('<site name="node%d" pos="0 0 %.9g"/>' % (n, s * prm.value("node_offset")))
            else:
                w('<site name="node%d" pos="%.9g %.9g %.9g"/>' % ((n,) + tuple(auth_local[n]["node"])))
            for r in geom.ROLE_NAMES:
                p = feat["hooks"][s][r] if auth_local is None else auth_local[n][r]
                w('<site name="hook%d_%s" pos="%.9g %.9g %.9g"/>' % ((n, r) + tuple(p)))
            g_ = feat["guide"][s] if auth_local is None else auth_local[n]["guide"]
            w('<site name="guide%d" pos="%.9g %.9g %.9g"/>' % ((n,) + tuple(g_)))
        w('<geom name="elec%d" class="col" type="mesh" mesh="col_elec" contype="%d" conaffinity="%d"/>' % (k, cbit, cbit if c["strut_collisions"] else 1))
        Ri = feat["imu_R"]
        qi = _quat_from_R(Ri)
        w('<site name="imu%d" pos="%.9g %.9g %.9g" quat="%.12g %.12g %.12g %.12g"/>' % ((k,) + tuple(feat["imu_pos"]) + tuple(qi)))
        w('</body>')
    w('</worldbody>')
    w('<tendon>')
    for name, kind, (na, sa), (nb, sb) in geom.tendon_table():
        if kind == "cable":
            # placeholder rest length 0.2 m; the Simulator sets every line's rest length and its material
            # stiffness EA / (L0 + line_inner_length) at construction and at every physics step
            L0 = 0.2
            k_, width, rgba = line_stiffness(L0, c["line_EA"], c["cable_k"]), 0.0006, "0.92 0.92 0.88 1"
            site_a, site_b = "guide%d" % na, "hook%d_%s" % (nb, sb)
        else:
            k_ = c["short_k"] if kind == "short" else c["long_k"]
            L0 = prm.value("short_sensor_rest") if kind == "short" else prm.value("long_sensor_rest")
            width = 0.0004
            rgba = ("0.95 0.6 0.1 0.6" if kind == "short" else "0.3 0.8 0.9 0.6") if c["show_sensors"] else "0 0 0 0"
            site_a, site_b = "hook%d_%s" % (na, sa), "hook%d_%s" % (nb, sb)
        w('<spatial name="%s" stiffness="%.9g" damping="%.9g" springlength="0 %.9g" width="%.6g" rgba="%s">'
          '<site site="%s"/><site site="%s"/></spatial>' % (name, k_, c["damping"], L0, width, rgba, site_a, site_b))
    w('</tendon>')
    w('<sensor>')
    for k in range(3):
        w('<accelerometer name="acc%d" site="imu%d"/><gyro name="gyr%d" site="imu%d"/>' % (k, k, k, k))
    w('</sensor>')
    w('</mujoco>')
    return "\n".join(L), {"poses": poses, "config": c}


class Ids:
    """Name -> id tables of a compiled ysim model."""

    def __init__(self, m):
        import mujoco
        n = lambda t, s: mujoco.mj_name2id(m, t, s)
        B, S, T, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_TENDON, mujoco.mjtObj.mjOBJ_GEOM
        self.body = [n(B, s) for s in geom.STRUTS]
        self.node_site = [n(S, "node%d" % i) for i in range(6)]
        self.cable = [n(T, "cable%d" % i) for i in range(6)]
        self.short = [n(T, "short%d" % i) for i in range(6)]
        self.long = [n(T, "long%d" % j) for j in range(3)]
        self.sensor_tendons = self.short + self.long
        self.imu_site = [n(S, "imu%d" % k) for k in range(3)]
        self.floor = n(G, "floor")
        # robot collision geoms only (stage 2 fix: world geoms from world_xml were included and shifted the
        # initial lift-to-floor; on the flat floor alone the set is unchanged)
        self.col_geoms = [i for i in range(m.ngeom) if m.geom_contype[i] and i != self.floor
                          and int(m.geom_bodyid[i]) in self.body]
        self.jnt_qpos = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.body]
        self.jnt_dof = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.body]
        import mujoco as mj
        self.acc = [m.sensor_adr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, "acc%d" % k)] for k in range(3)]
        self.gyr = [m.sensor_adr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, "gyr%d" % k)] for k in range(3)]
