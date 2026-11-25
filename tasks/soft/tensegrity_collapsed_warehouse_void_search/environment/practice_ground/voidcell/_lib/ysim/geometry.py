"""Strut geometry (measured on the authors' mesh) and the tendon net (the authors' MJCF topology).

Node numbering, strut numbering and tendon order follow the authors' MJCF
(the CableRobotGraphSim data set, crgs_3_bar_dataset/dataset_0/3bar_new_platform_all_cables.xml):
struts r01, r23, r45 carry nodes (0,1), (2,3), (4,5); actuated cables td_0..td_5 = motors 0..5 of the control code;
short sensors td_6..td_11 run parallel to cables 0..5; long passive sensors td_12..td_14.
Colours: red r01, green r23, blue r45 (control code comment, run_tensegrity_Astar.py lines 322-329, rss25).
"""
import json
import math

import numpy as np

from . import paths
from . import params as prm

STRUTS = ("r01", "r23", "r45")
STRUT_COLOUR = {"r01": "red", "r23": "green", "r45": "blue"}

# actuated cables: (motor node, line end node) ; authors' td_i sites s_<line>_b<motor> -> s_b<motor>_<line>
CABLES = ((5, 3), (3, 1), (1, 5), (2, 0), (0, 4), (4, 2))
# short sensors parallel to cable i: node pair = same as the cable; sensor i <-> td_{6+i}
# long passive sensors td_12..td_14
LONG = ((2, 5), (0, 3), (1, 4))
ROLE_NAMES = ("S_out", "I", "L", "S_in")


def node_strut(n):
    return n // 2


def node_sign(n, end_map="pz_even"):
    """+1: node at the mesh +z end of its strut, -1: at the -z end. end_map "pz_even" (default, params
    mesh_end_to_node, with evidence: the Arduino's micro-USB socket is at the mesh +z end and the odd nodes sit at
    the end far from the USB socket): even nodes at +z; "nz_even": the other assembly (diagnostic only)."""
    s = 1 if n % 2 == 0 else -1
    return s if end_map == "pz_even" else -s


def load_geometry():
    with open(paths.data_path("strut_geometry.json")) as fh:
        return json.load(fh)


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- hook roles (ysim 0.3.1: ASSUMPTION params.hook_roles)
# The assembly guide (PDF p26-27, printed p24-25) leaves the hooks free: 'It does not matter which of the hooks on the
# end caps the fishing line is connected to'; 'The long sensors must be connected to the opposite hook of the fishing
# line'; 'The short sensors must be connected in parallel to the fishing lines' (no hook named). So at every end the
# incoming line I takes any of the 4 hooks, the long sensor L the hook opposite it, and the two short sensors (S_out,
# parallel to this end's own outgoing line; S_in, parallel to the incoming line) the other two in either order:
# 8 assignments per end (allowed_end_assignments), 64 per strut type.
HOOK_RULES = ("authors_handedness", "guide_side_v030")
DEFAULT_HOOK_RULE = "authors_handedness"


def _hook_az(geo, tag):
    """The four hooks of one end sorted by azimuth about the strut axis (deg, from the centroids)."""
    hs = geo["features"]["hooks_" + tag]
    az = [math.degrees(math.atan2(h["centroid_mm"][1], h["centroid_mm"][0])) for h in hs]
    order = sorted(range(len(hs)), key=lambda j: az[j])
    return [hs[j] for j in order], [az[j] for j in order]


def allowed_end_assignments(geo=None, s=1):
    """The 8 role -> hook-index assignments the guide allows at one end (s = +1 mesh +z / even node, -1 mesh -z / odd
    node); hook indices in azimuth order (_hook_az). Order: I = hook 0..3, then the two orders of the short sensors."""
    geo = geo or load_geometry()
    _, a = _hook_az(geo, "pz" if s > 0 else "nz")
    out = []
    for i in range(4):
        opp = min(range(4), key=lambda j: abs(abs(_wrap(a[j] - a[i])) - 180.0))
        rest = [j for j in range(4) if j not in (i, opp)]
        for so, si in (rest, rest[::-1]):
            out.append({"I": i, "L": opp, "S_out": so, "S_in": si})
    return out


def hook_roles(geo=None, rule=None, assignment=None):
    """For each end (+1, -1): dict role -> hook position (m, strut frame), guide position and offset.

    rule "authors_handedness" (default, ysim 0.3.1; params hook_roles): at every end S_out = the hook nearest (in
    azimuth) the end's line guide, and about the strut's +z axis (from the odd to the even node) the roles follow
    S_out, L, S_in, I at +90 deg steps, the cyclic order of the authors' sites at all six ends of their model
    (authors_model.json). At the +z (even) ends this is the 0.3.0 assignment; at the -z (odd) ends I and L are
    exchanged against 0.3.0 (the real strut's ends are two-fold symmetric, so 0.3.0's guide-relative rule produced the
    mirror image of the authors' arrangement there).
    rule "guide_side_v030" (diagnostic): ysim 0.3.0, S_out nearest the guide, I = its neighbour on the guide side.
    assignment (diagnostic, overrides the rule): (even index, odd index) into allowed_end_assignments."""
    geo = geo or load_geometry()
    rule = rule or DEFAULT_HOOK_RULE
    if assignment is not None:
        out = {}
        for s, tag, idx in ((1, "pz", assignment[0]), (-1, "nz", assignment[1])):
            hooks, a = _hook_az(geo, tag)
            ch = allowed_end_assignments(geo, s)[int(idx)]
            g = np.array(geo["features"]["guide_" + tag]["centre_mm"])
            ga = math.degrees(math.atan2(g[1], g[0]))
            out[s] = {"roles": {r: np.array(hooks[ch[r]]["centroid_mm"]) / 1000.0 for r in ROLE_NAMES},
                      "guide": g / 1000.0, "guide_offset_deg": float(_wrap(ga - a[ch["S_out"]]))}
        return out
    if rule == "authors_handedness":
        out = {}
        for s, tag in ((1, "pz"), (-1, "nz")):
            hooks, a = _hook_az(geo, tag)
            g = np.array(geo["features"]["guide_" + tag]["centre_mm"])
            ga = math.degrees(math.atan2(g[1], g[0]))
            k = int(np.argmin([abs(_wrap(ga - x)) for x in a]))
            want = {"S_out": a[k], "L": a[k] + 90.0, "S_in": a[k] + 180.0, "I": a[k] + 270.0}
            roles = {}
            for r, w in want.items():
                j = int(np.argmin([abs(_wrap(w - b)) for b in a]))
                assert abs(_wrap(w - a[j])) < 5.0, "hook pattern not 90-degree regular"
                roles[r] = np.array(hooks[j]["centroid_mm"]) / 1000.0
            assert len({tuple(v) for v in roles.values()}) == 4
            out[s] = {"roles": roles, "guide": g / 1000.0, "guide_offset_deg": float(_wrap(ga - a[k]))}
        return out
    if rule != "guide_side_v030":
        raise ValueError("hook rule %r not in %s" % (rule, HOOK_RULES))
    out = {}
    for s, tag in ((1, "pz"), (-1, "nz")):
        hooks = geo["features"]["hooks_" + tag]
        g = np.array(geo["features"]["guide_" + tag]["centre_mm"])
        ga = math.degrees(math.atan2(g[1], g[0]))
        outward = [s * h["angle_deg"] for h in hooks]          # angle seen from outside the end
        g_out = s * ga
        diffs = [_wrap(g_out - a) for a in outward]
        k = int(np.argmin(np.abs(diffs)))
        side = 1.0 if diffs[k] > 0 else -1.0                  # guide side
        a0 = outward[k]
        want = {"S_out": a0, "I": a0 + side * 90.0, "L": a0 - side * 90.0, "S_in": a0 + 180.0}
        roles = {}
        for r, a in want.items():
            j = int(np.argmin([abs(_wrap(a - b)) for b in outward]))
            assert abs(_wrap(a - outward[j])) < 5.0, "hook pattern not 90-degree regular"
            roles[r] = np.array(hooks[j]["centroid_mm"]) / 1000.0
        assert len({tuple(v) for v in roles.values()}) == 4
        out[s] = {"roles": roles, "guide": g / 1000.0, "guide_offset_deg": float(diffs[k])}
    return out


def strut_features(geo=None, hook_rule=None, hook_assignment=None):
    """Feature points of one strut in its body frame (m); hook roles by hook_roles(rule, assignment)."""
    geo = geo or load_geometry()
    hr = hook_roles(geo, hook_rule, hook_assignment)
    z = prm.value("node_offset")
    ard = geo["features"]["arduino_board"]
    x_imu = np.array(ard["long_axis"]); x_imu /= np.linalg.norm(x_imu)
    if x_imu[2] < 0:
        x_imu = -x_imu
    z_imu = np.array(ard["normal"]); z_imu -= x_imu * (z_imu @ x_imu); z_imu /= np.linalg.norm(z_imu)
    y_imu = np.cross(z_imu, x_imu)
    return {"hook_rule": "assignment %s" % (tuple(hook_assignment),) if hook_assignment is not None else
            (hook_rule or DEFAULT_HOOK_RULE),
            "node": {1: np.array([0.0, 0.0, z]), -1: np.array([0.0, 0.0, -z])},
            "hooks": {s: hr[s]["roles"] for s in (1, -1)},
            "guide": {s: hr[s]["guide"] for s in (1, -1)},
            "guide_offset_deg": {s: hr[s]["guide_offset_deg"] for s in (1, -1)},
            "imu_pos": np.array(ard["centroid_mm"]) / 1000.0,
            "imu_R": np.stack([x_imu, y_imu, z_imu], axis=1)}   # columns = imu axes in strut frame


def tendon_table():
    """List of tendons: (name, kind, (strut_a, site_a), (strut_b, site_b)). Site names are local feature keys."""
    rows = []
    for i, (mot, line) in enumerate(CABLES):
        rows.append(("cable%d" % i, "cable", (mot, "guide"), (line, "I")))
    for i, (mot, line) in enumerate(CABLES):
        rows.append(("short%d" % i, "short", (mot, "S_out"), (line, "S_in")))
    for j, (a, b) in enumerate(LONG):
        rows.append(("long%d" % j, "long", (a, "L"), (b, "L")))
    return rows


def _authors_role_site(n, role):
    """The authors' MJCF site name of a role at node n (S_out, S_in, I, L, guide)."""
    cf = {m: l for m, l in CABLES}
    ct = {l: m for m, l in CABLES}
    lp = {a: b for a, b in LONG}
    lp.update({b: a for a, b in LONG})
    return {"S_out": "s_%d_%d" % (n, cf[n]), "S_in": "s_%d_%d" % (n, ct[n]), "I": "s_%d_b%d" % (n, ct[n]),
            "L": "s_%d_%d" % (n, lp[n]), "guide": "s_b%d_%d" % (n, cf[n])}[role]


def assignment_distance(assignment=None, rule=None, geo=None):
    """How close a hook assignment's attachment sites lie to the authors' simulation model (params hook_roles rule).

    Every mesh strut is placed on the authors' node axis of their initial pose (centre = node midpoint, +z = odd ->
    even node). M_strut: RMS 3D distance of its 10 attachment sites (S_out, S_in, I, L, line guide at both ends) to
    the authors' sites of the same roles with ONE roll per strut (closed-form least squares; absolute geometry, both
    ends). M_end: the same with a roll per end (each end's arrangement in the strut frame; rigid, no mirror image).
    M = sqrt((M_strut^2 + M_end^2) / 2). RMS over the three struts. Returns metres."""
    geo = geo or load_geometry()
    hr = hook_roles(geo, rule, assignment)
    d = load_authors_model()
    nodes = {int(k): np.array(v) for k, v in d["nodes_m"].items()}
    sites = {k: np.array(v) for k, v in d["sites_m"].items()}
    roles = ROLE_NAMES + ("guide",)

    def fit(P, Q):
        sxx = float(np.sum(P[:, 0] * Q[:, 0] + P[:, 1] * Q[:, 1]))
        sxy = float(np.sum(P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]))
        psi = math.atan2(sxy, sxx)
        c, s_ = math.cos(psi), math.sin(psi)
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return np.linalg.norm(P @ R.T - Q, axis=1)

    d_strut, d_end = [], []
    for k in range(3):
        ne, no = 2 * k, 2 * k + 1
        c = 0.5 * (nodes[ne] + nodes[no])
        u = nodes[ne] - nodes[no]
        u /= np.linalg.norm(u)
        e1 = np.cross(u, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        B = np.stack([e1, np.cross(u, e1), u], 0)
        P, Q = {}, {}
        for n, s in ((ne, 1), (no, -1)):
            P[s] = np.array([hr[s]["guide"] if r == "guide" else hr[s]["roles"][r] for r in roles])
            Q[s] = np.array([B @ (sites[_authors_role_site(n, r)] - c) for r in roles])
        d_strut += list(fit(np.vstack([P[1], P[-1]]), np.vstack([Q[1], Q[-1]])))
        for s in (1, -1):
            d_end += list(fit(P[s], Q[s]))
    ms = float(np.sqrt(np.mean(np.square(d_strut))))
    me = float(np.sqrt(np.mean(np.square(d_end))))
    return {"M": math.sqrt(0.5 * (ms * ms + me * me)), "M_strut": ms, "M_end": me}


def load_authors_model():
    """The data copied from the authors' MJCF (notes/sim_base_data/authors_model.json: node centres and role sites
    of the authors' initial pose in metres, tendon topology; source path, sha256 and extraction script recorded in
    the file). ysim 0.3.0 reads this table instead of the scratch source."""
    with open(paths.data_path("authors_model.json")) as fh:
        return json.load(fh)


def authors_initial_nodes():
    """Node positions (m, world) of the authors' MJCF initial pose and all their rod-body sites (m, world)."""
    d = load_authors_model()
    nodes = np.array([d["nodes_m"][str(i)] for i in range(6)], float)
    sites = {k: np.array(v, float) for k, v in d["sites_m"].items()}
    return nodes, sites
