"""The mechanism criterion of humanoid_leg_cad_design_from_photos (grade.py rules 9-11). The grader never runs FreeCAD's
solver: every quantity here is computed from the saved placements of the document and from the given parts' poses.

Pure functions on plain tables (numpy only), so the small cases of the rules can be fed to them directly. Units mm,
angles in degrees unless a name says rad; every point and axis is in the reference frame unless it is named local.

Ported pieces: the union-find of tasks/quadruped/quadruped_cad_design_from_photos/tests/quad_convert.py (_UF); the
band of tasks/space/freeflyer_cad_design_from_photos/tests/cadgrade/score.py; the forward kinematics of the foot
points in the order of tasks/quadruped/quadruped_cad_design_from_photos/tests/quad_grade.py (_foot_points: the
joints applied distal first); the closed-form inverse kinematics of the parallel ankle (an actuator turns a crank whose
ball drives a bar to a ball on the foot; the foot sits on two crossed axes, the second carried by the first) from the
numeric linkage model the leg's reference was measured with (circle-sphere intersection, the branch nearest the stored
pose).

Tables
  document   components [names]; joints [the extractor's records: name, label, type, suppressed, part1, part2, jcs1,
             jcs2, error]; holdings {component: {"tags": [...], "instances": [...]}} = the made parts assigned to the
             reference and the given instances assigned to reference placements, by the component that holds them;
             A = the alignment, document frame -> reference frame (4x4, rigid).
  reference  the "mechanism" block of reference.json (bodies, joints, poses, foot points, membership, chains, bars,
             ankle poses), validated by Reference._load_mechanism of grade.py.

The five sub-items (weights in grader_config.json mechanism.w):
  joints        M1. Rigid groups = union-find over the unsuppressed Fixed joints (an App::Part is one component, so one
                group); a group that holds no assigned made part and no assigned given instance is merged into its
                first neighbour by name (through any joint), repeatedly: joints on empty groups do not count. The
                group of each true rigid body is the group holding most of its tags (a tie: the larger sum of the
                reference volumes, then the first name); given parts do not decide groups. Each true joint is looked
                up between the groups of its two bodies over the unsuppressed Revolute joints only: same group or no
                Revolute path -> 0; one Revolute -> it; several in parallel on one step -> the worse one; n > 1 in
                series on the shortest path -> the best one divided by n; several shortest paths -> the worst path.
                A body whose tags are all missing: the joints on both sides of it are matched in order on the path
                between the two neighbouring bodies that have groups, times (joints / Revolute steps) when the path
                is longer. A Revolute is read in its frame on the side nearer the thigh (the true parent body's
                side); the two frames disagreeing by more than mechanism.frame_mismatch_note is counted. Score of a
                joint = band(distance from the true anchor to the axis line) x band(angle between the lines).
  foot_points   M2a. The foot's four sole corners in the reference, carried onto the submitted foot by a robust
                least-squares rigid fit of the foot's given instances (rod ends by their ball centres, the others by
                their centroids: a first fit, then the instances whose residual exceeds FOOT_FIT_DROP times the
                median residual are left out, at most down to FOOT_FIT_KEEP instances, the largest residuals first,
                and the fit is made once more), then turned through the 64 poses about the matched knee, pitch and
                roll axes of the document (each read with the sense of the true axis; an unmatched joint stays
                rigid), against the true corners: per pose the largest of the four errors; 0.5 band(median) + 0.5
                band(p90).
  membership    M2b. For each part of the five serial bodies (cranks, bars, rod ends and the shell excepted): the eight
                corners of the reference part's box through the 64 poses, once as its true body moves, once as the
                document's grouping moves it, both with the true axes (membership only, axis errors are M1's). A
                group that is the group of a true body moves as that body (the most proximal one when it is the group
                of several); any other group moves as the nearest such group through any joint, changed by the
                matched joints crossed on the way (crossing a matched joint from its parent side adds it, the other
                way removes it); a group joined to none of them does not move. Score = band(median over the poses of
                the largest corner error) x the share of matched joints among the true body's own joints.
  transmission  M3. Per chain: the actuator's directed output axis, the upper and lower ball centres, the pitch and roll
                axes through the two bore centres each; the bar length = the ball distance as saved; the crank angle
                change from the saved pose solved over the ankle poses; chain score = band(p90 of |submitted - true|);
                an unsolved pose counts transmission_unsolved_deg; a chain with a skeleton part missing scores 0.
  closure       M4. Per bar: | ball distance - (the bar's length along its own principal axis + 2 seat lengths) |.
Each item carries the floor segments it is multiplied with (combine).
"""
import math

import numpy as np

SPHERE_TOL = 1e-12
FOOT_FIT_DROP = 3.0       # M2a: an instance whose first-fit residual exceeds this times the median residual is left out
FOOT_FIT_KEEP = 6         # ... while at least this many instances stay in the second fit
FOOT_FIT_EPS_MM = 1e-9    # residuals within this of the cut are not above it (numerical noise of an exact document)


# ============================================================================ small helpers
def band(value, b):
    """1 when |value| <= b[0], 0 when |value| >= b[1], linear between; non-finite -> 0."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    full, zero = float(b[0]), float(b[1])
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if not (n > 1e-300) or not math.isfinite(n):
        return None
    return v / n


def rot(axis, ang):
    """Rotation matrix by `ang` rad about the unit vector `axis` (right hand)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def apply(T, P):
    T = np.asarray(T, float)
    P = np.asarray(P, float)
    if P.ndim == 1:
        return T[:3, :3] @ P + T[:3, 3]
    return P @ T[:3, :3].T + T[:3, 3]


def finite_matrix(m):
    try:
        a = np.asarray(m, float)
    except (TypeError, ValueError):
        return None
    if a.shape != (4, 4) or not np.all(np.isfinite(a)):
        return None
    return a


def frame_line(m):
    """(origin, unit z) of a 4x4 frame, or None when it is not finite or its z is degenerate."""
    a = finite_matrix(m)
    if a is None:
        return None
    z = unit(a[:3, 2])
    if z is None:
        return None
    return a[:3, 3].copy(), z


def line_distance(x, origin, z):
    v = np.asarray(x, float) - np.asarray(origin, float)
    return float(np.linalg.norm(v - z * float(v @ z)))


def line_angle_deg(a, b):
    """Angle between two lines (either sense), degrees."""
    c = abs(float(np.dot(a, b)))
    return math.degrees(math.acos(min(1.0, c)))


def kabsch(P, Q):
    """Proper rigid T minimising |T(P) - Q| over corresponding rows."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cq - R @ cp
    return T


def spread_ok(P, rel=1e-6):
    """True when the points are not collinear (a rigid fit onto them is unique)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return False
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return bool(s[0] > 1e-9 and s[1] > rel * s[0])


def rotate_about(P, point, d, ang):
    return (np.asarray(P, float) - point) @ rot(d, ang).T + point


def fk(P, axes, q):
    """Points P moved by the serial joints `axes` [(point, dir) or None] at angles q (rad): the distal joint first
    (product of exponentials about the axes at the saved pose); a None joint is rigid."""
    out = np.array(P, float)
    for ax, a in reversed(list(zip(axes, q))):
        if ax is None or a == 0.0:
            continue
        out = rotate_about(out, ax[0], ax[1], a)
    return out


class UF(object):
    """Union-find with a deterministic representative (the smaller name)."""

    def __init__(self, items=()):
        self.p = {}
        for x in items:
            self.find(x)

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


# ============================================================================ the document's rigid groups
class DocGraph(object):
    """Rigid groups of the document (union-find over unsuppressed Fixed joints, empty groups merged) and the joints
    between groups. A group is named by its smallest component name."""

    def __init__(self, components, joints, holdings, merge_empty=True, max_revolute=2000):
        self.components = sorted(set(str(c) for c in components))
        comps = set(self.components)
        self.notes = []
        self.stats = {"n_fixed": 0, "n_revolute": 0, "n_revolute_internal": 0, "n_other_joints": 0,
                      "n_joints_unreadable": 0, "n_empty_groups_merged": 0, "n_revolute_frames_one_side": 0}
        self.holdings = {c: {"tags": sorted(set((holdings.get(c) or {}).get("tags", []))),
                             "instances": sorted(set((holdings.get(c) or {}).get("instances", [])))} for c in self.components}
        uf = UF(self.components)
        usable = []
        for j in joints or []:
            if j.get("suppressed"):
                continue
            a, b = j.get("part1"), j.get("part2")
            if j.get("error") or a not in comps or b not in comps:
                self.stats["n_joints_unreadable"] += 1
                continue
            usable.append(j)
            if str(j.get("type")) == "Fixed" and a != b:
                uf.union(a, b)
                self.stats["n_fixed"] += 1
        self.uf = uf
        self.usable = usable
        if merge_empty:
            self._merge_empty()
        self._build()
        revs = [j for j in usable if str(j.get("type")) == "Revolute"]
        self.too_many = len(revs) > int(max_revolute)
        if self.too_many:
            self.notes.append("%d Revolute joints (more than %d): no joint is read" % (len(revs), int(max_revolute)))
            revs = []
        self.edges = []
        for j in sorted(revs, key=lambda x: (str(x.get("name")), str(x.get("label")))):
            g1, g2 = self.group_of(j["part1"]), self.group_of(j["part2"])
            if g1 == g2:
                self.stats["n_revolute_internal"] += 1
                continue
            f1, f2 = frame_line(j.get("jcs1")), frame_line(j.get("jcs2"))
            if f1 is None and f2 is None:
                self.stats["n_joints_unreadable"] += 1
                self.notes.append("Revolute %s: neither of its frames is finite: not read" % (j.get("label") or j.get("name")))
                continue
            if f1 is None or f2 is None:
                self.stats["n_revolute_frames_one_side"] += 1
            self.edges.append({"name": str(j.get("name")), "label": str(j.get("label") or j.get("name")), "g1": g1, "g2": g2,
                               "f1": f1 if f1 is not None else f2, "f2": f2 if f2 is not None else f1})
        self.stats["n_revolute"] = len(self.edges) + self.stats["n_revolute_internal"]
        self.stats["n_other_joints"] = sum(1 for j in usable if str(j.get("type")) not in ("Fixed", "Revolute"))
        self.rev_adj = {}
        for e in self.edges:
            self.rev_adj.setdefault(e["g1"], set()).add(e["g2"])
            self.rev_adj.setdefault(e["g2"], set()).add(e["g1"])
        self.stats["n_groups"] = len(self.groups)

    # ---------------------------------------------------------------- grouping
    def _build(self):
        groups = {}
        for c in self.components:
            groups.setdefault(self.uf.find(c), []).append(c)
        self.groups = {}
        self.name_of_root = {}
        for root, members in groups.items():
            name = min(members)
            self.name_of_root[root] = name
            self.groups[name] = sorted(members)
        self.content = {}
        for g, members in self.groups.items():
            tags, inst = set(), set()
            for c in members:
                tags |= set(self.holdings[c]["tags"])
                inst |= set(self.holdings[c]["instances"])
            self.content[g] = {"tags": sorted(tags), "instances": sorted(inst)}
        self.adj = {g: set() for g in self.groups}
        for j in self.usable:
            g1, g2 = self.group_of(j["part1"]), self.group_of(j["part2"])
            if g1 != g2:
                self.adj[g1].add(g2)
                self.adj[g2].add(g1)

    def _merge_empty(self):
        while True:
            self._build()
            merged = 0
            done = set()
            for g in sorted(self.groups):
                if g in done or self.content[g]["tags"] or self.content[g]["instances"]:
                    continue
                nb = sorted(self.adj[g])
                if not nb:
                    continue
                h = nb[0]
                self.uf.union(self.groups[g][0], self.groups[h][0])
                done.add(g)
                done.add(h)
                merged += 1
            self.stats["n_empty_groups_merged"] += merged
            if not merged:
                return

    def group_of(self, comp):
        if comp is None or comp not in self.uf.p:
            return None
        return self.name_of_root[self.uf.find(comp)]

    def between(self, u, v):
        return [e for e in self.edges if (e["g1"], e["g2"]) in ((u, v), (v, u))]

    def bfs(self, start, adj):
        dist = {start: 0}
        order = [start]
        k = 0
        while k < len(order):
            g = order[k]
            k += 1
            for h in sorted(adj.get(g, ())):
                if h not in dist:
                    dist[h] = dist[g] + 1
                    order.append(h)
        return dist


def body_groups(G, bodies, tag_group, tag_volume):
    """{body: group or None}: the group holding most of the body's tags (tie: larger reference volume, then name)."""
    out = {}
    for b in sorted(bodies):
        count, vol = {}, {}
        for t in bodies[b].get("tags", []):
            g = tag_group.get(t)
            if g is None:
                continue
            count[g] = count.get(g, 0) + 1
            vol[g] = vol.get(g, 0.0) + float(tag_volume.get(t, 0.0))
        if not count:
            out[b] = None
            continue
        out[b] = sorted(count, key=lambda g: (-count[g], -vol[g], g))[0]
    return out


# ============================================================================ M1: the four joints
def _edge_score(e, u, truth, A, cfg):
    """(score, record) of Revolute edge e read on the side of group u against a true joint."""
    f = e["f1"] if e["g1"] == u else e["f2"]
    o = apply(A, f[0])
    z = unit(np.asarray(A, float)[:3, :3] @ f[1])
    p, d = np.asarray(truth["point_mm"], float), unit(truth["dir"])
    off = line_distance(p, o, z)
    ang = line_angle_deg(z, d)
    s = band(off, cfg["joint_offset_mm"]) * band(ang, cfg["joint_angle_deg"])
    zz = z if float(z @ d) >= 0 else -z
    o1, z1 = e["f1"]
    o2, z2 = e["f2"]
    mism = (float(np.linalg.norm(o1 - o2)) > float(cfg["frame_mismatch_note"][0]) or
            line_angle_deg(z1, z2) > float(cfg["frame_mismatch_note"][1]))
    return s, {"edge": e["name"], "label": e["label"], "u": u, "v": e["g2"] if e["g1"] == u else e["g1"], "point": o,
               "dir": zz, "offset_mm": off, "angle_deg": ang, "score": s, "frame_mismatch": bool(mism)}


def _hop(G, u, v, truth, A, cfg):
    """The worst of the parallel Revolutes between u and v (u the side nearer the thigh)."""
    best = None
    for e in sorted(G.between(u, v), key=lambda x: x["name"]):
        s, rec = _edge_score(e, u, truth, A, cfg)
        rec["parallel"] = len(G.between(u, v))
        if best is None or s < best[0]:
            best = (s, rec)
    return best


def _shortest_dag(G, ga, gb):
    da = G.bfs(ga, G.rev_adj)
    if gb not in da:
        return None, None, None
    db = G.bfs(gb, G.rev_adj)
    n = da[gb]
    nodes = set(v for v in da if v in db and da[v] + db[v] == n)
    return n, da, nodes


def _run_paths(G, ga, gb, n, da, nodes, cap):
    """Shortest paths ga -> gb (lists of groups), depth first in name order, at most cap."""
    out = []

    def walk(path):
        if len(out) >= cap:
            return
        g = path[-1]
        if g == gb:
            out.append(list(path))
            return
        for h in sorted(G.rev_adj.get(g, ())):
            if h in nodes and da[h] == da[g] + 1:
                path.append(h)
                walk(path)
                path.pop()
    walk([ga])
    return out


def _single(G, ga, gb, n, da, nodes, truth, A, cfg):
    """k = 1: the minimum over the shortest paths of (the best step on the path) / n, a step being the worse of its
    parallel Revolutes."""
    order = sorted(nodes, key=lambda v: (da[v], v))
    val = {ga: (-1.0, None)}
    for v in order:
        if v == ga:
            continue
        best = None
        for u in sorted(G.rev_adj.get(v, ())):
            if u not in nodes or da.get(u) != da[v] - 1 or u not in val:
                continue
            hs, hrec = _hop(G, u, v, truth, A, cfg)
            cand = (hs, hrec) if hs > val[u][0] else val[u]
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            val[v] = best
    s, rec = val.get(gb, (0.0, None))
    return max(0.0, s) / float(n), rec


def _assign(S):
    """Order-preserving assignment of k joints to n steps maximising the sum of the scores, a match preferred to no
    match at equal sums (a Revolute on the path is matched even when its axis scores 0): (total, [step or None])."""
    k, n = len(S), len(S[0]) if S else 0
    eps = 1e-9
    F = np.zeros((k + 1, n + 1))
    for j in range(1, k + 1):
        for i in range(1, n + 1):
            F[j, i] = max(F[j - 1, i], F[j, i - 1], F[j - 1, i - 1] + S[j - 1][i - 1] + eps)
    pick = [None] * k
    j, i = k, n
    while j > 0 and i > 0:
        if F[j, i] == F[j - 1, i - 1] + S[j - 1][i - 1] + eps:
            pick[j - 1] = i - 1
            j, i = j - 1, i - 1
        elif F[j, i] == F[j - 1, i]:
            j -= 1
        else:
            i -= 1
    total = sum(S[jj][pick[jj]] for jj in range(k) if pick[jj] is not None)
    return float(total), pick


def lookup(G, groups_of_bodies, mech, A, cfg):
    """M1 per true joint: {name: {"score", "matched" (the Revolute read for it, or None), "n_steps", "factor",
    "why"}}, in the reference's joint order."""
    joints = mech["joints"]
    by_name = {j["name"]: j for j in joints}
    kids = {}
    for j in joints:
        kids.setdefault(j["parent"], []).append(j["name"])
    res = {j["name"]: {"score": 0.0, "matched": None, "n_steps": None, "factor": 1.0, "why": None} for j in joints}
    if G.too_many:
        for r in res.values():
            r["why"] = "too many Revolute joints"
        return res
    cap = int(cfg.get("max_paths", 256))
    done = set()
    for j in joints:
        if j["name"] in done:
            continue
        P = j["parent"]
        if groups_of_bodies.get(P) is None:
            done.add(j["name"])
            res[j["name"]]["why"] = "no group holds a tag of %s" % P
            continue
        seq, C = [j["name"]], j["child"]
        while groups_of_bodies.get(C) is None:
            ks = kids.get(C, [])
            if len(ks) != 1:
                C = None
                break
            seq.append(ks[0])
            C = by_name[ks[0]]["child"]
        done |= set(seq)
        if C is None:
            for nm in seq:
                res[nm]["why"] = "no group on the far side"
            continue
        ga, gb = groups_of_bodies[P], groups_of_bodies[C]
        if ga == gb:
            for nm in seq:
                res[nm]["why"] = "both sides in one rigid group"
            continue
        n, da, nodes = _shortest_dag(G, ga, gb)
        if n is None:
            for nm in seq:
                res[nm]["why"] = "no Revolute path between the two groups"
            continue
        k = len(seq)
        if k == 1:
            s, rec = _single(G, ga, gb, n, da, nodes, by_name[seq[0]], A, cfg)
            res[seq[0]].update(score=s, matched=rec, n_steps=n, factor=1.0 / n)
            continue
        paths = _run_paths(G, ga, gb, n, da, nodes, cap)
        worst = None
        for path in paths:
            steps = list(zip(path[:-1], path[1:]))
            S, R = [], []
            for nm in seq:
                row, rr = [], []
                for (u, v) in steps:
                    hs, hrec = _hop(G, u, v, by_name[nm], A, cfg)
                    row.append(hs)
                    rr.append(hrec)
                S.append(row)
                R.append(rr)
            total, pick = _assign(S)
            f = min(1.0, float(k) / float(n))
            if worst is None or total * f < worst[0]:
                worst = (total * f, pick, S, R, f)
        if worst is None:
            continue
        _t, pick, S, R, f = worst
        for jj, nm in enumerate(seq):
            if pick[jj] is None:
                res[nm].update(n_steps=n, factor=f, why="no Revolute left for it on the path")
                continue
            res[nm].update(score=S[jj][pick[jj]] * f, matched=R[jj][pick[jj]], n_steps=n, factor=f)
        if len(paths) >= cap:
            G.notes.append("more than %d shortest Revolute paths between two groups: the first %d read" % (cap, cap))
    return res


# ============================================================================ M2a: the foot points
def foot_points(look, mech, ref_pts, sub_pts, cfg):
    """(score, record). ref_pts / sub_pts: {placement id: point} of the foot's fitting instances (sub: only the
    assigned ones)."""
    names = [j["name"] for j in mech["joints"]]
    serial = list(mech["foot_joints"])
    order = [names.index(n) for n in serial]
    poses = np.radians(np.asarray(mech["poses"]["deg"], float))
    q = poses[:, order]
    P = np.asarray(mech["foot_points_mm"], float)
    truth_axes = [(np.asarray(mj["point_mm"], float), unit(mj["dir"])) for mj in mech["joints"]]
    tax = [truth_axes[i] for i in order]
    ids = [p for p in mech["foot_fit_instances"] if p in sub_pts]
    if len(ids) < 3 or not spread_ok([ref_pts[p] for p in ids]):
        return 0.0, {"why": "fewer than three foot instances to locate the foot (%d)" % len(ids), "n_fit": len(ids)}
    T, fit = robust_fit(ids, ref_pts, sub_pts)
    P0 = apply(T, P)
    sax = []
    for nm in serial:
        m = look[nm]["matched"]
        sax.append(None if m is None else (np.asarray(m["point"], float), np.asarray(m["dir"], float)))
    err = []
    for qq in q:
        a = fk(P0, sax, qq)
        b = fk(P, tax, qq)
        err.append(float(np.max(np.linalg.norm(a - b, axis=1))))
    err = np.asarray(err)
    med, p90 = float(np.median(err)), float(np.quantile(err, 0.9))
    s = 0.5 * band(med, cfg["foot_median_mm"]) + 0.5 * band(p90, cfg["foot_p90_mm"])
    return s, dict(fit, median_mm=med, p90_mm=p90, rigid=[nm for nm, ax in zip(serial, sax) if ax is None])


def robust_fit(ids, ref_pts, sub_pts):
    """(T, record): M2a's rigid fit of the reference points of the instances `ids` onto the submitted ones. A first
    least-squares fit over all of them; the instances whose residual exceeds FOOT_FIT_DROP x the median residual are
    left out (the largest residuals first, never below FOOT_FIT_KEEP instances; none when there are no more than
    FOOT_FIT_KEEP); with any left out, one more least-squares fit over the rest (kept only when the rest still locates
    the foot). Residual = the distance of a submitted point from its fitted reference point."""
    def fit_of(use):
        T = kabsch([ref_pts[p] for p in use], [sub_pts[p] for p in use])
        res = {p: float(np.linalg.norm(apply(T, ref_pts[p]) - sub_pts[p])) for p in ids}
        return T, res

    T, res0 = fit_of(ids)
    first_rms = float(np.sqrt(np.mean([res0[p] ** 2 for p in ids])))
    cut = FOOT_FIT_DROP * float(np.median([res0[p] for p in ids]))
    over = sorted((p for p in ids if res0[p] > cut + FOOT_FIT_EPS_MM), key=lambda p: (-res0[p], ids.index(p)))
    drop = over[:max(0, len(ids) - FOOT_FIT_KEEP)]
    kept = [p for p in ids if p not in drop]
    res = res0
    if drop and spread_ok([ref_pts[p] for p in kept]):
        T, res = fit_of(kept)
    else:
        drop, kept = [], list(ids)
    rms = float(np.sqrt(np.mean([res[p] ** 2 for p in kept])))
    return T, {"n_fit": len(kept), "fit_rms_mm": rms, "first_fit_rms_mm": first_rms, "dropped": list(drop),
               "dropped_first_residual_mm": [res0[p] for p in drop], "first_fit_cut_mm": cut}


# ============================================================================ M2b: membership
def true_sets(mech):
    """{body: set of joints between the root body and it}."""
    joints = mech["joints"]
    children = set(j["child"] for j in joints)
    roots = [b for b in mech["bodies"] if b not in children]
    S = {r: frozenset() for r in roots}
    changed = True
    while changed:
        changed = False
        for j in joints:
            if j["parent"] in S and j["child"] not in S:
                S[j["child"]] = S[j["parent"]] | {j["name"]}
                changed = True
    return S


def doc_sets(G, groups_of_bodies, look, mech):
    """{group: set of joints it moves with} under the document's grouping (module docstring, membership)."""
    Strue = true_sets(mech)
    bo = list(mech.get("body_order") or sorted(mech["bodies"]))
    order = sorted(mech["bodies"], key=lambda b: (len(Strue.get(b, ())), bo.index(b) if b in bo else len(bo), b))
    S = {}
    for b in order:
        g = groups_of_bodies.get(b)
        if g is not None and g not in S:
            S[g] = set(Strue.get(b, ()))
    body_groups_set = set(S)
    matched = {}
    for nm, r in look.items():
        m = r.get("matched")
        if m is not None:
            matched[(m["u"], m["v"])] = (nm, +1)
            matched[(m["v"], m["u"])] = (nm, -1)
    for g in sorted(G.groups):
        if g in S:
            continue
        # breadth first through any joint to the nearest body group
        prev = {g: None}
        queue = [g]
        found = None
        k = 0
        while k < len(queue) and found is None:
            x = queue[k]
            k += 1
            for h in sorted(G.adj.get(x, ())):
                if h in prev:
                    continue
                prev[h] = x
                if h in body_groups_set:
                    found = h
                    break
                queue.append(h)
        if found is None:
            S[g] = set()
            continue
        s = set(S[found])
        y = found
        while prev[y] is not None:            # walk from the body group back to g, crossing joints
            x = prev[y]
            hit = matched.get((y, x))
            if hit is not None:
                nm, sign = hit
                if sign > 0:
                    s.add(nm)
                else:
                    s.discard(nm)
            y = x
        S[g] = s
    return S


def membership(G, groups_of_bodies, look, mech, items, cfg):
    """(score, per item records). items: [{"id", "body", "group" (None when the part is missing), "corners" (8 x 3)}]."""
    names = [j["name"] for j in mech["joints"]]
    axes = [(np.asarray(j["point_mm"], float), unit(j["dir"])) for j in mech["joints"]]
    poses = np.radians(np.asarray(mech["poses"]["deg"], float))
    Strue = true_sets(mech)
    Sdoc = doc_sets(G, groups_of_bodies, look, mech)
    adj = {b: [j["name"] for j in mech["joints"] if b in (j["parent"], j["child"])] for b in mech["bodies"]}
    recs = []
    for it in items:
        rec = {"id": it["id"], "body": it["body"], "score": 0.0}
        b = it["body"]
        share = (sum(1 for nm in adj[b] if look[nm]["matched"] is not None) / float(len(adj[b]))) if adj[b] else 1.0
        rec["joint_share"] = share
        if it.get("group") is None:
            rec["why"] = "missing"
            recs.append(rec)
            continue
        st = Strue[b]
        sd = Sdoc.get(it["group"], set())
        C = np.asarray(it["corners"], float)
        err = []
        for qq in poses:
            qt = [a if n in st else 0.0 for n, a in zip(names, qq)]
            qd = [a if n in sd else 0.0 for n, a in zip(names, qq)]
            err.append(float(np.max(np.linalg.norm(fk(C, axes, qd) - fk(C, axes, qt), axis=1))))
        med = float(np.median(err))
        rec.update(median_mm=med, moves_with=sorted(sd), should_move_with=sorted(st),
                   score=band(med, cfg["membership_mm"]) * share)
        recs.append(rec)
    return (float(np.mean([r["score"] for r in recs])) if recs else 0.0), recs


# ============================================================================ M3: the transmission
class Chain(object):
    """One crank-bar chain of the parallel ankle, read from its skeleton points (reference frame). The bar length is
    the ball distance as saved, so the saved pose solves with a crank angle of 0."""

    def __init__(self, act_point, act_dir, up, lo, pitch_point, pitch_dir, roll_point, roll_dir):
        self.ok = False
        a = unit(act_dir)
        ap, ar = unit(pitch_dir), unit(roll_dir)
        if a is None or ap is None or ar is None:
            return
        up, lo = np.asarray(up, float), np.asarray(lo, float)
        p0 = np.asarray(act_point, float)
        self.C = p0 + float((up - p0) @ a) * a
        v = up - self.C
        self.r = float(np.linalg.norm(v))
        if not self.r > 1e-6:
            return
        self.a, self.e1 = a, v / self.r
        self.e2 = np.cross(a, self.e1)
        self.lo0 = lo
        self.L = float(np.linalg.norm(up - lo))
        self.Pp, self.ap = np.asarray(pitch_point, float), ap
        self.Pr, self.ar = np.asarray(roll_point, float), ar
        self.ok = all(np.all(np.isfinite(x)) for x in (self.C, self.e1, self.e2, self.Pp, self.Pr)) and math.isfinite(self.L)

    def lower(self, pitch, roll):
        Rp, Rr = rot(self.ap, pitch), rot(self.ar, roll)
        return self.Pp + Rp @ ((self.Pr - self.Pp) + Rr @ (self.lo0 - self.Pr))

    def ik(self, pitch, roll, ref=0.0):
        """Crank angle (rad) for a foot pose, the branch nearest ref; None when the bar cannot reach."""
        p = self.lower(pitch, roll) - self.C
        A_ = self.r * float(self.e1 @ p)
        B_ = self.r * float(self.e2 @ p)
        kk = (self.r ** 2 + float(p @ p) - self.L ** 2) / 2.0
        rho = math.hypot(A_, B_)
        if rho < 1e-12 or abs(kk / rho) > 1:
            return None
        base = math.atan2(B_, A_)
        d = math.acos(kk / rho)
        cands = [((c - ref + math.pi) % (2 * math.pi)) - math.pi + ref for c in (base + d, base - d)]
        return min(cands, key=lambda c: abs(c - ref))

    def dphi_deg(self, poses_rad):
        """Crank angle change from the saved pose (deg) per (pitch, roll); NaN where unsolved."""
        out = np.full(len(poses_rad), np.nan)
        if not self.ok:
            return out
        z = self.ik(0.0, 0.0)
        if z is None:
            return out
        for i, (p, r) in enumerate(poses_rad):
            v = self.ik(float(p), float(r))
            if v is not None:
                out[i] = math.degrees(v - z)
        return out


def axis_from_bores(b1, b2, ref_dir):
    """(midpoint, unit direction with the sense of ref_dir) of the line through two bore centres, or None."""
    b1, b2 = np.asarray(b1, float), np.asarray(b2, float)
    d = unit(b2 - b1)
    if d is None or float(np.linalg.norm(b2 - b1)) < 1e-6:
        return None
    if float(d @ np.asarray(ref_dir, float)) < 0:
        d = -d
    return 0.5 * (b1 + b2), d


def bore_lines(sk, mech, pitch_sense, roll_sense):
    """(pitch line, roll line) through the bore centres of sk, each (point, dir) with the given sense, or None."""
    pb, rb = mech["pitch_bores"], mech["roll_bores"]
    try:
        pit = axis_from_bores(sk["bore"][pb[0]], sk["bore"][pb[1]], pitch_sense)
        rol = axis_from_bores(sk["bore"][rb[0]], sk["bore"][rb[1]], roll_sense)
    except KeyError:
        return None, None
    return pit, rol


def chain_from(sk, ch, mech, pit, rol):
    """A Chain from skeleton points sk = {"act": {pid: (point, dir)}, "ball": {pid: point}, "bore": {pid: point}} and
    the two bore lines, or None when a part of it is missing."""
    c = mech["chains"][ch]
    try:
        ap, ad = sk["act"][c["actuator"]]
        up, lo = sk["ball"][c["upper"]], sk["ball"][c["lower"]]
    except KeyError:
        return None
    if pit is None or rol is None:
        return None
    return Chain(ap, ad, up, lo, pit[0], pit[1], rol[0], rol[1])


def reference_dphi(sk_ref, mech):
    """The true crank angle changes {chain: (25,) deg} and the true bore lines' senses (pitch, roll); raises
    ValueError when the reference's own skeleton does not solve (a broken task)."""
    pj = {j["name"]: j for j in mech["joints"]}
    pdir, rdir = unit(pj[mech["pitch_joint"]]["dir"]), unit(pj[mech["roll_joint"]]["dir"])
    tp, tr = bore_lines(sk_ref, mech, pdir, rdir)
    if tp is None or tr is None:
        raise ValueError("the reference's bore lines cannot be formed")
    poses = np.radians(np.asarray(mech["ankle_poses_deg"], float))
    truth = {}
    for ch in sorted(mech["chains"]):
        cr = chain_from(sk_ref, ch, mech, tp, tr)
        dt = cr.dphi_deg(poses) if (cr is not None and cr.ok) else None
        if dt is None or not np.all(np.isfinite(dt)):
            raise ValueError("the reference's chain %s does not solve over the ankle poses" % ch)
        truth[ch] = dt
    return truth, (tp[1], tr[1])


def transmission(sk_sub, truth, senses, mech, cfg):
    """(score, per chain records): truth and senses from reference_dphi."""
    poses = np.radians(np.asarray(mech["ankle_poses_deg"], float))
    cap = float(cfg["transmission_unsolved_deg"])
    pit, rol = bore_lines(sk_sub, mech, senses[0], senses[1])
    out, recs = [], {}
    for ch in sorted(mech["chains"]):
        cs = chain_from(sk_sub, ch, mech, pit, rol)
        if cs is None or not cs.ok:
            recs[ch] = {"score": 0.0, "why": "a part of the chain is missing" if cs is None else "degenerate chain"}
            out.append(0.0)
            continue
        ds = cs.dphi_deg(poses)
        d = np.abs(ds - truth[ch])
        uns = int(np.sum(~np.isfinite(d)))
        d = np.where(np.isfinite(d), d, cap)
        p90 = float(np.quantile(d, 0.9))
        s = band(p90, cfg["transmission_p90_deg"])
        recs[ch] = {"score": s, "p90_deg": p90, "max_deg": float(d.max()), "unsolved": uns, "crank_radius_mm": cs.r,
                    "bar_length_mm": cs.L}
        out.append(s)
    return float(np.mean(out)) if out else 0.0, recs


# ============================================================================ M4: closure
def closure(sk_sub, bar_len, mech, cfg):
    """(score, per bar records). bar_len: {tag: length along its principal axis or None}; mech["bars"] = {name:
    {"tag", "upper", "lower"}}."""
    seat = float(cfg["rod_end_seat_mm"])
    out, recs = [], {}
    for nm in sorted(mech["bars"]):
        c = mech["bars"][nm]
        L = bar_len.get(c["tag"])
        up, lo = sk_sub["ball"].get(c["upper"]), sk_sub["ball"].get(c["lower"])
        if L is None or up is None or lo is None:
            recs[c["tag"]] = {"score": 0.0, "why": "the bar or a ball of its chain is missing"}
            out.append(0.0)
            continue
        D = float(np.linalg.norm(np.asarray(up, float) - np.asarray(lo, float)))
        e = abs(D - (float(L) + 2.0 * seat))
        s = band(e, cfg["closure_mm"])
        recs[c["tag"]] = {"score": s, "error_mm": e, "ball_distance_mm": D, "bar_length_mm": float(L)}
        out.append(s)
    return float(np.mean(out)) if out else 0.0, recs


# ============================================================================ combination with the floors
def combine(items, floors, weights):
    """The mechanism criterion from its item lists {sub-item: [(score, [segments])]} and the segment floors
    {segment: floor}: per sub-item the mean of score x the mean floor of its segments, weighted over the sub-items
    the reference defines (the task's reference defines all five; grade.py refuses one that does not, unless it is
    marked as a partial test reference)."""
    tot = wsum = 0.0
    for k in sorted(weights):
        lst = items.get(k)
        if lst is None:
            continue
        w = float(weights[k])
        wsum += w
        if not lst:
            continue
        v = float(np.mean([s * float(np.mean([floors[g] for g in segs])) if segs else s for s, segs in lst]))
        tot += w * v
    return tot / wsum if wsum > 0 else 0.0
