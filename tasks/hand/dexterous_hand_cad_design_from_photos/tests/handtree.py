"""handtree: the mechanism module of dexterous_hand_cad_design_from_photos (the walk, the kinematic, mass, fit and
interference criteria per digit, the digit floor F_k). grade.py calls it after the rigid items, with the
loaded reference, the rigid groups, the palm groups, the alignment and the part and instance assignments in hand:

    crit = handtree.mechanism(grading)      {"kinematic", "mass", "fit", "interference"}, each in [0, 1], the digit
                                            floors applied (grade.py applies the shape floor F_s on top)

The hidden truth is the reference's mechanism block (hand_task.HandSpec). Every number is in grader_config.json
(blocks mechanism, mass, fit, interference; floors.fk). Everything positional is read in the aligned frame (the palm's
frame of the reference at configuration Z); lengths mm, angles rad unless named _deg.

The walk (one wrong or missing joint costs the hinge it belongs to and what cannot be reached without it):
 1. parts and given instances, rigid groups, palm groups, alignment: grade.py.
 2. hinge edges: every unsuppressed Revolute and Cylindrical joint is an edge between the groups of its two parts on its
    axis line (both frames' mean; frames more than frames_apart apart are reported); a Ball joint an edge with a centre
    and no axis; other joint types are ignored (the extractor lists them); an edge inside one group is ignored.
    Several edges between two groups (the palm groups count as one) are one edge on their mean axis when every two of
    them agree within coaxial_mm / coaxial_deg; else they disagree: the hinge they declare is read as missing (held
    rigid, its own items 0) and reported (hinges_disagreeing, edges_disagreeing), and the edge still joins the two
    groups (a digit candidate, the order of its links, a rigid step of the posing). A pass-through group (no structural,
    palm or servo part; joined to exactly two other groups by edges that agree) is taken out and its two edges become
    one. A group carries a recognised dowel pin when the pin is rigid in it (and the group holds more than the pin), or
    when a hole of one of its made parts or one of its recognised bearings lies on the pin's axis (within coaxial_mm /
    coaxial_deg, and along the pin within pin_along_slack_mm); two carriers that no declared edge joins get a pin edge
    on the pin's axis (pin_marked). A pin edge never is a first hinge and never stands for a servo hinge.
 3. digit candidates: the palm groups removed, each connected component of the edge graph that a Revolute or
    Cylindrical edge joins to a palm group; that edge is its first hinge; of several, the one whose group holds a
    structural part of the earliest chain link, ties by document order (first_hinge_surplus). Components without one
    float.
 4. slots: candidates to the reference's digits one to one (Hungarian) by d + slot_mm_per_deg x a, d the distance of a
    servo hinge centre of the digit from the candidate's first-hinge axis line, a the sense-free angle between the two,
    the smallest over the digit's servo hinges (the first one when the digit's earlier links sit in the palm), plus a
    tie-break of slot_tiebreak_per_mm per mm between the centre and the edge's frame point; admissible only within
    slot_max_mm and slot_max_deg.
 5. links: each chain link (and the link bar) is matched to the group among the candidate's and the palm's that holds
    most of its structural parts: a strict majority by count first; a tied link takes the tied group that no other
    link is matched to (links_tied); the reference volume decides only after that; the palm groups count as one;
    consecutive links
    in one group form a run; a link none of whose structural parts is there is absent; a structural part elsewhere is
    reported (membership).
 6. hinges: a hinge of the unit between two matched links is present when an edge joins their groups; a servo hinge
    needs a Revolute or Cylindrical edge, a pin hinge may be read from a declared edge, a Ball (its centre, the axis
    parallel to the finger's third hinge) or a pin edge; missing when both links are in one run, one of them is absent,
    or no such edge exists. Edges that match no hinge are reported (extra_hinge).
 7. driver, sense, zero, limits: the driver of a servo hinge is the recognised servo rigidly in one of its two groups
    (the palm's groups for a first hinge) whose output axis lies within driver_deg and driver_mm of the joint axis, the
    nearest to the joint frame; positive sense = the servo's +z when it is on the palm side, -z on the tip side
    (case_side reported where the unit has it on the other side); no driver: undriven; more than consistent_deg /
    consistent_mm off: inconsistent (reported; read on the servo's axis, as every driven hinge is). Zero is the saved
    pose; the declared interval is the joint's AngleMin / AngleMax relative to its saved angle, in the hinge's sense
    (child-first references and flipped frames give the same result; a full turn or more is the full turn).
 8. the loop of a finger is closed when the link bar is in a group of its own, its four hinges are present and their
    axes are parallel within loop_parallel_deg; the grader solves it in the plane normal to the third hinge's axis
    through the four axis points at the saved pose (circle-circle, the saved branch followed in steps of at most
    loop_step_deg); else it is open and the distal hinge is held rigid. The reference is solved the same way.
 9. posing, tip, validity: missing and undriven hinges and the distal hinge of an open loop are held rigid; the tip
    rides on the group of the digit's tactile module (recognised, in the candidate; of several, the one rigid in the
    group matched to the digit's last link, else the one farthest along the chain, ties by document order:
    tip_modules_surplus); a digit is
    invalid when one group is matched to non-consecutive chain links or the edges reach a later chain link's group
    before an earlier one's (hinge counts from the palm among the chain's groups), the order read on the matched
    links: a later link's group that holds a structural part of the earlier link (membership) does not count against
    the order.

Criteria (each digit's items; K = the mean over the reference's digits; an invalid digit and a digit
without a candidate score 0 on kinematic, mass, fit and interference):
  kinematic  axes (every hinge: band(true centre to the axis line) x band(angle), sense-free), key lengths (common
             normal distance and angle of consecutive chain hinges, the loop's ground, coupler arm and link bar, the tip's
             distance from the last hinge), fingertip over poses (poses_per_digit configurations of the independent
             hinges inside the true ranges, a scrambled Halton stream from the per-run seed, kept where the true digit is
             collision-free; the tip error in the aligned frame; a pose that the submitted loop cannot close counts
             unclosed_error_mm), coupling law (coupling_angles angles of the third hinge across its true range, the
             submitted loop turned in the unit's flexion sense, the mean |distal angle error|, an unclosed angle at the
             band's zero end), range (overlap over union of the declared and the true interval). A hinge the reference
             marks located false: its axis item, every key length it is an end of and its range item are not graded
             (out of their means; a component left with no item drops out with its weight, the others' weights
             renormalised); counted in hinge_items_not_graded (from the reference) and hinges_not_located.
  mass       per link (palm_weight / link_weight): w_mass band(mass error %) + w_com band(centre of mass), on the group
             matched to the link (the palm: every palm group; a run against the sum of its links; absent 0); made
             volume x density + data-page masses + extra solids at extra_density_g_mm3; dowel pins and bearings are left
             out on both sides (the grader accepts them in either of their links or in none).
  fit        w_given: each given instance band(median gap over the patches where it touches located surface of made
             parts in the unit)
             x band(share of its volume inside made parts and extra solids), by type weight; w_made: each made part
             band(share of its volume inside other made parts, given instances and extra solids); the unit's own
             overlaps exempt pair by pair.
  interference  w_within: the share of the kept poses at which the submitted digit, posed by its own hinges, has no pair
             of its groups (or a group and the palm) whose overlap grows beyond adjacent_ratio (groups joined by an edge)
             or other_ratio of the smaller one's volume over its reading at the saved pose; w_between: the same between
             the digits' groups at the poses where the true hand is collision-free as a whole.
  floors     F_k = band(median fingertip error; floors.fk) (0 for an invalid digit or one without a tip); the mass, fit
             and interference items of a digit times its F_k, the palm's items times the mean F_k.
"""
import json
import math
import os
import time

import numpy as np

import grade as G
import hand_task as HT

PALM = "@palm"

# every count of the walk, written into every verdict (0 when nothing fired), so that the metric names do not depend on
# the document
WALK_COUNTS = ("ball_joints", "case_side", "cylindrical_as_hinge", "digit_candidates", "digits_invalid",
               "edges_disagreeing", "edges_in_group", "extra_hinge", "first_hinge_surplus", "floating_groups",
               "frames_apart", "ground_not_on_palm", "hinges_disagreeing", "hinges_from_pins", "inconsistent",
               "joint_types_other", "joints_unread", "links_absent", "links_tied", "loop_open", "loop_skew",
               "membership", "no_tip", "pass_through_groups", "pin_marked", "servo_unfixed", "tip_link",
               "tip_modules_surplus", "unassigned_candidates", "undriven")
MECH_STATS = ("collision_booleans_submission", "collision_booleans_true", "digit_floor_mean", "fit_given", "fit_made",
              "hinge_items_not_graded", "hinges_not_located", "pose_seed_hi", "pose_seed_lo", "pose_seed_pinned",
              "true_poses_kept_between")
DIGIT_STATS = ("digit_assigned_%s", "digit_fk_%s", "digit_valid_%s", "kinematic_%s", "loop_closed_%s",
               "true_poses_kept_%s")
DIGIT_UNMEASURED = ("fingertip_median_mm_%s", "fingertip_p90_mm_%s")      # -1 where the digit is not graded
HINGE_STATUSES = ("present", "missing", "undriven")


def unlocated_items(spec):
    """(hinges the reference marks not located, kinematic items they take out of the grade: per such hinge its axis
    item, its range item when it is independent, two items per consecutive chain pair it is in, one per loop length and
    one for the tip's distance from the last chain hinge when it is that hinge)."""
    out = [hid for hid, h in spec.hinges.items() if not h.get("located", True)]
    n = 0
    for d in spec.order:
        D = spec.digits[d]
        loc = lambda hid: spec.hinges[hid].get("located", True)  # noqa: E731
        n += sum(1 for hid in D["hinges"] if not loc(hid))
        n += sum(1 for hid in D["independent"] if not loc(hid))
        chain = spec.chain_hinges(d)
        n += sum(2 for a, b in zip(chain[:-1], chain[1:]) if not (loc(a) and loc(b)))
        if D["loop"] is not None:
            lp = D["loop"]
            n += sum(1 for a, b in ((lp["driver"], lp["base_pivot"]), (lp["distal"], lp["link_pin"]),
                                    (lp["base_pivot"], lp["link_pin"])) if not (loc(a) and loc(b)))
        n += 0 if loc(chain[-1]) else 1
    return len(out), n


def metric_defaults(spec):
    """{metric name: value when nothing is read} of every statistic this module writes, for a HandSpec."""
    out = {k: 0.0 for k in WALK_COUNTS + MECH_STATS}
    for d in spec.order:
        for f in DIGIT_STATS:
            out[f % d] = 0.0
        for f in DIGIT_UNMEASURED:
            out[f % d] = -1.0
        for hid in spec.digits[d]["hinges"]:
            for s in HINGE_STATUSES:
                out["hinge_%s_%s" % (hid.replace(".", "_"), s)] = 0.0
    return out


# ============================================================================ small geometry
def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def line_angle_deg(a, b):
    """Sense-free angle between two directions, degrees in [0, 90]."""
    c = abs(float(np.dot(unit(a), unit(b))))
    return math.degrees(math.acos(min(1.0, c)))


def point_line(p, a, d):
    v = np.asarray(p, float) - np.asarray(a, float)
    d = unit(d)
    return float(np.linalg.norm(v - d * float(v @ d)))


def common_normal(p1, d1, p2, d2, parallel_deg=5.0):
    """The distance between two hinge axes: the length of their common normal; for axes within parallel_deg of
    parallel (mechanism.parallel_deg), the mean of each axis point's distance from the other axis (the common normal of
    two nearly parallel lines lies far from the hinges and says nothing about them)."""
    d1, d2 = unit(d1), unit(d2)
    if line_angle_deg(d1, d2) < float(parallel_deg):
        return 0.5 * (point_line(p2, p1, d1) + point_line(p1, p2, d2))
    c = np.cross(d1, d2)
    s = float(np.linalg.norm(c))
    return abs(float((np.asarray(p2, float) - np.asarray(p1, float)) @ c)) / s


def rot_about(point, axis, ang):
    """4 x 4 rotation by ang (rad, right hand about axis) about the line through point."""
    T = np.eye(4)
    if ang == 0.0:
        return T
    R = G.axis_angle_R(axis, ang)
    p = np.asarray(point, float)
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T


def plane_point(p, d, origin, n):
    """Where the line (p, d) crosses the plane through origin normal n (the projection of p for a line along n)."""
    p, d, n = np.asarray(p, float), unit(d), unit(n)
    den = float(d @ n)
    if abs(den) < 1e-9:
        return p - n * float((p - origin) @ n)
    return p + d * float((np.asarray(origin, float) - p) @ n) / den


def interval_iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# ============================================================================ the loop solver
class LoopError(ValueError):
    """The four-bar cannot close at this angle (beyond its toggle)."""


class PlanarLoop(object):
    """The planar four-bar of a finger: ground B (third hinge O2, base pivot O4), crank C (O2 to the distal hinge A),
    coupler D (A to the link pin Bp), rocker E (O4 to Bp), in the plane normal to n (the third hinge's positive axis)
    through O2. solve(q) -> (phi = D relative to C, chi = E relative to B) at the third-hinge angle q (C relative to B),
    on the branch of the saved pose followed from q = 0 in steps of at most step_rad."""

    def __init__(self, n, O2, A, O4, Bp, step_rad):
        self.n = unit(n)
        ref = np.array([1.0, 0.0, 0.0]) if abs(self.n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.e1 = unit(ref - self.n * float(ref @ self.n))
        self.e2 = np.cross(self.n, self.e1)
        self.origin = np.asarray(O2, float)
        pr = lambda p: np.array([float((p - self.origin) @ self.e1), float((p - self.origin) @ self.e2)])  # noqa: E731
        self.o2, self.a0, self.o4, self.b0 = (pr(np.asarray(x, float)) for x in (O2, A, O4, Bp))
        self.crank = float(np.linalg.norm(self.a0 - self.o2))
        self.coupler = float(np.linalg.norm(self.b0 - self.a0))
        self.rocker = float(np.linalg.norm(self.b0 - self.o4))
        self.ground = float(np.linalg.norm(self.o4 - self.o2))
        self.step = float(step_rad)
        self._cache = {}

    @staticmethod
    def _rot2(v, t):
        c, s = math.cos(t), math.sin(t)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    @staticmethod
    def _ang2(v0, v1):
        return math.atan2(v0[0] * v1[1] - v0[1] * v1[0], float(np.dot(v0, v1)))

    def _point(self, q, prev):
        a = self.o2 + self._rot2(self.a0 - self.o2, q)
        d = float(np.linalg.norm(a - self.o4))
        r1, r2 = self.rocker, self.coupler
        if d < 1e-12:
            raise LoopError("degenerate loop at %.6f rad" % q)
        x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
        h2 = r1 * r1 - x * x
        if h2 < -1e-9 * max(1.0, r1 * r1):
            raise LoopError("the loop cannot close at %.6f rad" % q)
        h = math.sqrt(max(0.0, h2))
        e = (a - self.o4) / d
        nn = np.array([-e[1], e[0]])
        c1, c2 = self.o4 + x * e + h * nn, self.o4 + x * e - h * nn
        b = c1 if float(np.linalg.norm(c1 - prev)) <= float(np.linalg.norm(c2 - prev)) else c2
        return a, b

    def solve(self, q):
        q = float(q)
        if q == 0.0:
            return 0.0, 0.0
        key = round(q, 15)
        if key in self._cache:
            got = self._cache[key]
            if isinstance(got, LoopError):
                raise got
            return got
        n = max(1, int(math.ceil(abs(q) / self.step - 1e-12)))
        b, a = self.b0, self.a0
        try:
            for k in range(1, n + 1):
                a, b = self._point(q * k / n, b)
        except LoopError as exc:
            self._cache[key] = exc
            raise
        psi = self._ang2(self.b0 - self.a0, b - a)
        chi = self._ang2(self.b0 - self.o4, b - self.o4)
        phi = (psi - q + math.pi) % (2.0 * math.pi) - math.pi
        self._cache[key] = (phi, chi)
        return phi, chi


# ============================================================================ seeds and poses
def pose_seeds():
    """(hi, lo) of the per-run pose stream: ARB_SEEDS_PIN="hi:lo" reproduces a run; else drawn fresh (the kit's hidden
    seed source when it is importable, else the OS)."""
    pin = os.environ.get("ARB_SEEDS_PIN")
    if pin:
        hi, lo = (int(x) for x in pin.split(":"))
        return hi, lo, True
    try:
        from robotics_grader import seed_rng
        rng = seed_rng()
        return int(rng.seed(31)), int(rng.seed(31)), False
    except Exception:  # noqa: BLE001 - the kit is not on the path (a development run): the OS
        return int.from_bytes(os.urandom(4), "little") >> 1, int.from_bytes(os.urandom(4), "little") >> 1, False


def halton(n, d, rng):
    from scipy.stats import qmc
    try:
        eng = qmc.Halton(d, scramble=True, rng=rng)
    except TypeError:                     # scipy < 1.15
        eng = qmc.Halton(d, scramble=True, seed=rng)
    return np.asarray(eng.random(int(n)), float)


# ============================================================================ collision bodies
class Bodies(object):
    """Rigid bodies made of placed tessellations: {body: [(geo, T0)]} with T0 the geo's frame -> the aligned frame at Z.
    overlap(a, b, Ta, Tb) = the summed common volume of their items with the bodies moved by Ta, Tb (4 x 4 motions from
    Z in the aligned frame); items are pruned by bounding sphere and box."""

    def __init__(self, items):
        self.items = {}
        self.volume = {}
        self.sphere = {}
        for b, lst in items.items():
            out = []
            for geo, T0 in lst:
                if geo is None:
                    continue
                man = G.placed_manifold(geo, T0)
                if man is None:
                    continue
                lo, hi = geo.bounds
                corners = G.apply(T0, G.box_probe(lo, hi))
                c = corners.mean(axis=0)
                r = float(np.max(np.linalg.norm(corners - c, axis=1)))
                out.append({"man": man, "corners": corners, "c": c, "r": r, "vol": float(geo.volume)})
            self.items[b] = out
            self.volume[b] = sum(it["vol"] for it in out)
            if out:
                P = np.vstack([it["corners"] for it in out])
                c = P.mean(axis=0)
                self.sphere[b] = (c, float(np.max(np.linalg.norm(P - c, axis=1))))
        self._moved = {}
        self.n_bool = 0
        self.t_bool = 0.0

    def reset(self):
        self._moved = {}

    def _item(self, b, k, T, key):
        ck = (b, k, key)
        got = self._moved.get(ck)
        if got is None:
            it = self.items[b][k]
            man = it["man"] if key is None else it["man"].transform(np.ascontiguousarray(T[:3, :], np.float32))
            P = G.apply(T, it["corners"]) if key is not None else it["corners"]
            got = (man, np.concatenate([P.min(axis=0), P.max(axis=0)]))
            self._moved[ck] = got
        return got

    def near(self, a, b, Ta, Tb):
        if a not in self.sphere or b not in self.sphere:
            return False
        ca, ra = self.sphere[a]
        cb, rb = self.sphere[b]
        return float(np.linalg.norm(G.apply(Ta, ca) - G.apply(Tb, cb))) <= ra + rb

    def overlap(self, a, b, Ta, Tb, ka, kb):
        """ka, kb: hashable keys of the motions (None: identity) for the per-pose cache of moved items."""
        if not self.near(a, b, Ta, Tb):
            return 0.0
        vol = 0.0
        for i, ia in enumerate(self.items.get(a, [])):
            ca = G.apply(Ta, ia["c"])
            for k, ib in enumerate(self.items.get(b, [])):
                if float(np.linalg.norm(ca - G.apply(Tb, ib["c"]))) > ia["r"] + ib["r"]:
                    continue
                ma, ba = self._item(a, i, Ta, ka)
                mb, bb = self._item(b, k, Tb, kb)
                if not G.aabb_overlap(ba, bb):
                    continue
                t0 = time.time()
                vol += G.common_volume(ma, mb)
                self.t_bool += time.time() - t0
                self.n_bool += 1
        return vol


# ============================================================================ the reference side
class RefMech(object):
    """The reference hand's mechanism prepared once per loaded reference: hinge lines, loop solvers, the true poser,
    link masses, collision bodies with their readings at Z, the fit's contact patches and the unit's own overlaps."""

    def __init__(self, ref, spec, cfg, mc):
        self.ref, self.spec, self.cfg, self.mc = ref, spec, cfg, mc
        M = mc["mechanism"]
        self.loops = {}
        for d in spec.order:
            D = spec.digits[d]
            if D["loop"] is None:
                continue
            H = spec.hinges
            lp = D["loop"]
            self.loops[d] = PlanarLoop(H[lp["driver"]]["axis"], H[lp["driver"]]["centre"], H[lp["distal"]]["centre"],
                                       H[lp["base_pivot"]]["centre"], H[lp["link_pin"]]["centre"],
                                       math.radians(float(M["loop_step_deg"])))
        self.tip_local = {}
        for d in spec.order:
            t = spec.digits[d]["tip"]
            W = ref.placements[t["placement"]]["W"]
            self.tip_local[d] = G.apply(G.inv(W), t["point_mm"])
        self._bodies = None
        self._fit = None
        self.cache = {}

    # ---------------------------------------------------------------- poses
    def link_transforms(self, q):
        """{link: motion from Z} at the independent angles q (missing = 0); raises LoopError beyond a toggle."""
        spec = self.spec
        T = {HT.PALM: np.eye(4)}
        for d in spec.order:
            D = spec.digits[d]
            prev = HT.PALM
            for i, L in enumerate(D["chain"]):
                h = spec.hinge_between(prev, L)
                if h["kind"] == "servo":
                    a = float(q.get(h["id"], 0.0))
                else:                                   # the distal hinge: the loop
                    a = self.loops[d].solve(float(q.get(D["loop"]["driver"], 0.0)))[0]
                T[L] = T[prev] @ rot_about(h["centre"], h["axis"], a)
                prev = L
            if D["bar"]:
                hb = spec.hinges[D["loop"]["base_pivot"]]
                chi = self.loops[d].solve(float(q.get(D["loop"]["driver"], 0.0)))[1]
                T[D["bar"]] = T[spec.hinges[hb["id"]]["parent"]] @ rot_about(hb["centre"], hb["axis"], chi)
        return T

    def tip(self, d, T):
        t = self.spec.digits[d]["tip"]
        L = self.spec.link_of_placement[t["placement"]]
        return G.apply(T[L], t["point_mm"])

    # ---------------------------------------------------------------- mass
    def link_mass(self, links, excluded):
        """(mass g, centre of mass) of a set of reference links at Z (excluded given types left out)."""
        ref, spec = self.ref, self.spec
        m, mc = 0.0, np.zeros(3)
        for L in links:
            for t in spec.links[L]["made"]:
                k = ref.kinds[ref.parts[t]["kind"]]
                mm = k["V"] * k["density"]
                m += mm
                mc += mm * G.apply(ref.parts[t]["W"], k["com"])
            for p in spec.links[L]["given"]:
                pl = ref.placements[p]
                if pl["type"] in excluded:
                    continue
                typ = ref.types[pl["type"]]
                m += typ["mass_g"]
                mc += typ["mass_g"] * G.apply(pl["W"], typ["centroid"])
        return m, (mc / m if m > 0 else mc)

    # ---------------------------------------------------------------- collision bodies of the true hand
    def bodies(self):
        if self._bodies is None:
            ref, spec = self.ref, self.spec
            items = {}
            for L, rec in spec.links.items():
                lst = []
                for t in rec["made"]:
                    lst.append((ref.kinds[ref.parts[t]["kind"]]["geo"], ref.parts[t]["W"]))
                for p in rec["given"]:
                    pl = ref.placements[p]
                    for c in ref.types[pl["type"]]["cons"]:
                        lst.append((c["geo"], pl["W"]))
                items[L] = lst
            self._bodies = Bodies(items)
            self.adjacent = set()
            for h in spec.hinges.values():
                self.adjacent.add((h["parent"], h["child"]))
                self.adjacent.add((h["child"], h["parent"]))
            self.z_overlap = {}
        return self._bodies

    def pair_overlap_z(self, a, b):
        key = tuple(sorted((a, b)))
        if key not in self.z_overlap:
            self.z_overlap[key] = self._bodies.overlap(key[0], key[1], np.eye(4), np.eye(4), None, None)
        return self.z_overlap[key]

    def collides(self, pairs, T, ka, ic):
        """True when a pair of links grows beyond its ratio over Z at the motions T (ka: the pose key)."""
        B = self.bodies()
        for a, b in pairs:
            if not B.near(a, b, T[a], T[b]):
                continue
            v = B.overlap(a, b, T[a], T[b], (ka, a), (ka, b)) - self.pair_overlap_z(a, b)
            thr = float(ic["adjacent_ratio"] if (a, b) in self.adjacent else ic["other_ratio"])
            if v > thr * max(1e-9, min(B.volume[a], B.volume[b])):
                return True
        return False

    # ---------------------------------------------------------------- the fit's reference data
    def fit_data(self):
        """Per reference placement: its contact patch (local points with the tag each touches), on located surface
        only (a patch point counts when the made part's surface sample nearest to it is marked located); the unit's own
        common volumes of placement-part and part-part pairs."""
        if self._fit is not None:
            return self._fit
        from scipy.spatial import cKDTree
        ref, cfg = self.ref, self.cfg
        F = self.mc["fit"]
        reach = float(F["contact_patch_mm"])
        disk = float(cfg["sampling"]["disk_mm"])
        placed_part, part_tree = {}, {}
        for t, p in ref.parts.items():
            k = ref.kinds[p["kind"]]
            placed_part[t] = G.Placed(k["geo"], p["W"], cfg)
            P, N, _tr, L = k["samples"]
            Pw = G.apply(p["W"], P)
            part_tree[t] = (Pw, N @ p["W"][:3, :3].T, cKDTree(Pw), np.asarray(L, bool))
        placed_pl = {pid: [G.Placed(c["geo"], pl["W"], cfg) for c in ref.types[pl["type"]]["cons"]]
                     for pid, pl in ref.placements.items()}
        # the unit's own contacts: the reference's pairs file where it ships one (exact B-rep gaps), else every part
        touches = None
        pinfo = ref.spec.get("pairs")
        if isinstance(pinfo, dict) and pinfo.get("file"):
            path = os.path.join(ref.dir, pinfo["file"])
            if pinfo.get("sha256") and G.sha256_file(path) != pinfo["sha256"]:
                raise G.TaskError("the reference pairs do not match their pin")
            with open(path) as fh:
                touches = json.load(fh).get("given_touches_made")
        patch, common_pt, common_tt = {}, {}, {}
        tags_all = sorted(ref.parts)
        for pid, pl in sorted(ref.placements.items()):
            typ = ref.types[pl["type"]]
            Q = np.vstack([c["geo"].dense(cfg)[0] for c in typ["cons"]])
            Qw = G.apply(pl["W"], Q)
            best = np.full(len(Qw), np.inf)
            who = np.full(len(Qw), -1)
            seen = np.zeros(len(Qw), bool)            # the nearest surface is located (the photographs show it)
            tags = tags_all
            box = np.concatenate([Qw.min(axis=0), Qw.max(axis=0)])
            for ti, t in enumerate(tags):
                pb = placed_part[t].box
                if not G.aabb_overlap(box, pb, reach + 0.5):
                    continue
                near = np.all((Qw >= pb[:3] - reach - disk) & (Qw <= pb[3:] + reach + disk), axis=1)
                if near.any():
                    Pw, Nw, tr, Lw = part_tree[t]
                    idx = np.nonzero(near)[0]
                    d = G.surface_distances(Qw[idx], Pw, Nw, tr, disk)
                    _dn, nn = tr.query(Qw[idx])
                    sel = d < best[idx]
                    best[idx[sel]] = d[sel]
                    who[idx[sel]] = ti
                    seen[idx[sel]] = Lw[np.asarray(nn, int)[sel]]
                for itp in placed_pl[pid]:
                    if G.aabb_overlap(itp.box, placed_part[t].box):
                        common_pt[(pid, t)] = common_pt.get((pid, t), 0.0) + G.common_volume(itp.man, placed_part[t].man)
            keep = (best <= reach) & seen
            if touches is not None:                 # only the parts the reference lists as touching this instance
                listed = set(touches.get(pid, []))
                keep &= np.array([who[i] >= 0 and tags[who[i]] in listed for i in range(len(who))], bool)
            patch[pid] = {"local": Q[keep], "tag": [tags[i] for i in who[keep]], "baseline_mm": float(np.median(best[keep]))
                          if keep.any() else None}
        tags = sorted(ref.parts)
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                a, b = placed_part[t1], placed_part[t2]
                if G.aabb_overlap(a.box, b.box):
                    v = G.common_volume(a.man, b.man)
                    if v > 0:
                        common_tt[(t1, t2)] = v
        self._fit = {"patch": patch, "common_pt": common_pt, "common_tt": common_tt}
        return self._fit


def ref_mech(G_):
    """The RefMech of the grading's reference (cached on the reference object)."""
    ref = G_.ref
    rm = getattr(ref, "_ref_mech", None)
    if rm is None:
        spec = HT.load_spec(ref)
        mc = HT.mech_config(G_.cfg)
        rm = RefMech(ref, spec, G_.cfg, mc)
        ref._ref_mech = rm
    return rm


# ============================================================================ the walk (the submission side)
class Edge(object):
    """An edge between two groups: kind line (a declared Revolute / Cylindrical axis), ball (a centre) or pin (a pin's
    axis); joints = [(joint record, group of part1, group of part2)], the groups after pass-through mapping."""

    def __init__(self, a, b, kind, point, direction, joints, pin=None, frame_point=None):
        self.a, self.b = a, b
        self.kind = kind
        self.point = np.asarray(point, float)
        self.dir = unit(direction) if direction is not None else None
        self.joints = list(joints)
        self.pin = pin
        self.frame_point = np.asarray(frame_point if frame_point is not None else point, float)
        self.types = sorted(set(j["type"] for j, _g1, _g2 in self.joints)) if self.joints else ["pin"]
        self.through = []
        self.conflict = False       # its joints disagree: the hinge they declare is read as missing
        self.n_raw = 1              # how many edges were combined into it

    def other(self, g):
        return self.b if g == self.a else self.a

    def pair(self):
        return tuple(sorted((self.a, self.b)))


class DigitReading(object):
    def __init__(self, name):
        self.name = name
        self.candidate = None
        self.groups = []           # the candidate's groups
        self.match = {}            # link -> group (PALM for the palm groups) or None
        self.hinges = {}           # hinge id -> reading
        self.valid = False
        self.invalid_reason = None
        self.loop_closed = False
        self.loop = None
        self.loop_sign = 1.0
        self.tip_group = None
        self.tip_point = None
        self.reports = []
        self.first_key = None


class Walk(object):
    def __init__(self, G_, rm):
        self.G, self.rm, self.spec, self.ref = G_, rm, rm.spec, G_.ref
        self.mc = rm.mc
        self.M = rm.mc["mechanism"]
        self.A = G_.A
        self.stats = {}
        self.notes = []
        self.palm = set(G_.palm)

    def note(self, msg):
        if len(self.notes) < 40:
            self.notes.append(msg)

    def node(self, g):
        """The walk's node of a group: the palm groups count as one."""
        return PALM if g in self.palm else g

    def doc_order(self, e):
        """The document order of an edge: the position of its first joint in the document (pin edges last)."""
        pos = [self.joint_pos.get(id(j), 10 ** 9) for j, _g1, _g2 in e.joints]
        return min(pos) if pos else 10 ** 9

    def count(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0.0) + float(n)

    # ---------------------------------------------------------------- groups and their members
    def collect(self):
        G_, ref, spec = self.G, self.ref, self.spec
        for k in WALK_COUNTS:
            self.stats.setdefault(k, 0.0)
        self.joint_pos = {id(j): n for n, j in enumerate(G_.sub.model.get("joints", []))}
        self.gsol = {}
        for s in G_.sub.solids:
            g = G_.groups.group(s["node"])
            if g is not None:
                self.gsol.setdefault(g, []).append(s)
        self.gtag = {}                       # reference tag -> group of its assigned solid
        for t, s in G_.part_of.items():
            g = G_.groups.group(s["node"])
            if g is not None:
                self.gtag[t] = g
        self.inst_group = {}
        for i, it in enumerate(G_.inst):
            typ = ref.types[it["type"]]
            s = G_.sol[it["slots"][typ["anchor"]]]
            self.inst_group[i] = G_.groups.group(s["node"])
        self.ginst = {}
        for i, g in self.inst_group.items():
            self.ginst.setdefault(g, []).append(i)
        struct = set(ref.structural_kinds)
        palm_kinds = set(ref.palm_kinds)
        self.gflags = {}
        for g, sols in self.gsol.items():
            f = {"structural": False, "palm": g in self.palm, "servo": False, "n": len(sols)}
            for s in sols:
                if s["tag"] is not None:
                    kid = ref.parts[s["tag"]]["kind"]
                    if kid in struct:
                        f["structural"] = True
                    if kid in palm_kinds:
                        f["palm_part"] = True
            for i in self.ginst.get(g, []):
                tn = G_.inst[i]["type"]
                if tn == HT.SERVO:
                    f["servo"] = True
                if tn in ref.palm_types:
                    f["palm_part"] = True
            self.gflags[g] = f
        self.servo_unfixed = []
        for i, it in enumerate(G_.inst):
            if it["type"] == HT.SERVO and self.gflags.get(self.inst_group[i], {}).get("n", 0) <= len(it["slots"]):
                self.servo_unfixed.append(i)
        self.stats["servo_unfixed"] = float(len(self.servo_unfixed))
        off = 0
        for gr in G_.sub.model.get("grounded", []):
            if gr.get("suppressed") or gr.get("foreign"):
                continue
            g = G_.groups.group(G_.groups.leaf_of(gr.get("node")) or "")
            if g is not None and g not in self.palm:
                off += 1
        self.stats["ground_not_on_palm"] = float(off)
        if off:
            self.note("%d Grounded joint(s) on a part that is not in the palm (ground_on_palm): grounding joins nothing" % off)

    # ---------------------------------------------------------------- the servo and pin geometry of instances
    def servo_axis(self, i):
        it = self.G.inst[i]
        typ = self.ref.types[it["type"]]
        oa = typ["output_axis"] or {"point": [0, 0, 0], "dir": [0, 0, 1]}
        W = self.A @ it["W"]
        return G.apply(W, np.asarray(oa["point"], float)), unit(W[:3, :3] @ np.asarray(oa["dir"], float))

    def axial_instance(self, i):
        """(centre, axis, half length) of a pin or a bearing instance in the aligned frame."""
        it = self.G.inst[i]
        typ = self.ref.types[it["type"]]
        W = self.A @ it["W"]
        a = unit(W[:3, :3] @ typ["sym_axis"])
        c = G.apply(W, typ["centroid"])
        V = G.apply(W, typ["cons"][0]["verts"])
        hl = float(np.max(np.abs((V - c) @ a))) if len(V) else 0.0
        return c, a, hl

    # ---------------------------------------------------------------- 2: hinge edges
    def joint_edge(self, j):
        try:
            J1, J2 = np.asarray(j.get("jcs1"), float), np.asarray(j.get("jcs2"), float)
        except (TypeError, ValueError):
            return None
        if J1.shape != (4, 4) or J2.shape != (4, 4) or not (np.all(np.isfinite(J1)) and np.all(np.isfinite(J2))):
            return None
        J1, J2 = self.A @ J1, self.A @ J2
        p1, p2 = J1[:3, 3], J2[:3, 3]
        z1, z2 = unit(J1[:3, 2]), unit(J2[:3, 2])
        if j["type"] == "Ball":
            if float(np.linalg.norm(p1 - p2)) > float(self.M["frames_apart_mm"]):
                self.count("frames_apart")
            c = 0.5 * (p1 + p2)
            return "ball", c, None, c
        apart = point_line(p2, p1, z1) > float(self.M["frames_apart_mm"]) or line_angle_deg(z1, z2) > float(
            self.M["frames_apart_deg"])
        if apart:
            self.count("frames_apart")
            self.note("joint %s: its two frames are apart at the saved pose; read on their mean axis" % j.get("label"))
        z = unit(z1 + (1.0 if float(z1 @ z2) >= 0 else -1.0) * z2)
        p = 0.5 * (p1 + p2)
        if j["type"] == "Cylindrical":
            self.count("cylindrical_as_hinge")
        return "line", p, z, p

    def read_joints(self):
        G_ = self.G
        raw = []
        for j in G_.sub.model.get("joints", []):
            if j.get("type") not in ("Revolute", "Cylindrical", "Ball"):
                continue
            if not G.Groups.usable(j):
                continue
            g1 = G_.groups.group(G_.groups.leaf_of(j.get("part1")) or "")
            g2 = G_.groups.group(G_.groups.leaf_of(j.get("part2")) or "")
            if g1 is None or g2 is None:
                self.count("joints_unread")
                self.note("joint %s: a reference reaches no part of the Assembly; not read" % j.get("label"))
                continue
            if g1 == g2:
                self.count("edges_in_group")
                continue
            got = self.joint_edge(j)
            if got is None:
                self.count("joints_unread")
                self.note("joint %s: its frames at the saved pose are not finite numbers (a placement is not); not "
                          "read as a hinge" % j.get("label"))
                continue
            kind, p, z, fp = got
            if j["type"] == "Ball":
                self.count("ball_joints")
            raw.append(Edge(g1, g2, kind, p, z, [(j, g1, g2)], frame_point=fp))
        self.stats["joint_types_other"] = float(len(G_.sub.model.get("joint_types_other", [])))
        self.edges = self.merge(raw)

    def disagreements(self):
        """The edges whose joints disagree, counted once on the final edges (merge runs again after pass-through
        groups and pin edges)."""
        bad = [e for e in self.edges if e.conflict and not (e.a in self.palm and e.b in self.palm)]
        self.stats["hinges_disagreeing"] = float(len(bad))
        self.stats["edges_disagreeing"] = float(sum(e.n_raw - 1 for e in bad))
        for e in bad:
            labels = [str(j.get("label")) for j, _g1, _g2 in e.joints][:4]
            self.note("%d joints between two groups disagree (more than %g mm or %g degrees apart: %s): the hinge they "
                      "declare is read as missing (held rigid), none of them is chosen" % (
                          e.n_raw, float(self.M["coaxial_mm"]), float(self.M["coaxial_deg"]), ", ".join(labels)))

    def agree(self, e, f):
        tol_mm, tol_deg = float(self.M["coaxial_mm"]), float(self.M["coaxial_deg"])
        if e.conflict or f.conflict:
            return False
        if e.kind != "ball" and f.kind != "ball":
            return line_angle_deg(e.dir, f.dir) <= tol_deg and max(point_line(f.point, e.point, e.dir),
                                                                     point_line(e.point, f.point, f.dir)) <= tol_mm
        if e.kind == "ball" and f.kind == "ball":
            return float(np.linalg.norm(e.point - f.point)) <= tol_mm
        line, ball = (e, f) if f.kind == "ball" else (f, e)
        return point_line(ball.point, line.point, line.dir) <= tol_mm

    def combine(self, members):
        """One edge on the mean axis of agreeing edges between the same two groups."""
        lines = [m for m in members if m.kind != "ball"]
        a, b = members[0].a, members[0].b
        joints = [jj for m in members for jj in m.joints]
        if lines:
            d0 = lines[0].dir
            ds = [m.dir * (1.0 if float(m.dir @ d0) >= 0 else -1.0) for m in lines]
            d = unit(np.sum(ds, axis=0))
            p0 = np.mean([m.point for m in lines], axis=0)
            kind = "pin" if all(m.kind == "pin" for m in lines) else "line"
        else:
            d, p0, kind = None, np.mean([m.point for m in members], axis=0), "ball"
        fp = np.mean([m.frame_point for m in members], axis=0)
        e = Edge(a, b, kind, p0, d, joints, pin=members[0].pin, frame_point=fp)
        e.through = [x for m in members for x in m.through]
        e.n_raw = sum(m.n_raw for m in members)
        e.conflict = any(m.conflict for m in members)
        return e

    def merge(self, raw):
        """Edges between the same two nodes (the palm groups count as one) become one edge: on their mean axis when
        every two of them agree; else one edge marked as a conflict (its joints disagree): it joins the two groups,
        and the hinge it stands for is missing. Nothing is chosen among disagreeing joints."""
        by = {}
        for e in raw:
            key = tuple(sorted((str(self.node(e.a)), str(self.node(e.b)))))
            by.setdefault(key, []).append(e)
        out = []
        for key in sorted(by):
            es = by[key]
            if len(es) == 1:
                out.append(es[0])
                continue
            ok = all(self.agree(es[i], es[k]) for i in range(len(es)) for k in range(i + 1, len(es)))
            e = self.combine(es)
            e.conflict = e.conflict or not ok
            out.append(e)
        return out

    # ---------------------------------------------------------------- pass-through groups
    def pass_through(self):
        n_removed = 0
        for _round in range(64):
            nb = {}
            for e in self.edges:
                nb.setdefault(e.a, []).append(e)
                nb.setdefault(e.b, []).append(e)
            done = False
            for g in sorted(nb):
                f = self.gflags.get(g, {})
                if g in self.palm or f.get("structural") or f.get("palm_part") or f.get("servo"):
                    continue
                es = nb[g]
                if len(es) != 2 or es[0].other(g) == es[1].other(g):
                    continue
                if es[0].kind == "pin" or es[1].kind == "pin" or not self.agree(es[0], es[1]):
                    continue
                n1, n2 = es[0].other(g), es[1].other(g)
                joints = []
                for e, far in ((es[0], n2), (es[1], n1)):
                    for j, g1, g2 in e.joints:
                        joints.append((j, far if g1 == g else g1, far if g2 == g else g2))
                lines = [e for e in es if e.kind != "ball"]
                base = lines[0] if lines else es[0]
                ne = Edge(n1, n2, base.kind, base.point if len(lines) != 2 else 0.5 * (lines[0].point + lines[1].point),
                          base.dir if len(lines) != 2 else unit(lines[0].dir + (1 if float(lines[0].dir @ lines[1].dir) >= 0
                                                                               else -1) * lines[1].dir),
                          joints, frame_point=0.5 * (es[0].frame_point + es[1].frame_point))
                ne.through = es[0].through + es[1].through + [g]
                self.edges = [e for e in self.edges if e is not es[0] and e is not es[1]]
                self.edges = self.merge(self.edges + [ne])
                n_removed += 1
                done = True
                break
            if not done:
                break
        self.stats["pass_through_groups"] = float(n_removed)

    # ---------------------------------------------------------------- pin edges
    def steps_of(self, s):
        if "steps" not in s:
            try:
                s["steps"] = G.axial_steps(s["shape"], self.G.cfg)
            except Exception:  # noqa: BLE001 - a part whose faces cannot be read carries nothing
                s["steps"] = []
        return s["steps"]

    def pin_edges(self):
        G_ = self.G
        tol_mm, tol_deg = float(self.M["coaxial_mm"]), float(self.M["coaxial_deg"])
        slack = float(self.M["pin_along_slack_mm"])
        joined = set(tuple(sorted((str(self.node(e.a)), str(self.node(e.b))))) for e in self.edges)
        brg = [i for i, it in enumerate(G_.inst) if it["type"] == "BRG"]
        brg_geo = {i: self.axial_instance(i) for i in brg}
        added = []
        for i, it in enumerate(G_.inst):
            if it["type"] != "PIN":
                continue
            c, a, hl = self.axial_instance(i)
            g0 = self.inst_group[i]
            carriers = set()
            if g0 is not None and self.gflags.get(g0, {}).get("n", 0) > len(it["slots"]):
                carriers.add(g0)
            for g, sols in self.gsol.items():
                if g == g0 or g in carriers:
                    continue
                hit = False
                for s in sols:
                    if s["tag"] is None or s.get("demoted"):
                        continue
                    for st in self.steps_of(s):
                        p = G.apply(self.A, st["point"])
                        d = self.A[:3, :3] @ st["dir"]
                        if line_angle_deg(d, a) > tol_deg or point_line(p, c, a) > tol_mm:
                            continue
                        if abs(float((p - c) @ a)) > hl + 0.5 * st["depth"] + slack:
                            continue
                        hit = True
                        break
                    if hit:
                        break
                if not hit:
                    for b in self.ginst.get(g, []):
                        if b not in brg_geo:
                            continue
                        cb, ab, hb = brg_geo[b]
                        if line_angle_deg(ab, a) <= tol_deg and point_line(cb, c, a) <= tol_mm and \
                                abs(float((cb - c) @ a)) <= hl + hb + slack:
                            hit = True
                            break
                if hit:
                    carriers.add(g)
            cs = sorted(carriers)
            for x in range(len(cs)):
                for y in range(x + 1, len(cs)):
                    pair = tuple(sorted((str(self.node(cs[x])), str(self.node(cs[y])))))
                    if pair in joined or pair[0] == pair[1]:
                        continue
                    joined.add(pair)
                    added.append(Edge(cs[x], cs[y], "pin", c, a, [], pin=i))
        if added:
            self.stats["pin_marked"] = float(len(set(e.pair() for e in added)))
            self.note("%d hinge(s) read from dowel pins that both groups carry, without a joint (pin_marked)" % (
                len(set(e.pair() for e in added))))
        else:
            self.stats["pin_marked"] = 0.0
        self.edges = self.merge(self.edges + added)
        self.disagreements()

    # ---------------------------------------------------------------- 3, 4: candidates and slots
    def candidates(self):
        adj = {}
        for e in self.edges:
            adj.setdefault(e.a, []).append(e)
            adj.setdefault(e.b, []).append(e)
        self.adj = adj
        seen, comps = set(), []
        nodes = sorted(set(g for e in self.edges for g in (e.a, e.b)) - self.palm)
        for g0 in nodes:
            if g0 in seen:
                continue
            comp, todo = set([g0]), [g0]
            seen.add(g0)
            while todo:
                g = todo.pop()
                for e in adj.get(g, []):
                    o = e.other(g)
                    if o in self.palm or o in seen:
                        continue
                    seen.add(o)
                    comp.add(o)
                    todo.append(o)
            comps.append(sorted(comp))
        self.cands = []
        floating = 0
        for comp in comps:
            cs = set(comp)
            first = [e for g in comp for e in adj.get(g, []) if e.kind == "line" and (e.other(g) in self.palm)]
            first = [e for k, e in enumerate(first) if all(e is not f for f in first[:k])]
            if not first:
                floating += len(comp)
                continue
            first.sort(key=lambda e: (self.chain_rank(e.a if e.a not in self.palm else e.b), self.doc_order(e)))
            self.cands.append({"groups": comp, "set": cs, "first": first})
        lone = [g for g in self.gsol if g not in self.palm and g not in seen]
        self.stats["floating_groups"] = float(floating + len(lone))
        self.stats["digit_candidates"] = float(len(self.cands))

    def chain_rank(self, g):
        """The earliest chain position (0 for a first link) among the structural parts rigid in group g, any digit;
        a group without one ranks last. Decides between several first hinges of a candidate by structure."""
        best = 10 ** 6
        for t, gg in self.gtag.items():
            if gg != g:
                continue
            L = self.spec.link_of_tag.get(t)
            if L is None or L == HT.PALM or t not in self.spec.links[L]["structural"]:
                continue
            d = self.spec.digit_of_link.get(L)
            if d is None or L not in self.spec.digits[d]["chain"]:
                continue
            best = min(best, list(self.spec.digits[d]["chain"]).index(L))
        return best

    def slot_cost(self, e, d):
        """(cost, hinge key) of the first-hinge edge e for digit d's slot, None if not admissible."""
        M = self.M
        best = None
        for hid in self.spec.digits[d]["hinges"]:
            h = self.spec.hinges[hid]
            if h["kind"] != "servo":
                continue
            dist = point_line(h["centre"], e.point, e.dir)
            ang = line_angle_deg(h["axis"], e.dir)
            if dist > float(M["slot_max_mm"]) or ang > float(M["slot_max_deg"]):
                continue
            c = dist + float(M["slot_mm_per_deg"]) * ang + float(M["slot_tiebreak_per_mm"]) * float(
                np.linalg.norm(h["centre"] - e.frame_point))
            if best is None or c < best[0]:
                best = (c, h["key"], e)
        return best

    def slots(self):
        digits = self.spec.order
        self.readings = {d: DigitReading(d) for d in digits}
        if not self.cands:
            self.stats["unassigned_candidates"] = 0.0
            return
        BIG = 1e9
        C = np.full((len(self.cands), len(digits)), BIG)
        pick = {}
        for ci, c in enumerate(self.cands):
            for di, d in enumerate(digits):
                for e in c["first"][:1]:         # the first hinge by structure (candidates); never the one a slot likes
                    got = self.slot_cost(e, d)
                    if got is not None and got[0] < C[ci, di]:
                        C[ci, di] = got[0]
                        pick[(ci, di)] = got
        unassigned = 0
        assigned = set()
        for ci, di in G.hungarian(C):
            if C[ci, di] >= BIG:
                continue
            d = digits[di]
            r = self.readings[d]
            c = self.cands[ci]
            cost, key, e = pick[(ci, di)]
            r.candidate = c
            r.groups = c["groups"]
            r.first_edge = e
            r.first_key = key
            r.slot_cost = cost
            assigned.add(ci)
            if len(c["first"]) > 1:
                self.count("first_hinge_surplus", len(c["first"]) - 1)
        unassigned = len(self.cands) - len(assigned)
        self.stats["unassigned_candidates"] = float(unassigned)
        if unassigned:
            self.note("%d chain(s) of parts hinged to the palm match no digit of the hand (unassigned)" % unassigned)

    # ---------------------------------------------------------------- 5-9: one digit
    def match_links(self, r):
        spec, G_ = self.spec, self.G
        D = spec.digits[r.name]
        pool = set(r.groups) | self.palm
        r.match = {HT.PALM: PALM}
        links = list(D["chain"]) + ([D["bar"]] if D["bar"] else [])
        counts = {}
        held = {PALM}                        # the palm link holds the palm groups
        tied = []
        for L in links:                      # a strict majority by count first
            cnt, vol = {}, {}
            for t in spec.links[L]["structural"]:
                g = self.gtag.get(t)
                if g is None or g not in pool:
                    continue
                key = PALM if g in self.palm else g
                cnt[key] = cnt.get(key, 0) + 1
                vol[key] = vol.get(key, 0.0) + self.ref.kinds[self.ref.parts[t]["kind"]]["V"]
            counts[L] = (cnt, vol)
            if not cnt:
                r.match[L] = None
                continue
            top = max(cnt.values())
            best = sorted(k for k in cnt if cnt[k] == top)
            if len(best) == 1:
                r.match[L] = best[0]
                held.add(best[0])
            else:
                tied.append((L, best))
        for L, best in tied:                 # then a tied link takes the tied group no other link is matched to
            cnt, vol = counts[L]
            free = [k for k in best if k not in held]
            pool_k = free if free else best  # the reference volume decides only after that
            k = sorted(pool_k, key=lambda k: (-vol[k], str(k)))[0]
            r.match[L] = k
            held.add(k)
            self.count("links_tied")
        for L in links:
            cnt, _vol = counts[L]
            if not cnt:
                r.reports.append("%s absent" % L.split(".", 1)[1])
                self.count("links_absent")
                continue
            best = r.match[L]
            for t in spec.links[L]["structural"]:
                g = self.gtag.get(t)
                if g is None:
                    continue
                key = PALM if g in self.palm else g
                if key != best:
                    self.count("membership")
                    r.reports.append("%s of link %s sits in another link (membership)" % (t, L.split(".", 1)[1]))
        _ = G_

    def edges_between(self, ma, mb, kinds):
        out = []
        for e in self.edges:
            if e.kind not in kinds:
                continue
            ea = PALM if e.a in self.palm else e.a
            eb = PALM if e.b in self.palm else e.b
            if {ea, eb} == {ma, mb}:
                out.append(e)
        return out

    def validity(self, r):
        D = self.spec.digits[r.name]
        chain = [HT.PALM] + list(D["chain"])
        m = [r.match.get(L) for L in chain]
        n = len(chain)
        for i in range(n):
            for k in range(i + 2, n):
                if m[i] is None or m[k] is None or m[i] != m[k]:
                    continue
                for j in range(i + 1, k):
                    if m[j] is not None and m[j] != m[i]:
                        return "links %s and %s are in one group while %s sits in another" % (
                            chain[i], chain[k], chain[j])
        nodes = set(x for x in m if x is not None)
        dist = {PALM: 0}
        frontier = [PALM]
        while frontier:
            nxt = []
            for g in frontier:
                for e in self.edges:
                    ea = PALM if e.a in self.palm else e.a
                    eb = PALM if e.b in self.palm else e.b
                    if g not in (ea, eb):
                        continue
                    o = eb if ea == g else ea
                    if o in nodes and o not in dist:
                        dist[o] = dist[g] + 1
                        nxt.append(o)
            frontier = nxt
        dd = [dist.get(x, np.inf) if x is not None else None for x in m]
        for i in range(1, n):
            for j in range(i + 1, n):
                if dd[i] is None or dd[j] is None or m[i] == m[j]:
                    continue
                if dd[j] < dd[i] and not self.holds_part_of(m[j], chain[i]):
                    return "the joints reach %s before %s" % (chain[j], chain[i])
        return None

    def holds_part_of(self, mg, L):
        """True when the group matched as mg (PALM for the palm groups) holds a structural part of link L. The order
        rule is read on the matched links: a later link's group reached before an earlier link's through a part of
        the earlier link fixed into it (membership) is that part's mistake, not an order the joints declare; the
        hinges it carries join no matched pair of links and are missing."""
        for t in self.spec.links[L]["structural"]:
            g = self.gtag.get(t)
            if g is not None and (PALM if g in self.palm else g) == mg:
                return True
        return False

    def read_hinges(self, r):
        spec = self.spec
        D = spec.digits[r.name]
        M = self.M
        used = set()
        for hid in D["hinges"]:
            h = spec.hinges[hid]
            hr = {"id": hid, "status": "missing", "reason": None, "edge": None, "line": None, "sense": None,
                  "declared": None, "driver": None, "inconsistent": False, "from": None}
            r.hinges[hid] = hr
            mP, mQ = r.match.get(h["parent"]), r.match.get(h["child"])
            if mP is None or mQ is None:
                hr["reason"] = "a link of the hinge is absent"
                continue
            if mP == mQ:
                hr["reason"] = "both links in one group (run)"
                continue
            if D["bar"] in (h["parent"], h["child"]) and not self.bar_own(r):
                hr["reason"] = "the link bar is fixed to another link"
                continue
            kinds = ("line",) if h["kind"] == "servo" else ("line", "ball", "pin")
            es = self.edges_between(mP, mQ, kinds)
            if not es:
                hr["reason"] = "no joint between the two links"
                continue
            if len(es) > 1 or any(x.conflict for x in es):
                for x in es:
                    used.add(id(x))
                hr["reason"] = "the joints between the two links disagree"
                r.reports.append("%s missing: its joints disagree" % h["key"])
                continue
            e = es[0]
            used.add(id(e))
            hr["edge"] = e
            hr["status"] = "present"
            hr["from"] = e.kind
            if e.kind == "pin":
                self.count("hinges_from_pins")
            if h["kind"] == "servo":
                self.drive(r, h, hr, mP, mQ)
            else:
                hr["line"] = (e.point, e.dir)
        # pin hinges: the sense parallel to the third hinge (a Ball: its axis too)
        h3 = r.hinges.get(r.name + ".h3")
        d3 = None
        if h3 is not None and h3["status"] in ("present", "undriven") and h3["sense"] is not None:
            d3 = h3["sense"]
        for hid in D["hinges"]:
            h, hr = spec.hinges[hid], r.hinges[hid]
            if h["kind"] != "pin" or hr["status"] != "present":
                continue
            ref_dir = d3 if d3 is not None else self.spec.hinges[r.name + ".h3"]["axis"]
            p, d = hr["line"]
            if d is None:
                d = ref_dir
            d = d * (1.0 if float(d @ ref_dir) >= 0 else -1.0)
            hr["line"] = (p, unit(d))
            hr["sense"] = unit(d)
        # extra hinges
        pool = set(r.groups)
        extra = 0
        for e in self.edges:
            if id(e) in used:
                continue
            if (e.a in pool or e.b in pool) and (e.a in pool or e.a in self.palm) and (e.b in pool or e.b in self.palm):
                extra += 1
        if extra:
            self.count("extra_hinge", extra)
            r.reports.append("%d joint(s) that match no hinge of the unit (extra_hinge)" % extra)

    def bar_own(self, r):
        """True when the finger's link bar sits in a group of its own: no other link is matched to its group and the
        group holds no structural part of another link."""
        bar = self.spec.digits[r.name]["bar"]
        mE = r.match.get(bar)
        if mE is None or mE == PALM:
            return False
        if any(L != bar and m == mE for L, m in r.match.items()):
            return False
        for t, g in self.gtag.items():
            L = self.spec.link_of_tag.get(t)
            if g == mE and L is not None and L != bar and t in self.spec.links[L]["structural"]:
                return False
        return True

    def side_groups(self, m):
        return set(self.palm) if m == PALM else {m}

    def drive(self, r, h, hr, mP, mQ):
        M, G_ = self.M, self.G
        e = hr["edge"]
        palm_side, tip_side = self.side_groups(mP), self.side_groups(mQ)
        best = None
        for i, it in enumerate(G_.inst):
            if it["type"] != HT.SERVO:
                continue
            g = self.inst_group.get(i)
            if g in palm_side:
                side = "parent"
            elif g in tip_side:
                side = "child"
            else:
                continue
            c, d = self.servo_axis(i)
            if line_angle_deg(d, e.dir) > float(M["driver_deg"]) or point_line(c, e.point, e.dir) > float(M["driver_mm"]):
                continue
            key = (float(np.linalg.norm(c - e.frame_point)), i)
            if best is None or key < best[0]:
                best = (key, i, side, c, d)
        if best is None:
            hr["status"] = "undriven"
            hr["reason"] = "no servo rigidly in either group on the joint's axis"
            hr["line"] = (e.point, e.dir)
            hr["sense"] = e.dir
            self.count("undriven")
            r.reports.append("%s undriven" % h["key"])
            return
        _k, i, side, c, d = best
        s = d if side == "parent" else -d
        hr["driver"] = i
        hr["case_side"] = side
        hr["line"] = (c, s)
        hr["sense"] = s
        if side != h["case_side"]:
            self.count("case_side")
            r.reports.append("%s: its servo's case is on the other side of the hinge (case_side)" % h["key"])
        if line_angle_deg(d, e.dir) > float(M["consistent_deg"]) or point_line(c, e.point, e.dir) > float(M["consistent_mm"]):
            hr["inconsistent"] = True
            self.count("inconsistent")
            r.reports.append("%s: the joint's axis is off its servo's output axis (inconsistent; read on the servo)" % h["key"])
        if h["independent"]:
            hr["declared"] = self.declared(hr, s, mP)

    def declared(self, hr, s, mP):
        """The declared interval in the hinge's sense, relative to the saved pose, or None (both limits not enabled)."""
        M = self.M
        e = hr["edge"]
        js = [x for x in e.joints if x[0].get("type") in ("Revolute", "Cylindrical")]
        both = [x for x in js if x[0].get("enable_angle_min") and x[0].get("enable_angle_max")]
        if not both:
            return None
        j, g1, _g2 = sorted(both, key=lambda x: str(x[0].get("name")))[0]
        try:
            amin, amax = float(j.get("angle_min_deg")), float(j.get("angle_max_deg"))
            th0 = float(j.get("angle_saved_deg") or 0.0)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(amin) and math.isfinite(amax) and math.isfinite(th0)):
            return None
        lo, hi = math.radians(amin - th0), math.radians(amax - th0)
        J1 = self.A @ np.asarray(j["jcs1"], float)
        z1 = unit(J1[:3, 2])
        child_first = not ((PALM if g1 in self.palm else g1) == mP)
        if child_first:
            lo, hi = -hi, -lo
        if float(z1 @ s) < 0:
            lo, hi = -hi, -lo
        tol = math.radians(float(M["full_turn_tol_deg"]))
        if hi - lo >= 2.0 * math.pi - tol:
            return (-math.pi, math.pi)
        lo, hi = max(lo, -math.pi), min(hi, math.pi)
        if not lo < hi:
            return (0.0, 0.0)
        return (lo, hi)

    def read_loop(self, r):
        D = self.spec.digits[r.name]
        if D["loop"] is None:
            return
        lp = D["loop"]
        bar = D["bar"]
        mE = r.match.get(bar)
        hs = [r.hinges[lp[k]] for k in ("driver", "distal", "base_pivot", "link_pin")]
        why = None
        if mE is None or mE == PALM:
            why = "the link bar is not in the digit"
        elif not self.bar_own(r):
            why = "the link bar is fixed to another link"
        if why is None and any(hh["status"] not in ("present", "undriven") or hh["line"] is None for hh in hs):
            why = "a hinge of the loop is missing"
        if why is None:
            n = hs[0]["sense"]
            if any(hh["from"] != "ball" and line_angle_deg(hh["line"][1], n) > float(self.M["loop_parallel_deg"])
                   for hh in hs[1:]):
                why = "skew"
        if why is not None:
            r.loop_closed = False
            self.count("loop_skew" if why == "skew" else "loop_open")
            r.reports.append("loop %s" % ("skew" if why == "skew" else "open: " + why))
            return
        n = hs[0]["sense"]
        O2 = hs[0]["line"][0]
        pts = [plane_point(hh["line"][0], hh["line"][1] if hh["from"] != "ball" else n, O2, n) for hh in hs[1:]]
        r.loop = PlanarLoop(n, O2, pts[0], pts[1], pts[2], math.radians(float(self.M["loop_step_deg"])))
        r.loop_closed = True
        u_true = self.spec.hinges[lp["driver"]]["axis"]
        r.loop_sign = 1.0 if float(n @ u_true) >= 0 else -1.0

    def read_tip(self, r):
        """The digit's tactile module, chosen by structure: of the recognised modules of the slot's type rigid in the
        candidate's groups, the one in the group matched to the digit's last link, else the one farthest along the
        chain; ties by document order. Never by its distance to the unit's tip."""
        D = self.spec.digits[r.name]
        G_ = self.G
        pool = set(r.groups)
        tloc = self.rm.tip_local[r.name]
        chain = list(D["chain"])
        last = chain[-1]
        mods = [i for i, it in enumerate(G_.inst) if it["type"] == D["tip_type"] and self.inst_group.get(i) in pool]
        if not mods:
            r.reports.append("no tactile module in the digit")
            self.count("no_tip")
            return

        def along(i):
            ks = [k for k, L in enumerate(chain) if r.match.get(L) is not None and r.match.get(L) == self.inst_group[i]]
            return max(ks) if ks else -1

        def doc(i):
            it = G_.inst[i]
            return min(it["slots"].values())
        i = sorted(mods, key=lambda i: (-along(i), doc(i)))[0]
        if len(mods) > 1:
            self.count("tip_modules_surplus", len(mods) - 1)
            r.reports.append("%d tactile modules in the digit: the one on its last link (else the farthest along the "
                             "chain, else the first in the document) is read" % len(mods))
        r.tip_group = self.inst_group[i]
        r.tip_point = G.apply(self.A @ G_.inst[i]["W"], tloc)
        if r.match.get(last) != r.tip_group:
            self.count("tip_link")
            r.reports.append("the tactile module rides on another link than the last (tip_link)")

    def read_digit(self, r):
        if r.candidate is None:
            r.valid = False
            r.invalid_reason = "no candidate"
            return
        self.match_links(r)
        why = self.validity(r)
        if why:
            r.valid = False
            r.invalid_reason = why
            self.count("digits_invalid")
            r.reports.append("invalid: " + why)
            return
        r.valid = True
        self.read_hinges(r)
        self.read_loop(r)
        self.read_tip(r)
        self.build_tree(r)

    # ---------------------------------------------------------------- posing
    def build_tree(self, r):
        """The posing order: chain links through their hinges, the bar through the base pivot, the rest rigidly."""
        D = self.spec.digits[r.name]
        spec = self.spec
        posed = {PALM}
        steps = []
        full = [HT.PALM] + list(D["chain"])
        for i in range(1, len(full)):
            L, P = full[i], full[i - 1]
            m, mp = r.match.get(L), r.match.get(P)
            if m is None or m in posed:
                continue
            h = spec.hinge_between(P, L)
            hr = r.hinges[h["id"]]
            if mp in posed and hr["status"] in ("present", "undriven") and hr["line"] is not None:
                steps.append((m, mp, h["id"]))
                posed.add(m)
        if D["bar"]:
            m = r.match.get(D["bar"])
            hb = D["loop"]["base_pivot"]
            mp = r.match.get(spec.hinges[hb]["parent"])
            if m is not None and m not in posed and mp in posed and r.loop_closed:
                steps.append((m, mp, hb))
                posed.add(m)
        pool = sorted(set(r.groups))
        for _ in range(len(pool) + 1):
            grew = False
            for e in self.edges:
                ea = PALM if e.a in self.palm else e.a
                eb = PALM if e.b in self.palm else e.b
                for x, y in ((ea, eb), (eb, ea)):
                    if x in posed and y not in posed and y in pool:
                        steps.append((y, x, None))
                        posed.add(y)
                        grew = True
            if not grew:
                break
        r.tree = steps

    def pose(self, r, q):
        """{group (PALM for the palm): motion from Z} of digit r at the independent angles q (each in the submitted
        hinge's own sense); raises LoopError when the submitted loop cannot close."""
        spec = self.spec
        D = spec.digits[r.name]
        T = {PALM: np.eye(4)}
        phi = chi = 0.0
        if r.loop_closed:
            h3 = r.hinges[D["loop"]["driver"]]
            q3 = float(q.get(D["loop"]["driver"], 0.0)) if h3["status"] == "present" else 0.0
            phi, chi = r.loop.solve(q3)
        for g, gp, hid in r.tree:
            M = np.eye(4)
            if hid is not None:
                h, hr = spec.hinges[hid], r.hinges[hid]
                a = 0.0
                if h["kind"] == "servo":
                    a = float(q.get(hid, 0.0)) if (hr["status"] == "present" and h["independent"]) else 0.0
                elif r.loop_closed and hid == D["loop"]["distal"]:
                    a = phi
                elif r.loop_closed and hid == D["loop"]["base_pivot"]:
                    a = chi
                M = rot_about(hr["line"][0], hr["sense"], a)
            T[g] = T[gp] @ M
        for g in r.groups:
            if g not in T:
                T[g] = np.eye(4)
        return T


# ============================================================================ the criteria
class Mechanism(object):
    def __init__(self, G_):
        self.G = G_
        self.rm = ref_mech(G_)
        self.spec = self.rm.spec
        self.mc = self.rm.mc
        self.M = self.mc["mechanism"]
        self.stats = G_.stats
        self.diags = G_.diags
        self.res = G_.res

    def run(self):
        t0 = time.time()
        W = Walk(self.G, self.rm)
        W.collect()
        W.read_joints()
        W.pass_through()
        W.pin_edges()
        W.candidates()
        W.slots()
        for d in self.spec.order:
            r = W.readings[d]
            try:
                W.read_digit(r)
            except Exception as exc:  # noqa: BLE001 - this digit cannot be read: it scores 0, the others go on
                G.item_error(self.stats, self.diags, "the walk of the %s" % d, exc)
                r.valid = False
                r.invalid_reason = "error"
        self.W = W
        self.stats["t_walk_s"] = time.time() - t0
        for k, v in W.stats.items():
            self.stats[k] = float(v)
        for n in W.notes[:12]:
            G.local_note(self.diags, n)
        self.draw_poses()
        t1 = time.time()
        self.true_collisions()
        self.stats["t_true_poses_s"] = time.time() - t1
        out = {}
        t2 = time.time()
        out["kinematic"] = self.guard("kinematic", self.kinematic)
        self.stats["t_kinematic_s"] = time.time() - t2
        t2 = time.time()
        out["mass"] = self.guard("mass", self.mass)
        out["fit"] = self.guard("fit", self.fit)
        self.stats["t_mass_fit_s"] = time.time() - t2
        t2 = time.time()
        out["interference"] = self.guard("interference", self.interference)
        self.stats["t_interference_s"] = time.time() - t2
        self.report()
        return out

    def guard(self, what, fn):
        try:
            return float(G.finite(fn()))
        except Exception as exc:  # noqa: BLE001 - an item that fails scores 0 (grader_item_errors)
            G.item_error(self.stats, self.diags, "the %s criterion" % what, exc)
            return 0.0

    # ---------------------------------------------------------------- poses
    def draw_poses(self):
        hi, lo, pinned = pose_seeds()
        self.stats["pose_seed_hi"] = float(hi)
        self.stats["pose_seed_lo"] = float(lo)
        self.stats["pose_seed_pinned"] = 1.0 if pinned else 0.0
        rng = np.random.default_rng(np.random.SeedSequence([hi, lo]))
        n = int(self.M["poses_per_digit"])
        margin = math.radians(float(self.M["toggle_margin_deg"]))
        self.poses = {}
        for d in HT.DIGITS:                       # a fixed order, whatever the reference holds
            if d not in self.spec.digits:
                continue
            D = self.spec.digits[d]
            ids = D["independent"]
            u = halton(n, len(ids), rng)
            lo_ = np.array([self.spec.hinges[h]["limits"][0] for h in ids])
            hi_ = np.array([self.spec.hinges[h]["limits"][1] for h in ids])
            Q = lo_ + u * (hi_ - lo_)
            if D["loop"] is not None:              # no pose within toggle_margin of the unit's toggles
                L = self.rm.loops[d]
                k = ids.index(D["loop"]["driver"])
                for row in Q:
                    for _ in range(64):
                        try:
                            L.solve(row[k] + math.copysign(margin, row[k]) if row[k] != 0 else 0.0)
                            break
                        except LoopError:
                            row[k] *= 0.5
            self.poses[d] = [dict(zip(ids, row)) for row in Q]

    def true_collisions(self):
        """Per digit, the poses at which the true digit is collision-free (with itself and the palm); the poses at
        which the true hand is collision-free between its digits as well."""
        rm, spec = self.rm, self.spec
        ic = self.mc["interference"]
        key = (self.stats["pose_seed_hi"], self.stats["pose_seed_lo"])
        cached = rm.cache.get(("true", key))
        if cached is not None:
            self.kept, self.kept_between = cached
            return
        rm.bodies()
        self.kept = {}
        n = int(self.M["poses_per_digit"])
        for d in spec.order:
            D = spec.digits[d]
            links = [HT.PALM] + list(D["chain"]) + ([D["bar"]] if D["bar"] else [])
            pairs = [(links[i], links[k]) for i in range(len(links)) for k in range(i + 1, len(links))]
            keep = []
            for k, q in enumerate(self.poses[d]):
                rm._bodies.reset()
                try:
                    T = rm.link_transforms(q)
                except LoopError:
                    continue
                if not rm.collides(pairs, T, ("w", d, k), ic):
                    keep.append(k)
            self.kept[d] = keep
        between = []
        dl = {d: list(spec.digits[d]["chain"]) + ([spec.digits[d]["bar"]] if spec.digits[d]["bar"] else [])
              for d in spec.order}
        cross = [(a, b) for i, d1 in enumerate(spec.order) for d2 in spec.order[i + 1:] for a in dl[d1] for b in dl[d2]]
        if len(spec.order) > 1:
            for k in range(n):
                if not all(k in self.kept[d] for d in spec.order):
                    continue
                q = {}
                for d in spec.order:
                    q.update(self.poses[d][k])
                rm._bodies.reset()
                try:
                    T = rm.link_transforms(q)
                except LoopError:
                    continue
                if not rm.collides(cross, T, ("b", k), ic):
                    between.append(k)
        self.kept_between = between
        rm._bodies.reset()
        rm.cache[("true", key)] = (self.kept, self.kept_between)
        self.stats["collision_booleans_true"] = float(rm._bodies.n_bool)
        self.stats["t_collision_booleans_true_s"] = rm._bodies.t_bool
        for d in spec.order:
            self.stats["true_poses_kept_%s" % d] = float(len(self.kept[d]))
        self.stats["true_poses_kept_between"] = float(len(between))

    # ---------------------------------------------------------------- floors
    def fk(self, d):
        r = self.W.readings[d]
        return getattr(r, "fk", 0.0) if r.valid else 0.0

    def palm_fk(self):
        return float(np.mean([self.fk(d) for d in self.spec.order])) if self.spec.order else 0.0

    def link_fk(self, link):
        dg = self.spec.digit_of_link.get(link)
        return self.palm_fk() if dg is None else self.fk(dg)

    # ---------------------------------------------------------------- kinematic
    def kinematic(self):
        vals = []
        self.digit_items = {}
        nh, ni = unlocated_items(self.spec)
        self.stats["hinges_not_located"] = float(nh)
        self.stats["hinge_items_not_graded"] = float(ni)
        for d in self.spec.order:
            r = self.W.readings[d]
            r.fk = 0.0
            try:
                items = self.digit_kinematic(r) if r.valid else None
            except Exception as exc:  # noqa: BLE001 - this digit's kinematic items 0
                G.item_error(self.stats, self.diags, "the kinematic items of the %s" % d, exc)
                items = None
            if items is None:
                vals.append(0.0)
                self.digit_items[d] = None
                continue
            self.digit_items[d] = items
            vals.append(items["score"])
        return float(np.mean(vals)) if vals else 0.0

    def digit_kinematic(self, r):
        spec, M, d = self.spec, self.M, r.name
        D = spec.digits[d]
        out = {}
        loc = lambda hid: spec.hinges[hid].get("located", True)  # noqa: E731     the reference's located mark
        n_out = 0
        # axes
        ax = []
        for hid in D["hinges"]:
            h, hr = spec.hinges[hid], r.hinges[hid]
            if not loc(hid):
                n_out += 1
                continue
            if hr["status"] != "present" or hr["line"] is None:
                ax.append(0.0)
                continue
            p, dd = hr["line"]
            ax.append(G.band(point_line(h["centre"], p, dd), M["axis_mm"]) * G.band(line_angle_deg(h["axis"], dd), M["axis_deg"]))
        out["axes"] = float(np.mean(ax)) if ax else None
        out["axes_each"] = ax
        # key lengths
        lens = []
        chain = spec.chain_hinges(d)

        def sub_line(hid):
            hr = r.hinges[hid]
            return hr["line"] if hr["status"] in ("present", "undriven") and hr["line"] is not None else None

        def true_line(hid):
            return spec.hinges[hid]["centre"], spec.hinges[hid]["axis"]
        for a, b in zip(chain[:-1], chain[1:]):
            if not (loc(a) and loc(b)):
                n_out += 2
                continue
            la, lb = sub_line(a), sub_line(b)
            ta, tb = true_line(a), true_line(b)
            if la is None or lb is None:
                lens += [0.0, 0.0]
                continue
            lens.append(G.band(common_normal(la[0], la[1], lb[0], lb[1], M["parallel_deg"]) - common_normal(ta[0], ta[1], tb[0], tb[1], M["parallel_deg"]),
                               M["length_mm"]))
            lens.append(G.band(line_angle_deg(la[1], lb[1]) - line_angle_deg(ta[1], tb[1]), M["length_deg"]))
        if D["loop"] is not None:
            lp = D["loop"]
            for a, b in ((lp["driver"], lp["base_pivot"]), (lp["distal"], lp["link_pin"]), (lp["base_pivot"], lp["link_pin"])):
                if not (loc(a) and loc(b)):
                    n_out += 1
                    continue
                la, lb = sub_line(a), sub_line(b)
                ta, tb = true_line(a), true_line(b)
                if la is None or lb is None:
                    lens.append(0.0)
                    continue
                lens.append(G.band(common_normal(la[0], la[1], lb[0], lb[1], M["parallel_deg"]) - common_normal(ta[0], ta[1], tb[0], tb[1], M["parallel_deg"]),
                                   M["length_mm"]))
        last = chain[-1]
        ll = sub_line(last)
        if not loc(last):
            n_out += 1
        elif ll is None or r.tip_point is None:
            lens.append(0.0)
        else:
            tl = true_line(last)
            lens.append(G.band(point_line(r.tip_point, ll[0], ll[1]) - point_line(D["tip"]["point_mm"], tl[0], tl[1]),
                               M["length_mm"]))
        out["lengths"] = float(np.mean(lens)) if lens else None
        # fingertip over poses
        kept = self.kept.get(d) or list(range(len(self.poses[d])))
        errs = []
        bad = float(M["unclosed_error_mm"])
        for k in kept:
            q = self.poses[d][k]
            Tt = self.rm.link_transforms(q)
            tt = self.rm.tip(d, Tt)
            if r.tip_point is None:
                errs.append(bad)
                continue
            try:
                Ts = self.W.pose(r, q)
            except LoopError:
                errs.append(bad)
                continue
            errs.append(float(np.linalg.norm(G.apply(Ts[self._tip_key(r)], r.tip_point) - tt)))
        errs = np.asarray(errs, float)
        med = float(np.median(errs)) if len(errs) else bad
        p90 = float(np.percentile(errs, 90)) if len(errs) else bad
        wm = float(M["fingertip_w_median"])
        out["fingertip"] = wm * G.band(med, M["fingertip_median_mm"]) + (1.0 - wm) * G.band(p90, M["fingertip_p90_mm"])
        out["fingertip_median_mm"], out["fingertip_p90_mm"] = med, p90
        r.fk = G.band(med, self.mc["fk"]) if r.tip_point is not None else 0.0
        # coupling law
        if D["loop"] is not None:
            if not r.loop_closed:
                out["coupling"] = 0.0
                out["coupling_mean_deg"] = None
            else:
                h3 = spec.hinges[D["loop"]["driver"]]
                qs = np.linspace(h3["limits"][0], h3["limits"][1], int(M["coupling_angles"]))
                zero = float(M["coupling_deg"][1])
                e = []
                for qt in qs:
                    phit = self.rm.loops[d].solve(float(qt))[0]
                    try:
                        phis = r.loop_sign * r.loop.solve(r.loop_sign * float(qt))[0]
                        e.append(min(zero, abs(math.degrees(phis - phit))))
                    except LoopError:
                        e.append(zero)
                out["coupling_mean_deg"] = float(np.mean(e))
                out["coupling"] = G.band(out["coupling_mean_deg"], M["coupling_deg"])
        # range
        rg = []
        tol = math.radians(float(M["range_tol_deg"]))
        rg_out = 0
        for hid in D["independent"]:
            h, hr = spec.hinges[hid], r.hinges[hid]
            if not loc(hid):
                n_out += 1
                rg_out += 1
                continue
            if hr["status"] != "present" or hr["declared"] is None:
                rg.append(0.0)
                continue
            lo, hi = hr["declared"]
            tlo, thi = h["limits"]
            lo = tlo if abs(lo - tlo) <= tol else lo
            hi = thi if abs(hi - thi) <= tol else hi
            rg.append(interval_iou((lo, hi), (tlo, thi)))
        out["range"] = float(np.mean(rg)) if rg else (None if rg_out else 0.0)
        w = M["weights_thumb"] if D["kind"] == "thumb" else M["weights_finger"]
        keys = ("axes", "lengths", "fingertip", "coupling", "range")
        if all(out.get(k, 0.0) is not None for k in keys):
            out["score"] = sum(float(w[k]) * float(out.get(k, 0.0)) for k in keys)
        else:                       # a component without a graded item (every one not located) drops out with its weight
            ks = [k for k in keys if out.get(k, 0.0) is not None]
            tw = sum(float(w[k]) for k in ks)
            out["score"] = sum(float(w[k]) * float(out.get(k, 0.0)) for k in ks) / tw if tw > 0 else 0.0
        out["items_not_graded"] = n_out
        return out

    def _tip_key(self, r):
        return PALM if r.tip_group in self.W.palm else r.tip_group

    # ---------------------------------------------------------------- mass
    def mass(self):
        spec, ref, G_, W = self.spec, self.G.ref, self.G, self.W
        Mc = self.mc["mass"]
        excl = set(Mc["excluded_types"])
        dens_x = float(Mc["extra_density_g_mm3"])
        # submitted mass per group, aligned frame
        assigned = {}
        for pid, it in G_.inst_of.items():
            assigned[id(it)] = pid
        gm = {}

        def add(g, m, c):
            key = PALM if g in W.palm else g
            a = gm.setdefault(key, [0.0, np.zeros(3)])
            a[0] += m
            a[1] += m * c
        given_sids = set()
        for i, it in enumerate(G_.inst):
            if id(it) not in assigned:
                continue
            given_sids.update(it["slots"].values())
            if it["type"] in excl:
                continue
            typ = ref.types[it["type"]]
            g = W.inst_group.get(i)
            if g is not None:
                add(g, typ["mass_g"], G.apply(self.G.A @ it["W"], typ["centroid"]))
        tagged = set(id(s) for s in G_.part_of.values())
        for s in G_.sub.solids:
            g = G_.groups.group(s["node"])
            if g is None or s["sid"] in given_sids:
                continue
            if "match" in s and not s.get("demoted"):
                # a recognised given solid outside the assigned instances: a pin or bearing beyond the count is left
                # out like the counted ones; any other is an extra solid
                if ref.classes[s["match"]["cls"]]["types"][0] in excl:
                    continue
            if id(s) in tagged:
                rho = ref.kinds[ref.parts[s["tag"]]["kind"]]["density"]
            else:
                rho = dens_x
            add(g, s["V"] * rho, G.apply(self.G.A, s["com"]))
        num = den = 0.0
        self.link_mass_items = {}
        for d in [None] + list(spec.order):
            if d is None:
                links = [HT.PALM]
            else:
                D = spec.digits[d]
                links = list(D["chain"]) + ([D["bar"]] if D["bar"] else [])
            r = W.readings.get(d) if d is not None else None
            for L in links:
                w = float(Mc["palm_weight"] if L == HT.PALM else Mc["link_weight"])
                den += w
                if d is not None and (r is None or not r.valid):
                    self.link_mass_items[L] = 0.0
                    continue
                m = PALM if L == HT.PALM else r.match.get(L)
                if m is None:
                    self.link_mass_items[L] = 0.0
                    continue
                run = [L2 for L2, m2 in (self._all_matches().items()) if m2 == m]
                if L not in run:
                    run.append(L)
                rmass, rcom = self.rm.link_mass(run, excl)
                got = gm.get(m)
                if got is None or got[0] <= 0 or rmass <= 0:
                    self.link_mass_items[L] = 0.0
                    continue
                smass, scom = got[0], got[1] / got[0]
                e_m = 100.0 * abs(smass / rmass - 1.0)
                e_c = float(np.linalg.norm(scom - rcom))
                sc = float(Mc["w_mass"]) * G.band(e_m, Mc["mass_pct"]) + float(Mc["w_com"]) * G.band(e_c, Mc["com_mm"])
                self.link_mass_items[L] = sc
                num += w * sc * self.link_fk(L)
        return num / den if den > 0 else 0.0

    def _all_matches(self):
        if not hasattr(self, "_matches"):
            out = {HT.PALM: PALM}
            for d in self.spec.order:
                r = self.W.readings[d]
                if r.valid:
                    for L, m in r.match.items():
                        if L != HT.PALM and m is not None:
                            out[L] = m
            self._matches = out
        return self._matches

    # ---------------------------------------------------------------- fit
    def _placed_part(self, s):
        if s is None:
            return None
        if "fit_placed" not in s:
            g = s.get("geo")
            if g is None:
                try:
                    g = self.G.sub.geo(s, int(self.G.cfg["sampling"]["seed"]) + 20000 + s["sid"])
                except Exception:  # noqa: BLE001
                    g = None
            s["fit_placed"] = G.Placed(g, self.G.A, self.G.cfg) if g is not None else None
        return s["fit_placed"]

    def _placed_inst(self, it, W=None):
        typ = self.G.ref.types[it["type"]]
        W = it["W"] if W is None else W
        return [G.Placed(c["geo"], self.G.A @ W, self.G.cfg) for c in typ["cons"] if c["geo"] is not None]

    def extras(self):
        G_ = self.G
        taken = set()
        for it in G_.inst_of.values():
            taken.update(it["slots"].values())
        tagged = set(id(s) for s in G_.part_of.values())
        out = []
        for s in G_.sub.solids:
            if id(s) in tagged or s["sid"] in taken:
                continue
            if "match" in s and not s.get("demoted"):
                cl = G_.ref.classes[s["match"]["cls"]]
                p = G.Placed(cl["rep"]["geo"], G_.A @ s["match"]["T"], G_.cfg) if cl["rep"]["geo"] is not None else None
            else:
                p = self._placed_part(s)
            if p is not None:
                out.append(p)
        return out

    def fit(self):
        G_, ref, cfg = self.G, self.G.ref, self.G.cfg
        F = self.mc["fit"]
        t0 = time.time()
        fd = self.rm.fit_data()
        self.stats["t_fit_reference_s"] = time.time() - t0
        disk = float(cfg["sampling"]["disk_mm"])
        extras = self.extras()
        parts = {t: self._placed_part(s) for t, s in G_.part_of.items()}
        insts = {pid: self._placed_inst(it) for pid, it in G_.inst_of.items()}
        cache = {}

        def common(key, A_, B_):
            if key not in cache:
                v = 0.0
                for a in A_:
                    for b in B_:
                        if a is not None and b is not None and G.aabb_overlap(a.box, b.box):
                            v += G.common_volume(a.man, b.man)
                cache[key] = v
            return cache[key]
        wsum = wacc = 0.0
        self.fit_items = {}
        tw = cfg["assembly"]["type_weights"]
        for pid in sorted(ref.placements):
            pl = ref.placements[pid]
            w = float(tw.get(pl["type"], 1.0))
            wsum += w
            it = G_.inst_of.get(pid)
            if it is None or not self._digit_ok(self.spec.link_of_placement.get(pid)):
                self.fit_items[pid] = 0.0
                continue
            mine = insts[pid]
            vol = sum(m.volume for m in mine)
            excess = 0.0
            for t, p in parts.items():
                if p is None or not any(G.aabb_overlap(m.box, p.box) for m in mine):
                    continue
                excess += max(0.0, common(("i", pid, t), mine, [p]) - fd["common_pt"].get((pid, t), 0.0))
            for k, e in enumerate(extras):
                if any(G.aabb_overlap(m.box, e.box) for m in mine):
                    excess += common(("ix", pid, k), mine, [e])
            over = G.band(100.0 * excess / max(vol, 1e-9), F["overlap_pct"])
            gap = 1.0
            pt = fd["patch"].get(pid)
            if pt is not None and len(pt["local"]):
                best = np.inf
                tl = np.asarray(pt["tag"])
                for Wa in it.get("alt", [it["W"]]):
                    P = G.apply(G_.A @ Wa, pt["local"])
                    dist = np.full(len(P), np.inf)
                    for t in sorted(set(pt["tag"])):
                        p = parts.get(t)
                        sel = tl == t
                        if p is None:
                            continue
                        D_, N_, tree = p.dense()
                        if tree is None:
                            continue
                        dist[sel] = G.surface_distances(P[sel], D_, N_, tree, disk)
                    best = min(best, float(np.median(dist)))
                gap = G.band(best, F["gap_mm"])
            sc = gap * over
            self.fit_items[pid] = sc
            wacc += w * sc * self.link_fk(self.spec.link_of_placement.get(pid))
        given = wacc / wsum if wsum > 0 else 0.0
        mvals = []
        tags = sorted(ref.parts)
        for t in tags:
            p = parts.get(t)
            if p is None or not self._digit_ok(self.spec.link_of_tag.get(t)):
                self.fit_items[t] = 0.0
                mvals.append(0.0)
                continue
            excess = 0.0
            for t2, p2 in parts.items():
                if t2 == t or p2 is None or not G.aabb_overlap(p.box, p2.box):
                    continue
                key = ("m",) + tuple(sorted((t, t2)))
                excess += max(0.0, common(key, [p], [p2]) - fd["common_tt"].get(tuple(sorted((t, t2))), 0.0))
            for pid, mine in insts.items():
                if not any(G.aabb_overlap(m.box, p.box) for m in mine):
                    continue
                excess += max(0.0, common(("i", pid, t), mine, [p]) - fd["common_pt"].get((pid, t), 0.0))
            for k, e in enumerate(extras):
                if G.aabb_overlap(p.box, e.box):
                    excess += common(("mx", t, k), [p], [e])
            sc = G.band(100.0 * excess / max(p.volume, 1e-9), F["overlap_pct"])
            self.fit_items[t] = sc
            mvals.append(sc * self.link_fk(self.spec.link_of_tag.get(t)))
        made = float(np.mean(mvals)) if mvals else 0.0
        self.stats["fit_given"] = given
        self.stats["fit_made"] = made
        return float(F["w_given"]) * given + float(F["w_made"]) * made

    def _digit_ok(self, link):
        d = self.spec.digit_of_link.get(link)
        if d is None:
            return True
        r = self.W.readings.get(d)
        return r is not None and r.valid

    # ---------------------------------------------------------------- interference
    def sub_bodies(self):
        """The submitted collision bodies: each assigned digit's groups and the palm (all palm groups as one body)."""
        G_, W = self.G, self.W
        items = {}
        given_of = {}
        for i, it in enumerate(G_.inst):
            for sid in it["slots"].values():
                given_of[sid] = it

        def add(g, geo, T0):
            key = PALM if g in W.palm else g
            items.setdefault(key, []).append((geo, T0))
        wanted = set(W.palm)
        for d in self.spec.order:
            r = W.readings[d]
            if r.valid:
                wanted.update(r.groups)
        for s in G_.sub.solids:
            g = G_.groups.group(s["node"])
            if g not in wanted:
                continue
            if s["sid"] in given_of and "match" in s:
                cl = G_.ref.classes[s["match"]["cls"]]
                add(g, cl["rep"]["geo"], G_.A @ s["match"]["T"])
            else:                                   # made parts and every other solid (obstacles)
                p = self._placed_part(s)
                if p is not None:
                    add(g, p.geo, G_.A)
        return Bodies(items)

    def interference(self):
        W, spec = self.W, self.spec
        ic = self.mc["interference"]
        B = self.sub_bodies()
        adj = set()
        for e in W.edges:
            a = PALM if e.a in W.palm else e.a
            b = PALM if e.b in W.palm else e.b
            adj.add((a, b))
            adj.add((b, a))
        zcache = {}

        def ov_z(a, b):
            key = tuple(sorted((str(a), str(b))))
            if key not in zcache:
                zcache[key] = B.overlap(a, b, np.eye(4), np.eye(4), None, None)
            return zcache[key]

        def fails(pairs, T, kk):
            for a, b in pairs:
                if a not in B.items or b not in B.items or not B.near(a, b, T[a], T[b]):
                    continue
                v = B.overlap(a, b, T[a], T[b], (kk, a), (kk, b)) - ov_z(a, b)
                thr = float(ic["adjacent_ratio"] if (a, b) in adj else ic["other_ratio"])
                if v > thr * max(1e-9, min(B.volume[a], B.volume[b])):
                    return True
            return False
        within, posed = {}, {}
        for d in spec.order:
            r = W.readings[d]
            if not r.valid:
                within[d] = 0.0
                continue
            bodies = [PALM] + [g for g in r.groups if g not in W.palm]
            pairs = [(bodies[i], bodies[k]) for i in range(len(bodies)) for k in range(i + 1, len(bodies))]
            kept = self.kept.get(d, [])
            ok = 0
            for k in kept:
                B.reset()
                try:
                    T = W.pose(r, self.poses[d][k])
                except LoopError:
                    continue
                posed[(d, k)] = T
                if not fails(pairs, T, ("w", d, k)):
                    ok += 1
            within[d] = ok / float(len(kept)) if kept else 1.0
        between = {d: 1.0 for d in spec.order}
        valid = [d for d in spec.order if W.readings[d].valid]
        if len(valid) > 1 and self.kept_between:
            counts = {d: 0 for d in valid}
            for k in self.kept_between:
                T = {PALM: np.eye(4)}
                bad = False
                for d in valid:
                    r = W.readings[d]
                    try:
                        Td = posed.get((d, k)) or W.pose(r, self.poses[d][k])
                    except LoopError:
                        bad = True
                        break
                    for g in r.groups:
                        if g not in W.palm:
                            T[g] = Td[g]
                if bad:
                    continue
                B.reset()
                for d in valid:
                    mine = [g for g in W.readings[d].groups if g not in W.palm]
                    others = [g for d2 in valid if d2 != d for g in W.readings[d2].groups if g not in W.palm]
                    pairs = [(a, b) for a in mine for b in others]
                    if not fails(pairs, T, ("b", k)):
                        counts[d] += 1
            for d in valid:
                between[d] = counts[d] / float(len(self.kept_between))
        self.stats["collision_booleans_submission"] = float(B.n_bool)
        self.stats["t_collision_booleans_submission_s"] = B.t_bool
        vals = []
        self.interference_items = {}
        for d in spec.order:
            r = W.readings[d]
            sc = (float(ic["w_within"]) * within.get(d, 0.0) + float(ic["w_between"]) * between.get(d, 0.0)) if r.valid else 0.0
            self.interference_items[d] = {"within": within.get(d, 0.0), "between": between.get(d, 0.0), "score": sc}
            vals.append(sc * self.fk(d))
        return float(np.mean(vals)) if vals else 0.0

    # ---------------------------------------------------------------- reporting
    def report(self):
        st = self.stats
        for k, v in metric_defaults(self.spec).items():
            if k not in st:
                st[k] = v
        out = {}
        for d in self.spec.order:
            r = self.W.readings[d]
            it = (self.digit_items or {}).get(d) if hasattr(self, "digit_items") else None
            rec = {"valid": bool(r.valid), "reason": r.invalid_reason, "first_hinge": r.first_key,
                   "groups": len(r.groups), "loop_closed": bool(r.loop_closed), "fk": float(self.fk(d)),
                   "reports": r.reports[:20],
                   "links": {L.split(".", 1)[-1]: (None if m is None else ("palm" if m == PALM else "group"))
                             for L, m in r.match.items() if L != HT.PALM},
                   "hinges": {hid.split(".", 1)[1]: hr["status"] for hid, hr in r.hinges.items()}}
            if it:
                rec["items"] = {k: it[k] for k in ("axes", "lengths", "fingertip", "coupling", "range", "score",
                                                   "fingertip_median_mm", "fingertip_p90_mm", "coupling_mean_deg")
                                if k in it}
                st["fingertip_median_mm_%s" % d] = G.finite(it["fingertip_median_mm"], -1.0)
                st["fingertip_p90_mm_%s" % d] = G.finite(it["fingertip_p90_mm"], -1.0)
                st["kinematic_%s" % d] = it["score"]
            if hasattr(self, "interference_items") and d in self.interference_items:
                rec["interference"] = self.interference_items[d]
            st["digit_valid_%s" % d] = 1.0 if r.valid else 0.0
            st["digit_fk_%s" % d] = float(self.fk(d))
            st["digit_assigned_%s" % d] = 1.0 if r.candidate is not None else 0.0
            st["loop_closed_%s" % d] = 1.0 if r.loop_closed else 0.0
            for hid, hr in r.hinges.items():
                for s in HINGE_STATUSES:
                    st["hinge_%s_%s" % (hid.replace(".", "_"), s)] = 1.0 if hr["status"] == s else 0.0
            out[d] = rec
            if r.reports:
                G.local_note(self.diags, "%s: %s" % (d, "; ".join(r.reports[:4])))
        self.res["digits"] = out
        st["digit_floor_mean"] = self.palm_fk()


def mechanism(grading):
    """The four mechanism criteria of a grading (grade.Grading after its rigid items)."""
    m = Mechanism(grading)
    grading.mech = m
    return m.run()
