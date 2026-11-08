"""Per-episode randomisation, the work order, faults, milestones and the work score (PUBLIC CODE).

Everything here is a pure function of (seed, config) and of the scene's physical state read through the `Scene`
interface in `physics.py`; nothing on the wire is an input.

The work order: K pairs. Pair k = two shoes (mirror images, the same parameters), a lace per shoe (pre-threaded
through row 1 and tacked there), one box blank. The next pair arrives the moment the previous box is shipped. Fault
drawn per shoe: `on_side` (the shoe lands lying on its side). (`uneven_tails`, an off-centre threading, is retired:
with the row-1 tack a tail cannot be evened out; `fault_uneven_p` is 0.)

Milestones per pair (credits sum to 1.0), judged on the physical state when the box is shipped, else at the end:
    lacing_a, lacing_b   0.30 each: rows laced in order from row 2 (both ends, alternating sides, every ring
                         entered from the OUTSIDE, the lace returning out of the cavity through the throat slot
                         between passes, counted consecutively until the first miss) / (rows - 1), times 0.7 +
                         0.3 x the snug fraction of those rows; 0 if an aglet was left inside the shoe (in the
                         cavity and never seen brought back out after its last pass) or a defective lace is still
                         through it
    erected              0.12: the four walls locked
    packed               0.13: both shoes inside the erected box, sole down, heel-to-toe, not resting on a wall
    closed               0.10: lid locked with both shoes inside
    shipped              0.05: the closed box's floor centre across the outfeed line
The work order is a LACED pair in a shipped box: packed, closed and shipped are credited in proportion to the pair's
laced shoes (each credit times (laced shoes) / 2, a shoe counting as laced when every row is laced, snug or not), so a
pair boxed unlaced earns the erected credit only. A safety violation voids the credit of the pair that was at work when
it happened (not the whole episode).
"""

import math

import numpy as np

__all__ = ["CREDITS", "LACED_SCALED", "draw_work_order", "lacing_score", "pair_milestones", "WorkOrder"]

CREDITS = {"lacing_a": 0.30, "lacing_b": 0.30, "erected": 0.12, "packed": 0.13, "closed": 0.10, "shipped": 0.05}
LACED_SCALED = ("packed", "closed", "shipped")        # credited times the fraction of the pair's shoes that are laced

UPPER_COLOURS = [  # name, rgba
    ("red", (0.80, 0.12, 0.10, 1.0)), ("blue", (0.12, 0.30, 0.85, 1.0)), ("green", (0.10, 0.60, 0.25, 1.0)),
    ("orange", (0.95, 0.55, 0.10, 1.0)), ("purple", (0.50, 0.15, 0.70, 1.0)), ("teal", (0.05, 0.60, 0.60, 1.0)),
    ("black", (0.08, 0.08, 0.10, 1.0)), ("navy", (0.10, 0.12, 0.40, 1.0)),
]
SOLE_COLOURS = [("white", (0.92, 0.92, 0.90, 1.0)), ("black", (0.10, 0.10, 0.10, 1.0)), ("gum", (0.75, 0.55, 0.30, 1.0)),
                ("grey", (0.55, 0.55, 0.55, 1.0))]
LACE_COLOURS = [("white", (0.95, 0.95, 0.92, 1.0)), ("black", (0.06, 0.06, 0.06, 1.0)), ("yellow", (0.95, 0.85, 0.15, 1.0)),
                ("pink", (0.95, 0.35, 0.60, 1.0)), ("lime", (0.55, 0.90, 0.20, 1.0)), ("sky", (0.35, 0.70, 0.95, 1.0))]


def draw_work_order(seed, cfg):
    """The episode's batch (one shoe model, box and lid for every pair) and its pairs (colours, arrival poses, faults),
    all from the seed."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0x5A0E])
    zones = cfg["zones"]
    L = float(rng.uniform(0.17, 0.21))
    batch = {"length": L, "width": float(rng.uniform(0.070, 0.080)), "n_rows": 2,
             "row_spacing": float(rng.uniform(0.016, 0.019)), "throat_gap": float(rng.uniform(0.024, 0.028)),
             "toe_cap_len": 0.045, "heel_len": 0.025, "first_row_offset": 0.012, "cavity_depth": 0.045}
    carton = {"L": L + float(rng.uniform(0.035, 0.05)), "W": 2 * batch["width"] + float(rng.uniform(0.012, 0.02)),
              "H": float(rng.uniform(0.094, 0.104))}
    tx, ty = zones["infeed"]["centre"]
    hx, hy = zones["infeed"]["half"]
    bx, by = zones["blank"]["centre"]
    pairs = []
    used = set()
    for k in range(int(cfg.get("pairs_per_episode", 3))):
        uppers = [c for c in UPPER_COLOURS if c[0] not in used] or UPPER_COLOURS
        upper = uppers[int(rng.integers(len(uppers)))]
        used.add(upper[0])
        sole = SOLE_COLOURS[int(rng.integers(len(SOLE_COLOURS)))]
        laces = [c for c in LACE_COLOURS if c[0] != upper[0]]
        lace = laces[int(rng.integers(len(laces)))]
        logo = UPPER_COLOURS[int(rng.integers(len(UPPER_COLOURS)))][1]
        shoes = []
        for s in range(2):
            on_side = bool(rng.random() < cfg.get("fault_on_side_p", 0.25))
            uneven = bool(rng.random() < cfg.get("fault_uneven_p", 0.0))
            x = tx - 0.115 + 0.23 * s + float(rng.uniform(-0.01, 0.01))
            y = ty + 0.10 + float(rng.uniform(-0.015, 0.015))
            yaw = float(rng.choice([0.0, math.pi]) + rng.uniform(-0.25, 0.25))
            tails = (0.5, 0.5)
            if uneven:
                tails = (0.35, 0.65) if rng.random() < 0.5 else (0.65, 0.35)
            shoes.append({"side": "AB"[s], "pos": [x, y], "yaw": yaw, "on_side": on_side, "side_sign": int(rng.choice([-1, 1])),
                          "uneven_tails": uneven, "tails": list(tails)})
        blank = {"pos": [bx + float(rng.uniform(-0.015, 0.015)), by + float(rng.uniform(-0.015, 0.015))],
                 "yaw": float(rng.choice([0.0, math.pi]) + rng.uniform(-0.15, 0.15))}
        # the lid tray, either way round: its centre within +-3 cm (x) and +-3 mm (y) of (tx, ty - 0.125), turned by at
        # most 0.03 rad, so that even the biggest batch's lid (0.28 x 0.20 m) lies >= 3.8 mm inside the tray's near rim
        # (its inner face at y = ty - 0.236) and rests flat on the plate
        lid = {"pos": [tx + float(rng.uniform(-0.03, 0.03)), ty - 0.125 + float(rng.uniform(-0.003, 0.003))],
               "yaw": float(rng.choice([0.0, math.pi]) + rng.uniform(-0.03, 0.03))}
        pairs.append({"index": k, "shoes": shoes, "blank": blank, "lid": lid,
                      "upper_rgba": list(upper[1]), "upper_colour": upper[0], "sole_rgba": list(sole[1]), "sole_colour": sole[0],
                      "lace_rgba": list(lace[1]), "lace_colour": lace[0], "logo_rgba": list(logo)})
    jitter = {"cradle": [float(rng.uniform(-0.02, 0.02)), float(rng.uniform(-0.02, 0.02)), float(rng.uniform(-0.12, 0.12))],
              "light": [float(rng.uniform(0.8, 1.2)) for _ in range(3)]}
    return {"seed": int(seed), "batch": batch, "carton": carton, "pairs": pairs, "jitter": jitter}


# ---------------------------------------------------------------- lacing

def _expected_sequence(n_rows, first_side):
    """Rings a tail must pass after leaving row 1 outward on `first_side`: (row, side, direction) with direction +1 =
    inside -> outside (out), -1 = outside -> inside (in). The throat-return criss-cross: every ring is entered from
    the outside (the lace comes back out of the cavity through the throat slot between two passes), the sides
    alternate starting opposite to the row-1 exit."""
    seq = []
    side = "R" if first_side == "L" else "L"
    for row in range(2, n_rows + 1):
        seq.append((row, side, -1))
        side = "R" if side == "L" else "L"
    return seq


def lacing_score(passes, n_rows, snug, aglets_inside, defective_through):
    """`passes`: the lace's ring passes in order along the lace from aglet A to aglet B, each (row, side, direction,
    t). `snug`: per row (2..n), whether the lace between that row's two rings and the previous row is snug (dict row ->
    bool). Returns (score in [0, 1], rows_credited, detail)."""
    if aglets_inside or defective_through or n_rows < 2:
        return 0.0, 0, {"reason": "aglet inside" if aglets_inside else ("defective lace" if defective_through else "no rows")}
    # split the pass list at row 1: the two passes of row 1 (out on each side) are the reference; everything before the
    # first row-1 pass belongs to tail A (reversed: walk away from row 1), everything after the second to tail B
    r1 = [i for i, p in enumerate(passes) if p[0] == 1]
    if len(r1) != 2 or passes[r1[0]][2] != -1 or passes[r1[1]][2] != +1:
        # along the lace from A to B the lace enters row 1 (as seen from A: outside -> inside) and leaves it (inside ->
        # outside); anything else means row 1 is no longer threaded the delivered way
        return 0.0, 0, {"reason": "row 1 not threaded", "passes": passes}
    tail_a = list(reversed(passes[:r1[0]]))                   # walking from row 1 toward aglet A
    tail_b = passes[r1[1] + 1:]
    side_a, side_b = passes[r1[0]][1], passes[r1[1]][1]
    if side_a == side_b:
        return 0.0, 0, {"reason": "row 1 passes on one side", "passes": passes}
    # directions as walked from row 1 outward: tail A's passes were recorded walking toward row 1, so invert them
    tail_a = [(r, s, -d, t) for (r, s, d, t) in tail_a]
    credited = 0
    exp_a, exp_b = _expected_sequence(n_rows, side_a), _expected_sequence(n_rows, side_b)
    for row in range(2, n_rows + 1):
        i = row - 2
        ok_a = i < len(tail_a) and tail_a[i][:3] == exp_a[i]
        ok_b = i < len(tail_b) and tail_b[i][:3] == exp_b[i]
        if ok_a and ok_b:
            credited += 1
        else:
            break
    if credited == 0:
        return 0.0, 0, {"reason": "row 2 not laced", "tail_a": tail_a[:3], "tail_b": tail_b[:3]}
    snug_frac = float(np.mean([1.0 if snug.get(r, False) else 0.0 for r in range(2, 2 + credited)]))
    score = (credited / float(n_rows - 1)) * (0.7 + 0.3 * snug_frac)
    return float(score), credited, {"snug_fraction": snug_frac, "tail_a": tail_a, "tail_b": tail_b}


# ---------------------------------------------------------------- milestones

def boxed(scene, pair):
    """The pair's carton state and, per shoe, whether it lies in the erected box as packing needs it: inside, sole down
    (within 15 deg), not resting on a wall."""
    k = pair["carton_id"]
    st = scene.carton_state(k)
    inside_ok = []
    for s in range(2):
        sh = pair["shoe_ids"][s]
        c = scene.shoe_centre(sh)
        inside_ok.append(st["erected"] and scene.inside_box(k, c) and scene.upright(sh, max_deg=15.0) and not scene.resting_on_wall(k, sh))
    return st, inside_ok


def pair_milestones(scene, pair, cfg, shipped=False):
    """Evaluate every milestone of a pair from the scene's physical state. Returns {milestone: value in [0, 1]} and the
    credit (sum of value x credit, the boxing milestones after lacing_a / lacing_b scaled by the laced fraction)."""
    ms = {}
    detail = {}
    n_rows = int(scene.batch["n_rows"])
    for s, key in ((0, "lacing_a"), (1, "lacing_b")):
        sh = pair["shoe_ids"][s]
        lace = pair["lace_ids"][s]
        passes, snug = scene.lace_passes(lace, sh)
        inside = scene.aglet_left_inside(lace, sh, (pair.get("brought_out") or {}).get(s, (False, False)))
        score, rows, d = lacing_score(passes, n_rows, snug, inside, False)
        ms[key] = score
        d["rows_credited"] = rows
        detail[key] = d
    st, inside_ok = boxed(scene, pair)
    ms["erected"] = 1.0 if st["erected"] else 0.0
    h0, h1 = scene.shoe_heading(pair["shoe_ids"][0]), scene.shoe_heading(pair["shoe_ids"][1])
    anti = float(np.dot(h0, h1)) <= -math.cos(math.radians(30.0))
    ms["packed"] = 1.0 if (all(inside_ok) and anti) else 0.0
    ms["closed"] = 1.0 if (st["closed"] and all(inside_ok)) else 0.0
    # shipped: closed as above (both shoes in the box) and across the outfeed line; WorkOrder.tick ships on exactly this
    ms["shipped"] = 1.0 if (shipped or (ms["closed"] and st["floor_pos"][0] >= cfg["zones"]["outfeed_x"])) else 0.0
    n_lace_rows = max(1, n_rows - 1)
    laced = sum(1.0 for key in ("lacing_a", "lacing_b") if detail[key]["rows_credited"] >= n_lace_rows) / 2.0
    credit = float(sum(CREDITS[m] * ms[m] * (laced if m in LACED_SCALED else 1.0) for m in CREDITS))
    detail["laced_fraction"] = laced
    detail["carton"] = {k2: v for k2, v in st.items() if k2 in ("erected", "closed", "wall_angles_deg", "locked", "floor_pos")}
    detail["inside"] = inside_ok
    detail["heel_to_toe"] = anti
    return ms, credit, detail


LATCH_EVERY_S = 0.5          # how often the work order looks at the lace ends (brought-out latch)


class WorkOrder(object):
    """The episode's pairs, their arrival and shipping, and the per-pair credits (frozen when a pair ships). The next
    pair arrives the moment the previous box is shipped."""

    def __init__(self, scene, order, cfg, log=None):
        self.scene = scene
        self.order = order
        self.cfg = cfg
        self.log = log or (lambda s: None)
        self.pairs = []
        self.next_index = 0
        self.shipped_count = 0

    @property
    def current(self):
        return self.pairs[-1] if self.pairs else None

    def start(self, t):
        self._arrive(t)

    def _arrive(self, t):
        if self.next_index >= len(self.order["pairs"]):
            return False
        spec = self.order["pairs"][self.next_index]
        rec = self.scene.deliver_pair(spec)
        rec.update({"index": self.next_index, "spec": spec, "arrived_at": float(t), "shipped": False, "shipped_at": None,
                    "milestones": None, "credit": 0.0, "events": [], "detail": None})
        self.pairs.append(rec)
        self.next_index += 1
        self.log("[cell] pair %d arrived at t=%.1f" % (rec["index"] + 1, t))
        return True

    def tick(self, t):
        """Shipping and arrivals, and the lace ends' brought-out latch; returns events."""
        events = []
        cur = self.current
        if cur is None or cur["shipped"]:
            return events
        if t >= cur.get("latch_t", 0.0):
            self._latch_ends(cur)
            cur["latch_t"] = t + LATCH_EVERY_S
        st = self.scene.carton_state(cur["carton_id"])
        # shipped = the lid closed with both shoes in the box (`boxed`) and the floor centre across the outfeed line; a
        # box that crosses the line otherwise stays where it is
        if st["closed"] and st["floor_pos"][0] >= self.cfg["zones"]["outfeed_x"] and all(boxed(self.scene, cur)[1]):
            ms, credit, detail = pair_milestones(self.scene, cur, self.cfg, shipped=True)
            cur.update({"shipped": True, "shipped_at": float(t), "milestones": ms, "credit": credit, "detail": detail})
            self.shipped_count += 1
            self.scene.take_away_pair(cur)
            events.append("box_shipped:%d" % (cur["index"] + 1))
            self.log("[cell] pair %d shipped at t=%.1f credit %.3f %s" % (cur["index"] + 1, t, credit, ms))
            if self._arrive(t):
                events.append("pair_arrived:%d" % self.next_index)
        return events

    def _latch_ends(self, rec):
        """Per shoe of the pair and per lace end: set once the end is seen brought out after its last ring pass
        (Scene.aglet_ends_out), cleared when the tail no longer passes a ring beyond row 1."""
        bo = rec.setdefault("brought_out", {0: [False, False], 1: [False, False]})
        for s in range(2):
            now = self.scene.aglet_ends_out(rec["lace_ids"][s], rec["shoe_ids"][s])
            for k in range(2):
                if now[k] is None:
                    bo[s][k] = False
                elif now[k]:
                    bo[s][k] = True

    def summary(self, violations=()):
        """Per-pair credits: shipped pairs keep their frozen milestones, the others are judged now. A safety violation
        (`violations`: dicts with a sim time `t`) voids the credit of the pair that was at work when it happened (its
        milestones are still reported, `safety_void` is set)."""
        out = []
        times = [float(v["t"]) for v in (violations or ())]
        for rec in self.pairs:
            if rec["shipped"]:
                ms, credit, detail = rec["milestones"], rec["credit"], rec["detail"]
            else:
                ms, credit, detail = pair_milestones(self.scene, rec, self.cfg, shipped=False)
                rec.update({"milestones": ms, "credit": credit, "detail": detail})
            t0 = float(rec["arrived_at"])
            t1 = float(rec["shipped_at"]) if rec["shipped"] else float("inf")
            void = any(t0 <= t <= t1 for t in times)
            out.append({"index": rec["index"], "shipped": rec["shipped"], "credit": 0.0 if void else credit, "safety_void": void,
                        "milestones": ms, "detail": detail, "arrived_at": rec["arrived_at"], "shipped_at": rec["shipped_at"],
                        "events": list(rec["events"])})
        K = len(self.order["pairs"])
        total = sum(p["credit"] for p in out)
        return {"pairs": out, "work_score": float(total / K) if K else 0.0, "pairs_shipped": self.shipped_count}
