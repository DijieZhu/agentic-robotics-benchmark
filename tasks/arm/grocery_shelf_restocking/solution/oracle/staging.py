"""Shelf restocking: the abstract restock problem, and the rule that keeps an episode.

A served lane is a front-access stack: only its front unit can be taken out, and a unit can only be set down in
front of the others. The tote holds TOTE_SLOTS units. An episode is kept (`keep`) when both hold:

1. `tote_only_solvable` is False: with full knowledge of every unit, no sequence of moves in which units wait only
   in the tote makes every served lane correct: a unit leaves a lane only for the tote, and is set down in a lane
   only at its final place (its own lane, the goal's code for that slot, every slot behind it final too). An
   exhaustive search; a search that exceeds MAX_STATES counts as solvable (not proven otherwise). So some unit
   has to be set down in a lane, any lane, and taken out again.
2. `online_plan` succeeds: a fixed strategy that decides each move only from what it has seen by then (the tote's
   units, each lane's unit count and free front space, and every unit it has taken out) restocks every lane,
   setting units down in front of another served lane of the same shelf where the tote has no room.

So staging in a lane is necessary (1) and an agent that cannot see behind a lane's front unit can still succeed (2).
The reference executes the moves of (2). Geometry (bay frame x, the unit's centre): a unit of the lane's own SKU
goes into the deepest of its slots (`shelf_world.slot_x`) that fits in front of the lane's front unit; a unit of
another SKU goes SLOT_GAP in front of the front unit (into an empty lane: its own deepest slot); never past the
faced position.

    moves = online_plan(unit, episode)    # [(kind, uid, src, dst, x)] or None
    keep(unit, episode) -> (bool, detail)
"""

from collections import deque

import shelf_world as W

TOTE_SLOTS = W.TOTE_GRID[0] * W.TOTE_GRID[1]
MAX_STATES = 300_000
FIT_GAP = 0.008                # m, the least gap between a unit set down and the unit behind it
FRONT_SLACK = 0.001            # m, how far past the faced position a staged unit may stand


class Problem:
    """The episode as units: uid -> (sku, code, home lane or None); each served lane's units front to back as
    (uid, x); the tote's uids."""

    def __init__(self, unit, ep):
        self.unit, self.ep, self.today = unit, ep, ep["today"]
        self.served = list(ep["served"])
        self.sku = {}           # uid -> sku name
        self.code = {}
        self.home = {}          # uid -> the served lane its SKU belongs to
        self.lanes = {}         # key -> [(uid, x)] front to back
        lane_of_sku = {ep["planogram"][k]: k for k in self.served}
        for key in self.served:
            sku = unit["skus"][ep["planogram"][key]]
            units = []
            for k, (slot, push, code) in enumerate(ep["old"][key]):
                uid = f"{key}/o{k}"
                self.sku[uid], self.code[uid], self.home[uid] = ep["planogram"][key], code, key
                units.append((uid, W.slot_x(unit, sku, slot) + push))
            self.lanes[key] = sorted(units, key=lambda u: u[1])
        self.tote = []
        for k, t in enumerate(ep["tote"]):
            uid = f"t{k}"
            # an extra (no served lane carries its SKU) has no home: it ends in the tote
            self.sku[uid], self.code[uid], self.home[uid] = t["sku"], t["code"], lane_of_sku.get(t["sku"])
            self.tote.append(uid)
        self.cap = {k: W.capacity(unit, unit["skus"][ep["planogram"][k]]) for k in self.served}
        self.goal = {}
        for key in self.served:
            mine = [u for u in self.sku if self.home[u] == key and self.code[u] >= self.today]
            self.goal[key] = sorted(self.code[u] for u in mine)
        self.shelf = {k: int(k.split(":")[0]) for k in self.served}

    def expired(self, uid):
        return self.code[uid] < self.today

    def hx(self, uid):
        return W.half_x(self.unit["skus"][self.sku[uid]])

    def place_x(self, key, stack, uid):
        """Where uid stands when set down in front of `stack` (lane key, front to back), or None."""
        sku = self.unit["skus"][self.sku[uid]]
        hx = W.half_x(sku)
        if abs(W.half_h(sku) - W.half_h(self.unit["skus"][self.ep["planogram"][key]])) > 1e-6:
            return None                         # only a unit of the shelf's height goes into its lanes
        if self.sku[uid] == self.ep["planogram"][key]:
            cap = self.cap[key]
            if not stack:
                return W.slot_x(self.unit, sku, cap - 1)
            limit = stack[0][1] - self.hx(stack[0][0]) - FIT_GAP - hx
            for k in range(cap - 1, -1, -1):
                x = W.slot_x(self.unit, sku, k)
                if x <= limit + 1e-9:
                    return x
            return None
        if not stack:
            return W.slot_x(self.unit, sku, W.capacity(self.unit, sku) - 1)
        x = stack[0][1] - self.hx(stack[0][0]) - W.SLOT_GAP - hx
        return x if x >= W.slot_x(self.unit, sku, 0) - FRONT_SLACK else None

    def slot_of(self, key, x):
        sku = self.unit["skus"][self.ep["planogram"][key]]
        x0 = W.slot_x(self.unit, sku, 0)
        return round((x - x0) / (W.slot_x(self.unit, sku, 1) - x0))

    def final(self, key, stack):
        """Every unit in the lane (front to back) at its final place: the lane's own unexpired units, the goal's
        last codes, in the deepest slots."""
        n, cap = len(stack), self.cap[key]
        if n > cap or any(self.home[u] != key or self.expired(u) for u, _x in stack):
            return False
        if [self.code[u] for u, _x in stack] != self.goal[key][len(self.goal[key]) - n:]:
            return False
        return all(self.slot_of(key, x) == cap - n + i for i, (_u, x) in enumerate(stack))

    def lane_done(self, key, stack, faced=True):
        """The lane correct (and, with faced, its front unit in the faced slot)."""
        if len(stack) != self.cap[key] or any(self.home[u] != key or self.expired(u) for u, _x in stack):
            return False
        codes = [self.code[u] for u, _x in stack]
        if codes != sorted(codes) or sorted(codes) != self.goal[key]:
            return False
        if not faced:
            return True
        sku = self.unit["skus"][self.ep["planogram"][key]]
        return abs(stack[0][1] - W.slot_x(self.unit, sku, 0)) < 0.003


# ------------------------------------------------------------------------------------------------ rule 1
def tote_only_solvable(p, max_states=MAX_STATES):
    """Breadth-first over every state reachable when units wait only in the tote (a unit set down in a lane only
    at its final place). True, False, or None when the search passed max_states."""
    keys = p.served

    def canon(lanes):
        return tuple(tuple((p.sku[u], p.code[u], round(x, 4)) for u, x in lanes[k]) for k in keys)

    def tote_of(lanes):
        inside = {u for k in keys for u, _x in lanes[k]}
        return [u for u in p.sku if u not in inside]

    start = {k: list(p.lanes[k]) for k in keys}
    seen = {canon(start)}
    queue = deque([start])
    while queue:
        lanes = queue.popleft()
        # correct is enough: a correct lane's front unit can always be set down again, faced, in place
        if all(p.lane_done(k, lanes[k], faced=False) for k in keys):
            return True
        tote = tote_of(lanes)
        nxt = []
        if len(tote) < TOTE_SLOTS:
            for k in keys:
                if lanes[k]:
                    n = dict(lanes)
                    n[k] = lanes[k][1:]
                    nxt.append(n)
        tried = set()
        for u in tote:
            k = p.home[u]
            if k is None:
                continue
            sig = (p.sku[u], p.code[u])
            if sig in tried:
                continue
            tried.add(sig)
            x = p.place_x(k, lanes[k], u)
            if x is not None and p.final(k, [(u, x)] + lanes[k]):
                n = dict(lanes)
                n[k] = [(u, x)] + lanes[k]
                nxt.append(n)
        for n in nxt:
            c = canon(n)
            if c not in seen:
                if len(seen) >= max_states:
                    return None
                seen.add(c)
                queue.append(n)
    return False


# ------------------------------------------------------------------------------------------------ rule 2
class Online:
    """The online strategy on the true episode; it reads a unit's code only once the unit is out of its lane (the
    tote's units it may read where they stand)."""

    def __init__(self, p):
        self.p = p
        self.lanes = {k: list(v) for k, v in p.lanes.items()}
        self.tote = list(p.tote)
        self.moves = []

    # moves ------------------------------------------------------------------------------------------
    def room(self):
        return TOTE_SLOTS - len(self.tote)

    def lane_to_tote(self, k):
        if self.room() < 1 or not self.lanes[k]:
            raise Fail(f"no tote room for {k}'s front")
        u, _x = self.lanes[k].pop(0)
        self.tote.append(u)
        self.moves.append(("lane_tote", u, k, None, None))

    def tote_to_lane(self, u, k):
        x = self.p.place_x(k, self.lanes[k], u)
        if x is None:
            raise Fail(f"{u} does not fit in {k}")
        self.tote.remove(u)
        self.lanes[k].insert(0, (u, x))
        self.moves.append(("tote_lane", u, None, k, x))

    def lane_to_lane(self, a, b):
        u = self.lanes[a][0][0]
        x = self.p.place_x(b, self.lanes[b], u)
        if x is None:
            raise Fail(f"{u} does not fit in {b}")
        self.lanes[a].pop(0)
        self.lanes[b].insert(0, (u, x))
        self.moves.append(("lane_lane", u, a, b, x))

    # what can be seen without taking a unit out -------------------------------------------------------
    def front_room(self, k, uids):
        """How many of uids (in this order) fit in front of lane k, set down one after the other."""
        stack, n = list(self.lanes[k]), 0
        for u in uids:
            x = self.p.place_x(k, stack, u)
            if x is None:
                break
            stack.insert(0, (u, x))
            n += 1
        return n

    # the strategy ---------------------------------------------------------------------------------------
    def order(self):
        """Lanes in an order chosen from what is visible (unit counts, free front space, the tote's units, the
        delivery, capacities), never from units still in a lane: the pair on one shelf, the first of them staging
        in the second; the lone lane first or last. The first candidate whose counts work out even if every unit
        still unseen were expired."""
        p = self.p
        by_shelf = {}
        for k in p.served:
            by_shelf.setdefault(p.shelf[k], []).append(k)
        pair = next((v for v in by_shelf.values() if len(v) == 2), None)
        if pair is None:
            raise Fail("no two served lanes on one shelf")
        x, y = sorted(pair, key=lambda k: (-len(self.lanes[k]), k))
        lone = [k for k in p.served if k not in pair]
        cands = []
        for a, b in ((x, y), (y, x)):
            if lone:
                cands += [(lone + [a, b], a, b), ([a, b] + lone, a, b)]
            else:
                cands.append(([a, b], a, b))
        for order, a, b in cands:
            if self.visible_ok(order, a, b):
                return order, {a: b}
        return cands[0][0], {cands[0][1]: cands[0][2]}

    def delivered(self, k):
        return sum(1 for u in self.tote if self.p.home[u] == k)

    def visible_ok(self, order, a, b):
        p, room = self.p, self.room()
        for k in order:
            m = len(self.lanes[k])
            if k == a:
                short = m - room
                if short > 0:
                    seq = self.stage_pick(a, b, short)
                    if seq is None or self.front_room(b, seq) < short:
                        return False
                    own = sum(1 for u in seq if p.home[u] == a)
                    room = p.cap[a] - own                    # the tote empty after a is out; its rebuild frees these
                    continue
            elif m > room:
                return False
            # pessimistic: every unit still unseen expired (it stays in the tote), at most the delivery
            room += self.delivered(k) - min(m, self.delivered(k)) if k != b else 0
            if k == b:
                room = room - m + p.cap[b] - (m - min(m, self.delivered(b)))
        return True

    def stage_pick(self, k, partner, short):
        """The tote units to stage in the partner before lane k is emptied, in the order they go in: other lanes'
        units first (deepest; they leave when the partner is restocked), then k's own delivered units, earliest
        code first (the rebuild takes the latest first, from the partner's front). None if too few."""
        p = self.p
        lane_h = W.half_h(p.unit["skus"][p.ep["planogram"][partner]])
        others = [u for u in self.tote if p.home[u] not in (k, partner)
                  and abs(W.half_h(p.unit["skus"][p.sku[u]]) - lane_h) < 1e-6]
        own = sorted([u for u in self.tote if p.home[u] == k and not p.expired(u)], key=lambda u: p.code[u])
        pick = (others + own)[:short]
        if len(pick) < short:
            return None
        return [u for u in pick if p.home[u] != k] + sorted([u for u in pick if p.home[u] == k], key=lambda u: p.code[u])

    def restock(self, k, partner):
        p = self.p
        m = len(self.lanes[k])
        short = m - self.room()
        if short > 0:
            if partner is None:
                raise Fail(f"{k}: {m} units out, {self.room()} tote slots, nowhere to stage")
            self.stage(k, partner, short)
        while self.lanes[k]:
            self.lane_to_tote(k)
        # rebuild, the latest code deepest; from the tote or the partner's front
        for code in reversed(p.goal[k]):
            front = self.lanes[partner][0][0] if partner and self.lanes[partner] else None
            if front is not None and p.home[front] == k and not p.expired(front) and p.code[front] == code:
                self.lane_to_lane(partner, k)
                continue
            u = next((u for u in self.tote if p.home[u] == k and not p.expired(u) and p.code[u] == code), None)
            while u is None and partner and self.lanes[partner]:
                # dig: the needed unit is staged deeper; its cover goes to the tote
                front = self.lanes[partner][0][0]
                if p.home[front] == k and not p.expired(front) and p.code[front] == code:
                    break
                self.lane_to_tote(partner)
                u = next((u for u in self.tote if p.home[u] == k and not p.expired(u) and p.code[u] == code),
                         None)
            if u is None:
                if not (partner and self.lanes[partner]):
                    raise Fail(f"{k}: no unit with code {code} at hand")
                self.lane_to_lane(partner, k)
            else:
                self.tote_to_lane(u, k)
        if not p.lane_done(k, self.lanes[k]):
            raise Fail(f"{k} not correct after its rebuild")

    def stage(self, k, partner, short):
        seq = self.stage_pick(k, partner, short)
        if seq is None:
            raise Fail(f"{k}: {short} to stage, too few tote units")
        if self.front_room(partner, seq) < len(seq):
            raise Fail(f"{partner} has no room for {len(seq)} staged units")
        for u in seq:
            self.tote_to_lane(u, partner)

    def run(self):
        order, partners = self.order()
        for k in order:
            self.restock(k, partners.get(k))
        if any(not self.p.lane_done(k, self.lanes[k]) for k in self.p.served):
            raise Fail("a lane not correct at the end")
        if any(self.p.home[u] is not None and not self.p.expired(u) for u in self.tote):
            raise Fail("an unexpired unit left in the tote")
        return self.moves


class Fail(Exception):
    pass


def online_plan(unit, ep):
    """The online strategy's moves, or (None, reason)."""
    try:
        return Online(Problem(unit, ep)).run(), None
    except Fail as e:
        return None, str(e)


def keep(unit, ep):
    p = Problem(unit, ep)
    moves, why = online_plan(unit, ep)
    if moves is None:
        return False, {"rule": 2, "why": why}
    tote = tote_only_solvable(p)
    if tote is not False:
        return False, {"rule": 1, "why": "tote alone suffices" if tote else "search too large"}
    staged = sum(1 for kind, u, _a, dst, _x in moves if dst is not None and p.home[u] != dst)
    return True, {"moves": len(moves), "staged": staged}

