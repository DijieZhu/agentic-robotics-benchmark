"""Shelf restocking: an episode and its credit.

An episode ends after EPISODE_STEPS control steps, when the controller sends the finish action ([0]), or at an
unstable simulation (credit 0). The credit, read from the simulation at the end (bay frame; a unit stands in a lane
when its centre is within LANE_TOL of the lane's centre line, between the lip and the back panel, on the lane's
board within Z_TOL, tilted less than TILT_MAX; a unit's code is expired when it is before the episode's `today`):

An episode earns credit only when EVERY served lane is correct: a delivery with a lane left wrong is not done, and
its credit is 0 whatever else was achieved.

- **correct** (CORRECT, all served lanes together): a served lane is correct when the units standing in it
  are exactly its unexpired old units and the delivered units of its SKU, in code order: from the front to the
  back, each unit's code no earlier than the one in front of it (equal codes in either order).
- **faced** (FACE, split evenly over the served lanes): a correct lane whose front unit's front face is within
  FACE_TOL of the lip's inner face, a box also turned face out (its yaw within FACE_YAW of the bay's).
- **kept** (KEEP): every unit of the lanes the delivery does not serve within KEEP_TOL of where it started, no unit
  lying down anywhere (tilted past 45 deg) or on the floor, and every expired unit and every extra (a tote unit
  that is not part of the delivery) standing in the tote.
"""

import math

import numpy as np

import shelf_sim as S
import shelf_world as W

EPISODE_STEPS = 15000           # 25 simulated minutes
CORRECT, FACE, KEEP = 0.7, 0.2, 0.1
LANE_TOL = 0.03                 # m
Z_TOL = 0.012                   # m
TILT_MAX = 10.0                 # deg
FACE_TOL = 0.015                # m
FACE_YAW = math.radians(20)
KEEP_TOL = 0.02                 # m
FINISH = [0]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Episode:
    def __init__(self, world):
        self.w = world
        self.done = False
        self.finished = False

    def step_done(self, finished=False):
        w = self.w
        self.finished = self.finished or finished
        if self.finished or w.t >= EPISODE_STEPS or w.unstable():
            self.done = True
        return self.done

    def standing(self, n):
        """(si, li) of the lane unit n stands in, or None."""
        w = self.w
        u = w.unit
        sku = u["skus"][w.sku_of[n]]
        p = w.to_bay(w.pos(n))
        if w.tilt(n) >= TILT_MAX or not (S.LIP_T < p[0] < S.LIP_T + u["depth"]):
            return None
        for si, sh in enumerate(u["shelves"]):
            if abs(p[2] - (sh["board"] + W.half_h(sku))) < Z_TOL:
                for li in range(W.N_LANES):
                    if abs(p[1] - W.lane_y(u, li)) < LANE_TOL:
                        return si, li
        return None

    def in_tote(self, n):
        """Unit n stands upright in the tote."""
        w = self.w
        sku = w.unit["skus"][w.sku_of[n]]
        p = w.pos(n)
        (cx, cy), (ix, iy) = W.TOTE["centre"], W.TOTE["inner_half"]
        return (w.tilt(n) < TILT_MAX and abs(p[0] - cx) < ix and abs(p[1] - cy) < iy
                and abs(p[2] - (W.CART["top"] + W.half_h(sku))) < Z_TOL)

    def score(self):
        w = self.w
        u, ep = w.unit, w.episode
        where = {n: self.standing(n) for n in w.bodies}
        depth = {n: float(w.to_bay(w.pos(n))[0]) for n in w.bodies}
        code = {n: m["code"] for n, m in w.meta.items()}
        lanes = {}
        for key in ep["served"]:
            si, li = map(int, key.split(":"))
            need = {n for n, m in w.meta.items() if m["lane"] == key and m["role"] in ("old", "new")
                    and not m["expired"]}
            inside = sorted((n for n, at in where.items() if at == (si, li)), key=lambda n: depth[n])
            in_order = all(code[a] <= code[b] for a, b in zip(inside, inside[1:]))
            correct = set(inside) == need and in_order
            faced = False
            if correct and inside:
                front = inside[0]
                k = u["skus"][w.sku_of[front]]
                faced = depth[front] - W.half_x(k) - S.LIP_T <= FACE_TOL
                if k["kind"] == "box":
                    faced = faced and abs(wrap(w.yaw(front) - w.bay["yaw"])) <= FACE_YAW
            lanes[key] = {"correct": bool(correct), "faced": bool(faced), "in_order": bool(in_order),
                          "missing": sorted(need - set(inside)), "extra": sorted(set(inside) - need),
                          "codes": [code[n] for n in inside]}
        stock = [n for n, m in w.meta.items() if m["role"] == "stock"]
        moved = [n for n in stock if np.linalg.norm(w.pos(n) - w.start_pos[n]) > KEEP_TOL]
        fallen = [n for n in w.bodies if w.tilt(n) > 45 or w.pos(n)[2] < 0.2]
        expired_out = [n for n, m in w.meta.items() if m["expired"] and not self.in_tote(n)]
        extra_out = [n for n, m in w.meta.items() if m["role"] == "extra" and not self.in_tote(n)]
        place = float(np.mean([v["correct"] for v in lanes.values()]))
        face = float(np.mean([v["faced"] for v in lanes.values()]))
        kept = bool(not moved and not fallen and not expired_out and not extra_out and place > 0)
        unstable = bool(w.unstable())
        complete = bool(lanes) and all(v["correct"] for v in lanes.values())
        credit = CORRECT + FACE * face + KEEP * kept if (complete and not unstable) else 0.0
        # the rule this task had until 2026-11-23 (a share of the lanes earns its share): reported beside the credit,
        # never scored
        partial = 0.0 if unstable else CORRECT * place + FACE * face + KEEP * kept
        return {"credit": round(credit, 4), "credit_partial": round(partial, 4), "complete": complete, "place": round(place, 4), "face": round(face, 4), "kept": kept,
                "lanes": lanes, "moved": moved, "fallen": fallen, "expired_out": expired_out,
                "extra_out": extra_out, "unstable": unstable,
                "steps": int(w.t), "finished": self.finished}
