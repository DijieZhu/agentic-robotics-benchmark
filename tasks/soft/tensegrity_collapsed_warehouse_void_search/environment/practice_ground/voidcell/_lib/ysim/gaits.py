"""Gait primitives exactly as the authors define them (control repository ChemStud24/tensegrity, branch rss25
@4b79620, src/run_tensegrity_Astar.py and src/symmetry_reduction_utils.py; RA-L Table I for the 11 primitives).

A target state is six normalised targets, one per motor; target x of motor i means a short-sensor length
min_length + x * RANGE_i. Motors 0-2 (cables of the odd triangle 1-3-5) use the LEFT range (RANGE135),
motors 3-5 (even triangle 0-2-4) the RIGHT range (RANGE024) (run_tensegrity_Astar.py lines 362-366, 566-568:
primitive 'L_R' sets RANGE135 = L, RANGE024 = R; RA-L Table I: Left Max = 100 + L mm, Right Max = 100 + R mm).
Seen from above and facing the rolling direction (+y of the primitive-table frame), the even triangle is on the
right, so the code's names agree with the geometry.
"""
from dataclasses import dataclass

# run_tensegrity_Astar.py lines 166-168 (states 0 and 1 are the rest state [1]*6; execution starts at state 1)
BASE = {
    "roll": [[1.0, 1.0, 0.1, 1.0, 1.0, 0.1], [0.0, 1.0, 1.0, 0.0, 1.0, 0.1]],
    "cw": [[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.7, 0.0, 1.0, 1.0]],
    "ccw": [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
}
REST = [1.0] * 6

# symmetry_reduction_utils.py lines 3-6 and 11-17
SYMMETRY = {(0, 2, 5): [0, 1, 2, 3, 4, 5], (0, 3, 5): [0, 1, 2, 3, 4, 5],
            (1, 2, 4): [1, 2, 0, 4, 5, 3], (1, 2, 5): [1, 2, 0, 4, 5, 3],
            (0, 3, 4): [2, 0, 1, 5, 3, 4], (1, 3, 4): [2, 0, 1, 5, 3, 4]}
NEXT_NODES = {(0, 2, 5): (1, 2, 4), (0, 3, 5): (1, 2, 4), (1, 2, 4): (0, 3, 4), (1, 2, 5): (0, 3, 4),
              (0, 3, 4): (0, 2, 5), (1, 3, 4): (0, 2, 5)}
# ysim 0.3.0: the six bottom triples the authors' control code can execute a primitive from (the keys of their
# symmetry table); every other triple raises NotRollableFace in Simulator.start_states (no silent mapping)
ROLLABLE_FACES = tuple(sorted(SYMMETRY))
# the two triangles of the six actuated lines (one end of every strut): the robot standing on its end
END_TRIANGLES = ((0, 2, 4), (1, 3, 5))
# the authors' table maps the two members of each pair identically (same symmetry permutation)
FACE_PAIRS = (((0, 2, 5), (0, 3, 5)), ((1, 2, 4), (1, 2, 5)), ((0, 3, 4), (1, 3, 4)))


class NotRollableFace(ValueError):
    """The robot rests on a bottom triple outside the authors' symmetry table (an end triangle or a contact set
    that is not a face): no published primitive is defined from it."""


def face_pair(bottom):
    """Index 0..2 of the symmetry class of a rollable bottom triple (the pair the authors' table treats alike)."""
    b = tuple(bottom)
    for k, pr in enumerate(FACE_PAIRS):
        if b in pr:
            return k
    raise NotRollableFace("bottom triple %s is not a rollable face %s" % (b, ROLLABLE_FACES))
# symmetry_reduction_utils.py lines 7-9 (reverse_mapping) and 28-32 (reverse_gait), used by run_tensegrity_Astar.py
# lines 561-563 for rolls only (reverse_the_gait): the already symmetry-mapped state is re-indexed once more.
# Added in ysim 0.2.0. The authors' own comment (lines 58-62): "I don't know what the reverse gait is
# supposed to look like since we've never done it :) ... update: it seems reasonable" - no published hardware run.
REVERSE = {(0, 2, 5): [3, 5, 4, 0, 2, 1], (0, 3, 5): [3, 5, 4, 0, 2, 1],
           (1, 2, 4): [4, 3, 5, 1, 0, 2], (1, 2, 5): [4, 3, 5, 1, 0, 2],
           (0, 3, 4): [5, 4, 3, 2, 1, 0], (1, 3, 4): [5, 4, 3, 2, 1, 0]}


@dataclass(frozen=True)
class Primitive:
    name: str
    kind: str            # roll | ccw | cw
    left: int            # RANGE135 [mm]
    right: int           # RANGE024 [mm]
    table_key: str       # key in calibration/new_platform_transformation_table.pkl
    index: int           # RA-L Table I row

    @property
    def tol(self):
        # run_tensegrity_Astar.py lines 569-572: tol 0.35 if a range >= 130 (rolls); cw/ccw keep 0.15
        return 0.35 if (self.kind == "roll" and (self.left >= 130 or self.right >= 130)) else 0.15

    @property
    def low_tol(self):
        return 0.15

    def states(self):
        return [list(REST)] + [list(s) for s in BASE[self.kind]] + [list(REST)]


# RA-L Table I order (= planner.py self.primitives order)
_ORDER = [("100_100", 0), ("120_120", 1), ("140_140", 2), ("100_120", 3), ("120_100", 4), ("100_140", 5),
          ("140_100", 6), ("120_140", 7), ("140_120", 8)]
PRIMITIVES = {}
for key, idx in _ORDER:
    l, r = (int(v) for v in key.split("_"))
    PRIMITIVES["roll_" + key] = Primitive("roll_" + key, "roll", l, r, "%s__%s" % (key, key), idx)
PRIMITIVES["ccw"] = Primitive("ccw", "ccw", 100, 100, "ccw__ccw", 9)
PRIMITIVES["cw"] = Primitive("cw", "cw", 100, 100, "cw__cw", 10)
NAMES = sorted(PRIMITIVES, key=lambda n: PRIMITIVES[n].index)


def get(name):
    return PRIMITIVES[name]


def map_state(state, bottom):
    """transform_gait(): motor k gets base target state[mapping[k]] for the current bottom-node triple."""
    mp = SYMMETRY.get(tuple(bottom))
    if mp is None:
        return None
    return [state[mp[k]] for k in range(6)]


def reverse_state(mapped_state, bottom):
    """reverse_gait(): re-index an already symmetry-mapped state with the reverse mapping of the bottom triple."""
    mp = REVERSE.get(tuple(bottom))
    if mp is None or mapped_state is None:
        return None
    return [mapped_state[m] for m in mp]


# The authors' simulator definition of the same 11 primitives (control repository, branch hybrid_mppi @d1e58a4,
# src/gnn_simulator/mujoco_physics_engine/tensegrity_mjc_simulation.py get_3bar_prims_se2_data(), lines 953-1000,
# PID defaults mujoco_physics_engine/pid.py lines 6-13): states (ending with the rest state), min_length per kind,
# (left, right) ranges, tol 0.1 + "within 10 mm" + "target 0 and pos < 0" done rules, k_p 6.0, k_i 0.01, k_d 0.5,
# symmetry order re-detected at every state from the lower end cap of each rod (run_w_target_gaits, lines 478-560).
SIM_BASE = {
    "roll": [[1.0, 1.0, 0.1, 1.0, 1.0, 0.1], [0.0, 1.0, 1.0, 0.0, 1.0, 0.1], [1.0] * 6],
    "ccw": [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [0.0] * 6, [1.0] * 6],
    "cw": [[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.5, 0.0, 1.0, 1.0], [1.0] * 6],
}
SIM_MIN_LENGTH = {"roll": 90.0, "ccw": 100.0, "cw": 80.0}
SIM_RANGES = {"ccw": (100, 100), "cw": (120, 120)}
SIM_TOL = 0.10
SIM_P = 6.0


def ground_endcaps(nodes):
    """detect_ground_endcaps(): the lower end cap of each rod, in rod order."""
    import numpy as np
    z = np.asarray(nodes)[:, 2]
    return tuple(2 * k + (0 if z[2 * k] <= z[2 * k + 1] else 1) for k in range(3))
