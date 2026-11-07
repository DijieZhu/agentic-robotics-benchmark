"""sar_protocol - the rules of aerial search-and-geolocation missions, in one place.

Shared byte-identically by the grading stage and the practice grader, so the practice run
and the scored run assemble missions the same way and apply the same actions, observation,
termination rules and mission result. The simulator lives behind `sim_client.SimClient`;
this module never imports it.

A mission: the drone starts at a pose above a scene and is given the descriptions of
three targets. Some are in the scene; possibly one is not (a decoy). The drone flies, and
with every reply it also carries its current report for each target: not yet located, found
at a position (in the mission's start frame), or declared absent. The mission ends on Stop,
on a collision, or when the action budget is spent; the last report received is final. A
present target earns credit only if the drone went and looked at it: at some point of the
mission it was within VISIT_RADIUS_M of the target horizontally, at most VISIT_HEIGHT_M above
it, and with the target inside the down camera's view (see `visits`).

stdlib only, 3.8-compatible syntax.
"""

import json
import math

__all__ = [
    "ACTIONS", "UNKNOWN", "FOUND", "ABSENT", "ProtocolError", "MissionOutcome",
    "agent_observation", "aggregate", "assemble_missions", "load_pool", "parse_reply",
    "perturb_targets", "privileged_view", "run_mission",
]

ACTIONS = ("stop", "forward", "left", "right", "ascend", "descend", "rotl", "rotr")
CAMERAS = ("front", "left", "right", "down")
UNKNOWN, FOUND, ABSENT = 0, 1, 2

# Scoring constants (the instruction states them; keep the two in sync).
FULL_CREDIT_M = 3.0       # a located target within this distance earns full credit
ZERO_CREDIT_M = 10.0      # ... falling linearly to nothing at this distance
DECOY_CREDIT = 1.0        # a decoy correctly declared absent: as much as a located target,
                          # so a perfect run scores 1
FALSE_ABSENT = -1.0       # a present target declared absent (decoys are ~1 target in 6, so
                          # random absence claims still lose on average)
VISIT_RADIUS_M = 3.0      # a present target counts only once the drone has been this close
VISIT_HEIGHT_M = 6.0      # ... horizontally, at most this high above it, and with the target
CAMERA_DROP_M = 1.0       # ... in the down camera's 90 degree view (the camera hangs 1 m low)
# Mission geometry: targets this far from the start horizontally, apart from each other by
# MIN_SPACING_M..MAX_SPACING_M and spread around the start by at least MIN_BEARING_DEG.
MIN_TARGET_M, MAX_TARGET_M = 40.0, 120.0
MIN_SPACING_M, MAX_SPACING_M = 25.0, 160.0
MIN_BEARING_DEG = 40.0
ENERGY_WEIGHT = 0.2       # up to 20 % off for flying the whole budget
COLLISION_FACTOR = 0.5    # a crash halves the mission


class ProtocolError(Exception):
    """The controller's reply is not a valid action/report. Aborts the mission."""


# --------------------------------------------------------------------------- the reply
def parse_reply(raw, n_targets):
    """`raw` is the reply's `action`: a flat list of 1 + 4 * n_targets numbers,

        [move, s_1, x_1, y_1, z_1, ..., s_n, x_n, y_n, z_n]

    where s_i is 0 (not located), 1 (found at x, y, z) or 2 (declared absent), and x, y, z
    are in the start frame (forward, right, down; metres). A bare move (an int, or a
    one-element list) reports nothing. Returns (move, reports)."""
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    if len(values) not in (1, 1 + 4 * n_targets):
        raise ProtocolError("action must hold 1 or {} numbers, got {}".format(
            1 + 4 * n_targets, len(values)))
    numbers = []
    for value in values:
        if isinstance(value, bool):
            raise ProtocolError("booleans are not numbers here")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ProtocolError("non-numeric action entry {!r}".format(value)[:120])
        if not math.isfinite(number):
            raise ProtocolError("non-finite action entry")
        numbers.append(number)
    move = numbers[0]
    if move != int(move) or not 0 <= int(move) < len(ACTIONS):
        raise ProtocolError("move must be an integer 0-7, got {!r}".format(values[0]))
    reports = [(UNKNOWN, None)] * n_targets
    if len(numbers) > 1:
        reports = []
        for i in range(n_targets):
            status, x, y, z = numbers[1 + 4 * i: 5 + 4 * i]
            if status not in (UNKNOWN, FOUND, ABSENT):
                raise ProtocolError("target {} status must be 0, 1 or 2".format(i))
            reports.append((int(status), [x, y, z] if status == FOUND else None))
    return int(move), reports


# --------------------------------------------------------------------------- geometry
def _yaw(q_xyzw):
    x, y, z, w = q_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def odometry(pose, start_pose):
    """Pose relative to the mission start, in the start's frame: {"position": [forward,
    right, down], "yaw"} (NED metres, yaw clockwise-positive, start heading = 0)."""
    sx, sy, sz = start_pose[:3]
    syaw = _yaw(start_pose[3:7])
    dx, dy, dz = pose[0] - sx, pose[1] - sy, pose[2] - sz
    c, s = math.cos(syaw), math.sin(syaw)
    return {"position": [round(c * dx + s * dy, 4), round(-s * dx + c * dy, 4), round(dz, 4)],
            "yaw": round(_wrap(_yaw(pose[3:7]) - syaw), 6)}


def world_to_start_frame(point, start_pose):
    syaw = _yaw(start_pose[3:7])
    dx, dy, dz = point[0] - start_pose[0], point[1] - start_pose[1], point[2] - start_pose[2]
    c, s = math.cos(syaw), math.sin(syaw)
    return [c * dx + s * dy, -s * dx + c * dy, dz]


def start_frame_to_world(point, start_pose):
    forward, right, down = point
    syaw = _yaw(start_pose[3:7])
    c, s = math.cos(syaw), math.sin(syaw)
    return [start_pose[0] + c * forward - s * right,
            start_pose[1] + s * forward + c * right,
            start_pose[2] + down]


# --------------------------------------------------------------------------- observation
def agent_observation(frame, mission, start_pose, steps_used, budget):
    """Perception, the target descriptions and odometry; never a true position."""
    return {
        "targets": [{"description": t["description"]} for t in mission["targets"]],
        "rgb": {cam: frame["rgb"][cam] for cam in CAMERAS},
        "depth": {cam: frame["depth"][cam] for cam in CAMERAS},
        "odom": odometry(frame["pose"], start_pose),
        "steps_used": int(steps_used),
        "budget": int(budget),
    }


def privileged_view(mission, start_pose):
    """Ground truth for the grader's privileged mode only (a valid oracle token): every
    object placed in the scene, by description, in the start frame. It describes the scene,
    not the request, so a probe's other descriptions get other answers from it. `run_mission`
    adds, per step, the simulator's `clearance`: how far each move could go before touching
    geometry."""
    return {"objects": [{"description": t["description"],
                         "position": [round(v, 4) for v in world_to_start_frame(t["position"], start_pose)]}
                        for t in mission["targets"] if t["position"] is not None]}


def perturb_targets(obs, descriptions):
    """The probe's perturbed copy: same images and odometry, different target texts."""
    perturbed = dict(obs)
    perturbed["targets"] = [{"description": d} for d in descriptions]
    return perturbed


# --------------------------------------------------------------------------- missions
def load_pool(path):
    """Slots and target kinds per map, as written by build_episode_pool.py."""
    with open(path) as handle:
        pool = json.load(handle)
    for name, entry in pool["maps"].items():
        if not entry.get("slots") or not entry.get("targets"):
            raise ValueError("pool map {} has no slots or no targets".format(name))
    return pool


def _hdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pick(rng, items, k, key=None):
    """k items drawn without replacement (the kit's stream has no sample/shuffle); with
    `key`, no two picked items share key(item)."""
    left, chosen, seen = list(items), [], set()
    while left and len(chosen) < k:
        item = left.pop(int(rng.random() * len(left)))
        if key is not None and key(item) in seen:
            continue
        chosen.append(item)
        if key is not None:
            seen.add(key(item))
    return chosen


def _bearing_gap(centre, a, b):
    ba = math.atan2(a[1] - centre[1], a[0] - centre[0])
    bb = math.atan2(b[1] - centre[1], b[0] - centre[0])
    return abs((ba - bb + math.pi) % (2.0 * math.pi) - math.pi)


def _spread(rng, slots, centre, k, min_dist, max_dist):
    """k slots min_dist..max_dist m from `centre`, MIN_SPACING_M..MAX_SPACING_M apart and at
    least MIN_BEARING_DEG apart as seen from `centre` (so they lie in different directions,
    and usually not all in view from the start)."""
    near = [s for s in slots if min_dist <= _hdist(s, centre) <= max_dist]
    chosen = []
    for s in _pick(rng, near, len(near)):
        if all(MIN_SPACING_M <= _hdist(s, t) <= MAX_SPACING_M and
               _bearing_gap(centre, s, t) >= math.radians(MIN_BEARING_DEG) for t in chosen):
            chosen.append(s)
            if len(chosen) == k:
                break
    return chosen


def _start_candidates(rng, slots, centre, min_dist, max_dist, count, tries=200):
    """Start points near `centre` (the first on it, the others within 30 m) from which every
    slot still lies min_dist..max_dist m away horizontally, with a random heading, at rising
    heights above the highest slot (10-16 m for the first, up to 34-40 m for the last), so
    that in a street canyon a later, higher one can clear the buildings. The simulator takes
    the first with free space around it, raised by up to 30 m if none has."""
    top = min(s[2] for s in slots)                    # NED: the highest has the smallest z
    found = []
    for _ in range(tries):
        if len(found) >= count:
            break
        angle = rng.random() * 2.0 * math.pi
        radius = 30.0 * rng.random() if found else 0.0
        x, y = centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle)
        if all(min_dist <= _hdist((x, y), s) <= max_dist for s in slots):
            rise = 10.0 + 24.0 * len(found) / max(1, count - 1) + 6.0 * rng.random()
            found.append([x, y, top - rise, (rng.random() * 2.0 - 1.0) * math.pi])
    return found


def assemble_missions(pool, rng, per_map, n_targets=3, min_dist=MIN_TARGET_M,
                      max_dist=MAX_TARGET_M, absent_probability=0.5, n_starts=12, max_tries=400):
    """Missions drawn from the pool with `rng` (anything with random() and choice()).

    Per mission, in one map: a start point is drawn over a random slot (open ground, so not
    inside a building), and target kinds of distinct categories (the map's own catalogue)
    are put on slots that lie `min_dist`..`max_dist` m (horizontally) from it, spread apart
    (`_spread`). With
    `absent_probability` one target is a decoy: a kind of the same map's catalogue whose
    category is not in the mission, so it is nowhere in the scene. The start is one of up to
    `n_starts` candidates near the drawn point; the simulator takes the first with free space
    around it."""
    missions = []
    for map_name in sorted(pool["maps"]):
        slots = pool["maps"][map_name]["slots"]
        kinds = pool["maps"][map_name]["targets"]
        made = 0
        for _ in range(max_tries):
            if made >= per_map:
                break
            n_absent = 1 if rng.random() < absent_probability else 0
            below = rng.choice(slots)
            centre = (below[0], below[1])
            placed = _spread(rng, slots, centre, n_targets - n_absent, min_dist, max_dist)
            if len(placed) < n_targets - n_absent:
                continue
            starts = _start_candidates(rng, placed, centre, min_dist, max_dist, n_starts)
            if not starts:
                continue
            picked = _pick(rng, kinds, n_targets, key=lambda k: k["category"])
            if len(picked) < n_targets:
                continue
            targets = [{"description": kind["description"], "category": kind["category"],
                        "asset": kind["asset"], "position": [float(v) for v in slot[:3]],
                        "rotation_wxyz": [float(v) for v in slot[3:7]]}
                       for kind, slot in zip(picked, placed)]
            if n_absent:
                decoy = picked[-1]
                targets.append({"description": decoy["description"], "category": decoy["category"],
                                "asset": decoy["asset"], "position": None, "rotation_wxyz": None})
            ordered = _pick(rng, targets, len(targets))
            missions.append({
                "map_name": map_name,
                "mission_id": "{}#{}".format(map_name, made),
                "start_candidates": starts,
                "targets": ordered,
            })
            made += 1
    return missions


def _quat_xyzw(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _sim_episode(mission, clearance=False):
    """What the simulator bridge needs: the objects to place, the candidate starts and some
    goal point for the environment's own bookkeeping (unused by the mission's scoring);
    `clearance` asks it for each move's free distance with every state (privileged mode)."""
    present = [t for t in mission["targets"] if t["position"] is not None]
    first = mission["start_candidates"][0]
    goal = present[0]["position"] if present else [first[0] + 1000.0, first[1], first[2]]
    return {
        "map_name": mission["map_name"], "episode_id": mission["mission_id"],
        "objects": [{"asset": t["asset"], "position": t["position"],
                     "rotation_wxyz": t["rotation_wxyz"]} for t in present],
        "start_candidates": [[c[0], c[1], c[2]] + _quat_xyzw(c[3])
                             for c in mission["start_candidates"]],
        "goal_position": goal,
        "description": mission["targets"][0]["description"],
        "mock_targets": [{"position": t["position"], "description": t["description"]}
                         for t in present],
        "clearance": bool(clearance),
    }


# --------------------------------------------------------------------------- scoring
class MissionOutcome(object):
    def __init__(self, target_scores, collision, path_length, budget, steps, termination,
                 errors=None, aborted=False, visited=None):
        self.target_scores = list(target_scores)
        self.visited = list(visited or [])
        self.collision = bool(collision)
        self.path_length = float(path_length)
        self.budget = int(budget)
        self.steps = int(steps)
        self.termination = str(termination)
        self.errors = list(errors or [])
        self.aborted = bool(aborted)

    @property
    def target_mean(self):
        """May be negative: a mission is not clamped, the run's mean is."""
        if not self.target_scores:
            return 0.0
        return sum(self.target_scores) / len(self.target_scores)

    @property
    def efficiency(self):
        return 1.0 - ENERGY_WEIGHT * min(1.0, self.path_length / max(1.0, float(self.budget)))

    @property
    def mission_score(self):
        if self.aborted:
            return 0.0
        mean = self.target_mean
        if mean <= 0.0:               # discounts never soften a penalty
            return mean
        return mean * self.efficiency * (COLLISION_FACTOR if self.collision else 1.0)

    def record(self):
        return {"mission_score": round(self.mission_score, 4),
                "targets": [round(s, 3) for s in self.target_scores],
                "errors_m": [None if e is None else round(e, 2) for e in self.errors],
                "visited": self.visited,
                "collision": self.collision, "aborted": self.aborted,
                "termination": self.termination, "steps": self.steps,
                "path_length_m": round(self.path_length, 2)}


def visits(mission, pose, visited):
    """Mark the present targets the drone at world `pose` is visiting (in place; decoys
    stay None): within VISIT_RADIUS_M horizontally, at most VISIT_HEIGHT_M above the target,
    and close enough below it that the target is inside the down camera's 90 degree view
    (horizontal distance at most the camera's height above the target)."""
    for i, target in enumerate(mission["targets"]):
        where = target["position"]
        if where is None or visited[i]:
            continue
        above = where[2] - pose[2]                 # NED: the drone's height over the target
        near = _hdist(pose, where)
        if near <= VISIT_RADIUS_M and above <= VISIT_HEIGHT_M and near <= above - CAMERA_DROP_M:
            visited[i] = True


def score_targets(mission, reports, start_pose, visited=None):
    """Per-target credit and, for located targets, the error in metres. A present target
    the drone never visited earns nothing for its report (`visited` None: no such rule)."""
    scores, errors = [], []
    if visited is None:
        visited = [True] * len(mission["targets"])
    for target, (status, point), went in zip(mission["targets"], reports, visited):
        if target["position"] is None:                       # a decoy
            scores.append(DECOY_CREDIT if status == ABSENT else 0.0)
            errors.append(None)
            continue
        if status == ABSENT:
            scores.append(FALSE_ABSENT)
            errors.append(None)
        elif status == FOUND:
            world = start_frame_to_world(point, start_pose)
            error = math.sqrt(sum((a - b) ** 2 for a, b in zip(world, target["position"])))
            errors.append(error)
            if not went:
                scores.append(0.0)
            elif error <= FULL_CREDIT_M:
                scores.append(1.0)
            elif error >= ZERO_CREDIT_M:
                scores.append(0.0)
            else:
                scores.append((ZERO_CREDIT_M - error) / (ZERO_CREDIT_M - FULL_CREDIT_M))
        else:
            scores.append(0.0)
            errors.append(None)
    return scores, errors


def run_mission(sim, mission, ask, budget, probe=None, privileged=False):
    """Fly one mission. `ask(t, obs)` returns the raw action; `probe(t, obs)` returns None
    (not probed) or the REAL raw action after a probed double query. With `privileged` (the
    grader's oracle-token mode) every observation also carries `privileged_view`. Wire
    failures and ProtocolError propagate (the caller charges an aborted mission)."""
    n = len(mission["targets"])
    episode = _sim_episode(mission, clearance=privileged)
    state = sim.reset(episode, budget=budget, success_radius=3.0)
    # the start frame is where the drone actually starts: the simulator may have raised the
    # candidate start to find free space around it
    start_pose = [float(v) for v in state["frame"]["pose"]]
    reports = [(UNKNOWN, None)] * n
    visited = [False if t["position"] is not None else None for t in mission["targets"]]
    visits(mission, state["frame"]["pose"], visited)
    truth = privileged_view(mission, start_pose) if privileged else None
    t = 0
    while not state["done"]:
        obs = agent_observation(state["frame"], mission, start_pose, state["step"], budget)
        if truth is not None:
            obs["privileged"] = dict(truth, clearance=state.get("clearance"))
        raw = probe(t, obs) if probe is not None else None
        if raw is None:
            raw = ask(t, obs)
        move, new_reports = parse_reply(raw, n)
        if isinstance(raw, (list, tuple)) and len(raw) > 1:
            reports = new_reports
        state = sim.step(ACTIONS[move])
        visits(mission, state["frame"]["pose"], visited)
        t += 1
        if t > 4 * budget + 8:
            raise RuntimeError("mission did not terminate; simulator state machine broken")
    scores, errors = score_targets(mission, reports, start_pose, visited)
    return MissionOutcome(scores, state["collision"], state["move_distance"], budget,
                          state["step"], state["termination"], errors, visited=visited)


def aggregate(outcomes):
    n = len(outcomes)
    if n == 0:
        return {"geolocation_score": 0.0}
    mean = lambda values: sum(values) / float(n)  # noqa: E731
    located = [s for o in outcomes for s in o.target_scores]
    return {
        # Clamped once, over the whole run, so the metric stays on a positive scale
        # while wrong absence claims still cancel credit earned elsewhere.
        "geolocation_score": max(0.0, mean([o.mission_score for o in outcomes])),
        "target_credit": sum(max(0.0, s) for s in located) / max(1, len(located)),
        "full_credit_targets": sum(1 for s in located if s >= 1.0) / float(max(1, len(located))),
        "visited_targets": sum(1 for o in outcomes for v in o.visited if v) / float(max(1, sum(
            1 for o in outcomes for v in o.visited if v is not None))),
        "collision_rate": mean([1.0 if o.collision else 0.0 for o in outcomes]),
        "mean_path_m": mean([o.path_length for o in outcomes]),
        "missions_scored": float(n),
    }
