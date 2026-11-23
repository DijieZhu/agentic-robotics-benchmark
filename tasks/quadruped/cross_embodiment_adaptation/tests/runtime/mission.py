"""One survey, one irreversible switch, one physical dock attempt."""
import math

import numpy as np

from articulated import (CAMERA_Z, Go1, MAX_SCOUT, MAX_TICKS, PALLET_X, Survey, Views, fixtures_and_launch, layout, look_scene,
                         site_states)
from mobility import fittings
from props import draw


class World:
    def __init__(self, seed, *, flat_control=False):
        self.rng, self.racks, self.goal, self.walls = layout(seed)
        self.posts, self.launch = fixtures_and_launch(seed, self.walls)
        self.props = draw(seed)
        self.fittings = [] if flat_control else fittings(seed, self.racks, self.props)
        self.seed = seed
        self.surveyed, self.found = site_states(seed, self.racks, self.goal, self.posts, self.walls)
        racks, goal, posts, pallet = self.surveyed
        self.racks, self.goal, self.posts, self.pallet = self.found      # what the mission is judged on
        # The charging station stands by the dock's surveyed place in both states: the survey shows nothing of a move.
        station = goal[1]-.4
        self.views = Views({"survey": look_scene(self.walls, goal, posts, self.fittings, pallet, station_y=station),
                            "found": look_scene(self.walls, self.goal, self.posts, self.fittings, self.pallet,
                                                station_y=station)})
        self.survey = Survey(self.walls, goal, posts, self.fittings, pallet, self.props.uav_scale, self.views)
        self.go1 = None
        self.phase = self.scout_actions = 0

    @property
    def done(self):
        return self.go1 is not None and self.go1.outcome != "running"

    def observe(self):
        data = self.go1.observe() if self.phase else {
            **self.survey.observe(), "scout_actions_remaining": MAX_SCOUT-self.scout_actions,
        }
        return {"phase": self.phase, "done": self.done, **data}

    def act(self, action):
        if self.done:
            raise ValueError("the mission has ended")
        if self.phase:
            self.go1.act(action)
        else:
            if self.scout_actions >= MAX_SCOUT:
                raise ValueError("scout action budget spent; advance to the Go1")
            if (not isinstance(action, list) or len(action) != 2
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in action)
                    or math.hypot(*action) > .60):
                raise ValueError("expected [dx, dy] within 0.60 m")
            self.survey.move(np.asarray(action))
            self.scout_actions += 1
        return self.observe()

    def advance(self):
        if self.phase or self.scout_actions < 4:
            raise ValueError("switch exactly once, after at least four scout moves")
        self.survey.close()
        self.go1 = Go1(self.rng, self.walls, self.goal, self.racks, self.posts, self.launch, self.fittings,
                       self.props, MAX_TICKS, self.pallet, self.seed, views=self.views,
                       uav=(float(self.survey.xy[0]), float(self.survey.xy[1]), CAMERA_Z+.12))  # it hovers where it ended
        self.phase = 1
        return self.observe()

    def look(self, detent):
        if not self.phase or self.done:
            raise ValueError("ground camera requires a live target deployment")
        return self.go1.look(detent)

    def metrics(self):
        dog = self.go1
        first, _, second = dog.choices if dog else (None, None, None)
        fell = dog is not None and dog.outcome == "fell"
        wall = dog.contact_wall if dog else None
        return {
            "success": int(dog is not None and dog.outcome == "docked"),
            "switched": self.phase,
            "scout_actions": self.scout_actions,
            "target_actions": dog.actions if dog else 0,
            "target_ticks": dog.ticks if dog else 0,
            "ground_looks": dog.looks if dog else 0,
            # Private ledger only; never an observation or a verifier reward field.
            "launch_pose_world": [*dog.origin.tolist(), dog.origin_yaw] if dog else None,
            "target_contact": int(dog is not None and dog.outcome == "contact"),
            "target_fell": int(fell),
            "target_budget_stop": int(dog is not None and dog.outcome == "budget"),
            "target_outcome": dog.outcome if dog else "not_deployed",
            "contact_wall": wall,
            "contact_part": dog.contact_part if dog else None,
            "duct_entered": int(bool(dog and "duct" in dog.fitting_entered)),
            "duct_cleared": int(bool(dog and "duct" in dog.fitting_cleared)),
            "deck_entered": int(bool(dog and "deck" in dog.fitting_entered)),
            "deck_cleared": int(bool(dog and "deck" in dog.fitting_cleared)),
            "deck_support_feet": len(dog.deck_feet) if dog else 0,
            "min_duct_gap_m": dog.min_duct_gap if dog else None,
            "contact_duct": int(bool(wall and wall.startswith("duct_"))),
            "contact_deck": int(bool(wall and wall.startswith("deck_"))),
            "contact_load_floor": int(wall == "floor"),
            "contact_first": int(bool(wall and wall.startswith("rack1_"))),
            "contact_middle": int(bool(wall and wall.startswith("rack2_"))),
            "contact_second": int(bool(wall and wall.startswith("rack3_"))),
            "contact_boundary": int(wall in ("back", "front", "north", "south")),
            "contact_bollard": int(bool(wall and wall.startswith("bollard"))),
            "corridor_entered": int(first is not None),
            "first_aisle_correct": int(first == "wide"),
            "first_shortcut_selected": int(first == "shortcut"),
            "fell_before_aisle": int(fell and first is None),
            "partition_crossed": int(bool(dog and dog.crossed[0])),
            "middle_crossed": int(bool(dog and dog.crossed[1])),
            "second_aisle_entered": int(second is not None),
            "second_aisle_correct": int(second == "wide"),
            "second_shortcut_selected": int(second == "shortcut"),
            "second_partition_crossed": int(bool(dog and dog.crossed[2])),
            "dock_zone_entered": int(bool(dog and dog.dock_zone_entered)),
            "passage_choices": dog.choices if dog else [None]*3,
            "passage_lateral_margin_m": dog.lateral_margin if dog else [None]*3,
            "max_forward_m": dog.max_x if dog else 0.,
            "travel_m": dog.travel if dog else 0.,
            "final_goal_distance": dog.dock_error if dog else -1.,
        }

    def truth(self):
        """The site as built, for the token holder only (the author's oracle): never an observation."""
        dog = self.go1
        return {
            "racks": [{"near": r.near, "far": r.far, "wide": list(r.wide),
                       "shortcut": list(r.shortcut) if r.shortcut else None} for r in self.racks],
            "goal": self.goal.tolist(), "posts": [list(map(float, p)) for p in self.posts if p is not None],
            "pallet": list(map(float, self.pallet)),
            "surveyed": {"goal": self.surveyed[1].tolist(), "pallet": list(map(float, self.surveyed[3])),
                         "posts": [list(map(float, p)) for p in self.surveyed[2]]},
            "fittings": [{"kind": f.kind, "x": f.x, "lo": f.lo, "hi": f.hi, "length": f.length,
                          "clearance": f.clearance, "rise": f.rise} for f in self.fittings],
            "launch": [float(v) for v in self.launch], "props": self.props.public(),
            "pose": [*map(float, dog.xy), float(dog.heading)] if dog else None,
            "launch_pose": [*dog.origin.tolist(), float(dog.origin_yaw)] if dog else None,
            "trunk_z": float(dog.data.xpos[dog.trunk, 2]) if dog else None,
        }

    def close(self):
        self.survey.close()
        if self.go1:
            self.go1.close()
        self.views.close()


class Pad:
    """The commissioning pad: the same loaded Go1 in a bare bay of the hall among bollards, no site.

    One episode is one deployment with its own draw of the properties (the ranges of props.RANGES), a random
    launch heading, the mast camera and a tick budget. The bay's four boundary barriers stand; there is no rack,
    fitting or dock, no truth, no score
    and no debrief: the observation is the one a graded deployment returns.
    """
    TICKS = 48000

    def __init__(self, seed):
        from articulated import fixtures_and_launch as place, layout as lay
        rng, _, _, walls = lay(seed)
        posts, launch = place(seed, walls)
        self.props = draw(seed, stream=3704)
        # The pad is the hall's bay inside its four boundary barriers: no rack, no fitting, no dock. A pallet on
        # edge and a dock beacon stand in it too, so that seeing them can be commissioned.
        bay = {name: walls[name] for name in ("back", "front", "south", "north")}
        reach_y = walls["north"][2]-.55
        place = np.random.default_rng([seed & ((1 << 63)-1), 3707])
        spots = []
        while len(spots) < 2:
            angle, reach = place.uniform(0, 2*math.pi), place.uniform(1.6, 2.6)
            spot = (launch[0]+reach*math.cos(angle), launch[1]+reach*math.sin(angle))
            if -.45 <= spot[0] <= 6.5 and abs(spot[1]) <= reach_y and all(math.dist(spot, other) >= .9 for other in spots):
                spots.append(spot)
        (px, py), beacon = spots
        pallet = (px-PALLET_X/2, px+PALLET_X/2, py-.40, py+.40)
        self.views = Views({"found": look_scene(bay, None, posts, [], pallet, tuple(beacon))})
        self.go1 = Go1(rng, bay, None, [], posts, launch, [], self.props, self.TICKS, pallet, seed, tuple(beacon),
                       views=self.views)
        self.phase = 1

    @property
    def done(self):
        return self.go1.outcome != "running"

    def observe(self):
        return {"phase": 1, "done": self.done, **self.go1.observe()}

    def act(self, action):
        if self.done:
            raise ValueError("this pad episode has ended; start another with pad()")
        self.go1.act(action)
        return self.observe()

    def look(self, detent):
        if self.done:
            raise ValueError("this pad episode has ended; start another with pad()")
        return self.go1.look(detent)

    def advance(self):
        raise ValueError("the pad has no survey; advance() is a graded call")

    def metrics(self):
        dog = self.go1
        return {"pad": 1, "target_ticks": dog.ticks, "target_actions": dog.actions, "ground_looks": dog.looks,
                "target_outcome": dog.outcome}

    def close(self):
        self.go1.close()
        self.views.close()
