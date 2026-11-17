"""drive_variants - graded stagings of the scenario kinds that practice never shows.

Grading stage only (the practice copy of drive_bridge has no such module). Every hidden
episode names a `variant` and a number of `distractors`; drive_bridge's ScenarioBuilder hands
staging to `stage()` here.

Variants, by kind (`standard` is the practice staging):

    broken_down_vehicle   debris                   street furniture and rubbish in the lane
                          shoulder_pass            a crash closes the car's lane and those to its
                                                   left; the way past is the shoulder on the right
                          spinning_car             a car overtaking in the next lane spins out and
                                                   stops side-on in the car's lane
                          reversing_car            a car ahead stops and reverses toward the car
                                                   until it hits it, unless the car moves over
                          dropped_cargo            a truck ahead sheds its load onto the lane
    occluded_pedestrian   walking_in_lane          walks ahead in the car's lane, same direction
    red_light_runner      crowd_against_light      a group crosses against the light while the car
                                                   has green; some turn back halfway
    oncoming_left_turn    overtaking_oncoming      an oncoming car overtakes a cyclist along the
                                                   centre line, a metre into the car's lane

Distractors need no reaction: a car parked on the shoulder, people standing on the
pavement, street furniture at the kerb, cones in the oncoming lane well before the hazard.
Hitting one counts like any other collision; braking or swerving for one only costs pace.

Every staging records what it drew in `b.staging` (the bridge puts it in `truth["staging"]`,
which never reaches the wire), so the grading side can see the range its episodes cover.

The variant table and `graded_specs` are plain Python (the grader imports them); everything that
touches CARLA runs inside the bridge.
"""

import math

VARIANTS = {
    "broken_down_vehicle": ("debris", "reversing_car", "dropped_cargo", "shoulder_pass",
                            "spinning_car"),
    "occluded_pedestrian": ("walking_in_lane",),
    "red_light_runner": ("crowd_against_light",),
    "oncoming_left_turn": ("overtaking_oncoming",),
}
#: The graded episodes, by variant: only the shifted stagings, three of each.
GRADED_MIX = tuple((variant, 3) for variant in (
    "debris", "walking_in_lane", "shoulder_pass", "spinning_car", "overtaking_oncoming",
    "reversing_car", "dropped_cargo", "crowd_against_light"))
MAX_DISTRACTORS = 2

STATIC_KINDS = ("construction_zone", "broken_down_vehicle", "occluded_pedestrian")
PROPS_DEBRIS = ("static.prop.trashcan01", "static.prop.trashcan03", "static.prop.box01",
                "static.prop.box03", "static.prop.shoppingcart", "static.prop.garbage01",
                "static.prop.plasticchair", "static.prop.haybale", "static.prop.clothcontainer")
PROPS_KERB = ("static.prop.mailbox", "static.prop.trashcan02", "static.prop.bench01",
              "static.prop.vendingmachine", "static.prop.plantpot01")
PROPS_CARGO = ("static.prop.box01", "static.prop.box02", "static.prop.box03")  # barrels roll on
BIKES = ("vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets")
TRUCKS = ("vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv",
          "vehicle.mercedes.sprinter")
#: shoulder_pass's warning signs, each with its right half-width (metres across the lane from
#: its origin: the larger of CARLA's box and the collision footprint that
#: review/measure_footprints.py measured, solution/footprints.json), and the car's width (the
#: box of vehicle.lincoln.mkz_2020). A sign keeps the author's place at the lane centre wherever
#: that leaves the car SIGN_SPARE_M of spare width to pass it on the shoulder.
SIGN_HALF_RIGHT_M = {"static.prop.warningaccident": 0.663, "static.prop.trafficwarning": 1.436}
EGO_WIDTH_M = 1.836
SIGN_SPARE_M = 1.0


#: Variants that need a road some towns lack, and the towns that have it (Town10HD's
#: shoulders are 2 m wide, too narrow to pass a crash on; Town04's run along highway rails
#: that leave less room than the lane says).
TOWNS_FOR = {"shoulder_pass": ("Town05",)}


def graded_cells(towns):
    """The graded episodes' (town, variant) cells in grading order: GRADED_MIX over `towns`
    (cycled; a variant listed in TOWNS_FOR cycles over the ones it allows), grouped by town so
    each town loads once. The fixed hidden set holds one episode per cell, in this order."""
    cells, index = [], 0
    for variant, count in GRADED_MIX:
        allowed = [t for t in towns if t in TOWNS_FOR.get(variant, towns)] or towns
        for k in range(count):
            cells.append((towns[index % len(towns)] if variant not in TOWNS_FOR else
                          allowed[k % len(allowed)], variant))
            index += 1
    cells.sort(key=lambda c: towns.index(c[0]))
    return cells


def graded_specs(rng, towns):
    """Episode specs over graded_cells(towns), each with a 62-bit seed and a distractor count
    from `rng` (a maintainer's generator: the grader itself runs the fixed hidden set)."""
    kind_of = {v: kind for kind, variants in VARIANTS.items() for v in variants}
    specs, index = [], 0
    for variant, count in GRADED_MIX:
        allowed = [t for t in towns if t in TOWNS_FOR.get(variant, towns)] or towns
        for k in range(count):
            seed = int(rng.random() * (1 << 31)) * (1 << 31) + int(rng.random() * (1 << 31))
            town = towns[index % len(towns)] if variant not in TOWNS_FOR else allowed[k % len(allowed)]
            specs.append({"town": town, "kind": kind_of[variant],
                          "variant": variant, "seed": seed,
                          "distractors": int(rng.random() * (MAX_DISTRACTORS + 1))})
            index += 1
    specs.sort(key=lambda s: towns.index(s["town"]))
    return specs


def junction_wanted(variant, default):
    """The junction the route has to take: only crowd_against_light (straight across) needs
    one."""
    return default if variant == "crowd_against_light" else None


def route(b, want):
    """A route laid out around a spot the variant needs, or None for the builder's own.
    overtaking_oncoming needs an oncoming lane right beside the car's (which most of Town04,
    divided highways, lacks) and shoulder_pass a shoulder or parking lane on the right: pick
    such a spot, walk back to the start and drive on past it to the route's length."""
    fits = SPOT_TESTS.get(b.variant)
    if fits is None:
        return None
    rng, RoadOption = b.rng, b.RoadOption
    cache = getattr(b.b, "_variant_spots", None)        # per loaded map and variant
    if cache is None or cache[0] is not b.map:
        cache = (b.map, {})
        b.b._variant_spots = cache
    if b.variant not in cache[1]:
        spots = [wp for wp in b.map.generate_waypoints(4.0) if fits(b, wp)]
        if b.variant == "shoulder_pass":
            standing = _map_obstacles(b)
            spots = [wp for wp in spots if _shoulder_clear(b, wp, standing)]
        cache[1][b.variant] = spots
    spots = cache[1][b.variant]
    if not spots:
        raise RuntimeError("no road for {} in this town".format(b.variant))
    hazard = rng.choice(spots)
    back, wp, run = [], hazard, 0.0
    target = rng.uniform(*b.HAZARD_M)
    while run < target:
        prev = wp.previous(2.0)
        if not prev:
            raise RuntimeError("the road ends behind the closure")
        nxt = rng.choice(prev)
        run += nxt.transform.location.distance(wp.transform.location)
        back.append(nxt)
        wp = nxt
    wps = list(reversed(back)) + [hazard]
    run_in = 0.0
    for prev, cur in zip(wps, wps[1:]):
        if cur.is_junction:
            if run_in < 25.0:
                raise RuntimeError("the route would start at a junction")
            break
        run_in += cur.transform.location.distance(prev.transform.location)
    total = run
    wp = hazard
    while total < b.ROUTE_M:
        nexts = wp.next(2.0)
        if not nexts:
            raise RuntimeError("the road ends ahead of the closure")
        if len(nexts) > 1:
            turns = [(b._turn(n), n) for n in nexts]
            nexts = [n for a, n in turns if abs(a) < 30] or [n for _, n in turns]
        nxt = rng.choice(nexts)
        total += nxt.transform.location.distance(wp.transform.location)
        wps.append(nxt)
        wp = nxt
    opts = [RoadOption.LANEFOLLOW]
    for prev, cur in zip(wps, wps[1:]):
        if cur.is_junction and not prev.is_junction:
            angle = b._turn(cur)
            opts.append(RoadOption.LEFT if angle < -45 else
                        RoadOption.RIGHT if angle > 45 else RoadOption.STRAIGHT)
        elif cur.is_junction:
            opts.append(opts[-1])
        else:
            opts.append(RoadOption.LANEFOLLOW)
    b.planned_index = len(back)
    return list(zip(wps, opts)), None


def _two_lane_straight(b, wp):
    """The car's lane is the one next to the centre line (the oncoming lane right beside it),
    with one or two lanes each way, unchanged and fairly straight from 30 m before to 40 m
    after, with no junction on that stretch."""
    same, opposite = _carriageway(b, wp)
    if opposite is None or same[-1].lane_id != wp.lane_id or len(same) > 2 or wp.is_junction:
        return False
    yaw = wp.transform.rotation.yaw
    probes = [w for d in (10.0, 20.0, 30.0) for w in wp.previous(d)[:1]] + \
        [w for d in (10.0, 20.0, 30.0, 40.0) for w in wp.next(d)[:1]]
    if len(probes) != 7:
        return False
    for probe in probes:
        lanes, opp = _carriageway(b, probe)
        turn = abs((probe.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0)
        if opp is None or len(lanes) != len(same) or probe.is_junction or turn > 20.0:
            return False
    return True


def _shoulder_fits(b, wp):
    """The car's lane is the rightmost driving lane, with a shoulder or parking lane at least
    2.4 m wide to its right from 30 m before to 25 m after, and no junction within 12 m
    (route() also wants that shoulder clear of the map's poles and signs: _shoulder_clear)."""
    kinds = (b.carla.LaneType.Shoulder, b.carla.LaneType.Parking)
    if wp.is_junction or any(w.is_junction for w in wp.previous(12.0) + wp.next(12.0)):
        return False
    probes = [wp] + wp.previous(30.0)[:1] + wp.previous(15.0)[:1] + wp.next(12.0)[:1] + \
        wp.next(25.0)[:1]
    if len(probes) != 5:
        return False
    for w in probes:
        right = w.get_right_lane()
        if w.is_junction or right is None or right.lane_type not in kinds or right.lane_width < 2.4:
            return False
    return True


def _obstacle_grid(points):
    """(x, y, radius) points in 10 m cells, for _shoulder_clear."""
    grid = {}
    for x, y, radius in points:
        grid.setdefault((int(x // 10.0), int(y // 10.0)), []).append((x, y, radius))
    return grid


def _map_obstacles(b):
    """Where the map's poles, traffic signs and traffic lights stand at a car's height, as an
    _obstacle_grid: CARLA keeps them as map objects (and traffic.* actors), which no ground-truth
    block lists. Town05 has speed-limit signs standing on its shoulders."""
    carla, points = b.carla, []
    for label in (carla.CityObjectLabel.Poles, carla.CityObjectLabel.TrafficSigns,
                  carla.CityObjectLabel.TrafficLight):
        for thing in b.world.get_environment_objects(label):
            c, e = thing.bounding_box.location, thing.bounding_box.extent
            ground = b.map.get_waypoint(c, project_to_road=True, lane_type=carla.LaneType.Any)
            z = ground.transform.location.z if ground is not None else c.z
            if c.z + e.z - z < 0.05 or c.z - e.z - z > 1.8:
                continue                          # above or below a car
            points.append((c.x, c.y, math.hypot(e.x, e.y)))
    for actor in b.world.get_actors().filter("traffic.*"):
        loc = actor.get_location()
        points.append((loc.x, loc.y, 0.1))
    return _obstacle_grid(points)


def _shoulder_clear(b, wp, standing):
    """Nothing of the map stands in the shoulder (or parking lane) beside the stretch a pass
    round a closure at `wp` uses, from 30 m before it to 40 m after, short of the 0.3 m along
    its outer edge (signs on the kerb line): there the shoulder is the way past."""
    kinds = (b.carla.LaneType.Shoulder, b.carla.LaneType.Parking)
    stretch = [w for d in range(2, 31, 2) for w in wp.previous(float(d))[:1]] + [wp] + \
        [w for d in range(2, 41, 2) for w in wp.next(float(d))[:1]]
    for w in stretch:
        right = w.get_right_lane()
        if right is None or right.lane_type not in kinds:
            continue
        tf = w.transform
        f, r = tf.get_forward_vector(), tf.get_right_vector()
        cx, cy = int(tf.location.x // 10.0), int(tf.location.y // 10.0)
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for x, y, radius in standing.get((i, j), ()):
                    dx, dy = x - tf.location.x, y - tf.location.y
                    across = dx * r.x + dy * r.y
                    if abs(dx * f.x + dy * f.y) <= 1.0 + radius and \
                            w.lane_width / 2.0 - radius <= across <= \
                            w.lane_width / 2.0 + right.lane_width - 0.3 + radius:
                        return False
    return True


SPOT_TESTS = {"overtaking_oncoming": _two_lane_straight, "shoulder_pass": _shoulder_fits}


# --------------------------------------------------------------------------- staging
def stage(b, route, junction):
    """Stage `b.kind` as `b.variant` on `route`, then the distractors before the hazard.
    Raises the bridge's redraw (via b._at_distance or RuntimeError) when a draw does not fit."""
    variant = b.variant
    if variant in SPOT_TESTS:
        index = b.planned_index
        globals()["_" + variant](b, route, index, route[index][0])
    elif b.kind in STATIC_KINDS:
        index, hwp = _hazard_waypoint(b, route, passable=_needs_pass(b.kind, variant))
        if variant == "standard":
            getattr(b, "_stage_" + b.kind)(hwp)
        else:
            globals()["_" + variant](b, route, index, hwp)
    else:
        index, entry = junction
        if variant == "standard":
            getattr(b, "_stage_" + b.kind)(route, index, entry)
        else:
            globals()["_" + variant](b, route, index, entry)
    # a spin-out is staged on a clear stretch: nothing else on or beside the road
    b.quiet = variant == "spinning_car"
    if not b.quiet:
        _distractors(b, route, index)
    return index


def _needs_pass(kind, variant):
    return variant in ("debris", "dropped_cargo", "walking_in_lane", "reversing_car",
                       "spinning_car")


def _hazard_waypoint(b, route, passable):
    index, hwp = b._at_distance(route, b.rng.uniform(*b.HAZARD_M))
    if hwp.is_junction or any(w.is_junction for w in hwp.previous(12.0) + hwp.next(12.0)):
        raise RuntimeError("hazard too close to a junction")
    if passable and not b._passable(hwp):
        raise RuntimeError("no passing lane")
    return index, hwp


def _ground(b, loc):
    """The road surface's height under a point (scripted actors sit on it, not above it)."""
    here = b.map.get_waypoint(loc, project_to_road=True, lane_type=b.carla.LaneType.Any)
    return here.transform.location.z if here is not None else loc.z


def _available(b, names):
    lib = b.world.get_blueprint_library()
    found = [n for n in names if len(lib.filter(n)) > 0]
    if not found:
        raise RuntimeError("none of {} exists in this CARLA".format(names[:3]))
    return found


def _yawed(b, tf, degrees):
    return b.carla.Transform(tf.location, b.carla.Rotation(pitch=tf.rotation.pitch,
                                                           yaw=tf.rotation.yaw + degrees,
                                                           roll=tf.rotation.roll))


def _car_blueprints(b):
    lib = b.world.get_blueprint_library().filter("vehicle.*")
    return [bp.id for bp in lib if int(bp.get_attribute("number_of_wheels")) == 4]


def _small_cars(b):
    """Four-wheelers no longer than an ordinary car (no vans, buses or trucks)."""
    lib = b.world.get_blueprint_library().filter("vehicle.*")
    big = ("sprinter", "fusorosa", "carlacola", "european_hgv", "firetruck", "ambulance", "t2",
           "cybertruck", "patrol", "charger_police")
    return [bp.id for bp in lib if int(bp.get_attribute("number_of_wheels")) == 4
            and not any(k in bp.id for k in big)]


def _walker(b, transform):
    walkers = b.world.get_blueprint_library().filter("walker.pedestrian.*")
    bp = b.rng.choice(walkers)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")
    walker = b.world.spawn_actor(bp, transform)
    b.b.actors.append(walker)
    return walker


# ---- construction_zone, broken_down_vehicle --------------------------------------
def _debris(b, route, index, wp):
    rng = b.rng
    props = _available(b, PROPS_DEBRIS)
    drawn = []
    for _ in range(rng.randint(2, 4)):
        name, ahead, lateral = rng.choice(props), rng.uniform(0.0, 4.0), rng.uniform(-1.2, 1.2)
        yaw = rng.uniform(0, 360)
        b.hazards.append(b.b._spawn(name, _yawed(b, b._offset(wp, ahead, lateral, 0.1), yaw)))
        drawn.append([name, round(ahead, 2), round(lateral, 2), round(yaw, 1)])
    name = rng.choice(props)
    b.hazards.append(b.b._spawn(name, b._offset(wp, 2.0, 0.0, 0.1)))
    drawn.append([name, 2.0, 0.0, 0.0])
    left = wp.get_left_lane()
    b.staging.update(props=drawn, lane_width=round(wp.lane_width, 2),
                     passing_lane=None if left is None else
                     ("same" if (left.lane_id > 0) == (wp.lane_id > 0) else "oncoming"))


def _carriageway(b, wp):
    """The lanes of the car's direction at `wp` (driving lanes with its lane_id sign, from the
    rightmost) and the first oncoming driving lane directly left of them, or None."""
    Driving = b.carla.LaneType.Driving
    same, w = [wp], wp
    while True:
        right = w.get_right_lane()
        if right is None or right.lane_type != Driving or (right.lane_id > 0) != (wp.lane_id > 0):
            break
        same.insert(0, right)
        w = right
    w = wp
    while True:
        left = w.get_left_lane()
        if left is None or left.lane_type != Driving:
            return same, None
        if (left.lane_id > 0) != (wp.lane_id > 0):
            return same, left
        same.append(left)
        w = left


# ---- occluded_pedestrian ------------------------------------------------------------
def _walking_in_lane(b, route, index, wp):
    """A pedestrian walking ahead in the car's lane, in the car's direction."""
    carla, rng = b.carla, b.rng
    lateral = rng.uniform(-0.3, 0.8)
    walker = _walker(b, b._offset(wp, 0.0, lateral, 1.0))
    b.hazards.append(walker)
    speed = rng.uniform(0.9, 1.6)
    left = wp.get_left_lane()
    b.staging.update(walker=walker.type_id, lateral=round(lateral, 2), speed=round(speed, 2),
                     lane_width=round(wp.lane_width, 2),
                     passing_lane=None if left is None else
                     ("same" if (left.lane_id > 0) == (wp.lane_id > 0) else "oncoming"))
    plan = [w for w, _ in route[index:index + 40]]

    def hook():
        here = walker.get_location()
        nearest = min(plan, key=lambda w: w.transform.location.distance(here)) if plan else wp
        f = nearest.transform.get_forward_vector()
        walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(f.x, f.y, 0.0),
                                                 speed=speed))
    b.b.scripted.append(hook)


def _room_beside(b, loc, half_right):
    """The spare width beside a prop standing at `loc` in the car's lane, for the car to pass it
    on the right: from the prop's right side to the outer edge of the shoulder or parking lane
    next to that lane (none there: the lane's own edge), less the car's width."""
    here = b.map.get_waypoint(loc)
    right = here.get_right_lane()
    kinds = (b.carla.LaneType.Shoulder, b.carla.LaneType.Parking)
    shoulder = right.lane_width if right is not None and right.lane_type in kinds else 0.0
    r = here.transform.get_right_vector()
    across = (loc.x - here.transform.location.x) * r.x + (loc.y - here.transform.location.y) * r.y
    return here.lane_width / 2.0 + shoulder - (across + half_right) - EGO_WIDTH_M


def _shoulder_pass(b, route, index, wp):
    """A crash: two cars stopped askew in the car's lane, every lane to its left closed at
    the same place (barriers), a warning sign 8 m before it at the lane centre. The shoulder
    (or parking lane) on the right is the only way past. Where a centred sign would leave the
    car less than SIGN_SPARE_M of spare width beside it, the sign stands on the lane's left
    like the wrecks instead (no spot in Town05 is that narrow)."""
    carla, rng = b.carla, b.rng
    Driving = carla.LaneType.Driving
    cars = _small_cars(b)
    # askew, but kept to the lane's left side so the shoulder stays wide enough to pass
    car1, lat1, yaw1 = rng.choice(cars), rng.uniform(-0.9, -0.7), rng.uniform(5.0, 15.0)
    wreck1 = b.b._spawn(car1, _yawed(b, b._offset(wp, 0.0, lat1, 0.3), yaw1))
    car2, ahead2, lat2 = rng.choice(cars), rng.uniform(5.0, 6.5), rng.uniform(-0.9, -0.7)
    yaw2 = rng.uniform(-15.0, -5.0)
    wrecks = [wreck1, b.b._spawn(car2, _yawed(b, b._offset(wp, ahead2, lat2, 0.3), yaw2))]
    for wreck in wrecks:        # on their wheels (physics on), handbrake set
        wreck.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
    b.hazards.extend(wrecks)
    w, closed = wp, 0
    for _ in range(4):
        # walk outward from the car's side, across the centre line if need be
        nxt = w.get_left_lane() if (w.lane_id > 0) == (wp.lane_id > 0) else w.get_right_lane()
        if nxt is None or nxt.lane_type != Driving:
            break
        for lateral in (-nxt.lane_width / 4.0, nxt.lane_width / 4.0):
            b.hazards.append(b.b._spawn("static.prop.streetbarrier",
                                        b._offset(nxt, 0.0, lateral, 0.1)))
        w, closed = nxt, closed + 1
    sign = _available(b, ("static.prop.warningaccident", "static.prop.trafficwarning"))
    # The sign keeps the author's place at the lane centre, unless the shoulder beside it is
    # too narrow for the car to pass with SIGN_SPARE_M to spare. The left lateral is drawn
    # either way, so no later draw depends on the choice.
    sign_name, left_lat = rng.choice(sign), rng.uniform(-0.9, -0.7)
    room = _room_beside(b, b._offset(wp, -8.0, 0.0, 0.1).location, SIGN_HALF_RIGHT_M[sign_name])
    sign_lat = 0.0 if room >= SIGN_SPARE_M else left_lat
    b.hazards.append(b.b._spawn(sign_name, b._offset(wp, -8.0, sign_lat, 0.1)))
    right = wp.get_right_lane()
    b.staging.update(
        shoulder_width=round(right.lane_width, 2) if right is not None else None,
        shoulder_type=str(right.lane_type) if right is not None else None,
        lane_width=round(wp.lane_width, 2), lanes_closed=closed,
        wrecks=[[w_.type_id, round(lat, 2), round(yaw, 1)] for w_, lat, yaw in
                zip(wrecks, (lat1, lat2), (yaw1, yaw2))],
        wreck_gap=round(ahead2, 2), sign=sign_name, sign_lateral=round(sign_lat, 2),
        sign_room=round(room, 2))


def _lead(b, route, wp, blueprints, speed_kmh):
    """A vehicle 25 m ahead in the car's lane, held until the car comes within 18 m, then
    driven along the route (and on past its end) by a BasicAgent."""
    from agents.navigation.basic_agent import BasicAgent
    carla, rng = b.carla, b.rng
    start_index, start_wp = b._at_distance(route, 25.0)
    tf = start_wp.transform
    car = b.b._spawn(rng.choice(blueprints),
                     carla.Transform(tf.location + carla.Location(z=0.5), tf.rotation))
    car.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))   # held while it settles
    b.hazards.append(car)
    agent = BasicAgent(car, target_speed=speed_kmh,
                       opt_dict={"ignore_traffic_lights": True, "ignore_vehicles": True})
    beyond, last = [], route[-1][0]
    for _ in range(40):
        nxt = last.next(2.0)
        if not nxt:
            break
        last = nxt[0]
        beyond.append((last, b.RoadOption.LANEFOLLOW))
    agent.set_global_plan(route[start_index:] + beyond, stop_waypoint_creation=True,
                          clean_queue=True)
    return car, agent


def _reversing_car(b, route, index, wp):
    """A car ahead in the car's lane stops at the hazard point and, once the car has stopped
    behind it, reverses toward it slowly (reversing lights on) for up to 35 m without looking:
    a driver backing up to a space they passed. Waiting behind it ends in a collision; the way
    out is the lane to the left."""
    carla, rng = b.carla, b.rng
    lead_kmh = rng.uniform(22.0, 30.0)
    car, agent = _lead(b, route, wp, _small_cars(b), lead_kmh)
    stop_at = wp.transform.location
    back_m = 35.0
    speed_back = rng.uniform(0.8, 1.2)
    dt = b.world.get_settings().fixed_delta_seconds or 0.05
    lights = carla.VehicleLightState
    state = {"go": False, "phase": "drive", "since": 0.0, "from": None}
    b.staging.update(lead=car.type_id, lead_kmh=round(lead_kmh, 2), back_speed=round(speed_back, 2),
                     reverse_started=False)

    def lane_walk(start, along, metres):
        """Points of one lane from `start`, `metres` along its own direction (along=+1) or
        against it (-1), every 2 m, across road ids."""
        out, w, run = [], start, 0.0
        while run < metres:
            nxt = w.next(2.0) if along > 0 else w.previous(2.0)
            if not nxt:
                break
            run += nxt[0].transform.location.distance(w.transform.location)
            w = nxt[0]
            out.append((w, run))
        return out

    def left_free():
        """The way out is open: the lane to the car's left holds no other vehicle from 60 m
        behind the car to 30 m ahead of it (walked along the lane, across road ids), and, when
        it is an oncoming lane, no vehicle in it would reach that stretch within 8 s."""
        ego_wp = b.map.get_waypoint(b.b.ego.get_location())
        if ego_wp is None:
            return True
        left = ego_wp.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving:
            return True
        oncoming = (left.lane_id > 0) != (ego_wp.lane_id > 0)
        # the car's forward is the left lane's direction, or against it for an oncoming lane
        ahead = lane_walk(left, -1 if oncoming else 1, 30.0 + (8.0 * 16.0 if oncoming else 0.0))
        behind = lane_walk(left, 1 if oncoming else -1, 60.0)
        stretch = [(left, 0.0)] + [(w, run) for w, run in ahead] + [(w, -run) for w, run in behind]
        width = left.lane_width / 2.0 + 0.5
        for other in b.world.get_actors().filter("vehicle.*"):
            if other.id in (car.id, b.b.ego.id):
                continue
            loc = other.get_location()
            near = min(stretch, key=lambda p: p[0].transform.location.distance(loc))
            if near[0].transform.location.distance(loc) > width:
                continue
            if near[1] <= 30.0:
                return False                    # in the lane, beside or close to the car
            v = other.get_velocity()
            f = near[0].transform.get_forward_vector()
            toward = v.x * f.x + v.y * f.y      # ahead of the car an oncoming lane runs toward it
            if toward > 0.1 and near[1] / toward < 8.0:
                return False
        return True

    def hook():
        now = b.b.tracker.time
        if not state["go"]:
            if car.get_location().distance(b.b.ego.get_location()) < 18.0:
                state["go"] = True
            else:
                car.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
                return
        phase = state["phase"]
        if phase == "drive" and car.get_location().distance(stop_at) < 2.5:
            state.update(phase="stop", since=now)
        if phase == "stop":
            car.apply_control(carla.VehicleControl(brake=1.0))
            # it backs up only once the car has stopped behind it (for a second) and has a
            # way round (the lane to its left free of traffic from 60 m behind the car to
            # 30 m ahead); until then it is just a stopped car
            if b.b._speed() < 0.2:
                state["ego_still"] = state.get("ego_still", 0.0) + dt
            else:
                state["ego_still"] = 0.0
            ready = state["ego_still"] >= 1.0 and left_free()
            if now - state["since"] > 1.0 and ready:
                # backed kinematically along its lane: a reverse gear under physics is not
                # dependable in CARLA
                tf = car.get_transform()
                state.update(phase="reverse", since=now, moved=0.0, tf=tf)
                car.set_simulate_physics(False)
                car.set_light_state(lights(lights.Reverse))
                b.staging.update(reverse_started=True, reverse_at_s=round(now, 2))
            return
        if phase == "reverse":
            state["moved"] = min(back_m, state["moved"] + speed_back * dt)
            tf = state["tf"]
            f = tf.get_forward_vector()
            car.set_transform(carla.Transform(tf.location - carla.Location(
                f.x * state["moved"], f.y * state["moved"], 0.0), tf.rotation))
            if state["moved"] >= back_m:
                state.update(phase="pause", since=now)
                car.set_simulate_physics(True)
                car.set_light_state(lights(lights.NONE))
            return
        if phase == "drive":
            car.apply_control(agent.run_step())
        else:
            car.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))   # parked
    b.b.scripted.append(hook)


def _dropped_cargo(b, route, index, wp):
    """A truck ahead in the car's lane sheds two to four pieces of its load as it passes the
    hazard point; they fall from its tail and come to rest in the car's lane, the next lane
    left open.

    The truck stands on its handbrake until it has stood still for a second and the car is
    within 30 m. Every piece is drawn here, at staging: its blueprint, where it comes to rest
    (1.5-4.0 m plus 1.2 m per piece behind the hazard point, its centre 0.3 m left to 0.9 m
    right of the lane centre with its footprint inside the lane), its final yaw and its spin.
    The fall is scripted (physics off: free fall from the tail, a decaying slide and tumble
    that never outruns the truck), so where the load lies follows from the seed."""
    carla, rng = b.carla, b.rng
    truck_kmh = rng.uniform(25.0, 32.0)
    truck, agent = _lead(b, route, wp, _available(b, TRUCKS), truck_kmh)
    cargo = _available(b, PROPS_CARGO)
    drop_at = wp.transform.location
    dt = b.world.get_settings().fixed_delta_seconds or 0.05
    plan = []
    for k in range(rng.randint(2, 4)):
        name, back = rng.choice(cargo), rng.uniform(1.5, 4.0) + 1.2 * k
        lateral, yaw = rng.uniform(-0.3, 0.9), rng.uniform(0.0, 360.0)
        spin = rng.choice((1.0, -1.0)) * rng.uniform(90.0, 360.0)
        tumble = rng.uniform(15.0, 45.0)
        rest_wp = (wp.previous(back) or [wp])[0]
        plan.append({"name": name, "back": back, "lateral": lateral, "yaw": yaw, "spin": spin,
                     "tumble": tumble, "rest_wp": rest_wp, "k": k})
    state = {"go": False, "dropped": False, "stood": 0.0}
    falling = []
    b.staging.update(truck=truck.type_id, truck_kmh=round(truck_kmh, 2), pieces=len(plan),
                     lane_width=round(wp.lane_width, 2),
                     planned=[[p["name"], round(p["back"], 2), round(p["lateral"], 2),
                               round(p["yaw"], 1)] for p in plan])

    def rest_transform(prop, p):
        """Where the piece comes to rest: flat on the road, inside the car's lane."""
        box = prop.bounding_box
        rel = math.radians(p["yaw"])
        half = abs(box.extent.x * math.sin(rel)) + abs(box.extent.y * math.cos(rel))
        room = max(0.0, p["rest_wp"].lane_width / 2.0 - half - 0.05)
        lateral = max(-min(0.3, room), min(min(0.9, room), p["lateral"]))
        tf = b._offset(p["rest_wp"], 0.0, lateral, 0.0)
        tf.location.z = _ground(b, tf.location) - box.location.z + box.extent.z
        p["rest_lateral"] = lateral
        return carla.Transform(tf.location, carla.Rotation(yaw=tf.rotation.yaw + p["yaw"]))

    def drop():
        tf = truck.get_transform()
        f, r = tf.get_forward_vector(), tf.get_right_vector()
        tail = truck.bounding_box.extent.x + 0.9
        v = truck.get_velocity()
        speed = max(1.0, math.sqrt(v.x * v.x + v.y * v.y))
        lib = b.world.get_blueprint_library()
        for p in plan:
            prop = None
            for lift in range(5):           # a deterministic sequence of release heights
                loc = tf.location + carla.Location(-f.x * tail + r.x * p["lateral"],
                                                   -f.y * tail + r.y * p["lateral"],
                                                   1.4 + 0.5 * p["k"] + 0.6 * lift)
                start = carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw + p["yaw"] - p["spin"]))
                prop = b.world.try_spawn_actor(lib.find(p["name"]), start)
                if prop is not None:
                    break
            if prop is None:
                p["spawned"] = False
                continue
            prop.set_simulate_physics(False)
            b.b.actors.append(prop)
            b.hazards.append(prop)
            b.b.tracker.hazard_ids.add(prop.id)     # hitting the load counts like the hazard
            rest = rest_transform(prop, p)
            height = max(0.2, start.location.z - rest.location.z)
            t_fall = math.sqrt(2.0 * height / 9.81)
            run = math.hypot(rest.location.x - start.location.x, rest.location.y - start.location.y)
            # the slide never starts faster than the truck goes, and ends no earlier than the fall
            p.update(spawned=True, prop=prop, start=start, rest=rest, t=0.0, t_fall=t_fall,
                     t_end=max(t_fall, 2.0 * run / speed), height=height)
            falling.append(p)
        b.staging["landed"] = [[p["name"], round(p["back"], 2), round(p.get("rest_lateral", p["lateral"]), 2)]
                               for p in plan if p.get("spawned")]

    def move(p):
        p["t"] = min(p["t_end"], p["t"] + dt)
        u = p["t"] / p["t_end"]
        ease = 1.0 - (1.0 - u) ** 2                    # decelerating slide
        a, c = p["start"].location, p["rest"].location
        fall = min(p["t"], p["t_fall"])
        z = max(c.z, a.z - 0.5 * 9.81 * fall * fall)
        settle = (1.0 - u) ** 2                        # the tumble dies out as it comes to rest
        rot = carla.Rotation(pitch=p["tumble"] * math.sin(3.0 * math.pi * u) * settle,
                             yaw=p["rest"].rotation.yaw - p["spin"] * (1.0 - ease),
                             roll=0.6 * p["tumble"] * math.sin(2.0 * math.pi * u) * settle)
        loc = carla.Location(a.x + (c.x - a.x) * ease, a.y + (c.y - a.y) * ease, z)
        p["prop"].set_transform(carla.Transform(loc, rot) if u < 1.0 else p["rest"])

    def hook():
        for p in list(falling):
            move(p)
            if p["t"] >= p["t_end"]:
                falling.remove(p)
        if not state["go"]:
            v = truck.get_velocity()
            still = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) < 0.1
            state["stood"] = state["stood"] + dt if still else 0.0
            near = truck.get_location().distance(b.b.ego.get_location()) < 30.0
            if state["stood"] >= 1.0 and near:
                state["go"] = True
                b.staging["released_at_s"] = round(b.b.tracker.time, 2)
            else:
                truck.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
                return
        if not state["dropped"] and truck.get_location().distance(drop_at) < 3.0:
            state["dropped"] = True
            b.staging["dropped_at_s"] = round(b.b.tracker.time, 2)
            drop()
        truck.apply_control(agent.run_step())
    b.b.scripted.append(hook)


def _crowd_against_light(b, route, index, entry):
    """The car has a green light straight across a junction; four to eight people step off
    the kerb on the car's right into the crossing just past the stop line, against their
    light, a little apart. One or two of them stop in the car's lane and turn back. A person
    walking back who has not reached the pavement within the time the way out took plus 2 s
    is put back on their spot, where they stay."""
    carla, rng = b.carla, b.rng
    lights = b.world.get_traffic_lights_from_waypoint(entry, 40.0)
    if not lights:
        raise RuntimeError("no traffic light at this junction")
    for light in lights:
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(120.0)
    kerb = entry.lane_width / 2.0
    right = entry.transform.get_right_vector()
    toward = carla.Vector3D(-right.x, -right.y, 0.0)
    back = carla.Vector3D(right.x, right.y, 0.0)
    crossing = 4.0 * entry.lane_width + 4.0
    people = []
    for k in range(rng.randint(4, 8)):
        spot = b._offset(entry, rng.uniform(1.5, 4.5), kerb + rng.uniform(1.8, 3.5), 1.0)
        try:
            people.append({"walker": _walker(b, spot), "delay": rng.uniform(0.0, 2.5),
                           "speed": rng.uniform(1.1, 1.7), "turn": False, "start": spot.location})
        except RuntimeError:
            continue
    if len(people) < 3:
        raise RuntimeError("no room on the pavement for the group")
    for p in rng.sample(people, rng.randint(1, 2)):
        p["turn"] = True
        p["stop_at"] = kerb + rng.uniform(0.5, 2.0)     # metres in from the kerb: the car's lane
    b.hazards.extend(p["walker"] for p in people)
    state = {"t0": None}
    b.staging.update(group=len(people), turners=sum(1 for p in people if p["turn"]),
                     lane_width=round(entry.lane_width, 2),
                     people=[[round(p["delay"], 2), round(p["speed"], 2),
                              round(p["start"].distance(entry.transform.location), 2),
                              round(p["stop_at"], 2) if p["turn"] else None] for p in people])

    def hook():
        ego = b.b.ego.get_location()
        if state["t0"] is None:
            if ego.distance(entry.transform.location) < b.b._speed() * 3.0 + 12.0:
                state["t0"] = b.b.tracker.time
            return
        elapsed = b.b.tracker.time - state["t0"]
        for p in people:
            w = p["walker"]
            if elapsed < p["delay"]:
                continue
            moved = w.get_location().distance(p["start"])
            if p["turn"]:
                phase = p.setdefault("phase", "out")
                if phase == "out" and moved >= p["stop_at"]:
                    p.update(phase="wait", since=elapsed, out=moved)
                if phase == "wait" and elapsed - p["since"] > 1.5:
                    p.update(phase="back", since=elapsed)
                if p["phase"] == "back":
                    if moved < 0.5:
                        p["phase"] = "done"
                    elif elapsed - p["since"] > p["out"] / p["speed"] + 2.0:
                        # stuck on the way back (a kerb, a car): back on the spot it came from
                        w.set_transform(carla.Transform(p["start"], w.get_transform().rotation))
                        p["phase"] = "done"
                        b.staging["returned_by_hand"] = b.staging.get("returned_by_hand", 0) + 1
                direction, speed = {"out": (toward, p["speed"]), "wait": (toward, 0.0),
                                    "back": (back, p["speed"]), "done": (back, 0.0)}[p["phase"]]
            else:
                p["done"] = p.get("done") or moved >= crossing
                direction, speed = toward, (0.0 if p["done"] else p["speed"])
            w.apply_control(carla.WalkerControl(direction=direction, speed=speed))
    b.b.scripted.append(hook)


def _overtaking_oncoming(b, route, index, wp):
    """An oncoming car pulls out to overtake a cyclist riding near its own kerb and comes along
    the centre line, about a metre into the car's lane, until it is clear of the cyclist.
    Both are moved kinematically along the oncoming lane. Stopping does not clear it: the
    car has to move over to its kerb side until the overtaker is back in its own lane."""
    carla, rng = b.carla, b.rng
    same, opposite = _carriageway(b, wp)
    dt = b.world.get_settings().fixed_delta_seconds or 0.05

    def ahead_of_car(lane_wp, metres):
        # the oncoming lane runs toward the car, so what lies ahead of the car is its `previous`
        out = lane_wp.previous(metres)
        if not out:
            raise RuntimeError("the oncoming lane ends")
        return out[0]

    def path(start, length):
        """Transforms every metre along the oncoming lane, in its direction of travel."""
        points, w = [start.transform], start
        for _ in range(int(length)):
            nxt = w.next(1.0)
            if not nxt:
                break
            w = nxt[0]
            points.append(w.transform)
        return points

    bike_wp = ahead_of_car(opposite, rng.uniform(18.0, 26.0))
    car_wp = ahead_of_car(bike_wp, rng.uniform(14.0, 18.0))
    bike = b.b._spawn(rng.choice(_available(b, BIKES)), carla.Transform(
        bike_wp.transform.location, bike_wp.transform.rotation), physics=False)
    car = b.b._spawn(rng.choice(_car_blueprints(b)), carla.Transform(
        car_wp.transform.location, car_wp.transform.rotation), physics=False)
    b.hazards.extend([bike, car])
    bike_path, car_path = path(bike_wp, 160), path(car_wp, 160)
    # where the bike starts, in the overtaker's path metres
    lead = bike_wp.transform.location.distance(car_wp.transform.location)
    clearance = car.bounding_box.extent.x + bike.bounding_box.extent.x
    intrusion = rng.uniform(0.8, 1.1)
    shift = opposite.lane_width / 2.0 + intrusion - car.bounding_box.extent.y
    v_bike, v_car = rng.uniform(3.0, 4.5), rng.uniform(9.0, 11.0)
    kerb_side = -(opposite.lane_width / 2.0 - 0.8)     # the cyclist keeps to its kerb
    release = rng.uniform(55.0, 70.0)
    state = {"go": False, "u": 0.0, "w": 0.0, "side": 0.0, "passed": False}
    b.staging.update(car=car.type_id, bike=bike.type_id, intrusion=round(intrusion, 2),
                     v_car=round(v_car, 2), v_bike=round(v_bike, 2), release_m=round(release, 1),
                     lane_width=round(wp.lane_width, 2), oncoming_width=round(opposite.lane_width, 2))

    def place(actor, points, u, side):
        i = min(int(u), len(points) - 2)
        a, c = points[i], points[i + 1]
        k = u - i
        loc = carla.Location(a.location.x + (c.location.x - a.location.x) * k,
                             a.location.y + (c.location.y - a.location.y) * k,
                             a.location.z)
        r = a.get_right_vector()
        loc += carla.Location(-r.x * side, -r.y * side, 0.0)     # toward the car's lane
        loc.z = _ground(b, loc)
        actor.set_transform(carla.Transform(loc, a.rotation))

    def hook():
        if not state["go"]:
            if b.b.ego.get_location().distance(car.get_location()) < release:
                state["go"] = True
            else:
                return
        state["u"] = min(state["u"] + v_car * dt, len(car_path) - 2)
        state["w"] = min(state["w"] + v_bike * dt, len(bike_path) - 2)
        # out once close behind the bike, back in only when clear past its front, never
        # sideways faster than a lane change: the two never overlap
        behind = lead + state["w"] - state["u"] - clearance
        if not state["passed"] and state["u"] - (lead + state["w"]) > clearance + 6.0:
            state["passed"] = True
        want = shift if (behind < 12.0 and not state["passed"]) else 0.0
        step = 1.2 * dt
        state["side"] += max(-step, min(step, want - state["side"]))
        place(bike, bike_path, state["w"], kerb_side)
        place(car, car_path, state["u"], state["side"])
    place(bike, bike_path, 0.0, kerb_side)
    b.b.scripted.append(hook)


def _spinning_car(b, route, index, wp):
    """A car in the lane to the left, overtaking the car, loses control at the hazard point:
    it spins through 180-270 degrees while sliding across into the car's lane and stops
    there, side-on. Moved kinematically along the left lane, then across."""
    carla, rng = b.carla, b.rng
    left = wp.get_left_lane()
    if left is None or left.lane_type != carla.LaneType.Driving or \
            (left.lane_id > 0) != (wp.lane_id > 0):
        raise RuntimeError("no lane to the left going the same way")
    if any(w.is_junction for d in range(5, 40, 5) for w in wp.next(float(d))[:1]):
        raise RuntimeError("a junction where the car would have to get round the wreck")
    dt = b.world.get_settings().fixed_delta_seconds or 0.05
    back = left.previous(30.0)
    if not back:
        raise RuntimeError("the left lane ends behind the hazard")
    points, w = [back[0].transform], back[0]
    for _ in range(70):
        nxt = w.next(1.0)
        if not nxt:
            break
        w = nxt[0]
        points.append(w.transform)
    if len(points) < 50:
        raise RuntimeError("the left lane is too short")
    car = b.b._spawn(rng.choice(_small_cars(b)), carla.Transform(
        points[0].location, points[0].rotation), physics=False)
    b.hazards.append(car)
    # from the left lane's centre to a metre right of the car's lane centre: side-on there it
    # closes the car's lane and leaves the left lane open to go round
    across = left.transform.location.distance(wp.transform.location) + 1.0
    spin = rng.choice((1.0, -1.0)) * rng.uniform(180.0, 270.0)
    v_car = rng.uniform(9.0, 11.0)
    slide_m, slide_s = rng.uniform(9.0, 13.0), rng.uniform(1.6, 2.2)
    state = {"go": False, "u": 0.0, "t": None}
    b.staging.update(car=car.type_id, spin=round(spin, 1), v_car=round(v_car, 2),
                     slide_m=round(slide_m, 2), slide_s=round(slide_s, 2),
                     rest_lateral=round(across - left.transform.location.distance(wp.transform.location), 2),
                     lane_width=round(wp.lane_width, 2))

    def pose(u, side, yaw_extra):
        i = min(int(u), len(points) - 2)
        a, c = points[i], points[i + 1]
        k = u - i
        loc = carla.Location(a.location.x + (c.location.x - a.location.x) * k,
                             a.location.y + (c.location.y - a.location.y) * k, a.location.z)
        r = a.get_right_vector()
        loc += carla.Location(r.x * side, r.y * side, 0.0)         # toward the car's lane
        loc.z = _ground(b, loc)
        rot = carla.Rotation(pitch=a.rotation.pitch, yaw=a.rotation.yaw + yaw_extra, roll=0.0)
        return carla.Transform(loc, rot)

    def hook():
        ego = b.b.ego.get_location()
        if not state["go"]:
            if car.get_location().distance(ego) < 20.0 or b.b.tracker.time > 6.0:
                state["go"] = True
            else:
                return
        if state["t"] is None:
            state["u"] = min(state["u"] + v_car * dt, len(points) - 2)
            if state["u"] >= 30.0:                      # the hazard point
                state["t"] = 0.0
            car.set_transform(pose(state["u"], 0.0, 0.0))
            return
        if state["t"] >= slide_s:
            return                                      # at rest, under physics
        state["t"] = min(slide_s, state["t"] + dt)
        f = state["t"] / slide_s
        ease = 1.0 - (1.0 - f) ** 2                     # decelerating slide
        car.set_transform(pose(30.0 + slide_m * ease, across * ease, spin * ease))
        if state["t"] >= slide_s:
            car.set_simulate_physics(True)              # let it settle on its wheels
            car.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
    b.b.scripted.append(hook)


def _straight(route, i, reach=8, max_turn_deg=5.0):
    """Whether the route runs straight around waypoint i: a parked car or a prop set beside
    a bend would stick out into the lane, and a distractor must need no reaction."""
    lo, hi = max(0, i - reach), min(len(route) - 1, i + reach)
    yaws = [route[k][0].transform.rotation.yaw for k in (lo, i, hi)]
    turn = max(abs((a - yaws[1] + 180.0) % 360.0 - 180.0) for a in (yaws[0], yaws[2]))
    return turn <= max_turn_deg


def _road_edge(b, wp):
    """Offset from wp's lane centre to the right edge of the road: half its lane plus every lane
    to its right a car may use (driving, shoulder, parking, bike lanes)."""
    LT = b.carla.LaneType
    edge, w = wp.lane_width / 2.0, wp
    for _ in range(6):
        nxt = w.get_right_lane()
        if nxt is None or nxt.lane_type not in (LT.Driving, LT.Shoulder, LT.Parking, LT.Biking) or \
                (nxt.lane_id > 0) != (wp.lane_id > 0):
            break
        edge += nxt.lane_width
        w = nxt
    return edge


def _distractors(b, route, hazard_index):
    """Things beside the route before the hazard that need no reaction. People and street
    furniture stand beyond the road's edge (past any shoulder, parking or bike lane, which can
    be the way round the hazard), a parked car on the shoulder or parking lane."""
    rng = b.rng
    run, spots = 0.0, []
    for i in range(1, hazard_index):
        run += route[i][0].transform.location.distance(route[i - 1][0].transform.location)
        if 20.0 <= run and i < hazard_index - 12 and not route[i][0].is_junction and \
                _straight(route, i):
            spots.append(route[i][0])
    if not spots:
        return
    kinds = ["parked_car", "pavement_people", "kerb_props", "oncoming_cones"]
    placed = b.staging.setdefault("distractors", [])
    for _ in range(b.distractors):
        wp = rng.choice(spots)
        kind = rng.choice(kinds)
        kerb = wp.lane_width / 2.0
        placed.append(kind)
        try:
            if kind == "parked_car":
                right = wp.get_right_lane()
                if right is not None and right.lane_type == b.carla.LaneType.Driving or \
                        b.variant == "shoulder_pass":
                    continue               # would block a lane (or the way round), not sit beside one
                car = b.b._spawn(rng.choice(_car_blueprints(b)),
                                 b._offset(wp, 0.0, kerb + 1.7, 0.3), physics=False)
                car.set_simulate_physics(False)
            elif kind == "pavement_people":
                edge = _road_edge(b, wp)
                for k in range(rng.randint(2, 4)):
                    _walker(b, b._offset(wp, rng.uniform(-3.0, 3.0), edge + rng.uniform(2.0, 3.0), 1.0))
            elif kind == "kerb_props":
                props = _available(b, PROPS_KERB)
                b.b._spawn(rng.choice(props), b._offset(wp, 0.0, _road_edge(b, wp) + 1.2, 0.1))
            else:
                left = wp.get_left_lane()
                if left is None or left.lane_type != b.carla.LaneType.Driving or \
                        b.variant in SPOT_TESTS:     # that lane is the way round, or the overtaker's
                    continue
                for k in range(3):
                    b.b._spawn("static.prop.constructioncone", b._offset(left, 2.0 * k, 0.0, 0.1))
        except RuntimeError:
            continue                       # a spawn collided: skip this distractor
