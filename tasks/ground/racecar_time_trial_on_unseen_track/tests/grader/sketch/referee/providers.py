"""Who builds the model and who runs its parts: the real packages (drive model, sensors, venue generator, ducts, obstacles) through their contract entry points
when they can be imported, else the stand-ins of `stubs.py`.

    prov = resolve("auto")            # or "stub" (every part a stand-in), or resolve("auto", sensors=MySuite)
    prov.sources                      # {"venue": "S3" | "stub", "car": ..., "sensors": ..., "ducts": ..., "others": ...}
    model = build_model(world, "avoid", prov)

A provider package that exists but fails to import with an error other than ImportError is not hidden: the error
propagates. Every model is composed in one MjSpec: venue (the venue generator's build_spec, ducts through the duct model), ego car (the drive model's add_car),
and for the variant `avoid` the boxes and slow cars, as the contract says.
"""

import importlib

from . import stubs

PARTS = ("venue", "car", "sensors", "ducts", "others")


class Providers(object):
    def __init__(self):
        self.venue_spec = stubs.venue_spec
        self.add_car = stubs.add_car
        self.CarRuntime = stubs.KinematicCar
        self.SensorSuite = stubs.IdealSensors
        self.DuctTracker = stubs.StaticDuctTracker
        self.add_boxes = stubs.add_boxes
        self.add_slow_cars = stubs.add_slow_cars
        self.BoxTracker = stubs.StubBoxTracker
        self.SlowCars = stubs.StubSlowCars
        self.pushable = False
        self.sources = {p: "stub" for p in PARTS}

    def describe(self):
        return dict(self.sources)


def _try(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:                    # only the package itself missing counts as "not there yet"
        if getattr(e, "name", None) in (name, "sketch"):
            return None
        raise


def resolve(mode="auto", **overrides):
    """The providers for `mode` "auto" (real packages where present) or "stub"; keyword overrides replace single
    attributes (tests: e.g. SensorSuite=..., DuctTracker=...)."""
    prov = Providers()
    if mode not in ("auto", "stub"):
        raise ValueError("mode is 'auto' or 'stub'")
    if mode == "auto":
        m = _try("sketch.venue")
        if m is not None and hasattr(m, "build_spec"):
            prov.venue_spec = m.build_spec
            prov.sources["venue"] = "S3"
        m = _try("sketch.car")
        if m is not None and hasattr(m, "add_car") and hasattr(m, "CarRuntime"):
            prov.add_car, prov.CarRuntime = m.add_car, m.CarRuntime
            prov.sources["car"] = "S1"
        m = _try("sketch.sensors")
        if m is not None and hasattr(m, "SensorSuite"):
            prov.SensorSuite = m.SensorSuite
            prov.sources["sensors"] = "S2"
        m = _try("sketch.ducts")
        if m is not None and hasattr(m, "DuctTracker"):
            prov.DuctTracker = m.DuctTracker
            prov.pushable = prov.sources["venue"] == "S3"         # the venue generator's build_spec calls the duct model's add_ducts
            prov.sources["ducts"] = "S4" if prov.pushable else "stub"
            if not prov.pushable:
                prov.DuctTracker = stubs.StaticDuctTracker
        m = _try("sketch.others")
        if m is not None and all(hasattr(m, a) for a in ("add_boxes", "add_slow_cars", "BoxTracker", "SlowCars")):
            prov.add_boxes, prov.add_slow_cars = m.add_boxes, m.add_slow_cars
            prov.BoxTracker, prov.SlowCars = m.BoxTracker, m.SlowCars
            prov.sources["others"] = "S5"
    for k, v in overrides.items():
        if not hasattr(prov, k):
            raise AttributeError("no provider attribute %r" % k)
        setattr(prov, k, v)
        part = {"venue_spec": "venue", "add_car": "car", "CarRuntime": "car", "SensorSuite": "sensors",
                "DuctTracker": "ducts", "add_boxes": "others", "add_slow_cars": "others", "BoxTracker": "others",
                "SlowCars": "others"}.get(k)
        if part:
            prov.sources[part] = "override"
    return prov


def build_spec(world, variant, prov, ego_pose):
    if variant not in ("track", "avoid"):
        raise ValueError("variant must be 'track' or 'avoid'")
    spec = prov.venue_spec(world, variant=variant, pushable=prov.pushable)
    prov.add_car(spec, prefix="car/", pose=tuple(ego_pose), tail_box=False)
    if variant == "avoid":
        prov.add_boxes(spec, world)
        prov.add_slow_cars(spec, world)
    return spec


def build_model(world, variant, prov, ego_pose=None, timestep=None):
    """Compile the model of a variant with the ego car at `ego_pose` (default the start pose)."""
    pose = world["truth"]["start_pose"]["pose"] if ego_pose is None else ego_pose
    spec = build_spec(world, variant, prov, pose)
    model = spec.compile()
    if timestep is not None:
        model.opt.timestep = float(timestep)
    return model
