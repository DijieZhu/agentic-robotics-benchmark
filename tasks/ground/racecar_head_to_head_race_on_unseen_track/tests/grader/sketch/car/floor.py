"""A faster floor lookup with the semantics of `sketch.examples.check.FloorMap`.

`CarRuntime` calls `grip_at(x, y)` under each wheel at every physics step (four calls per step). The contract's `FloorMap`
runs a point-in-polygon test for every patch on every call; `FloorGrip` first rejects patches whose bounding box does
not contain the point and only then calls the contract's own `point_in_polygon`, so the answers are those of `FloorMap` by
construction (test_s1_floor compares them on random points and on patch edges).

    car = CarRuntime(model, data, grip_at=FloorGrip(world).grip_at)
"""

from sketch.examples.check import point_in_polygon


class FloorGrip(object):

    def __init__(self, world):
        floor = world["truth"]["floor"]
        self.base_grip = float(floor["base_grip"])
        self.patches = []
        for p in floor["patches"]:
            xs = [float(q[0]) for q in p["polygon"]]
            ys = [float(q[1]) for q in p["polygon"]]
            self.patches.append((min(xs), max(xs), min(ys), max(ys), p["polygon"], p["id"], float(p["grip"])))
        self.patches.reverse()                     # the last listed patch that contains the point wins

    def zone_at(self, x, y):
        for x0, x1, y0, y1, poly, pid, _ in self.patches:
            if x0 <= x <= x1 and y0 <= y <= y1 and point_in_polygon(x, y, poly):
                return pid
        return None

    def grip_at(self, x, y):
        for x0, x1, y0, y1, poly, _, grip in self.patches:
            if x0 <= x <= x1 and y0 <= y <= y1 and point_in_polygon(x, y, poly):
                return grip
        return self.base_grip
