"""The plant's rope rig for the bay mission: one rope per flying robot, from the robot's own hook to the bucket's ring
that faces its pad. A port of the reference build's `rope_chain` (solution/reference/build_scene.py): a chain of
capsule links, plain rigid bodies on spherical joints, the hook joint and the ring joint outside every articulation,
laid the way a slack rope lies (down from the hook, along the floor, up to the ring).

pxr only (PhysX's own schemas are written by name, so it runs without Kit too); the grader's control probe calls it
through `mission_stage.build` before Kit opens the stage.
"""

import math

from pxr import Gf, Sdf, UsdGeom, UsdPhysics


class MissionStageError(Exception):
    """The agent's fleet cannot be put into the real bay as built: that run's mission aborts with this reason."""


def _march(path, n, seg):
    """Joint points along a polyline `path` (its first point the hook, its last the ring), each exactly one link (a
    chord) from the one before; returns the n points from the hook on and how far the last one misses one link from the
    ring (> 0: the path is longer than the rope)."""
    segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    b = path[-1]
    out = [path[0]]
    si, f0 = 0, 0.0  # the current segment and the fraction along it
    for _ in range(n - 1):
        c = out[-1]
        found = None
        while si < len(segs) and found is None:
            a, e = segs[si]
            d = [e[k] - a[k] for k in range(3)]
            m = [a[k] - c[k] for k in range(3)]
            A = sum(v * v for v in d)
            B = 2 * sum(d[k] * m[k] for k in range(3))
            C = sum(v * v for v in m) - seg * seg
            disc = B * B - 4 * A * C
            if A > 0 and disc >= 0:
                t = (-B + math.sqrt(disc)) / (2 * A)
                if f0 <= t <= 1.0:
                    found = [a[k] + t * d[k] for k in range(3)]
                    f0 = t
            if found is None:
                si, f0 = si + 1, 0.0
        if found is None:  # (the path ran out: stretch the rest along the last direction)
            found = [c[k] + (b[k] - c[k]) / max(1e-9, math.dist(c, b)) * seg for k in range(3)]
        out.append(found)
    return out, math.dist(out[-1], b) - seg


def _lay(h, b, u, floor_z):
    """A one-parameter family of paths from the hook h to the ring b, longer as u grows: u in [0, 1] the straight
    chord sagging down to the floor at its middle; u in [1, 2] the hook's end coming down to the floor too; u in [2, 4]
    the reference's slack lay (hook -> straight down -> a point on the floor half-way, (u - 2) m to the side -> ring)."""
    mid = [(h[k] + b[k]) / 2 for k in range(3)]
    if u <= 1.0:
        return [h, [mid[0], mid[1], mid[2] + (floor_z - mid[2]) * u], b]
    floor_mid = [mid[0], mid[1], floor_z]
    if u <= 2.0:
        t = u - 1.0
        return [h, [h[0], h[1], h[2] + (floor_z - h[2]) * t], floor_mid, b]
    dx, dy = b[0] - h[0], b[1] - h[1]
    dh = max(1e-9, math.hypot(dx, dy))
    side = (-dy / dh, dx / dh)  # (perpendicular to the hook-ring line, on the floor)
    w = u - 2.0
    return [h, [h[0], h[1], floor_z], [mid[0] + side[0] * w, mid[1] + side[1] * w, floor_z], b]


def _physx(prim, api, attrs):
    """PhysX's own API schema and its attributes, by name (PhysxSchema itself needs Kit)."""
    prim.AddAppliedSchema(api)
    for name, (typ, value) in attrs.items():
        prim.CreateAttribute(name, typ).Set(value)


def _quat_x_to(d):
    """A rotation taking +x onto the unit direction d."""
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*d)).GetQuat())


def _sphere_box(c, r, box):
    """Does the sphere (c, r) reach into the axis-aligned box (lo, hi)?"""
    lo, hi = box
    d2 = sum(max(lo[k] - c[k], 0.0, c[k] - hi[k]) ** 2 for k in range(3))
    return d2 < r * r


def lay_rope(stage, root, hook_body, hook_local, hook_world, bucket_body, ring_local, ring_world, carrier_bodies, rope, carrier_boxes=()):
    """The plant's rope at `root` (an Xform that resets the transform stack: the links' poses are world poses), from
    `hook_local` on `hook_body` (at `hook_world`) to `ring_local` on `bucket_body` (at `ring_world`). A link that starts
    inside (or within 1 cm of) one of the carrying robot's colliders (`carrier_boxes`, world boxes) is filtered against
    every body of that robot, so that the robot's geometry cannot throw the rope at the start; the others touch the
    robot as a rope does. Returns the links' paths and which were filtered."""
    n, L, r = int(rope["links"]), float(rope["length"]), float(rope["radius"])
    seg = L / n
    h, b = [float(v) for v in hook_world], [float(v) for v in ring_world]
    chord = math.dist(h, b)
    if chord > L - 0.05:
        raise MissionStageError(f"the hook stands {chord:.2f} m from its ring in the real bay; the plant's rope is {L:.2f} m")
    floor_z = r + 0.001
    lo, hi = 0.0, 4.0
    if _march(_lay(h, b, hi, floor_z), n, seg)[1] < 0:  # (even the widest slack lay leaves rope over: the widest it is)
        lo = hi
    for _ in range(60):  # the path's length so that the last link ends exactly on the ring
        mid = (lo + hi) / 2
        if _march(_lay(h, b, mid, floor_z), n, seg)[1] > 0:
            hi = mid
        else:
            lo = mid
    pts, miss = _march(_lay(h, b, (lo + hi) / 2, floor_z), n, seg)
    pts.append(b)
    xf = UsdGeom.Xform.Define(stage, root)
    xf.SetResetXformStack(True)
    links, filtered = [], []
    for i in range(n):
        p0, p1 = pts[i], pts[i + 1]
        d = [p1[k] - p0[k] for k in range(3)]
        ln = max(1e-9, math.sqrt(sum(v * v for v in d)))
        c, d = [(p0[k] + p1[k]) / 2 for k in range(3)], [v / ln for v in d]
        lp = f"{root}/Link_{i}"
        lx = UsdGeom.Xform.Define(stage, lp)
        lx.AddTranslateOp().Set(Gf.Vec3d(*c))
        lx.AddOrientOp().Set(_quat_x_to(d))
        UsdPhysics.RigidBodyAPI.Apply(lx.GetPrim())
        UsdPhysics.MassAPI.Apply(lx.GetPrim()).CreateMassAttr(float(rope["kg_per_m"]) * seg)
        _physx(lx.GetPrim(), "PhysxRigidBodyAPI", {
            "physxRigidBody:angularDamping": (Sdf.ValueTypeNames.Float, float(rope.get("angular_damping", 0.2))),
            "physxRigidBody:linearDamping": (Sdf.ValueTypeNames.Float, float(rope.get("linear_damping", 0.02))),
            "physxRigidBody:solverPositionIterationCount": (Sdf.ValueTypeNames.Int, int(rope.get("solver_position_iterations", 32))),
            "physxRigidBody:solverVelocityIterationCount": (Sdf.ValueTypeNames.Int, int(rope.get("solver_velocity_iterations", 4)))})
        cap = UsdGeom.Capsule.Define(stage, f"{lp}/geom")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(r)
        cap.CreateHeightAttr(seg - 2 * r)
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        _physx(cap.GetPrim(), "PhysxCollisionAPI", {
            "physxCollision:contactOffset": (Sdf.ValueTypeNames.Float, float(rope.get("contact_offset", 0.006))),
            "physxCollision:restOffset": (Sdf.ValueTypeNames.Float, float(rope.get("rest_offset", 0.0)))})
        if any(_sphere_box(c, seg / 2 + r + 0.01, b) for b in carrier_boxes):
            fp = UsdPhysics.FilteredPairsAPI.Apply(lx.GetPrim()).CreateFilteredPairsRel()
            for body in carrier_bodies:
                fp.AddTarget(body)
            filtered.append(i)
        links.append(lp)
    prev_body, prev_lp = hook_body, tuple(float(v) for v in hook_local)
    for i, lp in enumerate(links):
        j = UsdPhysics.SphericalJoint.Define(stage, f"{root}/Joint_{i}")
        j.CreateBody0Rel().SetTargets([prev_body])
        j.CreateBody1Rel().SetTargets([lp])
        j.CreateLocalPos0Attr(Gf.Vec3f(*prev_lp))
        j.CreateLocalPos1Attr(Gf.Vec3f(-seg / 2, 0, 0))
        if i == 0:
            j.CreateExcludeFromArticulationAttr(True)  # the chain is not part of the robot's articulation
        prev_body, prev_lp = lp, (seg / 2, 0, 0)
    j = UsdPhysics.SphericalJoint.Define(stage, f"{root}/Joint_{n}")  # the ring: a joint outside every articulation
    j.CreateBody0Rel().SetTargets([links[-1]])
    j.CreateBody1Rel().SetTargets([bucket_body])
    j.CreateLocalPos0Attr(Gf.Vec3f(seg / 2, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in ring_local]))
    j.CreateExcludeFromArticulationAttr(True)
    return dict(links=links, chord=round(chord, 4), miss=round(miss, 4), lay=round((lo + hi) / 2, 3), filtered=filtered)


def filter_bucket(stage, bucket_body, links):
    """The links never collide with the bucket (the last ones run up its side to the ring)."""
    rel = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(bucket_body)).CreateFilteredPairsRel()
    for lp in links:
        rel.AddTarget(lp)
