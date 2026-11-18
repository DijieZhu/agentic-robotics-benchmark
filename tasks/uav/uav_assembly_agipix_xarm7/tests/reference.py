"""Privileged, physical xArm reference: observations may be privileged, motion is not.

Every move uses the public cell tick, bounded actuators and real pad contacts.
This module never writes live part poses, equality activation or coupling state.
"""
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from motion import Motion, PlanningError, DOWN, body_subtree, axis_aligned_rotation


@dataclass
class Grasp:
    p: np.ndarray
    R: np.ndarray
    width: float
    label: str
    target_yaw: float = 0.
    squeeze: float = .006
    settle_s: float = .5
    # Camera only: how far the part may slide toward the finger tips before a
    # pad runs off its contact band (band top minus the pad's lower edge).
    slide_m: float = math.inf


def _log(*args):
    print("[reference]", *args, file=sys.stderr, flush=True)


def _tip(width):
    a = Motion.angle_for_width(width)
    return .059098 + math.sin(a)*.035465 + math.cos(a)*.042039 + .0595


def surface_grasps(cell, name):
    """Antipodal candidates on the imported surface in the settled pick frame.

    A protruding cable or lens must not move a grasp to the bounding-box centre.
    Endpoint normals and finite jaw width are measured from actual mesh rays;
    the planner subsequently checks the complete stock gripper and held part.
    """
    import trimesh
    kind = cell.graph["instances"][name]["kind"]
    part = cell.kit["parts"][kind]
    mesh = trimesh.load_mesh(cell.assets / "given/kit/meshes" / part["mesh"])
    top = cell.pose(name)[:3, :3].T @ DOWN
    candidates = []
    # Collision decomposition can slightly enlarge a fin or connector. Measure
    # its actual envelope too, so a command based only on one source ray does
    # not drive the pad several millimetres into a neighbouring feature.
    proxy_pieces = [
        trimesh.load_mesh(cell.assets / "given/kit/collision" / fn)
        for fn in part["collision"]]
    # The motor's grasp surface is its passive bell child. Its aggregate
    # source mesh already includes the child at the nominal hinge offset;
    # the collision footprint must use that same physical assembly.
    if part.get("rotor"):
        rotor=part["rotor"]
        for fn in rotor["collision"]:
            piece=trimesh.load_mesh(cell.assets / "given/kit/collision" / fn)
            piece.apply_translation(rotor["hinge_pos_m"])
            proxy_pieces.append(piece)
    proxy = trimesh.util.concatenate(proxy_pieces)

    def footprint(contact, R):
        # Stock pad_2 is 30 x 19 mm. The candidate contact denotes its distal
        # tip, so its material extends back towards the wrist along -tool Z.
        # Keep the complete pad envelope, with finer central rays for a
        # small curved lens or bell whose tangent patch is only a few mm wide.
        offsets=np.array([[x,0.,z] for z in (0.,-.002,-.004,-.006,-.008,-.010,-.012,-.014,-.016,-.018)
                          for x in (-.014,-.010,-.005,-.004,-.002,0.,.002,.004,.005,.010,.014)])
        centers=contact+offsets@R.T
        origins=centers-R[:,1]*.15
        directions=np.tile(R[:,1],(len(origins),1))
        source_hits,source_rays,source_faces=mesh.ray.intersects_location(origins,directions,multiple_hits=True)
        proxy_hits,proxy_rays,_=proxy.ray.intersects_location(origins,directions,multiple_hits=True)
        if not len(proxy_hits):return None
        sides=[];normals=[]
        for k in range(len(origins)):
            use=np.flatnonzero(source_rays==k)
            if len(use)<2:continue
            use=use[np.argsort(source_hits[use]@R[:,1])]
            a,b=use[0],use[-1]
            sides.append([source_hits[a]@R[:,1],source_hits[b]@R[:,1]])
            normals.append([-mesh.face_normals[source_faces[a]]@R[:,1],
                            mesh.face_normals[source_faces[b]]@R[:,1]])
        if len(sides)<8:return None
        sides=np.asarray(sides);normals=np.asarray(normals)
        envelope=np.array([min(proxy_hits@R[:,1]),max(proxy_hits@R[:,1])])
        width=float(envelope[1]-envelope[0])
        if not .006<=width<=.078:return None
        # Require a finite patch near each supporting outside plane, rather
        # than one opposing normal at a small connector or corner.
        near=(np.abs(sides-envelope)<.0012)&(normals>math.cos(math.radians(30)))
        support=near.sum(0)
        # A cylindrical motor bell meets a flat pad along a narrow axial
        # strip, so four supported rays can describe a valid finite strip.
        if min(support)<(4 if part.get("rotor") else 6):return None
        contact=contact+R[:,1]*(envelope.mean()-contact@R[:,1])
        return contact,width,int(min(support))
    for yaw in (0., math.pi/2, math.pi, -math.pi/2):
        R = top @ Rotation.from_euler("z", yaw).as_matrix()
        points_in_hand = mesh.vertices @ R
        low, high = points_in_hand.min(0), points_in_hand.max(0)
        # Sample the upper half of the physical part, with several depths for
        # stepped housings. Source geometry determines available material.
        zs = low[2] + np.array([.2, .35, .5, .65, .70, .73, .8]) * (high[2]-low[2])
        xs = np.linspace(low[0]+.002, high[0]-.002, 15)
        grid = np.array([[x, low[1]-.01, z] for z in zs for x in xs])
        origins = grid @ R.T
        hits, rays, faces = mesh.ray.intersects_location(
            origins, np.tile(R[:, 1], (len(origins), 1)), multiple_hits=True)
        for i in range(len(grid)):
            use = np.flatnonzero(rays == i)
            if len(use) < 2:
                continue
            ordered = use[np.argsort(hits[use] @ R[:, 1])]
            # Outermost hits span the whole part on this ray: inner recesses
            # are not falsely treated as the two outside gripping surfaces.
            a, b = ordered[0], ordered[-1]
            width = float((hits[b]-hits[a]) @ R[:, 1])
            if not .006 <= width <= .075:
                continue
            normal_a, normal_b = mesh.face_normals[faces[a]], mesh.face_normals[faces[b]]
            limit = math.cos(math.radians(25))
            if normal_a @ R[:, 1] > -limit or normal_b @ R[:, 1] < limit:
                continue
            measured=footprint((hits[a]+hits[b])*.5,R)
            if measured is None:continue
            contact,width,support=measured
            p = contact - R[:, 2]*(_tip(width)-.158)
            # Prefer broad surfaces near the source's main solid instead of
            # thin cable segments. This is a candidate rank, not a pose write.
            pad_center=contact-R[:,2]*.0095
            delta=(pad_center-np.asarray(part.get("inertial",{}).get("center_m",mesh.center_mass)))@R
            # Keep the COM near the pad's long-axis centre. A wide patch at
            # the far end of a board can still roll out under its own weight.
            priority = 2.*abs(delta[0]) + .05*abs(delta[1]) + .5*abs(delta[2]) - .00003*support
            label = f"surface-{yaw:.2f}-{i}"
            candidates.append((priority, Grasp(p, R, width, label, squeeze=.004)))
    seen = set()
    for _, candidate in sorted(candidates, key=lambda x: x[0]):
        key = tuple(np.round(candidate.p/.003).astype(int)) + tuple(np.round(candidate.R[:, 1], 2))
        if key not in seen:
            seen.add(key)
            yield candidate


def _lidar_housing_width(proxy, R, center):
    """Measured jaw span across the LiDAR's upper housing, or None.

    Rays along the jaw axis cover pad_2 and pad_1 behind it. Every hit must lie
    on the round housing (<= 26.2 mm from the axis): the drafted base shoulder,
    its fillet and the connector/cable must not reach a pad.
    """
    tip=center+R[:,2]*.0095
    offs=np.array([[x,0.,z] for z in np.linspace(0.,-.037,15) for x in np.linspace(-.0145,.0145,11)])
    points=tip+offs@R.T
    hits,rays,_=proxy.ray.intersects_location(points-R[:,1]*.15,np.tile(R[:,1],(len(points),1)),multiple_hits=True)
    if not len(rays):
        return None
    span=(np.asarray(hits)-points[rays])@R[:,1]
    if np.max(np.abs(span))>.0262:
        return None
    pad2=offs[rays][:,2]>=-.019
    if not pad2.any() or min(span[pad2].max(),-span[pad2].min())<.0252:
        return None
    return float(span[pad2].max()-span[pad2].min())


def lidar_housing_grasps(cell, name):
    """LiDAR: jaws across the rotationally symmetric upper housing only.

    Pad contacts are single points per convex pair, with no torsional friction,
    so a held part turns about its jaw line until the COM hangs below it. On the
    base body the 2.5 deg side draft puts those contacts on the bottom band and
    the connector keeps them >= 7 mm off the COM plane: that grasp hangs tilted
    over the cap (measured 27 deg after lift-off). Across the round housing the
    pads touch its y=0 tangent lines above the COM, so the hanging equilibrium is
    the upright mounting attitude. Bench to cap is a 90 deg turn about this same
    jaw axis. Top-down approaches come first: with the pad's long axis along the
    housing generator the sensor creeps least in the hand (~13 deg after
    lift-off, versus ~60 deg for a 30 deg lean towards the dome, the fallback).
    """
    import trimesh
    part=cell.kit["parts"]["lidar"]
    proxy=trimesh.util.concatenate([trimesh.load_mesh(cell.assets/"given/kit/collision"/fn)
                                    for fn in part["collision"]])
    if _lidar_upright(cell, name):
        yield from _lidar_upright_grasps(proxy)
        return
    down=cell.pose(name)[:3,:3].T@np.array([0.,0.,-1.])
    for theta in LIDAR_APPROACH_TILTS_DEG:
        for flip in (1.,-1.):
            jaw=np.array([flip,0.,0.])
            d=down-(down@jaw)*jaw;d/=np.linalg.norm(d)
            e=np.array([0.,0.,-1.]);e-=(e@jaw)*jaw+(e@d)*d;e/=np.linalg.norm(e)
            th=math.radians(theta)
            approach=math.cos(th)*d+math.sin(th)*e
            R=np.column_stack([np.cross(jaw,approach),jaw,approach])
            for yc in (0.,.002,-.002,.004):
                for zc in (.046,.044,.048):
                    center=np.array([0.,yc,zc])
                    width=_lidar_housing_width(proxy,R,center)
                    if width is None:
                        continue
                    tool=center+approach*.0095-approach*(_tip(width)-.158)
                    yield Grasp(tool,R,width,f"housing-th{theta:.0f}-y{yc*1000:.0f}-z{zc*1000:.0f}-{'+' if flip>0 else '-'}x",
                                squeeze=.004,settle_s=.5)


LIDAR_APPROACH_TILTS_DEG=(0.,15.,30.)

# Cable-free kit (mechanical-only scope): the LiDAR stands
# upright on its flat base on the bench, yawed to its mounting yaw.
LIDAR_UPRIGHT_MAX_TILT_DEG=20.
# Pad-tip heights (part z) on the knurled ring (r 25.3-25.7 mm, z 29.4-39.5 mm),
# preferred first: the held sensor creeps down in the pads at ~0.13 mm/s whatever
# the squeeze (soft-contact friction), so the ring
# length below the ring top (39.5 mm) is the carry budget (7.5 mm at 32 mm; the
# 34 mm tip's 5.5 mm ran out after a 37 s carry). Lower tips put the
# finger tips (1.5 mm past the pads) onto the base shoulder (top z 29.4 mm).
LIDAR_UPRIGHT_TIP_Z_M=(.032,.0315,.033,.034)
LIDAR_UPRIGHT_JAW_DEG=(0.,90.,45.,135.)


def _lidar_upright(cell, name):
    """True when the LiDAR stands upright on the bench (cable-free kit rest)."""
    return cell.pose(name)[2,2]>math.cos(math.radians(LIDAR_UPRIGHT_MAX_TILT_DEG))


def _lidar_upright_grasps(proxy):
    """Upright LiDAR (cable-free kit): top-down pinch across the knurled ring.

    The jaw line crosses the ring's tangent lines above the COM, so the sensor
    hangs in its upright mounting attitude from lift-off (zero gravity torque
    about the jaw line), and the bench yaw equals the mounting yaw: the carry
    needs no reorientation of the part. The same _lidar_housing_width()
    footprint check keeps both pads on the round housing only. The jaw
    orientation (and with it the wrist branch) is chosen by joint travel in
    place(): a 4.4 rad joint-7 unwinding made a 32 s carry.
    """
    approach=np.array([0.,0.,-1.])
    # Tip height outermost: place() ranks the first four feasible candidates (the
    # jaw orientations at the preferred tip) by joint travel (_least_travel_first).
    for tip_z in LIDAR_UPRIGHT_TIP_Z_M:
        for jaw_deg in LIDAR_UPRIGHT_JAW_DEG:
            for flip in (1.,-1.):
                a=math.radians(jaw_deg)
                jaw=flip*np.array([math.cos(a),math.sin(a),0.])
                R=np.column_stack([np.cross(jaw,approach),jaw,approach])
                center=np.array([0.,0.,tip_z+.0095])
                width=_lidar_housing_width(proxy,R,center)
                if width is None:
                    continue
                tool=center+approach*.0095-approach*(_tip(width)-.158)
                yield Grasp(tool,R,width,f"ring-top-j{jaw_deg:.0f}-z{tip_z*1000:.1f}-{'+' if flip>0 else '-'}",
                            squeeze=.004,settle_s=.5)


def _lidar_align(motion, pos, R, body, *, tolerance, max_passes):
    """Measured-pose alignment of the held LiDAR; its jaw-line tilt is gravity's.

    The two housing contacts leave the sensor free to turn about the jaw line,
    where it hangs ~0.5-1.7 deg from upright. Turning it with the hand only
    winds the hand around the hanging sensor, which creeps back within a second
    (measured: +0.5 deg in-hand per pass, never reaching the generic 0.57 deg;
    a corrected 0.18 deg crept back to 0.9 deg and shifted the origin 0.5 mm).
    Translate it to the target with that hanging tilt instead, and turn it
    back only beyond 2 deg. The mate accepts 3 deg.
    """
    motion.hold_body(body)
    b=motion.held[0]
    limit=math.radians(2.)
    for _ in range(max_passes):
        p_now,r_now=motion.body_pose(b)
        ep=float(np.linalg.norm(np.asarray(pos)-p_now))
        er=float(Rotation.from_matrix(np.asarray(R)@r_now.T).magnitude())
        if er>limit:
            desired=np.asarray(R)
        elif ep<=tolerance:
            return {"position_error_m":ep,"angle_error_rad":er}
        else:
            desired=r_now
        p,r=motion.body_goal(pos,desired,b)
        motion.line(p,r,speed=.01,clearance=False,allow_pad_body=b)
    p_now,r_now=motion.body_pose(b)
    ep=float(np.linalg.norm(np.asarray(pos)-p_now))
    er=float(Rotation.from_matrix(np.asarray(R)@r_now.T).magnitude())
    if ep>2*tolerance or er>math.radians(2.5):
        raise PlanningError(f"lidar alignment did not converge: {ep*1000:.3f} mm, {math.degrees(er):.3f} deg")
    _log("lidar alignment residual accepted",round(ep*1000,3),"mm",round(math.degrees(er),3),"deg")
    return {"position_error_m":ep,"angle_error_rad":er}
# Pad centres along the cheek's upper arm (part x), nearest the spine first
# that still leaves the inner finger plate (x +-16 mm) clear of it.
CHEEK_RIM_CENTERS_M = (-.012, -.0125, -.0115, -.013, -.011)


def cheek_rim_grasps(cell, name):
    """Sensor cheek (flat 4 mm C plate on its staging stand): jaws across its upper arm.

    The jaws close across the upper arm (part y) from above, with the inner
    finger in the C opening, which keeps the wrist above the installed guard.
    The arm's straight rim (x -18..+5.6 mm) is shorter than a stock pad (30 mm)
    and its finger plate (32 mm): at the free end a hook prong stands 1.9 mm
    proud of the inner edge, at the other end the spine closes the opening.
    The inner finger plate must clear the spine, so the pad spans the prong;
    the jaw gap is the collision proxy's span over the pad footprint (outer rim
    to the prong tip), i.e. the gap the closing pads actually stop at. (The
    earlier fixed 7.75 mm straight-rim gap at x -10 put the inner pad 1.9 mm
    inside the prong and the finger plate's side 0.2 mm inside the spine.)
    place()'s closed-jaw finger check (_grip_clear) takes the first clear one.
    Known limit (physics, not planning): no pinch across this arm is in-plane
    stable for the stock pads (its width falls as the plate turns and slides
    along them), so the cheek keeps creeping in the grip; place() keeps the
    hold short and the cheek upright (see "Cheek carry" below). (The earlier
    4.6 deg tooth face is corrected in the kit since final_assets_v7.)
    """
    import trimesh
    part = cell.kit["parts"][cell.graph["instances"][name]["kind"]]
    proxy = trimesh.util.concatenate([trimesh.load_mesh(cell.assets/"given/kit/collision"/fn)
                                      for fn in part["collision"]])
    for cx in CHEEK_RIM_CENTERS_M:
        rays = np.array([[cx+x, 0., z] for x in np.linspace(-.015, .015, 31) for z in (.0005, .002, .0035)])
        inner, _, _ = proxy.ray.intersects_location(rays+[0., .008, 0.], np.tile([0., 1., 0.], (len(rays), 1)))
        outer, _, _ = proxy.ray.intersects_location(rays+[0., .080, 0.], np.tile([0., -1., 0.], (len(rays), 1)))
        if not len(inner) or not len(outer):
            continue
        lo, hi = float(inner[:, 1].min()), float(outer[:, 1].max())
        if not .006 <= hi-lo <= .015:
            continue
        for yaw in (0., math.pi):
            R = Rotation.from_euler("z", yaw).as_matrix() @ DOWN
            yield Grasp(np.array([cx, (lo+hi)/2, .012359397152226951]), R, hi-lo, f"rim-x{1000*cx:.1f}-{yaw:.2f}")


# Axial-pinch pad-tip heights (camera only). 4 s still-hold creep after the
# lift: z24 0.094 mm/s 0.067 deg/s,
# z22 0.093/0.066 (2.7 mm slide during the lift), z16 0.31/0.10,
# z18 0.78/0.77; z26 falls out of the pads.
CAMERA_TIP_HEIGHTS_M = (.016, .018, .020, .022, .024)
CAMERA_TIP_PREFERRED_M = .024


def _camera_band_top(proxy, jaw, z_tip, tol, height=.0185):
    """Top of the shorter pad contact band of a jaw-axis camera pinch (camera only).

    For each jaw side, rays along the jaw in 0.25 mm rows from the pad's lower
    edge (z_tip) up: the face toward that pad touches it where it lies within
    tol of its most protruding point. Returns the lower of the two band tops
    (part frame z): the camera can slide down in the jaws until that edge
    reaches the pads' lower edge.
    """
    zs = np.arange(z_tip, z_tip+height, .00025)
    ys = np.linspace(-.0145, .0145, 7)
    tops = []
    for d in (jaw, -jaw):
        origins = np.array([[0., y, z] for z in zs for y in ys])-d*.06
        loc, ray, _ = proxy.ray.intersects_location(origins, np.tile(d, (len(origins), 1)), multiple_hits=False)
        if not len(loc):
            return z_tip
        depth = np.einsum("ij,j->i", loc-origins[ray], d)
        tops.append(float(loc[depth <= depth.min()+tol][:, 2].max()))
    return min(tops)


# Cheek seat: the last re-measurement is made this far above the seat, above the
# guard cradle's hook pocket (~1.8 mm deep), so the cheek is still free of contact.
CHEEK_NEAR_SEAT_M = .005
CHEEK_SEAT_HOLD_S = 1.5


def _cheek_align(motion, pos, R, body, *, tolerance, angle, max_passes, accept_mm, accept_deg, where):
    """align_body for the free (not yet seated) cheek; a small residual is accepted."""
    try:
        return motion.align_body(pos, R, body=body, tolerance=tolerance, angle=angle, max_passes=max_passes)
    except PlanningError as error:
        if "did not converge" not in str(error):
            raise
        b = motion.held[0]
        p_now, r_now = motion.body_pose(b)
        ep = float(np.linalg.norm(np.asarray(pos)-p_now))
        er = float(Rotation.from_matrix(np.asarray(R) @ r_now.T).magnitude())
        if ep > accept_mm/1000 or er > math.radians(accept_deg):
            raise
        _log("cheek", where, "alignment residual accepted", round(1000*ep, 3), "mm", round(math.degrees(er), 3), "deg")
        return {"position_error_m": ep, "angle_error_rad": er}


def _cheek_rebranch(motion, body, pre_p, goal_p, R, *, clearance, transport_clearance, speed):
    """_preinsert for the cheek near its seat: a new IK branch at pre_p (same hand pose)
    whose straight path to goal_p is clear, reached if possible by ONE direct joint move
    that keeps the approach level (CHEEK_LEVEL_TILT), else on the planned joint path that
    holds the approach off level least (of the first CHEEK_REBRANCH_PLANS branches that
    plan) at CHEEK_REBRANCH_SPEED, else by _preinsert as before. (_preinsert's RRT at .25 swung the
    cheek 18 deg about the jaw axis in 37 s at the hover, runs v10c9.)"""
    width = motion.width()
    q0 = motion.q
    solutions = motion.ik(pre_p, R, width=width, attempts=96, max_solutions=16, allow_pad_body=body,
                          clearance=clearance)
    for candidate in sorted(solutions, key=lambda q: float(np.max(np.abs(q-q0)))):
        prev, ok = candidate, True
        for u in np.linspace(0, 1, 34)[1:]:
            q = motion._ik_one((1-u)*np.asarray(pre_p)+u*np.asarray(goal_p), R, prev, width=width)
            if q is None or not motion.segment_free(prev, q, width=width, allow_pad_body=body, clearance=clearance):
                ok = False; break
            prev = q
        if not ok:
            continue
        duration, _exposure, _up, level = _cheek_joint_attitude(motion, q0, candidate, .6)
        if level <= CHEEK_LEVEL_TILT and motion.segment_free(q0, candidate, width=width, allow_pad_body=body,
                                                              clearance=transport_clearance):
            motion.execute([q0, candidate], speed=.6)
            _log("cheek re-branch: direct level joint move", round(duration, 1), "s, max |approach_z|", round(level, 2))
            return
    # Planned joint paths to the branches with a clear straight path: the one that holds
    # the approach off level least (duration x mean |approach_z|), executed briskly.
    best, found = None, 0
    for candidate in sorted(solutions, key=lambda q: float(np.max(np.abs(q-q0)))):
        prev, ok = candidate, True
        for u in np.linspace(0, 1, 34)[1:]:
            q = motion._ik_one((1-u)*np.asarray(pre_p)+u*np.asarray(goal_p), R, prev, width=width)
            if q is None or not motion.segment_free(prev, q, width=width, allow_pad_body=body, clearance=clearance):
                ok = False; break
            prev = q
        if not ok:
            continue
        try:
            path = motion.plan(candidate, width=width, allow_pad_body=body, clearance=transport_clearance)
        except PlanningError:
            continue
        exposure = sum(_cheek_joint_attitude(motion, path[i], path[i+1], CHEEK_REBRANCH_SPEED)[1]
                       for i in range(len(path)-1))
        if best is None or exposure < best[0]:
            best = (exposure, path)
        found += 1
        if found >= CHEEK_REBRANCH_PLANS:
            break
    if best is None:
        _log("cheek re-branch: no planned path either - _preinsert")
        _preinsert(motion, pre_p, goal_p, R, body, clearance=clearance, transport_clearance=transport_clearance,
                   attempts=96, speed=speed)
        return
    motion.execute(best[1], speed=CHEEK_REBRANCH_SPEED)
    _log("cheek re-branch: planned joint path,", len(best[1]), "waypoints, approach off level", round(best[0], 2), "s")


def _cheek_seat(cell, motion, mate, name, body, target, ins, pre_distance, wall_deadline=math.inf):
    """Sensor cheek only: correct the in-grip pivot just above the seat, descend, hold still.

    The rim grasp holds the cheek by the hook tooth and two short rim contacts, so it keeps
    turning in the grip about the jaw axis (soft pad friction, no torsional friction),
    ~0.6 deg/s through the carry (4 -> 20 deg in one run). A goal measured at the
    65 mm hover is ~2.6 deg and ~1 mm off after the 4.3 s descent. The coupling welds the
    pose the cheek is held in for 0.5 s within 1.5 mm, and this creep counts as still for
    it (< 2 deg/s): LOCKED up to 3 deg, irreversibly LOCKED_OFF beyond. Aligning against the
    cradle only winds the hand around the cheek (one run: 20 -> 36 deg in the grip, no
    convergence). So: align at the hover, descend to CHEEK_NEAR_SEAT_M above the seat
    (straight descent checked first, as in place()), align again there while still free,
    check the last 5 mm from the corrected pose (if the correction or the descent would
    graze the guard with the arm, re-branch there), make them with the re-measured grasp
    (~0.4 deg more creep) and hold still until the coupling locks; no alignment against
    the cradle.
    """
    key = mate["id"]
    held_in_hand = _camera_in_hand(cell, motion, name)

    def grip_kept(where):
        mm, deg = _in_hand_change(held_in_hand, _camera_in_hand(cell, motion, name))
        if deg > CHEEK_CARRY_TURN_DEG or not _gripped(cell, name):
            raise _CheekSlip(f"cheek turned {deg:.2f} deg in the grip in the {where} re-branch")
    _cheek_align(motion, target[:3,3]-ins*pre_distance, target[:3,:3], body, tolerance=.0007, angle=.01,
                 max_passes=10, accept_mm=1.5, accept_deg=2., where="hover")
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} during hover alignment")
    near = target[:3,3]-ins*CHEEK_NEAR_SEAT_M
    np_, nr = motion.body_goal(near, target[:3,:3], body)
    try:
        motion.line(np_, nr, allow_pad_body=body, clearance=False, execute=False)
    except PlanningError as error:
        _log("descent blocked from the transport branch:", name, str(error)[:160])
        hp, hr = motion.body_goal(target[:3,3]-ins*pre_distance, target[:3,:3], body)
        _cheek_rebranch(motion, body, hp, np_, nr, clearance=False, transport_clearance=True, speed=.25)
        grip_kept("hover")
        np_, nr = motion.body_goal(near, target[:3,:3], body)
    # Contact-free descent: twice the seat speed, so less pivot builds up before the near-seat check.
    motion.line(np_, nr, allow_pad_body=body, clearance=False, speed=.03)
    for attempt in range(3):
        try:
            _cheek_align(motion, near, target[:3,:3], body, tolerance=.0004, angle=.008, max_passes=6,
                         accept_mm=.8, accept_deg=1.2, where="near seat")
            pp, rr = motion.body_goal(target[:3,3], target[:3,:3], body)
            motion.line(pp, rr, allow_pad_body=body, clearance=False, execute=False)
            break
        except PlanningError as error:
            # The pivot correction turns the hand; on this IK branch the
            # correction or the last 5 mm can graze the guard (link5/link7).
            # Move instead, on a branch whose straight 5 mm descent
            # is clear (with arm margin if one exists), directly to the
            # corrected near-seat hand pose; then correct and check again.
            if attempt == 2 or "Cartesian path" not in str(error):
                raise
            _log("cheek near-seat move blocked:", name, str(error)[:120], "- new branch at the near-seat pose")
            np_, nr = motion.body_goal(near, target[:3,:3], body)
            pp, rr = motion.body_goal(target[:3,3], target[:3,:3], body)
            try:
                _cheek_rebranch(motion, body, np_, pp, rr, clearance=True, transport_clearance=False, speed=.15)
            except PlanningError:
                _cheek_rebranch(motion, body, np_, pp, rr, clearance=False, transport_clearance=False, speed=.15)
            grip_kept("near-seat")
    motion.line(pp, rr, allow_pad_body=body, clearance=False, speed=.015)
    q = motion.q
    states = []
    for _ in range(int(round(CHEEK_SEAT_HOLD_S/.01))):
        if time.monotonic() > wall_deadline:
            raise TimeoutError("reference wall cap during cheek seat hold")
        if cell.coupling.state[key] in ("LOCKED", "LOCKED_OFF"):
            break
        motion._drive(q); cell.tick()
        states.append(cell.coupling.state[key])
    P = cell.pose(name)
    err_deg = math.degrees(float(Rotation.from_matrix(target[:3,:3].T @ P[:3,:3]).magnitude()))
    _log("cheek seat", name, cell.coupling.state[key], "after hold", round(.01*len(states), 2), "s; error",
         round(1000*float(np.linalg.norm(P[:3,3]-target[:3,3])), 3), "mm", round(err_deg, 3), "deg")
    for _ in range(80):
        if time.monotonic() > wall_deadline:
            raise TimeoutError("reference wall cap during seat hold")
        motion._drive(q); cell.tick()


# Cheek carry (cheek only). The rim pinch is not in-plane stable: the pinch width across
# the upper arm (hook tooth tip and inner rim / r 13 arm-tip arc) falls as the plate turns
# and slides along the pads, with no local minimum anywhere on the arm for the stock pads
# (dev probe width_minima), so the squeeze walks it down that valley through the soft pad
# friction; and gravity turns it about the jaw axis whenever the approach is not level
# (moment ~ the approach's vertical component: COM 15.6 mm along the arm from the pad
# centre). Both creeps grow with time and feed each other. Held flat for ~20 s and turned
# slowly by the joint-space transport it pivoted 22 deg, swung loose and fell; a 45 s carry wound it 15 deg in-plane, ever faster, until the
# near-seat alignment could not converge. Upright in its mounting
# attitude (jaw axis vertical, approach level: weight on the lower pad, no moment about the
# jaw axis) it creeps ~0.05-0.1 deg/s, but even so it slides out of the jaws within a
# minute (dev probes upright_hold, runs v10c4/c6). So the whole hold is kept short: lift
# 4 cm briskly, then usually ONE direct joint move to a hover branch with a clear descent
# (approach off level < 3 s), else an upright turn followed by a level joint move or by
# straight legs at the mounting attitude (_cheek_turn_plan); hold still and measure the
# creep; lighter squeeze; seat (_cheek_seat, its re-branches kept level and brisk). A
# cheek that creeps, slips or cannot be aligned at its seat goes back onto its stand the
# way it came (_cheek_back_to_stand) for the next pinch; it is never dropped.
CHEEK_LIFT_SPEED = .05          # m/s off the stand (flat: the worst attitude, keep it short)
CHEEK_LIFT_M = .04              # lift off the stand before the turning joint move (place(): .10)
CHEEK_TURN_RISE_M = (.03, .08, .13)  # upright-turn height above the lift pose
CHEEK_TURN_YAWS_DEG = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
CHEEK_TURN_SPEED = 1.           # Motion.execute speed (peak joint rad/s) of the direct turn (2.4 rad: 4.5 s)
CHEEK_TURN_OPTIONS = 40         # turn configurations tried (least joint travel first)
CHEEK_PIVOT_EXPOSURE_S = 3.     # a turning joint move may hold the approach this long off level, as
                                # time x mean |vertical component of the approach| (flat: ~1 deg/s pivot)
CHEEK_FINGERTIPS_UP = .25       # largest upward component of the approach on any joint move
CHEEK_LEVEL_TILT = .26          # largest |vertical component of the approach| on an upright joint move
CHEEK_PIVOT_CHECK_DEG = (3., 5.)  # in-grip turns a preferred hover branch's descent also survives
CHEEK_REBRANCH_PLANS = 3        # planned re-branch paths compared (least approach-off-level kept)
CHEEK_REBRANCH_SPEED = .6       # their execution speed (Motion.execute; place()'s re-branch: .25)
CHEEK_CARRY_M_S = .1            # straight carry legs at the mounting attitude (smooth: peak x1.875)
CHEEK_CARRY_RAD_S = .5          # their yaw turn about the vertical
CHEEK_CARRY_QDOT = 1.2          # peak joint speed in a leg (runs v10c5: a leg past a wrist singularity
                                # spiked j5 to 3.7 rad/s, jerked the hand to 65 deg/s, slid the cheek 7 mm)
CHEEK_CARRY_MAX_S = 12.         # longest acceptable carry: upright the cheek still creeps, ever faster
                                # (runs v10c6: a 55 s near-singular leg slid it 23 mm along the pads, out)
CHEEK_CARRY_SQUEEZE_M = .003    # jaw command below the grasp width while upright (place(): .006);
                                # ~14 instead of ~25 N per pad, creep ~40% slower (probe upright_hold)
CHEEK_SETDOWN_M_S = .08         # put-back: lowering from the lift pose (flat: keep it short)
CHEEK_CARRY_RISE_M = .03        # carry height above the higher of the turn pose and the hover
CHEEK_SLIP_SETTLE_S = .3
CHEEK_SLIP_HOLD_S = 1.
CHEEK_SLIP_DEG_S = .5           # in-hand turn rate upright (probe: ~0.05-0.1 deg/s)
CHEEK_SLIP_MM_S = 1.            # in-hand slide rate upright (probe: ~0.1-0.2 mm/s)
CHEEK_SLIP_TURN_DEG = 10.       # in-hand turn since the clamp (4.3 deg clamp seating + the turn: 5-7)
CHEEK_CARRY_TURN_DEG = 6.       # in-hand turn in the carry (vs the still hold)
CHEEK_CARRY_SLIDE_MM = 4.       # in-hand slide at the pinch in the carry
CHEEK_PICK_ATTEMPTS = 3


class _CheekSlip(PlanningError):
    """The cheek turned or slid in the grip beyond the carry limits (it is still held)."""


def _in_hand_change(a, b, point=(0., 0., 0.)):
    """mm (of `point`, part frame) and deg between two in-hand poses (gripper-base frame)."""
    point = np.asarray(point, float)
    return (1000*float(np.linalg.norm(b[:3,:3] @ point+b[:3,3]-a[:3,:3] @ point-a[:3,3])),
            math.degrees(Rotation.from_matrix(b[:3,:3] @ a[:3,:3].T).magnitude()))


def _cheek_hold_still(cell, motion, name, seconds):
    q = motion.q
    for _ in range(int(round(seconds/.01))):
        motion._drive(q); cell.tick()
    return _camera_in_hand(cell, motion, name)


def _cheek_pivot_goals(motion, body, g, pre, target, degrees):
    """Hover and seat hand goals as if the cheek had turned `degrees` more about the jaw
    axis (gripper-base y) at the pinch, in the direction gravity turns it while the
    approach points down (its COM lies 15.6 mm along the pads from the pinch): the
    correction the hover alignment will make after the lift and the turn (~3.7 deg)."""
    b, _a, p, rel = motion.held
    pinch = rel @ np.array([g.p[0], g.p[1], .002])+p
    com = rel @ motion.m.body_ipos[b]+p
    sign = -1. if com[0] > pinch[0] else 1.
    out = []
    for deg in degrees:
        Ry = Rotation.from_euler("y", sign*deg, degrees=True).as_matrix()
        rel2, p2 = Ry @ rel, pinch+Ry @ (p-pinch)
        goals = []
        for pos in (pre, target[:3,3]):
            hand_R = target[:3,:3] @ rel2.T
            goals += [np.asarray(pos)+hand_R @ (motion.offset-p2), hand_R]
        out.append(tuple(goals))
    return out


def _cheek_branch_ok(motion, body, q, pp, goal_p, goal_R, retreat, open_width, also=()):
    """As _preinsert, from hover configuration q: the straight approach to the seat goal
    and the open-gripper exit after it are free of contact (planning copy). Free, not
    margin-clear: _cheek_seat dry-runs its descents the same way and re-branches near
    the seat if the alignment there turns the arm onto the guard (on the hover branches
    nearest the stand link5 passes the guard at ~3.7 mm, inside the 4 mm margin)."""
    width = motion.width()
    prev = q
    for u in np.linspace(0, 1, 34)[1:]:
        qn = motion._ik_one((1-u)*np.asarray(pp)+u*np.asarray(goal_p), goal_R, prev, width=width)
        if qn is None or not motion.segment_free(prev, qn, width=width, allow_pad_body=body, clearance=False):
            return False
        prev = qn
    motion._fk(prev, width)
    saved_held = motion.held
    motion.held = None
    try:
        for u in np.linspace(0, 1, 32)[1:]:
            qn = motion._ik_one(np.asarray(goal_p)+u*np.asarray(retreat), goal_R, prev, width=open_width)
            if qn is None or not motion.segment_free(prev, qn, width=open_width, allow_pad_body=body, clearance=False):
                return False
            prev = qn
    finally:
        motion.held = saved_held
    # The same branch must also descend free of contact after the hover alignment has
    # corrected the cheek's in-grip turn (alternative goals, _cheek_pivot_goals).
    for pp2, rr2, gp2, gr2 in also:
        prev = motion._ik_one(pp2, rr2, q, width=width)
        if prev is None:
            return False
        for u in np.linspace(0, 1, 34)[1:]:
            qn = motion._ik_one((1-u)*np.asarray(pp2)+u*np.asarray(gp2), gr2, prev, width=width)
            if qn is None or not motion.segment_free(prev, qn, width=width, allow_pad_body=body, clearance=False):
                return False
            prev = qn
    return True


def _cheek_legs(p0, r0, pp, rr):
    """Straight carry legs at the mounting attitude, from the turned pose to the hover:
    up to the carry height, across, down. The yaw about the vertical turns with the move
    across, while rising, in place before or after, or the long way round (in thirds of
    the move across): variants for _cheek_turn_plan, which needs only one that is clear
    and quick (a near-singular stretch makes a leg slow)."""
    z = max(float(p0[2]), float(pp[2]))+CHEEK_CARRY_RISE_M
    up, over, pp = np.array([p0[0], p0[1], z]), np.array([pp[0], pp[1], z]), np.asarray(pp, float)
    r0, rr = np.asarray(r0), np.asarray(rr)
    rv = Rotation.from_matrix(rr @ r0.T).as_rotvec()
    angle = float(np.linalg.norm(rv))
    variants = {"across+yaw": ((up, r0), (over, rr), (pp, rr)),
                "up+yaw": ((up, rr), (over, rr), (pp, rr)),
                "yaw over": ((up, r0), (over, r0), (over, rr), (pp, rr)),
                "yaw first": ((np.asarray(p0, float), rr), (up, rr), (over, rr), (pp, rr)),
                "yaw at top": ((up, r0), (up, rr), (over, rr), (pp, rr))}
    if angle > 1e-3:
        axis = rv/angle
        longway = angle-2*math.pi
        legs = [(up, r0)]
        for k in (1, 2):
            legs.append((up+(over-up)*k/3, Rotation.from_rotvec(axis*longway*k/3).as_matrix() @ r0))
        variants["long way"] = tuple(legs+[(over, rr), (pp, rr)])
    return variants


def _cheek_leg_timing(qs, p0, r0, leg_p, leg_R):
    """Duration, distance and angle of a straight leg (smooth profile): the hand at
    CHEEK_CARRY_M_S / CHEEK_CARRY_RAD_S on average, no joint above CHEEK_CARRY_QDOT."""
    dist = float(np.linalg.norm(np.asarray(leg_p)-np.asarray(p0)))
    angle = float(Rotation.from_matrix(np.asarray(leg_R) @ np.asarray(r0).T).magnitude())
    steps = np.abs(np.diff(np.asarray(qs), axis=0))
    q_time = 1.875*len(steps)*float(steps.max())/CHEEK_CARRY_QDOT if len(steps) else 0.
    return max(.2, dist/CHEEK_CARRY_M_S, angle/CHEEK_CARRY_RAD_S, q_time), dist, angle


def _cheek_leg(motion, body, leg_p, leg_R, *, clearance=True):
    """Execute one straight leg with _cheek_leg_timing."""
    qs = motion.line(leg_p, leg_R, allow_pad_body=body, clearance=clearance, execute=False)
    p0, r0 = motion.tool_pose()
    duration, dist, angle = _cheek_leg_timing(qs, p0, r0, leg_p, leg_R)
    motion.line(leg_p, leg_R, allow_pad_body=body, clearance=clearance, speed=max(dist/duration, 1e-6),
                angular_speed=max(angle/duration, 1e-6), smooth=True)


def _cheek_legs_end(motion, body, q, p0, r0, legs):
    """Planning-only run of straight legs from configuration q (pose p0, r0): end
    configuration and total duration (_cheek_leg_timing), or (None, inf)."""
    total = 0.
    try:
        for leg_p, leg_R in legs:
            qs = motion.line(leg_p, leg_R, allow_pad_body=body, clearance=True, start_q=q, execute=False)
            total += _cheek_leg_timing(qs, p0, r0, leg_p, leg_R)[0]
            q, p0, r0 = qs[-1], leg_p, leg_R
        return q, total
    except PlanningError:
        return None, math.inf


def _cheek_joint_attitude(motion, a, b, speed):
    """Planning copy: (duration, pivot exposure, largest upward and largest absolute
    vertical component of the approach) of the direct joint move a -> b at `speed`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    duration = max(.2, float(np.max(np.abs(b-a)))*1.875/speed)
    vertical = []
    for u in np.linspace(0, 1, 41):
        s = u*u*u*(10+u*(-15+6*u))          # Motion.execute's time scaling
        motion._fk((1-s)*a+s*b, motion.width())
        vertical.append(float(motion.pd.xmat[motion.hand].reshape(3, 3)[2, 2]))
    vertical = np.asarray(vertical)
    return duration, duration*float(np.mean(np.abs(vertical))), float(vertical.max()), float(np.abs(vertical).max())


def _cheek_turn_plan(motion, body, up_R, lift_p, pp, rr, goal_p, goal_R, retreat, open_width, also=()):
    """How the cheek gets from the lift pose to its hover, upright (planning only), in this
    order: (1) one direct joint move to a hover branch with a clear descent and exit, if
    it holds the approach off level only briefly (CHEEK_PIVOT_EXPOSURE_S, never far
    fingertips-up); else an upright turn (rise x yaw about the vertical x IK branch, least
    joint travel first, direct joint segment, same attitude limits) followed by (2) a
    direct joint move to a hover branch keeping the approach level (CHEEK_LEVEL_TILT) or
    (3) straight legs at the mounting attitude (_cheek_legs, <= CHEEK_CARRY_MAX_S).
    Returns ("direct", q_hover), ("turn+joint", q_turn, q_hover), ("legs", q_turn,
    variant) or None."""
    q0 = motion.q
    width = motion.width()
    hovers, robust = [], []
    for q in motion.ik(pp, rr, width=width, attempts=48, max_solutions=12, allow_pad_body=body, clearance=True):
        if _cheek_branch_ok(motion, body, q, pp, goal_p, goal_R, retreat, open_width):
            hovers.append(q)
            if _cheek_branch_ok(motion, body, q, pp, goal_p, goal_R, retreat, open_width, also=also):
                robust.append(q)
    # Branches whose descent survives the hover alignment's correction first (no re-branch
    # near the seat), then the others (_cheek_seat re-branches if it must).
    key = lambda q: float(np.max(np.abs(q-q0)))
    hovers = sorted(robust, key=key)+sorted([q for q in hovers if not any(q is r for r in robust)], key=key)
    for qh in hovers:
        duration, exposure, up, _level = _cheek_joint_attitude(motion, q0, qh, CHEEK_TURN_SPEED)
        if (exposure <= CHEEK_PIVOT_EXPOSURE_S and up <= CHEEK_FINGERTIPS_UP
                and motion.segment_free(q0, qh, width=width, allow_pad_body=body, clearance=True)):
            _log("cheek to its hover in one joint move", round(duration, 1), "s, approach off level", round(exposure, 2),
                 "s (of", len(hovers), "hover branches;", "robust" if any(qh is r for r in robust) else "not robust",
                 "to the alignment's correction)")
            return ("direct", qh)
    options = []
    for rise in CHEEK_TURN_RISE_M:
        for yaw in CHEEK_TURN_YAWS_DEG:
            Rz = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            for q in motion.ik(np.asarray(lift_p)+np.array([0., 0., rise]), Rz @ up_R, width=width,
                               attempts=24, max_solutions=6, allow_pad_body=body, clearance=True):
                options.append((float(np.max(np.abs(q-q0))), yaw, rise, q))
    options.sort(key=lambda o: o[0])
    tried = 0
    for delta, yaw, rise, q in options[:CHEEK_TURN_OPTIONS]:
        _duration, exposure, up, _level = _cheek_joint_attitude(motion, q0, q, CHEEK_TURN_SPEED)
        if (exposure > CHEEK_PIVOT_EXPOSURE_S or up > CHEEK_FINGERTIPS_UP
                or not motion.segment_free(q0, q, width=width, allow_pad_body=body, clearance=True)):
            continue
        tried += 1
        for qh in sorted(hovers, key=lambda h: float(np.max(np.abs(h-q)))):
            duration, _exposure, _up, level = _cheek_joint_attitude(motion, q, qh, CHEEK_CARRY_QDOT)
            if (level <= CHEEK_LEVEL_TILT
                    and motion.segment_free(q, qh, width=width, allow_pad_body=body, clearance=True)):
                _log("cheek upright turn", round(delta, 3), "rad, yaw", yaw, "deg, rise", rise, "m; joint move to the hover",
                     round(duration, 1), "s (direct turns tried", tried, "of", len(options), "configurations)")
                return ("turn+joint", q, qh)
        p_t, r_t = motion._fk(q, width)
        for variant, legs in _cheek_legs(p_t, r_t, pp, rr).items():
            end, duration = _cheek_legs_end(motion, body, q, p_t, r_t, legs)
            if (end is not None and duration <= CHEEK_CARRY_MAX_S
                    and _cheek_branch_ok(motion, body, end, pp, goal_p, goal_R, retreat, open_width)):
                _log("cheek upright turn", round(delta, 3), "rad, yaw", yaw, "deg, rise", rise, "m; straight carry",
                     variant, round(duration, 1), "s (direct turns tried", tried, "of", len(options), "configurations)")
                return ("legs", q, variant)
    return None


def _cheek_back_to_stand(cell, motion, name, body, turn, done, stand_T, width, hold_width=None):
    """Put the cheek back on its stand the way it came: executed carry legs reversed (to
    the turn pose; a joint move there if that fails), the turn reversed, lowered flat to
    5 mm then 0.3 mm above its stand pose (or released where it already touches it),
    released, gripper backed off."""
    motion.hold_body(body)
    try:
        for leg_p, leg_R in reversed(done):
            _cheek_leg(motion, body, leg_p, leg_R, clearance=False)
    except PlanningError as error:
        _log("cheek return legs blocked:", str(error)[:100], "- joint move to the turn pose")
    if np.max(np.abs(motion.q-turn[-1])) > 2e-3:
        for clearance in (True, False):
            try:
                motion.move_joints(turn[-1], width=motion.width(), allow_pad_body=body, speed=.5, clearance=clearance)
                break
            except PlanningError:
                if not clearance:
                    raise
    if hold_width is not None:
        motion.set_width(hold_width)    # the full pick squeeze again before it is flat
        motion.hold_body(body)
    motion.execute([motion.q, turn[0]], speed=CHEEK_TURN_SPEED)
    for above, speed in ((.005, CHEEK_SETDOWN_M_S), (.0003, .005), (.002, .005)):
        motion.hold_body(body)
        pp, rr = motion.body_goal(stand_T[:3,3]+np.array([0., 0., above]), stand_T[:3,:3], body)
        try:
            motion.line(pp, rr, allow_pad_body=body, clearance=False, speed=speed, smooth=above > .004)
            if above < .005:
                break
        except PlanningError as error:
            if above < .005:
                # It turned a little more in the jaws on the way down and already
                # touches its stand within a few millimetres: set it down here.
                _log("cheek set-down", round(1000*above, 1), "mm above the stand blocked:", str(error)[:80])
                continue
            # Turned in the grip, the cheek needs a turned hand at its stand pose: the
            # straight way down from the lift pose can meet the arm's own links.
            _log("cheek put-back descent blocked:", str(error)[:100], "- joint move to 5 mm above the stand")
            options = motion.ik(pp, rr, width=motion.width(), attempts=48, max_solutions=8, allow_pad_body=body,
                                clearance=False)
            if not options:
                raise
            for k, q in enumerate(options):
                try:
                    motion.move_joints(q, width=motion.width(), allow_pad_body=body, speed=.25, clearance=False)
                    break
                except PlanningError:
                    if k == len(options)-1:
                        raise
    motion.hold_body(None)
    motion.set_width(width)
    motion.wait(1.0)
    p_now, r_now = motion.tool_pose()
    motion.line(p_now-r_now[:,2]*.10, r_now, clearance=False, speed=.03)
    motion.wait(.5)
    _log("cheek put back", name, "stand drift mm",
         round(1000*float(np.linalg.norm(cell.pose(name)[:3,3]-stand_T[:3,3])), 3))


def _cheek_turn_carry(cell, motion, mate, name, body, g, stand_T, width, ins, pre_distance, retreat_distance,
                      contact_carriers, clamped, last_attempt=False):
    """Cheek only, right after the lift: to the hover upright (_cheek_turn_plan), a still
    hold after the turn, a grip check after every further move. Returns ("carried",
    (turn, legs)) at the hover, ("put back", None) after returning it to its stand (try
    the next pinch), or ("fallback", None) if on the last attempt no upright way exists
    (the generic transport follows)."""
    target = _target(cell, mate, g)
    pre = target[:3,3]-ins*pre_distance
    lift_p, _ = motion.tool_pose()
    motion.hold_body(body)

    def goals():
        pp, rr = motion.body_goal(pre, target[:3,:3], body)
        goal_p, goal_R = motion.body_goal(target[:3,3], target[:3,:3], body)
        return pp, rr, goal_p, goal_R
    _p, up_R = motion.body_goal(target[:3,3], target[:3,:3], body)
    pp, rr, goal_p, goal_R = goals()
    retreat, open_width = -goal_R[:,2]*retreat_distance, min(.085, g.width+.012)
    motion.set_mate_contact(contact_carriers, target[:3,3], target[:3,:3])
    also = _cheek_pivot_goals(motion, body, g, pre, target, CHEEK_PIVOT_CHECK_DEG)
    plan = _cheek_turn_plan(motion, body, up_R, lift_p, pp, rr, goal_p, goal_R, retreat, open_width, also)
    if plan is None:
        if last_attempt:
            _log("cheek: no upright way to the hover from here:", name, "- generic transport")
            return "fallback", None
        _log("cheek: no upright way to the hover from here:", name, "- back to the stand, next pinch")
        _cheek_back_to_stand(cell, motion, name, body, [motion.q, motion.q], [], stand_T, width)
        return "put back", None
    turn = [motion.q, plan[1]]
    motion.execute(turn, speed=CHEEK_TURN_SPEED)
    # Slide is measured at the pinch (the pad centre line at mid-thickness), not at the
    # part origin 37 mm away, which every in-hand turn would move as well.
    pinch = np.array([g.p[0], g.p[1], .002])
    a = _cheek_hold_still(cell, motion, name, CHEEK_SLIP_SETTLE_S)
    b = _cheek_hold_still(cell, motion, name, CHEEK_SLIP_HOLD_S)
    mm, deg = _in_hand_change(a, b, pinch)
    _turned_mm, turned = _in_hand_change(clamped, b, pinch)
    rate_mm, rate_deg = mm/CHEEK_SLIP_HOLD_S, deg/CHEEK_SLIP_HOLD_S
    held = (rate_deg <= CHEEK_SLIP_DEG_S and rate_mm <= CHEEK_SLIP_MM_S and turned <= CHEEK_SLIP_TURN_DEG
            and _gripped(cell, name))
    _log("cheek slip check", name, g.label, "upright creep deg/s", round(rate_deg, 3), "mm/s", round(rate_mm, 3),
         "turned since clamp deg", round(turned, 2), "held" if held else "CREEPS -> put back")
    if not held:
        _cheek_back_to_stand(cell, motion, name, body, turn, [], stand_T, width)
        return "put back", None
    motion.set_width(g.width-CHEEK_CARRY_SQUEEZE_M)
    motion.hold_body(body)
    if plan[0] == "direct":
        return "carried", (turn, [])
    pp, rr, goal_p, goal_R = goals()      # with the cheek as it is held now
    done = []
    try:
        if plan[0] == "turn+joint":
            if not motion.segment_free(motion.q, plan[2], width=motion.width(), allow_pad_body=body, clearance=True):
                raise PlanningError("upright joint move to the hover no longer clear")
            motion.execute([motion.q, plan[2]], speed=CHEEK_CARRY_QDOT)
            checks = [None]
        else:
            p0, r0 = motion.tool_pose()
            checks = _cheek_legs(p0, r0, pp, rr)[plan[2]]
        for leg in checks:
            if leg is not None:
                start = motion.tool_pose()
                _cheek_leg(motion, body, *leg)
                done.append(start)
            mm, deg = _in_hand_change(b, _camera_in_hand(cell, motion, name), pinch)
            if deg > CHEEK_CARRY_TURN_DEG or mm > CHEEK_CARRY_SLIDE_MM or not _gripped(cell, name):
                raise _CheekSlip(f"in-hand turn {deg:.2f} deg, slide {mm:.2f} mm")
    except (PlanningError, _CheekSlip) as error:
        _log("cheek carry stopped:", name, str(error)[:140], "- back to the stand")
        _cheek_back_to_stand(cell, motion, name, body, turn, done, stand_T, width, g.width-g.squeeze)
        return "put back", None
    _log("cheek at the hover", name, "in-hand turn in the carry deg", round(deg, 2), "slide mm", round(mm, 2))
    return "carried", (turn, done)


def camera_grasps(cell, name):
    """Axial pinch for the board camera (camera only): board front <-> lens front.

    The jaws close along the lens axis (part x) on the 38 mm board's front
    face and on the lens front, with the hand running down the board plane
    (approach = part -z), so the hand is exactly DOWN at the mount and the
    camera's COM lies in the jaw axis' vertical plane: gravity has no moment
    about the jaw axis, and the 24 mm span resists the other two axes.

    Pinches across the lens axis are not used. With the stock pads (condim 3,
    no torsional friction) MuJoCo's soft friction lets a held part creep about
    the jaw axis when gravity has a moment about it: the 17.6 mm lens ring
    (one collision piece per pad, 3.4 mm contact) turned freely (88 deg in
    transport) and the board's 2 mm side edges crept 1.2-1.5 deg/s at 18-76 N
    at the mount orientation. The axial pinch needs the board
    front and the lens free: it is only yielded when the bench pose leaves
    both above the table (the camera leaning on its board edge and lens rim),
    not for the kit's board-down rest.
    """
    import trimesh
    part = cell.kit["parts"][cell.graph["instances"][name]["kind"]]
    proxy = trimesh.util.concatenate([
        trimesh.load_mesh(cell.assets/"given/kit/collision"/fn) for fn in part["collision"]])
    T = cell.pose(name)
    approach = np.array([0., 0., -1.])
    if (T[:3, :3] @ approach)[2] > -math.cos(math.radians(45)):
        _log("camera: no bench pose access for the axial pinch (board front/lens on the table); no grasp")
        return
    s_grid = np.linspace(-.0165, .0165, 12)
    t_grid = np.linspace(-.0229, .0171, 15)
    pad = np.array([[s_, 0., t_] for t_ in t_grid for s_ in s_grid])
    candidates = []
    for sign in (1., -1.):
        jaw = np.array([sign, 0., 0.])
        R = np.column_stack([np.cross(jaw, approach), jaw, approach])
        for z_tip in CAMERA_TIP_HEIGHTS_M:
            width0 = .0244
            p = np.array([0., 0., z_tip])-approach*(_tip(width0)-.158)
            # Both pads' lowest corners stay 2 mm above the table plane (z = 0).
            corners = np.array([p+R @ np.array([s, y, t]) for s in (-.015, .015)
                                for y in (-.017, .017) for t in (-.0214, .0156)])
            lowest = float(((corners @ T[:3, :3].T)[:, 2]).min()+T[2, 3])
            if lowest < .002:
                continue
            origins = p+pad @ R.T-jaw*.05
            hits, _, _ = proxy.ray.intersects_location(
                origins, np.tile(jaw, (len(origins), 1)), multiple_hits=True)
            if not len(hits):
                continue
            along = (hits-p) @ jaw
            low, high = float(along.min()), float(along.max())
            width = high-low
            if not .022 <= width <= .027:
                continue
            support = min(int(np.sum(along < low+.0012)), int(np.sum(along > high-.0012)))
            if support < 4:
                continue
            slide = _camera_band_top(proxy, jaw, z_tip, .0012)-z_tip
            # Pads ABOVE the lens-glass apex (z 19 mm) first: there the front pad
            # bears on the lens ring (pieces 4/5) and the back pad on five board
            # pieces; at z16/z18 the front pad meets only the glass apex (one
            # piece, one point) and the camera creeps down/pitches 0.3-0.8 mm/s
            # in the jaws.
            # z24 creeps slowly (0.1 mm/s) but its ring band is only 3.8 mm
            # tall: it ran off in the carry, so the slip check
            # also compares the expected carry slide with slide_m.
            candidates.append((100*abs(z_tip-CAMERA_TIP_PREFERRED_M)-.01*support,
                               Grasp(p+jaw*(low+high)/2, R, width,
                                     f"axial-z{1000*z_tip:.0f}-{'+' if sign > 0 else '-'}",
                                     # 12-14 N holds still at the mount
                                     # orientation; at ~35 N the curved lens
                                     # front squeezed out of the pads.
                                     squeeze=.004, settle_s=1., slide_m=slide)))
    for _, g in sorted(candidates, key=lambda c: c[0]):
        yield g


def camera_board_grasps(cell, name):
    """Flat-to-flat pinch of the camera board's top band against its mount plate (camera only).

    Jaws along the lens axis (part x) as for the axial pinch, hand DOWN, but
    with the pads above the lens ring (z > 27.8 mm): the front pad on the
    board's front face, the back pad on the mount plate behind the board.
    Both are flat multi-piece contacts with ~10 mm overlap, so in-hand creep
    neither concentrates on one point nor runs a pad off a 3-4 mm ring strip.
    Only yielded when the plate exists behind the board (measured jaw span
    6-12 mm); with the 12 mm tab the span is the bare 1.7 mm board, below
    what the jaws can squeeze (they close to 1.65 mm at the .85 rad joint
    limit; probe: 0 N, the camera stays on the bench): none.
    """
    import trimesh
    part = cell.kit["parts"][cell.graph["instances"][name]["kind"]]
    proxy = trimesh.util.concatenate([
        trimesh.load_mesh(cell.assets/"given/kit/collision"/fn) for fn in part["collision"]])
    T = cell.pose(name)
    approach = np.array([0., 0., -1.])
    if (T[:3, :3] @ approach)[2] > -math.cos(math.radians(45)):
        return
    top = float(proxy.bounds[1][2])
    candidates = []
    for sign in (1., -1.):
        jaw = np.array([sign, 0., 0.])
        R = np.column_stack([np.cross(jaw, approach), jaw, approach])
        for z_tip in CAMERA_BOARD_TIP_HEIGHTS_M:
            zs = np.linspace(z_tip+.0005, min(top-.0005, z_tip+.0185), 8)
            ys = np.linspace(-.0145, .0145, 11)
            origins = np.array([[0., y, z] for z in zs for y in ys])-jaw*.06
            hits, rays, _ = proxy.ray.intersects_location(origins, np.tile(jaw, (len(origins), 1)), multiple_hits=True)
            if not len(hits):
                continue
            along = hits @ jaw
            low, high = float(along.min()), float(along.max())
            width = high-low
            if not .006 <= width <= .012:
                continue
            support = min(int(np.sum(along < low+.0006)), int(np.sum(along > high-.0006)))
            if support < 20:
                continue
            slide = _camera_band_top(proxy, jaw, z_tip, .0006)-z_tip
            contact = np.array([0., 0., z_tip])+jaw*(low+high)/2
            p = contact-approach*(_tip(width)-.158)
            candidates.append((1000*abs(z_tip-CAMERA_BOARD_TIP_HEIGHTS_M[0])-.001*support,
                               Grasp(p, R, width, f"board-z{1000*z_tip:.0f}-{'+' if sign > 0 else '-'}",
                                     squeeze=.004, settle_s=1., slide_m=slide)))
    for _, g in sorted(candidates, key=lambda c: c[0]):
        yield g


# Board-top pinch pad-tip heights: above the lens ring top (27.8 mm) with the
# finger tips (1.5 mm past the pads) clear of the lens barrel (top 25 mm).
CAMERA_BOARD_TIP_HEIGHTS_M = (.029, .028, .030)


def grasps(cell, name):
    """Antipodal box/cylinder grasps, followed by collision-mesh ray candidates."""
    kind = cell.graph["instances"][name]["kind"]
    part = cell.kit["parts"][kind]
    if kind == "lidar":
        yield from lidar_housing_grasps(cell, name)
        return
    if kind == "camera":
        yield from camera_board_grasps(cell, name)
        yield from camera_grasps(cell, name)
        return
    if kind == "battery":
        yield from battery_grasps(cell, name)
        return
    if part.get("grasp_mode") == "surface":
        yield from surface_grasps(cell, name)
        return
    bb = np.asarray(part["bounds_m"])
    center = (bb[0]+bb[1])/2
    if kind in ("prop_cw", "prop_ccw"):
        # Pinch the pitched blade across its thickness, away from the three
        # roots and hub. CCW is the source mesh's reflected CW geometry.
        mirror = 1. if kind == "prop_cw" else -1.
        R = Rotation.from_euler("z",240 if mirror > 0 else 120,degrees=True).as_matrix()
        R = R @ Rotation.from_euler("x",45*mirror,degrees=True).as_matrix() @ DOWN
        p = np.array([-.023529567724055858,-.026829716583874622*mirror,.013179790526569504])
        # Blade deep in the pads (see PROP_GRASP_DEPTH_M): the tool point
        # moves along the approach, the pinched blade section stays the same.
        p = p + R[:,2]*PROP_GRASP_DEPTH_M
        for target_yaw in (0.,math.pi/2,-math.pi/2,math.pi):
            for yaw in (0.,120.,240.):
                symmetry=Rotation.from_euler("z",yaw,degrees=True).as_matrix()
                yield Grasp(symmetry @ p,symmetry @ R,.00586788925370038,
                            f"pitched-blade-normal-{yaw:.0f}",target_yaw=target_yaw,settle_s=1.)
    if kind == "landing_strut":
        import trimesh
        shaft_mesh=trimesh.load_mesh(cell.assets/"given/kit/meshes"/part["mesh"])
        def shaft_grasp(R,height,label,target_yaw=0.):
            # Measure the actual shaft over the pad's longitudinal footprint;
            # the native aluminium shaft is narrower than the old 8 mm proxy.
            centerline=np.array([center[0],center[1],height])
            centers=[centerline+offset*R[:,0] for offset in (-.012,0.,.012)]
            origins=np.asarray(centers)-R[:,1]*.05
            hits,rays,faces=shaft_mesh.ray.intersects_location(origins,np.tile(R[:,1],(3,1)),multiple_hits=True)
            mids=[];spans=[]
            for k in range(3):
                use=np.flatnonzero(rays==k)
                if len(use)<2:return None
                use=use[np.argsort(hits[use]@R[:,1])];a,b=use[0],use[-1]
                if shaft_mesh.face_normals[faces[a]]@R[:,1]>-math.cos(math.radians(25)) or shaft_mesh.face_normals[faces[b]]@R[:,1]<math.cos(math.radians(25)):
                    return None
                spans.append(float((hits[b]-hits[a])@R[:,1]));mids.append((hits[a]+hits[b])/2)
            width=max(spans)
            if not .004<=width<=.020:return None
            # Centre the real shaft in pad_2's 19 mm height. The old fixed
            # 8 mm tool offset put this narrower shaft near the distal tip;
            # its initial 6 mm slip then rolled it out during reorientation.
            pad_center_offset=_tip(width)-.158-.0095
            return Grasp(np.mean(mids,axis=0)-R[:,2]*pad_center_offset,R,width,label,
                         target_yaw,squeeze=.004,settle_s=1.)
        # Align the long pad dimension with the settled shaft's horizontal
        # projection; a valid resting pose need not point along world X.
        body_R=cell.pose(name)[:3,:3]
        shaft_horizontal=body_R[:,2].copy();shaft_horizontal[2]=0.
        if np.linalg.norm(shaft_horizontal)<1e-6:
            shaft_horizontal=np.array([1.,0.,0.])
        else:
            shaft_horizontal/=np.linalg.norm(shaft_horizontal)
        downward=np.array([0.,0.,-1.])
        world_hand=np.column_stack([shaft_horizontal,np.cross(downward,shaft_horizontal),downward])
        Rtop = body_R.T @ world_hand
        for target_yaw in (0.,math.pi/2,-math.pi/2,math.pi):
            for height in (.040,.035,.045):
                candidate=shaft_grasp(Rtop,height,f"stable-top-{height:.3f}-yaw{target_yaw:.2f}",target_yaw)
                if candidate is not None:yield _bench_tip_lift(cell,name,candidate)
        # Grasp the measured shaft beneath the upper flange. A horizontal hand can
        # reach under the fixed base and insert the upright leg from below.
        for yaw in (-math.pi/2, 0, math.pi/2, math.pi):
            R = Rotation.from_euler("z", yaw).as_matrix() @ np.array([[0,0,1],[0,-1,0],[1,0,0.]])
            for height in (.040, .045, .035, .030):
                candidate=shaft_grasp(R,height,f"shaft-side-{yaw:.2f}-{height:.3f}")
                if candidate is not None:yield candidate
        return
    if kind in ("esc", "flight_controller", "compute", "battery", "lidar", "camera", "motor"):
        for yaw in (0, math.pi/2, math.pi, -math.pi/2):
            Rz = Rotation.from_euler("z", yaw).as_matrix()
            width = (bb[1]-bb[0])[1 if abs(math.sin(yaw)) < .5 else 0]
            if width > .078:
                continue
            engagement = min(.014, (bb[1,2]-bb[0,2])*.5)
            z = bb[1,2]-engagement+_tip(width)-.158
            yield Grasp(np.array([center[0],center[1],z]), Rz @ DOWN, width, f"top-{yaw:.2f}")
        return
    if kind == "sensor_cap":
        # The full 30 mm pad touches the outer arc at its center and the
        # inner arc near both ends. Its effective span is about 10 mm;
        # a single center ray incorrectly predicts a 4.8 mm strip at x=-20 mm.
        for yaw in (math.pi/2,-math.pi/2):
            R=Rotation.from_euler("z",yaw).as_matrix() @ DOWN
            yield Grasp(np.array([-.0175,0.,_tip(.010)-.158-.003]),R,.010,
                        f"pad-footprint-arc-{yaw:.2f}",squeeze=.003,settle_s=2.)
    if kind == "sensor_cheek":
        yield from cheek_rim_grasps(cell, name)
    # Thin manufactured plates and propellers: cast on the actual union of the
    # documented convex pieces and require opposing contact normals. The tool
    # descends only after checking the complete hand at its preshaped width.
    import trimesh
    meshes = [trimesh.load_mesh(cell.assets/"given/kit/collision"/fn) for fn in part["collision"]]
    mesh = trimesh.util.concatenate(meshes)
    z = .001 if kind not in ("prop_cw", "prop_ccw") else .004
    candidates = []
    xs = np.linspace(bb[0,0]+.004,bb[1,0]-.004,13)
    ys = np.linspace(bb[0,1]+.004,bb[1,1]-.004,13)
    centers = [np.array([0.,0.,z])] + [np.array([x,y,z]) for x in xs for y in ys]
    for yaw in (0, math.pi/2, math.pi, -math.pi/2, math.pi/4, -math.pi/4):
        Rz = Rotation.from_euler("z", yaw).as_matrix()
        axis = Rz[:,1]
        origins = np.array([p-axis*.5 for p in centers])
        points, rays, faces = mesh.ray.intersects_location(origins,np.tile(axis,(len(origins),1)),multiple_hits=True)
        for i,p in enumerate(centers):
            use = np.flatnonzero(rays == i)
            if len(use) < 2:
                continue
            order = use[np.argsort(points[use] @ axis)]
            # Adjacent solid spans allow edge/rim grasps rather than requiring
            # the gripper to span a plate's full bounding box.
            for u,v in list(zip(order[:-1], order[1:])) + ([(order[0],order[-1])] if len(order)>2 else []):
                a,b = points[u],points[v]
                width = float((b-a) @ axis)
                if not (.0035 if kind == "sensor_cap" else .006) <= width <= .075:
                    continue
                n1,n2 = mesh.face_normals[faces[u]],mesh.face_normals[faces[v]]
                if n1 @ axis > -math.cos(math.radians(15)) or n2 @ axis < math.cos(math.radians(15)):
                    continue
                pp = (a+b)*.5
                pp[2] = .017 if kind in ("prop_cw","prop_ccw") else _tip(width)-.158-.003
                if kind == "sensor_cap" and width>.020:
                    # The cap's end rims sit directly above the cheek edges.
                    # Engage its upper millimetre so the metal backers clear
                    # both installed supports during precision placement.
                    pp[2]=_tip(width)-.158+.0005
                priority = float(np.linalg.norm(pp[:2]))
                candidates.append((priority, Grasp(pp,Rz @ DOWN,width,f"ray-{yaw:.2f}-{pp[0]:.3f}-{pp[1]:.3f}")))
    seen = set()
    for _,g in sorted(candidates,key=lambda x:x[0]):
        key=tuple(np.round(g.p/.005).astype(int))+tuple(np.round(g.R[:,1],2))
        if key not in seen:
            seen.add(key); yield g


# Stock finger link vs its pads: the finger plate ends 1.5 mm beyond the pad
# tips, and the parallel linkage reaches furthest from the hand (pad tips at
# .1736 m) near an 18 mm aperture, i.e. while the open jaws approach a thin shaft.
FINGER_TIP_BEYOND_PAD_M = .0015
LINKAGE_REACH_MAX_M = .059098+math.hypot(.035465, .042039)+.0595
BENCH_TIP_CLEARANCE_M = .0005


def _bench_tip_lift(cell, name, grasp):
    """Raise a straight top-down bench grasp until the finger tips clear the table.

    Landing struts only. The stable-top grasp centres the pads on the lying
    shaft's axis, 10.4 mm above the bench: the pad tips end 0.9 mm above it and
    the finger plates 0.6-0.9 mm inside it (open approach and grip; the finger
    links are visual-only in the cell, so only the planning model's finger
    colliders see this). Moving the tool point back along the approach keeps
    the shaft's full diameter between the pads (the contact lines stay at its
    equator) and the finger tips BENCH_TIP_CLEARANCE_M above the table plane
    (z = 0) at any aperture.
    """
    p, R = _world(cell.pose(name), grasp)
    if R[2, 2] > -.999:
        return grasp
    low = p[2]+R[2, 2]*(LINKAGE_REACH_MAX_M-.158+FINGER_TIP_BEYOND_PAD_M)
    lift = BENCH_TIP_CLEARANCE_M-low
    if lift > 0:
        grasp.p = grasp.p-grasp.R[:, 2]*lift
        grasp.label += f"-lift{1000*lift:.1f}"
    return grasp


def _world(T, grasp):
    return T[:3,3]+T[:3,:3] @ grasp.p, T[:3,:3] @ grasp.R


def _target(cell,mate,grasp):
    T=cell.coupling.target(mate).copy()
    if grasp.target_yaw:
        T[:3,:3] = T[:3,:3] @ Rotation.from_euler("z",grasp.target_yaw).as_matrix()
    return T


def _least_travel_first(cell, motion, mate, candidates, pre_offset, count=4):
    """Camera / flight controller / upright LiDAR only: order the leading grasps by joint travel.

    Their best candidates come in mirrored pairs (the same contact patch with
    the jaws swapped) whose priorities tie to ~1e-9, so bench micro-drift used
    to decide the arm branch. With joint 7 wound near its limit, the wrong
    twin spent ~30 s unwinding the wrist (FC 75.6 s instead of ~44 s). Rank
    the first `count` IK-feasible candidates by the joint travel to the pick
    and from there to the mount hover; the remaining candidates follow in
    their original order. Planning only: nothing moves here.
    """
    T = cell.pose(mate["part"])
    q_now = motion.q
    head = []
    for g in candidates:
        p, R = _world(T, g)
        gp, gR = _world(_target(cell, mate, g), g)
        pick = motion.ik(p, R, width=min(.085, g.width+.012), attempts=10, max_solutions=1, clearance=False)
        mount = motion.ik(gp+pre_offset, gR, width=g.width, attempts=10, max_solutions=1,
                          seed=pick[0]) if pick else []
        if mount:
            travel = float(np.max(np.abs(pick[0]-q_now))+np.max(np.abs(mount[0]-pick[0])))
            head.append((travel, len(head), g))
            _log("travel", mate["part"], g.label, round(travel, 2))
        if len(head) >= count:
            break
    # Near-ties (within TRAVEL_TIE_RAD) keep the original grasp priority: two FC
    # patches 2 mm apart (surface-1.57-96 / -97) tie to <0.02 rad, and the
    # arm's history then decided between the validated 38.5 mm Cube pinch and a
    # 40.5 mm pinch that slipped out in transport.
    # Camera too: its +twins at z16..z24 lie within 0.03 rad, so sub-mrad
    # travel differences (CPU-dependent bench settling) picked z16 on one
    # host and z18 on another; keep camera_grasps' priority among them.
    # Upright LiDAR (cable-free kit): the same rule among its jaw orientations.
    tie = TRAVEL_TIE_RAD if cell.graph["instances"][mate["part"]]["kind"] in ("flight_controller", "camera", "lidar") else 0.
    best = min((h[0] for h in head), default=0.)
    near = sorted((h for h in head if h[0] <= best+tie), key=lambda h: h[1])
    far = sorted((h for h in head if h[0] > best+tie), key=lambda h: h[:2])
    if cell.graph["instances"][mate["part"]]["kind"] == "camera":
        # The whole decision, travel to 1 urad (the "travel" lines round to
        # 0.01 rad, which hid the z16/z18 near-tie in development).
        _log("camera order", [(g.label, round(t, 6)) for t, _, g in near+far], "tie window <=", round(best+tie, 6))
    for _, _, g in near+far:
        yield g
    yield from candidates


TRAVEL_TIE_RAD = .1


# Per-pad squeeze assumed by the closed-jaw grasp check: the live pads press
# 0.05-0.6 mm into a gripped part (soft contacts under the closing force).
GRIP_CHECK_SQUEEZE_M = .0005


def _grip_clear(motion, q, grasp, body, name):
    """Closed-jaw check of a grasp at its pick, with the finger link colliders.

    The jaws close on the part at the measured contact width, squeeze each pad
    a few tenths of a millimetre into it, and closing lowers the fingertips
    (parallel linkage). At the contact width and at the squeezed width only
    pad-face contact with the part is accepted (Motion.collision): a finger
    side, tip or back inside the part (a cheek rim, a leg flange, the LiDAR
    housing edge) or any contact with another body rejects the candidate
    before the arm moves. Planning only.
    """
    for w in (grasp.width, max(0., grasp.width-2*GRIP_CHECK_SQUEEZE_M)):
        if motion.collision(q, w, allow_pad_body=body, clearance=False):
            _log("reject grasp", name, grasp.label, "closed width", round(1000*w, 2), "mm:", motion.last_contacts[:3])
            return False
    return True


def _open_clear(motion, body, width, seconds=.6):
    """Open the jaws on a released part only as far as the swept finger links
    stay clear of everything but pad-face contact with that part.

    The opening moves each finger's back and sides ~6 mm outwards. A contact
    that already exists at the closed jaw may remain while it does not get
    deeper (as Motion.line's escape rule); a new or deepening one ends the
    sweep. If even the first millimetre is blocked, open that millimetre
    (logged): the grip must be released before the retreat. With a clear
    sweep this is set_width().
    """
    motion.sync()
    current=motion.width()
    target=current
    motion.collision(motion.q,width=current,allow_pad_body=body,clearance=False)
    previous={}
    for a,b,d in motion.last_contacts:
        key=tuple(sorted((a,b)));previous[key]=min(d,previous.get(key,math.inf))
    for w in np.linspace(current,width,10)[1:]:
        motion.collision(motion.q,width=w,allow_pad_body=body,clearance=False)
        now={}
        for a,b,d in motion.last_contacts:
            key=tuple(sorted((a,b)));now[key]=min(d,now.get(key,math.inf))
        if any(key not in previous or d<previous[key]-5e-5 for key,d in now.items()):
            _log("release aperture limited at",round(1000*w,2),"mm by",motion.last_contacts[:2])
            break
        previous=now
        target=w
    if target<current+.001 and width>current+.001:
        _log("release aperture blocked; minimal opening",round(1000*(current+.001),2),"mm")
        target=current+.001
    motion.set_width(target, seconds=seconds)
    return target


def _grip_forces(cell, name):
    members = body_subtree(cell.m,cell.ids[name])
    sides={"left":0.,"right":0.}
    for i,c in enumerate(cell.d.contact):
        for g,h in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
            gn = cell.m.geom(int(g)).name or ""
            if cell.m.geom_bodyid[h] in members and "finger_pad" in gn:
                force=np.zeros(6)
                mujoco.mj_contactForce(cell.m,cell.d,i,force)
                sides[gn.split("_")[0]]+=max(0.,float(force[0]))
    return sides


def _gripped(cell, name):
    return min(_grip_forces(cell,name).values())>.05


def _prop_squeeze(cell, motion, name, *, min_force=1.5, max_s=4.0):
    """Wait until the stock jaws have actually pinched a thin propeller blade.

    Near full closure the gripper servo's residual force is below the joint
    friction of the stock linkage, so the jaws creep shut over seconds. The
    source blade is ~1 mm thick and twisted: the pads first touch it lightly
    at two points, and the squeeze needs 1-3 s to build. Lifting at first
    touch drags the blade out of the pads. Only waits; never commands more.
    """
    history=[]
    t_end=cell.d.time+max_s
    while cell.d.time<t_end:
        motion.wait(.1)
        forces=_grip_forces(cell,name)
        history.append((round(float(cell.d.time),2),round(1000*motion.width(),2),
                        round(forces["left"],2),round(forces["right"],2)))
        if (len(history)>=3 and min(forces.values())>=min_force
                and abs(history[-3][1]-history[-1][1])<.05):
            break
    _log("prop squeeze",name,history[::2]+history[-1:])
    return history[-1]


def _prop_release(motion, body, p, R, width):
    """Props only: let go of a pinch that did not lock and back out of it.

    Opens the jaws to the preshape width (finger sweep checked), waits for the
    prop to settle back flat on its pedestal, then retracts along the approach
    to the pre-grasp (the same checked line as the pick, reversed).
    """
    _open_clear(motion,body,width)
    motion.wait(1.)
    motion.line(p-R[:,2]*.10,R,width=motion.width(),allow_pad_body=body,clearance=False,speed=.03)


class _HeldMargin:
    """Temporarily enlarge the held part's planning clearance (camera carry).

    Motion.hold_body sizes the held geoms' contact-query margins from
    held_clearance; enlarge both together and restore them afterwards.
    None leaves the planner untouched.
    """
    def __init__(self, motion, clearance):
        self.motion, self.clearance = motion, clearance

    def __enter__(self):
        m = self.motion
        self.saved = (m.held_clearance, m.pm.geom_margin.copy())
        if self.clearance is not None and m.held is not None:
            m.held_clearance = self.clearance
            members = m.body_group(m.held[0])
            for g in range(m.m.ngeom):
                if m.m.geom_bodyid[g] in members:
                    m.pm.geom_margin[g] = max(m.pm.geom_margin[g], self.clearance+m._margin_max)
        return self

    def __exit__(self, *exc):
        self.motion.held_clearance = self.saved[0]
        self.motion.pm.geom_margin[:] = self.saved[1]
        return False


def _preinsert(motion, pre_p, goal_p, R, body, *, retreat=None, open_width=None, attempts=64, clearance=True, transport_clearance=None, speed=.25,
               transport_held_clearance=None):
    """Choose an IK branch for a complete straight approach and optional exit.

    transport_held_clearance (camera only): held-part margin for the transport
    move to the pre pose; the straight approach keeps the normal margin.
    """
    if transport_clearance is None:
        transport_clearance=clearance
    solutions=motion.ik(pre_p,R,width=motion.width(),attempts=attempts,max_solutions=16,allow_pad_body=body,clearance=clearance)
    errors=[]
    for candidate in solutions:
        prev=candidate
        ok=True
        for u in np.linspace(0,1,34)[1:]:
            q=motion._ik_one((1-u)*np.asarray(pre_p)+u*np.asarray(goal_p),R,prev,width=motion.width())
            if q is None or not motion.segment_free(prev,q,width=motion.width(),allow_pad_body=body,clearance=clearance):
                ok=False;break
            prev=q
        if not ok:
            continue
        if retreat is not None:
            # The detached component stays at its final pose in the planning
            # copy while the open gripper retreats. Verify this exit on the
            # same IK branch before moving the real arm into a dead end.
            motion._fk(prev,motion.width())
            saved_held=motion.held
            motion.held=None
            try:
                for u in np.linspace(0,1,32)[1:]:
                    q=motion._ik_one(np.asarray(goal_p)+u*np.asarray(retreat),R,prev,width=open_width)
                    if q is None or not motion.segment_free(prev,q,width=open_width,allow_pad_body=body,clearance=clearance):
                        ok=False;break
                    prev=q
            finally:
                motion.held=saved_held
            if not ok:
                continue
        try:
            with _HeldMargin(motion,transport_held_clearance):
                motion.move_joints(candidate,width=motion.width(),allow_pad_body=body,speed=speed,clearance=transport_clearance)
            return
        except PlanningError as e:
            errors.append(str(e))
    raise PlanningError(f"no collision-free Cartesian approach branch for {body or 'empty gripper'}: {errors[:2]} last {motion.last_contacts[:3]}")


def _prop_axis_error(cell, name, target):
    """Lateral bore-to-shaft-axis error, height above seat and tilt of a held prop."""
    T=cell.pose(name)
    axis=target[:3,2]
    delta=T[:3,3]-target[:3,3]
    height=float(delta@axis)
    lateral=float(np.linalg.norm(delta-height*axis))
    tilt=math.degrees(math.acos(float(np.clip(T[:3,2]@axis,-1,1))))
    return lateral,height,tilt


# Prop seat geometry (fixture frame): hub bottom 0.12 mm above the prop origin,
# rotor shaft end 12 mm above the seat. Above PROP_GATE_M the bore is not yet on
# the shaft, so the servo only descends while the hub is centred on the axis.
PROP_ENTRY_M=.014
PROP_GATE_M=.011
# The bell's convex top meets the 13-sided bore edge ~0.17 mm below nominal;
# end the descent with the hub resting on it, not held above it.
PROP_REST_M=-.00025
# Free-space joint speed for the empty-hand move to a propeller pick.
PROP_FREE_SPEED=.6


# Arm self-clearance kept along a propeller insertion (the planning copy's
# link-link shell is 2 x 4 mm). The in-grasp pivot keeps turning the hand
# during the servo descent, so a branch must also stay contact-free for a few
# degrees of pivot either way; the servo swivels the elbow below a 3 mm gap.
PROP_SELF_SHELL=.008
PROP_SELF_ACCEPT=.003
PROP_SERVO_SELF_MIN=.003
# Servo outer integral (1/s) and its clamp.
PROP_SERVO_KI=1.5
PROP_SERVO_BIAS_MAX=.003
# The prop's final yaw is free: turning it about its own axis at the hover
# gives the hand (45 deg pitched-blade pinch) a different orientation. Pivot
# variants: ~12 deg build up during transport, a few more during the descent.
PROP_YAW_TURNS_DEG=(0.,30.,-30.,60.,-60.,90.,-90.,120.,-120.,150.,-150.,180.)
PROP_TRANSPORT_PIVOT_DEG=(12.,)
PROP_HOVER_PIVOT_DEG=(6.,)
# How much deeper than the original pitched-blade grasp the blade sits in the
# pads (along the approach). Originally the blade's chord, which runs nearly
# along the approach, filled only the distal half of pad_2: with the 45 deg hand
# the pad plane carries 0.71 g, so the pinched blade crept towards the finger
# tips at ~1.1 mm/s (MuJoCo soft friction) and a carry of more than ~10 s could
# end with the blade at the pad edge (one development seed dropped it in
# transport). With the whole chord inside pad_1 + pad_2 the jaws stop on a
# wider envelope (3 N instead of 1-1.8 N after the lift) and the blade hardly
# creeps along the approach (10 and 15 mm deep picks
# alike; 10 mm keeps the lower finger 8.7 mm above the bench). The staging
# stand's pedestal leaves room under the blades for the lower finger.
PROP_GRASP_DEPTH_M=.010
# On its pedestal the deep pitched-blade pinch first holds the blade lightly
# (~0.05 N per pad) while the jaws creep shut and the prop turns ~9 deg into
# the four-corner pinch; then it locks (>= 1 N) within a few tenths of a
# second. For every blade of both prop kinds that happens 3.6-5.1 s after the
# close. The old fixed 3 s squeeze
# ended right at that transition, so the clamp read 0.04-0.4 N by chance
# (one run, prop_2: 0.049/0.039 N -> "two-sided pad contact
# missing"). Wait for the lock (same exit test as the lift-off squeeze).
PROP_STAND_SQUEEZE_S=7.
PROP_STAND_LOCK_N=1.
# Still below this after the stand squeeze: the pinch has not locked. Release
# (the prop settles back on its pedestal), back out along the approach and
# take the next blade; at most PROP_PICK_ATTEMPTS blades.
PROP_CLAMP_MIN_N=.3
PROP_PICK_ATTEMPTS=3


def _prop_robot_names(motion):
    """(robot body names, gripper body names): the arm's self-distance ignores
    pairs inside the gripper (its two fingers sit ~2 mm apart on a blade)."""
    names=getattr(motion,"_prop_robot_names",None)
    if names is None:
        robot=frozenset(motion.m.body(int(b)).name for b in motion.robot_bodies)
        hand=frozenset(motion.m.body(int(b)).name for b in body_subtree(motion.m,motion.hand))
        names=motion._prop_robot_names=(robot,hand)
    return names


def _prop_contacts(motion, q, width, body):
    """One planning query: (contacts that penetrate, arm self-distance).

    Same pair rules as Motion.collision; with clearance on, pairs inside their
    planning shells are listed with their signed distances, so penetration and
    the arm's link self-distance (link-link and link-gripper pairs, capped at
    the 8 mm shell) come from one query."""
    motion.collision(q,width,body,True)
    robot,hand=_prop_robot_names(motion)
    pen=[c for c in motion.last_contacts if c[2]<-2e-5]
    own=[c[2] for c in motion.last_contacts
         if c[0] in robot and c[1] in robot and not (c[0] in hand and c[1] in hand)]
    return pen,(min(own) if own else PROP_SELF_SHELL)


def _prop_variant(held, delta, width):
    """The held transform `held` turned by delta about the jaw axis through
    the pad contact (the direction the pinched blade pivots in)."""
    b,adr,p,rel=held
    if not delta:
        return held
    c=np.array([0.,0.,_tip(width)-.005-PROP_GRASP_DEPTH_M])
    Ry=Rotation.from_euler("y",delta).as_matrix()
    return (b,adr,c+Ry@(np.asarray(p)-c),Ry@np.asarray(rel))


def _prop_tool(motion, held, pos, R):
    """Tool pose that puts the held part (transform `held`) at pos/R."""
    _b,_a,p,rel=held
    hand_R=np.asarray(R)@np.asarray(rel).T
    return np.asarray(pos)+hand_R@(motion.offset-np.asarray(p)),hand_R


def _prop_branch_check(motion, body, q0, target, desR, width, open_width, *, h_top, h_end, pivots_deg=(6.,)):
    """Plan-only check of one arm branch for the whole propeller insertion.

    From configuration q0 (at or near the hover) the straight descent of the
    held prop along its shaft axis, in 3 mm steps, must stay contact-free with
    the arm's link self-distance >= PROP_SELF_ACCEPT; with the measured grasp
    turned by +-each of pivots_deg about the jaw axis the same descent must stay
    contact-free; and from the seat one of the exits of _prop_exit must be
    clear with the jaws open and the prop left on its seat. Never moves the
    robot. Returns (ok, min self-distance of the measured-grasp descent capped
    at the 8 mm shell, reason)."""
    axis=target[:3,2]
    saved=motion.held
    worst=math.inf
    heights=np.linspace(h_top,h_end,max(2,int(math.ceil((h_top-h_end)/.003))+1))
    deltas=(0.,)+tuple(math.radians(sign*p) for p in pivots_deg for sign in (1.,-1.))
    bits=getattr(motion,"_finger_bits",None)
    try:
        for delta in deltas:
            held=_prop_variant(saved,delta,width)
            motion.held=held
            margin=PROP_SELF_ACCEPT if not delta else 0.
            prev=np.asarray(q0,float)
            for j,hk in enumerate(heights):
                tp,tR=_prop_tool(motion,held,target[:3,3]+axis*hk,desR)
                q=motion._ik_one(tp,tR,prev,width=width)
                if q is None:
                    return False,worst,f"no IK at {1000*hk:.0f} mm (pivot {math.degrees(delta):+.0f} deg)"
                if j and np.max(np.abs(q-prev))>.3:
                    return False,worst,f"branch jump at {1000*hk:.0f} mm (pivot {math.degrees(delta):+.0f} deg)"
                pen,own=_prop_contacts(motion,q,width,body)
                if not delta:
                    worst=min(worst,own)
                if pen:
                    return False,worst,f"contact {pen[0][:2]} at {1000*hk:.0f} mm (pivot {math.degrees(delta):+.0f} deg)"
                if own<margin-1e-6:
                    return False,worst,f"self-distance {1000*own:.1f} mm at {1000*hk:.0f} mm (pivot {math.degrees(delta):+.0f} deg)"
                prev=q
            if not delta:
                # Exit from the seat: the prop stays where the last held FK put
                # it; as after release it is an ordinary obstacle again.
                motion._fk(prev,width)
                motion.held=None
                if bits is not None:
                    ct,ca=motion.pm.geom_contype.copy(),motion.pm.geom_conaffinity.copy()
                    motion.pm.geom_contype[:],motion.pm.geom_conaffinity[:]=bits
                try:
                    pp,rr=motion._fk(prev,open_width)
                    rr=rr.copy()
                    exits=[-rr[:,2]*d for d in (.08,.06,.04,.03)]+[rr[:,0]*d for d in (.05,.045)]
                    exits.append((rr[:,0]-rr[:,2])*(.06/math.sqrt(2)))
                    clear=any(_prop_line_free(motion,prev,v,open_width) for v in exits)
                finally:
                    if bits is not None:
                        motion.pm.geom_contype[:],motion.pm.geom_conaffinity[:]=ct,ca
                if not clear:
                    return False,worst,"no clear exit from the seat"
        return True,worst,""
    finally:
        motion.held=saved
        motion.sync()


def _prop_branch_search(motion, body, target, base_R, h_top, width, open_width, *, pivots_deg, q_ref=None,
                        yaw_turns=PROP_YAW_TURNS_DEG, ik_per_yaw=6, keep=3):
    """Hover branches for the held prop that pass _prop_branch_check.

    Candidates are the prop's free-yaw turns about its own axis x the
    contact-free IK branches of the hover pose, tried nearest to q_ref in joint
    space first (the shortest arm move). Returns ([(q, yaw_deg, desR,
    min self-distance, joint distance)], rejection reasons, candidate count)."""
    axis=target[:3,2]
    q_ref=motion.q if q_ref is None else np.asarray(q_ref,float)
    candidates=[]
    for yaw in yaw_turns:
        R=Rotation.from_rotvec(axis*math.radians(yaw)).as_matrix()@np.asarray(base_R)
        tp,tR=_prop_tool(motion,motion.held,target[:3,3]+axis*h_top,R)
        # Contact-free IK only: the pinched blade sits inside the hand's
        # planning shell; _prop_branch_check applies the arm's own margin.
        for qc in motion.ik(tp,tR,width=width,attempts=32,max_solutions=ik_per_yaw,allow_pad_body=body,clearance=False):
            candidates.append((float(np.linalg.norm(qc-q_ref)),yaw,R,qc))
    candidates.sort(key=lambda c:c[0])
    passed,reasons=[],[]
    for dist,yaw,R,qc in candidates:
        ok,worst,why=_prop_branch_check(motion,body,qc,target,R,width,open_width,h_top=h_top,
                                        h_end=PROP_REST_M,pivots_deg=pivots_deg)
        if not ok:
            reasons.append(f"yaw {yaw:+.0f}: {why}")
            continue
        passed.append((qc,yaw,R,worst,dist))
        if len(passed)>=keep:
            break
    # Prefer the full 8 mm clearance among the nearest passing branches.
    passed.sort(key=lambda c:(c[3]<PROP_SELF_SHELL-1e-6,c[4]))
    return passed,reasons,len(candidates)


def _prop_preinsert(motion, body, target, pre_distance, open_width, *, speed=.65):
    """Transport the held prop to a hover branch verified for the whole
    insertion even after the in-grasp pivot expected during transport
    (PROP_TRANSPORT_PIVOT_DEG, then PROP_HOVER_PIVOT_DEG if none passes).
    Returns (yaw turn deg, min self-distance). Raises PlanningError."""
    width=motion.width()
    errors=[]
    for pivots in (PROP_TRANSPORT_PIVOT_DEG,PROP_HOVER_PIVOT_DEG):
        passed,reasons,n=_prop_branch_search(motion,body,target,target[:3,:3],pre_distance,width,open_width,pivots_deg=pivots)
        for qc,yaw,R,worst,dist in passed:
            try:
                motion.move_joints(qc,width=width,allow_pad_body=body,speed=speed,clearance=True)
                return yaw,worst,pivots
            except PlanningError as error:
                errors.append(str(error)[:100])
        errors.extend(reasons[:2])
    raise PlanningError(f"no verified insertion branch at the hover: {errors[:4]}")


def _prop_select_branch(cell, motion, mate, name, body, target, open_width):
    """Re-check the arm branch at the hover with the grasp as it is NOW.

    The pinched blade pivots in the jaws during transport (11-13 deg seen), so
    the hover alignment turns the hand well away from the planned branch; with
    the 45 deg pitched-blade hand that can fold the elbow (joint4 ~0.08) so far
    that link5 meets link2 on the way down, for every IK branch of that hand
    pose. If the present branch fails _prop_branch_check, move at the hover
    (planned joint move, prop held) to the nearest branch that passes, turning
    the prop about its own (free) axis if needed. Returns True if it moved."""
    motion.hold_body("part_"+name)
    width=motion.width()
    P=cell.pose(name)
    desR,_=axis_aligned_rotation(P[:3,:3],target[:3,:3])
    axis=target[:3,2]
    h_top=float((P[:3,3]-target[:3,3])@axis)
    ok,worst,why=_prop_branch_check(motion,body,motion.q,target,desR,width,open_width,h_top=h_top,
                                    h_end=PROP_REST_M,pivots_deg=PROP_HOVER_PIVOT_DEG)
    if ok:
        _log("prop branch",name,"hover branch ok; min self-distance mm",round(1000*worst,2))
        return False
    _log("prop branch",name,"hover branch rejected:",why)
    q_before=motion.q
    passed,reasons,n=_prop_branch_search(motion,body,target,desR,h_top,width,open_width,pivots_deg=PROP_HOVER_PIVOT_DEG)
    for qc,yaw,R,worst,dist in passed:
        try:
            motion.move_joints(qc,width=width,allow_pad_body=body,speed=.4,clearance=True)
        except PlanningError as error:
            reasons.append(str(error)[:100])
            continue
        _log("prop branch",name,"re-configured at the hover: yaw turn deg",yaw,"joint move",round(dist,2),
             "min self-distance mm",round(1000*worst,2),"dq",np.round(qc-q_before,2))
        return True
    # Nothing better reachable: keep the present branch; the servo's elbow
    # swivel (_prop_swivel) remains the last guard.
    _log("prop branch",name,"no better hover branch",n,"candidates:",reasons[:3])
    return False


def _prop_swivel(motion, q_goal, tool_p, hand_R, width, body, q_cmd, max_step=.05):
    """Elbow self-motion for one servo step: same tool pose, joint-3 offset
    seeds; the solution with the largest arm self-distance that penetrates
    nothing and stays within max_step of the last command, or None."""
    best=None
    for k in range(1,7):
        for sign in (1.,-1.):
            seed=np.asarray(q_goal,float).copy();seed[2]+=sign*.015*k
            q=motion._ik_one(tool_p,hand_R,seed,width=width)
            if q is None or np.max(np.abs(q-q_cmd))>max_step:
                continue
            pen,own=_prop_contacts(motion,q,width,body)
            if pen:
                continue
            if best is None or own>best[1]:
                best=(q,own)
        if best is not None and best[1]>=PROP_SELF_SHELL:
            break
    return best


def _prop_report(cell, mate, name, target, stage):
    lateral,height,tilt=_prop_axis_error(cell,name,target)
    _log("prop insert",name,stage,"lateral_mm",round(1000*lateral,3),"height_mm",round(1000*height,3),
         "tilt_deg",round(tilt,3),"state",cell.coupling.state[mate["id"]],"t",round(float(cell.d.time),2))
    return lateral,height,tilt


def _prop_servo(cell, motion, mate, name, body, target, *, end_h, fast=.025, slow=.006,
                lateral_gate=.00015, max_s=12., wall_deadline=math.inf):
    """Closed-loop descent of a held propeller hub onto its rotor shaft.

    The 1.6 mm blade is pinched at a few pad points and slowly pivots about the
    jaw axis under its own weight (soft contact friction), ~1 mm/s at the hub.
    An open-loop line therefore drifts off a shaft that the bore clears by only
    0.43 mm. Every 50 ms this re-measures the live part, refreshes the held
    transform and commands a small, collision-checked IK step that puts the hub
    back on the shaft axis while descending. Above the shaft end the descent
    pauses until the hub is centred. Only arm actuator targets are written.
    """
    axis=target[:3,2]
    P=cell.pose(name)
    h_ref=float((P[:3,3]-target[:3,3])@axis)
    q_cmd=motion.q
    width=motion.width()
    # Seed every servo IK step from the straight-descent branch planned now
    # with the measured grasp (the branch transport planning verified for the
    # exit). Seeding only from the last command lets the 7-DOF null space
    # drift over a hundred steps, e.g. into a link2/link5 self-contact.
    def make_branch(q_from,h_from):
        R0,_=axis_aligned_rotation(cell.pose(name)[:3,:3],target[:3,:3])
        hs=np.linspace(h_from,end_h,max(2,int(math.ceil((h_from-end_h)/.002))+1))
        qs=[np.asarray(q_from,float).copy()]
        for hk in hs[1:]:
            bp,bR=motion.body_goal(target[:3,3]+axis*hk,R0,body)
            qk=motion._ik_one(bp,bR,qs[-1],width=width)
            if qk is None:
                raise PlanningError(f"{name} no straight descent branch at height {1000*hk:.1f} mm")
            qs.append(qk)
        return hs,np.asarray(qs)
    branch_h,branch_q=make_branch(q_cmd,h_ref)
    def branch_seed(hh):
        return np.array([np.interp(hh,branch_h[::-1],branch_q[::-1,j]) for j in range(7)])
    swivels=0
    t_end=float(cell.d.time)+max_s
    paused=0
    history=[]
    # Outer integral on the measured hub's lateral error. While the servo
    # follows the in-grasp pivot the arm reference keeps moving, which freezes
    # Motion._drive's own integral, and the joints' friction dead band then
    # leaves the hand ~0.8 mm short of every command. Frozen once the bore is
    # on the shaft (the shaft then holds the hub; never push it sideways).
    bias=np.zeros(3)
    while True:
        if float(cell.d.time)>t_end:
            _prop_report(cell,mate,name,target,"servo-timeout")
            raise PlanningError(f"{name} servo descent timed out")
        if time.monotonic()>wall_deadline:
            raise TimeoutError("reference wall cap during prop servo")
        if cell.coupling.state[mate["id"]] in ("LOCKED","LOCKED_OFF"):
            break  # fastened at rest on the bell: stop commanding the descent
        P=cell.pose(name)
        delta=P[:3,3]-target[:3,3]
        h=float(delta@axis)
        lateral=float(np.linalg.norm(delta-h*axis))
        history.append(h)
        # A hub that stops descending while commanded down is obstructed
        # (e.g. resting on the shaft end): stop instead of pushing.
        if len(history)>20 and history[-21]-h<.0001 and h_ref<h-.0003 and h>end_h+.0005:
            _prop_report(cell,mate,name,target,"blocked")
            raise PlanningError(f"{name} hub blocked {1000*(h-end_h):.1f} mm above its seat")
        if h_ref<=PROP_GATE_M or lateral<=lateral_gate:
            step=(fast if h_ref>PROP_ENTRY_M else slow)*.05
            # Never lead the measured hub by more than 1.5 mm.
            h_ref=max(end_h,h_ref-step,h-.0015)
        else:
            paused+=1
        if h>PROP_GATE_M:
            bias+=PROP_SERVO_KI*.05*(delta-h*axis)
            norm=float(np.linalg.norm(bias))
            if norm>PROP_SERVO_BIAS_MAX:
                bias*=PROP_SERVO_BIAS_MAX/norm
        desired_R,_=axis_aligned_rotation(P[:3,:3],target[:3,:3])
        tool_p,hand_R=motion.body_goal(target[:3,3]+axis*h_ref-bias,desired_R,body)
        # Bound each 50 ms correction (1.5 mm, 0.8 deg) for a smooth command.
        cur_p,cur_R=motion._fk(q_cmd,width)
        dp=tool_p-cur_p
        if np.linalg.norm(dp)>.0015:
            tool_p=cur_p+dp*(.0015/np.linalg.norm(dp))
        rv=Rotation.from_matrix(hand_R@cur_R.T).as_rotvec()
        if np.linalg.norm(rv)>math.radians(.8):
            hand_R=Rotation.from_rotvec(rv*(math.radians(.8)/np.linalg.norm(rv))).as_matrix()@cur_R
        seed=q_cmd+.3*(branch_seed(h_ref)-q_cmd)
        q_goal=motion._ik_one(tool_p,hand_R,seed,width=width)
        if q_goal is None:
            raise PlanningError(f"{name} servo IK failed at height {1000*h:.2f} mm")
        pen,own=_prop_contacts(motion,q_goal,width,body)
        if pen or own<PROP_SERVO_SELF_MIN:
            # The arm's own links (e.g. link5 towards link2) are closing in:
            # swivel the elbow for this step with the tool pose unchanged.
            alt=_prop_swivel(motion,q_goal,tool_p,hand_R,width,body,q_cmd)
            if alt is not None and (pen or alt[1]>own):
                q_goal,own=alt;pen=[]
                swivels+=1
                # Re-seed the remaining descent from the swivelled posture.
                branch_h,branch_q=make_branch(q_goal,max(end_h,h_ref))
        if pen:
            _prop_report(cell,mate,name,target,"servo-collision")
            raise PlanningError(f"{name} servo step in collision: {pen[:3]}")
        for k in range(1,6):
            motion._drive(q_cmd+(q_goal-q_cmd)*k/5);cell.tick()
        q_cmd=q_goal
        if h_ref<=end_h and h-end_h<.0003 and lateral<.0004:
            break
    _log("prop servo",name,"paused_cycles",paused,"elbow_swivels",swivels,"lateral_bias_mm",np.round(1000*bias,3))
    return q_cmd


def _prop_insert(cell, motion, mate, name, body, target, ins, pre_distance, wall_deadline=math.inf, open_width=.018):
    """Lower a held propeller's 13-sided bore over its passive rotor shaft.

    The bore (inradius 2.43 mm) clears the 2 mm shaft by only ~0.43 mm and
    neither has a chamfer, so a hub that is not centred on the axis when it
    reaches the shaft end lands on top of it (see _prop_servo).
    """
    _prop_report(cell,mate,name,target,"arrived")
    # Coarse correction of the transport pivot; _prop_servo centres the hub.
    try:
        motion.align_body(target[:3,3]-ins*pre_distance,target[:3,:3],body=body,
                          tolerance=.0015,angle=.03,max_passes=3,axis_only=True)
    except PlanningError as error:
        if "did not converge" not in str(error):
            raise
        _log("prop insert",name,"coarse hover alignment incomplete:",str(error))
    if not _gripped(cell,name):
        raise PlanningError(f"lost grip on {name} during hover alignment")
    if _prop_select_branch(cell,motion,mate,name,body,target,open_width):
        try:
            motion.align_body(target[:3,3]-ins*pre_distance,target[:3,:3],body=body,
                              tolerance=.0015,angle=.03,max_passes=3,axis_only=True)
        except PlanningError as error:
            if "did not converge" not in str(error):
                raise
            _log("prop insert",name,"coarse hover alignment incomplete:",str(error))
        if not _gripped(cell,name):
            raise PlanningError(f"lost grip on {name} during hover re-configuration")
    _prop_report(cell,mate,name,target,"hover")
    q=_prop_servo(cell,motion,mate,name,body,target,end_h=PROP_REST_M,wall_deadline=wall_deadline)
    _prop_report(cell,mate,name,target,"seated")
    return q


def _prop_seat(cell, motion, mate, name, body, q, released_width, wall_deadline=math.inf,
               hold_s=.8, settle_s=3.):
    """Wait for the at-rest fastening of a hub resting on its bell.

    If the pinch keeps the prop from being still against the passive rotor,
    open the jaws (aperture sweep checked) and let the prop rest on the shaft
    and bell under its own weight; the fastening then locks the resting pose.
    """
    key=mate["id"]
    def locked():
        return cell.coupling.state[key] in ("LOCKED","LOCKED_OFF")
    for _ in range(int(round(hold_s/.01))):
        if time.monotonic()>wall_deadline:
            raise TimeoutError("reference wall cap during seat hold")
        motion._drive(q);cell.tick()
        if locked():
            _log("prop seat",name,"locked while held")
            return
    motion.sync()
    current=motion.width()
    target_width=current
    for w in np.linspace(current,released_width,10)[1:]:
        if motion.collision(q,width=w,allow_pad_body=body,clearance=False):
            break
        target_width=w
    _log("prop seat",name,"release to",round(1000*target_width,2),"mm; state",cell.coupling.state[key])
    command=float(np.clip(100*Motion.angle_for_width(target_width)/.333,0,255))
    start=float(cell.d.ctrl[motion.ga])
    ramp=60
    for k in range(int(round(settle_s/.01))):
        if time.monotonic()>wall_deadline:
            raise TimeoutError("reference wall cap during seat release")
        if k<ramp:
            cell.d.ctrl[motion.ga]=start+(command-start)*(k+1)/ramp
        motion._drive(q);cell.tick()
        if locked() and k>=ramp:
            _log("prop seat",name,"locked after release at",round(float(cell.d.time),2))
            return


def _prop_open(motion, body, width):
    """Open the jaws beside an installed propeller only as far as the swept
    finger linkage stays clear of everything but the blade it just released."""
    motion.sync()
    current=motion.width()
    target=current
    for w in np.linspace(current,width,10)[1:]:
        if motion.collision(motion.q,width=w,allow_pad_body=body,clearance=False):
            break
        target=w
    motion.set_width(target)
    return target


def _prop_line_free(motion, q0, vector, width):
    """Dry-run of Motion.line's precheck from configuration q0 (no pad contact allowed)."""
    p0,r0=motion._fk(q0,width)
    p0,r0=p0.copy(),r0.copy()
    motion.collision(q0,width,None,False)
    initial={}
    for a,b,d in motion.last_contacts:
        key=tuple(sorted((a,b)));initial[key]=min(d,initial.get(key,math.inf))
    n=max(1,int(math.ceil(np.linalg.norm(vector)/.003)))
    prev=q0
    for u in np.linspace(0,1,n+1)[1:]:
        q=motion._ik_one(p0+u*np.asarray(vector),r0,prev,width=width)
        if q is None:
            return False
        steps=max(1,int(math.ceil(np.max(np.abs(q-prev))/.007)))
        for v in np.linspace(0,1,steps+1)[1:]:
            if motion.collision((1-v)*prev+v*q,width,None,False):
                for a,b,d in motion.last_contacts:
                    key=tuple(sorted((a,b)))
                    if key not in initial or d<initial[key]-5e-5:
                        return False
        prev=q
    return True


def _prop_exit(cell, motion, name, released_width):
    """Withdraw the open jaws from an installed propeller without touching it.

    Exits stay in the pad plane: back along the approach (tool -z) or off the
    blade along its span (tool +x, outward). The in-grasp pivot changes the
    hand orientation after transport planning verified its exit, and the seat
    posture can then leave link5 near link2. If every straight exit is blocked,
    swivel the elbow in small IK steps with the tool pose fixed (a self-motion
    of the 7-DOF arm) until one is clear. No pad contact with the prop allowed.
    """
    motion.sync()
    width=max(released_width,motion.width())
    pp,rr=motion.tool_pose()
    exits=[-rr[:,2]*d for d in (.08,.06,.04,.03)]+[rr[:,0]*d for d in (.05,.045)]
    exits.append((rr[:,0]-rr[:,2])*(.06/math.sqrt(2)))
    q0=motion.q
    postures=[[q0]]
    for sign in (-1.,1.):
        path=[q0];q=q0.copy()
        for _ in range(12):
            seed=q.copy();seed[2]+=sign*.05
            qk=motion._ik_one(pp,rr,seed,width=width)
            if qk is None or abs(qk[2]-q[2])<.002:
                break
            if not motion.segment_free(q,qk,width=width,allow_pad_body=None,clearance=False):
                break
            path.append(qk);q=qk
            postures.append(list(path))
    for path in postures:
        for vector in exits:
            if not _prop_line_free(motion,path[-1],vector,width):
                continue
            if len(path)>1:
                _log("retreat",name,"elbow self-motion dq3",round(float(path[-1][2]-q0[2]),3))
                motion.execute(path,speed=.1)
            p_now,_=motion.tool_pose()
            motion.line(p_now+vector,rr,width=width,allow_pad_body=None,clearance=False,speed=.03)
            _log("retreat",name,"exit_mm",np.round(1000*(rr.T@vector),1),"t",round(float(cell.d.time),2))
            return
    raise PlanningError(f"cannot retract from {name}: no clear exit after elbow self-motion")


def _prepare_width(motion, width):
    """Open only where the complete linkage's swept aperture stays clear."""
    current=motion.width()
    def clear():
        motion.sync()
        return all(not motion.collision(motion.q,width=w,clearance=True)
                   for w in np.linspace(current,width,12))
    if clear():
        return
    p,R=motion.tool_pose()
    errors=[]
    for height in (.08,.14):
        try:
            motion.move_pose(p+np.array([0.,0.,height]),R,width=current,speed=.25)
            if clear():
                return
        except PlanningError as error:
            errors.append(str(error))
    motion.move_pose(np.array([-.10,0.,.50]),DOWN,width=current,attempts=48,speed=.25)
    if not clear():
        raise PlanningError(f"no clear aperture sweep for {width}: {errors[:2]}")


# ----------------------------------------------------------------------------
# Landing struts (kind landing_strut). Part-specific; nothing here runs for
# other kinds.
#
# * Creep: MuJoCo's soft friction lets a pinched part slip at about
#   (1-d)/d*g/b ~ 1 mm/s along any pad-tangent direction that carries gravity,
#   whatever the grip force. A strut hanging from a downward-pointing hand walks
#   out of the fingertips, so it is lifted and turned upright right after the
#   pick (slow, smooth Cartesian moves: fast turns make the arm sag into the
#   staged guard). Upright, the shaft slides a little until the flange fillet
#   rests on the top edges of both pads, a support that does not creep.
# * Rattle: the shaft collider is a 9-sided prism, so one pad always meets a
#   facet and the other a vertex edge. MuJoCo reports that edge as a single
#   contact jumping between the pad ends every step: a hanging strut rocks at
#   ~50 Hz (60-120 deg/s) and the at-rest coupling never sees it still. At the
#   seat the hand therefore drives 1 mm past flush: the pad tops push the fillet
#   up and clamp the flange face flat against the plate, which stops the rocking.
# * Twist: the stable-top pick fixes the strut's twist in the hand, so with the
#   4-fold mount only four seat approach directions exist. Where none of them is
#   reachable (leg_4's corner) the strut is stood up on the bench and grasped
#   again from a direction that maps onto a reachable approach.
STRUT_PICK_STATION = .045       # shaft station (local z) of the pad centre at the pick
STRUT_PAD_HALF = .015           # half length of a stock pad along tool x
STRUT_LIFT = .075               # lift before turning upright: foot swing and hand over the staged guard
STRUT_HOVER = .020              # hover below the seat before the vertical insertion
STRUT_PRESS = .0010             # drive the seat this far past flush: the pad tops press the
                                # flange fillet up, clamping the flange flat against the plate
STRUT_HOLD_SQUEEZE = .0007      # lighter jaw command once the strut is upright
STRUT_OUTSIDE = .08             # horizontal stand-off of the hover outside the plate footprint


STRUT_FILLET = .0609            # local z where the flange fillet (cone) starts above the shaft
STRUT_SEATED_STATION = STRUT_FILLET-STRUT_PAD_HALF   # pad centre once the fillet rests on the pad tops


def _strut_seated(g):
    """The same physical grasp once the upright shaft has slid down onto the pad tops."""
    return Grasp(g.p+np.array([0., 0., STRUT_SEATED_STATION-STRUT_PICK_STATION]), g.R, g.width,
                 g.label+"-seated", g.target_yaw, g.squeeze, g.settle_s)


def _strut_in_hand(motion, body):
    """Measured strut origin (m) and shaft axis in the hand frame."""
    b = motion.m.body(body).id
    Rh = motion.d.xmat[motion.hand].reshape(3, 3)
    p = Rh.T @ (motion.d.xpos[b]-motion.d.xpos[motion.hand])
    axis = Rh.T @ motion.d.xmat[b].reshape(3, 3)[:, 2]
    return p, axis


def _strut_turn_pose(part_R, p, R, g):
    """Hand pose that turns the lifted strut upright about its grasped shaft section."""
    tcp = p+np.array([0., 0., STRUT_LIFT])
    upright, angle = axis_aligned_rotation(part_R, np.eye(3))
    turn = upright @ part_R.T
    centre = tcp+R[:, 2]*(_tip(g.width)-.158-.0095)
    return centre+turn @ (tcp-centre), turn @ R, angle


STRUT_RRT_TIMEOUT = 30.         # RRT-Connect nominal budget for strut moves (x PLAN_EXTENDS_PER_S extends; base default 12)


STRUT_EMPTY_SPEED = .5          # peak joint speed of the empty-hand strut approaches (rad/s; carries .25)
MOTOR_EXTRA_HOLD_TICKS = 120    # a motor still HOLDING after place()'s fixed 80-tick hold may hold this much longer


def _strut_goto(motion, targets, *, width, allow_pad_body=None, what="strut move"):
    """Plan (RRT-Connect, larger budget) and execute an EMPTY-HAND move to the first reachable target.

    The base planner's default budget is short for these moves; every candidate
    branch is tried with a larger budget before giving up. Only empty-hand moves
    (pick and regrasp approaches) come here; they run at STRUT_EMPTY_SPEED: the
    arm then tracks its collision-checked path within 0.3 mm at every link
    (tools/probe_speed_v7.py, leg_4 approach: 0.16 mm at .25, 0.30 mm at .5;
    planning clearance 4 mm), and a joint-space approach that unwinds the wrist
    takes half the time (63 -> 32 s).
    """
    errors = []
    for q in targets:
        try:
            path = motion.plan(q, width=width, allow_pad_body=allow_pad_body, timeout=STRUT_RRT_TIMEOUT)
        except PlanningError as error:
            errors.append(str(error)[:120])
            continue
        motion.execute(path, speed=STRUT_EMPTY_SPEED)
        return
    raise PlanningError(f"{what}: no path to any of {len(targets)} branches: {errors[:3]}")


def _strut_approach_branches(motion, pre_p, goal_p, R, body=None, *, clearance=False, attempts=64, seed=None):
    """IK branches at pre_p whose straight approach to goal_p is clear (as in _preinsert).

    With ``seed`` (a planned branch) the branches nearest to it come first.
    """
    out = []
    for candidate in motion.ik(pre_p, R, width=motion.width(), attempts=attempts, max_solutions=16,
                               allow_pad_body=body, clearance=clearance, seed=seed):
        prev, ok = candidate, True
        for u in np.linspace(0, 1, 34)[1:]:
            q = motion._ik_one((1-u)*np.asarray(pre_p)+u*np.asarray(goal_p), R, prev, width=motion.width())
            if q is None or not motion.segment_free(prev, q, width=motion.width(), allow_pad_body=body,
                                                    clearance=clearance):
                ok = False
                break
            prev = q
        if ok:
            out.append(candidate)
    if seed is not None:
        out.sort(key=lambda q: float(np.linalg.norm(np.asarray(q)-seed)))
    return out


def _strut_pick_branches(cell, motion, name, g, p, R, width, *, prefer_margin=False, avoid_wound=False):
    """IK branches above the pick whose descent, lift and upright turn are all clear.

    The branch the arm arrives on decides where its elbow ends up after the
    turn (a fresh run comes from the ESC and met the staged guard with link4).
    The empty-hand descent, then lift and turn at the closed jaw, are dry-run
    from each candidate branch before the arm moves. With ``prefer_margin`` the
    branches are ordered by the joint-limit margin of the arm after the turn
    (a strut that is stood up and regrasped next must be carried with a wrist
    that is not wound up near its limit: final05's leg_4 arrived with joint 2
    at 1.82 and joint 7 at -5.18 rad and no upright stand carry existed). With
    ``avoid_wound`` (direct struts) the IK order is kept except that branches
    whose +-2 pi joints end the turn within STRUT_WIND_MARGIN of a limit go
    last (in one run leg_2 arrived with joint 7 at -6.283 and no carry
    existed for any yaw). The full margin order would cost ~35 s of pick
    approach per strut (v1/v2 runs: leg_1 76 s instead of 40 s), and the
    tilt-constrained carry RRT copes with the remaining postures.
    """
    turn_p, turn_R, _ = _strut_turn_pose(cell.pose(name)[:3, :3], p, R, g)
    branches, margins, windings, errors = [], [], [], []
    for q in motion.ik(p-R[:, 2]*.10, R, width=width, attempts=48, max_solutions=12):
        try:
            qd = motion.line(p, R, width=width, clearance=False, start_q=q, execute=False)[-1]
            ql = motion.line(p+np.array([0., 0., STRUT_LIFT]), R, width=g.width, clearance=False,
                             start_q=qd, execute=False)[-1]
            qt = motion.line(turn_p, turn_R, width=g.width, clearance=True, start_q=ql, execute=False)[-1]
        except PlanningError as error:
            errors.append(str(error)[:120])
            continue
        branches.append(q)
        margins.append(_joint_margin(motion, qt))
        windings.append(_joint_winding_margin(motion, qt))
    if not branches:
        raise PlanningError(f"{name}: no pick branch with a clear lift and upright turn: {errors[:2]}")
    if prefer_margin:
        order = sorted(range(len(branches)), key=lambda i: -margins[i])
        _log("pick branches", name, "post-turn joint margins", [round(margins[i], 3) for i in order])
        branches = [branches[i] for i in order]
    elif avoid_wound:
        order = sorted(range(len(branches)), key=lambda i: (windings[i] < STRUT_WIND_MARGIN,
                                                            -windings[i] if windings[i] < STRUT_WIND_MARGIN else i))
        _log("pick branches", name, "post-turn winding margins", [round(windings[i], 3) for i in order])
        branches = [branches[i] for i in order]
    return branches


STRUT_WIND_MARGIN = .02         # +-2 pi joints after the upright turn: keep >= 2 % of their range (.25 rad) to a limit


def _joint_winding_margin(motion, q):
    """Smallest normalised distance of the +-2 pi joints (1, 3, 5, 7) to their limits."""
    idx = [0, 2, 4, 6]
    lo, hi = motion.limits[idx, 0], motion.limits[idx, 1]
    qq = np.asarray(q, float)[idx]
    return float(np.min(np.minimum(qq-lo, hi-qq)/(hi-lo)))


def _joint_margin(motion, q):
    """Smallest normalised distance of q to its joint limits (0 at a limit, .5 centred)."""
    lo, hi = motion.limits[:, 0], motion.limits[:, 1]
    return float(np.min(np.minimum(q-lo, hi-q)/(hi-lo)))


def _strut_pick_upright(cell, motion, name, body, g, p, R, width, *, prefer_margin=False, avoid_wound=False):
    """Pick the resting strut from above, lift it and turn it upright, flange up."""
    _prepare_width(motion, width)
    motion.set_width(width)
    motion.wait(1.5)
    width = motion.width()
    _log("preshape", name, "measured_gap", round(width, 4))
    _strut_goto(motion, _strut_pick_branches(cell, motion, name, g, p, R, width, prefer_margin=prefer_margin,
                                             avoid_wound=avoid_wound),
                width=width, what=f"{name} pick")
    motion.line(p, R, width=width, clearance=False, speed=.025)
    motion.close(g.width, squeeze=g.squeeze)
    motion.wait(max(g.settle_s, .5))
    _log("clamp", name, "gap", round(motion.width(), 4), "forces", _grip_forces(cell, name))
    if not _gripped(cell, name):
        raise PlanningError(f"two-sided pad contact missing for {name}")
    motion.hold_body(body)
    t_pick = float(cell.d.time)
    # Lift quickly and turn without an intermediate hold: while the hand points
    # down, gravity drags the pinched shaft towards the fingertips.
    motion.line(p+np.array([0., 0., STRUT_LIFT]), R, allow_pad_body=body, clearance=False, speed=.05,
                hold=10, smooth=True)
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} after lift")
    # Turn the strut upright about its grasped shaft section, flange up, using
    # the measured part axis and the commanded lift pose.
    turn_p, turn_R, angle = _strut_turn_pose(cell.pose(name)[:3, :3], p, R, g)
    motion.line(turn_p, turn_R, allow_pad_body=body, clearance=True, speed=.05, angular_speed=.5, smooth=True)
    # A full 4 mm squeeze makes the faceted shaft rattle and walk towards the
    # fingertips; a lighter command keeps several N per pad.
    motion.set_width(g.width, seconds=.3, squeeze=STRUT_HOLD_SQUEEZE)
    pin, axis = _strut_in_hand(motion, body)
    _log("upright", name, "turn_deg", round(math.degrees(angle), 1), "horizontal_s",
         round(float(cell.d.time)-t_pick, 2), "in_hand_mm", np.round(pin*1000, 2),
         "axis_hand", np.round(axis, 3))
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} while turning upright")


STRUT_REGRASP_STATION = .0445   # pad centre station of a side regrasp (pad tops 1.4 mm under the fillet)
STRUT_REGRASP_DEPTH = 0.        # shaft axis at the TCP: 15 mm of pad remain towards the tip
STRUT_REGRASP_STANDOFF = .06    # straight horizontal approach of the side regrasp
STRUT_STAND_EXIT = (.06, .04, .025)   # open-hand exits after standing a strut (fingertips clear at 20 mm)
STRUT_REGRASP_LIFT_CHECK = .05  # the planned regrasp posture must be able to lift this far with clearance


def _strut_side_hand(point, theta, station, depth=STRUT_REGRASP_DEPTH):
    """Horizontal hand (tool x up) holding an upright shaft from world direction theta."""
    z = np.array([math.cos(theta), math.sin(theta), 0.])
    x = np.array([0., 0., 1.])
    R = np.column_stack([x, np.cross(z, x), z])
    return np.array([point[0], point[1], point[2]+station])-z*depth, R


def _strut_yaw(R):
    return math.atan2(R[1, 0], R[0, 0])


class _Detached:
    """Plan hypothetical grasps with no held body attached in the planning copy."""
    def __init__(self, motion):
        self.motion = motion

    def __enter__(self):
        self.held, self.motion.held = self.motion.held, None

    def __exit__(self, *exc):
        self.motion.held = self.held


def _strut_mount_angles(cell, motion, mate, body, width):
    """World approach angles with a collision-free upright seat and hover branch."""
    target = cell.coupling.target(mate)
    ins = np.asarray(mate["insertion_axis"], float)
    angles = []
    with _Detached(motion):
        for k in range(24):
            theta = 2*math.pi*k/24
            p, R = _strut_side_hand(target[:3, 3], theta, STRUT_SEATED_STATION)
            if (motion.ik(p, R, width=width, attempts=12, max_solutions=1, allow_pad_body=body) and
                    motion.ik(p-ins*STRUT_HOVER, R, width=width, attempts=12, max_solutions=1,
                              allow_pad_body=body)):
                angles.append(theta)
    return angles


def _strut_stand_spots(cell, name, clearance=.05):
    """Free bench spots near the fixture for standing a strut upright.

    Candidates lie on a 40 mm grid in front of the fixture; a spot is free when
    its 25 mm foot circle keeps `clearance` from every other part's resting
    bounding box and from the fixture pedestal.
    """
    others = []
    for n, inst in cell.graph["instances"].items():
        if n == name or n in cell.coupling.assigned.values():
            continue
        part = cell.kit["parts"][inst["kind"]]
        T = cell.pose(n)
        bb = np.asarray(part["bounds_m"], float)
        corners = np.array([[x, y, z] for x in bb[:, 0] for y in bb[:, 1] for z in bb[:, 2]]) @ T[:3, :3].T+T[:3, 3]
        if corners[:, 2].min() > .12:
            continue   # parts already on the raised assembly
        others.append((corners[:, :2].min(0), corners[:, :2].max(0)))
    fixture = np.asarray(cell.graph["fixture_pos_m"][:2], float)
    others.append((fixture-.025, fixture+.025))
    # Keep the clamped chassis' footprint free too: the strut must leave the
    # stand spot straight upwards for its upright carry route.
    plate = np.asarray(cell.kit["parts"]["base"]["bounds_m"], float)[:, :2]+cell.pose("base")[:2, 3]
    others.append((plate[0]+.02, plate[1]-.02))
    here = cell.pose(name)[:2, 3]
    spots = []
    for x in np.arange(-.26, .07, .04):
        for y in np.arange(-.18, .181, .04):
            xy = np.array([x, y])
            if all(np.any(xy+.0125+clearance < lo) or np.any(xy-.0125-clearance > hi) for lo, hi in others):
                spots.append(np.array([x, y, 0.]))
    spots.sort(key=lambda s: float(np.linalg.norm(s[:2]-here)))
    return spots


def _lowest_point(cell, body):
    """Lowest world z of a body's collision meshes (live pose)."""
    m, d = cell.m, cell.d
    b = m.body(body).id
    low = math.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
        z = v @ d.geom_xmat[g].reshape(3, 3)[2]+d.geom_xpos[g][2]
        low = min(low, float(z.min()))
    return low


def _strut_regrasp(cell, motion, mate, name, body, g):
    """Stand the upright strut on its foot and take a side grasp at a chosen twist.

    The stable-top pick fixes the shaft's twist in the hand, and the 4-fold
    mount then allows only four approach directions at the seat. When none is
    reachable, the strut is set down upright on the bench and grasped again from
    a direction that maps onto a reachable seat approach.
    """
    open_width = min(.085, g.width+.012)
    mounts = _strut_mount_angles(cell, motion, mate, body, g.width)
    _log("regrasp mount angles", name, [round(math.degrees(a)) for a in mounts])
    if not mounts:
        raise PlanningError(f"{name}: no upright seat approach at any twist")
    # Prefer the middle of the widest reachable window (15 deg scan steps).
    step = 2*math.pi/24
    index = {int(round(a/step)) % 24 for a in mounts}
    def depth(theta):
        i, d = int(round(theta/step)) % 24, 0
        while d < 12 and (i+d+1) % 24 in index and (i-d-1) % 24 in index:
            d += 1
        return d
    mounts.sort(key=lambda a: -depth(a))
    leg = cell.pose(name)
    psi0 = _strut_yaw(leg[:3, :3])
    plan = None
    spots = _strut_stand_spots(cell, name)[:10]
    free_at = {}
    # First pass: the regrasp posture must leave room to lift the strut
    # STRUT_REGRASP_LIFT_CHECK with clearance (in one run a posture folded
    # near the robot base, link1/link_base-link6 inside the 8 mm self-clearance,
    # refused every lift and the carry). Second pass: the old criterion.
    for strict in (True, False):
        for si, spot in enumerate(spots):
            if si not in free_at:
                # Side-regrasp directions that reach the standing strut at this spot.
                with _Detached(motion):
                    free = set()
                    for d in range(0, 360, 15):
                        p2, R2 = _strut_side_hand(spot, math.radians(d), STRUT_REGRASP_STATION)
                        if (motion.ik(p2, R2, width=open_width, attempts=8, max_solutions=1) and
                                motion.ik(p2-R2[:, 2]*STRUT_REGRASP_STANDOFF, R2, width=open_width,
                                          attempts=8, max_solutions=1)):
                            free.add(d)
                free_at[si] = free
                _log("regrasp spot", name, np.round(spot[:2], 3), "side directions", sorted(free))
            free = free_at[si]
            if not free:
                continue
            for turn_deg in (0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90):
                psi = psi0+math.radians(turn_deg)
                options = []
                for theta3 in mounts:
                    for k in range(4):
                        theta2 = math.degrees(theta3+psi-k*math.pi/2) % 360
                        if int(round(theta2/15)) % 24*15 in free:
                            options.append((theta3, k, theta2))
                if not options:
                    continue
                # Stand it exactly upright (the hand absorbs any in-hand pivot).
                stand_R = Rotation.from_euler("z", psi).as_matrix()
                place_p, place_R = motion.body_goal(spot+np.array([0., 0., .003]), stand_R, body)
                q_place = motion.ik(place_p, place_R, width=motion.width(), attempts=12, max_solutions=1,
                                    allow_pad_body=body)
                if not q_place:
                    continue
                try:
                    # Final 3 mm descent with the strut, then the open hand's exit.
                    q_low = motion.line(place_p-np.array([0., 0., .0028]), place_R, allow_pad_body=body,
                                        clearance=False, start_q=q_place[0], execute=False)[-1]
                    with _Detached(motion):
                        motion.line(place_p-np.array([0., 0., .0028])-place_R[:, 2]*STRUT_STAND_EXIT[-1],
                                    place_R, width=open_width, start_q=q_low, execute=False)
                except PlanningError:
                    continue
                with _Detached(motion):
                    for theta3, k, theta2 in options:
                        p2, R2 = _strut_side_hand(spot, math.radians(theta2), STRUT_REGRASP_STATION)
                        for q_pre in motion.ik(p2-R2[:, 2]*STRUT_REGRASP_STANDOFF, R2, width=open_width,
                                               attempts=12, max_solutions=4 if strict else 1):
                            try:
                                q_reg = motion.line(p2, R2, width=open_width, clearance=False, start_q=q_pre,
                                                    execute=False)[-1]
                                if strict:
                                    motion.line(p2+np.array([0., 0., STRUT_REGRASP_LIFT_CHECK]), R2, width=g.width,
                                                clearance=True, start_q=q_reg, execute=False)
                            except PlanningError:
                                continue
                            plan = (spot, turn_deg, stand_R, theta3, k, q_pre)
                            break
                        if plan:
                            break
                if plan:
                    break
            if plan:
                break
        if plan:
            break
        _log("regrasp plan", name, "no posture with a clear lift; relaxed search")
    if plan is None:
        raise PlanningError(f"{name}: no stand spot with a side regrasp onto a reachable seat")
    spot, turn_deg, stand_R, theta3, k, q_regrasp = plan
    _log("regrasp plan", name, "spot", np.round(spot, 3), "turn_deg", turn_deg, "seat_deg",
         round(math.degrees(theta3)), "yaw_k", k, "lift_checked", strict)
    # Set the strut down upright, foot first, with the planned twist.
    touch_p, touch_R = motion.body_goal(spot+np.array([0., 0., .003]), stand_R, body)
    _strut_stand_carry(motion, name, body, spot, stand_R, touch_p, touch_R)
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} during the stand carry")
    motion.line(touch_p, touch_R, allow_pad_body=body, clearance=False, speed=.02)
    # Lower the last few mm until the lowest collider point is 0.1 mm above the
    # bench; letting go then leaves the strut standing on its foot. Re-measure
    # the grasp first so the planning copy sees the strut where it really is.
    motion.hold_body(body)
    tcp, Rh = motion.tool_pose()
    motion.line(tcp-np.array([0., 0., _lowest_point(cell, body)-.0001]), Rh, allow_pad_body=body,
                clearance=False, speed=.004)
    motion.hold_body(None)
    _open_clear(motion, body, open_width, seconds=1.)
    motion.wait(.5)
    tcp, Rh = motion.tool_pose()
    errors = []
    for distance in STRUT_STAND_EXIT:
        try:
            motion.line(tcp-Rh[:, 2]*distance, Rh, allow_pad_body=body, clearance=True, speed=.03)
            break
        except PlanningError as error:
            errors.append(str(error)[:120])
    else:
        raise PlanningError(f"cannot leave the standing {name}: {errors}")
    leg = cell.pose(name)
    tilt = math.degrees(math.acos(float(np.clip(leg[2, 2], -1, 1))))
    _log("stood", name, "pos", np.round(leg[:3, 3], 4), "tilt_deg", round(tilt, 2),
         "twist_deg", round(math.degrees(_strut_yaw(leg[:3, :3])), 2))
    if tilt > 3.:
        raise PlanningError(f"{name} did not stay upright on the bench ({tilt:.1f} deg)")
    # Side regrasp, corrected for the measured standing pose.
    psi = _strut_yaw(leg[:3, :3])
    theta2 = theta3+psi-k*math.pi/2
    p2, R2 = _strut_side_hand(leg[:3, 3], theta2, STRUT_REGRASP_STATION)
    _strut_goto(motion, _strut_approach_branches(motion, p2-R2[:, 2]*STRUT_REGRASP_STANDOFF, p2, R2,
                                                 seed=q_regrasp),
                width=motion.width(), what=f"{name} regrasp approach")
    motion.line(p2, R2, width=motion.width(), clearance=False, speed=.025)
    motion.close(g.width, squeeze=g.squeeze)
    motion.wait(.5)
    _log("regrasp clamp", name, "deg", round(math.degrees(theta2), 1), "gap", round(motion.width(), 4),
         "forces", _grip_forces(cell, name))
    if not _gripped(cell, name):
        raise PlanningError(f"two-sided pad contact missing for {name} after the regrasp")
    motion.hold_body(body)
    # Lift it clear of the bench before the carry is planned: the low regrasp
    # pose among the staged parts leaves the planner little room (final04).
    for rise in (.10, .05, .025):
        try:
            motion.line(p2+np.array([0., 0., rise]), R2, allow_pad_body=body, clearance=rise > .03,
                        speed=.03, smooth=rise > .03)
            break
        except PlanningError as error:
            _log("regrasp lift shortened", name, rise, str(error)[:80])
    motion.set_width(g.width, seconds=.3, squeeze=STRUT_HOLD_SQUEEZE)
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} after the regrasp lift")
    leg = cell.pose(name)
    tcp, Rt = motion.tool_pose()
    G = np.linalg.inv(leg) @ np.r_[np.c_[Rt, tcp], [[0, 0, 0, 1]]]
    return Grasp(G[:3, 3], G[:3, :3], g.width, f"side-regrasp-{math.degrees(theta2):.0f}",
                 target_yaw=k*math.pi/2, squeeze=g.squeeze, settle_s=.5)


STRUT_TILT_OK = 25.             # accept a carry path whose strut stays this close to upright (deg)


def _strut_path_tilt(motion, path, width):
    """Largest strut tilt from vertical along a joint path (planning copy)."""
    b = motion.held[0]
    worst = 0.
    for a, c in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.max(np.abs(np.asarray(c)-a))/.03)))
        for t in np.linspace(0, 1, n+1):
            motion._fk((1-t)*np.asarray(a)+t*np.asarray(c), width)
            z = motion.pd.xmat[b].reshape(3, 3)[2, 2]
            worst = max(worst, math.degrees(math.acos(float(np.clip(z, -1, 1)))))
    return worst


STRUT_RRT_ITERATIONS = 150      # deterministic search budget of the upright carry RRT
STRUT_RRT_WALL_CAP = 900.       # wall-clock hang guard only (the iteration budget ends the search)


def _strut_upright_rrt(motion, body, width, goal_p, goal_yaw, *, iterations=STRUT_RRT_ITERATIONS):
    """Task-space RRT over the held strut's position and yaw, strut kept upright.

    Nodes are IK solutions (measured grasp) for an upright strut; each edge is a
    short joint segment that is collision-checked and whose strut tilt stays
    within STRUT_TILT_OK. Deterministic: samples come from motion.rng.
    """
    b, _adr, p_rel, rel = motion.held
    rng = motion.rng
    def solve(leg_p, yaw, seed):
        hand_R = Rotation.from_euler("z", yaw).as_matrix() @ rel.T
        q = motion._ik_one(leg_p+hand_R @ (motion.offset-p_rel), hand_R, seed, width=width)
        if q is None or np.max(np.abs(q-seed)) > .5 or motion.collision(q, width, body, True):
            return None
        return q
    def edge(qa, qb, resolution=.015):
        return (motion.segment_free(qa, qb, width=width, allow_pad_body=body, clearance=True,
                                    resolution=resolution) and
                _strut_path_tilt(motion, [qa, qb], width) <= STRUT_TILT_OK)
    q0 = motion.q
    motion._fk(q0, width)
    P = [motion.pd.xpos[b].copy()]
    Y = [_strut_yaw(motion.pd.xmat[b].reshape(3, 3))]
    Q, parent = [q0], [-1]
    lo, hi = np.array([-.30, -.45, .05]), np.array([.32, .45, .40])
    started = time.monotonic()
    for _iteration in range(iterations):
        if time.monotonic()-started > STRUT_RRT_WALL_CAP:
            break
        if rng.random() < .3:
            sp, sy = np.asarray(goal_p, float), goal_yaw
        else:
            sp, sy = rng.uniform(lo, hi), rng.uniform(-math.pi, math.pi)
        dp = sp-np.asarray(P)
        dy = np.array([math.remainder(sy-y, 2*math.pi) for y in Y])
        i = int(np.argmin(np.linalg.norm(dp, axis=1)+.05*np.abs(dy)))
        # RRT-Connect style: keep stepping towards the sample until blocked.
        reached = False
        for _ in range(40):
            dpi = sp-P[i]
            dyi = math.remainder(sy-Y[i], 2*math.pi)
            length = float(np.linalg.norm(dpi))
            if length < 1e-4 and abs(dyi) < 1e-3:
                break
            f = min(1., .03/(length+1e-9), .25/(abs(dyi)+1e-9))
            cp, cy = P[i]+f*dpi, Y[i]+f*dyi
            q = solve(cp, cy, Q[i])
            if q is None or not edge(Q[i], q):
                break
            P.append(cp); Y.append(cy); Q.append(q); parent.append(i)
            i = len(Q)-1
            if np.linalg.norm(cp-goal_p)+.05*abs(math.remainder(cy-goal_yaw, 2*math.pi)) < .03:
                reached = True
                break
        if reached:
            cp, cy, q = P[-1], Y[-1], Q[-1]
            qg = solve(np.asarray(goal_p, float), goal_yaw, q)
            if qg is None or not edge(q, qg):
                continue
            path, k = [qg], len(Q)-1
            while k >= 0:
                path.append(Q[k]); k = parent[k]
            path.reverse()
            out, k = [path[0]], 0
            while k < len(path)-1:
                j = len(path)-1
                while j > k+1 and not edge(path[k], path[j], .007):
                    j -= 1
                if j == k+1 and not edge(path[k], path[j], .007):
                    raise PlanningError("upright carry path failed its fine check")
                out.append(path[j]); k = j
            return out
    raise PlanningError(f"upright carry RRT found no path ({len(Q)} nodes, {time.monotonic()-started:.0f} s)")


STRUT_TILT_RRT_EXTENDS = 1500   # deterministic extend budget of the tilt-constrained joint-space RRT
STRUT_TILT_RRT_STEP = .18       # joint-space step (rad, as Motion.plan)
STRUT_TILT_SHORTCUT = 15.       # shortcuts of a tilt-RRT path keep the strut this upright (deg)


def _strut_tilt_rrt(motion, body, width, goals, *, extends=STRUT_TILT_RRT_EXTENDS):
    """Joint-space RRT-Connect whose every edge keeps the held strut within STRUT_TILT_OK.

    The upright task-space RRT explores only the strut's position and yaw: the
    arm's elbow swivel and branch follow the start through the IK seeds, so a
    carry that needs another arm posture is never found (two runs:
    leg_2 found nothing for any yaw). Here the trees grow in joint space from
    the start and from every goal branch; each new node is projected back onto
    the upright set (IK seeded from the stepped configuration, strut position
    and yaw kept, tilt zero) and each edge is collision-checked with clearance
    and tilt-checked. The connect phase stops as soon as a step brings the
    other tree no closer (the projection can hold it in place, which otherwise
    spends the whole budget on one target). Deterministic: motion.rng only and
    a fixed extend budget (the wall clock is only a hang guard).
    """
    b, _adr, p_rel, rel = motion.held
    rng = motion.rng
    q0 = motion.q
    goals = [np.asarray(q, float) for q in goals]
    if not goals:
        raise PlanningError("tilt RRT: no goal branch")
    lo = np.maximum(motion.limits[:, 0], np.minimum(q0, np.min(goals, 0))-1.5)
    hi = np.minimum(motion.limits[:, 1], np.maximum(q0, np.max(goals, 0))+1.5)
    step = STRUT_TILT_RRT_STEP
    def project(q):
        motion._fk(q, width)
        leg_p = motion.pd.xpos[b].copy()
        hand_R = Rotation.from_euler("z", _strut_yaw(motion.pd.xmat[b].reshape(3, 3))).as_matrix() @ rel.T
        return motion._ik_one(leg_p+hand_R @ (motion.offset-p_rel), hand_R, q, width=width, max_iter=40)
    def edge(qa, qb, limit=STRUT_TILT_OK):
        return (motion.segment_free(qa, qb, width=width, allow_pad_body=body, clearance=True) and
                _strut_path_tilt(motion, [qa, qb], width) <= limit)
    trees = [([q0], [-1]), (list(goals), [-1]*len(goals))]
    spent, swapped, started = 0, False, time.monotonic()
    def extend(tree, target):
        nonlocal spent
        spent += 1
        nodes, parents = tree
        i = int(np.argmin([np.linalg.norm(n-target) for n in nodes]))
        delta = target-nodes[i]
        distance = float(np.linalg.norm(delta))
        if distance <= step:
            q = np.asarray(target, float)
        else:
            q = project(nodes[i]+delta*(step/distance))
            if q is None or np.linalg.norm(q-nodes[i]) > 2*step:
                return None, False
        if not edge(nodes[i], q):
            return None, False
        nodes.append(q); parents.append(i)
        return len(nodes)-1, distance <= step
    def trace(tree, index):
        out = []
        while index >= 0:
            out.append(tree[0][index]); index = tree[1][index]
        return out[::-1]
    while spent < extends and time.monotonic()-started < STRUT_RRT_WALL_CAP:
        a, other = trees
        if rng.random() < .2:
            sample = other[0][-1]
        else:
            sample = rng.uniform(lo, hi)
            projected = project(sample)
            if projected is not None:
                sample = projected
        ia, _ = extend(a, sample)
        if ia is not None:
            closest = math.inf
            while spent < extends:
                ib, reached = extend(other, a[0][ia])
                if ib is None:
                    break
                gap = float(np.linalg.norm(other[0][ib]-a[0][ia]))
                if not reached and gap > closest-.01:
                    break
                closest = gap
                if reached:
                    path = trace(a, ia)+trace(other, ib)[-2::-1]
                    if swapped:
                        path.reverse()
                    out, i = [path[0]], 0
                    while i < len(path)-1:
                        j = len(path)-1
                        while j > i+1 and not edge(path[i], path[j], STRUT_TILT_SHORTCUT):
                            j -= 1
                        out.append(path[j]); i = j
                    return out
        trees.reverse(); swapped = not swapped
    raise PlanningError(f"tilt RRT found no path ({len(trees[0][0])+len(trees[1][0])} nodes, "
                        f"{spent} extends, {time.monotonic()-started:.0f} s)")


def _strut_stand_carry(motion, name, body, spot, stand_R, touch_p, touch_R):
    """Carry the upright strut to 25 mm above its stand spot without tipping it.

    Upright task-space RRT first, once more after an 80 mm lift, then the
    tilt-constrained joint-space RRT; the plain joint-space RRT fallback only
    takes a path whose strut tilt stays within STRUT_TILT_OK (final05: an
    unchecked fallback path tipped the strut to ~80 deg and it slid out of the
    pads).
    """
    goal = spot+np.array([0., 0., .025])
    width = motion.width()
    errors = []
    for attempt in range(2):
        if attempt and not _strut_rise(motion, body, name, (.08, .05)):
            break
        try:
            path = _strut_upright_rrt(motion, body, width, goal, _strut_yaw(stand_R))
        except PlanningError as error:
            errors.append(str(error)[:120])
            _log("stand carry rejected", name, "attempt", attempt, errors[-1])
            continue
        motion.execute(path, speed=.25)
        return
    above_p, _ = motion.body_goal(goal, stand_R, body)
    branches = _strut_approach_branches(motion, above_p, touch_p, touch_R, body, attempts=48)
    try:
        path = _strut_tilt_rrt(motion, body, width, branches)
        _log("stand carry tilt rrt", name, "waypoints", len(path), "max_tilt_deg",
             round(_strut_path_tilt(motion, path, width), 1))
        motion.execute(path, speed=.25)
        return
    except PlanningError as error:
        errors.append(str(error)[:120])
        _log("stand carry tilt rrt rejected", name, errors[-1])
    for q in branches:
        try:
            path = motion.plan(q, width=width, allow_pad_body=body, timeout=STRUT_RRT_TIMEOUT)
        except PlanningError as error:
            errors.append(str(error)[:120])
            continue
        tilt = _strut_path_tilt(motion, path, width)
        if tilt > STRUT_TILT_OK:
            errors.append(f"RRT path tilts the strut {tilt:.0f} deg")
            _log("stand carry fallback rejected", name, errors[-1])
            continue
        _log("stand carry fallback rrt", name, "waypoints", len(path), "max_tilt_deg", round(tilt, 1))
        motion.execute(path, speed=.25)
        return
    raise PlanningError(f"{name}: no stand carry keeps the strut upright: {errors[-3:]}")


def _strut_carry_once(cell, motion, mate, body, g, ins, yaws):
    """Carry the upright strut to the hover beneath its seat.

    The strut goes to a point outside the chassis footprint at hover height
    (STRUT_OUTSIDE back along the approach) and then straight in under its
    corner. A plain joint-space RRT path can tip it far over; while the hand
    points down gravity walks the pinched shaft towards the fingertips (a05: 62
    deg tilt, 5 mm walk, 14 deg pivot), so the carry is planned by a task-space
    RRT that keeps the strut upright, then (another arm posture needed) by the
    tilt-constrained joint-space RRT, and only last by the plain RRT whose path
    stays within STRUT_TILT_OK. Yaws needing the least hand turn go first.
    """
    width = motion.width()
    leg_now = cell.pose(mate["part"])
    yaw_now = _strut_yaw(leg_now[:3, :3])
    def turn(yaw):
        g.target_yaw = yaw
        return abs(math.remainder(_strut_yaw(_target(cell, mate, g)[:3, :3])-yaw_now, 2*math.pi))
    def goal(yaw):
        g.target_yaw = yaw
        target = _target(cell, mate, g)
        pre = target[:3, 3]-ins*STRUT_HOVER
        hover_p, hover_R = motion.body_goal(pre, target[:3, :3], body)
        out = hover_R[:, 2].copy()
        out[2] = 0.
        out /= np.linalg.norm(out)
        motion.set_mate_contact(["part_"+cell.coupling.carrier(mate)], target[:3, 3], target[:3, :3])
        return target, pre, hover_p, hover_R, out
    def outside_branches(hover_p, hover_R, out, count):
        branches = []
        for q_out in motion.ik(hover_p-out*STRUT_OUTSIDE, hover_R, width=width, attempts=48, max_solutions=count,
                               allow_pad_body=body):
            try:
                motion.line(hover_p, hover_R, allow_pad_body=body, start_q=q_out, execute=False)
            except PlanningError:
                continue
            branches.append(q_out)
        return branches
    def go(what, yaw, path, hover_p, hover_R, tilt=None):
        _log(what, mate["part"], "yaw", round(yaw, 3), "waypoints", len(path), "max_tilt_deg",
             round(_strut_path_tilt(motion, path, width) if tilt is None else tilt, 1))
        motion.execute(path, speed=.25)
        try:
            motion.line(hover_p, hover_R, allow_pad_body=body, clearance=True, speed=.03, smooth=True)
        except PlanningError as error:
            # The straight line-in was checked from the planned end; the tracked
            # end can sit a fraction of a millimetre off it. Plan it instead.
            _log("carry line-in replanned", mate["part"], str(error)[:100])
            hover = motion.ik(hover_p, hover_R, width=width, attempts=24, max_solutions=4, allow_pad_body=body,
                              seed=motion.q)
            motion.execute(_strut_tilt_rrt(motion, body, width, hover), speed=.25)
    order = sorted(yaws, key=turn)
    errors = []
    for yaw in order:
        target, pre, hover_p, hover_R, out = goal(yaw)
        try:
            path = _strut_upright_rrt(motion, body, width, pre-out*STRUT_OUTSIDE, _strut_yaw(target[:3, :3]))
            motion.line(hover_p, hover_R, allow_pad_body=body, start_q=path[-1], execute=False)
        except PlanningError as error:
            errors.append(str(error)[:160])
            _log("carry yaw rejected", mate["part"], round(yaw, 3), errors[-1])
            continue
        return go("carry", yaw, path, hover_p, hover_R)
    # Joint-space carries are executed at .25 rad/s: try first the yaw whose
    # outside branches lie closest to the arm (a wound wrist can make one yaw
    # a 6 rad carry and another a 3 rad one).
    q_now = motion.q
    candidates = []
    for yaw in order:
        target, pre, hover_p, hover_R, out = goal(yaw)
        branches = outside_branches(hover_p, hover_R, out, 6)
        if branches:
            distance = min(float(np.max(np.abs(q-q_now))) for q in branches)
            candidates.append((distance, yaw, branches))
        else:
            errors.append("tilt RRT: no goal branch")
            _log("carry tilt rrt rejected", mate["part"], round(yaw, 3), errors[-1])
    for distance, yaw, branches in sorted(candidates, key=lambda c: c[0]):
        target, pre, hover_p, hover_R, out = goal(yaw)
        try:
            path = _strut_tilt_rrt(motion, body, width, branches)
        except PlanningError as error:
            errors.append(str(error)[:160])
            _log("carry tilt rrt rejected", mate["part"], round(yaw, 3), errors[-1])
            continue
        return go("carry tilt rrt", yaw, path, hover_p, hover_R)
    # Last resort: the plain joint-space RRT with the least strut tilt, and only
    # within STRUT_TILT_OK (a tipped strut walks out of the pads: final05 lost one
    # on an unchecked path).
    best = None
    for yaw in order:
        target, pre, hover_p, hover_R, out = goal(yaw)
        for q_out in outside_branches(hover_p, hover_R, out, 4):
            try:
                path = motion.plan(q_out, width=width, allow_pad_body=body, clearance=True,
                                   timeout=STRUT_RRT_TIMEOUT)
            except PlanningError as error:
                errors.append(str(error)[:160])
                continue
            tilt = _strut_path_tilt(motion, path, width)
            if tilt > STRUT_TILT_OK:
                errors.append(f"RRT path tilts the strut {tilt:.0f} deg")
                continue
            if best is None or tilt < best[0]:
                best = (tilt, yaw, path, hover_p, hover_R)
    if best is None:
        raise PlanningError(f"no carry path for {mate['part']}: {errors[:3]}")
    tilt, yaw, path, hover_p, hover_R = best
    goal(yaw)
    go("carry fallback rrt", yaw, path, hover_p, hover_R, tilt)


def _strut_carry(cell, motion, mate, body, g, ins, yaws):
    """Carry the upright strut to its hover; if no path exists, lift it 80 mm and retry once."""
    try:
        return _strut_carry_once(cell, motion, mate, body, g, ins, yaws)
    except PlanningError as error:
        _log("carry retry after lift", mate["part"], str(error)[:160])
    _strut_rise(motion, body, mate["part"], (.08, .05, .025))
    return _strut_carry_once(cell, motion, mate, body, g, ins, yaws)


def _strut_rise(motion, body, name, rises):
    """Lift the held strut straight up by the first clear distance of `rises`.

    Near the robot base a full lift can be refused by the arm's own clearance
    (link_base-link6 at 6.5 mm in one run); a shorter lift, or none, still
    leaves the next plan its own escape. Returns the distance lifted (0 if none).
    """
    tcp, R = motion.tool_pose()
    for rise in rises:
        try:
            motion.line(tcp+np.array([0., 0., rise]), R, allow_pad_body=body, clearance=True, speed=.03, smooth=True)
            return rise
        except PlanningError as error:
            _log("strut lift refused", name, rise, str(error)[:100])
    return 0.


def _place_strut(cell, motion, mate, *, choice=0, wall_deadline=math.inf):
    name, body = mate["part"], "part_"+mate["part"]
    T = cell.pose(name)
    carrier = cell.coupling.carrier(mate)
    contact_carriers = ["part_"+carrier]
    ins = cell.pose(carrier)[:3, :3] @ np.asarray(mate["insertion_axis"], float)
    label = f"stable-top-{STRUT_PICK_STATION:.3f}"
    candidates = [g for g in grasps(cell, name) if g.label.startswith(label)]
    if not candidates:
        raise PlanningError(f"no stable-top shaft grasp for {name}")
    g0 = candidates[0]
    p, R = _world(T, g0)
    width = min(.085, g0.width+.012)
    if not motion.ik(p, R, width=width, attempts=10, max_solutions=1, clearance=False):
        raise PlanningError(f"{name}: pick pose unreachable")
    if not motion.ik(p-R[:, 2]*.10, R, width=width, attempts=12, max_solutions=1):
        raise PlanningError(f"{name}: pick approach unreachable")
    # Upright insertion branches for the seated grasp, checked before touching the strut.
    yaws = []
    for g in candidates:
        seated = _strut_seated(g)
        target = _target(cell, mate, seated)
        gp, gR = _world(target, seated)
        hover = motion.ik(gp-ins*STRUT_HOVER, gR, width=g.width, attempts=16, max_solutions=1, allow_pad_body=body)
        seat = hover and motion.ik(gp, gR, width=g.width, attempts=16, max_solutions=1, allow_pad_body=body)
        _log("strut branch", name, g.label, "hover", bool(hover), "seat", bool(seat))
        if hover and seat:
            yaws.append(g.target_yaw)
    if choice:
        yaws = yaws[choice % len(yaws):]+yaws[:choice % len(yaws)]
    g = g0
    _log("pick", name, g.label, "width", round(g.width, 4), "pose", np.round(p, 4), "yaws", np.round(yaws, 3))
    _strut_pick_upright(cell, motion, name, body, g, p, R, width, prefer_margin=not yaws, avoid_wound=bool(yaws))
    if not yaws:
        # No seat approach exists for the twist the bench pose gives the
        # stable-top grasp: stand the strut up and take it again from the side.
        g = _strut_regrasp(cell, motion, mate, name, body, g)
        yaws = [g.target_yaw]
    _strut_carry(cell, motion, mate, body, g, ins, yaws)
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} during transport")
    target = _target(cell, mate, g)
    motion.set_mate_contact(contact_carriers, target[:3, 3], target[:3, :3])
    pin, axis = _strut_in_hand(motion, body)
    _log("hover", name, "in_hand_mm", np.round(pin*1000, 2), "axis_hand", np.round(axis, 3))
    motion.align_body(target[:3, 3]-ins*STRUT_HOVER, target[:3, :3], body=body,
                      tolerance=.0005, angle=.008, max_passes=10)
    if not _gripped(cell, name):
        raise PlanningError(f"lost grip on {name} during hover alignment")
    flush = target[:3, 3]
    pp, rr = motion.body_goal(flush, target[:3, :3], body)
    motion.line(pp, rr, allow_pad_body=body, clearance=False, speed=.015)
    pin, axis = _strut_in_hand(motion, body)
    _log("flush", name, "error_mm", np.round(1000*(cell.pose(name)[:3, 3]-target[:3, 3]), 3),
         "in_hand_mm", np.round(pin*1000, 2), "state", cell.coupling.state[mate["id"]])
    # Press: same measured grasp, PRESS beyond flush along the insertion axis.
    motion.line(pp+ins*STRUT_PRESS, rr, allow_pad_body=body, clearance=False, speed=.005)
    q = motion.q
    for step in range(150):
        if time.monotonic() > wall_deadline:
            raise TimeoutError("reference wall cap during seat hold")
        motion._drive(q); cell.tick()
        if step % 25 == 0:
            _log("press", name, step, cell.coupling.state[mate["id"]], "error_mm",
                 np.round(1000*(cell.pose(name)[:3, 3]-target[:3, 3]), 3), "forces", _grip_forces(cell, name))
        if cell.coupling.state[mate["id"]] in ("LOCKED", "LOCKED_OFF") and step >= 10:
            break
    state = cell.coupling.state[mate["id"]]
    event = next((e for e in cell.events[::-1] if e.get("mate") == mate["id"]), {})
    _log("seat", name, state, "error_mm", round(1000*float(np.linalg.norm(cell.pose(name)[:3, 3]-target[:3, 3])), 3),
         "lock_error_mm", round(1000*event.get("error_m", float("nan")), 3), "lock_angle_deg",
         round(event.get("angle_deg", float("nan")), 3))
    if state != "LOCKED":
        raise PlanningError(f"{name} did not fasten correctly ({state})")
    motion.hold_body(None)
    released_width = _open_clear(motion, body, min(.085, g.width+.012))
    pp, rr = motion.tool_pose()
    errors = []
    for distance in (.08, .06, .04):
        try:
            motion.line(pp-rr[:, 2]*distance, rr, allow_pad_body=body, clearance=True, speed=.03)
            break
        except PlanningError as e:
            errors.append(str(e))
    else:
        raise PlanningError(f"cannot retract from {name}: {errors}")
    if cell.violations:
        raise PlanningError(f"safety failure after {name}: {cell.violations[:2]}")
    return dict(part=name, grasp=g.label, candidate=0, sim_time_s=float(cell.d.time), state=state,
                lock_error_mm=1000*event.get("error_m", float("nan")), lock_angle_deg=event.get("angle_deg", float("nan")))


# Camera still-hold slip check after the 10 cm lift (camera only): settle,
# then hold still and measure the camera's motion in the gripper frame. A
# pinch that creeps faster, or whose expected slide over the carry (slid so
# far + rate x CAMERA_CARRY_S) would use more than CAMERA_SLIDE_KEEP of its
# contact band (Grasp.slide_m), is put back and the next candidate is tried;
# a slipping camera is never carried. Lift to lock took 31-34 s (two runs:
# 3.2-3.6 mm slide, 1-2 deg about the jaw axis); the carry keeps a larger
# held-part margin for that in-hand motion.
CAMERA_SLIP_SETTLE_S = .3
CAMERA_SLIP_HOLD_S = 1.0
CAMERA_SLIP_MM_S = .2
CAMERA_SLIP_DEG_S = .15
CAMERA_CARRY_S = 40.
CAMERA_SLIDE_KEEP = .6
CAMERA_PICK_ATTEMPTS = 3
CAMERA_CARRY_CLEARANCE_M = .008


def _camera_in_hand(cell, motion, name):
    """Camera pose in the gripper-base frame (measured live)."""
    Th = np.eye(4)
    Th[:3,:3] = cell.d.xmat[motion.hand].reshape(3,3)
    Th[:3,3] = cell.d.xpos[motion.hand]
    return np.linalg.inv(Th) @ cell.pose(name)


def _camera_slip(cell, motion, name, clamped):
    """Hold the lifted camera still; return its in-hand creep rates (mm/s, deg/s)
    and how far it has slid in the jaws since the clamp (mm; clamped = the
    in-hand pose before the lift)."""
    q=motion.q
    for _ in range(int(round(CAMERA_SLIP_SETTLE_S/.01))):
        motion._drive(q);cell.tick()
    a=_camera_in_hand(cell,motion,name)
    for _ in range(int(round(CAMERA_SLIP_HOLD_S/.01))):
        motion._drive(q);cell.tick()
    b=_camera_in_hand(cell,motion,name)
    mm=1000*float(np.linalg.norm(b[:3,3]-a[:3,3]))/CAMERA_SLIP_HOLD_S
    deg=math.degrees(Rotation.from_matrix(a[:3,:3].T @ b[:3,:3]).magnitude())/CAMERA_SLIP_HOLD_S
    slid=1000*float(np.linalg.norm(b[:3,3]-clamped[:3,3]))
    return mm,deg,slid


def _camera_put_back(cell, motion, name, body, bench_T, R, width):
    """Set a creeping camera back on its bench pose, open, back off.

    It keeps creeping in the jaws on the way down, so the hand stops 5 mm
    above the bench pose, re-measures the camera in the jaws and lowers it
    slowly to 0.3 mm above that pose: the release drops it a few tenths of a
    millimetre (a 1 mm goal dropped it ~0.6 mm, 0.38 mm impact into the bench).
    """
    for above,speed in ((.005,.025),(.0003,.005)):
        motion.hold_body(body)
        pp,rr=motion.body_goal(bench_T[:3,3]+np.array([0.,0.,above]),bench_T[:3,:3],body)
        motion.line(pp,rr,allow_pad_body=body,clearance=False,speed=speed)
    released_mm=1000*float(cell.pose(name)[2,3]-bench_T[2,3])
    motion.hold_body(None)
    motion.set_width(width)
    motion.wait(1.0)
    p_now,r_now=motion.tool_pose()
    motion.line(p_now-r_now[:,2]*.10,r_now,clearance=False,speed=.03)
    motion.wait(.5)
    _log("camera put back", "released mm above the bench pose", round(released_mm,2),
         "bench drift mm", round(1000*float(np.linalg.norm(cell.pose(name)[:3,3]-bench_T[:3,3])),3))


# Battery (lead-free kit: plain 140 x 45 x 39 mm pack, 0.49 kg). The stock pads
# close on the top edge of its side faces (the loaded fingers toe in), a line
# 17.8 mm above the COM; any COM offset along the pack turns it about that line.
# With the generic 4 mm squeeze (~7 N per pad) it turned at ~0.3 deg/s in the
# jaws, and the hover alignment, turning the hand under it, wound it to 7-33 deg
# (three full runs). A 5 mm squeeze gives
# ~17 N per pad and ~0.025 deg/s (6 mm ~32 N and 8 mm
# ~50 N creep alike, 0.022 / 0.005 deg/s, but 6 mm already sums to ~115 N of pad
# contact force on the pack, against the cell's 150 N sustained part-force limit).
BATTERY_SQUEEZE_M = .005
BATTERY_SLIP_SETTLE_S = .3
BATTERY_SLIP_HOLD_S = 1.0
BATTERY_SLIP_DEG_S = .15        # in-hand turn rate after the lift (still hold)
BATTERY_SLIP_TURN_DEG = 1.5     # in-hand turn since the clamp
BATTERY_PICK_ATTEMPTS = 3
BATTERY_ALIGN_ACCEPT_DEG = 1.5  # residual accepted if the alignment does not converge (mate: 3 deg)
BATTERY_ALIGN_ACCEPT_TOL = 2.   # x the position tolerance


def battery_grasps(cell, name):
    """Battery: the generic flat-face pinches across the 45 mm width, each moved
    along its pad axis so the pad centre lies on the measured COM (the jaw
    midplane already passes through it), with a firm squeeze. Candidates more
    than 15 mm from the COM are dropped; duplicates after the move are removed."""
    com = np.asarray(cell.kit["parts"]["battery"]["inertial"]["center_m"])
    seen = set()
    for g in surface_grasps(cell, name):
        pad_c = g.p + g.R[:, 2]*(_tip(g.width)-.158) - g.R[:, 2]*.0095
        shift = float((com-pad_c) @ g.R[:, 0])
        if abs(shift) > .015:
            continue
        g.p = g.p + shift*g.R[:, 0]
        key = tuple(np.round(g.p/.001).astype(int)) + tuple(np.round(g.R[:, 1], 2))
        if key in seen:
            continue
        seen.add(key)
        g.squeeze = BATTERY_SQUEEZE_M
        g.label += "-com"
        yield g


def _battery_slip(cell, motion, name, clamped):
    """Hold the lifted battery still; return its in-hand turn rate (deg/s) and
    its in-hand turn since the clamp (deg)."""
    q = motion.q
    for _ in range(int(round(BATTERY_SLIP_SETTLE_S/.01))):
        motion._drive(q); cell.tick()
    a = _camera_in_hand(cell, motion, name)
    for _ in range(int(round(BATTERY_SLIP_HOLD_S/.01))):
        motion._drive(q); cell.tick()
    b = _camera_in_hand(cell, motion, name)
    rate = math.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())/BATTERY_SLIP_HOLD_S
    turned = math.degrees(Rotation.from_matrix(clamped[:3, :3].T @ b[:3, :3]).magnitude())
    return rate, turned


def _battery_put_back(cell, motion, name, body, bench_T, width):
    """Set a turning battery back on its bench pose (5 mm above, then slowly to
    0.3 mm above), open, back off 10 cm."""
    for above, speed in ((.005, .025), (.0003, .005)):
        motion.hold_body(body)
        pp, rr = motion.body_goal(bench_T[:3, 3]+np.array([0., 0., above]), bench_T[:3, :3], body)
        motion.line(pp, rr, allow_pad_body=body, clearance=False, speed=speed)
    motion.hold_body(None)
    motion.set_width(width)
    motion.wait(1.0)
    p_now, r_now = motion.tool_pose()
    motion.line(p_now-r_now[:, 2]*.10, r_now, clearance=False, speed=.03)
    motion.wait(.5)
    _log("battery put back", "bench drift mm",
         round(1000*float(np.linalg.norm(cell.pose(name)[:3, 3]-bench_T[:3, 3])), 3))


def _battery_align(motion, pos, R, body, *, tolerance, max_passes):
    """motion.align_body for the battery; a residual within BATTERY_ALIGN_ACCEPT_*
    (the mate accepts 1.5 mm / 3 deg) is accepted instead of raising, so slow
    residual creep in the jaws cannot end the placement."""
    try:
        return motion.align_body(pos, R, body=body, tolerance=tolerance, angle=.01, max_passes=max_passes)
    except PlanningError as error:
        if "did not converge" not in str(error):
            raise
        p_now, r_now = motion.body_pose(motion.held[0])
        ep = float(np.linalg.norm(np.asarray(pos)-p_now))
        er = float(Rotation.from_matrix(np.asarray(R) @ r_now.T).magnitude())
        if ep > BATTERY_ALIGN_ACCEPT_TOL*tolerance or er > math.radians(BATTERY_ALIGN_ACCEPT_DEG):
            raise
        _log("battery alignment residual accepted", round(ep*1000, 3), "mm", round(math.degrees(er), 3), "deg")
        return {"position_error_m": ep, "angle_error_rad": er}


def place(cell, motion, mate, *, choice=0, wall_deadline=math.inf, _camera_rejected=frozenset(),
          _prop_rejected=frozenset(), _battery_rejected=frozenset(), _cheek_rejected=frozenset()):
    name, body = mate["part"], "part_"+mate["part"]
    kind=cell.graph["instances"][name]["kind"]
    if kind == "landing_strut":
        return _place_strut(cell, motion, mate, choice=choice, wall_deadline=wall_deadline)
    is_prop = kind in ("prop_cw","prop_ccw")
    pre_distance=.040 if kind == "lidar" else .065
    retreat_distance=.040 if kind == "lidar" else .080
    T = cell.pose(name)
    carrier=cell.coupling.carrier(mate)
    contact_carriers=["part_"+carrier]
    if name == "sensor_cap":
        # The roof seats on both installed cheek rims; only its near-seat
        # contact envelope is exempted, exactly as for the named carrier.
        contact_carriers.append("part_"+cell.coupling.assigned.get("mount_cheek_right","cheek_right"))
    if kind == "lidar":
        # Seated on the cap plate, the LiDAR base is only the cap's 1.5 mm
        # above both cheek rims (inside the 2 mm held-part planning margin).
        # Exempt the cheeks only in the same near-seat envelope as the carrier.
        for key,default in (("mount_cheek_left","cheek_left"),("mount_cheek_right","cheek_right")):
            contact_carriers.append("part_"+cell.coupling.assigned.get(key,default))
    nominal_carrier=cell.graph["instances"].get(mate["carrier"].split("/")[0],{})
    nominal_R=Rotation.from_euler("xyz",nominal_carrier.get("target_euler_xyz_deg",[0,0,0]),degrees=True).as_matrix()
    # Graph arrows are in the nominal fixture frame. Follow carrier drift,
    # without applying its nominal CAD rotation a second time.
    ins = cell.pose(carrier)[:3,:3] @ nominal_R.T @ np.asarray(mate["insertion_axis"],float)
    selected = None
    count = 0
    pick_prepared = False
    pickup_errors = []
    candidates = grasps(cell,name)
    if kind == "camera" and _camera_rejected:
        candidates = (c for c in candidates if c.label not in _camera_rejected)
    if is_prop and _prop_rejected:
        candidates = (c for c in candidates if c.label not in _prop_rejected)
    if kind == "battery" and _battery_rejected:
        candidates = (c for c in candidates if c.label not in _battery_rejected)
    if kind == "sensor_cheek" and _cheek_rejected:
        candidates = (c for c in candidates if c.label not in _cheek_rejected)
    if kind in ("camera", "flight_controller") or (kind == "lidar" and _lidar_upright(cell, name)):
        candidates = _least_travel_first(cell,motion,mate,candidates,-ins*pre_distance)
    for i,g in enumerate(candidates):
        if time.monotonic() > wall_deadline:
            raise TimeoutError("reference wall cap during grasp search")
        p,R = _world(T,g)
        target = _target(cell,mate,g)
        gp,gR = _world(target,g)
        width = .085 if kind == "sensor_cap" and g.width>.020 else min(.085,g.width+.012)
        # Test both ends; the actual transport is planned again with its measured
        # held part after pickup. Final insertion is checked outside-in.
        pick=motion.ik(p,R,width=width,attempts=10,max_solutions=1,clearance=False)
        if not pick:
            continue
        if not _grip_clear(motion,pick[0],g,body,name):
            continue
        if not motion.ik(gp-ins*pre_distance,gR,width=g.width,attempts=10,max_solutions=1,allow_pad_body=body):
            continue
        if not motion.ik(gp,gR,width=g.width,attempts=10,max_solutions=1,allow_pad_body=body,clearance=False):
            continue
        if not motion.ik(p-R[:,2]*.10,R,width=width,attempts=12,max_solutions=1):
            continue
        if count < choice:
            count += 1; continue
        if is_prop or kind == "camera":
            # A blade's endpoints can be clear while its oblique pickup path
            # clips a neighbouring staged component. Try the equivalent blade
            # before touching the prop, with the actual settled jaw aperture.
            # The camera's oblique approach down its board plane passes the
            # installed guard ring; take a candidate whose whole approach is clear.
            try:
                _prepare_width(motion,width)
                motion.set_width(width)
                motion.wait(1.5)
                width=motion.width()
                _preinsert(motion,p-R[:,2]*.10,p,R,None,clearance=False,
                           transport_clearance=True,attempts=128,
                           speed=PROP_FREE_SPEED if is_prop else .25)
            except PlanningError as error:
                if cell.violations:
                    raise
                pickup_errors.append(str(error))
                _log("reject pickup path",name,g.label,str(error))
                continue
            pick_prepared=True
        selected=(i,g,p,R,gp,gR,width);break
    if selected is None:
        raise PlanningError(f"no pick+preinsert grasp for {name}; pickup failures {pickup_errors[:3]}; last contacts {motion.last_contacts[:3]}")
    i,g,p,R,gp,gR,width=selected
    _log("pick",name,"candidate",i,g.label,"width",round(g.width,4),"pose",np.round(p,4))
    if not pick_prepared:
        _prepare_width(motion,width)
        motion.set_width(width)
        # Large opening changes continue moving after the command ramp.
        motion.wait(1.5)
        width=motion.width()
    _log("preshape",name,"measured_gap",round(width,4))
    approach = R[:,2]
    if not pick_prepared:
        _preinsert(motion,p-approach*.10,p,R,None,clearance=False,transport_clearance=True)
    if is_prop:
        # Light part on its stand's pedestal: close most of the checked
        # approach briskly and only the last 20 mm slowly.
        motion.line(p-approach*.02,R,width=width,clearance=False,speed=.06)
    motion.line(p,R,width=width,clearance=False,speed=.025 if not is_prop else .02)
    motion.close(g.width,squeeze=g.squeeze)
    if is_prop:
        _prop_squeeze(cell,motion,name,min_force=PROP_STAND_LOCK_N,max_s=PROP_STAND_SQUEEZE_S)
    else:
        motion.wait(max(g.settle_s,.5))
    _log("clamp",name,"gap",round(motion.width(),4),"forces",_grip_forces(cell,name))
    if is_prop and min(_grip_forces(cell,name).values())<PROP_CLAMP_MIN_N:
        rejected=_prop_rejected|{g.label}
        _log("prop pinch did not lock",name,g.label,"- release onto the stand, back out, next blade")
        _prop_release(motion,body,p,R,width)
        if len(rejected)>=PROP_PICK_ATTEMPTS:
            raise PlanningError(f"no locked pinch on {name}: rejected {sorted(rejected)}")
        return place(cell,motion,mate,choice=choice,wall_deadline=wall_deadline,_prop_rejected=rejected)
    if not _gripped(cell,name):
        raise PlanningError(f"two-sided pad contact missing for {name}")
    motion.hold_body(body)
    if is_prop:
        # The pinch locked on the pedestal (the blade turned ~9-10 deg into it).
        # Lift 3 mm clear of the stand and let the blade settle into the pads
        # in the air before the full lift.
        motion.line(p-approach*.003,R,allow_pad_body=body,clearance=False,speed=.01)
        _prop_squeeze(cell,motion,name,min_force=1.0,max_s=2.0)
        if not _gripped(cell,name):
            raise PlanningError(f"lost grip on {name} at lift-off")
    clamped=_camera_in_hand(cell,motion,name) if kind in ("camera", "battery", "sensor_cheek") else None
    # The pinched blade pivots slowly in the jaws (see _prop_servo); do not
    # linger in the lift.
    motion.line(p-approach*(CHEEK_LIFT_M if kind == "sensor_cheek" else .10),R,allow_pad_body=body,clearance=False,
                speed=.05 if is_prop else CHEEK_LIFT_SPEED if kind == "sensor_cheek" else .025)
    if not _gripped(cell,name):
        raise PlanningError(f"lost grip on {name} after lift")
    if kind == "camera":
        creep_mm,creep_deg,slid_mm=_camera_slip(cell,motion,name,clamped)
        carry_mm=slid_mm+creep_mm*CAMERA_CARRY_S
        held=(creep_mm<=CAMERA_SLIP_MM_S and creep_deg<=CAMERA_SLIP_DEG_S
              and carry_mm<=CAMERA_SLIDE_KEEP*1000*g.slide_m and _gripped(cell,name))
        _log("camera slip check",g.label,"creep mm/s",round(creep_mm,3),"deg/s",round(creep_deg,3),
             "slid mm",round(slid_mm,2),"expected carry slide mm",round(carry_mm,2),"band mm",round(1000*g.slide_m,1),
             "held" if held else "CREEPS -> put back")
        if not held:
            _camera_put_back(cell,motion,name,body,T,R,width)
            rejected=_camera_rejected|{g.label}
            if len(rejected)>=CAMERA_PICK_ATTEMPTS:
                raise PlanningError(f"no camera pinch held still: rejected {sorted(rejected)}")
            return place(cell,motion,mate,choice=choice,wall_deadline=wall_deadline,_camera_rejected=rejected)
        motion.hold_body(body)
    if kind == "battery":
        rate,turned=_battery_slip(cell,motion,name,clamped)
        held=rate<=BATTERY_SLIP_DEG_S and turned<=BATTERY_SLIP_TURN_DEG and _gripped(cell,name)
        _log("battery slip check",g.label,"turn deg/s",round(rate,3),"turned since clamp deg",round(turned,3),
             "held" if held else "TURNS -> put back")
        if not held:
            _battery_put_back(cell,motion,name,body,T,width)
            rejected=_battery_rejected|{g.label}
            if len(rejected)>=BATTERY_PICK_ATTEMPTS:
                raise PlanningError(f"no battery pinch held still: rejected {sorted(rejected)}")
            return place(cell,motion,mate,choice=choice,wall_deadline=wall_deadline,_battery_rejected=rejected)
        motion.hold_body(body)
    cheek_turn=None
    if kind == "sensor_cheek":
        outcome,cheek_turn=_cheek_turn_carry(cell,motion,mate,name,body,g,T,width,ins,pre_distance,
                                             retreat_distance,contact_carriers,clamped,
                                             last_attempt=len(_cheek_rejected)+1>=CHEEK_PICK_ATTEMPTS)
        if outcome == "put back":
            rejected=_cheek_rejected|{g.label}
            if len(rejected)>=CHEEK_PICK_ATTEMPTS:
                raise PlanningError(f"no cheek rim pinch held: rejected {sorted(rejected)} (cheek back on its stand)")
            return place(cell,motion,mate,choice=choice,wall_deadline=wall_deadline,_cheek_rejected=rejected)
    base_yaw=g.target_yaw
    symmetry=int(mate.get("axial_symmetry",1))
    free_yaw=symmetry==0
    if free_yaw:
        target_yaws=(base_yaw,base_yaw+math.pi/2,base_yaw-math.pi/2,base_yaw+math.pi)
    elif symmetry>1:
        target_yaws=tuple(base_yaw+2*math.pi*k/symmetry for k in range(symmetry))
    else:
        target_yaws=(base_yaw,)
    transport_errors=[]
    # Propellers first try only hover branches verified for the whole
    # insertion (_prop_preinsert); the plain transport is the fallback.
    # A cheek carried upright is already at its hover: no transport.
    for verified in (() if cheek_turn is not None else (True,False) if is_prop else (False,)):
        # The verified search covers every free-yaw turn itself: run it once.
        for yaw in (target_yaws[:1] if verified else target_yaws):
            g.target_yaw=yaw
            target = _target(cell,mate,g)
            pre = target[:3,3]-ins*pre_distance
            pp,rr=motion.body_goal(pre,target[:3,:3],body)
            goal_p,goal_R=motion.body_goal(target[:3,3],target[:3,:3],body)
            motion.set_mate_contact(contact_carriers,target[:3,3],target[:3,:3],axis_only=free_yaw)
            _log("transport",name,"pre",np.round(pre,4),"yaw",round(yaw,3))
            retreat=-goal_R[:,2]*(.06 if kind == "landing_strut" else retreat_distance)
            try:
                if verified:
                    turn,worst,pivots=_prop_preinsert(motion,body,target,pre_distance,min(.085,g.width+.012),speed=.65)
                    # The prop's final yaw is free; record the chosen turn.
                    g.target_yaw=yaw+math.radians(turn)
                    _log("prop branch",name,"verified transport branch: yaw turn deg",turn,"pivot check deg",pivots,
                         "min self-distance mm",round(1000*worst,2))
                else:
                    _preinsert(motion,pp,goal_p,goal_R,body,retreat=retreat,open_width=min(.085,g.width+.012),
                               clearance=not is_prop,transport_clearance=True,
                               attempts=96 if is_prop else 64,speed=.65 if is_prop else .25,
                               transport_held_clearance=CAMERA_CARRY_CLEARANCE_M if kind == "camera" else None)
                break
            except PlanningError as error:
                transport_errors.append(str(error))
        else:
            continue
        break
    else:
        if cheek_turn is None:
            raise PlanningError(f"all axial orientations failed for {name}: {transport_errors}")
    if not _gripped(cell,name):
        raise PlanningError(f"lost grip on {name} during transport")
    target=_target(cell,mate,g)
    motion.set_mate_contact(contact_carriers,target[:3,3],target[:3,:3],axis_only=free_yaw)
    if is_prop:
        q=_prop_insert(cell,motion,mate,name,body,target,ins,pre_distance,wall_deadline=wall_deadline,
                       open_width=min(.085,g.width+.012))
        _prop_seat(cell,motion,mate,name,body,q,min(.085,g.width+.012),wall_deadline)
    elif kind == "sensor_cheek":
        try:
            _cheek_seat(cell,motion,mate,name,body,target,ins,pre_distance,wall_deadline=wall_deadline)
        except PlanningError as error:
            # Still free above its seat (the coupling has not welded it) and still
            # in the grip: back onto its stand for the next pinch, never dropped.
            if (cheek_turn is None or not (isinstance(error,_CheekSlip) or "did not converge" in str(error))
                    or cell.coupling.state[mate["id"]] in ("LOCKED","LOCKED_OFF") or not _gripped(cell,name)):
                raise
            _log("cheek seat not possible:",name,str(error)[:120],"- back to the stand")
            hp,hr=motion.body_goal(target[:3,3]-ins*pre_distance,target[:3,:3],body)
            motion.line(hp,hr,allow_pad_body=body,clearance=False,speed=.03)
            _cheek_back_to_stand(cell,motion,name,body,cheek_turn[0],cheek_turn[1],T,width,g.width-g.squeeze)
            rejected=_cheek_rejected|{g.label}
            if len(rejected)>=CHEEK_PICK_ATTEMPTS:
                raise PlanningError(f"cheek seat did not converge for {sorted(rejected)} (cheek back on its stand)") from error
            return place(cell,motion,mate,choice=choice,wall_deadline=wall_deadline,_cheek_rejected=rejected)
    else:
        if kind == "landing_strut" or cell.kit["parts"][kind].get("grasp_mode") == "surface":
            # Real contact grasps can creep during transport. Correct the full
            # measured pose while still clear of the receiving surface, rather
            # than trying to rotate a misaligned flange after it touches the base.
            if kind == "lidar":
                _lidar_align(motion,target[:3,3]-ins*pre_distance,target[:3,:3],body,tolerance=.0007,max_passes=10)
            elif kind == "battery":
                _battery_align(motion,target[:3,3]-ins*pre_distance,target[:3,:3],body,tolerance=.0007,max_passes=10)
            else:
                motion.align_body(target[:3,3]-ins*pre_distance,target[:3,:3],body=body,
                                  tolerance=.0007,angle=.01,max_passes=10,axis_only=free_yaw)
            if not _gripped(cell,name):
                raise PlanningError(f"lost grip on {name} during hover alignment")
        pp,rr=motion.body_goal(target[:3,3],target[:3,:3],body)
        try:
            motion.line(pp,rr,allow_pad_body=body,clearance=False,execute=False)
        except PlanningError as error:
            # The part moved in the hand during the carry (re-measured by
            # body_goal): from the transport plan's IK branch the straight
            # descent to the re-measured seat goal is blocked. Choose again,
            # as for the transport, an IK branch at the hover whose straight
            # descent is clear, move there, then descend. Planning-only check
            # first: a clear descent is executed exactly as before.
            _log("descent blocked from the transport branch:",name,str(error)[:160])
            hp,hr=motion.body_goal(target[:3,3]-ins*pre_distance,target[:3,:3],body)
            _preinsert(motion,hp,pp,rr,body,clearance=False,transport_clearance=True,attempts=96,speed=.25)
            pp,rr=motion.body_goal(target[:3,3],target[:3,:3],body)
        motion.line(pp,rr,allow_pad_body=body,clearance=False,speed=.015)
        if cell.coupling.state[mate["id"]] not in ("LOCKED","LOCKED_OFF"):
            if kind == "lidar":
                _lidar_align(motion,target[:3,3],target[:3,:3],body,tolerance=.0005,max_passes=8)
            elif kind == "battery":
                _battery_align(motion,target[:3,3],target[:3,:3],body,tolerance=.0005,max_passes=8)
            else:
                motion.align_body(target[:3,3],target[:3,:3],body=body,
                                  tolerance=.0005,angle=.01,max_passes=8,axis_only=free_yaw)
        q=motion.q
        for step in range(80):
            if time.monotonic()>wall_deadline:
                raise TimeoutError("reference wall cap during seat hold")
            motion._drive(q);cell.tick()
        if kind == "motor":
            # A motor still HOLDING after the fixed hold is counting its 0.5 s
            # at-rest window (one run: motor_4 HOLDING at the end):
            # keep the same hold until it locks. A motor that already locked
            # gets no extra tick, so working placements are unchanged.
            for step in range(MOTOR_EXTRA_HOLD_TICKS):
                if cell.coupling.state[mate["id"]] in ("LOCKED","LOCKED_OFF"):
                    break
                if time.monotonic()>wall_deadline:
                    raise TimeoutError("reference wall cap during seat hold")
                motion._drive(q);cell.tick()
    state=cell.coupling.state[mate["id"]]
    _log("seat",name,state,"error_mm",1000*np.linalg.norm(cell.pose(name)[:3,3]-target[:3,3]))
    if state != "LOCKED":
        raise PlanningError(f"{name} did not fasten correctly ({state})")
    motion.hold_body(None)
    released_width=min(.085,g.width+.012)
    if is_prop:
        released_width=_prop_open(motion,body,released_width)
    else:
        released_width=_open_clear(motion,body,released_width)
    if is_prop:
        # Linkage motion continues after the opening ramp; check the retreat
        # at its final aperture so a widening knuckle cannot graze a neighbour.
        motion.wait(1.0)
    pp,rr=motion.tool_pose()
    if is_prop:
        _prop_exit(cell,motion,name,released_width)
    elif kind == "landing_strut":
        # A long back-out can fold into the arm or approach another propeller.
        # Precheck shorter escapes, then let RRT plan the next free-space move.
        errors=[]
        for distance in (.08,.06,.04):
            try:
                motion.line(pp-rr[:,2]*distance,rr,width=None,
                            allow_pad_body=body,clearance=True,speed=.03)
                break
            except PlanningError as e:
                errors.append(str(e))
        else:
            raise PlanningError(f"cannot retract from {name}: {errors}")
    else:
        motion.line(pp-rr[:,2]*retreat_distance,rr,allow_pad_body=body,clearance=False,speed=.03)
    if cell.violations:
        raise PlanningError(f"safety failure after {name}: {cell.violations[:2]}")
    return dict(part=name,grasp=g.label,candidate=i,sim_time_s=float(cell.d.time),state=state)


def assemble(cell, *, only=None, choices=None, wall_cap_s=7200):
    deadline=time.monotonic()+wall_cap_s
    motion=Motion(cell)
    choices=choices or {}
    mates={m["part"]:m for m in cell.graph["mates"]}
    order = ["esc"] + [n for i in range(1,5) for n in (f"leg_{i}",f"motor_{i}")] + [
        f"prop_{i}" for i in range(1,5)] + [
        "guard","camera","flight_controller","compute","compute_cover","battery",
        "cheek_left","cheek_right","sensor_cap","lidar"]
    if only is not None:
        order = list(only)
    results=[]
    for name in order:
        if time.monotonic()>deadline:
            raise TimeoutError("reference assembly wall cap")
        results.append(place(cell,motion,mates[name],choice=choices.get(name,0),wall_deadline=deadline))
    result=cell.result();result["placements"]=results
    return result
