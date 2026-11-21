"""Berth of the reference dock v1 (v0.3): magnets, pogo pins, magnet retraction, lance / cup guide geometry, the
dock-relative (fiducial) localisation and the MjSpec edits that make the part-level dock and robot models dock.

Values (the reference build's dock report, ``iface.dock.*`` in the shared values file):
  * 4 magnets per berth on two carriers, each F(g) = 12.1 N / (1 + g / 1 mm)^2 between the magnet face and the
    robot's strike plate, working gap 0.39 mm (the panel skin) -> 25.1 N; the 20 pogo pins (one slide, 12.01 N at
    their 1.0 mm working compression, 1.4 mm travel) push back 12.0 N: net pull-off 13.1 N;
  * retraction: the carriers move 30 mm away at 6.2 mm/s (linear actuators); after 0.62 mm (0.1 s) the pins win and
    push the robot off at 0.034 m/s; the carriers re-extend after ``hold_s`` (a few seconds);
  * guide: two lances per berth (17.4 deg cones, 36.2 mm proud) entering the robot's two cups (mouth r 13.42 mm,
    17.4 deg, 22.3 mm deep, 13.2 mm through-hole): lateral capture 11.2 mm.
Model: the magnets and pins are force laws (``BerthDock``, a wrench source) on the berth axis, because the base
models' pin and magnet hulls are display parts (the pin hull would start 2.9 mm inside the robot, the magnets
collide with their own dock); the lances are the dock model's convex hulls; the robot's cups are added to its
collision model here (``rig_dock_spec``): the solid dock-face proxies of the reference build are replaced by boxes
that leave the two lance channels open plus a funnel of thin boxes on each cup cone.  These are adapter edits,
listed in the returned summary; the reference build itself is not changed.  From reference build v1.2 on both edits are
in the build (its 'refbuild' text, ``spacekit.refbuild``): ``rig_dock_spec`` then changes nothing and
``attach_dock_v1`` finds the display parts already contact-free.

``DockRelativeLocalizer`` wraps a map localiser: while a fiducial (the dock's target or a bag-mouth plate) is
referenced and seen by the chosen camera within range and field of view, the estimate error is the fiducial's
(RMSE of the error norm 1 cm at 0.6 m, scaling with range, floor 2 mm [G]); otherwise the map localiser's.  From v0.4
that error follows the map localiser's consistent model (a smooth state times the reported std;
``error_model='legacy'`` keeps v0.3).
"""
from __future__ import annotations

import math
import os
import re

import mujoco
import numpy as np

from .rotations import mat_to_quat, quat_from_rotvec, quat_mul, quat_to_mat

__all__ = ["DOCK_V1", "dock_model_path", "attach_dock_v1", "rig_dock_spec", "BerthDock", "DockRelativeLocalizer",
           "fiducial_visible", "berth_contact_pose", "camera_in_body", "plate_points"]

DOCK_V1 = dict(
    working_gap_m=0.39e-3,          # magnet face behind the panel skin  [V, C]
    magnet_peak_N=12.1,             # per magnet at contact  [V DR3 10.9 lbf read as the catalogue contact pull]
    magnet_g1_m=1.0e-3,             # F = F0 / (1 + g / g1)^2  [V]
    magnet_radius_m=0.00476,        # 3/8 in ring magnets; lateral overlap of magnet and strike plate  [V, G law]
    magnet_yz_m=((0.0286, 0.0286), (0.0286, -0.0286), (-0.0286, 0.0286), (-0.0286, -0.0286)),   # about the berth axis
    pins_k_N_m=12010.2,             # 20 pins as one spring: 12.01 N at 1.0 mm  [V]
    pins_damping_N_s_m=2.0,         # the base model's pin-slide damping  [V MJCF]
    pins_working_compression_m=1.0e-3, pins_travel_m=1.4e-3,
    net_pull_off_N=13.08, ejection_m_s=0.0339,                    # [V]
    retract_m=0.030, retract_speed_m_s=0.0062, hold_s=8.0,        # carriers' stroke [V DR3], speed [V 0.62 mm / 0.1 s]
    cup_mouth_r=0.01342, cup_half_angle_deg=17.4, cup_depth=0.0223, hole_r=0.0066,   # robot cups [V]
    cup_yz=((0.0, 0.03175), (0.0, -0.03175)),                     # cup axes on the robot dock face (body y, z)
    dock_face_x=-0.1461,            # robot dock-plate face (body x): 212.71 - 66.61 mm (the base proxy box is at -0.1460)
    lateral_capture_m=0.0112, roll_capture_deg=20.6, yaw_capture_deg=9.2,
    contact_pose_dock=((0.21271, 0.254, 0.0), (1.0, 0.0, 0.0, 0.0)),   # berth 1, robot body in the dock frame
    berth2_pose_dock=((0.21271, -0.254, 0.0), (0.0, 1.0, 0.0, 0.0)),
    eps_contact_m=0.2e-3,           # pins compressed by this much = the dock's loopback pins see the robot [G]
)

def dock_model_path(default: str | None = None) -> str | None:
    """Path of the reference build's dock scene: ``SPACEKIT_DOCK_MJCF`` if set, else ``default`` (callers -- scene
    adapters, tests -- pass their build location; the package carries no host paths)."""
    return os.environ.get("SPACEKIT_DOCK_MJCF", default)


def berth_contact_pose(dock_pos, dock_quat, berth: int = 1):
    """World pose (pos, quat) of the robot body resting on berth 1 or 2 of a dock placed at (dock_pos, dock_quat)."""
    pr, qr = DOCK_V1["contact_pose_dock"] if berth == 1 else DOCK_V1["berth2_pose_dock"]
    R = quat_to_mat(dock_quat)
    return np.asarray(dock_pos, float) + R @ np.asarray(pr, float), quat_mul(np.asarray(dock_quat, float), np.asarray(qr, float))


def attach_dock_v1(spec, dock_pos=(0.0, 0.0, 0.0), dock_quat=(1.0, 0.0, 0.0, 0.0), xml: str | None = None,
                   visual: bool = True, prefix: str = "") -> dict:
    """Attach the reference build's dock (the 'dock' body of its dock scene; the robot of that scene is dropped) to
    ``spec`` at (dock_pos, dock_quat).  The magnet carriers and the pin block become fixed display parts without
    contacts (their forces are ``BerthDock``'s).  visual=False drops the dock's display geoms.  Returns a summary."""
    path = xml or dock_model_path()
    if path is None:
        raise FileNotFoundError("no dock model: pass xml= or set SPACEKIT_DOCK_MJCF")
    src = mujoco.MjSpec.from_file(path)
    # reference build v1.2: the dock scene says in its 'refbuild' text whether its display parts are already
    # contact-free; the text (it also names the scene's robot, which is dropped here) is not attached to the host
    from .refbuild import build_info
    binfo = build_info(src)
    for t in list(src.texts):
        src.delete(t)
    core = src.body("core")
    if core is not None:
        src.delete(core)
    dock = src.body("dock")
    removed_j, silenced, dropped = [], [], 0
    for b in dock.find_all(mujoco.mjtObj.mjOBJ_BODY):
        for j in list(b.joints):
            removed_j.append(j.name)
            src.delete(j)
    for g in list(src.geoms):
        par = g.parent
        if par is None:
            continue
        moving = par.name != "dock" and par.name.startswith("dock_")
        # display hulls that are not contact surfaces: moving parts, pins, magnets, and the four berth captive-screw
        # heads (one convex hull spanning all four = an 87 mm slab 1.6 mm proud; the heads sit in the robot's relief
        # pockets)
        if (g.contype or g.conaffinity) and (moving or "pin_" in g.name or "magnet" in g.name or "captive_screw" in g.name):
            g.contype, g.conaffinity = 0, 0
            silenced.append(g.name)
        if not visual and not (g.contype or g.conaffinity):
            if not any(k in g.name for k in ("target",)):
                src.delete(g)
                dropped += 1
    used = {g.meshname for g in src.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for me in list(src.meshes):           # the dock scene also carries the robot's mesh assets
        if me.name not in used:
            src.delete(me)
    # the source's keyframes belong to its robot: re-parse without them (deleting them from the spec does not drop
    # the parser's pending keyframes, which would be merged into the host)
    src.meshdir = os.path.join(os.path.dirname(os.path.abspath(path)), src.meshdir or ".")
    src = mujoco.MjSpec.from_string(re.sub(r"<keyframe>.*?</keyframe>", "", src.to_xml(), flags=re.S))
    frame = spec.worldbody.add_frame(pos=list(dock_pos), quat=list(dock_quat))
    spec.attach(src, frame=frame, prefix=prefix)
    return dict(xml=path, removed_joints=removed_j, silenced_geoms=silenced, dropped_visual=dropped,
                dock_pose=(list(map(float, dock_pos)), list(map(float, dock_quat))),
                display_parts_in_model="berth_display_only" in binfo["corrections"], refbuild=binfo["text"])


def _box(body, name, lo, hi, quat=None, group=3, rgba=(0.6, 0.3, 0.3, 0.3), contact=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list((hi - lo) / 2)
    g.pos = list((hi + lo) / 2)
    if quat is not None:
        g.quat = list(quat)
    g.group = group
    g.rgba = list(rgba)
    g.mass = 0.0
    g.density = 0.0
    if contact:
        for k, v in contact.items():
            setattr(g, k, v)
    return g


def rig_dock_spec(spec, robot_root: str = "core", n_funnel: int = 16, funnel_t: float = 0.0012, prefix: str = "") -> dict:
    """Edit the robot's collision proxies at its dock face so the dock's lances can enter the cups (see module doc):
    the dock-plate proxy and the aft 38.5 mm of the mid-band proxy are replaced by boxes around two 27 mm square
    channels on the cup axes, each channel gets a funnel of ``n_funnel`` thin boxes on the cup cone, a ring on the
    13.2 mm through-hole and a face annulus between the cup mouth and the channel walls.  Returns a summary."""
    P = DOCK_V1
    root = spec.body(prefix + robot_root)
    names = {g.name: g for g in root.geoms}
    plate = names.get(prefix + "c_core_dock_plate")
    mid = names.get(prefix + "c_core_midband")
    # reference build v1.2: the cups are in the model (its 'refbuild' text, or the proxy boxes themselves);
    # nothing to do -- applying the edit again would double the dock-face proxy
    from .refbuild import has_correction
    cups = [g.name for g in root.geoms if g.name.startswith(tuple(prefix + k for k in ("c_core_midband_fwd", "c_core_dockface_", "c_core_cup")))]
    if plate is None and mid is None and (cups or has_correction(spec, "dock_face_cups")):
        if not cups:
            raise ValueError("the model says it has the dock-face cups, but the core carries none")
        x_face = float(P["dock_face_x"])
        return dict(removed=[], added=0, x_face=x_face, x_channel_end=x_face + P["cup_depth"] + 0.0156, names=cups,
                    already_in_model=True)
    if plate is None or mid is None:
        raise ValueError("robot dock-face proxies (c_core_dock_plate, c_core_midband) not found")
    ref = dict(friction=list(mid.friction), solref=list(mid.solref), solimp=list(mid.solimp), condim=mid.condim,
               contype=mid.contype, conaffinity=mid.conaffinity, priority=mid.priority)
    mlo = np.asarray(mid.pos) - np.asarray(mid.size)
    mhi = np.asarray(mid.pos) + np.asarray(mid.size)
    plo = np.asarray(plate.pos) - np.asarray(plate.size)
    phi = np.asarray(plate.pos) + np.asarray(plate.size)
    if np.any(np.abs(np.asarray(mid.quat) - [1, 0, 0, 0]) > 1e-9) or np.any(np.abs(np.asarray(plate.quat) - [1, 0, 0, 0]) > 1e-9):
        raise ValueError("dock-face proxies must be body-aligned boxes")
    x_face = float(P["dock_face_x"])                    # the measured face (the base proxy box face is 0.1 mm short of it)
    x_cut = x_face + P["cup_depth"] + 0.0156            # channel depth: past the lance tips at contact (body x -0.1099)
    spec.delete(plate)
    lo_mid = mlo.copy()
    lo_mid[0] = x_cut
    added = []
    added.append(_box(root, prefix + "c_core_midband_fwd", lo_mid, mhi, contact=ref).name)
    h = 0.0135                                          # half width of the square lance channels
    yz = P["cup_yz"]
    z1, z2 = sorted(c[1] for c in yz)
    for tag, lo, hi in (("aft", (x_face, mlo[1], mlo[2]), (x_cut, mhi[1], mhi[2])),):
        y0, y1, za, zb = lo[1], hi[1], lo[2], hi[2]
        parts = [((y0, za), (-h, zb)), ((h, za), (y1, zb)),                     # beside the channels
                 ((-h, z1 + h), (h, z2 - h)),                                    # between the channels
                 ((-h, z2 + h), (h, zb)), ((-h, za), (h, z1 - h))]               # above / below
        for k, ((ya, z_a), (yb, z_b)) in enumerate(parts):
            if yb - ya > 1e-4 and z_b - z_a > 1e-4:
                added.append(_box(root, prefix + f"c_core_dockface_{tag}{k}", (lo[0], ya, z_a), (hi[0], yb, z_b), contact=ref).name)
    tana = math.tan(math.radians(P["cup_half_angle_deg"]))
    r0, depth, rh = P["cup_mouth_r"], P["cup_depth"], P["hole_r"]
    L = depth / math.cos(math.radians(P["cup_half_angle_deg"]))
    for c, (cy, cz) in enumerate(yz):
        for i in range(n_funnel):
            phi_i = 2 * math.pi * (i + 0.5) / n_funnel
            u = np.array([0.0, math.cos(phi_i), math.sin(phi_i)])          # radial direction
            r_mid = r0 - 0.5 * depth * tana
            w = 2 * math.pi * (r0 + funnel_t) / n_funnel * 0.55             # half width of the slat (overlapping)
            # slat on the cone: axis along the generatrix (x and radial), thickness outward of the cone surface
            gdir = np.array([1.0, 0.0, 0.0]) * math.cos(math.radians(P["cup_half_angle_deg"])) - u * math.sin(math.radians(P["cup_half_angle_deg"]))
            ndir = np.cross(gdir, np.cross(u, gdir))
            ndir /= np.linalg.norm(ndir)                                    # outward normal of the cone surface
            tdir = np.cross(ndir, gdir)
            ctr = np.array([x_face + depth / 2 + 0.0004, cy, cz]) + u * (r_mid - 0.0004 * tana) + ndir * (funnel_t / 2)
            R = np.column_stack([gdir, tdir, ndir])
            if np.linalg.det(R) < 0:
                R[:, 1] = -R[:, 1]
            g = root.add_geom()
            g.name = prefix + f"c_core_cup{c}_funnel{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [L / 2 - 0.0008, w, funnel_t / 2]       # starts 0.4 mm inside the face (no contact with the plate)
            g.pos = list(ctr)
            g.quat = list(mat_to_quat(R))
            g.group, g.rgba, g.mass, g.density = 3, [0.8, 0.5, 0.2, 0.4], 0.0, 0.0
            for k, v in ref.items():
                setattr(g, k, v)
            added.append(g.name)
            # through-hole ring
            ctr2 = np.array([0.5 * (x_face + depth + x_cut), cy, cz]) + u * (rh + funnel_t / 2)
            R2 = np.column_stack([[1.0, 0, 0], np.cross(u, [1.0, 0, 0]), u])
            if np.linalg.det(R2) < 0:
                R2[:, 1] = -R2[:, 1]
            g2 = root.add_geom()
            g2.name = prefix + f"c_core_cup{c}_hole{i}"
            g2.type = mujoco.mjtGeom.mjGEOM_BOX
            g2.size = [(x_cut - x_face - depth) / 2, 2 * math.pi * (rh + funnel_t) / n_funnel * 0.55, funnel_t / 2]
            g2.pos = list(ctr2)
            g2.quat = list(mat_to_quat(R2))
            g2.group, g2.rgba, g2.mass, g2.density = 3, [0.8, 0.5, 0.2, 0.4], 0.0, 0.0
            for k, v in ref.items():
                setattr(g2, k, v)
            added.append(g2.name)
        # face annulus between the cup mouth and the square channel (the dock face around the cup)
        for i in range(n_funnel):
            phi_i = 2 * math.pi * (i + 0.5) / n_funnel
            u = np.array([0.0, math.cos(phi_i), math.sin(phi_i)])
            r_in = r0 + funnel_t
            r_out = h * math.sqrt(2.0) + 0.001
            ctr = np.array([x_face + 0.0020, cy, cz]) + u * 0.5 * (r_in + r_out)      # 0.5 mm behind the face
            R = np.column_stack([[1.0, 0, 0], np.cross(u, [1.0, 0, 0]), u])
            if np.linalg.det(R) < 0:
                R[:, 1] = -R[:, 1]
            g3 = root.add_geom()
            g3.name = prefix + f"c_core_cup{c}_face{i}"
            g3.type = mujoco.mjtGeom.mjGEOM_BOX
            g3.size = [0.0015, 2 * math.pi * r_out / n_funnel * 0.55, 0.5 * (r_out - r_in)]
            g3.pos = list(ctr)
            g3.quat = list(mat_to_quat(R))
            g3.group, g3.rgba, g3.mass, g3.density = 3, [0.6, 0.3, 0.3, 0.3], 0.0, 0.0
            for k, v in ref.items():
                setattr(g3, k, v)
            added.append(g3.name)
    spec.delete(mid)
    return dict(removed=[prefix + "c_core_dock_plate", prefix + "c_core_midband"], added=len(added), x_face=x_face,
                x_channel_end=x_cut, names=added, already_in_model=False)


class BerthDock:
    """Wrench source: magnets and pogo pins of one berth, with the magnet retraction (see module doc).

    berth_pose: (pos, quat) of the robot body resting on the berth (``berth_contact_pose``); the berth axis x points
    out of the dock (egress).  The robot's strike plates and pin pads are taken at the robot body's dock face
    (``DOCK_V1['dock_face_x']``) at the magnet / berth-axis offsets; their displacement from the contact pose along the
    axis sets the gaps.  Interface used by the primitives (as ``primitives.DockMagnets``): ``gap`` (robot origin
    along the axis from the contact pose), ``lateral``, ``force`` (total magnet pull), ``retract(t, s)``,
    plus ``pins_N``, ``carrier`` (retraction [m]), ``eps_contact`` (the dock sees the robot: pins compressed)."""

    def __init__(self, robot_body: int, berth_pose, params: dict | None = None, enabled: bool = True, far_skip: bool = True):
        self.P = dict(DOCK_V1, **(params or {}))
        self.far_skip = bool(far_skip)
        self.n_skipped = 0
        self.b = int(robot_body)
        self.p = np.asarray(berth_pose[0], float)
        self.R = quat_to_mat(np.asarray(berth_pose[1], float))
        self.x = self.R[:, 0]
        P = self.P
        xf = P["dock_face_x"]
        self.mag_pts = np.array([[xf, y, z] for y, z in P["magnet_yz_m"]])     # robot body frame (at the contact pose
        self.pin_pt = np.array([xf, 0.0, 0.0])                                    #  the body frame = the berth frame)
        self._rq_mag = float(np.max(np.linalg.norm(self.mag_pts, axis=1)))
        self._rq_pin = float(np.linalg.norm(self.pin_pt))
        self.enabled = enabled
        self.carrier = 0.0               # retraction [m]
        self._retract_cmd = False
        self._extend_at = None
        self._t_last = None
        self.gap, self.lateral, self.force, self.pins_N = np.inf, np.inf, 0.0, 0.0
        self.mag_N = np.zeros(4)
        self.eps_contact = False

    def retract(self, t: float, seconds: float | None = None) -> None:
        """Retract the magnets now; they re-extend ``seconds`` later (default ``hold_s``)."""
        self._retract_cmd = True
        self._extend_at = float(t) + float(self.P["hold_s"] if seconds is None else seconds)

    def _advance_carrier(self, t: float) -> None:
        if self._t_last is None:
            self._t_last = t
        dt = max(0.0, t - self._t_last)
        self._t_last = t
        if self._extend_at is not None and t >= self._extend_at:
            self._retract_cmd, self._extend_at = False, None
        tgt = self.P["retract_m"] if self._retract_cmd else 0.0
        step = self.P["retract_speed_m_s"] * dt
        self.carrier += float(np.clip(tgt - self.carrier, -step, step))

    def geometry(self, data):
        d = data.xpos[self.b] - self.p
        g = float(d @ self.x)
        lat = float(np.linalg.norm(d - g * self.x))
        return g, lat

    def _out_of_reach(self, Rb) -> bool:
        """Exact early-out (v0.4): True when no magnet can be within its 0.06 m range or overlap its strike plate and
        the pins cannot be compressed, whatever the attitude -- every robot-side point q moves at most 2 sin(theta / 2)
        |q| from its contact position relative to the body origin's displacement (theta = attitude difference from the
        contact pose).  The skipped evaluation would have added zero force and torque."""
        P = self.P
        tr = float(np.einsum("ij,ij->", self.R, Rb))
        s2 = math.sqrt(max(0.0, 3.0 - tr))               # 2 sin(theta / 2)
        g, lat = self.gap, self.lateral
        eps = 1e-9
        mag_far = (P["working_gap_m"] + g - s2 * self._rq_mag > 0.06 + eps) or (lat - s2 * self._rq_mag >= 2 * P["magnet_radius_m"] + eps)
        pin_far = (g - s2 * self._rq_pin >= P["pins_working_compression_m"] + eps) or (lat - s2 * self._rq_pin >= 0.02 + eps)
        return mag_far and pin_far

    def accumulate(self, model, data, W) -> None:
        self._advance_carrier(float(data.time))
        self.gap, self.lateral = self.geometry(data)
        self.force, self.pins_N = 0.0, 0.0
        self.mag_N[:] = 0.0
        self.eps_contact = False
        if not self.enabled:
            return
        P = self.P
        Rb = data.xmat[self.b].reshape(3, 3)
        if self.far_skip and self._out_of_reach(Rb):
            self.n_skipped += 1
            z = np.zeros(3)
            W[self.b, 0:3] += z                         # the same (zero) update as the full evaluation
            W[self.b, 3:6] += z
            return
        pb = data.xpos[self.b]
        com = data.xipos[self.b]                 # xfrc_applied acts at the body's own CoM (as PropulsionWrench)
        Ft = np.zeros(3)
        Tt = np.zeros(3)
        rm = P["magnet_radius_m"]
        for j, q in enumerate(self.mag_pts):
            w = pb + Rb @ q                                   # strike plate point (world)
            rel = w - (self.p + self.R @ q)                   # its displacement from the contact position (berth frame)
            g = P["working_gap_m"] + float(rel @ self.x) + self.carrier
            if g > 0.06:
                continue
            g = max(g, 0.0)
            lat = float(np.linalg.norm(rel - (rel @ self.x) * self.x))
            ov = _disc_overlap(lat, rm) if lat < 2 * rm else 0.0
            F = P["magnet_peak_N"] / (1.0 + g / P["magnet_g1_m"]) ** 2 * ov
            if F <= 0:
                continue
            f = -F * self.x
            Ft += f
            Tt += np.cross(w - com, f)
            self.mag_N[j] = F
            self.force += F
        w = pb + Rb @ self.pin_pt
        rel = w - (self.p + self.R @ self.pin_pt)
        comp = P["pins_working_compression_m"] - float(rel @ self.x)
        lat = float(np.linalg.norm(rel - (rel @ self.x) * self.x))
        if comp > 0 and lat < 0.02:
            comp = min(comp, P["pins_travel_m"])
            vb = data.cvel[self.b, 3:6] + np.cross(data.cvel[self.b, 0:3], w - com)   # point velocity (cvel at the
            F = max(0.0, P["pins_k_N_m"] * comp - P["pins_damping_N_s_m"] * float(vb @ self.x) * -1.0)   # subtree CoM)
            f = F * self.x
            Ft += f
            Tt += np.cross(w - com, f)
            self.pins_N = F
            self.eps_contact = comp >= P["eps_contact_m"]
        W[self.b, 0:3] += Ft
        W[self.b, 3:6] += Tt

    def net_hold_N(self) -> float:
        """Magnet pull minus pin push of the last evaluation [N]."""
        return self.force - self.pins_N


def _disc_overlap(d: float, r: float) -> float:
    """Overlap area fraction of two equal discs of radius r at centre distance d."""
    if d >= 2 * r:
        return 0.0
    if d <= 0:
        return 1.0
    a = 2 * r * r * math.acos(d / (2 * r)) - 0.5 * d * math.sqrt(max(0.0, 4 * r * r - d * d))
    return a / (math.pi * r * r)


def fiducial_visible(cam_pos, cam_axis, points, normal, half_fov_deg: float = 40.0, range_m=(0.05, 1.5),
                     max_incidence_deg: float = 75.0, min_points: int = 3):
    """Analytic visibility of a planar fiducial given as marker points (N, 3) with the plate normal: the points inside
    the camera's view cone and range, on a plate facing the camera.  Returns (visible, distance to the nearest
    visible point [m] or None)."""
    P = np.atleast_2d(np.asarray(points, float))
    v = P - np.asarray(cam_pos, float)
    dist = np.linalg.norm(v, axis=1)
    cosang = (v @ np.asarray(cam_axis, float)) / np.maximum(dist, 1e-9)
    inc = (-v @ np.asarray(normal, float)) / np.maximum(dist, 1e-9)
    ok = ((dist >= range_m[0]) & (dist <= range_m[1]) & (cosang >= math.cos(math.radians(half_fov_deg)))
          & (inc >= math.cos(math.radians(max_incidence_deg))))
    if int(ok.sum()) < min_points:
        return False, None
    return True, float(dist[ok].min())


def camera_in_body(model, cam_name: str, body: int):
    """(offset (3,), optical axis (3,)) of a model camera in the frame of ``body`` (MuJoCo cameras look along -z)."""
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    mujoco.mj_camlight(model, d)
    c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if c < 0:
        raise KeyError(cam_name)
    Rb = d.xmat[body].reshape(3, 3)
    Rc = d.cam_xmat[c].reshape(3, 3)
    return Rb.T @ (d.cam_xpos[c] - d.xpos[body]), Rb.T @ (-Rc[:, 2])


def plate_points(model, data, geom_name: str, normal_hint, nu: int = 7, nv: int = 3):
    """Marker points on the face of a box / mesh-hull geom (its bounding box) whose normal is closest to
    ``normal_hint`` (world), and that normal."""
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if g < 0:
        raise KeyError(geom_name)
    R = data.geom_xmat[g].reshape(3, 3)
    hint = np.asarray(normal_hint, float)
    ax = int(np.argmax(np.abs(R.T @ hint)))
    sg = 1.0 if float(R[:, ax] @ hint) > 0 else -1.0
    c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    o = [a for a in range(3) if a != ax]
    pts = []
    for i in range(nu):
        for j in range(nv):
            loc = np.array(c, float)
            loc[ax] += sg * h[ax]
            loc[o[0]] += (-1 + 2 * (i + 0.5) / nu) * h[o[0]]
            loc[o[1]] += (-1 + 2 * (j + 0.5) / nv) * h[o[1]]
            pts.append(data.geom_xpos[g] + R @ loc)
    return np.array(pts), R[:, ax] * sg


class DockRelativeLocalizer:
    """Map localiser plus a fiducial-relative mode (see module doc).

    base: a ``sensors.Localizer`` (the map-based estimate; None = truth).  ``reference(name, points, normal,
    cam_offset_body, cam_axis_body)`` starts using a fiducial (world marker points of its plate and the plate normal)
    seen by a camera at ``cam_offset_body`` looking along ``cam_axis_body`` (robot body frame); ``release()`` returns to
    the map.  rmse_m: RMSE of the position error norm at ``rmse_range_m`` (1 cm at 0.6 m), proportional to range,
    floor ``floor_m``; att_rmse_rad likewise (the reported covariance); ``update`` has the Localizer signature; its dict
    adds ``source`` ('map' | 'fiducial:<name>' | 'fiducial(hold):<name>' | 'map(no view)').  The base localiser keeps
    running underneath.

    error_model (v0.4):
      'consistent' (default) -- the same model as the map localiser's (``sensors.Localizer``): per axis the error is
        the reported std at the current range times a unit-variance, critically damped second-order Gauss-Markov state
        that runs in an error clock ds/dt = 1 / corr_time + |v| / corr_length + |w| / corr_angle (corr_time,
        corr_length, corr_angle [G], the map's defaults), so a robot holding a pose on a plate sees a nearly constant
        error that shrinks and grows with the range instead of fresh noise every tick; the velocity / rate estimates
        carry the error's own derivative (plus the small independent noise of v0.3).  No fiducial in view within
        ``hold_s`` of the last fix: the position error integrates an Ornstein-Uhlenbeck velocity error (correlation
        time vio_tau [G], long-run diffusion ``hold_drift``), the attitude error is held; back in view the held error
        is carried over and re-converges with the clock at least 2 / tau for 2 tau.
      'legacy' -- v0.3 (bit-identical): first-order Gauss-Markov error with correlation time ``tau`` (1 s), random
        walk while holding, velocity / rate noise independent of it.
    """

    ERROR_MODELS = ("consistent", "legacy")

    def __init__(self, rng: np.random.Generator, base=None, rmse_m: float = 0.010, rmse_range_m: float = 0.6,
                 floor_m: float = 0.002, att_rmse_rad: float = math.radians(0.5), tau: float = 1.0,
                 half_fov_deg: float = 40.0, range_m=(0.05, 1.5), hold_s: float = 20.0, hold_drift: float = 1e-6,
                 error_model: str = "consistent", corr_time: float = 60.0, corr_length: float | None = 1.0,
                 corr_angle: float | None = 1.0, vio_tau: float = 10.0):
        if error_model not in self.ERROR_MODELS:
            raise ValueError(f"error_model {error_model!r} not in {self.ERROR_MODELS}")
        self.rng, self.base = rng, base
        self.hold_s, self.hold_q = float(hold_s), float(hold_drift)     # after the last fix: inertial hold, then map
        self.t_fix = None
        self.rmse, self.r0, self.floor, self.att_rmse, self.tau = rmse_m, rmse_range_m, floor_m, att_rmse_rad, tau
        self.half_fov, self.range = half_fov_deg, range_m
        self.ref = None
        self.e_pos = np.zeros(3)
        self.e_att = np.zeros(3)
        self.dt = 1.0 / 62.5
        self.forced_lost = False
        self.last_source = "map"
        self.last_range = None
        self.error_model = error_model
        self.corr_time, self.corr_length, self.corr_angle, self.vio_tau = corr_time, corr_length, corr_angle, vio_tau
        # consistent model: unit state (row 0 the value, row 1 its derivative in clock units) of the 6 error axes
        # (position x y z, attitude x y z), the reported per-axis std at the previous fix (None: no fix since the last
        # reference / hold), the error's rate d(e_pos, e_att)/dt, the hold's velocity error, the re-acquisition time
        self._z = np.zeros((2, 6))
        self._g = None
        self.de = np.zeros(6)
        self._u = None
        self._t_reacq = None
        self.last_clock = None

    def reference(self, name: str, points, normal, cam_offset_body, cam_axis_body) -> None:
        """Use the fiducial ``name`` (see class doc)."""
        sp, sa = self._sigma(self.r0)
        self.ref = dict(name=name, points=np.atleast_2d(np.asarray(points, float)), normal=np.asarray(normal, float),
                        cam=np.asarray(cam_offset_body, float), axis=np.asarray(cam_axis_body, float))
        if self.error_model == "legacy":
            self.e_pos = self.rng.normal(0.0, sp, 3)
            self.e_att = self.rng.normal(0.0, sa, 3)
        else:
            # a fresh fiducial: the unit state from its stationary distribution (value and derivative)
            self._z = self.rng.normal(0.0, 1.0, (2, 6))
            self.e_pos, self.e_att = sp * self._z[0, 0:3], sa * self._z[0, 3:6]
            self._g, self._u, self._t_reacq = None, None, None
            self.de = np.zeros(6)
        self.t_fix = None

    def release(self) -> None:
        """Back to the map localisation."""
        self.ref = None
        self.t_fix = None
        self._g, self._u, self._t_reacq = None, None, None

    def clock_rate(self, vel, om) -> float:
        """ds/dt of the error clock (consistent model, as ``sensors.Localizer.clock_rate``) [1/s]."""
        w0 = 1.0 / self.corr_time
        if self.corr_length:
            w0 += float(np.linalg.norm(vel)) / self.corr_length
        if self.corr_angle:
            w0 += float(np.linalg.norm(om)) / self.corr_angle
        return w0

    def _fix_consistent(self, t, dist, vel, om) -> None:
        """One tick with the fiducial in view (consistent model): e = g(range) z, z a GM2 state in the error clock."""
        from .sensors import Localizer
        sp, sa = self._sigma(dist)
        g1 = np.array([sp, sp, sp, sa, sa, sa])
        sd = self.clock_rate(vel, om)
        if self._g is None:                     # first fix after reference(): the unit state as drawn, no rate jump
            g0, z = g1, self._z
        elif self._u is not None:               # back in view after an inertial hold: carry the held error over
            g0 = g1
            z = np.empty((2, 6))
            z[0] = np.concatenate([self.e_pos, self.e_att]) / g1
            z[1] = np.clip(self.de / (max(sd, 2.0 / self.tau) * g1), -3.0, 3.0)
            self._u, self._t_reacq = None, t
        else:
            g0, z = self._g, self._z
        if self._t_reacq is not None:
            if t - self._t_reacq < 2.0 * self.tau:
                sd = max(sd, 2.0 / self.tau)    # re-convergence on the fiducial (the v0.3 correlation time)
            else:
                self._t_reacq = None
        self.last_clock = sd
        Phi, Lc = Localizer._gm2_step(sd * self.dt)
        z = Phi @ z + Lc @ self.rng.normal(0.0, 1.0, (2, 6))
        e, ds = g1 * z[0], g1 * z[1]
        self.de = ds * sd + z[0] * (g1 - g0) / self.dt                     # d(g z)/dt
        self.e_pos, self.e_att = e[0:3], e[3:6]
        self._z, self._g = z, g1

    def _hold_consistent(self) -> None:
        """One tick of the inertial hold (consistent model): the position error integrates an OU velocity error."""
        su = math.sqrt(self.hold_q / (2.0 * self.vio_tau))
        if self._u is None:                     # entering the hold: the velocity error from its stationary law
            self._u = su * self.rng.normal(0.0, 1.0, 3)
        a = math.exp(-self.dt / self.vio_tau)
        u1 = a * self._u + math.sqrt(max(0.0, 1.0 - a * a)) * su * self.rng.normal(0.0, 1.0, 3)
        self.e_pos = self.e_pos + 0.5 * self.dt * (self._u + u1)
        self._u = u1
        self.de = np.concatenate([u1, np.zeros(3)])
        self.last_clock = None

    def _sigma(self, dist):
        k = max(float(dist) / self.r0, 0.0)
        sp = max(self.rmse * k, self.floor) / math.sqrt(3.0)
        sa = max(self.att_rmse * k, self.att_rmse * self.floor / self.rmse) / math.sqrt(3.0)
        return sp, sa

    def update(self, t, model, data, body: int, occlusion=None) -> dict:
        from .sensors import true_state
        base = self.base.update(t, model, data, body, occlusion) if self.base is not None else None
        if self.ref is None:
            self.last_source = "map"
            return base if base is not None else self._truth(model, data, body)
        pos, quat, vel, om = true_state(model, data, body)
        R = quat_to_mat(quat)
        vis, dist = fiducial_visible(pos + R @ self.ref["cam"], R @ self.ref["axis"], self.ref["points"], self.ref["normal"],
                                     self.half_fov, self.range)
        legacy = self.error_model == "legacy"
        if not vis:
            if self.t_fix is None or t - self.t_fix > self.hold_s:
                self.last_source = "map(no view)"
                return base if base is not None else self._truth(model, data, body)
            # no fiducial in view since a recent fix: the estimator coasts on its inertial / visual odometry
            if legacy:
                self.e_pos = self.e_pos + self.rng.normal(0.0, math.sqrt(self.hold_q * self.dt), 3)
            else:
                self._hold_consistent()
            src = "fiducial(hold):" + self.ref["name"]
            sp = sa = None
        else:
            if legacy:
                sp, sa = self._sigma(dist)
                a = math.exp(-self.dt / self.tau)
                q = math.sqrt(1.0 - a * a)
                self.e_pos = a * self.e_pos + q * self.rng.normal(0.0, sp, 3)
                self.e_att = a * self.e_att + q * self.rng.normal(0.0, sa, 3)
            else:
                self._fix_consistent(t, dist, vel, om)
                sp, sa = self._sigma(dist)
            self.t_fix = t
            src = "fiducial:" + self.ref["name"]
        if sp is None:
            sp, sa = self._sigma(self.r0)
        if legacy:
            v_est, w_est = vel + self.rng.normal(0.0, 0.001, 3), om + self.rng.normal(0.0, 2e-4, 3)
        else:
            # the error's own derivative (body rate of q (x) exp(e_att): w + w x e_att + d(e_att)/dt) + the v0.3 noise
            v_est = vel + self.de[0:3] + self.rng.normal(0.0, 0.001, 3)
            w_est = om + self.de[3:6] + np.cross(om, self.e_att) + self.rng.normal(0.0, 2e-4, 3)
        out = dict(pos=pos + self.e_pos, quat=quat_mul(quat, quat_from_rotvec(self.e_att)),
                   vel=v_est, omega=w_est,
                   cov_pos=np.eye(3) * sp * sp, cov_att=np.eye(3) * sa * sa,
                   confidence=2 if self.forced_lost else 0, richness=1.0, pos_error=float(np.linalg.norm(self.e_pos)),
                   att_error=float(np.linalg.norm(self.e_att)), source=src)
        self.last_source, self.last_range = out["source"], dist
        return out

    def _truth(self, model, data, body):
        from .sensors import true_state
        pos, quat, vel, om = true_state(model, data, body)
        return dict(pos=pos, quat=quat, vel=vel, omega=om, cov_pos=np.eye(3) * 1e-8, cov_att=np.eye(3) * 1e-8, confidence=0,
                    richness=1.0, pos_error=0.0, att_error=0.0, source="truth")
