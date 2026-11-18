"""The cell-side world of one site (task 07; DESIGN v1.2 sections 1.4, 2.1-2.3, 7, 13; package P08).

Built from a site package (subseacell.site) out of the devices of packages P04-P07 into THREE models (subseacell.twin):
  physics model    the digest vehicle (sea trim, every convex piece + jaw shells, no appearance geometry; P05), the
                   stand-in manifold's collision geometry (mudmats, frame, roof proxy plate, headers, hubs, branches,
                   flanges) + the roof grating bars (no collision), the panel skid (face, frame, guard rail, 8 port
                   receptacles, 18 hinged valves; P05), the hot stab and its receptacle sliders (P07), the marker block in
                   the jaws (P05), the dropped-object stand with its Morison hydrodynamics (P07), the deployment cage
                   (P07); the site's current (P06) drives the vehicle, the free objects and the plume.
                   Sea bed (P24a, decision P19-1): the kit's sea-bed PLANE stays for the contacts (MuJoCo planes are
                   infinite for collisions) but is hidden from every ray (group 3); the rays of the DVL, the sonar and
                   the camera see SEABED_BOX instead, an 800 x 800 x 1 m non-colliding box centred under the structure
                   whose top is the plane (a plane answers rays only within its size, and mj_multiRay culls it by its
                   centre point).  Roof (P24a, decision P18-3): the DVL sees the roof proxy plate (group 0, opaque,
                   acoustic class steel), whose top is the top of the grating bars, so the grating returns like one
                   surface; the bars stay for the camera and the sonar.
  render twin      the camera's model (vehicle appearance meshes, textured tag plates, paint), posed from the physics
                   qpos only when a frame is taken.
  acoustic model   the scanning sonar's model (structure-level geometry + mocap copies of the moving bodies), posed
                   only when a ping is taken.  Amendment 01 (decision CAL-2): the deployment cage with its two sonar
                   reflector poles is in it (cage_acoustic: mirrored like a moving body, it is a body of its own).
  static scenery   (amendment 02, decisions AM2-1 to AM2-6; subseacell.scenery) the site package's scenery block -
                   rocks, the production flowline with its tie-in spool and pipe supports, the test flowline, concrete
                   mattresses - is world geometry in all three models at the same pose: colliders in static zone
                   bodies (contact parameters of the sea bed), drawn by the render twin, sonar targets in the acoustic
                   model (rocks rebuilt from the same OBJ), acoustic classes for the DVL and the sonar.  A site package
                   without the block gives the same three models as before, bit for bit.
Runtime: the firmware (P04: neutral modes, degradation, protective behaviours), the grasp rules (P05), the hot-stab
state machine (P07), the process graph and the plume with its sonar / camera couplings (P06).  Every random stream is
seeded from the site; time passes only in advance(); the same site and seed give the same qpos history, event log,
frames and pings bit for bit.

    W = SiteWorld(site)                         # site: subseacell.site.Site, a package path or its bytes
    W.fw.set_hold(...); W.advance(1.0)          # motion calls go to the firmware (P09 wraps them into the API)
    fr = W.frame((640, 480)); p = W.ping()      # camera frame (render twin) / one sonar ping (acoustic model)
    W.gripper_close(); W.set_lights(1.0); W.set_camera_tilt(rad)
    W.state()                                   # the firmware's state() reply (public keys)
    W.truth(); W.events; W.fingerprint()        # grader side: true state, event log, determinism hashes

Physics step order (the phase 1 worlds' order, P04 / P05 / P07 interface notes):
    every control period (10 ms): hooks(world, t) [program-side controllers of tests / the reference]; fw.control(t)
    bank.tick(); gripper.tick_pre(); stab.tick_pre(); sim.step()          [the stand's hydrodynamics inside sim.step]
    fw.ingest(suite.step(t)); gripper.tick_post(); stab.tick_post(); fw.physics_tick(t)   [protective behaviours]
    every 0.1 s: process graph (valve angles -> conduction -> segment pressures), plume, valve history, qpos hash
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import mujoco

import uwkit
from uwkit.sensors.suite import SensorSuite, SensorWorld, SuiteConfig
from uwkit.sensors.acoustics import WaterColumn
from uwkit.sensors.sonar import ScanningSonarConfig
from uwkit.sensors import water as UWAT

from . import vehicle as VH
from . import devices as DV
from . import grasp as GR
from . import stab as ST
from . import objects as OB
from . import protect as PT
from . import firmware as FW
from . import flow as FL
from . import process as PR
from . import plume as PL
from . import sensing as SE
from . import site as SITE
from . import twin as TW
from . import scenery as SCN

WATER_TEMPERATURE_C = 10.0
WATER_SALINITY_PSU = 35.0
SLOW_DT_S = 0.1                          # process / plume grid (process.update_hz = plume.step_hz = 10)
LAYDOWN_LINE_R = 0.02
SEABED_PLANE = "seabed"                  # the kit's sea-bed plane (uwkit build_vehicle(floor_depth=...)): contacts only
SEABED_BOX = "seabed_box"                # what the rays see of the sea bed (P24a): non-colliding, top = the plane
SEABED_BOX_HALF_M = (400.0, 400.0, 0.5)  # 800 x 800 x 1 m, centred under the structure origin
ROOF_PROXY_ACLS = "steel"                # the DVL's acoustic class of the roof proxy plate (the grating is steel)
ROOF_PROXY_RGBA = (0.85, 0.68, 0.12, 1.0)  # physics model only (never drawn): alpha > 0 so that rays meet the plate
GRIPPER_PUBLIC = {"open": "open", "closing": "open", "opening": "open", "closed_empty": "closed_empty",
                  "stalled_on_object": "stalled_on_object"}
CAGE_ACLS = "steel"                      # the cage's steel plates and its two reflector poles (sonar)
AC_PREFIX = "ac_"                        # subseacell.twin.AcousticModel names a mirrored body's geoms ac_<physics name>


def cage_acoustic(model, cage: dict):
    """The deployment cage in the sonar's structural model (amendment 01, decision CAL-2 of DECISIONS_P2_BEH.md 11):
    objects.add_cage builds it as a body of its own (not body 0), so the static copy of acoustic_scene() never saw it.
    It is mirrored like the moving bodies (subseacell.twin.AcousticModel copies every colliding geom of the body and
    poses it from the physics data before each ping; the cage never moves, so the copy stays where the physics model
    has it).  Returns (moving entry, {acoustic-model geom name: class}): the deck pads keep the class the physics model
    gives them (objects.add_cage acoustic_overrides), every other cage geom - plates, columns, the two poles - steel."""
    bid = model.body(cage["name"]).id
    over = cage.get("acoustic_overrides", {})
    classes = {}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid or (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0):
            continue
        n = model.geom(g).name
        classes[AC_PREFIX + n] = over.get(n, CAGE_ACLS)
    return dict(body=cage["name"]), classes


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


class Timer:
    """Thread-CPU accounting per component (the cell is single-threaded; frames also use the process clock because
    the software rasteriser may run threads)."""
    __slots__ = ("cpu",)

    def __init__(self):
        self.cpu = {}

    def add(self, k, v):
        self.cpu[k] = self.cpu.get(k, 0.0) + v


# ======================================================================================================= fast paths
# Per-step implementations of three device rules that give the same numbers with less interpreter work (the cell's
# budget is 0.33 CPU s per simulated second on one core; DESIGN 13).  Semantics are the owners' (P05 / P07);
# tests/test_world.py checks each against the original implementation.
class FastSiteField(FL.CUR.CurrentField):
    """The site current (subseacell.flow.StationFlow.field) evaluated in plain floats for the few points the vehicle's
    and the free objects' hydrodynamics ask for every physics step (5); more points go to the original field.  Same
    model and constants (tide + gust, 1/7 boundary layer normalised at z_ref, in-structure factor), equal to the
    original to rounding (tests/test_world.py); the gust process is the original object (one sample grid)."""

    SMALL = 12

    def __init__(self, flow: FL.StationFlow):
        self.flow = flow
        self.full = flow.field
        bl = flow.free
        self.tide = flow.tide
        self.gust = flow.gust
        self.seabed = float(bl.seabed)
        self.delta, self.k = float(bl.delta), float(bl.k)
        self.zref = bl.z_ref
        self.norm = float(bl._norm)
        env, pose = flow.env, flow.pose
        self.o = [float(x) for x in pose.o]
        self.R = [float(x) for x in pose.R.reshape(-1)]
        self.hx, self.hy, self.ramp = env.lx / 2.0, env.ly / 2.0, env.ramp_m
        self.top, self.bot = env.roof_z, -env.mudmat_thk
        self.f_in = float(flow.fs.f_inside)

    @property
    def needs_rng(self):
        return False

    def velocity(self, pos, t):
        p = np.asarray(pos, float)
        if p.ndim == 1:
            p = p.reshape(1, 3)
        n = p.shape[0]
        if n > self.SMALL:
            return self.full.velocity(p, t)
        h = self.tide.horizontal(t)
        g = self.gust.velocity(p[:1], t)[0]
        hm = self.tide.mean
        vx, vy, vz = (float(h[0]) + float(hm[0])) + float(g[0]), (float(h[1]) + float(hm[1])) + float(g[1]), \
            (0.0 + float(hm[2])) + float(g[2])
        R, o = self.R, self.o
        out = np.empty((n, 3))
        for i, (x, y, z) in enumerate(p.tolist()):
            hb = self.seabed + z
            if hb < 0.0:
                hb = 0.0
            f = (hb / self.delta) ** self.k if hb < self.delta else 1.0
            if self.zref is not None:
                f = f / self.norm
            dx, dy, dz = x - o[0], y - o[1], z - o[2]
            q0 = dx * R[0] + dy * R[3] + dz * R[6]
            q1 = dx * R[1] + dy * R[4] + dz * R[7]
            q2 = dx * R[2] + dy * R[5] + dz * R[8]
            ax = min(max((self.hx - abs(q0)) / self.ramp + 0.5, 0.0), 1.0)
            ay = min(max((self.hy - abs(q1)) / self.ramp + 0.5, 0.0), 1.0)
            az = min(max((self.top - q2) / self.ramp + 0.5, 0.0), 1.0) * min(max((q2 - self.bot) / self.ramp + 0.5, 0.0), 1.0)
            s = 1.0 - (1.0 - self.f_in) * (ax * ay * az)
            out[i, 0] = vx * f * s
            out[i, 1] = vy * f * s
            out[i, 2] = vz * f * s
        return out


class FastValveBank(DV.ValveBank):
    """subseacell.devices.ValveBank with tick() vectorised over the valves (same arithmetic, same order of the
    comparisons, so the hinge torques and frictions are bit-identical)."""

    def __init__(self, model, data, infos):
        super().__init__(model, data, infos)

    def _pull(self):
        V = list(self.v.values())
        self._qa = np.array([s["qa"] for s in V], np.int64)
        self._da = np.array([s["da"] for s in V], np.int64)
        self._brk = np.array([float(s["spec"].breakaway_Nm) for s in V])
        self._run = np.array([float(s["spec"].running_Nm) for s in V])
        self._det = np.array([float(s["detent"]) for s in V])
        self._zone = np.array([float(s["zone"]) for s in V])
        self._rest = np.array([float(s["rest_t"]) for s in V])
        self._stuck = np.array([bool(s["stuck"]) for s in V])

    def reset(self):
        super().reset()
        self._pull()

    def tick(self, dt=None):
        m, d = self.m, self.d
        dt = float(m.opt.timestep if dt is None else dt)
        band, rate = math.radians(DV.VALUES["rearm_within_deg"]), math.radians(DV.VALUES["rearm_rate_deg_s"])
        rest_s = DV.VALUES["rearm_rest_s"]
        q = d.qpos[self._qa]
        w = d.qvel[self._da]
        near = np.minimum(np.abs(q - self.lo), np.abs(q - self.hi)) < band
        self._rest = np.where(near & (np.abs(w) < rate), self._rest + dt, 0.0)
        unstick = self._stuck & ~near
        stick = (~self._stuck) & near & (self._rest >= rest_s)
        self._stuck = (self._stuck & ~unstick) | stick
        m.dof_frictionloss[self._da] = np.where(self._stuck, self._brk, self._run)
        tau = np.where(q < self.lo + self._zone, -self._det, np.where(q > self.hi - self._zone, self._det, 0.0))
        d.qfrc_applied[self._da] = tau

    def push_state(self):
        """Write the vectorised state back into the per-valve dicts (for readers of ValveBank.v)."""
        for k, s in enumerate(self.v.values()):
            s["rest_t"], s["stuck"] = float(self._rest[k]), bool(self._stuck[k])


class FastGripper(GR.Gripper):
    """subseacell.grasp.Gripper with weld_load() reading only the held weld's rows (J^T f restricted to the hull's six
    DOFs) instead of J^T over every constraint row, 3-vector algebra in plain floats; same formula, equal to the
    original to rounding (tests/test_world.py)."""

    def weld_load(self):
        m, d = self.m, self.d
        if self._sparse:
            return super().weld_load()
        h = self.handles[self.held[1]]
        eq = h["eq"]
        rows = np.flatnonzero(d.efc_id[:int(d.ne)] == eq)
        nv = m.nv
        J = d.efc_J[:d.nefc * nv].reshape(d.nefc, nv)
        r6 = (d.efc_force[rows] @ J[rows, self.vadr:self.vadr + 6]).tolist()
        F0, F1, F2, w0, w1, w2 = r6
        R = d.xmat[self.hull].tolist()                       # row-major 3 x 3
        Mo0 = R[0] * w0 + R[1] * w1 + R[2] * w2
        Mo1 = R[3] * w0 + R[4] * w1 + R[5] * w2
        Mo2 = R[6] * w0 + R[7] * w1 + R[8] * w2
        b0, b1, b2 = self._axis
        a0 = R[0] * b0 + R[1] * b1 + R[2] * b2
        a1 = R[3] * b0 + R[4] * b1 + R[5] * b2
        a2 = R[6] * b0 + R[7] * b1 + R[8] * b2
        xH = d.xpos[self.hull].tolist()
        tc = d.site_xpos[self.tcp_sid].tolist()
        ta = self.tip_ahead
        r0, r1, r2 = tc[0] + a0 * ta - xH[0], tc[1] + a1 * ta - xH[1], tc[2] + a2 * ta - xH[2]
        Mt0 = Mo0 - (r1 * F2 - r2 * F1)
        Mt1 = Mo1 - (r2 * F0 - r0 * F2)
        Mt2 = Mo2 - (r0 * F1 - r1 * F0)
        ax = -(F0 * a0 + F1 * a1 + F2 * a2)
        tor = -(Mt0 * a0 + Mt1 * a1 + Mt2 * a2)
        l0, l1, l2 = -F0 - ax * a0, -F1 - ax * a1, -F2 - ax * a2
        n0, n1, n2 = -Mt0 - tor * a0, -Mt1 - tor * a1, -Mt2 - tor * a2
        ed = m.eq_data[eq].tolist()
        p1 = [xH[i] + R[3 * i] * ed[3] + R[3 * i + 1] * ed[4] + R[3 * i + 2] * ed[5] for i in range(3)]
        bb = h["bid"]
        xb = d.xpos[bb].tolist()
        Rb = d.xmat[bb].tolist()
        p2 = [xb[i] + Rb[3 * i] * ed[0] + Rb[3 * i + 1] * ed[1] + Rb[3 * i + 2] * ed[2] for i in range(3)]
        dp = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        return dict(lateral=math.sqrt(l0 * l0 + l1 * l1 + l2 * l2), bending=math.sqrt(n0 * n0 + n1 * n1 + n2 * n2),
                    axial=ax, torsion=tor, residual_mm=math.sqrt(dp) * 1e3)

    def reset(self):
        super().reset()
        self._sparse = bool(mujoco.mj_isSparse(self.m))
        self._axis = [float(x) for x in self.axis_B]


class FastStabMachine(ST.StabMachine):
    """subseacell.stab.StabMachine whose tick_post skips the load measurement while the stab is Locked, not held by
    the gripper and touched by nothing: then the only load on the latch is the stab's own wet weight and current drag
    (a few N), far below the 35 N extraction pull, so the original rule could not fire either (the extraction run
    timer is reset as the original does for a pull below the threshold)."""

    def __init__(self, model, data, stab, receptacles, **kw):
        super().__init__(model, data, stab, receptacles, **kw)
        self._gids = np.array(self.geom_ids, np.int64)
        self.n_skipped = 0

    def tick_post(self):
        if self.state == "Locked":
            g = self.gripper
            held = g is not None and g.held is not None and g.held[1] == self.name
            n = int(self.d.ncon)
            touched = n > 0 and bool(np.isin(self.d.contact.geom[:n], self._gids).any())
            if not held and not touched:
                self.t_extract_run = 0.0
                self._apply_bits()
                self.n_skipped += 1
                return
        super().tick_post()


class DebrisSleep:
    """World rule (grader side; not in the agent's documents): the dropped-object stand sleeps while nothing is near it.
    Asleep, its contacts are off, its weight is compensated, its velocity is held at zero and its Morison model is
    skipped - it stays exactly where it is (the generator only accepts stand poses that stay put for the whole episode
    in the site's current with full physics: DESIGN 2.3.3 static predicate).  It wakes as soon as the bounding sphere of
    the vehicle, the marker or the stab comes within `margin_m` of its own (no contact is possible before that) and
    falls asleep again after `rest_s` at rest with nothing near (hysteresis `hyst_m`).  Saves about 0.17 CPU s per
    simulated second (13 condim-6 contacts and the per-member hydrodynamics)."""

    def __init__(self, world, body="debris", others=("hull", "marker", "stab"), margin_m=0.25, hyst_m=0.25,
                 rest_s=10.0, v_rest=1e-3, w_rest=0.02):
        m = world.m
        self.W, self.m, self.d = world, m, world.d
        self.bid = m.body(body).id
        jt = int(m.body_jntadr[self.bid])
        self.va = int(m.jnt_dofadr[jt])
        self.gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.bid]
        self.bits = [(int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in self.gids]
        self.gravcomp0 = float(m.body_gravcomp[self.bid])
        self.r_self = self._radius([self.bid])
        hull = m.body("hull").id
        veh = [b for b in range(m.nbody) if int(m.body_rootid[b]) == hull]
        self.others = []
        for name in others:
            b = m.body(name).id
            self.others.append((b, self._radius(veh if b == hull else [b]) if b == hull else self._radius([b])))
        self.margin, self.hyst, self.rest_s, self.v_rest, self.w_rest = margin_m, hyst_m, rest_s, v_rest, w_rest
        self.stab_bid = m.body("stab").id if "stab" in others else -1
        self.asleep = False
        self.t_rest = 0.0
        self.n_wake = self.n_sleep = 0

    def _radius(self, bodies):
        """Bounding radius about the first body's frame origin (current kinematics)."""
        m, d = self.m, self.d
        o = d.xpos[bodies[0]]
        r = 0.0
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g]):
                r = max(r, float(np.linalg.norm(d.geom_xpos[g] - o)) + float(m.geom_rbound[g]))
        return r

    def stab_static(self) -> bool:
        """The hot stab is latched in a receptacle (PARK-A, PARK-B or a test port) and not held by the gripper: it cannot
        move, so it cannot touch the stand (P20's refinement for the training tank, decision P08-1)."""
        W = self.W
        sm = getattr(W, "stab", None)
        if sm is None or sm.state != "Locked":
            return False
        g = W.gripper.held
        return not (g is not None and g[1] == "stab")

    def near(self, extra=0.0):
        """Is the bounding sphere of the vehicle, the marker or a free hot stab within margin (+ extra) of the stand's?
        A stab latched in a receptacle and not held does not count (stab_static)."""
        d = self.d
        c = d.xpos[self.bid]
        skip_stab = self.stab_bid >= 0 and self.stab_static()
        for b, r in self.others:
            if skip_stab and b == self.stab_bid:
                continue
            p = d.xpos[b]
            dx, dy, dz = float(p[0] - c[0]), float(p[1] - c[1]), float(p[2] - c[2])
            lim = r + self.r_self + self.margin + extra
            if dx * dx + dy * dy + dz * dz < lim * lim:
                return True
        return False

    def sleep(self):
        m, d = self.m, self.d
        for g in self.gids:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        m.body_gravcomp[self.bid] = 1.0
        d.qvel[self.va:self.va + 6] = 0.0
        self.asleep = True
        self.t_rest = 0.0
        self.n_sleep += 1
        self.W.events.append(dict(t=float(d.time), src="world", ev="debris_sleep"))

    def wake(self):
        m, d = self.m, self.d
        for g, (ct, ca) in zip(self.gids, self.bits):
            m.geom_contype[g] = ct
            m.geom_conaffinity[g] = ca
        m.body_gravcomp[self.bid] = self.gravcomp0
        d.qvel[self.va:self.va + 6] = 0.0
        self.asleep = False
        self.t_rest = 0.0
        self.n_wake += 1
        self.W.debris_hydro.reset()
        self.W.events.append(dict(t=float(d.time), src="world", ev="debris_wake"))

    def restart(self):
        """Episode start: the stand asleep at its pose (logged), counters cleared."""
        if self.asleep:
            self.asleep = False
        self.sleep()
        self.n_wake = self.n_sleep = 0

    def pre_step(self):
        if self.asleep:
            if self.near():
                self.wake()
            else:
                self.d.qvel[self.va:self.va + 6] = 0.0

    def post_step(self, dt):
        if self.asleep:
            return
        d = self.d
        g = self.W.gripper.held
        if g is not None and g[0] == "eye":
            self.t_rest = 0.0
            return
        v = d.qvel[self.va:self.va + 6]
        if float(v[:3] @ v[:3]) < self.v_rest ** 2 and float(v[3:] @ v[3:]) < self.w_rest ** 2 \
                and not self.near(self.hyst):
            self.t_rest += dt
            if self.t_rest >= self.rest_s - 1e-9:
                self.sleep()
        else:
            self.t_rest = 0.0


class _HydroGate:
    """The stand's Morison model, skipped while the stand sleeps."""

    def __init__(self, fh, sleeper_ref):
        self.fh, self.ref = fh, sleeper_ref

    def apply(self, t):
        s = self.ref()
        if s is None or not s.asleep:
            self.fh.apply(t)


class SiteWorld:
    """scenery (amendment 02, subseacell.scenery): None = the site package's site.json["scenery"] block if it has one;
    False = none; a scenery_layout/1 dict = that block instead (maintainer tests and cost studies).  scenery_assets: the
    asset folder (default subseacell.scenery.asset_root()).  scenery_bodies: "zones" (the task world: static bodies per
    sea-bed tile) or "world" (every piece in the world body, the L09 prototype's way: cost comparison only)."""

    scenery = None          # the scenery's bookkeeping (subseacell.scenery.add_to_spec) when the world has one

    def __init__(self, site, *, hydro=None, config: Optional[dict] = None, camera_sizes=((640, 480), (320, 240)),
                 cameras: bool = True, mesh_root: Optional[str] = None, timing: bool = False, sensor_seed=None,
                 scenery=None, scenery_assets: Optional[str] = None, scenery_bodies: str = "zones"):
        gc.collect()                                            # every cell starts clean (U9 / MuJoCo GL objects)
        c_build0, w_build0 = time.process_time(), time.perf_counter()
        self.build = {}
        last = [c_build0, w_build0]

        def mark(name):
            c, w = time.process_time(), time.perf_counter()
            self.build[name] = dict(cpu_s=c - last[0], wall_s=w - last[1])
            last[0], last[1] = c, w
        if isinstance(site, (str, bytes, bytearray)):
            site = SITE.load(site if isinstance(site, str) else bytes(site))
        self.site = site
        sd = site.dict()
        self.sd = sd
        self.cfg = dict(config or {})
        self.timing = bool(timing)
        self.T = Timer()
        self.mesh_root = mesh_root
        label = sd["label"]
        self.sensor_seed = int(sd["sensor_seed"] if sensor_seed is None else sensor_seed)
        # ------------------------------------------------------------------ vehicle (physics: collision only)
        if hydro is None:
            hydro, self.hydro_info = site.hydro_draw()
        else:
            self.hydro_info = dict(given=str(getattr(hydro, "name", hydro)))
        self.seabed_depth = float(sd["seabed_depth_m"])
        self.z_bed = -self.seabed_depth
        cg = sd["cage"]
        vb = VH.build_vehicle(hydro, floor_depth=self.seabed_depth,
                              pos=(float(cg["pos_w"][0]), float(cg["pos_w"][1]), self.z_bed + 1.0))
        self.vb = vb
        s = vb.spec
        mark("vehicle_spec")
        # ------------------------------------------------------------------ static structure (stand-in manifold)
        self.pose = site.pose
        self.manifold_geoms = self._build_manifold(s)
        self._build_seabed(s)
        # ------------------------------------------------------------------ panel skid: face, frame, rail, ports, valves
        self.P_o, self.P_R = site.panel_pose_w()
        vspecs = {v["key"]: DV.ValveSpec(v["tag"], float(v["breakaway_Nm"]), float(v["running_Nm"]), float(v["theta0_deg"]))
                  for v in sd["valves"]}
        self.panel = DV.build_panel(s, site.skid, self.P_o, self.P_R, valves=vspecs,
                                    ports={p["key"]: p["name"] for p in sd["ports"]}, contact_rules=OB.SURFACE_RULES)
        self.valve_tags = {v["tag"]: v for v in sd["valves"]}
        # a valve handle turns about its fixed stem between hard stops: it cannot reach the panel face, the rail or any
        # other static geometry, so those pairs are excluded from the collision test (physics-neutral; 36 us a step)
        for inf in self.panel["valves"].values():
            s.add_exclude(bodyname1="world", bodyname2=inf["body"])
        # ------------------------------------------------------------------ hot stab + receptacle sliders
        self.stab_info = ST.add_stab(s, "stab", pos=list(self.P_o + self.P_R[:, 2] * 0.3), vb=vb)
        self.receptacles = ST.add_receptacles(s, self.panel, "stab")
        # ------------------------------------------------------------------ marker (in the jaws at the start)
        self.marker_handle = DV.add_marker(vb, "marker")
        # ------------------------------------------------------------------ dropped-object stand
        self.debris_spec = OB.make_spec(sd["debris"]["kind"])
        dpos, dquat, _ = site.debris_pose_w()
        self.debris_info = OB.add_frame(s, "debris", self.debris_spec, list(dpos), list(dquat), visual_eyes=False)
        # ------------------------------------------------------------------ deployment cage (its coded plate: render twin)
        self.cage = OB.add_cage(s, "cage", cg["pos_w"], float(cg["yaw"]))
        # ------------------------------------------------------------------ static scenery (amendment 02; after every
        # original body, so that every original body and geom keeps its index)
        self.scenery = None
        block = sd.get("scenery") if scenery is None else (scenery or None)
        if block:
            self._add_scenery(s, block, scenery_assets, scenery_bodies)
        # ------------------------------------------------------------------ grasp constraints
        self.handles = dict(self.panel["handles"])
        self.handles["marker"] = self.marker_handle
        self.handles["stab"] = self.stab_info["handle"]
        self.eyes = OB.frame_eyes(self.debris_info)
        GR.add_constraints(s, self.handles, self.eyes)
        # the cage's coded plate is appearance (the render twin draws it); MjSpec.delete() last (MuJoCo 3.14: name
        # lookups after a deletion can return the wrong element)
        tg = [g for g in s.bodies if g.name == "cage"][0]
        for g in [g for g in tg.geoms if g.name == "cage_tag_plate"]:
            s.delete(g)
        mark("scene_spec")
        # ------------------------------------------------------------------ current and simulator
        self.flow = FL.StationFlow.from_site(sd)
        self.sim = uwkit.UnderwaterSim(vb, current=FastSiteField(self.flow), seed=_seed(label + "/sim") & 0x7FFFFFFF,
                                       battery_soc0=float(sd.get("battery_soc0", 0.9)))
        hw = vb.meta.get("hotel_W") or {}
        for k in ("sonar", "dvl"):
            if k in hw:
                self.sim.set_hotel(k, float(hw[k]))
        m, d = self.sim.model, self.sim.data
        self.m, self.d = m, d
        if self.scenery is not None:
            self.scenery["compiled"] = SCN.check_compiled(m, self.scenery)
        mark("compile_sim")
        # ------------------------------------------------------------------ devices and rules
        self.events = []
        self.bank = FastValveBank(m, d, list(self.panel["valves"].values()))
        self.gripper = FastGripper(self.sim, self.handles, self.eyes, on_event=self._on("grasp"))
        self.process = PR.Process.from_site(sd)
        self.port_branch = {p["name"]: p.get("branch") for p in sd["ports"]}
        self.stab = FastStabMachine(m, d, self.stab_info, self.receptacles, pressure_bar=self._port_pressure,
                                    gripper=self.gripper, on_event=self._on("stab"))
        self.debris_hydro = OB.FrameHydro(m, d, "debris", self.debris_spec, self.sim.current)
        self.debris_sleep = None
        OB.install_hydro(self.sim, [_HydroGate(self.debris_hydro, lambda: self.debris_sleep)])
        chart = sd.get("chart") or {}
        centre = chart.get("origin_xy") or list(self.pose.o[:2])
        self.protect = PT.Protection(m, d, PT.ProtectConfig.from_values(self.cfg),
                                     site=PT.SiteFrame(centre_xy=(float(centre[0]), float(centre[1])), mudline_z=self.z_bed))
        # ------------------------------------------------------------------ on-board navigation sensors (physics model)
        self.column = WaterColumn(temperature_c=WATER_TEMPERATURE_C, salinity_psu=WATER_SALINITY_PSU)
        self.water_optics = SE.site_water(float(sd["water_c550"]))
        self.acoustic_classes = self._acoustic_classes()
        scfg = SuiteConfig.for_vehicle(m, with_camera=False, scanning_sonar=None, usbl=None)
        rng = np.random.default_rng([self.sensor_seed, 1])
        self.suite = SensorSuite(m, d, SensorWorld(water=self.water_optics, ambient=UWAT.DARK, column=self.column,
                                                   material_overrides=self.acoustic_classes), rng, config=scfg)
        self.fw = FW.Firmware(self.sim, self.suite, vb.vehicle, config=FW.FirmwareConfig.from_values(self.cfg),
                              protect=self.protect)
        self.k_ctrl = max(1, int(round(self.fw.dt / self.sim.dt)))
        self.k_slow = max(1, int(round(SLOW_DT_S / self.sim.dt)))
        mark("devices_firmware")
        # ------------------------------------------------------------------ plume (spin-up: the leak is older than the dive)
        self.plume = PL.Plume.from_site(sd, self.flow)
        self.plume.attach_process(self.process)
        self.plume.advance_to(0.0)
        mark("plume_spinup")
        # ------------------------------------------------------------------ acoustic model + scanning sonar
        self.acoustic = TW.AcousticModel(m, self.acoustic_scene())
        self.sonar = SE.PlumeSonar(self.acoustic.model, self.acoustic.materials(),
                                   np.random.default_rng([self.sensor_seed, 2]), plume=self.plume,
                                   acoustic=SE.AcousticSpec.from_site(sd), water=self.column,
                                   config=ScanningSonarConfig(), geomgroup=None, exclude_body=self.acoustic.mount_id)
        mark("acoustic_model")
        # ------------------------------------------------------------------ render twin + cameras
        self.twin = TW.RenderTwin(s, m, self.twin_scene())
        mark("render_twin")
        self.cams = {}
        self.camera_sizes = [tuple(int(v) for v in sz) for sz in camera_sizes]
        if cameras:
            base = SuiteConfig.for_vehicle(m).camera
            for i, (w, h) in enumerate(self.camera_sizes):
                ccfg = replace(base, width=w, height=h, show_geomgroups=(TW.GROUP_FOUL,))
                self.cams[(w, h)] = SE.PlumeCamera(self.twin.model, self.water_optics,
                                                   np.random.default_rng([self.sensor_seed, 3, w, h]),
                                                   ambient=UWAT.DARK, config=ccfg, plume=self.plume,
                                                   bubbles=SE.BubbleOptics())
        mark("renderers")
        # ------------------------------------------------------------------ start pose and reset
        self._start = OB.cage_start_pose(self.cage, OB.vehicle_lowest_z(m, d))
        self.debris_sleep = DebrisSleep(self)
        self.reset()
        mark("reset")
        self.build["total"] = dict(cpu_s=time.process_time() - c_build0, wall_s=time.perf_counter() - w_build0)

    # ====================================================================================== build helpers
    def _build_manifold(self, s):
        """Physics geometry of the stand-in manifold: the primitives flagged physics (collision) or dvl (seen by the
        DVL only).  Returns {role: [geom names]}."""
        out = {}
        wb = s.worldbody
        R, o = self.pose.R, self.pose.o
        for pr in self.site.manifold["primitives"]:
            if not (pr["physics"] or pr.get("dvl")):
                continue
            kw = dict(name=pr["name"], rgba=list(pr["rgba"]), friction=list(OB.SURFACE_FRICTION))
            if pr["role"] == "mudmat":
                kw.update(solref=list(OB.DEBRIS_CONTACT["solref"]), solimp=list(OB.DEBRIS_CONTACT["solimp"]))
            if pr["role"] == "roof_proxy":
                # group 0 and opaque: the DVL's rays see the plate (decision P18-3: over the roof the DVL tracks the
                # roof; MuJoCo's ray casts skip a geom without material whose rgba alpha is 0, and the package gives the
                # proxy alpha 0); the render twin deletes it (twin_scene), the acoustic model skips it (not acoustic)
                kw.update(group=TW.GROUP_SCENE, friction=list(OB.ROOF_FRICTION), rgba=list(ROOF_PROXY_RGBA))
            if not pr["physics"]:
                kw.update(contype=0, conaffinity=0)
            if pr["kind"] == "box":
                A = np.eye(3) if pr.get("axes") is None else np.asarray(pr["axes"], float).T
                g = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pr["half"]),
                                pos=list(self.pose.to_w(pr["centre"])), quat=list(SITE.quat_of(R @ A)), **kw)
            else:
                gt = mujoco.mjtGeom.mjGEOM_CAPSULE if pr["kind"] == "capsule" else mujoco.mjtGeom.mjGEOM_CYLINDER
                g = wb.add_geom(type=gt, size=[pr["radius"], 0, 0],
                                fromto=list(self.pose.to_w(pr["a"])) + list(self.pose.to_w(pr["b"])), **kw)
            out.setdefault(pr["role"], []).append(g.name)
        return out

    def _build_seabed(self, s):
        """The sea bed the rays see (P24a, decision P19-1).  The kit's plane (50 x 50 m half size at the world origin)
        stays the contact surface - MuJoCo planes are infinite for collisions, so the contact physics is unchanged -
        but goes to group 3, which no ray of the three models sees; SEABED_BOX (800 x 800 x 1 m, centred under the
        structure origin, top face = the plane, no contacts) takes its place for the DVL, the sonar (the acoustic model
        copies it) and the camera (the render twin copies the spec and drops the plane).  Same acoustic class as the
        plane (the name rule "seabed" -> sand)."""
        planes = [g for g in s.worldbody.geoms if g.name == SEABED_PLANE]
        if len(planes) != 1:
            raise RuntimeError("the vehicle build has no single sea-bed plane")
        pl = planes[0]
        if abs(float(pl.pos[2]) - self.z_bed) > 1e-9:
            raise RuntimeError("the sea-bed plane is not at the site's sea-bed depth")
        pl.group = TW.GROUP_PROXY
        hx, hy, hz = SEABED_BOX_HALF_M
        o = self.pose.o
        s.worldbody.add_geom(name=SEABED_BOX, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                             pos=[float(o[0]), float(o[1]), self.z_bed - hz], rgba=list(pl.rgba),
                             group=TW.GROUP_SCENE, contype=0, conaffinity=0)
        self.seabed_box = dict(name=SEABED_BOX, centre_xy=[float(o[0]), float(o[1])], half_m=list(SEABED_BOX_HALF_M),
                               top_z=self.z_bed)

    # ------------------------------------------------------------------ static scenery (amendment 02)
    def _add_scenery(self, s, block, assets=None, bodies="zones"):
        """Add a scenery_layout/1 block to the physics spec (subseacell.scenery): pieces in static zone bodies, contact
        parameters of the sea-bed plane.  Also for subclasses with a construction of their own (the practice
        facilities): call it on the spec before the physics compile, then _scenery_classes() and _scenery_acoustic()."""
        lay = SCN.parse(block)
        planes = [g for g in s.worldbody.geoms if g.name == SEABED_PLANE]
        if len(planes) != 1:
            raise RuntimeError("scenery: the vehicle build has no single sea-bed plane")
        o = self.pose.o
        self.scenery = SCN.add_to_spec(s, lay, SCN.asset_root(assets), SCN.contact_of(planes[0]), bodies=bodies,
                                       origin_xy=(float(o[0]), float(o[1])))
        self.scenery["layout"] = lay
        return self.scenery

    def _scenery_classes(self, cls: dict) -> dict:
        """The scenery's acoustic classes (one per piece name) added to a class table (DVL and sonar)."""
        if self.scenery is not None:
            cls.update(self.scenery["classes"])
        return cls

    def _scenery_acoustic(self, scene: dict) -> dict:
        """The scenery in an acoustic scene: its primitives as static geoms, its rocks as static meshes; scenery names
        that a body-0 loop picked up (bodies="world") are taken out of the plain static list first."""
        if self.scenery is None:
            return scene
        names = set(self.scenery["pieces"])
        prim, meshes = SCN.acoustic_entries(self.m, self.scenery)
        scene["static_geoms"] = [n for n in scene["static_geoms"] if n not in names] + prim
        scene["static_meshes"] = list(scene.get("static_meshes", ())) + meshes
        return scene

    def scenery_names(self) -> list:
        return [] if self.scenery is None else list(self.scenery["pieces"])

    def _acoustic_classes(self) -> dict:
        cls = {}
        for pr in self.site.manifold["primitives"]:
            if pr.get("acls"):
                cls[pr["name"]] = pr["acls"]
            elif pr["role"] == "roof_proxy":
                cls[pr["name"]] = ROOF_PROXY_ACLS
        for role, names in self.panel["geoms"].items():
            for n in names:
                cls[n] = "steel"
        cls.update(self.cage.get("acoustic_overrides", {}))
        return self._scenery_classes(cls)

    def acoustic_scene(self) -> dict:
        """What the sonar's structural model contains (see subseacell.twin.AcousticModel)."""
        m = self.m
        skip = set(self.panel["geoms"].get("port_receptacle", ()))
        skip.update(n for n, pr in ((p["name"], p) for p in self.site.manifold["primitives"]) if not pr["acoustic"])
        if self.scenery is not None:
            skip.update(self.scenery["pieces"])
        static = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != 0:
                continue
            n = m.geom(g).name
            if not n or n in skip or int(m.geom_group[g]) == TW.GROUP_PROXY:
                continue
            static.append(n)
        cyl = []
        sk_ports = {p["key"]: p for p in self.site.skid["ports"]}
        for name, p in self.panel["ports"].items():
            r = 0.5 * float(sk_ports[p["key"]]["outer_diameter_mouth"])
            cyl.append(dict(name=f"port_{name}_steel", a=list(p["mouth_w"]), b=list(p["seat_w"]), radius=r, acls="steel"))
        moving = [dict(body=inf["body"], acls="steel") for inf in self.panel["valves"].values()]
        moving += [dict(body="stab", acls="steel"), dict(body="marker", acls="plastic"), dict(body="debris", acls="steel")]
        cage_mv, cage_cls = cage_acoustic(m, self.cage)              # the cage and its reflector poles (CAL-2)
        moving.append(cage_mv)
        classes = dict(self.acoustic_classes)
        classes.update(cage_cls)
        return self._scenery_acoustic(dict(static_geoms=static, cylinders=cyl, moving=moving, classes=classes,
                                           sonar_site="sonar_head"))

    def twin_scene(self) -> dict:
        """What the render twin paints (see subseacell.twin.RenderTwin)."""
        sd, site = self.sd, self.site
        oP, RP = self.P_o, self.P_R
        plates = []
        tags = {(t["of"], t["key"]): t for t in sd["tags"]}
        for t in site.skid["tags"]:
            st = tags[(t["of"], t["key"])]
            X, Y, _ = t["centre"]
            plates.append(dict(name=f"{t['of']}_{t['key']}", centre_w=list(oP + RP @ np.array([X, Y, 0.0])),
                               R_w=RP.tolist(), size=list(t["size"]), thickness=float(t["thickness"]),
                               image=site.texture(st["texture"]), fouling=st.get("fouling"), code_id=st["code_id"],
                               text=st.get("text", "")))
        cgt = self.cage["tag"]
        n = np.asarray(cgt["normal_w"], float)
        up = np.array([0.0, 0.0, 1.0])
        Rc = np.stack([np.cross(up, n), up, n], 1)
        th = float(self.cage["params"]["tag"]["thickness"])
        plates.append(dict(name="cage", centre_w=list(np.asarray(cgt["centre_w"], float) - n * th), R_w=Rc.tolist(),
                           size=list(cgt["size_m"]), thickness=th, image=site.texture(sd["cage"]["texture"]),
                           fouling=None, code_id=sd["cage"]["code_id"], text=""))
        paddles = []
        for inf in self.panel["valves"].values():
            h = inf["handle"]
            paddles.append(dict(body=h["body"], geom=h["paddle_geom"], centre_b=h["centre_b"], R_bp=h["R_bp"],
                                size=h["paddle_m"]))
        h = self.stab_info["handle"]
        paddles.append(dict(body=h["body"], geom=h["paddle_geom"], centre_b=h["centre_b"], R_bp=h["R_bp"], size=h["paddle_m"]))
        fs = self.debris_spec
        F = OB.FRAME_DEFAULTS
        rr = F["eye_ring_r"]
        eyes = []
        for en, c in fs.eyes_b.items():
            c = np.asarray(c, float)
            base = np.array([c[0], fs.W / 2, fs.r]) if en == "drag" else np.array([c[0], c[1], fs.H - fs.r])
            eyes.append(dict(name=f"debris_eye_{en}", body="debris", centre_b=c.tolist(), u_b=[0.0, 1.0, 0.0],
                             v_b=[0.0, 0.0, 1.0], ring_r=rr, post=[base.tolist(), (c - np.array([0.0, 0.0, rr])).tolist()]))
        lay = site.skid["laydown"]
        (x0, x1), (z0, z1) = lay["x"], lay["z"]
        yl = 0.004
        corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        segs = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            segs.append(dict(name=f"laydown_paint_{i}", a=list(oP + RP @ np.array([a[0], yl, a[1]])),
                             b=list(oP + RP @ np.array([b[0], yl, b[1]])), radius=LAYDOWN_LINE_R, rgba=TW.LAYDOWN_RGBA))
        boxes = []
        for pr in site.manifold["primitives"]:
            if pr["look"] and not pr["physics"] and not pr.get("dvl") and pr["kind"] == "box":
                A = np.eye(3) if pr.get("axes") is None else np.asarray(pr["axes"], float).T
                boxes.append(dict(name=pr["name"], centre=list(self.pose.to_w(pr["centre"])), half=pr["half"],
                                  R=self.pose.R @ A, rgba=pr["rgba"]))
        return dict(plates=plates, paddles=paddles, delete=["roof_proxy", SEABED_PLANE], hide=[], eyes=eyes,
                    segments=segs, boxes=boxes, mesh_root=self.mesh_root)

    # ====================================================================================== reset / events
    def _on(self, src):
        def cb(e):
            ev = _jsonable(dict(e))
            ev["src"] = src
            ev["t"] = float(self.d.time)
            self.events.append(ev)
        return cb

    def _port_pressure(self, port, t):
        b = self.port_branch.get(port)
        return 0.0 if b is None else self.process.segment_pressure(int(b))

    def reset(self):
        """Episode start: vehicle seated on the cage deck with the marker in its jaws, valves as found, stab locked in
        its parking port, stand at its pose, process at p0, firmware idle."""
        m, d, sim = self.m, self.d, self.sim
        self.events = []
        pos, quat = self._start
        sim.reset(pos=pos, quat=quat)
        self.bank.reset()
        mujoco.mj_forward(m, d)
        self.gripper.reset()
        self.stab.reset_locked(self.sd["stab"].get("initial_port", "PARK-A"))
        self.debris_hydro.reset()
        self.gripper.attach_initial("marker")              # logs the initial grasp (rule "initial")
        self.protect.reset()
        if self.debris_sleep is not None:
            self.debris_sleep.restart()
        self._n_protect_logged = 0
        mujoco.mj_forward(m, d)
        self.k = 0
        self.lights = 0.0
        self.sim.set_lights(0.0)
        self.hooks = []
        self.frame_hooks = []
        self.ended = False
        self.end_reason = None
        self.qhash = hashlib.sha256()
        self.n_frames = {sz: 0 for sz in self.camera_sizes}
        self.n_pings = 0
        self.frame_log = []
        self.valve_names = [v["tag"] for v in sorted(self.sd["valves"], key=lambda v: (v["kind"], v["branch"]))]
        self._vidx = [self.bank.names.index(n) for n in self.valve_names]
        a0 = self.bank.angles_deg()
        self.angles0 = {n: a0[n] for n in self.valve_names}
        self.angle_hist = [[a0[n] for n in self.valve_names]]
        self._cond = {n: a0[n] < PR.CLOSED_MIN_DEG for n in self.valve_names}
        self.cpu_steps = 0.0
        self.T = Timer()
        self.n_steps = 0
        self._t_slow = float(sim.time)               # simulated time of the last 0.1 s grid point run (tick_slow_to)
        self.fw.ingest(self.suite.step(sim.time))

    # ====================================================================================== time
    def advance(self, dt: float) -> int:
        """Run the physics for dt seconds (rounded to whole 2 ms steps).  Returns the steps run (fewer if the episode
        ended: vehicle disabled, out of bounds, undervoltage)."""
        n = int(round(float(dt) / self.sim.dt))
        sim, d, fw, G, SM, bank, DS = self.sim, self.d, self.fw, self.gripper, self.stab, self.bank, self.debris_sleep
        dt_s = sim.dt
        tt = time.thread_time
        timing = self.timing
        T = self.T
        c_start = tt()
        hook_cpu = 0.0
        ran = 0
        for _ in range(n):
            if self.ended:
                break
            if self.k % self.k_ctrl == 0:
                if self.hooks:
                    a = tt()
                    for h in list(self.hooks):
                        h(self, sim.time)
                    hook_cpu += tt() - a
                if timing:
                    a = tt()
                    sim.set_thrusters(fw.control(sim.time))
                    T.add("firmware", tt() - a)
                else:
                    sim.set_thrusters(fw.control(sim.time))
            if timing:
                a = tt()
                bank.tick(); G.tick_pre(); SM.tick_pre(); DS.pre_step()
                b = tt(); T.add("devices", b - a)
                sim.step()
                a = tt(); T.add("physics", a - b)
                fw.ingest(self.suite.step(sim.time))
                b = tt(); T.add("sensors_nav", b - a)
                G.tick_post(); SM.tick_post(); DS.post_step(dt_s)
                a = tt(); T.add("devices", a - b)
                fw.physics_tick(sim.time)
                b = tt(); T.add("protect", b - a)
            else:
                bank.tick(); G.tick_pre(); SM.tick_pre(); DS.pre_step()
                sim.step()
                fw.ingest(self.suite.step(sim.time))
                G.tick_post(); SM.tick_post(); DS.post_step(dt_s)
                fw.physics_tick(sim.time)
            self.k += 1
            ran += 1
            if len(self.protect.events) > self._n_protect_logged:
                for e in self.protect.events[self._n_protect_logged:]:
                    ev = _jsonable(dict(e))
                    ev["src"], ev["ev"] = "protect", ev.pop("kind")
                    self.events.append(ev)
                self._n_protect_logged = len(self.protect.events)
            if self.k % self.k_slow == 0:
                a = tt()
                self._slow_tick()
                self._t_slow = sim.time
                if timing:
                    T.add("process_plume_log", tt() - a)
            if self.protect.ended:
                self.ended, self.end_reason = True, self.protect.end_reason
        self.n_steps += ran
        self.cpu_steps += tt() - c_start - hook_cpu
        T.add("hooks", hook_cpu)
        return ran

    def tick_slow_to(self, t: float) -> int:
        """Run the 0.1 s grid (process graph, plume, valve history, qpos hash; the event log's sampling when one is
        attached) up to simulated time t WITHOUT stepping the physics - for test benches and observability rigs that
        place the vehicle kinematically.  Every grid point k * 0.1 s with  last grid point < k * 0.1 <= t  is run in
        order with the clock set to it; the clock then stands at t.  Returns the number of grid points run."""
        t = float(t)
        last = float(getattr(self, "_t_slow", 0.0))
        k = int(math.floor(last / SLOW_DT_S + 1e-6)) + 1
        n = 0
        while True:
            g = round(k * SLOW_DT_S, 6)
            if g > t + 1e-9:
                break
            self.d.time = g
            self._slow_tick()
            self._t_slow = g
            k += 1
            n += 1
        self.d.time = t
        return n

    def _slow_tick(self):
        d, t = self.d, self.sim.time
        ang = self.bank.angles_deg()
        self.process.step_to(t, angles_deg=ang)
        self.plume.advance_to(t)
        row = [ang[n] for n in self.valve_names]
        self.angle_hist.append(row)
        for n, a in zip(self.valve_names, row):
            c = a < PR.CLOSED_MIN_DEG
            if c != self._cond[n]:
                self._cond[n] = c
                self.events.append(dict(t=float(t), src="process", ev="conduction", tag=n, conducting=bool(c),
                                        angle_deg=float(a)))
        self.qhash.update(d.qpos.tobytes())
        self.qhash.update(d.qvel.tobytes())
        if not np.all(np.isfinite(d.qvel)):
            raise FloatingPointError("simulation diverged")

    # ====================================================================================== sensors
    def frame(self, size=(640, 480)):
        """One camera frame of the render twin at the current time (no simulated time passes here)."""
        size = tuple(int(v) for v in size)
        cam = self.cams.get(size)
        if cam is None:
            raise ValueError(f"no camera of size {size} in this world")
        c0, w0 = time.process_time(), time.perf_counter()
        self.twin.sync(self.d)
        cam.set_lamp_level(self.lights)
        fr = cam.capture(self.twin.data, self.sim.time, force=True)
        c1, w1 = time.process_time(), time.perf_counter()
        self.n_frames[size] += 1
        p, C = cam.camera_pose(self.twin.data)
        rec = dict(t=float(self.sim.time), size=list(size), lamp=float(self.lights), tilt_deg=float(math.degrees(fr.tilt_rad)),
                   cam_pos=p.tolist(), cpu_s=c1 - c0, wall_s=w1 - w0)
        self.frame_log.append(rec)
        for h in list(self.frame_hooks):
            h(self, fr, size, cam)
        return fr

    def sonar_configure(self, **kw):
        self.sonar.configure(**kw)

    def ping_period(self) -> float:
        return self.sonar.ping_period()

    def ping(self):
        """One scanning-sonar ping of the acoustic model at the current time; the head then steps (no simulated time
        passes here - the caller advances by ping_period())."""
        c0 = time.thread_time()
        self.acoustic.sync(self.d)
        p = self.sonar.ping_now(self.sim.time, self.acoustic.data)
        self.last_ping_cpu = time.thread_time() - c0
        self.n_pings += 1
        return p

    # ====================================================================================== vehicle-side actions
    def set_lights(self, level: float):
        self.lights = float(min(max(float(level), 0.0), 1.0))
        self.sim.set_lights(self.lights)

    def set_camera_tilt(self, rad: float):
        self.sim.set_camera_tilt(float(rad))

    def gripper_close(self):
        return self.gripper.close()

    def gripper_open(self):
        return self.gripper.open()

    def gripper_public(self) -> str:
        return GRIPPER_PUBLIC[self.gripper.state]

    def state(self, budget=None) -> dict:
        return self.fw.state(gripper=self.gripper_public(), budget=budget)

    # ====================================================================================== grader side
    def tcp_w(self):
        sid = self.gripper.tcp_sid
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def body_pose(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()

    def truth(self) -> dict:
        """The true state the scoring reads (grader side only)."""
        d = self.d
        st = self.sim.state()
        mk = OB.body_speed(self.m, d, "marker")
        dbs = OB.body_speed(self.m, d, "debris")
        hull_v = OB.body_speed(self.m, d, "hull")
        return dict(t=float(self.sim.time), ended=self.ended, end_reason=self.end_reason,
                    valves={n: float(a) for n, a in zip(self.valve_names, self.angle_hist[-1])},
                    leak_pressure_bar=self.process.leak_pressure(), leak_isolated=bool(self.process.leak_isolated()),
                    stab=self.stab.public(), gripper=dict(state=self.gripper.state, held=self.gripper.held),
                    marker=dict(pos=self.body_pose("marker")[0].tolist(), speed=mk[0]),
                    debris=dict(pos=self.body_pose("debris")[0].tolist(), speed=dbs[0]),
                    vehicle=dict(pos=st.pos_w.tolist(), quat=st.quat_wb.tolist(), speed=hull_v[0],
                                 in_cage=OB.in_cage(self.cage, st.pos_w)),
                    battery=dict(soc=float(self.sim.battery.status()["soc"]), V=float(self.sim.battery.voltage)),
                    protect=dict(count=int(self.protect.count), ended=bool(self.protect.ended)),
                    plume_particles=int(self.plume.n), frames={f"{w}x{h}": n for (w, h), n in self.n_frames.items()},
                    pings=int(self.n_pings))

    def event_log_bytes(self) -> bytes:
        return SITE.dumps(_jsonable(self.events))

    def fingerprint(self) -> dict:
        V = np.asarray(self.angle_hist, np.float64)
        d = self.d
        return dict(t=float(self.sim.time), steps=int(self.n_steps),
                    qpos_chain_sha256=self.qhash.hexdigest(),
                    qpos_final_sha256=hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest(),
                    valve_hist_sha256=hashlib.sha256(V.tobytes()).hexdigest(),
                    events_sha256=hashlib.sha256(self.event_log_bytes()).hexdigest(), n_events=len(self.events),
                    plume_sha256=self.plume.state_digest())

    def timing_report(self) -> dict:
        sim_s = self.n_steps * self.sim.dt
        return dict(sim_s=sim_s, steps=self.n_steps, cell_cpu_s=self.cpu_steps,
                    cell_cpu_per_sim_s=self.cpu_steps / sim_s if sim_s > 0 else None,
                    components_cpu_s={k: round(v, 4) for k, v in sorted(self.T.cpu.items())},
                    plume_cpu_s=dict(step=self.plume.c_step_s, index=self.plume.c_index_s),
                    sonar_hook_cpu_s=self.sonar.c_extra, build=self.build)

    def model_stats(self) -> dict:
        m = self.m
        veh = {m.body(n).id for n in TW.VEHICLE_BODIES}
        n_scene = sum(1 for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in veh)
        n_nocoll_scene = sum(1 for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in veh
                             and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)
        out = dict(physics=dict(ngeom=int(m.ngeom), nbody=int(m.nbody), nq=int(m.nq), nv=int(m.nv), nmesh=int(m.nmesh),
                                ntex=int(m.ntex), scene_geoms=n_scene, scene_noncolliding=n_nocoll_scene,
                                vehicle_geom_groups=np.bincount(m.geom_group[[g for g in range(m.ngeom)
                                                                             if int(m.geom_bodyid[g]) in veh]],
                                                                minlength=6).tolist()),
                   twin=self.twin.stats(), acoustic=self.acoustic.stats())
        if self.scenery is not None:
            out["scenery"] = dict(pieces=len(self.scenery["pieces"]), bodies=self.scenery["bodies"],
                                  zone_bodies=len(self.scenery["zone_bodies"]), rocks=len(self.scenery["meshes"]),
                                  rule_version=self.scenery.get("rule_version"))
        return out

    def close(self):
        for cam in self.cams.values():
            try:
                cam.close()
            except Exception:
                pass
        self.cams = {}
