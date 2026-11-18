"""The composed mission world: a generated site with the robot attached, the sensor head, the on-board controller,
the sensors, the link model, the dose model, the report queue, budgets, events and the run logs.

Only the facade (facade.py) and the privileged Truth object (truth.py) talk to it; a mission program never does.

Composition (the site loader's compose(), then this module):
  * the site's scene.xml with the robot model's MJCF attached at the pad under the name prefix 'robot/';
  * on the pan-tilt head's tilt body three boresighted cameras ('robot/nav' 640x480 90 deg with depth,
    'robot/inspect' 1280x960 18 deg, 'robot/thermal' 160x120 50 deg) and a spot head lamp 'robot/lamp'
    (bricks.sensors.camera.add_sensor_head); the robot's own 'head_cam' stays unused;
  * the robot's collision primitives move from geom group 3 to group 4 (the site draws its traversal elements in
    group 3, which the robot's cameras must see; the robot's red collision boxes must not be drawn);
  * every instrument in its true state (the site loader's apply_truth_states), the thermal units' temperatures and their
    surface types' emissivities in a bricks TemperatureTable, the bricks planar lidar at the robot's lidar site with
    the robot's body excluded;
  * the renderers in a render_guard.GuardedRenderPool (safe close, validated segmentation renders).

Time passes only in step(): one tick is 10 ms = 10 physics steps of 1 ms (the controller's control period).

The start: the robot is set down on the deployment pad within 5 cm and 2 degrees of the drawing's datum
(a draw of the mission's own noise stream); its odometry starts at the datum, so the set-down error is part of the
odometry error from the first tick. Every random draw of a mission comes from streams seeded by the mission's seed
(the cell derives it from the site package), none from the renderer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time

import numpy as np
import yaml

from . import paths

paths.setup()

import mujoco  # noqa: E402

from bricks.sensors import camera as bcam  # noqa: E402
from bricks.sensors import lidar as blidar  # noqa: E402
from bricks.sensors import thermal as bthermal  # noqa: E402
from site_gen.loader import apply_truth_states, compose, instrument_records, open_site  # noqa: E402
from site_gen.loader import scene_option as site_scene_option  # noqa: E402

from .catalog import REPORT_KIND, emissivity as catalog_emissivity  # noqa: E402
from .controllers import has_own_protections, make_controller  # noqa: E402
from .evidence import EvidenceModel  # noqa: E402
from .render_guard import GuardedRenderPool  # noqa: E402

MODE_IDS = {"stand": 0, "walk": 1, "high_step": 2, "crouch_walk": 3, "sit": 4}
SUBSTEPS = 10
UPRIGHT_TILT_DEG = 30.0        # the return factor's "upright": roll and pitch below this (no condition on the height)


# ============================================================================================== configuration
from .config import _merge, load_cfg  # noqa: E402,F401  (re-exported)


# ============================================================================================== helpers
def quat_to_rpy(q):
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def thermal_emissivity(ip: dict) -> float:
    """The emissivity a temperature point's units render with: the site truth's (the generator writes the plant
    catalogue's value for the point's surface type), else the catalogue's value for the surface the truth or the item's
    parameters name (tests/review/cell/test_thermal.py checks it against public_catalog())."""
    tr = ip.get("truth") or {}
    if tr.get("emissivity") is not None:
        return float(tr["emissivity"])
    return catalog_emissivity(tr.get("surface") or (ip.get("params") or {}).get("surface"))


# ============================================================================================== radiation
class DoseField:
    """The site's dose-rate field, the same model as site_gen.radiation.dose_rate with the bricks RadiationField
    (inverse square with an r_min floor, exp(-mu L) through 3-D boxes centred at z 1.3 m, half height 1.4 m, plus the
    background), vectorised over the boxes for one point at a time. Units: the site's (uSv/h)."""

    def __init__(self, site: dict):
        r = site["radiation"]
        self.background = float(r["background_uSv_h"])
        self.r_min = float(r["r_min"])
        b = np.asarray(r["boxes"], float).reshape(-1, 6)
        self.c = np.c_[b[:, 0], b[:, 1], np.full(len(b), 1.3)]
        self.h = np.c_[b[:, 2], b[:, 3], np.full(len(b), 1.4)]
        self.cos, self.sin = np.cos(b[:, 4]), np.sin(b[:, 4])
        self.mu = b[:, 5]
        self.src = [(np.asarray(s["pos"], float), float(s["strength"])) for s in site["sources"]]

    def _att(self, s, p):
        """sum mu * chord of the segment s -> p through every box."""
        def local(v):
            dx, dy = v[0] - self.c[:, 0], v[1] - self.c[:, 1]
            return np.stack([dx * self.cos + dy * self.sin, -dx * self.sin + dy * self.cos, v[2] - self.c[:, 2]], 1)
        o = local(s)
        e = local(p)
        d = e - o
        t0 = np.zeros(len(o))
        t1 = np.ones(len(o))
        for a in range(3):
            da, oa, ha = d[:, a], o[:, a], self.h[:, a]
            par = np.abs(da) < 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (-ha - oa) / da
                tb = (ha - oa) / da
            lo = np.where(par, np.where(np.abs(oa) <= ha, -np.inf, np.inf), np.minimum(ta, tb))
            hi = np.where(par, np.inf, np.maximum(ta, tb))
            t0 = np.maximum(t0, lo)
            t1 = np.minimum(t1, hi)
        L = np.clip(t1 - t0, 0.0, None) * np.linalg.norm(p - s)
        return float(np.sum(self.mu * L))

    def rate(self, p) -> float:
        p = np.asarray(p, float)
        out = self.background
        for s, S in self.src:
            r2 = max(float(np.sum((p - s) ** 2)), self.r_min ** 2)
            out += S * math.exp(-self._att(s, p)) / r2
        return out


# ============================================================================================== odometry drift
class OdometryDrift:
    """Extra odometry error on top of the controller's leg odometry: a scale error of the distance travelled and a
    heading error whose rate performs a bounded random walk (~2 % of distance, <= 1 deg/min).

    The odometry the program sees starts at the drawing's datum (x0, y0, yaw0_odo) while the robot (and the
    controller's estimate) starts at its true set-down pose: `yaw_offset` (datum yaw - true start yaw) is the initial
    heading error, so the increments of the controller's estimate are turned into the datum's frame."""

    def __init__(self, cfg: dict, rng: np.random.Generator, x0, y0, yaw0, yaw_offset: float = 0.0):
        c = cfg["odometry_drift"]
        self.k = float(c["scale_error"]) * (1.0 if rng.random() < 0.5 else -1.0)
        self.rmax = math.radians(float(c["heading_rate_max_deg_min"])) / 60.0
        self.tau = float(c["heading_rate_tau_s"])
        self.sig = 0.5 * self.rmax / math.sqrt(self.tau / 2.0)
        self.rate = float(rng.uniform(-1.0, 1.0)) * self.rmax
        self.rng = rng
        self.err = float(yaw_offset)
        self.x, self.y = float(x0), float(y0)
        self.prev = (float(x0), float(y0))

    def update(self, x, y, dt):
        dx, dy = x - self.prev[0], y - self.prev[1]
        self.prev = (x, y)
        self.rate += -self.rate / self.tau * dt + self.sig * math.sqrt(dt) * float(self.rng.standard_normal())
        self.rate = max(-self.rmax, min(self.rmax, self.rate))
        self.err += self.rate * dt
        c, s = math.cos(self.err), math.sin(self.err)
        self.x += (1.0 + self.k) * (c * dx - s * dy)
        self.y += (1.0 + self.k) * (s * dx + c * dy)

    def pose(self, yaw_ctl):
        return self.x, self.y, wrap(yaw_ctl + self.err)


# ============================================================================================== the world
class MissionWorld:
    def __init__(self, site, controller: str | None = None, seed: int = 0, cfg: dict | None = None,
                 out_dir: str | None = None, robot_xml: str | None = None, start_pose=None,
                 sim_time_limit: float | None = None, shared_dir: str | None = None):
        """site: a stored site package (<name>.tar.xz), a generated site folder, or a site_gen.loader.SitePackage.
        seed: the mission's noise seed (every random stream of the mission). start_pose: (x, y, yaw) of the true
        set-down pose; None = the datum plus the set-down error of cfg['start'] drawn from the seed.
        sim_time_limit: None = cfg's. shared_dir: the shared textures of the archives (None = the loader's)."""
        controller = controller or paths.DEFAULT_CONTROLLER
        self.cfg = cfg or load_cfg()
        # the render bundle (version 2): required where the task runs (SITE_GEN_DRESS=1), checked file by file before
        # anything is built; a missing or altered bundle raises here (mission.bundle)
        from .bundle import check as _check_bundle
        self.render_bundle = _check_bundle()
        self.pkg = open_site(site, shared_dir=shared_dir or paths.SHARED_ASSETS)
        self.site_dir = self.pkg.source
        self.seed = int(seed)
        self.controller_kind = controller
        self.out_dir = out_dir
        self.wall0 = time.time()
        self.site = self.pkg.site
        self.prior = self.pkg.prior
        self._site_obj = None                          # the site's truth queries (Truth only), loaded on demand
        self.sim_time_limit = float(sim_time_limit if sim_time_limit is not None
                                    else self.cfg["time"]["sim_time_limit_s"])
        self.rng = np.random.default_rng([self.seed, 20261113])
        d = self.prior.get("datum", {})
        self.datum = (float(d.get("pos", [0, 0])[0]), float(d.get("pos", [0, 0])[1]), float(d.get("yaw", 0.0)))
        sp = start_pose
        if sp is None:
            sp = self._set_down_pose(self.datum)
        self.start_pose = tuple(float(v) for v in sp)
        self._compose(robot_xml or paths.ROBOT_XML)
        self.evidence = EvidenceModel(self.model, self.data, self.site, self.cfg, exclude_body=self.head_tilt_id)
        self.ctl = make_controller(controller, self.model, self.data, prefix="robot/", seed=self.seed)
        self.own_protections = has_own_protections(self.ctl)
        self.ctl.reset(pos=self.start_pose[:2], yaw=self.start_pose[2])
        self.t0 = float(self.data.time)
        self._init_sensors()
        self._init_state()
        if self.out_dir:
            os.makedirs(os.path.join(self.out_dir, "frames"), exist_ok=True)
        self.log_event("start", pose=list(self.start_pose), controller=controller, seed=self.seed)
        self._link_check(force=True)
        self._log_traj()

    # ------------------------------------------------------------------ the start
    def _set_down_pose(self, datum):
        """The true set-down pose: the datum moved by a draw within cfg['start'] (5 cm, 2 deg)."""
        S = self.cfg.get("start") or {}
        r_max, a_max = float(S.get("pos_m", 0.0)), math.radians(float(S.get("yaw_deg", 0.0)))
        rng = np.random.default_rng([self.seed, 50])
        r = r_max * math.sqrt(float(rng.random()))
        phi = 2.0 * math.pi * float(rng.random())
        dyaw = a_max * float(rng.uniform(-1.0, 1.0))
        return (datum[0] + r * math.cos(phi), datum[1] + r * math.sin(phi), wrap(datum[2] + dyaw))

    @property
    def site_obj(self):
        if self._site_obj is None:
            from site_gen.loader import Site
            self._site_obj = Site(self.pkg)
        return self._site_obj

    def lidar_offset_body(self):
        """The lidar's optical centre in the body frame (m)."""
        Tb = self.T_world_base()
        p = np.asarray(self.data.site_xpos[self.lidar_site], float)
        return (np.linalg.inv(Tb) @ np.r_[p, 1.0])[:3].round(4).tolist()

    # ------------------------------------------------------------------ composition
    def _compose(self, robot_xml):
        spec, _, info = compose(self.pkg, robot_xml=robot_xml, start_pose=self.start_pose)
        S = self.cfg["sensors"]
        head = spec.body("robot/head_tilt")
        nav = bcam.CameraModel("nav", S["nav"]["width"], S["nav"]["height"], S["nav"]["hfov_deg"], depth=True,
                               depth_range=tuple(S["nav"]["depth_range"]))
        ins = bcam.CameraModel("inspect", S["inspect"]["width"], S["inspect"]["height"], S["inspect"]["hfov_deg"],
                               depth=False, blur_px=0.6)
        th = bthermal.ThermalModel("thermal", S["thermal"]["width"], S["thermal"]["height"], S["thermal"]["hfov_deg"])
        self.cam_models = {"nav": nav, "inspect": ins, "thermal": th}
        bcam.add_sensor_head(spec, head, "robot", cameras=(nav, ins, th),
                             offsets={k: tuple(v) for k, v in S["offsets"].items()}, lamp=True, lamp_on=False)
        for g in spec.geoms:
            if g.name.startswith("robot/") and g.group == 3:
                g.group = 4
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        q0 = np.asarray(info["qpos0"], float)
        if len(q0) != self.model.nq:
            raise RuntimeError("qpos layout changed by the sensor head (%d vs %d)" % (len(q0), self.model.nq))
        self.data.qpos[:] = q0
        apply_truth_states(self.model, self.data, instrument_records(self.site))
        mujoco.mj_forward(self.model, self.data)
        xml = spec.to_xml()
        self.model_hash = hashlib.sha1(xml.encode()).hexdigest()[:16]
        self.spec = spec
        m = self.model
        self.base_id = m.body("robot/base").id
        self.head_tilt_id = m.body("robot/head_tilt").id
        bodies = set(range(m.nbody))
        rb = [b for b in bodies if self._in_subtree(b, self.base_id)]
        self.robot_bodies = np.array(sorted(rb), int)
        self.robot_geom = np.isin(m.geom_bodyid, self.robot_bodies)
        self.lamp_id = m.light("robot/lamp").id
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or "" for i in range(m.nlight)]
        self.room_lights = [i for i, n in enumerate(names) if n.startswith("light/") and not n.startswith("light/fill")]
        self.cam_ids = {k: m.camera("robot/" + k).id for k in ("nav", "inspect", "thermal")}
        self.lidar_site = m.site("robot/lidar").id
        # traversal elements: geom id -> class (for the stand-in trip stop)
        self.elem_class = {}
        for e in self.site["elements"]:
            for gname in e.get("geoms", []):
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    self.elem_class[gid] = (e["class"], e["id"])

    def _in_subtree(self, b, root):
        m = self.model
        while b > 0:
            if b == root:
                return True
            b = int(m.body_parentid[b])
        return False

    # ------------------------------------------------------------------ sensors
    def _init_sensors(self):
        m, S = self.model, self.cfg["sensors"]
        # every renderer of the mission: closed in its own GL context, rendering with the cyclic GC paused, and every
        # segmentation render validated against its scene (re-rendered once, else sensor_unavailable; render_guard)
        self.pool = GuardedRenderPool(m)
        self.scene_opt = site_scene_option(show_overhead=True)
        rs = np.random.default_rng([self.seed, 1])
        self.cams = {k: bcam.Camera(m, "robot/" + k, self.cam_models[k], self.pool, np.random.default_rng([self.seed, 10 + i]),
                                    scene_option=self.scene_opt) for i, k in enumerate(("nav", "inspect"))}
        # the thermal scene: a temperature point's units at their true temperatures, each with the emissivity of the
        # point's surface type (the plant catalogue's table, as the round sheet names the surface); every other
        # surface at the ambient (+ a small fixed spread) with the configured default emissivity; the reflected
        # temperature of every surface is the ambient (bricks TemperatureTable)
        self.ttable = bthermal.TemperatureTable(m, ambient_c=float(S["thermal"]["ambient_c"]),
                                                emissivity=float(S["thermal"]["emissivity"]), seed=self.seed)
        temps, emis = {}, {}
        for ip in self.site["ips"] + self.site["decoys"]:
            if ip.get("type") == "thermal":
                eps = thermal_emissivity(ip)
                tr = ip.get("truth") or {}
                units = tr.get("units") or []
                for u in units:
                    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, u["geom"]) >= 0:
                        temps[u["geom"]] = float(u["temp_C"])
                        emis[u["geom"]] = eps
                if not units and tr.get("temp_C") is not None:
                    # a temperature point without a unit list (the ladder's decoys: truth.temp_C on the hot face of
                    # params.hot_geom) at its own true temperature, as the site data gives it
                    hg = tr.get("geom") or (ip.get("params") or {}).get("hot_geom")
                    if hg and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, hg) >= 0:
                        temps[hg] = float(tr["temp_C"])
                        emis[hg] = eps
        self.ttable.set(temps, emis)
        self.thermal_emissivity = emis
        self.thermal = bthermal.ThermalCamera(m, "robot/thermal", self.cam_models["thermal"], self.ttable, self.pool,
                                              np.random.default_rng([self.seed, 20]), scene_option=self.scene_opt)
        L = S["lidar"]
        self.lidar_model = blidar.LidarModel("lidar2d", 1, int(L["beams"]), r_min=float(L["r_min"]),
                                             r_max=float(L["r_max"]))
        self.lidar = blidar.Lidar(m, "robot/lidar", self.lidar_model, exclude_root="robot/base",
                                  rng=np.random.default_rng([self.seed, 30]), geomgroup=(1, 1, 1, 0, 0, 0))
        # dose
        D = self.cfg["dose"]
        self.field = DoseField(self.site)
        self.k_gy = float(D["gyh_per_site_unit"])
        # one configured rating for every site (no per-site number derived from a reference route). A
        # config of an older run folder (3 x the site's route dose) is still understood for replays.
        if "rating_gy" in D:
            self.rating_gy = float(D["rating_gy"])
        else:
            route_dose = float(self.site.get("route", {}).get("dose_total_uSv", 0.0))
            self.rating_gy = max(float(D.get("rating_min_site_units", 20.0)),
                                 route_dose / float(D.get("rating_route_fraction", 1.0 / 3.0))) * self.k_gy
        self.sens = float(D["sensitivity_cps_per_gyh"])
        self.dose_rng = np.random.default_rng([self.seed, 40])
        # link nodes (as built: the broken repeater does not relay)
        Lk = self.site["link"]
        C = self.cfg["link"]
        self.link_nodes = [{"id": "BASE", "pos": np.asarray(Lk["base_station"]["antenna"], float),
                            "range": float(C["base_range_m"])}]
        for r in Lk.get("repeaters", []):
            if r.get("working", False):
                self.link_nodes.append({"id": r["id"], "pos": np.asarray(r["pos"], float),
                                        "range": float(C["repeater_range_m"])})
        _ = rs

    # ------------------------------------------------------------------ mission state
    def _init_state(self):
        self.tick = 0
        self.done = False
        self.done_reason = None
        self.events = []
        self.frames = {}
        self.n_frames = {"nav": 0, "inspect": 0, "depth": 0, "thermal": 0, "lidar": 0}
        self.frame_seq = 0
        self.reports = []
        self.n_reports = {"instrument": 0, "mark": 0, "source": 0}
        self.link_up = False
        self.link_node = None
        self.link_dist = None
        self.cmd = np.zeros(3)
        self.prog_mode = "stand"
        self.body_h = None
        self.body_pitch = 0.0
        self.head_cmd = (0.0, 0.0)
        self.lamp_on = False
        self.dose_gy = 0.0                  # true accumulated dose (ends the mission at the rating; grader side)
        self.dose_meas_gy = 0.0             # the dosimeter's own accumulated dose, from its counts (what get_state shows)
        self.counts_pending = 0
        self.dwell_pending = 0.0
        self.dose_log = {"t": [], "pos": [], "rate": [], "counts": []}
        self.lidar_log = {"t": [], "T": [], "ranges": []}
        self.marks = [{"tag": ip["tag"], "pos": ip["pose"]["pos"][:2], "since": None} for ip in self.site["ips"]
                      if ip["type"] == "dose_point"]
        self.traj = {k: [] for k in ("t", "qpos", "mode", "head", "lamp", "odom", "true", "dose_rate", "dose",
                                     "link", "cmd")}
        self.last_mode = "stand"
        self.override = None                # (until_t, reason) of a protective behaviour
        # the hold after a `blocked` (version 2): until hold_until (simulated s) the robot takes no walking
        # command and its status stays `blocked`; the program's walking command is dropped at the report (the gait
        # drops its own); a mode or body command given meanwhile goes to the gait (with zero velocity) and stays
        self.hold_s = float((self.cfg.get("hold") or {}).get("after_blocked_s", 0.0))
        self.hold_until = None
        self.hold_pending = False
        self.n_holds = 0
        self.trip_t = -1e9
        self.tilt_since = None
        self.status_prev = "ok"
        self.ctl_events_seen = len(self.ctl.get_state().get("events") or []) if self.own_protections else 0
        self.rate_now = 0.0
        self.st = self.ctl.get_state()
        # the odometry starts at the datum; the controller's estimate starts at the true set-down pose
        x0, y0, yaw0 = self.datum
        self.drift = OdometryDrift(self.cfg, np.random.default_rng([self.seed, int(self.cfg["odometry_drift"]["seed_salt"])]),
                                   x0, y0, yaw0, yaw_offset=wrap(yaw0 - float(self.st["odom_yaw"])))
        self.drift.prev = tuple(float(v) for v in self.st["odom_pos"][:2])
        self.drift.x, self.drift.y = x0, y0
        self.odom_distance0 = float(self.st["odom_distance"])   # the firmware's settle at set-down is not travel
        self.distance_true = 0.0
        self._prev_true = self.true_pose()
        self.facade_wall = 0.0
        self.render_wall = {"nav": 0.0, "inspect": 0.0, "depth": 0.0, "thermal": 0.0, "lidar": 0.0}
        self.physics_wall = 0.0
        self._dose_sample()

    @property
    def time(self) -> float:
        return float(self.data.time) - self.t0

    # ------------------------------------------------------------------ truth helpers (world side)
    def true_pose(self) -> dict:
        tr = self.ctl.truth()
        roll, pitch, yaw = quat_to_rpy(tr["quat"])
        p = tr["pos"]
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw": yaw, "roll": roll, "pitch": pitch}

    def T_world_base(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[self.base_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self.base_id]
        return T

    # ------------------------------------------------------------------ events
    def log_event(self, typ, **kw):
        ev = {"t": round(self.time, 4), "type": typ}
        ev.update(kw)
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------------ commands
    def hold_active(self) -> bool:
        """True while the hold after a `blocked` lasts: no walking command is taken, the status reads blocked."""
        return self.hold_until is not None and self.time < self.hold_until - 1e-9

    def hold_left(self) -> float:
        return max(0.0, float(self.hold_until) - self.time) if self.hold_active() else 0.0

    def _start_hold(self):
        """A `blocked` was reported now: the hold runs hold_s of simulated time from this instant (not extended by a
        second report, which cannot come while the robot stands)."""
        if self.hold_s <= 0.0 or self.hold_active():
            return
        # the walking command is dropped (the manual: "it backs off, stands, and drops the velocity command"): nothing
        # moves after the hold until the program sends a walking command again. Keeping it would let a mode or body
        # command made during the hold revive a 5 s old walking command when the hold ends, a lurch toward whatever
        # had stopped the robot
        self.cmd = np.zeros(3)
        self.hold_until = self.time + self.hold_s
        self.hold_pending = False
        self.n_holds += 1
        self.log_event("hold_start", until=round(self.hold_until, 4))          # run log only

    def _end_hold(self):
        self.hold_until = None
        self.log_event("hold_end", pending=bool(self.hold_pending))           # run log only
        if self.hold_pending:
            # what the program asked for during the hold (a mode, a body height) is already the gait's; the velocity
            # stays zero (the walking command was dropped at the report) until the program's next walking command
            self.hold_pending = False
            self._push_command()

    def _push_command(self):
        if self.override is not None:
            return
        mode = self.prog_mode
        cmd = self.cmd
        if self.hold_active():
            # the hold: the mode and body height go to the gait now, the velocity stays zero until the hold ends
            self.hold_pending = True
            cmd = np.zeros(3)
        moving = bool(np.any(np.abs(cmd) > 1e-4))
        if mode in ("stand", "sit") and moving:
            mode = None                    # the controller switches a moving command in stand to walk
        self.ctl.set_command(float(cmd[0]), float(cmd[1]), float(cmd[2]), body_height=self.body_h, mode=mode)

    def set_velocity(self, vx, vy, wz):
        """The program's walking command. During the hold it is refused (returns False) and changes nothing."""
        if self.hold_active():
            return False
        self.cmd = np.array([vx, vy, wz], float)
        if self.prog_mode in ("stand", "sit") and np.any(np.abs(self.cmd) > 1e-4):
            self.prog_mode = "walk"
        self._push_command()
        return True

    def set_mode(self, m):
        if m not in MODE_IDS:
            raise ValueError(m)
        if m != self.prog_mode:
            self.body_h = None             # each mode starts at its default body height
        self.prog_mode = m
        if m in ("stand", "sit"):
            self.cmd = np.zeros(3)
        self._push_command()

    def set_body(self, height=None, pitch=None):
        if height is not None:
            self.body_h = float(height)
        if pitch is not None:
            self.ctl.set_body_pitch(float(pitch))
            # get_state's pitch_cmd is what the firmware holds after its cut to +-20 deg (it once showed the
            # value asked for)
            self.body_pitch = float(getattr(self.ctl, "pitch_user", float(pitch)))
        self._push_command()

    def set_head(self, pan=None, tilt=None):
        lim = self.cfg["sensors"]["head_limits"]
        p, t = self.head_cmd
        if pan is not None:
            p = float(np.clip(pan, *lim["pan"]))
        if tilt is not None:
            t = float(np.clip(tilt, *lim["tilt"]))
        self.head_cmd = (p, t)
        self.ctl.set_head(p, t)
        return p, t

    def set_lamp(self, on):
        self.lamp_on = bool(on)

    # ------------------------------------------------------------------ stepping
    def step(self, n_ticks=1):
        n = int(max(1, min(int(n_ticks), int(self.cfg["time"]["max_ticks_per_step"]))))
        out = []
        t_w = time.perf_counter()
        for _ in range(n):
            if self.done:
                break
            for _k in range(SUBSTEPS):
                self.ctl.step()
            self.tick += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                # a physics fault is the cell's failure, not the program's (the grader runs the mission once more)
                self.log_event("physics_fault")
                self.end("physics_fault", out)
                break
            self._after_tick(out)
        self.physics_wall += time.perf_counter() - t_w
        return out

    def _after_tick(self, out):
        dt = float(self.cfg["time"]["tick_s"])
        st = self.ctl.get_state()
        self.st = st
        op = st["odom_pos"]
        self.drift.update(float(op[0]), float(op[1]), dt)
        tp = self.true_pose()
        self.distance_true += math.hypot(tp["x"] - self._prev_true["x"], tp["y"] - self._prev_true["y"])
        self._prev_true = tp
        mode = st["mode"]
        if mode != self.last_mode:
            self.log_event("mode_change", mode=mode, prev=self.last_mode)
            self.last_mode = mode
        if self.hold_until is not None and not self.hold_active():
            self._end_hold()
        self._protections(st, tp, out)
        if self.tick % int(round(float(self.cfg["dose"]["sample_s"]) / dt)) == 0:
            self._dose_sample()
        if self.tick % int(round(float(self.cfg["link"]["check_every_s"]) / dt)) == 0:
            self._link_check(out=out)
        if self.tick % int(round(1.0 / (dt * float(self.cfg["log"]["traj_hz"])))) == 0:
            self._log_traj()
        if not self.done and self.dose_gy >= self.rating_gy:
            self.end("dose_limit", out)
        if not self.done and self.time >= self.sim_time_limit - 1e-9:
            self.end("time_limit", out)

    # ------------------------------------------------------------------ protections
    def _protections(self, st, tp, out):
        P = self.cfg["protect"]
        t = self.time
        tilt = math.degrees(max(abs(tp["roll"]), abs(tp["pitch"])))
        # lost: tilted beyond the limit for the hold time (true attitude: the IMU agrees to 0.1 deg)
        if tilt > float(P["lost_tilt_deg"]):
            if self.tilt_since is None:
                self.tilt_since = t
            elif t - self.tilt_since >= float(P["lost_hold_s"]):
                self.end("robot_lost", out)
                return
        else:
            self.tilt_since = None
        if self.own_protections:
            # the controller's own protective behaviours: every new entry of its event list is passed on (its
            # 'status' stays 'blocked' after a successful retry until a mode change, so a status transition would
            # miss the second and third blocked report of a retry; a review of the gait found it)
            evs = st.get("events") or []
            for _te, typ in evs[self.ctl_events_seen:]:
                if typ in ("blocked", "protective_sit"):
                    out.append({"t": round(t, 3), "type": typ})
                    info = st.get("blocked") if typ == "blocked" else None
                    self.log_event(typ, source="controller", mode=st.get("mode"),
                                   reason=(info or {}).get("reason") if info else None)
                    if typ == "blocked":
                        self._start_hold()
            self.ctl_events_seen = len(evs)
            s = st.get("status", "ok")
            if s != self.status_prev:
                self.log_event("controller_status", status=s, prev=self.status_prev)     # run log only
                self.status_prev = s
            return
        if not P.get("enabled", True):
            return
        if self.override is not None:
            until, why = self.override
            if t >= until:
                self.override = None
                if why == "backoff":
                    self.cmd = np.zeros(3)
                self.ctl.set_command(0.0, 0.0, 0.0, body_height=self.body_h, mode=None)
                self._push_command()
            elif why == "backoff" and t >= until - 0.25:
                self.ctl.set_command(0.0, 0.0, 0.0, body_height=self.body_h, mode=None)
            return
        if tilt > float(P["protective_tilt_deg"]):
            self.override = (t + float(P["protective_pause_s"]), "protective_sit")
            self.cmd = np.zeros(3)
            self.ctl.set_command(0.0, 0.0, 0.0, body_height=None, mode=None)
            out.append({"t": round(t, 3), "type": "protective_sit"})
            self.log_event("protective_sit", tilt_deg=round(tilt, 1))
            return
        if P.get("trip_stop", True) and t - self.trip_t > float(P["trip_debounce_s"]):
            hit = self._unrated_contact(st["mode"])
            if hit is not None:
                self.trip_t = t
                self.override = (t + float(P["backoff_s"]), "backoff")
                self.ctl.set_command(-float(P["backoff_speed"]), 0.0, 0.0, body_height=self.body_h, mode=None)
                out.append({"t": round(t, 3), "type": "blocked"})
                self.log_event("blocked", element=hit[1], element_class=hit[0], mode=st["mode"])
                self._start_hold()

    def _unrated_contact(self, mode):
        d = self.data
        if d.ncon == 0:
            return None
        rated = set(self.cfg["protect"]["rated"].get(mode, []))
        g = d.contact.geom[:d.ncon]
        for g1, g2 in g:
            g1, g2 = int(g1), int(g2)
            if self.robot_geom[g1] == self.robot_geom[g2]:
                continue
            other = g2 if self.robot_geom[g1] else g1
            ec = self.elem_class.get(other)
            if ec is not None and ec[0] not in rated:
                return ec
        return None

    # ------------------------------------------------------------------ dose
    def dosimeter_pos(self) -> np.ndarray:
        off = np.asarray(self.cfg["dose"]["dosimeter_offset"], float)
        return self.data.xpos[self.base_id] + self.data.xmat[self.base_id].reshape(3, 3) @ off

    def _dose_sample(self):
        dt = float(self.cfg["dose"]["sample_s"])
        p = self.dosimeter_pos().copy()
        rate = self.field.rate(p) * self.k_gy
        self.rate_now = rate
        self.dose_gy += rate * dt / 3600.0
        c = int(self.dose_rng.poisson(max(self.sens * rate * dt, 0.0)))
        # the reported dose integrates the counts (Poisson), not the true rate: a noise-free integral would hand the
        # program the exact field by differencing two get_state() calls (review 2026-11-13)
        self.dose_meas_gy += c / (self.sens * 3600.0)
        self.counts_pending += c
        self.dwell_pending += dt
        # survey-mark dwell bookkeeping (run log only)
        for mk in self.marks:
            inside = math.hypot(p[0] - mk["pos"][0], p[1] - mk["pos"][1]) <= float(self.cfg["evidence"]["mark"]["radius_m"])
            if inside and mk["since"] is None:
                mk["since"] = self.time
                self.log_event("mark_dwell_start", mark=mk["tag"])
            elif not inside and mk["since"] is not None:
                self.log_event("mark_dwell_end", mark=mk["tag"], dwell_s=round(self.time - mk["since"], 2))
                mk["since"] = None
        L = self.dose_log
        L["t"].append(self.time)
        L["pos"].append(p)
        L["rate"].append(rate)
        L["counts"].append(c)

    def take_dosimeter(self):
        c, w = self.counts_pending, self.dwell_pending
        self.counts_pending, self.dwell_pending = 0, 0.0
        return c, w

    # ------------------------------------------------------------------ link
    def link_visible(self, p=None):
        """(up, node id, distance) for the lidar site (or point p): line of sight to a working node in range,
        ray cast against group-0 geometry (structure, closed doors, equipment)."""
        m, d = self.model, self.data
        p = np.asarray(d.site_xpos[self.lidar_site] if p is None else p, float)
        gg = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        geomid = np.zeros(1, np.int32)
        best = (False, None, None)
        margin = float(self.cfg["link"]["end_margin_m"])
        for nd in self.link_nodes:
            v = nd["pos"] - p
            dist = float(np.linalg.norm(v))
            if dist > nd["range"]:
                continue
            hit = mujoco.mj_ray(m, d, p, v / dist, gg, 1, -1, geomid)
            if hit < 0 or hit >= dist - margin:
                if not best[0] or dist < best[2]:
                    best = (True, nd["id"], dist)
        return best

    def _link_check(self, out=None, force=False):
        up, nid, dist = self.link_visible()
        self.link_node, self.link_dist = nid, dist
        if up != self.link_up or force:
            prev = self.link_up
            self.link_up = up
            if up != prev or force:
                typ = "link_up" if up else "link_down"
                if out is not None:
                    out.append({"t": round(self.time, 3), "type": typ})
                self.log_event(typ, node=nid)
        if self.link_up:
            for r in self.reports:
                if r["status"] == "queued":
                    self._deliver(r, out)

    def _deliver(self, r, out=None):
        r["status"] = "delivered"
        r["t_sent"] = round(self.time, 4)
        if out is not None:
            out.append({"t": round(self.time, 3), "type": "report_sent", "report_id": r["id"]})
        self.log_event("report_sent", report_id=r["id"], tag=r["tag"], kind=r["kind"])

    # ------------------------------------------------------------------ reports
    def report(self, kind, tag, value, frame_ids=(), extra=None):
        """Record a report (sent now when the link is up, else queued). Refusals raise facade.CallError."""
        from .facade import CallError
        if self.done:
            raise CallError("mission_over", "the mission has ended (%s)" % (self.done_reason,))
        if kind not in self.n_reports:
            raise ValueError("unknown report kind %r" % (kind,))
        if self.n_reports[kind] >= int(self.cfg["budgets"]["reports"][kind]):
            raise CallError("budget", "%s report budget used (%d)" % (kind, int(self.cfg["budgets"]["reports"][kind])))
        fids = [str(f) for f in (frame_ids or [])][:3]
        self.n_reports[kind] += 1
        r = {"id": len(self.reports) + 1, "kind": kind, "tag": str(tag), "value": value, "frame_ids": fids,
             "t": round(self.time, 4), "t_sent": None, "status": "queued", "pose_true": self.true_pose()}
        if extra:
            r.update(extra)
        self.reports.append(r)
        self.log_event("report_made", report_id=r["id"], tag=r["tag"], kind=kind, frame_ids=fids,
                       value=_jsonable(value))
        sent = False
        if self.link_up:
            self._deliver(r)
            sent = True
        return {"ok": True, "sent": sent, "report_id": r["id"]}

    def report_kind_of(self, tag):
        for it in self.prior["work_order"]["items"]:
            if it["tag"] == tag:
                return REPORT_KIND.get(it["type"], "instrument")
        return "instrument"

    # ------------------------------------------------------------------ frames
    def _lights(self, cam_id):
        m = self.model
        pos = self.data.cam_xpos[cam_id]
        n = int(self.cfg["sensors"]["room_lights_per_frame"]) - (1 if self.lamp_on else 0)
        dd = sorted((float(np.linalg.norm(m.light_pos[i][:2] - pos[:2])), i) for i in self.room_lights)
        act = {i for _, i in dd[:n]}
        for i in range(m.nlight):
            m.light_active[i] = 1 if i in act else 0
        m.light_active[self.lamp_id] = 1 if self.lamp_on else 0

    def _frame_record(self, sensor, cam_id, K, w, h, extra=None):
        self.frame_seq += 1
        fid = "F%05d" % self.frame_seq
        T = bcam.camera_pose_optical(self.model, self.data, cam_id)
        tp = self.true_pose()
        hd = self.ctl.get_state()["head"]
        rec = {"frame_id": fid, "sensor": sensor, "t": round(self.time, 4), "T_world_cam": T.tolist(),
               "K": np.asarray(K).tolist(), "width": int(w), "height": int(h), "head": [float(hd[0]), float(hd[1])],
               "base_true": tp, "lamp": self.lamp_on}
        if extra:
            rec.update(extra)
        if sensor in ("inspect", "thermal"):
            # grader side: the evidence record of every listed instrument near the view, computed now (the robot's
            # own body can occlude, and it moves on)
            evs = {}
            for tag in self.evidence.near_view(rec):
                r = self.evidence.record(rec, tag)
                if r is not None:
                    evs[tag] = r
            rec["evidence"] = evs
        self.frames[fid] = rec
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, T[:3, :3].ravel())
        self.log_event("frame", frame_id=fid, sensor=sensor, cam_pos=[round(float(v), 4) for v in T[:3, 3]],
                       cam_quat_optical=[round(float(v), 6) for v in q])
        return fid, T

    def _prep_render(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def capture(self, sensor, depth=False):
        t_w = time.perf_counter()
        self._prep_render()
        cid = self.cam_ids[sensor]
        self._lights(cid)
        cam = self.cams[sensor]
        fr = cam.capture(self.data, depth=depth)
        fid, T = self._frame_record(sensor, cid, fr.K, cam.cm.width, cam.cm.height,
                                    {"gain": float(fr.gain), "depth": bool(depth)})
        if self.out_dir and self.cfg["log"]["save_frames"]:
            import cv2
            p = os.path.join(self.out_dir, "frames", fid + ".png")
            cv2.imwrite(p, cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR))
            self.frames[fid]["file"] = os.path.relpath(p, self.out_dir)
            if fr.depth is not None:
                dm = np.nan_to_num(fr.depth * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16)
                pd = os.path.join(self.out_dir, "frames", fid + "_depth.png")
                cv2.imwrite(pd, dm)
                self.frames[fid]["depth_file"] = os.path.relpath(pd, self.out_dir)
        self.render_wall[sensor] += time.perf_counter() - t_w
        return fid, fr, T

    def capture_depth(self):
        t_w = time.perf_counter()
        self._prep_render()
        cid = self.cam_ids["nav"]
        cam = self.cams["nav"]
        r = self.pool.get(cam.cm.width, cam.cm.height)
        r.update_scene(self.data, camera=cid, scene_option=self.scene_opt)
        r.enable_depth_rendering()
        try:
            z = r.render().astype(np.float32).copy()
        finally:
            r.disable_depth_rendering()
        a, b = cam.cm.depth_noise
        z = z + cam.rng.standard_normal(z.shape).astype(np.float32) * (a + b * z * z)
        lo, hi = cam.cm.depth_range
        z[(~np.isfinite(z)) | (z < lo) | (z > hi) | (cam.rng.random(z.shape) < cam.cm.depth_dropout)] = np.nan
        fid, T = self._frame_record("depth", cid, cam.cm.K, cam.cm.width, cam.cm.height)
        if self.out_dir and self.cfg["log"]["save_frames"]:
            import cv2
            pd = os.path.join(self.out_dir, "frames", fid + "_depth.png")
            cv2.imwrite(pd, np.nan_to_num(z * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16))
            self.frames[fid]["depth_file"] = os.path.relpath(pd, self.out_dir)
        self.render_wall["depth"] += time.perf_counter() - t_w
        return fid, z, cam.cm.K, T

    def capture_thermal(self):
        t_w = time.perf_counter()
        self._prep_render()
        cid = self.cam_ids["thermal"]
        T_c = self.thermal.capture(self.data)
        tm = self.cam_models["thermal"]
        fid, T = self._frame_record("thermal", cid, tm.K, tm.width, tm.height)
        if self.out_dir and self.cfg["log"]["save_frames"]:
            import cv2
            p = os.path.join(self.out_dir, "frames", fid + "_thermal.png")
            cv2.imwrite(p, np.clip((T_c + 50.0) * 100.0, 0, 65535).astype(np.uint16))
            self.frames[fid]["file"] = os.path.relpath(p, self.out_dir)
        self.render_wall["thermal"] += time.perf_counter() - t_w
        return fid, T_c, tm.K, T

    def scan_lidar(self):
        t_w = time.perf_counter()
        sc = lidar_scan_rays(self.lidar, self.data)
        self.lidar_log["t"].append(self.time)
        self.lidar_log["T"].append(np.asarray(sc.T_world_lidar, np.float32))
        self.lidar_log["ranges"].append(np.asarray(sc.ranges[0], np.float32))
        self.render_wall["lidar"] += time.perf_counter() - t_w
        return sc

    # ------------------------------------------------------------------ logs
    def _log_traj(self):
        T = self.traj
        st = self.st
        T["t"].append(self.time)
        T["qpos"].append(self.data.qpos.astype(np.float32).copy())
        T["mode"].append(MODE_IDS.get(st["mode"], -1))
        hd = st["head"]
        T["head"].append((float(hd[0]), float(hd[1])))
        T["lamp"].append(1 if self.lamp_on else 0)
        T["odom"].append(self.drift.pose(float(st["odom_yaw"])))
        tp = self.true_pose()
        T["true"].append((tp["x"], tp["y"], tp["z"], tp["yaw"], tp["roll"], tp["pitch"]))
        T["dose_rate"].append(self.rate_now)
        T["dose"].append(self.dose_gy)
        T["link"].append(1 if self.link_up else 0)
        T["cmd"].append(tuple(float(v) for v in self.cmd))

    def end(self, reason, out=None):
        if self.done:
            return
        self.done = True
        self.done_reason = reason
        # a last look at the link at the end instant: the periodic check runs every 0.5 s, so a report queued until the
        # robot re-entered coverage a moment ago would otherwise be dropped although the link is up (review 2026-11-13)
        self._link_check(out=out)
        for r in self.reports:
            if r["status"] == "queued":
                r["status"] = "lost"
        if out is not None and reason in ("robot_lost", "dose_limit", "time_limit"):
            out.append({"t": round(self.time, 3), "type": reason})
        self.log_event(reason if reason in ("robot_lost", "dose_limit", "time_limit", "finish") else "end",
                       reason=reason)
        self._log_traj()

    def close(self):
        """Release the renderers, each in its own GL context (safe to call twice)."""
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.close()

    def render_incidents(self) -> list:
        """What the render guard had to do in this mission (re-rendered or failed segmentation renders)."""
        pool = getattr(self, "pool", None)
        return list(getattr(pool, "incidents", None) or [])

    def end_state(self) -> dict:
        """The robot's state when the mission ends, for the return factor: upright = roll and pitch below 30 degrees,
        whatever the body's height (a robot that sits down on the pad has
        returned); pad_distance = the body centre's horizontal distance from the pad centre."""
        tp = self.true_pose()
        pad = self.prior.get("datum", {}).get("pos", [0.0, 0.0])
        d = math.hypot(tp["x"] - pad[0], tp["y"] - pad[1])
        upright = math.degrees(max(abs(tp["roll"]), abs(tp["pitch"]))) < UPRIGHT_TILT_DEG
        return {"pose": tp, "pad_distance": d, "upright": bool(upright), "reason": self.done_reason}

    def state_digest(self) -> str:
        """sha256 of the mission's final state: simulated time, qpos, qvel, the true dose, the dosimeter's dose and
        every report (kind, tag, value, cited frames, made / sent times, status). Two runs of one program on one
        site agree on it when the mission was reproduced bit for bit."""
        h = hashlib.sha256()
        h.update(np.float64(self.time).tobytes())
        h.update(np.asarray(self.data.qpos, np.float64).tobytes())
        h.update(np.asarray(self.data.qvel, np.float64).tobytes())
        h.update(np.float64(self.dose_gy).tobytes())
        h.update(np.float64(self.dose_meas_gy).tobytes())
        rep = [[r["kind"], r["tag"], _jsonable(r["value"]), r["frame_ids"], r["t"], r["t_sent"], r["status"]]
               for r in self.reports]
        h.update(json.dumps(rep, sort_keys=True, default=str).encode())
        return h.hexdigest()

    def save_traj(self, path):
        T = self.traj
        np.savez_compressed(path, t=np.asarray(T["t"], np.float64), qpos=np.asarray(T["qpos"], np.float32),
                            mode=np.asarray(T["mode"], np.int8), head=np.asarray(T["head"], np.float32),
                            lamp=np.asarray(T["lamp"], np.int8), odom=np.asarray(T["odom"], np.float32),
                            true=np.asarray(T["true"], np.float32), dose_rate=np.asarray(T["dose_rate"], np.float32),
                            dose=np.asarray(T["dose"], np.float64), link=np.asarray(T["link"], np.int8),
                            cmd=np.asarray(T["cmd"], np.float32),
                            dose_t=np.asarray(self.dose_log["t"], np.float64),
                            dose_pos=np.asarray(self.dose_log["pos"], np.float32),
                            dose_rate_10hz=np.asarray(self.dose_log["rate"], np.float32),
                            dose_counts=np.asarray(self.dose_log["counts"], np.int32),
                            mode_names=np.array(list(MODE_IDS.keys())))


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def lidar_scan_rays(lid, data):
    """The planar lidar's sweep with one mujoco.mj_ray per beam (version 2).
    The brick's Lidar.scan casts the whole sweep with one mj_multiRay, whose culling misses geoms of static bodies
    that hold several geoms (MuJoCo 3.14; the fault the evidence model met in version 1, `evidence.first_hits`): on 11
    of 13 development buildings 1.8 % of the beams passed through benches, tables, workbenches, desks, sinks, shower
    stalls, a fume hood, carts, cages, a boot rack, pallet loads and corridor fire extinguishers (group 0 boxes and
    cylinders of the bodies room/<id>) and returned the wall behind them. Here the scene's first hit of every beam is
    one mj_ray (the brick's geom groups, static geoms included, the sensor housing's body excluded, beyond r_max no
    hit); everything after it is the brick's own: the robot's own body blocks a beam (no return), the range noise and
    the dropout from the brick's seeded stream, the range limits, the scan record. The sensors see the
    geometry the camera sees."""
    m, lm = lid.model, lid.lm
    p = data.site_xpos[lid.site].copy()
    R = data.site_xmat[lid.site].reshape(3, 3)
    vec = (lid.flat @ R.T).astype(np.float64)
    n = len(vec)
    gid = np.full(n, -1, np.int32)
    dist = np.full(n, -1.0, np.float64)
    gg = lid.geomgroup if lid.geomgroup is not None else np.ones(6, np.uint8)
    g1 = np.zeros(1, np.int32)
    p64 = np.asarray(p, np.float64)
    for i in range(n):
        d = mujoco.mj_ray(m, data, p64, vec[i], gg, 1, lid.own_body, g1)
        if d >= 0.0 and d <= lm.r_max:
            gid[i], dist[i] = g1[0], d
    sb = lid.selfblock
    if sb.excl.any():
        blocked = (gid >= 0) & sb.excl[np.maximum(gid, 0)]
        if sb.need_pass:
            d_self = sb.distances(data, p, vec, lm.r_max)
            d_scene = np.where((gid >= 0) & (dist >= 0), dist, np.inf)
            blocked |= np.isfinite(d_self) & (d_self <= d_scene)
        gid[blocked] = blidar.SELF_BLOCKED
        dist[blocked] = -1.0
    r = dist.astype(np.float32)
    r[(gid < 0) | (dist < 0)] = np.nan
    if lid.noise:
        ok = np.isfinite(r)
        r[ok] += (lid.rng.standard_normal(int(ok.sum())) * (lm.noise_a + lm.noise_b * r[ok])).astype(np.float32)
        r[lid.rng.random(n) < lm.dropout] = np.nan
    r[(r < lm.r_min) | (r > lm.r_max)] = np.nan
    ranges = r.reshape(lm.channels, lm.beams)
    ok = np.isfinite(r)
    pts = (lid.flat[ok] * r[ok, None]).astype(np.float32)
    ring = np.repeat(np.arange(lm.channels), lm.beams)[ok]
    az = np.tile(np.arange(lm.beams) * (2 * math.pi / lm.beams), lm.channels)[ok]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return blidar.LidarScan(ranges, pts, ring, az, T, gid.reshape(lm.channels, lm.beams))
