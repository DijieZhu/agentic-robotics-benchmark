"""server.py -- CabinCellServer: the grader-owned cell of one caretaking run and the program's calls on the wire.

One run = one site package (``<site_id>.site.tar.xz``, the site package format; an unpacked folder or a site dict in
tests) + one program.  ``reset`` builds the cell:

  site -> scene   the station builder (station.sites) through cabincell.site.build: stowage units at their
                  unit sites, the blocker and the pouches, the cargo bags, the per-circuit lighting; the robot and its
                  dock from the verified asset root (assetroot); MuJoCo arena 64 MB;
  state           the package's settled state (state0.json of the site's model with the robot, signature checked),
                  else the builder's initial state; an optional perturbation of every free body's start (run.perturb_um);
  wiring          station.site_rig.station_components (cabin air units and inlets, the stowage units' blowers, one
                  SuctionFaces at 50 Hz, the cabin + unit air field, the station model with the inhibited units);
                  firmware.stack_for_site (rig, localiser stack with the lighting richness, the dock of each berth,
                  the item aerodynamics with the robot's jets, the task firmware, berthing cone); the station service
                  (mission-second clock, its lighting hook drives the localisation and the renderer);
  pre-roll        PREROLL_S on the berth, then mission second 0: the firmware's clock (cell.budget.sim_time_s), the
                  capture budget hook, the recorder (events.py) and the contact events start.

``handle(method, params)`` answers one call (never raises): a call outside the whitelist answers
{"error": "forbidden"} (site_spec cell.forbidden_reply) and is counted; a whitelisted one goes to ``api.CellAPI``.  The
run ends when the program returns or crashes, when the site's simulated time is used up (TIME_UP), when the program
time (wall clock minus the cell's work) passes cell.budget.program_compute_s, or after cell.budget.silence_s without a
call.  ``finalize()`` then runs the firmware's stop() and hold (not on the dock), lets the world settle for
cell.settle_after_exit_s, takes the last world-state sample at that instant ('end') and returns the record
(the record format).  The in-process driver (``run_inprocess``) and the wire driver (``run_wire``: the program as a
child process on JSON lines, cellkit.wire.serve_episode with wire_guard.GuardedProcess) apply the same budgets.

In the shape of the quadruped plant inspection task's plantcell/server.py (the call boundary, the record, the
drivers); every part is this folder's own module, the settle is 60 s, the record is the record format's.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import resource
import sys
import tarfile
import time
import traceback

if __package__ in (None, ""):                         # run as a script: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "cabincell"

from . import CELL_VERSION, DATA_DIR, HERE, TESTS_DIR  # noqa: E402  (puts _lib and the cell dir on sys.path)
from . import api as API  # noqa: E402
from . import budgets as BU  # noqa: E402
from . import events as EV  # noqa: E402

import numpy as np  # noqa: E402

__all__ = ["CabinCellServer", "Cell", "run_inprocess", "run_wire", "open_site", "site_sha256", "noise_seed_for",
           "write_record", "model_signature", "PREROLL_S", "ARENA_BYTES"]

PREROLL_S = 2.0                 # scene initialisation on the berth before mission second 0 [G] (2 s)
ARENA_BYTES = 64 << 20          # the robot model's own <size memory="64M"/>
STATION_SERVICE_PERIOD_S = 0.1  # the station service's clock is driven at this period (at least once per second)
PRIVILEGED_METHODS = ("reference_run",)
AFTER_DONE_MAX_CALLS = 200     # calls answered after TIME_UP (in-process; the wire loop has its own limits)
EXIT_OF_REASON = {"eof": "returned", "returned": "returned", "crashed": "crashed", "time_up": "time_up",
                  "program_time_cap": "compute_budget", "compute_budget": "compute_budget", "silence": "silence",
                  "stalled": "silence", "protocol": "crashed", "internal_error": "cell_error", "cell_error": "cell_error",
                  "done": "time_up", "closed": "returned"}
SITE_PACKAGE_SCHEMA = "caretaking.site_package/1"
STATE0_SCHEMA = "caretaking.state0/1"
STATION_MEMBERS = ("mission/config_record.json", "mission/log.csv", "mission/advisory_log.json")


# ============================================================================================== site packages
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def site_sha256(path_or_members) -> str:
    """sha256 of a site archive's bytes; of a folder or a member dict: of the sorted member names and their sha256."""
    if isinstance(path_or_members, (str, os.PathLike)) and os.path.isfile(path_or_members):
        return _sha(open(path_or_members, "rb").read())
    members = path_or_members if isinstance(path_or_members, dict) else open_site(path_or_members)[1]
    h = hashlib.sha256()
    for name in sorted(members):
        h.update(f"{name} {_sha(members[name])}\n".encode())
    return h.hexdigest()


def open_site(src):
    """(site dict, members {name: bytes}, package sha256) of a site package (.site.tar.xz), an unpacked folder, a
    site.json file or a site dict (members then only {'site.json'}).  A package's MANIFEST.json hashes are checked."""
    if isinstance(src, dict):
        site = json.loads(json.dumps(src))
        raw = (json.dumps(site, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
        members = {"site.json": raw}
        return site, members, site_sha256(members)
    path = os.fspath(src)
    members = {}
    if os.path.isfile(path) and path.endswith(".tar.xz"):
        with tarfile.open(path, "r:xz") as tf:
            for ti in tf.getmembers():
                if not ti.isfile():
                    raise ValueError(f"{path}: member {ti.name} is not a regular file")
                if ti.name.startswith("/") or ".." in ti.name.split("/"):
                    raise ValueError(f"{path}: unsafe member name {ti.name!r}")
                members[ti.name] = tf.extractfile(ti).read()
        sha = _sha(open(path, "rb").read())
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                members[os.path.relpath(p, path).replace(os.sep, "/")] = open(p, "rb").read()
        sha = site_sha256(members)
    elif os.path.isfile(path):
        members = {"site.json": open(path, "rb").read()}
        sha = site_sha256(members)
    else:
        raise FileNotFoundError(path)
    if "MANIFEST.json" in members:
        man = json.loads(members["MANIFEST.json"].decode("utf-8"))
        if man.get("schema") != SITE_PACKAGE_SCHEMA:
            raise ValueError(f"{path}: MANIFEST schema {man.get('schema')!r}")
        for name, want in (man.get("members") or {}).items():
            if name not in members or _sha(members[name]) != want:
                raise ValueError(f"{path}: member {name} missing or its sha256 differs from the manifest")
    if "site.json" not in members:
        raise ValueError(f"{path}: no site.json")
    return json.loads(members["site.json"].decode("utf-8")), members, sha


def noise_seed_for(sha: str) -> int:
    """The run's noise seed when none is given: from the site package's sha256 (a 62-bit integer)."""
    return int(sha[:16], 16) >> 2


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def write_record(path: str, record: dict) -> str:
    """Write a record as JSON (sorted keys, one line per top-level value); returns its sha256."""
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    with open(path, "wb") as fh:
        fh.write(raw)
    return _sha(raw)


# ============================================================================================== the cell
class Cell:
    """The scene, wiring, firmware, station service, recorder and calls of one run (module doc)."""

    def __init__(self, site: dict, members: dict, site_sha: str, *, seed: int, perturb_um: float = 0.0, label: str = "",
                 cpu: str | None = None, renderer=None, overrides: dict | None = None, runtime: dict | None = None,
                 config: dict | None = None, log=None):
        import warnings

        import mujoco

        from . import grade as GR
        from . import predicates as P
        from . import station_service as SS
        from . import telemetry_synth as TS
        from .site import build as SB
        import firmware as FW
        from station import data as SD
        from station import scene as SC
        from station import site_rig as SR
        from station import sites as SI

        c0, w0 = time.process_time(), time.time()
        self.log = log or (lambda s: None)
        self.site, self.members, self.site_sha = site, members, site_sha
        self.seed, self.perturb_um, self.label = int(seed), float(perturb_um), str(label)
        self.cpu = cpu or _cpu_name()
        self.runtime = runtime or BU.load_runtime()
        self.config = config or GR.load_config()
        self.budgets = BU.Budgets(self.runtime["values"], overrides)
        self.renderer = renderer if renderer is not None else API.NullRenderer()
        mp = P.measure_params(self.config["values"])
        # ---- station members of the package (a package's members must equal a fresh synthesis)
        if all(k in members for k in STATION_MEMBERS):
            bad = TS.verify_members(site, members)
            if bad:
                raise ValueError(f"site package {site.get('site_id')}: {bad}")
        else:
            members = dict(members)
            members.update(TS.mission_members(site))
            self.members = members
        # ---- scene (cabincell.site.build.build with the arena of the robot model)
        S = SB.station_site(site)
        S["robot"] = dict(S.get("robot") or {}, present=True)
        paths = SB.robot_paths()
        side = SB.pad_side(site)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec, info = SI.build_site_spec(S, None, paths["robot_xml"], paths["dock_xml"], pad_fid_side=side)
            spec.memory = ARENA_BYTES
            model = spec.compile()
        SC.apply_near_clip(model)
        info["site_full"] = json.loads(json.dumps(site))
        info["pad_side"] = side
        self.model, self.info = model, info
        data = SB.initial_data(model, info)
        self.state0 = self._apply_state0(model, data, info, members, S)
        self.data = data
        self.perturbation = self._perturb(self.perturb_um) if self.perturb_um > 0 else []
        # ---- wiring
        comps = SR.station_components(model, data, info)
        self.comps = comps
        rb = site.get("robot") or {}
        self.soc0 = float(rb.get("battery_pct", 100.0)) / 100.0
        tf = FW.stack_for_site(model, data, info, comps, seed=self.seed, soc0=self.soc0)
        self.tf = tf
        self.body = tf.body
        mujoco.mj_forward(model, data)
        comps.suction.wrench(model, data)
        comps.suction.update_networks()
        # ---- pre-roll on the berth, then mission second 0
        tf.rig.run(PREROLL_S)
        self.t0 = float(data.time)
        tf.t0 = self.t0
        tf.clock = FW.Clock(data, self.t0, float(self.budgets.sim_time_s))
        tf.capture_budget = self.budgets.take_capture
        tf.calls = []
        comps.station.t = 0.0                    # the station's clock = mission seconds
        self.svc = SS.StationService(site, comps.station, on_lights=self._on_lights)
        svc = self.svc
        for mod, st in sorted(svc.lighting_effective().items()):
            self.renderer.set_lighting(mod, st)            # the renderer starts from the site's effective lighting
        tf.rig.sim.add_periodic(lambda sim: svc.on_step(comps.station.t), STATION_SERVICE_PERIOD_S, "cell_station_service")
        feats = SD.features(info.get("scene", "mission"))
        self.world = EV.WorldModel(model, info, site, comps, tf, P, mp)
        self.contacts = EV.GeomContactEvents(model, int(model.body_rootid[tf.body]), feats, lambda: self.t0)
        tf.rig.sim.add_periodic(lambda sim: self.contacts.update(sim.data), model.opt.timestep, "cell_contacts")
        keep = [(z["min"], z["max"]) for z in SD.zones(info.get("scene", "mission")) if z.get("type") == "keepout"]
        self.recorder = EV.Recorder(self, self.world, float(self.budgets.record_period_s), keep)
        self.recorder.attach(tf.rig.sim)
        self.t_docked, self.battery_at_docked = None, None
        self.energy0 = float(sum(tf.battery.used.values()))
        self.api = API.CellAPI(self)
        self.recorder.start()
        self.time_up = False
        self.ended = False
        self.program_end = None
        self.settled = False
        self.build_cpu_s = time.process_time() - c0
        self.build_wall_s = time.time() - w0
        self.run_cpu0 = time.process_time()
        self.run_wall0 = time.time()

    # -- state ------------------------------------------------------------------------------------------------
    def _apply_state0(self, model, data, info, members, S) -> dict | None:
        """The package's settled state (the site package format; written by the generator for the site's model with the robot on
        its berth, the site generator's settle.state0_of): nq, nv and the model signature must match, then qpos, qvel and time are
        taken as they are."""
        import mujoco
        raw = members.get("state0.json")
        if raw is None:
            return None
        st = json.loads(raw.decode("utf-8"))
        if st.get("schema") != STATE0_SCHEMA:
            raise ValueError(f"state0.json schema {st.get('schema')!r}")
        if int(st["nq"]) != model.nq or int(st["nv"]) != model.nv or st.get("model_signature") != model_signature(model):
            raise ValueError("state0.json was written for another model (nq / nv / names differ)")
        data.qpos[:] = np.asarray(st["qpos"], float)
        data.qvel[:] = np.asarray(st["qvel"], float)
        data.time = float(st.get("time", 0.0))
        mujoco.mj_forward(model, data)
        return dict(mode="direct", nq=int(model.nq), sha256=_sha(raw))

    def _perturb(self, um: float) -> list:
        """Every free body's start position + um micrometres in a seeded random direction (the hidden-set rule's robustness run)."""
        import mujoco
        m, d = self.model, self.data
        rng = np.random.default_rng([int(self.seed), 1234])
        out = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                a = int(m.jnt_qposadr[j])
                d.qpos[a:a + 3] += u * um * 1e-6
                out.append(dict(body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]), dir=u.tolist()))
        mujoco.mj_forward(m, d)
        return out

    def _on_lights(self, module: str, state: str) -> None:
        self.tf.set_lighting(module, state)
        self.renderer.set_lighting(module, state)

    # -- time ---------------------------------------------------------------------------------------------------
    def t_mission(self) -> float:
        return float(self.data.time) - self.t0

    def time_left(self) -> float:
        return float(self.tf.clock.left())

    def out_of_time(self) -> bool:
        """The site's simulated time is used up: less than half a physics step is left (the stepper rounds a run to
        whole steps, so a smaller remainder would never pass)."""
        return self.time_left() <= 0.5 * float(self.model.opt.timestep)

    def mark_time_up(self) -> None:
        self.time_up = True

    def on_docked(self) -> None:
        if self.t_docked is None:
            self.t_docked = round(self.t_mission(), 4)
            self.battery_at_docked = round(100.0 * float(self.tf.battery.soc), 6)

    # -- documents ----------------------------------------------------------------------------------------------
    def mission_document(self) -> dict:
        """mission() of the robot API: the package's mission members, the as-surveyed map, start charge and berth."""
        from . import telemetry_synth as TS
        M = self.members
        dec = lambda k: M[k].decode("utf-8") if k in M else None
        reg = json.loads(dec("mission/register.json")) if "mission/register.json" in M else _public_register(self.site)
        rb = self.site.get("robot") or {}
        return dict(work_order=dec("mission/work_order.txt") or "", log=TS.mission_log(dec("mission/log.csv")),
                    config_record=json.loads(dec("mission/config_record.json")),
                    advisory_log=json.loads(dec("mission/advisory_log.json")), register=reg, map=station_map(),
                    battery_pct=float(rb.get("battery_pct", 100.0)), berth=int(rb.get("berth", 1)))

    # -- the end ------------------------------------------------------------------------------------------------
    def settle(self) -> None:
        """The firmware's stop() and hold (not on the dock), the world for cell.settle_after_exit_s, the last sample."""
        if self.settled:
            return
        self.program_end = round(min(self.t_mission(), float(self.budgets.sim_time_s)) if self.out_of_time() else self.t_mission(), 4)
        if not self.tf.layer.docked:
            self.tf.stop()
        self.tf.rig.run(float(self.budgets.settle_after_exit_s))
        self.recorder.sample_now()
        self.settled = True

    def record(self, exit_: str, program_compute_s: float, wall_extra: dict | None = None) -> dict:
        """The record of the run (the record format, caretaking.record/1)."""
        self.settle()
        tf, R, svc = self.tf, self.recorder, self.svc
        rows = R.rows
        end = float(rows[-1]["t"])
        api = self.api
        calls = [dict(t=e["t"], call=e["call"], code=e["code"]) for e in getattr(self, "call_log", [])]
        energy = float(sum(tf.battery.used.values())) - self.energy0
        run_cpu = time.process_time() - self.run_cpu0
        sim_run = max(end, 1e-9)
        met = dict(cell_cpu_s=run_cpu, cell_wall_s=time.time() - self.run_wall0, cpu_per_sim_s=run_cpu / sim_run,
                   build_cpu_s=self.build_cpu_s, build_wall_s=self.build_wall_s, preroll_s=PREROLL_S,
                   recorder_cpu_s=R.sample_cpu_s, rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                   record_samples=len(rows), contact_events=self.contacts.n_events,
                   contact_events_recorded=len(self.contacts.events), frames=len(api.frames),
                   firmware_relatches=int(getattr(tf, "relatches", 0)), soc_end_pct=100.0 * float(tf.battery.soc))
        for k, v in sorted(self.contacts.by_class.items()):
            met[f"contacts_{k}"] = v
        for k, v in sorted(getattr(self, "call_counts", {}).items()):
            met[f"calls_{k}"] = v
        met.update(api.hint_metrics())
        met.update(wall_extra or {})
        met = {k: float(v) for k, v in met.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
        rec = dict(
            schema=EV.RECORD_SCHEMA, site_id=str(self.site.get("site_id")),
            run=dict(seed=int(self.seed), perturb_um=float(self.perturb_um), cpu=str(self.cpu), label=str(self.label)),
            spec=dict(runtime_sha256=self.runtime["sha256"], grader_config_sha256=self.config.get("sha256"),
                      site_package_sha256=self.site_sha),
            items=self.world.items_record(), faces=self.world.faces_record(),
            times=dict(program_end=float(self.program_end), end=end, settle_s=float(self.budgets.settle_after_exit_s), exit=exit_),
            budgets_used=self.budgets.used(float(self.program_end), max(0.0, float(program_compute_s)), svc.light_requests),
            state=rows, frames=api.frames, report=api.report, station_requests=[dict(e) for e in svc.requests],
            fan_stops=svc.fan_stops(), violations=[dict(e) for e in svc.violations],
            dock=dict(t_docked=self.t_docked, battery_pct_at_docked=self.battery_at_docked, docked_at_end=bool(tf.layer.docked),
                      item_contacts_after_dock={k: bool(v) for k, v in R.item_dock.items()}),
            energy_wh=max(0.0, energy), loss_of_control=R.loss_of_control(), hard_contacts=self.contacts.record(),
            keepout_s=round(float(R.keepout_s), 6), calls=calls,
            final=dict(inlets=dict(rows[-1]["inlets"]), fans={k: bool(v) for k, v in sorted(svc.fans_running().items())},
                       docked=bool(tf.layer.docked)),
            hashes=dict(state_sha256=R.state_sha256(), qpos_sha256=EV.qpos_sha256(self.data)),
            metrics=met)
        return _jsonable(rec)


def model_signature(model) -> dict:
    """{nbody, njnt, names_sha256} of a model: sha256 of 'body:<name>' for every body then 'joint:<name>' for every
    joint, one per line (the generator's definition, tests/review/sitegen/gen/settle.py model_signature)."""
    import mujoco
    names = ["body:" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(model.nbody)]
    names += ["joint:" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "") for j in range(model.njnt)]
    return dict(nbody=int(model.nbody), njnt=int(model.njnt), names_sha256=_sha("\n".join(names).encode("utf-8")))


def _public_register(site: dict) -> dict:
    """The register as the program sees it (no 'present', no 'item'; the site package format)."""
    R = site.get("register") or {}
    ent = [{k: v for k, v in e.items() if k not in ("present", "item")} for e in (R.get("entries") or [])]
    return dict(entries=ent)


def station_map() -> dict | None:
    """The as-surveyed map of ~/given/station_map/ (zones, fixtures, landmarks) when the cell can see a copy:
    CABINCELL_STATION_MAP, else tests/setup_payload/given/station_map; None when neither exists."""
    for d in (os.environ.get("CABINCELL_STATION_MAP"), os.path.join(TESTS_DIR, "setup_payload", "given", "station_map")):
        if d and os.path.isdir(d):
            out = {}
            for k in ("zones", "fixtures", "landmarks"):
                p = os.path.join(d, k + ".json")
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as fh:
                        out[k] = json.load(fh)
            return out
    return None


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ============================================================================================== the server
class CabinCellServer(object):
    """The call boundary of one run (module doc).  Surface used by the drivers and the verifier:

        server = CabinCellServer(config=None, privileged=False, limits=None, renderer=None, log=None, reference_dir=None)
        server.reset(site, seed=None, label="", perturb_um=0.0, cpu=None)     site: package path, folder, site.json, dict
        server.handle(method, params) -> result dict | {"error": code[, "detail"]}        (never raises)
        server.done, server.done_reason, server._end(reason), server.work_wall, server.wall_start,
        server.program_used, server.wall_used, server.steps_after_done                 (cellkit.wire.serve_episode)
        server.finalize(exit_code=None) -> the record (the record format)          server.close()

    limits: test overrides of the budgets (budgets.Budgets); the verifier passes none."""

    def __init__(self, config=None, privileged=False, limits=None, renderer=None, log=None, reference_dir=None, runtime=None):
        self.config, self.runtime = config, runtime
        self.privileged = bool(privileged)
        self.limits = dict(limits or {})
        self.renderer = renderer
        self.log = log or (lambda s: None)
        self.reference_dir = reference_dir
        self.cell = None
        self.done = True
        self.done_reason = None
        self.wall_start = time.time()
        self.work_wall = 0.0
        self.program_used = 0.0
        self.wall_used = 0.0
        self.steps_after_done = 0
        self.internal_errors = 0
        self.exit_code = None
        self._last_reply = None
        self._record = None
        self.reference_result = None
        self.calls_after_done = 0

    # ------------------------------------------------------------------ run
    def reset(self, site, seed=None, label="", perturb_um=0.0, cpu=None) -> dict:
        self.close()
        t0 = time.time()
        site_d, members, sha = open_site(site)
        self.cell = Cell(site_d, members, sha, seed=(noise_seed_for(sha) if seed is None else int(seed)), perturb_um=perturb_um,
                         label=label, cpu=cpu, renderer=self.renderer, overrides=self.limits, runtime=self.runtime,
                         config=self.config, log=self.log)
        self.cell.call_log, self.cell.call_counts = [], {}
        self.done, self.done_reason = False, None
        self.wall_start = time.time()
        self.work_wall = 0.0
        self.program_used = 0.0
        self.internal_errors = 0
        self.exit_code = None
        self._last_reply = None
        self._record = None
        self.calls_after_done = 0
        self.build_wall_s = time.time() - t0
        return {"ok": True, "site_id": site_d.get("site_id"), "seed": self.cell.seed}

    def close(self) -> None:
        if self.cell is not None:
            try:
                self.cell.renderer.close()
            except Exception:  # noqa: BLE001
                pass

    def _end(self, reason) -> None:
        """The run is over (the wire loop's reason, or the cell's own); the settle happens in finalize()."""
        if not self.done:
            self.done = True
            self.done_reason = str(reason)
        elif self.done_reason is None:
            self.done_reason = str(reason)

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        t0 = time.time()
        try:
            return self._dispatch(method, params, t0)
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
            self.internal_errors += 1
            self.log("[cell] internal error in %s\n%s" % (str(method)[:64], traceback.format_exc()))
            if self.cell is not None and str(method) in API.ACTION_CALLS:
                self._end("internal_error")
            return {"error": "internal_error"}
        finally:
            self.work_wall += time.time() - t0
            self._last_reply = time.time()

    def _dispatch(self, method, params, t0):
        c = self.cell
        name = method if isinstance(method, str) else "(not a name)"
        if c is None:
            return {"error": "no_run", "detail": "the run has not started"}
        if name in PRIVILEGED_METHODS:
            if not self.privileged:
                return self._forbidden(name)
            return getattr(self, name)()
        if name not in API.WHITELIST:
            return self._forbidden(name)
        if self.done and self.done_reason not in ("time_up",):
            return {"error": "run_over", "detail": f"the run has ended ({self.done_reason})"}
        if self.done:
            self.calls_after_done += 1               # after TIME_UP: queries answer, actions TIME_UP, then the run closes
            if self.calls_after_done > AFTER_DONE_MAX_CALLS:
                return {"error": "run_over", "detail": "the run has ended (time_up)"}
        # in-process budgets (the wire loop applies the same two itself)
        if self._last_reply is not None and t0 - self._last_reply > float(c.budgets.silence_s):
            self._end("silence")
            return {"error": "run_over", "detail": "silence budget"}
        if self.program_time(t0) > float(c.budgets.program_compute_s):
            self._end("program_time_cap")
            return {"error": "run_over", "detail": "program compute budget"}
        c.call_counts[name] = c.call_counts.get(name, 0) + 1
        res = c.api.call(name, params if params is not None else {})
        if name not in API.LOGGED_EVERY_CALL:
            c.call_log.append(dict(t=c.api.t(), call=name, code=res.get("code")))
        if c.time_up:
            self._end("time_up")
        return API_encode(res)

    def _forbidden(self, name):
        c = self.cell
        c.budgets.forbidden()
        c.call_log.append(dict(t=c.api.t(), call=name[:64], code="forbidden"))
        return {"error": str(c.budgets.forbidden_reply)}

    def program_time(self, now=None) -> float:
        now = time.time() if now is None else now
        return (now - self.wall_start) - self.work_wall

    # ------------------------------------------------------------------ privileged
    def reference_run(self):
        """The privileged reference program inside the cell (reference_dir/reference.py, run(server))."""
        import importlib.util
        if not self.reference_dir:
            return {"error": "no_reference", "detail": "this cell has no reference program"}
        path = os.path.join(self.reference_dir, "reference.py")
        spec = importlib.util.spec_from_file_location("cabincell_reference", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        self.reference_result = mod.run(self)
        return _jsonable({"ok": True, "wall_s": round(time.time() - t0, 2), "t": self.cell.api.t()})

    # ------------------------------------------------------------------ the record
    def finalize(self, exit_code=None) -> dict:
        """The record (the settle runs here once)."""
        if self._record is not None:
            return self._record
        if exit_code is not None:
            self.exit_code = exit_code
        if not self.done:
            self._end("returned")
        reason = self.done_reason or "returned"
        exit_ = EXIT_OF_REASON.get(reason, "cell_error")
        if exit_ == "returned" and self.exit_code not in (None, 0):
            exit_ = "crashed"
        if self.cell.time_up and exit_ in ("returned", "crashed"):
            exit_ = "time_up"
        extra = dict(program_wall_s=self.wall_used, cell_work_s=self.work_wall, internal_errors=self.internal_errors)
        self._record = self.cell.record(exit_, self.program_used if self.program_used else self.program_time(), extra)
        return self._record


def API_encode(res):
    return _jsonable(res)


# ============================================================================================== drivers
def run_inprocess(server: CabinCellServer, solve, robot) -> dict:
    """Run ``solve(robot)`` in this process (``robot``: a cabinkit Robot bound to ``server.handle``, e.g.
    ``cabinkit.client.connect_handler(server.handle)``); a raising program is a crash.  Returns the record."""
    server.wall_start = time.time()
    server.work_wall = 0.0
    code = 0
    try:
        solve(robot)
    except Exception:  # noqa: BLE001 - a crashing program is graded on the world it left
        code = 1
        server.log("[program] " + traceback.format_exc())
    server.wall_used = time.time() - server.wall_start
    server.program_used = server.program_time()
    if not server.done:
        server._end("returned" if code == 0 else "crashed")
    return server.finalize(exit_code=code)


def run_wire(server: CabinCellServer, argv, cwd, user=None, env=None, stderr_path=None, cpu_s=None, nonet=False, log=None) -> dict:
    """Run the program as a child process on JSON lines (cellkit.wire.serve_episode with the guard of
    wire_guard.GuardedProcess); the program-time and silence budgets are the wire loop's.  Returns the record."""
    from cellkit import wire
    from .wire_guard import GuardedProcess
    b = server.cell.budgets
    proc = GuardedProcess(argv, cwd, user=user, env=env, stderr_path=stderr_path, cpu_s=cpu_s, nonet=nonet)
    try:
        reason = wire.serve_episode(server, proc, float(b.program_compute_s), float(b.silence_s), log=log)
    finally:
        proc.terminate()
    rc = proc.proc.returncode
    if server.done_reason in ("eof", None) and rc not in (None, 0):
        server.done_reason = "crashed"
    server.log(f"[wire] end {reason}, exit code {rc}")
    return server.finalize(exit_code=rc)


# ============================================================================================== command line
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="run one program on one site in the caretaking cell; write the record")
    ap.add_argument("--site", required=True, help="<site>.site.tar.xz, an unpacked package, or site.json")
    ap.add_argument("--program", required=True, help="the program file (defines solve(robot))")
    ap.add_argument("--out", required=True, help="record JSON")
    ap.add_argument("--client-lib", required=True, help="the folder that holds cabinkit/ (given/lib)")
    ap.add_argument("--wire", action="store_true", help="run the program as a child process (cellkit wire)")
    ap.add_argument("--cellkit-dir", default=None, help="folder that holds cellkit/ (wire mode)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--perturb-um", type=float, default=0.0)
    ap.add_argument("--limits", default=None, help="JSON object of budget overrides (tests)")
    a = ap.parse_args(argv)
    logs = []
    srv = CabinCellServer(limits=json.loads(a.limits) if a.limits else None, log=logs.append)
    srv.reset(a.site, seed=a.seed, label=a.label, perturb_um=a.perturb_um)
    prog = os.path.abspath(a.program)
    if a.wire:
        if a.cellkit_dir and a.cellkit_dir not in sys.path:
            sys.path.insert(0, a.cellkit_dir)
        sys.path.insert(0, a.client_lib)
        from cabinkit import client as CK
        argv_p = CK.launcher_argv(sys.executable, os.path.abspath(a.client_lib), os.path.dirname(prog),
                                  os.path.splitext(os.path.basename(prog))[0])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        rec = run_wire(srv, argv_p, os.path.dirname(prog), env=env, stderr_path=a.out + ".stderr.txt", log=logs.append)
    else:
        import importlib.util
        sys.path.insert(0, a.client_lib)
        from cabinkit import client as CK
        spec = importlib.util.spec_from_file_location("cabincell_program", prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rec = run_inprocess(srv, mod.solve, CK.connect_handler(srv.handle))
    sha = write_record(a.out, rec)
    print(json.dumps(dict(record=a.out, sha256=sha, exit=rec["times"]["exit"], end=rec["times"]["end"],
                          state_sha256=rec["hashes"]["state_sha256"], qpos_sha256=rec["hashes"]["qpos_sha256"],
                          cpu_per_sim_s=rec["metrics"].get("cpu_per_sim_s"), logs=logs[-5:]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
