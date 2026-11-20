"""Reference-robot adapter: spec values + a compiled MuJoCo model of the free-flyer -> a wired simulation.

The model may be the part-level reference build or the test placeholder; both follow
``NAME_MAP`` (or pass a different map).  Plant parameters come from ``FreeFlyerSpec``: the firmware
reference by default, overridden key by key from a parameter-values JSON file (the shared values file of
the reference robot) when it provides them.  Nozzle positions / directions are read from the model's nozzle sites when present (so an as-built
model with moved nozzles produces as-built thrust lines), else from the spec table.

``FreeFlyerRig`` wires plant (fan propulsion), firmware (controller + allocator + executive), localisation,
IMU, battery, station frame, cabin air, nozzle jets and grille suction around one free-flyer in a scene and
steps them deterministically (physics every dt, firmware at 62.5 Hz, slow forces every ``slow_every`` steps).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import cabin as _cabin
from .control import ControlState, Firmware
from .propulsion import FanPropulsion, PropulsionParams, PropulsionWrench, impeller_power, load_firmware_reference
from .sensors import Battery, Imu, Localizer, true_state
from .stepping import Simulator

__all__ = ["NAME_MAP", "VALUE_KEYS", "NOZZLE_TABLE_KEYS", "FreeFlyerSpec", "FreeFlyerAdapter", "FreeFlyerRig"]

# Names in the free-flyer model (the reference builds v0, v1 and v1.1 use these; the test placeholder too).
NAME_MAP = {
    "root_body": "core",
    "root_joint": "freeflyer_root",
    "nozzle_sites": [f"nozzle_1_{k}" for k in range(1, 7)] + [f"nozzle_2_{k}" for k in range(1, 7)],
    "imu_site": "site_imu",
    "arm_joints": ["arm_tilt", "arm_pan"],
    "arm_actuators": ["act_arm_tilt", "act_arm_pan"],
    "gripper_actuator": "act_gripper",
    "gripper_joints": ["grip_left_proximal", "grip_left_distal", "grip_right_proximal", "grip_right_distal"],
    "gripper_bodies": ["gripper_left_proximal", "gripper_left_distal", "gripper_right_proximal", "gripper_right_distal"],
    "gripper_site": "gripper_center",
    "dock_site": "site_dock_connector",
    "cameras": {"forward_wide": "cam_fwd_wide", "aft_wide": "cam_aft_wide", "forward_imaging": "cam_fwd_img",
                "forward_depth": "cam_fwd_depth", "aft_depth": "cam_aft_depth"},   # neutral ids
}

# Keys looked up in the values file (first match wins) for each plant quantity.  The shared values file of the
# reference robot keeps flat dotted keys under "values" ({"value": ..., "unit": ..., "source": ...}) and lists
# under "tables"; nested dicts and top-level dotted keys are accepted too.
VALUE_KEYS = {
    "mass_kg": ["mass.ref.total", "ff.mass_properties.mass_kg", "ff.mass_kg"],
    "com_m": ["mass.ref.com", "ff.mass_properties.com_m", "ff.com_m"],
    "inertia_kgm2": ["mass.ref.inertia_diag", "ff.mass_properties.inertia_kgm2", "ff.inertia_kgm2"],
    "leak_area_m2": ["prop.nozzle.leak_area", "ff.propulsion.leak_area_m2"],
    "fan_levels_rpm": ["prop.fan.speeds", "ff.propulsion.fan_levels_rpm"],
    "drag": ["env.drag.model"],
    "battery_capacity_wh": ["pwr.battery.energy", "ff.battery.capacity_wh"],
    "battery_count": ["pwr.battery.default_count", "pwr.battery.count", "ff.battery.count"],
    # firmware configuration of the reference configuration (platform + arm module); the reference
    # build v1.1 keeps these values (its lead-ins are mass-compensated), spec_values 0.3.0 only added keys
    "firmware_mass_kg": ["mass.refcfg.total"],
    "firmware_com_m": ["mass.refcfg.com"],
    "firmware_inertia_kgm2": ["mass.refcfg.inertia"],
}
_FIRMWARE_MASS_KEYS = ("firmware_mass_kg", "firmware_com_m", "firmware_inertia_kgm2")
NOZZLE_TABLE_KEYS = ["nozzles"]          # under "tables": rows with pos_m, exhaust_dir, width_in / width_m, discharge_coeff


def _lookup(d: dict, dotted: str):
    for container in (d.get("values", {}), d):
        if isinstance(container, dict) and dotted in container:
            v = container[dotted]
            return v["value"] if isinstance(v, dict) and "value" in v else v
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, dict) and "value" in cur:
        return cur["value"]
    return cur


@dataclass
class FreeFlyerSpec:
    """Plant parameters of the free-flyer plus the firmware reference (its own copy of the constants)."""

    propulsion: PropulsionParams
    firmware: dict
    mass_kg: float
    com_m: np.ndarray
    inertia_kgm2: np.ndarray
    drag_cd: float = 1.05
    drag_area_m2: float = 0.092903
    battery_count: int = 4
    battery_capacity_wh: float = 49.0
    sources: dict = field(default_factory=dict)

    @classmethod
    def reference(cls) -> "FreeFlyerSpec":
        """All values from the firmware reference (data/firmware_reference.json)."""
        ref = load_firmware_reference()
        mp = ref["mass_properties_configured"]
        return cls(propulsion=PropulsionParams.from_dict(ref["propulsion"]), firmware=ref, mass_kg=float(mp["mass_kg"]),
                   com_m=np.asarray(mp["com_m"], float), inertia_kgm2=np.asarray(mp["inertia_kgm2"], float),
                   sources={"*": "firmware_reference"})

    @classmethod
    def reference_configuration(cls) -> "FreeFlyerSpec":
        """The reference robot as freeflyer_uncrewed_station_caretaking flies it: firmware reference constants (propulsion, controller, modes,
        impeller inertia 0.001 kg m^2 -- the firmware's own) with the configured mass properties of the
        reference configuration (platform + perching arm module stowed, 10.450 kg; ``mass.refcfg.*``)
        and the default battery count (``pwr.battery.default_count``), from ``data/reference_configuration.json``
        (a copy of the reference robot's values).  The plant itself takes mass and inertia from the MuJoCo model."""
        spec = cls.reference()
        rc = json.loads((Path(__file__).resolve().parent / "data" / "reference_configuration.json").read_text())
        spec.firmware["mass_properties_configured"] = copy.deepcopy(rc["mass_properties_configured"])
        spec.battery_count = int(rc["battery_count"])
        spec.sources.update(firmware_mass="reference_configuration (" + ", ".join(rc["source"]["keys"][:3]) + ")",
                            battery_count="reference_configuration (pwr.battery.default_count)")
        return spec

    @classmethod
    def from_values(cls, path, base: "FreeFlyerSpec | None" = None, firmware_config: str = "refcfg") -> "FreeFlyerSpec":
        """Override the plant values found in a values JSON file (missing keys keep the base value).
        firmware_config='refcfg' also configures the firmware with the file's reference-configuration mass properties
        (``mass.refcfg.*``) when present; 'firmware' keeps the firmware reference's own configuration."""
        spec = copy.deepcopy(base) if base is not None else cls.reference()
        p = Path(path)
        if not p.exists():
            return spec
        d = json.loads(p.read_text())
        found = {}
        for q, keys in VALUE_KEYS.items():
            for k in keys:
                v = _lookup(d, k)
                if v is not None:
                    found[q] = (k, v)
                    break
        P = spec.propulsion
        fm = {q: v for q, (k, v) in found.items() if q in _FIRMWARE_MASS_KEYS}
        if firmware_config == "refcfg" and len(fm) == 3:
            J = np.asarray(fm["firmware_inertia_kgm2"], float)
            spec.firmware["mass_properties_configured"] = dict(
                mass_kg=float(fm["firmware_mass_kg"]), com_m=[float(x) for x in fm["firmware_com_m"]],
                inertia_kgm2=(np.diag(J) if J.ndim == 1 else J).tolist())
            spec.sources["firmware_mass"] = "values_file:mass.refcfg.*"
        for q, (k, v) in found.items():
            if q in _FIRMWARE_MASS_KEYS:
                continue
            if q in ("mass_kg", "battery_capacity_wh"):
                setattr(spec, q, float(v))
            elif q == "battery_count":
                spec.battery_count = int(max(v) if isinstance(v, (list, tuple)) else v)
            elif q == "com_m":
                spec.com_m = np.asarray(v, float)
            elif q == "inertia_kgm2":
                a = np.asarray(v, float)
                spec.inertia_kgm2 = np.diag(a) if a.ndim == 1 else a
            elif q == "fan_levels_rpm":
                P.imp_levels = np.asarray(v, float) * 2 * math.pi / 60.0
            elif q == "leak_area_m2":
                P.leak_area = np.asarray(v, float)
            elif q == "drag":
                spec.drag_cd = float(v.get("Cd", spec.drag_cd))
                spec.drag_area_m2 = float(v.get("area_m2", spec.drag_area_m2))
            spec.sources[q] = f"values_file:{k}"
        tables = d.get("tables", {})
        for tk in NOZZLE_TABLE_KEYS:
            rows = tables.get(tk)
            if isinstance(rows, list) and len(rows) == 12:
                rows = sorted(rows, key=lambda r: r.get("index", 0))
                P.nozzle_position = np.asarray([r["pos_m"] for r in rows], float)
                P.nozzle_exhaust_dir = np.asarray([r["exhaust_dir"] for r in rows], float)
                P.discharge_coeff = np.asarray([r["discharge_coeff"] for r in rows], float).reshape(2, 6)
                w = [r.get("width_m", r.get("width_in", 0.0) * 0.0254) for r in rows[:6]]
                P.nozzle_width = np.asarray(w, float)
                spec.sources["nozzle_table"] = f"values_file:tables.{tk}"
                break
        return spec


class FreeFlyerAdapter:
    """Binds a compiled MjModel of a scene to the free-flyer's names (``name_map``, optional ``prefix``).

    Attributes: body (root body id), joint, qadr, dadr (free joint addresses), nozzle_pos / nozzle_dir
    (12, 3) in the root body frame (exhaust directions), arm / gripper ids (or empty), imu_site,
    cameras {role: name}.  Nozzle geometry is read from the nozzle sites if they exist, else from spec.
    """

    def __init__(self, model, spec: FreeFlyerSpec | None = None, name_map: dict | None = None, prefix: str = ""):
        self.model = model
        self.map = copy.deepcopy(NAME_MAP if name_map is None else name_map)
        self.spec = spec if spec is not None else FreeFlyerSpec.reference()
        self.prefix = prefix
        nm = self._n
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm(self.map["root_body"]))
        if self.body < 0:
            raise ValueError(f"root body {nm(self.map['root_body'])} not found")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm(self.map["root_joint"]))
        if jid < 0:
            jid = int(model.body_jntadr[self.body])
        self.joint = jid
        self.qadr = int(model.jnt_qposadr[jid])
        self.dadr = int(model.jnt_dofadr[jid])
        self.subtree = np.nonzero(model.body_rootid == model.body_rootid[self.body])[0]
        self.nozzle_from_sites = False
        self.nozzle_pos = self.spec.propulsion.nozzle_position.copy()
        self.nozzle_dir = self.spec.propulsion.nozzle_exhaust_dir.copy()
        sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm(s)) for s in self.map.get("nozzle_sites", [])]
        if sids and min(sids) >= 0:
            self.nozzle_from_sites = True
            self.nozzle_sites = np.array(sids)
            d = mujoco.MjData(model)
            mujoco.mj_kinematics(model, d)
            Rb = d.xmat[self.body].reshape(3, 3)
            pb = d.xpos[self.body]
            for i, s in enumerate(sids):
                self.nozzle_pos[i] = Rb.T @ (d.site_xpos[s] - pb)
                self.nozzle_dir[i] = Rb.T @ d.site_xmat[s].reshape(3, 3)[:, 2]
        self.arm_joints = self._ids(mujoco.mjtObj.mjOBJ_JOINT, self.map.get("arm_joints", []))
        self.arm_actuators = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, self.map.get("arm_actuators", []))
        g = self.map.get("gripper_actuator")
        self.gripper_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm(g)) if g else -1
        self.gripper_joints = self._ids(mujoco.mjtObj.mjOBJ_JOINT, self.map.get("gripper_joints", []))
        gb = self._ids(mujoco.mjtObj.mjOBJ_BODY, self.map.get("gripper_bodies", []))
        self.gripper_geoms = [gg for gg in range(model.ngeom) if model.geom_bodyid[gg] in set(gb)
                              and (model.geom_contype[gg] or model.geom_conaffinity[gg])]
        s = self.map.get("imu_site")
        self.imu_site = nm(s) if s and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm(s)) >= 0 else None
        self.cameras = {k: nm(v) for k, v in self.map.get("cameras", {}).items()
                        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, nm(v)) >= 0}

    def _n(self, name: str) -> str:
        return self.prefix + name

    def _ids(self, obj, names):
        ids = [mujoco.mj_name2id(self.model, obj, self._n(n)) for n in names]
        return [i for i in ids if i >= 0]

    @property
    def mass(self) -> float:
        """Total mass of the free-flyer subtree in the model [kg]."""
        return float(self.model.body_subtreemass[self.body])

    def mass_properties(self, data=None):
        """(mass, com_body (3,), inertia about the CoM in body axes (3, 3)) of the subtree at its current pose."""
        m = self.model
        d = data if data is not None else mujoco.MjData(m)
        if data is None:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
        Rb = d.xmat[self.body].reshape(3, 3)
        pb = d.xpos[self.body]
        M = 0.0
        c = np.zeros(3)
        for b in self.subtree:
            M += m.body_mass[b]
            c += m.body_mass[b] * d.xipos[b]
        c /= M
        J = np.zeros((3, 3))
        for b in self.subtree:
            R = d.ximat[b].reshape(3, 3)
            Jb = R @ np.diag(m.body_inertia[b]) @ R.T
            r = d.xipos[b] - c
            J += Jb + m.body_mass[b] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return M, Rb.T @ (c - pb), Rb.T @ J @ Rb

    def set_pose(self, data, pos, quat, vel=(0, 0, 0), omega_body=(0, 0, 0)) -> None:
        """Place the robot (free joint) at pos / quat (w,x,y,z) with world velocity and body rate."""
        data.qpos[self.qadr:self.qadr + 3] = pos
        data.qpos[self.qadr + 3:self.qadr + 7] = quat
        data.qvel[self.dadr:self.dadr + 3] = vel
        data.qvel[self.dadr + 3:self.dadr + 6] = omega_body

    def true_state(self, data):
        """(pos, quat, vel_world, omega_body) of the root body frame origin."""
        return true_state(self.model, data, self.body)


class FreeFlyerRig:
    """One free-flyer with plant, firmware and environment, stepped deterministically (see module doc).

    spec: a FreeFlyerSpec, or None / 'reference_configuration' (default: the firmware configured with the reference
    configuration's mass properties, ``FreeFlyerSpec.reference_configuration()``), or 'firmware' (the firmware
    reference's own configured mass properties = the platform without the arm, ``FreeFlyerSpec.reference()``).
    Main options: mode (flight mode name), allocator ('flight' | 'nnls'), rng (numpy Generator; None =
    noise-free truth to the firmware), station_frame (RotatingStationFrame or None), flow (velocity field
    or None), drag_bodies (extra body ids with drag; the robot always has drag unless drag=False),
    jet_targets (body ids blown by the nozzles), suction (list of GrilleSuction), slow_every (steps
    between refreshes of the slow forces), spin_up (start with the fans at the mode's speed),
    thrust_scale (None = firmware plant's 1.25), gyro ('physical' | 'firmware'), fan_efficiency (power),
    stop_feedforward (v0.4, ``control.Firmware``: 'zero' default, 'hold' = v0.3).
    """

    def __init__(self, model, data, spec: "FreeFlyerSpec | str | None" = None, name_map: dict | None = None, prefix: str = "",
                 mode: str = "standard", allocator: str = "flight", rng: np.random.Generator | None = None,
                 station_frame=None, flow=None, drag: bool = True, drag_bodies=(), drag_shapes=None, jet_targets=(),
                 suction=(), slow_every: int = 10, spin_up: bool = True, thrust_scale: float | None = None,
                 gyro: str = "physical", fan_efficiency: float = 0.52, localizer: Localizer | None = None,
                 battery: Battery | None = None, extra_sources=(), extra_slow_sources=(),
                 overspeed_stop: float | None = 0.5, overspeed_cutoff: float | None = 0.75, cutoff_cuts_power: bool = True,
                 stop_feedforward: str = "zero"):
        self.model, self.data = model, data
        if spec is None or spec == "reference_configuration":
            spec = FreeFlyerSpec.reference_configuration()
        elif spec == "firmware":
            spec = FreeFlyerSpec.reference()
        self.spec = spec
        self.adapter = FreeFlyerAdapter(model, self.spec, name_map, prefix)
        P = self.spec.propulsion.copy()
        P.nozzle_position = self.adapter.nozzle_pos.copy()
        P.nozzle_exhaust_dir = self.adapter.nozzle_dir.copy()
        self.plant = FanPropulsion(P, com=(0.0, 0.0, 0.0), gyro=gyro, thrust_scale=thrust_scale)
        self.fan_efficiency = fan_efficiency
        self.prop_src = PropulsionWrench(self.adapter.body, self.plant, P.nozzle_position)
        self.firmware = Firmware(None, allocator=allocator, mode=mode, ref=self.spec.firmware,
                                 overspeed_stop=overspeed_stop, overspeed_cutoff=overspeed_cutoff,
                                 stop_feedforward=stop_feedforward)
        self.cutoff_cuts_power = bool(cutoff_cuts_power)
        self.power_events = []
        self.rng = rng
        self.localizer = localizer if localizer is not None else (Localizer(rng) if rng is not None else None)
        self.imu = None
        if self.adapter.imu_site is not None and rng is not None:
            self.imu = Imu(model, self.adapter.imu_site, rng, frame=station_frame)
        self.battery = battery if battery is not None else Battery(self.spec.battery_count, self.spec.battery_capacity_wh)
        self.frame = station_frame
        self.flow = flow
        slow = []
        if station_frame is not None:
            slow.append(station_frame)
        self.air = None
        if drag:
            bodies = [self.adapter.body] + [int(b) for b in drag_bodies]
            shapes = [_cabin.DragShape("fixed", [self.spec.drag_area_m2], cd=self.spec.drag_cd)]
            shapes += list(drag_shapes) if drag_shapes is not None else [_cabin.shape_from_body(model, int(b)) for b in drag_bodies]
            self.air = _cabin.CabinAir(model, bodies, shapes, flow=flow)
            slow.append(self.air)
        self.jets = None
        if len(jet_targets):
            self.jets = _cabin.NozzleJets(model, self.adapter.body, P.nozzle_position, P.nozzle_exhaust_dir,
                                          lambda: self.plant.thrust, list(jet_targets),
                                          exit_area_fn=lambda: self.plant.area * np.concatenate(P.discharge_coeff),
                                          ambient=flow)
            slow.append(self.jets)
        self.suction = list(suction)
        slow += self.suction
        slow += list(extra_slow_sources)
        self.sim = Simulator(model, data, sources=[self.prop_src] + list(extra_sources), slow_sources=slow,
                             slow_every=slow_every)
        self.period = 1.0 / float(self.spec.firmware["control"]["rate_hz"])
        self.sim.add_periodic(self._tick, self.period, "firmware")
        self.est = None
        self.log = []
        self.record_every = 0
        self._tick_count = 0
        mujoco.mj_forward(model, data)
        if spin_up:
            self.spin_up()

    # -- setup -----------------------------------------------------------------------------------------
    def spin_up(self) -> None:
        """Put both fans at the current mode's commanded speed with a settled speed loop."""
        lvl = self.firmware.speed_level()
        P = self.plant.p
        for m in self.plant.modules:
            if lvl == 0:
                w = 0.0
            else:
                cmd = int(P.imp_cmd_per_rad_s * P.imp_levels[lvl - 1] + 0.5)
                w = cmd / P.imp_cmd_per_rad_s
            m.impeller_speed = m.last_speed = m.prev_speed_rate = w
            v_ss = w / P.imp_speed_k + (P.imp_friction * w / P.imp_torque_k) * P.imp_r
            ki = m.speed_pid.ki
            m.speed_pid.integral = v_ss / ki if ki else 0.0
            m.speed_pid.last = 0.0

    def estimate(self) -> ControlState:
        """Current estimate handed to the firmware (truth when no localiser / rng)."""
        if self.localizer is None:
            pos, quat, vel, om = self.adapter.true_state(self.data)
            return ControlState(pos, quat, vel, om, 0)
        e = self.localizer.update(self.data.time, self.model, self.data, self.adapter.body)
        self.loc_out = e
        return ControlState(e["pos"], e["quat"], e["vel"], e["omega"], e["confidence"])

    def impeller_power_w(self) -> float:
        """Electrical power of both fan motors [W] from the steady-state model (propulsion.impeller_power)."""
        tot = 0.0
        for m in self.plant.modules:
            if m.powered and m.impeller_speed > 1.0:
                tot += impeller_power(self.plant.p, m.impeller_speed, m.total_area(), self.fan_efficiency)["power_w"]
        return tot

    def _arm_mech_power(self) -> float:
        d = self.data
        p = 0.0
        for j in self.adapter.arm_joints:
            dof = self.model.jnt_dofadr[j]
            p += abs(float(d.qfrc_actuator[dof] * d.qvel[dof]))
        return p

    def _tick(self, sim) -> None:
        d, m = self.data, self.model
        # kinematics of the current state (after mj_step2 the positional quantities are one step old);
        # no collision or constraint pass needed here
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        est = self.estimate()
        self.est = est
        _, _, _, om_true = self.adapter.true_state(d)
        out = self.firmware.tick(d.time, est, self.plant.impeller_speed)
        if self.cutoff_cuts_power and self.firmware.result == "overspeed_cutoff" and any(self.plant.powered):
            self.set_propulsion_power(False)          # the computer commands the power unit off
        self.plant.step(out["impeller_cmd"], out["servo_cmd"], om_true)
        self.battery.update(self.period, self.impeller_power_w(), self._arm_mech_power())
        if self.imu is not None:
            self.imu_out = self.imu.read(self.model, d)
        self._tick_count += 1
        if self.record_every and self._tick_count % self.record_every == 0:
            self.log.append(self.telemetry())

    # -- commands --------------------------------------------------------------------------------------
    def set_propulsion_power(self, on: bool, modules=(0, 1)) -> None:
        """Power / cut the propulsion channels (fan drive, nozzle servos, module board) of the given modules: a cut
        fan coasts, its flaps hold (``propulsion.FanModule.set_power``).  The over-speed cutoff calls this."""
        self.plant.set_power(on, modules)
        self.power_events.append((float(self.data.time), bool(on), tuple(int(i) for i in modules)))

    def set_mode(self, name: str) -> None:
        """Change flight mode (speed level changes are prepared by the executive)."""
        self.firmware.set_mode(name)

    def move_to(self, pos, quat, faceforward: bool = True, desired: dict | None = None):
        """Plan and start a move from the current estimate to (pos, quat)."""
        est = self.est if self.est is not None else self.estimate()
        return self.firmware.move([(np.asarray(pos, float), np.asarray(quat, float))], faceforward, self.data.time, est,
                                  desired)

    def hold(self) -> None:
        """Station keeping at the current pose."""
        self.firmware.hold()

    def stop(self) -> None:
        """Stop motion."""
        self.firmware.stop()

    def idle(self) -> None:
        """Zero thrust command (drift)."""
        self.firmware.idle()

    def set_arm(self, targets) -> None:
        """Position targets [rad] of the arm joint actuators."""
        for a, q in zip(self.adapter.arm_actuators, targets):
            self.data.ctrl[a] = q

    def set_gripper(self, target: float) -> None:
        """Position target of the gripper actuator."""
        if self.adapter.gripper_actuator >= 0:
            self.data.ctrl[self.adapter.gripper_actuator] = target

    def step(self, n: int = 1) -> None:
        """Advance n physics steps."""
        self.sim.step(n)

    def run(self, duration: float) -> None:
        """Advance by duration seconds."""
        self.sim.run_until(self.data.time + duration)

    def run_until_done(self, timeout: float, states=("stopped", "idle")) -> bool:
        """Step until the executive reaches one of ``states`` (after leaving 'moving'), or timeout."""
        t_end = self.data.time + timeout
        while self.data.time < t_end:
            self.step(int(round(self.period / self.model.opt.timestep)))
            if self.firmware.state in states:
                return True
        return False

    def telemetry(self) -> dict:
        """Snapshot for logs: time, true pose, estimate, firmware state, thrusts, fan speeds, battery."""
        pos, quat, vel, om = self.adapter.true_state(self.data)
        fw = self.firmware.last
        return dict(t=float(self.data.time), pos=pos, quat=quat, vel=vel, omega=om,
                    est_pos=None if self.est is None else np.asarray(self.est.pos).copy(),
                    state=self.firmware.state, result=self.firmware.result, mode=self.firmware.mode.name,
                    force_cmd=np.asarray(fw.get("force", np.zeros(3))).copy(),
                    torque_cmd=np.asarray(fw.get("torque", np.zeros(3))).copy(),
                    thrust=self.plant.thrust.copy(), fan_speed=self.plant.impeller_speed.copy(),
                    battery_soc=self.battery.soc, power_w=self.battery.last_power_w,
                    traj_error_pos=float(fw.get("traj_error_pos", 0.0)), traj_error_att=float(fw.get("traj_error_att", 0.0)))
