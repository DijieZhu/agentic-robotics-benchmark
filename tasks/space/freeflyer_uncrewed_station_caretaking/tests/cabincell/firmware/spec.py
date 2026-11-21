"""The task firmware's numbers.

Two sources, both read by key, never written here as literals:

* the runtime copy of the task's numbers, ``tests/cabincell/data/site_spec.json`` (generated, never edited by
  hand; checked against ``site_spec.sha256`` when it is loaded);
* the reference robot's value table, ``tests/cabincell/data/robot/spec_values.json`` (hidden side only).

``FirmwareSpec.load()`` returns the frozen set the firmware uses; ``apply_dock_defaults(spec)`` puts the berthing cone of
the task (6 deg about the berth axis, 10 deg in attitude) into the simulation package's dock defaults at initialisation.
Contract 1.2.0 carries every firmware constant as a ``firmware.*`` key (the 15 that this module kept in a
pending list until then are read from the runtime copy too, with the same values); ``firmware.fiducial.check_hz`` is the
reference robot's fiducial rate and is refused when it differs from the robot's value table.  ``near_clip_m`` is the
frame recipe's near clip (``render.near_clip_m``, the renderer's key): the camera-visibility tests apply the same rule
as the frames.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = os.path.dirname(HERE)
RUNTIME = os.path.join(CELL, "data", "site_spec.json")
RUNTIME_SHA = os.path.join(CELL, "data", "site_spec.sha256")
ROBOT_VALUES = os.path.join(CELL, "data", "robot", "spec_values.json")


class SpecError(RuntimeError):
    """The runtime copy does not match its checksum or misses a key the firmware needs."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_site_spec(path: str = RUNTIME, sha_path: str = RUNTIME_SHA) -> dict:
    """The parsed runtime copy after its sha256 check."""
    data = open(path, "rb").read()
    want = open(sha_path, "rb").read().decode("ascii").split()[0]
    got = _sha(data)
    if got != want:
        raise SpecError(f"site_spec.json sha256 {got} != {want} ({sha_path})")
    return json.loads(data.decode("ascii"))


def load_robot_values(path: str = ROBOT_VALUES) -> dict:
    """``values`` of the reference robot's value table (each entry a dict with ``value``)."""
    with open(path) as fh:
        return json.load(fh)["values"]


def _v(values: dict, key: str):
    if key not in values:
        raise SpecError(f"key {key!r} missing from the runtime site_spec")
    return values[key]


def _rv(robot: dict, key: str):
    if key not in robot:
        raise SpecError(f"key {key!r} missing from the reference robot's values")
    return robot[key]["value"]


@dataclasses.dataclass(frozen=True)
class FirmwareSpec:
    # capture
    gate_pos_m: float
    gate_axis_deg: float
    detector_hz: float
    lock_mean_of: int
    no_bar_s: float
    stalled_s: float
    capture_timeout_s: float
    close_depth_item_m: float
    close_depth_rail_m: float
    approach_mode: str
    approach_speed_m_s: float
    detector_sigma_pos_m: float
    detector_sigma_axis_deg: float
    lost_bar_s: float
    progress_m: float
    align_out_m: float
    align_tol_m: float
    align_speed_m_s: float
    align_att_tol_deg: float
    capture_deployed_tol_deg: float
    close_settle_s: float
    # zones
    inflation_stowed_m: float
    inflation_arm_m: float
    arm_deployed_tol_deg: float
    arm_envelope_margin_m: float
    approach_grow_m: float
    approach_depth_m: float
    dock_approach_grow_m: float
    dock_approach_depth_m: float
    # moves
    move_timeout_extra_s: float
    # fly_rel
    view_cone_deg: float
    min_visible_markers: int
    loss_hold_s: float
    rel_settle_s: float
    fiducial_check_hz: float
    # camera visibility: the frames' near clip
    near_clip_m: float
    # dock (this task's berthing cone)
    dock_cone_deg: float
    dock_att_deg: float
    # power and lamps
    lamp_power_w: float
    other_w: float
    low_flag_pct: float
    battery_hz: float
    battery_pct_resolution: float
    # lighting on the localisation
    light_factor: tuple
    # status() telemetry of the arm (reference robot)
    encoder_deg: float
    servo_hz: float
    current_lsb_a: float
    torque_per_amp_nm: float
    # rates
    sparse_map_hz: float
    # provenance
    site_spec_sha256: str
    contract_version: str

    @classmethod
    def from_values(cls, rt: dict, robot: dict, site_spec_sha256: str = "") -> "FirmwareSpec":
        V = rt["values"]
        gate = _v(V, "firmware.capture.gate")
        cd = _v(V, "firmware.capture.close_depth_mm")
        ap = _v(V, "firmware.capture.approach")
        dn = _v(V, "firmware.capture.detector_noise")
        mapper = _rv(robot, "mob.mapper")
        az = _v(V, "world.stowage_unit.approach_zone")
        lf = _v(V, "firmware.lighting.feature_factor")
        rates = _rv(robot, "loc.rates")
        servo = _rv(robot, "arm.joint_servo")
        dz = _v(V, "firmware.dock.approach_zone")
        fid_hz = float(_v(V, "firmware.fiducial.check_hz"))
        if fid_hz != float(rates["fiducial_Hz"]):
            raise SpecError(f"firmware.fiducial.check_hz {fid_hz} != the reference robot's loc.rates fiducial_Hz {rates['fiducial_Hz']}")
        return cls(
            gate_pos_m=float(gate["position_cm"]) / 100.0, gate_axis_deg=float(gate["axis_deg"]),
            detector_hz=float(_v(V, "firmware.capture.detector_hz")), lock_mean_of=int(_v(V, "firmware.capture.lock_mean_of")),
            no_bar_s=float(_v(V, "firmware.capture.no_bar_s")), stalled_s=float(_v(V, "firmware.capture.stalled_s")),
            capture_timeout_s=float(_v(V, "firmware.capture.timeout_s")),
            close_depth_item_m=float(cd["item_bar"]) / 1000.0, close_depth_rail_m=float(cd["map_handrail"]) / 1000.0,
            approach_mode=str(ap["mode"]).lower(), approach_speed_m_s=float(ap["speed_m_s"]),
            detector_sigma_pos_m=float(dn["position_mm"]) / 1000.0, detector_sigma_axis_deg=float(dn["axis_deg"]),
            lost_bar_s=float(_v(V, "firmware.capture.lost_bar_s")), progress_m=float(_v(V, "firmware.capture.progress_m")),
            align_out_m=float(_v(V, "firmware.capture.align_out_m")), align_tol_m=float(_v(V, "firmware.capture.align_tol_m")),
            align_speed_m_s=float(_v(V, "firmware.capture.align_speed_m_s")),
            align_att_tol_deg=float(_v(V, "firmware.capture.align_att_tol_deg")),
            capture_deployed_tol_deg=float(_v(V, "firmware.capture.deployed_tol_deg")),
            close_settle_s=float(_v(V, "firmware.capture.close_settle_s")),
            inflation_stowed_m=float(mapper["robot_radius_m"]) + float(mapper["collision_distance_m"]),
            inflation_arm_m=float(_v(V, "firmware.zone.inflation_arm_m")),
            arm_deployed_tol_deg=float(_v(V, "firmware.zone.arm_deployed_tol_deg")),
            arm_envelope_margin_m=float(_v(V, "firmware.zone.arm_envelope_margin_m")),
            approach_grow_m=float(az["outline_margin_m"]), approach_depth_m=float(az["normal_m"]),
            dock_approach_grow_m=float(dz["outline_margin_m"]), dock_approach_depth_m=float(dz["normal_m"]),
            move_timeout_extra_s=float(_v(V, "firmware.fly_to.timeout_extra_s")),
            view_cone_deg=float(_v(V, "firmware.fly_rel.view_cone_deg")),
            min_visible_markers=int(_v(V, "firmware.fly_rel.min_visible_markers")),
            loss_hold_s=float(_v(V, "firmware.fly_rel.loss_hold_s")), rel_settle_s=float(_v(V, "firmware.fly_rel.settle_s")),
            fiducial_check_hz=fid_hz,
            near_clip_m=float(_v(V, "render.near_clip_m")),
            dock_cone_deg=float(_v(V, "firmware.dock.cone_half_angle_deg")),
            dock_att_deg=float(_v(V, "firmware.dock.att_tolerance_deg")),
            lamp_power_w=float(_v(V, "firmware.lamp.power_w")), other_w=float(_v(V, "firmware.battery.other_w")),
            low_flag_pct=float(_v(V, "firmware.battery.low_flag_pct")),
            battery_hz=float(_rv(robot, "pwr.eps.telemetry_rate")),
            battery_pct_resolution=float(_v(V, "firmware.battery.pct_resolution")),
            light_factor=tuple(sorted((str(k), float(v)) for k, v in lf.items())),
            encoder_deg=float(servo["dr3_variant"]["resolution_deg"]), servo_hz=float(servo["telemetry_Hz"]),
            current_lsb_a=float(_v(V, "firmware.arm.current_lsb_a")),
            torque_per_amp_nm=float(_v(V, "firmware.arm.torque_per_amp_nm")),
            sparse_map_hz=float(rates["sparse_map_Hz"]),
            site_spec_sha256=site_spec_sha256, contract_version=str(rt.get("contract_version", "")),
        )

    @classmethod
    def load(cls, path: str = RUNTIME, sha_path: str = RUNTIME_SHA, robot_path: str = ROBOT_VALUES) -> "FirmwareSpec":
        rt = load_site_spec(path, sha_path)
        return cls.from_values(rt, load_robot_values(robot_path), _sha(open(path, "rb").read()))

    def lighting(self) -> dict:
        """{state: map-feature factor} ('off' = failed light circuit)."""
        return dict(self.light_factor)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["light_factor"] = self.lighting()
        return d


_DEFAULT = None


def default_spec() -> FirmwareSpec:
    """The spec of the published task (cached)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = FirmwareSpec.load()
    return _DEFAULT


def apply_dock_defaults(spec: FirmwareSpec) -> dict:
    """The berthing cone of this task (6 / 10 deg) is set in the simulation package's dock defaults when the
    firmware is initialised (the package keeps its own default).  Returns the values before the change."""
    from spacekit import primitives as PR
    before = dict(cone_half_angle_deg=PR.DOCK_DEFAULTS["cone_half_angle_deg"], att_tolerance_deg=PR.DOCK_DEFAULTS["att_tolerance_deg"])
    PR.DOCK_DEFAULTS["att_tolerance_deg"] = float(spec.dock_att_deg)
    if not math.isclose(float(PR.DOCK_DEFAULTS["cone_half_angle_deg"]), float(spec.dock_cone_deg)):
        PR.DOCK_DEFAULTS["cone_half_angle_deg"] = float(spec.dock_cone_deg)
    return before
