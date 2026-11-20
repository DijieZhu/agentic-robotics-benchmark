"""Zone rules of the task firmware (robot_manual.md section 8, robot_api.md).

All checks use the simulation package's validator (``spacekit.zones.ZoneMap``: the firmware validator's voxel map,
dilated by the stowed inflation and, as a second map, by the arm-deployed inflation) and its band rule (a segment that
starts inside the inflated band may leave it or move without getting closer, ``band_exit=True``).  This module adds
the three task rules the package does not have:

* ``check_fly_rel``: a fly_rel segment is checked like a fly_to segment (arm-deployed map while the arm is deployed),
  except that its samples inside the approach zone of the plate's own stowage unit (or of the dock) are checked on the
  stowed map.  The segment is cut into runs of samples inside / outside that zone and every run is validated by the
  package's ``check_positions`` on its own map (an arm-map free sample is free on the stowed map, so a run that enters
  the zone starts free).  fly_to never has the exemption.
* ``check_envelope``: arm_deploy() / arm() toward a deployed attitude are refused when the robot's position (the
  estimate) is not free on the map dilated by the deployed robot's envelope margin (firmware.zone.arm_envelope_margin_m;
  the validator's model of the deployed robot is its body point plus that radius, the same model as the move check).
* approach zones: oriented boxes ``dict(center, R (columns a, b, n), half)``; ``ApproachZone.contains(P)``.
"""
from __future__ import annotations

import numpy as np

from spacekit.zones import ZoneMap, resample_positions

RATE_HZ = 10.0      # the validator's resampling rate [R spacekit zones]


class ApproachZone:
    """Oriented box (centre, rotation whose columns are the box axes, half sizes)."""

    def __init__(self, owner: str, center, R, half):
        self.owner = str(owner)
        self.c = np.asarray(center, float)
        self.R = np.asarray(R, float).reshape(3, 3)
        self.half = np.asarray(half, float)

    @classmethod
    def from_dict(cls, owner: str, d: dict) -> "ApproachZone":
        return cls(owner, d["center"], d["R"], d["half"])

    def contains(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        loc = (P - self.c) @ self.R
        return np.all(np.abs(loc) <= self.half + 1e-9, axis=1)

    def as_dict(self) -> dict:
        return dict(owner=self.owner, center=self.c.tolist(), R=self.R.tolist(), half=self.half.tolist())


def box_zone(owner: str, center, axes, outline_half, grow: float, depth: float) -> ApproachZone:
    """Approach zone of a part on a wall: the outline (half sizes along the two in-plane axes) grown by ``grow``, from the
    wall plane (through ``center``) to ``depth`` along the normal (third axis)."""
    a, b, n = (np.asarray(v, float) / np.linalg.norm(v) for v in axes)
    c = np.asarray(center, float) + n * (depth / 2.0)
    return ApproachZone(owner, c, np.column_stack([a, b, n]), [outline_half[0] + grow, outline_half[1] + grow, depth / 2.0])


class ZoneRules:
    """The task's zone checks around one ``ZoneMap`` (stowed inflation + arm-deployed inflation)."""

    def __init__(self, zones, spec, approach_zones: dict | None = None):
        self.spec = spec
        self.map = ZoneMap(zones, inflation=spec.inflation_stowed_m, inflation_arm=spec.inflation_arm_m)
        if abs(spec.arm_envelope_margin_m - spec.inflation_arm_m) < 1e-12:
            self.env_map = self.map
        else:
            self.env_map = ZoneMap(zones, inflation=spec.inflation_stowed_m, inflation_arm=spec.arm_envelope_margin_m)
        self.approach = dict(approach_zones or {})

    # ---------------------------------------------------------------- fly_rel
    def check_fly_rel(self, seg, owner: str | None, arm_deployed: bool, band_exit=True) -> tuple:
        """(result 'ok' | 'keep_in' | 'keep_out', first violating sample or -1, detail).  Stowed arm: the stowed map
        everywhere (the exemption only concerns the arm-deployed inflation)."""
        P = resample_positions(seg, RATE_HZ)
        if not len(P):
            return "ok", -1, dict(n=0)
        if not arm_deployed:
            r, k = self.map.check_positions(P, band_exit=band_exit, arm=False)
            return r, k, dict(n=len(P), runs=[("stowed", 0, len(P))])
        Z = self.approach.get(owner) if owner is not None else None
        inside = Z.contains(P) if Z is not None else np.zeros(len(P), bool)
        runs, k0 = [], 0
        for k in range(1, len(P) + 1):
            if k == len(P) or inside[k] != inside[k0]:
                runs.append((bool(inside[k0]), k0, k))
                k0 = k
        detail = dict(n=len(P), runs=[("zone" if z else "arm", a, b) for z, a, b in runs], inside=int(inside.sum()))
        for z, a, b in runs:
            r, k = self.map.check_positions(P[a:b], band_exit=band_exit, arm=not z)
            if r != "ok":
                return r, a + k, detail
        return "ok", -1, detail

    # ---------------------------------------------------------------- arm envelope
    def check_envelope(self, pos) -> str:
        """'ok' when the deployed robot at ``pos`` keeps the envelope margin from every occupied cell."""
        return self.env_map.check_point(np.asarray(pos, float), arm=True)

    def point(self, pos, arm: bool) -> str:
        return self.map.check_point(np.asarray(pos, float), arm=arm)

    def owner_of(self, pos) -> list:
        """Owners of the approach zones that contain ``pos``."""
        return sorted(o for o, Z in self.approach.items() if bool(Z.contains(pos)[0]))
