"""Acoustic surface classes and the geom -> class map used by sonar, DVL and altimeter models.

Backscattering strength per unit area and unit solid angle (linear, "sigma"), incidence angle i from the normal:

    sigma(i) = mu * cos(i)^2  +  s * exp(-0.5 * (i / w)^2)

  * first term: Lambert's law (rough surface), mu = 10^(lambert_db / 10)
  * second term: coherent / specular lobe of smooth hard surfaces (steel, concrete), peak s = 10^(specular_db/10),
    width w.
Values are order-of-magnitude design values for 0.1-1 MHz (self-defined; consistent with the usual ranking
mud < sand < gravel < rock < structures; not fitted to measurements).  They are properties of the site, not of the sensors.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import mujoco


@dataclass(frozen=True)
class AcousticMaterial:
    name: str
    lambert_db: float
    specular_db: float = -200.0
    specular_width_deg: float = 4.0
    note: str = ""

    @property
    def mu(self) -> float:
        return 10.0 ** (self.lambert_db / 10.0)

    @property
    def spec(self) -> float:
        return 10.0 ** (self.specular_db / 10.0)


ACOUSTIC_MATERIALS: dict[str, AcousticMaterial] = {m.name: m for m in [
    AcousticMaterial("soft_mud", -36.0, note="fluid mud, absorbs; DVL loses lock early"),
    AcousticMaterial("silt", -31.0),
    AcousticMaterial("sand", -25.0),
    AcousticMaterial("gravel", -20.0, note="gravel / shell hash"),
    AcousticMaterial("rock", -16.0),
    AcousticMaterial("concrete", -15.0, specular_db=0.0, specular_width_deg=5.0),
    AcousticMaterial("steel", -19.0, specular_db=10.0, specular_width_deg=3.0, note="painted or bare steel"),
    AcousticMaterial("biofouled", -11.0, specular_db=-6.0, specular_width_deg=8.0, note="steel/concrete with growth"),
    AcousticMaterial("plastic", -24.0, specular_db=-6.0, specular_width_deg=4.0, note="HDPE, rope, nets"),
    AcousticMaterial("absorber", -45.0, note="acoustic absorber / anechoic, near-invisible"),
]}

# Default name rules: (regex on lower-case "geomname|materialname", class).  First match wins.
DEFAULT_RULES: list[tuple[str, str]] = [
    (r"absorb|anechoic", "absorber"),
    (r"mud", "soft_mud"),
    (r"silt|clay", "silt"),
    (r"gravel|shell", "gravel"),
    (r"foul|barnacle|growth|mussel", "biofouled"),
    (r"concrete|mattress|caisson|quay|pile", "concrete"),
    (r"steel|metal|iron|pipe|tank|hull|valve|manifold|frame|anode", "steel"),
    (r"rock|boulder|stone|reef", "rock"),
    (r"plastic|hdpe|rope|net|buoy|float", "plastic"),
    (r"sand|seabed|seafloor|floor|bottom|ground|terrain", "sand"),
]


class MaterialMap:
    """Per-geom acoustic class lookup (vectorised arrays indexed by geom id)."""

    def __init__(self, m: mujoco.MjModel, rules: Optional[Sequence[tuple[str, str]]] = None,
                 default: str = "rock", overrides: Optional[dict[str, str]] = None,
                 table: Optional[dict[str, AcousticMaterial]] = None):
        self.table = dict(ACOUSTIC_MATERIALS if table is None else table)
        self.names = list(self.table)
        rules = list(DEFAULT_RULES if rules is None else rules)
        overrides = overrides or {}
        idx = np.zeros(m.ngeom, np.int32)
        for g in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            mid = int(m.geom_matid[g])
            mname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or "") if mid >= 0 else ""
            cls = overrides.get(gname) or (overrides.get(mname) if mname else None)
            if cls is None:
                key = f"{gname}|{mname}".lower()
                for pat, c in rules:
                    if re.search(pat, key):
                        cls = c
                        break
            cls = cls or default
            idx[g] = self.names.index(cls)
        self.geom_class = idx
        mats = [self.table[n] for n in self.names]
        self._mu = np.array([x.mu for x in mats])
        self._spec = np.array([x.spec for x in mats])
        self._w = np.radians(np.array([x.specular_width_deg for x in mats]))

    def class_of(self, geom_id: int) -> str:
        return self.names[int(self.geom_class[geom_id])] if geom_id >= 0 else ""

    def sigma(self, geom_ids: np.ndarray, cos_i: np.ndarray) -> np.ndarray:
        """Backscattering cross-section per unit area per sr (linear).  geom_ids < 0 -> 0."""
        gid = np.asarray(geom_ids)
        ok = gid >= 0
        k = self.geom_class[np.where(ok, gid, 0)]
        ci = np.clip(np.abs(cos_i), 0.0, 1.0)
        inc = np.arccos(ci)
        s = self._mu[k] * ci * ci + self._spec[k] * np.exp(-0.5 * (inc / self._w[k]) ** 2)
        return np.where(ok, s, 0.0)
