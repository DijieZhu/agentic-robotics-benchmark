"""The session ends of the zone rule and of the contact rule (safety.keep_in, safety.body_radius_m, safety.contact,
sessions.zone_file). The zone file in force for a session is the edition sessions.zone_file names for it; the world
holds byte copies of the editions in data/zones/ with their sha256 in data/zones/MANIFEST.json, checked here before
the zone map is built.

















"""
import hashlib
import json
import os
import re

import numpy as np

from .config import V, DATA
from . import kit_zones, kit_zones_parse, zone_exact

ZONES_DIR = os.path.join(DATA, "zones")
MANIFEST = os.path.join(ZONES_DIR, "MANIFEST.json")
EDITION_RE = re.compile(r"^the maker's edition (\S+\.bin) \(")
_MAPS = {}          # (edition sha256, radius) -> the kit's ZoneMap (one per process; the diagnostic)
_EXACT = {}         # edition sha256 -> ExactZones (one per process)


def table_key(session_name, day):
    """the session's key in sessions.zone_file: its own name, or 'development <day>' for a development session."""
    zf = V("sessions.zone_file")
    if session_name in zf:
        return session_name
    k = "development %s" % day
    if k in zf:
        return k
    raise KeyError("sessions.zone_file names no zone file for %s (day %s)" % (session_name, day))


def edition_in_force(session_name, day):
    txt = V("sessions.zone_file")[table_key(session_name, day)]["zone_file"]
    m = EDITION_RE.match(txt)
    if not m:
        raise NotImplementedError("the session takes no edition of the maker's file (the campaign volume is not built): %r" % txt)
    return m.group(1)


def paper_names_edition(paper_zone_file, edition):
    """the paper's zone file (a payload path) names the edition: its date (the edition name's first field) is one of the
    path's folders."""
    return isinstance(paper_zone_file, str) and edition.split("_", 1)[0] in paper_zone_file.replace("\\", "/").split("/")


class ZoneRule:
    def __init__(self, session_name, day, paper_zone_file=None):
        self.edition = edition_in_force(session_name, day)
        self.table_key = table_key(session_name, day)
        with open(MANIFEST) as f:
            man = json.load(f)["editions"]
        if self.edition not in man:
            raise FileNotFoundError("data/zones holds no copy of the edition %s" % self.edition)
        with open(os.path.join(ZONES_DIR, self.edition), "rb") as f:
            raw = f.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != man[self.edition]["sha256"]:
            raise ValueError("data/zones/%s is not the manifest's copy" % self.edition)
        self.radius = float(V("safety.body_radius_m"))
        assert np.isfinite(self.radius) and 0.0 < self.radius < zone_exact.NEAR_M
        # in force when the session's paper names a zone file in force (it must name the table's edition);
        # a session the table names under its own name (an acceptance session) must have one
        self.paper_zone_file = paper_zone_file
        if paper_zone_file:
            if not paper_names_edition(paper_zone_file, self.edition):
                raise ValueError("the paper of %s names the zone file %r, not the edition in force %s"
                                 % (session_name, paper_zone_file, self.edition))
            self.in_force = True
        else:
            if self.table_key == session_name:
                raise ValueError("%s is named in sessions.zone_file but its paper names no zone file in force" % session_name)
            self.in_force = False
        zones = None
        if self.sha256 not in _EXACT:
            zones = kit_zones_parse.parse_zones(raw)["zones"]
            _EXACT[self.sha256] = zone_exact.ExactZones(zones)
        self.exact = _EXACT[self.sha256]
        key = (self.sha256, self.radius)
        if key not in _MAPS:
            zones = zones or kit_zones_parse.parse_zones(raw)["zones"]
            _MAPS[key] = kit_zones.ZoneMap(zones, inflation=self.radius, inflation_z=self.radius)
        self.map = _MAPS[key]

    def keep_in(self, p):
        """the exact test: None when the sphere passes; else 'keep_in' or 'keep_out'."""
        return self.exact.breach(p, self.radius)

    def clearance(self, p):
        """d(p) = min(d_in, d_out), exact below zone_exact.NEAR_M (the record)."""
        return min(self.exact.point(p))

    def contact(self, p):
        return self.exact.contact(p)

    def kit_keep_in(self, p):
        """the kit's voxel validator beside (a diagnostic): 'ok', 'keep_in' or 'keep_out'."""
        return self.map.check_point(np.asarray(p, float))

    def describe(self):
        return {"edition": self.edition, "sha256": self.sha256, "table_key": self.table_key, "radius_m": self.radius,
                "in_force": self.in_force, "paper_zone_file": self.paper_zone_file,
                "test": "exact: the closed ball of radius safety.body_radius_m about the body origin inside the union of the closed "
                        "keep-in boxes and meeting no keep-out box (world/zone_exact.py)",
                "beside": "the kit's voxel validator (ZoneMap.check_point, inflation = the radius), a diagnostic"}
