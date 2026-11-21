"""The standard of each tag: the maker's triangles of its file, placed by reference/placement.json, in millimetres.

    std = standard.load(maker_root, placement_path)            # pins checked; placement re-derived and compared
    V, F = std.tag_mesh_mm("P03")                              # body frame at configuration Z, mm
    T = std.tag_triangles_mm("P03")                            # (M, 3, 3) corners V0, V1, V2
    S = std.file_triangles_mm("pmc.dae")                       # the file's own frame (node matrices applied), mm
    M = std.T_body_file_mm("P03")                              # 4 x 4, rotation as in placement.json, translation mm

The triangle order of a tag is its file's flat triangle list (dae.py); a pair's two tags (P02 / P03, P04 / P05) share
one file and therefore one numbering. Transforms are applied elementwise in a fixed order (dae.apply_matrix).
"""
import numpy as np

from . import dae
from . import pinned
from . import placement as placement_mod
from .errors import MakerStdError

TAGS = tuple(t for t, _k, _e in placement_mod.TAG_OF_ELEMENT)


class Standard:
    def __init__(self, maker_root, placement, pins):
        self.maker_root = maker_root
        self.placement = placement
        self.pins = pins
        self._tags = {t["tag"]: t for t in placement["tags"]}
        if tuple(sorted(self._tags)) != tuple(sorted(TAGS)):
            raise MakerStdError("placement.json lists tags %s, expected %s" % (sorted(self._tags), list(TAGS)))
        self._meshes = {}

    @property
    def tags(self):
        return TAGS

    def maker_file(self, tag):
        return self._tags[tag]["maker_file"]

    def mesh(self, maker_file):
        if maker_file not in self._meshes:
            self._meshes[maker_file] = dae.read(self.maker_root, maker_file, self.pins)
        return self._meshes[maker_file]

    def T_body_file_m(self, tag):
        return [list(map(float, row)) for row in self._tags[tag]["T_body_file_m"]]

    def T_body_file_mm(self, tag):
        T = self.T_body_file_m(tag)
        return [[T[0][0], T[0][1], T[0][2], T[0][3] * 1000.0],
                [T[1][0], T[1][1], T[1][2], T[1][3] * 1000.0],
                [T[2][0], T[2][1], T[2][2], T[2][3] * 1000.0],
                [0.0, 0.0, 0.0, 1.0]]

    def file_triangles_mm(self, maker_file):
        return self.mesh(maker_file).triangles_mm()

    def tag_mesh_mm(self, tag):
        m = self.mesh(self.maker_file(tag))
        return dae.apply_matrix(self.T_body_file_mm(tag), m.vertices_mm()), m.F

    def tag_triangles_mm(self, tag):
        V, F = self.tag_mesh_mm(tag)
        return V[F]

    def triangle_counts(self):
        return {t: self.mesh(self.maker_file(t)).triangle_count for t in TAGS}


def load(maker_root, placement_path, pins=None, check_placement=True):
    """The standard under maker_root. check_placement re-derives placement.json from the pinned texts (needs the
    description and context files under maker_root) and stops on any difference; without it the stored file is
    read as it is (a copy of the maker root that keeps only the meshes and the description)."""
    if pins is None:
        pins = pinned.load_pins(maker_root)
    if check_placement:
        pl = placement_mod.verify(maker_root, placement_path, pins)
    else:
        pl = placement_mod.load(placement_path)
    return Standard(maker_root, pl, pins)


def points_mm(triangles_mm, tri, u, v):
    """Points on given triangles: V0 + u (V1 - V0) + v (V2 - V0) (u, v as floats)."""
    T = np.asarray(triangles_mm)[np.asarray(tri, dtype=np.int64)]
    u = np.asarray(u, dtype=np.float64)[:, None]
    v = np.asarray(v, dtype=np.float64)[:, None]
    return (T[:, 0] + u * (T[:, 1] - T[:, 0])) + v * (T[:, 2] - T[:, 0])
