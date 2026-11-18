"""A catalogue as data (public code; every number comes from the quad-catalogue/1 passed in).

The grader passes the truth catalogue of its GT folder; an agent can pass its own transcription of the datasheet pages
and the quote (the same format, schemas.catalogue_schema). Each list is in the quote's order; the global quote order is
modules, packs, drivers, hubs, and it breaks the price ties of the optimum and of the repair.

    cat = Catalogue(data_or_path)
    cat.page_at(pid, v)       -> (peak torque N.m, no-load speed rad/s, peak current A) at the module voltage v
    cat.pack_window(pid)      -> (cells x cut-off, cells x full) V
    cat.fits(pid, inner_m)    -> the sorted outline fits the sorted inner box
    cat.bom_cost(bom)         -> the list price of an internal BOM (USD, 2 decimals)
"""
import copy
import json

from . import schemas as S
from .quadmodel import ProcedureError

_LISTS = (("modules", "actuator"), ("packs", "battery"), ("drivers", "driver"), ("hubs", "hub"))


class Catalogue:
    def __init__(self, data):
        if isinstance(data, str):
            with open(data) as fh:
                data = json.load(fh)
        probs = S.validate(data, "catalogue")
        if probs:
            raise ProcedureError("invalid catalogue: " + "; ".join(probs[:8]))
        self.data = copy.deepcopy(data)
        self.vendors = dict(self.data["vendors"])
        self.modules = list(self.data["modules"])
        self.packs = list(self.data["packs"])
        self.drivers = list(self.data["drivers"])
        self.hubs = list(self.data["hubs"])
        self._by = {}
        self._role = {}
        self.order = []
        for key, role in _LISTS:
            for item in getattr(self, key):
                pid = item["part_id"]
                if pid in self._by:
                    raise ProcedureError("catalogue part %r is listed twice" % pid)
                self._by[pid] = item
                self._role[pid] = role
                self.order.append(pid)
        self._index = {pid: i for i, pid in enumerate(self.order)}

    def has(self, pid):
        return pid in self._by

    def get(self, pid):
        return self._by[pid]

    def role(self, pid):
        return self._role[pid]

    def price(self, pid):
        return float(self._by[pid]["price_usd"])

    def index(self, pid):
        return self._index[pid]

    def quote_roles(self):
        return dict(self._role)

    def inspected(self, vendor):
        return bool(self.vendors[vendor]["inspected"])

    def page_at(self, pid, v):
        """Page values at the module voltage v (B-14 g): two points -> linear interpolation or extrapolation of each
        value; one point -> peak torque and peak current as printed, no-load speed in proportion to the voltage."""
        pts = sorted(self._by[pid]["points"], key=lambda p: float(p["voltage_v"]))
        keys = ("peak_torque_nm", "no_load_speed_rad_s", "peak_current_a")
        v = float(v)
        if len(pts) == 1:
            p = pts[0]
            v1 = float(p["voltage_v"])
            if not v1 > 0:
                raise ProcedureError("part %s: the voltage of its one page point is not positive" % pid)
            return float(p["peak_torque_nm"]), float(p["no_load_speed_rad_s"]) * v / v1, float(p["peak_current_a"])
        p1, p2 = pts[0], pts[-1]
        v1, v2 = float(p1["voltage_v"]), float(p2["voltage_v"])
        if not v2 > v1:
            raise ProcedureError("part %s: its two page points have the same voltage" % pid)
        return tuple(float(p1[k]) + (v - v1) * (float(p2[k]) - float(p1[k])) / (v2 - v1) for k in keys)

    def pack_window(self, pid):
        p = self._by[pid]
        n = int(p["cells"])
        cv = p["cell_voltage_v"]
        return (n * float(cv["cutoff"]), n * float(cv["full"]))

    @staticmethod
    def window_inside(window, vrange):
        return float(vrange[0]) - 1e-9 <= float(window[0]) and float(window[1]) <= float(vrange[1]) + 1e-9

    def fits(self, pid, inner_m):
        out = sorted(float(x) for x in self._by[pid]["outline_m"])
        inn = sorted(float(x) for x in inner_m)
        return all(a <= b + 1e-9 for a, b in zip(out, inn))

    def driver_count(self, pid):
        return -(-len(S.JOINTS) // int(self._by[pid]["channels"]))

    def bom_cost(self, bom):
        """round(4 x the three module prices + pack + driver_count x driver + hub, 2) of an internal BOM."""
        mods = sum(self.price(bom["modules"][t]) for t in S.JOINT_TYPES)
        total = (len(S.LEGS) * mods + self.price(bom["battery"]) + int(bom["driver_count"]) * self.price(bom["driver"])
                 + self.price(bom["hub"]))
        return round(total, 2)
