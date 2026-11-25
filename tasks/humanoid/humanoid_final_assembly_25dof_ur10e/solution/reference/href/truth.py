"""The true poses, read through the privileged calls of the oracle token (what an agent would have to perceive with
its cameras): every module's root pose and joints, every mate's carrier and part body pose, the hidden interfaces
(seat in the carrier body, interface frame in the part body, the design bolt position), the coupling's states."""
import math

import numpy as np

from . import tf


class Truth(object):
    def __init__(self, session):
        self.s = session
        dp = session.truth_describe()
        self.describe = dp
        self.ifaces = {}
        pos = {r["seat_class"]: int(r["bolt_positions"]) for r in (dp.get("capture_table") or {}).get("rows", [])}
        for x in dp["interfaces"]:
            x = dict(x)
            x["S"] = np.array(x["seat_in_carrier"], float)
            x["F"] = np.array(x["frame_in_part"], float)
            x["positions"] = pos[x["class"]]
            x["pitch"] = 2.0 * math.pi / x["positions"]
            self.ifaces[x["id"]] = x
        self.last = None

    def read(self):
        self.last = self.s.truth_state()
        return self.last

    def module_T(self, mid, st=None):
        st = st or self.last or self.read()
        r = st["modules"][mid]
        return tf.T_of(r["pos"], r["quat"])

    def body_T(self, iface, role, st=None):
        st = st or self.last or self.read()
        r = st["interface_bodies"][iface][role]
        return tf.T_of(r["pos"], r["quat"])

    def design_part_T(self, iface, k_offset=0, st=None):
        """the world pose of the part body at the design pose on its carrier as it stands now (k_offset bolt positions
        turned about the seat axis: 0 = the design; the reference never asks for another)"""
        x = self.ifaces[iface]
        Tc = self.body_T(iface, "carrier", st)
        return Tc @ x["S"] @ tf.rotz(k_offset * x["pitch"]) @ tf.inv(x["F"])

    def seat_errors(self, iface, st=None):
        """the coupling's reading of the part's interface frame against its seat now: axial, lateral (m), tilt,
        clock (rad), the nearest bolt position, the residual phase (rad)"""
        x = self.ifaces[iface]
        Tc, Tp = self.body_T(iface, "carrier", st), self.body_T(iface, "part", st)
        E = tf.inv(x["S"]) @ tf.inv(Tc) @ Tp @ x["F"]
        a, l, t, c, k, r = tf.seat_errors(E, x["pitch"])
        return {"axial_m": a, "lateral_m": l, "tilt_rad": t, "clock_rad": c, "k_near": k % x["positions"], "residual_rad": r,
                "E": E}

    def state_of(self, iface, st=None):
        st = st or self.last or self.read()
        return (st.get("states") or {}).get(iface)

    def record_of(self, iface, st=None):
        st = st or self.last or self.read()
        for r in st.get("records") or []:
            if r["id"] == iface:
                return r
        return None
