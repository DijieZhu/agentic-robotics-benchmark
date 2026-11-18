"""Poses of the kit's bodies with every unit at its designed slot (the assembled robot at the zero pose), robot frame."""
import numpy as np


def designed_candidates(kit):
    """{slot id: the candidate designed for that slot} (its unit's designed leg is the slot's leg, or the carrier unit's)"""
    B = kit["bodies"]
    out = {}
    for s in kit["slots"]:
        carrier_unit = B[s["carrier"]]["unit"]
        want = s.get("leg") or kit["units"][carrier_unit]["designed_leg"]
        out[s["id"]] = [c for c in s["candidates"] if kit["units"][c["unit"]]["designed_leg"] == want][0]
    return out


def design_poses(kit):
    """({body: 4x4 robot-frame pose}, {slot: designed candidate}) with every unit at its designed slot"""
    B = kit["bodies"]
    T = {"base": np.eye(4)}
    for bn, bb in B.items():
        if bb["parent"] == "base":
            T[bn] = np.array(bb["T_design"])
    designed = designed_candidates(kit)
    changed = True
    while changed:
        changed = False
        for s in kit["slots"]:
            c = designed[s["id"]]
            if s["carrier"] in T and c["part"] not in T:
                T[c["part"]] = T[s["carrier"]] @ np.array(c["T_part_in_carrier"])
                changed = True
                for bn, bb in B.items():
                    if bb["parent"] == c["part"]:
                        T[bn] = T[c["part"]] @ np.linalg.inv(np.array(B[c["part"]]["T_design"])) @ np.array(bb["T_design"])
    return T, designed
