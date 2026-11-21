"""ref_truth.py -- the ONLY place where the reference program reads the cell (token privilege).

Five privileged quantities are allowed; each method below is tagged with the one it reads (``PRIV``), and every
call is counted (``Truth.reads``, returned in the reference's summary):

  P1  the robot's true pose (for the capture hints: the true bar pose expressed in the true body frame);
  P2  the true poses of the grasp features (capture hints, pre-grasp poses) -- with their identity: which item a
      feature belongs to, the item's register reference (what the operator reads off the item's class and colour in
      the forward imaging frame) and the inlet the item lies on (what the survey frame shows);
  P3  'something in the gripper' (the grader's cage test with the tendon commanded closed; and the same cage with
      the gripper open: is a bar still between the fingers -- 'the finger cage is empty' before closing);
  P4  the attitude of the item in the hand (the deposit is planned on the held item's pose in the palm frame);
  P5  whether an item is on a suction face after a placement (sealed area, share of its own footprint).

No free item is ever read: ``feature``, ``held_pose`` and ``on_face`` answer only for an item that is held or that
lies on a suction face (sealed area > 0 on some face: a return inlet, a holding screen or a pouch pad); for an item
that floats they answer None (and the refusal is counted).  Nothing here changes the cell.

Public constants read through the cell (not privileged; tagged ``PUBLIC``): the runtime site_spec values (budgets,
reference.* recipe values, the registered stowage-unit sites -- the map), the palm transform of the deployed arm (the
robot manual's grasp point relative to the body).
"""
from __future__ import annotations

import numpy as np

PRIVILEGES = {
    "P1": "the robot's true pose",
    "P2": "the true poses (and identity) of the grasp features",
    "P3": "something in the gripper (the finger cage)",
    "P4": "the attitude of the item in the hand",
    "P5": "whether an item is on a suction face after a placement",
}


def priv(tag: str):
    """Tag a Truth method with the privilege it reads (or 'PUBLIC')."""
    def deco(fn):
        fn.PRIV = tag
        return fn
    return deco


class Truth:
    """The privileged reads of one run (module doc)."""

    def __init__(self, server):
        self._cell = server.cell
        self.reads = {}
        self.refusals = {}

    def _count(self, name: str) -> None:
        self.reads[name] = self.reads.get(name, 0) + 1

    def _refuse(self, name: str) -> None:
        self.refusals[name] = self.refusals.get(name, 0) + 1

    # ------------------------------------------------------------------ public constants
    @priv("PUBLIC")
    def runtime_values(self) -> dict:
        """The runtime site_spec values (checked against their sha256 file by the cell)."""
        self._count("runtime_values")
        return self._cell.runtime["values"]

    @priv("PUBLIC")
    def palm_geometry(self) -> dict:
        """The deployed palm in the body frame (p, R) and the palm plane offset (robot manual constants)."""
        self._count("palm_geometry")
        lay = self._cell.tf.layer
        pb, Rb = lay.T_body_palm_deployed
        return dict(pb=np.asarray(pb, float).copy(), Rb=np.asarray(Rb, float).copy(), z_plane=float(lay.z_plane))

    # ------------------------------------------------------------------ P1
    @priv("P1")
    def robot_pose(self):
        """(position, rotation body -> world) of the robot body, true."""
        self._count("robot_pose")
        c = self._cell
        b = c.tf.body
        return np.asarray(c.data.xpos[b], float).copy(), np.asarray(c.data.xmat[b], float).reshape(3, 3).copy()

    # ------------------------------------------------------------------ helpers (no reads of their own)
    def _bars(self):
        return [b for b in self._cell.tf.detector.bars if getattr(b, "kind", "") in ("strap", "loop", "handle")]

    def _attached(self, item: str) -> bool:
        """The item is held (tendon closed, bar in the cage) or lies on some suction face (sealed area > 0)."""
        c = self._cell
        if c.world.in_gripper(c.data, item):
            return True
        seal = c.world.seal_by_face(c.data)
        return any(a.get(item, 0.0) > 0.0 for a in seal.values())

    def _bar_in_cage(self, b) -> bool:
        c = self._cell
        lay = c.tf.layer
        P = c.world.P
        pr, Rr = lay.palm_frame()
        cc, a, _ = b.pose(c.data)
        pts = P.to_palm(P.bar_points(cc, a, b.length), pr, Rr)
        return bool(P.bar_in_cage(pts, c.world.mp["cage_mm"], lay.z_plane))

    # ------------------------------------------------------------------ P2
    @priv("P2")
    def features(self) -> list:
        """Identity of every grasp feature of the site's items: [dict(name, item, kind, length)] (no poses)."""
        self._count("features")
        return [dict(name=b.name, item=b.name.split(":")[0], kind=b.kind, length=float(b.length)) for b in self._bars()]

    @priv("P2")
    def feature(self, name: str):
        """(centre, axis, face normal) of a grasp feature, true -- only while its item is held or on a suction face."""
        self._count("feature")
        item = name.split(":")[0]
        if not self._attached(item):
            self._refuse("feature")
            return None
        b = next(x for x in self._bars() if x.name == name)
        cc, a, n = b.pose(self._cell.data)
        return np.asarray(cc, float).copy(), np.asarray(a, float).copy(), np.asarray(n, float).copy()

    @priv("P2")
    def register_ref(self, item: str) -> str | None:
        """The register reference of an item (what the operator reads off its class and colour in the frame)."""
        self._count("register_ref")
        it = self._cell.world.items.get(item)
        return None if it is None else it.get("register_ref")

    # ------------------------------------------------------------------ P3
    @priv("P3")
    def holding(self, item: str | None = None) -> bool:
        """The item (any item when None) in the closed gripper (tendon commanded closed, a bar in the cage)."""
        self._count("holding")
        c = self._cell
        names = [item] if item else list(c.world.items)
        return any(c.world.in_gripper(c.data, k) for k in names)

    @priv("P3")
    def bar_between_fingers(self, item: str | None = None) -> bool:
        """Some bar of the item (any item when None) inside the finger cage, whatever the tendon (the open hand still
        around a handle; 'the cage is empty' before the gripper closes)."""
        self._count("bar_between_fingers")
        names = [item] if item else None
        return any(self._bar_in_cage(b) for b in self._bars() if names is None or b.name.split(":")[0] in names)

    # ------------------------------------------------------------------ P4
    @priv("P4")
    def held_pose(self, item: str) -> dict | None:
        """The held item in the palm frame: every body's (position, rotation) relative to the palm, and the grasped
        bar's centre and axis -- only while the item is held (or its bar is still between the fingers)."""
        self._count("held_pose")
        c = self._cell
        if not (c.world.in_gripper(c.data, item) or any(self._bar_in_cage(b) for b in self._bars() if b.name.startswith(item + ":"))):
            self._refuse("held_pose")
            return None
        lay = c.tf.layer
        pr, Rr = lay.palm_frame()
        pr, Rr = np.asarray(pr, float), np.asarray(Rr, float)
        it = c.world.items[item]
        bodies = []
        for b in it["bodies"]:
            p = Rr.T @ (np.asarray(c.data.xpos[b], float) - pr)
            R = Rr.T @ np.asarray(c.data.xmat[b], float).reshape(3, 3)
            bodies.append((p, R))
        bars = {}
        for b in self._bars():
            if b.name.startswith(item + ":"):
                cc, a, n = b.pose(c.data)
                bars[b.name] = (Rr.T @ (np.asarray(cc, float) - pr), Rr.T @ np.asarray(a, float), Rr.T @ np.asarray(n, float),
                                bool(self._bar_in_cage(b)))
        return dict(bodies=bodies, bars=bars, half=tuple(float(h) for h in it["half"]))

    # ------------------------------------------------------------------ P5
    @priv("P5")
    def on_face(self, item: str, face: str) -> dict | None:
        """After a placement: the item's sealed area on ``face`` (SB-<M>-S / -P1 / -P2), on any other face, the share
        of its own footprint, held / between the fingers -- None when the item lies on no face and is not held (it
        floats: not read further)."""
        self._count("on_face")
        c = self._cell
        seal = c.world.seal_by_face(c.data)
        held = bool(c.world.in_gripper(c.data, item))
        fingers = any(self._bar_in_cage(b) for b in self._bars() if b.name.startswith(item + ":"))
        mine = {f: a[item] for f, a in seal.items() if a.get(item, 0.0) > 0.0}
        if not mine and not held and not fingers:
            self._refuse("on_face")
            return None
        fp = float(c.world.items[item]["footprint_m2"])
        on = float(mine.get(face, 0.0))
        return dict(face=face, sealed_m2=on, own_fraction=(on / fp if fp > 0 else 0.0), other=dict((f, v) for f, v in mine.items() if f != face),
                    held=held, between_fingers=bool(fingers))
