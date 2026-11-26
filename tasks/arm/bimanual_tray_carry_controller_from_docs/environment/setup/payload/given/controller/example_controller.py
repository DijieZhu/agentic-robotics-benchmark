"""A minimal example controller: it holds the pose in which the pair is handed over and never declines.

It shows the calls and the frames only. It carries no tray: the first job fails at its capture and the shift ends
with a keep-out event at that tray's removal.

    setup(unit_record, site) -> {"gripper_mode": {"left": "posforce", "right": "posforce"}}
    step(feedback)           -> {"frames": [16 frames], "decline": False}

Module order: left J1..J7, left gripper, right J1..J7, right gripper. setup reads each module's position range PMAX
from its register dump in the unit's record. The arm modules get MIT frames that hold the module angles of the first
feedback, kept inside +-PMAX, with the maker's first gain set of its driver configuration; the two grippers, in
force-position mode, hold their first angles with a small current limit. Standard library only.
"""

KP = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
KD = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)
GRIPPERS = (7, 15)


def register_value(dump, name):
    """The value of register `name` ("PMAX", "VMAX", "TMAX", ...) in a module's register dump as the record prints it
    (the row whose parameter ends in "(name)"; its last column)."""
    tag = "(" + name + ")"
    for line in dump.splitlines():
        cols = line.split()
        if cols and line[:30].rstrip().endswith(tag):
            return float(cols[-1])
    raise KeyError(name)


class Controller:
    def __init__(self):
        self.hold = None
        self.pmax = None

    def setup(self, unit_record, site):
        self.hold = None
        self.pmax = [register_value(m["register_dump"], "PMAX") for m in unit_record["modules"]]
        return {"gripper_mode": {"left": "posforce", "right": "posforce"}}

    def step(self, feedback):
        mods = feedback["modules"]
        if self.hold is None:
            self.hold = [max(-p, min(p, m["q"])) for m, p in zip(mods, self.pmax)]
        frames = []
        for i in range(16):
            if i in GRIPPERS:
                frames.append({"p": self.hold[i], "v_lim": 1.0, "i_lim": 0.05})
            else:
                j = i % 8
                frames.append({"p": self.hold[i], "v": 0.0, "kp": KP[j], "kd": KD[j], "t": 0.0})
        return {"frames": frames, "decline": False}
