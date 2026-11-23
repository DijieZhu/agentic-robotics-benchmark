"""parcelcell: the label-facing station of a parcel sorting line (MuJoCo).

A fixed dual-arm platform (two 7-joint arms, a parallel gripper each) stands
at a straight conveyor; parcels of random size, kind, weight and label face
arrive at random times. The package holds the simulator and the task's cell:

    from parcelcell.server import ParcelCellServer
    srv = ParcelCellServer("grader_config.json")
    srv.reset("shift.json")                       # a shift package
    srv.handle("step", {"n": 50})                 # what the wire does
    record = srv.finalize()

Modules: ``robot`` (platform + grippers, joint order), ``station`` (belt and
its frame), ``stream`` (parcels, shifts), ``cell`` (the generic simulator:
run-time-written parcel slots), ``world`` (one shift compiled as one model),
``visuals`` (what parcels look like), ``config`` (every number of the cell),
``shiftgen`` (shift generator and packages), ``cameras`` + ``look`` (the
camera system and its renderers), ``toolpose`` (the tool-pose command
level), ``safety`` (the safety watch), ``episode`` (the shift's record and score), ``server``
(the cell behind the wire), ``client`` (the program's side of the wire),
``sensors`` / ``truth`` (simulator-level public and privileged state).

Imports are lazy, so ``parcelcell.client`` loads without MuJoCo.
"""

CELL_VERSION = "parcelcell-3.0"
__version__ = "1.1.0"

_LAZY = {
    "ParcelCell": ("cell", "ParcelCell"), "CellConfig": ("cell", "CellConfig"),
    "CONTROL_DT": ("cell", "CONTROL_DT"), "PHYSICS_DT": ("cell", "PHYSICS_DT"),
    "ARM_JOINTS": ("robot", "ARM_JOINTS"), "JOINT_NAMES": ("robot", "JOINT_NAMES"),
    "home_qpos": ("robot", "home_qpos"),
    "DEFAULT_CAMERAS": ("sensors", "DEFAULT_CAMERAS"), "CameraSpec": ("sensors", "CameraSpec"),
    "StationConfig": ("station", "StationConfig"),
    "Arrival": ("stream", "Arrival"), "Parcel": ("stream", "Parcel"), "Shift": ("stream", "Shift"),
    "StreamConfig": ("stream", "StreamConfig"), "sample_parcel": ("stream", "sample_parcel"),
    "sample_shift": ("stream", "sample_shift"),
    "ShiftWorld": ("world", "ShiftWorld"),
}

__all__ = sorted(_LAZY) + ["CELL_VERSION"]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module("." + mod, __name__), attr)
    raise AttributeError("module 'parcelcell' has no attribute %r" % name)
