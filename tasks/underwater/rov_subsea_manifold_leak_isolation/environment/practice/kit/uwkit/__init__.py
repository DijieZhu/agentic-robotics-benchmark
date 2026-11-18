"""uwkit - shared underwater simulation kit for the four underwater benchmark tasks.

    import uwkit
    vb  = uwkit.build_vehicle()                 # reference mission vehicle from the digest (fresh-water ballast)
    sim = uwkit.UnderwaterSim(vb, seed=0)       # MuJoCo + hydro layer + thrusters + battery + current
    ap  = uwkit.Autopilot(vb.vehicle)           # on-board firmware (gains tuned on the nominal plant)

Modules: frames, params, refdata (digest files), hydrostatics (pure numpy), sim, dynamics/ (hydro, thrusters,
battery, current), control/ (allocation, autopilot), models/ (vehicle builder + data files), sensors/ and nav/
(imported on demand), viz/ (render-only twin), hidden/ (grader-side coefficient data; never shipped), packaging
(maintainer tool: which files go where, byte copies for the task folders).

Subsets: a task folder may hold only part of the package (uwkit/packaging.py, purposes 'verifier', 'practice-service',
'agent-sandbox').  `import uwkit` then exposes what is present; `uwkit.SUBSET` names the missing top-level modules.

Dependencies: numpy + mujoco (3.14.0); gsw only for hydrostatics.rho_teos10.  Python >= 3.10.
"""
import importlib.util as _ilu

__version__ = "0.3.0"
SUBSET = []                                  # top-level modules absent from this copy (subset packages)


def _have(*names) -> bool:
    return all(_ilu.find_spec(f"{__name__}.{n}") is not None for n in names)


from . import frames, hydrostatics  # noqa: E402,F401

if _have("refdata"):
    from . import refdata  # noqa: F401
else:
    SUBSET.append("refdata")

if _have("params", "models", "sim", "control", "dynamics"):
    from .params import (G, HydroCoeffs, VehicleSpec, Water, load_hydro, load_vehicle_spec,  # noqa: F401
                         sample_hydro, blend_hydro, default_hydro, hydro_sets)
    from .models.vehicle import build_vehicle, add_free_object, reference_vehicle_spec  # noqa: F401
    from .sim import UnderwaterSim, VehicleState  # noqa: F401
    from .control.autopilot import Autopilot, AutopilotGains, NavEstimate, Setpoint, run_closed_loop  # noqa: F401
    from .control.allocation import Allocator  # noqa: F401
else:
    SUBSET.append("simulator")
    G = hydrostatics.G_STD
