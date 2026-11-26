"""openarm_unit: the shared simulated unit of the OpenArm 2.0 pair (the maker's v2 model on its pedestal). PUBLIC.

One module per file. This file only fixes and records the process environment that every scored path runs in,
at import time:

- ``OPENBLAS_CORETYPE=Haswell`` and ``OPENBLAS_L2_SIZE=256`` are set for whatever a library may still call (the
  second is required with the first). OpenBLAS reads them when it is loaded: they take effect only when numpy
  is loaded after this import, or when the process inherited both values already. ``ENVIRONMENT["openblas_in_effect"]``
  says which: it is False when numpy was loaded first and the inherited environment lacked either value.
- ``PYTHONHASHSEED`` must be ``0`` on a scored path; it is read at interpreter start, so it is recorded
  here (``ENVIRONMENT["hashseed_zero"]``), not set.
- flush-to-zero and denormals-are-zero must be off: asserted here on a denormal operand and a denormal result.
A process with either mode on cannot produce the specified bits and is refused at import.

``require_scored_environment()`` raises unless all three hold; a runner of a scored path (a grader, a checker)
calls it before its first step.

Nothing else is imported here: importing the package does not import MuJoCo, and no other module is pulled in with it.
"""
import os
import sys

__version__ = "0.1.0"

# The values a scored process runs with.
REQUIRED_ENV = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256", "PYTHONHASHSEED": "0"}
_BLAS_KEYS = ("OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE")

_NUMPY_LOADED_BEFORE = "numpy" in sys.modules
_INHERITED = {k: os.environ.get(k) for k in _BLAS_KEYS}

# OPENBLAS_CORETYPE=Haswell with OPENBLAS_L2_SIZE=256, for whatever a library may still call.
for _k in _BLAS_KEYS:
    os.environ[_k] = REQUIRED_ENV[_k]


class FloatEnvironmentError(RuntimeError):
    """Flush-to-zero or denormals-are-zero is on in this process (a scored path forbids both)."""


class ScoredEnvironmentError(RuntimeError):
    """The process cannot run a scored path: OpenBLAS was loaded without the pinned variables, or the hash seed is not 0."""


def _check_denormals():
    """Return (ftz_off, daz_off) measured on numpy float64 arrays (the SSE / AVX path every ufunc uses)."""
    import numpy as np

    smallest_normal = np.array([2.2250738585072014e-308], dtype=np.float64)
    halved = smallest_normal / np.float64(2.0)            # a denormal RESULT: 0.0 under flush-to-zero
    denormal = np.array([5e-324], dtype=np.float64)       # a denormal OPERAND: read as 0.0 under denormals-are-zero
    kept = denormal * np.float64(1.0)
    return bool(halved[0] != 0.0), bool(kept[0] != 0.0)


def check_float_environment():
    """Raise FloatEnvironmentError unless flush-to-zero and denormals-are-zero are both off."""
    ftz_off, daz_off = _check_denormals()
    if not (ftz_off and daz_off):
        raise FloatEnvironmentError(
            f"flush-to-zero off: {ftz_off}; denormals-are-zero off: {daz_off}; both must be off on a scored path")
    return {"ftz_off": ftz_off, "daz_off": daz_off}


ENVIRONMENT = {
    "numpy_loaded_before": _NUMPY_LOADED_BEFORE,
    "OPENBLAS_inherited": dict(_INHERITED),
    "OPENBLAS_CORETYPE": os.environ["OPENBLAS_CORETYPE"],
    "OPENBLAS_L2_SIZE": os.environ["OPENBLAS_L2_SIZE"],
    "openblas_in_effect": (not _NUMPY_LOADED_BEFORE)
    or all(_INHERITED[k] == REQUIRED_ENV[k] for k in _BLAS_KEYS),
    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    "hashseed_zero": os.environ.get("PYTHONHASHSEED") == REQUIRED_ENV["PYTHONHASHSEED"],
}
ENVIRONMENT.update(check_float_environment())


def require_scored_environment():
    """Raise ScoredEnvironmentError unless the scored environment holds (OpenBLAS variables in effect, hash seed 0,
    flush-to-zero and denormals-are-zero off, the last re-measured now). Returns ENVIRONMENT."""
    check_float_environment()
    bad = [k for k in ("openblas_in_effect", "hashseed_zero") if not ENVIRONMENT[k]]
    if bad:
        raise ScoredEnvironmentError(
            f"not a scored environment: {bad} (inherited OpenBLAS variables {ENVIRONMENT['OPENBLAS_inherited']}, "
            f"numpy loaded before the package: {ENVIRONMENT['numpy_loaded_before']}, "
            f"PYTHONHASHSEED {ENVIRONMENT['PYTHONHASHSEED']!r})")
    return ENVIRONMENT
