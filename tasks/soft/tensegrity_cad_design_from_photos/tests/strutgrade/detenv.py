"""Deterministic environment of the strut grader: thread counts, CPU affinity, the OpenBLAS kernel and
the seeds.

Two steps, because the BLAS kernel and thread counts are read when numpy is first imported:

    import strutgrade.detenv as detenv          # this module imports only the standard library at import time
    detenv.early(n_cpus)                        # BEFORE numpy, scipy, trimesh, manifold3d or OCP are imported
    ...
    record = detenv.pin(cfg)                    # after the configuration is read: affinity, a check of the above

early() sets OMP / OPENBLAS / MKL / NUMEXPR / VECLIB thread counts to 1, TBB's to n_cpus (manifold3d's embedded
oneTBB otherwise starts a thread per core) and OPENBLAS_CORETYPE=Haswell unless the caller set
it (PR #83: numpy's OpenBLAS computes wrong on Intel AVX512_BF16 without the pin), and pins the process to the first
n_cpus CPUs of its affinity mask. pin() repeats the affinity from the configuration and records whether numpy was
imported before early() ran (statistic `detenv_numpy_before_pin`; a run where it is 1 is still graded, its kernel
was whatever the caller's environment set).

Seeds: seed_for(cfg, *names) = (sampling.seed + the first 8 bytes of sha256("/".join(names))) mod 2**31. A seed
depends only on the configured base seed and on what it is for (a design key, a solid's id, a stage), never on the
order things are graded in.
"""
import hashlib
import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
CORETYPE = "Haswell"
_STATE = {"early": False, "numpy_before": None, "cpus": []}


def limit_cpus(n):
    """Pin this process to its first n allowed CPUs (OCC's mesher and manifold3d's TBB size their pools to the host).
    Returns the CPU list (empty where affinity is not supported)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))[:max(1, int(n))]
        os.sched_setaffinity(0, cpus)
        return cpus
    except (AttributeError, OSError):
        return []


def early(n_cpus=4):
    """Set the thread and kernel environment before numpy is imported (module docstring)."""
    _STATE["numpy_before"] = "numpy" in sys.modules
    for k in THREAD_VARS:
        os.environ[k] = "1"
    os.environ["TBB_NUM_THREADS"] = str(max(1, int(n_cpus)))
    os.environ.setdefault("OPENBLAS_CORETYPE", CORETYPE)
    os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "0")
    _STATE["cpus"] = limit_cpus(n_cpus)
    _STATE["early"] = True
    return dict(_STATE)


def pin(cfg):
    """Affinity from the configuration (cfg["detenv"]["grader_cpus"], the environment variable ARB_GRADER_CPUS wins)
    and a record of the deterministic environment for the statistics."""
    D = cfg.get("detenv", {})
    n = int(os.environ.get("ARB_GRADER_CPUS", D.get("grader_cpus", 4)))
    if not _STATE["early"]:
        early(n)
    else:
        _STATE["cpus"] = limit_cpus(n)
    rec = {"detenv_cpus": float(len(_STATE["cpus"])),
           "detenv_numpy_before_pin": 1.0 if _STATE["numpy_before"] else 0.0,
           "detenv_openblas_coretype_haswell": 1.0 if os.environ.get("OPENBLAS_CORETYPE") == CORETYPE else 0.0}
    return rec


def seed_for(cfg, *names):
    """A fixed seed for one purpose (module docstring)."""
    base = int(cfg["sampling"]["seed"])
    h = int.from_bytes(hashlib.sha256("/".join(str(n) for n in names).encode()).digest()[:8], "little")
    return int((base + h) % (2 ** 31))


def versions():
    """Library versions that change the numbers (recorded in the statistics' notes)."""
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "trimesh", "manifold3d"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:  # noqa: BLE001
            out[mod] = "missing"
    try:
        from OCP.Standard import Standard_Version  # noqa: F401
        out["ocp"] = "present"
    except Exception:  # noqa: BLE001
        try:
            import OCP  # noqa: F401
            out["ocp"] = "present"
        except Exception:  # noqa: BLE001
            out["ocp"] = "missing"
    return out
