import os
import sys

FORCED_ENV = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_L2_SIZE": "256",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MPLBACKEND": "Agg",
    "SOURCE_DATE_EPOCH": "0",
}

ENV_SET_BEFORE_NUMPY = "numpy" not in sys.modules
_INHERITED_OK = all(os.environ.get(k) == v for k, v in FORCED_ENV.items())
for _k, _v in FORCED_ENV.items():
    if _k != "PYTHONHASHSEED":
        os.environ[_k] = _v
sys.dont_write_bytecode = True

DOCGEN = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(DOCGEN)
ARM = os.path.dirname(os.path.dirname(TASK_DIR))
UNIT_DIR = os.path.join(ARM, "shared", "openarm_unit")
UNIT_PKG = os.path.join(UNIT_DIR, "openarm_unit")
CONTRACT = os.path.join(TASK_DIR, "contract")
REF = os.path.join(TASK_DIR, "ref")
DUTY = os.path.join(TASK_DIR, "duty")
HIDDEN = os.path.join(TASK_DIR, "hidden")
RUNTIME = os.path.join(TASK_DIR, "runtime")
RULES = os.path.join(TASK_DIR, "rules")
PAYLOAD = os.path.join(TASK_DIR, "payload")
GIVEN = os.path.join(PAYLOAD, "given")
MANIFEST = os.path.join(DOCGEN, "manifest.json")
REGISTRY_DEV = os.path.join(DOCGEN, "registry")
PROFILE_PATH = os.path.join(CONTRACT, "profile_stage05.json")
DATA = os.path.join(DOCGEN, "data")
MAKER = os.environ.get("OPENARM05_MAKER_DIR", "enactic")

for _p in (UNIT_DIR, DOCGEN):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def environment_ok():
    return _INHERITED_OK and ENV_SET_BEFORE_NUMPY


def reexec_if_needed():
    if environment_ok():
        return
    env = dict(os.environ)
    env.update(FORCED_ENV)
    os.execve(sys.executable, [sys.executable, "-B"] + sys.argv, env)


def load_module(name, path):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def contract():
    return load_module("t05_contract_constants", os.path.join(CONTRACT, "constants.py"))


def shift():
    contract()
    if REF not in sys.path:
        sys.path.append(REF)
    import shift as SH
    return SH


def quantities():
    shift()
    import quantities as Q
    return Q
