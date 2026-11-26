import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
CONTRACT = os.path.join(TASK_DIR, "contract")

SOURCES = {"frames.py": "frames.py", "toy_chain.py": "toy_chain.py", "check_controller.py": "check_controller.py",
           "example_controller.py": "example_controller.py"}

PUBLISHED_DOC = '''"""The frame codec of the controller interface (standard library only).

  - the fields of the two frame kinds a controller sends, their ranges and bit widths, and the quantiser the modules
    apply (clip to the range, then truncation to the field's code): mit_codes, mit_effective, posforce_codes,
    posforce_effective give what a module receives for a frame;
  - the eight data bytes of each frame kind (mit_bytes, posforce_bytes) and the values of a module's state reply
    (state_fields);
  - the check of a step's reply (parse_step_reply): a reply that is an object but not a valid reply counts as a late
    cycle (ReplyMalformed); a reply that is not an object ends the controller (ReplyNotObject);
  - the feedback object of one cycle (feedback, module_feedback).

The ranges of a module are its own registers PMAX, VMAX and TMAX, as the unit's delivery record prints them.
A number field of a frame is a JSON number that is finite and not a boolean; a NaN, an infinity, a boolean, a string,
an integer beyond the float range, or a force-position p beyond the float32 range makes the reply malformed.
"""'''


def _strip_docstring(text):
    i = text.index('"""')
    j = text.index('"""', i + 3)
    return text[j + 3:]


def published_files():
    out = {}
    for name, src in SOURCES.items():
        with open(os.path.join(HERE, src), encoding="utf-8") as f:
            text = f.read()
        if name == "frames.py":
            text = PUBLISHED_DOC + _strip_docstring(text)
        out[name] = text
    return out


def runtime_constants():
    if CONTRACT not in sys.path:
        sys.path.insert(0, CONTRACT)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import constants as K
    import frames as FR
    import wire as WI
    cls = {m["class"] for m in K.MODULES}
    return {
        "cycle_s": K.CYCLE_S, "feedback_age_cycles": K.FEEDBACK_AGE_CYCLES,
        "modules": [{"index": m["index"], "label": m["label"], "product": m["product"], "class": m["class"]}
                    for m in K.MODULES],
        "mit_keys": list(FR.MIT_KEYS), "posforce_keys": list(FR.POSFORCE_KEYS), "mit_bits": dict(FR.MIT_BITS),
        "kp_range": list(FR.KP_RANGE), "kd_range": list(FR.KD_RANGE), "pmax": K.PMAX,
        "vmax": {c: K.VMAX[c] for c in sorted(cls)}, "tmax": {c: K.TMAX[c] for c in sorted(cls)},
        "steps": {"p": FR.step_of(-K.PMAX, K.PMAX, 16),
                  "v": {c: FR.step_of(-K.VMAX[c], K.VMAX[c], 12) for c in sorted(cls)},
                  "kp": FR.step_of(*FR.KP_RANGE, 12), "kd": FR.step_of(*FR.KD_RANGE, 12),
                  "t": {c: FR.step_of(-K.TMAX[c], K.TMAX[c], 12) for c in sorted(cls)},
                  "v_lim": 1.0 / FR.V_LIM_SCALE, "i_lim": 1.0 / FR.I_LIM_SCALE},
        "v_lim_range": list(FR.V_LIM_RANGE), "i_lim_range": list(FR.I_LIM_RANGE),
        "reply_bits": dict(FR.REPLY_BITS), "feedback_keys": list(FR.FEEDBACK_KEYS),
        "site_keys": list(FR.SITE_KEYS), "module_feedback_keys": list(FR.MODULE_FEEDBACK_KEYS),
        "gripper_modes": list(K.GRIPPER_MODES),
        "setup_budget": dict(K.SETUP_BUDGET), "step_budget_cpu_s": K.STEP_BUDGET_CPU_S,
        "step_fatal": dict(K.STEP_FATAL), "on_time_share_min": K.ON_TIME_SHARE_MIN,
        "late_time_stop_share": K.LATE_TIME_STOP_SHARE,
        "reply_line_max_bytes": WI.MAX_LINE, "reply_line_max_mib": WI.MAX_LINE / float(1 << 20),
        "import_wall_s": WI.IMPORT_S, "child_threads": WI.CHILD_THREADS, "mujoco_gl": WI.MUJOCO_GL,
        "python_flags": list(WI.PY_FLAGS),
    }


def write_published(folder):
    os.makedirs(folder, exist_ok=True)
    for name, text in published_files().items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(text)


def self_test(folder, python=None, controller=None):
    argv = [python or sys.executable, "-I", "-B", "check_controller.py"]
    if controller:
        argv.append(controller)
    p = subprocess.run(argv, cwd=folder, capture_output=True, text=True, timeout=600,
                       env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode, p.stdout + p.stderr
