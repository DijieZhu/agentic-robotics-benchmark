# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
import json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DEFAULT_PARAMS = os.path.join(HERE, 'PARAMETERS.json')
DEFAULT_SENSOR_PARAMS = os.path.join(HERE, 'SENSOR_PARAMETERS.json')
DEFAULT_SESSION_INPUTS = os.path.join(HERE, 'SESSION_INPUTS.json')
_cfg = {'spec': None, 'sessions_dir': None, 'key': None, 'params': DEFAULT_PARAMS, 'sensor_params': DEFAULT_SENSOR_PARAMS, 'scenarios_dir': None, 'session_inputs': DEFAULT_SESSION_INPUTS}
_spec = None
_params = None
SIGNED = ['+x', '-x', '+y', '-y', '+z', '-z']

def configure(spec=None, sessions_dir=None, key=None, params=None, sensor_params=None, scenarios_dir=None, session_inputs=None):
    global _spec, _params
    for k, v in (('spec', spec), ('sessions_dir', sessions_dir), ('key', key), ('params', params), ('sensor_params', sensor_params), ('scenarios_dir', scenarios_dir), ('session_inputs', session_inputs)):
        if v is not None:
            _cfg[k] = os.path.abspath(v) if v != '' else ''
    _spec = None
    _params = None

def optional_path(name):
    return _cfg.get(name)

def path(name):
    p = _cfg[name]
    if p is None:
        raise RuntimeError('world.config.configure(%s=...) has not been called' % name)
    return p

def load_spec():
    global _spec
    if _spec is None:
        with open(path('spec')) as f:
            _spec = json.load(f)['values']
    return _spec
READ_KEYS = set()

def V(key):
    READ_KEYS.add(key)
    v = load_spec()[key]['value']
    if v is None:
        raise KeyError('table entry %s is pending without a value' % key)
    return v

def pending_read():
    sp = load_spec()
    return {k: sp[k]['status'] for k in sorted(READ_KEYS) if str(sp[k].get('status', '')).startswith('pending')}

def params():
    global _params
    if _params is None:
        with open(path('params')) as f:
            base = json.load(f)
        sp = _cfg.get('sensor_params')
        if sp and os.path.exists(sp):
            with open(sp) as f:
                over = json.load(f)['parameters']
            filled = []
            for k in sorted(over):
                e = base['parameters'].get(k)
                if e is None or e.get('value') is None:
                    m = dict(e or {})
                    m.update(over[k])
                    base['parameters'][k] = m
                    filled.append(k)
            base['sensor_parameters_overlay'] = {'file': os.path.basename(sp), 'filled': filled}
        _params = base
    return _params
inputs = params

def param(name):
    e = params()['parameters'][name]
    if e.get('status', '').startswith('pending') and e.get('value') is None:
        raise KeyError('world parameter %s is %s' % (name, e['status']))
    return e['value']

def key_bytes():
    with open(path('key'), 'rb') as f:
        return f.read()

def no_numbers_from_file():
    return {}

def torque_gains(level):
    if level == 1:
        assert V('world.torque_gain.level1') == 'equal to level 2'
        level = 2
    g = V('world.torque_gain.level%d' % level)
    return [float(g[s]) for s in SIGNED]

def force_gains(level, inp=None):
    inp = inp or params()
    fg = inp['parameters']['force_gain']['value']
    if level in (1, 2):
        return [float(fg['level2_and_level1'][s]) for s in SIGNED]
    return [float(fg['level3'][s]) for s in SIGNED]
