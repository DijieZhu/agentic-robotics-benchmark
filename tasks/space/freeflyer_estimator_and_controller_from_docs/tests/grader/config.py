"""The grader's configuration (tests/grader_config.json, written by the task's generator from the value of the public
entries of the task's table of numbers and from the grader's frame: where its files are and its own operating limits).

No knob has a default in code: every key of VALUE_KEYS, LAYOUT_KEYS and RUNTIME_KEYS must be in the document, and no
other key may be; a missing, extra or malformed key is a ConfigError, which the verification turns into a task error.
A value that the table still marks pending may be null in the document; the grader refuses to score with a null value it
needs (score.bands above all) and the verification is then void.
"""
import math
import re

FORMAT = "estimator-controller-grader-config/1"

# the public entries the grader reads (values copied as the table holds them)
VALUE_KEYS = (
    "score.weights", "score.bands", "score.end_window_s", "score.gates",
    "score.leg", "score.slew", "score.hold", "score.view", "score.refusals", "score.session", "score.L", "score.completed",
    "jobs.kinds", "jobs.capture_s", "jobs.camera_mount_m", "jobs.hold_s",
    "interface.submission_cap", "interface.libraries", "interface.on_time_share_gate", "interface.step_period_ms",
    "interface.verify_timeout_s", "interface.sandbox",
    "safety.overspeed_m_s", "monitor.recovery_s",
    "sessions.names", "sessions.acceptance_count",
)
# where the grader finds its files in the stage folder (the grader's frame); token_file is the access token's name
# inside the controller folder, token_sha256 the stage's copy of its sha256 (grader/access.py)
LAYOUT_KEYS = ("hidden_set", "world_root", "world_table", "sessions_dir", "seed_key", "runner_dir", "submission_dir",
               "controller_file", "token_file", "token_sha256")
# the verification's own operating limits (the grader's frame); none of them decides a score
RUNTIME_KEYS = ("reserve_s", "late_gate_max_gap_s", "snapshot_root", "work_prefix", "megabyte_bytes")

# the formula's terms (score.leg, score.slew, score.hold, score.view): the measures of each kind
MEASURES = {"leg": ("end_pos", "end_att", "overshoot"), "slew": ("end_att", "end_pos", "overshoot"),
            "hold": ("pos_rms", "att_rms", "speed_rms", "rate_rms"), "view": ("pointing", "centre", "rate_rms")}
KIND_WEIGHT = {"leg": "legs", "slew": "slews", "hold": "holds", "view": "views"}
REFUSAL_WEIGHT = "refusals"
MODES = ("nominal", "quiet", "precision")
# the cells the sessions fly (sessions.blocks): legs, slews and holds in every mode, views in the nominal mode
CELL_MODES = {"leg": MODES, "slew": MODES, "hold": MODES, "view": ("nominal",)}
# the measure whose completion limit is its zero end in every mode, dropped or not (score.completed)
CENTRE = ("view", "centre")


class ConfigError(Exception):
    pass


def all_cells():
    return sorted("%s/%s/%s" % (k, m, x) for k in MEASURES for m in CELL_MODES[k] for x in MEASURES[k])


def _finite(x, what, positive=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ConfigError("%s is not a finite number: %r" % (what, x))
    if positive and not float(x) > 0.0:
        raise ConfigError("%s is not above 0: %r" % (what, x))
    return float(x)


def check_bands(b):
    """score.bands in its form: {"cells": {"kind/mode/measure": {"full_end": FE, "zero_end": Z}}} for exactly the 33
    cells, every end a finite number above 0 in the measure's unit (m, rad, m/s, rad/s)."""
    if not isinstance(b, dict) or set(b) != {"cells"} or not isinstance(b["cells"], dict):
        raise ConfigError("score.bands is not of the form {cells: {kind/mode/measure: {full_end, zero_end}}}")
    cells = b["cells"]
    want = set(all_cells())
    if set(cells) != want:
        raise ConfigError("score.bands holds the cells %s and lacks %s" % (sorted(set(cells) - want)[:5],
                                                                          sorted(want - set(cells))[:5]))
    out = {}
    for k in sorted(cells):
        c = cells[k]
        if not isinstance(c, dict) or set(c) != {"full_end", "zero_end"}:
            raise ConfigError("score.bands cell %s is not {full_end, zero_end}" % k)
        out[k] = (_finite(c["full_end"], k + " full_end", True), _finite(c["zero_end"], k + " zero_end", True))
    return out


class Config(object):
    def __init__(self, doc):
        self.doc = doc
        self.values = doc["values"]
        self.layout = doc["layout"]
        self.runtime = doc["runtime"]
        self.pending = list(doc["pending_in_table"])

    def v(self, key):
        return self.values[key]

    def need(self, key):
        """a value the grader cannot score without (a null pending value voids the verification)."""
        x = self.values[key]
        if x is None:
            raise ConfigError("the configuration's %s is null (pending in the table): no score can be computed" % key)
        return x

    # the numbers, each by its key
    def weights(self):
        w = self.need("score.weights")
        names = set(KIND_WEIGHT.values()) | {REFUSAL_WEIGHT}
        if not isinstance(w, dict) or set(w) != names:
            raise ConfigError("score.weights does not name exactly %s" % sorted(names))
        return {k: _finite(x, "score.weights " + k) for k, x in w.items()}

    def bands(self):
        """score.bands, checked; the zero end of the view's optical-centre displacement must be the cap that
        score.completed states (its completion limit in every mode): a band document with another zero end there is
        a defect of the hidden side (a task error), never a score."""
        out = check_bands(self.need("score.bands"))
        key = "%s/nominal/%s" % CENTRE
        cap = self.centre_cap_m()
        if out[key][1] != cap:
            raise ConfigError("score.bands gives the zero end %r m for %s, not the cap %r m that score.completed states "
                              "for the optical-centre displacement" % (out[key][1], key, cap))
        return out

    def centre_cap_m(self):
        """the cap of the view's optical-centre displacement in metres, read from the text of score.completed (the
        completion limit 'always its zero end (<number> m)'); a text without that number stops the grader."""
        txt = self.need("score.completed")
        m = re.search(r"whose completion limit is always its zero end \(([0-9]+(?:\.[0-9]+)?) m\)", txt)
        if not m:
            raise ConfigError("score.completed does not state the optical-centre displacement's zero end in metres")
        return _finite(float(m.group(1)), "the optical-centre cap of score.completed", True)

    def end_window(self):
        ew = self.need("score.end_window_s")
        if not isinstance(ew, list) or len(ew) != 2:
            raise ConfigError("score.end_window_s is not a pair")
        a, b = _finite(ew[0], "score.end_window_s[0]"), _finite(ew[1], "score.end_window_s[1]")
        if not 0.0 <= a < b:
            raise ConfigError("score.end_window_s is not an interval after done")
        return a, b

    def capture_s(self):
        return _finite(self.need("jobs.capture_s"), "jobs.capture_s", True)

    def hold_s(self):
        return _finite(self.need("jobs.hold_s"), "jobs.hold_s", True)

    def mount(self):
        m = self.need("jobs.camera_mount_m")
        if not isinstance(m, list) or len(m) != 3:
            raise ConfigError("jobs.camera_mount_m is not a vector of 3")
        return [_finite(x, "jobs.camera_mount_m") for x in m]

    def kinds(self):
        k = self.need("jobs.kinds")
        if sorted(k) != sorted(MEASURES):
            raise ConfigError("jobs.kinds %r are not the formula's kinds %s" % (k, sorted(MEASURES)))
        return list(k)

    def step_s(self):
        return _finite(self.need("interface.step_period_ms"), "interface.step_period_ms", True) / 1000.0

    def overspeed(self):
        return _finite(self.need("safety.overspeed_m_s"), "safety.overspeed_m_s", True)

    def recovery_s(self):
        r = self.need("monitor.recovery_s")
        if not isinstance(r, dict) or set(r) != set(MODES):
            raise ConfigError("monitor.recovery_s is not given per mode")
        return {m: _finite(r[m], "monitor.recovery_s " + m, True) for m in MODES}

    def on_time_share_gate(self):
        return _finite(self.need("interface.on_time_share_gate"), "interface.on_time_share_gate")

    def submission_cap(self):
        c = self.need("interface.submission_cap")
        if not isinstance(c, dict) or set(c) != {"files", "megabytes"}:
            raise ConfigError("interface.submission_cap is not {files, megabytes}")
        return int(_finite(c["files"], "files", True)), _finite(c["megabytes"], "megabytes", True)

    def libraries(self):
        lib = self.need("interface.libraries")
        if not isinstance(lib, list) or not all(isinstance(x, str) for x in lib):
            raise ConfigError("interface.libraries is not a list of names")
        return list(lib)

    def sessions(self):
        names = self.need("sessions.names")
        n = int(_finite(self.need("sessions.acceptance_count"), "sessions.acceptance_count", True))
        if not isinstance(names, list) or len(names) != n or len(set(names)) != n:
            raise ConfigError("sessions.names is not a list of sessions.acceptance_count distinct names")
        return list(names)

    def gates(self):
        g = self.need("score.gates")
        if not isinstance(g, list) or len(g) != 6:
            raise ConfigError("score.gates is not the list of six gates the grader applies")
        return list(g)

    def overshoot_terms(self):
        """(a, b) of the leg's and the slew's factor (a + b O), read from the formulas' text (score.leg, score.slew);
        the grader stops if the table's formulas are not the ones its code implements."""
        out = []
        for key, head in (("score.leg", "completed x L(end position error) x L(end attitude error) x ("),
                          ("score.slew", "completed x L(end attitude error) x L(end position error) x (")):
            txt = self.need(key)
            m = re.fullmatch(re.escape(head) + r"([0-9.]+) \+ ([0-9.]+) O\)", txt)
            if not m:
                raise ConfigError("%s is not the formula the grader implements: %r" % (key, txt))
            out.append((float(m.group(1)), float(m.group(2))))
        if out[0] != out[1]:
            raise ConfigError("score.leg and score.slew name different overshoot factors")
        return out[0]

    def check_formulas(self):
        """the texts of the rules the code implements (score.hold, score.view, score.refusals, score.session, score.L,
        score.completed): a changed rule stops the grader instead of being scored by the old code."""
        want = {"score.hold": ("completed(job before)", "L(position RMS about the hold's target pose)",
                               "L(speed RMS)", "L(rate RMS)", "the goal of the job before"),
                "score.view": ("captured x L(pointing error) x L(optical-centre displacement) x L(rate RMS in the capture)",),
                "score.refusals": ("J x (share of accepted feasible jobs completed)",
                                   "J = share of requests refused + share of feasible jobs accepted - 1"),
                "score.session": ("weighted mean over the kinds", "a kind without jobs drops out, weights renormalised",
                                  "stay in the mean with 0", "a session whose setup overran scores 0"),
                "score.L": ("1 at the full end, 0 at the zero end, linear between",),
                "score.completed": ("each of its own measures is inside its completion limit",
                                    "twice its full end",
                                    "except the optical-centre displacement of a view, whose completion limit is always "
                                    "its zero end", "a hold is tested in the same way")}
        for key, parts in want.items():
            txt = self.need(key)
            for p in parts:
                if p not in txt:
                    raise ConfigError("%s no longer says %r: the grader's code implements the earlier rule" % (key, p))
        return True

    def verify_timeout_s(self):
        return _finite(self.need("interface.verify_timeout_s"), "interface.verify_timeout_s", True)

    def address_space_bytes(self):
        sb = self.need("interface.sandbox")
        return int(_finite(sb["address_space_per_process_GiB"], "address space", True) * 1024 ** 3)


def from_document(doc):
    """the Config of a configuration document; ConfigError on any departure from the form."""
    if not isinstance(doc, dict):
        raise ConfigError("the configuration is not an object")
    want = {"format", "values", "pending_in_table", "layout", "runtime", "generated_from"}
    if set(doc) != want:
        raise ConfigError("the configuration's blocks are %s, not %s" % (sorted(doc), sorted(want)))
    if doc["format"] != FORMAT:
        raise ConfigError("the configuration's format is %r, not %r" % (doc["format"], FORMAT))
    for block, keys in (("values", VALUE_KEYS), ("layout", LAYOUT_KEYS), ("runtime", RUNTIME_KEYS)):
        got = doc[block]
        if not isinstance(got, dict):
            raise ConfigError("the configuration's %s is not an object" % block)
        missing = sorted(set(keys) - set(got))
        extra = sorted(set(got) - set(keys))
        if missing or extra:
            raise ConfigError("the configuration's %s lacks %s and holds %s (no knob has a default; no other key may be "
                              "there)" % (block, missing, extra))
    if not isinstance(doc["pending_in_table"], list) or not set(doc["pending_in_table"]) <= set(VALUE_KEYS):
        raise ConfigError("pending_in_table is not a list of the grader's keys")
    for k in VALUE_KEYS:
        if doc["values"][k] is None and k not in doc["pending_in_table"]:
            raise ConfigError("%s is null but not pending in the table" % k)
    for k in LAYOUT_KEYS:
        p = doc["layout"][k]
        if not isinstance(p, str) or not p or p.startswith("/") or ".." in p.split("/"):
            raise ConfigError("layout %s is not a path inside the stage folder: %r" % (k, p))
    for k in ("reserve_s", "late_gate_max_gap_s", "megabyte_bytes"):
        _finite(doc["runtime"][k], "runtime " + k, True)
    for k in ("snapshot_root", "work_prefix"):
        if not isinstance(doc["runtime"][k], str) or not doc["runtime"][k]:
            raise ConfigError("runtime %s is not a name" % k)
    cfg = Config(doc)
    cfg.weights()
    cfg.kinds()
    cfg.end_window()
    cfg.mount()
    cfg.sessions()
    cfg.gates()
    cfg.overshoot_terms()
    cfg.check_formulas()
    cfg.centre_cap_m()
    if cfg.values["score.bands"] is not None:
        cfg.bands()
    return cfg
