import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
REF = os.path.join(TASK_DIR, "ref")
RUNTIME = os.path.join(TASK_DIR, "runtime")
DUTY = os.path.join(TASK_DIR, "duty")
HIDDEN = os.path.join(TASK_DIR, "hidden")
RULES = os.path.join(TASK_DIR, "rules")
for _p in (RUNTIME, DUTY, HIDDEN, REF, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shift as SH
import quantities as QT

K = SH.K

RULE_FILES = {
    "BAND_RULE.txt": "3f4dce4e24ba02f674f6f94ebeff57e5ec1173834fa2dbfb5efdd181178791a4",
    "band_rule.json": "05a20420ac335172ce86f716d39b18bae38f77f57c34e41da20b4c226d602bbb",
    "band_rule_amendment_1_floors.json": "115526c4fd34ae475c2f56192a6b06e732948a9855c1ca7d3506480e38c3e424",
    "BAND_RULE_AMENDMENT_1_FLOORS.txt": "90b0d9dbdf9e3fe79831003b9dba07dbfd2ab996962c4ff74c7e630d3057ce72",
}
AMENDMENT_FILE = ("BAND_RULE_AMENDMENT_1.txt", "a80a32481265bd575f6c44181ba630299720d4a6ef2bc7ac2b5e18b5b0f5c0b8")

QUANTITY_KEYS = {"e_z": "e_z_mm", "theta": "tilt_deg", "rho_t": "rho_mm", "e_xy": "e_xy_mm", "d_m": "d_m_mm",
                 "P_r": "P_r_N"}
RULE_NAME = {v: k for k, v in QUANTITY_KEYS.items()}
TERM_OF = {"e_z": "H", "theta": "H", "rho_t": "R", "e_xy": "A", "d_m": "P", "P_r": "place"}
ZERO_END_MUTANT = {"e_z_mm": ("M-load", "documents"), "tilt_deg": ("M-load", "documents"),
                   "rho_mm": ("M-fric", 1.0), "d_m_mm": ("M-fric", 1.0), "e_xy_mm": ("M-geom", 1.0),
                   "P_r_N": ("M-shelf", "drawn")}
ZERO_END_MUTANTS = (("M-load", "documents"), ("M-fric", 1.0), ("M-geom", 1.0), ("M-shelf", "drawn"))
THREE_X = None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest_of(obj):
    return sha256_bytes(canonical(obj).encode("ascii"))


def plain(v):
    return SH._plain(v)


class RuleFileMismatch(RuntimeError):
    ""


def check_rule_files(rules_dir=RULES):
    out = {}
    for name, want in list(RULE_FILES.items()) + [AMENDMENT_FILE]:
        got = sha256_file(os.path.join(rules_dir, name))
        if got != want:
            raise RuleFileMismatch(f"{name}: sha256 {got} differs from the pinned {want}")
        out[name] = got
    return out


def load_rule(rules_dir=RULES):
    check_rule_files(rules_dir)
    with open(os.path.join(rules_dir, "band_rule.json"), encoding="utf-8") as f:
        br = json.load(f)
    with open(os.path.join(rules_dir, "band_rule_amendment_1_floors.json"), encoding="utf-8") as f:
        fl = json.load(f)
    return br, fl


def floor_table(rules_dir=RULES):
    br, fl = load_rule(rules_dir)
    out = {}
    for rn, key in QUANTITY_KEYS.items():
        term = TERM_OF[rn]
        c = br["terms"][term]["quantities"][rn]["floor"]["value"]
        if rn in fl["floors"]:
            f = fl["floors"][rn]
            if abs(float(f["committed_floor"]) - float(c)) > 1e-12:
                raise RuleFileMismatch(f"{rn}: the floors file's committed floor {f['committed_floor']} is not the "
                                       f"committed rule's {c}")
            v = max(float(c), float(f["T"]))
            if abs(v - float(f["floor"])) > 1e-12:
                raise RuleFileMismatch(f"{rn}: floor {f['floor']} is not max(committed, T) = {v}")
            out[key] = {"floor": float(f["floor"]), "unit": f["unit"], "committed": float(c), "T": float(f["T"]),
                        "source": "rules/band_rule_amendment_1_floors.json floors." + rn + ".floor (item 1)"}
        else:
            kept = fl["floors_kept"].get("rho" if rn == "rho_t" else rn)
            if kept is not None and abs(float(kept["floor"]) - float(c)) > 1e-12:
                raise RuleFileMismatch(f"{rn}: floors_kept {kept['floor']} is not the committed floor {c}")
            out[key] = {"floor": float(c), "unit": br["terms"][term]["quantities"][rn]["unit"], "committed": float(c),
                        "T": None, "source": f"rules/band_rule.json terms.{term}.quantities.{rn}.floor (section 6)"}
    return out


def weights_of_record(rules_dir=RULES):
    br, _ = load_rule(rules_dir)
    w = br["weights_of_record"]
    return {"Q": dict(w["Q"]), "unit": dict(w["unit"]), "tray_place": dict(w["tray_place"])}


def three_x_ratio(rules_dir=RULES):
    br, _ = load_rule(rules_dir)
    return float(br["three_x_rule"]["ratio"])


def full_end_factor(rules_dir=RULES):
    br, _ = load_rule(rules_dir)
    return float(br["full_end"]["reference_factor"])


def full_end_mismatches(decision, factor=None):
    fac = full_end_factor() if factor is None else float(factor)
    bad = []
    for k, q in sorted(decision["quantities"].items()):
        p = q.get("reference_p90")
        want = float(q["floor"]) if p is None else max(float(q["floor"]), fac * float(p))
        if float(q["F"]) != want:
            bad.append(f"{k}: F {q['F']!r} is not max(floor {q['floor']!r}, {fac!r} x p90 {p!r}) = {want!r}")
    return bad
