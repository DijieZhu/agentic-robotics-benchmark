"""The agent's documents, generated from the cell (a maintainer's tool; the grader never runs it).

    python3 -B tests/review/docs/make_docs.py --write     # render and write every generated file
    python3 -B tests/review/docs/make_docs.py --check     # exit 1 if a shipped file differs from its render

Generated files (paths relative to the task folder):

    instruction.md                                          templates/instruction.md
    environment/setup/payload/given/cell/cell_manual.md     templates/cell_manual.md
    environment/setup/payload/given/cell/cell_api.md        templates/cell_api.md
    environment/setup/payload/given/cell/cell_config.json   public_config() below
    environment/setup/payload/given/robot/, given/station/  make_models.py

A template holds text and ``{{key}}`` placeholders; every value comes from ``facts.collect()``
(the cell's configuration, a built cell, the generator, ``measured.json``). A template may not
hold a number of its own: ``template_literals()`` lists every digit outside a placeholder and
outside a fenced code block, and each must match one of ``LITERALS`` (structural numbers with a
reason). ``test_docs.py`` holds the binding: shipped = rendered, no unbound literal, the public
config equals the cell's, the given models equal their build.

Needs the cell package (``tests/parcelcell``) and MuJoCo; run it with the task's Python.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
TESTS = os.path.dirname(REVIEW)
TASK = os.path.dirname(TESTS)
sys.path.insert(0, HERE)
sys.path.insert(0, TESTS)
sys.dont_write_bytecode = True

import facts as FACTS  # noqa: E402

TEMPLATES = os.path.join(HERE, "templates")
GIVEN_CELL = os.path.join(TASK, "environment", "setup", "payload", "given", "cell")
OUTPUTS = {
    "instruction.md": os.path.join(TASK, "instruction.md"),
    "cell_manual.md": os.path.join(GIVEN_CELL, "cell_manual.md"),
    "cell_api.md": os.path.join(GIVEN_CELL, "cell_api.md"),
}
CONFIG_OUT = os.path.join(GIVEN_CELL, "cell_config.json")

PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")

# Structural literals a template may hold (regexes on the template text with every placeholder
# replaced by the sign §), each with its reason. Anything else with a digit is an unbound number.
LITERALS = [
    (r"^#+ \d+\. ", "section heading number"),
    (r"^\s*\d+\. ", "ordered list item"),
    (r"section \d+", "a reference to a section of the manual"),
    (r"sections? \d+ and \d+", "a reference to two sections"),
    (r"2F-85", "the gripper's product name"),
    (r"\b(8|16)-bit\b", "image format"),
    (r"\buint8\b|\bfloat32\b", "array type"),
    (r"\b4x4\b", "matrix shape"),
    (r"\[2\]", "a two-element vector (left, right)"),
    (r"\b0 \(open\)|\b1 \(closed\)|\(0\) \.\. 1|= 0 \(open\) \.\. 1 \(closed\)", "the gripper command's ends"),
    (r"from 0 \(open\) to 1 \(closed\)|0 \(open\) to 1 \(closed\)", "the gripper command's ends"),
    (r"\[0, 1\]", "the gripper command's range"),
    (r"0 = no reading", "the depth image's no-reading value"),
    (r"\(0\.001\)", "depth_unit_m, a fixed wire constant"),
    (r"`depth_invalid` \(0\)", "a fixed wire constant"),
    (r"row 0 at the top", "pixel convention"),
    (r"\(H/2\) / tan\(fovy/2\)|W/2 - 0\.5|H/2 - 0\.5", "the intrinsics formula"),
    (r"\(\(u - cx\) / fx, \(v - cy\) / fy, 1\)", "the ray formula"),
    (r"\bz = 0\b|\bt = 0\b", "the floor plane / the shift's start"),
    (r"stroke of 0 to\b", "the stroke's closed end"),
    (r"`cv2`", "a module name"),
    (r"0 to § m \(typically", "the random extra's lower end"),
    (r"extra of 0 to § m \(exponential", "the random extra's lower end"),
    (r"at least 1\b", "the ticks argument's lower end"),
    (r"\b1 to §", "the first development shift"),
    (r"exactly § or 2 numbers", "the grip vector's length"),
    (r"\bmax\(0, credit\)", "the score's floor"),
    (r"a W or an X parcel costs 1 and a U parcel 2|a W parcel costs 1 and a U parcel 2|costs 1 more|costs 1\b|"
     r"earns 1\b|scores 0\b", "credit units of the score"),
    (r"every shift scores 0", "the import check's consequence"),
    (r"BSD-2-Clause", "a licence name"),
    (r"OpenCV axes|OpenCV", "a convention name"),
    (r"`os\.exec\*`|`os\.spawn\*`", "call names"),
    (r"python3", "the interpreter's name"),
    (r"gripper_2f85", "a folder name"),
    (r"_b64\b", "a reply field's name (base64)"),
    (r"H x W x 3", "an image array's shape"),
    (r"`ticks: 0`", "the reply after the shift's end"),
    (r"\bn=1\b", "the call's default argument"),
    (r"command 1\b", "the gripper's closed command"),
    (r"net 0 for a W parcel, -1 for a U parcel", "credit units of the score (a parcel read and lost afterwards)"),
]


def render_text(template: str, facts: dict) -> str:
    def sub(mo):
        k = mo.group(1)
        if k not in facts:
            raise KeyError("template placeholder {{%s}} has no fact" % k)
        return str(facts[k])
    return PLACEHOLDER.sub(sub, template)


LIST_ITEM = re.compile(r"^(\s*)([*-]|\d+\.)\s+")


def reflow(md: str, width: int = 120) -> str:
    """Re-wrap the paragraphs and list items of a rendered document to ``width`` columns (values substituted into a
    template change the line lengths). Headings, tables, fenced code and blank lines are kept as they are."""
    import textwrap
    lines, out, i, code = md.split("\n"), [], 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            code = not code
            out.append(ln)
            i += 1
            continue
        if code or not ln.strip() or ln.lstrip().startswith(("|", "#")):
            out.append(ln)
            i += 1
            continue
        mo = LIST_ITEM.match(ln)
        if mo:
            prefix, indent, own = mo.group(0), " " * len(mo.group(0)), len(mo.group(1))
            words = [ln[len(prefix):].strip()]
        else:
            lead = ln[:len(ln) - len(ln.lstrip())]
            prefix, indent, own = lead, lead, None
            words = [ln.strip()]
        i += 1
        while i < len(lines):
            nx = lines[i]
            if (not nx.strip() or nx.strip().startswith("```") or nx.lstrip().startswith(("|", "#"))
                    or LIST_ITEM.match(nx)):
                break
            if own is not None and len(nx) - len(nx.lstrip()) <= own:
                break
            words.append(nx.strip())
            i += 1
        out += textwrap.wrap(" ".join(words), width=width, initial_indent=prefix, subsequent_indent=indent,
                             break_long_words=False, break_on_hyphens=False) or [prefix.rstrip()]
    return "\n".join(out)


def _strip_code(text: str) -> str:
    """The text with fenced code blocks blanked (their lines kept, emptied)."""
    out, inside = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return "\n".join(out)


def template_literals(template: str):
    """[(line number, text)] of every digit outside placeholders and code blocks that no
    ``LITERALS`` entry explains."""
    txt = _strip_code(PLACEHOLDER.sub("§", template))
    bad = []
    for no, line in enumerate(txt.split("\n"), 1):
        masked = line
        for rx, _why in LITERALS:
            masked = re.sub(rx, lambda mo: "#" * len(mo.group(0)), masked)
        if re.search(r"\d", masked):
            bad.append((no, line.strip()))
    return bad


FACE_WORDS = {0: "top", 1: "bottom", 2: "+x side", 3: "-x side", 4: "+y side", 5: "-y side"}


def public_config(ctx: dict) -> dict:
    """The station's configuration as far as it is public (the agent's cell_config.json)."""
    from make_models import BELT_FRICTION_RANGE, PAD_FRICTION_RANGE
    cfg, run, info = ctx["cfg"], ctx["runner"], ctx["describe"]
    w, sc = cfg["world"], cfg["line"]["shifts"]
    classes = sc["belt_speed_classes"]
    from parcelcell.config import score_config
    from parcelcell.shiftgen import line_list_length, line_setting
    from parcelcell.stream import Shift
    b = dict(cfg["budgets"])
    b.pop("privileged_program_time_s", None)
    b.pop("privileged_program_time_factor", None)
    cams = {}
    for n, c in sorted(cfg["cameras"]["list"].items()):
        e = {"model_camera": c["model_camera"], "size": list(c["size"]), "fovy_deg": c["fovy_deg"],
             "depth_range_m": list(c["depth_range_m"])}
        if "flange_site" in c:
            e["flange_site"] = c["flange_site"]
        cams[n] = e
    out = {
        "cell_version": info["cell_version"],
        "note": "the station's configuration as far as it is public (cell_manual.md gives every value in words). "
                "A shift lasts at most its duration (it ends when the last parcel of its list is settled). "
                "Program time per shift = program_time_factor x the shift's duration + program_time_extra_s; "
                "frames per shift = frames_per_shift_s x the duration, pooled over the four cameras; the CPU time "
                "cap = cpu_cap_factor x the program-time budget.",
        "tick_s": info["tick_s"], "physics_step_s": info["physics_step_s"],
        "station": {"belt_top_z": w["station"]["belt_height"], "belt_centre_x": w["station"]["belt_center_x"],
                    "belt_width": w["station"]["belt_width"], "frame_half_length": w["station"]["visible_half_len"],
                    "work_window_half_length": w["station"]["work_window_half"],
                    "entry_y": w["entry_y"], "scan_y": w["scan_y"], "exit_y": w["exit_y"],
                    "fall_below_belt_m": ctx["fall"][0], "fall_beyond_m": ctx["fall"][1],
                    "hardware": info["station"]["hardware"]},
        "belt": {"speed_m_s": [min(v[0] for v in classes.values()), max(v[1] for v in classes.values())],
                 "speed": "constant through a shift, set per shift in this range; not on the wire",
                 "indexing_zone_y": list(cfg["line"]["indexing_zone_y"]),
                 "modes": "running (the belt runs through the shift) or indexing (it stands while the centre of the "
                          "parcel in the station is inside indexing_zone_y); one mode per shift, not on the wire"},
        "robot": {"start_posture": list(w["start_posture"]),
                  "command_speed_limit_rad_s": w["target_speed_limit_rad_s"],
                  "tool_pose": {k: w["tool_pose"][k] for k in ("iterations", "joint_margin_rad", "reached_m",
                                                               "reached_deg", "gravity_compensation")}},
        "gripper": {"object_detected": {"short_m": w["gripper_object"]["short_m"],
                                        "still_m": w["gripper_object"]["still_m"],
                                        "hold_s": w["gripper_object"]["hold_s"]},
                    "free_opening_m": list(w["gripper_object"]["free_opening_m"]),
                    "free_opening_at_command": [round(0.05 * k, 2) for k in range(len(w["gripper_object"]["free_opening_m"]))]},
        "cameras": {"exposure_period_s": cfg["cameras"]["exposure_ticks"] * info["tick_s"],
                    "frame_age_s": cfg["cameras"]["delay_ticks"] * info["tick_s"], "list": cams,
                    "poses": "the installation drawing's (describe()); the installed cameras differ within "
                             "installation_tolerance: calibrate them on site",
                    "installation_tolerance": cfg["cameras"]["installation_tolerance"]},
        "sensors": {"rgb_noise_sigma": cfg["sensors"]["rgb_noise_sigma"],
                    "depth_noise_rel": cfg["sensors"]["depth_noise_rel"],
                    "depth_dropout": cfg["sensors"]["depth_dropout"],
                    "tau_noise_rel": cfg["sensors"]["tau_noise_rel"],
                    "tau_offset_rel": cfg["sensors"]["tau_offset_rel"],
                    "tau": "the drives' torque readings: the motor torque + zero-mean noise (tau_noise_rel x the "
                           "joint's torque limit, 1 sigma, per joint and tick) + a slowly wandering offset (at most "
                           "tau_offset_rel x the limit); q and dq exact"},
        "friction_ranges": {"belt_cover": list(BELT_FRICTION_RANGE), "gripper_pads": list(PAD_FRICTION_RANGE),
                            "note": "the data sheet's ranges; the station's values lie inside"},
        "parcels": {"kind_shares": dict(sc["kind_shares"]),
                    "rigid_size_m": sc["rigid_size_m"], "soft_size_m": sc["soft_size_m"],
                    "mass_kg": list(sc["mass_kg"]), "friction": list(sc["friction"]),
                    "com_offset_max_m": sc["com_offset_max_m"], "label_min_face_side_m": 0.08,
                    "label_uv_max": sc["label_uv_max"],
                    "label_face_chance": "equal among the faces whose two sides are both at least "
                                         "label_min_face_side_m (the top or the bottom when none is)",
                    "waybill_papers_mm": sorted([list(v) for v in __import__("parcelcell.parcelkit.waybill", fromlist=["PAPERS"]).PAPERS.values()], key=lambda p: -p[0] * p[1]),
                    "set_down": {"y_m": list(sc["set_down_y"]), "x": "the belt's centre line",
                                 "yaw": "any (uniform on the full circle)", "drop_m": 0.002},
                    "list": {"rule": "one parcel at a time: the next is set down at the first tick after the one before "
                                     "it is settled (judged at the scan line, fallen or damaged); the list holds "
                                     "floor(duration_s / (rated_ride_m / belt speed + rated_handling_s)) parcels (the "
                                     "station's rated cycle)",
                             "rated_ride_m": sc["rated_ride_m"], "rated_handling_s": sc["rated_handling_s"],
                             "due": "a parcel is due at the scan line by its set-down time + (scan_y - its set-down "
                                    "y) / belt speed + due_margin_s; a later crossing is not readable",
                             "due_margin_s": cfg["line"]["score"]["due_margin_s"],
                             "length_range": [line_list_length(min(v[0] for v in classes.values()), sc),
                                              line_list_length(max(v[1] for v in classes.values()), sc)]}},
        "score": score_config(cfg, Shift(setting=line_setting(cfg, False))),
        "safety": dict(cfg["safety"]),
        "budgets": b,
        "shifts_graded": run["shifts_graded"],
        "cpu_cap_factor": run["cpu_cap_factor"],
        "import_check": {"wall_s": run["import_check_wall_s"], "cpu_s": run["import_check_cpu_s"]},
        "route_gate": run["route_gate"],
        "commissioning": {"development_shifts": run["commissioning"]["development_shifts"],
                          "bench_shifts": run["commissioning"]["bench_shifts"],
                          "full_shifts": run["commissioning"]["full_shifts"],
                          "shifts": run["commissioning"]["shifts_per_session"],
                          "open_at_a_time": run["commissioning"]["open_at_a_time"],
                          "per_shift": "the graded shift's budgets"},
    }
    from parcelcell.stream import LABEL_MIN_FACE
    out["parcels"]["label_min_face_side_m"] = LABEL_MIN_FACE
    return json.loads(json.dumps(out))


def config_bytes(cfg_obj: dict) -> bytes:
    return (json.dumps(cfg_obj, indent=1, sort_keys=True) + "\n").encode()


def render_all(cfg_path=None, measured_path=None):
    """{output path: bytes} of every generated document, plus the context."""
    F, ctx = FACTS.collect(cfg_path, measured_path)
    out = {}
    for name, path in OUTPUTS.items():
        with open(os.path.join(TEMPLATES, name)) as fh:
            out[path] = reflow(render_text(fh.read(), F)).encode()
    out[CONFIG_OUT] = config_bytes(public_config(ctx))
    return out, F, ctx


def markers(text: str):
    return re.findall(r"\[TO BE [A-Z ]+: [^\]]*\]", re.sub(r"\s+", " ", text))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--measured", default=None)
    a = ap.parse_args(argv)
    for name in OUTPUTS:
        with open(os.path.join(TEMPLATES, name)) as fh:
            bad = template_literals(fh.read())
        for no, line in bad:
            print("unbound number in templates/%s line %d: %s" % (name, no, line))
        if bad and a.write:
            sys.exit("refusing to write: bind every number to a fact first")
    files, F, ctx = render_all(a.config, a.measured)
    rc = 0
    for path, data in files.items():
        rel = os.path.relpath(path, TASK)
        old = open(path, "rb").read() if os.path.exists(path) else None
        if a.write:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if old != data:
                with open(path, "wb") as fh:
                    fh.write(data)
                print("wrote", rel)
            else:
                print("unchanged", rel)
        elif old != data:
            print("differs from its render:", rel)
            rc = 1
        left = markers(data.decode())
        for m in left:
            print("  placeholder left in %s: %s" % (rel, m))
    return rc


if __name__ == "__main__":
    sys.exit(main())
