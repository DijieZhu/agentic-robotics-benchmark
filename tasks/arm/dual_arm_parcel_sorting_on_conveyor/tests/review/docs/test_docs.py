"""The binding between the agent's documents and the cell (pytest; a maintainer's check, nothing runs it at verify).

    cd tests/review/docs && PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider test_docs.py

* every generated file equals its render from the cell (``make_docs.py``, ``make_models.py``);
* no template holds a number of its own (``make_docs.template_literals``);
* the numbers of the manual, read back from the rendered text, equal the grader's configuration (an independent
  read: regular expressions on the text against tests/grader_config.json, not through ``facts.py``);
* the public configuration holds every public block and none of the grader-only ones;
* the given model files compile and agree with the cell;
* the text rules of a published task (no machine path, account or host, no ruling or work-package id, none of the
  forbidden names);
* the publication gate: no placeholder left (expected to fail until the measurements and the final look exist).
"""

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import make_docs  # noqa: E402
import make_models  # noqa: E402

TASK = make_docs.TASK
TESTS = make_docs.TESTS


@pytest.fixture(scope="module")
def rendered():
    files, F, ctx = make_docs.render_all()
    return files, F, ctx


@pytest.fixture(scope="module")
def manual(rendered):
    return rendered[0][make_docs.OUTPUTS["cell_manual.md"]].decode()


@pytest.fixture(scope="module")
def cfg():
    sys.path.insert(0, TESTS)
    from parcelcell.config import load_config
    return load_config(os.path.join(TESTS, "grader_config.json"))


def test_shipped_files_equal_their_render(rendered):
    files = rendered[0]
    for path, data in files.items():
        assert os.path.exists(path), path
        with open(path, "rb") as fh:
            assert fh.read() == data, "%s differs from its render: run make_docs.py --write" % path


def test_templates_hold_no_number_of_their_own():
    for name in make_docs.OUTPUTS:
        with open(os.path.join(make_docs.TEMPLATES, name)) as fh:
            bad = make_docs.template_literals(fh.read())
        assert not bad, "unbound numbers in templates/%s: %s" % (name, bad)


def _num(text, rx):
    mo = re.search(rx, text, re.S)
    assert mo, "not found in the manual: %r" % rx
    return [float(g) for g in mo.groups()]


def test_manual_numbers_equal_the_config(manual, cfg):
    w, s, sa, sc, b, sn, c = (cfg["world"], cfg["world"]["station"], cfg["safety"], cfg["score"], cfg["budgets"],
                              cfg["sensors"], cfg["cameras"])
    sh = cfg["line"]["shifts"]
    flat = re.sub(r"\s+", " ", manual)
    checks = [
        (r"Top at z = ([\d.]+) m, centre line at x = ([\d.]+) m, ([\d.]+) m wide", [s["belt_height"],
                                                                                   s["belt_center_x"],
                                                                                   s["belt_width"]]),
        (r"\| set-down \| (-?[\d.]+) to (-?[\d.]+) \|", list(sh["set_down_y"])),
        (r"\| indexing zone \| (-?[\d.]+) to (-?[\d.]+) \|", list(cfg["line"]["indexing_zone_y"])),
        (r"\| scan line \| ([\d.]+) \|", [w["scan_y"]]),
        (r"\| belt end \| ([\d.]+) \|", [w["exit_y"]]),
        (r"Along \|y\| < ([\d.]+) m", [s["work_window_half"]]),
        (r"set per shift between ([\d.]+) and ([\d.]+) m/s", [min(v[0] for v in sh["belt_speed_classes"].values()),
                                                              max(v[1] for v in sh["belt_speed_classes"].values())]),
        (r"\*\*Tool-pose level\.\*\* .{0,2000}? within ([\d.]+) mm and ([\d.]+) degrees", [1000 * w["tool_pose"]["reached_m"],
                                                                                   w["tool_pose"]["reached_deg"]]),
        (r"at most ([\d.]+) rad/s, that is ([\d.]+) rad per tick", [w["target_speed_limit_rad_s"],
                                                                    w["target_speed_limit_rad_s"] * 0.02]),
        (r"\| structure \|[^|]*\| ([\d.]+) N \|", [sa["structure_N"]]),
        (r"\| arm to arm \|[^|]*\| ([\d.]+) N \|", [sa["self_N"]]),
        (r"\| belt \|[^|]*\| ([\d.]+) N \|", [sa["belt_N"]]),
        (r"for (\d+) consecutive ticks \((\d+) ms\) stops the robot", [sa["press_ticks"], 20 * sa["press_ticks"]]),
        (r"for ([\d.]+) s of simulated time every `step`", [sa["stop_lock_s"]]),
        (r"above ([\d.]+) N for (\d+) consecutive ticks, or the", [sa["damage_N"], sa["press_ticks"]]),
        (r"more than ([\d.]+) N in total .{0,120}? for (\d+) consecutive ticks", [sa["squeeze_N"], sa["squeeze_ticks"]]),
        (r"outward normal within ([\d.]+) degrees of vertical\)", [sc["label_tilt_deg"]]),
        (r"touched the belt surface at least once in the last ([\d.]+) s .{0,80}?at most ([\d.]+) mm above the belt",
         [sc["rest_window_s"], 1000.0 * sc["rest_clearance_max_m"]]),
        (r"it turns slower than ([\d.]+) rad/s,\s+and it moves at less than ([\d.]+) m/s", [sc["max_angular_speed_rad_s"],
                                                                                          sc["max_rel_speed_m_s"]]),
        (r"\*\*The shift\.\*\* From t = 0 to at most `describe\(\)\[\"shift\"\]\[\"duration_s\"\]`: ([\d.]+) s for every "
         r"graded shift and every full development shift \(([\d.]+) s for a bench shift\)",
         [sh["duration_s"], sh["bench_duration_s"]]),
        (r"\*\*Program time\.\*\* (?:[\d.]+ x )?[Tt]he shift's duration \+ ([\d.]+) s", [b["program_time_extra_s"]]),
        (r"\*\*Silence\.\*\* At most ([\d.]+) s", [b["silence_s"]]),
        (r"\*\*Frames\.\*\* ([\d.]+) x the shift's duration", [b["frames_per_shift_s"]]),
        (r"One `step` runs at most (\d+) ticks", [b["max_ticks_per_step"]]),
        (r"at most (\d+) more steps, (\d+) more requests and ([\d.]+) s more", [b["after_over_steps"],
                                                                                b["after_over_requests"],
                                                                                b["after_over_s"]]),
        (r"expose together every (\d+) ms", [20 * c["exposure_ticks"]]),
        (r"A frame is delivered (\d+) ms after its exposure", [20 * c["delay_ticks"]]),
        (r"sensor noise of sigma ([\d.]+) levels", [sn["rgb_noise_sigma"]]),
        (r"noise of ([\d.]+) % of the distance \(one sigma\), ([\d.]+) % of the pixels", [100 * sn["depth_noise_rel"],
                                                                                        100 * sn["depth_dropout"]]),
        (r"rigid x ([\d.]+)-([\d.]+) m, y ([\d.]+)-([\d.]+) m, z ([\d.]+)-([\d.]+) m",
         [v for r in sh["rigid_size_m"] for v in r]),
        (r"soft x ([\d.]+)-([\d.]+) m, y ([\d.]+)-([\d.]+) m, z ([\d.]+)-([\d.]+) m",
         [v for r in sh["soft_size_m"] for v in r]),
        (r"\*\*Mass\.\*\* ([\d.]+)-([\d.]+) kg", list(sh["mass_kg"])),
        (r"coefficient ([\d.]+)-([\d.]+) for\s+rigid parcels", list(sh["friction"])),
        (r"horizontally by up to ([\d.]+) m along each side", [sh["com_offset_max_m"]]),
        (r"at the belt's ride over ([\d.]+) m \(from the middle of the set-down range to the scan line\) plus ([\d.]+) s,",
         [sh["rated_ride_m"], sh["rated_handling_s"]]),
        (r"plus its free ride \(from its set-down y to the scan line at the shift's belt speed\) plus ([\d.]+) s",
         [cfg["line"]["score"]["due_margin_s"]]),
        (r"its centre on the belt's centre line \(x = ([\d.]+) m\) between y = (-?[\d.]+) and (-?[\d.]+) m",
         [s["belt_center_x"]] + list(sh["set_down_y"])),
        (r"Constant through a shift and set per shift between ([\d.]+) and ([\d.]+) m/s",
         [min(v[0] for v in sh["belt_speed_classes"].values()), max(v[1] for v in sh["belt_speed_classes"].values())]),
        (r"offset from the face's centre by up to ([\d.]+) % of the free margin", [100 * sh["label_uv_max"]]),
        (r"start_posture\"\]` = \[([^\]]+)\]", None),
    ]
    for rx, want in checks:
        if want is None:
            mo = re.search(rx, flat)
            assert mo, rx
            got = [float(x) for x in mo.group(1).split(",")]
            assert got == pytest.approx(list(w["start_posture"]), abs=1e-9)
            continue
        got = _num(flat, rx)
        assert got == pytest.approx([float(x) for x in want], abs=1e-9), (rx, got, want)
    for name, cam in c["list"].items():
        row = re.search(r"\| `%s` \| \w+ \| (\d+) x (\d+) \| ([\d.]+) \| [\d.]+ \| ([\d.]+)-([\d.]+) \|" % name, flat)
        assert row, name
        assert [float(x) for x in row.groups()] == pytest.approx(
            [cam["size"][0], cam["size"][1], cam["fovy_deg"], cam["depth_range_m"][0], cam["depth_range_m"][1]])


def test_api_and_instruction_numbers(rendered, cfg):
    files = rendered[0]
    api = re.sub(r"\s+", " ", files[make_docs.OUTPUTS["cell_api.md"]].decode())
    ins = re.sub(r"\s+", " ", files[make_docs.OUTPUTS["instruction.md"]].decode())
    b, sa = cfg["budgets"], cfg["safety"]
    assert "at most %d per call" % b["max_ticks_per_step"] in api
    assert "at most %g s of program time from a reply" % b["silence_s"] in api
    rule = ("the shift's duration + %g s" % b["program_time_extra_s"] if b["program_time_factor"] == 1.0 else
            "%g x the shift's duration + %g s" % (b["program_time_factor"], b["program_time_extra_s"]))
    assert "a program time of %s" % rule in ins
    assert re.search(r"\*\*Program time\.\*\* " + re.escape(rule), re.sub(r"\s+", " ", files[make_docs.OUTPUTS[
        "cell_manual.md"]].decode()), re.I)
    assert "above %g N with the station's structure" % sa["structure_N"] in ins
    assert "above %g N for" % sa["damage_N"] in ins
    assert "more than %g N in total" % sa["squeeze_N"] in ins
    assert "%g degrees from vertical" % cfg["score"]["label_tilt_deg"] in ins


GRADER_ONLY = ("belt_speed_classes", "mix_classes", "gap_extra_m", "gap_margin_m", "bench_parcels",
               "bench_gap_extra_m", "solver", "finger_pads", "parcel_contact", "look", "privileged_program_time_s",
               "privileged_program_time_factor", "parcel_visuals", "hardware_boxes", "robot_mounts",
               "runner_account", "socket_sample_s", "settle_s", "belt_friction", "indexing_share")


def test_public_config(rendered, cfg):
    pub = json.loads(rendered[0][make_docs.CONFIG_OUT].decode())
    want_score = json.loads(json.dumps(cfg["score"]))
    want_score["duty"] = cfg["line"]["score"]["duty"]
    want_score["due_margin_s"] = cfg["line"]["score"]["due_margin_s"]
    assert pub["safety"] == cfg["safety"] and pub["score"] == want_score
    b = dict(cfg["budgets"])
    b.pop("privileged_program_time_s", None)
    b.pop("privileged_program_time_factor", None)
    assert pub["budgets"] == b
    txt = json.dumps(pub)
    for k in GRADER_ONLY:
        assert '"%s"' % k not in txt, k
    assert pub["robot"]["start_posture"] == cfg["world"]["start_posture"]
    assert pub["station"]["scan_y"] == cfg["world"]["scan_y"]


def test_development_shifts_are_the_ones_the_manual_describes():
    """The commissioning service's development shifts (environment/commission_service/sites.json, when the
    packaging has written it) are those of gen_shifts.py's rule, in the number the documents state."""
    sites_path = os.path.join(TASK, "environment", "commission_service", "sites.json")
    if not os.path.exists(sites_path):
        pytest.skip("no commissioning service yet")
    sys.path.insert(0, make_docs.REVIEW)
    import gen_shifts
    import common
    cfg = common.load_cfg()
    sj = json.load(open(sites_path))
    seeds = [int(s["seed"]) for s in sj["sites"]]
    bench, full = gen_shifts.line_dev_rule(cfg, bytes.fromhex(sj["salt_hex"]))
    assert len(bench) == gen_shifts.DEV_BENCH and len(full) == gen_shifts.DEV_FULL
    assert seeds == [c["seed"] for c, _ in bench + full]
    assert [s["sha256"] for s in sj["sites"]] == [c["sha256"] for c, _ in bench + full]
    assert [(c["belt_class"], c["indexing"]) for c, _ in full] == list(gen_shifts.LINE_FULL)
    assert [(c["belt_class"], c["indexing"]) for c, _ in bench] == [(b[0], b[1]) for b in gen_shifts.LINE_BENCH]
    assert all(pkg.get("setting", {}).get("name") == "line" for _, pkg in bench + full)
    assert all(c["features"]["classes"]["X"] == 0 for c, _ in bench + full)


def test_given_models_agree_with_the_cell():
    files, _ = make_models.build()
    for rel, data in files.items():
        p = os.path.join(make_models.GIVEN, rel)
        assert os.path.exists(p) and open(p, "rb").read() == data, "given/%s differs from its build" % rel
    assert make_models.check_models() == []


def test_ranges_contain_the_truth_and_the_factors_stay_hidden(cfg):
    """The data sheet's friction ranges contain the cell's values; the factor file stays on the hidden side; the
    documents give the tolerances, never an installed camera's error; the public configuration holds no
    installation."""
    lo, hi = make_models.PAD_FRICTION_RANGE
    assert lo < cfg["world"]["finger_pads"]["friction"] < hi
    lo, hi = make_models.BELT_FRICTION_RANGE
    assert lo < cfg["world"]["station"]["belt_friction"] < hi
    fac = make_models.nominal_factors()
    shipped = open(os.path.join(TASK, "instruction.md")).read()
    for dp, _dn, fns in os.walk(os.path.join(TASK, "environment", "setup", "payload")):
        for f in fns:
            if f.lower().endswith((".md", ".json", ".xml", ".py")):
                shipped += open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
    assert fac["bodies"] and fac["joints"]
    for dp, _dn, fns in os.walk(os.path.join(TASK, "environment")):
        assert os.path.basename(make_models.NOMINAL_FILE) not in fns, dp          # the factors: hidden side only
    for e in cfg["installation"]["cameras"].values():
        for v in e["d_pos_m"] + e["d_rotvec_rad"]:
            assert repr(v) not in shipped
    pub = json.loads(open(make_docs.CONFIG_OUT).read())
    assert "installation" not in pub and pub["cameras"]["installation_tolerance"] == cfg["cameras"]["installation_tolerance"]


def _w(*cps):
    """A scan word from its code points: the words this check refuses are never written out in a published
    file (it is published itself)."""
    return re.escape("".join(chr(c) for c in cps))


# names that never appear (case-insensitive unless marked), paths of the build machine, account and host names,
# outside documents, and ruling or work-package ids; every word is stored as code points
FORBIDDEN = ["(?i)" + _w(108, 101, 110, 115),
             "(?i)" + _w(115, 101, 114, 118, 105, 116, 111, 114),
             _w(0x84DD, 0x601D),
             r"(?i)\b" + _w(100, 101, 109, 45, 49) + r"\b",
             r"\b" + _w(80, 77, 83) + r"\b",
             _w(47, 100, 97, 116, 97, 47), _w(47, 109, 110, 116, 47), _w(47, 116, 109, 112, 47),
             _w(47, 104, 111, 109, 101, 47) + r"(?!user\b)",
             "(?i)" + _w(120, 105, 97, 116, 97, 111), "(?i)" + _w(97, 112, 111, 108, 108, 111),
             _w(100, 111, 99, 115, 47, 112, 108, 97, 110), _w(45, 100, 101, 118, 47),
             r"\b[RW]\d{1,3}\b", r"\bC\d\b(?! class)", r"\bG\d{1,2}\b", r"\bE\d{1,2}\b",
             "(?i)" + _w(121, 97, 108, 101)]


def _published_files():
    roots = [os.path.join(TASK, "instruction.md"), os.path.join(TASK, "environment", "setup", "payload"),
             os.path.join(TESTS, "review")]
    for r in roots:
        if os.path.isfile(r):
            yield r
            continue
        for dp, dn, fn in os.walk(r):
            assert "__pycache__" not in dn, dp
            for f in fn:
                if f.lower().endswith((".md", ".py", ".json", ".xml", ".txt")) or f == "LICENSE":
                    yield os.path.join(dp, f)


def test_text_rules():
    bad = []
    for p in _published_files():
        txt = open(p, encoding="utf-8", errors="replace").read()
        for rx in FORBIDDEN:
            for mo in re.finditer(rx, txt):
                ctx = txt[max(0, mo.start() - 30):mo.end() + 30].replace("\n", " ")
                bad.append("%s: %r in ...%s..." % (os.path.relpath(p, TASK), mo.group(0), ctx))
    assert not bad, "\n".join(bad[:40])


@pytest.mark.xfail(reason="publication gate: placeholders stay until the measurements and the final look exist",
                   strict=False)
def test_publication_gate(rendered):
    left = []
    for path, data in rendered[0].items():
        left += ["%s: %s" % (os.path.relpath(path, TASK), m) for m in make_docs.markers(data.decode())]
    assert not left, "\n".join(left)
