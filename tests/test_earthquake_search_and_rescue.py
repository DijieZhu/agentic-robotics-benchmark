"""ground/earthquake_search_and_rescue: the facts that several files state must agree (stdlib only, no image).

  * the training ground's allowance: the service's configuration, the instruction and the robot manual;
  * the hidden set: the grader's configuration, the pinned seeds, the selection record and the instruction;
  * the byte copies: the training ground's cell = the grader's cell, the grader's copy of the givens = the givens,
    and the givens' hashes;
  * the public contract: the instruction names every weight-file extension and training-call string that the
    submission checks refuse;
  * the numerics pin: the variables the run regime sets are the ones the instruction and the image name.
"""
import ast
import hashlib
import json
import pathlib
import re

TASK = pathlib.Path(__file__).resolve().parents[1] / "tasks" / "ground" / "earthquake_search_and_rescue"
MISSIONS = 80
SITES = 8


def _text(rel):
    return (TASK / rel).read_text(encoding="utf-8")


def _files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def _constants(rel):
    out = {}
    for node in ast.parse(_text(rel)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_the_training_allowance_is_stated_alike():
    sites = json.loads(_text("environment/training_ground/sites.json"))
    assert sites["missions"] == MISSIONS and sites["open"] == 2
    assert "allows **%d missions** in total, 2 at a time" % MISSIONS in _text("instruction.md")
    assert "the training ground's %d\n  missions" % MISSIONS in _text("instruction.md")
    assert "allows %d missions in total and 2 at a time" % MISSIONS in _text("environment/setup/payload/given/robot_manual.md")
    kinds = [s["kind"] for s in sites["sites"]]
    assert kinds == ["lanes"] + ["training"] * 6


def test_the_training_ground_gives_no_truth_and_no_score():
    src = _text("environment/training_ground/trainingd.py")
    for word in ("surv_torso", "score()", "report_error_m", "kit_dist_m", "mission_score", "person_position"):
        assert word not in src, word


def test_the_hidden_set_is_eight_sites_from_the_selection_record():
    cfg = json.loads(_text("tests/grader_config.json"))
    seeds = json.loads(_text("tests/hidden/seeds.json"))["seeds"]
    sel = json.loads(_text("tests/hidden/selection.json"))
    assert cfg["episodes"] == SITES == len(seeds) == len(set(seeds))
    assert seeds == sel["chosen"]["hidden"]["seeds"]
    assert all(s in sel["candidates"]["hidden"] for s in seeds)
    training = [s["seed"] for s in json.loads(_text("environment/training_ground/sites.json"))["sites"] if s["kind"] == "training"]
    assert training == sel["chosen"]["training"]["seeds"]
    assert not set(training) & set(sel["candidates"]["hidden"])
    assert "on **%d hidden sites**" % SITES in _text("instruction.md")
    assert json.loads(_text("tests/anchor.json"))["full_at"] == 0.98


def test_the_training_ground_runs_the_graders_cell_and_numbers():
    assert _files(TASK / "tests" / "rubblecell") == _files(TASK / "environment" / "training_ground" / "rubblecell")
    cfg = json.loads(_text("tests/grader_config.json"))
    tcfg = json.loads(_text("environment/training_ground/sites.json"))["config"]
    for k in ("sim_cap_s", "program_time_cap_s", "silence_s", "max_ticks_per_step", "camera_budget_frames",
              "lidar_budget_sweeps", "thermal_budget_frames", "score", "safety", "grip", "link", "temps"):
        assert tcfg[k] == cfg[k], k


def test_the_graders_copy_of_the_givens_is_the_givens():
    given = _files(TASK / "environment" / "setup" / "payload")
    assert given == _files(TASK / "tests" / "setup_payload")
    assert given == json.loads(_text("tests/given_hashes.json"))


def test_the_instruction_names_what_the_submission_checks_refuse():
    c = _constants("environment/setup/payload/training/submission_checks.py")
    ins = _text("instruction.md")
    for ext in c["FORBIDDEN_EXTENSIONS"]:
        assert "`%s`" % ext in ins, ext
    for text in c["FORBIDDEN_STRINGS"]:
        assert "`%s`" % text in ins, text
    assert "at most %d MB" % c["MAX_SUBMISSION_MB"] in ins
    assert "%d s of wall clock and %d CPU-seconds" % (c["IMPORT_WALL_S"], c["IMPORT_CPU_S"]) in ins
    for mod in ("socket", "subprocess", "asyncio", "ctypes"):
        assert mod in c["FORBIDDEN_MODULES"] and "`%s`" % mod in ins


def test_the_numerics_pin_is_stated_and_set_alike():
    src = _text("environment/setup/payload/training/submission_checks.py")
    pin = dict(re.findall(r'"((?:OPENBLAS|OMP|MKL|NPY|OPENCV)_[A-Z0-9_]+)": "([^"]+)"', src))
    assert pin["OPENBLAS_CORETYPE"] == "Haswell" and pin["OPENBLAS_NUM_THREADS"] == "1" and pin["OPENCV_IPP"] == "avx2"
    ins, docker = _text("instruction.md"), _text("environment/Dockerfile")
    for k in ("OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES", "OPENCV_IPP"):
        assert "%s=%s" % (k, pin[k]) in ins.replace('"', "") or "%s=\"%s\"" % (k, pin[k]) in ins, k
        assert ("%s=%s" % (k, pin[k]) in docker) or ('%s="%s"' % (k, pin[k]) in docker), k
    ref = json.loads(_text("tests/numerics.json"))
    assert ref["math"] and all(len(v) == 16 or v == "absent" for v in ref["math"].values())


def test_the_look_pack_is_one_pinned_archive_fetched_by_the_image_build():
    lines = _text("environment/assets.sha256").splitlines()
    files = [ln.split("  ", 1) for ln in lines if ln and not ln.startswith("#")]
    assert [f[1] for f in files] == ["look/rubblecell_look_pack.tar"]
    assert len(files[0][0]) == 64
    head = dict(ln[2:].split(": ", 1) for ln in lines if ln.startswith("# "))
    assert head["prefix"] == "tasks/ground/earthquake_search_and_rescue/payload" and len(head["revision"]) == 40
    docker = _text("environment/Dockerfile")
    assert "fetch_assets.py /tmp/task-assets/assets.sha256" in docker and "rubblecell_look_pack.tar" in docker
    assert "rm -rf /tmp/task-assets" in docker                       # no layer keeps the archive
    assert "chmod -R u+rwX,g+rX,o-rwx /opt/rubblecell-look" in docker  # not readable by the agent's account


def test_the_selection_record_is_of_this_cell_and_this_reference():
    sel = json.loads(_text("tests/hidden/selection.json"))
    version = re.search(r'^CELL_VERSION = "([^"]+)"', _text("tests/rubblecell/__init__.py"), re.M).group(1)
    assert sel["cell_version"] == version
    assert all(r["cell_version"] == version for r in sel["runs"]) and len(sel["runs"]) == 48
    last = sel["amendments"][-1]
    assert last["n"] == 3
    digest = hashlib.sha256((TASK / "tests/reference/reference.py").read_bytes()).hexdigest()
    assert digest in last["reference"]
    chosen = sel["chosen"]["hidden"]
    assert all(r["admitted"] for r in sel["runs"] if r["pool"] == "hidden" and r["index"] in chosen["indices"])
