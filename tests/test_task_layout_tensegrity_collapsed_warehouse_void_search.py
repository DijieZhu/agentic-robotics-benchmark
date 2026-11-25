"""tensegrity_collapsed_warehouse_void_search: the Harbor layout and what may reach the agent's image.

The grader's files (tests/), the reference submission (solution/), the hidden site packages and the oracle token must
not reach the image or the agent home: no file of tests/ or solution/ is byte-identical to a file of the image's build
context (environment/ less setup/) or of the givens (environment/setup/payload/), with exactly the declared
exceptions: the practice ground's copy of the cell (environment/practice_ground/voidcell/ = tests/voidcell/ plus
tests/voidscore/ as voidcell/voidscore/, installed for the practice ground's own account only, mode 700), the given
client library (environment/setup/payload/given/lib/voidcell/client.py = tests/voidcell/client.py) and
tests/setup_payload/ (the grader's byte copy of the givens). The practice ground holds no reference program, no token
and no hidden site package. Every given in the agent home is a small text file or picture in git. The plate bundles
(the search camera's offline environment of each site) are not in git: the calibration and practice sites' four come
from the benchmark's asset dataset through environment/assets.sha256 and the vendored fetcher (byte-identical to the
repository's shared/task_assets/fetch_assets.py), installed into the practice ground (readable by its own account
only) in the same image step that removes the download; the hidden sites' ten are one tarball pinned in
grader_config.json (hidden_plates), fetched by verify.py, and only its local copy tests/assets/plates.tar.xz may sit in
the task folder before it is published. No other plate file and no development bundle is in the task folder.
tests/given_hashes.json, the grader's pin of the givens, equals the givens. The image installs only requirements.txt,
practice_ground/ and the asset list with its fetcher, and derives from the pinned base. Stdlib only; nothing is
downloaded or run.
"""
import hashlib
import json
import os
import re
import tarfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "tensegrity_collapsed_warehouse_void_search"
TASK = REPO / "tasks" / "soft" / SLUG
ENV = TASK / "environment"
PAYLOAD = ENV / "setup" / "payload"
PRACTICE = ENV / "practice_ground"
TESTS = TASK / "tests"
SOLUTION = TASK / "solution"
EMPTY = hashlib.sha256(b"").hexdigest()
FETCHER = REPO / "shared" / "task_assets" / "fetch_assets.py"
PRACTICE_BUNDLES = ("cal", "p1", "p2", "p3")
# the files of a plate bundle (the plate backend's format): none of them may sit in the task folder outside tests/assets/
PLATE_FILES = ("meta.json", "plate_ldr.png", "plate_lin.npz", "depth.npz", "depth4.npz", "sunfrac.npz", "air.npz",
               "sunvis.npz", "light.json")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tree(root):
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = _sha(p)
    return out


class TaskLayout(unittest.TestCase):
    def test_top_level_is_the_harbor_layout(self):
        names = {p.name for p in TASK.iterdir()} - {".ale-cache"}
        self.assertEqual(names, {"task.toml", "instruction.md", "environment", "tests", "solution"})

    def test_manifest(self):
        cfg = tomllib.loads((TASK / "task.toml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["schema_version"], "1.4+ale.1")
        self.assertEqual(cfg["task"]["name"], "grounded-intelligence-initiative/" + SLUG)
        self.assertEqual(cfg["metadata"]["platform"], "soft")
        self.assertEqual(cfg["artifacts"], ["/home/user/submission"])
        self.assertEqual((cfg["environment"]["cpus"], cfg["environment"]["memory_mb"]), (8, 16384))
        self.assertEqual((cfg["agent"]["timeout_sec"], cfg["agent"]["network_mode"]), (14400, "no-network"))
        self.assertEqual(cfg["verifier"]["environment_mode"], "separate")
        # the floor for ten serial worst-case episodes plus one re-run: at least 4.5 h
        self.assertGreaterEqual(cfg["verifier"]["timeout_sec"], 16200)
        ve = cfg["verifier"]["environment"]
        self.assertEqual((ve["cpus"], ve["memory_mb"], ve["gpus"]), (8, 16384, 0))

    def test_no_bytecode_in_the_task(self):
        found = [str(p.relative_to(TASK)) for p in TASK.rglob("*")
                 if p.name in ("__pycache__", ".pytest_cache") or p.suffix == ".pyc"]
        self.assertEqual(found, [])

    def test_givens_are_small_files_in_git(self):
        self.assertFalse((PAYLOAD.parent / "assets").exists())
        tree = _tree(PAYLOAD)
        self.assertTrue(tree)
        total = 0
        for rel in tree:
            size = (PAYLOAD / rel).stat().st_size
            total += size
            self.assertLess(size, 512 * 1024, rel)
            self.assertTrue(rel.endswith((".md", ".json", ".png", ".py")), rel)
        self.assertLess(total, 2 * 1024 * 1024)
        for want in ("given/robot_handbook.md", "given/station_api.md", "given/site_briefing.md",
                     "given/camera_datasheet.md", "given/calibration_site/calibration_site.md",
                     "given/calibration_site/site_plan.json", "given/calibration_site/site_plan.png",
                     "given/lib/voidcell/__init__.py", "given/lib/voidcell/client.py", "example/operate.py",
                     "training/run_mission.py"):
            self.assertIn(want, tree)

    def test_the_plate_bundles_are_fetched_not_in_git(self):
        # the practice side: the asset list names the four bundles only, under plates/<name>/, pinned at one revision
        self.assertEqual(_sha(ENV / "fetch_assets.py"), _sha(FETCHER))
        head, rows = {}, []
        for ln in (ENV / "assets.sha256").read_text(encoding="utf-8").splitlines():
            if ln.startswith("#"):
                k, _, v = ln[1:].partition(":")
                head[k.strip()] = v.strip()
            elif ln.strip():
                digest, rel = ln.split(None, 1)
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
                rows.append(rel.strip())
        self.assertEqual(head.get("prefix"), "tasks/soft/%s/payload" % SLUG)
        self.assertRegex(head.get("revision", ""), r"^[0-9a-f]{40}$")
        self.assertNotEqual(head.get("revision"), "0" * 40, "the asset list still names the placeholder revision")
        self.assertTrue(rows)
        self.assertEqual(sorted({r.split("/")[1] for r in rows}), sorted(PRACTICE_BUNDLES))
        self.assertTrue(all(r.split("/")[0] == "plates" and r.count("/") >= 2 for r in rows), rows)
        for name in PRACTICE_BUNDLES:
            self.assertIn("plates/%s/meta.json" % name, rows)
        # the hidden side: one tarball of the asset dataset, pinned in grader_config.json (url, sha256, files)
        cfg = json.loads((TESTS / "grader_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["render_backend"], "plate_filament")
        self.assertEqual(cfg["plate_root"], "hidden/plates")
        spec = cfg["hidden_plates"]
        self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(isinstance(spec["files"], int) and not isinstance(spec["files"], bool) and spec["files"] > 0)
        m = re.match(r"^https://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)$", spec["url"])
        self.assertTrue(m, spec["url"])
        self.assertEqual(m.group(1), head.get("repo"))
        self.assertEqual(m.group(2), head.get("revision"))
        self.assertEqual(m.group(3), "tasks/soft/%s/verifier/plates.tar.xz" % SLUG)
        # the local copy (only until the tarball is published): exactly the pinned bytes, regular files of h01 ... h10
        assets = TESTS / "assets"
        if assets.exists():
            self.assertEqual(sorted(p.name for p in assets.iterdir()), ["plates.tar.xz"])
            self.assertEqual(_sha(assets / "plates.tar.xz"), spec["sha256"])
            with tarfile.open(assets / "plates.tar.xz", "r:xz") as tf:
                members = tf.getmembers()
            self.assertTrue(all(mb.isfile() for mb in members))
            self.assertEqual(len(members), spec["files"])
            self.assertEqual(sorted({mb.name.split("/")[0] for mb in members}), ["h%02d" % i for i in range(1, 11)])
        # the practice ground's sites.json names the same backend and its own plate root
        practice = json.loads((PRACTICE / "sites.json").read_text(encoding="utf-8"))
        self.assertEqual(practice["config"]["render_backend"], "plate_filament")
        self.assertEqual(practice["config"]["plate_root"], "plates")
        # no plate file and no development bundle anywhere else in the task folder
        stray = [str(p.relative_to(TASK)) for p in TASK.rglob("*")
                 if p.is_file() and not p.is_relative_to(assets) and (p.suffix == ".ktx" or p.name in PLATE_FILES)]
        self.assertEqual(stray, [])
        self.assertFalse((ENV / "assets").exists())
        self.assertFalse((PRACTICE / "plates").exists())
        self.assertFalse((TESTS / "hidden" / "plates").exists())

    def test_the_given_client_is_the_graders(self):
        self.assertEqual(_sha(PAYLOAD / "given" / "lib" / "voidcell" / "client.py"), _sha(TESTS / "voidcell" / "client.py"))

    def test_practice_ground_is_the_cell_copy_without_the_hidden_side(self):
        pg = _tree(PRACTICE / "voidcell")
        cell = {r: h for r, h in pg.items() if not r.startswith("voidscore/")}
        score = {r[len("voidscore/"):]: h for r, h in pg.items() if r.startswith("voidscore/")}
        self.assertEqual(cell, _tree(TESTS / "voidcell"))
        self.assertEqual(score, _tree(TESTS / "voidscore"))
        names = [r for r in _tree(PRACTICE) if re.search(r"(reference|voidref|oracle|token|hidden|selection|seeds|"
                                                         r"anchor|limits\.json|grader_config)", r)]
        self.assertEqual(names, [])

    def test_grader_reference_and_hidden_sites_never_reach_the_image(self):
        hidden = {"tests/" + r: h for r, h in _tree(TESTS).items()}
        hidden.update({"solution/" + r: h for r, h in _tree(SOLUTION).items()})
        visible = {}
        for rel, digest in _tree(ENV).items():
            if not rel.startswith("setup/"):
                visible.setdefault(digest, []).append("environment/" + rel)
        for rel, digest in _tree(PAYLOAD).items():
            visible.setdefault(digest, []).append("payload/" + rel)
        bad = []
        for rel, digest in sorted(hidden.items()):
            if digest not in visible or digest == EMPTY:
                continue
            where = visible[digest]
            if rel.startswith("tests/setup_payload/"):
                continue                                            # the grader's copy of the givens
            if rel.startswith(("tests/voidcell/", "tests/voidscore/")) and all(
                    w.startswith("environment/practice_ground/voidcell/") or w == "payload/given/lib/voidcell/client.py"
                    for w in where):
                continue                                            # the declared copies of the cell
            bad.append((rel, where))
        self.assertEqual(bad, [])
        for rel in _tree(TESTS / "hidden"):
            self.assertNotIn(_sha(TESTS / "hidden" / rel), visible, rel)
        for rel in _tree(TESTS / "reference"):
            self.assertNotIn(_sha(TESTS / "reference" / rel), visible, rel)

    def test_setup_payload_is_the_givens_byte_for_byte(self):
        self.assertEqual(_tree(TESTS / "setup_payload"), _tree(PAYLOAD))

    def test_given_hashes_pin_the_givens(self):
        pin = json.loads((TESTS / "given_hashes.json").read_text(encoding="utf-8"))
        self.assertEqual(pin, _tree(PAYLOAD))

    def test_the_oracle_token_lives_in_solution_only(self):
        token = (SOLUTION / "reference" / "oracle_token").read_bytes().strip()
        self.assertGreaterEqual(len(token), 32)
        pinned = (TESTS / "hidden" / "oracle_token.sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(token).hexdigest(), pinned)
        for p in TASK.rglob("*"):
            if p.is_file() and p != SOLUTION / "reference" / "oracle_token":
                data = p.read_bytes()
                self.assertNotIn(token, data, str(p.relative_to(TASK)))
                if p.is_relative_to(ENV):
                    self.assertNotIn(pinned.encode(), data, str(p.relative_to(TASK)))

    def test_hidden_set_is_pinned(self):
        seeds = json.loads((TESTS / "hidden" / "seeds.json").read_text(encoding="utf-8"))
        cfg = json.loads((TESTS / "grader_config.json").read_text(encoding="utf-8"))
        self.assertEqual(len(seeds["sites"]), cfg["episodes"])
        for row in seeds["sites"]:
            self.assertEqual(_sha(TESTS / "hidden" / row["archive"]), row["sha256"], row["tag"])
        practice = json.loads((PRACTICE / "sites.json").read_text(encoding="utf-8"))
        for row in practice["sites"]:
            self.assertEqual(_sha(PRACTICE / row["package"]), row["sha256"], row["name"])
        hidden_hashes = {row["sha256"] for row in seeds["sites"]}
        self.assertFalse(hidden_hashes & {row["sha256"] for row in practice["sites"]})

    def test_dockerfile_installs_only_the_environment(self):
        text = (ENV / "Dockerfile").read_text(encoding="utf-8")
        body = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        froms = [ln for ln in body if re.match(r"^FROM\b", ln)]
        self.assertEqual(froms, ["FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1"])
        copies = [ln.split()[1:] for ln in body if re.match(r"^(COPY|ADD)\b", ln)]
        self.assertEqual(copies, [["--from=ghcr.io/astral-sh/uv:0.11", "/uv", "/usr/local/bin/uv"],
                                  ["requirements.txt", "/tmp/requirements.txt"],
                                  ["practice_ground", "/opt/voidcell-practice"],
                                  ["fetch_assets.py", "assets.sha256", "/opt/task-assets/"]])
        # the practice ground's bundles: fetched to a temporary folder, installed into the practice ground (before the
        # step that gives the folder to its own account), the download and the list removed in the same step
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        runs = [i for i in re.split(r"(?m)^(?=[A-Z]+\b)", code) if i.startswith("RUN")]
        fetch = [r for r in runs if "fetch_assets.py" in r]
        self.assertEqual(len(fetch), 1)
        self.assertIn("plate_install.py practice /tmp/plates-fetch/plates", fetch[0])
        self.assertIn("/opt/voidcell-practice/plates /opt/voidcell-practice/sites.json", fetch[0])
        self.assertIn("rm -rf /tmp/plates-fetch /opt/task-assets", fetch[0])
        self.assertLess(text.index("plate_install.py practice"), text.index("chown -R voidcell:voidcell /opt/voidcell-practice"))
        self.assertIn("! runuser -u user -- ls /opt/voidcell-practice/plates", text)
        self.assertNotIn("/opt/task-assets/payload", text)
        self.assertFalse([ln for ln in body if re.match(r"^\s*ENTRYPOINT\b", ln)])
        self.assertIn("chown -R voidcell:voidcell /opt/voidcell-practice", text)
        self.assertIn("chmod -R u+rwX,go-rwx /opt/voidcell-practice", text)
        self.assertIn("! runuser -u user -- ls /opt/voidcell-practice", text)
        setup = (ENV / "setup" / "setup.sh").read_text(encoding="utf-8")
        self.assertIn("runuser -u voidcell", setup)
        reqs = [ln for ln in (ENV / "requirements.txt").read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")]
        self.assertTrue(reqs and all(re.match(r"^[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.\-+]*$", ln) for ln in reqs), reqs)
        self.assertIn("mujoco==3.14.0", reqs)


if __name__ == "__main__":
    unittest.main()
