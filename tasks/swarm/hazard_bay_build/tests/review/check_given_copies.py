"""Maintainers' check: what the agent is given of the grader's own is the grader's.

    python3 tests/review/check_given_copies.py

- the structure check: environment/setup/payload/given/check_structure.py = tests/scan.py, byte for byte;
- the catalog: given/catalog/catalog.json = grader_config.json's `catalog`, every key and number;
- the frames' format: the grader's writer (tests/sensor_rig.py: `Slot`, `manifest`) writes frames of every format into
  a temporary folder and given/frame_reader.py reads them back: bytes and headers equal; a slot being written (seq odd)
  or holding another frame is refused; the manifest's intrinsics and beams are the catalog's.
"""

import filecmp
import importlib.util
import json
import os
import struct
import sys
import tempfile

import numpy as np

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(TASK, "tests")
GIVEN = os.path.join(TASK, "environment", "setup", "payload", "given")
PAIRS = (("tests/scan.py", "check_structure.py"),)
sys.path.insert(0, TESTS)
import sensor_rig  # noqa: E402

bad = [f"DIFFERS: {t} vs given/{g}" for t, g in PAIRS if not filecmp.cmp(os.path.join(TASK, t), os.path.join(GIVEN, g), shallow=False)]
cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
cat = json.load(open(os.path.join(GIVEN, "catalog", "catalog.json")))
if cat != cfg["catalog"]:
    bad.append("DIFFERS: given/catalog/catalog.json vs grader_config.json catalog")

spec = importlib.util.spec_from_file_location("frame_reader", os.path.join(GIVEN, "frame_reader.py"))
fr_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr_mod)
scfg = cfg["sensors"]
models = ["CAM-RGB-S", "THERM-A", "TOF-D", "STEREO-X", "LIDAR-2D", "LIDAR-3D", "CAM-RGB-W"]
plan = dict(sensors=[], not_streamed=[dict(robot="UGV", prim="/World/Fleet/UGV/Extra", model="CAM-RGB-W", why="beyond the robot's first 4 cameras")])
for k, m in enumerate(models):
    sm = sensor_rig.stream_of(m, cfg["catalog"]["sensors"][m])
    plan["sensors"].append(dict(index=k, id=f"s{k:02d}", stem=f"s{k:02d}", robot="UAV_1", prim=f"/World/Fleet/UAV_1/S{k}", model=m, kind=sm["kind"],
                                body="/World/Fleet/UAV_1/Body", pos_in_body=[0.1, 0.0, 0.05], quat_in_body=[1.0, 0.0, 0.0, 0.0], stream=sm))
rng = np.random.default_rng(7)
with tempfile.TemporaryDirectory() as d:
    man = sensor_rig.manifest(plan, scfg)
    sensor_rig.write_excl(os.path.join(d, "manifest.json"), json.dumps(man).encode())
    slots, want = [], {}
    for s in plan["sensors"]:
        sm = s["stream"]
        slots.append([sensor_rig.Slot(os.path.join(d, f"{s['stem']}.{j}"), sm["width"], sm["height"], sm["channels"], sm["dtype"]) for j in range(2)])
    for n in range(3):  # frame sets 0, 1, 2: slot 0 is overwritten by frame set 2
        for i, s in enumerate(plan["sensors"]):
            sm = s["stream"]
            if sm["dtype"] == "uint8":
                a = rng.integers(0, 256, (sm["height"], sm["width"], 4), dtype=np.uint8)  # (RGBA, as rendered: cut to RGB)
                exp = a[..., :3].copy()
            else:
                a = rng.standard_normal((sm["height"], sm["width"]), dtype=np.float32) * 3 + 300
                exp = a.copy()
            slots[i][n % 2].write(n, 6 * n, 0.1 * n, a)
            want[(s["id"], n)] = exp
    fr = fr_mod.Frames({"dir": d, "manifest": "manifest.json"})
    if [s["id"] for s in fr.sensors] != [s["id"] for s in plan["sensors"]]:
        bad.append("the reader's sensors are not the manifest's")
    for s in plan["sensors"]:
        for n in (1, 2):
            arr, head = fr.read(s["id"], {"n": n, "t": 6 * n, "time": 0.1 * n, "slot": n % 2})
            if (arr is None or arr.tobytes() != want[(s["id"], n)].tobytes() or head["n"] != n or head["t"] != 6 * n
                    or abs(head["time"] - 0.1 * n) > 1e-12):
                bad.append(f"{s['id']} ({s['model']}) frame {n}: read back differs")
        arr, head = fr.read(s["id"], {"n": 0, "t": 0, "time": 0.0, "slot": 0})
        if arr is not None:
            bad.append(f"{s['id']}: frame 0 read back after frame 2 replaced it")
    sl = slots[0][1]  # a slot being written (seq odd) is refused
    struct.pack_into("<Q", sl.mm, 8, sl.seq + 1)
    arr, head = fr.read(plan["sensors"][0]["id"], {"n": 1, "t": 6, "time": 0.1, "slot": 1}, tries=3)
    if arr is not None:
        bad.append("a slot being written (seq odd) was read")
    struct.pack_into("<Q", sl.mm, 8, sl.seq)
    for e, s in zip(man["sensors"], plan["sensors"]):  # the manifest against the catalog
        c = cfg["catalog"]["sensors"][s["model"]]
        if s["kind"] == "lidar":
            b = e["beams"]
            if (b["rows"], b["columns"]) != (c["stream"]["rows"], c["stream"]["columns"]) or abs(b["azimuth_step_deg"] * b["columns"] - 360.0) > 1e-9:
                bad.append(f"{s['model']}: beams {b}")
        else:
            fx = c["focal"] / c["hap"] * c["res"][0]
            if abs(e["intrinsics"]["fx"] - fx) > 1e-9 or (e["width"], e["height"]) != tuple(c["res"]):
                bad.append(f"{s['model']}: intrinsics {e['intrinsics']}")
    del fr
    for row in slots:
        for sl in row:
            sl.close()
for b in bad:
    print(b)
print("OK" if not bad else "PROBLEMS")
sys.exit(1 if bad else 0)
