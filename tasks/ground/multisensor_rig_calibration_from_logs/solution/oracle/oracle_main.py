"""The reference: token-privileged. The grader hands the true parameters and the fault list of a rig to a
submission whose .oracle_token it recognises (tests/verify.py, `--truth <dir>`); this program writes them out in
the output format. It reads nothing of the rig's logs and computes nothing."""
import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--truth", default="")
a = ap.parse_args()
if not a.truth:
    raise SystemExit("the reference needs the grader's --truth folder (it is privileged; it is not a calibrator)")
shutil.copyfile(os.path.join(a.truth, "ground_truth.yaml"), os.path.join(a.output, "calibration.yaml"))
with open(os.path.join(a.truth, "faults.json")) as f:
    doc = json.load(f)
faults = doc["faults"] if isinstance(doc, dict) else doc
with open(os.path.join(a.output, "fault_report.json"), "w") as f:
    json.dump({"faults": faults}, f)
