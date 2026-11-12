"""Audit the frozen small-box evaluation and write its bounded results report."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DEV = set(range(5)) | set(range(100, 105))
HOLDOUT = set(range(200, 205))
MASSES = {50, 100, 200}
EXPECTED_CRITERIA = {
    "min_tray_lift_m": .08, "min_loaded_translation_m": .30,
    "delivery_xy_tolerance_m": .035, "resting_speed_m_s": .025,
    "resting_angular_speed_rad_s": .10, "gripper_clearance_m": .25,
    "stable_control_steps": 20,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    cases = []
    frozen = None
    for root in args.batches:
        root = root.resolve()
        summary = json.loads((root / "summary.json").read_text())
        require(len(summary["cases"]) == summary["planned_trials"], f"Incomplete batch: {root}")
        hashes = {key: summary[key] for key in (
            "controller_sha256", "transport_helper_sha256", "reference_sha256", "dependency_sha256")}
        if frozen is None:
            frozen = hashes
        require(hashes == frozen, f"Mixed source versions: {root}")
        for case in summary["cases"]:
            require(case["success"] and case["exit_code"] == 0, f"Failed case: {case}")
            path = Path(case["result_path"])
            r = json.loads(path.read_text())
            require(r["success"] and r["criteria"] == EXPECTED_CRITERIA, f"Result/criteria mismatch: {path}")
            require(r["script_sha256"] == hashes["controller_sha256"]
                    and r["transport_helper_sha256"] == hashes["transport_helper_sha256"]
                    and r["reference_script_sha256"] == hashes["reference_sha256"]
                    and r["dependency_sha256"] == hashes["dependency_sha256"], f"Hash mismatch: {path}")
            require(r["seed"] == case["seed"]
                    and math.isclose(r["payload_mass_kg"], case["mass_g"] / 1000, rel_tol=1e-8), f"Mass/seed mismatch: {path}")
            require(all(r[k] for k in ("payload_required", "loaded_verified", "lift_verified",
                    "loaded_transport_verified", "payload_contained", "tray_on_counter", "payload_tray_contact")), f"Missing physical evidence: {path}")
            require(not r["tray_grasp_contact"] and r["retention_violations"] == 0
                    and r["max_tray_lift_m"] >= .08 and r["max_loaded_translation_m"] >= .30
                    and r["final_delivery_error_m"] < .035 and r["gripper_distance_m"] > .25
                    and abs(r["final_tray_position"][2] - r["destination"][2]) < .012
                    and r["final_tray_speed_m_s"] < .025 and r["final_payload_speed_m_s"] < .025
                    and r["final_tray_angular_speed_rad_s"] < .10 and r["stable_success_steps"] >= 20,
                    f"Failed recorded acceptance: {path}")
            require(not r["runtime_attachments"] and not r["runtime_object_teleportation"], f"Runtime state shortcut: {path}")
            trace = [json.loads(line) for line in (path.parent / "trace.jsonl").read_text().splitlines()]
            require(len(trace) == r["steps"] and [x["step"] for x in trace] == list(range(1, r["steps"] + 1)), f"Incomplete trace: {path}")
            require(all(x["payload_contained"] and x["retention_violations"] == 0 for x in trace), f"Trace retention failure: {path}")
            for stage in ("lift_loaded_tray", "transport_loaded_tray"):
                require(any(x["stage"] == stage and x["tray_grasp"] for x in trace), f"No grasp evidence at {stage}: {path}")
            video = path.parent / r["video_file"]
            require(video.is_file() and video.stat().st_size > 0, f"Missing video: {video}")
            depths = list(path.parent.glob("*_depth_m.npy"))
            require(len(depths) == 3, f"Missing RGB-D camera artifacts: {path}")
            for depth_path in depths:
                depth = np.load(depth_path)
                require(depth.shape[:2] == (256, 256) and np.isfinite(depth).all()
                        and (depth > 0).all(), f"Invalid metric depth: {depth_path}")
                require(depth_path.with_name(depth_path.name.replace("_depth_m.npy", "_initial.png")).is_file(),
                        f"Missing RGB frame for {depth_path}")
            cases.append({"mass_g": case["mass_g"], "seed": case["seed"],
                          "split": "development" if case["seed"] in DEV else "holdout",
                          "result_path": str(path), "video_path": str(video), "result": r})
    expected = {(mass, seed) for mass in MASSES for seed in DEV | HOLDOUT}
    require(len(cases) == len(expected) and {(x["mass_g"], x["seed"]) for x in cases} == expected,
            "Evaluation must contain each of the 45 planned mass/seed pairs exactly once")
    here = Path(__file__).resolve().parent
    source_keys = {"preloaded_box_transport.py": frozen["controller_sha256"],
                   "loaded_tray_transport.py": frozen["transport_helper_sha256"],
                   "auto_pick_place.py": frozen["reference_sha256"], **frozen["dependency_sha256"]}
    for filename, digest in source_keys.items():
        require(hashlib.sha256((here / filename).read_bytes()).hexdigest() == digest,
                f"Current source differs from evaluated source: {filename}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    aggregate = {"status": "passed", "development_passed": 30, "development_total": 30,
                 "holdout_passed": 15, "holdout_total": 15, "source_hashes": frozen,
                 "batches": [str(x.resolve()) for x in args.batches], "cases": cases}
    (out / "evaluation.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    lines = ["# Small-box transport feasibility evaluation", "",
             "Final frozen protocol: **45/45 passed** (30/30 development and regression; 15/15 new held-out cases).", "",
             "Development seeds: 0–4 and 100–104. New holdout seeds: 200–204. Each seed uses 50, 100, and 200 g payloads.", "",
             "| Payload | Development | New holdout | Minimum lift | Minimum airborne transport | Maximum tilt | Maximum final XY error |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for mass in sorted(MASSES):
        rows = [x["result"] for x in cases if x["mass_g"] == mass]
        lines.append(f"| {mass} g | 10/10 | 5/5 | {min(x['max_tray_lift_m'] for x in rows)*100:.2f} cm | "
                     f"{min(x['max_loaded_translation_m'] for x in rows)*100:.2f} cm | "
                     f"{max(x['max_tray_tilt_deg'] for x in rows):.3f}° | {max(x['final_delivery_error_m'] for x in rows)*1000:.3f} mm |")
    lines += ["", "All cases recorded zero containment violations from the first controller action through withdrawal, "
              "physical lift and grasped airborne transport, released placement, and at least 20 stable control steps. "
              "The report additionally checked trace completeness, control-step containment, grasp evidence, masses, source hashes, "
              "video presence, and three RGB/finite positive metric-depth camera artifacts per case.", "",
              "The 94 × 68 × 46 mm rigid box is grasped across its 68 mm short side. Empty box mass is 8.6 g. "
              "The fixed package is approximately 23 × 59 × 36 mm. Mass and inertia are scaled together before controller execution; "
              "200 g represents dense ballast (about 4.2 g/cm³ based on bounding volume), rather than ordinary snack material. "
              "Contact parameters, robot force limits, and geometry are unchanged. Initial reset height is specified inside the box; "
              "there are no runtime object teleports or attachments.", "",
              "## Scope and retained failures", "",
              "These are preloaded-container transport component tests using private simulation poses and contacts, native PandaOmron, "
              "and fixed kitchen layout/style 1/1. Seeds provide small initialization jitter; object identity and geometry are fixed. "
              "This does not establish broad scene generalization, a sensor-only policy, robot loading ability, or a completed long-horizon benchmark.", "",
              "Earlier tray-rim failures, fruit preloading failures, the interrupted small-package diagnostic batch, "
              "and initial generator/interface failures remain under outputs/. The previous holdout exposed invalid preloading "
              "at seeds 100 and 104; those seeds became development data. They are not counted as held out here. "
              "This final matrix reran every planned pair after the reset-height and full-episode retention-monitor fixes.", "",
              "## Artifacts", ""]
    for root in args.batches:
        path = root.resolve() / "summary.md"
        lines.append(f"- [{root.name}](<{path}>)")
    representative = next(x for x in cases if x["mass_g"] == 200 and x["seed"] == 204)
    lines += [f"- [200 g / new holdout seed 204 video](<{representative['video_path']}>)", "", "## Frozen sources", ""]
    lines += [f"- `{name}`: `{digest}`" for name, digest in source_keys.items()]
    lines += ["", "Environment: RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1, Python 3.11 on Apple Silicon macOS.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(out / "report.md")
    print("45/45 cases and recorded-evidence audit: PASS")


if __name__ == "__main__":
    main()
