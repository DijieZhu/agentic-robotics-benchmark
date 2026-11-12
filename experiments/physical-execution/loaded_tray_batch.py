"""Run a fixed loaded-tray mass/seed matrix without dropping failed trials."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_summary(root, summary):
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Carrier transport mass and seed matrix", "", f"Controller SHA256: `{summary['controller_sha256']}`", "",
             "| Payload (g) | Seed | Result | Last stage | Reason |",
             "| --- | --- | --- | --- | --- |"]
    for case in summary["cases"]:
        r = case.get("result", {})
        reason = r.get("error", case.get("error", "")).replace("|", "/")
        lines.append(f"| {case['mass_g']:g} | {case['seed']} | {'PASS' if case['success'] else 'FAIL'} | {r.get('last_stage', 'infrastructure')} | {reason} |")
    lines += ["", f"Completed {len(summary['cases'])}/{summary['planned_trials']} trials; "
              f"passed {sum(c['success'] for c in summary['cases'])}.", ""]
    (root / "summary.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--masses-g", type=float, nargs="+", default=[50, 100, 200])
    p.add_argument("--label", default="evaluation")
    p.add_argument("--timeout-sec", type=float, default=240)
    p.add_argument("--script", choices=["loaded_tray_transport.py", "loaded_box_transport.py", "preloaded_box_transport.py"],
                   default="loaded_tray_transport.py")
    args = p.parse_args()
    script = HERE / args.script
    helper = HERE / "loaded_tray_transport.py"
    dependencies = [HERE / "loaded_box_transport.py"] if args.script == "preloaded_box_transport.py" else []
    reference = HERE / "auto_pick_place.py"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = HERE / "outputs" / f"transport_batch_{args.label}_{stamp}"
    root.mkdir(parents=True)
    for source in {script, helper, reference, Path(__file__), *dependencies}:
        shutil.copy2(source, root / source.name)
    summary = {
        "controller_sha256": digest(script), "reference_sha256": digest(reference),
        "transport_helper_sha256": digest(helper), "entry_script": args.script,
        "dependency_sha256": {p.name: digest(p) for p in dependencies},
        "masses_g": args.masses_g, "seeds": args.seeds,
        "planned_trials": len(args.masses_g) * len(args.seeds), "cases": [],
    }
    print(f"Batch outputs: {root}", flush=True)
    write_summary(root, summary)
    for mass in args.masses_g:
        for seed in args.seeds:
            if (digest(script) != summary["controller_sha256"]
                or digest(reference) != summary["reference_sha256"]
                or digest(helper) != summary["transport_helper_sha256"]
                or any(digest(p) != summary["dependency_sha256"][p.name] for p in dependencies)):
                raise RuntimeError("Controller changed during the batch; refusing mixed-version evaluation")
            case_root = root / f"mass{mass:g}g_seed{seed}"
            case_root.mkdir()
            log = case_root / "console.log"
            case = {"mass_g": mass, "seed": seed, "success": False, "log": str(log)}
            print(f"START {mass:g} g / seed {seed}", flush=True)
            try:
                with log.open("w") as output:
                    process = subprocess.run(
                        [sys.executable, "-I", str(script), "--seed", str(seed),
                         "--payload-mass-g", str(mass), "--output-root", str(case_root)],
                        cwd=HERE, stdout=output, stderr=subprocess.STDOUT,
                        timeout=args.timeout_sec, check=False,
                    )
                case["exit_code"] = process.returncode
                records = list(case_root.glob("*/result.json"))
                if len(records) != 1:
                    raise RuntimeError(f"Expected one result record, got {len(records)}")
                record = records[0]
                result = json.loads(record.read_text())
                case["result_path"] = str(record)
                case["result"] = result
                if result["script_sha256"] != summary["controller_sha256"]:
                    raise RuntimeError("Result controller hash differs from the frozen batch")
                if result["reference_script_sha256"] != summary["reference_sha256"]:
                    raise RuntimeError("Result helper hash differs from the frozen batch")
                if result["transport_helper_sha256"] != summary["transport_helper_sha256"]:
                    raise RuntimeError("Transport helper differs from the frozen batch")
                if result["dependency_sha256"] != summary["dependency_sha256"]:
                    raise RuntimeError("Carrier controller differs from the frozen batch")
                if not math.isclose(result["payload_mass_kg"], mass / 1000, rel_tol=1e-8):
                    raise RuntimeError("Simulated payload mass differs from the requested mass")
                case["success"] = process.returncode == 0 and result["success"] is True
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
            summary["cases"].append(case)
            write_summary(root, summary)
            r = case.get("result", {})
            print(f"{'PASS' if case['success'] else 'FAIL'} {mass:g} g / seed {seed}: "
                  f"{r.get('error', case.get('error', 'completed'))}", flush=True)
    passed = sum(c["success"] for c in summary["cases"])
    print(f"FINAL {passed}/{summary['planned_trials']} passed; {root / 'summary.json'}", flush=True)
    return 0 if passed == summary["planned_trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
