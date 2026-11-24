"""The format of the two output files, filled with the rough prior and no faults.

    python3 ~/tools/prior_baseline.py --input <rig_dir> --output <out_dir>

It calibrates nothing; it only shows a valid calibration.yaml and fault_report.json (the CAD prior, the nominal
camera, zero clocks and biases, no faults, no change).
"""

import argparse
import json
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.input, "rough_prior.yaml")) as fh:
        prior = yaml.safe_load(fh)
    with open(os.path.join(a.input, "manifest.json")) as fh:
        cam = json.load(fh)["camera"]
    os.makedirs(a.output, exist_ok=True)
    cal = {
        "T_lidar_camera": prior["T_lidar_camera"],
        "T_lidar_imu": prior["T_lidar_imu"],
        "camera": {"fx": cam["nominal_focal_px"], "fy": cam["nominal_focal_px"],
                   "cx": cam["nominal_principal_point_px"][0], "cy": cam["nominal_principal_point_px"][1],
                   "distortion": [0.0, 0.0, 0.0, 0.0, 0.0], "readout_s": 0.02},
        "time": {"camera": {"offset_s": 0.0, "drift_ppm": 0.0},
                 "lidar": {"offset_s": 0.0, "drift_ppm": 0.0}},
        "imu": {"gyro_bias_rps": [0.0, 0.0, 0.0], "accel_bias_mps2": [0.0, 0.0, 0.0]},
    }
    with open(os.path.join(a.output, "calibration.yaml"), "w") as fh:
        yaml.safe_dump(cal, fh, sort_keys=False)
    with open(os.path.join(a.output, "fault_report.json"), "w") as fh:
        json.dump({"faults": []}, fh, indent=1)


if __name__ == "__main__":
    main()
