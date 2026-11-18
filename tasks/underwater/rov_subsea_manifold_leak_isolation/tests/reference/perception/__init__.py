"""Perception library of task 07 (package P14): the code shared by the reference program, the unprivileged pilot and
the grader-side observability check.  numpy + scipy only (no simulator, no uwkit, no cell import); it is NOT part of
the agent's given files.

    readers     the FIXED readers: plate codes (correction limit 2), painted tags (tagtext v3), handle state
                (handles v4 AND v5 must agree; own-gripper guard); panel-standard numbering; read_panel (code first,
                painted tag only where the code is covered); own_gripper_mask
    codes, tagtext, handles, imgproc, panel_std, panel_codes.json
                the gate G4 readers (proto/g4_optics), package-relative imports, P14 additions appended (tagtext v3,
                handles v5); G4_ORIGIN = their sha256 at the gate's close
    vehicle     the manual's camera mount / tilt hinge / FOV / gripper numbers and the gripper drawing as data
    pose        camera model, 4-DoF (AHRS attitude) and 6-DoF (attitude-free) code PnP, TCP vs stem
    blade       two-colour blade fit with the own gripper masked (last 0.1 m of a grasp approach)
    spots       bright-point (bubble) detection, moving-point filter, projection on a horizontal plane
    sonar_map   sonar pings -> horizontal maps, section components, structure mask from the drawing (ManifoldGA)
    plume_fit   plume source localisation (G3 pilot v3: two section heights x 5 visits + camera exit + flanges)
    current     current estimates: free-drift DVL fit and hold-force inversion with a tank calibration

Conventions: continuous pixel coordinates (pixel (i, j) centre at (i + 0.5, j + 0.5)); camera OpenCV axes; body x
forward / y left / z up; tilt up positive; panel-local P = X right, Y up, Z out of the panel, origin on the stem.
The tag and handle readers assume the camera roll relative to the panel is within +-45 deg (G4): rolled 90 deg about
the gripper axis only the code decoder is valid.
"""
from __future__ import annotations

import hashlib
import os

VERSION = "1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))

# The fixed readers (DESIGN 2.4) = readers.py and every module it uses; P19 checks these against records/P2_P14.md.
# All library files are pinned in READERS.sha256 (tests/test_perception.py::test_reader_files_pinned_and_numpy_only).
READER_FILES = ("readers.py", "codes.py", "tagtext.py", "handles.py", "imgproc.py", "panel_std.py", "panel_codes.json",
                "blade.py", "pose.py", "vehicle.py")
LIBRARY_FILES = READER_FILES + ("__init__.py", "spots.py", "sonar_map.py", "plume_fit.py", "current.py")
# gate G4 origin of the reader files (proto/g4_optics at the gate's close; records/G4.md section 1.3)
G4_ORIGIN = {"panel_std.py": "3acc607ee5be8887", "imgproc.py": "0b356597052799ef", "codes.py": "40fd57b38bdfe18b",
             "tagtext.py": "f4d31ac96e460b95", "handles.py": "063359e95001c030", "panel_codes.json": "5085583bf146ca42"}


def file_sha256(name: str) -> str:
    with open(os.path.join(HERE, name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def reader_sha256() -> dict:
    return {n: file_sha256(n) for n in READER_FILES}


def library_sha256() -> dict:
    return {n: file_sha256(n) for n in LIBRARY_FILES}
