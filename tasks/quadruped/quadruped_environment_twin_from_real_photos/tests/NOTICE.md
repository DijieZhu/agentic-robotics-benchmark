# NOTICE - sources and licences of this task's photos, truth and tools (verify / maintainers only; never shown to the agent)

## The photos and everything derived from the laser scan

The surveyed site, its photos and the material the truth is built from come from the **ETH3D** benchmark: the
high-resolution multi-view training scene `delivery_area` (T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler,
K. Schindler, M. Pollefeys, A. Geiger, "A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera
Videos", IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017; https://www.eth3d.net/). The ETH3D
data is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International,
https://creativecommons.org/licenses/by-nc-sa/4.0/): "The data provided here is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License" (https://www.eth3d.net/). This notice is the
attribution the licence asks for. The material is provided by its licensors as-is, without warranties.

- **Used:** the archives `delivery_area_dslr_undistorted.7z` (the 44 undistorted DSLR images and their calibration:
  intrinsics and poses), `delivery_area_scan_clean.7z` (the laser scans and their alignment),
  `delivery_area_dslr_depth.7z`, `delivery_area_dslr_occlusion.7z` and `delivery_area_dslr_scan_eval.7z` (per-image
  depth and occlusion data used to check the truth), and `delivery_area_dslr_jpg.7z` (only the capture times of the
  original images, which fix the photo order), from `https://www.eth3d.net/data/`.
- **Agent-visible:** `photos/site_001.jpg` .. `site_044.jpg`: the 44 undistorted JPEG images of the scene, byte for
  byte as distributed (no metadata beyond the JFIF header), renamed in capture order (the original file numbers
  `DSC_0675` .. `DSC_0718` map to `site_001` .. `site_044` in that order); in development in
  `environment/setup/payload/photos/` and `tests/setup_payload/photos/`, published through the benchmark's asset
  repository. The givens never name the source.
- **Modifications (the truth in `tests/gt/` and the oracle in `solution/answer/`):**
  - the site frame (z up, z = 0 at the floor under the camera path, x along the main hall's long wall) is a rigid
    transform of the dataset's frame fitted to the scan;
  - the true cameras (`cameras_true.json`, the photo cameras of the true twin) are the dataset's calibration
    expressed in that frame, with the principal point converted to OpenCV pixel centres (minus 0.5 px);
  - the static building (floor, walls, pillars, dock, kerbs, rack, cages, doors, fixtures) and every loose unit are
    unions of convex pieces fitted by the maintainers to the scan and the photos, hidden sides completed from each
    class's known shape; the floor is one flat slab at z = 0, the units were shifted to rest on it (at most 2.7 cm);
    the two mesh-cage gates are slide joints along their rails;
  - masses, frictions, class labels, zones, the scored region, the per-sample "determined" flags and every other
    truth file (`items_surface.npz`, `c3_samples.npz`, `slice_truth.npz`, `support_truth.json`, `probe_truth.json`,
    `routes.json`, `robustness.json`, `sfm_selftest.json`) are this task's own, computed from that twin, the scan and
    the photos.
- **ShareAlike:** these adaptations (`tests/gt/`, `tests/setup_payload/photos/`, `solution/answer/`, and the photos
  in the asset repository) are published under the same licence, CC BY-NC-SA 4.0, with this source and modification
  note; the rest of the task is the benchmark's own.

## The probe robot

The probe robot's description (`robot/robot/model/robot.xml` and its meshes in the agent's `robot/`, mirrored in
`tests/setup_payload/robot/`) is the nominal public design of the open-source Solo12 quadruped of the Open Dynamic Robot
Initiative (numbers and meshes from https://github.com/open-dynamic-robot-initiative/robot_properties_solo at commit
`2201f454`, with a small inspection payload of this benchmark's own), Copyright (c) 2019, New York University and Max
Planck Gesellschaft, under the **BSD 3-Clause License**; the licence text travels with the model
(`robot/robot/LICENSE-robot-description-BSD-3-Clause.txt`). The controller and the walk harness are this benchmark's
own. Agent-visible files do not name the robot's origin (and, by the licence's third clause, the holders' names are
not used to endorse or promote this task).

## OpenMVS in the image

The task image contains the unmodified command-line tools of **OpenMVS 2.4.0** (https://github.com/cdcseacave/openMVS),
licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0), installed from the conda-forge build
`openmvs 2.4.0 cpu_h59d1794_202` with the libraries it loads (their licences: `/opt/openmvs/PACKAGES.tsv` and
`/opt/openmvs/licenses/` in the image). The repository contains none of its code: `environment/Dockerfile` installs
it from the package list `environment/openmvs-conda-explicit.txt` when the image is built. Whoever distributes a built
image conveys these binaries and must give access to their corresponding source (`/opt/openmvs/README.md` names it);
the other programs in the image are separate programs and are not covered by the AGPL.

- **Date:** prepared by the maintainers in November 2026.
