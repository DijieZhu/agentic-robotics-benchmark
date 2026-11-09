# NOTICE - sources of this task's photo and truth (verify / maintainers only; never shown to the agent)

The photographed scene, its photo and the material the truth is built from come from the **HOPE** dataset (Household
Objects for Pose Estimation; S. Tyree, J. Tremblay, T. To, J. Cheng, T. Mosier, J. Smith, S. Birchfield, "6-DoF Pose
Estimation of Household Objects for Robotic Manipulation: An Accessible Dataset and Benchmark", IROS 2022,
arXiv:2203.05701; https://github.com/swtyree/hope-dataset), Copyright (C) 2020 NVIDIA Corporation, licensed under
**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International,
https://creativecommons.org/licenses/by-nc-sa/4.0/). The copy used is the BOP distribution of HOPE, Hugging Face
dataset `bop-benchmark/hope` at revision `ddd0a26ca3460085e93b648748160fb25e4b5566` (`hope_val_realsense.zip`,
`hope_models.zip`, `hope_base.zip`). This notice is the attribution the licence asks for. The material is provided by
its licensors as-is, without warranties.

- **Used:** validation scene 000009 (RealSense D415 colour images 000000-000004 with their registered depth,
  `scene_camera.json`, `scene_gt.json`, `scene_gt_info.json`, the visible masks), the object models of the 15
  products in the scene (`models_eval/`, `models/` with their textures, `models_info.json`) and the product names of
  `dataset_info.md`.
- **Agent-visible:** `photo.png` (`rgb/000000.png` of that scene, unmodified; in development in
  `environment/setup/payload/`, published through the benchmark's asset repository) and `product_names.json`
  (the 28 product names in NVIDIA's spellings). The givens never name the source.
- **Modifications (the truth in `tests/gt/` and the oracle in `solution/answer/`):**
  - every item's collision shape is the convex hull of the evaluation model, in metres, in the model frame (the six
    box-shaped products clipped by 1 mm); no texture or visual mesh is shipped;
  - the annotated poses are expressed in a gravity-aligned world fitted to the console's top, then fixed by at most
    3 mm / 3 deg so that the twin rests under the task's standard simulation, settled and frozen;
  - the console, its compartments and its drawer are boxes fitted to the fused depth of the five images;
  - the depth reference (`depth_ref.png`, `depth_class.png`) is the median of the five depth frames, kept where they
    agree with each other and with the fitted console;
  - the support and probe truths (`support_truth.json`, `probe_truth.json`) are simulation outcomes of that twin under
    the task's robustness variants;
  - densities, frictions and the camera's pixel convention (principal point minus 0.5 px, OpenCV pixel centres) are
    this task's own.
- **ShareAlike:** these adaptations (`tests/gt/`, `tests/setup_payload/photo.png`, `solution/answer/`, and the
  photo in the asset repository) are published under the same licence, CC BY-NC-SA 4.0, with this source and
  modification note; the rest of the task is the benchmark's own.
- **Date:** prepared by the maintainers in November 2026.
