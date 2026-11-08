# Detect 6-DoF parallel-jaw grasps on objects you have never seen

A robot with a parallel-jaw gripper has to pick objects out of a cluttered tabletop from a single RGB-D frame, and the objects it meets at deployment are not the ones anyone trained on. You are given a working but weak detector for this setting and real captured data to develop against. Make the detector better: after you finish, your detector is re-run on frames of unseen objects, the grasps it proposes are scored with the benchmark's force-closure evaluation, and you are credited with how much it improves on the detector you were given.

## What You Submit

A Python module at `/home/user/submission/solver.py` that defines

```python
predict(rgb, depth, intrinsics, depth_scale) -> numpy.ndarray   # shape (N, 17)
```

- `rgb` is `uint8 (720, 1280, 3)`, `depth` is `uint16 (720, 1280)` in units of `1/depth_scale` metres (`depth_scale` is 1000.0), and `intrinsics` is the `3x3` camera matrix. Your function receives these four arguments and nothing else: no object identities, poses or meshes come with a graded frame.
- Each returned row is `[score, width, height, depth, R00..R22, tx, ty, tz, object_id]` in the camera frame, in metres, with `R` row-major. Column 0 of `R` is the approach direction and column 1 the closing direction. `object_id` is ignored, and a higher `score` ranks first.
- Put any modules, parameter files or trained models your solver loads beside it in `/home/user/submission/`, and load them relative to `solver.py`. Keep the folder under 200 MB. Nothing outside `/home/user/submission/` is kept. There is no submit step; whatever is there when you stop is graded.

## Environment

- Python: use `/opt/grasp/bin/python` (Python 3.10). It has numpy 1.23.5, scipy, scikit-learn 1.6.1, scikit-image, open3d 0.18, trimesh, opencv-python, pillow, matplotlib, h5py, torch 2.4.1 (CPU) and graspnetAPI 1.2.11. Your solver is graded on exactly this interpreter and these packages; nothing else can be installed. `python3` on `PATH` is a different interpreter without these packages.
- 8 CPUs, 64 GB of memory, no GPU, and no network, both while you work and while your solver is graded.
- You have 4 hours.

## What You Have

- `/data/graspnet/grasp_label/`: dense force-closure annotations for ten objects, subsampled to a tenth of their sampled surface points. Each file gives the annotated points in the object's frame and, for each point, the gripper poses tried there (300 approach directions, 12 in-plane rotations, 4 insertion depths, with the opening width each needs), the friction coefficient at which each one holds (-1 where it does not), and whether it collides with that object in isolation. Collisions in a cluttered scene are not annotated. This is the only grasp supervision you have.
- `/data/graspnet/scenes/`: real RealSense captures of cluttered tabletops from GraspNet-1Billion, eight viewpoints each. Every scene directory holds `realsense/rgb/`, `realsense/depth/`, `realsense/camK.npy`, `realsense/camera_poses.npy`, `realsense/cam0_wrt_table.npy` and `object_id_list.txt`.
  - Twenty development scenes contain the ten annotated objects among others. Their `realsense/meta/` gives, per frame, the depth scale and the camera-frame poses of the annotated objects only (`cls_indexes` is the object id plus one, `poses` is 3x4 per object), which places the annotations in each frame.
  - Twelve practice scenes have RGB-D and camera data only.
- `/data/graspnet/manifest.json`: which scenes and viewpoints each part of the data covers (`roles.train` for the development scenes, `roles.selfcheck` for the practice scenes) and the annotated objects (`label_objects`).
- There are no object meshes, so graspnetAPI's evaluation code is installed but has nothing to evaluate against here.
- `~/starter/solver.py`: the detector you are given. It back-projects the depth image, removes the table plane, estimates surface normals, and proposes grasps that approach sampled points along the inward normal at four fixed rotations with one fixed opening width and insertion depth, discarding any whose fingers or base would pass through the observed point cloud. It ranks the survivors by how flat the surface is around the grasp point. It already satisfies the interface below; copy it to `/home/user/submission/` to start from it.
- `~/practice_check/check.py`: runs whatever is in `/home/user/submission/solver.py` on the practice frames, with the graded run's thread settings, network guard and time limits, and reports, per frame, whether the output meets the contract below and how long the call took. Run it as `/opt/grasp/bin/python ~/practice_check/check.py [--views K]`. Free and unlimited. It does not score grasps.

## How It Is Graded

Your solver is run over 448 hidden frames: 16 viewpoints from each of 28 GraspNet-1Billion scenes whose 26 objects appear nowhere in your data. It runs as your user account in 8 processes at once on the 8 CPUs, with `OMP_NUM_THREADS=1`, the working directory set to `/home/user/submission/`, and no network access (no AF_INET or AF_INET6 sockets at all, so no localhost TCP either). Each process imports `solver.py` once and then gets one frame at a time: the grader hands it the current frame's four arguments only after it has answered the previous frame, and no frame is ever on disk where your code could read it. The frames come in one fixed shuffled order that mixes the scenes, so consecutive calls usually show different scenes; each call has to stand on its own frame.

- Importing `solver.py` must finish within 120 s, and each `predict()` call must return within 10 s, as a pick cell would wait for a frame's grasps on one CPU core. The grader times every call on its own clock, from the moment it starts handing over the frame until the array is back.
- A call that is not back within 10 s scores 0 for that frame, whatever it returns later. Its process is then killed and a fresh process, which imports `solver.py` again, takes the next frame. A process that dies on a frame is replaced the same way. If an import fails or runs over, every frame of the process's block that has not been answered yet scores 0.
- Each process works through a block of 56 frames with a time budget of 56 x 10 s plus 4 x 120 s (its first import and three re-imports). Every restart pays its import out of that budget; once the budget is spent, the rest of the block scores 0. A solver that never runs over never comes near it.
- A frame also scores 0 if `predict()` raises, or if its array is empty, wrongly shaped, longer than 5000 rows, or contains a non-finite value, a negative `width`, a `height` or `depth` outside (0, 0.10] m, or an `R` that is not a rotation (orthonormal with determinant +1, to 1e-4).

Each frame is then scored with graspnetAPI 1.2.11's `eval_grasp`, unmodified, exactly as `GraspNetEval.eval_scene` calls it. `width` is first clipped to at most 0.10 m. The evaluator then applies non-maximum suppression at 0.03 m and 30 degrees, assigns every grasp to the nearest object in the ground-truth scene, keeps each object's ten highest-`score` grasps, and keeps only those whose `score` is at least the 50th highest among what remains. Each kept grasp is placed in the scene with a gripper of 0.01 m finger width, 0.02 m finger height and a 0.02 m base behind the grasp point, using your `width` for the opening and your `depth` for how far the fingers reach past it. It fails if that body collides with any object or the table, or if too few object points lie between the fingers; otherwise it is tested for force closure against the object's mesh at friction coefficients 0.2, 0.4, 0.6, 0.8, 1.0 and 1.2. Sorting the kept grasps by `score`, a frame scores the mean over those six coefficients and over `k` from 1 to 50 of the fraction of successes among the top `k`, where slots beyond the number of kept grasps count as failures.

Your mean AP is the mean of that per-frame AP over all 448 frames, in percent. The graded quantity is how much your mean AP exceeds the mean AP that `~/starter/solver.py`, unchanged, scores on the same 448 frames, floored at 0, measured as a fraction of what a reference detector gains over it. The reference is a privileged detector: during its own run the grader hands it the ground-truth poses and meshes of each frame, the frames in scene order and up to 300 s per call, none of which a submission receives. It runs the evaluator's own collision and force-closure tests on its candidates and keeps the ones that pass, so it marks what a solver could reach with perfect knowledge of the scene, and a solver without that knowledge is not expected to come close to it. Submitting the provided detector unchanged earns nothing; more mean AP always earns more, up to a cap.
