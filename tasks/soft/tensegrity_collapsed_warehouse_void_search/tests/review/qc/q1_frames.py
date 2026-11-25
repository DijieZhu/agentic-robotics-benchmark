"""QC Q1 (frames): a fixed 20-frame sequence on a frozen hidden site, rendered by the task's cell, and the
comparison of two such runs made in two processes.

    q1_frames.py OUT.json [--site h03] [--plate-root DIR] [--frames-npz FRAMES.npz]
    q1_frames.py compare A.json B.json [--out CMP.json]

Run: one in-process cell (tests/voidcell/server.py, mode hidden, the task's grader_config.json render settings and
the hidden time limit) on a frozen hidden package with its pinned seed. The sequence: 20 x (one camera frame, then a
fixed command: a primitive with the true bottom triple, run until idle, or a plain step), so the frames show the robot
at 20 different states and every exposure is a different index. With the plate backend (render_backend
plate_filament) the server needs the installed bundles: --plate-root is the absolute path of a plate root that holds
the hidden sites' bundles (verify.py installs them only inside its own stage; grader_config.json's plate_root is a
path under the stage, so it is not used here). Writes every frame's index, time, the cell's own Frame.sha256 (rgb and
depth), the depth's own sha256 and the robot window's position (the plate backend's 128 x 128 px window around the
robot and its floor shadow; none when the robot is outside the frame), the robot's state hash after the sequence and
the process id; --frames-npz also stores the 20 frames' rgb and depth arrays (a hidden site's picture: keep it in a
mode-700 folder and delete it after the comparison).

Compare (the plate backend's reading of frame determinism): the robot layer inside the window carries the renderer's
per-frame dithering, which follows the process's call history, so frames are compared as: the depth sha256 equal on
every frame; the RGB bytes OUTSIDE the window equal on every frame; INSIDE the window, on every frame, >= 99 % of the
pixels within 8 levels (the largest channel difference); the window positions and the final state hash equal.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
TESTS = os.path.join(TASK, "tests")

SEQ = ("roll_120_120", None, "ccw", "roll_100_140", None, "cw", "roll_140_100", "ccw", None, "roll_120_140",
       "roll_100_120", None, "cw", "roll_140_140", "ccw", None, "roll_100_100", "cw", None, "roll_120_100")
WIN_LEVELS = 8
WIN_SHARE = 0.99


def window_of(srv):
    """(u0, v0, n) of the plate backend's robot window for the last frame, or None."""
    ses = getattr(srv.cam, "session", None)
    last = getattr(ses, "last", None) or {}
    w = last.get("window")
    if w is None:
        return None
    return [int(w[0]), int(w[1]), int(getattr(ses, "win"))]


def run(a):
    sys.path.insert(0, TESTS)
    from voidcell.server import VoidCellServer  # noqa: E402
    from voidcell import client as vclient  # noqa: E402
    import numpy as np
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    lim = json.load(open(os.path.join(TESTS, "hidden", "limits.json")))
    seeds = json.load(open(os.path.join(TESTS, "hidden", "seeds.json")))
    row = [r for r in seeds["sites"] if r["tag"] == a.site][0]
    c = {k: cfg[k] for k in ("limits", "pose_source", "render_backend", "msaa_rgb") if k in cfg}
    c["limits"] = dict(c.get("limits") or {}, time_limit_s=float(lim["time_limit_s"]))
    if a.plate_root:
        if not os.path.isabs(a.plate_root) or not os.path.isdir(a.plate_root):
            raise SystemExit("q1: --plate-root must be an absolute path of an existing folder")
        c["plate_root"] = a.plate_root
    elif c.get("render_backend") == "plate_filament":
        raise SystemExit("q1: the plate backend needs --plate-root (the installed bundles of the hidden sites)")
    srv = VoidCellServer(config=c, mode="hidden", render=True)
    srv.reset(os.path.join(TESTS, "hidden", row["archive"]), seed=int(row["seed"]), kind="hidden")
    st = vclient.connect_handler(srv.handle)
    frames, rgbs, depths = [], [], []
    for cmd in SEQ:
        f = st.get_frame()
        h = hashlib.sha256(f["rgb"].tobytes() + f["depth"].tobytes()).hexdigest()
        frames.append({"index": int(f["index"]), "t": float(f["t"]), "sha256_rgb_depth": h,
                       "sha256_depth": hashlib.sha256(f["depth"].tobytes()).hexdigest(),
                       "cell_sha256": srv.frame_hashes[-1][1], "window": window_of(srv)})
        if a.frames_npz:
            rgbs.append(np.asarray(f["rgb"]).copy())
            depths.append(np.asarray(f["depth"]).copy())
        if cmd is None:
            st.step(1.0)
            continue
        b = srv.robot_truth.bottom()
        if not b["rollable"]:
            st.step(1.0)
            continue
        st.run_primitive(cmd, list(b["nodes"]))
        while st.status()["busy"]:
            st.step(0.5)
        st.step(0.3)
    out = {"site": a.site, "pid": os.getpid(), "n_frames": len(frames), "frames": frames,
           "state_hash": srv.S.state_hash(), "t_end": float(srv.X.t()),
           "render_backend": srv.cam.backend.version() if srv.cam is not None else None,
           "plate_root_given": bool(a.plate_root), "frames_npz": os.path.abspath(a.frames_npz) if a.frames_npz else None,
           "env": {k: os.environ.get(k) for k in ("LP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                  "MUJOCO_GL", "PYOPENGL_PLATFORM", "OPENBLAS_CORETYPE")}}
    srv.close()
    if a.frames_npz:
        np.savez_compressed(a.frames_npz, rgb=np.stack(rgbs), depth=np.stack(depths))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"n": len(frames), "t_end": out["t_end"], "distinct": len({x["sha256_rgb_depth"] for x in frames}),
                      "windows": sum(1 for x in frames if x["window"])}))


def compare(a):
    import numpy as np
    A, B = json.load(open(a.a)), json.load(open(a.b))
    FA, FB = np.load(A["frames_npz"]), np.load(B["frames_npz"])
    rows = []
    for k, (fa, fb) in enumerate(zip(A["frames"], B["frames"])):
        ra, rb = FA["rgb"][k].astype(np.int16), FB["rgb"][k].astype(np.int16)
        inside = np.zeros(ra.shape[:2], bool)
        for w in (fa["window"], fb["window"]):
            if w:
                u0, v0, n = w
                inside[v0:v0 + n, u0:u0 + n] = True
        d = np.abs(ra - rb).max(axis=-1) if ra.ndim == 3 else np.abs(ra - rb)
        n_in = int(inside.sum())
        within = int((d[inside] <= WIN_LEVELS).sum()) if n_in else 0
        rows.append(dict(k=k, same_index=fa["index"] == fb["index"], same_window=fa["window"] == fb["window"],
                         depth_equal=fa["sha256_depth"] == fb["sha256_depth"],
                         depth_array_equal=bool(np.array_equal(FA["depth"][k], FB["depth"][k])),
                         outside_equal=bool((d[~inside] == 0).all()), window_px=n_in,
                         window_within_share=round(within / n_in, 6) if n_in else None,
                         window_max_diff=int(d[inside].max()) if n_in else None,
                         rgb_hash_equal=fa["sha256_rgb_depth"] == fb["sha256_rgb_depth"]))
    ok = (len(rows) == len(SEQ) == A["n_frames"] == B["n_frames"] and A["pid"] != B["pid"] and
          A["state_hash"] == B["state_hash"] and
          all(r["same_index"] and r["same_window"] and r["depth_equal"] and r["depth_array_equal"] and r["outside_equal"]
              and (r["window_within_share"] is None or r["window_within_share"] >= WIN_SHARE) for r in rows))
    res = dict(n=len(rows), processes_differ=A["pid"] != B["pid"], state_equal=A["state_hash"] == B["state_hash"],
               depth_equal=sum(r["depth_equal"] and r["depth_array_equal"] for r in rows),
               outside_equal=sum(r["outside_equal"] for r in rows), windows=sum(1 for r in rows if r["window_px"]),
               window_min_share=min([r["window_within_share"] for r in rows if r["window_within_share"] is not None],
                                    default=None),
               window_max_diff=max([r["window_max_diff"] for r in rows if r["window_max_diff"] is not None], default=None),
               rgb_hash_equal=sum(r["rgb_hash_equal"] for r in rows), distinct_frames=len({f["sha256_rgb_depth"]
                                                                                          for f in A["frames"]}),
               lines=dict(levels=WIN_LEVELS, share=WIN_SHARE), rows=rows, pass_=bool(ok))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    return 0 if ok else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("a")
        ap.add_argument("b")
        ap.add_argument("--out", default=None)
        return compare(ap.parse_args())
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--site", default="h03")
    ap.add_argument("--plate-root", default=None)
    ap.add_argument("--frames-npz", default=None)
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
