"""One frame of each agent camera at the program's start (the oracle must exercise the camera path and
its cost must be in the budget): get_camera("top", depth) and get_camera("wrist", depth) through the cell's render
layer, a frame that does not come back fails loudly (required) or is logged (optional: a cell built without a
renderer, the suite's short test). The decoded frames are written as PNG when HREF_FRAMES_DIR names a folder."""
import base64
import io
import os
import time

from .errors import ReferenceFailure


def first_frames(session, log, required=True, depth=True, names=("top", "wrist")):
    out = {}
    folder = os.environ.get("HREF_FRAMES_DIR")
    for name in names:
        t0 = time.time()
        try:
            r = session.camera(name, depth=depth)
        except Exception as e:  # noqa: BLE001 - the client raises CellError on an error reply
            r = {"error": str(e)}
        dt = time.time() - t0
        ok = isinstance(r, dict) and bool(r.get("ok"))
        rec = {"ok": ok, "wall_s": round(dt, 3)}
        if ok:
            rec.update({"width": r.get("width"), "height": r.get("height"), "frame_index": r.get("frame_index"),
                        "depth": bool(r.get("sensor_depth_u16_mm_b64"))})
            if folder:
                os.makedirs(folder, exist_ok=True)
                for key, fn in (("rgb_png_b64", "%s_rgb.png" % name), ("sensor_depth_u16_mm_b64", "%s_depth_mm.png" % name)):
                    if r.get(key):
                        with open(os.path.join(folder, fn), "wb") as f:
                            f.write(base64.b64decode(r[key]))
                rec["saved"] = folder
        else:
            rec["error"] = (r or {}).get("error") if isinstance(r, dict) else repr(r)
        log.write("camera", camera=name, **rec)
        out[name] = rec
        if required and not ok:
            raise ReferenceFailure(log.stack[-1] if log.stack else "cameras", "camera %s gave no frame" % name, rec)
    return out
