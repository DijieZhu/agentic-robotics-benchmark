"""Author-only physical fixtures. No placement or cameras here feed the pilot."""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from check_rescue import World
from oracle_run import aperture_width
from rescue_scene import DECK


def trial(seed, idx, yaw, shift, loaded, photo=None, air_heading=0):
    w = World("rescue",seed,render=photo is not None)
    try:
        col = w.layout.wells[idx][0]
        wx,wy = w.layout.xy((col,4))
        x,y = wx+shift,wy-shift
        z = DECK+1.75
        slot = next(i for i in (0,1) if w.fits(i))
        def aircraft(x,y,z,heading=0.):
            w.command[:] = w.data.qpos[:4] = w.data.ctrl[:] = (x,y,z,heading)
            mujoco.mj_forward(w.model,w.data)
        # Mount via the real dock; place the assembly in a specified test
        # pose, then use the regular latch and normal dynamic descent.
        dx,dy,dz = w.layout.dock(slot)
        aircraft(dx,dy,dz+.206,math.radians(air_heading))
        w.grip(1.)
        assert w.tool == slot
        adr = int(w.model.joint(f"collet{slot}_free").qposadr[0])
        w.data.qpos[adr:adr+3] += [x-dx,y-dy,z-(dz+.206)]
        if loaded:
            a = math.radians(yaw)/2
            w.data.qpos[4:11] = [x,y,z-.31,math.cos(a),0.,0.,math.sin(a)]
        aircraft(x,y,z,math.radians(air_heading))
        if loaded:
            w.grip(1.)
            assert w.held
        measured = None
        if photo is not None:
            for i,im in enumerate(w.images()):
                Image.fromarray(im).save(photo/f"well{idx}-yaw{yaw}-held{int(loaded)}-{i}.png")
                if i==0:
                    try:
                        measured = aperture_width(im.astype(np.int16),1.75)
                    except RuntimeError:
                        if not loaded or yaw!=0:
                            raise  # aligned board can occupy the entire down view
                    if measured is not None:
                        assert abs(measured-2*w.layout.well_half(col))<.08, measured
        for _ in range(4):
            w.act([0.,0.,0.,0.,0.])
        for _ in range(16):
            if w.position[2]<1.55 or w.done:
                break
            w.act([0.,0.,-1.,0.,0.])
        clear = bool(not w.done and w.position[2]<1.55)
        assert clear == (not loaded or idx==1), (idx,yaw,shift,loaded,w.failure,w.impact_geom)
        if not clear:
            assert w.impact_body=="pod" and w.impact_obstacle.startswith("aperture_")
        rotation = w.data.geom_xmat[w.board].reshape(3,3)
        ext = (np.abs(rotation) @ w.model.geom_size[w.board])[:2]
        board_gap = w.layout.well_half(col)-max(ext+abs(w.payload[:2]-[wx,wy]))
        return dict(well=idx,yaw=yaw,air_heading_deg=air_heading,offset_m=shift,loaded=loaded,down=clear,
                    width_m=2*w.layout.well_half(col),image_width_m=measured,
                    board_projected_gap_m=float(board_gap) if loaded else None,
                    signed_envelope_gap_m=w.well_gaps,contact=[w.impact_geom,w.impact_obstacle])
    finally:
        w.close()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder",type=Path)
    ap.add_argument("--air-headings",action="store_true")
    args = ap.parse_args()
    args.folder.mkdir(exist_ok=True,parents=True)
    trials = []
    for idx in (0,1):
        if args.air_headings:
            for heading in (-45,30,90):
                for loaded in (False,True):
                    entry = trial(61907136,idx,heading+35,.08,loaded,air_heading=heading)
                    trials.append(entry)
                    print(json.dumps(entry),flush=True)
            continue
        for yaw in (0,45,90):
            for shift in (0.,.12):
                for loaded in (False,True):
                    entry = trial(61907136,idx,yaw,shift,loaded,
                                  args.folder if yaw in (0,90) and shift==0 else None)
                    trials.append(entry)
                    print(json.dumps(entry),flush=True)
    (args.folder/"dynamic.json").write_text(json.dumps(trials,indent=2))
