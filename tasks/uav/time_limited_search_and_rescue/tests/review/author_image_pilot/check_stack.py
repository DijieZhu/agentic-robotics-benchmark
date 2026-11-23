"""Matched stack-memory controls. All world access is on the referee side."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from check_litter import OnDemandWorld, Referee
from check_rescue import World, check_known_returns, run
from rescue_scene import DECK, distances, edge


def flight(seed, planar, folder, full=False):
    w = (World if full else OnDemandWorld)("rescue",seed)
    rec = Referee(w,capture=True)
    name = "xy" if planar else "xyz"
    try:
        error = None
        try:
            p = run(rec,planar=planar)
        except RuntimeError as exc:
            error = str(exc)
            tb = exc.__traceback__
            while tb.tb_frame.f_code is not run.__code__:
                tb = tb.tb_next
            p = tb.tb_frame.f_locals["p"]
        if w.metrics()["success"]:
            check_known_returns(p.trace)
        def actual(frame):
            x,y,z = rec.poses[frame]
            return int(np.argmin(abs(w.columns-x))),int(np.argmin(abs(w.rows-y))),int(z>=DECK)
        images, scans, aliases = {}, {}, []
        for e in p.trace:
            event, frame = e["event"],e["frame"] if "frame" in e else None
            if event in ("sled_found","flange_survey","depot_choice","sled_reacquired","litter_axis",
                         "pickup_attempt","entry_jambs","exit_jambs","exhausted"):
                images[event] = frame
            if event == "well_flight":
                for k,f in (("approach",e["before"]),("inside",e["middle"]),("other",frame)):
                    images.setdefault(("loaded_" if e["loaded"] else "")+"well_"+k,f)
            if event != "room_scan":
                continue
            cell = actual(frame)
            e["referee_pose"] = rec.poses[frame].tolist()
            scans.setdefault(cell,(frame,e["open"]))
            for direction,previous in e["skipped"]:
                other = actual(previous)
                there = tuple(a+b for a,b in zip(cell,direction))
                if (other[:2]==cell[:2] and other[2]!=cell[2] and
                        edge(cell,there) in w.layout.halls and not direction[2]):
                    aliases.append(dict(frame=frame,prior=previous,cell=cell,other=other,
                                        direction=direction,loaded=e["loaded"]))
        overlaps = []
        for (x,y,k),(frame,ports) in scans.items():
            if k or (x,y,1) not in scans:
                continue
            other,above = scans[x,y,1]
            different = {tuple(d[:2]) for d in ports if not d[2]} != {tuple(d[:2]) for d in above if not d[2]}
            overlaps.append((different,x,y,frame,other))
        if overlaps:
            _,_,_,images["same_xy_lower"],images["same_xy_upper"] = max(overlaps)
        if aliases:
            images.update(alias_prior=aliases[0]["prior"],alias_now=aliases[0]["frame"])
        images["stop"] = w.steps
        for label,step in images.items():
            pair = rec.frames[step]
            if not full and step != w.steps and pair.frames is None:
                continue
            for camera in (0,1):
                (folder/f"{name}-{label}-{camera}.png").write_bytes(base64.b64decode(pair[camera]))
        result = dict(seed=str(seed),mode=name,full=full,error=error,failure=w.failure,
                      impact=w.impact_geom,structure=w.impact_obstacle,target=w.layout.target,
                      wide=w.layout.flange>.06,trim=w.bias_sign,stages=images,grasp=rec.grasp_frame,
                      shortest=distances((w.layout.inlet,1,0),w.layout.halls)[w.layout.target],
                      overlap_scans=overlaps,physical_aliases=aliases,**w.diagnostics(),**w.metrics())
        for label,content in (("outcome",result),("trace",p.trace),("actions",rec.actions)):
            (folder/f"{name}-{label}.json").write_text(json.dumps(content,indent=None if label=="actions" else 2))
        print(json.dumps(result),flush=True)
        return result,rec.actions
    finally:
        w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed",type=int)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--full",action="store_true")
    ap.add_argument("--paired",action="store_true")
    args = ap.parse_args()
    args.folder.mkdir(exist_ok=True,parents=True)
    pos,a = flight(args.seed,False,args.folder,args.full)
    if args.paired:
        neg,b = flight(args.seed,True,args.folder,args.full)
        common = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        print(json.dumps(dict(seed=str(args.seed),identical_prefix=common)),flush=True)
        if neg["physical_aliases"]:
            assert common >= neg["physical_aliases"][0]["frame"] or neg["success"]
    assert pos["success"] and not pos["error"] and not pos["failure"], pos
    assert pos["well_crossings"] and pos["overlap_xy"] >= 2 and not pos["fit_misses"]
