"""Geometry-only referee: clearance, opaque slabs and actual flights into a ceiling."""
import argparse
import json
import math

import mujoco
import numpy as np
from PIL import Image

from check_rescue import World
from rescue_scene import DECK, SLAB, distances, edge, layout_for


def topology(l):
    start = (l.inlet,1,0)
    d = distances(start,l.halls)
    ground = {e for e in l.halls if e[0][2]==e[1][2]==0}
    assert len(d)==30 and len(distances(start,ground))==15 and l.target[2]==1
    assert {k for x,y,k in l.rooms}=={0,1}
    assert all((x,y,1-k) in l.rooms for x,y,k in l.rooms)
    vertical = {e for e in l.halls if e[0][2]!=e[1][2]}
    assert {a[:2] for a,b in vertical}==set(l.wells) and len(vertical)==2
    # Either well alone supplies a complete path, including a freight-only
    # round trip. No sampled site requires using the small service opening.
    for link in vertical:
        paths = distances(start,l.halls-{link})
        assert l.target in paths and (l.outlet,1,0) in paths
    assert l.target[:2] not in l.wells
    return d[l.target]


def via_each(l):
    result = {}
    for i,k in enumerate(("service","freight")):
        other = l.wells[1-i]
        halls = l.halls-{edge((*other,0),(*other,1))}
        foot = distances((l.inlet,1,0),halls)
        carry = distances(l.target,halls)
        result[k] = dict(inlet_to_upper=foot[(*l.wells[i],1)],inlet_to_sled=foot[l.target],
                         sled_to_cargo=carry[(l.outlet,1,0)])
    return result


def check(seed,out,paired=None):
    w = World("rescue",seed,render=out is not None)
    try:
        l = w.layout
        assert set(w.observe())=={"images","frame_id","steps_remaining","done"}
        assert w.metrics()["rescue_progress"]==0
        assert all(np.isfinite(float(v)) for v in w.metrics().values())
        shortest = topology(l)
        sign = 1 if l.outlet==4 else -1
        row = 3
        x,y = l.xy((2,row))
        def ray(start,direction):
            gid = np.array([-1],dtype=np.int32)
            dist = mujoco.mj_ray(w.model,w.data,np.array(start,dtype=float),
                                 np.array(direction,dtype=float),None,1,-1,gid)
            return float(dist),w.model.geom(int(gid[0])).name if gid[0]>=0 else None
        # Off the central HOIST WELL floor sign, before the threshold apron.
        below = ray((x,y+.20,1.30),(0,0,1))
        above = ray((x,y+.20,DECK+1.30),(0,0,-1))
        blocked = ray((x,y,1.30),(sign,0,0))
        passage = ray((x,y,DECK+1.30),(sign,0,0))
        assert below[1].startswith("slab") and above[1].startswith("slab")
        assert abs(below[0]-(DECK-SLAB-1.30))<.001 and abs(above[0]-1.30)<.001
        assert blocked[0]<1.5 and passage[0]>2.4,(blocked,passage)
        col,_ = l.wells[1]
        wx,wy = l.xy((col,4))
        through = ray((wx,wy,1.30),(0,0,1))
        assert through[1].startswith("roof") and through[0]>3.5
        skipped_rays = []
        if paired is not None:
            record = json.loads((paired/str(seed)/"xy-outcome.json").read_text())
            for skip in record["physical_aliases"]:
                cx,cy,level = skip["cell"]
                direction = skip["direction"]
                pose = (*l.xy((cx,cy)),level*DECK+1.30)
                clear,obstacle = ray(pose,direction)
                assert clear>2.4,(seed,skip,clear,obstacle)
                skipped_rays.append(dict(frame=skip["frame"],cell=skip["cell"],port=direction,
                                         unobstructed_m=clear,next_solid=obstacle))
        # Proximity in XY is not evidence of seeing through the slab.
        for level in (1-l.target[2],l.target[2]):
            w.data.qpos[:4] = (*l.pod[:2],level*DECK+1.30,0.)
            mujoco.mj_forward(w.model,w.data)
            w.observation_progress()
            assert ("located" in w.progress) == (level == l.target[2])
        boards = w.solids[np.array(w.kinds)=="slab"]
        def gap(offset):
            smallest = 2.
            for angle in np.linspace(0,np.pi,25):
                w.data.qpos[4:11] = (wx+offset,wy-offset,DECK-SLAB/2,math.cos(angle/2),0,0,math.sin(angle/2))
                mujoco.mj_forward(w.model,w.data)
                smallest = min(smallest,*(mujoco.mj_geomDistance(w.model,w.data,w.board,g,2.,None) for g in boards))
            return smallest
        centered,drifted = gap(0.),gap(.12)
        assert centered>.23 and drifted>.11
        def place(x,y,z,yaw):
            w.command[:] = (x,y,z,yaw)
            w.data.qpos[:4] = w.data.ctrl[:] = w.command
            mujoco.mj_forward(w.model,w.data)
        if out is not None:
            for label,px,py,z,yaw in (("lower",x,y,1.30,-sign*np.pi/2),
                                      ("upper",x,y,DECK+1.30,-sign*np.pi/2),
                                      ("well",wx,wy,DECK+.65,0)):
                place(px,py,z,yaw)
                for i,pixels in enumerate(w.images()):
                    Image.fromarray(pixels).save(out/f"{seed}-{label}-{i}.png")
        place(x,y,1.30,0.)
        for _ in range(12):
            if w.act([0.,0.,1.,0.,0.])["done"]:
                break
        assert w.failure=="slab" and w.impact_body=="uav"
        return dict(seed=str(seed),shortest=shortest,via_each=via_each(l),wells=l.wells,
                    loading=l.loading,target=l.target,bridge=(2,row),lower_blocked=blocked,
                    upper_open=passage,ceiling_above=below,floor_below=above,open_well=through,
                    centered_board_gap_m=centered,offset_12cm_gap_m=drifted,
                    real_ceiling_contact=(w.failure,w.impact_geom,w.impact_obstacle),
                    opposite_storey_located=False,same_storey_located=True,
                    skipped_open_door_rays=skipped_rays)
    finally:
        w.close()


if __name__=="__main__":
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument("seeds",nargs="+",type=int)
    parser.add_argument("--images",type=Path)
    parser.add_argument("--paired",type=Path,help="ray-check recorded cross-storey skips from this run")
    args=parser.parse_args()
    mins=[topology(layout_for(i)) for i in range(512)]
    print(json.dumps(dict(generated_layouts=len(mins),shortest_range=[min(mins),max(mins)])),flush=True)
    for i,seed in enumerate(args.seeds):
        print(json.dumps(check(seed,args.images if i==0 else None,args.paired)),flush=True)
