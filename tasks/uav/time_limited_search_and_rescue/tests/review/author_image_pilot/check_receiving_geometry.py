"""Independent real camera and physical receiving fixtures, not pilot inputs."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from check_rescue import World
from oracle_run import dispatch_arrow,station_view
from rescue_scene import layout_for


def sites():
    result = []
    for mirrored in (0,1):
        for i in range(100):
            raw = hashlib.sha256(f"civilian39-receiving-v1:{mirrored}:{i}".encode()).digest()
            base = (int.from_bytes(raw[:7],"big")<<3) + 3*mirrored
            site = layout_for(base>>1)
            if (site.flange>.06)==bool(mirrored):
                for bit in (0,1):
                    result.append(dict(seed=base+4*bit,mirror=mirrored,trim=2*mirrored-1,
                                       draw=i,wide=bool(mirrored),side=2*bit-1))
                break
    assert len(result)==4
    return result


def audit(sites,out):
    pairs = []
    for left,right in zip(sites[::2],sites[1::2]):
        a,b = (layout_for(s["seed"]>>1) for s in (left,right))
        assert right["seed"]==left["seed"]^4 and a.loading[1]==-1 and b.loading[1]==1
        neutral = (a.loading[0],0,a.loading[2])
        assert replace(a,loading=neutral)==replace(b,loading=(b.loading[0],0,b.loading[2]))
        record = asdict(replace(a,loading=neutral))
        record["halls"] = sorted(record["halls"])
        raw = json.dumps(record,sort_keys=True,separators=(",",":"))
        pairs.append(dict(seeds=[left["seed"],right["seed"]],shared_layout_sha256=hashlib.sha256(raw.encode()).hexdigest()))
    (out/"layout-audit.json").write_text(json.dumps(pairs,indent=2))


def exercise(s,out):
    w = World("rescue",s["seed"])
    try:
        cx,cy = w.layout.pocket
        side,offset = w.layout.loading[1:]
        assert side==s["side"] and w.metrics()["rescue_progress"]==0
        def photo(name):
            pair = w.images()
            for i,im in enumerate(pair):
                Image.fromarray(im).save(out/f"{name}-{i}.png")
            return pair[1].astype(np.int16)
        def place(x,y,z,yaw):
            w.command[:] = w.data.qpos[:4] = w.data.ctrl[:] = (x,y,z,yaw)
            mujoco.mj_forward(w.model,w.data)
        early = photo("dispatch")
        reading = dispatch_arrow(early)
        rows = []
        for dx in (-1,1):
            lane = cy+dx*offset
            place(cx,lane,1.65,-dx*math.pi/2)
            view = photo(f"approach{dx}")
            state = station_view(view)
            gray = np.repeat(view.mean(axis=2,keepdims=True),3,axis=2)
            gid = np.array([-1],dtype=np.int32)
            dist = mujoco.mj_ray(w.model,w.data,np.array([cx+dx*.45,lane,1.2]),np.array([float(dx),0,0]),
                                 None,1,-1,gid)
            # Actual board centered lengthwise in the throat, not a radius tag.
            w.data.qpos[4:11] = (cx+dx*2.10,lane,1.2,2**-.5,0.,0.,2**-.5)
            mujoco.mj_forward(w.model,w.data)
            gap = min(mujoco.mj_geomDistance(w.model,w.data,w.board,g,1.,None) for g in w.solids)
            rows.append(dict(side=dx,reading=state,gray_reading=station_view(gray),ray_m=dist,
                             ray_geom=w.model.geom(int(gid[0])).name,board_clearance_m=gap))
            # Move only the test board to floor level at that station. A closed
            # station has no floor-level mat even behind its shutter.
            w.data.qpos[4:7] = (*w.layout.pad_at(dx),.083)
            mujoco.mj_forward(w.model,w.data)
            rows[-1]["supported_centered"] = w.supported_on_mat()
            w.data.qpos[4] += .31
            mujoco.mj_forward(w.model,w.data)
            rows[-1]["supported_off_center"] = w.supported_on_mat()
        result = dict(seed=s["seed"],dispatch=reading,
                      gray_dispatch=dispatch_arrow(np.repeat(early.mean(2,keepdims=True),3,axis=2)),doors=rows)
        (out/"geometry.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        assert reading==result["gray_dispatch"]==side
        for r in rows:
            ok = r["side"]==side
            assert r["reading"]==("open" if ok else "closed") and (ok or r["gray_reading"]=="closed")
            assert (r["board_clearance_m"]>.30 if ok else r["board_clearance_m"]<=0.)
            assert (r["ray_m"]>3 if ok else r["ray_geom"].startswith("receiving_shutter"))
            assert r["supported_centered"]==ok and not r["supported_off_center"]
    finally:
        w.close()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out",type=Path)
    ap.add_argument("--geometry",action="store_true")
    ap.add_argument("--layout-audit",action="store_true")
    args = ap.parse_args()
    file = args.out/"sites.json"
    if args.layout_audit:
        audit(json.loads(file.read_text()),args.out)
    elif not args.geometry:
        with file.open("x") as f:
            json.dump(sites(),f,indent=2)
    else:
        for s in json.loads(file.read_text()):
            out = args.out/("geometry-"+str(s["seed"]))
            out.mkdir()
            exercise(s,out)
