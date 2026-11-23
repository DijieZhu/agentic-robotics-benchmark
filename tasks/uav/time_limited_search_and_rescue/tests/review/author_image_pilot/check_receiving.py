"""Private matched receiving controls. Only the referee sees the world or seed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from check_litter import OnDemandWorld,Referee
from check_rescue import check_known_returns
import oracle_run as pilot


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2)+"\n")


def verify_saved(site,out):
    read = lambda name: json.loads((out/name).read_text())
    pair = read("pair.json")
    r,f = read("site-outcome.json"),read("fixed-outcome.json")
    a = read("site-actions.json")
    assert len(a)==r["steps"] and r["success"] and not r["error"] and not r["failure"]
    assert r["dispatch"]==site["side"] and r["inspections"][-1]["side"]==site["side"]
    assert r["inspections"][-1]["state"]=="open" and r["settled_mat_margin_m"]>=0
    check_known_returns(read("site-trace.json"))
    assert len(pair["state_sha256"])==len(pair["rgb_sha256"])==64
    for mode,n in pair["identical_actions"].items():
        b = read(mode+"-actions.json")
        assert n==next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        assert n>=r["stage"]["entry_jambs"]>pair["frame"]>r["grasp"]
    if site["side"]>0:
        assert f["success"] and not f["error"] and read("fixed-actions.json")==a
    else:
        assert f["contact"]==["litter_board","receiving_shutter_1"] and not f["error"] and f["lifted"]
        l = read("local-outcome.json")
        assert l["success"] and not l["failure"] and not l["error"]
        assert [(e["side"],e["state"]) for e in l["inspections"]]==[(1,"closed"),(-1,"open")]
    return dict(seed=site["seed"],delivered=r["steps"],verified=True)


def mission(site,out):
    out.mkdir()
    start = time.monotonic()
    w = OnDemandWorld("rescue",site["seed"])
    rec = Referee(w,capture=True)
    p = pilot.Pilot(rec)
    original = pilot.unload
    outcomes,tapes,fork = {},{},{}

    def save(mode,controller,error=None):
        events = {"dispatch_view","sled_found","flange_survey","depot_choice","sled_reacquired",
                  "pickup_attempt","entry_jambs","exit_jambs"}
        frames = {e["event"]:e["frame"] for e in controller.trace if e["event"] in events}
        seen = [e for e in controller.trace if e["event"]=="station_inspection"]
        frames.update({f"door-{i}":s["frame"] for i,s in enumerate(seen)})
        frames["stop"] = w.steps
        rec.observe()["images"][0]
        for k,f in frames.items():
            if rec.frames[f].frames is not None:
                for cam in (0,1):
                    rec.export(f,out/f"{mode}-{k}-{cam}.png",cam)
        r = dict(mode=mode,seed=site["seed"],error=error,failure=w.failure,
                 contact=[w.impact_geom,w.impact_obstacle],grasp=rec.grasp_frame,
                 dispatch=controller.trace[0].get("side"),inspections=seen,stage=frames,
                 rest_at=w.payload.tolist(),actual_deck=w.layout.receiving_pad.tolist(),
                 elapsed_with_controls_s=round(time.monotonic()-start,2),**w.metrics())
        dump(out/f"{mode}-outcome.json",r)
        dump(out/f"{mode}-trace.json",controller.trace)
        (out/f"{mode}-actions.json").write_text(json.dumps(rec.actions))
        outcomes[mode],tapes[mode] = r,rec.actions.copy()
        print(json.dumps({k:r[k] for k in ("mode","steps","success","failure","contact","error","inspections")}),
              flush=True)

    def branch(p,at,d,hold):
        assert w.held and "lifted" in w.progress
        fork["frame"] = w.steps
        def digest():
            return hashlib.sha256(b"".join(a.tobytes() for a in (
                w.data.qpos,w.data.qvel,w.data.qacc_warmstart,w.data.ctrl,w.data.eq_active,w.model.eq_data))).hexdigest()
        fork["state_sha256"] = digest()
        fork["rgb_sha256"] = hashlib.sha256(p.obs["images"][0].encode()).hexdigest()
        physical = copy.copy(w.data)
        eq = w.model.eq_data.copy()
        state = copy.deepcopy({k:v for k,v in vars(w).items()
                               if k not in ("data","model","renderer","_observation")})
        entries = rec.actions.copy(),rec.frames.copy(),rec.poses.copy()
        # Missing early information is not an automatic failure: on each
        # negative-side incident the local-only pilot inspects + first.
        for mode in (["fixed","local"] if site["side"]<0 else ["fixed"]):
            alt = copy.deepcopy(p,{id(rec):rec,id(p.obs):p.obs})
            alt.fixed_side = 1 if mode=="fixed" else None
            if mode=="local":
                alt.receiving_hint = None
            error = None
            try:
                original(alt,at,d,hold)
            except RuntimeError as exc:
                error = str(exc)
            save(mode,alt,error)
            w.__dict__.update(copy.deepcopy(state))
            w.data,w._observation = copy.copy(physical),None
            w.model.eq_data[:] = eq
            rec.actions,rec.frames,rec.poses = (v.copy() for v in entries)
            p.obs = rec.observe()
            assert digest()==fork["state_sha256"] and w.held and w.steps==fork["frame"]
            assert hashlib.sha256(p.obs["images"][0].encode()).hexdigest()==fork["rgb_sha256"]
        return original(p,at,d,hold)

    try:
        error = None
        with patch.object(pilot,"Pilot",return_value=p),patch.object(pilot,"unload",branch):
            try:
                pilot.run(rec)
            except RuntimeError as exc:
                error = str(exc)
        save("site",p,error)
        common = {}
        for mode,tape in tapes.items():
            if mode!="site":
                a = tapes["site"]
                common[mode] = next((i for i,(u,v) in enumerate(zip(a,tape)) if u!=v),min(len(a),len(tape)))
        dump(out/"pair.json",dict(**fork,identical_actions=common))
        r = outcomes["site"]
        assert r["success"] and not r["failure"] and not error and r["dispatch"]==site["side"],r
        assert r["inspections"][-1]["state"]=="open" and r["inspections"][-1]["side"]==site["side"]
        assert all(i>=r["stage"]["entry_jambs"]>fork["frame"]>r["grasp"] for i in common.values())
        check_known_returns(p.trace)
        f = outcomes["fixed"]
        if site["side"]>0:
            assert f["success"] and tapes["site"]==tapes["fixed"]
        else:
            assert not f["success"] and f["contact"][1]=="receiving_shutter_1" and f["lifted"] and not f["error"]
            l = outcomes["local"]
            assert l["success"] and not l["failure"] and [x["state"] for x in l["inspections"]]==["closed","open"]
    finally:
        w.close()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    ap.add_argument("index")
    args = ap.parse_args()
    sites = json.loads((args.root/"sites.json").read_text())
    if args.index=="verify":
        rows = [verify_saved(s,args.root/str(s["seed"])) for s in sites]
        dump(args.root/"screen-audit.json",rows)
        print(json.dumps(rows))
    else:
        site = sites[int(args.index)]
        mission(site,args.root/str(site["seed"]))
