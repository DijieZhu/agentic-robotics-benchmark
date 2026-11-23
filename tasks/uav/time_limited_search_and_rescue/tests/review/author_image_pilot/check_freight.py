"""Private paired controls for load-aware routing, using a shared physical pickup."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from check_litter import OnDemandWorld, Referee
from check_rescue import check_known_returns
from rescue_scene import layout_for
import oracle_run as pilot


def sites():
    cases = {}
    for counter in range(200):
        seed = int.from_bytes(hashlib.sha256(f"civilian39-freight-v1:{counter}".encode()).digest()[:7],"big")
        layout = layout_for(seed>>1)
        key = (seed&1,(seed>>1)&1,layout.flange>.06)
        cases.setdefault(key,dict(seed=seed,draw=counter,stratum=key,target=layout.target))
        if len(cases)==8:
            return list(cases.values())
    raise AssertionError("incomplete strata")


def write(path, data):
    path.write_text(json.dumps(data,indent=2)+"\n")


def receipt(w,rec,p,folder,mode,error,started):
    events = {"sled_found","flange_survey","depot_choice","sled_reacquired","pickup_attempt",
              "return_route","entry_jambs","exit_jambs","exhausted"}
    steps = {e["event"]:e["frame"] for e in p.trace if e["event"] in events}
    mouths = [e for e in p.trace if e["event"]=="aperture_measurement"]
    route = [e for e in p.trace if e["event"]=="return_route"]
    steps.update({f"mouth-{i}":e["frame"] for i,e in enumerate(mouths)})
    steps["stop"] = w.steps
    rec.observe()["images"][0]
    for event,step in steps.items():
        if rec.frames[step].frames is not None:
            for cam in range(2):
                rec.export(step,folder/f"{mode}-{event}-{cam}.png",cam)
    result = dict(mode=mode,wall_s=round(time.monotonic()-started,2),error=error,
                  progress=steps,grasp=rec.grasp_frame,apertures=mouths,return_route=route,
                  target=w.layout.target,flange=w.layout.flange,trim=w.bias_sign,
                  contact=[w.impact_geom,w.impact_obstacle],failure=w.failure,
                  **w.diagnostics(),**w.metrics())
    write(folder/f"{mode}-outcome.json",result)
    write(folder/f"{mode}-trace.json",p.trace)
    (folder/f"{mode}-actions.json").write_text(json.dumps(rec.actions))
    print(json.dumps({k:result[k] for k in ("mode","steps","success","grasp","failure",
                                            "contact","error","wall_s")}),flush=True)
    return result


def paired(seed, folder):
    folder.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    w = OnDemandWorld("rescue",seed)
    rec = Referee(w,capture=True)
    p = pilot.Pilot(rec)
    loaded_return = pilot.loaded_return
    neg_actions = []
    checkpoint = {}

    def fork(p,bay,trail,memory,cargo,replay,hold):
        assert w.held and "lifted" in w.progress
        checkpoint["step"] = w.steps
        def state_digest():
            arrays = (w.data.qpos,w.data.qvel,w.data.qacc_warmstart,w.data.ctrl,
                      w.data.eq_active,w.model.eq_data,w.command)
            return hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
        checkpoint["state_sha256"] = state_digest()
        checkpoint["down_png_sha256"] = hashlib.sha256(p.obs["images"][0].encode()).hexdigest()
        # Both continuations start with the same simulated integrator/weld
        # state, and copies of the same independently obtained RGB pilot memory.
        data = copy.copy(w.data)
        eq = w.model.eq_data.copy()
        state = copy.deepcopy({k:v for k,v in vars(w).items()
                               if k not in ("model","data","renderer","_observation")})
        records = rec.actions.copy(),rec.frames.copy(),rec.poses.copy()
        other = copy.deepcopy(p,{id(rec):rec,id(p.obs):p.obs})
        error = None
        try:
            loaded_return(other,bay,trail,copy.deepcopy(memory),cargo,True,hold)
        except RuntimeError as exc:
            error = str(exc)
        receipt(w,rec,other,folder,"replay",error,started)
        neg_actions.extend(rec.actions)
        w.__dict__.update(state)
        w.data,w._observation = data,None
        w.model.eq_data[:] = eq
        rec.actions,rec.frames,rec.poses = records
        p.obs = rec.observe()
        assert w.steps==checkpoint["step"] and w.held and state_digest()==checkpoint["state_sha256"]
        assert hashlib.sha256(p.obs["images"][0].encode()).hexdigest()==checkpoint["down_png_sha256"]
        return loaded_return(p,bay,trail,memory,cargo,False,hold)

    try:
        error = None
        with patch.object(pilot,"Pilot",return_value=p),patch.object(pilot,"loaded_return",fork):
            try:
                pilot.run(rec)
            except RuntimeError as exc:
                error = str(exc)
        pos = receipt(w,rec,p,folder,"aware",error,started)
        common = next((i for i,(a,b) in enumerate(zip(rec.actions,neg_actions)) if a!=b),
                      min(len(rec.actions),len(neg_actions)))
        write(folder/"pair.json",dict(seed=seed,identical_actions=common,**checkpoint))
        if checkpoint:
            assert common >= checkpoint["step"] > rec.grasp_frame
        if pos["success"]:
            check_known_returns(p.trace)
        return pos
    finally:
        w.close()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder",type=Path)
    ap.add_argument("--index",type=int)
    args = ap.parse_args()
    manifest = args.folder/"sites.json"
    if args.index is None:
        with manifest.open("x") as f:
            json.dump(sites(),f,indent=2)
    else:
        site = json.loads(manifest.read_text())[args.index]
        result = paired(site["seed"],args.folder/str(site["seed"]))
        assert result["success"] and not result["failure"] and not result["error"]
