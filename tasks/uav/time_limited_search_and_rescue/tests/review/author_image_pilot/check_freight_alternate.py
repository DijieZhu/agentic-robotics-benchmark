"""Positive control: try the second observed well first, then replay its route."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

from check_freight import write
from check_litter import Referee
from check_rescue import World, check_known_returns
import oracle_run


def alternate(seed,out):
    out.mkdir(parents=True,exist_ok=False)
    first = None
    explore = oracle_run.explore
    def other_first(p,start,planar,cargo=None,memory=None,**kwargs):
        def skip_first(at,step):
            nonlocal first
            if not step[2]:
                return True
            if first is None:
                first = at[:2]
            return at[:2]!=first
        if cargo is None:
            kwargs["allowed"] = skip_first
        return explore(p,start,planar,cargo,memory,**kwargs)

    started = time.monotonic()
    w = World("rescue",seed)  # both cameras on every action, including all holds
    rec = Referee(w,capture=True)
    try:
        with patch.object(oracle_run,"explore",other_first):
            p = oracle_run.run(rec,replay=True)
        write(out/"trace.json",p.trace)
        write(out/"actions.json",rec.actions)
        result = dict(seed=seed,skipped_first=first,full_render=True,
                      elapsed_s=round(time.monotonic()-started,2),failure=w.failure,
                      aperture=[v for v in p.trace if v["event"]=="aperture_measurement"],
                      reuse=[v for v in p.trace if v["event"]=="return_route"],
                      **w.diagnostics(),**w.metrics())
        write(out/"outcome.json",result)
        for event in ("aperture_measurement","pickup_attempt","return_route","exit_jambs"):
            e = next((e for e in p.trace if e["event"]==event),None)
            if e:
                for i in (0,1):
                    rec.export(e["frame"],out/f"{event}-{i}.png",i)
        assert result["success"] and len(result["aperture"])==1 and result["reuse"][0]["reuse"]
        assert all(abs(x-w.well_x[1])<.3 for _,_,_,x,_ in w.well_events)
        check_known_returns(p.trace)
    finally:
        w.close()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed",type=int)
    ap.add_argument("folder",type=Path)
    args = ap.parse_args()
    alternate(args.seed,args.folder)
