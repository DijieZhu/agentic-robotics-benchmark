"""Static submission grading; never imports or executes the submission."""
import json
import math
from pathlib import Path
import jsonschema
import yaml
from procedure import evaluate, passing_designs, price, line_changes

CRITERIA = {"kinematic_replication": .30, "requirements_sheet": .10,
            "derived_requirements": .20, "bom_acceptance": .10, "bom_economy": .30}


def finite_tree(x):
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError("nonfinite number")
    if isinstance(x, dict):
        for v in x.values(): finite_tree(v)
    if isinstance(x, list):
        for v in x: finite_tree(v)


def load_submission(folder, schemas):
    out = {}
    for filename in ("requirements.yaml", "bom.json", "kinematic_sketch.json"):
        p = Path(folder) / filename
        if not p.is_file() or p.is_symlink() or p.stat().st_size > 262144:
            raise ValueError(f"missing, linked, or oversized {filename}")
        text = p.read_text()
        doc = yaml.safe_load(text) if filename.endswith("yaml") else json.loads(text)
        finite_tree(doc)
        jsonschema.validate(doc, schemas[filename])
        out[filename] = doc
    names = [x["id"] for x in out["requirements.yaml"]["facets"]]
    if len(names) != len(set(names)): raise ValueError("duplicate facet id")
    return out


def ramp(error, full, zero):
    return max(0.0, min(1.0, (zero-error)/(zero-full)))


def grade(docs, truth):
    geometry, cat, duty = (truth[k] for k in ("geometry", "catalogue", "duty"))
    sheet, bom, sketch = (docs[k] for k in ("requirements.yaml", "bom.json", "kinematic_sketch.json"))
    metrics = {}
    values = []
    for k, cell in sketch["dimensions_mm"].items():
        err = abs(cell["nominal"]-geometry["dimensions_mm"][k])
        metrics[f"geometry_error_mm:{k}"] = err
        values.append(ramp(err, .5, 3.5))
    for field,label in (('rotor_cant_deg','rotor_cant'),('rotor_radial_tilt_deg','rotor_radial_tilt')):
        for station,cell in sketch[field].items():
            err=abs(cell['nominal']-geometry[field][station])
            metrics[f'geometry_error_deg:{label}_{station}']=err
            values.append(ramp(err,1.,4.))
    spin = sum(sketch["spin"][s] == v for s, v in truth["spin"].items()) / 4
    c1 = .95*sum(values)/len(values)+.05*spin
    by_id = {r["id"]: r for r in sheet["facets"]}
    credit = 0
    for r in truth["facets"]:
        sub = by_id.get(r["id"], {})
        match = sub.get("unit") == r["unit"]
        if isinstance(r["value"], (int, float)):
            v = sub.get("value")
            match &= isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v-r["value"]) <= r["tolerance"]
        else: match &= sub.get("value") == r["value"]
        trace = bool(set(sub.get("sources", [])) & set(r["sources"]))
        credit += float(match and trace)
    c2 = 2*credit / (len(truth["facets"])+len(sheet["facets"]))
    result = evaluate(bom, geometry, cat, duty)
    derived_credit = []
    for k in truth["derived_specs"]:
        expected=result["derived"][k]
        sub = sheet["derived"].get(k)
        error = 1.0
        if sub:
            denom = max(abs(expected), truth["derived_specs"][k]["floor"])
            error = abs(sub["value"]-expected)/denom
            if sub["unit"] != truth["derived_specs"][k]["unit"]: error = 1.0
        metrics[f"derived_relative_error:{k}"] = error
        derived_credit.append(ramp(error, .03, .15))
    c3 = sum(derived_credit)/len(derived_credit)
    good = passing_designs(geometry, cat, duty)
    if not good: raise RuntimeError("fixture has no passing configuration")
    optimum = min(r["price"] for _, r in good)
    if not math.isclose(optimum, truth["optimum_report"]["price"], abs_tol=1e-8):
        raise RuntimeError("optimum replay mismatch")
    # A failed line is replaced; each changed motor station or shared procurement line incurs ECO.
    effective = min(r["price"] + duty["eco_usd"]*line_changes(bom, b) for b, r in good)
    c5 = ramp(max(0, effective-optimum), 0, duty["eco_usd"])
    criteria = dict(zip(CRITERIA, (c1, c2, c3, float(result["pass"]), c5)))
    metrics.update(criteria)
    metrics.update({"cost_usd": price(bom, cat), "effective_cost_usd": effective,
                    "optimum_cost_usd": optimum, "gate_g0_structure": 1.0})
    metrics.update({f"acceptance:{k}": float(v) for k,v in result["checks"].items()})
    metrics.update({f"thrust_utilization:{s}": v for s,v in zip(truth["spin"],result["thrust_utilization"])})
    value = sum(CRITERIA[k]*v for k,v in criteria.items())
    return value, metrics
