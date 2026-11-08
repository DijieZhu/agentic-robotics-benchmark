You are the requirements engineer for eight upgraded replicas of a guarded indoor research quadrotor.
The reference vehicle has left the lab. Recover its as-built geometry from the fixed photo survey, reconcile the
customer brief, build log, vendor pages and incoming-lot reports, then deliver a traceable specification and the
least-cost propulsion/power BOM that meets the prescribed acceptance procedure with the new payload.

Read `brief.md`, `build_log.md`, `submission_contract.md`, `quote.csv`, and the photos/document scans in your home.
`source_notice.md` identifies the open hardware source and the synthetic fixture boundary. There is no network or
practice vehicle. The CAD, camera calibration, numeric vendor catalogue and measurements of this unit are unavailable.
You may use the documented vendor outlines and physical scale references in the photographs.

Write exactly these deliverables under `/home/user/submission/`:

* `requirements.yaml`: the traced requirement facets and derived sizing quantities, with values, units and sources.
* `bom.json`: one motor at each of FL/FR/RL/RR, one common propeller type (two CW and two CCW), one four-channel ESC,
  and one battery. Quantities are implicit; prices are per vehicle.
* `kinematic_sketch.json`: dimensions and individual motor angles with tolerances, their evidence sources, and propeller spin directions.

Use the JSON schemas in `schemas/`. `/opt/venv/bin/python3 schema_check.py submission` checks structure only.
`procedure.py` is the exact public acceptance calculation, parameterized by your reconstructed geometry, catalogue
and duty. Its input keys and units are documented in `submission_contract.md`. It contains no measurements.
The Python environment is `/opt/venv/bin/python3` (NumPy, SciPy, Pillow, PyYAML and jsonschema).

Grading compares your sketch to the actual unit (30%), the traced requirement sheet (10%), and the derived loads
for YOUR selected BOM on the actual unit (20%). The BOM must pass all acceptance checks (10%). Economy (30%) uses
the cheapest passing design on the actual unit, including the stated engineering-change procedure for a wrong BOM.
The public contract specifies all tolerances and scoring rules. Missing, malformed or nonfinite submissions receive
zero. Finish all writes before ending the session; leftover processes and changes after the session are rejected.
