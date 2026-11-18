"""Gates G0 (structure) and G2 (no hedging) of the grader: the byte copy schema_check.py of the public check, run on
the submission snapshot with the schema copies next to it. Gate G1 (submission integrity) is verify.py's.

    g0, g2 = check(docs)        verify.py's interface: the G0 and G2 problem texts of documents that
                                load.load_submission read without a problem
    out = evaluate(sub_dir)     reading included: {"g0", "g2", "problems", "notes", "docs", "stats", "diagnostic"}

g0 and g2 are 1.0 or 0.0. `docs` are the documents that loaded; they are scored only when both gates are 1.0. Every
message is printable ASCII (lone surrogates and other non-ASCII characters escaped), so it can go into the verdict
as it is. Nothing here raises on anything an agent can write; the time is bounded by the check's own limits. No
timing goes into the stats, so two runs on one submission give the same output.
"""
import os

from . import schema_check as SC

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
MAX_DIAGNOSTIC_PROBLEMS = 40


def evaluate(sub_dir, schema_dir=SCHEMA_DIR):
    try:
        rep = SC.check_submission(sub_dir, schema_dir)
    except Exception as e:  # noqa: BLE001  (defence in depth: the check is written never to raise)
        rep = {"g0": False, "g2": True, "notes": [], "docs": {},
               "problems": [{"gate": "G0", "file": "-", "message": SC.ascii_text(
                   "the structure check cannot report on the submission: %s" % type(e).__name__)}]}
    problems = rep["problems"]
    g0 = 1.0 if rep["g0"] else 0.0
    g2 = 1.0 if rep["g2"] else 0.0
    n0 = sum(1 for p in problems if p["gate"] == "G0")
    n2 = sum(1 for p in problems if p["gate"] == "G2")
    shown = problems[:MAX_DIAGNOSTIC_PROBLEMS]
    diag = "; ".join("%s %s" % (p["gate"], p["message"]) for p in shown)
    if len(problems) > len(shown):
        diag += "; %d more" % (len(problems) - len(shown))
    return {"g0": g0, "g2": g2, "problems": problems, "notes": rep["notes"], "docs": rep["docs"],
            "stats": {"gate_g0_structure": g0, "gate_g2_no_hedging": g2, "g0_problems": float(n0),
                      "g2_problems": float(n2)},
            "diagnostic": diag or "G0 and G2 hold"}


def check(docs, schema_dir=SCHEMA_DIR):
    """([G0 problem text], [G2 problem text]) of loaded documents (schemas, then the G2 rules)."""
    try:
        problems, _notes = SC.check_documents(docs, schema_dir)
    except Exception as e:  # noqa: BLE001  (defence in depth)
        return [SC.ascii_text("the structure check cannot report on the submission: %s" % type(e).__name__)], []
    return ([p["message"] for p in problems if p["gate"] == "G0"],
            [p["message"] for p in problems if p["gate"] == "G2"])
