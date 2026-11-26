import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C
import gtrace as TR

SH, K = C.SH, C.K


def _draw():
    import draw as DR
    return DR


def _sheet():
    import sheet as SHEET
    return SHEET


class UnitContext:

    def __init__(self, root, label):
        DR = _draw()
        self.root = root
        self.label = label
        self.rec = DR.load_candidate(root, label)
        if self.rec.get("outcome") != "admitted":
            raise ValueError(f"{label}: not an admitted candidate")
        self.reg = DR.open_registry(root)
        st, ses = self.reg.load_unit(label)
        self.state, self.session = st, ses[0]
        if st.digest != self.rec["unit_digest"]:
            raise ValueError(f"{label}: the stored state is not the candidate's")
        self.site = self.rec["site"]
        sh = self.rec["shift"]
        self.slots = [dict(s) for s in sh["slots"]]
        self.jobs = [dict(j) for j in sh["jobs"]]
        S = _sheet()
        self.sheet = S.build_sheet([(sl["job"], j["targets"]) for sl, j in zip(self.slots, self.jobs)])
        if sh.get("sheet_digest") is not None and S.digest(self.sheet) != sh["sheet_digest"]:
            raise ValueError(f"{label}: the rebuilt shift sheet's digest differs from the record's")
        self.cell = DR.CELL

    def build(self, profile=None):
        prof = TR.grader_trace_fields(self.cell) if profile is None else profile
        return self.reg.build(self.label, world=SH.make_world(self.site, cell=self.cell), policy="stage05",
                              truth_profile=prof)

    def public(self, unit):
        import runtime as RT
        rec, site_pub = RT.setup_data(unit, self.reg.noise_key(self.label), self.sheet)
        return rec, site_pub, ["runtime.setup_data: docgen/record.py delivery_record (the printed delivery record)",
                               "runtime.setup_data: docgen/celldata.py cell sheet and the unit's shift sheet"]

    def identity(self):
        return {"label": self.label, "unit_digest": self.rec["unit_digest"], "candidate_digest": self.rec["digest"],
                "sheet_digest": self.rec["shift"].get("sheet_digest"), "n_slots": len(self.slots)}


def admitted_labels(root):
    DR = _draw()
    d = os.path.join(root, "candidates")
    out = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".json"):
            r = DR.load_candidate(root, n[:-5])
            if r.get("outcome") == "admitted":
                out.append(r["label"])
    return out
