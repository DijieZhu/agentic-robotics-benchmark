"""Criterion M, mass properties (weight 0.05): total mass, centre of mass and principal moments of
inertia of the strut in scope, submitted against original:

  M = w[0] 0.3 band(|m / m_ref - 1|; mass.mass_rel 2 % / 25 %) + w[1] 0.35 band(|c - c_ref|; mass.com_mm 0.5 / 5.0)
      + w[2] 0.35 band(max_i |I_i / I_ref,i - 1|; mass.inertia_rel 4 % / 50 %)

Original: the 13 design meshes at their instance poses, each at its material's density (families.json materials:
printed PLA 1.24, SLA resin 1.20, silicone 1.08, board laminate 1.85 g/cm3), and the purchased
placements at their data-sheet masses (purchased.json mass_g: gearmotor 11 g, rod 6.483 g), spread uniformly over the
vendor solid. Submission, in the strut frame (A): every credited tagged part at its tag's design density; every
recognised purchased solid as the grader's copy of its constituent at its recognised pose, at its type's density; every
other solid (extra solids, demoted tagged parts) at the printed-part density (mass.extra_material). What is out of scope
counts on neither side. Principal moments are the sorted eigenvalues of the inertia tensor about the centre of mass.
"""
import numpy as np

from .geom import apply, band

NAME = "mass"


def _moved(V, c, I, T, rho):
    R = np.asarray(T, float)[:3, :3]
    return (V * rho, apply(T, np.asarray(c, float)), rho * (R @ np.asarray(I, float) @ R.T))


def combine(items):
    """items: (mass g, centroid, inertia about the centroid in g mm2) -> (mass, centre of mass, sorted principal
    moments)."""
    M = float(sum(m for m, _c, _I in items))
    if M <= 0:
        return 0.0, np.zeros(3), np.zeros(3)
    com = sum(m * c for m, c, _I in items) / M
    It = np.zeros((3, 3))
    for m, c, I in items:
        d = c - com
        It += I + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return M, com, np.sort(np.linalg.eigvalsh(0.5 * (It + It.T)))


def _type_density(ref, tname):
    t = ref.types[tname]
    return float(t["mass_g"]) / float(t["V"])


def reference_items(ref):
    exp = []
    for tag in ref.tags:
        d = ref.designs[ref.family_of[tag]]
        exp.append(_moved(d["volume"], d["com"], d["I"], ref.instances[tag]["W"], d["density_g_mm3"]))
    for pid, pl in sorted(ref.placements.items()):
        rho = _type_density(ref, pl["type"])
        for c in ref.types[pl["type"]]["cons"]:
            exp.append(_moved(c["V"], c["com"], c["I"], pl["W"], rho))
    return exp


def submission_items(ctx):
    ref = ctx.ref
    A = ctx.alignment["A"]
    rho_extra = float(ref.materials[ctx.cfg["mass"]["extra_material"]]["density_g_cm3"]) / 1000.0
    type_of = {}
    for it in ctx.instances:
        for sid in it["slots"].values():
            type_of[sid] = it["type"]
    act, n_extra = [], 0
    for s in ctx.solids:
        if s["tag"] is not None:
            rho = ref.designs[ref.family_of[s["tag"]]]["density_g_mm3"]
            act.append(_moved(s["V"], s["com"], s["I"], A, rho))
        elif "match" in s:
            rep = ref.classes[s["match"]["cls"]]["rep"]
            rho = _type_density(ref, type_of.get(s["sid"], rep["type"]))
            act.append(_moved(rep["V"], rep["com"], rep["I"], A @ s["match"]["T"], rho))
        else:
            act.append(_moved(s["V"], s["com"], s["I"], A, rho_extra))
            n_extra += 1
    return act, n_extra


def item(ctx):
    """Criterion M (module docstring)."""
    Mc = ctx.cfg["mass"]
    ref = ctx.ref
    if getattr(ref, "_mass_ref", None) is None:
        ref._mass_ref = combine(reference_items(ref))
    bm, bc, bi = ref._mass_ref
    act, n_extra = submission_items(ctx)
    am, ac, ai = combine(act)
    e_m = abs(am / bm - 1.0) if bm > 0 else float("inf")
    e_c = float(np.linalg.norm(ac - bc))
    e_i = float(np.max(np.abs(ai / np.maximum(bi, 1e-9) - 1.0))) if am > 0 else float("inf")
    w = Mc["weights"]
    score = (float(w[0]) * band(e_m, Mc["mass_rel"]) + float(w[1]) * band(e_c, Mc["com_mm"]) +
             float(w[2]) * band(e_i, Mc["inertia_rel"]))
    ctx.stats.update(mass_g=float(am), mass_ref_g=float(bm), mass_error_rel=float(min(e_m, 1e9)),
                     com_error_mm=float(min(e_c, 1e9)), inertia_error_rel=float(min(e_i, 1e9)),
                     mass_extra_solids=float(n_extra))
    diags = []
    if score < 1.0 - 1e-9:
        diags.append("Mass %.2f g (original %.2f g), centre of mass %.2f mm off, principal moments off by %.1f %%" % (
            am, bm, e_c, 100.0 * min(e_i, 1e9)))
    return {"score": float(score), "per_part": {"_totals": {"mass_g": float(am), "mass_ref_g": float(bm),
                                                             "com_error_mm": e_c, "inertia_error_rel": e_i}},
            "diagnostics": diags}
