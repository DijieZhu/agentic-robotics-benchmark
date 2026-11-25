"""w3gen: the site family W3 of task 07 version 1.1 (collapsed warehouse void search; "W3+C" in the documents).

Hidden-side maintainer tool. W3 = W2's open ground (bands A, R1, B by W2's rules, module w2gen.family) + one whole
slab over the north part of the site + three look-alike covered P mouths in the front row (one leads in, two are
plugged a short way inside) + a structured void behind them (fore-gallery G1 with a dead-end pocket, connector N,
optional back corridor K, chamber C in a back corner holding search point 3) + fixed rubble boxes in the straight
runs under the slab, each leaving one P-wide gap beside it (rule "A", module rubble). Every site is made of static
boxes only; one seed gives one site package (site.json, site.xml, qc.json). No robot geometry is in this package.

Modules: family (seed -> the three counterfactual sites, one per mouth, and the site of the true mouth), rubble
(the rubble rule), mjcf (site dict -> MuJoCo XML; W2's writer), predicates (G1-G19), cli (command line).
w2gen is imported, never changed: the version 1.0 sites must stay reproducible.
"""
GENERATOR_VERSION = "w3gen 0.1.0"
