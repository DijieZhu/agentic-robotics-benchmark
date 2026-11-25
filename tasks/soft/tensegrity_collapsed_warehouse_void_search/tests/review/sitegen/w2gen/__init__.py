"""w2gen: the blockout site family W2 of task 07 (collapsed warehouse void search).

Hidden-side maintainer tool (the site family and its predicates; tests/README.md describes them). Every site is made
of static boxes only; one seed gives one site package (site.json, site.xml, qc.json). No robot geometry is in this
package.

Modules: envelope (robot envelope numbers and where they came from), family (the W2 rules: seed -> site dict),
grid (occupancy and shortest paths), mjcf (site dict -> MuJoCo XML), predicates (G1-G11), views (figures and the
short film for the owner), cli (command line).
"""
GENERATOR_VERSION = "w2gen 0.1.0"
