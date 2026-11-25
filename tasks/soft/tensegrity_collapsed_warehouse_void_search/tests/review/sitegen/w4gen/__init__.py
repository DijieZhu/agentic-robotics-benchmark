"""w4gen: the site family W4 of task 07 version 2.0 (collapsed warehouse void search).

Hidden-side maintainer tool. W4 = W3 (module w3gen, imported, never changed: the open ground of bands A, R1, B, the
whole slab, three look-alike covered mouths, the structured void behind them and the rubble boxes of rule A in its
straight runs) with the open floor's grey blocks replaced by heaps: 3-6 rubble heaps (class R) and mounds of fill
(class M) from the heap library, each leaning on a wall, the ridge or the slab's front edge, placed from a new visible
random stream. Under the slab nothing changes. Every heap's collision body is an inline MuJoCo height field (the
piece's 5 mm cells times its scale, sunk 10 mm, a ring of zero cells, max == size_z); one seed gives one site package
(site.json, site.xml, qc.json) and, for the maintainers only, the table heap_pieces.json (which library piece made
each heap).

Modules: library (the heap registry, the pieces, the placement transform), family (seed -> the three counterfactual
sites and the site), mjcf (site dict + grids -> MuJoCo XML), measure (measurements on the compiled model),
predicates (G1-G24), cli (command line). w2gen and w3gen are imported, never changed.

The generator's version string is defined once, in family.py (family.GENERATOR_VERSION); `w4gen.GENERATOR_VERSION`
reads it from there on first use, so importing the package alone does not import the family.
"""


def __getattr__(name):
    if name == "GENERATOR_VERSION":
        from .family import GENERATOR_VERSION
        return GENERATOR_VERSION
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
