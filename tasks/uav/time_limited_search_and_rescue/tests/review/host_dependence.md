# Does a program's flight depend on the grading host?

Measured on two machines of different CPU families (AMD Zen 3 with AVX2, AMD Zen 5 with AVX-512), same task image.

**The simulator and the pictures do not.** One recorded action tape replayed on both machines (1,410 actions, both
cameras hashed at every action): all 2,820 frames byte-identical. Frames after a reload inside one process (a new site
after three others were flown and closed, as `new_site` does) are byte-identical with a fresh process.

**A program's own arithmetic does, unless it is pinned.** An image-driven pilot (the author's, kept in
`author_image_pilot/`; used here only as an instrument) was flown in process on four development sites on each
machine:

| Program environment | Action tapes, machine A against machine B |
|---|---|
| as the image was (OpenBLAS and NumPy choose their kernels from the CPU) | different on 4 of 4 sites; actions differ from the sixth decimal, the frames follow, and one flight took 2,464 actions on one machine and 2,465 on the other. The four outcomes were the same. |
| `NPY_DISABLE_CPU_FEATURES` (AVX-512 off) alone | still different (1 site tried) |
| `OPENBLAS_CORETYPE=HASWELL` and `NPY_DISABLE_CPU_FEATURES` (AVX-512 off) | identical tapes and identical frames on 4 of 4 sites |

So the grader starts the agent's program (and the aircraft service) with those two variables, and the image sets them
in the agent's sandbox, so that what a program computes while it is being written is what it computes when graded.
The instruction states both. A program that brings its own compiled numerics, or unsets the variables, is outside
this guarantee.

The privileged reference reads no image and uses no BLAS call that matters: its action counts are the same on both
machines with or without the pin.

**Since 2026-11-22 (furnished scene, Filament renderer on the CPU).** The pictures were measured again with
`tests/review/camera_digests.py` on the same two machines: two development sites, 164 made-up states each (every
room cell of both storeys on four headings), both cameras: 656 images byte-identical between the machines, and on each
machine identical when drawn again in reverse order and by fresh processes. The daylight computed at a site's build
and the door aprons' wear pictures are part of those images. The action-tape comparison of the image-driven
instrument above was not repeated on the new pictures: with identical frames and the numerics pin its inputs are
identical on both machines, as they were.
