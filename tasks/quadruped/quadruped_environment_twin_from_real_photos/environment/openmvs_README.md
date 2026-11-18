# OpenMVS command-line tools (CPU build)

OpenMVS 2.4.0, the conda-forge CPU build (`openmvs 2.4.0 cpu_h59d1794_202`), copied here with only the shared
libraries its tools load. The tools run from any directory without activating anything; `/opt/openmvs/bin` is on
`PATH` (after the system's own directories).

Tools: `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, `TextureMesh`, `TransformScene`
(`<tool> --help` lists the options). There is no GPU: every tool runs on the CPU.

- **Threads.** The machine reports more processors than the sandbox may use. Pass `--max-threads 8` (at most the
  sandbox's CPU count) to every tool, and `num_threads=8` to pycolmap's options; otherwise dozens of threads compete
  for the same CPUs and everything runs slower.
- **Work folder.** Every tool writes a `<tool>-<time>.log` into the current directory: run them from a work folder
  of your own, never from the home directory or `submission/`.

Typical use after a sparse reconstruction with pycolmap (a COLMAP model in `sparse/0` of the photos in `images/`):

```
python3 -c "import pycolmap; pycolmap.undistort_images('dense', 'sparse/0', 'images', num_threads=8)"
cd dense
InterfaceCOLMAP -i . -o scene.mvs --image-folder images --max-threads 8
DensifyPointCloud scene.mvs --resolution-level 2 --max-threads 8     # -> scene_dense.ply
ReconstructMesh scene_dense.mvs --max-threads 8                       # optional surface mesh
```

`--resolution-level` halves the images that many times before matching. On 8 CPUs a dense cloud of a dozen photos at
level 2 takes a few minutes; all photos of a site at level 1 take on the order of 20-30 minutes. The multi-threaded
tools are not bit-for-bit deterministic between runs.

## Licences and sources

OpenMVS is free software under the GNU Affero General Public License version 3 (AGPL-3.0); its licence text is in
`licenses/openmvs/`. The libraries beside it keep their own licences: `PACKAGES.tsv` lists every conda package that
contributed a file here (name, version, build, licence) and `licenses/` holds their licence texts. The exact packages
installed are listed, with their download URLs and checksums, in `conda-explicit.txt`.

Corresponding source of these binaries: the OpenMVS v2.4.0 source release (https://github.com/cdcseacave/openMVS,
tag `v2.4.0`) built by the conda-forge recipe `openmvs-feedstock` (https://github.com/conda-forge/openmvs-feedstock)
at the build `cpu_h59d1794_202`; the sources of the other packages are on conda-forge (https://conda-forge.org) under
the names and versions in `PACKAGES.tsv`. The image runs these programs unmodified; nothing else in it links to them.
