"""Convert the Menagerie Panda's visual OBJ meshes to binary STL (about half the size; MuJoCo loads both) and
rewrite the model's mesh references. Usage: python3 obj2stl.py <menagerie/franka_emika_panda> <out_dir>.

environment/Dockerfile runs it at image build on the folder pinned in panda_assets.sha256 and checks the output
against panda_converted.sha256, so the conversion must stay deterministic: same input, same bytes."""
import os, re, shutil, struct, sys
import numpy as np

def read_obj(path):
    verts, faces = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):          # fan-triangulate polygons
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, float), np.asarray(faces, int)

def write_stl(path, v, f):
    tri = v[f]                                            # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    data = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["n"] = n; data["v"] = tri
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80); fh.write(struct.pack("<I", len(f))); fh.write(data.tobytes())

src, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "assets"), exist_ok=True)
for fn in sorted(os.listdir(os.path.join(src, "assets"))):
    p = os.path.join(src, "assets", fn)
    if fn.endswith(".obj"):
        v, f = read_obj(p); write_stl(os.path.join(out, "assets", fn[:-4] + ".stl"), v, f)
    elif fn.endswith(".stl"):
        shutil.copyfile(p, os.path.join(out, "assets", fn))
for fn in ("panda.xml", "panda_nohand.xml", "hand.xml", "LICENSE", "README.md"):
    text = open(os.path.join(src, fn), encoding="utf-8").read()
    if fn.endswith(".xml"):
        text = re.sub(r'file="([^"]+)\.obj"', r'file="\1.stl"', text)
    open(os.path.join(out, fn), "w", encoding="utf-8").write(text)
print("done:", sum(os.path.getsize(os.path.join(out, "assets", f)) for f in os.listdir(os.path.join(out, "assets"))) // 1024, "KiB of assets")
