"""Triangle meshes of MuJoCo geoms in the geom's own frame, with per-corner normals and texture coordinates.

Every function returns (points (n, 3), faces (m, 3) int, normals (m, 3, 3), st (m, 3, 2)). Texture coordinates use
the USD convention: st = (0, 0) is the bottom-left corner of the image file.

Primitives get texture coordinates that MuJoCo's own viewer does not define exactly:
  box       each face carries the whole image once (times texrepeat);
  plane     the image is repeated texrepeat times over the plane, or texrepeat times per metre with texuniform;
  sphere, ellipsoid, capsule, cylinder side: longitude around z (u) and height or latitude (v); caps are planar.
Meshes use the texture coordinates the mesh file carries.
"""
import math

import numpy as np


def _pack(points, tris, normal_fn, st_fn):
    P = np.asarray(points, np.float64)
    F = np.asarray(tris, np.int64).reshape(-1, 3)
    corners = P[F]
    N = normal_fn(corners)
    S = st_fn(corners, F)
    return P.astype(np.float32), F.astype(np.int32), N.astype(np.float32), S.astype(np.float32)


def box(sx, sy, sz, rep=(1.0, 1.0)):
    """24 corners (4 per face), outward winding; every face maps the image once (times texrepeat)."""
    s = np.array([sx, sy, sz], float)
    pts, tris, nrm, st = [], [], [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[a] = sign
            u = np.zeros(3)
            u[b] = 1.0
            w = np.cross(n, u)
            quad = [n * s[a] - u * s[b] - w * s[c], n * s[a] + u * s[b] - w * s[c],
                    n * s[a] + u * s[b] + w * s[c], n * s[a] - u * s[b] + w * s[c]]
            k = len(pts)
            pts.extend(quad)
            uvq = [(0.0, 0.0), (rep[0], 0.0), (rep[0], rep[1]), (0.0, rep[1])]
            for t in ((0, 1, 2), (0, 2, 3)):
                tris.append([k + t[0], k + t[1], k + t[2]])
                nrm.append([n, n, n])
                st.append([uvq[t[0]], uvq[t[1]], uvq[t[2]]])
    P = np.asarray(pts, np.float32)
    return P, np.asarray(tris, np.int32), np.asarray(nrm, np.float32), np.asarray(st, np.float32)


def plane(sx, sy, rep=(1.0, 1.0), uniform=False):
    """A two-triangle quad facing +z, size 2 sx x 2 sy."""
    P = np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]], int)

    def st(c, _F):
        if uniform:
            return np.stack([c[..., 0] * rep[0], c[..., 1] * rep[1]], -1)
        return np.stack([(c[..., 0] + sx) / (2 * sx) * rep[0], (c[..., 1] + sy) / (2 * sy) * rep[1]], -1)

    return _pack(P, F, lambda c: np.broadcast_to([0.0, 0.0, 1.0], c.shape).copy(), st)


def _grid(nu, nv):
    tris = []
    for i in range(nv):
        for j in range(nu):
            a, b = i * (nu + 1) + j, i * (nu + 1) + j + 1
            c, d = (i + 1) * (nu + 1) + j + 1, (i + 1) * (nu + 1) + j
            tris.append([a, b, c])
            tris.append([a, c, d])
    return tris


def ellipsoid(a, b, c, nu=64, nv=32, rep=(1.0, 1.0)):
    """UV sphere scaled to semi-axes (a, b, c); a sphere is a = b = c. u = longitude, v = latitude (south = 0)."""
    pts, uvs = [], []
    for i in range(nv + 1):
        th = math.pi * i / nv                    # 0 at the south pole
        for j in range(nu + 1):
            ph = 2 * math.pi * j / nu
            pts.append((a * math.sin(th) * math.cos(ph), b * math.sin(th) * math.sin(ph), -c * math.cos(th)))
            uvs.append((j / nu * rep[0], i / nv * rep[1]))
    P = np.asarray(pts, float)
    UV = np.asarray(uvs, float)
    F = np.asarray(_grid(nu, nv), int)
    keep = [k for k, t in enumerate(F) if len({tuple(np.round(P[v], 12)) for v in t}) == 3]
    F = F[keep]
    inv = np.array([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)])

    def nfn(cor):
        n = cor * inv
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)

    return _pack(P, F, nfn, lambda cor, FF: UV[FF])


def cylinder(r, h, nu=64, rep=(1.0, 1.0)):
    """Side, bottom cap and top cap of a cylinder of radius r and half-height h about z."""
    pts, tris, nrm, st = [], [], [], []
    for j in range(nu):
        t0, t1 = 2 * math.pi * j / nu, 2 * math.pi * (j + 1) / nu
        c0, s0, c1, s1 = math.cos(t0), math.sin(t0), math.cos(t1), math.sin(t1)
        q = [(r * c0, r * s0, -h), (r * c1, r * s1, -h), (r * c1, r * s1, h), (r * c0, r * s0, h)]
        n = [(c0, s0, 0), (c1, s1, 0), (c1, s1, 0), (c0, s0, 0)]
        u = [(j / nu * rep[0], 0), ((j + 1) / nu * rep[0], 0), ((j + 1) / nu * rep[0], rep[1]), (j / nu * rep[0], rep[1])]
        k = len(pts)
        pts.extend(q)
        for t in ((0, 1, 2), (0, 2, 3)):
            tris.append([k + t[0], k + t[1], k + t[2]])
            nrm.append([n[t[0]], n[t[1]], n[t[2]]])
            st.append([u[t[0]], u[t[1]], u[t[2]]])
        for z, sgn in ((h, 1.0), (-h, -1.0)):
            k = len(pts)
            pts.extend([(0, 0, z), (r * c0, r * s0, z), (r * c1, r * s1, z)])
            tri = [k, k + 1, k + 2] if sgn > 0 else [k, k + 2, k + 1]
            tris.append(tri)
            nrm.append([(0, 0, sgn)] * 3)
            st.append([((pts[v][0] / r + 1) / 2 * rep[0], (pts[v][1] / r + 1) / 2 * rep[1]) for v in tri])
    return (np.asarray(pts, np.float32), np.asarray(tris, np.int32), np.asarray(nrm, np.float32),
            np.asarray(st, np.float32))


def capsule(r, h, nu=64, nv=16, rep=(1.0, 1.0)):
    """A cylinder of half-height h with hemispherical ends of radius r about z (smooth normals)."""
    rings = []                                   # (z, ring radius, normal z, normal radial)
    for i in range(nv + 1):
        a = -math.pi / 2 + (math.pi / 2) * i / nv
        rings.append((-h + r * math.sin(a), r * math.cos(a), math.sin(a), math.cos(a)))
    for i in range(nv + 1):
        a = (math.pi / 2) * i / nv
        rings.append((h + r * math.sin(a), r * math.cos(a), math.sin(a), math.cos(a)))
    total = 2 * h + 2 * r
    pts, nr, uvs = [], [], []
    for (z, rr, nz, nrad) in rings:
        for j in range(nu + 1):
            ph = 2 * math.pi * j / nu
            pts.append((rr * math.cos(ph), rr * math.sin(ph), z))
            nr.append((nrad * math.cos(ph), nrad * math.sin(ph), nz))
            uvs.append((j / nu * rep[0], (z + h + r) / total * rep[1]))
    P, NR, UV = np.asarray(pts, float), np.asarray(nr, float), np.asarray(uvs, float)
    F = np.asarray(_grid(nu, len(rings) - 1), int)
    keep = [k for k, t in enumerate(F) if len({tuple(np.round(P[v], 12)) for v in t}) == 3]
    F = F[keep]
    N = NR[F]
    return P.astype(np.float32), F.astype(np.int32), N.astype(np.float32), UV[F].astype(np.float32)


def mesh_arrays(m, mesh_id, rep=(1.0, 1.0)):
    """(points, faces, normals per corner, st per corner or None) of a compiled MuJoCo mesh, in the mesh frame.

    MuJoCo flips v when it loads texture coordinates (its v = 0 is the first row of the image data) and scales them
    by the material's texrepeat (measured: texrepeat changes MuJoCo's own render of a mesh), so the USD st is
    (u * rx, 1 - v * ry)."""
    va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
    fa, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
    P = np.asarray(m.mesh_vert[va:va + vn], np.float32)
    F = np.asarray(m.mesh_face[fa:fa + fn], np.int32)
    N = None
    nn = int(m.mesh_normalnum[mesh_id])
    if nn > 0:
        na = int(m.mesh_normaladr[mesh_id])
        N = np.asarray(m.mesh_normal[na:na + nn], np.float32)[np.asarray(m.mesh_facenormal[fa:fa + fn], np.int64)]
    if N is None or not np.all(np.isfinite(N)):
        tri = P[F].astype(np.float64)
        fnrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fnrm /= np.maximum(np.linalg.norm(fnrm, axis=1, keepdims=True), 1e-12)
        N = np.repeat(fnrm[:, None, :], 3, axis=1).astype(np.float32)
    S = None
    tn = int(m.mesh_texcoordnum[mesh_id])
    if tn > 0:
        ta = int(m.mesh_texcoordadr[mesh_id])
        T = np.asarray(m.mesh_texcoord[ta:ta + tn], np.float32)[np.asarray(m.mesh_facetexcoord[fa:fa + fn], np.int64)]
        S = np.stack([T[..., 0] * rep[0], 1.0 - T[..., 1] * rep[1]], -1).astype(np.float32)
    return P, F, N, S
