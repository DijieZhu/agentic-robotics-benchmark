"""Writers: OBJ + MTL, a USD prim as ASCII .usda (no USD library needed), PNG maps."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def write_png(path, arr):
    Image.fromarray(np.ascontiguousarray(arr)).save(path, compress_level=6)


def write_obj(path, V, N, UV, F, mtl_name, material, maps):
    """One mesh, one material. ``maps``: dict basecolor / normal / orm -> file name relative to the OBJ."""
    stem = os.path.splitext(os.path.basename(path))[0]
    mtl = os.path.join(os.path.dirname(path), mtl_name)
    with open(mtl, "w") as f:
        f.write("# parcelkit material. Base colour sRGB; normal map OpenGL (+Y up), tangent space;\n")
        f.write("# ORM map: R ambient occlusion, G roughness, B metallic (linear).\n")
        f.write("newmtl %s\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 10\nillum 2\n" % material)
        f.write("map_Kd %s\n" % maps["basecolor"])
        f.write("norm %s\n" % maps["normal"])
        f.write("map_Bump -bm 1.0 %s\n" % maps["normal"])
        f.write("map_Pr %s\n" % maps["orm"])
        f.write("# map_Pr is the ORM texture: read its green channel as roughness\n")
    with open(path, "w") as f:
        f.write("# parcelkit visual mesh, metres, parcel frame (z up); %d vertices, %d triangles\n" % (len(V), len(F)))
        f.write("mtllib %s\no %s\n" % (mtl_name, stem))
        f.write("".join("v %.6f %.6f %.6f\n" % tuple(v) for v in V))
        f.write("".join("vt %.6f %.6f\n" % tuple(t) for t in UV))
        f.write("".join("vn %.5f %.5f %.5f\n" % tuple(n) for n in N))
        f.write("usemtl %s\ns 1\n" % material)
        G = F + 1
        f.write("".join("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (a, a, a, b, b, b, c, c, c) for a, b, c in G))


def _arr(fmt, A):
    return "[" + ", ".join(fmt % tuple(r) for r in A) + "]"


def write_usda(path, V, N, UV, F, maps, prim="Parcel", material="parcel_look", doc="", custom=None):
    """A self-contained USD layer: Xform </Parcel> (defaultPrim) with Mesh 'visual' and a UsdPreviewSurface material
    reading the three maps (paths relative to the layer). metersPerUnit 1, upAxis Z."""
    lo, hi = V.min(0), V.max(0)
    cd = ""
    if custom:
        items = []
        for k, v in custom.items():
            if isinstance(v, str):
                items.append('        string %s = "%s"' % (k, v.replace('"', "'")))
            elif isinstance(v, (int, np.integer)):
                items.append('        int %s = %d' % (k, int(v)))
            elif isinstance(v, float):
                items.append('        double %s = %.9g' % (k, v))
            elif isinstance(v, (list, tuple)) and len(v) == 3:
                items.append('        double3 %s = (%.9g, %.9g, %.9g)' % ((k,) + tuple(float(x) for x in v)))
            elif isinstance(v, (list, tuple)) and len(v) == 4:
                items.append('        double4 %s = (%.9g, %.9g, %.9g, %.9g)' % ((k,) + tuple(float(x) for x in v)))
        cd = "    customData = {\n" + "\n".join(items) + "\n    }\n"
    m = "/%s/Looks/%s" % (prim, material)
    txt = []
    txt.append('#usda 1.0\n(\n    defaultPrim = "%s"\n    doc = "%s"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
               % (prim, doc.replace('"', "'")))
    txt.append('def Xform "%s" (\n    kind = "component"\n%s)\n{\n' % (prim, cd))
    txt.append('    def Mesh "visual" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {\n')
    txt.append('        uniform bool doubleSided = 0\n')
    txt.append('        float3[] extent = [(%.6f, %.6f, %.6f), (%.6f, %.6f, %.6f)]\n' % (tuple(lo) + tuple(hi)))
    txt.append('        int[] faceVertexCounts = [%s]\n' % ", ".join(["3"] * len(F)))
    txt.append('        int[] faceVertexIndices = [%s]\n' % ", ".join(str(int(i)) for i in F.reshape(-1)))
    txt.append('        rel material:binding = <%s>\n' % m)
    txt.append('        normal3f[] normals = %s (\n            interpolation = "vertex"\n        )\n'
               % _arr("(%.5f, %.5f, %.5f)", N))
    txt.append('        point3f[] points = %s\n' % _arr("(%.6f, %.6f, %.6f)", V))
    txt.append('        texCoord2f[] primvars:st = %s (\n            interpolation = "vertex"\n        )\n'
               % _arr("(%.6f, %.6f)", UV))
    txt.append('        uniform token subdivisionScheme = "none"\n    }\n\n')
    txt.append('    def Scope "Looks"\n    {\n')
    txt.append('        def Material "%s"\n        {\n' % material)
    txt.append('            token outputs:surface.connect = <%s/PBR.outputs:surface>\n\n' % m)
    txt.append('            def Shader "PBR"\n            {\n'
               '                uniform token info:id = "UsdPreviewSurface"\n'
               '                color3f inputs:diffuseColor.connect = <%s/base.outputs:rgb>\n'
               '                float inputs:metallic.connect = <%s/orm.outputs:b>\n'
               '                normal3f inputs:normal.connect = <%s/nrm.outputs:rgb>\n'
               '                float inputs:occlusion.connect = <%s/orm.outputs:r>\n'
               '                float inputs:roughness.connect = <%s/orm.outputs:g>\n'
               '                int inputs:useSpecularWorkflow = 0\n'
               '%s'
               '                token outputs:surface\n            }\n\n' % (m, m, m, m, m,
                   ('                float inputs:clearcoat.connect = <%s/cc.outputs:r>\n'
                    '                float inputs:clearcoatRoughness = 0.06\n' % m) if "cc" in maps else ""))
    txt.append('            def Shader "st"\n            {\n'
               '                uniform token info:id = "UsdPrimvarReader_float2"\n'
               '                token inputs:varname = "st"\n'
               '                float2 outputs:result\n            }\n\n')
    for name, key, cs, extra in [x for x in (("base", "basecolor", "sRGB", ""), ("orm", "orm", "raw", ""),
                                 ("nrm", "normal", "raw",
                                  '                float4 inputs:bias = (-1, -1, -1, 0)\n'
                                  '                float4 inputs:scale = (2, 2, 2, 1)\n'), ("cc", "cc", "raw", "")) if x[1] in maps]:
        txt.append('            def Shader "%s"\n            {\n'
                   '                uniform token info:id = "UsdUVTexture"\n'
                   '%s'
                   '                asset inputs:file = @%s@\n'
                   '                token inputs:sourceColorSpace = "%s"\n'
                   '                float2 inputs:st.connect = <%s/st.outputs:result>\n'
                   '                token inputs:wrapS = "clamp"\n'
                   '                token inputs:wrapT = "clamp"\n'
                   '                float outputs:b\n                float outputs:g\n                float outputs:r\n'
                   '                float3 outputs:rgb\n            }\n\n' % (name, extra, maps[key], cs, m))
    txt.append('        }\n    }\n}\n')
    with open(path, "w") as f:
        f.write("".join(txt))


def write_layout_usda(path, entries, doc="parcelkit layout", extra_prims=""):
    """A stage that references parcel layers. entries: list of dict(name, usda (path relative to ``path``),
    pos [x, y, z], quat_wxyz [w, x, y, z]). metersPerUnit 1, upAxis Z, defaultPrim /World."""
    txt = ['#usda 1.0\n(\n    defaultPrim = "World"\n    doc = "%s"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
           % doc.replace('"', "'"), 'def Xform "World"\n{\n']
    for e in entries:
        q = e.get("quat_wxyz", (1, 0, 0, 0))
        txt.append('    def Xform "%s" (\n        prepend references = @%s@\n    )\n    {\n'
                   '        quatf xformOp:orient = (%.7f, %.7f, %.7f, %.7f)\n'
                   '        double3 xformOp:translate = (%.6f, %.6f, %.6f)\n'
                   '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n    }\n'
                   % ((e["name"], e["usda"]) + tuple(float(v) for v in q) + tuple(float(v) for v in e["pos"])))
    txt.append(extra_prims)
    txt.append("}\n")
    with open(path, "w") as f:
        f.write("".join(txt))
