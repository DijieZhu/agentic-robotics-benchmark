"""A compiled MuJoCo scene (one state) -> one self-contained OpenUSD stage that OVRTX path-traces.

Runs in the task's Python (mujoco + usd-core); never imports ovrtx. Writes <out>/scene.usdc and <out>/textures/*.png:

  /World/Geoms/g<id>   one Mesh per visible geom at its world pose: mesh geoms with the mesh's normals and texture
                       coordinates, primitives tessellated with sensible texture coordinates (shapes.py)
  /World/Looks/M<k>    UsdPreviewSurface per material: colour (MuJoCo colours are display values, linearised here),
                       base colour, normal, ORM / roughness / metallic and emissive textures, roughness, metallic,
                       opacity (rgba alpha), emission
  /World/Lights        MJCF lights in physical units: spot and point lights as small sphere lights whose on-axis
                       intensity is the light's `intensity` in candela; directional lights as distant lights whose
                       illuminance is `intensity` in lux; a light without `intensity` gets one from its diffuse
                       colour (DEFAULT_CANDELA_PER_DIFFUSE, DEFAULT_LUX_PER_DIFFUSE); plus a uniform dome light
  /World/<camera>      the camera as an exact OpenCV pinhole (fx, fy, cx, cy of camera.py)
  /Render/Product      one PathTracing RenderProduct with LdrColor and HdrColor outputs

Which geoms are drawn: those in the given geom groups (MuJoCo's viewer shows groups 0, 1 and 2) whose alpha is not
zero. Not rendered, each with a warning: height-field and SDF geoms, flex objects and skins; cube and skybox
textures; occlusion, opacity and user-role material textures (the colour's alpha sets the opacity); image lights;
and MuJoCo's headlight (it has no physical counterpart).
"""
import hashlib
import math
import os

import numpy as np

from . import camera as CAM
from . import shapes as SH

DEFAULTS = {
    "spp": 64,                  # path-traced samples per pixel
    "max_bounces": 6,
    "dome_lux": 60.0,           # illuminance of the uniform dome light on an upward-facing surface; 0 = no dome
    "dome_rgb": (1.0, 1.0, 1.0),
    "exposure_ev": 0.0,         # added to the camera exposure (+1 = twice as bright)
    "iso": 100.0, "fstop": 2.0, "shutter_s": 1.0 / 25.0,
    "denoise": False,
    "lens": "opencv",           # "opencv": OmniLensDistortionOpenCvPinholeAPI; "aperture": USD apertures and offsets
    # OVRTX's lens schema counts pixels from the image edge (OpenCV counts from the first pixel centre): cx, cy get
    # +0.5. Measured with sphere centroids: 0.03 px max with +0.5, 0.73 px with 0; the aperture route assumes square
    # pixels (66 px off at the image edge for fx/fy = 1.15), so it is not used.
    "principal_offset_px": 0.5,
    "near": 0.01, "far": 1000.0,
}
DEFAULT_CANDELA_PER_DIFFUSE = 1000.0
DEFAULT_LUX_PER_DIFFUSE = 400.0
EMISSION_NITS = 100.0           # luminance of an emission = 1 surface of colour white
# OVRTX's UsdPreviewSurface emits about this many nits per unit of emissiveColor. A calibration constant: one
# emissiveColor = 1 render gave HdrColor 11.33, and lights of known candela and lux give 0.0098 HdrColor per nit.
EMISSIVE_COLOR_NITS = 1156.0
LENS_PITCH = 0.36               # sensor width in tenths of a metre (36 mm); only ratios matter for a pinhole
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)
GEOM_NAMES = {0: "plane", 1: "height-field", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box",
              7: "mesh", 8: "SDF"}
# material texture roles the exporter does not draw (mjtTextureRole: USER 0, OCCLUSION 2, OPACITY 6)
UNRENDERED_ROLES = {0: "user-role", 2: "occlusion", 6: "opacity"}


def srgb_to_linear(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, None)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _matrix(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


class Exporter:
    def __init__(self, m, d, out_dir, groups=(0, 1, 2), opts=None):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender, UsdShade, Vt
        self.pxr = dict(Gf=Gf, Sdf=Sdf, Usd=Usd, UsdGeom=UsdGeom, UsdLux=UsdLux, UsdRender=UsdRender,
                        UsdShade=UsdShade, Vt=Vt)
        self.m, self.d = m, d
        self.out = os.path.abspath(out_dir)
        self.tex_dir = os.path.join(self.out, "textures")
        os.makedirs(self.tex_dir, exist_ok=True)
        self.groups = set(int(g) for g in groups)
        self.o = dict(DEFAULTS, **(opts or {}))
        self.warnings = []
        self.tex_files = {}
        self.materials = {}
        self.counts = {"geoms": 0, "triangles": 0, "materials": 0, "textures": 0, "lights": 0}
        self.path = os.path.join(self.out, "scene.usdc")
        if os.path.exists(self.path):
            os.remove(self.path)
        self.stage = Usd.Stage.CreateNew(self.path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        # attached OVRTX populates only schema attributes unless the root layer asks for every authored one
        self.stage.GetRootLayer().customLayerData = {"populateAllAuthoredAttributes": True}
        for p in ("/World/Looks", "/World/Geoms", "/World/Lights"):
            UsdGeom.Scope.Define(self.stage, p)

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _attr(self, prim, name, typ, value):
        a = prim.CreateAttribute(name, getattr(self.pxr["Sdf"].ValueTypeNames, typ), False)
        a.Set(value)
        return a

    # ---- textures and materials -------------------------------------------------------------------------------
    def texture_file(self, t):
        if t in self.tex_files:
            return self.tex_files[t]
        m = self.m
        w, h, c = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
        a = int(m.tex_adr[t])
        data = np.asarray(m.tex_data[a:a + w * h * c], np.uint8).reshape(h, w, c)
        if c == 1:
            data = np.repeat(data, 3, axis=2)
        elif c >= 4:
            data = data[..., :4]
        name = hashlib.sha256(data.tobytes()).hexdigest()[:16] + ".png"
        path = os.path.join(self.tex_dir, name)
        if not os.path.exists(path):
            from PIL import Image
            Image.fromarray(np.ascontiguousarray(data)).save(path, compress_level=1)
            self.counts["textures"] += 1
        self.tex_files[t] = path
        return path

    def _texture_node(self, mat_path, name, t, raw, scale=None, bias=None):
        P = self.pxr
        sh = P["UsdShade"].Shader.Define(self.stage, "%s/%s" % (mat_path, name))
        sh.CreateIdAttr("UsdUVTexture")
        sh.CreateInput("file", P["Sdf"].ValueTypeNames.Asset).Set(
            P["Sdf"].AssetPath("./textures/" + os.path.basename(self.texture_file(t))))
        st = P["UsdShade"].Shader.Get(self.stage, mat_path + "/St")
        sh.CreateInput("st", P["Sdf"].ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        sh.CreateInput("sourceColorSpace", P["Sdf"].ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        sh.CreateInput("wrapS", P["Sdf"].ValueTypeNames.Token).Set("repeat")
        sh.CreateInput("wrapT", P["Sdf"].ValueTypeNames.Token).Set("repeat")
        if scale is not None:
            sh.CreateInput("scale", P["Sdf"].ValueTypeNames.Float4).Set(P["Gf"].Vec4f(*[float(v) for v in scale]))
        if bias is not None:
            sh.CreateInput("bias", P["Sdf"].ValueTypeNames.Float4).Set(P["Gf"].Vec4f(*[float(v) for v in bias]))
        sh.CreateOutput("rgb", P["Sdf"].ValueTypeNames.Float3)
        for ch in "rgba":
            sh.CreateOutput(ch, P["Sdf"].ValueTypeNames.Float)
        return sh

    def rgba(self, g):
        """MuJoCo's rule: a geom rgba that differs from the default wins over its material's rgba."""
        m = self.m
        mid = int(m.geom_matid[g])
        grgba = tuple(float(v) for v in m.geom_rgba[g])
        if mid < 0 or any(abs(a - b) > 1e-6 for a, b in zip(grgba, DEFAULT_RGBA)):
            return grgba
        return tuple(float(v) for v in m.mat_rgba[mid])

    def material(self, g, rgba, has_uv=True):
        """The USD material of geom g (created once per MuJoCo material, colour and texture-coordinate presence)."""
        import mujoco
        m = self.m
        mid = int(m.geom_matid[g])
        key = (mid, tuple(round(v, 6) for v in rgba), bool(has_uv))
        if key in self.materials:
            return self.materials[key]
        P = self.pxr
        path = "/World/Looks/M%d" % len(self.materials)
        mat = P["UsdShade"].Material.Define(self.stage, path)
        sh = P["UsdShade"].Shader.Define(self.stage, path + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        rd = P["UsdShade"].Shader.Define(self.stage, path + "/St")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", P["Sdf"].ValueTypeNames.Token).Set("st")
        rd.CreateOutput("result", P["Sdf"].ValueTypeNames.Float2)
        lin = [float(v) for v in srgb_to_linear(rgba[:3])]
        rough, metal, emission = 0.5, 0.0, 0.0
        roles = {}
        if mid >= 0:
            r, mt = float(m.mat_roughness[mid]), float(m.mat_metallic[mid])
            if r >= 0:
                rough = r
            else:                             # Phong shininess -> GGX roughness (exponent 128 * shininess)
                n = 128.0 * max(0.0, float(m.mat_shininess[mid]))
                rough = math.sqrt(math.sqrt(2.0 / (n + 2.0)))
            if mt >= 0:
                metal = mt
            emission = max(0.0, float(m.mat_emission[mid]))
            mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or str(mid)
            for role in range(int(mujoco.mjtTextureRole.mjNTEXROLE)):
                t = int(m.mat_texid[mid, role])
                if t < 0:
                    continue
                if role in UNRENDERED_ROLES:
                    self.warn("material %s: its %s texture is not rendered%s" % (
                        mname, UNRENDERED_ROLES[role],
                        " (the colour's alpha sets the opacity)" if role == int(mujoco.mjtTextureRole.mjTEXROLE_OPACITY)
                        else ""))
                    continue
                if not has_uv:
                    self.warn("geom %s: its mesh has no texture coordinates, so its material's textures are not "
                              "drawn (colour only)" % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or g))
                    continue
                if int(m.tex_type[t]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
                    self.warn("material %s: cube and skybox textures are not rendered (only 2D textures)" % mname)
                    continue
                roles[role] = t
        else:
            rough = math.sqrt(math.sqrt(2.0 / (128.0 * 0.5 + 2.0)))
        R = mujoco.mjtTextureRole
        base = roles.get(int(R.mjTEXROLE_RGB), roles.get(int(R.mjTEXROLE_RGBA)))
        F = P["Sdf"].ValueTypeNames
        if base is not None:
            t = self._texture_node(path, "BaseColor", base, False, scale=lin + [1.0])
            sh.CreateInput("diffuseColor", F.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", F.Color3f).Set(P["Gf"].Vec3f(*lin))
        if int(R.mjTEXROLE_ORM) in roles:
            t = self._texture_node(path, "Orm", roles[int(R.mjTEXROLE_ORM)], True)
            sh.CreateInput("roughness", F.Float).ConnectToSource(t.ConnectableAPI(), "g")
            sh.CreateInput("metallic", F.Float).ConnectToSource(t.ConnectableAPI(), "b")
        else:
            if int(R.mjTEXROLE_ROUGHNESS) in roles:
                t = self._texture_node(path, "Roughness", roles[int(R.mjTEXROLE_ROUGHNESS)], True)
                sh.CreateInput("roughness", F.Float).ConnectToSource(t.ConnectableAPI(), "r")
            else:
                sh.CreateInput("roughness", F.Float).Set(float(np.clip(rough, 0.0, 1.0)))
            if int(R.mjTEXROLE_METALLIC) in roles:
                t = self._texture_node(path, "Metallic", roles[int(R.mjTEXROLE_METALLIC)], True)
                sh.CreateInput("metallic", F.Float).ConnectToSource(t.ConnectableAPI(), "r")
            else:
                sh.CreateInput("metallic", F.Float).Set(float(np.clip(metal, 0.0, 1.0)))
        if int(R.mjTEXROLE_NORMAL) in roles:
            t = self._texture_node(path, "Normal", roles[int(R.mjTEXROLE_NORMAL)], True,
                                   scale=[2.0, 2.0, 2.0, 1.0], bias=[-1.0, -1.0, -1.0, 0.0])
            sh.CreateInput("normal", F.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        if int(R.mjTEXROLE_EMISSIVE) in roles:
            k = EMISSION_NITS * max(emission, 1.0) / EMISSIVE_COLOR_NITS
            t = self._texture_node(path, "Emissive", roles[int(R.mjTEXROLE_EMISSIVE)], False, scale=[k, k, k, 1.0])
            sh.CreateInput("emissiveColor", F.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        elif emission > 0:
            k = EMISSION_NITS * emission / EMISSIVE_COLOR_NITS
            sh.CreateInput("emissiveColor", F.Color3f).Set(P["Gf"].Vec3f(*[k * c for c in lin]))
        if rgba[3] < 0.999:
            sh.CreateInput("opacity", F.Float).Set(float(np.clip(rgba[3], 0.0, 1.0)))
        sh.CreateOutput("surface", F.Token)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.materials[key] = mat
        self.counts["materials"] += 1
        return mat

    # ---- geometry ---------------------------------------------------------------------------------------------
    def shape(self, g):
        m = self.m
        typ = int(m.geom_type[g])
        s = np.asarray(m.geom_size[g], float)
        mid = int(m.geom_matid[g])
        rep = tuple(float(v) for v in m.mat_texrepeat[mid]) if mid >= 0 else (1.0, 1.0)
        uniform = bool(m.mat_texuniform[mid]) if mid >= 0 else False
        if typ == 7:
            return SH.mesh_arrays(m, int(m.geom_dataid[g]), rep)
        if typ == 6:
            return SH.box(s[0], s[1], s[2], rep)
        if typ == 2:
            return SH.ellipsoid(s[0], s[0], s[0], rep=rep)
        if typ == 4:
            return SH.ellipsoid(s[0], s[1], s[2], rep=rep)
        if typ == 5:
            return SH.cylinder(s[0], s[1], rep=rep)
        if typ == 3:
            return SH.capsule(s[0], s[1], rep=rep)
        if typ == 0:
            sx = s[0] if s[0] > 0 else 50.0
            sy = s[1] if s[1] > 0 else 50.0
            return SH.plane(sx, sy, rep, uniform)
        return None

    def geoms(self):
        import mujoco
        m, d, P = self.m, self.d, self.pxr
        Vt, Gf, UsdGeom, UsdShade = P["Vt"], P["Gf"], P["UsdGeom"], P["UsdShade"]
        skipped = {}                                   # geom type -> names of drawable geoms that are not rendered
        for g in range(m.ngeom):
            if int(m.geom_group[g]) not in self.groups:
                continue
            rgba = self.rgba(g)
            if rgba[3] <= 1e-3:
                continue
            sh = self.shape(g)
            if sh is None:
                skipped.setdefault(int(m.geom_type[g]), []).append(
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "geom %d" % g)
                continue
            V, F, N, S = sh
            if len(F) == 0:
                continue
            mat = self.material(g, rgba, S is not None)
            mesh = UsdGeom.Mesh.Define(self.stage, "/World/Geoms/g%05d" % g)
            M = _matrix(np.asarray(d.geom_xmat[g], float).reshape(3, 3), np.asarray(d.geom_xpos[g], float))
            mesh.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))       # USD matrices are row-vector (transposed)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F.reshape(-1), np.int32)))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            lo, hi = V.min(axis=0), V.max(axis=0)
            mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in lo]), Gf.Vec3f(*[float(x) for x in hi])]))
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(N.reshape(-1, 3), np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            if S is not None:
                pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", P["Sdf"].ValueTypeNames.TexCoord2fArray,
                                                             UsdGeom.Tokens.faceVarying)
                pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(S.reshape(-1, 2), np.float32)))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            self.counts["geoms"] += 1
            self.counts["triangles"] += int(len(F))
        for typ, names in sorted(skipped.items()):
            self.warn("%d %s geom(s) are not rendered: %s%s" % (len(names), GEOM_NAMES.get(typ, "type-%d" % typ),
                                                                 ", ".join(names[:8]), ", ..." if len(names) > 8 else ""))
        if m.nflex:
            self.warn("%d flex object(s) are not rendered" % m.nflex)
        if m.nskin:
            self.warn("%d skin(s) are not rendered" % m.nskin)
        if int(m.vis.headlight.active):
            self.warn("the MuJoCo headlight is not rendered (no physical counterpart); light the scene with MJCF lights")

    # ---- lights -----------------------------------------------------------------------------------------------
    def lights(self):
        import mujoco
        m, d, P = self.m, self.d, self.pxr
        UsdLux, Gf, Sdf = P["UsdLux"], P["Gf"], P["Sdf"]
        T = mujoco.mjtLightType
        for i in range(m.nlight):
            if not bool(m.light_active[i]):
                continue
            typ = int(m.light_type[i])
            diffuse = np.asarray(m.light_diffuse[i], float)
            peak = float(diffuse.max())
            color = (diffuse / peak) if peak > 0 else np.ones(3)
            intensity = float(m.light_intensity[i])
            pos = np.asarray(d.light_xpos[i], float)
            dirv = np.asarray(d.light_xdir[i], float)
            dirv = dirv / max(np.linalg.norm(dirv), 1e-12)
            # local -z along the light direction
            z = -dirv
            x = np.cross([0.0, 0.0, 1.0], z)
            if np.linalg.norm(x) < 1e-6:
                x = np.cross([0.0, 1.0, 0.0], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            M = _matrix(np.stack([x, y, z], 1), pos)
            path = "/World/Lights/L%d" % i
            if typ == int(T.mjLIGHT_DIRECTIONAL):
                lux = intensity if intensity > 0 else DEFAULT_LUX_PER_DIFFUSE * peak
                lt = UsdLux.DistantLight.Define(self.stage, path)
                lt.CreateAngleAttr(1.0)
                prim = lt.GetPrim()
                prim.AddAppliedSchema("PhotometricDistantLightAPI")
                self._attr(prim, "photometric:illuminance", "Float", float(lux))
                desc = {"type": "directional", "lux": lux}
            elif typ in (int(T.mjLIGHT_SPOT), int(T.mjLIGHT_POINT)):
                cd = intensity if intensity > 0 else DEFAULT_CANDELA_PER_DIFFUSE * peak
                lt = UsdLux.SphereLight.Define(self.stage, path)
                lt.CreateRadiusAttr(float(max(0.002, m.light_bulbradius[i])))
                prim = lt.GetPrim()
                prim.AddAppliedSchema("PhotometricAreaLightAPI")
                # on-axis illuminance at 1 m in lux equals the luminous intensity in candela
                self._attr(prim, "photometric:power", "Float", 0.0)
                self._attr(prim, "photometric:illuminance", "Float", float(cd))
                self._attr(prim, "photometric:illuminance:distance", "Float", 1.0)
                desc = {"type": "point", "candela": cd}
                if typ == int(T.mjLIGHT_SPOT):
                    sh = UsdLux.ShapingAPI.Apply(prim)
                    sh.CreateShapingConeAngleAttr(float(np.clip(m.light_cutoff[i], 0.1, 180.0)))
                    sh.CreateShapingFocusAttr(float(max(0.0, m.light_exponent[i])))
                    desc = {"type": "spot", "candela": cd, "cone_half_angle_deg": float(m.light_cutoff[i])}
            else:
                self.warn("image lights are not rendered")
                continue
            lt.CreateColorAttr(Gf.Vec3f(*[float(c) for c in color]))
            lt.CreateIntensityAttr(1.0)
            UsdLux.ShadowAPI.Apply(lt.GetPrim()).CreateShadowEnableAttr(bool(m.light_castshadow[i]))
            lt.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
            self.counts["lights"] += 1
            desc.update({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or "light%d" % i})
            self.counts.setdefault("light_list", []).append(desc)
        lux = float(self.o["dome_lux"])
        if lux > 0:
            dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Dome")
            dome.CreateColorAttr(Gf.Vec3f(*[float(c) for c in self.o["dome_rgb"]]))
            dome.CreateIntensityAttr(1.0)
            prim = dome.GetPrim()
            prim.AddAppliedSchema("PhotometricDomeLightAPI")
            self._attr(prim, "photometric:illuminance", "Float", lux)

    # ---- camera and render product ----------------------------------------------------------------------------
    def camera(self, cid, size=None):
        import mujoco
        m, d, P, o = self.m, self.d, self.pxr, self.o
        W, H, fx, fy, cx, cy = CAM.intrinsics(m, cid, size)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid) or "camera%d" % cid
        cam = P["UsdGeom"].Camera.Define(self.stage, "/World/RenderCamera")
        prim = cam.GetPrim()
        R = np.asarray(d.cam_xmat[cid], float).reshape(3, 3)     # MuJoCo and USD cameras both look down -z, y up
        cam.AddTransformOp().Set(P["Gf"].Matrix4d(_matrix(R, d.cam_xpos[cid]).T.tolist()))
        p = LENS_PITCH / W
        cam.CreateProjectionAttr("perspective")
        cam.CreateClippingRangeAttr(P["Gf"].Vec2f(float(o["near"]), float(o["far"])))
        cam.CreateFocalLengthAttr(float(fx * p))
        cam.CreateHorizontalApertureAttr(float(W * p))
        cam.CreateVerticalApertureAttr(float(H * p * fx / fy))
        cam.CreateHorizontalApertureOffsetAttr(float(((W - 1) / 2.0 - cx) * p))
        cam.CreateVerticalApertureOffsetAttr(float((cy - (H - 1) / 2.0) * p * fx / fy))
        cam.CreateFStopAttr(0.0)
        cam.CreateFocusDistanceAttr(1.0)
        # The EV offset scales the exposure time. OVRTX honours the camera's `exposure` attribute as well (measured:
        # -2 on either gives a quarter of the brightness); it stays 0 so that only one of the two carries the offset.
        cam.CreateExposureAttr(0.0)
        cam.CreateExposureIsoAttr(float(o["iso"]))
        cam.CreateExposureTimeAttr(float(o["shutter_s"]) * 2.0 ** float(o["exposure_ev"]))
        cam.CreateExposureFStopAttr(float(o["fstop"]))
        cam.CreateExposureResponsivityAttr(1.0)
        self._attr(prim, "omni:rtx:autoExposure:enabled", "Bool", False)
        if o["lens"] == "opencv":
            prim.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
            k = "omni:lensdistortion:opencvPinhole:"
            self._attr(prim, "omni:lensdistortion:model", "Token", "opencvPinhole")
            self._attr(prim, k + "imageSize", "Int2", P["Gf"].Vec2i(W, H))
            dc = float(o["principal_offset_px"])
            for key, val in (("fx", fx), ("fy", fy), ("cx", cx + dc), ("cy", cy + dc)):
                self._attr(prim, k + key, "Float", float(val))
            for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"):
                self._attr(prim, k + key, "Float", 0.0)
        return {"name": name, "W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "position": [float(v) for v in d.cam_xpos[cid]], "xmat": [float(v) for v in d.cam_xmat[cid]]}

    def render_product(self, W, H):
        P, o = self.pxr, self.o
        UsdRender, Sdf = P["UsdRender"], P["Sdf"]
        UsdGeomScope = P["UsdGeom"].Scope
        UsdGeomScope.Define(self.stage, "/Render")
        UsdGeomScope.Define(self.stage, "/Render/Vars")
        rvars = []
        for nm in ("LdrColor", "HdrColor"):
            v = UsdRender.Var.Define(self.stage, "/Render/Vars/" + nm)
            v.CreateSourceNameAttr(nm)
            rvars.append(v.GetPath())
        prod = UsdRender.Product.Define(self.stage, "/Render/Product")
        prod.CreateCameraRel().SetTargets([Sdf.Path("/World/RenderCamera")])
        prod.CreateResolutionAttr(P["Gf"].Vec2i(W, H))
        prod.CreateOrderedVarsRel().SetTargets(rvars)
        prim = prod.GetPrim()
        spp = int(o["spp"])
        self._attr(prim, "omni:rtx:rendermode", "Token", "PathTracing")
        self._attr(prim, "omni:rtx:pt:samplesPerPixel", "UInt", spp)
        self._attr(prim, "omni:rtx:pt:samplesPerIteration", "Int", int(min(spp, 32)))
        self._attr(prim, "omni:rtx:pt:adaptiveSampling:enabled", "Bool", False)
        self._attr(prim, "omni:rtx:pt:denoising:enabled", "Bool", bool(o["denoise"]))
        self._attr(prim, "omni:rtx:pt:radianceCache:enabled", "Bool", False)
        self._attr(prim, "omni:rtx:pt:limits:maxBounces", "UInt", int(o["max_bounces"]))
        self._attr(prim, "omni:rtx:pt:multigpu:enabled", "Bool", False)
        self._attr(prim, "deviceIds", "UIntArray", P["Vt"].UIntArray([0]))

    def save(self):
        self.stage.GetRootLayer().Save()


def export_scene(m, d, out_dir, camera="photo_cam", size=None, groups=(0, 1, 2), opts=None):
    """Write <out_dir>/scene.usdc for camera `camera` (a name) of the model `m` in state `d`. Returns a summary."""
    import mujoco
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        raise CAM.CameraError("the model has no camera named %r (cameras: %s)" % (camera, ", ".join(
            n for n in names if n) or "none"))
    ex = Exporter(m, d, out_dir, groups, opts)
    cam = ex.camera(cid, size)
    ex.geoms()
    ex.lights()
    ex.render_product(cam["W"], cam["H"])
    ex.save()
    return {"stage": ex.path, "camera": cam, "counts": ex.counts, "warnings": ex.warnings, "options": ex.o}
