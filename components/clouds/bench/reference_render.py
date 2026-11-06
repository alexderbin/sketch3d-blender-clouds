"""The sample cloud scene one-to-one (library/clouds/refvol/refvol_cloud_reference_scene.xml): camera, sun (2.6; 2.5; 2.3), sky (0.03; 0.07; 0.23), ground 0.2,
cloud σ = 4·ρ per unit, HG g = 0.877, albedo 1. The reference is a frame of the reference render (up to 10⁶ bounces, the .exr alongside) against precomputed light + Eevee;
the reference scene's axes (y up) → Blender (x, −z, y). The numbers are the cloud's brightness over the reference mask (fractions p10/p50/p90, the mean error).
blender … -P core/blender/run.py -- components/clouds/bench/reference_render.py -- <folder> [variants=oct,diff1,diff2] [coarse=4] [sigma=4]"""
import json, math, os, runpy, sys, time
import numpy as np
import bpy
from mathutils import Matrix, Vector

from components.clouds import forms, light
C = {**vars(forms), **vars(light)}
a = sys.argv[sys.argv.index("--") + 1:]; OUT = os.path.abspath(a[0]); opt = dict(x.split("=", 1) for x in a[1:]); os.makedirs(OUT, exist_ok=True)
REF = "library/clouds/refvol"; SIG = float(opt.get("sigma", 4.0)); G = 0.877
ESUN = np.array([2.6, 2.5, 2.3]); LSKY = np.array([0.03, 0.07, 0.23])
M2B = lambda p: Vector((p[0], -p[2], p[1]))
VARS = {"oct": dict(ms_mode="octaves", octaves=5, ms_a=0.62, ms_b=0.5),                  # the former light (octaves + sky and ground by depth)
        "diff1": dict(ms_mode="diff", octaves=1), "diff2": dict(ms_mode="diff", octaves=2, ms_a=0.5, ms_b=0.5),
        "d4": dict(ms_mode="diff", octaves=1, g_ms=G, coarse=4), "d2": dict(ms_mode="diff", octaves=1, g_ms=G, coarse=2),
        "p4": dict(ms_mode="diff", octaves=1, g_ms=G, coarse=4, flux=1, detail=0), "p2": dict(ms_mode="diff", octaves=1, g_ms=G, coarse=2, flux=1, detail=0),
        "c1": dict(ms_mode="diff", octaves=1, g_ms=G, coarse=4, flux=1, detail=1.0), "c2": dict(ms_mode="diff", octaves=1, g_ms=G, coarse=4, flux=1, detail=2.0)}

import openvdb as vdb
g0 = vdb.read(f"{REF}/refvol_cloud_{opt.get('file', 'eighth')}.vdb", "density")
lo, hi = g0.evalActiveVoxelBoundingBox(); dims = tuple(hi[i] - lo[i] + 1 for i in range(3))
arr = np.zeros(dims, np.float32); g0.copyToArray(arr, ijk=lo)
rho = np.ascontiguousarray(arr.transpose(0, 2, 1)[:, ::-1, :]); del arr; vs = float(g0.transform.voxelSize()[0])
loc = M2B(g0.transform.indexToWorld((lo[0], lo[1], hi[2])))                   # the centre of cell rho[0, 0, 0]
S = M2B((0.5826, 0.7660, 0.2717)).normalized()                                # toward the sun
Lgnd = 0.2 * (ESUN * S.z + math.pi * LSKY) / math.pi                           # ground 0.2 under the sun and the sky
P = {"light": {"sun_alt": {"h_max": 1.0, "rgb": [ESUN.tolist()] * 2}, "sky_e_rgb": (math.pi * LSKY).tolist(), "ground_up_rgb": Lgnd.tolist()}}
res = {"n_cells": int(rho.size), "step": round(vs, 3), "angle_to_sun": round(math.degrees(math.asin(S.z)), 1)}

sc = bpy.context.scene
for ob in list(sc.objects): bpy.data.objects.remove(ob)
O, T, U = M2B((648.064, -82.473, -63.856)), M2B((6.021, 100.043, -43.679)), M2B((0.273, 0.962, -0.009))
f = (T - O).normalized(); r_ = f.cross(U).normalized(); u_ = r_.cross(f)
cd = bpy.data.cameras.new("cam"); cd.sensor_fit = "HORIZONTAL"; cd.angle = math.radians(54.43); cd.clip_start = 1.0; cd.clip_end = 1e5
cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
cam.matrix_world = Matrix(((r_.x, u_.x, -f.x, O.x), (r_.y, u_.y, -f.y, O.y), (r_.z, u_.z, -f.z, O.z), (0, 0, 0, 1)))
view_c = float(f.dot(S))
corn = [loc + Vector((i * dims[0], j * dims[2], k * dims[1])) * vs for i in (0, 1) for j in (0, 1) for k in (0, 1)]
dd = [((c_ - O).dot(f)) for c_ in corn]; res["depth"] = [round(min(dd)), round(max(dd))]
w = sc.world or bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = next(n_ for n_ in w.node_tree.nodes if n_.type == "BACKGROUND"); bg.inputs["Color"].default_value = (*LSKY, 1); bg.inputs["Strength"].default_value = 1.0
vt = sc.view_settings; vt.view_transform = "Standard"; vt.look = "None"; vt.exposure = 0; vt.gamma = 1
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 666, 360, 100
sc.render.engine = "BLENDER_EEVEE"; ee = sc.eevee; ee.taa_render_samples = 64
for k_, v_ in (("volumetric_start", max(1.0, min(dd) - 5)), ("volumetric_end", max(dd) + 5), ("volumetric_tile_size", "2"), ("volumetric_samples", 256),
               ("volumetric_sample_distribution", 0.0), ("use_volumetric_shadows", False)):
    if hasattr(ee, k_): setattr(ee, k_, v_)


def material():
    m = bpy.data.materials.new("baked"); nt = m.node_tree
    for n_ in list(nt.nodes): nt.nodes.remove(n_)
    def att(nm):
        n_ = nt.nodes.new("ShaderNodeAttribute"); n_.attribute_name = nm; return n_
    ab = nt.nodes.new("ShaderNodeVolumeAbsorption"); ab.inputs["Color"].default_value = (0, 0, 0, 1)
    dm = nt.nodes.new("ShaderNodeMath"); dm.operation = "MULTIPLY"; dm.inputs[1].default_value = SIG; nt.links.new(att("density").outputs["Fac"], dm.inputs[0]); nt.links.new(dm.outputs[0], ab.inputs["Density"])
    cc = nt.nodes.new("ShaderNodeCombineColor")
    for q, nm in enumerate(("em_r", "em_g", "em_b")): nt.links.new(att(nm).outputs["Fac"], cc.inputs[q])
    em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(cc.outputs[0], em.inputs["Color"])
    ad = nt.nodes.new("ShaderNodeAddShader"); nt.links.new(ab.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
    mo = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(ad.outputs[0], mo.inputs["Volume"]); return m


def pixels(path, wh):
    im = bpy.data.images.load(path); im.colorspace_settings.name = "Non-Color"; px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)[::-1, :, :3]
    bpy.data.images.remove(im)
    if px.shape[1] != wh[0]: k = px.shape[1] // wh[0]; px = px.reshape(wh[1], k, wh[0], k, 3).mean((1, 3))
    return px


ref = pixels(os.path.abspath(f"{REF}/refvol_cloud_reference_scene.exr"), (333, 180)); lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
mask = np.abs(ref - LSKY).sum(-1) > 0.05; lr = lum(ref)[mask]
res["reference"] = {"p10": round(float(np.percentile(lr, 10)), 3), "p50": round(float(np.percentile(lr, 50)), 3), "p90": round(float(np.percentile(lr, 90)), 3)}
MAT = material()
for vn in opt.get("variants", "oct,diff1,diff2").split(","):
    kw = dict(VARS[vn]); kw.setdefault("coarse", int(opt.get("coarse", 4))); info = {}
    if kw.pop("flux", 0): kw["view_dir"] = np.array(-f)
    for ob in [o_ for o_ in sc.objects if o_.type == "VOLUME"]: bpy.data.objects.remove(ob)
    t = time.time(); E = C["light"](rho, np.zeros(3), vs, P, np.array(S), view_c, sigma=SIG, g=G, ground=1.0, info=info, **kw)
    res[f"{vn}_light_s"] = round(time.time() - t, 1); res[f"{vn}_cg"] = info.get("cg_iterations")
    grids = []
    for nm, x in (("density", rho), ("em_r", E[..., 0]), ("em_g", E[..., 1]), ("em_b", E[..., 2])):
        g_ = vdb.FloatGrid(); g_.copyFromArray(np.ascontiguousarray(x)); g_.transform = vdb.createLinearTransform(voxelSize=vs); g_.name = nm; grids.append(g_)
    del E; vp = os.path.join(OUT, f"refvol_{vn}.vdb"); vdb.write(vp, grids=grids); del grids
    v = bpy.data.volumes.new(vn); v.filepath = vp; ob = bpy.data.objects.new(vn, v); sc.collection.objects.link(ob); ob.location = loc
    v.materials.append(MAT)
    for fmt, ext in (("OPEN_EXR", "exr"), ("PNG", "png")):
        sc.render.image_settings.file_format = fmt; sc.render.filepath = os.path.join(OUT, f"{vn}.{ext}")
        t = time.time(); bpy.ops.render.render(write_still=True); res[f"{vn}_eevee_s"] = round(time.time() - t, 1)
    ours = pixels(os.path.join(OUT, f"{vn}.exr"), (333, 180)); lo_ = lum(ours)[mask]
    res[vn] = {"p10": round(float(np.percentile(lo_, 10)), 3), "p50": round(float(np.percentile(lo_, 50)), 3), "p90": round(float(np.percentile(lo_, 90)), 3),
               "error": round(float(np.abs(lo_ - lr).mean()), 3), "shift": round(float((lo_ - lr).mean()), 3)}
    print("RENDER", vn, res[vn], flush=True)
json.dump(res, open(os.path.join(OUT, "times.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps(res, ensure_ascii=False))
