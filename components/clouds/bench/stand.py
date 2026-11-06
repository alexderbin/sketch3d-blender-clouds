"""The "Clouds" test stand — a showcase for evaluation (the clouds component); recipe=<name> — roles from recipes.yaml through the frame solver
(compose.py), the layout check is printed. The older knobs are for trials: a camera on a hill, the horizon at the bottom of the frame, the sky across the whole frame; cumulus
built from the shape of the sample cloud (rotation, reflection, scale) are placed by composition — the hero at 5 km, a medium shot, small ones near the horizon;
precomputed light (diffusion + pattern), the node's air haze by distance (far.transmit transmission, the colour is the sky at the horizon by azimuth,
no brighter than 2 medians of the horizon: a halo near the sun), two ridges as silhouettes in the same haze.
blender … -P core/blender/run.py -- components/clouds/bench/stand.py -- <folder> [recipe=spacious|rich|storm|type_*] [light=day,side,golden,back,after] [ci=0] [n_clouds=26] [gain=1] [kind=standard]
    [dens=1 — a fraction of the reference scene's density] [detail=1] [detail_far=m — the pattern twice as weak] [n_octaves=1] [ms_b=0.5] [haze=1] [composition=fair|streets|towers|rim|frame|masses|sky|sky2|sky5|sky6|sky8|sky11|sky12] [seed=3] [camera_z=400] [tilt] [floor=0] [haze_k=1] [mass=lobes|stamps] [wisps=0] [style=neutral_painted] [vdb=0 — the grids are deleted] [sun_az=° from the axis] [sun_elev=°] [time=<the node's TIME>] [weather=clear|overcast|storm|rain] [style=none] [lightning=N] [lightning_strength=30] [sky_k=1 — the sky light on the clouds (a storm veil ≈ 0.3)] [ground_k=1] [mass_dens=0.35] [ts_blur=1] [floor_bumps=0] [floor_covering=0.88] [bottom=m of bottom feathering] [sink=m — the mass bases into the floor] [silhouette=1 — notan.png] [style=neutral_painted|warm_painted]"""
import json, math, os, runpy, sys, time
import numpy as np
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from core.light import far, bl_far

from components.clouds import forms, light, compose as K
C = {**vars(forms), **vars(light)}
a = sys.argv[sys.argv.index("--") + 1:]; OUT = os.path.abspath(a[0]); opt = dict(x.split("=", 1) for x in a[1:]); os.makedirs(OUT, exist_ok=True)
SIG = 0.06; CAM = Vector((0, 0, float(opt.get("camera_z", 400.0)))); AZC, TILT = 90.0, 14.0
SUN = {"day": ("day", 320.0, 45.0), "side": ("day", 10.0, 30.0), "golden": ("sunset", 60.0, 7.0), "back": ("day", 100.0, 18.0), "after": ("afterglow", 95.0, -1.3)}


def layout(rng, n):
    """A field of cumulus: the hero on the left at 5 km, the others at 6–45 km (more of the distant ones), azimuth ±34°; at the same distance they do not
    occlude each other."""
    out = [(5200.0, 15.0, 2400.0, 1500.0, 0, False, (1.0, 1.0, 1.0))]
    while len(out) < n:
        d = 6000 + 39000 * math.sqrt(rng.random()); daz = rng.uniform(-34, 34); W = rng.uniform(700, 1800); zb = 1300 + rng.uniform(-100, 150)
        hw = math.degrees(W / 2 / d)
        if any(abs(daz - q[1]) < 0.9 * (hw + math.degrees(q[2] / 2 / q[0])) and max(d, q[0]) / min(d, q[0]) < 1.6 for q in out): continue
        out.append((d, daz, W, zb, int(rng.integers(4)), bool(rng.random() < 0.5), (rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3), rng.uniform(0.55, 1.1))))
    return out


LAYOUT = layout(np.random.default_rng(int(opt.get("seed", 3))), int(opt.get("n_clouds", 26)))
MASSES = {   # masses as in paintings: (distance m, azimuth from the axis °, length m, height m, peak, base m, seed); the sun (time, azimuth shift, elevation)
    "masses": ([(20000, 16, 18000, 8000, 0.25, 1500, 1), (32000, -14, 16000, 6500, 0.75, 1450, 2), (40000, -30, 10000, 4000, 0.4, 1400, 3)], ("sunset", -55, 9)),
    "sky": ([(18000, 22, 16000, 7000, 0.3, 1900, 1), (27000, -20, 14000, 6000, 0.65, 1900, 2), (40000, 4, 12000, 5000, 0.5, 1900, 3)], ("sunset", -60, 12)),
    "sky2": ([(9000, 31, 9000, 5500, 0.6, 1900, 11), (24000, -8, 16000, 8000, 0.35, 1900, 12), (15000, -29, 9000, 4500, 0.4, 1900, 13),
              (33000, 15, 10000, 4000, 0.5, 1900, 14)], ("sunset", -62, 12)),
    "sky5": ([(6200, 37, 7000, 6200, 0.5, 1750, 11), (24000, -8, 16000, 8000, 0.35, 1900, 12), (15000, -29, 9000, 4500, 0.4, 1900, 13),
              (33000, 15, 10000, 4000, 0.5, 1900, 14)], ("sunset", -62, 12)),        # like sky2, the near mass is a frame at the left edge
    "sky6": ([(6500, 44, 4500, 5200, 0.5, 1750, 11), (24000, -8, 16000, 8000, 0.35, 1900, 12), (15000, -29, 9000, 4500, 0.4, 1900, 13),
              (33000, 15, 10000, 4000, 0.5, 1900, 14)], ("sunset", -62, 12)),        # layout of sky2, the frame is a narrow one at the left edge
    # roles: (distance, azimuth, length, height, peak, base, seed, mass_lobes keys) — the main one on the right
    # third with a lean, the second one lower and left of the centre overlaps its base, the distant low bank is a
    # calm strip at the horizon, a dark frame at the left edge
    "sky8": ([(19000, -16, 4000, 6500, 0.45, 1700, 21, dict(accents=3, lean=-0.6, kids_frac=0.5)),     # main: the right third (U 0.54–0.93, top 0.84)
              (23000, 1, 3500, 3400, 0.3, 1700, 22, dict(accents=2, lean=0.4, kids_frac=0.4)),       # second: lower, farther, tucks under the main one
              (38000, -2, 36000, 1700, 0.6, 1700, 23, dict(accents=4, kids_frac=0.3, aspect=0.9)),   # a bank at the horizon
              (6000, 47, 600, 5200, 0.5, 1650, 24, dict(accents=2, lean=0.5, kids_frac=0.5))], ("sunset", -62, 12)),   # frame: the edge ≈ 0.12 of the frame
    "sky11": ([(19000, -17, 3600, 7600, 0.5, 1700, 21, dict(tower=-14, accents=3, kids_frac=0.5, face=0.25)),   # main — a tower leaning toward the centre
               (23000, 1, 3500, 3400, 0.3, 1700, 22, dict(accents=2, lean=0.4, kids_frac=0.4, face=0.25)),
               (38000, -2, 36000, 1700, 0.6, 1700, 23, dict(accents=4, kids_frac=0.3, aspect=0.9, face=0.25)),
               (6000, 50, 400, 4200, 0.5, 1650, 24, dict(accents=2, lean=0.5, kids_frac=0.4, aspect=0.32, face=0.25))], ("sunset", -62, 12)),   # a lighter frame
    "sky12": ([(19000, -17, 4200, 7600, 0.5, 1700, 21, dict(tower=-12, accents=3, kids_frac=0.5, face=0.25)),   # main — a tower, the right third
               (12500, 24, 5000, 3600, 0.35, 1650, 25, dict(accents=3, lean=-0.4, kids_frac=0.45, face=0.25)),   # counterweight: a large low one on the left
               (24000, 0, 3000, 2600, 0.4, 1700, 22, dict(accents=2, lean=0.3, kids_frac=0.4, face=0.25)),       # third: small, between them, farther
               (38000, -2, 36000, 1700, 0.6, 1700, 23, dict(accents=4, kids_frac=0.3, aspect=0.9, face=0.25))], ("sunset", -62, 12)),
    "masses_day": ([(20000, 16, 18000, 8000, 0.25, 1500, 1), (32000, -14, 16000, 6500, 0.75, 1450, 2), (40000, -30, 10000, 4000, 0.4, 1400, 3)], ("day", -75, 32))}
ROT = {}                                                                  # rotation of the masses about the vertical, °
if opt.get("composition") in ("vortex", "canyon"):                        # vortex — a ring of walls around the camera; canyon — two rows of walls along the view
    TILT = float(opt.get("tilt", 8.0)); SUN["comp"] = ("sunset", (AZC + float(opt.get("sun_az", -40.0))) % 360, float(opt.get("sun_elev", 10.0)))
if "recipe" in opt:                                                       # component recipe: roles in fractions of the frame → the world (compose.solve)
    CC, ROLES, RR = K.layout(opt["recipe"]); sc_ = RR.get("scene", {})
    for w_ in K.check(CC, ROLES): print("LAYOUT:", w_, flush=True)
    for k_, v_ in (("camera_z", RR["camera"].get("z", 3000)), ("tilt", RR["camera"].get("tilt", 3)), ("style", RR.get("style", "warm_painted")), ("haze_k", RR.get("haze", 2.0)),
                   ("sun_az", RR["sun"][0]), ("sun_elev", None if "time" in opt else RR["sun"][1]), ("floor", 1 if sc_.get("floor") else 0), ("floor_covering", sc_.get("floor", 0.95)),
                   ("floor_bumps", sc_.get("humps", 0)), ("wisps", sc_.get("wisps", 0)), ("sink", sc_.get("sink", 0)), ("mass_dens", 0.35), ("ts_blur", 2),
                   ("detail", 0.3), ("detail_far", 8000)):
        if v_ is not None: opt.setdefault(k_, str(v_))
    CAM = Vector((0, 0, float(opt["camera_z"])))
    MASSES["__recipe"] = ([(r["d"], r["daz"], r["L"], r["H"], r["peak"], r["zb"], r["seed"], r["kw"]) for r in ROLES if r["form"] == "mass"], ("sunset", 0, 0))
    LAYERS = [r for r in ROLES if r["form"] != "mass"]; opt["composition"] = "__recipe"
if opt.get("composition") in MASSES:                                      # masses: their own SH layers, the field — a few small ones in front
    TILT = float(opt.get("tilt", 15.0)); ML, (tm_, daz_, el_) = MASSES[opt["composition"]]
    SUN["comp"] = (tm_, (AZC + float(opt.get("sun_az", daz_))) % 360, float(opt.get("sun_elev", el_)))
    LAYOUT = [(6500, -6, 700, 1350, 1, False, (1.1, 1.0, 0.6)), (7800, 30, 900, 1400, 2, True, (1.0, 1.2, 0.7)), (9000, -28, 600, 1350, 3, False, (1.0, 1.0, 0.55))]
sc = bpy.context.scene
for ob in list(sc.objects): bpy.data.objects.remove(ob)
cd = bpy.data.cameras.new("cam"); cd.lens = float(opt.get("lens", 28)); cd.clip_end = 200000; cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
f = Vector((math.cos(math.radians(TILT)) * math.cos(math.radians(AZC)), math.cos(math.radians(TILT)) * math.sin(math.radians(AZC)), math.sin(math.radians(TILT))))
cam.location = CAM; cam.rotation_euler = f.to_track_quat("-Z", "Y").to_euler()


def ridge(name, R, h0, h1, seed, n=600):
    """A ridge as a silhouette at distance R: the profile is a sum of sines (h0…h1 m), the glow = ground·T + haze·(1 − T) (set by the light)."""
    r_ = np.random.default_rng(seed); az = np.radians(np.linspace(AZC - 55, AZC + 55, n)); u = np.linspace(0, 1, n)
    prof = sum(r_.uniform(0.3, 1.0) / k * np.sin(2 * math.pi * (k * 1.7 * u + r_.random())) for k in (1, 2, 3, 5, 8, 13, 21))
    h = h0 + (h1 - h0) * (prof - prof.min()) / (prof.max() - prof.min())
    V = [(R * math.cos(a_), R * math.sin(a_), z_) for a_, hh in zip(az, h) for z_ in (-3000.0, float(hh))]   # the bottom is below the horizon: otherwise there is a gap down to the bottom of the world
    F = [(2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1) for i in range(n - 1)]
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    m = bpy.data.materials.new(name); nt = m.node_tree
    for n_ in list(nt.nodes): nt.nodes.remove(n_)
    em = nt.nodes.new("ShaderNodeEmission"); mo = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], mo.inputs["Surface"])
    me.materials.append(m); return ob, em, R, float(h.mean())


RIDGES = [ridge("ridge_fore", 3500.0, 180.0, 390.0, 13), ridge("ridge_near", 14000.0, 150.0, 650.0, 11), ridge("ridge_far", 27000.0, 400.0, 1400.0, 12)]
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1600, 900, 100
sc.render.engine = "BLENDER_EEVEE"; ee = sc.eevee; ee.taa_render_samples = 64
for k_, v_ in (("volumetric_start", 2000.0), ("volumetric_end", 50000.0), ("volumetric_tile_size", "2"), ("volumetric_samples", 256), ("use_volumetric_shadows", False)):
    if hasattr(ee, k_): setattr(ee, k_, v_)


def material():
    m = bpy.data.materials.new("cloud_baked"); nt = m.node_tree
    for n_ in list(nt.nodes): nt.nodes.remove(n_)
    def att(nm):
        n_ = nt.nodes.new("ShaderNodeAttribute"); n_.attribute_name = nm; return n_
    ab = nt.nodes.new("ShaderNodeVolumeAbsorption"); ab.inputs["Color"].default_value = (0, 0, 0, 1)
    dm = nt.nodes.new("ShaderNodeMath"); dm.operation = "MULTIPLY"; dm.inputs[1].default_value = SIG
    nt.links.new(att("density").outputs["Fac"], dm.inputs[0]); nt.links.new(dm.outputs[0], ab.inputs["Density"])
    cc = nt.nodes.new("ShaderNodeCombineColor")
    for q, nm in enumerate(("em_r", "em_g", "em_b")): nt.links.new(att(nm).outputs["Fac"], cc.inputs[q])
    em = nt.nodes.new("ShaderNodeEmission")
    fv = nt.nodes.new("ShaderNodeValue"); fv.name = "flash"; fv.outputs[0].default_value = float(opt.get("lightning_strength", 0.0))   # flash strength (keys — animation)
    fm = nt.nodes.new("ShaderNodeMath"); fm.operation = "MULTIPLY"; nt.links.new(att("em_f").outputs["Fac"], fm.inputs[0]); nt.links.new(fv.outputs[0], fm.inputs[1])
    fc = nt.nodes.new("ShaderNodeVectorMath"); fc.operation = "SCALE"; fc.inputs[0].default_value = (0.78, 0.84, 1.0); nt.links.new(fm.outputs[0], fc.inputs["Scale"])
    sm = nt.nodes.new("ShaderNodeVectorMath"); sm.operation = "ADD"; nt.links.new(cc.outputs[0], sm.inputs[0]); nt.links.new(fc.outputs[0], sm.inputs[1])
    nt.links.new(sm.outputs[0], em.inputs["Color"])
    ad = nt.nodes.new("ShaderNodeAddShader"); nt.links.new(ab.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
    mo = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(ad.outputs[0], mo.inputs["Volume"]); return m


def put(name, rho, o, vs, loc, E, flash=None):
    import openvdb as vdb
    grids = []
    for nm, arr in (("density", rho), ("em_r", E[..., 0]), ("em_g", E[..., 1]), ("em_b", E[..., 2])) + ((("em_f", flash),) if flash is not None else ()):
        g_ = vdb.FloatGrid(); g_.copyFromArray(np.ascontiguousarray(arr)); g_.transform = vdb.createLinearTransform(voxelSize=vs); g_.name = nm; grids.append(g_)
    vp = os.path.join(OUT, "vdb", f"{name}.vdb"); os.makedirs(os.path.dirname(vp), exist_ok=True); vdb.write(vp, grids=grids)
    vol = bpy.data.volumes.new(name); vol.filepath = vp; ob = bpy.data.objects.new(name, vol); sc.collection.objects.link(ob)
    ob.location = (loc[0] + o[0], loc[1] + o[1], loc[2] + o[2]); vol.materials.append(MAT); return ob


def haze_lit(cam_p, ctr, S, rh, rv, n=16):
    """The fraction of the haze in front of a cloud that is lit by the sun: points along the ray camera → cloud (the weight is the haze density at the point),
    a ray to the sun from each; the cloud is an ellipsoid (rh horizontally, rv vertically). Against the sun the air in front of the cloud lies in its shadow:
    without this the haze flooded the cloud with an even halo light."""
    w_, l_ = 0.0, 0.0
    for u in (np.arange(n) + 0.5) / n:
        p_ = cam_p + u * (ctr - cam_p); q = (p_ - ctr) / np.array([rh, rh, rv]); s_ = np.asarray(S) / np.array([rh, rh, rv])
        b_ = q @ s_; c_ = q @ q - 1.0; disc = b_ * b_ - (s_ @ s_) * c_
        hit = disc > 0 and (-b_ + math.sqrt(disc)) / (s_ @ s_) > 0
        wk = math.exp(-p_[2] / 1500.0); w_ += wk; l_ += wk * (0.0 if hit else 1.0)
    return l_ / max(w_, 1e-9)


# styles from reference palettes (sRGB → linear): neutral_painted — a painted cloud sky and Aivazovsky's
# "Clouds over a Calm Sea": cloud light ≈ (205, 195, 170), shadow ≈ (135, 128, 132), sky at the zenith ≈ (155,
# 172, 185), at the horizon ≈ (200, 195, 185)
STYLES = {"warm_painted": dict(sun=(1.85, 1.45, 0.92), amb=(0.23, 0.24, 0.31), gnd=(0.48, 0.42, 0.34), zen=(0.17, 0.33, 0.56), hor=(0.62, 0.55, 0.45)),
          "neutral_painted": dict(sun=(1.6, 1.42, 1.08), amb=(0.27, 0.25, 0.29), gnd=(0.45, 0.42, 0.38), zen=(0.17, 0.33, 0.56), hor=(0.578, 0.546, 0.485))}   # the zenith is bluer than the palette: the frame shows the top of the sky


def style_world(ST):
    """Sky of a style: a gradient from the horizon to the zenith by the elevation of the direction (the Standard kind — colours as in the palette)."""
    w = sc.world; nt = w.node_tree
    for n_ in list(nt.nodes): nt.nodes.remove(n_)
    tc = nt.nodes.new("ShaderNodeTexCoord"); sx = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tc.outputs["Generated"], sx.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["From Min"].default_value = -0.15; mr.inputs["From Max"].default_value = 0.55; nt.links.new(sx.outputs["Z"], mr.inputs["Value"])
    cr = nt.nodes.new("ShaderNodeValToRGB"); el = cr.color_ramp.elements
    below = tuple(0.5 * h + 0.5 * g * 0.8 for h, g in zip(ST["hor"], ST["gnd"]))   # below the horizon — a dim haze above the clouds (holes in the floor show no sky)
    el[0].color = (*below, 1); el[1].color = (*ST["zen"], 1); e_ = el.new(0.15 / 0.70); e_.color = (*ST["hor"], 1)
    nt.links.new(mr.outputs["Result"], cr.inputs["Fac"]); bg = nt.nodes.new("ShaderNodeBackground"); nt.links.new(cr.outputs["Color"], bg.inputs["Color"])
    wo = nt.nodes.new("ShaderNodeOutputWorld"); nt.links.new(bg.outputs[0], wo.inputs["Surface"])
    for o_ in [o_ for o_ in sc.objects if o_.type == "LIGHT"]: o_.hide_render = True


def sky_probe(path):
    """Clear sky (a linear .exr, without clouds) — the colour of the haze at the horizon by azimuth."""
    sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.filepath = path; rx, ry = sc.render.resolution_x, sc.render.resolution_y
    sc.render.resolution_x, sc.render.resolution_y = 320, 180; bpy.ops.render.render(write_still=True); sc.render.resolution_x, sc.render.resolution_y = rx, ry
    im = bpy.data.images.load(path); px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)[..., :3]; bpy.data.images.remove(im); return px


MAT = material(); WD = C["refvol"](); res = {}; t = time.time(); SH = []
for i, (d, daz, W, zb, k, fl, st) in enumerate(LAYOUT):
    az = math.radians(AZC + daz); x, y = d * math.cos(az), d * math.sin(az); vs = float(np.clip(d / 400.0, 8.0, 60.0))
    if st[2] > 1.2: rho, o, vs = C["refvol_tower"](WD, W, vs, np.random.default_rng(i), int(round(st[2] + 0.5)), float(opt.get("dens", 1.0)), SIG)   # a towering one — in tiers
    else: rho, o, vs = C["refvol_cloud"](WD, W, vs, k, fl, float(opt.get("dens", 1.0)), SIG, st)
    SH.append((f"wd{i:02d}", rho, o, vs, (x, y, zb)))
if opt.get("composition") == "canyon":                                     # canyon: rows of walls at the sides, narrowing and rising into the distance; the sun at the end
    half_w = float(opt.get("width", 3200.0)) / 2
    for side in (-1, 1):
        y_ = 2500.0; j = 0
        while y_ < float(opt.get("length", 32000.0)):
            L_ = float(opt.get("step", 4200.0)) * (1 + 0.15 * j); H_ = float(opt.get("height", 6500.0)) * (0.7 + 0.5 * min(1.0, y_ / 25000.0)) * (0.85 + 0.3 * ((j * 7 + (side > 0) * 3) % 5) / 4)
            xw = side * (half_w + 0.35 * L_ * 0.45 + 0.04 * y_)                     # the canyon's opening grows slightly with distance
            rho, o, vs = C["mass_lobes"](L_, H_, np.random.default_rng(300 + j + (side > 0) * 50), float(np.clip(math.hypot(xw, y_) / 300.0, 20.0, 120.0)), 0.4,
                                         dens=float(opt.get("mass_dens", 0.35)), base_soft=float(opt.get("bottom", 0.0)),
                                         levels=int(opt.get("levels", 3)) + (1 if math.hypot(xw, y_) < 9000 else 0))   # the near ones have smaller puffs
            nm_ = f"mass{len(ROT) + 100}"; SH.append((nm_, rho, o, vs, (xw, y_, 1900.0 - float(opt.get("sink", 0.0))))); ROT[nm_] = 90.0
            y_ += 0.8 * L_; j += 1
if opt.get("composition") == "vortex":
    N_ = int(opt.get("walls", 9)); Rr = float(opt.get("radius", 10000.0)); cy_ = float(opt.get("center", 3000.0))
    for j in range(N_):                                                   # walls along the tangent, the height grows around the circle — one turn of a spiral
        a_ = 2 * math.pi * j / N_ + 0.35; px, py = Rr * math.cos(a_), cy_ + Rr * math.sin(a_)
        sp_ = float(opt.get("spiral", 0.6))                              # the coil's amplitude: 0 — walls of one height
        L_ = 2 * math.pi * Rr / N_ * float(opt.get("overlap", 1.2)); H_ = float(opt.get("height", 5200.0)) * (1.15 - sp_ + sp_ * j / N_)
        rho, o, vs = C["mass_lobes"](L_, H_, np.random.default_rng(200 + j), float(np.clip(math.hypot(px, py) / 300.0, 20.0, 120.0)), 0.4,
                                     dens=float(opt.get("mass_dens", 0.35)), base_soft=float(opt.get("bottom", 0.0)))
        SH.append((f"mass{j}", rho, o, vs, (px, py, 1900.0 - float(opt.get("sink", 0.0))))); ROT[f"mass{j}"] = math.degrees(a_) + 90.0
if opt.get("composition") in MASSES:
    for j, (d, daz, L, Hm, pk, zb, sd, *kw_) in enumerate(ML):
        az = math.radians(AZC + daz); vs = float(np.clip(d / 300.0, 20.0, 120.0))
        if opt.get("mass", "lobes") == "lobes": rho, o, vs = C["mass_lobes"](L, Hm, np.random.default_rng(sd), vs, pk, dens=float(opt.get("mass_dens", 0.35)),
                                                                                base_soft=float(opt.get("bottom", 0.0)), **(kw_[0] if kw_ else {}))   # a silhouette of large soft puffs; cell τ < 1 — no banding
        else: rho, o, vs = C["cloud_mass2"](WD, L, Hm, np.random.default_rng(sd), vs, pk, float(opt.get("dens", 1.0)), SIG)
        SH.append((f"mass{j}", rho, o, vs, (d * math.cos(az), d * math.sin(az), zb - float(opt.get("sink", 0.0)))))
    if opt.get("floor_bumps", "0") == "1":                                   # bumps of the floor near the camera: the "sea of clouds" is not flat
        for j, (d, daz, L_) in enumerate(((10000, -26, 5000), (14000, 8, 6000), (17000, 30, 5500), (21000, -14, 7000))):   # below the frame's horizon
            az = math.radians(AZC + daz); rho, o, vs = C["mass_lobes"](L_, 0.08 * L_, np.random.default_rng(60 + j), float(np.clip(d / 300.0, 15.0, 60.0)), 0.5, 5,
                                                                         dens=float(opt.get("mass_dens", 0.35)), levels=2, aspect=2.5, accents=2, kids_frac=0.5)   # gentle, not columns
            SH.append((f"hump{j}", rho, o, vs, (d * math.cos(az), d * math.sin(az), 1950.0)))
    if opt.get("wisps", "0") == "1":                                      # wisps of haze in front of the camera: small, soft and sparse
        for j, (d, daz, L_, H_, zc) in enumerate(((3000, -24, 700, 350, 2250), (4200, 18, 900, 400, 2300), (5600, -4, 800, 350, 2150))):   # below the frame's horizon, small
            az = math.radians(AZC + daz); rho, o, vs = C["mass_lobes"](L_ * 1.6, H_ * 0.5, np.random.default_rng(40 + j), 12.0, 0.5, 3, dens=0.1, levels=2, aspect=2.8, accents=2, kids_frac=0.4)   # flat sparse strands
            SH.append((f"wisp{j}", rho, o, vs, (d * math.cos(az), d * math.sin(az), zc)))
del WD
if opt.get("floor", "0") == "1":                                           # a floor of clouds under the camera: a layer of cells
    rho, o, vs = C["strato_layer"](6, ext=70000.0, thick=600.0, cell=1600.0, cover=float(opt.get("floor_covering", 0.88)), vs=60.0, erode=0.4, size_var=0.5); SH.append(("floor_layer", rho, o, vs, (0.0, 28000.0, 1500.0)))
if opt.get("ci", "0") == "1": rho, o, vs = C["cirrus_layer"](9, ext=60000.0, cover=0.45); SH.append(("ci_layer", rho, o, vs, (0.0, 30000.0, 8500.0)))
res["shapes_s"] = round(time.time() - t, 1); res["n_cells"] = int(sum(s[1].size for s in SH))
for li in opt.get("light", "comp" if "composition" in opt else "day,side,golden,back,after").split(","):
    tm, az, el = SUN[li]
    if opt.get("time"): tm = opt["time"]; el = float(opt.get("sun_elev", far.TIME[tm][0]))   # time of day of the node: the sun elevation comes from TIME
    P = far.resolve(far.load_spec("scenes/loc_fort"), "scenes/loc_fort", {"time": tm, "weather": opt.get("weather", "clear"), "sun_az": az, "sun_el": el})
    for ob in [o_ for o_ in sc.objects if o_.type in ("LIGHT", "VOLUME")]: bpy.data.objects.remove(ob)
    for ob_, em_, R_, h_ in RIDGES: ob_.hide_render = True
    bl_far.attach(sc, P); S = np.array(P["sun"]["dir"]); sky = sky_probe(os.path.join(OUT, f"sky_{li}.exr"))
    ST = None if opt.get("style") == "none" else STYLES.get(opt.get("style", ""))
    if ST:                                                                 # style: the cloud light and the sky as palette numbers (linear)
        import copy
        P = copy.deepcopy(P); L_ = P["light"]; L_["sun_alt"] = {"h_max": 1.0, "rgb": [list(ST["sun"])] * 2}; L_["sky_e_rgb"] = [math.pi * v for v in ST["amb"]]
        L_["night_sky_e_rgb"] = [0.0, 0.0, 0.0]; L_["ground_up_rgb"] = list(ST["gnd"]); style_world(ST)
        sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"; sc.view_settings.exposure = 0.0
    if opt.get("kind") == "standard": sc.view_settings.view_transform = "Standard"                   # tone test: sRGB without the AgX shoulder
    p0 = world_to_camera_view(sc, cam, Vector((0, 10000.0, CAM.z + 260.0))); row = sky[int(np.clip(p0.y, 0, 1) * (sky.shape[0] - 1))]
    cap = 2.0 * float(np.median(row @ np.array([0.2126, 0.7152, 0.0722])))                         # haze no brighter than 2 medians of the horizon
    hz_col = lambda p_: (lambda L_: L_ * min(1.0, cap / max(float(L_ @ np.array([0.2126, 0.7152, 0.0722])), 1e-9)))(sky[int(np.clip(p_.y, 0, 1) * (sky.shape[0] - 1)), int(np.clip(p_.x, 0, 1) * (sky.shape[1] - 1))])
    Lg = np.array(P["light"]["ground_up_rgb"]); Lmed = np.median(row, 0) * min(1.0, cap / 2 / max(float(np.median(row, 0) @ np.array([0.2126, 0.7152, 0.0722])), 1e-9))
    for ob_, em_, R_, h_ in ([] if CAM.z > 1500 else RIDGES):             # there are no ridges above the clouds (below the floor)
        T_ = far.transmit(P["fog"]["layers"], float(CAM.z), R_, h_ - float(CAM.z)); c_ = Lg * T_ + Lmed * (1 - T_)
        em_.inputs["Color"].default_value = (*c_, 1); em_.inputs["Strength"].default_value = 1.0; ob_.hide_render = False
    PB = P                                                                   # light for baking the clouds: a veil above the storm — the sky is weaker
    if opt.get("sky_k"):
        import copy; PB = copy.deepcopy(P); k_ = float(opt["sky_k"])
        PB["light"]["sky_e_rgb"] = [k_ * v for v in PB["light"]["sky_e_rgb"]]; PB["light"]["night_sky_e_rgb"] = [k_ * v for v in PB["light"].get("night_sky_e_rgb", [0, 0, 0])]
    t = time.time()
    for name, rho, o, vs, loc in SH:
        half = np.array(rho.shape) * vs / 2; ctr = np.array(loc) + o + half; v = ctr - np.array(CAM); rr_ = math.radians(ROT.get(name, 0.0))
        Rz = np.array([[math.cos(rr_), -math.sin(rr_), 0], [math.sin(rr_), math.cos(rr_), 0], [0, 0, 1]]); S_l, v_l = Rz.T @ S, Rz.T @ v
        c = float(v_l / np.linalg.norm(v_l) @ S_l)
        dd = float(np.linalg.norm(v)); det = float(opt.get("detail", 1.0)) / (1.0 + (dd / float(opt.get("detail_far", 1e9))) ** 2)   # the pattern fades with distance
        fl = []; info_ = {}
        if int(opt.get("lightning", 0)) and name.startswith("mass") and int(name[4:]) < 2:   # lightning — in the main and the second mass
            rg = np.random.default_rng(7 + int(name[4:])); occ = np.argwhere((rho > 0.25) & (np.arange(rho.shape[2])[None, None, :] < 0.45 * rho.shape[2]))
            fl = [(tuple(occ[i_]), 3.0) for i_ in rg.choice(len(occ), min(int(opt["lightning"]), len(occ)), replace=False)] if len(occ) else []
        E = C["light"](rho, o, vs, PB, S_l, c, sigma=SIG, z0=loc[2], octaves=int(opt.get("n_octaves", 1)), g=0.85, ms_a=0.5, ms_b=float(opt.get("ms_b", 0.5)), ms_mode="diff", flashes=fl, info=info_,
                       coarse=2 if name.endswith("_layer") else 4, view_dir=-v_l, ground=float(opt.get("ground_k", 1.0)), gain=float(opt.get("gain", 1.0)), detail=det,
                       ts_blur=int(opt.get("ts_blur", 1)))
        T = far.transmit(P["fog"]["layers"], float(CAM.z), float(math.hypot(v[0], v[1])), float(v[2])) ** float(opt.get("haze_k", 1.0)) if opt.get("haze", "1") == "1" else 1.0   # air haze in front of the cloud
        hz = Vector((ctr[0], ctr[1], CAM.z + 0.026 * math.hypot(v[0], v[1])))                         # the sky at the horizon (1.5°) at the cloud's azimuth
        Lh = np.array(ST["hor"]) if ST else hz_col(world_to_camera_view(sc, cam, hz))
        lit = haze_lit(np.array(CAM), ctr, S, 0.42 * max(rho.shape[0], rho.shape[1]) * vs, 0.35 * rho.shape[2] * vs)
        E = E * T + (SIG * rho[..., None] * (1 - T) * (Lh * (0.35 + 0.65 * lit))[None, None, None, :]).astype(np.float32)
        res.setdefault(f"{li}_haze_light", {})[name] = round(lit, 2)
        res.setdefault(f"{li}_haze_T", {})[name] = round(T, 3)
        FL = info_.get("flash")
        if FL is not None: FL = FL * T                                       # the flash is dimmed by the haze, like the light
        ob_ = put(f"{name}_{li}", rho, o, vs, loc, E, FL); del E
        if name in ROT: ob_.rotation_euler = (0.0, 0.0, rr_); ob_.location = tuple(ctr - Rz @ half)   # rotation about the grid centre
    res[f"{li}_light_s"] = round(time.time() - t, 1)
    sc.render.image_settings.file_format = "PNG"; sc.render.filepath = os.path.join(OUT, f"show_{li}.png"); t = time.time(); bpy.ops.render.render(write_still=True)
    res[f"{li}_eevee_s"] = round(time.time() - t, 1); print("RENDER", li, res[f"{li}_eevee_s"], flush=True)
if opt.get("silhouette", "0") == "1":                                       # composition check (notan): patches by distance — closer is darker, without light
    for ob in [o_ for o_ in sc.objects if o_.type == "VOLUME"]:
        d_ = float((ob.matrix_world.translation - CAM).length); c_ = float(np.clip(0.06 + 0.55 * math.log(max(d_, 2000) / 2000) / math.log(25), 0.06, 0.62))
        m = bpy.data.materials.new("flat"); nt = m.node_tree
        for n_ in list(nt.nodes): nt.nodes.remove(n_)
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "density"; mk = nt.nodes.new("ShaderNodeMath"); mk.operation = "MULTIPLY"; mk.inputs[1].default_value = 2.0
        nt.links.new(at.outputs["Fac"], mk.inputs[0]); ab = nt.nodes.new("ShaderNodeVolumeAbsorption"); ab.inputs["Color"].default_value = (0, 0, 0, 1); nt.links.new(mk.outputs[0], ab.inputs["Density"])
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (c_, c_, c_, 1); nt.links.new(mk.outputs[0], em.inputs["Strength"])
        ad = nt.nodes.new("ShaderNodeAddShader"); nt.links.new(ab.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        mo = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(ad.outputs[0], mo.inputs["Volume"]); ob.data.materials.clear(); ob.data.materials.append(m)
    w = sc.world.node_tree
    for n_ in list(w.nodes): w.nodes.remove(n_)
    bg = w.nodes.new("ShaderNodeBackground"); bg.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1); wo = w.nodes.new("ShaderNodeOutputWorld"); w.links.new(bg.outputs[0], wo.inputs["Surface"])
    for ob_, *_ in RIDGES: ob_.hide_render = True
    sc.view_settings.view_transform = "Standard"; sc.view_settings.exposure = 0.0; sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = os.path.join(OUT, "notan.png"); bpy.ops.render.render(write_still=True)
if opt.get("vdb", "0") != "1":                                           # the grids are intermediate (rebuilt within seconds): not kept on disk
    import shutil; shutil.rmtree(os.path.join(OUT, "vdb"), ignore_errors=True)
json.dump(res, open(os.path.join(OUT, "times_show.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps({k: v for k, v in res.items() if "haze" not in k}, ensure_ascii=False))
