"""Blender part of Clouds: the "absorption + emission" volume material (the light is baked into the grid: em_r, em_g, em_b), the volume from numpy fields
(VDB through Blender Python's openvdb), the build of the step's roles. Only inside Blender."""
import math
import os

import numpy as np
import bpy


PAINTS = ("rim", "leak", "glow")                                                    # paints on top of the physics: their own grids and knobs


def material(sigma=0.06, name="cloud_baked", flash=0.0):
    m = bpy.data.materials.new(name); nt = m.node_tree
    for n_ in list(nt.nodes): nt.nodes.remove(n_)
    def att(nm):
        n_ = nt.nodes.new("ShaderNodeAttribute"); n_.attribute_name = nm; return n_
    ab = nt.nodes.new("ShaderNodeVolumeAbsorption"); ab.inputs["Color"].default_value = (0, 0, 0, 1)
    dm = nt.nodes.new("ShaderNodeMath"); dm.operation = "MULTIPLY"; dm.inputs[1].default_value = sigma
    nt.links.new(att("density").outputs["Fac"], dm.inputs[0]); nt.links.new(dm.outputs[0], ab.inputs["Density"])
    cc = nt.nodes.new("ShaderNodeCombineColor")
    for q, nm in enumerate(("em_r", "em_g", "em_b")): nt.links.new(att(nm).outputs["Fac"], cc.inputs[q])
    em = nt.nodes.new("ShaderNodeEmission")
    fv = nt.nodes.new("ShaderNodeValue"); fv.name = "flash"; fv.outputs[0].default_value = flash      # lightning strength (keys — flashes in time)
    fm = nt.nodes.new("ShaderNodeMath"); fm.operation = "MULTIPLY"; nt.links.new(att("em_f").outputs["Fac"], fm.inputs[0]); nt.links.new(fv.outputs[0], fm.inputs[1])
    fc = nt.nodes.new("ShaderNodeVectorMath"); fc.operation = "SCALE"; fc.inputs[0].default_value = (0.78, 0.84, 1.0); nt.links.new(fm.outputs[0], fc.inputs["Scale"])
    sm = nt.nodes.new("ShaderNodeVectorMath"); sm.operation = "ADD"; nt.links.new(cc.outputs[0], sm.inputs[0]); nt.links.new(fc.outputs[0], sm.inputs[1])
    for pn in PAINTS:                                                          # paint: its own em grids × the knob (default 1); no grid — 0
        pc = nt.nodes.new("ShaderNodeCombineColor")
        for q, ch in enumerate("rgb"): nt.links.new(att(f"{pn}_{ch}").outputs["Fac"], pc.inputs[q])
        pv = nt.nodes.new("ShaderNodeValue"); pv.name = pn; pv.outputs[0].default_value = 1.0
        ps = nt.nodes.new("ShaderNodeVectorMath"); ps.operation = "SCALE"; nt.links.new(pc.outputs[0], ps.inputs[0]); nt.links.new(pv.outputs[0], ps.inputs["Scale"])
        s2 = nt.nodes.new("ShaderNodeVectorMath"); s2.operation = "ADD"; nt.links.new(sm.outputs[0], s2.inputs[0]); nt.links.new(ps.outputs[0], s2.inputs[1]); sm = s2
    nt.links.new(sm.outputs[0], em.inputs["Color"])
    ad = nt.nodes.new("ShaderNodeAddShader"); nt.links.new(ab.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
    mo = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(ad.outputs[0], mo.inputs["Volume"]); return m


def put(name, rho, E, o, vs, loc, mat, folder, coll, flash=None, paints=None):
    """Volume: the grids density + em_* (+ em_f — the lightning glow per unit strength, + <paint>_r/g/b) into a VDB (folder folder), an object in collection
    coll; the grid corner is loc + o."""
    import openvdb as vdb
    grids = []; extra = [(f"{pn}_{ch}", Pp[..., q]) for pn, Pp in (paints or {}).items() for q, ch in enumerate("rgb")]
    for nm, arr in (("density", rho), ("em_r", E[..., 0]), ("em_g", E[..., 1]), ("em_b", E[..., 2])) + ((("em_f", flash),) if flash is not None else ()) + tuple(extra):
        tol_ = 1e-3 * float(np.max(arr)) if nm not in ("density", "em_r", "em_g", "em_b") and arr.size else 0.0   # paints and the flash are sparse: values near zero are not written
        g_ = vdb.FloatGrid(); g_.copyFromArray(np.ascontiguousarray(arr), tolerance=tol_); g_.transform = vdb.createLinearTransform(voxelSize=vs); g_.name = nm; grids.append(g_)
    os.makedirs(folder, exist_ok=True); vp = os.path.join(folder, f"{name}.vdb"); vdb.write(vp, grids=grids)
    vol = bpy.data.volumes.new(name); vol.filepath = vp; ob = bpy.data.objects.new(name, vol); coll.objects.link(ob)
    ob.location = (loc[0] + o[0], loc[1] + o[1], loc[2] + o[2]); vol.materials.append(mat); ob["sk_kind"] = "cloud"; return ob


def build(plan, P, coll, folder, dens=0.35, detail=0.3):
    """The step plan's roles → volumes: the shape of the kind, precomputed light from the light plan P (sun, sky, ground), toward the plan's camera."""
    from components.clouds import api
    mat = material(); cam = np.array(plan["camera"]["pos"]); S = np.array(P["sun"]["dir"]); n = 0
    for i, r in enumerate(plan["roles"]):
        if r["form"] != "mass": continue
        a = math.radians(plan["camera"]["az"] + r["daz"]); loc = (cam[0] + r["d"] * math.cos(a), cam[1] + r["d"] * math.sin(a), r["zb"])
        rho, o, vs = api.form(r, float(np.clip(r["d"] / 300.0, 20.0, 120.0)), dens)
        ctr = np.array(loc) + o + np.array(rho.shape) * vs / 2
        E = api.bake(rho, o, vs, P, S, cam - ctr, z0=r["zb"], detail=detail)
        put(f"cloud_{i:02d}", rho, E, o, vs, loc, mat, folder, coll); n += 1
    return n
