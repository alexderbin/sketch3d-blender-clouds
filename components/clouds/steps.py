"""The "clouds" step: a composition recipe (recipes.yaml) relative to the scene camera → roles in the world (plan, Python); the shapes of the kinds, precomputed light
from the scene's light plan, Eevee volumes (build, Blender). The composition is defined in the frame, so without a camera the step is skipped."""
import json
import math
import os

import numpy as np


def plan_clouds(M, keys, ctx):
    from components.clouds import compose
    from core import layers
    from core.light import far
    cams = layers.cameras(ctx["scene"])
    if not cams: return {"skipped": "no cameras: the cloud composition is defined in a camera's frame"}
    c0 = cams[int(keys.get("camera") or 0) % len(cams)]; Mw = np.asarray(c0["matrix_world"], float); f = -Mw[:3, 2]
    cam = compose.Cam(float(Mw[2, 3]), math.degrees(math.atan2(f[1], f[0])), math.degrees(math.asin(np.clip(f[2], -1, 1))), float(c0.get("lens_mm", 28.0)))
    R = compose.load("recipes.yaml")[keys["recipe"]]; types = compose.load("types.yaml")
    roles = [dict(compose.solve(cam, r, R.get("base", 1700.0), types), name=r.get("name", r["type"])) for r in R["roles"]]
    fp = os.path.join(ctx["out"], "far_plan.json"); far.plan_of(ctx["scene"], {}, out=fp)
    plan = {"recipe": keys["recipe"], "camera": {"pos": Mw[:3, 3].tolist(), "az": cam.az, "tilt": cam.tilt, "lens": cam.k * 18.0},
            "roles": roles, "far_plan": fp, "warnings": compose.check(cam, roles), "density": keys.get("density", 0.35), "detail": keys.get("detail", 0.3)}
    low = [r["name"] for r in roles if r["H"] <= 200.0]
    if low: plan["warnings"].append(f"the tops of the roles are below the camera's horizon (tilt {cam.tilt:.0f}°): {', '.join(low)} — the recipe is meant for a view into the sky")
    json.dump(plan, open(os.path.join(ctx["out"], "clouds_plan.json"), "w"), ensure_ascii=False, indent=1, default=float)
    return plan


def build_clouds(ctx):
    from components.clouds import bl_clouds
    plan = ctx["plan"]
    if plan.get("skipped"): return {"skipped": plan["skipped"]}
    P = json.load(open(plan["far_plan"]))
    n = bl_clouds.build(plan, P, ctx["collection"], os.path.join(ctx["out"], "vdb"), plan["density"], plan["detail"])
    return {"clouds": n, "warnings": plan["warnings"]}
