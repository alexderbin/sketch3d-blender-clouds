"""Public entry of Clouds (ARCHITECTURE.md: neighbours use only api). Without Blender: recipe → roles in the world, the shape of a kind → a density
field, precomputed light → emission for an "absorption + emission" volume. The Blender part is the test stand (bench/stand.py)."""
import numpy as np

from components.clouds import compose, forms, light as _light


def layout(recipe):
    """A recipe from recipes.yaml → (camera, roles in the world, recipe); roles: d, daz, L, H, zb, kw, peak, seed, form, type, name."""
    return compose.layout(recipe)


def check(cam, roles):
    """Layout warnings before Blender: touches, tops level with each other."""
    return compose.check(cam, roles)


def realism(r, dist=0.0, near=8000.0):
    """The component's "realism" knob 0…1 (the target look: small puffs with lit faces): 0 — soft masses (3 puff levels,
    shading pattern 0.3), 0.5 — 4 levels, 0.8, 1 — 5 levels, 1.2, a finer grid, edge noise 0.4. The extra levels and the fine grid go to masses closer
    than near (the distant ones are small in the frame). → {levels_add, detail, ts_blur, erode, vs_k}."""
    r = float(min(max(r, 0.0), 1.0)); close = dist < near
    return {"levels_add": int(round(2 * r)) if close else 0, "detail": 0.3 + 0.9 * r, "ts_blur": 2 if r < 0.25 else 1,
            "erode": 0.4 * r, "vs_k": (1.0 - 0.25 * r) if close else 1.0}


def form(role, vs, dens=0.35, base_soft=0.0):
    """Density field of a role of kind mass: (ρ, o, vs). Layers (layer, cirrus) — forms.strato_layer / cirrus_layer by the kind's knobs."""
    if role["form"] != "mass": raise ValueError(f"kind {role['type']}: form {role['form']} is a layer, not a mass")
    return forms.mass_lobes(role["L"], role["H"], np.random.default_rng(role["seed"]), vs, role["peak"], dens=dens, base_soft=base_soft, **role["kw"])


def bake(rho, o, vs, P, sun_dir, view_dir, z0=0.0, detail=0.3, ts_blur=2, sigma=0.06, ground=1.0):
    """Precomputed light (diffusion + single scattering + pattern): the emission σρ·J (RGB, float32) in units of the light plan P."""
    v = np.asarray(view_dir, float); s = np.asarray(sun_dir, float)
    return _light.light(rho, o, vs, P, s, float(-v @ s / max(np.linalg.norm(v) * np.linalg.norm(s), 1e-9)), sigma=sigma, z0=z0, octaves=1, g=0.85,
                        ms_mode="diff", coarse=4, view_dir=v, ground=ground, detail=detail, ts_blur=ts_blur)
