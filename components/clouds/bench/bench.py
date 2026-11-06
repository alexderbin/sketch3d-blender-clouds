"""Reference for Clouds without Blender: (1) diffusion energy — on a homogeneous sphere sky + ground inside = 1, the lit edge ≈ 0.2–0.4 E☉;
(2) the frame solver — the roles of the recipes without remarks (touches, tops at the same height), sizes within reasonable limits; (3) the shapes of
the kinds — the field is not empty and does not hit the grid edge. .venv/bin/python -m components.clouds.bench.bench"""
import sys

import numpy as np

from components.clouds import compose, forms, light


def energy():
    n = 64; I = np.indices((n, n, n)).transpose(1, 2, 3, 0) - n / 2 + 0.5; rho = (np.linalg.norm(I, axis=-1) < 26).astype(np.float32)
    ts = light.tau_sun(rho, 1.0, [0, 0, 1], 4.0, reach=200); F, k = light.diffusion(rho, 1.0, ts, 4.0, 0.877, 4, view_dir=[0, 0, 1]); c = n // 2
    top = float(F[c, c, c + 24, 0]); amb = float(F[c, c, c, 1] + F[c, c, c, 2])
    return [f"energy: sky+ground inside {amb:.3f} (expected 1 ± 0.05)"] if abs(amb - 1) > 0.05 else [], f"edge {top:.2f} E☉, CG {k}"


def recipes():
    bad = []
    for name, R in compose.load("recipes.yaml").items():
        if R.get("camera") == "scene": continue                               # the scene camera — checked on a real camera (castle.py)
        cam, roles, _ = compose.layout(name)
        bad += [f"{name}: {w}" for w in compose.check(cam, roles)]
        bad += [f"{name}: {r['name']} height {r['H']:.0f} m" for r in roles if not 200 <= r["H"] <= 16000]
    return bad


def shapes():
    bad = []; types = compose.load("types.yaml")
    for t, T in types.items():
        if T.get("form", "mass") != "mass": continue
        kw = dict(T.get("kw", {})); rho, o, vs = forms.mass_lobes(3000.0, 2500.0 if "tower" not in kw else 7000.0, np.random.default_rng(1), 60.0, T.get("peak", 0.4), dens=0.35, **kw)
        edge = max(rho[0].max(), rho[-1].max(), rho[:, 0].max(), rho[:, -1].max(), rho[:, :, -1].max())
        if rho.max() <= 0 or edge > 0.05: bad.append(f"kind {t}: empty or hits the edge ({edge:.2f})")
    return bad


if __name__ == "__main__":
    b1, note = energy(); bad = b1 + recipes() + shapes()
    print("energy:", note); print("\n".join(bad) if bad else "reference: no remarks"); sys.exit(1 if bad else 0)
