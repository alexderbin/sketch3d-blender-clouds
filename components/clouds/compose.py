"""Composition of clouds in the frame: a recipe (recipes.yaml) sets the roles in fractions of the frame — the centre U, the width, the top V, the distance —
and the cloud kind (types.yaml); the solver converts them to the world: the azimuth from U, the mass height from the top and distance, the length from the
width (the mass width on screen is the length + 2 dome radii; for a tower — ×2.2 of the base). The camera is a pinhole (lens, tilt).
Checks — silhouettes (bench/stand.py silhouette=1 → notan.png) and the roles' boxes in the frame (frames)."""
import math
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SENSOR, ASPECT = 36.0, 16 / 9


def load(name):
    """The component's types.yaml or recipes.yaml."""
    return yaml.safe_load(open(os.path.join(HERE, name)))


class Cam:
    """A pinhole camera: position, gaze azimuth (° from +x counter-clockwise), tilt (°), lens (mm); the frame U → right, V → up."""
    def __init__(self, z=400.0, az=90.0, tilt=0.0, lens=28.0):
        self.p = np.array([0.0, 0.0, z]); self.az, self.tilt, self.k = az, tilt, lens / (SENSOR / 2)
        a, t = math.radians(az), math.radians(tilt)
        self.f = np.array([math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t)])
        self.r = np.array([math.sin(a), -math.cos(a), 0.0]); self.u = np.cross(self.r, self.f)

    def project(self, P):
        d = np.atleast_2d(P) - self.p; z = d @ self.f
        return np.stack([(d @ self.r) / z * self.k * 0.5 + 0.5, (d @ self.u) / z * self.k * ASPECT * 0.5 + 0.5, z], -1)


def solve(cam, role, base, types):
    """A role in fractions of the frame → the world: {d, daz (° from the view axis, + — to the left), L, H, zb, kw (shape knobs), type}."""
    T = dict(types[role["type"]]); d = float(role["dist"])
    if T.get("form", "mass") != "mass":                                       # a layer is not in fractions of the frame: distance, extent, height
        return dict(type=role["type"], form=T["form"], d=d, daz=float(role.get("daz", 0.0)), L=float(role.get("ext", 30000.0)), H=0.0,
                    zb=float(role.get("alt", base)), kw=dict(T.get("kw", {})), seed=int(role.get("seed", 1)))
    daz = -math.degrees(math.atan((2 * role["U"] - 1) / cam.k))
    el_top = cam.tilt + math.degrees(math.atan((2 * role["top"] - 1) / (cam.k * ASPECT)))
    zb = float(role.get("base", base)); H = max(200.0, d * math.tan(math.radians(el_top)) + cam.p[2] - zb)
    W = role["width"] * 2 * d / cam.k                                          # frame width at distance d × the fraction
    form = T.pop("form", "mass")
    if form == "mass":
        kw = T.pop("kw", {}); kw = dict(kw, **role.get("kw", {}))
        kw.setdefault("levels", 3 + (role["width"] > 0.25) + (role["width"] > 0.5))   # puff levels by the size in the frame: large in the frame — smaller puffs
        if "tower" in kw: L = W / 2.2                                         # tower: the base and shoulders ≈ 2.2 bases
        elif kw.get("lens"): L = W
        else: L = max(0.2 * W, W - 2 * kw.get("aspect", 0.45) * 0.8 * H)
        if kw.get("anvil"): aw, dr = kw["anvil"]; kw["anvil"] = (aw * W, dr * W); L = W * 0.3   # anvil — a fraction of the role width
        return dict(type=role["type"], form=form, d=d, daz=daz, L=L, H=H, zb=zb, kw=kw, peak=role.get("peak", T.get("peak", 0.4)), seed=int(role.get("seed", 1)))
    return dict(type=role["type"], form=form, d=d, daz=daz, L=W, H=H, zb=zb, kw=dict(T.get("kw", {}), **role.get("kw", {})), seed=int(role.get("seed", 1)))


def layout(recipe, cam=None, base=None):
    """A recipe → (camera, roles in the world, recipe); camera: scene — the caller supplies the camera and the bases (base, m) (the scene camera)."""
    R = load("recipes.yaml")[recipe]; types = load("types.yaml"); c = R["camera"]
    if cam is None:
        if c == "scene": raise ValueError(f"recipe {recipe}: scene camera — pass cam")
        cam = Cam(c.get("z", 400.0), c.get("az", 90.0), c.get("tilt", 0.0), c.get("lens", 28.0))
    if base is not None: R = dict(R, base=base)
    return cam, [dict(solve(cam, r, R.get("base", 1700.0), types), name=r.get("name", r["type"])) for r in R["roles"]], R


def frames(cam, roles):
    """The roles' boxes in the frame (U0, U1, top V) from the mass's bounding box — to check the layout before Blender (touches, overlaps)."""
    out = []
    for r in roles:
        if r.get("form", "mass") != "mass": continue                          # layers span the full width, there are no touches
        a = math.radians(cam.az + r["daz"]); x, y = r["d"] * math.cos(a), r["d"] * math.sin(a); dx, dy = -math.sin(a), math.cos(a)
        Le = r["L"] * (2.2 if "tower" in r["kw"] else 1.0) + (0 if "tower" in r["kw"] or r["kw"].get("lens") else 2 * r["kw"].get("aspect", 0.45) * 0.8 * r["H"])
        P = [(x + s * Le / 2 * dx, y + s * Le / 2 * dy, z) for s in (-1, 1) for z in (r["zb"], r["zb"] + r["H"])]
        q = cam.project(np.array(P)); out.append((r["name"], round(float(q[:, 0].min()), 2), round(float(q[:, 0].max()), 2), round(float(q[:, 1].max()), 2)))
    return out


def check(cam, roles, tol=0.02):
    """Layout before Blender: a touch — the edges of two roles closer than tol of the frame (neither an overlap nor a gap); tops at the same height (± tol)
    for neighbours. → a list of remarks."""
    F = frames(cam, roles); out = []
    for i in range(len(F)):
        for j in range(i + 1, len(F)):
            a, b = F[i], F[j]
            for e1, e2 in ((a[2], b[1]), (b[2], a[1])):
                if abs(e1 - e2) < tol and 0 < min(a[2], b[2]) and max(a[1], b[1]) < 1: out.append(f"touch: {a[0]} and {b[0]} ({e1:.2f} / {e2:.2f})")
            if abs(a[3] - b[3]) < tol and max(a[1], b[1]) < min(a[2], b[2]) + 0.1: out.append(f"tops at the same height: {a[0]} and {b[0]} ({a[3]:.2f})")
    return out
