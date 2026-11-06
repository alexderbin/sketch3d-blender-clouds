"""Rainbow — a per-frame post effect (without Blender): the angle of the camera ray from the antisolar point → the primary arc (red 42.3°, green 41.7°,
blue 40.7°, width ~0.9°), lighter inside the arc, Alexander's dark band (42–50°), the secondary arc (50.4–53.4°, colours reversed, ×0.43). Brighter against
a dark background (rain, a dark cloud), fading against a light sky; visible when the sun is behind and not higher than 42°.
.venv/bin/python -m components.clouds.rainbow <test-stand frames folder> [<camera>…] [strength=0.6] [scene=<scene folder>]"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image


def rays(cam, W, H):
    """Camera rays in the world: (H, W, 3) from matrix_world, the lens and the sensor width."""
    M = np.array(cam["matrix_world"]); k = cam.get("lens_mm", 35.0) / (cam.get("sensor_width_mm", 36.0) / 2)
    u = (np.arange(W) + 0.5) / W * 2 - 1; v = ((np.arange(H) + 0.5) / H * 2 - 1) * H / W
    d = M[:3, 0][None, None] * u[None, :, None] / k + M[:3, 1][None, None] * (-v)[:, None, None] / k - M[:3, 2][None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def bow(theta):
    """Rainbow colour (linear RGB, fractions) by the angle from the antisolar point, °."""
    g = lambda c, w: np.exp(-((theta - c) / w) ** 2)
    prim = np.stack([g(42.3, 0.75), g(41.7, 0.7), g(40.7, 0.7)], -1) * 1.0
    sec = np.stack([g(50.6, 1.0), g(51.6, 1.0), g(52.9, 1.0)], -1) * 0.43       # secondary: red on the inside
    inside = np.clip((40.5 - theta) / 4.0, 0, 1)[..., None] * 0.025 * np.array([1.0, 1.0, 1.0])   # slightly lighter inside the arc
    dark = -np.clip(1 - np.abs(theta - 46.2) / 3.8, 0, 1)[..., None] * 0.08    # Alexander's band is darker
    return prim + sec + inside * (theta < 40.5)[..., None] + dark


def terrain_dist(cam, W, H, scene, step=60.0, far=30000.0):
    """Distance to the ground along the frame's rays (a march over the scene's height map; a miss — infinity): the arc appears on rain above the ground, not on near mountains."""
    from components.clouds import terrain as T
    z, x0, y0, st = T.heights(scene); n = z.shape[0]; d = rays(cam, W, H).reshape(-1, 3); p0 = np.array(cam["matrix_world"])[:3, 3]
    hit = np.full(len(d), np.inf, np.float32); alive = np.ones(len(d), bool); t = 50.0
    while t < far and alive.any():
        P = p0[None] + d[alive] * t; i = np.clip(((P[:, 0] - x0) / st).astype(int), 0, n - 1); j = np.clip(((P[:, 1] - y0) / st).astype(int), 0, n - 1)
        h = P[:, 2] <= z[i, j]; idx = np.nonzero(alive)[0][h]; hit[idx] = t; alive[idx] = False; t += step * (1 + t / 4000.0)
    return hit.reshape(H, W)


def apply(png, cam, sun_dir, strength=0.6, scene=None):
    im = np.asarray(Image.open(png).convert("RGB"), np.float32) / 255.0; H, W = im.shape[:2]
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    d = rays(cam, W, H); anti = -np.asarray(sun_dir, float); anti /= np.linalg.norm(anti)
    th = np.degrees(np.arccos(np.clip(d @ anti, -1, 1)))
    Y = lin @ np.array([0.2126, 0.7152, 0.0722]); dark_bg = np.clip((0.45 - Y) / 0.35, 0.15, 1.0)[..., None]   # brighter on a dark background
    near = np.ones((H, W, 1), np.float32)
    if scene: near = np.clip((terrain_dist(cam, W, H, scene) - 3000.0) / 4000.0, 0, 1)[..., None]   # near ground — no arc
    out = np.clip(lin + strength * bow(th) * dark_bg * near * np.median(Y[Y > 0]) * 1.6, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    a = sys.argv[1:]; d0 = a[0]; opt = dict(x.split("=", 1) for x in a[1:] if "=" in x); names = [x for x in a[1:] if "=" not in x] or ["far", "medium"]
    P = json.load(open(os.path.join(d0, "far_plan.json"))); S = P["sun"]["dir"]
    for nm in names:
        cam = json.load(open(os.path.join(d0, "lit", nm, "far.json")))["camera"]
        apply(os.path.join(d0, f"{nm}.png"), cam, S, float(opt.get("strength", 0.6)), opt.get("scene")).save(os.path.join(d0, f"{nm}_rainbow.png")); print(nm, "rainbow")
