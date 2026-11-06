"""Sky box (a storm front, a skybox of storm clouds): the 6 faces of the test stand's cube (`skybox: 1` → sky_px…sky_nz, 90° each, after the node's processing and
the lightning channel) → an equirectangular 2:1 panorama. A pixel's direction goes to the face it looks at most directly; the projection uses the face camera
(lightning.project), sampling is bilinear. Without Blender.
.venv/bin/python -m components.clouds.skybox <run folder> [width=4096] [suffix=]"""
import json
import os
import sys

import numpy as np
from PIL import Image

FACES = ("sky_px", "sky_nx", "sky_py", "sky_ny", "sky_pz", "sky_nz")


def bilinear(img, x, y):
    H, W = img.shape[:2]; x = np.clip(x - 0.5, 0, W - 1.001); y = np.clip(y - 0.5, 0, H - 1.001); x0, y0 = x.astype(int), y.astype(int); fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def equirect(d0, width=4096, sfx=""):
    from components.clouds.lightning import project
    cams, imgs = {}, {}
    for f in FACES:
        p = os.path.join(d0, f"{f}{sfx}.png")
        if not os.path.exists(p): raise SystemExit(f"no face {p}")
        cams[f] = json.load(open(os.path.join(d0, "lit", f, "far.json")))["camera"]; imgs[f] = np.asarray(Image.open(p).convert("RGB"), np.float32)
    H = width // 2; lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi; lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    LON, LAT = np.meshgrid(lon, lat); D = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1).reshape(-1, 3)
    F = np.stack([-np.array(cams[f]["matrix_world"])[:3, 2] for f in FACES])        # the faces' view directions
    best = np.argmax(D @ F.T, 1); out = np.zeros((len(D), 3), np.float32)
    for k, f in enumerate(FACES):
        m = best == k
        if not m.any(): continue
        M = np.array(cams[f]["matrix_world"]); h_, w_ = imgs[f].shape[:2]
        xy, z = project(cams[f], M[:3, 3][None] + D[m] * 1000.0, w_, h_); out[m] = bilinear(imgs[f], xy[:, 0], xy[:, 1])
    return Image.fromarray(np.clip(out.reshape(H, width, 3), 0, 255).astype(np.uint8))


if __name__ == "__main__":
    a = sys.argv[1:]; opt = dict(x.split("=", 1) for x in a[1:] if "=" in x); sfx = opt.get("suffix", "")
    im = equirect(a[0], int(opt.get("width", 4096)), sfx); p = os.path.join(a[0], f"skybox{sfx}.png"); im.save(p)
    im.resize((1600, 800), Image.LANCZOS).save(os.path.join(a[0], f"skybox{sfx}_preview.jpg"), quality=90); print(p, im.size)
