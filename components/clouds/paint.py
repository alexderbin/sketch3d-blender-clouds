"""Tuning the paints without baking: the test stand's passes (`paint_passes: 1` → passes/<layer>/lit/<camera>/lit.exr: the base, the paints rim/leak/glow, the flash)
are blended in linear light — base + Σ weight·(layer − base) — and go through the light node's processing (far.py post: haze, tone); the lightning channel goes
on top (bolt.json, strength = the flash weight). A weight of 1 is as in the recipe.
.venv/bin/python -m components.clouds.paint <run folder> rim=1 leak=2 glow=1 flash=1 [<camera>…] [name=tuned]"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def combined(path):
    import OpenEXR
    f = OpenEXR.File(path); part = f.parts[0]; key = next(k for k in part.channels if k.endswith("Combined") or k in ("RGBA", "RGB"))
    return f, part, key, np.asarray(part.channels[key].pixels, np.float32)


def mix(d0, weights, names, tag="tuned", sat=None, tint=None):
    """A blend of passes → <d0>/<tag>/lit/<camera>/ → far.py post → <d0>/<camera>_<tag>.png (+ the lightning channel); sat — the layer's saturation
    (the layer's contribution: luminance + sat·(colour − luminance)); tint — recolouring of the layer (the contribution's luminance stays, the colour is new: a palette without baking)."""
    dirs = []
    for nm in names:
        b = os.path.join(d0, "passes", "base", "lit", nm)
        if not os.path.exists(os.path.join(b, "lit.exr")): continue
        f, part, key, A = combined(os.path.join(b, "lit.exr")); acc = A.copy()
        for ly, w in weights.items():
            p_ = os.path.join(d0, "passes", ly, "lit", nm, "lit.exr")
            if w and os.path.exists(p_):
                Dl = combined(p_)[3][..., :3] - A[..., :3]; k_ = (sat or {}).get(ly, 1.0)
                lum_ = np.array([0.2126, 0.7152, 0.0722], np.float32)
                if (tint or {}).get(ly) is not None: c_ = np.asarray(tint[ly], np.float32); Dl = (Dl @ lum_)[..., None] * c_ / float(c_ @ lum_)
                if k_ != 1.0: Y_ = (Dl @ lum_)[..., None]; Dl = Y_ + k_ * (Dl - Y_)
                acc[..., :3] += w * Dl
        acc[..., :3] = np.maximum(acc[..., :3], 0.0)
        d = os.path.join(d0, tag, "lit", nm); os.makedirs(d, exist_ok=True)
        part.channels[key].pixels = acc; f.write(os.path.join(d, "lit.exr"))
        for fn in ("sky.exr", "far.json"): shutil.copy(os.path.join(b, fn), os.path.join(d, fn))
        dirs.append((nm, d))
    r = subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-m", "core.light.far", "post"] + [d for _, d in dirs], capture_output=True, text=True, cwd=ROOT)
    if r.returncode: raise SystemExit("far.py post: " + r.stderr[-400:])
    from components.clouds import lightning as LN
    bj = os.path.join(d0, "bolt.json"); B, D = (LN.load(bj) if os.path.exists(bj) else (None, {}))
    for nm, d in dirs:
        dst = os.path.join(d0, f"{nm}_{tag}.png"); shutil.copy(os.path.join(d, "lit.png"), dst)
        if B is not None and weights.get("flash", 0) > 0:                         # the channel goes on top, its strength follows the flash weight
            cam = json.load(open(os.path.join(d, "far.json")))["camera"]
            LN.draw(dst, cam, B, D.get("tau", {}).get(nm), D.get("scene"), gain=float(weights["flash"])).save(dst)
        print(nm, dst)


if __name__ == "__main__":
    a = sys.argv[1:]; opt = dict(x.split("=", 1) for x in a[1:] if "=" in x)
    W_ = {k: float(v) for k, v in opt.items() if k in ("rim", "leak", "glow", "flash")}
    S_ = {k[:-4]: float(v) for k, v in opt.items() if k.endswith("_sat")}
    T_ = {k[:-4]: [float(x) for x in v.split(",")] for k, v in opt.items() if k.endswith("_rgb")}
    mix(a[0], W_, [x for x in a[1:] if "=" not in x] or ["far", "medium", "up", "bolt", "high"], opt.get("name", "tuned"), S_, T_)
