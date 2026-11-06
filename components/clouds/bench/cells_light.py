"""A fast test stand for the light of a storm made of small clouds: the shape forms.storm_cells, the light light.light without the sun (the dusk plan of any
run) with discharges inside (lightning.intracloud) — knob variants in rows: the sky on the left (diffuse light), the flashes on the right; a side view,
alpha compositing of the emission, its own exposure. Without Blender: a 48 m grid takes seconds, 24 m ~20 s per variant.
.venv/bin/python -m components.clouds.bench.cells_light <output.jpg> [cell=24] '[[sky_detail, sky_r, bolt_ms], …]' [<far_plan.json>]"""
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from components.clouds import forms, light as LT, lightning as LN

SIG = 0.06


def render(rho, em):
    """A view along y: the emission em (σρ·L) with absorption → an image (x, z)."""
    a = 1 - np.exp(-SIG * rho * VS); T = np.cumprod(np.concatenate([np.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
    return (T[..., None] * em / np.maximum(SIG * rho, 1e-9)[..., None] * a[..., None]).sum(1)


def tone(img, w=560):
    Y = img @ np.array([0.2126, 0.7152, 0.0722]); g = 0.5 / max(float(np.percentile(Y[Y > 0], 97)), 1e-9); x = img * g; x = x / (1 + x) * 1.6
    im = Image.fromarray((np.clip(x.transpose(1, 0, 2)[::-1], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    return im.resize((w, int(w * img.shape[1] / img.shape[0])), Image.LANCZOS)


if __name__ == "__main__":
    out = sys.argv[1]; VS = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
    VAR = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [[0.0, None, 0.5], [5.0, 150.0, 2.0]]
    P = json.load(open(sys.argv[4] if len(sys.argv) > 4 else "out/dev/clouds/castle2/castle_t_storm_cells/far_plan_bake.json"))
    rho, o, VS = forms.storm_cells(4500.0, 2700.0, np.random.default_rng(5), VS, kids=4, dens=0.7)
    Pc, Wc = LN.samples(LN.intracloud(0.0, 0.0, 4500.0, 2700.0, 0.0, np.random.default_rng(12), n=3), 160.0)
    ch = tuple(zip((Pc - o) / VS - 0.5, Wc * 60.0)); tiles = []
    for sd, sr, cm in VAR:
        t = time.time(); info = {}
        E = LT.light(rho, o, VS, P, np.array(P["sun"]["dir"]), 0.0, sigma=SIG, octaves=1, g=0.85, ms_mode="diff", coarse=4,
                     view_dir=np.array([0, 1.0, 0]), detail=1.0, ts_blur=1, info=info, channel=ch, sky_detail=sd, sky_r=sr, ch_ms=cm)
        tiles += [(tone(render(rho, E)), f"sky: sky_detail {sd}, sky_r {sr}"),
                  (tone(render(rho, info["flash"][..., None] * np.array([0.78, 0.84, 1.0], np.float32))), f"flashes: bolt_ms {cm}")]
        print(sd, sr, cm, round(time.time() - t, 1), "s", flush=True)
    h = tiles[0][0].size[1] + 20; S = Image.new("RGB", (2 * 570, (len(tiles) // 2) * h), (15, 15, 18)); d = ImageDraw.Draw(S)
    for k, (im, tx) in enumerate(tiles):
        x0, y0 = (k % 2) * 570, (k // 2) * h; S.paste(im, (x0, y0 + 18)); d.text((x0 + 4, y0 + 3), tx, fill=(230, 230, 230))
    S.save(out, quality=90); print(out, rho.shape)
