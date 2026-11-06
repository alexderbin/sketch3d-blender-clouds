"""A fast test stand for a storm ceiling in perspective (like a camera "under the storm"): a piece of the ceiling (forms.storm_ceiling), the plan's light +
the dawn rim (a paint, as in the castle.py stand), camera rays from below at ~30° — its own ray march with a step of 0.7 cell (trilinear ρ and
emission, absorption + emission). Without Blender, ~45 s per variant. A yardstick for close-range detail: Eevee's depth layers blur it, while a strict view
from below shows a pattern that does not appear in the frame.
.venv/bin/python -m components.clouds.bench.ceiling_persp <output.jpg> '<JSON list of variants>' [<dusk far_plan.json>]
a variant — storm_ceiling knobs: {"cover": 0.8, "vs": 24, "kids": 2, "protrude": 0.7, "lump": [420, 170, 70]}"""
import json, math, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from components.clouds import forms, light as LT
SIG = 0.06; OUT = sys.argv[1]; VAR = json.loads(sys.argv[2])
P = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "out/dev/clouds/castle2/castle_t_storm_sky_magic3/far_plan.json"))   # the light plan of dusk from any run
F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14); ims = []
def tri(a, idx):                                                                 # trilinear, outside — 0
    return LT.sample(a, idx.astype(np.float32))
for v in VAR:
    t = time.time(); kw = {k: (tuple(x) if isinstance(x, list) else x) for k, x in v.items() if k in ("cover", "cell", "lump", "kids", "vs", "erode", "billow", "protrude", "lumpy", "roll", "scud")}
    rho, o, vs = forms.storm_ceiling(7, ext=v.get("ext", 5000.0), thick=1400.0, **kw)
    sun = np.array(P["sun"]["dir"]); E = LT.light(rho, o, vs, P, sun, 0.0, sigma=SIG, z0=1875, octaves=1, g=0.85, ms_mode="diff", coarse=2, view_dir=np.array([0, -0.6, -0.8]),
                                                    detail=0.3, ts_blur=1, info={}, ground=0.3)
    Ls = float(np.dot(P["light"]["sky_e_rgb"], [0.2126, 0.7152, 0.0722])) / math.pi
    ar, er = math.radians(float(P["sun"]["az"])), math.radians(-1.0); rd = np.array([math.cos(ar) * math.cos(er), math.sin(ar) * math.cos(er), math.sin(er)])
    tr = LT.tau_sun(rho, vs, rd, SIG)
    for ax in range(3): tr = (np.roll(tr, 1, ax) + tr + np.roll(tr, -1, ax)) / 3.0
    Mr = 1.0
    if v.get("rim_var"):                                                         # dawn patches: where it broke through stronger, where the mass is in shadow (the rhythm of light)
        Xc = o[0] + (np.arange(rho.shape[0]) + 0.5) * vs; Yc = o[1] + (np.arange(rho.shape[1]) + 0.5) * vs
        Mr = (1 - v["rim_var"]) + v["rim_var"] * np.clip((forms.vnoise2(Xc + 7e3, Yc + 3e3, 1 / 1500.0, 21) - 0.3) / 0.4, 0, 1)[:, :, None, None] * 1.6
    E = E + (SIG * rho[..., None] * np.exp(-2.0 * tr)[..., None] * Mr * (12.0 * Ls * np.array([1.0, 0.36, 0.12]))).astype(np.float32)   # dawn rim (like B × 1.5)
    if v.get("fill"):                                                            # cold fill of the shadows: the sky from the opposite side (warm — cold)
        af, ef = ar + math.pi, math.radians(v.get("fill_el", -10.0)); fd = np.array([math.cos(af) * math.cos(ef), math.sin(af) * math.cos(ef), math.sin(ef)])   # from below and aside: the undersides on that side
        tf = LT.tau_sun(rho, vs, fd, SIG)
        for _ in range(5):
            for ax in range(3): tf = (np.roll(tf, 1, ax) + tf + np.roll(tf, -1, ax)) / 3.0   # without noise and sampling steps
        E = E + (SIG * rho[..., None] * np.exp(-1.0 * tf)[..., None] * (v["fill"] * Ls * np.array(v.get("fill_rgb", [0.42, 0.52, 1.0])))).astype(np.float32); del tf
    W, H = 360, 200; cam = np.array([0.0, -2600.0, -1700.0])                     # camera: 1.7 km below the base, to the side; looking up ~30°
    f = np.array([0.0, math.cos(math.radians(30)), math.sin(math.radians(30))]); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    k = 1.0 / math.tan(math.radians(35)); uu = (np.arange(W) + 0.5) / W * 2 - 1; vv = ((np.arange(H) + 0.5) / H * 2 - 1) * H / W
    D = f[None, None] + (uu[None, :, None] * r[None, None] - vv[:, None, None] * u[None, None]) / k; D /= np.linalg.norm(D, axis=-1, keepdims=True)
    T = np.ones((H, W)); acc = np.zeros((H, W, 3)); ds = 0.7 * vs
    for s in np.arange(1500.0, 9000.0, ds):                                      # from 1.5 to 9 km along the ray
        Pw = cam[None, None] + D * s; idx = (Pw - o[None, None]) / vs - 0.5
        rr = tri(rho, idx.reshape(-1, 3)).reshape(H, W); 
        if not (rr > 0).any(): continue
        ee = np.stack([tri(np.ascontiguousarray(E[..., q]), idx.reshape(-1, 3)).reshape(H, W) for q in range(3)], -1)
        a_ = np.exp(-SIG * rr * ds); acc += T[..., None] * np.where(rr[..., None] > 1e-6, ee / np.maximum(SIG * rr, 1e-9)[..., None], 0) * (1 - a_)[..., None]; T *= a_
    sky = np.array(P["light"]["sky_e_rgb"]) / math.pi * 1.2; img = acc + T[..., None] * sky
    Y = img @ np.array([0.2126, 0.7152, 0.0722]); g = 0.45 / max(float(np.percentile(Y, 95)), 1e-9); x = img * g; x = x / (1 + x) * 1.5
    im = Image.fromarray((np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).resize((540, 300), Image.LANCZOS)
    print(v, rho.shape, round(time.time() - t), "s", flush=True); ims.append((im, v))
S = Image.new("RGB", (2 * 550, ((len(ims) + 1) // 2) * 324), (20, 20, 20)); d = ImageDraw.Draw(S)
for k_, (im, v) in enumerate(ims):
    x0, y0 = (k_ % 2) * 550, (k_ // 2) * 324; S.paste(im, (x0, y0 + 22)); d.text((x0 + 4, y0 + 4), json.dumps(v)[:70], fill=(230, 230, 230), font=F)
S.save(OUT, quality=90); print(OUT)
