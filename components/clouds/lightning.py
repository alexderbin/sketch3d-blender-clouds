"""Lightning (the look: dark masses, light seeping through from inside in thin places, thin veins of lightning in the glow).
The channel is a polyline built by midpoint displacement: large bends from the top down to fine jitter (the kink of small segments ~16°, as in Reed & Wyvill
1994); branches go down from the upper part of the channel; the "spider" — branches along the bottom of the cloud (a discharge outside the cloud is the brightest);
inside the cloud there are branches of their own (they are not visible — they light the cloud from inside). The light for the clouds is the channel points for baking
(`light.light(channel=…)`), for the ground — lamps along the channel. The channel itself is a per-frame drawing: a thin core, a halo and an air glow;
occlusion by the relief and the clouds (τ from a channel point to the camera). Without Blender.
.venv/bin/python -m components.clouds.lightning <frames folder> [<camera>…] [scene=<scene folder>] — redraw from bolt.json"""
import json
import math
import os
import sys

import numpy as np

STYLE = {"core": 1.3,                                     # core width of the main channel, px at a 960 frame (branches — by strength, √)
         "core_rgb": (1.0, 0.97, 1.0), "core_k": 1.8,     # the core is overexposed — white
         "glow": ((1.0, 0.85, "halo"), (3.5, 0.45, "halo"), (12.0, 0.2, "air"), (48.0, 0.08, "air")),   # (σ px, strength, colour)
         "halo_rgb": (0.6, 0.76, 1.0), "air_rgb": (0.5, 0.52, 1.0),
         "veins": 0.25, "veins_k": 0.7}                    # branches inside the cloud — as veins through the glow: τ × veins, strength × veins_k


def _perp(d, rng, flat=0.0):
    """A unit vector ⟂ d; flat — the fraction of the vertical that is suppressed (the spider along the cloud bottom does not go up and down)."""
    a = rng.normal(size=3); a[2] *= 1.0 - flat; a -= a.dot(d) * d; n = np.linalg.norm(a)
    return a / n if n > 1e-9 else _perp(d, rng)


def displace(A, B, rng, seg=12.0, amp=0.24, decay=0.8, flat=0.0):
    """Midpoint displacement A→B down to segments of ~seg m: the midpoint of a segment is shifted across by N(0, a·segment length), a drops by a factor decay per
    level (large bends at the top, fine jitter at the bottom: small segments kink by ~10–20°)."""
    A, B = np.asarray(A, float), np.asarray(B, float); P = [A, B]; a = amp
    for _ in range(max(1, int(math.ceil(math.log2(max(np.linalg.norm(B - A) / seg, 2.0)))))):
        Q = [P[0]]
        for p, q in zip(P[:-1], P[1:]):
            d = q - p; L = float(np.linalg.norm(d)); Q += [(p + q) / 2 + _perp(d / L, rng, flat) * rng.normal(0.0, a * L), q]
        P = Q; a *= decay
    return np.array(P)


def _branch(P, i, rng, ang, length, gz=None, flat=0.0, z=None, seg=12.0, amp=0.22):
    """A branch from vertex i of the polyline P at angle ang (°) to its course, of length length; z — the height of the end (the spider), gz — the ground below."""
    d = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]; d /= np.linalg.norm(d); u = _perp(d, rng, 0.6 if flat == 0 else 0.95)
    a = math.radians(ang); e = math.cos(a) * d + math.sin(a) * u
    if flat: e[2] = 0.0
    E = P[i] + e / np.linalg.norm(e) * length
    if z is not None: E[2] = z
    if gz is not None: E[2] = max(E[2], gz(E[0], E[1]) + 15.0)
    return displace(P[i], E, rng, seg, amp=amp, flat=flat)


def bolt(top, ground, base_z, rng, across, gz=None, crawl=4, branches=6, inner=3, reach=None, seg=12.0):
    """Lightning cloud-to-ground. top — the channel point in the cloud, ground — the strike point, base_z — the cloud base above the channel, across — across the view
    (xy: the spider spreads to the sides of the frame), gz(x, y) — the ground height (branches do not go into the ground), reach — (x, y, r): the spider does not
    leave the area under the cloud. → [{"pts", "w", "kind"}]: main — the channel, branch — branches down, crawl — the spider, cloud — inside the cloud."""
    top, ground = np.asarray(top, float), np.asarray(ground, float); out = []
    M = displace(top, ground, rng, seg); out.append({"pts": M, "w": np.ones(len(M)), "kind": "main"})
    fade = lambda n, w0, k=0.9: w0 * (1 - k * np.linspace(0, 1, n))
    lo = ground[2] + 0.3 * (base_z - ground[2])                                     # branches start in the upper 70 % of the visible descent, more often near the top
    idx = [i for i in range(2, len(M) - 2) if lo < M[i][2] < base_z - 20.0]
    if idx and branches:
        p = np.array([M[i][2] - ground[2] for i in idx]) ** 1.5; p /= p.sum()
        for i in rng.choice(idx, size=min(branches, len(idx)), replace=False, p=p):
            L = rng.uniform(0.2, 0.5) * (M[i][2] - ground[2]) / 0.6; B = _branch(M, i, rng, rng.uniform(25, 55), L, gz, seg=seg)
            out.append({"pts": B, "w": fade(len(B), rng.uniform(0.3, 0.5)), "kind": "branch"})
            if rng.random() < 0.5 and len(B) > 8:                                   # a branch of a branch
                j = int(rng.integers(2, int(0.6 * len(B)))); B2 = _branch(B, j, rng, rng.uniform(25, 50), rng.uniform(0.3, 0.5) * L * (1 - j / len(B)), gz, seg=seg)
                out.append({"pts": B2, "w": fade(len(B2), 0.5 * out[-1]["w"][j]), "kind": "branch"})
    k0 = int(np.argmin(np.abs(M[:, 2] - (base_z - 90.0))))                         # just below where it leaves the cloud — the root of the spider
    ax = np.array([across[0], across[1], 0.0]); ax /= np.linalg.norm(ax); fw = np.cross([0.0, 0.0, 1.0], ax)
    for c in range(crawl):                                                         # spider: along the cloud bottom, toward the sides of the frame and slightly into the depth
        a = rng.normal(0.0, 0.5) + (0.0 if c < 2 else rng.choice([-1, 1]) * rng.uniform(0.9, 1.4))
        e = (1 if c % 2 == 0 else -1) * (math.cos(a) * ax + math.sin(a) * fw); L = rng.uniform(900.0, 2600.0)
        if reach is not None:                                                      # the end is under the cloud
            for _ in range(20):
                E = M[k0] + e * L
                if math.hypot(E[0] - reach[0], E[1] - reach[1]) < reach[2]: break
                L *= 0.85
        E = M[k0] + e * L; E[2] = base_z - rng.uniform(80.0, 200.0)                 # below the base: a channel outside the cloud is visible
        Cp = displace(M[k0], E, rng, seg, amp=0.2, flat=0.85); wc = fade(len(Cp), rng.uniform(0.45, 0.65), 0.88); out.append({"pts": Cp, "w": wc, "kind": "crawl"})
        for _ in range(int(rng.integers(2, 4))):                                   # small branches of the spider
            j = int(rng.integers(3, max(4, int(0.75 * len(Cp)))))
            C2 = _branch(Cp, j, rng, rng.uniform(30, 75), rng.uniform(0.25, 0.5) * L * (1 - j / len(Cp)), flat=0.85, z=base_z - rng.uniform(80, 220), seg=seg)
            out.append({"pts": C2, "w": fade(len(C2), 0.55 * wc[j]), "kind": "crawl"})
    for c in range(inner):                                                         # inside the cloud: branches spreading outward (they light the cloud from inside)
        j = int(rng.integers(0, max(1, k0)))
        Ci = _branch(M, j, rng, rng.uniform(60, 90), rng.uniform(500.0, 1200.0), flat=0.7, z=base_z + rng.uniform(60.0, 1.2 * max(top[2] - base_z, 100.0)), seg=seg)
        out.append({"pts": Ci, "w": fade(len(Ci), rng.uniform(0.6, 0.9), 0.7), "kind": "cloud"})
    return out


def intracloud(cx, cy, L, H, zb, rng, n=2, seg=14.0):
    """Intracloud discharges of a storm mass (the storm's logic is inside the cloud: the cloud glows from inside, no channel is visible): n polylines spreading
    outward in the lower half (0.15–0.5 of the height above the base), 0.3–0.6 of the width long, with branches. → [{"pts", "w", "kind": "cloud"}]."""
    out = []
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi); Lc = rng.uniform(0.3, 0.6) * L; z = zb + rng.uniform(0.15, 0.5) * H
        p0 = np.array([cx + rng.normal(0, 0.12 * L) - 0.5 * Lc * math.cos(a), cy + rng.normal(0, 0.12 * L) - 0.5 * Lc * math.sin(a), z])
        p1 = p0 + np.array([Lc * math.cos(a), Lc * math.sin(a), rng.normal(0, 0.05 * H)])
        M = displace(p0, p1, rng, seg, amp=0.2, flat=0.6); out.append({"pts": M, "w": np.full(len(M), 1.0), "kind": "cloud"})
        for _ in range(int(rng.integers(2, 5))):                                    # branches outward
            j = int(rng.integers(2, len(M) - 2)); B = _branch(M, j, rng, rng.uniform(30, 70), rng.uniform(0.2, 0.45) * Lc, flat=0.6, seg=seg)
            out.append({"pts": B, "w": np.linspace(0.7, 0.1, len(B)), "kind": "cloud"})
    return out


def place(cands, gz, cams, scene=None, avoid=(), n=12):
    """The place where the channel leaves the cloud: cands — [(x, y, z of the base)] under the cloud; the score is the visible length of the descent to the ground in
    the cameras' frames (px, behind the relief — none), on a third of the frame, not on the castle (avoid — world points). → (x, y, z of the base, ground)."""
    best = None
    for x, y, zb in cands:
        zg = float(gz(x, y)); h = zb - zg
        if h < 150.0: continue
        P = np.stack([np.full(n, x), np.full(n, y), zg + 15.0 + (np.arange(n) + 0.5) / n * (h - 30.0)], -1); sc = 0.0
        for cam in cams:
            W, H = cam.get("resolution", (960, 540)); xy, z = project(cam, P, W, H)
            ok = (z > 50.0) & (xy[:, 0] > 0.06 * W) & (xy[:, 0] < 0.94 * W) & (xy[:, 1] > 0.04 * H) & (xy[:, 1] < 0.96 * H)
            if scene and ok.any(): ok &= terrain_vis(cam, P, scene) > 0
            seg = ok[1:] & ok[:-1]; L = float(np.linalg.norm(np.diff(xy, axis=0), axis=1)[seg].sum())
            if not L: continue
            u = float(np.mean(xy[ok, 0])) / W; third = 1.0 - 0.6 * min(abs(u - 1 / 3), abs(u - 2 / 3)) / 0.17
            for A in avoid:
                axy, az_ = project(cam, np.asarray(A, float)[None], W, H)
                if az_[0] > 0 and abs(axy[0, 0] / W - u) < 0.08: third *= 0.2
            sc += L * max(third, 0.1) / H
        if best is None or sc > best[0]: best = (sc, x, y, zb, zg)
    return best[1:] if best and best[0] > 0 else None


def samples(B, step=100.0, k=None):
    """Channel points for baking every ~step m: world xyz and the weight = segment strength × length (m); k — a multiplier by kind."""
    k = k or {"main": 1.0, "branch": 1.0, "crawl": 1.0, "cloud": 1.3}; P, W = [], []
    for b in B:
        p = np.asarray(b["pts"], float); s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        n = max(1, int(round(s[-1] / step))); t = (np.arange(n) + 0.5) * s[-1] / n
        P.append(np.stack([np.interp(t, s, p[:, q]) for q in range(3)], -1)); W.append(np.interp(t, s, np.asarray(b["w"], float)) * s[-1] / n * k.get(b["kind"], 1.0))
    return np.concatenate(P), np.concatenate(W)


def project(cam, P, W, H):
    """World → frame pixels (the centre of pixel i is i + 0.5) and the depth along the view; as in rainbow.rays (the sensor width is by width)."""
    M = np.array(cam["matrix_world"], float); pc = (np.asarray(P, float) - M[:3, 3]) @ M[:3, :3]
    z = -pc[:, 2]; k = cam.get("lens_mm", 35.0) / (cam.get("sensor_width_mm", 36.0) / 2); zz = np.maximum(z, 1e-6)
    u, v = k * pc[:, 0] / zz, -k * pc[:, 1] / zz
    return np.stack([(u + 1) / 2 * W, (v * W / H + 1) / 2 * H], -1), z


def terrain_vis(cam, P, scene, step=25.0):
    """Whether a point is visible from the camera over the relief (the scene's height map): 1 — visible, 0 — behind a mountain."""
    from components.clouds import terrain as T
    z, x0, y0, st = T.heights(scene); n0, n1 = z.shape; p0 = np.array(cam["matrix_world"], float)[:3, 3]
    d = np.asarray(P, float) - p0; L = np.linalg.norm(d, axis=1); u = d / np.maximum(L, 1e-6)[:, None]; vis = np.ones(len(d)); t = 30.0
    while True:
        al = (t < L - 30.0) & (vis > 0)
        if not al.any(): break
        Q = p0 + u[al] * t; i = np.clip(((Q[:, 0] - x0) / st).astype(int), 0, n0 - 1); j = np.clip(((Q[:, 1] - y0) / st).astype(int), 0, n1 - 1)
        vis[np.nonzero(al)[0][Q[:, 2] < z[i, j] - 2.0]] = 0.0; t += step * (1 + t / 3000.0)
    return vis


def tau_grid(rho, vs, to_idx, A, B, sigma, n=96):
    """σ∫ρ ds along segments A→B (world) inside the cloud grid; to_idx — world → cell indices (a cell centre is an integer)."""
    from components.clouds.light import sample
    ia, ib = to_idx(np.asarray(A, float)), to_idx(np.asarray(B, float)); d = ib - ia; lo, hi = -0.5, np.array(rho.shape) - 0.5
    t0, t1 = np.zeros(len(ia)), np.ones(len(ia))
    for ax in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - ia[:, ax]) / d[:, ax], (hi[ax] - ia[:, ax]) / d[:, ax]
        inside = (ia[:, ax] >= lo) & (ia[:, ax] <= hi[ax]); nz = d[:, ax] != 0
        t0 = np.maximum(t0, np.where(nz, np.minimum(ta, tb), np.where(inside, -np.inf, np.inf)))
        t1 = np.minimum(t1, np.where(nz, np.maximum(ta, tb), np.where(inside, np.inf, -np.inf)))
    ok = t1 > t0; tau = np.zeros(len(ia))
    if ok.any():
        u = (np.arange(n) + 0.5) / n; T_ = t0[ok, None] + (t1 - t0)[ok, None] * u[None, :]
        Q = ia[ok, None, :] + T_[..., None] * d[ok, None, :]
        s = sample(rho, Q.reshape(-1, 3).astype(np.float32)).reshape(Q.shape[:2])
        tau[ok] = sigma * s.mean(1) * np.linalg.norm(d[ok], axis=1) * (t1 - t0)[ok] * vs
    return tau


def _gauss(a, s):
    """A Gaussian with σ = s px by three box filters (cumulative sums), the edges extended."""
    w = max(1, int(round(math.sqrt(12 * s * s / 3 + 1)))) | 1; r = w // 2
    for _ in range(3):
        for ax in (0, 1):
            pad = [(0, 0), (0, 0)]; pad[ax] = (r + 1, r); c = np.cumsum(np.pad(a, pad, mode="edge"), ax)
            a = (np.take(c, np.arange(w, c.shape[ax]), ax) - np.take(c, np.arange(0, c.shape[ax] - w), ax)) / w
    return a


FLASH = ((0.0, 0.0), (0.15, 0.1), (1.0, 1.0), (0.35, 0.5), (0.85, 0.9), (0.25, 0.4), (0.1, 0.2), (0.5, 0.6), (0.12, 0.25), (0.0, 0.1),
         (0.0, 0.05), (0.0, 0.0))   # 12 fps: (channel strength, cloud strength) — leader, strike, afterglow, repeated strikes, fade-out


def draw(png, cam, B, tau=None, scene=None, style=None, ss=4, gain=1.0, vis=None):
    """The channel on a frame over a finished frame (sRGB; a path or a 0–1 array): the core is thin lines (segment strength × visibility), the halo and air glow are
    Gaussians of the core; visibility is the relief (scene) × exp(−τ of the clouds to the camera); gain — the channel strength (the flash over time), vis — ready
    visibility per branch (do not recompute the relief). → Image."""
    from PIL import Image, ImageDraw                                              # PIL is available only in the project's Python (Blender has none)
    S = dict(STYLE, **(style or {})); im = (np.asarray(Image.open(png).convert("RGB"), np.float32) / 255.0) if isinstance(png, str) else np.asarray(png, np.float32)
    H, W = im.shape[:2]
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4); sc = W / 960.0
    core = Image.new("F", (W * ss, H * ss), 0.0); dr = ImageDraw.Draw(core)
    for bi, b in enumerate(B):
        P = np.asarray(b["pts"], float); xy, z = project(cam, P, W, H); w = np.asarray(b["w"], float) * gain
        if vis is not None: v_ = vis[bi]
        else:
            v_ = (z > 5.0).astype(float)
            if scene: v_ *= terrain_vis(cam, P, scene)
            if tau is not None: v_ *= np.exp(-np.asarray(tau[bi], float) * (S["veins"] if b["kind"] == "cloud" else 1.0)) * (S["veins_k"] if b["kind"] == "cloud" else 1.0)
        for k in range(len(P) - 1):
            I = 0.5 * (w[k] + w[k + 1]) * 0.5 * (v_[k] + v_[k + 1])
            if I < 0.02 or min(z[k], z[k + 1]) <= 5.0: continue
            wd = max(0.55, S["core"] * sc * math.sqrt(0.5 * (w[k] + w[k + 1]) / max(gain, 1e-6))) * ss   # the width depends on the segment strength, not on the flash
            a_, b_ = tuple(xy[k] * ss), tuple(xy[k + 1] * ss); dr.line([a_, b_], fill=float(I), width=max(1, int(round(wd))))
            if wd > 2.5: r_ = wd / 2; dr.ellipse([b_[0] - r_, b_[1] - r_, b_[0] + r_, b_[1] + r_], fill=float(I))   # a joint without a gap
    C = np.asarray(core.resize((W, H), Image.BOX), np.float32)
    out = lin + S["core_k"] * C[..., None] * np.array(S["core_rgb"])
    w_core = max(S["core"] * sc, 1.0)
    for s_, k_, col in S["glow"]:
        g = _gauss(C, s_ * sc) * (math.sqrt(2 * math.pi) * s_ * sc / w_core)      # the halo peak of a single line ≈ 1
        out = out + k_ * g[..., None] * np.array(S[col + "_rgb"])
    kn = 0.8; out = np.where(out > kn, kn + (1 - kn) * (1 - np.exp(-(out - kn) / (1 - kn))), out)   # a soft shoulder: the core goes to white
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * np.maximum(out, 0) ** (1 / 2.4) - 0.055)
    return Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def save(path, B, tau=None, extra=None):
    json.dump({"branches": [{"pts": np.round(np.asarray(b["pts"]), 2).tolist(), "w": np.round(np.asarray(b["w"]), 4).tolist(), "kind": b["kind"]} for b in B],
               "tau": {k: [np.round(np.asarray(t), 4).tolist() for t in v] for k, v in (tau or {}).items()}, **(extra or {})}, open(path, "w"))


def load(path):
    d = json.load(open(path)); return [{"pts": np.array(b["pts"]), "w": np.array(b["w"]), "kind": b["kind"]} for b in d["branches"]], d


if __name__ == "__main__":
    a = sys.argv[1:]; d0 = a[0]; opt = dict(x.split("=", 1) for x in a[1:] if "=" in x); names = [x for x in a[1:] if "=" not in x] or ["far", "medium", "high"]
    B, D = load(os.path.join(d0, "bolt.json")); st = {k: float(v) for k, v in opt.items() if k in ("core", "core_k", "veins", "veins_k")}
    out_sfx = opt.get("suffix", "")
    from PIL import Image
    lin = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4); srgb = lambda a: np.where(a <= 0.0031308, a * 12.92, 1.055 * np.maximum(a, 0) ** (1 / 2.4) - 0.055)
    for nm in names:
        raw = os.path.join(d0, f"{nm}_raw.png")
        if not os.path.exists(raw): continue
        cam = json.load(open(os.path.join(d0, "lit", nm, "far.json")))["camera"]; sc_ = opt.get("scene", D.get("scene")); tau = D.get("tau", {}).get(nm)
        draw(raw, cam, B, tau, sc_, st).save(os.path.join(d0, f"{nm}{out_sfx}.png")); print(nm, "lightning")
        nf = os.path.join(d0, "noflash", f"{nm}.png")
        if opt.get("clip") and os.path.exists(nf):                                # the flash over time: clouds — a blend of frames, the channel — strength by a curve
            S_ = dict(STYLE, **st); V = []
            for bi, b in enumerate(B):
                P = np.asarray(b["pts"], float); _, z = project(cam, P, 1, 1); v_ = (z > 5.0).astype(float) * (terrain_vis(cam, P, sc_) if sc_ else 1.0)
                if tau is not None: v_ = v_ * np.exp(-np.asarray(tau[bi], float) * (S_["veins"] if b["kind"] == "cloud" else 1.0)) * (S_["veins_k"] if b["kind"] == "cloud" else 1.0)
                V.append(v_)
            A0, A1 = (lin(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0) for f in (nf, raw)); fr = []
            for ib, ic in FLASH:
                base = srgb(A0 + ic * (A1 - A0)); fr.append(draw(base, cam, B, style=st, gain=ib, vis=V) if ib > 0 else Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8)))
            fr[0].save(os.path.join(d0, f"{nm}_flash.gif"), save_all=True, append_images=fr[1:] + fr[-1:] * 6, duration=83, loop=0); print(nm, "clip", len(fr), "n_frames")
            import shutil, subprocess, tempfile                                    # mp4 — without the palette steps of gif
            if shutil.which("ffmpeg"):
                with tempfile.TemporaryDirectory() as td:
                    for i_, f_ in enumerate(fr + fr[-1:] * 6): f_.save(os.path.join(td, f"{i_:03d}.png"))
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", os.path.join(td, "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-crf", "16", os.path.join(d0, f"{nm}_flash.mp4")], check=False)
