"""Precomputed cloud light: toward the sun — steps with doubling, single HG scattering, multiple scattering by diffusion on a coarse grid
(Stam 1995, δ-similarity, the Marshak boundary, P1 radiance toward the camera, exchange through gaps), a shading pattern finer than a cell; the result is
the emission σρ·J (RGB) for an "absorption + emission" volume (Eevee)."""
import math
import numpy as np
from components.clouds.forms import block_mean


def sample(rho, idx):
    """Trilinear ρ at cell indices (…, 3); outside the grid — 0."""
    n = np.array(rho.shape); i0 = np.floor(idx).astype(np.int64); w = (idx - i0).astype(np.float32); out = np.zeros(idx.shape[:-1], np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                j = i0 + np.array([dx, dy, dz]); ok = np.all((j >= 0) & (j < n), -1); jj = np.where(ok[..., None], j, 0)
                ww = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1]) * (w[..., 2] if dz else 1 - w[..., 2])
                out += np.where(ok, rho[jj[..., 0], jj[..., 1], jj[..., 2]], 0.0) * ww
    return out


def tau_sun(rho, vs, s, sigma, reach=3000.0, chunk=400000, jitter=True):
    """Optical depth toward the sun from each occupied cell: steps with doubling (½, 1, 2, 4 … cells) up to reach, trilinear; the sample points are randomly
    shifted by up to a cell within a step (jitter) — steps gave iso-line rings on the surface."""
    s = np.asarray(s, np.float64); s /= np.linalg.norm(s)
    t = [0.5 * vs]; dt = [0.75 * vs]
    while t[-1] < reach: dt.append(dt[-1] * 1.5); t.append(t[-1] + 0.5 * (dt[-2] + dt[-1]))
    t, dt = np.array(t), np.array(dt); occ = np.argwhere(rho > 0.002); out = np.zeros(rho.shape, np.float32)
    for a in range(0, len(occ), chunk):
        c = occ[a:a + chunk].astype(np.float32)
        acc = np.zeros(len(c), np.float32); u = (np.random.default_rng(a).random(len(c), dtype=np.float32) - 0.5) if jitter else np.zeros(len(c), np.float32)
        for tk, dk in zip(t, dt):
            acc += sample(rho, c + ((tk + u * dk) / vs)[:, None] * s[None, :].astype(np.float32)) * dk
        out[tuple(occ[a:a + chunk].T)] = acc * sigma
    return out


def tau_point(rho, vs, P, sigma, chunk=400000, steps=28):
    """Optical depth from each occupied cell to point P (cell indices; the point is usually outside the grid — a fire on the ground): the sample points on the
    segment are denser near the cell (a quadratic grid along the path fraction), outside the grid the density is 0. → (τ, distance in m)."""
    occ = np.argwhere(rho > 0.002); tau = np.zeros(rho.shape, np.float32); P = np.asarray(P, np.float32)
    dist = np.full(rho.shape, 1e9, np.float32)                                    # empty cells are "infinitely far" (otherwise 1/d² is huge — sparks)
    u = ((np.arange(steps) + 0.5) / steps) ** 2; du = np.diff(np.r_[0.0, u])
    for a in range(0, len(occ), chunk):
        c = occ[a:a + chunk].astype(np.float32); d = P[None, :] - c; L = np.linalg.norm(d, axis=1)
        acc = np.zeros(len(c), np.float32)
        for uk, dk in zip(u, du): acc += sample(rho, c + uk * d) * dk
        tau[tuple(occ[a:a + chunk].T)] = acc * L * vs * sigma; dist[tuple(occ[a:a + chunk].T)] = L * vs
    return tau, dist


def hg(c, g): return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * c) ** 1.5)


def phase2(c, k=1.0, gf=0.8, gb=-0.35, wb=0.3):
    """Cloud phase function with two lobes (forward — the rim against the sun, backward — the bright side toward the sun when lit from behind; with a single HG
    g = 0.7 the sun-facing side of the cloud went dark and looked like smoke); k — smoothing for the higher octaves."""
    return (1 - wb) * hg(c, gf * k) + wb * hg(c, gb * k)


def upsample(c, f, shape):
    """Coarse grid (block centres) → fine, trilinear, axis by axis; the last axis c holds the columns."""
    out = c
    for ax, n in enumerate(shape):
        t = np.clip((np.arange(n) + 0.5) / f - 0.5, 0, c.shape[ax] - 1); i0 = np.minimum(np.floor(t).astype(np.int64), c.shape[ax] - 2 if c.shape[ax] > 1 else 0)
        w = (t - i0).astype(np.float32); sh = [1] * out.ndim; sh[ax] = n; w = w.reshape(sh)
        out = np.take(out, i0, ax) * (1 - w) + np.take(out, np.minimum(i0 + 1, c.shape[ax] - 1), ax) * w
    return out.astype(np.float32)


def _gaps(act, ax):
    """Gaps along an axis: cell i, empty space behind it, → the first occupied cell j farther along the axis; the number of empty cells between them is L."""
    A_ = np.moveaxis(act, ax, -1); n = A_.shape[-1]
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(A_, np.arange(n), n), -1), -1), -1)
    ok = A_[..., :-1] & ~A_[..., 1:] & (nxt[..., 1:] < n); idx = np.nonzero(ok); i_, j_ = idx[-1], nxt[..., 1:][ok]
    orig = lambda last: tuple(list(idx[:-1])[:ax] + [last] + list(idx[:-1])[ax:])
    return orig(i_), orig(j_), j_ - i_ - 1


def diffusion(rho, vs, ts, sigma, g=0.85, f=4, iters=600, tol=2e-4, thr=0.05, view_dir=None, gap=(0.8, 3.0), extra=()):
    """Multiple scattering by diffusion (Stam 1995; δ-similarity: σ' = σ(1−g), isotropic scattering): the flux φ on a coarse grid (cell f·vs) — the source inside
    the cloud is the sun after the first scattering σ'ρ·exp(−(1−g)τ☉), at the boundary with empty space — the Marshak condition (the input is the sky from above,
    the ground from below, the sides half and half). A gap between puffs: a face sees the wall opposite with the weight gap[0]·exp(−(L−1)/gap[1]) (L is the gap
    width in cells; a narrow gap is transparent, as if the puffs touched), the rest is sky; without this the light went into every fold and the cloud darkened
    with a fine grid. Conjugate gradients, matrix-free. Returns the P1 radiance (φ + 3·J·ω)/(4π) toward the camera (view_dir points from the cloud to the camera;
    without it — φ/4π) on the fine grid, columns: the sun (per unit of illumination), the sky, the ground (per unit of radiance). Without the flux J the edge is
    darker: by Marshak, φ/2 leaves outward.
    The octaves lacked energy deep inside the cloud (the sum aⁱ·exp(−bⁱτ) dies out within about ten τ) — folds and shadow came out as "smoke"."""
    h = f * vs; sp = sigma * (1 - g)
    st = sp * block_mean(rho, f); act = st * h > thr
    q = block_mean(sp * rho * np.exp(-(1 - g) * ts), f) * h                  # the source per cell (the flux through the faces is divided by h²)
    r = np.where(act, 1.5 * h * st, 0.0)                                       # half a cell: h/(2D), D = 1/(3σ')
    ne = len(extra); shp = act.shape; diag = np.zeros(shp); b = np.zeros(shp + (3 + ne,)); Gf = []; Bd = []; Pr = []   # extra — its own sources (the moon, lightning)
    for ax in range(3):
        a0 = [slice(None)] * 3; a1 = [slice(None)] * 3; a0[ax] = slice(0, -1); a1[ax] = slice(1, None); a0, a1 = tuple(a0), tuple(a1)
        G = np.where(act[a0] & act[a1], 1.0 / np.maximum(r[a0] + r[a1], 1e-9), 0.0); Gf.append(G); diag[a0] += G; diag[a1] += G
        I, J, L = _gaps(act, ax); w = gap[0] * np.exp(-(L - 1) / gap[1]) if gap else np.zeros(len(L))
        Gp = w / np.maximum(r[I] + r[J], 1e-9); diag[I] += Gp; diag[J] += Gp
        Pr.append((ax, np.ravel_multi_index(I, shp), np.ravel_multi_index(J, shp), Gp))
        for side, wv in ((1, (1.0, 0.0) if ax == 2 else (0.5, 0.5)), (-1, (0.0, 1.0) if ax == 2 else (0.5, 0.5))):   # (sky, ground) at the input
            nb = np.zeros(shp, bool); sl = [slice(None)] * 3; e = [slice(None)] * 3
            if side == 1: sl[ax] = slice(0, -1); nb[tuple(sl)] = ~act[a1]; e[ax] = -1
            else: sl[ax] = slice(1, None); nb[tuple(sl)] = ~act[a0]; e[ax] = 0
            nb[tuple(e)] = True; Gb = np.where(nb & act, 1.0 / (r + 2.0), 0.0)  # a face toward empty space: 1/(h/2D + 2), φ_b = 4π·L
            Gb[I if side == 1 else J] *= 1 - w; diag += Gb; pb = 4 * math.pi * np.array([0.0, wv[0], wv[1]] + [0.0] * ne)
            b[..., 1] += Gb * pb[1]; b[..., 2] += Gb * pb[2]; Bd.append((ax, side, Gb, pb))
    b[..., 0] = q
    for i_, e_ in enumerate(extra): b[..., 3 + i_] = block_mean(e_, f) * h               # the source of a column is a fine-grid field (per unit)
    b *= act[..., None]; diag = np.where(act, diag, 1.0)[..., None]

    def A(x):
        y = diag * x
        for ax, G in enumerate(Gf):
            a0 = [slice(None)] * 4; a1 = [slice(None)] * 4; a0[ax] = slice(0, -1); a1[ax] = slice(1, None); a0, a1 = tuple(a0), tuple(a1)
            y[a0] -= G[..., None] * x[a1]; y[a1] -= G[..., None] * x[a0]
        yf, xf = y.reshape(-1, y.shape[-1]), x.reshape(-1, x.shape[-1])
        for _, I, J, Gp in Pr: yf[I] -= Gp[:, None] * xf[J]; yf[J] -= Gp[:, None] * xf[I]
        return y * act[..., None]
    x = np.zeros_like(b); res = b.copy(); z = res / diag; p = z.copy(); rz = (res * z).sum((0, 1, 2)); b2 = np.maximum((b * b).sum((0, 1, 2)), 1e-30)
    for k in range(iters):
        Ap = A(p); al = rz / np.maximum((p * Ap).sum((0, 1, 2)), 1e-30); x += al * p; res -= al * Ap
        if np.all((res * res).sum((0, 1, 2)) / b2 < tol * tol): break
        z = res / diag; rz2 = (res * z).sum((0, 1, 2)); p = z + (rz2 / np.maximum(rz, 1e-30)) * p; rz = rz2
    if view_dir is not None:                                                   # flux J across faces (toward empty space — by Marshak, across a gap — toward the wall), P1 toward the camera
        w_ = np.asarray(view_dir, np.float64); w_ = w_ / np.linalg.norm(w_); Jw = np.zeros_like(x); Jf, xf = Jw.reshape(-1, x.shape[-1]), x.reshape(-1, x.shape[-1])
        for ax, G in enumerate(Gf):
            a0 = [slice(None)] * 4; a1 = [slice(None)] * 4; a0[ax] = slice(0, -1); a1[ax] = slice(1, None); a0, a1 = tuple(a0), tuple(a1)
            Fc = G[..., None] * (x[a0] - x[a1]); Jw[a0] += 0.5 * w_[ax] * Fc; Jw[a1] += 0.5 * w_[ax] * Fc
        for ax, I, J, Gp in Pr: Fc = Gp[:, None] * (xf[I] - xf[J]); Jf[I] += 0.5 * w_[ax] * Fc; Jf[J] += 0.5 * w_[ax] * Fc
        for ax, side, Gb, pb in Bd: Jw += 0.5 * w_[ax] * side * Gb[..., None] * (x - pb)
        x = np.maximum(x + 3.0 * Jw * act[..., None], 0.0)
    for _ in range(2):                                                         # empty cells at the edge get the mean of their neighbours (otherwise the edge is pulled to zero)
        s = np.zeros_like(x); c = np.zeros(shp)
        for ax in range(3):
            for d in (1, -1): s += np.roll(x, d, ax); c += np.roll(act, d, ax)
        fill = ~act & (c > 0); x[fill] = s[fill] / c[fill][..., None]; act = act | fill
    x[~act] = 4 * math.pi * np.array([0.0, 0.5, 0.5] + [0.0] * ne)            # far from the bulk — free space: half sky, half ground
    wc = block_mean((rho > 0.002).astype(np.float32), f)[..., None]           # normalised interpolation: the weight is the fraction of occupied cells
    U = upsample(x * wc / (4 * math.pi), f, rho.shape) / np.maximum(upsample(wc, f, rho.shape), 1e-4)   # (the edge is not mixed with the void)
    for ax in range(3):                                                        # a window of f + 1 along each axis: trilinear + box — no kinks
        w = f + 1; pd = [(0, 0)] * 4; pd[ax] = (w // 2 + 1, w - w // 2 - 1); c_ = np.cumsum(np.pad(U, pd, mode="edge"), ax, dtype=np.float32)
        U = (np.take(c_, np.arange(w, c_.shape[ax]), ax) - np.take(c_, np.arange(0, c_.shape[ax] - w), ax)) / w   # (on the coarse grid's faces — banding)
    return U.astype(np.float32), k + 1


def light(rho, o, vs, P, sun_dir, view_c, sigma=0.06, g=0.7, ms_a=0.5, ms_b=0.5, gain=1.0, z0=0.0, octaves=4, powder=0.0, ground=0.5, ground_neutral=False,
          ms_mode="octaves", g_ms=0.85, coarse=4, view_dir=None, gap=(0.8, 3.0), detail=1.0, ts_blur=1, info=None, flashes=(), points=(), channel=(),
          ch_dir=None, ch_detail=None, ch_ms=1.0, sky_detail=0.0, sky_r=None):
    """Light of each cell (units of the node's plan): the sun at the cell's height × Σ over octaves aⁱ·exp(−bⁱτ)·HG(c, g·0.5ⁱ) × "powder" + ½·sky·exp(−¼τ↑) +
    ½·ground·exp(−¼τ↓) → the emission σρ·L (RGB). view_c — the cosine between the view of the cloud and the sun (a distant cloud — one value).
    "Powder" (Schneider 2015): right at the rim it is darker toward the sun and lighter deeper in; visible when looking away from the sun (c < 0).
    ms_mode="diff": the octaves are only the near orders (octaves = 1–2), the far ones, the sky and the ground go through diffusion (diffusion); the shading
    pattern finer than the coarse cell is the ratio of exp(−k·τ☉) on the fine grid to its mean in the cell (k = detail/(σ·h), h is the coarse cell: one knob for
    any density and step):
    the cell's energy stays the same while the puffs become visible again (an opaque volume shows only the source function, a smooth one looks like "cotton wool").
    sky_detail — the same pattern for scattered light (sky, ground, flashes inside): without a sun (a storm, dusk) the diffusion on the coarse grid is even —
    small clouds and gaps shine equally. The occlusion measure is the neighbourhood density within radius sky_r m (by default 2 coarse cells; for small clouds —
    their radius): above a cell — light from the sky, below — from the ground; exp(−sky_detail·ρ̄/ρmax) relative to the mean in the coarse cell — the same energy,
    open areas lighter, gaps darker."""
    ts = tau_sun(rho, vs, sun_dir, sigma)
    if ms_mode == "diff":                                                      # τ☉ smoothed with a 3³ filter ts_blur times: voxel steps gave rings on the surface
        for _ in range(ts_blur):
            for ax in range(3): ts = (np.roll(ts, 1, ax) + ts + np.roll(ts, -1, ax)) / 3.0
    z = (np.arange(rho.shape[2]) + 0.5) * vs + o[2] + z0
    sa = P["light"]["sun_alt"]; H = np.linspace(0, sa["h_max"], len(sa["rgb"])); A = np.array(sa["rgb"])
    Esun = np.stack([np.interp(z, H, A[:, q]) for q in range(3)], -1)
    ph = (lambda i: phase2(view_c, 0.5 ** i)) if g == "dual" else (lambda i: hg(view_c, g * 0.5 ** i))
    ms = sum((ms_a ** i) * np.exp(-(ms_b ** i) * ts) * ph(i) for i in range(octaves))
    if powder > 0: ms = ms * (1 - powder * (0.5 - 0.5 * view_c) * np.exp(-2.0 * ts))
    Lsky = np.array(P["light"]["sky_e_rgb"]) / math.pi + np.array(P["light"].get("night_sky_e_rgb", [0, 0, 0])) / math.pi
    Lgnd = np.array(P["light"]["ground_up_rgb"])
    if ground_neutral: Lgnd = np.full(3, float(np.dot([0.2126, 0.7152, 0.0722], Lgnd)))   # a grey bounce: a warm one makes a "dirty" underside
    if ms_mode == "diff":
        sp = sigma * (1 - g_ms); extra = []; Mo = P.get("moon", {}); md = np.asarray(Mo.get("dir", [0, 0, -1]), float)
        moon = bool(Mo.get("on")) and md[2] > 0.02 and max(Mo.get("lamp_rgb", [0])) > 0
        if moon:                                                               # the moon is a second source: its own shadow and its own scattering
            tm = tau_sun(rho, vs, md, sigma)
            for _ in range(ts_blur):
                for ax in range(3): tm = (np.roll(tm, 1, ax) + tm + np.roll(tm, -1, ax)) / 3.0
            extra.append(sp * rho * np.exp(-(1 - g_ms) * tm))
        if flashes:                                                            # lightning — point sources inside the mass (cell indices, radius[, strength])
            fl = np.zeros(rho.shape, np.float32); I3 = np.indices(rho.shape, dtype=np.float32)
            for e_ in flashes:
                (fi, fj, fk), rr = e_[0], e_[1]; fl += (e_[2] if len(e_) > 2 else 1.0) * np.exp(-((I3[0] - fi) ** 2 + (I3[1] - fj) ** 2 + (I3[2] - fk) ** 2) / (2 * rr * rr)).astype(np.float32)
            extra.append(fl * (rho > 0.002)); del I3
        CH = None
        if channel:                                                            # lightning channel: points (cell indices, strength·m²) — on the coarse grid:
            f_ = coarse; rc = block_mean(rho, f_); Ec = np.zeros(rc.shape, np.float32); Sc = np.zeros(rc.shape, np.float32); d0 = 1.5 * f_ * vs
            for Pp, w_ in channel:                                             # single scattering w·e^−τ/d² and a diffusion source (δ-similarity)
                tp, dp = tau_point(rc, vs * f_, (np.asarray(Pp, np.float32) + 0.5) / f_ - 0.5, sigma); inv = 1.0 / np.maximum(dp * dp, d0 * d0)
                Ec += (w_ * np.exp(-tp) * inv).astype(np.float32); Sc += (w_ * np.exp(-(1 - g_ms) * tp) * inv).astype(np.float32)
            extra.append(sp * rho * upsample(Sc[..., None], f_, rho.shape)[..., 0]); CH = upsample(Ec[..., None], f_, rho.shape)[..., 0]; del rc, Ec, Sc
            dk_ = detail if ch_detail is None else ch_detail                   # ch_detail — the flash's own pattern strength (sculpting the puffs by the channel's light)
            if dk_ > 0 and ch_dir is not None:                                 # the shading pattern toward the channel — as for the sun: the ratio to the cell mean
                tc = tau_sun(rho, vs, ch_dir, sigma)
                for _ in range(ts_blur):
                    for ax in range(3): tc = (np.roll(tc, 1, ax) + tc + np.roll(tc, -1, ax)) / 3.0
                occ_ = (rho > 0.002).astype(np.float32); A_ = np.exp(-dk_ / (sigma * vs * coarse) * tc) * occ_
                m_ = upsample((block_mean(A_, coarse) / np.maximum(block_mean(occ_, coarse), 1e-6))[..., None], coarse, rho.shape)[..., 0]
                CH *= np.where(occ_ > 0, np.clip(A_ / np.maximum(m_, 1e-6), 0.25, 3.0), 1.0); del tc, A_, m_, occ_
        PT = []                                                                    # point sources outside the cloud (fires): ((indices), I rgb W/sr in plan units)
        for Pp, I in points:
            tp, dp = tau_point(rho, vs, Pp, sigma); Ep = np.exp(-tp) / np.maximum(dp * dp, 1.0); PT.append((Ep, np.asarray(I, float)))
            extra.append(sp * rho * np.exp(-(1 - g_ms) * tp) / np.maximum(dp * dp, 1.0))
        F, n = diffusion(rho, vs, ts, sigma, g_ms, coarse, view_dir=view_dir, gap=gap, extra=extra)
        if info is not None: info["cg_iterations"] = n
        rs_ = {}
        if sky_detail > 0:                                                     # the pattern of scattered light: neighbourhood density → occlusion
            from scipy.ndimage import gaussian_filter, shift as nd_shift
            occ_s = (rho > 0.002).astype(np.float32); cnt_s = np.maximum(block_mean(occ_s, coarse), 1e-6)
            sr_ = max(1.0, (sky_r or 2.0 * coarse * vs) / vs); nb = gaussian_filter(rho.astype(np.float32), sr_ / 2, mode="constant")
            nb /= max(float(rho.max()), 1e-6)
            for key_, dz_ in (("sky", -0.6 * sr_), ("gnd", 0.6 * sr_)):          # above a cell — the sky, below — the ground (not for flashes: the light is from inside,
                                                                                   # occlusion by neighbours would darken the small clouds into discs)
                A_ = np.exp(-sky_detail * nd_shift(nb, (0, 0, dz_), order=1, mode="nearest")) * occ_s
                m_ = upsample((block_mean(A_, coarse) / cnt_s)[..., None], coarse, rho.shape)[..., 0]
                rs_[key_] = np.where(occ_s > 0, np.clip(A_ / np.maximum(m_, 1e-6), 0.25, 3.0), 1.0).astype(np.float32); del A_, m_
            del nb, occ_s, cnt_s
            F[..., 1] *= rs_["sky"]; F[..., 2] *= rs_["gnd"]
        if detail > 0:
            occ = (rho > 0.002).astype(np.float32); cnt = np.maximum(block_mean(occ, coarse), 1e-6)
            A_ = np.exp(-detail / (sigma * vs * coarse) * ts) * occ              # toward the sun only (τ☉ is already smoothed): the sky shines as a hemisphere
            m_ = upsample((block_mean(A_, coarse) / cnt)[..., None], coarse, rho.shape)[..., 0]
            F[..., 0] *= np.where(occ > 0, np.clip(A_ / np.maximum(m_, 1e-6), 0.25, 3.0), 1.0); del A_, m_
        L = Esun[None, None, :, :] * (ms[..., None] + F[..., 0:1]) + F[..., 1:2] * Lsky + ground * F[..., 2:3] * Lgnd
        if moon:
            vd = np.asarray(view_dir, float); vc = float(-vd @ md / max(np.linalg.norm(vd), 1e-9)) if view_dir is not None else view_c
            L = L + np.asarray(Mo["lamp_rgb"], float) * (np.exp(-tm) * hg(vc, g if g != "dual" else 0.8) + F[..., 3])[..., None]; del tm
        if (flashes or channel) and info is not None:                              # lightning glow (per unit of the material's strength): flashes + channel
            fl_ = F[..., 3 + moon] if flashes else np.zeros(rho.shape, np.float32)
            if channel: fl_ = fl_ + CH / (4 * math.pi) + ch_ms * F[..., 3 + moon + bool(flashes)]   # ch_ms — the fraction of multiple-scattered (soft) glow
            info["flash"] = (sigma * rho * fl_ * gain).astype(np.float32)
        for k_, (Ep, I) in enumerate(PT):                                          # fire: single scattering (almost isotropic from below) + diffusion
            L = L + I * (Ep / (4 * math.pi) + F[..., 3 + moon + bool(flashes) + bool(channel) + k_])[..., None]
        return (sigma * rho[..., None] * L * gain).astype(np.float32)
    cs = lambda ax_, rev: (np.flip(np.cumsum(np.flip(rho, ax_), ax_), ax_) if rev else np.cumsum(rho, ax_)) * sigma * vs
    tz, tg = cs(2, True), cs(2, False)                                       # up (toward the sky) and down (toward the ground)
    tside = np.minimum(np.minimum(cs(0, True), cs(0, False)), np.minimum(cs(1, True), cs(1, False)))   # toward the nearest side
    tsky = np.minimum(tz, tside)                                             # the sky comes in from the nearest side — from above or from the side
    L = Esun[None, None, :, :] * ms[..., None] + 0.5 * Lsky * np.exp(-0.25 * tsky)[..., None] + ground * Lgnd * np.exp(-0.25 * tg)[..., None]
    return (sigma * rho[..., None] * L * gain).astype(np.float32)
