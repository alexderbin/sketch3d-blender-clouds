"""Cloud shapes — numpy only: a density field ρ on a grid (the z axis is up), origin o (m), step vs (m). The kinds are in types.yaml.
A mass the way an artist makes it (mass_lobes: a silhouette of large soft puffs on domes; a tower with a gesture), cell layers (stratocumulus,
altocumulus), cirrus strokes."""
import math
import numpy as np


def vnoise2(X, Y, f, seed, L=256):
    lat = np.random.default_rng(seed).random((L, L)).astype(np.float32)
    def ax(v):
        t = v * f; i = np.floor(t).astype(np.int64); u = (t - i).astype(np.float32); return i % L, (i + 1) % L, u * u * (3 - 2 * u)
    (x0, x1, ux), (y0, y1, uy) = ax(X), ax(Y); ux, uy = ux[:, None], uy[None, :]
    g = lambda i, j: lat[i[:, None], j[None, :]]
    return (g(x0, y0) * (1 - ux) + g(x1, y0) * ux) * (1 - uy) + (g(x0, y1) * (1 - ux) + g(x1, y1) * ux) * uy


def vnoise2p(XX, YY, f, seed, L=256):
    """2D value noise at arbitrary points (arrays of one shape)."""
    lat = np.random.default_rng(seed).random((L, L)).astype(np.float32)
    tx, ty = XX * f, YY * f; ix, iy = np.floor(tx).astype(np.int64), np.floor(ty).astype(np.int64)
    ux, uy = (tx - ix).astype(np.float32), (ty - iy).astype(np.float32); ux, uy = ux * ux * (3 - 2 * ux), uy * uy * (3 - 2 * uy)
    g = lambda i, j: lat[i % L, j % L]
    return (g(ix, iy) * (1 - ux) + g(ix + 1, iy) * ux) * (1 - uy) + (g(ix, iy + 1) * (1 - ux) + g(ix + 1, iy + 1) * ux) * uy


def vnoise3(X, Y, Z, f, seed, L=64):
    lat = np.random.default_rng(seed).random((L, L, L)).astype(np.float32)
    def ax(v):
        t = v * f; i = np.floor(t).astype(np.int64); u = (t - i).astype(np.float32); return i % L, (i + 1) % L, u * u * (3 - 2 * u)
    (x0, x1, ux), (y0, y1, uy), (z0, z1, uz) = ax(X), ax(Y), ax(Z)
    g = lambda i, j, k: lat[i[:, None, None], j[None, :, None], k[None, None, :]]
    ux, uy, uz = ux[:, None, None], uy[None, :, None], uz[None, None, :]
    c00 = g(x0, y0, z0) * (1 - ux) + g(x1, y0, z0) * ux; c10 = g(x0, y1, z0) * (1 - ux) + g(x1, y1, z0) * ux
    c01 = g(x0, y0, z1) * (1 - ux) + g(x1, y0, z1) * ux; c11 = g(x0, y1, z1) * (1 - ux) + g(x1, y1, z1) * ux
    return (c00 * (1 - uy) + c10 * uy) * (1 - uz) + (c01 * (1 - uy) + c11 * uy) * uz


def fbm3(X, Y, Z, f, seed, octaves=4):
    s, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = s + amp * vnoise3(X, Y, Z, f * 2 ** o, seed + 17 * o); tot += amp; amp *= 0.5
    return s / tot


def worley(P, cell, seed, L=64):
    """Worley F1 for points P (N, 3): the distance to the nearest random point of the cells of size cell (27 neighbours)."""
    T = np.random.default_rng(seed).random((L, L, L, 3)).astype(np.float32)
    c = np.floor(P / cell).astype(np.int64); best = np.full(len(P), 1e9, np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cc = c + np.array([dx, dy, dz]); f = (cc + T[cc[:, 0] % L, cc[:, 1] % L, cc[:, 2] % L]) * cell
                best = np.minimum(best, np.sqrt(((P - f) ** 2).sum(1)))
    return best


def strato_layer(seed, ext=28000.0, thick=420.0, cell=1100.0, cover=0.62, vs=40.0, erode=0.5, size_var=0.0, lumps=0.0, lump_r=300.0):
    """Stratocumulus: cells (the distance to the nearest centre, Worley F1) — a dome per cell, gaps between them; eroded by noise."""
    rng = np.random.default_rng(seed)
    X = np.arange(-ext / 2, ext / 2, vs, dtype=np.float32); Y = np.arange(-ext / 2, ext / 2, vs, dtype=np.float32); Z = np.arange(0, thick + vs, vs, dtype=np.float32)
    nc = int(ext / cell) + 5; F = (np.arange(nc) - 2) * cell - ext / 2
    jit = rng.uniform(0.1, 0.9, (nc, nc, 2)) * cell; CX = F[:, None] + jit[..., 0]; CY = F[None, :] + jit[..., 1]
    hc = rng.uniform(0.5, 1.0, (nc, nc)).astype(np.float32)                   # each cell has its own height
    rc = rng.uniform(1 - size_var, 1 + 0.3 * size_var, (nc, nc)).astype(np.float32)   # its own size (fleecy puffs differ in size)
    wx = (vnoise2(X + 3e4, Y + 1e4, 1 / 2600.0, seed + 7) - 0.5) * cell * 1.2   # coordinate warping: rows and cells are uneven
    wy = (vnoise2(X + 5e4, Y + 2e4, 1 / 2600.0, seed + 8) - 0.5) * cell * 1.2
    XX, YY = X[:, None] + wx, Y[None, :] + wy
    ix = np.clip(((XX + ext / 2) / cell).astype(int) + 2, 1, nc - 2); iy = np.clip(((YY + ext / 2) / cell).astype(int) + 2, 1, nc - 2)
    d1 = np.full(XX.shape, 1e9, np.float32); h1 = np.zeros(XX.shape, np.float32); r1 = np.ones(XX.shape, np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = CX[ix + dx, iy + dy], CY[ix + dx, iy + dy]
            d = np.sqrt(((XX - cx) * 0.75) ** 2 + (YY - cy) ** 2)                 # cells are stretched along x (with the wind)
            dn = d / rc[ix + dx, iy + dy]                                         # distance as a fraction of its own size
            near = dn < d1; d1 = np.where(near, dn, d1); h1 = np.where(near, hc[ix + dx, iy + dy], h1)
    big = vnoise2(X + 9e4, Y + 7e4, 1 / 9000.0, seed + 5) * 0.6 + vnoise2(X, Y, 1 / 3000.0, seed + 6) * 0.4   # where the layer is denser, where it breaks up
    top = np.clip(1 - d1 / (cell * 0.78), 0, 1) ** 0.5 * h1 * np.clip((big - (1 - cover)) * 3.0, 0, 1)   # dome height of a cell 0–1
    zz = Z[None, None, :] / thick
    rho = np.clip((top[..., None] - zz) * 4.0, 0, 1) * np.clip(zz * 6.0, 0, 1)
    n = fbm3(X + 300, Y + 200, Z * 3 + 50, 1 / max(260.0, cell * 0.24), seed + 21, 3); e = erode * (1 - np.abs(2 * n - 1)); rho = np.clip((rho - e) / np.maximum(1 - e, 1e-3), 0, 1)
    if lumps > 0:                                                              # the veil's bottom in lumps (a storm sky from below — rolls and lumps, not a clean cut):
        under = rho[:, :, min(2, rho.shape[2] - 1)] > 0.3; dep = np.zeros(under.shape, np.float32)   # hemispheres hang below the base
        cand = np.argwhere(under); k_ = int(lumps * len(cand) * vs * vs / (math.pi * (1.1 * lump_r) ** 2))
        for i_, j_ in cand[rng.choice(len(cand), size=min(k_, len(cand)), replace=False)] if k_ else []:
            r = lump_r * rng.uniform(0.5, 1.5); m = int(r / vs) + 1; a0, a1, b0, b1 = max(0, i_ - m), min(len(X), i_ + m + 1), max(0, j_ - m), min(len(Y), j_ + m + 1)
            dd = ((X[a0:a1, None] - X[i_]) ** 2 + (Y[None, b0:b1] - Y[j_]) ** 2) / (r * r)
            dep[a0:a1, b0:b1] = np.maximum(dep[a0:a1, b0:b1], 0.8 * r * np.sqrt(np.clip(1 - dd, 0, 1)) * rng.uniform(0.6, 1.0))
        nb = int(dep.max() / vs) + 2; Zb = -(np.arange(nb, 0, -1) - 0.5) * vs     # layers below the base (z < 0), bottom to top
        low = np.clip((dep[..., None] + Zb[None, None, :]) / (0.6 * vs) + 0.5, 0, 1) * under[..., None]
        rho = np.concatenate([low * 0.9, rho], 2); Z = np.r_[Zb, Z]
    return rho.astype(np.float32), np.array([X[0], Y[0], Z[0]], np.float32), vs


def cirrus_layer(seed, ext=40000.0, thick=500.0, vs=80.0, cover=0.5, dens=0.12):
    """Cirrus: a thin layer of fibres — noise stretched 8 times along the wind (x), with warping; the density is low (the cloud is translucent)."""
    X = np.arange(-ext / 2, ext / 2, vs, dtype=np.float32); Y = np.arange(-ext / 2, ext / 2, vs, dtype=np.float32); Z = np.arange(0, thick + vs, vs, dtype=np.float32)
    w = (vnoise2(X + 1e4, Y + 3e4, 1 / 6000.0, seed) - 0.5) * 3000.0
    XX, YY = np.broadcast_to(X[:, None] / 8.0 + 2e4, w.shape), Y[None, :] + w
    n = np.zeros((X.size, Y.size), np.float32); amp = 1.0
    for o in range(4):
        n += amp * vnoise2p(XX, YY, 2 ** o / 1400.0, seed + 11 * o); amp *= 0.5
    n /= 1.875; big = vnoise2(X + 6e4, Y, 1 / 12000.0, seed + 2)
    s = np.clip((n - (0.62 - 0.25 * cover + 0.3 * (0.5 - big))) * 5.0, 0, 1) ** 1.5
    zz = Z[None, None, :] / thick; prof = np.clip(np.sin(np.pi * zz), 0, 1)
    return (dens * s[..., None] * prof).astype(np.float32), np.array([X[0], Y[0], Z[0]], np.float32), vs


def block_mean(x, f):
    """Mean over f×f×f blocks (the edge is padded with empty space)."""
    x = np.pad(x, [(0, (-n) % f) for n in x.shape]); n = [d // f for d in x.shape]
    return x.reshape(n[0], f, n[1], f, n[2], f).mean((1, 3, 5))


def _to_vs(r, k_):
    """Resampling a grid to another step by the nearest sample (k_ = old step / new)."""
    for ax in range(3): r = np.take(r, np.minimum((np.arange(max(1, int(round(r.shape[ax] * k_)))) / k_).astype(np.int64), r.shape[ax] - 1), ax)
    return r


def refvol(path="library/clouds/refvol/refvol_cloud_eighth.vdb"):
    """The sample cloud volume (CC BY-SA 3.0) in Blender axes (y up in the reference scene → z; its z axis → −y): density and step in file units; in the
    reference scene σ = 4·ρ per unit (the cloud is ~417 units)."""
    import openvdb as vdb
    g0 = vdb.read(path, "density"); lo, hi = g0.evalActiveVoxelBoundingBox(); dims = tuple(hi[i] - lo[i] + 1 for i in range(3))
    arr = np.zeros(dims, np.float32); g0.copyToArray(arr, ijk=lo)
    return np.ascontiguousarray(arr.transpose(0, 2, 1)[:, ::-1, :]), float(g0.transform.voxelSize()[0])


def refvol_cloud(base, W, vs_want, k_rot=0, flip=False, dens=1.0, sigma=0.06, stretch=(1.0, 1.0, 1.0), w_ref=None):
    """Cumulus made from the sample cloud volume: a rotation by 90°·k about the vertical, a reflection, stretching along the axes (by the nearest sample: flat —
    sz < 1, elongated — sx ≠ sy), width W (m), step ≈ vs_want (block reduction), the density as in the reference scene at this scale
    (σ = 4/m_per_unit · dens), recomputed for the material's σ. → (ρ, o, vs)."""
    rho0, u = base; r = np.rot90(rho0, k_rot, (0, 1)); r = r[::-1] if flip else r
    for ax, k_ in enumerate(stretch):
        if abs(k_ - 1.0) > 1e-3: r = np.take(r, np.minimum((np.arange(int(round(r.shape[ax] * k_))) / k_).astype(np.int64), r.shape[ax] - 1), ax)
    m = W / (r.shape[0] * u); f = max(1, int(round(vs_want / (u * m))))
    r = block_mean(r, f) if f > 1 else r; vs = u * m * f
    m_d = m if w_ref is None else w_ref / (rho0.shape[0] * u)                  # w_ref — the density as for a cloud of this width (one per mass)
    r = np.ascontiguousarray(r * (4.0 / m_d * dens / sigma), np.float32)
    return r, np.array([-r.shape[0] * vs / 2, -r.shape[1] * vs / 2, 0.0], np.float32), vs


def refvol_tower(base, W, vs_want, rng, levels=3, dens=1.0, sigma=0.06):
    """Towering cumulus (congestus) — puffs stacked on one another rather than a stretched cloud ("tongues of flame"): the base is the sample cloud of
    width W, above it 1–2 narrower copies (×0.62–0.75), each at half the height of the previous one, with its own rotation and a small shift;
    the common grid is the maximum of the densities. → (ρ, o, vs)."""
    r0, _, vs = refvol_cloud(base, W, vs_want, int(rng.integers(4)), bool(rng.random() < 0.5), dens, sigma, (1.0, 1.0, 0.9))
    parts = [(r0, 0.0, 0.0, 0.0)]; z = 0.5 * r0.shape[2] * vs; Wi = W
    for _ in range(1, levels):
        Wi *= rng.uniform(0.62, 0.75)
        ri, _, vi = refvol_cloud(base, Wi, vs_want, int(rng.integers(4)), bool(rng.random() < 0.5), dens, sigma, (1.0, 1.0, rng.uniform(0.95, 1.2)))
        k_ = vi / vs
        for ax in range(3): ri = np.take(ri, np.minimum((np.arange(int(round(ri.shape[ax] * k_))) / k_).astype(np.int64), ri.shape[ax] - 1), ax)
        dx, dy = rng.normal(0, 0.05 * W, 2); parts.append((ri, dx, dy, z)); z += 0.5 * ri.shape[2] * vs
    lo = np.min([[dx / vs - r.shape[0] / 2, dy / vs - r.shape[1] / 2] for r, dx, dy, _ in parts], 0)
    hi = np.max([[dx / vs + r.shape[0] / 2, dy / vs + r.shape[1] / 2] for r, dx, dy, _ in parts], 0)
    nz = int(max(z_ / vs + r.shape[2] for r, _, _, z_ in parts)) + 1
    out = np.zeros((int(hi[0] - lo[0]) + 2, int(hi[1] - lo[1]) + 2, nz), np.float32)
    for r, dx, dy, z_ in parts:
        i0, j0, k0 = int(dx / vs - r.shape[0] / 2 - lo[0]), int(dy / vs - r.shape[1] / 2 - lo[1]), int(z_ / vs)
        sl = out[i0:i0 + r.shape[0], j0:j0 + r.shape[1], k0:k0 + r.shape[2]]; np.maximum(sl, r[:sl.shape[0], :sl.shape[1], :sl.shape[2]], out=sl)
    return out, np.array([lo[0] * vs, lo[1] * vs, 0.0], np.float32), vs


def mass_lobes(L, H, rng, vs, peak=0.3, n_domes=6, scales=(0.42, 0.24, 0.13, 0.07, 0.04), soft=0.12, dens=1.5, levels=3, base_soft=0.0,
               aspect=0.45, accents=0, lean=0.0, kids_frac=1.0, tower=None, face=0.0, anvil=None, lens=0, domes=None, base_lumps=0.0, lump_r=0.06, realism=0.0):
    """A mass the way an artist paints it (Kuindzhi, Aivazovsky): a silhouette of large soft puffs of 2–3 sizes on domes of a large
    form — a dome, puffs of 0.42 radius on its top, 0.24 on their tops, finer 0.13; smooth merging (a sum of exp), a soft edge (soft · puff), no small detail.
    The base is flat (z ≥ 0). The silhouette hierarchy (backlight showed "beads" — an edge made of identical semicircles): accents > 0 — the dome's puffs are
    gathered into that many groups near the top (a calm edge between them), the sizes follow a power law (many small, a few large), a fraction kids_frac of them
    gets children; lean — the tilt (groups and heads lean toward +x when lean > 0); aspect — the dome's width to height (a low mound — 2–3, otherwise "columns").
    tower = tilt in degrees — a tower with a gesture instead of an envelope: domes along a curved ridge (L is the base width), narrower toward the top, a head at
    the top, shoulders at the base; face — the fraction of puffs allowed on the front side (−y, toward the camera): a large puff in front read as a disc.
    anvil = (width, drift) m — the anvil of a cumulonimbus: a flat smooth hat above the tower, blown off by the wind (+x), the ice sparser (×0.45).
    lens = n — lenticular: n smooth lenses in a stack without puffs. domes — custom domes [(x, y, z, a, b, c, puffs)]. base_lumps — the bottom of a storm
    in lumps (dark hanging lumps with light between them): the fraction of the base under lumps of radius lump_r·H (pouches and rolls along x), the grid
    extends below the base. realism 0–1: a wider soft edge, cut by "fleece" of three octaves of fine noise — small puffs along the edge;
    the light on the fine grid lights their faces. → (ρ, o, vs)."""
    us = np.sort(np.r_[peak, rng.uniform(0.03, 0.97, n_domes - 1)]); env = lambda u: H * (0.3 + 0.7 * math.exp(-((u - peak) / 0.3) ** 2))
    D = []
    if domes is not None: D = [tuple(d_) for d_ in domes]
    elif lens:                                                                 # lenses in a stack: smooth, without puffs, slightly shifted
        for i in range(lens):
            sc_ = 1 - 0.18 * i; D.append((L / 2 + rng.normal(0, 0.04 * L), 0.0, H * (0.5 + 1.4 * i) / (1.4 * lens), 0.5 * L * sc_, 0.32 * L * sc_, 0.5 * H / lens, 0))
    elif tower is not None:                                                      # the tower's ridge: z is a fraction of the height, a shift along an arc (the gesture), narrower toward the top
        top = math.tan(math.radians(tower)) * H; n = 5
        for i in range(n):
            t = i / (n - 1); a = 0.5 * L * (1 - 0.3 * t) * rng.uniform(0.92, 1.08)          # a third narrower toward the top: a narrow "neck" — a "teddy bear"
            D.append((L / 2 + top * t * t, rng.normal(0, 0.03 * L), t * 0.7 * H, a, 0.75 * a, 0.36 * H, 9))
        D.append((L / 2 + top * 1.02, 0.0, 0.76 * H, 0.27 * L, 0.22 * L, 0.2 * H, 9))                  # the head is smaller, merged with the body
        for sgn, w_ in ((-1, rng.uniform(0.5, 0.65)), (1, rng.uniform(0.35, 0.5))):                      # the shoulders are different and low
            D.append((L / 2 + sgn * 0.6 * L, rng.normal(0, 0.05 * L), 0.0, w_ * L, 0.7 * w_ * L, 0.2 * H * rng.uniform(0.8, 1.15), 9))
    else:
        for u in us:
            h = env(u) * rng.uniform(0.85, 1.05); a = aspect * h * rng.uniform(0.85, 1.15); D.append((u * L, rng.normal(0, 0.06 * L), 0.0, a, 0.7 * a, 0.85 * h, 9))
    if anvil is not None:                                                      # anvil: a flat hat above the tower along the wind + an "overshoot" dome
        aw, drift = anvil; xt = max(d_[0] for d_ in D if d_[2] > 0.5 * H) if any(d_[2] > 0.5 * H for d_ in D) else L / 2
        for k_ in range(-1, 5):                                                # a wedge along the wind (and a short projection against it): thicker above the tower, thinner toward the edge,
            t_ = k_ / 4; c_ = 0.16 * H * (1 - 0.65 * abs(t_)); a_ = aw * (0.22 + 0.12 * abs(t_)) / 2   # flat on top, the edge sags
            D.append((xt + (drift * 1.1 + aw * 0.3) * t_, rng.normal(0, 0.03 * aw), H - c_ - 0.03 * H * t_ * t_, a_, 0.85 * a_, c_, 3 if abs(k_) < 2 else 1))
        D.append((xt, 0.0, 0.9 * H, 0.16 * L, 0.13 * L, 0.12 * H, 4))               # the "overshoot" dome above the tower
    S = []                                                                     # puff spheres (x, y, z, r)
    def kids(x, y, z, r, k, lvl):
        if lvl >= levels: return
        acc = [(rng.uniform(0.0, 0.9), rng.normal(0.0 if lean >= 0 else math.pi, 0.9) if lean else rng.uniform(0, 2 * math.pi)) for _ in range(accents)]
        for _ in range(k):
            if lvl > 0 and rng.random() > kids_frac: continue
            if lvl == 0 and acc and rng.random() < 0.5:                        # a puff goes to an accent group near the top (half of them; the rest are texture)
                t0, f0 = acc[rng.integers(len(acc))]; th = float(np.clip(t0 + rng.normal(0, 0.22), 0.0, 1.45)); fi = f0 + rng.normal(0, 0.45)
            else: th = math.acos(rng.uniform(0.15, 1.0)); fi = rng.uniform(0, 2 * math.pi)
            front = math.sin(fi) < -0.5
            if face < 1.0 and front and rng.random() > max(face, 0.5 if levels > 3 else 0.0): fi = -fi; front = False   # from the front — to the silhouette or backward
            base_r = r * scales[lvl] / (scales[lvl - 1] if lvl else 1.0)
            rr = base_r * ((0.55 + 1.1 * rng.random() ** 3) if accents else rng.uniform(0.8, 1.2))   # a power-law spread: a few large ones
            if front and face < 1.0: rr *= 0.6                                 # on the front — smaller: texture, not a disc
            if lvl == 0: px, py, pz = x + a_ * math.sin(th) * math.cos(fi), y + b_ * math.sin(th) * math.sin(fi), z + c_ * math.cos(th)
            else: px, py, pz = x + r * math.sin(th) * math.cos(fi), y + r * math.sin(th) * math.sin(fi), z + r * math.cos(th)
            sink = 0.3 if face >= 1.0 else 0.45                                # sunk deeper: they stick out less as circles
            S.append((px, py, pz - sink * rr, rr)); kids(px, py, pz - sink * rr, rr, max(2, k - 3), lvl + 1)
    for cx, cy, cz, a_, b_, c_, nk in D:
        if nk: kids(cx + lean * 0.25 * a_, cy, cz, min(a_, c_), nk + 5 * (levels > 3), 0)
    S = S or [(D[0][0], D[0][1], D[0][2], 0.3 * min(d_[5] for d_ in D))]      # without puffs (lenses) — a reference sphere: the merging scale and the box
    rmin = min(s[3] for s in S); kb = 0.35 * rmin; pad = 0.2 * max(d[3] for d in D)
    x0, x1 = min(min(d[0] - d[3] for d in D), min(s[0] - s[3] for s in S)) - pad, max(max(d[0] + d[3] for d in D), max(s[0] + s[3] for s in S)) + pad
    y0, y1 = min(min(d[1] - d[4] for d in D), min(s[1] - s[3] for s in S)) - pad, max(max(d[1] + d[4] for d in D), max(s[1] + s[3] for s in S)) + pad
    z1 = max(max(d[2] + d[5] for d in D), max(s[2] + s[3] for s in S)) + pad; r0 = lump_r * H; zs = -1.4 * r0 if base_lumps > 0 else 0.0
    X = (np.arange(int((x1 - x0) / vs) + 1) + 0.5) * vs + x0; Y = (np.arange(int((y1 - y0) / vs) + 1) + 0.5) * vs + y0; Z = (np.arange(int((z1 - zs) / vs) + 1) + 0.5) * vs + zs
    acc = np.zeros((len(X), len(Y), len(Z)), np.float64)
    def add(cx, cy, cz, ax, by, cz_, ellip, tgt=None, kb=kb):
        tgt = acc if tgt is None else tgt; m = 4 * kb
        ix = slice(max(0, int((cx - ax - m - x0) / vs)), int((cx + ax + m - x0) / vs) + 2); iy = slice(max(0, int((cy - by - m - y0) / vs)), int((cy + by + m - y0) / vs) + 2)
        iz = slice(max(0, int((cz - cz_ - m - zs) / vs)), int((cz + cz_ + m - zs) / vs) + 2)
        dx, dy, dz = X[ix, None, None] - cx, Y[None, iy, None] - cy, Z[None, None, iz] - cz
        d = (np.sqrt((dx / ax) ** 2 + (dy / by) ** 2 + (dz / cz_) ** 2) - 1) * min(ax, by, cz_) if ellip else np.sqrt(dx * dx + dy * dy + dz * dz) - ax
        tgt[ix, iy, iz] += np.exp(np.minimum(-d / kb, 60.0))
    core = 0.8 if levels > 3 else 1.0                                          # large in the frame: the dome core is smaller — the surface is made of puffs
    for cx, cy, cz, a_, b_, c_, nk in D: add(cx, cy, cz, a_ * (core if nk else 1), b_ * (core if nk else 1), c_ * (core if nk else 1), True)
    for px, py, pz, r in S: add(px, py, pz, r, r, r, False)
    soft_m = min(soft * scales[0] * 0.5 * H, 0.9 * rmin)                      # the edge is soft but not softer than a small puff (otherwise the small levels drown)
    sdf = -kb * np.log(np.maximum(acc, 1e-30))
    if realism > 0:                                                            # realism: a wider edge cut by fleece (inside the shape it is 1, there it does not cut)
        sh_ = np.clip(-sdf / (soft_m * (1 + 2.5 * realism)), 0, 1)
        n_ = fbm3(X, Y, Z, 1.0 / max(3.0 * vs, 0.5 * rmin), int(rng.integers(1e6)), 3); e_ = 0.65 * realism * (1 - (1 - np.abs(2 * n_ - 1))); del n_
        rho = (dens * np.clip((sh_ - e_) / np.maximum(1 - e_, 1e-3), 0, 1)).astype(np.float32); del sh_, e_
    else: rho = (dens * np.clip(-sdf / soft_m, 0, 1)).astype(np.float32)
    if base_lumps > 0: base_soft = min(base_soft, 0.05 * H)                   # under the lumps the bottom is denser: the lumps cling to the base
    if base_soft > 0: rho *= np.clip(Z / base_soft, 0, 1).astype(np.float32)[None, None, :]   # the bottom is feathered: a flat bottom above the floor would be a sharp line
    elif zs < 0: rho *= (Z >= 0).astype(np.float32)[None, None, :]
    if anvil is not None: rho *= np.clip(1.0 - 0.55 * (Z - 0.72 * H) / (0.2 * H), 0.45, 1.0).astype(np.float32)[None, None, :]   # the anvil's ice is sparser than the tower's water
    if base_lumps > 0:                                                         # lumps under the base: pouches and rolls (along x), only under the cloud
        k_ = min(len(Z) - 1, int((0.06 * H - zs) / vs)); under = np.argwhere(rho[:, :, k_] > 0.3 * dens); acc_l = np.zeros(rho.shape, np.float64); kbl = 0.25 * r0
        n_ = int(base_lumps * len(under) * vs * vs / (math.pi * (1.1 * r0) ** 2))
        for i_, j_ in under[rng.choice(len(under), size=min(n_, len(under)), replace=False)] if n_ else []:
            r = r0 * rng.uniform(0.55, 1.4); roll = rng.random() < 0.3
            add(X[i_], Y[j_], rng.uniform(-0.45, 0.15) * r, (2.4 if roll else 1.0) * r, r, 0.8 * r, True, acc_l, kbl)
        sdf_l = -kbl * np.log(np.maximum(acc_l, 1e-30)); rho = np.maximum(rho, (0.9 * dens * np.clip(-sdf_l / (0.35 * r0), 0, 1)).astype(np.float32)); del acc_l
    return rho, np.array([x0 - L / 2, y0, zs], np.float32), vs


def storm_cells(L, H, rng, vs, r=None, lean=0.2, anvil=None, fill=0.35, soft=0.4, kids=3, dens=1.0, layers=2, rough=0.45, cover=1.4, flat=0.5):
    """A storm cloud made of small clouds (instead of one giant lump): the envelope of a cumulonimbus — a tower ridge (lean — the top drifts toward +x, with
    the wind), shoulders at the base, an anvil (width relative to L, thickness relative to H) — smooth and sparser (ice). Along the tower's surface there are
    small clouds of radius r[0]…r[1] (many small, a few large; larger toward the top), cover — how many times the small clouds cover the surface, layers — the
    number of layers inward; each small cloud has kids puffs on top (a small "cauliflower") — softly merged with it (a sum), and the bottom of a small cloud
    is flattened (flat). Between the small clouds the union is a maximum, not a sum: they do not merge into a lump, there are seams and gaps between them; the
    edge is rough (rough — noise). Inside there is a sparse core fill: flashes spread through the mass, the cloud is not empty. The base is flat. The anvil
    (anvil=(width, thickness)) is smooth — off by default: at the scene's mass height (~3 km) it reads as a hat. → (ρ, o, vs)"""
    r0, r1 = r if r is not None else (max(4.0 * vs, 0.025 * L), max(9.0 * vs, 0.09 * L))
    E = []                                                                     # envelope: (x, y, z, a, b, c, kind: 0 — tower, 1 — anvil)
    for i in range(6):
        t = i / 5; a = 0.5 * L * (1 - 0.45 * t) * rng.uniform(0.9, 1.1)
        E.append((lean * H * t * t, rng.normal(0, 0.03 * L), (0.1 + 0.62 * t) * H, a, 0.8 * a, 0.22 * H, 0))
    for sgn in (-1, 1):                                                        # the shoulders are different and low
        w_ = rng.uniform(0.3, 0.5) * L; E.append((sgn * 0.5 * L, rng.normal(0, 0.06 * L), 0.1 * H, w_, 0.75 * w_, 0.14 * H, 0))
    if anvil:                                                                  # the anvil goes downwind from the top, thinner and narrower toward the edge (not a "mushroom")
        aw, at = anvil; xt = lean * H
        for k in range(5):
            t = k / 4; s_ = 0.5 * aw * L * (0.5 - 0.22 * t)
            E.append((xt + 0.6 * aw * L * (t - 0.12), rng.normal(0, 0.06 * L), H * (1 - 0.45 * at) - 0.06 * H * t, s_, 0.75 * s_ * rng.uniform(0.8, 1.2),
                      0.5 * at * H * (1 - 0.45 * t), 1))
    E = np.array(E, float); T_ = E[E[:, 6] == 0]; A_ = E[E[:, 6] == 1]
    def inside(P, ell, skip=-1):                                               # points inside the ellipsoids (except its own)
        q = (((P[:, None, :] - ell[None, :, :3]) / ell[None, :, 3:6]) ** 2).sum(-1)
        if skip >= 0: q[:, skip] = 9.0
        return (q < 0.98).any(1)
    S = []                                                                     # small clouds: [(x, y, z, r) of a small cloud, then of its puffs]
    rm = 0.5 * (r0 + r1)
    for k, (cx, cy, cz, a, b, c, _) in enumerate(T_):
        ar = 4 * math.pi * (((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) / 3) ** (1 / 1.6)
        m = int(cover * ar / (math.pi * rm * rm)); u = rng.normal(size=(m, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        P = np.c_[cx + a * u[:, 0], cy + b * u[:, 1], cz + c * u[:, 2]]; nrm = u / np.array([a, b, c]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        keep = ~inside(P, T_, k) & (P[:, 2] > 0.03 * H); P, nrm = P[keep], nrm[keep]   # only the tower's outer surface, above the base
        for lay in range(layers):
            n_ = len(P) if lay == 0 else len(P) // 2; idx = rng.permutation(len(P))[:n_]
            rr = r0 * (r1 / r0) ** (rng.random(n_) ** 2.0) * (0.75 + 0.5 * np.clip(P[idx, 2] / H, 0, 1)) * (1.3 if lay else 1.0)
            C_ = P[idx] - nrm[idx] * (0.35 * rr[:, None] + lay * 0.9 * r1)
            for (x_, y_, z_), r_, n3 in zip(C_, rr, nrm[idx]):
                g_ = [(x_, y_, z_, r_)]
                for _ in range(kids if lay == 0 else 0):                       # puffs from the top outward
                    d_ = n3 + 0.8 * rng.normal(size=3) + np.array([0.0, 0.0, 0.6]); d_ /= np.linalg.norm(d_); rk = r_ * rng.uniform(0.35, 0.55)
                    g_.append((x_ + d_[0] * 0.65 * r_, y_ + d_[1] * 0.65 * r_, z_ + d_[2] * 0.65 * r_, rk))
                S.append(np.array(g_, float))
    Sa = np.concatenate(S); pad = 2 * vs
    lo = np.minimum((Sa[:, :3] - Sa[:, 3:4]).min(0), (E[:, :3] - E[:, 3:6]).min(0)) - pad; hi = np.maximum((Sa[:, :3] + Sa[:, 3:4]).max(0), (E[:, :3] + E[:, 3:6]).max(0)) + pad
    lo[2] = 0.0
    X, Y, Z = [(np.arange(int((hi[q] - lo[q]) / vs) + 1) + 0.5) * vs + lo[q] for q in range(3)]
    acc = np.zeros((len(X), len(Y), len(Z)), np.float32)
    for g_ in S:                                                               # a small cloud with puffs: core 1, the edge at soft·r, the bottom flattened; inside a small cloud —
        b0 = (g_[:, :3] - g_[:, 3:4]).min(0); b1 = (g_[:, :3] + g_[:, 3:4]).max(0)      # a sum (merged), between small clouds — a maximum (seams)
        ix = slice(max(0, int((b0[0] - lo[0]) / vs)), int((b1[0] - lo[0]) / vs) + 2); iy = slice(max(0, int((b0[1] - lo[1]) / vs)), int((b1[1] - lo[1]) / vs) + 2)
        iz = slice(max(0, int(b0[2] / vs)), int(b1[2] / vs) + 2)
        if iz.start >= len(Z) or iz.stop <= 0: continue
        f_ = 0.0
        for x_, y_, z_, r_ in g_:
            dz = Z[None, None, iz] - z_; dz = np.where(dz < 0, dz * (1 + flat), dz)
            f_ = f_ + np.clip((r_ - np.sqrt((X[ix, None, None] - x_) ** 2 + (Y[None, iy, None] - y_) ** 2 + dz * dz)) / (soft * r_), 0, 1)
        np.maximum(acc[ix, iy, iz], np.minimum(f_, 1.0).astype(np.float32), out=acc[ix, iy, iz])
    if rough > 0:                                                              # a rough edge: noise cuts only the edge of the small clouds (the core stays 1)
        n_ = fbm3(X, Y, Z, 1.0 / max(1.5 * r0, 3.0 * vs), int(rng.integers(1e6)), 2).astype(np.float32)
        acc = np.clip(acc * (1 + rough) - rough * n_, 0, 1); del n_
    core = np.zeros_like(acc); amb = np.zeros_like(acc)
    for cx, cy, cz, a, b, c, kind in E:                                        # the tower core and the anvil: depth inside the ellipsoid, m
        q_ = np.sqrt(((X[:, None, None] - cx) / a) ** 2 + ((Y[None, :, None] - cy) / b) ** 2 + ((Z[None, None, :] - cz) / c) ** 2)
        dep = ((1 - q_) * min(a, b, c)).astype(np.float32)
        if kind == 0: np.maximum(core, np.clip((dep - 0.5 * r1) / r1, 0, 1), out=core)
        else: np.maximum(amb, np.clip(dep / (0.35 * c), 0, 1), out=amb)
    rho = np.maximum(acc, fill * core)
    if len(A_):                                                                # anvil: smooth, ice sparser, the edge is rough too
        n_ = fbm3(X, Y, Z, 1.0 / max(2.5 * r0, 3.0 * vs), int(rng.integers(1e6)), 3).astype(np.float32)
        rho = np.maximum(rho, 0.5 * np.clip(amb * 1.6 - 0.9 * n_, 0, 1)); del n_
    rho *= np.clip(Z / (1.5 * vs), 0, 1).astype(np.float32)[None, None, :]     # flat base
    return (dens * rho).astype(np.float32), np.array(lo, np.float32), vs


def storm_ceiling(seed, ext=6000.0, thick=1400.0, vs=30.0, cover=0.7, cell=1800.0, lump=(420.0, 170.0, 70.0), dens=0.7, erode=0.55, billow=110.0,kids=2, protrude=0.4,
                  lumpy=0.0, roll=0.0, scud=0.0):
    """A storm sky seen from below (dark masses, light seeping through in thin places): the thickness varies by place — noise ~cell (the mass is thick and dark
    from below, thin in the gaps — light passes through from above), the bottom is lumps of three sizes (lump hemispheres, denser under the mass), the edges
    are puffy — eroded by 3D "fleece" noise (1 − |2n − 1|, ~billow m). A veil of ~1.6 km cells seen from below — smooth bumps within 20° of the frame.
    lumpy 0–1 — the rhythm: where the bottom is smooth and where it is richly bumpy (a mask of ~0.6·cell; an even carpet of identical "pouches" is boring);
    roll — the lumps stretched along the wind (+x), rolls; scud — ragged wisps under the base (40–220 m lower, sparser than the cloud). → (ρ, o, vs);
    z = 0 is the base, the lumps are below."""
    rng = np.random.default_rng(seed); X = np.arange(-ext / 2, ext / 2, vs, dtype=np.float32) + vs / 2; Y = X.copy()
    m = vnoise2(X + 1e4, Y + 2e4, 1 / cell, seed) * 0.65 + vnoise2(X, Y, 1 / (0.4 * cell), seed + 1) * 0.35
    T = np.clip((m - (1 - cover)) / 0.3, 0, 1).astype(np.float32)               # 0 — a gap, 1 — full thickness
    rr = np.hypot(X[:, None], Y[None, :]) / (ext / 2) + 0.24 * (vnoise2(X + 3e4, Y + 3e4, 1 / (0.12 * ext), seed + 9) - 0.5)   # the edge is a ragged circle, not the square of the grid
    T *= np.clip((0.95 - rr) / 0.22, 0, 1).astype(np.float32)
    dep = np.zeros(T.shape, np.float32)
    Lm = np.clip((vnoise2(X + 5e4, Y + 6e4, 1 / (0.6 * cell), seed + 11) - 0.25) / 0.5, 0, 1).astype(np.float32)   # where it is bumpy (1), where smooth (0)
    lm_at = lambda x_, y_: (1 - lumpy) + lumpy * float(Lm[min(len(X) - 1, max(0, int((x_ - X[0]) / vs))), min(len(Y) - 1, max(0, int((y_ - Y[0]) / vs)))])
    for r0 in lump:                                                            # lumps: large under the mass, small ones everywhere there is cloud
        cand = np.argwhere(T > (0.45 if r0 == lump[0] else 0.25)); k_ = int(0.8 * len(cand) * vs * vs / (math.pi * (1.1 * r0) ** 2))   # large ones — under the mass
        for i_, j_ in cand[rng.choice(len(cand), size=min(k_, len(cand)), replace=False)] if k_ else []:
            R = r0 * rng.uniform(0.55, 1.4) * (0.6 + 0.4 * T[i_, j_]); h0 = rng.uniform(0.6, 1.0) * T[i_, j_]
            sub = [(0.0, 0.0, R)] + ([(rng.normal(0, 0.55 * R), rng.normal(0, 0.55 * R), R * rng.uniform(0.45, 0.8)) for _ in range(int(rng.integers(3, 7)))] if r0 == lump[0] else [])
            def bump(cx_, cy_, r, base, h, lvl):                               # a lump: a hemisphere downward from base; its children are on its bottom (a hierarchy, like the puffs of a mass)
                mm = int((1.6 + 1.5 * roll) * r / vs) + 1; ex, ey = rng.uniform(0.7, 1.4) * (1 + roll * rng.uniform(0.5, 1.5)), rng.uniform(0.7, 1.4)   # rolls — along x
                ic, jc = int((cx_ - X[0]) / vs), int((cy_ - Y[0]) / vs); a0, a1, b0, b1 = max(0, ic - mm), min(len(X), ic + mm + 1), max(0, jc - mm), min(len(Y), jc + mm + 1)
                if a1 <= a0 or b1 <= b0: return
                dd = ((X[a0:a1, None] - cx_) / ex) ** 2 / (r * r) + ((Y[None, b0:b1] - cy_) / ey) ** 2 / (r * r)
                prof = 0.85 * r * np.clip(1 - dd, 0, 1) ** 0.6 * h
                dep[a0:a1, b0:b1] = np.maximum(dep[a0:a1, b0:b1], np.where(dd < 1, base + prof, 0.0))
                if lvl >= kids or r < 1.6 * vs: return
                lk = lm_at(cx_, cy_)                                           # rhythm: in smooth places there are few children
                for _ in range(int(round((rng.integers(6, 11) if lvl == 0 else rng.integers(4, 8)) * lk))):   # the "cloudlet look": the children protrude from the parent's bottom
                    a_ = rng.uniform(0, 2 * math.pi); q_ = math.sqrt(rng.uniform(0.0, 0.85)); kx, ky = cx_ + q_ * r * ex * math.cos(a_), cy_ + q_ * r * ey * math.sin(a_)
                    rk = r * rng.uniform(0.3, 0.45); bk = base + 0.85 * r * max(0.0, 1 - q_ * q_) ** 0.6 * h - (0.85 - protrude) * rk   # protrusion — protrude·rk
                    bump(kx, ky, rk, bk, rng.uniform(0.7, 1.0), lvl + 1)
            for dx_, dy_, r in sub:                                            # a large one — in a cluster: an uneven edge, like a cauliflower's
                bump(X[i_] + dx_, Y[j_] + dy_, r, 0.0, h0, 0 if r0 == lump[0] else kids)
    Z = np.arange(-float(dep.max()) - vs - (260.0 if scud > 0 else 0.0), thick + vs, vs, dtype=np.float32) + vs / 2
    top = thick * (0.08 + 0.92 * T ** 1.3)                                      # the mass is thick, the gap thin (τ ~2–5 — light from above passes through)
    sb = max(1.3 * vs, 0.35 * billow)                                           # the bottom is soft over ~1.3 cells: softer, and the lumps' children drown (the erosion cuts them anyway)
    shape = np.clip((Z[None, None, :] + dep[..., None]) / sb, 0, 1) * np.clip((top[..., None] - Z[None, None, :]) / (0.25 * thick), 0, 1) * (T[..., None] > 0.03)
    n = fbm3(X + 700, Y + 300, Z + 50, 1 / billow, seed + 5, 3); b = 1 - np.abs(2 * n - 1)   # fleece: a puffy edge
    e = erode * (1 - b); rho = dens * np.clip((shape - e) / np.maximum(1 - e, 1e-3), 0, 1) * (0.25 + 0.75 * T[..., None])
    if scud > 0:                                                               # wisps under the base: elongated ragged pieces, sparser than the cloud
        cand = np.argwhere(T > 0.4); k_ = int(scud * len(cand) * vs * vs / (math.pi * 300.0 ** 2))
        for i_, j_ in cand[rng.choice(len(cand), size=min(k_, len(cand)), replace=False)] if k_ else []:
            rx = rng.uniform(60, 200); ry = rx * rng.uniform(0.4, 0.8); rz = rx * rng.uniform(0.25, 0.4); cz = -float(dep[i_, j_]) - rng.uniform(40, 220)
            m_ = int(rx / vs) + 2; a0, a1, b0, b1 = max(0, i_ - m_), min(len(X), i_ + m_ + 1), max(0, j_ - m_), min(len(Y), j_ + m_ + 1)
            k0 = max(0, int((cz - rz - Z[0]) / vs)); k1 = min(len(Z), int((cz + rz - Z[0]) / vs) + 2)
            if a1 <= a0 or b1 <= b0 or k1 <= k0: continue
            d_ = np.sqrt(((X[a0:a1, None, None] - X[i_]) / rx) ** 2 + ((Y[None, b0:b1, None] - Y[j_]) / ry) ** 2 + ((Z[None, None, k0:k1] - cz) / rz) ** 2)
            v_ = 0.45 * dens * np.clip(1 - d_, 0, 1) ** 0.5 * (b[a0:a1, b0:b1, k0:k1] > 0.45)   # ragged: the fleece noise cuts the piece
            rho[a0:a1, b0:b1, k0:k1] = np.maximum(rho[a0:a1, b0:b1, k0:k1], v_)
    return rho.astype(np.float32), np.array([X[0] - vs / 2, Y[0] - vs / 2, Z[0] - vs / 2], np.float32), vs


def rain_shaft(W, H, rng, vs, slant=0.15, dens=1.0):
    """A rain curtain: a column under the base of the cloud — the density fades toward the edge (a Gaussian of 0.5·W), the edge is ragged and the curtains denser
    (noise ~0.35·W), streaks along the path of the drops (noise ~45 m across, slanted by the wind), breaks along the height (a wave of ~0.3·H shifted by place);
    drift by the wind (slant — the shift in x per metre of fall), it fades near the ground and near the cloud. The z axis points down from the base: k = 0 is at the cloud.
    A uniform grey column (a Gaussian × vertical streaks) read as a wall from below."""
    n = max(4, int(W / vs)); nz = max(4, int(H / vs)); X = (np.arange(n) + 0.5) * vs - W / 2
    shift = slant * (np.arange(nz) + 0.5) * vs; sh = np.round(shift / vs).astype(int); ext = int(sh.max()) + 1
    xs = (np.arange(n + ext) + 0.5) * vs - W / 2 - ext * vs; ys = X                # the field is wider by the drift: layer k takes its own shift
    s1, s2, s3 = (int(rng.integers(1e6)) for _ in range(3))
    lo = np.clip((vnoise2(xs, ys, 1.0 / (0.35 * W), s1) - 0.2) / 0.55, 0, 1)        # curtains and a ragged edge
    st = np.clip((vnoise2(xs, ys, 1.0 / 45.0, s2) - 0.3) / 0.45, 0, 1)              # drop streaks
    ph = vnoise2(xs, ys, 1.0 / (0.2 * W), s3) * 2 * math.pi                          # the shift of the break wave by place
    G = lo * (0.3 + 0.7 * st)
    XX = X[:, None, None] - shift[None, None, :]; YY = X[None, :, None]; core = np.exp(-(XX ** 2 + YY ** 2) / (0.5 * W) ** 2)
    zf = (np.arange(nz) + 0.5) / nz; fade = np.clip((1 - zf) / 0.25, 0, 1) * np.clip(zf / 0.05, 0, 1)   # fades at the ground, smoothly at the cloud
    rho = np.empty((n, n, nz), np.float32)
    for k in range(nz):                                                            # a layer is the field with the drift shift; the breaks are a wave along the height
        j = ext - sh[k]; wave = 0.65 + 0.35 * np.sin(2 * math.pi * zf[k] * H / (0.3 * H) + ph[j:j + n])
        rho[:, :, k] = dens * core[:, :, k] * G[j:j + n] * wave * fade[k]
    return rho[:, :, ::-1].copy(), np.array([-W / 2, -W / 2, -H], np.float32), vs                # z up: the column from the ground to the base


def mammatus(L, rng, vs, thick=260.0, r0=180.0, dens=0.35):
    """Mammatus: a slab of veil with hundreds of hanging pouches below it (ellipsoids elongated downward, spacing ~2.1·r0), merged by a sum of exp.
    The slab's bottom is at a height of 2.2·r0 from the grid bottom (place it with its base at the needed height). → (ρ, o, vs)."""
    n = max(8, int(L / vs)); zb = 2.2 * r0; nz = max(6, int((zb + thick + r0) / vs)); kb = 0.35 * r0
    X = (np.arange(n) + 0.5) * vs - L / 2; Z = (np.arange(nz) + 0.5) * vs
    d_slab = np.maximum(zb - Z, Z - (zb + thick)); acc = np.repeat(np.repeat(np.exp(np.minimum(-d_slab / kb, 60.0))[None, None, :], n, 0), n, 1)
    pts = []
    for _ in range(int((L / (2.1 * r0)) ** 2 * 3)):                               # dart throwing: spacing ~2.1·r0
        p = rng.uniform(-L / 2 + r0, L / 2 - r0, 2)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > (2.1 * r0) ** 2 for q in pts[-400:]): pts.append(p)
    for px, py in pts:
        r = r0 * rng.uniform(0.7, 1.2); cz = zb - 0.45 * r; m = int(1.6 * r / vs) + 2
        i0, j0, k0 = int((px + L / 2) / vs), int((py + L / 2) / vs), int(cz / vs)
        ix, iy, iz = slice(max(0, i0 - m), i0 + m), slice(max(0, j0 - m), j0 + m), slice(max(0, k0 - m), k0 + m)
        dx, dy, dz = X[ix, None, None] - px, X[None, iy, None] - py, Z[None, None, iz] - cz
        d = np.sqrt((dx / r) ** 2 + (dy / r) ** 2 + (dz / (1.2 * r)) ** 2) * r - r
        acc[ix, iy, iz] += np.exp(np.minimum(-d / kb, 60.0))
    sdf = -kb * np.log(np.maximum(acc, 1e-30)); rho = (dens * np.clip(-sdf / (0.25 * r0), 0, 1)).astype(np.float32)
    return rho, np.array([-L / 2, -L / 2, 0.0], np.float32), vs
