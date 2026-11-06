"""Relief summary for clouds: the levels of valleys and ridges, the amplitude, peaks (prominence, dominance), the mountain spacing λ — from the scene's far_heights.npz
(a height grid over tens of km). Without Blender. The cloud levels by regime are fractions of the amplitude (levels)."""
import json
import math
import os

import numpy as np


def _maxf(z, r):
    """Maximum in a (2r + 1)² window — by axis shifts (without scipy: Blender's Python)."""
    out = z.copy()
    for ax in (0, 1):
        acc = out.copy()
        for k in range(1, r + 1): acc = np.maximum(acc, np.maximum(np.roll(out, k, ax), np.roll(out, -k, ax)))
        out = acc
    return out


_H = {}


def heights(scene):
    """The scene's height map: far_heights.npz (tens of km, 40 m) + the near relief and the belt (core.surface.ground) where they exist —
    the massif near the castle lives in the belt and is absent from the far map. → (z, x0, y0, step)."""
    if scene in _H: return _H[scene]
    f = np.load(os.path.join(scene, "far_heights.npz")); z = f["z"].astype(np.float64).copy(); x0, y0, st = (float(v) for v in f["o"])
    try:
        from core.surface.ground import Ground
        G = Ground(scene); b = G.far_bounds() if hasattr(G, "far_bounds") else None
        if b is not None:
            bx0, by0, bx1, by1 = (float(v) for v in b)
            i0, i1 = max(0, int((bx0 - x0) / st)), min(z.shape[0], int((bx1 - x0) / st) + 1); j0, j1 = max(0, int((by0 - y0) / st)), min(z.shape[1], int((by1 - y0) / st) + 1)
            Xs = x0 + np.arange(i0, i1) * st; Ys = y0 + np.arange(j0, j1) * st; XX, YY = np.meshgrid(Xs, Ys, indexing="ij")
            z[i0:i1, j0:j1] = np.asarray(G.height(XX.ravel(), YY.ravel()), float).reshape(XX.shape)
    except Exception as e_: print("the belt relief was not merged:", e_)
    _H[scene] = (z, x0, y0, st); return _H[scene]


def summary(scene, center=(0.0, 0.0), radius=20000.0, peak_r=600.0):
    """→ {z_v, z_r, A, lam, peaks: [{x, y, z, prom, dom}], grid}: the levels are percentiles of heights within the view radius, the peaks are local maxima of a peak_r window
    with a prominence ≥ max(150, 0.15·A) (prominence ≈ height − the 20th percentile within 2 km)."""
    z, x0, y0, st = heights(scene)
    n = z.shape[0]; X = x0 + np.arange(n) * st; Y = y0 + np.arange(z.shape[1]) * st
    R = np.hypot(X[:, None] - center[0], Y[None, :] - center[1]); inside = R < radius
    z_v, z_r = float(np.percentile(z[inside], 10)), float(np.percentile(z[inside], 92)); A = max(z_r - z_v, 1.0)
    r = max(1, int(peak_r / st)); loc = (z >= _maxf(z, r)) & inside & (z > z_v + 0.3 * A)
    low = np.percentile                                                          # prominence — relative to the 2 km neighbourhood
    peaks = []
    for i, j in zip(*np.nonzero(loc)):
        k = int(2000 / st); w = z[max(0, i - k):i + k + 1, max(0, j - k):j + k + 1]
        prom = float(z[i, j] - low(w, 20))
        if prom >= max(150.0, 0.15 * A): peaks.append({"x": float(X[i]), "y": float(Y[j]), "z": float(z[i, j]), "prom": round(prom)})
    peaks.sort(key=lambda p: -p["z"])
    for p in peaks:                                                              # dominance — the distance to the nearest higher peak
        higher = [math.hypot(p["x"] - q["x"], p["y"] - q["y"]) for q in peaks if q["z"] > p["z"]]
        p["dom"] = round(min(higher)) if higher else round(radius)
    nn = [min(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) for q in peaks if q is not p) for p in peaks] if len(peaks) > 1 else [radius]
    return {"z_v": round(z_v), "z_r": round(z_r), "A": round(A), "lam": round(float(np.median(nn))), "peaks": peaks,
            "grid": {"x0": x0, "y0": y0, "step": st, "n": n}}


LEVELS = {   # regime → (base, top) as fractions of the amplitude A from z_v; None — independent of the relief
    "fair": (1.0 + 0.3, 1.0 + 1.2), "peaks": (0.7, 1.15), "belt": (0.42, 0.62), "fog": (0.0, 0.3), "deck": (0.9, 1.05), "storm": (0.9, 3.9), "glow": (0.9, 3.9), "mist": (0.0, 0.3), "mammatus": (1.0, 1.2)}


def levels(S, regime):
    lo, hi = LEVELS[regime]; return S["z_v"] + lo * S["A"], S["z_v"] + hi * S["A"]


if __name__ == "__main__":
    import sys
    S = summary(sys.argv[1]); print(json.dumps({k: v for k, v in S.items() if k not in ("peaks", "grid")}, ensure_ascii=False), "n_vertices", len(S["peaks"]))
    for p in S["peaks"][:8]: print("  ", p)
