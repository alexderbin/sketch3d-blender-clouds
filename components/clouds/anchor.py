"""Clouds from the ground: a weather regime + a relief summary → clouds in the world tied to landforms. Without Blender.
fair — cumulus above the peaks (the base above the ridges, downwind, no more often than the mountain spacing, the size from the spacing and the prominence),
the main one — above the main peak behind the castle; peaks — clouds at the peaks (the base below the summits, hugging the upper slopes); belt / fog — a layer
of cells cut by the relief (where the ground is above the layer there is no cloud): a belt along the slopes and fog in the valleys. Above the castle — empty."""
import math

import numpy as np

from components.clouds import terrain as T


def height_at(scene, X, Y):
    """Ground height from far_heights.npz (bilinear; outside the grid — the edge level)."""
    z, x0, y0, st = T.heights(scene); n = z.shape[0]                               # with the near relief and the belt
    fx = np.clip((np.asarray(X) - x0) / st, 0, n - 1.001); fy = np.clip((np.asarray(Y) - y0) / st, 0, z.shape[1] - 1.001)
    i, j = fx.astype(int), fy.astype(int); u, v = fx - i, fy - j
    return z[i, j] * (1 - u) * (1 - v) + z[i + 1, j] * u * (1 - v) + z[i, j + 1] * (1 - u) * v + z[i + 1, j + 1] * u * v


def plan(scene, regime, castle, view_az, rng, cover=0.5, wind_az=None, R=(500.0, 22000.0), quiet=600.0, cam=None, deck=1.0, deck_form=None, ceil_l=14000.0, shelf=True, near=0.0, hero_k=1.0):
    """→ (summary, [cloud]): a cloud is {form: mass|layer, x, y, zb, L, H, type, kw, seed, anchor}; view_az — the heading of the main camera (toward the castle):
    the main peak is in a ±35° sector behind the castle. cover — the fraction of candidate peaks under clouds (fair)."""
    S = T.summary(scene, center=castle[:2]); lam = S["lam"]; va = math.radians(view_az)
    def behind(p):                                                                  # behind the castle as seen from the main camera: the angle between the heading and the direction to the peak
        a = math.atan2(p["y"] - castle[1], p["x"] - castle[0]); return abs((a - va + math.pi) % (2 * math.pi) - math.pi)
    vis = [p["z"] for p in S["peaks"] if behind(p) < math.radians(40) and math.hypot(p["x"] - castle[0], p["y"] - castle[1]) < 8000]
    if vis: S["z_r"] = round(max(S["z_r"], 0.95 * max(vis))); S["A"] = S["z_r"] - S["z_v"]   # ridges — no lower than the peaks of the main view
    A = S["A"]; zb, zt = T.levels(S, regime)
    w = math.radians(view_az + 90.0 if wind_az is None else wind_az); wind = np.array([math.cos(w), math.sin(w)])   # the wind is across the view
    out = []
    storm = regime in ("storm", "glow"); wet = regime == "storm"                   # glow — heavy masses and a veil, without rain and a roll cloud
    if storm:                                                                        # storm: a veil above, masses low and dense, a roll cloud in front
        zb_s = S["z_r"] - 0.1 * A
        vx, vy = (math.cos(math.radians(cam[3])), math.sin(math.radians(cam[3]))) if cam is not None else (0.0, 1.0)
        if deck <= 0: pass                                                         # without a veil
        elif deck_form == "ceiling":                                               # storm ceiling: a thick layer fitted to the place, lumps in clusters, small puffs (forms.storm_ceiling)
            out.append({"form": "layer", "type": "stratocumulus", "x": castle[0] + vx * 2500.0, "y": castle[1] + vy * 2500.0, "zb": S["z_r"] + 1.6 * A, "L": ceil_l, "H": 1400.0,
                        "kw": {"form": "ceiling", "thick": 1400.0, "cell": 1800.0, "cover": deck}, "seed": int(rng.integers(1e6)), "anchor": "veil"})
        else:
            out.append({"form": "layer", "type": "stratocumulus", "x": castle[0] + vx * 4000.0, "y": castle[1] + vy * 4000.0, "zb": S["z_r"] + 1.6 * A, "L": 22000.0, "H": 600.0,
                        "kw": {"thick": 700.0, "cell": 1.2 * lam, "cover": deck, "lumps": 0.7, "lump_r": 300.0}, "seed": int(rng.integers(1e6)), "anchor": "veil"})   # front: it brightens behind the storm; deck < 1 — gaps; the bottom in lumps
        if cam is not None and wet and shelf:                                      # roll cloud: a long low bank across the view, spanning the full width of the frame
            ds = math.hypot(castle[0] - cam[0], castle[1] - cam[1]) + 3500.0; va_ = math.radians(cam[3])
            Ls = 2 * ds * math.tan(math.radians(cam[4]) / 2) * 1.3
            out.append({"form": "mass", "type": "bank_calm", "x": cam[0] + ds * math.cos(va_), "y": cam[1] + ds * math.sin(va_), "zb": S["z_v"] + 0.5 * A,
                        "L": Ls, "H": 0.45 * A, "rot": cam[3] + 90.0, "kw": {"aspect": 2.2, "accents": 0, "kids_frac": 0.3, "n_domes": 9, "face": 1.0, "levels": 3},
                        "seed": int(rng.integers(1e6)), "anchor": "rampart", "peak": 0.5})
    if regime == "mammatus" and cam is not None:                                    # mammatus: a slab with pouches above the valley in front of the camera
        va_ = math.radians(cam[3]); dm = 5000.0
        out.append({"form": "mammatus", "x": cam[0] + dm * math.cos(va_), "y": cam[1] + dm * math.sin(va_), "zb": S["z_r"] + 0.5 * A, "L": 11000.0,
                    "H": 600.0, "seed": int(rng.integers(1e6)), "anchor": "mammatus"})
        return S, out
    if regime == "mist":                                                            # mountain mist: fog in the valleys + a thin belt higher on the slopes
        for rg_, cov_, th_ in (("fog", 0.97, None), ("belt", 0.8, 180.0)):
            zb_, zt_ = T.levels(S, rg_)
            out.append({"form": "layer", "type": "stratocumulus", "x": castle[0], "y": castle[1], "zb": zb_, "L": 2 * R[1], "H": zt_ - zb_,
                        "kw": {"thick": th_ or max(150.0, zt_ - zb_), "cell": 0.9 * lam, "cover": cov_}, "seed": int(rng.integers(1e6)), "anchor": rg_})
        return S, out
    if regime in ("belt", "fog"):                                                   # the layer is cut by the relief in the test stand (height_at)
        out.append({"form": "layer", "type": "stratocumulus", "x": castle[0], "y": castle[1], "zb": zb, "L": 2 * R[1], "H": zt - zb,
                    "kw": {"thick": max(250.0, zt - zb), "cell": 0.9 * lam, "cover": 0.95 if regime == "fog" else 0.88}, "seed": int(rng.integers(1e6)), "anchor": regime})
        if regime == "fog": return S, out
        zb, zt = T.levels(S, "fair")                                               # above the belt — cumulus of their own level (above the ridges)
    if storm: zb, zt = zb_s, zb_s + 3.0 * A                                         # storm masses — from the ridges, tall
    pk = [p for p in S["peaks"] if R[0] < math.hypot(p["x"] - castle[0], p["y"] - castle[1]) < R[1]]
    def in_frame(p):                                                                # in the main camera's frame: (angle from the axis, distance)
        a = math.atan2(p["y"] - cam[1], p["x"] - cam[0]); return abs((a - math.radians(cam[3]) + math.pi) % (2 * math.pi) - math.pi), math.hypot(p["x"] - cam[0], p["y"] - cam[1])
    if cam is not None and near > 0:                                               # masses no closer than near to the camera: close to storms only the bottom is visible
        pk = [p for p in pk if in_frame(p)[1] >= near]
    if cam is not None:                                                             # the main one is the largest in the camera's frame: the angular height above the eye
        hero = max((p for p in pk if in_frame(p)[0] < 0.45 * math.radians(cam[4]) and 1500.0 < in_frame(p)[1] < (near + 12000.0 if near > 0 else 15000.0) and p["z"] >= S["z_v"] + 0.5 * A),
                   key=lambda p: (p["z"] - cam[2]) / in_frame(p)[1], default=None)
    else:
        hero = max((p for p in pk if behind(p) < math.radians(35) and p["z"] >= S["z_v"] + 0.6 * A and math.hypot(p["x"] - castle[0], p["y"] - castle[1]) > 1500.0),
                   key=lambda p: (p["z"] - S["z_v"]) / math.hypot(p["x"] - castle[0], p["y"] - castle[1]), default=None)   # a weighty peak behind the castle
    counter = None
    if cam is not None and hero is not None:                                       # a counterweight — the other side of the frame, smaller than the main one
        sgn = lambda p: math.copysign(1.0, ((math.atan2(p["y"] - cam[1], p["x"] - cam[0]) - math.radians(cam[3]) + math.pi) % (2 * math.pi) - math.pi))
        counter = max((p for p in pk if p is not hero and in_frame(p)[0] < 0.48 * math.radians(cam[4]) and sgn(p) != sgn(hero)
                       and in_frame(p)[0] > 0.12 * math.radians(cam[4]) and 1500.0 < in_frame(p)[1] < 20000.0),
                      key=lambda p: (p["z"] - cam[2]) / in_frame(p)[1], default=None)
    cand = sorted(pk, key=lambda p: -(p["prom"] * (1 + p["dom"] / 5000.0)))
    if cam is not None:                                                            # those in the main camera's frame go first (a composition built from the ground)
        cand = sorted(cand, key=lambda p: 0 if in_frame(p)[0] < 0.55 * math.radians(cam[4]) else 1)
    if counter: cand = [counter] + [p for p in cand if p is not counter]
    if hero: cand = [hero] + [p for p in cand if p is not hero]
    taken = []
    for p in cand[:max(1, int(cover * len(cand)))]:
        if any(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) < 1.3 * lam for q in taken): continue   # no more often than the mountain spacing
        big = p is hero; s = (1.0 if big else 0.75 if p is counter else rng.uniform(0.45, 0.9)) * min(1.0, p["prom"] / max(A, 1.0) + 0.4)
        if cam is not None and not big and in_frame(p)[0] < 0.55 * math.radians(cam[4]): s *= 1.25           # in the frame — larger
        if regime == "peaks":
            if p["z"] < zb + 0.15 * A: continue                                     # at peaks — only where the top is higher than the base
            W = max(1.5 * lam * s, 2.2 * p["prom"]); H = (p["z"] - zb) + 0.25 * A * s; d = 0.15 * lam
            kw = {"aspect": 1.4, "accents": 2, "kids_frac": 0.4, "n_domes": 3, "face": 0.25}
        else:
            W = max(1.4 * lam * s * (1.4 if big else 1.0), 1.2 * p["prom"]); H = (zt - zb) * (0.55 + 0.45 * s) * (1.3 if big else 1.0)
            d = rng.uniform(0.3, 0.8) * 1000.0; kw = {"aspect": 0.8, "accents": 3, "kids_frac": 0.45, "n_domes": 4, "face": 0.25}
        if big and cam is not None:                                                 # above the main peak of the frame: behind it (from the camera), the size is derived from it
            W = max(W, 2.4 * p["prom"], 1.2 * lam); H = max(H, 0.8 * (p["z"] - S["z_v"])); bx, by = p["x"] - cam[0], p["y"] - cam[1]; bn = math.hypot(bx, by)
            x, y = p["x"] + bx / bn * 0.3 * W, p["y"] + by / bn * 0.3 * W
            if storm: W, H, kw = W * 1.6 * hero_k, max(H, 2.2 * A) * hero_k, {"aspect": 0.45, "n_domes": 6, "levels": 4, "kids_frac": 1.0, "face": 1.0, "base_lumps": 0.7}   # the bottom in lumps; 4 levels — the camera is close under the storm
            out.append({"form": "mass", "type": "mediocris", "x": x, "y": y, "zb": zb if storm else max(zb, p["z"] - 0.25 * A), "L": W, "H": H, "kw": kw,
                        "seed": int(rng.integers(1e6)), "anchor": f"peak {p['z']:.0f} m (main)", "peak": 0.5, "flash": storm}); taken.append(p)
            if wet: out.append({"form": "rain", "x": x, "y": y, "zb": zb + 150.0, "L": 0.8 * W, "H": zb - S["z_v"] + 200.0, "seed": int(rng.integers(1e6)), "anchor": "rain"})
            continue
        x, y = p["x"] + wind[0] * d + rng.normal(0, 0.1 * lam), p["y"] + wind[1] * d + rng.normal(0, 0.1 * lam)
        if math.hypot(x - castle[0], y - castle[1]) < quiet + W / 2: continue      # above the castle — empty
        if storm: W, H, kw = W * 1.5, H * 2.0, {"aspect": 0.45, "n_domes": 6, "levels": 3, "kids_frac": 1.0, "face": 1.0, "base_lumps": 0.7}
        if wet and p is counter: out.append({"form": "rain", "x": x, "y": y, "zb": zb + 150.0, "L": 0.7 * W, "H": zb - S["z_v"] + 200.0, "seed": int(rng.integers(1e6)), "anchor": "rain"})
        out.append({"form": "mass", "type": "mediocris", "x": x, "y": y, "zb": zb + rng.normal(0, 0.03 * A), "L": W, "H": H, "kw": kw,
                    "seed": int(rng.integers(1e6)), "anchor": f"peak {p['z']:.0f} m" + (" (main)" if big else ""), "peak": 0.5,
                    "flash": storm and p is counter})                                # the counterweight in a storm is sheet lightning: flashes inside, no channel visible
        taken.append(p)
    return S, out
