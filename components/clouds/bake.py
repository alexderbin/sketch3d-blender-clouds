"""Baking clouds with a process pool: the shape of a plan cloud and the precomputed light (diffusion, lightning, colours, haze up to the camera) are
pure numpy without Blender, and the clouds are independent → one process per cloud. The test stand writes jobs (shared.pkl + job_NNN.pkl), this module
computes them with a pool and writes result_NNN.npz; Blender only writes the VDB and renders the frame. The logic is the same as it was in the stand
(castle.py finish), with the same random seeds.
.venv/bin/python -m components.clouds.bake <jobs folder> [n_processes=3]"""
import math
import os
import pickle
import sys
import time

import numpy as np

from components.clouds import forms, light as LT, lightning as LN


def mass_form(cl, sh, real=False):
    """A plan mass → (ρ, o, vs): realism is the component's knob (api.realism), the hybrid uses towers from a sample volume, the draft uses a ×2 grid."""
    from components.clouds import api as CA
    LS, cam_p, DV, DRAFT = sh["LS"], np.asarray(sh["cam_p"]), sh["DV"], sh["DRAFT"]
    d_m = math.hypot(cl["x"] - cam_p[0], cl["y"] - cam_p[1])
    if LS.get("cells") and cl.get("flash"):                                    # a storm made of small clouds — as in the test stand
        return forms.storm_cells(cl["L"], cl["H"], np.random.default_rng(cl["seed"]), float(np.clip(d_m / 300.0, 12.0, 80.0)) * DV * float(LS.get("vs_cells", 1.0)), dens=float(LS.get("dens", 0.35)),
                                 **{k[6:]: LS[k] for k in LS if k.startswith("cells_")})
    if real:
        WD = (np.load(sh["refvol"]), sh["refvol_u"])
        return forms.refvol_tower(WD, 0.85 * cl["L"], float(np.clip(d_m / 300.0, 12.0, 80.0)) * DV, np.random.default_rng(cl["seed"]), levels=3,
                                dens=float(LS.get("real_dens", 1.0)), sigma=sh["SIG"])
    rl = CA.realism(float(LS.get("realism", 0.5 if (LS.get("hybrid") and d_m < 8000.0) else 0.0)), d_m)
    kw = dict(cl["kw"]); kw["levels"] = min(3 if DRAFT else 5, int(kw.get("levels", 3)) + rl["levels_add"]); kw["realism"] = 0.0 if DRAFT else rl["erode"]
    return forms.mass_lobes(cl["L"], cl["H"], np.random.default_rng(cl["seed"]), float(np.clip(d_m / 300.0, 12.0, 80.0)) * rl["vs_k"] * DV, cl.get("peak", 0.5),
                            dens=float(LS.get("dens", 0.35)), base_soft=0.15 * cl["H"], **kw)


def form_cloud(cl, sh, real=False):
    """Shape of a plan cloud of any kind → (ρ, o, vs, loc)."""
    from components.clouds import anchor
    LS, DV, folder = sh["LS"], sh["DV"], sh["folder"]; loc = (cl["x"], cl["y"], cl["zb"])
    if cl["form"] == "layer":                                                  # a layer cut by the relief: where the ground is higher there is no cloud
        if cl["kw"].get("form") == "ceiling":
            rho, o, vs = forms.storm_ceiling(cl["seed"], ext=cl["L"], thick=cl["kw"]["thick"], vs=float(LS.get("ceil_vs", 32.0)) * DV, cover=cl["kw"]["cover"], cell=cl["kw"]["cell"],
                                             protrude=float(LS.get("ceil_protrude", 0.4)))
        else:
            rho, o, vs = forms.strato_layer(cl["seed"], ext=cl["L"], thick=cl["kw"]["thick"], cell=cl["kw"]["cell"], cover=cl["kw"]["cover"], vs=40.0 * DV, erode=0.4,
                                            size_var=0.5, lumps=cl["kw"].get("lumps", 0.0), lump_r=cl["kw"].get("lump_r", 300.0))
        X = loc[0] + o[0] + (np.arange(rho.shape[0]) + 0.5) * vs; Y = loc[1] + o[1] + (np.arange(rho.shape[1]) + 0.5) * vs
        gz = anchor.height_at(folder, np.repeat(X[:, None], len(Y), 1), np.repeat(Y[None, :], len(X), 0)); Z = loc[2] + o[2] + (np.arange(rho.shape[2]) + 0.5) * vs
        rho = (rho * np.clip((Z[None, None, :] - gz[:, :, None] - 10.0) / 60.0, 0, 1)).astype(np.float32)
        if cl["anchor"] == "fog":                                              # fog: denser near the ground, a soft top
            rho *= np.clip(1.15 - (Z - loc[2]) / max(cl["kw"]["thick"], 1.0), 0.15, 1.0).astype(np.float32)[None, None, :] ** 0.7
    elif cl["form"] == "mammatus":                                             # udder: a slab with pouches; the slab's bottom sits at the base
        rho, o, vs = forms.mammatus(cl["L"], np.random.default_rng(cl["seed"]), 28.0 * DV); loc = (cl["x"], cl["y"], cl["zb"] - 2.2 * 180.0)
    elif cl["form"] == "rain":                                                 # a rain curtain under the mass
        rho, o, vs = forms.rain_shaft(cl["L"], cl["H"], np.random.default_rng(cl["seed"]), 30.0 * DV, dens=float(LS.get("rain", 0.12)))
    else: rho, o, vs = mass_form(cl, sh, real)
    return rho, o, vs, loc


def bake_cloud(job, sh, rho, o, vs, loc):
    """Precomputed light, colours, haze up to the main camera (like the stand's finish) → {E, FL, PT, T, tau, entry}."""
    from core.light import far
    LS, RL, P, SIG = sh["LS"], sh["RL"], sh["P"], sh["SIG"]; cam_p = np.asarray(sh["cam_p"], float); S = np.asarray(sh["S"], float)
    i, rot, layer, detail = job["i"], float(job.get("rot", 0.0)), job["layer"], job.get("detail")
    plan = sh[job.get("plan", "P")]
    cl_ = np.array(rho.shape) * float(vs) / 2; ctr = np.array(loc) + o + cl_; v = ctr - cam_p
    Rz = np.array([[math.cos(math.radians(rot)), -math.sin(math.radians(rot)), 0], [math.sin(math.radians(rot)), math.cos(math.radians(rot)), 0], [0, 0, 1]])
    S_l, v_l = Rz.T @ S, Rz.T @ v; c = float(v_l / np.linalg.norm(v_l) @ S_l)
    fl = []; info = {}
    if int(LS.get("flashes", 0)) and job.get("flash_ok"):                       # flashes inside the mass, in its lower half
        rg = np.random.default_rng(7 + i); zc_ = np.arange(rho.shape[2])[None, None, :]
        occ = np.argwhere((rho > 0.3) & (zc_ > 0.15 * rho.shape[2]) & (zc_ < 0.45 * rho.shape[2]))
        fl = [(tuple(occ[j]), 3.0, float(LS.get("flash", 14.0)) if LS.get("bolt") else 1.0) for j in rg.choice(len(occ), min(int(LS["flashes"]), len(occ)), replace=False)] if len(occ) else []
    pts = [((Rz.T @ (np.array(fp_) - np.array(loc) - o)) / vs, I_) for fp_, I_ in sh.get("FIRES", [])]
    ch, chd = [], None; tau = {}; t_ = {"n_cells": int(rho.size)}; t0 = time.time()   # seconds per stage go into the cloud's report (to see where the pool's tail is)
    to_idx = lambda X: ((np.asarray(X, float) - ctr) @ Rz + cl_) / vs - 0.5     # world → cell indices (the cloud's axes)
    if job.get("ic") is not None: ch += list(zip(to_idx(job["ic"][0]), job["ic"][1]))
    B = sh.get("BOLT")
    if B is not None:                                                          # lightning: the channel's light for the near ones, the channel's τ up to the cameras for all
        gap_ = float(np.min(np.linalg.norm(B["P"] - ctr, axis=1))) - float(np.linalg.norm(cl_))
        if gap_ < 5000.0:
            k3 = 1 if gap_ < 2500.0 else 3; ch += list(zip(to_idx(B["P"][::k3]), B["W"][::k3] * k3))
            cen = (B["P"] * B["W"][:, None]).sum(0) / B["W"].sum(); chd = Rz.T @ (cen - ctr); chd = chd / np.linalg.norm(chd)
        for c_ in sh["cams"]:
            cp_ = np.array(c_["matrix_world"])[:3, 3]
            tau[c_["name"]] = [LN.tau_grid(rho, vs, to_idx, b_["pts"], np.repeat(cp_[None], len(b_["pts"]), 0), SIG) for b_ in B["B"]]
    t_["channel τ"] = round(time.time() - t0, 1); t_["channel points"] = len(ch); t0 = time.time()
    E = LT.light(rho, o, vs, plan, S_l, c, sigma=SIG, points=pts, z0=loc[2], octaves=1, g=0.85, ms_mode="diff", coarse=4 if not layer else 2, view_dir=-v_l,
                 ground=float(LS.get("ground_k", 1.0)), detail=RL["detail"] if detail is None else detail,
                 ts_blur=(1 if (detail or 0) >= 0.7 else 2) if detail is not None else (int(LS.get("ts_blur", RL["ts_blur"])) if "realism" not in LS else RL["ts_blur"]),
                 flashes=fl, info=info, channel=tuple(ch), ch_dir=chd, ch_detail=float(LS.get("bolt_detail", 1.0)), ch_ms=float(LS.get("bolt_ms", 0.5)), sky_detail=float(LS.get("sky_detail", 0.0)), sky_r=LS.get("sky_r"))
    t_["light"] = round(time.time() - t0, 1); t0 = time.time()
    Ls_ = float(np.dot(P["light"]["sky_e_rgb"], [0.2126, 0.7152, 0.0722])) / math.pi; PT = {}
    if LS.get("rim"):                                                          # "magic" dawn light — a paint: its own thickness, without neighbours and the horizon
        ar_, er_ = math.radians(float(P["sun"]["az"]) + float(LS.get("rim_daz", 0.0))), math.radians(float(LS.get("rim_el", -3.0)))
        tr_ = LT.tau_sun(rho, vs, Rz.T @ np.array([math.cos(ar_) * math.cos(er_), math.sin(ar_) * math.cos(er_), math.sin(er_)]), SIG)
        for _ in range(RL["ts_blur"] if "realism" in LS else int(LS.get("ts_blur", 2))):
            for ax in range(3): tr_ = (np.roll(tr_, 1, ax) + tr_ + np.roll(tr_, -1, ax)) / 3.0
        kf_ = 1.0 + float(LS.get("rim_far", 1.0)) * float(np.clip((np.linalg.norm(v) - 6000.0) / 10000.0, 0, 1))
        PT["rim"] = (SIG * rho[..., None] * np.exp(-float(LS.get("rim_a", 0.6)) * tr_)[..., None] * (float(LS["rim"]) * Ls_ * kf_ * np.array(LS.get("rim_rgb", [1.0, 0.45, 0.2])))).astype(np.float32)
        del tr_
    if LS.get("leak"):                                                         # light in thin places: from above through the cloud's own thickness
        tz_ = np.flip(np.cumsum(np.flip(rho, 2), 2), 2) * SIG * vs
        PT["leak"] = (SIG * rho[..., None] * np.exp(-float(LS.get("leak_a", 0.35)) * tz_)[..., None] * (float(LS["leak"]) * Ls_ * np.array(LS.get("leak_rgb", [1.0, 0.62, 0.38])))).astype(np.float32)
        del tz_
    if job.get("glow") and LS.get("glow"):                                     # distant clouds glow toward the camera
        tg_ = LT.tau_sun(rho, vs, -v_l / np.linalg.norm(v_l), SIG)
        PT["glow"] = (SIG * rho[..., None] * np.exp(-float(LS.get("glow_a", 1.0)) * tg_)[..., None] * (float(LS["glow"]) * Ls_ * np.array(LS.get("glow_rgb", [1.0, 0.7, 0.42])))).astype(np.float32)
        del tg_
    T = far.transmit(P["fog"]["layers"], float(cam_p[2]), float(math.hypot(v[0], v[1])), float(v[2])) ** float(LS.get("haze_k", 1.0))
    a_deg = math.degrees(math.atan2(v[1], v[0])) % 360; az_h, rgb_h = sh["az_h"], np.asarray(sh["rgb_h"])
    Lh = np.array([np.interp(a_deg, az_h, rgb_h[:, q], period=360) for q in range(3)])
    E = E * T + (SIG * rho[..., None] * (1 - T) * Lh[None, None, None, :]).astype(np.float32)   # air haze in front of the cloud
    FL = info.get("flash"); FL = FL * T if FL is not None else None; PT = {k_: v_ * T for k_, v_ in PT.items()}
    t_["paints"] = round(time.time() - t0, 1)
    entry = {"role": job["name"], "d": round(float(np.linalg.norm(v))), "H": round(float(rho.shape[2] * vs)), "T": round(T, 2), "sec": t_}
    if FL is not None:
        oc_ = rho > 0.02; Y_ = E[oc_] @ np.array([0.2126, 0.7152, 0.0722]); m_ = max(float(np.median(Y_)), 1e-6)
        entry["flash/light"] = [round(float(np.percentile(FL[oc_], q_)) / m_, 2) for q_ in (50, 90, 99)]
    return {"E": E, "FL": FL, "PT": PT, "T": T, "tau": tau, "entry": entry, "ctr": ctr, "cl_": cl_}


def run_job(args):
    d0, name = args; t = time.time()
    sh = pickle.load(open(os.path.join(d0, "shared.pkl"), "rb")); job = pickle.load(open(os.path.join(d0, name), "rb"))
    rho, o, vs, loc = form_cloud(job["cl"], sh, job.get("real", False)); t_f = round(time.time() - t, 1)
    r = bake_cloud(job, sh, rho, o, vs, loc); r["entry"]["sec"]["shape"] = t_f
    out = os.path.join(d0, name.replace("job_", "result_").replace(".pkl", ".pkl"))
    pickle.dump({"i": job["i"], "rho": rho, "o": o, "vs": vs, "loc": loc, "rot": job.get("rot", 0.0), **{k: r[k] for k in ("E", "FL", "PT", "T", "tau", "entry", "ctr", "cl_")}},
                open(out + ".part", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(out + ".part", out)
    return job["i"], round(time.time() - t, 1), int(rho.size)


def main(d0, workers=3):
    """A process pool over the jobs of a folder (the large ones first: they take longer); printed — the cloud, seconds, cells."""
    import multiprocessing as mp
    names = sorted(n for n in os.listdir(d0) if n.startswith("job_") and n.endswith(".pkl"))
    names.sort(key=lambda n: -pickle.load(open(os.path.join(d0, n), "rb")).get("cost", 0.0))
    ctx = mp.get_context("spawn"); t = time.time()
    with ctx.Pool(max(1, int(workers))) as pool:
        for i, s_, n_ in pool.imap_unordered(run_job, [(d0, n) for n in names]): print(f"cloud {i}: {s_} s, cells {n_}", flush=True)
    print(f"baking: {len(names)} clouds, {round(time.time() - t)} s, processes {workers}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]; opt = dict(x.split("=", 1) for x in a[1:] if "=" in x)
    main(a[0], int(opt.get("n_processes", 3)))
