"""Denoise-then-decode benchmark with ONE fixed decoder (retrained, 189 classes): untuned control and every fine-tuned denoiser, same noise per complex.
Per set (real 103, held-out 163 at latent noise levels t = 0.05 / 0.15 / 0.3, extra real 884 -> the complexes with a scored non-canonical residue at t = 0.1): top-1 accuracy of the decoder's residue naming from the denoiser's predicted latents
at non-canonical positions (residues outside the 189 vocabulary masked), by residue group, against the control, plus the oracle (decoding the TRUE latents)."""
import os, json, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = f"{D}/denoiser_trial/saved/eval_dec"
MODELS = [("control", "Untuned control", "#8d99ae"), ("A_all", "A", "#2a6f97"), ("B_first", "B", "#c9a227"), ("C_long", "C", "#b23a48"), ("D_rank64", "D", "#3a7d44"), ("E_rank64_ff", "E", "#7b5ea7"),
          ("F_pool2", "F box", "#e07a1f"), ("F_H200", "F H200", "#17a2b8"), ("F2_H200", "F2", "#0b6e4f"), ("G_ncw10", "G", "#d62768")]
DF13 = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split()); GEN12 = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
rng = np.random.default_rng(0); names = [n for n, _, _ in MODELS if n != "control"]; lab = {n: l for n, l, _ in MODELS}; col = {n: c for n, _, c in MODELS}
def load(st): return {n: json.load(open(f"{S}/{st}_{n}.json"))["results"] for n, _, _ in MODELS if os.path.exists(f"{S}/{st}_{n}.json")}
def agg(R, keys, grp, field):
    """per entry token-weighted mean of `field` over the codes of group grp (None = all non-canonical, 'CAN' = canonical): returns {model: (value array, n array)}"""
    out = {}
    for n, r in R.items():
        v = []; w = []
        for k in keys:
            cs = r[k]["codes"]
            sel = {c: e for c, e in cs.items() if (c == "_CANONICAL" if grp == "CAN" else (not c.startswith("_") and (grp is None or c in grp)))}
            tn = sum(e["n"] for e in sel.values()); v.append(sum(e["n"] * e[field] for e in sel.values()) / tn if tn else np.nan); w.append(tn)
        out[n] = (np.array(v), np.array(w, float))
    return out
def diff(vals, n, scale=100.0):
    cv, cw = vals["control"]; mv, mw = vals[n]; ok = ~np.isnan(cv) & ~np.isnan(mv); w = cw[ok]; d = (mv - cv)[ok]
    est = scale * np.average(d, weights=w); idx = np.arange(len(d)); bs = [scale * np.average(d[i], weights=w[i]) for i in (rng.choice(idx, len(idx)) for _ in range(1200))]
    return est, np.percentile(bs, [2.5, 97.5]), int(w.sum())
SETS = [("real", None, "103 real complexes"), ("heldout", "0.05", "163 held-out, latent t=0.05"), ("heldout", "0.15", "163 held-out, t=0.15"), ("heldout", "0.3", "163 held-out, t=0.3"), ("ham", "0.1", "extra real (non-canonical complexes), t=0.1")]
# the real set was scored at t = 0.05 / 0.15 / 0.3 as well
SETS = [("real", "0.05", "103 real, t=0.05"), ("real", "0.15", "103 real, t=0.15"), ("real", "0.3", "103 real, t=0.3")] + SETS[1:]
rows = {}; report = []
for st, tk, ttl in SETS:
    R = load(st)
    if len(R) < 2 or "control" not in R: continue
    keys = sorted(set.intersection(*[{k for k, v in r.items() if "codes" in v and (tk is None or k.endswith("@" + tk))} for r in R.values()]))
    rows[(st, tk)] = dict(title=ttl, n=len(keys))
    for g, gn in ((None, "all non-canonical"), (DF13, "13 D-forms"), (GEN12, "12 general codes"), ("CAN", "canonical")):
        a = agg(R, keys, g, "acc"); o = agg(R, keys, g, "acc_oracle"); ok = ~np.isnan(a["control"][0]); ntok = int(a["control"][1][ok].sum())
        ctrl = 100 * np.average(a["control"][0][ok], weights=a["control"][1][ok]) if ok.any() else np.nan
        orc = 100 * np.average(o["control"][0][ok], weights=o["control"][1][ok]) if ok.any() else np.nan
        ch = {n: diff(a, n) for n in names if n in R}
        rows[(st, tk)][gn] = dict(ntok=ntok, control=ctrl, oracle=orc, change=ch)
        report.append(f"{ttl:34s} | {gn:18s} | {ntok:6d} positions | control {ctrl:5.1f}% (decode of true latents {orc:5.1f}%) | " + " ".join(f"{lab[n]} {c[0]:+.1f}" for n, c in ch.items()))
print("\n".join(report))
json.dump({f"{k[0]}@{k[1]}": {g: {kk: (vv if kk != "change" else {n: [float(c[0]), [float(x) for x in c[1]], c[2]] for n, c in vv.items()}) for kk, vv in v.items()} if isinstance(v, dict) else v for g, v in r.items()} for k, r in rows.items()}, open(f"{S}/decode_summary.json", "w"), indent=1, default=float)
keys_show = [k for k in rows if k[0] in ("heldout", "real", "ham")]
fig, axs = plt.subplots(2, 3, figsize=(20, 10.4), gridspec_kw=dict(hspace=0.45, wspace=0.3))
show = [(("heldout", "0.05"), "all non-canonical"), (("heldout", "0.05"), "13 D-forms"), (("heldout", "0.05"), "12 general codes"), (("real", "0.05"), "all non-canonical"), (("ham", "0.1"), "all non-canonical"), (("real", "0.05"), "canonical")]
for ax, (k, g) in zip(axs.ravel(), show):
    if k not in rows: ax.axis("off"); continue
    v = rows[k][g]; y = np.arange(len(names))
    for i, n in enumerate(names):
        if n in v["change"]: m, ci, _ = v["change"][n]; ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=col[n], capsize=3, ms=7)
    ax.axvline(0, color="#555", lw=1); ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in names]); ax.invert_yaxis(); ax.set_xlabel("accuracy change vs untuned control (percentage points)", fontsize=9)
    ax.set_title(f"{rows[k]['title']}: {g}\n(control {v['control']:.1f}%, {v['ntok']} positions)", fontsize=10, loc="left", fontweight="bold")
fig.suptitle("Denoise then decode with the SAME retrained decoder: residue naming accuracy from each denoiser's predicted latents (positive = better than the untuned control)", fontsize=12, fontweight="bold", y=0.995)
fig.savefig(f"{D}/figures/fig44_decode_same_decoder.png", bbox_inches="tight", dpi=130); plt.close(fig)
