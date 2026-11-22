"""Figure: untuned control vs trial A, B, C (best weights of each) scored on EVERY held-out complex, same noise per complex for all models (paired)."""
import os, json, sys, numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune"; E = f"{D}/denoiser_trial/saved/eval"
MODELS = [("control", "Untuned control", "#8d99ae"), ("A_all", "Trial A", "#2a6f97"), ("B_first", "Trial B", "#c9a227"), ("C_long", "Trial C", "#b23a48")]
extra = [m for m in sys.argv[1:]]
EXTRA = {"D_rank64": ("Trial D", "#3a7d44", "same data and layers as C\nLoRA rank 64 (alpha 128)\nbest = step 1,050 (early stop 2,250)"),
         "E_rank64_ff": ("Trial E", "#7b5ea7", "rank 64 on attention plus\nfeed-forward layers and gate\n(70 adapted layers)"),
         "F_pool2": ("Trial F (box)", "#e07a1f", "C settings on the 6,457-\ncomplex pool; lr 1e-4, batch 1\nbest = step 750 (stopped 981)"),
         "G_ncw10": ("Trial G", "#d62768", "C recipe + 10x weight on non-canonical\npositions, noise weight capped, lr halved\non plateau; best = step 2,700 (time cap)"),
         "F2_H200": ("Trial F2", "#0b6e4f", "F (H200) best weights continued\nat lr 5e-5; best = step 300\n(early stop at 800)"),
         "F_H200": ("Trial F (H200)", "#17a2b8", "same pool; batch 4, lr 2e-4\nbest = step 800\n(early stop at 1,600)")}
for n in extra:
    if os.path.exists(f"{E}/{n}.json"): MODELS.append((n, EXTRA.get(n, (n, "#3a7d44", ""))[0], EXTRA.get(n, (n, "#3a7d44", ""))[1]))
R = {n: json.load(open(f"{E}/{n}.json"))["results"] for n, _, _ in MODELS if os.path.exists(f"{E}/{n}.json")}
MODELS = [m for m in MODELS if m[0] in R]
files = sorted(set.intersection(*[{f for f, v in R[n].items() if "total" in v} for n in R]))
TERMS = ["bb_ca", "local_latents", "sequence_prediction"]
def arr(n, key="total"): return np.array([R[n][f][key] for f in files])
ctrl = arr("control")
cut = np.percentile(ctrl, 95)  # the 5% of complexes whose noise draw is at the extreme low-noise end dominate the mean for every model alike
keep = ctrl <= cut
src = np.array(["first batch" if f.split("_")[1].startswith("ncf") else "top-up" for f in files])
rng = np.random.default_rng(0)
def boot(d, n=2000):
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return np.percentile(m, [2.5, 97.5])
print(f"complexes scored in all models: {len(files)} (first batch {int((src == 'first batch').sum())}, top-up {int((src == 'top-up').sum())}); trimmed set (control loss <= p95): {int(keep.sum())}")
rows = {}
for n, lab, col in MODELS:
    a = arr(n); d = (a - ctrl)[keep]
    rows[n] = dict(mean_all=a.mean(), mean_trim=a[keep].mean(), diff=d.mean(), ci=boot(d), win=float((d < 0).mean()), med_ratio=float(np.median(a[keep] / ctrl[keep])),
                   terms={t: arr(n, t)[keep].mean() for t in TERMS}, diff_first=(a - ctrl)[keep & (src == "first batch")].mean(), diff_top=(a - ctrl)[keep & (src == "top-up")].mean())
    print(f"{lab:16s} mean {a[keep].mean():8.3f} | paired change vs control {d.mean():+.3f} (95% CI {rows[n]['ci'][0]:+.3f} to {rows[n]['ci'][1]:+.3f}) | lower than control on {100 * (d < 0).mean():.0f}% of complexes | first-batch {rows[n]['diff_first']:+.3f} top-up {rows[n]['diff_top']:+.3f}")
json.dump({n: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for n, r in rows.items()}, open(f"{E}/summary.json", "w"), indent=1, default=float)

f = plt.figure(figsize=(19, 10.2)); gs = f.add_gridspec(2, 3, height_ratios=[0.9, 2.2], hspace=0.34, wspace=0.3)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
from matplotlib.patches import FancyBboxPatch
info = [("Untuned control", "RFD4 flow model as is\n(189-class bundle)\nno training", "#e9ecef"),
        ("Trial A", "all 1,266 training complexes\nLoRA rank 16, attention only\nbest = step 500 (of 900)", "#cfe3f1"),
        ("Trial B", "first batch only: 228 complexes\nsame LoRA as A\nbest = step 800 (of 1,000)", "#f6e7b4"),
        ("Trial C", "all 1,266 training complexes\nsame LoRA, 30-complex checks\nbest = step 600 (of 1,143)", "#efc6cc")]
for n, _, col in MODELS:
    if n in EXTRA: info.append((EXTRA[n][0], EXTRA[n][2], {"D_rank64": "#cfe8d0", "E_rank64_ff": "#ddd0ee"}.get(n, "#e0e0e0")))
W = 98 / len(info)
for i, (t, d, c) in enumerate(info):
    x = 1 + i * W
    s.add_patch(FancyBboxPatch((x, 1), W - 2.5, 8, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + (W - 2.5) / 2, 7.4, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + (W - 2.5) / 2, 3.6, d, ha="center", va="center", fontsize=8 if len(info) > 4 else 8.5)
s.set_title(f"Same {len(files)} held-out complexes (protein families never trained on), same noise on each complex, every model scored the same way", fontsize=10.5, loc="left", fontweight="bold")
a1 = f.add_subplot(gs[1, 0]); x = np.arange(len(MODELS))
vals = [rows[n]["mean_trim"] for n, _, _ in MODELS]
a1.bar(x, vals, color=[c for _, _, c in MODELS], edgecolor="#555")
for i, v in enumerate(vals): a1.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
lo = min(vals) - 1.5; a1.set_ylim(lo, max(vals) + 1.2); a1.set_xticks(x); a1.set_xticklabels([l for _, l, _ in MODELS], fontsize=9); a1.set_ylabel("held-out flow loss (lower is better)")
a1.set_title("(a) Mean loss (axis cut)", fontsize=10.5, loc="left", fontweight="bold")
a2 = f.add_subplot(gs[1, 1]); mm = [m for m in MODELS if m[0] != "control"]; y = np.arange(len(mm))
for i, (n, l, c) in enumerate(mm):
    r = rows[n]; a2.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]], fmt="o", color=c, capsize=4, ms=9)
    a2.text(r["ci"][1] + 0.05, i, f"better on {100 * r['win']:.0f}% of complexes", va="center", fontsize=8.5)
a2.axvline(0, color="#555", lw=1); a2.set_yticks(y); a2.set_yticklabels([l for _, l, _ in mm]); a2.invert_yaxis(); a2.set_xlabel("change vs untuned control, per complex (negative = better)")
a2.set_title("(b) Paired change with 95% interval", fontsize=10.5, loc="left", fontweight="bold")
a3 = f.add_subplot(gs[1, 2]); w = 0.8 / max(1, len(mm))
for i, (n, l, c) in enumerate(mm):
    ch = [100 * (rows[n]["terms"][t] / rows["control"]["terms"][t] - 1) for t in TERMS]
    a3.bar(np.arange(len(TERMS)) + (i - (len(mm) - 1) / 2) * w, ch, w, color=c, edgecolor="#555", label=l)
a3.axhline(0, color="#555", lw=1); a3.set_xticks(range(len(TERMS))); a3.set_xticklabels(["backbone CA\n(MSE)", "latent code\n(MSE)", "residue type\n(cross-entropy)"], fontsize=9); a3.set_ylabel("change vs control (%)"); a3.legend(frameon=False, fontsize=8.5)
a3.set_title("(c) By loss term", fontsize=10.5, loc="left", fontweight="bold")
f.text(0.01, 0.005, f"Test = the {len(files)} held-out complexes of the training set (12% of protein families); the 5% with the most extreme noise draw ({len(files) - int(keep.sum())} complexes) are left out of the means for every model alike. Flow loss, not generated structures. One run per model.", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig38_trials_heldout_test.png", bbox_inches="tight", dpi=130); plt.close(f)
