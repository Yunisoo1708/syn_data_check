"""Identity-withheld decode test: the residue identity is NOT given to the denoiser or the decoder (MODE v1). Decoder run 3 (decoder_v5_493747.pt, trained with the identity given) against the blind decoder run 4 (decoder_v6_496342.pt, identity hidden in training).
Per residue group (13 D-forms, 12 general codes, other vocabulary codes) and test set (held-out synthetic, real 103, extra real 884): top-1 accuracy at non-canonical positions of (a) the decoder alone on the TRUE latents (oracle) and (b) denoiser + decoder (control, G, G2, G3)."""
import os, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = f"{D}/denoiser_trial/saved"
DF13 = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split()); GEN12 = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
SETS = [("heldout", "held-out synthetic"), ("real", "real 103"), ("ham", "extra real 884")]
MODELS = [("control", "control", "#8d99ae"), ("G_ncw10", "G", "#d62768"), ("G2_pool", "G2", "#6a4c93"), ("G3_bal", "G3", "#e8590c")]
def grp(c): return "D-forms" if c in DF13 else ("general codes" if c in GEN12 else "other codes")
def agg(dirname, model, st):
    p = f"{S}/{dirname}/v1_{st}_{model}.json"
    if not os.path.exists(p): return None
    R = json.load(open(p))["results"]; t = {g: [0, 0.0, 0.0] for g in ("D-forms", "general codes", "other codes")}
    for k, v in R.items():
        for c, e in v.get("codes", {}).items():
            if c.startswith("_"): continue
            x = t[grp(c)]; x[0] += e["n"]; x[1] += e["n"] * e["acc"]; x[2] += e["n"] * e["acc_oracle"]
    return {g: (100 * x[1] / x[0], 100 * x[2] / x[0], x[0]) if x[0] else None for g, x in t.items()}
tab = {(d, m, s): agg(d, m, s) for d in ("eval_blind", "eval_blind6") for m, _, _ in MODELS for s in [x[0] for x in SETS]}
print("accuracy at non-canonical positions, identity withheld (denoiser+decoder | decoder alone on true latents)")
for g in ("D-forms", "general codes", "other codes"):
    for st, sl in SETS:
        r = {d: tab[(d, "control", st)] for d in ("eval_blind", "eval_blind6")}
        if r["eval_blind"] is None or r["eval_blind"][g] is None: continue
        line = f"{g:14s} {sl:20s} n={r['eval_blind'][g][2]:4d} | decoder alone: run3 {r['eval_blind'][g][1]:5.1f}%  run4 {r['eval_blind6'][g][1]:5.1f}% | with denoiser (run3 / run4 decoder): " + "  ".join(
            f"{lab} {tab[('eval_blind', m, st)][g][0]:4.1f}/{tab[('eval_blind6', m, st)][g][0]:4.1f}" for m, lab, _ in MODELS if tab[("eval_blind6", m, st)])
        print(line)
f = plt.figure(figsize=(19, 10)); gs = f.add_gridspec(2, 3, height_ratios=[0.5, 2.4], hspace=0.4, wspace=0.25)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Residue identity hidden", "the residue identity that the pipeline normally gives for\nnon-canonical residues is replaced by <M> for the denoiser\nand the decoder (token type kept)", "#ffd6a5"),
       ("Two decoders", "run 3: trained with the identity given (can copy it)\nrun 4: retrained with the identity hidden\nsame data, started from run 3", "#ade8f4"),
       ("Two readings", "decoder alone: true latents of the complex (its own ceiling)\ndenoiser + decoder: latents predicted by control, G, G2, G3\ntop-1 accuracy of the residue name", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 1), 31, 8, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 7.4, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 3.7, d, ha="center", va="center", fontsize=8.3)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: can the models name a non-canonical residue when its identity is not given?", fontsize=10.5, loc="left", fontweight="bold")
for gi, g in enumerate(("D-forms", "general codes", "other codes")):
    ax = f.add_subplot(gs[1, gi]); w = 0.09; labels = []
    bars = [("decoder alone, run 3", "eval_blind", None, 1, "#bbbbbb", ""), ("decoder alone, run 4", "eval_blind6", None, 1, "#555555", "//")]
    for m, lab, col in MODELS: bars += [(f"{lab} + run 3", "eval_blind", m, 0, col, ""), (f"{lab} + run 4", "eval_blind6", m, 0, col, "//")]
    for bi, (lbl, d, m, ci, col, hat) in enumerate(bars):
        xs, ys = [], []
        for si, (st, sl) in enumerate(SETS):
            r = tab[(d, m or "control", st)]
            if r is None or r[g] is None: continue
            xs.append(si + (bi - len(bars) / 2) * w); ys.append(r[g][ci])
        ax.bar(xs, ys, w * 0.95, color=col, hatch=hat, edgecolor="#333", linewidth=0.5, label=lbl if gi == 0 else None)
    n = tab[("eval_blind", "control", "real")][g][2] if tab[("eval_blind", "control", "real")] and tab[("eval_blind", "control", "real")][g] else 0
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels([f"{sl}\n({tab[('eval_blind', 'control', st)][g][2]} positions)" for st, sl in SETS], fontsize=9); ax.set_ylim(0, 100); ax.set_ylabel("top-1 accuracy, identity withheld (%)")
    ax.set_title(g, fontsize=11, loc="left", fontweight="bold")
f.axes[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
f.text(0.01, 0.005, "Latent time t = 0.15 (extra real 0.1), same noise per complex for every model; one run per model. Chance among the 142 non-canonical classes is below 1%. Hatched = decoder run 4 (identity hidden in training).", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig55_blind_decoder.png", bbox_inches="tight", dpi=130); plt.close(f)
