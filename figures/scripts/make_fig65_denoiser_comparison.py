"""Decoder run 8 + each denoiser (control, G, G2, G3, G4): generation test, 11 real pockets, 88 designs per run (G and G3 pooled over two seeds = 176 designs). Bars = mean (+- 1 standard error over designs); right column = after the best-of-4 clash filter (rule B, Results.MD 67)."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json")); B = json.load(open(f"{D}/gen_test/diag/best_of_n.json"))
RUNS = {"control": ["control_v10"], "G": ["G_ncw10_v10", "G_ncw10_v10_s777"], "G2": ["G2_pool_v10"], "G3": ["G3_bal_v10", "G3_bal_v10_s777"], "G4": ["G4_clash_v10"]}
COL = {"control": "#999999", "G": "#2a9d8f", "G2": "#219ebc", "G3": "#6a4c93", "G4": "#e76f51"}
def vals(keys, k): return np.array([r[k] for key in keys for r in S[key]["rows"] if r[k] == r[k]])
f = plt.figure(figsize=(19, 11)); gs = f.add_gridspec(3, 3, height_ratios=[0.4, 1, 1], hspace=0.55, wspace=0.27)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#dde5b6", ec="#555"))
s.text(50, 5, "Test: 11 real pockets x 8 designs per denoiser, non-canonical identities given, decoder run 8, raw unrelaxed outputs. Denoisers: control = untuned; G, G2, G3 = LoRA fine-tunes\n(G, G3 pooled over two seeds = 176 designs); G4 = G3 + clash loss. Error bars = 1 standard error over designs (11 pockets only: differences of a few points are not significant).", ha="center", va="center", fontsize=8.6)
s.set_title("Which denoiser is best with decoder run 8?", fontsize=10.5, loc="left", fontweight="bold")
M = [("clash", "(a) designs with an atom pair < 2 A to the receptor (%, lower = better)", 100, True, (0, 45)), ("bond_ok", "(b) peptide bonds of sensible length (%)", 100, False, (0, 105)), ("chir_D", "(c) D-forms with the right handedness (%)", 100, False, (0, 112)),
     ("ca_rmsd", "(d) median CA distance to the native peptide (A, lower = better)", 1, "median", (0, 7)), ("contacts_rel", "(e) median contacts relative to native", 1, "median", (0, 1.9))]
for i, (k, title, mult, flag, yl) in enumerate(M):
    ax = f.add_subplot(gs[1, i] if i < 3 else gs[2, i - 3]); xs = np.arange(len(RUNS))
    for j, (nm, keys) in enumerate(RUNS.items()):
        v = vals(keys, k)
        if flag == "median": h = float(np.median(v)); err = 0
        else: h = mult * v.mean(); err = mult * v.std(ddof=1) / np.sqrt(len(v))
        ax.bar(j, h, 0.7, color=COL[nm], edgecolor="#333", linewidth=0.5, yerr=err if err else None, capsize=3)
        ax.text(j, h + (err or 0), f"{h:.1f}" if mult == 100 else f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(list(RUNS)); ax.set_ylim(*yl); ax.set_title(title, fontsize=9.0, loc="left", fontweight="bold")
ax = f.add_subplot(gs[2, 2]); ks = ["control", "G", "G2", "G3"]; w = 0.26
for j, (lab, key, col) in enumerate([("clash-free share", "clash_free_share", "#e76f51"), ("sensible bonds", "bond_ok", "#2a9d8f"), ("D right", "chir_D", "#6a4c93")]):
    ys = []
    for nm in ks: ys.append(100 * np.mean([B[k]["N"]["4B"][key] for k in RUNS[nm] if k in B]))
    ax.bar(np.arange(len(ks)) + (j - 1) * w, ys, w * 0.95, color=col, edgecolor="#333", linewidth=0.5, label=lab)
    for x, v in zip(np.arange(len(ks)) + (j - 1) * w, ys): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_ylim(0, 125); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3)
ax.set_title("(f) after the best-of-4 clash filter (rule B)", fontsize=9.0, loc="left", fontweight="bold")
f.text(0.01, 0.005, "Reading: no denoiser wins everything. G has the closest pose and best handedness but the most close contacts and fewest sensible bonds; G2 and G3 have fewer close contacts; G4 (clash loss) is not better than G3. With the clash filter G3 is the best all-rounder.", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig65_denoiser_comparison.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig65")
