"""Run 9 (clash loss) vs run 8 and run 5b: (a-c) generation test with real pockets (G, G2, G3, control; 88 designs each): peptide bonds of sensible length, D-forms with the right handedness, designs with an atom pair closer than 2 A to the receptor; (d-f) decoder alone on the true latents of the real complexes: D-forms right, atom RMSD of D-forms, atom RMSD of canonical residues."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json"))
DEC = [("v7b", "run5b", "run 5b: + atom-coordinate loss", "#bbbbbb"), ("v10", "run8", "run 8: + peptide-bond loss (default so far)", "#2a9d8f"), ("v11", "run9", "run 9: + clash loss (noisy latent, sigma 0.2)", "#e76f51")]
DEN = [("G_ncw10", "G"), ("G2_pool", "G2"), ("G3_bal", "G3"), ("control", "control")]
def dg(run): return json.load(open(f"{D}/gen_test/diag/atoms_{run}_real.json"))["summary"]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.5, wspace=0.25)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Clash loss", "decoded atoms of the peptide vs TRUE atoms of the receptor chains:\npenalty = max(0, 2.5 A - distance)^2, weight 5; applied on the true latent\nand on a noised latent (sigma 0.2 x std) to mimic denoiser output", "#ffd6a5"),
       ("Training", "1 full H200, 4 epochs, start = run 8; run 8 loss kept (bond loss 10);\nnoised-pass bond term OFF (it gave a loss of 14 in the first\nattempt, which was cancelled)", "#ade8f4"),
       ("Tests", "generation: 11 real pockets, identity of the non-canonical\nresidues given, 88 designs per denoiser (G, G2, G3, control), raw outputs;\ndecoder alone: true latents of the real complexes", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: a loss on peptide-receptor atom distances of the decoded atoms", fontsize=10.5, loc="left", fontweight="bold")
gen = [("bond_ok", "(a) peptide bonds of sensible length 1.2-1.5 A (generation)", 100, "%"), ("chir_D", "(b) D-forms with the right handedness (generation)", 100, "%"), ("clash", "(c) designs with an atom pair < 2 A to the receptor (lower = better)", 100, "%")]
n = len(DEC); w = 0.26; nd = len(DEN)
for i, (k, title, mult, unit) in enumerate(gen):
    ax = f.add_subplot(gs[1, i])
    for j, (suf, rn, lab, col) in enumerate(DEC):
        vals = [S[f"{m}_{suf}"]["agg"][k] * mult if f"{m}_{suf}" in S else np.nan for m, _ in DEN]
        xs = np.arange(nd) + (j - (n - 1) / 2) * w
        ax.bar(xs, np.nan_to_num(vals), w * 0.95, color=col, edgecolor="#333", linewidth=0.5)
        for x, v in zip(xs, vals):
            if not np.isnan(v): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(nd)); ax.set_xticklabels([l for _, l in DEN]); ax.set_ylim(0, 112); ax.set_ylabel(unit); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
f.legend(handles=[Patch(fc=c, ec="#333", label=l) for _, _, l, c in DEC], frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.025))
dd = [(lambda d: 100 * d["D_decoded_right"], "(d) D-forms decoded right, true latents (%)"), (lambda d: d["atom_rmsd_A"]["D-forms"], "(e) atom RMSD of D-forms (A, lower = better)"), (lambda d: d["atom_rmsd_A"]["canonical"], "(f) atom RMSD of canonical residues (A, lower = better)")]
for i, (fn, title) in enumerate(dd):
    ax = f.add_subplot(gs[2, i]); vals = [fn(dg(rn)) for _, rn, _, _ in DEC]
    ax.bar(range(n), vals, color=[c for *_, c in DEC], edgecolor="#333", linewidth=0.5)
    for x, v in enumerate(vals): ax.text(x, v, f"{v:.1f}" if i == 0 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(["run 5b", "run 8", "run 9"]); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
    if i == 0: ax.set_ylim(0, 112)
f.text(0.01, 0.005, "Generation: raw outputs, 88 designs per run, same seed, unrelaxed. Decoder alone: real complexes. Result: run 9 did not lower close contacts (G 30->35%, G2 19->21%, G3 20->24%, control 22->22%) and lowered sensible bonds (G2 87->84%, G3 87->81%, control 87->78%); D handedness slightly up on 3 of 4.", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig62_clash_loss.png", bbox_inches="tight", dpi=130); plt.close(f)
for suf, rn, lab, col in DEC:
    d = dg(rn); print(lab.ljust(50), "D right %.1f%%" % (100 * d["D_decoded_right"]), {k: round(v, 2) for k, v in d["atom_rmsd_A"].items()}, " | gen bonds", [round(100 * S[f"{m}_{suf}"]["agg"]["bond_ok"]) for m, _ in DEN if f"{m}_{suf}" in S], "D", [round(100 * S[f"{m}_{suf}"]["agg"]["chir_D"]) for m, _ in DEN if f"{m}_{suf}" in S], "clash", [round(100 * S[f"{m}_{suf}"]["agg"]["clash"]) for m, _ in DEN if f"{m}_{suf}" in S])
