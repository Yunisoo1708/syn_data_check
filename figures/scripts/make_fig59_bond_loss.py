"""Run 8 (peptide-bond loss) vs run 3, 5b, 7: (a-c) generation test with real pockets (G, G2, G3; 88 designs each): peptide bonds of sensible length, D-forms with the right handedness, designs with an atom pair closer than 2 A; (d-f) decoder alone on the true latents of the real 103 complexes: D-forms right, atom RMSD of D-forms, atom RMSD of canonical residues."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json"))
DEC = [("v5", "run3", "run 3: residue-type loss only", "#bbbbbb"), ("v7b", "run5b", "run 5b: + atom-coordinate loss (default so far)", "#219ebc"), ("v9", "run7", "run 7: + chirality loss", "#6a4c93"), ("v10", "run8", "run 8: + peptide-bond loss (H200)", "#2a9d8f")]
DEN = [("G_ncw10", "G"), ("G2_pool", "G2"), ("G3_bal", "G3")]
def dg(run): return json.load(open(f"{D}/gen_test/diag/atoms_{run}_real.json"))["summary"]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.5, wspace=0.25)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Bond loss", "for every pair of neighbours that are bonded in the true structure:\npenalty = max(0, |C(i)-N(i+1) - 1.33 A| - 0.04 A)^2 on the DECODED atoms,\nweight 10, added to the run 7 loss (CE + atom coordinates + chirality)", "#ffd6a5"),
       ("Training", "1 full H200, 4 epochs (37 min, about 0.6 USD), start = run 5b;\n46k peptides + 7,534 mirror D-peptides + 11,291 complexes;\nnoised-latent bond term tried first and dropped (diverged)", "#ade8f4"),
       ("Tests", "generation: 11 real pockets, identity of the non-canonical\nresidues given, 88 designs per denoiser (G, G2, G3), raw outputs;\ndecoder alone: true latents of the real 103 complexes", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: a loss on the peptide-bond length of the decoded atoms", fontsize=10.5, loc="left", fontweight="bold")
gen = [("bond_ok", "(a) peptide bonds of sensible length 1.2-1.5 A (generation)", 100, "%"), ("chir_D", "(b) D-forms with the right handedness (generation)", 100, "%"), ("clash", "(c) designs with an atom pair < 2 A to the receptor (lower = better)", 100, "%")]
n = len(DEC); w = 0.2
for i, (k, title, mult, unit) in enumerate(gen):
    ax = f.add_subplot(gs[1, i])
    for j, (suf, rn, lab, col) in enumerate(DEC):
        vals = [S[f"{m}_{suf}"]["agg"][k] * mult for m, _ in DEN]
        xs = np.arange(3) + (j - (n - 1) / 2) * w
        ax.bar(xs, vals, w * 0.95, color=col, edgecolor="#333", linewidth=0.5)
        for x, v in zip(xs, vals): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in DEN]); ax.set_ylim(0, 112); ax.set_ylabel(unit); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
f.legend(handles=[Patch(fc=c, ec="#333", label=l) for _, _, l, c in DEC], frameon=False, fontsize=8.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.025))
dd = [(lambda d: 100 * d["D_decoded_right"], "(d) D-forms decoded right, true latents (%)"), (lambda d: d["atom_rmsd_A"]["D-forms"], "(e) atom RMSD of D-forms (A, lower = better)"), (lambda d: d["atom_rmsd_A"]["canonical"], "(f) atom RMSD of canonical residues (A, lower = better)")]
for i, (fn, title) in enumerate(dd):
    ax = f.add_subplot(gs[2, i]); vals = [fn(dg(rn)) for _, rn, _, _ in DEC]
    ax.bar(range(n), vals, color=[c for *_, c in DEC], edgecolor="#333", linewidth=0.5)
    for x, v in enumerate(vals): ax.text(x, v, f"{v:.1f}" if i == 0 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(["run 3", "run 5b", "run 7", "run 8"]); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
    if i == 0: ax.set_ylim(0, 112)
f.text(0.01, 0.005, "Generation: raw outputs, 88 designs per run, same seed. Decoder alone: real 103 complexes (55 with a D-form). Run 8 lifts sensible bonds to 83-87% (run 5b 67-74%) with D handedness 92-100%; on true latents it is slightly weaker on D-forms and canonical atoms.", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig59_bond_loss.png", bbox_inches="tight", dpi=130); plt.close(f)
for suf, rn, lab, col in DEC:
    d = dg(rn); print(lab.ljust(46), "D right %.1f%%" % (100 * d["D_decoded_right"]), {k: round(v, 2) for k, v in d["atom_rmsd_A"].items()}, " | gen bonds", [round(100 * S[f"{m}_{suf}"]["agg"]["bond_ok"]) for m, _ in DEN], "D", [round(100 * S[f"{m}_{suf}"]["agg"]["chir_D"]) for m, _ in DEN], "clash", [round(100 * S[f"{m}_{suf}"]["agg"]["clash"]) for m, _ in DEN])
