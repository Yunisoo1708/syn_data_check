"""All decoder runs side by side: (a to c) generation test with real pockets (G, G2, G3 x 5 decoders, 88 designs per run): D-forms with the right handedness, peptide bonds of sensible length, designs with an atom pair closer than 2 A; (d to f) decoder alone on the TRUE latents of the real 103 complexes: D-forms decoded with the right handedness, atom RMSD of D-forms, atom RMSD of canonical residues."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json"))
DEC = [("v5", "run3", "run 3: residue-type loss only", "#bbbbbb"), ("v7", "run5", "run 5: + atom-coordinate loss", "#8ecae6"), ("v7b", "run5b", "run 5b: run 5 continued", "#219ebc"), ("v8", "run6", "run 6: NC x3 + chirality 0.5", "#e8590c"), ("v9", "run7", "run 7: NC x2 + chirality 0.15", "#6a4c93")]
DEN = [("G_ncw10", "G"), ("G2_pool", "G2"), ("G3_bal", "G3")]
def dg(run): return json.load(open(f"{D}/gen_test/diag/atoms_{run}_real.json"))["summary"]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.5, wspace=0.25)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Decoder runs", "run 3: residue-type loss only (identity can be copied)\nruns 5 to 7: + loss on the decoded atom positions of the 46k peptide pool,\nmirror D-peptides and synthetic complexes; 6 and 7 add a handedness loss", "#ffd6a5"),
       ("Generation test", "11 real pockets, peptide generated with the identity of its\nnon-canonical residues given; 88 designs per run;\ndenoisers G, G2, G3 x 5 decoders, raw outputs", "#ade8f4"),
       ("Decoder alone", "real 103 complexes: encode the native complex, decode the TRUE\nlatents, compare atoms and C-alpha handedness with the native\n(native D-forms are 100% right)", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: what training the decoder's atom output on D-forms changed", fontsize=10.5, loc="left", fontweight="bold")
gen = [("chir_D", "(a) D-forms with the right handedness (generation)", 100, "%"), ("bond_ok", "(b) peptide bonds of sensible length (generation)", 100, "%"), ("clash", "(c) designs with an atom pair < 2 A to the receptor (lower = better)", 100, "%")]
for i, (k, title, mult, unit) in enumerate(gen):
    ax = f.add_subplot(gs[1, i]); w = 0.16
    for j, (suf, rn, lab, col) in enumerate(DEC):
        vals = [S[f"{m}_{suf}"]["agg"][k] * mult for m, _ in DEN]
        ax.bar(np.arange(3) + (j - 2) * w, vals, w * 0.95, color=col, edgecolor="#333", linewidth=0.5, label=lab if i == 0 else None)
        for x, v in zip(np.arange(3) + (j - 2) * w, vals): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in DEN]); ax.set_ylim(0, 112); ax.set_ylabel(unit); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
from matplotlib.patches import Patch
f.legend(handles=[Patch(fc=c, ec="#333", label=l) for _, _, l, c in DEC], frameon=False, fontsize=8.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.025))
dd = [("D right", lambda d: 100 * d["D_decoded_right"], "(d) D-forms decoded right, true latents (%)"), ("D rmsd", lambda d: d["atom_rmsd_A"]["D-forms"], "(e) atom RMSD of D-forms (A, lower = better)"), ("can rmsd", lambda d: d["atom_rmsd_A"]["canonical"], "(f) atom RMSD of canonical residues (A, lower = better)")]
for i, (nm, fn, title) in enumerate(dd):
    ax = f.add_subplot(gs[2, i]); vals = [fn(dg(rn)) for _, rn, _, _ in DEC]
    ax.bar(range(5), vals, color=[c for *_, c in DEC], edgecolor="#333", linewidth=0.5)
    for x, v in enumerate(vals): ax.text(x, v, f"{v:.1f}" if i == 0 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["run 3", "run 5", "run 5b", "run 6", "run 7"]); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
    if i == 0: ax.set_ylim(0, 112)
f.text(0.01, 0.005, "Generation: raw outputs, 88 designs per run (11 complexes x 8), same seed. Decoder alone: real 103 complexes (55 with a D-form). Run 5b = best balance (handedness 94 to 96% with bond geometry kept); run 7 = handedness first (97 to 99%).", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig58_decoder_runs.png", bbox_inches="tight", dpi=130); plt.close(f)
for suf, rn, lab, col in DEC:
    d = dg(rn); print(lab.ljust(34), "D right %.1f%%" % (100 * d["D_decoded_right"]), "atom RMSD", {k: round(v, 2) for k, v in d["atom_rmsd_A"].items()})
