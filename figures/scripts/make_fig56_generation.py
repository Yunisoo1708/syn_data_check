"""Generation test: 4 denoisers (control, G, G2, G3) x 2 decoders (run 3 = identity given in training, run 4 = blind) on 11 real peptide-protein complexes (test only), 8 designs each (88 per run), same seed.
Receptor pocket fixed, hotspots = receptor residues within 4.5 A of the native peptide, the non-canonical residues of the native peptide given at their positions, everything else designed."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json"))
MODELS = [("control", "control", "#8d99ae"), ("G_ncw10", "G", "#d62768"), ("G2_pool", "G2", "#6a4c93"), ("G3_bal", "G3", "#e8590c")]
METR = [("nc_kept", "given non-canonical residues kept", 100, "%"), ("chir_D", "D-forms with the right handedness", 100, "%"), ("chir_L", "canonical L-residues with the right handedness", 100, "%"), ("nc_atoms", "non-canonical residues with the native atom count", 100, "%"),
        ("free_rec", "sequence identity to native at free positions", 100, "%"), ("ca_rmsd_med", "median CA distance to the native peptide (lower = better)", 1, "A"),
        ("contacts_rel_med", "median peptide-receptor contacts vs native (1 = same)", 1, "x"), ("bond_ok", "peptide bonds of sensible length (1.2 to 1.5 A)", 100, "%"), ("clash", "designs with an atom pair closer than 2 A (lower = better)", 100, "%")]
print("run".ljust(14), " ".join(m[0][:9].rjust(10) for m in METR))
for key in S: print(key.ljust(14), " ".join(f"{S[key]['agg'][m[0]] * m[2]:10.2f}" for m in METR))
f = plt.figure(figsize=(19, 12.5)); gs = f.add_gridspec(4, 3, height_ratios=[0.55, 1, 1, 1], hspace=0.5, wspace=0.25)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Input", "11 real peptide-protein complexes (test only)\nreceptor pocket fixed; hotspots = pocket residues\nwithin 4.5 A of the native peptide; the non-canonical\nresidues of the peptide are GIVEN at their positions", "#ffd6a5"),
       ("Generate", "the peptide backbone, side chains and the free residues\nare generated from noise (60 steps), 8 designs per complex\n4 denoisers x 2 decoders, same seed", "#ade8f4"),
       ("Check", "against the native peptide after superposing on the receptor:\nidentity kept, handedness, atom count, backbone distance,\ncontacts, bond lengths, clashes (raw outputs, no refinement)", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: design the peptide around a fixed pocket with non-canonical residues specified, then check the raw designs", fontsize=10.5, loc="left", fontweight="bold")
for i, (k, title, mult, unit) in enumerate(METR):
    ax = f.add_subplot(gs[1 + i // 3, i % 3]); w = 0.36
    for j, (dec, hat) in enumerate((("v5", ""), ("v6", "//"))):
        vals = [S[f"{m}_{dec}"]["agg"][k] * mult for m, _, _ in MODELS]
        ax.bar(np.arange(4) + (j - 0.5) * w, vals, w * 0.95, color=[c for _, _, c in MODELS], hatch=hat, edgecolor="#333", linewidth=0.6, label=("decoder run 3 (identity given in training)" if dec == "v5" else "decoder run 4 (blind)") if i == 0 else None)
        for x, v in zip(np.arange(4) + (j - 0.5) * w, vals): ax.text(x, v, f"{v:.0f}" if mult == 100 else f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in MODELS]); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold"); ax.set_ylabel(unit)
    if mult == 100: ax.set_ylim(0, 112)
from matplotlib.patches import Patch
f.axes[1].legend(handles=[Patch(fc="#cccccc", ec="#333", label="decoder run 3 (identity given in training)"), Patch(fc="#cccccc", ec="#333", hatch="//", label="decoder run 4 (blind)")], frameon=False, fontsize=8, loc="lower left")
f.text(0.01, 0.005, "88 designs per run (11 complexes x 8). Raw model outputs, no relaxation or refolding; handedness = sign of the C-alpha signed volume (N, C, CB), reference from the native receptor L-residues. One run per setting.", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig56_generation_test.png", bbox_inches="tight", dpi=130); plt.close(f)
