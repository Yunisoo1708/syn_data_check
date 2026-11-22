"""How the synthetic data were made (all numbers from Results.MD): sections 1-2 (peptide library), 10, 28-34 (NCFlow conversion of natural complexes), 36 (training format), 40 (limits), 49 (mirror-image D-peptides), 52-52c (pool counts, balanced subset).
Real non-canonical complexes are NOT part of any of it (test only)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D = "C:/Research/RFD4_fine_tune"
BLU, GRN, GLD, RED, GRY, ORG = "#cfe8ef", "#b7e4c7", "#e9c46a", "#f7d6d6", "#dddddd", "#f4a261"
f = plt.figure(figsize=(20, 13)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, txt, fc, fs=8.2, bold=False, ec="#444", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.5", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arr(x0, y0, x1, y1, c="#444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=c, lw=1.4))

def row(y, h, items, x0=1.5, w=12.3, gap=1.6):
    for i, (t, c) in enumerate(items):
        x = x0 + i * (w + gap); box(x, y, w, h, t, c)
        if i < len(items) - 1: arr(x + w + 0.05, y + h / 2, x + w + gap - 0.05, y + h / 2)

ax.text(1.5, 97.5, "How the synthetic training data were made", fontsize=15, fontweight="bold")
ax.text(1.5, 94.8, "Two sources, both synthetic. Real non-canonical (NCAA) complexes are never used for training (test only).", fontsize=9.5, color="#444")

# ---------- A: peptide pool ----------
ax.text(1.5, 91, "A", fontsize=15, fontweight="bold", color="#264653"); ax.text(4, 91, "Source 1: free peptides (teach the decoder to read and build NCAA residues)", fontsize=11.5, fontweight="bold")
row(78, 10, [("125,000 sequences\n32 real NCAA monomers\n+ canonical, 6-30 residues", BLU), ("RDKit builds a\nstarting 3D conformer", BLU), ("RF3 predicts the\nstructure (Boltz-2 for\npart of the pool)", BLU), ("repair stereocentres\nthat came out with\nthe wrong hand", BLU),
              ("strict test of every\nresidue: bonds +-0.2 A,\nangles, clashes, C-N link\n1.33 +-0.2 A, handedness", GRN), ("pool: 46,379 peptides\n(43,465 RF3 passed,\nabout 1 in 3)", GLD), ("+ 7,534 mirror-image\nD-peptides: take an L\npeptide, x -> -x,\nswap codes L<->D", GLD)])
ax.text(1.5, 74.6, "Why mirror images: a D-residue made by relabelling an L-residue sits on an L backbone, so the decoder never saw a real D-backbone. The mirror image of a valid L peptide is a valid D peptide (decoder run 3, Results.MD 49).", fontsize=8.4, color="#444")

# ---------- B: complexes ----------
ax.text(1.5, 69, "B", fontsize=15, fontweight="bold", color="#264653"); ax.text(4, 69, "Source 2: synthetic complexes (an NCAA put into a real peptide-protein complex, one residue at a time)", fontsize=11.5, fontweight="bold")
row(54, 11, [("natural complex\nfrom the TRAIN split:\ncanonical peptide, 5-30\nres, 3,038 entries,\nabout 700 families", BLU), ("pick ONE peptide\nresidue (least touching\nfirst); candidate NCAAs\nare analogues of that\nresidue's type", BLU), ("NCFlow draws 4 new\nresidues per round, up to\n3 rounds (model trained\non synthetic data only)", BLU), ("mirror repair\n(a draw comes out\nmirrored about half\nthe time)", BLU),
              ("5 gates, all must pass\n(see below)", GRN), ("accepted: keep it and\nmove to the next residue\n(at most 8 swaps per\ncomplex, natural pose kept)", GLD), ("write the training CIF:\n12 A pocket around the\npeptide, protein chains\nsplit at gaps, split\nguard, geometry re-check", GLD)])
# gates detail
box(50.2, 38.5, 15.4, 13.5, "", "#eaf6ee", ec="#2a9d8f")
ax.text(57.9, 50.6, "the 5 gates", ha="center", fontsize=8.8, fontweight="bold")
ax.text(50.8, 44.2, "1 strict chemistry test\n2 no receptor atom < 2.2 A\n3 >= 90% of residue and polar\n    contacts kept\n4 CA drift <= 1.0 A (any CA 1.5)\n5 contact ratio 0.9 to 1.3", fontsize=7.6, va="center")
arr(57.9, 52, 57.9, 51.7)

# ---------- C: draws outcome chart ----------
ax2 = f.add_axes([0.06, 0.09, 0.22, 0.17])
lab = ["passed all\ngates", "failed strict\nchemistry", "clashed with\nthe protein", "lost\ninteractions", "drifted"]; val = [4490, 4023, 1915, 1759, 285]
ax2.barh(range(5), val, color=["#2a9d8f", "#e76f51", "#e76f51", "#e76f51", "#e76f51"], edgecolor="#333", lw=0.5); ax2.set_yticks(range(5)); ax2.set_yticklabels(lab, fontsize=8); ax2.invert_yaxis()
for i, v in enumerate(val): ax2.text(v + 60, i, f"{v:,} ({100 * v / 12472:.0f}%)", va="center", fontsize=8)
ax2.set_xlim(0, 6500); ax2.set_title("What happened to 12,472 draws (264 pilot complexes)", fontsize=9.2, loc="left", fontweight="bold"); ax2.spines[["top", "right"]].set_visible(False)
ax.text(1.5, 33.5, "Result of the pilot: 1,852 of 2,110 planned swaps accepted (88%)", fontsize=8.6, fontweight="bold")

# ---------- D: example ----------
ax.text(32, 32.2, "Example of the swaps on one peptide", fontsize=9.8, fontweight="bold")
def strip(x, y, res, mark=()):
    for i, r in enumerate(res):
        box(x + i * 3.8, y, 3.4, 3.8, r, ORG if i in mark else "#dcdcdc", fs=8.6, bold=i in mark)
ax.text(32, 27.6, "natural", fontsize=8.6, fontweight="bold"); strip(38, 26.3, ["T", "F", "G", "V", "L", "P", "E", "R"])
ax.text(32, 21.6, "after swaps", fontsize=8.6, fontweight="bold"); strip(38, 20.3, ["T", "F", "SAR", "MVA", "DLE", "HYP", "E", "R"], mark=(2, 3, 4, 5))
ax.text(32, 17.2, "each canonical residue is replaced only by an analogue of its own type:", fontsize=8, color="#444")
ax.text(32, 14.6, "Gly -> SAR   Val -> MVA   Leu -> MLE / 2ML / D-Leu   Pro -> HYP / FP9\nAla -> AIB / MAA   any -> D-form", fontsize=8, color="#444", va="top")
ax.text(32, 9.6, "backbone and pose stay those of the natural complex (no new binding mode)", fontsize=8, color="#444")

# ---------- E: output + where used ----------
box(32, 1.5, 35, 6.5, "Output pool: 8,947 complexes, 22,289 NCAA positions, 31 codes\nbalanced subset: 2,010 complexes, 25 codes x 234-260 positions", GLD, fs=8.6, bold=True, lw=1.6)
box(68.5, 1.5, 30, 6.5, "used for: decoder runs (A + B) and denoiser\nLoRA fine-tuning (B); split by protein family,\nheld out = 1,009 complexes", GRN, fs=8.2)
arr(67.5, 4.8, 68.4, 4.8)

box(68.5, 12, 30, 40, "", "#fdf1f1", ec="#e76f51")
ax.text(83.5, 49.5, "Limits to keep in mind", ha="center", fontsize=9.8, fontweight="bold", color="#9d2f1b")
ax.text(69.3, 33.5, "- chemistry is narrow: 4 families (side-chain modified,\n  alpha,alpha-disubstituted, N-alkyl, D-alpha); no beta,\n  gamma, ornithine, statine via this route\n- natural pose kept: the data teach recognition and\n  local geometry, not new binding modes\n- gates are geometric; Rosetta interface energy of\n  converted peptides is about +7 REU worse than natural\n- D-forms are over-represented (32% of complexes)\n- the NCAA identity is a label chosen by the generator's\n  rule, not by binding (so no signal for \"which NCAA\")\n- the 12 A pocket crop and one swap at a time limit\n  context", fontsize=7.9, va="center")
f.savefig(f"{D}/figures/fig74_synthetic_data.png", dpi=130, bbox_inches="tight"); plt.close(f); print("saved fig74")
