"""Figure for the RF3 'peptide geometry hint without position' test."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def fig(OUT, C):
    f = plt.figure(figsize=(15, 8.8))
    gs = f.add_gridspec(2, 3, height_ratios=[0.8, 1.3], hspace=0.55, wspace=0.34)
    ax = f.add_subplot(gs[0, :])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("RF3 test: give RF3 the peptide's 3D shape, but not where it sits", loc="left", fontsize=12.5, fontweight="bold")

    def box(x, y, w, h, text, fc, ec="#495057"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.3)

    box(0.1, 1.2, 3.4, 1.9, "Baseline\nreceptor = template\npeptide: sequence only", "#dbe9f4")
    box(3.9, 1.2, 3.5, 1.9, "Test 1: RDKit shape\nreceptor = template\n+ our built peptide shape\n(each residue re-centred)", "#fff3cd", "#c9a227")
    box(7.8, 1.2, 3.5, 1.9, "Test 2: real shape\nreceptor = template\n+ the REAL peptide shape\n(upper bound, 15 complexes)", "#d8f0dc", "#3a7d44")
    box(11.7, 1.2, 3.2, 1.9, "Why it should be fair:\nRF3 centres every residue\nshape on its own origin,\nso no position leaks", "#f8d7da", "#b23a48")
    for x0, x1 in ((3.5, 3.9), (7.4, 7.8), (11.3, 11.7)):
        ax.add_patch(FancyArrowPatch((x0, 2.15), (x1, 2.15), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.text(0.1, 0.45, "Same 32 real complexes as before (Test 2 only on the 15 whose real peptide could be matched atom by atom). One H200 job, 47 predictions.", fontsize=9.5)

    cols = [C["grey"], C["other"], C["good"]]

    def panel(a, labels, vals, title, ylab, ymax, pct=False, ref=None):
        b = a.bar(labels, vals, color=cols[:len(vals)], width=0.6)
        for r, v in zip(b, vals):
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + ymax * 0.02, ("{:.0f}%".format(v) if pct else "{:.1f}".format(v)), ha="center", fontsize=10, fontweight="bold")
        a.set_ylim(0, ymax)
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        a.tick_params(axis="x", labelsize=9)
        if ref:
            a.axhline(ref, color="#495057", ls="--", lw=1.1)
            a.text(len(vals) - 0.5, ref + ymax * 0.02, "real PDB peptides: {}%".format(ref), ha="right", fontsize=9)

    l32 = ["baseline", "RDKit\nshape"]
    a1 = f.add_subplot(gs[1, 0])
    panel(a1, l32, [59, 62], "Peptide within 4 A of the real one\n(all 32 complexes)", "% of complexes", 100, True)
    a2 = f.add_subplot(gs[1, 1])
    panel(a2, ["baseline", "RDKit\nshape", "real\nshape"], [67, 73, 67], "Same measure on the 15 complexes\nwhere the real shape could be given", "% of complexes", 100, True)
    a3 = f.add_subplot(gs[1, 2])
    panel(a3, l32, [50, 38], "Predicted peptide passes the\nstrict chemistry test (32 complexes)", "% of complexes", 100, True, ref=55)
    f.text(0.02, 0.005, "Result: no real gain. Pose accuracy moves by one complex in 32 (noise), and the chemistry test gets slightly worse. Even the real shape does not help.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig8_rf3_shape_hint.png"), bbox_inches="tight")
    plt.close(f)
