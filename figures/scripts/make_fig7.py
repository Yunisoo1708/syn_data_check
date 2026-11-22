"""Figure for the RF3 'peptide as template' test."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


def fig(OUT, C):
    f = plt.figure(figsize=(15, 8.6))
    gs = f.add_gridspec(2, 3, height_ratios=[0.8, 1.3], hspace=0.5, wspace=0.34)
    ax = f.add_subplot(gs[0, :])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("RF3 test: does giving RF3 the peptide's own structure as a template help?", loc="left", fontsize=12.5, fontweight="bold")

    def box(x, y, w, h, text, fc, ec="#495057"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    box(0.2, 1.4, 4.2, 1.7, "Input A (before)\nreceptor = template\npeptide structure given\nbut NOT used as template", "#dbe9f4")
    box(5.2, 1.4, 4.2, 1.7, "Input B (this test)\nreceptor = template\npeptide structure ALSO\nmarked as a template", "#fff3cd", "#c9a227")
    box(10.2, 1.4, 4.6, 1.7, "Problem: the peptide structure sits\n300 A from the receptor (so the real\npose does not leak), and RF3 reads\nthat distance as part of the template", "#f8d7da", "#b23a48")
    ax.add_patch(FancyArrowPatch((4.4, 2.25), (5.2, 2.25), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.add_patch(FancyArrowPatch((9.4, 2.25), (10.2, 2.25), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.text(0.2, 0.6, "32 real complexes, one H200 job each; same complexes and metrics as the earlier template test.", fontsize=9.5)

    labels = ["no\ntemplate", "receptor\ntemplate\n(A)", "receptor +\npeptide\ntemplate (B)"]
    cols = [C["bad"], C["good"], C["other"]]

    def bars(a, vals, title):
        b = a.bar(labels, vals, color=cols, width=0.6)
        for r, v in zip(b, vals):
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, "{}%".format(v), ha="center", fontsize=10, fontweight="bold")
        a.set_ylim(0, 100)
        a.set_ylabel("% of 32 complexes")
        a.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        a.tick_params(axis="x", labelsize=9)

    bars(f.add_subplot(gs[1, 0]), [0, 59, 0], "Peptide within 4 A of the real one")
    bars(f.add_subplot(gs[1, 1]), [9, 84, 53], "Peptide centre within 5 A\n(found the right site)")
    a3 = f.add_subplot(gs[1, 2])
    bars(a3, [19, 50, 25], "Predicted peptide passes the\nstrict chemistry test")
    a3.axhline(55, color="#495057", ls="--", lw=1.1)
    a3.text(2.45, 57, "real PDB peptides: 55%", ha="right", fontsize=9)
    f.text(0.02, 0.005, "Result: marking the far-away peptide as a template made things worse on every measure (probably because of the far-away placement; not tested separately).", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig7_rf3_peptide_template.png"), bbox_inches="tight")
    plt.close(f)
