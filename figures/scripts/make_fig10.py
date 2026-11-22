"""Figure for the RF3 pose-substitution test (levels 1 and 2)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GROUPS = ["control\n(same residues)\n32 complexes", "1 residue\nswapped\n20", "2 residues\nswapped\n10", "3 residues\nswapped\n5"]
BASE_LT2 = [50, 55, 30, 20]; POSE_LT2 = [81, 85, 80, 80]
BASE_REC = [0.83, 0.83, 0.83, 0.84]; POSE_REC = [0.94, 0.94, 0.97, 0.97]
BASE_ST = [47, 40, 60, 80]; POSE_ST = [47, 50, 30, 0]
SW_PASS = ["", "75%", "75%", "67%"]


def fig(OUT, C):
    f = plt.figure(figsize=(15, 9))
    gs = f.add_gridspec(2, 3, height_ratios=[0.75, 1.3], hspace=0.62, wspace=0.34)
    ax = f.add_subplot(gs[0, :])
    ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("RF3 pose substitution: our residues in the real peptide's binding pose", loc="left", fontsize=12.5, fontweight="bold")

    def box(x, y, w, h, text, fc, ec="#495057"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.3)

    box(0.1, 1.2, 3.5, 1.9, "Real complex\nreceptor + real peptide\n(real binding pose)", "#dbe9f4")
    box(4.0, 1.2, 3.5, 1.9, "Level 1 (control)\nsame residues, both chains\nmarked as templates", "#d8f0dc", "#3a7d44")
    box(7.9, 1.2, 3.5, 1.9, "Level 2\n1, 2 or 3 residues swapped for\nnon-canonical ones, real\nbackbone kept as template", "#fff3cd", "#c9a227")
    box(11.8, 1.2, 3.1, 1.9, "RF3 refolds;\nwe score pose, contacts\nand strict chemistry", "#f8d7da", "#b23a48")
    for x0, x1 in ((3.6, 4.0), (7.5, 7.9), (11.4, 11.8)):
        ax.add_patch(FancyArrowPatch((x0, 2.15), (x1, 2.15), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.text(0.1, 0.45, "Baseline for every bar pair = RF3 with the receptor template only, same complexes. Swaps are the accepted ones from the round-by-round pilot. One H200 job, 68 predictions.", fontsize=9.5)

    xs = list(range(4)); w = 0.38

    def pair(a, base, pose, title, ylab, ymax, fmt):
        a.bar([x - w / 2 for x in xs], base, w, color=C["grey"], label="receptor template only")
        a.bar([x + w / 2 for x in xs], pose, w, color=C["ours"], label="pose transplant")
        for x, b, p in zip(xs, base, pose):
            a.text(x - w / 2, b + ymax * 0.02, fmt(b), ha="center", fontsize=8.5)
            a.text(x + w / 2, p + ymax * 0.02, fmt(p), ha="center", fontsize=8.5)
        a.set_xticks(xs); a.set_xticklabels(GROUPS, fontsize=8)
        a.set_ylim(0, ymax); a.set_ylabel(ylab)
        a.set_title(title, fontsize=10.5, loc="left", fontweight="bold")

    a1 = f.add_subplot(gs[1, 0]); pair(a1, BASE_LT2, POSE_LT2, "Peptide within 2 A of the real pose", "% of complexes", 125, lambda v: f"{v:.0f}")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    a2 = f.add_subplot(gs[1, 1]); pair(a2, BASE_REC, POSE_REC, "Interface found (median recall)", "recall of real interface residues", 1.15, lambda v: f"{v:.2f}")
    a3 = f.add_subplot(gs[1, 2]); pair(a3, BASE_ST, POSE_ST, "Whole peptide passes the strict chemistry test", "% of complexes", 105, lambda v: f"{v:.0f}")
    a3.axhline(55, color="#495057", ls="--", lw=1.1); a3.text(3.45, 57, "real PDB peptides: 55%", ha="right", fontsize=8.5)
    f.text(0.02, 0.005, "Result: the pose is kept (about 0.4-0.55 A from the real one) and the interface is found, but chemistry does not improve: swapped residues pass 67-75% and the whole-peptide pass falls with more swaps.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig10_pose_substitution.png"), bbox_inches="tight")
    plt.close(f)
