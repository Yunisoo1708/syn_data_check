"""Figure for the conversion pilot (explained step by step)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def fig(OUT, C):
    f = plt.figure(figsize=(15, 9.2))
    gs = f.add_gridspec(2, 2, height_ratios=[1.0, 1.3], hspace=0.42, wspace=0.28)
    ax = f.add_subplot(gs[0, :])
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.9, 5)
    ax.axis("off")
    ax.set_title("Pilot: turn a real canonical peptide into a non-canonical one, one residue per round", loc="left", fontsize=12.5, fontweight="bold")
    rows = [("Start: real canonical peptide", []), ("Round 1: swap 1 residue", [2]), ("Round 2: swap 2 residues", [2, 5]), ("Round 3: swap 3 residues", [2, 5, 7])]
    colors = {2: "#e07a5f", 5: "#3d85c6", 7: "#6aa84f"}
    for r, (label, conv) in enumerate(rows):
        y = 4.2 - r * 1.1
        ax.text(0.1, y, label, fontsize=10.5, va="center", fontweight="bold")
        for k in range(9):
            x = 5.2 + k * 1.0
            fc = colors[k] if k in conv else "#ced4da"
            ax.add_patch(Circle((x, y), 0.36, fc=fc, ec="#495057", lw=1.2))
            if k < 8:
                ax.plot([x + 0.36, x + 0.64], [y, y], color="#495057", lw=1.4)
    ax.text(14.9, -0.2, "grey = canonical residue\ncolour = swapped for a\nnon-canonical residue", fontsize=10, va="top", ha="right", fontweight="bold")
    ax.text(0.1, -0.2, "How one swap is done: keep the real backbone atoms (N, CA, C) of that residue, build the new residue from its chemistry (RDKit),\nline it up on that backbone, pick the conformer that packs best in the real receptor, and accept it only if nothing clashes\nand its geometry is valid.", fontsize=9.5, va="top")

    a1 = f.add_subplot(gs[1, 0])
    xs = [0, 1, 2]
    w = 0.36
    b1 = a1.bar([x - w / 2 for x in xs], [0.08, 0.66, 1.56], w, color=C["ours"], label="our method (best-packing conformer)")
    b2 = a1.bar([x + w / 2 for x in xs], [0.08, 1.31, 4.21], w, color=C["grey"], label="a random conformer (no scoring)")
    for rects in (b1, b2):
        for r in rects:
            a1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.05, "{:.2f}".format(r.get_height()), ha="center", fontsize=9)
    a1.axhline(1.5, color=C["bad"], ls="--", lw=1.2)
    a1.text(-0.45, 1.62, "1.5 A = still a good match", color=C["bad"], fontsize=9, ha="left")
    a1.set_xticks(xs)
    a1.set_xticklabels(["small side chain\n(1-2 atoms, 142 residues)", "medium\n(3-5 atoms, 168)", "large\n(6+ atoms, 66)"], fontsize=9)
    a1.set_ylabel("error vs the real residue (A)\nlower = better")
    a1.set_ylim(0, 4.9)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a1.set_title("Test 1: rebuild a REAL non-canonical residue\nand measure how close it lands to the real one", fontsize=10.5, loc="left", fontweight="bold")

    a2 = f.add_subplot(gs[1, 1])
    b = a2.bar(["Round 1\n1 swap", "Round 2\n2 swaps", "Round 3\n3 swaps"], [78, 81, 87], color=[colors[2], colors[5], colors[7]], width=0.55)
    for r, v, n in zip(b, [78, 81, 87], [145, 98, 69]):
        a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, "{}%\n({} complexes)".format(v, n), ha="center", fontsize=9.5, fontweight="bold")
    a2.set_ylim(0, 112)
    a2.set_ylabel("% of swaps that gave a valid,\nclash-free residue")
    a2.set_title("Test 2: swap residues in real canonical complexes\nand check the result is chemically reasonable", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.02, 0.005, "Reading it: Test 1 shows the building step is accurate where the answer is known. Test 2 shows most swaps are acceptable, but acceptable does not prove the peptide still binds.", fontsize=9.5)
    f.savefig(os.path.join(OUT, "fig5_conversion_pilot.png"), bbox_inches="tight")
    plt.close(f)
