"""Figures that explain each test (method schematic + result). Run: python make_figures.py   -> figures/*.png"""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = RFD4_fine_tune/figures
os.makedirs(OUT, exist_ok=True)
C = {"ours": "#2a6f97", "other": "#c9a227", "bad": "#b23a48", "good": "#3a7d44", "grey": "#8d99ae", "light": "#e9ecef"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def box(ax, xy, w, h, text, fc="#e9ecef", ec="#495057", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, color="#495057"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))


def bars(ax, labels, vals, colors, title, ylabel, fmt="{:.0f}%", ylim=None, annot=None):
    b = ax.bar(labels, vals, color=colors, width=0.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(textwrap.fill(title, 34), fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel(textwrap.fill(ylabel, 30), fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelsize=8)
    return b


def grouped(ax, cats, series, colors, labels, title, ylabel, fmt="{:.0f}", ylim=None):
    n = len(series)
    w = 0.8 / n
    for i, (s, c, l) in enumerate(zip(series, colors, labels)):
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(cats))]
        b = ax.bar(xs, s, w * 0.95, color=c, label=l)
        for r, v in zip(b, s):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_title(textwrap.fill(title, 34), fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel(textwrap.fill(ylabel, 30), fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8.5)


# ---------------------------------------------------------------- 1. backbone tests
def fig_backbone():
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.4], hspace=0.5, wspace=0.42)
    # schematic row
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("Backbone test: can a residue with ANY backbone be rebuilt between its two real neighbours?", loc="left", fontsize=12, fontweight="bold")
    box(ax, (0.2, 1.0), 2.4, 1.1, "Real neighbour\nresidue i-1\n(kept fixed)", "#dbe9f4")
    box(ax, (5.8, 1.0), 2.4, 1.1, "Residue i\nREMOVED\n(whole backbone)", "#f8d7da", "#b23a48")
    box(ax, (11.4, 1.0), 2.4, 1.1, "Real neighbour\nresidue i+1\n(kept fixed)", "#dbe9f4")
    for x0, x1 in ((2.6, 5.8), (8.2, 11.4)):
        ax.plot([x0, x1], [1.55, 1.55], color="#495057", lw=1.2, ls="--")
    ax.text(4.2, 1.75, "5 pinned atoms:\nCa, C, O (i-1)", ha="center", fontsize=8.5)
    ax.text(9.8, 1.75, "N, Ca (i+1)", ha="center", fontsize=8.5)
    ax.text(7.0, 0.55, "Rebuild from the residue's chemistry (RDKit conformers, acetyl and N-methylamide caps pinned to the neighbours),\nkeep the best-packing one, compare with the real residue (Test A) or put a different residue there (Test B)", ha="center", fontsize=9)
    ax.text(7.0, 2.55, "Test A: rebuild the REAL residue  |  Test B: swap a canonical residue for a non-canonical one, neighbours unmoved", ha="center", fontsize=9.5, color="#2a6f97", fontweight="bold")
    # A
    ax1 = fig.add_subplot(gs[1, 0])
    cls = ["canon.\n(control)", "side-chain\nmodified", "alpha,\nalpha", "N-\nalkyl", "D-\nalpha", "beta", "gamma"]
    v = [0.23, 0.22, 0.25, 0.20, 0.22, 0.25, 0.70]
    col = [C["grey"], C["ours"], C["ours"], C["ours"], C["ours"], C["good"], C["good"]]
    bars(ax1, cls, v, col, "A. Backbone error when rebuilding the real residue", "median backbone RMSD (A)", "{:.2f}", (0, 0.9))
    ax1.text(0.02, 0.97, "beta = canonical accuracy;\ngamma only 6 cases", transform=ax1.transAxes, va="top", fontsize=8.5)
    # A2 geometry pass
    ax2 = fig.add_subplot(gs[1, 1])
    p = [98, 96, 96, 100, 94, 95, 100]
    bars(ax2, cls, p, col, "A. Rebuilt residue passes the strict geometry test", "% of cases", "{:.0f}%", (0, 118))
    # B
    ax3 = fig.add_subplot(gs[1, 2])
    cls2 = ["alpha,\nalpha", "beta", "D-\nalpha", "side-chain\nmodified", "N-\nalkyl"]
    acc = [91, 83, 79, 75, 22]
    col2 = [C["ours"], C["good"], C["ours"], C["ours"], C["bad"]]
    bars(ax3, cls2, acc, col2, "B. Swap into a real canonical gap, neighbours unmoved", "% accepted (valid geometry, no clash)", "{:.0f}%", (0, 118))
    ax3.text(0.98, 0.96, "N-alkyl: the N-substituent\nclashes -> neighbours must move", transform=ax3.transAxes, ha="right", va="top", fontsize=8.5, color=C["bad"])
    fig.savefig(os.path.join(OUT, "fig1_backbone_tests.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. NCFlow
def fig_ncflow():
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.4], hspace=0.5, wspace=0.42)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("NCFlow check: it places side-chain atoms on a FIXED backbone", loc="left", fontsize=12, fontweight="bold")
    box(ax, (0.2, 1.0), 3.0, 1.2, "Real complex\n(old residue's\nN, CA, C, O kept)", "#dbe9f4")
    box(ax, (4.6, 1.0), 3.0, 1.2, "NCFlow predicts all\nother atoms\n(4 samples, best pLDDT)", "#fff3cd", "#c9a227")
    box(ax, (9.0, 1.0), 4.8, 1.2, "Compare: real residue (reverse test)\nor 'is it reasonable?' (forward test)", "#d8f3dc", "#3a7d44")
    arrow(ax, (3.2, 1.6), (4.6, 1.6)); arrow(ax, (7.6, 1.6), (9.0, 1.6))
    ax.text(7.0, 0.35, "Its pool builder accepts only residues with an N-C-C(=O)-O alpha backbone, and the N, CA, C, O positions are copied from the parent", ha="center", fontsize=9, color=C["bad"])
    a1 = fig.add_subplot(gs[1, 0])
    grouped(a1, ["side-chain\nmodified", "alpha,\nalpha", "N-\nalkyl", "D-\nalpha", "ALL"], [[0.16, 0.08, 0.31, 0.30, 0.17], [0.89, 0.14, 0.84, 0.18, 0.52]],
            [C["other"], C["ours"]], ["NCFlow", "my RDKit placement"], "Reverse test: side-chain error vs the real residue", "median RMSD (A)  (lower = better)", "{:.2f}", (0, 1.1))
    a2 = fig.add_subplot(gs[1, 1])
    grouped(a2, ["alpha,\nalpha", "side-chain\nmod. (n=4)", "N-\nalkyl", "D-\nalpha", "ALL"], [[100, 50, 38, 0, 37], [100, 50, 95, 76, 86]],
            [C["other"], C["ours"]], ["NCFlow", "my RDKit placement"], "Forward test: residue on a real canonical position", "% accepted (valid geometry, no clash)", "{:.0f}%", (0, 125))
    a3 = fig.add_subplot(gs[1, 2])
    cls = ["alpha", "N-alkyl", "alpha,\nalpha", "beta", "gamma"]
    acc = [12, 5, 8, 0, 0]
    tot = [12, 5, 8, 6, 1]
    colors = [C["good"] if a == t else C["bad"] for a, t in zip(acc, tot)]
    b = a3.bar(cls, [100 * a / t for a, t in zip(acc, tot)], color=colors, width=0.62)
    for r, a, t in zip(b, acc, tot):
        a3.text(r.get_x() + r.get_width() / 2, r.get_height() + 2, f"{a} of {t}", ha="center", fontsize=9, fontweight="bold")
    a3.set_ylim(0, 125); a3.set_ylabel("% of the 32 residues accepted")
    a3.set_title("Which backbones NCFlow accepts", fontsize=10, loc="left", fontweight="bold")
    a3.tick_params(axis="x", labelsize=8.5)
    fig.savefig(os.path.join(OUT, "fig2_ncflow_check.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. complex pose
def fig_complex():
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.4], hspace=0.5, wspace=0.42)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("Complex test: which model places a non-canonical peptide on the real binding site?", loc="left", fontsize=12, fontweight="bold")
    box(ax, (0.2, 1.0), 3.0, 1.2, "Real PDB complex\nreceptor + peptide\nwith ncAA codes", "#dbe9f4")
    box(ax, (4.6, 1.0), 3.4, 1.2, "Boltz-2 / RF3 / DiffDock\n(+/- real receptor\nas template)", "#fff3cd", "#c9a227")
    box(ax, (9.4, 1.0), 4.4, 1.2, "Score: peptide CA RMSD < 4 A, peptide centre\nwithin 5 A of the real one, strict geometry\ntest on the predicted peptide", "#d8f3dc", "#3a7d44")
    arrow(ax, (3.2, 1.6), (4.6, 1.6)); arrow(ax, (8.0, 1.6), (9.4, 1.6))
    models = ["RF3\nno template", "Boltz-2\nno template", "RF3 +\ntemplate", "Boltz-2 +\ntemplate"]
    cols = [C["bad"], C["other"], C["good"], C["ours"]]
    a1 = fig.add_subplot(gs[1, 0])
    bars(a1, models, [0, 47, 59, 72], cols, "Peptide within 4 A of the real one", "% of 32 complexes", "{:.0f}%", (0, 100))
    a2 = fig.add_subplot(gs[1, 1])
    bars(a2, models, [9, 59, 84, 84], cols, "Peptide centre within 5 A (right site)", "% of 32 complexes", "{:.0f}%", (0, 100))
    a3 = fig.add_subplot(gs[1, 2])
    bars(a3, models, [19, 9, 50, 19], cols, "Predicted peptide passes the strict chemistry test", "% of 32 complexes", "{:.0f}%", (0, 100))
    a3.axhline(55, color="#495057", ls="--", lw=1.1)
    a3.text(3.4, 57, "real PDB peptides: 55%", ha="right", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig3_complex_pose.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. autoencoder
def fig_autoencoder():
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    a0 = fig.add_subplot(gs[0, 0]); a0.axis("off"); a0.set_xlim(0, 10); a0.set_ylim(0, 10)
    a0.set_title("Autoencoder test on REAL PDB peptides", loc="left", fontsize=12, fontweight="bold")
    box(a0, (0.3, 7.2), 9.2, 1.8, "Real peptide coordinates\n(residue names hidden)", "#dbe9f4")
    box(a0, (0.3, 4.3), 9.2, 1.8, "Frozen encoder + trained decoder\nnames every residue (176 classes)", "#fff3cd", "#c9a227")
    box(a0, (0.3, 1.4), 9.2, 1.8, "Compare with the real names\n2,652 positions, 189 peptides, 32 residues", "#d8f3dc", "#3a7d44")
    arrow(a0, (4.9, 7.2), (4.9, 6.1)); arrow(a0, (4.9, 4.3), (4.9, 3.2))
    a1 = fig.add_subplot(gs[0, 1])
    grouped(a1, ["all positions", "canonical", "your 32\nnon-canonical"], [[86.2, 98.5, 25.1], [96.6, 98.1, 90.1], [98.5, 98.4, 99.6]],
            [C["grey"], C["other"], C["good"]], ["untrained new classes", "run 1 (20 epochs)", "run 2 (90 epochs)"], "Recovery of the real residue names", "% correct", "{:.1f}", (0, 118))
    a2 = fig.add_subplot(gs[0, 2])
    ep = [0, 8, 16, 24, 32, 40, 52, 65, 69]
    val = [0.223, 0.0695, 0.0501, 0.0382, 0.0297, 0.0260, 0.0188, 0.0139, 0.0131]
    a2.plot([e + 20 for e in ep], val, "-o", color=C["ours"], lw=1.8, ms=4)
    a2.set_yscale("log"); a2.set_xlabel("epoch (run 1 = 0-19, run 2 = 20-89)"); a2.set_ylabel("validation loss (log scale)")
    a2.axvline(20, color=C["grey"], ls="--", lw=1)
    a2.text(21, 0.2, "run 2 starts\n(loss 0.39)", fontsize=8.5, va="top")
    a2.set_title("Training: loss still falling, no overfit", fontsize=10.5, loc="left", fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig4_autoencoder.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. conversion pilot
def fig_pilot():
    import make_fig5
    make_fig5.fig(OUT, C)


# ---------------------------------------------------------------- 6. Boltz-2 variants
def fig_variants():
    fig = plt.figure(figsize=(14, 5.8))
    gs = fig.add_gridspec(1, 3, wspace=0.3, width_ratios=[1.0, 1.1, 1.1])
    a0 = fig.add_subplot(gs[0, 0]); a0.axis("off"); a0.set_xlim(0, 10); a0.set_ylim(0, 10)
    a0.set_title("Can Boltz-2 be improved? (72 complexes)", loc="left", fontsize=12, fontweight="bold")
    items = [("baseline: no alignment, 1 sample", "#e9ecef"), ("5 samples, pick by confidence", "#fff3cd"), ("more steps + steering", "#fff3cd"), ("cyclic/covalent bonds (18)", "#fff3cd"), ("real receptor as template", "#d8f3dc")]
    for i, (t, f) in enumerate(items):
        box(a0, (0.3, 8.0 - i * 1.8), 9.2, 1.3, t, f, fs=9)
    a1 = fig.add_subplot(gs[0, 1])
    lab = ["baseline", "5 samples\n(top conf.)", "5 samples\n(oracle)", "template"]
    bars(a1, lab, [32, 31, 42, 57], [C["grey"], C["other"], "#e6d38a", C["good"]], "Peptide within 4 A of the real one", "% of 72 complexes", "{:.0f}%", (0, 80))
    a2 = fig.add_subplot(gs[0, 2])
    grouped(a2, ["peptide < 4 A", "centre < 5 A"], [[32, 58], [57, 86]], [C["grey"], C["good"]], ["baseline", "with real receptor template"],
            "Only the receptor template helps clearly", "% of 72 complexes", "{:.0f}%", (0, 100))
    fig.text(0.02, 0.01, "More recycling/steps + steering: 32% vs 34% baseline (65 complexes, no gain). Covalent bonds: median peptide error 12.0 -> 9.2 A but success count unchanged (18 complexes).", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig6_boltz2_variants.png"), bbox_inches="tight")
    plt.close(fig)


def fig_rf3p():
    import make_fig7
    make_fig7.fig(OUT, C)


def fig_rf3g():
    import make_fig8
    make_fig8.fig(OUT, C)


def fig_ncf9():
    import make_fig9
    make_fig9.fig(OUT, C)


def fig_pose10():
    import make_fig10
    make_fig10.fig(OUT, C)


def fig_rl11():
    import make_fig11
    make_fig11.fig(OUT, C)


def fig_ncf12():
    import make_fig12
    make_fig12.fig(OUT, C)


def fig_hand13():
    import make_fig13
    make_fig13.fig(OUT, C)


def fig_pk14():
    import make_fig14
    make_fig14.fig(OUT, C)


def fig_cv15():
    import make_fig15
    make_fig15.fig(OUT, C)


def fig_nat16():
    import make_fig16
    make_fig16.fig(OUT, C)


def fig_map18():
    import make_fig18
    make_fig18.fig(OUT, C)


def fig_prot19():
    import make_fig19
    make_fig19.fig(OUT, C)


for f in (fig_backbone, fig_ncflow, fig_complex, fig_autoencoder, fig_pilot, fig_variants, fig_rf3p, fig_rf3g, fig_ncf9, fig_pose10, fig_rl11, fig_ncf12, fig_hand13, fig_pk14, fig_cv15, fig_nat16, fig_map18, fig_prot19):
    f()
print("figures written to", OUT, sorted(os.listdir(OUT)))
