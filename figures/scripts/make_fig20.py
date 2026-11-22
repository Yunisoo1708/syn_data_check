"""Figure: RF3 repair test (RF3 + receptor template predictions; NCFlow redraws the residues that fail the strict test) and the relaxed-energy sanity test."""
import os, csv, json, statistics as S
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out/repair_summary.tsv"
N = "C:/Research/RFD4_fine_tune/data/natural_complexes"


def fig(OUT, C):
    rows = list(csv.DictReader(open(P), delimiter="\t"))
    fl = float
    f = plt.figure(figsize=(16, 9.5))
    gs = f.add_gridspec(2, 3, height_ratios=[0.7, 1.25], hspace=0.55, wspace=0.32)
    ax = f.add_subplot(gs[0, :]); ax.set_xlim(0, 16); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("RF3 repair test: let NCFlow redraw only the residues of an RF3 peptide that fail the strict test", loc="left", fontsize=13, fontweight="bold")
    for x, w, t, fc in ((0.1, 3.6, "32 RF3 + receptor-template\npredictions of real complexes\n(17 have failing residues)", "#dbe9f4"), (4.3, 3.7, "find each residue that fails\nthe strict test or touches\nthe protein (< 2.2 A)", "#fff3cd"),
                        (8.6, 3.8, "NCFlow redraws it: up to 3 rounds\nof 4 draws, mirror repair,\nkeep the first that passes", "#f8d7da"), (13.0, 2.9, "repeat once;\nscore the whole peptide", "#d8f0dc")):
        ax.add_patch(FancyBboxPatch((x, 0.4), w, 1.9, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#495057", lw=1.2))
        ax.text(x + w / 2, 1.35, t, ha="center", va="center", fontsize=9.2)
    for x0, x1 in ((3.7, 4.3), (8.0, 8.6), (12.4, 13.0)):
        ax.add_patch(FancyArrowPatch((x0, 1.35), (x1, 1.35), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))

    def pair(a, labels, before, after, title, ylab, ymax, fmt="{:.0f}"):
        a.bar([0, 1], [before, after], color=["#b23a48", "#3a7d44"], width=0.55)
        for i, v in enumerate([before, after]):
            a.text(i, v + ymax * 0.02, fmt.format(v), ha="center", fontsize=11, fontweight="bold")
        a.set_xticks([0, 1]); a.set_xticklabels(labels); a.set_ylim(0, ymax); a.set_ylabel(ylab); a.set_title(title, fontsize=10.5, loc="left", fontweight="bold")

    a = f.add_subplot(gs[1, 0])
    pair(a, ["RF3 as predicted", "after the repair"], sum(int(r["pass_before"]) for r in rows), sum(int(r["pass_after"]) for r in rows),
         "(a) Peptides that pass the whole\nstrict test (of 32)", "number of complexes", 38)
    b = f.add_subplot(gs[1, 1])
    fb, fa = sum(int(r["fail_before"]) for r in rows), sum(int(r["fail_after"]) for r in rows)
    cb, ca = sum(int(r["clash_before"]) for r in rows), sum(int(r["clash_after"]) for r in rows)
    w = 0.36
    b.bar([-w / 2, 1 - w / 2], [fb, cb], w, color="#b23a48", label="as predicted")
    b.bar([w / 2, 1 + w / 2], [fa, ca], w, color="#3a7d44", label="after the repair")
    for xx, v in ((-w / 2, fb), (w / 2, fa), (1 - w / 2, cb), (1 + w / 2, ca)):
        b.text(xx, v + 1, str(v), ha="center", fontsize=10, fontweight="bold")
    b.set_xticks([0, 1]); b.set_xticklabels(["residues failing the\nstrict test (of 573)", "atom pairs closer than\n2.2 A to the protein"]); b.set_ylim(0, 52)
    b.legend(frameon=False, fontsize=9); b.set_title("(b) What was wrong", fontsize=10.5, loc="left", fontweight="bold")
    c = f.add_subplot(gs[1, 2])
    vals = [100 * S.median(fl(r["contact_kept"]) for r in rows), 100 * S.median(fl(r["polar_kept"]) for r in rows if r["polar_kept"] != ""), 100 * S.median(fl(r["atom_ratio"]) for r in rows)]
    mins = [100 * min(fl(r["contact_kept"]) for r in rows), 100 * min(fl(r["polar_kept"]) for r in rows if r["polar_kept"] != ""), 100 * min(fl(r["atom_ratio"]) for r in rows)]
    c.bar(range(3), vals, color="#2a6f97", width=0.55, label="median")
    c.scatter(range(3), mins, color="#222", zorder=3, marker="v", s=50, label="worst complex")
    for i, v in enumerate(vals):
        c.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
    c.set_xticks(range(3)); c.set_xticklabels(["residue contacts\nkept (5 A)", "polar contacts\nkept (3.5 A)", "atom contacts\nvs before (4.5 A)"]); c.set_ylim(0, 125); c.set_ylabel("% of the RF3 interaction")
    c.legend(frameon=False, fontsize=8.5, loc="upper right"); c.set_title(f"(c) The pose and interface are kept\n(CA movement: median {S.median(fl(r['ca_move']) for r in rows):.2f} A, largest {max(fl(r['ca_move_max']) for r in rows):.1f} A)", fontsize=10.5, loc="left", fontweight="bold")
    att, fix, dr = sum(int(r["attempts"]) for r in rows), sum(int(r["fixed"]) for r in rows), sum(int(r["draws"]) for r in rows)
    f.text(0.02, 0.005, f"Result: {fix} of {att} failing residues were fixed with {dr} draws (about {dr / max(fix, 1):.0f} per fix, free GPUs); {sum(int(r['pass_after']) for r in rows)} of 32 peptides now pass the whole strict test, the interface is kept.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig20_rf3_repair.png"), bbox_inches="tight")
    plt.close(f)
