"""Figure for the NCFlow raw / fine-tune backbone-substitution test."""
import os, json, statistics as S
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SRC = "C:/Research/ncflow_ref/ncf_bb_ft_out.json"
FAMS = [("canonical", ["canonical"]), ("alpha,\nmodified", ["alpha, side-chain modified", "alpha,alpha-disubstituted"]),
        ("D-alpha", ["D-alpha"]), ("N-alkyl", ["N-alkyl"]), ("beta,\ngamma", ["beta", "gamma"])]


def stats(R, tag, fams, split="test"):
    r = [x for x in R if x["tag"] == tag and x["split"] == split and x["family"] in fams and x.get("rmsd_all") is not None]
    if not r:
        return 0, 0, 0
    return len(r), S.median(x["rmsd_bb"] for x in r), 100 * sum(x["bad_bonds"] > 0 for x in r) / len(r)


def fig(OUT, C):
    d = json.load(open(SRC))
    R, log = d["results"], d["train_log"]
    f = plt.figure(figsize=(15, 9))
    gs = f.add_gridspec(2, 3, height_ratios=[0.75, 1.3], hspace=0.6, wspace=0.34)
    ax = f.add_subplot(gs[0, :])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("NCFlow test: can it generate a whole residue, backbone included, between two real neighbours?", loc="left", fontsize=12.5, fontweight="bold")

    def box(x, y, w, h, text, fc, ec="#495057"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.3)

    box(0.1, 1.2, 3.5, 1.9, "Task\nremove one real residue,\nkeep its two neighbours and\nthe receptor; NCFlow must\ngenerate ALL its atoms", "#dbe9f4")
    box(4.0, 1.2, 3.4, 1.9, "Raw model\nreleased checkpoint,\nbackbone atoms set free\n(no training by us)", "#f8d7da", "#b23a48")
    box(7.8, 1.2, 3.5, 1.9, "Fine-tuned\n1500 steps on 251 of our cases\n(about 4 minutes on an H200),\nsplit by PDB entry", "#fff3cd", "#c9a227")
    box(11.7, 1.2, 3.2, 1.9, "Scored on held-out\nentries (89 cases) vs the\nreal residue: error and\nbad bond lengths", "#d8f0dc", "#3a7d44")
    for x0, x1 in ((3.6, 4.0), (7.4, 7.8), (11.3, 11.7)):
        ax.add_patch(FancyArrowPatch((x0, 2.15), (x1, 2.15), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.text(0.1, 0.45, "Held-out = whole PDB entries not seen in training. Beta and gamma have few cases (12 held-out), so those bars are rough.", fontsize=9.5)

    xs = range(len(FAMS))
    w = 0.38
    rows = [(stats(R, "raw_base", fm), stats(R, "finetuned_base", fm)) for _, fm in FAMS]
    a1 = f.add_subplot(gs[1, 0])
    a1.bar([x - w / 2 for x in xs], [r[0][1] for r in rows], w, color=C["bad"], label="raw")
    a1.bar([x + w / 2 for x in xs], [r[1][1] for r in rows], w, color=C["good"], label="fine-tuned")
    for x, r in zip(xs, rows):
        a1.text(x - w / 2, r[0][1] + 0.02, "{:.2f}".format(r[0][1]), ha="center", fontsize=8.5)
        a1.text(x + w / 2, r[1][1] + 0.02, "{:.2f}".format(r[1][1]), ha="center", fontsize=8.5)
    a1.set_xticks(list(xs))
    a1.set_xticklabels([n for n, _ in FAMS], fontsize=8)
    a1.set_ylabel("backbone error vs real (A)\nlower = better")
    a1.set_ylim(0, 1.15)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a1.set_title("Backbone error, held-out cases", fontsize=10.5, loc="left", fontweight="bold")

    a2 = f.add_subplot(gs[1, 1])
    a2.bar([x - w / 2 for x in xs], [r[0][2] for r in rows], w, color=C["bad"], label="raw")
    a2.bar([x + w / 2 for x in xs], [r[1][2] for r in rows], w, color=C["good"], label="fine-tuned")
    for x, r in zip(xs, rows):
        a2.text(x - w / 2, r[0][2] + 1.5, "{:.0f}".format(r[0][2]), ha="center", fontsize=8.5)
        a2.text(x + w / 2, r[1][2] + 1.5, "{:.0f}".format(r[1][2]), ha="center", fontsize=8.5)
    a2.set_xticks(list(xs))
    a2.set_xticklabels([n for n, _ in FAMS], fontsize=8)
    a2.set_ylabel("% of cases with at least\none bad bond length")
    a2.set_ylim(0, 80)
    a2.set_title("Chemically broken residues, held-out cases", fontsize=10.5, loc="left", fontweight="bold")

    a3 = f.add_subplot(gs[1, 2])
    st = [s for s, _ in log]
    a3.plot(st, [v for _, v in log], color=C["grey"], lw=1, marker="o", ms=3)
    a3.set_xlabel("training step")
    a3.set_ylabel("training loss (noisy)")
    a3.set_title("Training loss: halves, still noisy", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.02, 0.005, "Result: fine-tuning cuts backbone error about 2.4x and removes nearly all broken bond lengths, including for beta; N-alkyl stays the weakest.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig9_ncflow_finetune.png"), bbox_inches="tight")
    plt.close(f)
