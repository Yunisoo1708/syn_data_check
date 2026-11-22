"""Figures for the natural-complex list and the backbone-swap idea: fig16 (the 31 complexes by peptide length, resolution and binding mode),
fig17 (backbone changes and what they do to the side-chain positions; a design judgement, not a measurement)."""
import os, csv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

TSV = "C:/Research/RFD4_fine_tune/data/natural_complexes/natural_complexes.tsv"
GROUP = {"helix": "helix", "polyproline II": "polyproline II (SH3-type)", "hairpin / turn": "turn, cyclic", "cyclic (disulfide hairpin)": "turn, cyclic"}
COL = {"helix": "#2a6f97", "polyproline II (SH3-type)": "#c9a227", "turn, cyclic": "#7b4fa3", "extended / strand / substrate": "#3a7d44"}


def grp(c):
    return GROUP.get(c, "extended / strand / substrate")


def fig16(OUT, C):
    rows = list(csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"))
    f = plt.figure(figsize=(15, 8.5))
    gs = f.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.25)
    a = f.add_subplot(gs[0, 0])
    seen = set()
    for r in rows:
        g = grp(r["cls"]); L = int(r["pep_len"]); y = float(r["res"])
        a.scatter(L, y, s=90, color=COL[g], edgecolor="#222", zorder=3, label=g if g not in seen else None); seen.add(g)
        a.annotate(r["pdb"], (L, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    a.set_xlabel("peptide length (residues)"); a.set_ylabel("resolution (A, lower = sharper)"); a.invert_yaxis()
    a.legend(frameon=False, fontsize=9, loc="upper right"); a.grid(alpha=0.25)
    a.set_title("31 natural peptide-protein complexes to start from\n(X-ray, 2.0 A or better, canonical residues only, one peptide per protein family)", fontsize=11.5, loc="left", fontweight="bold")
    b = f.add_subplot(gs[0, 1])
    cnt = {}
    for r in rows:
        cnt[grp(r["cls"])] = cnt.get(grp(r["cls"]), 0) + 1
    ks = sorted(cnt, key=lambda k: -cnt[k])
    b.barh(range(len(ks)), [cnt[k] for k in ks], color=[COL[k] for k in ks])
    for i, k in enumerate(ks):
        b.text(cnt[k] + 0.2, i, str(cnt[k]), va="center", fontsize=10)
    b.set_yticks(range(len(ks))); b.set_yticklabels(ks, fontsize=9); b.invert_yaxis(); b.set_xlabel("number of complexes")
    b.set_title("By binding mode\n(from the literature; confirm on coordinates)", fontsize=10.5, loc="left", fontweight="bold")
    f.savefig(os.path.join(OUT, "fig16_natural_complex_list.png"), bbox_inches="tight")
    plt.close(f)


def fig17(OUT, C):
    cols = ["side-chain position\nstays the same", "backbone gets\nlonger / shifts", "backbone NH donor\nis lost", "can we build it\nwith our pipeline now"]
    rows = [
        ("N-methyl amide", ["good", "no", "yes", "yes: NCFlow + RF3 (N-alkyl class)"]),
        ("alpha-methyl (Aib-type)", ["good", "no", "no", "yes: alpha,alpha class"]),
        ("amide isosteres (ester, thioamide,\nreduced, olefin)", ["good", "no", "partly", "not yet: residue library needed"]),
        ("beta3-homologation\n(same side chain, +1 CH2)", ["moves; fixed by\nalpha/beta pattern", "yes: +1 atom", "no", "yes: beta class (weakest)"]),
        ("gamma residues", ["moves", "yes: +2 atoms", "no", "yes: gamma class (few data)"]),
        ("peptoid (side chain on N)", ["shifts one bond", "no", "yes", "not yet"]),
        ("single D-residue", ["mirrored to the\nopposite face", "no", "no", "yes, but handedness is\nthe weak point"]),
        ("retro-inverso (reverse + all D)", ["good for helices\nand strands", "no", "no", "not yet"]),
        ("stapling / macrocycle", ["good (uses the\nback face)", "no", "no", "not yet"]),
    ]
    tone = {"good": "#d8f0dc", "no": "#d8f0dc", "yes": "#fff3cd", "partly": "#fff3cd"}

    def color(j, v):
        if j == 0:
            return "#d8f0dc" if v.startswith("good") else ("#f8d7da" if ("mirrored" in v or "shifts" in v) else "#fff3cd")
        if j == 1:
            return "#d8f0dc" if v == "no" else "#fff3cd"
        if j == 2:
            return "#d8f0dc" if v == "no" else "#fff3cd"
        return "#d8f0dc" if v.startswith("yes: ") and "weakest" not in v and "few" not in v and "handedness" not in v else ("#fff3cd" if v.startswith("yes") else "#f8d7da")
    f = plt.figure(figsize=(15, 8.5))
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.text(0.3, 9.2, "Changing only the backbone: which options keep the side-chain interactions with the protein?", fontsize=13.5, fontweight="bold")
    ax.text(0.3, 8.8, "Design judgement from the chemistry (not a measurement). Green = helps keep the interaction or is available now, amber = needs care, red = changes the interaction.", fontsize=9.5)
    x0 = [0.3, 3.9, 6.5, 9.0, 11.2]
    w = [3.5, 2.5, 2.4, 2.1, 3.6]
    ax.add_patch(FancyBboxPatch((x0[0], 7.85), w[0], 0.75, boxstyle="round,pad=0.02", fc="#e9ecef", ec="#495057"))
    ax.text(x0[0] + w[0] / 2, 8.22, "backbone change", ha="center", va="center", fontsize=9.5, fontweight="bold")
    xs = [3.9, 6.5, 9.0, 11.2]; ws = [2.5, 2.4, 2.1, 3.6]
    for j, t in enumerate(cols):
        ax.add_patch(FancyBboxPatch((xs[j], 7.85), ws[j], 0.75, boxstyle="round,pad=0.02", fc="#e9ecef", ec="#495057"))
        ax.text(xs[j] + ws[j] / 2, 8.22, t, ha="center", va="center", fontsize=8.8, fontweight="bold")
    for i, (name, vals) in enumerate(rows):
        y = 7.1 - i * 0.78
        ax.add_patch(FancyBboxPatch((x0[0], y), w[0], 0.68, boxstyle="round,pad=0.02", fc="#f8f9fa", ec="#adb5bd"))
        ax.text(x0[0] + 0.1, y + 0.34, name, va="center", fontsize=9)
        for j, v in enumerate(vals):
            ax.add_patch(FancyBboxPatch((xs[j], y), ws[j], 0.68, boxstyle="round,pad=0.02", fc=color(j if j < 3 else 3, v), ec="#adb5bd"))
            ax.text(xs[j] + ws[j] / 2, y + 0.34, v, ha="center", va="center", fontsize=8.3)
    ax.text(0.3, 0.45, "Reading it: the safest start is the top three rows (same atoms in the same places, only the amide changes). Length-changing backbones (beta, gamma) need the backbone re-routed\n"
                       "between side-chain anchors that stay fixed: that is the task for the denoiser (side-chain atoms as fixed motif, new backbone generated).", fontsize=9.3)
    f.savefig(os.path.join(OUT, "fig17_backbone_options.png"), bbox_inches="tight")
    plt.close(f)


def fig(OUT, C):
    fig16(OUT, C)
    fig17(OUT, C)
