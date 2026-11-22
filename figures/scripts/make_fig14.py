"""Figure: draw several samples and keep one that passes the strict check (pass@k), synthetic / real / RF3 sets."""
import os, json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

D = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out"
COL = {"ft2": "#2a6f97", "rl": "#3a7d44", "chi": "#7b4fa3"}
LAB = {"ft2": "ft2", "rl": "reward fine-tune", "chi": "reward + handedness input"}


def fig(OUT, C):
    pk = json.load(open(f"{D}/passk.json"))
    f = plt.figure(figsize=(15, 9))
    gs = f.add_gridspec(2, 3, height_ratios=[0.8, 1.3], hspace=0.5, wspace=0.3)
    ax = f.add_subplot(gs[0, :]); ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Sample-and-check: draw several versions of the residue, keep one that passes the strict check", loc="left", fontsize=13, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.1, 1.2), 2.6, 1.8, boxstyle="round,pad=0.02,rounding_size=0.05", fc="#dbe9f4", ec="#495057", lw=1.2))
    ax.text(1.4, 2.1, "hidden residue\nin a peptide", ha="center", va="center", fontsize=9.5)
    verdict = [("try 1", "#f8d7da", "fails\n(wrong hand)"), ("try 2", "#f8d7da", "fails\n(link too long)"), ("try 3", "#d8f0dc", "passes"), ("try 4", "#e9ecef", "not needed")]
    for i, (t, fc, v) in enumerate(verdict):
        x = 3.3 + i * 2.35
        ax.add_patch(FancyBboxPatch((x, 1.2), 2.0, 1.8, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#495057", lw=1.2))
        ax.text(x + 1.0, 2.5, t, ha="center", fontsize=9.5, fontweight="bold"); ax.text(x + 1.0, 1.8, v, ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((2.7, 2.1), (3.3, 2.1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.add_patch(FancyBboxPatch((12.8, 1.2), 2.1, 1.8, boxstyle="round,pad=0.02,rounding_size=0.05", fc="#fff3cd", ec="#c9a227", lw=1.2))
    ax.text(13.85, 2.1, "keep try 3\n(the check needs\nonly the residue's\ndefinition)", ha="center", va="center", fontsize=8.8)
    ax.add_patch(FancyArrowPatch((12.1, 2.1), (12.8, 2.1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))
    ax.text(0.1, 0.55, "The check uses the residue's definition (bond lengths, angles, handedness, link to neighbours), not the true answer, so it can be used when designing.", fontsize=9.5)

    ks = [1, 2, 3, 4]
    a = f.add_subplot(gs[1, 0:2])
    for t in ("ft2", "rl", "chi"):
        v = pk[f"{t}|syn_test"][:4]
        a.plot(ks, v, marker="o", color=COL[t], lw=2, label=LAB[t])
        for k, y in zip(ks, v):
            a.text(k, y + 2.0, f"{y:.0f}", ha="center", fontsize=8, color=COL[t])
    stereo = 125 / 180
    a.plot(ks, [100 * ((1 - stereo) + stereo * (1 - 0.5 ** k)) for k in ks], ls="--", color="#222", label="ceiling if handedness (a coin flip) were the only problem")
    a.set_xticks(ks); a.set_xlabel("number of samples drawn per residue"); a.set_ylim(30, 105); a.set_xlim(0.7, 4.4)
    a.set_ylabel("% of residues with at least one passing sample")
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    a.set_title("(a) Synthetic peptides (180 residues)", fontsize=10.5, loc="left", fontweight="bold")

    b = f.add_subplot(gs[1, 2])
    for s, ls, lab in (("real_test", "-", "real (89)"), ("rf3", ":", "RF3 complexes (39)")):
        v = pk[f"rl|{s}"][:4]
        b.plot(ks, v, marker="o", color=COL["rl"], ls=ls, lw=2, label=lab)
        for k, y in zip(ks, v):
            b.text(k, y + 2.0, f"{y:.0f}", ha="center", fontsize=8, color=COL["rl"])
    b.set_xticks(ks); b.set_xlabel("number of samples drawn per residue"); b.set_ylim(30, 105); b.set_xlim(0.7, 4.4)
    b.set_ylabel("% of residues with at least one passing sample"); b.legend(frameon=False, fontsize=8.5, loc="lower right")
    b.set_title("(b) Real residues and RF3 complexes\n(reward fine-tune)", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.02, 0.005, "Result: one sample passes about half the time; four samples pass about 81% (synthetic) to 93% (real). The curve flattens below the dashed ceiling because some residues fail for another reason in every sample.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig14_sample_and_check.png"), bbox_inches="tight")
    plt.close(f)
