"""Figure: handedness (L / D) of NCFlow-generated residues, with and without the handedness input; coin-flip level; best-of-4 by check."""
import os, json, csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

D = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out"
ORDER = ["raw", "ft1", "ft2", "rl", "chi"]
LABEL = {"raw": "raw", "ft1": "ft1", "ft2": "ft2", "rl": "reward", "chi": "reward + handedness input"}
COL = {"raw": "#b23a48", "ft1": "#c9a227", "ft2": "#2a6f97", "rl": "#3a7d44", "chi": "#7b4fa3"}
SETS = [("real_test", "real residues"), ("rf3", "RF3 complexes"), ("syn_test", "synthetic peptides")]


def fig(OUT, C):
    h = json.load(open(f"{D}/handed.json"))
    diag = json.load(open(f"{D}/handed_diag.json")) if os.path.exists(f"{D}/handed_diag.json") else {}
    f = plt.figure(figsize=(15, 9.5))
    gs = f.add_gridspec(2, 3, height_ratios=[0.8, 1.3], hspace=0.55, wspace=0.32)
    ax = f.add_subplot(gs[0, :]); ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Handedness: can NCFlow draw the right mirror image of a residue?", loc="left", fontsize=13, fontweight="bold")
    for cx, flip, lab in ((1.6, 1, "L form"), (4.6, -1, "D form (mirror image)")):
        ax.add_patch(Circle((cx, 2.2), 0.28, fc="#8d99ae", ec="#222"))
        for dx, dy, c in ((-0.9 * flip, -0.6, "#2a6f97"), (0.9 * flip, -0.6, "#b23a48"), (0.0, 0.95, "#3a7d44")):
            ax.plot([cx, cx + dx], [2.2, 2.2 + dy], color="#222", lw=1.4)
            ax.add_patch(Circle((cx + dx, 2.2 + dy), 0.22, fc=c, ec="#222"))
        ax.text(cx, 1.0, lab, ha="center", fontsize=9.5, fontweight="bold")
    ax.plot([3.1, 3.1], [1.0, 3.5], color="#777", ls=":")
    ax.text(0.2, 3.65, "Same atoms, same bonds, opposite handedness.", fontsize=10)
    ax.text(6.4, 3.0, "Reward fine-tune: removes bond-length and bond-angle failures.\nIt cannot teach handedness by itself.", fontsize=10)
    ax.text(6.4, 1.9, "Test here: for residues that have a stereocentre, is the redrawn residue\nthe right mirror image (the strict test's handedness check)?", fontsize=10)
    ax.text(6.4, 0.8, "A model that does not know the answer is right half of the time (50%).", fontsize=10, fontweight="bold", color="#b23a48")

    b = f.add_subplot(gs[1, 0])
    n = len(ORDER); w = 0.8 / n
    for k, t in enumerate(ORDER):
        vals = [h[f"{t}|{s}"]["wrong_top"] for s, _ in SETS]
        xs = [i - 0.4 + w * (k + 0.5) for i in range(len(SETS))]
        b.bar(xs, vals, w, color=COL[t], label=LABEL[t])
        for x, v in zip(xs, vals):
            b.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    b.axhline(50, color="#222", ls="--", lw=1.3); b.text(-0.45, 51.5, "coin flip", ha="left", fontsize=9)
    b.set_xticks(range(len(SETS))); b.set_xticklabels([f"{l}\n({h['raw|' + s]['n']} with\nstereocentre)" for s, l in SETS], fontsize=8.5)
    b.set_ylim(0, 100); b.set_ylabel("% redrawn with the WRONG handedness"); b.legend(frameon=False, fontsize=7.5, loc="upper left")
    b.set_title("(a) Wrong handedness, by model", fontsize=10.5, loc="left", fontweight="bold")

    c = f.add_subplot(gs[1, 1])
    names = [("chi|syn_test", "input as given"), ("chizero|syn_test", "input set\nto zero"), ("chiflip|syn_test", "input flipped\n(mirror signal)")]
    vals = []
    for k, _ in names:
        src = h if k == "chi|syn_test" else diag
        vals.append(src[k]["wrong_any_sample"] if k in src else None)
    c.bar(range(3), [v if v is not None else 0 for v in vals], color=["#7b4fa3", "#adb5bd", "#e07a5f"], width=0.6)
    for i, v in enumerate(vals):
        c.text(i, (v or 0) + 1.5, f"{v:.0f}" if v is not None else "pending", ha="center", fontsize=9)
    c.axhline(50, color="#222", ls="--", lw=1.3)
    c.set_xticks(range(3)); c.set_xticklabels([n_ for _, n_ in names], fontsize=8.5)
    c.set_ylim(0, 100); c.set_ylabel("% wrong handedness (all samples)")
    c.set_title("(b) Does the model use the input?\n(synthetic peptides, handedness model)", fontsize=10.5, loc="left", fontweight="bold")

    d = f.add_subplot(gs[1, 2])
    top, any4 = {}, {}
    for t in ORDER:
        rows = [r for r in csv.DictReader(open(f"{D}/strict_{t}.tsv"), delimiter="\t") if r.get("g_checked") == "1" and r["set"] == "syn_test"]
        top[t] = 100 * sum(r["g_strict"] == "1" for r in rows) / len(rows); any4[t] = 100 * sum(r["any_sample_strict"] == "1" for r in rows) / len(rows)
    w2 = 0.38
    d.bar([i - w2 / 2 for i in range(n)], [top[t] for t in ORDER], w2, color="#adb5bd", label="sample NCFlow ranks first")
    d.bar([i + w2 / 2 for i in range(n)], [any4[t] for t in ORDER], w2, color="#3a7d44", label="best of 4 picked by the strict check")
    for i, t in enumerate(ORDER):
        d.text(i - w2 / 2, top[t] + 1.5, f"{top[t]:.0f}", ha="center", fontsize=8); d.text(i + w2 / 2, any4[t] + 1.5, f"{any4[t]:.0f}", ha="center", fontsize=8)
    d.set_xticks(range(n)); d.set_xticklabels(["raw", "ft1", "ft2", "reward", "handed-\nness input"], fontsize=8.5)
    d.set_ylim(0, 125); d.set_ylabel("% passing everything (synthetic)"); d.legend(frameon=False, fontsize=8, loc="upper left")
    d.set_title("(c) Checking the samples fixes more than training", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.02, 0.005, "Result: on synthetic peptides every model is at coin-flip handedness (about 50% wrong), with or without the new input; picking the best of 4 samples with the check is what works.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig13_handedness.png"), bbox_inches="tight")
    plt.close(f)
