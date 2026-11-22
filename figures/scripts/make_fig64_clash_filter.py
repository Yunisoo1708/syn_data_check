"""Best-of-N clash filter at generation time (decoder run 8): draw N designs per pocket, keep one without an atom pair < 2 A to the receptor. (a) share of pockets with a clash-free design among N draws; (b-d) quality of the kept designs (bonds, D handedness, CA distance) against all designs; (e) clash-free designs per pocket for G3, two seeds. Data: gen_test/gen_summary.json, gen_test/diag/best_of_n.json (script gen_test/best_of_n_filter.py)."""
import json, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json")); B = json.load(open(f"{D}/gen_test/diag/best_of_n.json"))
RUNS = [("G_ncw10_v10", "G (seed 1234)", "#2a9d8f", "-"), ("G2_pool_v10", "G2", "#219ebc", "-"), ("G3_bal_v10", "G3", "#6a4c93", "-"), ("control_v10", "control", "#999999", "-"), ("G_ncw10_v10_s777", "G (fresh seed 777)", "#2a9d8f", "--"), ("G3_bal_v10_s777", "G3 (fresh seed 777)", "#6a4c93", "--")]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.5, wspace=0.27)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Filter", "per pocket, draw N of the 8 designs; keep one with NO peptide-receptor atom pair\nunder 2 A (rule A, random among clash-free) or the clash-free one with the most\nsensible peptide bonds (rule B); N = 1, 2, 4; Monte Carlo over the draws", "#ffd6a5"),
       ("Data", "decoder run 8 with each denoiser, 11 real pockets x 8 designs = 88 designs per run;\nfirst batch seed 1234 (G, G2, G3, control), FRESH batch seed 777 (G, G3)\nto check that the numbers are not specific to one seed", "#ade8f4"),
       ("Costs", "no training; N designs are generated per pocket (about 8 min for 88 designs on the\nbox GPU, so 4 draws per pocket = 44 designs, about 4 min); the clash check is a\nsimple distance test on the output", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=7.0)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: keep one clash-free design out of N draws per pocket (no training)", fontsize=10.5, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 0])
for k, lab, col, ls in RUNS:
    xs = [1, 2, 4]; ys = [100 * B[k]["N"][f"{n}A"]["clash_free_share"] for n in xs]
    ax.plot(xs, ys, ls, color=col, marker="o", label=lab)
    ax.text(4.08, ys[-1], f"{ys[-1]:.0f}", fontsize=7.5, va="center", color=col)
ax.set_xticks([1, 2, 4]); ax.set_xlabel("N designs drawn per pocket"); ax.set_ylabel("% of pockets with a clash-free design"); ax.set_ylim(60, 103); ax.set_xlim(0.8, 4.6)
ax.set_title("(a) pockets with a clash-free design among N draws", fontsize=9.3, loc="left", fontweight="bold"); ax.legend(fontsize=7, frameon=False, loc="lower right")
def bars(ax, key, title, ylabel, mult=100, fmt="{:.0f}", unfiltered=None):
    n = len(RUNS); w = 0.27; cols = ["#bbbbbb", "#e76f51", "#2a9d8f"]
    for j, (lab, col) in enumerate([("all designs", cols[0]), ("kept, rule A (N=4)", cols[1]), ("kept, rule B (N=4)", cols[2])]):
        vals = []
        for k, *_ in RUNS:
            if j == 0: vals.append(unfiltered(k))
            else: vals.append(B[k]["N"]["4A" if j == 1 else "4B"][key] * mult)
        xs = np.arange(n) + (j - 1) * w; ax.bar(xs, vals, w * 0.95, color=col, edgecolor="#333", linewidth=0.5, label=lab)
        for x, v in zip(xs, vals): ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(n)); ax.set_xticklabels(["G", "G2", "G3", "control", "G s777", "G3 s777"], fontsize=8); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold"); ax.set_ylabel(ylabel)
ax = f.add_subplot(gs[1, 1]); bars(ax, "bond_ok", "(b) peptide bonds of sensible length (%) of the kept designs", "%", unfiltered=lambda k: 100 * B[k]["unfiltered"]["bond_ok"]); ax.set_ylim(0, 128); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3)
ax = f.add_subplot(gs[1, 2]); bars(ax, "chir_D", "(c) D-forms with the right handedness (%) of the kept designs", "%", unfiltered=lambda k: 100 * B[k]["unfiltered"]["chir_D"]); ax.set_ylim(0, 112)
ax = f.add_subplot(gs[2, 0]); bars(ax, "ca_rmsd_med", "(d) median CA distance to the native peptide (A)", "A", mult=1, fmt="{:.2f}", unfiltered=lambda k: float(np.median([r["ca_rmsd"] for r in S[k]["rows"]]))); ax.set_ylim(0, 7)
ax = f.add_subplot(gs[2, 1]); bars(ax, "contacts_rel_med", "(e) median contacts relative to the native peptide", "ratio", mult=1, fmt="{:.2f}", unfiltered=lambda k: float(np.median([r["contacts_rel"] for r in S[k]["rows"]]))); ax.set_ylim(0, 2)
ax = f.add_subplot(gs[2, 2]); names = sorted({r["name"] for r in S["G3_bal_v10"]["rows"]})
for j, (k, lab, col) in enumerate([("G3_bal_v10", "seed 1234", "#6a4c93"), ("G3_bal_v10_s777", "seed 777 (fresh)", "#b5a3d6")]):
    c = collections.Counter(); [c.update({r["name"]: 1}) for r in S[k]["rows"] if r["clash"] == 0]
    ax.bar(np.arange(len(names)) + (j - 0.5) * 0.4, [c.get(n, 0) for n in names], 0.38, color=col, edgecolor="#333", linewidth=0.5, label=lab)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split("_")[0] for n in names], rotation=60, fontsize=7); ax.set_ylim(0, 9); ax.axhline(8, color="#888", lw=0.5, ls=":")
ax.set_ylabel("clash-free designs of 8"); ax.set_title("(f) clash-free designs per pocket, G3 (8 drawn per pocket)", fontsize=9.3, loc="left", fontweight="bold"); ax.legend(fontsize=7, frameon=False, loc="lower right")
g3, g3b = B["G3_bal_v10"], B["G3_bal_v10_s777"]; g, gb = B["G_ncw10_v10"], B["G_ncw10_v10_s777"]
f.text(0.01, 0.005, "Result: with 4 draws per pocket a clash-free design exists for G3 in %.0f%% (seed 1234) and %.0f%% (fresh seed) of pockets (G: %.0f%% / %.0f%%); the kept designs keep bonds, handedness and pose; rule B also lifts bonds (G3 %.0f%% -> %.0f%%). 88 designs per run, unrelaxed." % (100 * g3["N"]["4A"]["clash_free_share"], 100 * g3b["N"]["4A"]["clash_free_share"], 100 * g["N"]["4A"]["clash_free_share"], 100 * gb["N"]["4A"]["clash_free_share"], 100 * g3["unfiltered"]["bond_ok"], 100 * g3["N"]["4B"]["bond_ok"]), fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig64_clash_filter.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig64")
