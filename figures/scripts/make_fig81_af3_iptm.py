"""Fig 81: AF3 ipTM as a plausibility filter (Results.MD 83). Data: af3_filter/collected.json and summary.json (analyze_af3.py)."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
A = "C:/Research/RFD4_fine_tune/af3_filter"; R = json.load(open(f"{A}/collected.json")); S = json.load(open(f"{A}/summary.json"))
GR = [("native", "native", "#2a9d8f"), ("ala", "NCAA\n->ALA", "#8ab17d"), ("rnd", "random\nNCAAs", "#e9c46a"), ("decoy", "other\nreceptor", "#bbbbbb"), ("shuf", "shuffled", "#777777"), ("control", "control", "#999999"), ("G3", "G3", "#6a4c93"), ("S1", "S1", "#e76f51")]
f = plt.figure(figsize=(19, 14.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.26, 1, 0.7], hspace=0.62, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10); s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, "Can AF3 ipTM act as a filter for how plausible a complex is?  310 AF3 runs (single sequence, native receptor given as template), 10 GPCR-like pockets, peptide = chain B with the NCAAs as CCD residues.\n"
       "Groups: native peptide; native with NCAAs set to ALA; native with random NCAAs; native peptide on another pocket's receptor; shuffled peptide; and the free-choice designs of the three denoisers (80 each, Results.MD 81).\n"
       "Questions: (a,b) does ipTM separate real from fake?   (c) does it agree with the designed pose?   (d) does keeping the high-ipTM designs of each pocket give better designs by the Rosetta / contact scores?", ha="center", va="center", fontsize=8.8)
s.set_title("AF3 ipTM as a filter", fontsize=10.5, loc="left", fontweight="bold")
# (a)
ax = f.add_subplot(gs[1, 0]); rng = np.random.default_rng(0)
for j, (g, lab, c) in enumerate(GR):
    v = [r["iptm"] for r in R if r["group"] == g]
    ax.boxplot([v], positions=[j], widths=0.55, showfliers=False, patch_artist=True, boxprops=dict(facecolor=c, alpha=0.55), medianprops=dict(color="k"))
    ax.scatter(j + rng.uniform(-0.17, 0.17, len(v)), v, s=7, color="#222", alpha=0.5, zorder=3)
ax.set_xticks(range(len(GR))); ax.set_xticklabels([l for _, l, _ in GR], fontsize=7.6); ax.axhline(0.7, color="#c1121f", ls="--", lw=0.8)
ax.set_ylabel("AF3 ipTM"); ax.set_ylim(0.2, 1.02); ax.set_title("(a) ipTM by group (each dot = one run)", fontsize=9.4, loc="left", fontweight="bold")
au = S["auc_native_vs"]
ax.text(0.02, 0.02, "AUC, native vs: shuffled %.2f | random NCAAs %.2f\nNCAA->ALA %.2f | other receptor %.2f" % (au["shuf"], au["rnd"], au["ala"], au["decoy"]), transform=ax.transAxes, fontsize=7.6, va="bottom")
# (b)
ax = f.add_subplot(gs[1, 1]); M = S["per_pocket_mean_iptm"]; ex = list(M["native"]); x = np.arange(len(ex))
ax.plot(x, [M["native"][e] for e in ex], "o-", color="#2a9d8f", label="native")
ax.plot(x, [M["shuf"][e] for e in ex], "s-", color="#777", label="shuffled")
ax.plot(x, [np.mean([M[g][e] for g in ("control", "G3", "S1")]) for e in ex], "D-", color="#6a4c93", label="designs (3 denoisers)")
ax.plot(x, [M["rnd"][e] for e in ex], "^--", color="#e9c46a", label="native + random NCAAs")
ax.set_xticks(x); ax.set_xticklabels([e[:4] for e in ex], rotation=45, fontsize=8); ax.set_ylabel("mean ipTM per pocket"); ax.set_ylim(0.0, 1.0)
ax.legend(fontsize=7.6, frameon=False, loc="lower left", ncol=2)
d1, d2, d3 = S["design_minus_native"], S["design_minus_shuf"], S["design_minus_rnd"]
ax.text(0.98, 0.14, "designs - native: %.2f / %.2f / %.2f\ndesigns - shuffled: +%.2f / +%.2f / +%.2f\ndesigns - random NCAAs: %.2f / %.2f / %.2f\n(control / G3 / S1)" % (d1["control"], d1["G3"], d1["S1"], d2["control"], d2["G3"], d2["S1"], d3["control"], d3["G3"], d3["S1"]), transform=ax.transAxes, fontsize=7.6, ha="right", va="bottom")
ax.set_title("(b) the same ten pockets", fontsize=9.4, loc="left", fontweight="bold")
# (c)
ax = f.add_subplot(gs[1, 2])
for g, c in {"native": "#2a9d8f", "control": "#999999", "G3": "#6a4c93", "S1": "#e76f51"}.items():
    v = [r for r in R if r["group"] == g and r.get("pep_rmsd") is not None]
    ax.scatter([r["pep_rmsd"] for r in v], [r["iptm"] for r in v], s=22 if g != "native" else 60, color=c, alpha=0.65 if g != "native" else 0.95, edgecolor="#333", lw=0.3, label=g if g != "native" else "native peptide vs its own pose")
ax.axvline(3, color="#c1121f", ls="--", lw=0.8); ax.axhline(0.7, color="#c1121f", ls="--", lw=0.8)
ax.set_xlabel("distance of AF3 peptide to the designed pose (A, CA RMSD)"); ax.set_ylabel("AF3 ipTM"); ax.legend(fontsize=7.4, frameon=False, loc="lower right")
sp = S["spearman_within_pocket"]; ax.text(0.97, 0.97, "within-pocket Spearman, ipTM vs pose RMSD: %.2f (p < 0.001)" % sp["rmsd"]["rho"], transform=ax.transAxes, fontsize=7.6, ha="right", va="top")
ax.set_title("(c) does AF3 reproduce the designed pose?", fontsize=9.4, loc="left", fontweight="bold")
# (d) four mini panels
sub = gs[2, :].subgridspec(1, 4, wspace=0.38)
keys = [("e_slot", "interaction energy of the\nNCAA slots (REU, lower = better)"), ("e_int", "E_int of the whole peptide\n(REU, lower = better)"), ("contacts", "receptor contacts\nof the NCAA slot"), ("rmsd", "AF3 pose vs designed pose\n(A, lower = better)")]
for i, (k, lab) in enumerate(keys):
    ax = f.add_subplot(sub[0, i]); xs = [0.25, 0.5]
    for x0, fr in zip(xs, ("0.25", "0.5")):
        d = S["filter"][fr][k]
        ax.fill_between([x0 - 0.09, x0 + 0.09], d["rand_lo"], d["rand_hi"], color="#ddd", zorder=1)
        ax.plot([x0 - 0.09, x0 + 0.09], [d["all"]] * 2, color="#333", lw=1.8, zorder=2); ax.scatter([x0], [d["kept"]], s=70, color="#e76f51", zorder=3, edgecolor="#333")
    ax.set_xlim(0.1, 0.65); ax.set_xticks(xs); ax.set_xticklabels(["top 25%\n(60 designs)", "top 50%\n(120 designs)"], fontsize=8); ax.set_title(lab, fontsize=8.6, loc="left")
    if i == 0: ax.text(0.0, 1.3, "(d) keep the designs with the highest ipTM within each pocket.  grey = range of 2,000 random keeps of the same size, black = all 240 designs, orange = kept", transform=ax.transAxes, fontsize=9.4, fontweight="bold")
f.text(0.01, 0.03, "Reading: ipTM separates a real peptide from a scrambled one very well (AUC 0.99) but only moderately from the same peptide with random NCAAs (0.78),\n"
       "with ALA (0.76) or on another receptor (0.76): with the native receptor as template it is saturated for peptides that fit the pocket at all.\n"
       "Our designs score 0.64-0.67 against 0.88 for the native peptides, 0.57 for shuffled and 0.73 for native-with-random-NCAAs (the last keeps the native context, so it is not like-for-like).\n"
       "Only 28-40% of the designs reach ipTM 0.7 (native 100%, shuffled 15%). As a filter among the designs it is weak: it agrees with the AF3-to-design pose (rho -0.26) and a little with\n"
       "the slot energy (rho -0.19), not with whole-peptide E_int (-0.08, n.s.) or contacts (0.04). The kept 25% have a slot energy -3.7 against -0.9 and a pose 0.7 A closer, but both sit at the\n"
       "edge of the random range; E_int does not improve. S1 is over-represented in the kept set (47% against 33%).", fontsize=8.6, fontweight="bold", va="top")
f.savefig("C:/Research/RFD4_fine_tune/figures/fig81_af3_iptm_filter.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved")
