"""Figure: the interaction-preserving rewrite protocol (gates) and how often the current generator would meet it (conversion run of section 22)."""
import os, csv, statistics as S, collections
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out"
ARMS = ["top1", "check", "repair"]
LAB = {"top1": "top sample\n(no check)", "check": "check +\nretries", "repair": "check + retries\n+ mirror repair"}
COL = {"top1": "#b23a48", "check": "#2a6f97", "repair": "#3a7d44"}


def fig(OUT, C):
    rows = list(csv.DictReader(open(f"{D}/iter3_arms.tsv"), delimiter="\t"))
    fl = float
    f = plt.figure(figsize=(16, 10.5))
    gs = f.add_gridspec(2, 3, height_ratios=[0.95, 1.2], hspace=0.5, wspace=0.3)
    ax = f.add_subplot(gs[0, :]); ax.set_xlim(0, 16); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("Rewriting a residue under a rule: keep at least 0.9 of the original interaction map, keep the pose", loc="left", fontsize=13, fontweight="bold")
    steps = [("candidate residue\n(NCFlow, or a fallback\nbuilder)", "#dbe9f4"), ("1 chemistry\nstrict test passes\n(bonds, angles, clashes,\nlink, handedness)", "#fff3cd"),
             ("2 no clash with the\nprotein (< 2.2 A)", "#fff3cd"), ("3 interaction map\nresidue contacts >= 0.9\npolar contacts >= 0.9\n(one swap loses <= 0.05)", "#f8d7da"),
             ("4 pose\nCA drift <= 1.0 A\natom contacts 0.9 to 1.3\nof the start", "#f8d7da"), ("accept the best\ncandidate (highest\nretention), else try the\nnext fallback", "#d8f0dc")]
    x = 0.1
    for t, fc in steps:
        w = 2.5
        ax.add_patch(FancyBboxPatch((x, 1.5), w, 2.1, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="#495057", lw=1.2))
        ax.text(x + w / 2, 2.55, t, ha="center", va="center", fontsize=8.8)
        if x + w + 0.15 < 15.5:
            ax.add_patch(FancyArrowPatch((x + w, 2.55), (x + w + 0.2, 2.55), arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#495057"))
        x += w + 0.2
    ax.text(0.1, 0.9, "Interaction map = (a) residue pairs peptide-protein within 5 A, (b) polar pairs N/O to N/O within 3.5 A, (c) heavy-atom contacts within 4.5 A; each compared with the natural complex.\n"
                      "Fallback ladder when nothing passes: more draws (up to 36) -> mirror repair -> alternative residue (up to 2) -> geometric builder (to build) -> leave canonical and report it.",
            fontsize=9.3, va="top")

    # (b) share of final peptides meeting each criterion
    b = f.add_subplot(gs[1, 0])
    crit = [("contacts\n>= 0.9", lambda l: fl(l["contact_retention"]) >= 0.9), ("polar\n>= 0.9", lambda l: fl(l["polar_kept"]) >= 0.9), ("drift\n<= 1 A", lambda l: fl(l["ca_drift"]) <= 1.0),
            ("no protein\nclash", lambda l: int(l["rec_clash_pairs"]) == 0), ("strict\ntest", lambda l: int(l["n_fail_all"]) == 0),
            ("all five", None)]
    w = 0.26
    for k, arm in enumerate(ARMS):
        rs = [r for r in rows if r["arm"] == arm]
        items = sorted({r["item"] for r in rs})
        last = [max([r for r in rs if r["item"] == it], key=lambda r: int(r["round"])) for it in items]
        vals = []
        for name, fn in crit:
            if fn is None:
                vals.append(100 * sum(all(c[1](l) for c in crit[:-1]) for l in last) / len(last))
            else:
                vals.append(100 * sum(fn(l) for l in last) / len(last))
        xs = [i - 0.3 + w * k for i in range(len(crit))]
        b.bar(xs, vals, w, color=COL[arm], label=LAB[arm].replace("\n", " "))
        for xx, v in zip(xs, vals):
            b.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    b.set_xticks(range(len(crit))); b.set_xticklabels([n for n, _ in crit], fontsize=8); b.set_ylim(0, 125); b.set_ylabel("% of the 31 finished peptides")
    b.legend(frameon=False, fontsize=7.5, loc="upper right"); b.set_title("(a) Would today's generator meet the rule?\n(final peptides, rule applied afterwards)", fontsize=10.5, loc="left", fontweight="bold")

    # (c) cumulative retention by round
    c = f.add_subplot(gs[1, 1])
    for arm, ls in (("check", "-"), ("repair", "--")):
        rs = [r for r in rows if r["arm"] == arm and int(r["round"]) > 0]
        for key, colr, nm in (("contact_retention", "#2a6f97", "residue contacts"), ("polar_kept", "#c0392b", "polar contacts")):
            xs, ys = [], []
            for rd in range(1, 13):
                v = [fl(r[key]) for r in rs if int(r["round"]) == rd]
                if len(v) >= 6:
                    xs.append(rd); ys.append(S.median(v))
            c.plot(xs, ys, ls=ls, color=colr, lw=2, marker="o", ms=4, label=f"{nm}, {arm}")
    c.axhline(0.9, color="#222", ls=":", lw=1.4); c.text(1.0, 0.905, "0.9", ha="left", fontsize=9)
    c.set_xlabel("number of swaps done (rounds)"); c.set_ylabel("share of the original interaction kept (median)"); c.set_ylim(0.6, 1.02)
    c.legend(frameon=False, fontsize=7.5, loc="lower left"); c.set_title("(b) The map erodes as swaps accumulate", fontsize=10.5, loc="left", fontweight="bold")

    # (d) loss per accepted swap
    d = f.add_subplot(gs[1, 2])
    labs, med, p90, cols = [], [], [], []
    for arm in ("check", "repair"):
        by = collections.defaultdict(dict)
        for r in rows:
            if r["arm"] == arm:
                by[r["item"]][int(r["round"])] = r
        for key, nm in (("contact_retention", "contacts"), ("polar_kept", "polar")):
            dl = []
            for it, dd in by.items():
                for k in sorted(dd):
                    if k > 0 and k - 1 in dd and dd[k]["n_converted"] != dd[k - 1]["n_converted"]:
                        dl.append(fl(dd[k - 1][key]) - fl(dd[k][key]))
            dl.sort()
            labs.append(f"{nm}\n{arm}"); med.append(S.median(dl)); p90.append(dl[int(0.9 * len(dl))]); cols.append(COL[arm])
    xs = range(len(labs))
    d.bar([i - 0.2 for i in xs], med, 0.38, color=cols, label="median")
    d.bar([i + 0.2 for i in xs], p90, 0.38, color=cols, alpha=0.55, label="90th percentile")
    for i in xs:
        d.text(i - 0.2, med[i] + 0.003, f"{med[i]:.3f}", ha="center", fontsize=8); d.text(i + 0.2, p90[i] + 0.003, f"{p90[i]:.3f}", ha="center", fontsize=8)
    d.axhline(0.05, color="#222", ls=":", lw=1.4); d.text(3.6, 0.052, "limit per swap 0.05", ha="right", fontsize=8.5)
    d.set_xticks(list(xs)); d.set_xticklabels(labs, fontsize=8.5); d.set_ylabel("loss of the interaction map per accepted swap"); d.set_ylim(0, 0.16)
    d.legend(frameon=False, fontsize=8, loc="upper left"); d.set_title("(c) One swap usually costs little, sometimes a lot", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.02, 0.005, "Result: with the rule applied afterwards, only 8 to 10 of 31 finished peptides would pass (polar contacts are the bottleneck); the rule has to be built into the choice of each swap, not checked at the end.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig19_rewrite_protocol.png"), bbox_inches="tight")
    plt.close(f)
