"""Figure: iterative conversion of 31 real canonical-rich peptides on their receptors into non-canonical ones, three ways (top sample / check + retries / check + retries + mirror repair)."""
import os, csv, pickle, statistics as S, collections
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out"
ARMS = ["top1", "check", "repair"]
LAB = {"top1": "top sample only\n(no check)", "check": "check + retries", "repair": "check + retries\n+ mirror repair"}
COL = {"top1": "#b23a48", "check": "#2a6f97", "repair": "#3a7d44"}


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def fig(OUT, C):
    rows = list(csv.DictReader(open(f"{D}/iter3_arms.tsv"), delimiter="\t"))
    last, first = {}, {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        for r in rs:
            it = r["item"]
            if (arm, it) not in last or int(r["round"]) > int(last[(arm, it)]["round"]):
                last[(arm, it)] = r
            if r["round"] == "0":
                first[(arm, it)] = r
    items = sorted({it for a, it in last})
    pk = {a: pickle.load(open(f"{D}/iter3_rl_{a}.pkl", "rb")) for a in ARMS}
    f = plt.figure(figsize=(16, 11.5))
    gs = f.add_gridspec(3, 3, height_ratios=[0.55, 1.15, 1.15], hspace=1.0, wspace=0.32)
    ax = f.add_subplot(gs[0, :]); ax.set_xlim(0, 16); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title("Turning real canonical peptides on their proteins into non-canonical ones, one residue at a time", loc="left", fontsize=13, fontweight="bold")
    for x, w, t, fc in ((0.1, 3.5, "31 real peptide-protein\ncomplexes (227 swaps)", "#dbe9f4"), (4.1, 3.7, "swap one canonical residue per round;\nNCFlow redraws it in the real pocket", "#fff3cd"),
                        (8.3, 3.9, "three ways: top sample only,\ncheck + retries (up to 36 draws),\ncheck + retries + mirror repair", "#f8d7da"), (12.7, 3.2, "measure chemistry and\nthe interaction with the\nprotein", "#d8f0dc")):
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.2, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#495057", lw=1.2))
        ax.text(x + w / 2, 0.8, t, ha="center", va="center", fontsize=9.3)
    for x0, x1 in ((3.6, 4.1), (7.8, 8.3), (12.2, 12.7)):
        ax.add_patch(FancyArrowPatch((x0, 0.8), (x1, 0.8), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))

    def bars(a, vals, title, ylab, fmt="{:.0f}", ylim=None, cols=None):
        b = a.bar(range(len(ARMS)), vals, color=cols or [COL[x] for x in ARMS], width=0.62)
        for i, v in enumerate(vals):
            a.text(i, v + (ylim or max(vals)) * 0.02, fmt.format(v), ha="center", fontsize=9)
        a.set_xticks(range(len(ARMS))); a.set_xticklabels([LAB[x] for x in ARMS], fontsize=8.5)
        a.set_ylabel(ylab); a.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        if ylim:
            a.set_ylim(0, ylim)

    # (a) outcome of the swaps: accepted & passes / accepted but fails / left canonical
    a = f.add_subplot(gs[1, 0])
    ok, bad, skip = [], [], []
    for arm in ARMS:
        o = b_ = s_ = 0
        for it, v in pk[arm].items():
            for sn in v["snaps"][1:]:
                i = sn["info"]
                if i.get("skipped"):
                    s_ += 1
                elif i.get("passed"):
                    o += 1
                else:
                    b_ += 1
        ok.append(o); bad.append(b_); skip.append(s_)
    tot = [x + y + z for x, y, z in zip(ok, bad, skip)]
    pc = lambda L: [100 * x / t for x, t in zip(L, tot)]
    a.bar(range(3), pc(ok), color="#3a7d44", label="residue swapped and passes the check", width=0.62)
    a.bar(range(3), pc(bad), bottom=pc(ok), color="#b23a48", label="swapped but fails the check", width=0.62)
    a.bar(range(3), pc(skip), bottom=[x + y for x, y in zip(pc(ok), pc(bad))], color="#adb5bd", label="left canonical (no passing sample)", width=0.62)
    for i in range(3):
        a.text(i, pc(ok)[i] / 2 if pc(ok)[i] > 8 else 3, f"{pc(ok)[i]:.0f}", ha="center", color="white", fontsize=9, fontweight="bold")
        if pc(bad)[i] > 8:
            a.text(i, pc(ok)[i] + pc(bad)[i] / 2, f"{pc(bad)[i]:.0f}", ha="center", color="white", fontsize=9, fontweight="bold")
        if pc(skip)[i] > 8:
            a.text(i, pc(ok)[i] + pc(bad)[i] + pc(skip)[i] / 2, f"{pc(skip)[i]:.0f}", ha="center", fontsize=9, fontweight="bold")
    a.set_xticks(range(3)); a.set_xticklabels([LAB[x] for x in ARMS], fontsize=8.5); a.set_ylim(0, 100); a.set_ylabel("% of the 227 planned swaps")
    a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    a.set_title("(a) What happened to each swap", fontsize=10.5, loc="left", fontweight="bold")

    # (b) residues failing the strict test in the finished peptides (start: before any swap)
    b = f.add_subplot(gs[1, 1])
    fr = [100 * sum(int(last[(x, it)]["n_fail_all"]) for it in items) / sum(int(last[(x, it)]["n_checked"]) for it in items) for x in ARMS]
    st = 100 * sum(int(first[("top1", it)]["n_fail_all"]) for it in items) / sum(int(first[("top1", it)]["n_checked"]) for it in items)
    bars(b, fr, "(b) Residues failing the strict test\nin the finished peptides", "% of all residues", ylim=85)
    b.axhline(st, color="#222", ls="--", lw=1.2); b.text(2.4, st + 2, f"before any swap: {st:.0f}%", ha="right", fontsize=8.5)

    # (c) cost
    c = f.add_subplot(gs[1, 2])
    dr = [sum(int(last[(x, it)]["draws"]) for it in items) / sum(int(last[(x, it)]["n_swaps_planned"]) for it in items) for x in ARMS]
    bars(c, dr, "(c) Cost: draws per planned swap", "samples drawn per swap", fmt="{:.1f}", ylim=max(dr) * 1.25)

    # (d) interaction with the protein
    d = f.add_subplot(gs[2, 0])
    names = [("contact_retention", "residue contacts\nkept (5 A)"), ("polar_kept", "polar contacts\nkept (3.5 A)"), ("contact_atoms_ratio", "atom contacts\nvs start (4.5 A)")]
    w = 0.26
    for k, x in enumerate(ARMS):
        vals = [100 * S.median(fnum(last[(x, it)][key]) for it in items) for key, _ in names]
        xs = [i - 0.3 + w * k for i in range(3)]
        d.bar(xs, vals, w, color=COL[x], label=LAB[x].replace("\n", " "))
        for xx, v in zip(xs, vals):
            d.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    d.set_xticks(range(3)); d.set_xticklabels([n_ for _, n_ in names], fontsize=8.5); d.set_ylim(0, 150); d.set_ylabel("% of the starting interaction (median)")
    d.legend(frameon=False, fontsize=7.5, loc="upper left"); d.set_title("(d) Interaction with the protein kept", fontsize=10.5, loc="left", fontweight="bold")

    # (e) receptor clashes + drift
    e = f.add_subplot(gs[2, 1])
    cl = [sum(1 for it in items if int(last[(x, it)]["rec_clash_pairs"]) > 0) for x in ARMS]
    bars(e, cl, "(e) Finished peptides with an atom closer\nthan 2.2 A to the protein (of 31)", "number of peptides", fmt="{:.0f}", ylim=max(max(cl) * 1.3, 3))

    # (f) why swaps are skipped
    g = f.add_subplot(gs[2, 2])
    CH = [("chir", "wrong\nhand"), ("link", "link"), ("clash", "atoms\nclose"), ("rec_clash", "protein\nclash"), ("bond", "bond\nlength"), ("angle", "bond\nangle")]
    w2 = 0.38
    for k, x in enumerate(("check", "repair")):
        cnt = collections.Counter()
        for it, v in pk[x].items():
            for sn in v["snaps"][1:]:
                i = sn["info"]
                if i.get("skipped"):
                    for kk, vv in (i.get("fails") or {}).items():
                        if kk in dict(CH):
                            cnt[kk] += vv
        vals = [cnt[kk] for kk, _ in CH]
        xs = [i - w2 / 2 + w2 * k for i in range(len(CH))]
        g.bar(xs, vals, w2, color=COL[x], label=LAB[x].replace("\n", " "))
        for xx, v in zip(xs, vals):
            g.text(xx, v + 8, f"{v}", ha="center", fontsize=7.5)
    g.set_xticks(range(len(CH))); g.set_xticklabels([n_ for _, n_ in CH], fontsize=8); g.set_ylabel("failed draws (swaps left canonical)")
    g.legend(frameon=False, fontsize=8); g.set_title("(f) Why swaps were left canonical", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.02, 0.005, "Result: without the check most swapped residues are chemically wrong; the check removes that at the price of leaving some residues canonical, and the interaction with the protein is kept.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig15_conversion_with_check.png"), bbox_inches="tight")
    plt.close(f)
