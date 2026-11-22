"""Figure: NCFlow strict test on held-out residues (real, RF3-predicted complexes, synthetic peptides) and the iterative canonical -> non-canonical conversion.
Reads the TSV files written by ncf_strict_run.py / ncf_iter_analyze.py; every model present in the files gets a bar (raw, ft1, ft2, rl)."""
import os, csv, statistics as S, collections
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out"
ORDER = ["raw", "ft1", "ft2", "rl", "chi"]
LABEL = {"raw": "raw (as released)", "ft1": "ft1: 252 real", "ft2": "ft2: real + 4,000 synthetic", "rl": "reward (ft2 + chemistry penalty)", "chi": "reward + handedness input"}
COL = {"raw": "#b23a48", "ft1": "#c9a227", "ft2": "#2a6f97", "rl": "#3a7d44", "chi": "#7b4fa3"}
SETS = [("real_test", "real\n(89)"), ("rf3", "RF3 complexes\n(39)"), ("syn_test", "synthetic\n(180)")]
FAMS = ["alpha, side-chain modified", "alpha,alpha-disubstituted", "N-alkyl", "D-alpha", "beta", "gamma"]
FAMN = ["alpha,\nmodified", "alpha,\nalpha", "N-alkyl", "D-alpha", "beta", "gamma"]


def load(tag):
    p = f"{D}/strict_{tag}.tsv"
    if not os.path.exists(p):
        return None
    return [r for r in csv.DictReader(open(p), delimiter="\t") if r.get("g_checked") == "1"]


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def fig(OUT, C):
    data = {t: load(t) for t in ORDER}
    tags = [t for t in ORDER if data[t]]
    it = [r for r in csv.DictReader(open(f"{D}/iter_arms.tsv"), delimiter="\t") if not r.get("error")]
    itags = [t for t in ORDER if any(r["arm"] == t for r in it)]
    f = plt.figure(figsize=(16, 11.5))
    gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1.15, 1.15], hspace=0.7, wspace=0.3)
    ax = f.add_subplot(gs[0, :]); ax.set_xlim(0, 16); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title("NCFlow: how well does it redraw a hidden residue? (the strict chemistry check)", loc="left", fontsize=13, fontweight="bold")
    for x, w, t, fc in ((0.1, 3.6, "hide one residue\nin a peptide", "#dbe9f4"), (4.2, 3.6, "NCFlow redraws it\n(best of 4 tries by its own score)", "#fff3cd"),
                        (8.3, 3.8, "strict check: bonds, angles,\nclashes, link to neighbour,\nhandedness all must pass", "#f8d7da"), (12.6, 3.3, "compare: raw, ft1, ft2\nand the reward fine-tune", "#d8f0dc")):
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.2, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#495057", lw=1.2))
        ax.text(x + w / 2, 0.8, t, ha="center", va="center", fontsize=9.3)
    for x0, x1 in ((3.7, 4.2), (7.8, 8.3), (12.1, 12.6)):
        ax.add_patch(FancyArrowPatch((x0, 0.8), (x1, 0.8), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#495057"))

    n = len(tags); w = 0.8 / n
    # (a) strict pass by set
    a = f.add_subplot(gs[1, 0])
    for k, t in enumerate(tags):
        vals = []
        for s, _ in SETS:
            sub = [r for r in data[t] if r["set"] == s]
            vals.append(100 * sum(1 for r in sub if r["g_strict"] == "1") / max(1, len(sub)))
        xs = [i - 0.4 + w * (k + 0.5) for i in range(len(SETS))]
        a.bar(xs, vals, w, color=COL[t], label=LABEL[t])
        for x, v in zip(xs, vals):
            a.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ctrl = []
    for s, _ in SETS:
        sub = [r for r in data[tags[0]] if r["set"] == s]
        ctrl.append(100 * sum(1 for r in sub if r.get("t_strict") == "1") / max(1, len(sub)))
    for i, c in enumerate(ctrl):
        a.plot([i - 0.42, i + 0.42], [c, c], color="#222", ls="--", lw=1.3)
    a.plot([], [], color="#222", ls="--", label="the original residue (best possible)")
    a.set_xticks(range(len(SETS))); a.set_xticklabels([n_ for _, n_ in SETS], fontsize=9)
    a.set_ylim(0, 175); a.set_ylabel("% of redrawn residues passing everything"); a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    a.set_title("(a) Pass rate on held-out residues", fontsize=10.5, loc="left", fontweight="bold")

    # (b) why they fail on the synthetic set
    b = f.add_subplot(gs[1, 1])
    CH = [("f_bond", "bond\nlength"), ("f_angle", "bond\nangle"), ("f_clash", "atoms\ntoo close"), ("f_link", "link to\nnext\nresidue"), ("f_chir", "wrong\nhand-\nedness")]
    for k, t in enumerate(tags):
        sub = [r for r in data[t] if r["set"] == "syn_test"]
        vals = [100 * sum(1 for r in sub if r["g_" + c] == "0") / max(1, len(sub)) for c, _ in CH]
        xs = [i - 0.4 + w * (k + 0.5) for i in range(len(CH))]
        b.bar(xs, vals, w, color=COL[t])
        for x, v in zip(xs, vals):
            b.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    b.set_xticks(range(len(CH))); b.set_xticklabels([n_ for _, n_ in CH], fontsize=8.5)
    b.set_ylim(0, 100); b.set_ylabel("% of residues failing this check")
    b.set_title("(b) What fails (synthetic peptides)", fontsize=10.5, loc="left", fontweight="bold")

    # (c) per family, synthetic
    c_ = f.add_subplot(gs[1, 2])
    for k, t in enumerate(tags):
        vals = []
        for fam in FAMS:
            sub = [r for r in data[t] if r["set"] == "syn_test" and r["family"] == fam]
            vals.append(100 * sum(1 for r in sub if r["g_strict"] == "1") / max(1, len(sub)))
        xs = [i - 0.4 + w * (k + 0.5) for i in range(len(FAMS))]
        c_.bar(xs, vals, w, color=COL[t])
        for x, v in zip(xs, vals):
            c_.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    c_.set_xticks(range(len(FAMS))); c_.set_xticklabels(FAMN, fontsize=8.5)
    c_.set_ylim(0, 100); c_.set_ylabel("% passing everything")
    c_.set_title("(c) Pass rate by residue type (synthetic, 30 each)", fontsize=10.5, loc="left", fontweight="bold")

    # iterative conversion
    def conv(tag):
        rows = [r for r in it if r["arm"] == tag and int(r["round"]) > 0]
        return rows
    n2 = len(itags); w2 = 0.8 / max(n2, 1)
    d = f.add_subplot(gs[2, 0])
    buckets = [(1, 3, "swaps 1-3"), (4, 6, "swaps 4-6"), (7, 99, "swaps 7+")]
    for k, t in enumerate(itags):
        vals = []
        for lo, hi, _ in buckets:
            sub = [r for r in conv(t) if lo <= int(r["round"]) <= hi]
            vals.append(100 * sum(fnum(r["n_conv_fail"]) for r in sub) / max(1, sum(fnum(r["n_conv_checked"]) for r in sub)))
        xs = [i - 0.4 + w2 * (k + 0.5) for i in range(len(buckets))]
        d.bar(xs, vals, w2, color=COL[t])
        for x, v in zip(xs, vals):
            d.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    d.set_xticks(range(len(buckets))); d.set_xticklabels([l for _, _, l in buckets], fontsize=9)
    d.set_ylim(0, 110); d.set_ylabel("% of swapped residues failing")
    d.set_title("(d) Iterative conversion: swapped residues\nthat fail the check", fontsize=10.5, loc="left", fontweight="bold")

    sub2 = gs[2, 1].subgridspec(1, 2, wspace=0.5)
    for j, (key, ttl, unit) in enumerate((("conv_bond_rms", "bond-length error", "A"), ("conv_angle_rms", "bond-angle error", "deg"))):
        e = f.add_subplot(sub2[0, j])
        vals = [S.mean(fnum(r[key]) for r in conv(t) if fnum(r[key]) is not None) for t in itags]
        e.bar(range(len(itags)), vals, color=[COL[t] for t in itags], width=0.7)
        for i, v in enumerate(vals):
            e.text(i, v * 1.02, f"{v:.3f}" if unit == "A" else f"{v:.1f}", ha="center", fontsize=8)
        e.set_xticks(range(len(itags))); e.set_xticklabels(itags, fontsize=8.5)
        e.set_ylabel(f"average error ({unit})"); e.set_ylim(0, max(vals) * 1.25)
        e.set_title(ttl, fontsize=9.5, loc="left")
    f.text(0.385, 0.385, "(e) Strain of the swapped residues (lower = better)", fontsize=10.5, fontweight="bold")

    sub3 = gs[2, 2].subgridspec(1, 2, wspace=0.5)
    for j, (key, ttl, unit) in enumerate((("contact_retention", "contacts kept", "%"), ("ca_drift", "peptide drift", "A"))):
        e = f.add_subplot(sub3[0, j])
        vals = []
        for t in itags:
            rr = [fnum(r[key]) for r in conv(t) if fnum(r[key]) is not None]
            vals.append((100 if unit == "%" else 1) * S.median(rr))
        e.bar(range(len(itags)), vals, color=[COL[t] for t in itags], width=0.7)
        for i, v in enumerate(vals):
            e.text(i, v * 1.02, f"{v:.0f}" if unit == "%" else f"{v:.2f}", ha="center", fontsize=8)
        e.set_xticks(range(len(itags))); e.set_xticklabels(itags, fontsize=8.5)
        e.set_ylim(0, 115 if unit == "%" else max(vals) * 1.3)
        e.set_ylabel("% of the starting contacts" if unit == "%" else "CA movement (A)")
        e.set_title(ttl, fontsize=9.5, loc="left")
    f.text(0.70, 0.385, "(f) Does the peptide stay in place?", fontsize=10.5, fontweight="bold")
    f.text(0.02, 0.005, "Result: the reward fine-tune removes bond-length and bond-angle failures (0%) and lifts the synthetic pass rate to 51%, but wrong handedness (38%) and the link to the next residue (25%) remain.", fontsize=10, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig12_ncflow_strict_test.png"), bbox_inches="tight")
    plt.close(f)
