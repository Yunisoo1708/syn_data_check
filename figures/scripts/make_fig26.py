"""Figure: docking of 46k-pool peptides into the pockets of the 264 train complexes (rigid CA-trace matching, window search, residue-level design) with the gate funnel and the self-test of the score."""
import os, pickle, collections, statistics as S
import numpy as np
import matplotlib.pyplot as plt

D = "C:/Research/RFD4_fine_tune/data/pool_dock"
COL = {"null": "#8d99ae", "rigid": "#2a6f97", "design": "#c9a227", "native": "#b23a48"}


def load(n):
    p = f"{D}/{n}"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}


def best_funnel(cands, native, key="score"):
    """list of booleans per gate level: any candidate passes gates up to level k"""
    lv = [False] * 4
    for c in cands:
        if c["rmin"] < 2.2:
            continue
        lv[0] = True
        if c["recall"] < 0.6:
            continue
        lv[1] = True
        if c[key] <= 0.8 * native:
            lv[2] = True
        if c[key] <= native:
            lv[3] = True
    return lv


def fig(OUT, C):
    st, dk, dw, ds = load("selftest_all.pkl"), load("dock_all.pkl"), load("dockw_all.pkl"), load("design_all.pkl")
    f, ax = plt.subplots(2, 2, figsize=(14.5, 10.5), gridspec_kw=dict(wspace=0.28, hspace=0.42))
    a = ax[0, 0]
    ok = [v for v in st.values() if "error" not in v]
    lv = [(0.2, "start 1.5 A / 11 deg off"), (0.35, "start 2.5 A / 20 deg off")]
    for i, (ang, lab) in enumerate(lv):
        r = [t["ca_rmsd"] for v in ok for t in v["trials"] if t["ang"] == ang]
        a.bar(i * 2, 100 * np.mean([x < 1 for x in r]), color=COL["rigid"], edgecolor="#555", width=0.8)
        a.bar(i * 2 + 1, 100 * np.mean([x < 2 for x in r]), color=COL["null"], edgecolor="#555", width=0.8)
        a.text(i * 2, 100 * np.mean([x < 1 for x in r]) + 1.5, f"{100 * np.mean([x < 1 for x in r]):.0f}%", ha="center", fontsize=9, fontweight="bold")
        a.text(i * 2 + 1, 100 * np.mean([x < 2 for x in r]) + 1.5, f"{100 * np.mean([x < 2 for x in r]):.0f}%", ha="center", fontsize=9, fontweight="bold")
    a.set_xticks([0.5, 2.5]); a.set_xticklabels([l for _, l in lv], fontsize=9); a.set_ylim(0, 112); a.set_ylabel("natural peptides re-docked to the real pose (%)")
    a.set_title(f"(a) Self-test of the score and optimiser\n({len(ok)} natural peptides; blue < 1 A, grey < 2 A CA RMSD)", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[0, 1]
    arms = []
    for lab, key, src in [("random windows\n(null)", "wrand", dw), ("best-matching windows\nrigid", "window", dw)]:
        n = tot = 0
        fl = np.zeros(4)
        for it, v in src.items():
            if "error" in v:
                continue
            c = [x for x in v["cands"] if x["arm"] == key]
            fl += np.array(best_funnel(c, v["native"]), float); tot += 1
        arms.append((lab, fl / max(1, tot) * 100, tot))
    if ds:
        fl = np.zeros(4); tot = 0
        for it, v in dw.items():
            if "error" in v:
                continue
            tot += 1
            d = ds.get(it)
            cands = [dict(rmin=x["rmin"], recall=x["recall"], score=x["final_score"]) for x in (d["designs"] if d and "error" not in d else [])]
            fl += np.array(best_funnel(cands, v["native"]), float)
        arms.append(("after residue-level\ndesign (2 starts)", fl / max(1, tot) * 100, tot))
    w = 0.26
    cols = [COL["null"], COL["rigid"], COL["design"]]
    for k, (lab, fl, tot) in enumerate(arms):
        b.bar(np.arange(4) + (k - 1) * w, fl, w, color=cols[k], edgecolor="#555", label=f"{lab.replace(chr(10), ' ')} (n={tot})")
        for i, v in enumerate(fl):
            b.text(i + (k - 1) * w, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    b.set_xticks(range(4)); b.set_xticklabels(["no receptor\nclash", "+ contacts 60% of\nnatural pocket", "+ score within\n80% of natural", "+ score as good\nas natural"], fontsize=8.5)
    b.set_ylabel("pockets with at least one candidate (%)"); b.set_ylim(0, 118); b.legend(frameon=False, fontsize=8, loc="upper right")
    b.set_title("(b) Gate funnel over the train pockets", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[1, 0]
    xs, y1, y2 = [], [], []
    for it, v in dw.items():
        if "error" in v:
            continue
        cc = [x for x in v["cands"] if x["arm"] == "window" and x["rmin"] >= 2.2 and x["recall"] >= 0.6]
        d = ds.get(it) if ds else None
        dd = [x for x in (d["designs"] if d and "error" not in d else []) if x["rmin"] >= 2.2 and x["recall"] >= 0.6]
        if cc:
            xs.append(v["native"]); y1.append(min(x["score"] for x in cc)); y2.append(min([x["final_score"] for x in dd] or [np.nan]))
    xs, y1, y2 = map(np.array, (xs, y1, y2))
    c.scatter(xs, y1, s=20, color=COL["rigid"], alpha=0.8, label="rigid placement", edgecolor="none")
    if len(y2):
        c.scatter(xs, y2, s=20, color=COL["design"], alpha=0.8, label="after design", edgecolor="none")
    lim = [min(np.nanmin(xs), np.nanmin(y1)) - 2 if len(xs) else -40, 2]
    c.plot(lim, lim, color="#222", lw=1); c.set_xlim(lim); c.set_ylim(lim)
    c.set_xlabel("score of the NATURAL peptide (Vina-style, lower = better)"); c.set_ylabel("best docked pool peptide")
    c.legend(frameon=False, fontsize=9); c.text(0.05, 0.93, "below the line = better than the natural peptide", transform=c.transAxes, fontsize=8.5)
    c.set_title("(c) Best candidate per pocket against the natural binder", fontsize=10.5, loc="left", fontweight="bold")
    d_ = ax[1, 1]
    cnt = collections.Counter()
    for it, d in (ds or {}).items():
        if "error" in d or not d["designs"]:
            continue
        best = min(d["designs"], key=lambda x: x["final_score"])
        cnt.update(set(best["codes"]))
    items = cnt.most_common(32)
    d_.bar(range(len(items)), [v for _, v in items], color=COL["design"], edgecolor="#555")
    d_.set_xticks(range(len(items))); d_.set_xticklabels([k for k, _ in items], rotation=90, fontsize=7.5)
    d_.set_ylabel("designed peptides containing the residue"); d_.set_title(f"(d) Which non-canonical residues the designs use\n({len(cnt)} different codes; one best design per pocket)", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.02, "Score = AutoDock-Vina-style interface proxy computed from elements and distances (works for any residue); it is a proxy for affinity, not a measurement. Pool peptides and residues come from structures that passed the strict geometry test; rigid placement keeps them valid.", fontsize=9, fontweight="bold", wrap=True)
    f.savefig(os.path.join(OUT, "fig26_pool_docking.png"), bbox_inches="tight")
    plt.close(f)
