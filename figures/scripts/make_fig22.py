"""Figure: the frozen split v1 (groups by split and tier, benchmark vs natural, release year)."""
import os, json, collections
import matplotlib.pyplot as plt

P = "C:/Research/RFD4_fine_tune/splits/split_v1.json"
COL = {"train": "#8d99ae", "dev_test": "#2a6f97", "blind_test": "#b23a48"}


def fig(OUT, C):
    d = json.load(open(P))
    G = d["groups"]
    f, ax = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(wspace=0.3))
    a = ax[0]
    for i, s in enumerate(("train", "dev_test", "blind_test")):
        gs = [g for g in G if g["split"] == s]
        a.bar(i, len(gs), color=COL[s], width=0.6)
        a.text(i, len(gs) + 2, f"{len(gs)} groups\n{sum(len(g['entries']) for g in gs)} entries", ha="center", fontsize=9.5, fontweight="bold")
    a.set_xticks(range(3)); a.set_xticklabels(["train", "dev test", "blind test\n(do not inspect)"]); a.set_ylim(0, 150); a.set_ylabel("groups (one receptor family or similar peptide each)")
    a.set_title("(a) Groups per split", fontsize=11, loc="left", fontweight="bold")
    b = ax[1]
    w = 0.38
    for i, s in enumerate(("train", "dev_test", "blind_test")):
        gs = [g for g in G if g["split"] == s]
        nb, nn = sum(g["n_benchmark"] for g in gs), sum(g["n_natural"] for g in gs)
        b.bar(i - w / 2, nb, w, color="#c9a227", label="non-canonical benchmark complexes" if i == 0 else None)
        b.bar(i + w / 2, nn, w, color="#3a7d44", label="natural canonical-peptide complexes" if i == 0 else None)
        b.text(i - w / 2, nb + 2, str(nb), ha="center", fontsize=9); b.text(i + w / 2, nn + 2, str(nn), ha="center", fontsize=9)
    b.set_xticks(range(3)); b.set_xticklabels(["train", "dev test", "blind test"]); b.set_ylim(0, 150); b.set_ylabel("entries"); b.legend(frameon=False, fontsize=8.5, loc="upper right")
    b.set_title("(b) What is in each split", fontsize=11, loc="left", fontweight="bold")
    c = ax[2]
    years = list(range(1995, 2027))
    bottom = [0] * len(years)
    for s in ("train", "dev_test", "blind_test"):
        cnt = collections.Counter()
        for g in G:
            if g["split"] == s:
                for m in g["members"].values():
                    cnt[int(m["release"][:4])] += 1
        vals = [cnt.get(y, 0) for y in years]
        c.bar(years, vals, bottom=bottom, color=COL[s], label=s.replace("_", " "))
        bottom = [x + y for x, y in zip(bottom, vals)]
    c.axvline(2024.5, color="#222", ls=":"); c.text(2024.3, max(bottom) * 0.9, "cutoff 2025-01-01", ha="right", fontsize=8.5)
    c.set_xlabel("release year of the entry"); c.set_ylabel("entries"); c.legend(frameon=False, fontsize=8.5, loc="upper left")
    c.set_title("(c) Test targets include every post-cutoff family", fontsize=11, loc="left", fontweight="bold")
    f.text(0.01, -0.03, "Frozen split v1: no test or blind receptor, nor a homolog, may be used to train, to start a conversion or to generate data. Split file hash: " + open("C:/Research/RFD4_fine_tune/splits/split_v1.sha256.txt").read().split()[0][:16] + "...", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig22_split_v1.png"), bbox_inches="tight")
    plt.close(f)
