"""Figure: frozen split v2 (all real non-canonical complexes = test; train / val from canonical natural complexes unrelated to them, used for synthetic data)."""
import os, json, collections
import matplotlib.pyplot as plt

P = "C:/Research/RFD4_fine_tune/splits/split_v2.json"
ORDER = ["train", "val", "test", "excluded_test_family"]
COL = {"train": "#8d99ae", "val": "#2a6f97", "test": "#b23a48", "excluded_test_family": "#e9ecef"}
LAB = {"train": "train\n(receptors for\nsynthetic data)", "val": "validation\n(synthetic)", "test": "TEST: real\nnon-canonical\ncomplexes", "excluded_test_family": "canonical complexes\nrelated to a test one\n(not used)"}


def fig(OUT, C):
    d = json.load(open(P))
    f, ax = plt.subplots(1, 3, figsize=(16.5, 5.8), gridspec_kw=dict(wspace=0.32))
    a = ax[0]
    cnt = d["counts"]["entries_by_split"]
    for i, s in enumerate(ORDER):
        a.bar(i, cnt.get(s, 0), color=COL[s], edgecolor="#555", width=0.65)
        a.text(i, cnt.get(s, 0) + 40, str(cnt.get(s, 0)), ha="center", fontsize=10, fontweight="bold")
    a.set_xticks(range(4)); a.set_xticklabels([LAB[s] for s in ORDER], fontsize=8); a.set_ylabel("PDB entries"); a.set_ylim(0, 2500)
    a.set_title("(a) Entries per role", fontsize=11, loc="left", fontweight="bold")
    b = ax[1]
    g = collections.Counter()
    for x in d["groups"]:
        g[x["split"]] += 1
    gs = d["counts"]["groups_by_split"]
    labs = [("train", "train"), ("val", "val"), ("test_family", "test")]
    for i, (k, s) in enumerate(labs):
        b.bar(i, gs.get(k, 0), color=COL[s], edgecolor="#555", width=0.6)
        b.text(i, gs.get(k, 0) + 8, str(gs.get(k, 0)), ha="center", fontsize=10, fontweight="bold")
    b.set_xticks(range(3)); b.set_xticklabels(["train", "validation", "test families"]); b.set_ylabel("groups (receptor family or similar peptide)"); b.set_ylim(0, 700)
    b.set_title("(b) Groups: no group is on both sides", fontsize=11, loc="left", fontweight="bold")
    c = ax[2]
    years = list(range(1995, 2027))
    bottom = [0] * len(years)
    for s in ORDER:
        cn = collections.Counter()
        for x in d["groups"]:
            for e, m in x["members"].items():
                if d["split_of_entry"].get(e) == s:
                    cn[int(m["release"][:4])] += 1
        vals = [cn.get(y, 0) for y in years]
        c.bar(years, vals, bottom=bottom, color=COL[s], label=s.replace("_", " "))
        bottom = [u + v for u, v in zip(bottom, vals)]
    c.set_xlabel("release year"); c.set_ylabel("entries"); c.legend(frameon=False, fontsize=8, loc="upper left")
    c.set_title("(c) Release years", fontsize=11, loc="left", fontweight="bold")
    f.text(0.01, -0.04, "Split v2: all 278 real non-canonical complexes are test only; a model never sees them or their families. Train and validation data are synthetic (RF3 + NCFlow) on the canonical complexes at left. Hash " + open("C:/Research/RFD4_fine_tune/splits/split_v2.sha256.txt").read().split()[0][:16] + "...", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig23_split_v2.png"), bbox_inches="tight")
    plt.close(f)
