"""Figure: pilot generator with rl_syn on 40 train complexes (laptop share): gate outcomes, accepted swaps by parent residue, quality of accepted swaps."""
import os, json, collections
import numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune"
COL = {"chem": "#b23a48", "clash": "#c9a227", "interaction": "#8d99ae", "pose": "#6c757d", "passed": "#2a6f97"}
LAB = {"chem": "fails strict chemistry", "clash": "clashes with protein", "interaction": "loses interactions", "pose": "pose drift", "passed": "passes all gates"}
def fig(OUT, rows_file="pilot_laptop_swaps.json", name="fig29_pilot_laptop.png"):
    R = json.load(open(f"{D}/{rows_file}"))
    f, ax = plt.subplots(1, 4, figsize=(20, 5.2), gridspec_kw=dict(wspace=0.38))
    a = ax[0]; tot = collections.Counter()
    for r in R:
        for g in COL: tot[g] += r["fails"].get(g, 0)
    left = 0
    for g in COL:
        a.barh(0, tot[g], left=left, color=COL[g], edgecolor="#fff", label=f"{LAB[g]}: {tot[g]}"); left += tot[g]
    a.set_yticks([]); a.set_xlabel("draws"); a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    a.set_title(f"(a) {int(left)} draws for {len(R)} planned swaps", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[1]; by = collections.defaultdict(lambda: [0, 0])
    for r in R: by[r["parent"]][0] += 1; by[r["parent"]][1] += r["ok"]
    ks = sorted(by, key=lambda k: -by[k][0])
    b.bar(range(len(ks)), [100 * by[k][1] / by[k][0] for k in ks], color="#2a6f97", edgecolor="#555")
    for i, k in enumerate(ks): b.text(i, 100 * by[k][1] / by[k][0] + 1.5, f"{by[k][1]}/{by[k][0]}", ha="center", fontsize=6.5, rotation=90)
    b.set_xticks(range(len(ks))); b.set_xticklabels(ks, rotation=90, fontsize=8); b.set_ylim(0, 118); b.set_ylabel("swaps accepted (%)")
    b.set_title("(b) Accepted by residue replaced", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[2]; ok = [r for r in R if r["ok"]]
    for key, col, lab in [("ret_c", "#2a6f97", "residue contacts kept"), ("ret_p", "#c9a227", "polar contacts kept")]:
        v = np.array([r[key] for r in ok if r[key] is not None]); c.hist(v, bins=np.linspace(0.85, 1.0, 16), alpha=0.7, color=col, edgecolor="#555", label=f"{lab} (median {np.median(v):.2f})")
    c.axvline(0.9, color="#222", ls="--", lw=1); c.legend(frameon=False, fontsize=8, loc="upper left"); c.set_xlabel("fraction of the natural complex's contacts kept (gate 0.9)"); c.set_ylabel("accepted swaps")
    c.set_title("(c) Contacts kept after each swap", fontsize=10.5, loc="left", fontweight="bold")
    d = ax[3]
    cnt = collections.Counter(r["code"] for r in ok); it = cnt.most_common(31)
    d.bar(range(len(it)), [v for _, v in it], color="#c9a227", edgecolor="#555"); d.set_xticks(range(len(it))); d.set_xticklabels([k for k, _ in it], rotation=90, fontsize=8)
    d.set_ylabel("accepted swaps"); d.set_title(f"(d) Non-canonical residues produced ({len(cnt)} codes)", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.02, "Every accepted residue passed the strict chemistry test, had no receptor clash (<2.2 A), kept at least 90% of residue and polar contacts, and stayed in the natural pose (CA drift <= 1.0 A). Train complexes only; rl_syn trained on synthetic data only.", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(f)
if __name__ == "__main__": fig(f"{D}/figures")
