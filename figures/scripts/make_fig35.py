"""Figure: real-pose test (189 real PDB peptides, residue naming from coordinates) of the decoder residue head trained on masked synthetic complexes only."""
import os, csv, json, collections, numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune"
def load(n): return list(csv.DictReader(open(f"{D}/data/pilot_train_masked/ae_eval_{n}.tsv"), delimiter="\t"))
B, S = load("bundle"), load("syn")
M = json.load(open(f"{D}/data/pilot_out/manifest.json")); tr = collections.Counter(s["code"] for m in M for s in m["swaps"])
def rec(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        d[r["ccd"]][0] += int(r["correct"]); d[r["ccd"]][1] += 1
    return d
def cat(rows, c):
    x = [int(r["correct"]) for r in rows if r["category"] == c]; return 100 * np.mean(x), len(x)
f, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(wspace=0.42, width_ratios=[0.8, 2.0, 1.1]))
a = ax[0]; cs = ["canonical", "trained_ncaa", "other_vocab_ncaa"]; lab = ["canonical", "32 trained\nnon-canonical", "other vocab\nnon-canonical"]
for k, (rows, col, nm) in enumerate([(B, "#8d99ae", "fresh 176-class model"), (S, "#2a6f97", "trained on masked synthetic complexes")]):
    v = [cat(rows, c)[0] for c in cs]; a.bar(np.arange(3) + (k - 0.5) * 0.38, v, 0.38, color=col, edgecolor="#555", label=nm)
    for i, x in enumerate(v): a.text(i + (k - 0.5) * 0.38, x + 1.5, f"{x:.0f}", ha="center", fontsize=9)
a.set_xticks(range(3)); a.set_xticklabels(lab, fontsize=9); a.set_ylim(0, 112); a.set_ylabel("real positions named correctly (%)"); a.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.82))
a.set_title("(a) Real-pose accuracy (2,659 positions)", fontsize=10.5, loc="left", fontweight="bold")
b = ax[1]; rb, rs = rec(B), rec(S); codes = sorted([c for c in rs if rs[c][1] >= 5 and c in set(r["ccd"] for r in S if r["category"] == "trained_ncaa")], key=lambda c: -rs[c][1])
x = np.arange(len(codes)); b.bar(x - 0.2, [100 * rb[c][0] / rb[c][1] for c in codes], 0.4, color="#8d99ae", edgecolor="#555"); b.bar(x + 0.2, [100 * rs[c][0] / rs[c][1] for c in codes], 0.4, color=["#2a6f97" if tr[c] else "#b23a48" for c in codes], edgecolor="#555")
b.set_xticks(x); b.set_xticklabels([f"{c}\n({rs[c][1]})" for c in codes], fontsize=7.5, rotation=90, ha="center"); b.set_ylabel("recall on real positions (%)"); b.set_ylim(0, 108)
b.set_title("(b) Per code: grey fresh, blue trained, red trained code absent from the synthetic data", fontsize=9.5, loc="left", fontweight="bold")
c = ax[2]; xs = [tr[k] for k in rs if rs[k][1] >= 5 and k in set(r["ccd"] for r in S if r["category"] == "trained_ncaa")]; ys = [100 * rs[k][0] / rs[k][1] for k in rs if rs[k][1] >= 5 and k in set(r["ccd"] for r in S if r["category"] == "trained_ncaa")]
c.scatter(xs, ys, s=40, color="#c9a227", edgecolor="#555")
for k in rs:
    if rs[k][1] >= 5 and k in set(r["ccd"] for r in S if r["category"] == "trained_ncaa"): c.annotate(k, (tr[k], 100 * rs[k][0] / rs[k][1]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
c.set_xlabel("accepted swaps of the code in the synthetic training data"); c.set_ylabel("recall on real positions (%)"); c.set_ylim(-5, 108); c.set_title("(c) Recall vs training examples", fontsize=10.5, loc="left", fontweight="bold")
f.text(0.01, -0.04, "Real benchmark = real peptides from PDB, never used for training (test only). The model sees coordinates and atoms, never the residue name. Training: 600 steps, residue-type head only, 254 synthetic complexes with 13 out-of-vocabulary D-forms masked.", fontsize=9, fontweight="bold")
f.savefig(os.path.join(D, "figures", "fig35_real_pose_synthetic_head.png"), bbox_inches="tight"); plt.close(f)
