"""Figure: real-pose test of the whole-decoder, 189-class model (13 new D-forms) vs the run2 overlay baseline."""
import os, csv, json, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"
E = f"{D}/data/pilot_train_189/eval_v3"
def load(n): return list(csv.DictReader(open(f"{E}/ae_eval_{n}.tsv"), delimiter="\t"))
B, V = load("bundle"), load("v3")
M = json.load(open(f"{D}/data/pilot_out/manifest.json")); tr = collections.Counter(s["code"] for m in M for s in m["swaps"])
NEW = "DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split()
def cat(rows, c):
    x = [int(r["correct"]) for r in rows if r["category"] == c]; return 100 * np.mean(x), len(x)
def rec(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows: d[r["ccd"]][0] += int(r["correct"]); d[r["ccd"]][1] += 1
    return d
GR, BL = "#8d99ae", "#2a6f97"
f = plt.figure(figsize=(19, 9.2))
gs = f.add_gridspec(2, 3, height_ratios=[0.4, 2.3], hspace=0.25, wspace=0.32)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
boxes = [(1, 20, "46,379 strict-passed" + chr(10) + "peptides (pool)", "#dde5b6"), (25, 22, "261 synthetic complexes" + chr(10) + "(NCFlow + RF3), 968 D-form swaps", "#dde5b6"), (51, 24, "189-class decoder (13 D-forms added)" + chr(10) + "whole decoder 128M numbers, start = run2", "#ade8f4"), (79, 20, "TEST: 189 real PDB peptides" + chr(10) + "coordinates in, residue name out", "#ffd6a5")]
for x, w, t, c in boxes:
    s.add_patch(FancyBboxPatch((x, 2), w, 6, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + w / 2, 5, t, ha="center", va="center", fontsize=9)
for x0, x1 in [(21.5, 24.5), (47.5, 50.5), (75.5, 78.5)]:
    s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.text(63, 9.2, "each step: pool batch 256 + complex batch 16", fontsize=8, ha="center")
s.set_title("Method: train whole decoder on pool + synthetic complexes, test on real structures never used for training", fontsize=10.5, loc="left", fontweight="bold")
a = f.add_subplot(gs[1, 0]); cs = ["canonical", "trained_ncaa", "other_vocab_ncaa", "new_dform"]; lab = ["canonical", "32 trained\nnon-canonical", "other vocab\nnon-canonical", "13 new\nD-forms"]
for k, (rows, col, nm) in enumerate([(B, GR, "run2 overlay (new rows untrained)"), (V, BL, "whole decoder, 189 classes")]):
    v = [cat(rows, c)[0] for c in cs]; a.bar(np.arange(4) + (k - 0.5) * 0.38, v, 0.38, color=col, edgecolor="#555", label=nm)
    for i, x in enumerate(v): a.text(i + (k - 0.5) * 0.38, x + 1.5, f"{x:.0f}", ha="center", fontsize=9)
a.set_xticks(range(4)); a.set_xticklabels(lab, fontsize=9); a.set_ylim(0, 118); a.set_ylabel("real positions named correctly (%)"); a.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
a.set_title("(a) Accuracy by group (2,775 real positions)", fontsize=10.5, loc="left", fontweight="bold")
rb, rv = rec(B), rec(V); codes = [c for c in sorted(NEW, key=lambda c: -rv[c][1]) if rv[c][1] > 0]
b = f.add_subplot(gs[1, 1]); x = np.arange(len(codes))
b.bar(x - 0.2, [100 * rb[c][0] / rb[c][1] for c in codes], 0.4, color=GR, edgecolor="#555"); b.bar(x + 0.2, [100 * rv[c][0] / rv[c][1] for c in codes], 0.4, color=BL, edgecolor="#555")
b.set_xticks(x); b.set_xticklabels([f"{c}\n({rv[c][1]})" for c in codes], fontsize=8); b.set_ylim(0, 108); b.set_ylabel("recall on real positions (%)")
b.set_title("(b) 13 new D-forms: grey before, blue after (n)", fontsize=9.5, loc="left", fontweight="bold")
c_ = f.add_subplot(gs[1, 2])
xs = [tr[k] for k in codes]; ys = [100 * rv[k][0] / rv[k][1] for k in codes]
c_.scatter(xs, ys, s=[20 + 6 * rv[k][1] for k in codes], color="#c9a227", edgecolor="#555")
for k, xx, yy in zip(codes, xs, ys): c_.annotate(k, (xx, yy), fontsize=8, xytext=((-24, 5) if k == "MED" else (4, 4)), textcoords="offset points")
c_.set_xlabel("accepted synthetic swaps of the D-form in training data"); c_.set_ylabel("recall on real positions (%)"); c_.set_ylim(-5, 108)
c_.set_title("(c) Recall vs synthetic examples", fontsize=10, loc="left", fontweight="bold")
f.text(0.01, 0.01, "Real benchmark = real PDB peptides, test only, never used for training. Weights = epoch 0 of the 22 epochs run (best on held-out data; later epochs overfitted the 235 training complexes). One run, no repeats.", fontsize=9, fontweight="bold")
f.savefig(os.path.join(D, "figures", "fig36_real_pose_189_decoder.png"), bbox_inches="tight", dpi=130); plt.close(f)
