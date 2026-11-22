"""Figure: run 2 of the whole-decoder retrain (189 classes, 1,429 synthetic complexes + pool, early stopped, epoch-0 weights) vs run 1 vs untrained new rows, on real PDB peptides."""
import os, csv, json, collections, numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune"; E = f"{D}/data/pilot_train_189/eval_v4"
WEAK = "DHI DIL DTY DAS DGN DSG".split()
def load(n): return list(csv.DictReader(open(f"{E}/ae_eval_{n}.tsv"), delimiter="\t"))
R = {n: load(n) for n in ("bundle", "v3", "v4")}
def cat(rows, c):
    x = [int(r["correct"]) for r in rows if r["category"] == c]; return 100 * np.mean(x), sum(x), len(x)
def rec(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows: d[r["ccd"]][0] += int(r["correct"]); d[r["ccd"]][1] += 1
    return d
LAB = {"bundle": "new rows untrained", "v3": "run 1 (261 complexes)", "v4": "run 2 (1,429 complexes)"}
COL = {"bundle": "#8d99ae", "v3": "#2a6f97", "v4": "#c9a227"}
f = plt.figure(figsize=(19, 8.6)); gs = f.add_gridspec(2, 3, height_ratios=[0.38, 2.4], hspace=0.28, wspace=0.3)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
from matplotlib.patches import FancyBboxPatch
for x, w, t, c in [(1, 21, "1,429 synthetic complexes (12% of protein\nfamilies held out) + 46,379-peptide pool", "#dde5b6"), (27, 22, "whole decoder, 189 classes, batch 64/8\nstops after 3 epochs without held-out gain", "#ade8f4"), (55, 19, "weights of epoch 0\n(best held-out loss)", "#ade8f4"), (80, 19, "TEST: 189 real PDB peptides\n2,765 positions, never trained on", "#ffd6a5")]:
    s.add_patch(FancyBboxPatch((x, 2), w, 6, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + w / 2, 5, t, ha="center", va="center", fontsize=9)
for x0, x1 in [(22.5, 26.5), (49.5, 54.5), (74.5, 79.5)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: retrain the decoder on more synthetic data, test on real structures", fontsize=10.5, loc="left", fontweight="bold")
a = f.add_subplot(gs[1, 0]); cs = ["canonical", "trained_ncaa", "other_vocab_ncaa", "new_dform"]; lab = ["canonical", "32 trained\nnon-canonical", "other vocab", "13 new\nD-forms"]
for k, n in enumerate(("bundle", "v3", "v4")):
    v = [cat(R[n], c)[0] for c in cs]; a.bar(np.arange(4) + (k - 1) * 0.27, v, 0.27, color=COL[n], edgecolor="#555", label=LAB[n])
    for i, x in enumerate(v): a.text(i + (k - 1) * 0.27, x + 1.2, f"{x:.0f}", ha="center", fontsize=7.5)
a.set_xticks(range(4)); a.set_xticklabels(lab, fontsize=9); a.set_ylim(0, 120); a.set_ylabel("real positions named correctly (%)"); a.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
a.set_title("(a) Accuracy by group", fontsize=10.5, loc="left", fontweight="bold")
b = f.add_subplot(gs[1, 1]); recs = {n: rec(R[n]) for n in R}; x = np.arange(len(WEAK) + 1)
for k, n in enumerate(("bundle", "v3", "v4")):
    v = [100 * recs[n][c][0] / recs[n][c][1] for c in WEAK] + [100 * sum(recs[n][c][0] for c in WEAK) / sum(recs[n][c][1] for c in WEAK)]
    b.bar(x + (k - 1) * 0.27, v, 0.27, color=COL[n], edgecolor="#555")
b.set_xticks(x); b.set_xticklabels([f"{c}\n({recs['v4'][c][1]})" for c in WEAK] + ["six pooled\n(34)"], fontsize=8.5); b.set_ylim(0, 110); b.set_ylabel("recall on real positions (%)")
b.axhline(95, color="#b23a48", lw=1, ls="--"); b.text(len(WEAK) + 0.45, 96, "95% stop line", color="#b23a48", fontsize=8, ha="right")
for k, n in enumerate(("bundle", "v3", "v4")):
    w = sum(recs[n][c][0] for c in WEAK); b.text(len(WEAK) + (k - 1) * 0.27, 100 * w / 34 + 1.5, f"{w}", ha="center", fontsize=8)
b.set_title("(b) The six weak D-forms (positions in brackets)", fontsize=10.5, loc="left", fontweight="bold")
c = f.add_subplot(gs[1, 2]); rows = [l for l in open(f"{D}/dec_v4_488524.log", errors="ignore") if l.startswith("epoch ")]
ep = []; vc = []; vp = []
import re
for l in rows:
    m = re.match(r"epoch (\d+): val pool loss ([\d.]+) acc [\d.]+ \| val complexes loss ([\d.]+)", l)
    if m: ep.append(int(m.group(1))); vp.append(float(m.group(2))); vc.append(float(m.group(3)))
c.plot(ep, vc, "o-", color="#b23a48", label="held-out complex loss"); c.plot(ep, vp, "s-", color="#2a6f97", label="held-out pool loss")
c.set_xticks(ep); c.set_xlabel("epoch"); c.set_ylabel("loss"); c.legend(frameon=False, fontsize=8); c.annotate("best = epoch 0\n(saved)", (0, vc[0]), xytext=(0.5, vc[0] + 0.012), fontsize=8)
c.set_title("(c) Run 2 held-out loss (stopped after epoch 3)", fontsize=10.5, loc="left", fontweight="bold")
f.text(0.01, 0.005, "Real positions are test only. Counts: weak six pooled 6/34 untrained, 9/34 run 1, 10/34 run 2. One run each. The same run-1 weights scored 77/112 and 74/112 new D-forms in two evaluations, so differences of about 3 positions are noise.", fontsize=8.5, fontweight="bold")
f.savefig(os.path.join(D, "figures", "fig37_decoder_run2_real_pose.png"), bbox_inches="tight", dpi=130); plt.close(f)
