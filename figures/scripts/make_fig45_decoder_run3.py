"""Figure: decoder run 3 (mirror-image D-peptides added to the pool) vs run 2 vs run 1 vs the untrained bundle, on the real benchmark peptides (2,750 positions with the 189-class labels; TEST ONLY)."""
import os, csv, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; E = f"{D}/data/pilot_train_189/eval_v5"
WEAK = "DHI DIL DTY DAS DGN DSG".split(); D13 = "DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split()
def load(n): return list(csv.DictReader(open(f"{E}/ae_eval_{n}.tsv"), delimiter="\t"))
R = {n: load(n) for n in ("bundle", "v3", "v4", "v5") if os.path.exists(f"{E}/ae_eval_{n}.tsv")}
LAB = {"bundle": "new rows untrained", "v3": "run 1 (261 complexes)", "v4": "run 2 (1,429 complexes)", "v5": "run 3 (+7,534 mirror-image D-peptides)"}
COL = {"bundle": "#8d99ae", "v3": "#2a6f97", "v4": "#c9a227", "v5": "#b23a48"}
def cat(rows, c):
    x = [int(r["correct"]) for r in rows if r["category"] == c]; return 100 * np.mean(x), sum(x), len(x)
def rec(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows: d[r["ccd"]][0] += int(r["correct"]); d[r["ccd"]][1] += 1
    return d
f = plt.figure(figsize=(20, 9.0)); gs = f.add_gridspec(2, 3, height_ratios=[0.5, 2.4], hspace=0.32, wspace=0.3)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Pool of peptides (train split)", "pool peptides made of canonical or D-alpha residues:\n7,534 of the 37,105 training peptides", "#dde5b6"),
       ("Mirror image", "negate x of every atom, swap each residue for its mirror\ncounterpart (SER <-> DSN, TYR <-> DTY, ... 19 pairs)\n= a valid D-peptide with true D backbone geometry", "#ade8f4"),
       ("Decoder run 3", "same settings as run 2 + the mirrored peptides in the\ntraining side only (validation and test identical to run 2)", "#ffd6a5"),
       ("TEST: real PDB peptides", "2,750 positions never trained on,\nincluding 112 positions of the 13 new D-forms", "#f4c2c2")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 25; s.add_patch(FancyBboxPatch((x, 1), 23, 8, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 11.5, 7.5, t, ha="center", va="center", fontsize=10.5, fontweight="bold"); s.text(x + 11.5, 3.7, d, ha="center", va="center", fontsize=8)
for x0 in (24.2, 49.2, 74.2): s.annotate("", xy=(x0 + 1.5, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: teach the decoder real D-peptide geometry by mirroring pool peptides, then test on real peptides", fontsize=10.5, loc="left", fontweight="bold")
a = f.add_subplot(gs[1, 0]); cs = ["canonical", "trained_ncaa", "other_vocab_ncaa", "new_dform"]; lab = ["canonical", "32 trained\nnon-canonical", "other vocab", "13 new\nD-forms"]; ks = list(R)
for k, n in enumerate(ks):
    v = [cat(R[n], c)[0] for c in cs]; w = 0.8 / len(ks); a.bar(np.arange(4) + (k - (len(ks) - 1) / 2) * w, v, w, color=COL[n], edgecolor="#555", label=LAB[n])
    for i, x in enumerate(v): a.text(i + (k - (len(ks) - 1) / 2) * w, x + 1.0, f"{x:.0f}", ha="center", fontsize=7)
a.set_xticks(range(4)); a.set_xticklabels(lab, fontsize=9); a.set_ylim(0, 118); a.set_ylabel("real positions named correctly (%)"); a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
a.set_title("(a) Accuracy by group", fontsize=10.5, loc="left", fontweight="bold")
b = f.add_subplot(gs[1, 1]); recs = {n: rec(R[n]) for n in R}; x = np.arange(len(WEAK) + 1); ref = recs["v4"]
for k, n in enumerate(ks):
    v = [100 * recs[n][c][0] / max(1, recs[n][c][1]) for c in WEAK] + [100 * sum(recs[n][c][0] for c in WEAK) / max(1, sum(recs[n][c][1] for c in WEAK))]
    w = 0.8 / len(ks); b.bar(x + (k - (len(ks) - 1) / 2) * w, v, w, color=COL[n], edgecolor="#555")
b.set_xticks(x); b.set_xticklabels([f"{c}\n({ref[c][1]})" for c in WEAK] + [f"six pooled\n({sum(ref[c][1] for c in WEAK)})"], fontsize=8.5); b.set_ylim(0, 112); b.set_ylabel("recall on real positions (%)")
b.axhline(95, color="#b23a48", lw=1, ls="--"); b.text(len(WEAK) + 0.45, 96.5, "95% stop line", color="#b23a48", fontsize=8, ha="right")
for k, n in enumerate(ks):
    w_ = sum(recs[n][c][0] for c in WEAK); b.text(len(WEAK) + (k - (len(ks) - 1) / 2) * (0.8 / len(ks)), 100 * w_ / sum(ref[c][1] for c in WEAK) + 1.5, f"{w_}", ha="center", fontsize=7.5)
b.set_title("(b) The six weak D-forms (positions in brackets)", fontsize=10.5, loc="left", fontweight="bold")
c = f.add_subplot(gs[1, 2]); x = np.arange(len(D13));
for k, n in enumerate(("v4", "v5")):
    v = [100 * recs[n][d][0] / max(1, recs[n][d][1]) for d in D13]; c.bar(x + (k - 0.5) * 0.4, v, 0.4, color=COL[n], edgecolor="#555", label=LAB[n])
c.set_xticks(x); c.set_xticklabels([f"{d}\n{recs['v4'][d][1]}" for d in D13], fontsize=7.5); c.set_ylim(0, 112); c.set_ylabel("recall on real positions (%)"); c.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
c.set_title("(c) All 13 new D-forms, run 2 vs run 3 (positions below the code)", fontsize=10.5, loc="left", fontweight="bold")
n2 = {n: cat(R[n], "new_dform") for n in R}; w6 = {n: sum(recs[n][d][0] for d in WEAK) for n in R}
f.text(0.01, 0.005, f"Real positions are test only. New D-forms: run 2 {n2['v4'][1]}/{n2['v4'][2]}, run 3 {n2['v5'][1]}/{n2['v5'][2]}; weak six pooled: run 2 {w6['v4']}/34, run 3 {w6['v5']}/34. One run each; the saved weights of each run are the epoch with the best held-out synthetic loss (run 3: epoch 1). Positions per code are small (1 to 21).", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig45_decoder_run3_mirror.png", bbox_inches="tight", dpi=130); plt.close(f)
print({n: [round(cat(R[n], c_)[0], 1) for c_ in cs] for n in R}, "weak six:", w6)
