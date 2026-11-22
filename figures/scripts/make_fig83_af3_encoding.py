"""Fig 83: does a different input encoding fix AF3's D-handedness? (Results.MD 84, addendum). Modification-in-chain (310-model test) vs peptide as one bonded CCD ligand (30-job pilot: 10 natives, 20 S1 designs with D-forms)."""
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
f, (s, ax) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[0.42, 1], hspace=0.25))
s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10); s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, "Two ways to give AF3 a peptide with D-residues:\n(A) protein chain + CCD modification at the position (the encoding of the NCAA dataset)   (B) the peptide as one ligand of CCD residues, bonded C to N.\nQuestion: does (B) keep the D hand?  Measured: share of D-residues with the right hand, L-residues with the right hand, peptide bonds intact.", ha="center", va="center", fontsize=8)
s.set_title("Does another AF3 input encoding fix the D-handedness?", fontsize=11, loc="left", fontweight="bold")
labels = ["D-residues,\nnative peptides", "D-residues,\nS1 designs", "canonical L\nresidues", "peptide bonds\nintact"]
A = [33, 14, 100, 98]; B = [61, 0, 91, 80]; nA = ["9 jobs", "65 jobs", "2,138 res.", "all"]; nB = ["9 jobs", "20 jobs", "90 res.", "215 bonds"]
x = np.arange(4); w = 0.38
ax.bar(x - w / 2, A, w, color="#6a4c93", edgecolor="#333", label="(A) modification in the chain"); ax.bar(x + w / 2, B, w, color="#e76f51", edgecolor="#333", label="(B) bonded ligand")
for i in range(4):
    ax.text(i - w / 2, A[i] + 1.5, f"{A[i]}%\n{nA[i]}", ha="center", fontsize=8.4); ax.text(i + w / 2, B[i] + 1.5, f"{B[i]}%\n{nB[i]}", ha="center", fontsize=8.4)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 125); ax.set_ylabel("%"); ax.axhline(50, color="#c1121f", ls="--", lw=0.8); ax.legend(frameon=False, fontsize=9, loc="upper right")
f.text(0.01, -0.03, "Reading: (B) helps the native D-residues (33% to 61%, but only 9 jobs) and not the S1 designs (0 of 20 jobs, 12 of the 21 D-residues are D-isoleucine, which has a second stereocentre), and it costs a lot elsewhere:\nL-residue hand 100% to 91%, peptide bonds 98% to 80%. So the encoding is not a fix; AF3 stays unreliable for D-forms. Small pilot, one seed, best-ranked sample.", fontsize=8.8, fontweight="bold", va="top")
f.savefig("C:/Research/RFD4_fine_tune/figures/fig83_af3_encoding.png", bbox_inches="tight", dpi=130); print("saved")
