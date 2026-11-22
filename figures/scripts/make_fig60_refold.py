"""AF3 refold of 20 run 8 designs (G denoiser, 2 per complex, 10 complexes) + 10 native peptides as control, on the box. (a) method; (b) receptor pocket RMSD of each AF3 model to the native receptor (validity gate: template not followed above 4 A); (c) peptide CA RMSD of valid refolds to the native peptide and to the design; (d) AF3 ipTM designs vs natives."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/refold/refold_summary.json"))
names = list(S); kind = [S[n]["kind"] for n in names]
f = plt.figure(figsize=(17, 9.5)); gs = f.add_gridspec(2, 3, height_ratios=[0.45, 1], hspace=0.45, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Inputs", "receptor = full native chain, sequence + native structure as TEMPLATE;\npeptide = designed sequence, non-canonical residues as CCD modifications;\ncontrol = the NATIVE peptide with the same receptor (10 of 11 complexes)", "#ffd6a5"),
       ("AF3 on the box", "single sequence (no MSA), 3 diffusion samples, 10 recycles, 5060 Ti;\n30 jobs: 20 designs (run 8 decoder, denoiser G, designs 0 and 1) + 10 natives", "#ade8f4"),
       ("Checks", "gate: AF3 receptor pocket within 4 A of native (else the pose cannot be compared);\nthen peptide CA RMSD to the native peptide and to the design,\nidentity kept, ipTM", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: refold designs with AlphaFold 3 and compare with the design and the native peptide", fontsize=10.5, loc="left", fontweight="bold")
col = {"design": "#2a9d8f", "native": "#bbbbbb"}
ax = f.add_subplot(gs[1, 0]); order = np.argsort([S[n]["rec_rmsd_pocket"] for n in names])
ax.barh(range(len(names)), [S[names[i]]["rec_rmsd_pocket"] for i in order], color=[col[kind[i]] for i in order], edgecolor="#333", linewidth=0.4); ax.axvline(4, color="#c0392b", ls="--"); ax.text(4.2, 1, "gate 4 A", color="#c0392b", fontsize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels([names[i] for i in order], fontsize=6); ax.set_xlabel("receptor pocket RMSD to native (A)"); ax.set_title("(b) did AF3 keep the receptor? (grey = native control)", fontsize=9.3, loc="left", fontweight="bold")
ok = [n for n in names if S[n]["rec_rmsd_pocket"] < 4.0]
ax = f.add_subplot(gs[1, 1]); w = 0.38
dn = [n for n in ok if S[n]["kind"] == "design"]; nn = [n for n in ok if S[n]["kind"] == "native"]
ax.bar(np.arange(len(dn)) - w / 2, [S[n]["pep_ca_rmsd_design"] for n in dn], w, color="#2a9d8f", edgecolor="#333", label="refold vs design")
ax.bar(np.arange(len(dn)) + w / 2, [S[n]["pep_ca_rmsd_native"] for n in dn], w, color="#e9c46a", edgecolor="#333", label="refold vs native peptide")
for i, n in enumerate(nn): ax.scatter([], [])
ax.set_xticks(range(len(dn))); ax.set_xticklabels(dn, rotation=60, ha="right", fontsize=7); ax.set_ylabel("peptide CA RMSD (A)"); ax.legend(fontsize=8, frameon=False)
nat_r = {n: S[n]["pep_ca_rmsd_native"] for n in nn}; ax.set_title("(c) valid refolds only (%d of 20 designs): natives refolded: %s" % (len(dn), ", ".join(f"{k.split('-')[0]} {v:.1f}" for k, v in nat_r.items())), fontsize=8, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 2])
for k, c in (("design", "#2a9d8f"), ("native", "#bbbbbb")):
    v = [S[n]["iptm"] for n in names if S[n]["kind"] == k]; ax.scatter(np.random.default_rng(0).normal(0 if k == "design" else 1, 0.05, len(v)), v, color=c, edgecolor="#333", s=40, label=f"{k} (median {np.median(v):.2f})")
ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["designs", "natives"]); ax.set_ylabel("AF3 ipTM"); ax.legend(fontsize=8, frameon=False, loc="lower center"); ax.set_title("(d) AF3 confidence", fontsize=9.3, loc="left", fontweight="bold")
f.text(0.01, 0.005, "Result: AF3 did not keep the template receptor in most jobs (even for the native controls), so only a few pose comparisons are valid; the test is inconclusive on pose. Specified residues are kept in 100% of refolds (forced by the CCD input).", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig60_af3_refold.png", bbox_inches="tight", dpi=130); plt.close(f)
dg = [S[n] for n in names if S[n]["kind"] == "design"]; nv = [S[n] for n in names if S[n]["kind"] == "native"]
print("designs: n", len(dg), "valid", sum(x["rec_rmsd_pocket"] < 4 for x in dg), "median iptm %.2f" % np.median([x["iptm"] for x in dg]), "| natives valid", sum(x["rec_rmsd_pocket"] < 4 for x in nv), "median iptm %.2f" % np.median([x["iptm"] for x in nv]))
