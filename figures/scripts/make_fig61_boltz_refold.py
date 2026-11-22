"""Boltz-2 refold with the receptor held by a forced template: 20 run 8 designs (denoiser G, 2 per complex, 10 complexes) + 10 native peptides. (b) receptor pocket RMSD per job, AF3 (section 63) vs Boltz-2; (c) peptide CA RMSD of the refold to the design and to the native peptide per complex (dots = native control refolded vs native); (d) ipTM."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; B = json.load(open(f"{D}/gen_test/refold/boltz_summary.json")); A = json.load(open(f"{D}/gen_test/refold/refold_summary.json"))
names = list(B); cx = sorted({n.split("-")[0] for n in names})
f = plt.figure(figsize=(17, 9.5)); gs = f.add_gridspec(2, 3, height_ratios=[0.45, 1], hspace=0.5, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Inputs", "same 30 jobs as the AF3 test: designed peptide (non-canonical residues as CCD),\nfull native receptor chain, receptor structure given as a FORCED template\n(Boltz-2 template force, threshold 2 A); control = native peptide", "#ffd6a5"),
       ("Boltz-2 on the box", "single sequence (no MSA), potentials on, 3 diffusion samples,\nbest sample by Boltz confidence; 5060 Ti", "#ade8f4"),
       ("Checks", "receptor pocket RMSD (gate 4 A); peptide CA RMSD of the refold to the\ndesign and to the native peptide (agree < 3 A); identity kept;\nD handedness; ipTM", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: refold the designs with Boltz-2 while holding the receptor fixed", fontsize=10.5, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 0]); rng = np.random.default_rng(1)
for j, (nm, SS, c) in enumerate((("AF3", A, "#e76f51"), ("Boltz-2", B, "#2a9d8f"))):
    v = [SS[n]["rec_rmsd_pocket"] for n in names]; ax.scatter(rng.normal(j, 0.06, len(v)), v, color=c, edgecolor="#333", s=36, label=f"{nm}: {sum(x < 4 for x in v)} of {len(v)} below 4 A")
ax.axhline(4, color="#c0392b", ls="--"); ax.set_xticks([0, 1]); ax.set_xticklabels(["AF3", "Boltz-2"]); ax.set_ylabel("receptor pocket RMSD to native (A)"); ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.set_title("(b) is the receptor held? (30 jobs each)", fontsize=9.3, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 1]); w = 0.27
for k, c in enumerate(cx):
    for j, kk in enumerate(("d0", "d1")):
        n = f"{c}-{kk}"; ax.bar(k + (j - 0.5) * w * 1.1 - 0.15, B[n]["pep_ca_rmsd_design"], w, color="#2a9d8f", edgecolor="#333", linewidth=0.4, label="design refold vs design" if (k, j) == (0, 0) else None)
        ax.bar(k + (j - 0.5) * w * 1.1 + 0.25, B[n]["pep_ca_rmsd_native"], w, color="#e9c46a", edgecolor="#333", linewidth=0.4, label="design refold vs native peptide" if (k, j) == (0, 0) else None)
    ax.scatter([k + 0.55], [B[f"{c}-native"]["pep_ca_rmsd_native"]], marker="D", color="#264653", zorder=5, s=34, label="NATIVE control refold vs native" if k == 0 else None)
ax.axhline(3, color="#c0392b", ls="--"); ax.set_xticks(range(len(cx))); ax.set_xticklabels([c.split("_")[0] for c in cx], rotation=45, fontsize=8); ax.set_ylabel("peptide CA RMSD (A)"); ax.legend(fontsize=7, frameon=False, loc="upper right"); ax.set_title("(c) peptide pose, receptor held (bars: design 0 and 1 per complex)", fontsize=9.3, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 2])
for j, (nm, SS) in enumerate((("AF3", A), ("Boltz-2", B))):
    for q, (kd, c) in enumerate((("design", "#2a9d8f"), ("native", "#bbbbbb"))):
        v = [SS[n]["iptm"] for n in names if SS[n]["kind"] == kd]; ax.scatter(rng.normal(j * 2.5 + q * 0.9, 0.06, len(v)), v, color=c, edgecolor="#333", s=34)
        ax.text(j * 2.5 + q * 0.9, 1.0, f"{np.median(v):.2f}", ha="center", fontsize=8)
ax.set_xticks([0, 0.9, 2.5, 3.4]); ax.set_xticklabels(["AF3\ndesigns", "AF3\nnatives", "Boltz\ndesigns", "Boltz\nnatives"], fontsize=8); ax.set_ylim(0, 1.08); ax.set_ylabel("ipTM (median on top)"); ax.set_title("(d) predictor confidence", fontsize=9.3, loc="left", fontweight="bold")
dg = [B[n] for n in names if B[n]["kind"] == "design"]; nv = [B[n] for n in names if B[n]["kind"] == "native"]
agree = sum(x["pep_ca_rmsd_design"] < 3 for x in dg); nat_ok = sum(x["pep_ca_rmsd_native"] < 3 for x in nv)
f.text(0.01, 0.005, f"Boltz-2 with a held receptor: receptor kept in {sum(x['rec_rmsd_pocket'] < 4 for x in B.values())} of 30 jobs; the refold agrees with the design (CA RMSD < 3 A) in {agree} of 20 designs; the native peptide is recovered in {nat_ok} of 10 controls. Specified residues kept 100% (forced by input); D handedness right in every refold (native controls too).", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig61_boltz_refold.png", bbox_inches="tight", dpi=130); plt.close(f)
for k in (3, 5):
    print(f"CA RMSD < {k} A: design refold vs design {sum(x['pep_ca_rmsd_design'] < k for x in dg)}/20, vs native {sum(x['pep_ca_rmsd_native'] < k for x in dg)}/20 | native control {sum(x['pep_ca_rmsd_native'] < k for x in nv)}/10 | design itself vs native {sum(x.get('design_ca_rmsd_native', 99) < k for x in dg)}/20")
print("median ipTM Boltz designs %.2f natives %.2f | AF3 designs %.2f natives %.2f" % (np.median([x['iptm'] for x in dg]), np.median([x['iptm'] for x in nv]), np.median([A[n]['iptm'] for n in names if A[n]['kind'] == 'design']), np.median([A[n]['iptm'] for n in names if A[n]['kind'] == 'native'])))
print("median pocket RMSD Boltz %.2f, bond_ok mean %.2f, D_right values %s" % (np.median([x['rec_rmsd_pocket'] for x in B.values()]), np.mean([x['bond_ok'] for x in B.values()]), sorted({x['D_right'] for x in B.values()}, key=str)))
print("median refold-vs-design %.1f, refold-vs-native (designs) %.1f, native control %.1f, design-vs-native %.1f" % (np.median([x['pep_ca_rmsd_design'] for x in dg]), np.median([x['pep_ca_rmsd_native'] for x in dg]), np.median([x['pep_ca_rmsd_native'] for x in nv]), np.median([x['design_ca_rmsd_native'] for x in dg])))
