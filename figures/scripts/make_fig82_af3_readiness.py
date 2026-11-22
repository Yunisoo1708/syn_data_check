"""Fig 82: readiness of AF3 for D-amino acids and canonical-like NCAAs (Results.MD 84). Data: af3_filter/readiness.json, readiness_rows.json (readiness_af3.py on the 310 ipTM-test models)."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
A = "C:/Research/RFD4_fine_tune/af3_filter"; S = json.load(open(f"{A}/readiness.json")); BC = S["by_code"]; BF = S["by_family"]
FC = {"canonical L": "#2a9d8f", "D-alpha": "#e76f51", "N-alkyl": "#264653", "side-chain modified": "#e9c46a", "PTM": "#8ab17d", "beta/gamma": "#bbbbbb", "alpha,alpha-disubst.": "#999999"}
FAMOF = {}
for fam, codes in {"D-alpha": "DAL DAR DAS DCY DGL DGN DHI DIL DLE DLY DPN DPR DSG DSN DTH DTR DTY DVA", "N-alkyl": "SAR 04Q MAA MLE MVA", "side-chain modified": "23F 4FB FP9 HYP ORN DBU", "PTM": "SEP TPO PTR", "beta/gamma": "B3L B3A BAL XCP STA", "alpha,alpha-disubst.": "AIB 2ML 1AC"}.items():
    for c in codes.split(): FAMOF[c] = fam
f = plt.figure(figsize=(19, 12.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.3, 1, 0.12], hspace=0.55, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10); s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, "Is AF3 ready for D-amino acids and for NCAAs with a canonical-like backbone?  Check on the 310 AF3 models of Results.MD 83 (2,573 peptide residues, 42 codes, no code rejected by AF3).\n"
       "Three checks per residue of the peptide: (1) hand of the C-alpha (N, CA, C, CB triple product) against the sign of the chemical definition of the code (the experimental D-residues of the test pockets match it 100%: DAL 7/7, DIL 4/4, DPN 3/3, DTR 5/5),\n"
       "(2) is the peptide bond to the next residue present and 1.20-1.50 A long, (3) AF3's confidence (pLDDT) of the residue against the canonical residues of the same peptide.", ha="center", va="center", fontsize=8.8)
s.set_title("AF3 readiness for D-forms and canonical-like NCAAs", fontsize=10.5, loc="left", fontweight="bold")
# (a) chirality per code
ax = f.add_subplot(gs[1, :2])
codes = [c for c, v in BC.items() if v["chir_ok"] is not None and v["n_chir"] >= 5 and c in FAMOF or c in ("ALA", "LEU", "PHE")]
codes = [c for c in codes if BC[c]["chir_ok"] is not None]; codes = sorted(codes, key=lambda c: (FAMOF.get(c, "canonical L") != "canonical L", FAMOF.get(c, ""), BC[c]["chir_ok"]))
cols = [FC[FAMOF.get(c, "canonical L")] for c in codes]; ax.bar(range(len(codes)), [100 * BC[c]["chir_ok"] for c in codes], color=cols, edgecolor="#333", lw=0.4)
for i, c in enumerate(codes): ax.text(i, 100 * BC[c]["chir_ok"] + 1.5, "n=%d" % BC[c]["n_chir"], ha="center", fontsize=7, rotation=90, va="bottom")
ax.axhline(50, color="#c1121f", ls="--", lw=0.8); ax.text(len(codes) - 0.5, 51.5, "coin flip", color="#c1121f", fontsize=8, ha="right")
ax.set_xticks(range(len(codes))); ax.set_xticklabels(codes, rotation=60, fontsize=8); ax.set_ylim(0, 125); ax.set_ylabel("% of residues with the right hand")
ax.set_title("(a) hand of the C-alpha, per CCD code (codes with 5 or more residues)", fontsize=9.4, loc="left", fontweight="bold")
import matplotlib.patches as mp; ax.legend(handles=[mp.Patch(color=c, label=k) for k, c in FC.items() if k in set(FAMOF.get(x, "canonical L") for x in codes)], fontsize=7.6, frameon=False, ncol=2, loc="upper right")
# (b) bonds + plddt per family
ax = f.add_subplot(gs[1, 2]); fams = [k for k in ("canonical L", "D-alpha", "N-alkyl", "side-chain modified", "PTM") if k in BF]; x = np.arange(len(fams)); w = 0.38
ax.bar(x - w / 2, [100 * BF[k]["bond_ok"] for k in fams], w, color=[FC[k] for k in fams], edgecolor="#333", lw=0.4, label="peptide bond to next residue ok (%)")
for i, k in enumerate(fams): ax.text(i - w / 2, 100 * BF[k]["bond_ok"] + 0.5, "%.0f%%" % (100 * BF[k]["bond_ok"]), ha="center", fontsize=7.8)
ax.set_ylim(60, 106); ax.set_xticks(x); ax.set_xticklabels([k.replace(" ", "\n") for k in fams], fontsize=7.8); ax.set_ylabel("% of residues with the peptide bond intact")
a2 = ax.twinx(); a2.scatter(x + w / 2, [0] + [S["plddt_minus_canonical_same_peptide"].get(k, np.nan) for k in fams[1:]], marker="D", color="#333", s=36, zorder=4); a2.set_ylim(-5, 15); a2.set_ylabel("pLDDT minus canonical residues of the same peptide (diamonds)")
ax.set_title("(b) peptide bonds and confidence", fontsize=9.4, loc="left", fontweight="bold")
f.text(0.01, 0.02, "Reading: AF3 accepts every code and builds the bond into the chain, and it is reliable for L-residues (100%) and phosphorylated residues (SEP, TPO: 24 of 24), but it does NOT enforce D-handedness: D-alpha residues have the right hand in only 17% of the cases\n"
       "(DIL 9% of 80, DHI 0 of 13, DTY 2 of 15, DAL 3 of 14; DPN 5 of 11 is a coin flip), and N-methylated MAA 6 of 11. Peptide bonds are fine for D-forms (95%) and weaker for N-methyl residues (92%, MAA 73%). AF3 is confident about the wrong hand (pLDDT +4.6 over the canonical residues).\n"
       "Consequence: an AF3 ipTM or pose for a peptide with D-forms is computed for a molecule with the wrong chirality at most D positions; use it only after a chirality check or with the mirror-image trick, as the earlier mirror repair does for RF3.", fontsize=8.6, fontweight="bold", va="top")
f.savefig("C:/Research/RFD4_fine_tune/figures/fig82_af3_readiness.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved")
