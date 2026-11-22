"""Per non-canonical position test of the DECODER on the TRUE latents of 468 real complexes (1,066 NCAA positions; gen_test/diag_per_code.py): vanilla (untouched bundle decoder), run 3, run 5b, run 8. (a-d) by residue group; (e) per code for run 8 against vanilla; (f) residue named correctly when the identity is hidden."""
import json, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; M = [("vanilla", "vanilla (untouched)", "#bbbbbb"), ("run3", "run 3", "#6a4c93"), ("run5b", "run 5b", "#219ebc"), ("run8", "run 8 (default)", "#2a9d8f")]
P = {m: json.load(open(f"{D}/gen_test/diag/percode_{m}.json"))["rows"] for m, *_ in M}
G = ["D-forms", "general codes", "other non-canonical", "all"]
def sel(m, g): return [x for x in P[m] if g == "all" or x["group"] == g]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.4, 1, 1], hspace=0.55, wspace=0.27)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#dde5b6", ec="#555"))
s.text(50, 5, "Test: encode each real complex with the frozen encoder, decode the TRUE latent (no denoiser), compare every non-canonical residue with the native one (468 complexes with an NCAA, 1,066 positions, never trained on).\nResidue identity GIVEN to the decoder unless stated. RMSD = all heavy atoms of the residue (A); bond error = mean change in length of the covalent bonds inside the residue (A); handedness = C-alpha signed volume equals the native sign.", ha="center", va="center", fontsize=8.4)
s.set_title("How well is each non-canonical residue rebuilt? (decoder alone, specification given)", fontsize=10.5, loc="left", fontweight="bold")
def grouped(ax, fn, title, ylabel, fmt, ylim=None, log=False):
    w = 0.2
    for j, (m, lab, col) in enumerate(M):
        vals = [fn(sel(m, g)) for g in G]; xs = np.arange(len(G)) + (j - 1.5) * w
        ax.bar(xs, vals, w * 0.95, color=col, edgecolor="#333", linewidth=0.5, label=lab)
        for x, v in zip(xs, vals): ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(G))); ax.set_xticklabels(["D-forms\n(n=336)", "general codes\n(n=250)", "other\n(n=470)", "all\n(n=1,056)"], fontsize=8); ax.set_title(title, fontsize=9.2, loc="left", fontweight="bold"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
ax = f.add_subplot(gs[1, 0]); grouped(ax, lambda r: float(np.median([x["rmsd_A"] for x in r])), "(a) median all-atom RMSD to the native residue (A, lower = better)", "A", "{:.2f}", (0, 5.5)); ax.legend(fontsize=7, frameon=False, loc="upper right")
ax = f.add_subplot(gs[1, 1]); grouped(ax, lambda r: 100 * np.mean([x["rmsd_A"] < 1.0 for x in r]), "(b) residues rebuilt within 1 A of the native (%)", "%", "{:.0f}", (0, 112))
ax = f.add_subplot(gs[1, 2]); grouped(ax, lambda r: 100 * np.mean([x["atoms_ok"] for x in r]), "(c) decoder emits exactly the native atoms (%)", "%", "{:.0f}", (0, 112))
ax = f.add_subplot(gs[2, 0]); grouped(ax, lambda r: float(np.nanmedian([x["bond_dev_A"] for x in r])), "(d) bond-length error inside the residue (A, lower = better)", "A", "{:.2f}", (0, 3.2))
ax = f.add_subplot(gs[2, 1]); grouped(ax, lambda r: 100 * np.mean([x["chir_ok"] for x in r if x["chir_ok"] is not None]), "(e) C-alpha handedness right (%, 50 = coin flip)", "%", "{:.0f}", (0, 112))
ax = f.add_subplot(gs[2, 2]); cnt = collections.Counter(x["code"] for x in P["run8"]); codes = [c for c, n in cnt.most_common(26)]
r8 = [float(np.median([x["rmsd_A"] for x in P["run8"] if x["code"] == c])) for c in codes]; rv = [float(np.median([x["rmsd_A"] for x in P["vanilla"] if x["code"] == c])) for c in codes]
xs = np.arange(len(codes)); ax.bar(xs - 0.2, rv, 0.4, color="#bbbbbb", edgecolor="#333", linewidth=0.4, label="vanilla"); ax.bar(xs + 0.2, r8, 0.4, color="#2a9d8f", edgecolor="#333", linewidth=0.4, label="run 8"); ax.axhline(1.0, color="#c1121f", lw=0.8, ls="--")
ax.set_xticks(xs); ax.set_xticklabels([f"{c} ({cnt[c]})" for c in codes], rotation=70, fontsize=6.5); ax.set_ylabel("median RMSD (A)"); ax.legend(fontsize=7, frameon=False); ax.set_title("(f) the 26 most frequent codes (n): vanilla vs run 8", fontsize=9.2, loc="left", fontweight="bold")
a = [x for x in P["run8"]]; nh = 100 * np.mean([x["name_ok"] for x in a])
f.text(0.01, 0.005, "Result: fine-tuning moves the median residue error from 3.9 to 0.65 A (0 to 82% within 1 A), atoms from 8 to 97% correct, bond error from 2.4 to 0.27 A, handedness from a coin flip (54%) to 89-92%. Weak codes: CGU, PTR, STA, KCR, SLL (1.1-1.4 A). With the identity hidden the decoder names only " + f"{nh:.0f}" + "% of the residues from the latent: the specification must be given.", fontsize=8.4, fontweight="bold")
f.savefig(f"{D}/figures/fig66_per_code_decoder.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig66")
