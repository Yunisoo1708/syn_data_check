"""Figure: NCFlow trained with real data (ft2, rl) vs trained on synthetic data only (ft_syn, rl_syn), same held-out sets, strict per-residue test."""
import os, matplotlib.pyplot as plt, numpy as np
M = ["raw", "ft2\n(real+syn)", "ft_syn\n(syn only)", "rl\n(real+syn)", "rl_syn\n(syn only)"]
C = ["#8d99ae", "#b23a48", "#2a6f97", "#d98b95", "#7fb3d5"]
d = {  # from ncf_summary.py, 16 Nov 2026: strict, any-of-4, bad bond, link off, backbone err
 "real (n=89)":      {"strict": [28, 72, 75, 74, 74], "any4": [50, 91, 90, 93, 91], "link": [62, 17, 17, 15, 16], "bb": [0.49, 0.26, 0.27, 0.30, 0.28]},
 "synthetic (n=180)": {"strict": [10, 43, 48, 51, 52], "any4": [19, 81, 80, 81, 82], "link": [82, 29, 28, 25, 26], "bb": [1.0, 0.50, 0.56, 0.61, 0.67]}}
def fig(OUT):
    f, ax = plt.subplots(2, 3, figsize=(16, 8.4), gridspec_kw=dict(wspace=0.28, hspace=0.5))
    for r, (name, v) in enumerate(d.items()):
        for c, (k, ttl, yl) in enumerate([("strict", "strict pass, top sample", "residues passing (%)"), ("any4", "strict pass, best of 4 samples", "residues passing (%)"), ("link", "peptide link off", "residues with defect (%)")]):
            a = ax[r, c]; a.bar(range(5), v[k], color=C, edgecolor="#555")
            for i, x in enumerate(v[k]): a.text(i, x + 1.5, f"{x}", ha="center", fontsize=9)
            a.set_xticks(range(5)); a.set_xticklabels(M, fontsize=7.5); a.set_ylim(0, 100); a.set_ylabel(yl, fontsize=9)
            a.set_title(f"{name}: {ttl}", fontsize=10, loc="left", fontweight="bold")
    f.text(0.01, 0.01, "Same held-out cases for every model. Red = trained with real residues (252 real + 3,950 synthetic); blue = trained on the 3,952 synthetic cases only. Raw = released weights.", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig28_syn_only_vs_real.png"), bbox_inches="tight"); plt.close(f)
if __name__ == "__main__": fig("figures")
