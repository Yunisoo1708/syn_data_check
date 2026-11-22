"""Figure: raw vs synthetic-only fine-tuned NCFlow models, strict per-residue test on held-out real residues and held-out synthetic cases."""
import os, matplotlib.pyplot as plt, numpy as np
C = {"raw": "#8d99ae", "ft_syn": "#2a6f97", "rl_syn": "#c9a227"}
# values from ncf_summary.py on strict_real_new.tsv / strict_syn_new.tsv (16 Nov 2026)
sets = ["real_test (n=89)", "real_train (n=251)", "syn_test (n=180)"]
strict = {"raw": [27, 30, None], "ft_syn": [75, 76, 48], "rl_syn": [74, 78, 52]}
anyS = {"raw": [49, 53, None], "ft_syn": [90, 95, 80], "rl_syn": [91, 94, 82]}
bad = {"raw": [36, 36], "ft_syn": [3, 3], "rl_syn": [1, 0]}
link = {"raw": [63, 60], "ft_syn": [17, 12], "rl_syn": [16, 12]}
pro = {"raw": [3, 23, 3, 8, 0, 3], "ft_syn": [14, 25, 7, 13, 0, 4], "rl_syn": [11, 25, 6, 13, 0, 4]}
def fig(OUT):
    f, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.3))
    a = ax[0]; w = 0.27
    for k, m in enumerate(C):
        for i, v in enumerate(strict[m]):
            if v is None: continue
            a.bar(i + (k - 1) * w, v, w, color=C[m], edgecolor="#555", label=m if i == 0 else None)
            a.text(i + (k - 1) * w, v + 1.5, f"{v}", ha="center", fontsize=8.5)
    a.set_xticks(range(3)); a.set_xticklabels(sets, fontsize=9); a.set_ylim(0, 105); a.set_ylabel("residues passing the strict chemistry test (%)")
    a.legend(frameon=False, fontsize=9); a.set_title("(a) Strict pass rate, top-pLDDT sample", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[1]
    lab = ["bad bond\n(real_test)", "bad bond\n(real_train)", "link off\n(real_test)", "link off\n(real_train)"]
    for k, m in enumerate(C):
        vals = bad[m] + link[m]
        b.bar(np.arange(4) + (k - 1) * w, vals, w, color=C[m], edgecolor="#555")
        for i, v in enumerate(vals): b.text(i + (k - 1) * w, v + 1, f"{v}", ha="center", fontsize=8.5)
    b.set_xticks(range(4)); b.set_xticklabels(lab, fontsize=8.5); b.set_ylabel("residues with the defect (%)"); b.set_ylim(0, 72)
    b.set_title("(b) Bond and peptide-link defects", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[2]
    codes = ["HYP", "FP9", "DPR"]
    for k, m in enumerate(C):
        v = pro[m]; rate = [100 * v[2 * j] / v[2 * j + 1] for j in range(3)]
        c.bar(np.arange(3) + (k - 1) * w, rate, w, color=C[m], edgecolor="#555")
        for j in range(3): c.text(j + (k - 1) * w, rate[j] + 1.5, f"{v[2*j]}/{v[2*j+1]}", ha="center", fontsize=8)
    c.set_xticks(range(3)); c.set_xticklabels(codes); c.set_ylim(0, 105); c.set_ylabel("strict pass (%), residue count above bar")
    c.set_title("(c) Proline analogues", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.03, "Training of ft_syn and rl_syn used only the 3,952 synthetic cases; real_test and real_train residues were never used to train. 'syn_test' cases are held-out synthetic ones (raw not rerun on them here).", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig27_ncflow_synthetic_only.png"), bbox_inches="tight"); plt.close(f)
if __name__ == "__main__": fig("figures")
