"""Figure: can the synthetic complexes be used by the RFD4 NCAA training recipe? Stage 0 check, vocabulary coverage, smoke tests."""
import os, matplotlib.pyplot as plt
OUT = "C:/Research/RFD4_fine_tune/figures"
f, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1, 1.3]))
a = ax[0]; v = [878, 4, 2]; a.bar(range(3), v, color=["#2a6f97", "#c9a227", "#b23a48"], edgecolor="#555")
for i, x in enumerate(v): a.text(i, x + 15, str(x), ha="center", fontsize=10, fontweight="bold")
a.set_xticks(range(3)); a.set_xticklabels(["pass", "chirality\nfail", "link\nfail"]); a.set_ylabel("non-canonical residues"); a.set_title("(a) Stage 0 geometry check\n(geo_check.py, peptide-only files)", fontsize=10.5, loc="left", fontweight="bold")
b = ax[1]; b.bar([0, 1], [884, 968], color=["#2a6f97", "#8d99ae"], edgecolor="#555")
for i, x in enumerate([884, 968]): b.text(i, x + 15, f"{x}\n({100*x/1852:.0f}%)", ha="center", fontsize=10, fontweight="bold")
b.set_xticks([0, 1]); b.set_xticklabels(["in the 176-class\nvocabulary (18 codes)", "not in the vocabulary\n(13 D-forms)"]); b.set_ylim(0, 1150); b.set_ylabel("accepted swaps"); b.set_title("(b) Vocabulary coverage of the swaps", fontsize=10.5, loc="left", fontweight="bold")
c = ax[2]; c.axis("off")
rows = [("Peptide-only files (clean_pep_cif)", "2 steps run, loss 92.7 -> 47.0", "#2a6f97"), ("Full complexes, continuous numbering", "FAILS: bond 6.57 A across chain gap", "#b23a48"),
        ("Full complexes, split at gaps", "validator passes; GPU out of memory (8 GB)", "#c9a227"), ("Pocket crop 12 A, split at gaps", "2 steps run, loss 42.4 -> 50.4 (8 GB card at limit)", "#2a6f97")]
for i, (t, r, col) in enumerate(rows):
    y = 0.85 - i * 0.22; c.add_patch(plt.Rectangle((0, y - 0.07), 0.03, 0.14, color=col, transform=c.transAxes)); c.text(0.06, y + 0.03, t, fontsize=10.5, fontweight="bold", transform=c.transAxes); c.text(0.06, y - 0.05, r, fontsize=9.5, transform=c.transAxes)
c.set_title("(c) Smoke tests of ncaa_fine_tuning.py (2 steps, batch 2)", fontsize=10.5, loc="left", fontweight="bold")
f.text(0.01, -0.03, "Loss values after 2 steps say only that training runs, not that the model learns. 6 of the 264 files hold only codes outside the vocabulary and were not covered by Stage 0.", fontsize=9, fontweight="bold")
f.savefig(os.path.join(OUT, "fig34_training_check.png"), bbox_inches="tight"); plt.close(f)
