"""Figure: readiness board for the first batch of complex data for the denoiser fine-tune (status from Results.MD sections 4 to 24)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

G, A, R = "#d8f0dc", "#fff3cd", "#f8d7da"
ROWS = [
    ("Peptide-only structures that pass the strict test", "46,379 in the pool (all 176 classes covered)", G),
    ("Model that understands non-canonical backbones", "autoencoder recovers 98.5% of residues, 99.6% of non-canonical ones", G),
    ("Complex pose for a given receptor (RF3 + receptor template)", "59% within 4 A of the real pose, 84% at the right site (32 complexes)", A),
    ("Chemistry of those complexes", "NCFlow repair: 15 -> 31 of 32 peptides pass the strict test; interface kept", G),
    ("Interaction-preserving conversion of a real complex", "rule not built into each swap yet: 8-10 of 31 finished peptides meet it", A),
    ("Energy sanity check (relaxed interface energy)", "works on natural complexes (Ala scan: 3.0 vs 0.1 REU); control and Tillicum copy still running", A),
    ("Independent pose check (re-dock the converted sequence)", "not run", R),
    ("Receptor diversity", "about 110 receptors in hand (83 benchmark + 30 natural), many peptide variants each", A),
    ("Training recipe for the denoiser on complexes", "inputs, targets, evaluation not defined or tested yet", R),
    ("Cost of a pilot batch (a few hundred complexes)", "free GPUs and CPUs: 0 USD; an H200 for RF3 would be about 3 USD", G),
]


def fig(OUT, C):
    f = plt.figure(figsize=(15, 8.2))
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15); ax.set_ylim(0, 9.4); ax.axis("off")
    ax.text(0.3, 9.0, "Are we ready to generate the first batch of data for the denoiser?", fontsize=14, fontweight="bold")
    ax.text(0.3, 8.6, "Green = ready, amber = usable with a known gap, red = missing. Verdict: ready for a small pilot batch, not for the large one.", fontsize=10)
    for i, (a, b, c) in enumerate(ROWS):
        y = 7.7 - i * 0.74
        ax.add_patch(FancyBboxPatch((0.3, y), 5.6, 0.62, boxstyle="round,pad=0.02", fc="#f8f9fa", ec="#adb5bd"))
        ax.text(0.45, y + 0.31, a, va="center", fontsize=9.4)
        ax.add_patch(FancyBboxPatch((6.0, y), 8.7, 0.62, boxstyle="round,pad=0.02", fc=c, ec="#adb5bd"))
        ax.text(6.15, y + 0.31, b, va="center", fontsize=9.4)
    ax.text(0.3, 0.45, "Plan: pilot of about 300 converted complexes (real and natural complexes, interaction rule built into each swap, energy and re-dock as extra filters), train a small test run to check the recipe, then scale.", fontsize=9.8, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig21_data_readiness.png"), bbox_inches="tight")
    plt.close(f)
