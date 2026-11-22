"""Free-sequence generation test (gen_test/spec_free.yaml, run_free.sh, analyze_free.py): the receptor pocket is fixed and NO residue identity of the peptide is given in the spec. Finding: the NCAA slots still come out with the native code in 100% of the designs (also with the untuned denoiser and the blind decoder): the design pipeline hands the identity of non-canonical residues to the model whatever the spec says, so this test cannot measure the choice of an NCAA. Panels: (a) NCAA slots with the native code, (b) canonical-slot recovery, (c-e) structure quality of run 8 vs the blind decoder."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; F = json.load(open(f"{D}/gen_test/diag/free.json"))
DN = [("control", "control", "#999999"), ("G", "G", "#2a9d8f"), ("G3", "G3", "#6a4c93")]
f = plt.figure(figsize=(19, 10)); gs = f.add_gridspec(2, 3, height_ratios=[0.45, 1], hspace=0.5, wspace=0.27)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, "Test: 11 real pockets x 8 designs per run, receptor fixed, the C_SEQ condition of the WHOLE peptide removed (spec_free.yaml); denoisers control, G, G3 x decoder run 8 (identity-conditioned) and the blind decoder (run 4).\nQuestion: does the model pick the right NCAA at a slot when nobody tells it the identity?   Result: it cannot be answered with this setup. The output names of the NCAA slots equal the native ones in 100% of the 176 slots of EVERY run, also with the untuned\ndenoiser and the blind decoder: the design pipeline gives the identity of a non-canonical residue to the model whatever the spec says. A real test needs the pipeline (or the input CIF) changed.", ha="center", va="center", fontsize=8.0)
s.set_title("Free-sequence generation: the NCAA identity is still handed to the model", fontsize=10.5, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 0]); vals = [100 * F[f"{k}_run8"]["exact"] for k, *_ in DN]; ax.bar(range(3), vals, color=[c for *_, c in DN], edgecolor="#333", linewidth=0.5)
for x, v in enumerate(vals): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax.axhline(100 * F["control_run8"]["majority_baseline"], color="#c1121f", ls="--", lw=0.8); ax.text(2.45, 100 * F["control_run8"]["majority_baseline"] + 2, "most common native code (DAL) = 32%", ha="right", fontsize=7, color="#c1121f")
ax.set_xticks(range(3)); ax.set_xticklabels([n for n, *_ in DN]); ax.set_ylim(0, 112); ax.set_ylabel("%"); ax.set_title("(a) NCAA slots that carry the native code (176 per run)", fontsize=9.2, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 1]); vals = [100 * F[f"{k}_run8"]["canonical_slot_exact"] for k, *_ in DN]; ax.bar(range(3), vals, color=[c for *_, c in DN], edgecolor="#333", linewidth=0.5)
for x, v in enumerate(vals): ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.axhline(5, color="#888", ls=":", lw=0.8); ax.text(2.45, 5.5, "uniform guess over 20 = 5%", ha="right", fontsize=7, color="#666")
ax.set_xticks(range(3)); ax.set_xticklabels([n for n, *_ in DN]); ax.set_ylim(0, 25); ax.set_ylabel("%"); ax.set_title("(b) canonical slots: native residue recovered (free design)", fontsize=9.2, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 2]); w = 0.2; labs = ["bonds OK", "D right", "no close contact"]
for j, (dec, col, lab) in enumerate([("run8", "#2a9d8f", "decoder run 8"), ("blind", "#e76f51", "blind decoder (run 4)")]):
    vals = [np.mean([100 * F[f"{k}_{dec}"]["bond_ok"] for k, *_ in DN]), np.mean([100 * F[f"{k}_{dec}"]["chir_D"] for k, *_ in DN]), np.mean([100 * (1 - F[f"{k}_{dec}"]["clash"]) for k, *_ in DN])]
    xs = np.arange(3) + (j - 0.5) * 0.38; ax.bar(xs, vals, 0.36, color=col, edgecolor="#333", linewidth=0.5, label=lab)
    for x, v in zip(xs, vals): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_ylim(0, 115); ax.set_ylabel("% (mean of the 3 denoisers)"); ax.legend(fontsize=7, frameon=False, loc="upper right"); ax.set_title("(c) structure quality: run 8 vs the blind decoder", fontsize=9.2, loc="left", fontweight="bold")
f.text(0.01, 0.005, "Reading: this run did not test the choice of an NCAA (identity still given). What it does show: free canonical positions are recovered only 9-13% (chance 5%), and the blind decoder gives much worse structures than run 8.", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig68_free_sequence.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig68")
