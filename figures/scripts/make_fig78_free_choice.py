"""Free-choice generation (Results.MD 81): the pipeline patch hides the identity of the peptide's NCAA tokens (token type kept); the NCAA choice comes from the denoiser's readout (restricted to NCAA classes) and decoder run 8 builds the atoms.
11 real pockets x 8 designs per denoiser = 176 NCAA slots (152 are D-forms in the native). Denoisers: untuned control, G3 (default), S1 (G3 + readout trained on hidden NCAAs). Data: gen_test/diag/free_choice.json (identity-given reference: free_identity_given_backup.json)."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; F = json.load(open(f"{D}/gen_test/diag/free_choice.json")); G = json.load(open(f"{D}/gen_test/diag/free_identity_given_backup.json"))
K = [("control_choice_run8", "control", "#999999"), ("G3_choice_run8", "G3", "#6a4c93"), ("S1_choice_run8", "S1", "#e76f51")]
f = plt.figure(figsize=(19, 10.5)); gs = f.add_gridspec(2, 3, height_ratios=[0.42, 1], hspace=0.55, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, "Before (Results.MD 71): the library gave the NCAA identity to the model even when the spec had no sequence condition (probe: C_SEQ = DAL, DPN at the denoiser input) -> native code returned in 100% of 176 slots.\nNow: a patch of the pipeline hides the identity of the peptide's NCAA tokens (token type kept, so only NCAA classes are allowed); the denoiser's own readout fills in the identity, decoder run 8 builds the atoms.\nTest: 11 real pockets x 8 designs per denoiser, 176 NCAA slots (152 are D-forms in the native peptide). No residue identity is given in the spec.", ha="center", va="center", fontsize=8.6)
s.set_title("Free choice of the non-canonical residue: what the model picks when it is not told", fontsize=10.5, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 0]); v = [100 * F[k]["exact"] for k, *_ in K]; ax.bar(range(3), v, color=[c for *_, c in K], edgecolor="#333", lw=0.5)
for i, y in enumerate(v): ax.text(i, y + 0.6, f"{y:.1f}%", ha="center", fontsize=9)
ax.axhline(100 * F["control_choice_run8"]["majority_baseline"], color="#c1121f", ls="--", lw=0.9); ax.text(2.45, 100 * F["control_choice_run8"]["majority_baseline"] + 1, "always DAL = 32%", ha="right", fontsize=8, color="#c1121f")
ax.text(2.45, 36.5, "identity given (before): 100%", ha="right", fontsize=8, color="#555"); ax.set_ylim(0, 42); ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in K]); ax.set_ylabel("%")
ax.set_title("(a) NCAA slots that get the exact native code", fontsize=9.4, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 1]); v = [100 * F[k]["D_designed_at_D_slot"] for k, *_ in K]; ax.bar(range(3), v, color=[c for *_, c in K], edgecolor="#333", lw=0.5)
for i, y in enumerate(v): ax.text(i, y + 0.8, f"{y:.0f}%", ha="center", fontsize=9)
ax.axhline(100 * 19 / 142, color="#888", ls=":", lw=1); ax.text(-0.45, 100 * 19 / 142 - 3.2, "chance: 19 of 142 NCAA classes are D-forms = 13%", ha="left", fontsize=7.8, color="#222")
ax.set_ylim(0, 55); ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in K]); ax.set_ylabel("%"); ax.set_title("(b) a D-form is chosen where the native residue is a D-form", fontsize=9.4, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 2]); codes = ["DAL", "DPN", "DTR", "DIL", "MLE", "SAR", "MAA", "4FB", "04Q", "23F", "DTY", "XCP", "SEP", "DHI"]; w = 0.2
rows = [("native (given)", {c: n for c, n in G["control_run8"]["designed_nc_codes"]}, "#2a9d8f")] + [(n, {c: m for c, m in F[k]["designed_nc_codes"]}, col) for k, n, col in K]
for j, (lab, d, col) in enumerate(rows):
    ax.bar(np.arange(len(codes)) + (j - 1.5) * w, [100 * d.get(c, 0) / 176 for c in codes], w * 0.95, color=col, edgecolor="#333", lw=0.3, label=lab)
ax.set_xticks(range(len(codes))); ax.set_xticklabels(codes, rotation=55, fontsize=8); ax.set_ylabel("% of the 176 NCAA slots"); ax.legend(fontsize=7.4, frameon=False); ax.set_title("(c) which NCAA codes are picked (top 8 per model; missing = below that)", fontsize=9.0, loc="left", fontweight="bold")
f.text(0.01, -0.03, "Reading: with the identity really hidden the exact native code is recovered in only 0.6 to 4% of slots (always-DAL 32%) and at D-form slots none of the picks has the native residue's parent amino acid: the models do not yet choose the 'right' NCAA, and every denoiser collapses to a handful of codes (SAR, MAA, DIL, 4FB).\nOne small real effect: S1 puts a D-form where the native is a D-form in 41% of the D-form slots against 20% for control and G3 (chance 13%), and it also gives the best D handedness (91%); the readout training moved the choice toward the native class, not toward the native code. Structure of the designs: sensible bonds 84-87%, close contacts 17-24%.", fontsize=8.4, fontweight="bold")
f.savefig(f"{D}/figures/fig78_free_choice.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig78")
