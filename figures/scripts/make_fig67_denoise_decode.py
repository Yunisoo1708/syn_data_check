"""DENOISER + DECODER (run 8) rebuild of non-canonical residues on real complexes (test only; gen_test/eval_dd_atoms.py): real 103 + extra real complexes with an NCAA; the generated tokens (CA and latent) are noised to a fixed time t (0 = pure noise, 1 = clean; same seeded noise for every model), the denoiser predicts the clean complex, decoder run 8 builds the atoms, and each NCAA residue is compared with the native one. Identity of NCAAs and the receptor are given, as in the generation test."""
import json, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; DN = [("control", "control (untuned)", "#999999"), ("G_ncw10", "G", "#2a9d8f"), ("G2_pool", "G2", "#219ebc"), ("G3_bal", "G3", "#6a4c93")]
rows = {}
for k, *_ in DN:
    r = []
    for st in ("real", "ham"):
        p = f"{D}/gen_test/diag/dd_{k}_{st}.json"
        if os.path.exists(p): r += json.load(open(p))["rows"]
    rows[k] = r
TS = sorted({x["t"] for x in rows["control"]})
def sel(k, t, g=None): return [x for x in rows[k] if x["t"] == t and (g is None or x["group"] == g)]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.4, 1, 1], hspace=0.55, wspace=0.27)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#dde5b6", ec="#555"))
n = len(sel("control", TS[-1]))
s.text(50, 5, f"Test: every NCAA residue of the real complexes that hold one ({n} generated NCAA positions per noise level; test only). CA and latent of the generated tokens are noised to time t (0 = pure noise, 1 = clean), the denoiser predicts the clean\ncomplex, decoder run 8 builds the atoms; the NCAA residue is compared with the native one. Receptor and NCAA identity are given (as in the generation test). 'oracle' = decoder on the TRUE latent (no denoiser).", ha="center", va="center", fontsize=8.2)
s.set_title("Do the denoiser and the decoder together rebuild non-canonical residues accurately?", fontsize=10.5, loc="left", fontweight="bold")
def curve(ax, fn, title, ylabel, ylim=None, oracle=None):
    for k, lab, col in DN:
        ys = [fn(sel(k, t)) for t in TS]; ax.plot(TS, ys, "-o", color=col, label=lab, ms=4)
    if oracle is not None: ax.plot(TS, [oracle(sel("control", t)) for t in TS], "--", color="#c1121f", label="oracle (true latent)")
    ax.set_xlabel("noise time t (0 = pure noise, 1 = clean)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9.2, loc="left", fontweight="bold")
    if ylim: ax.set_ylim(*ylim)
ax = f.add_subplot(gs[1, 0]); curve(ax, lambda r: float(np.median([x["rmsd_dd_A"] for x in r])), "(a) median all-atom RMSD of the NCAA residue (A)", "A", (0, 20), lambda r: float(np.median([x["rmsd_oracle_A"] for x in r]))); ax.legend(fontsize=7, frameon=False)
ax = f.add_subplot(gs[1, 1]); curve(ax, lambda r: 100 * np.mean([x["rmsd_dd_A"] < 1.0 for x in r]), "(b) residues within 1 A of the native (%)", "%", (0, 100), lambda r: 100 * np.mean([x["rmsd_oracle_A"] < 1.0 for x in r]))
ax = f.add_subplot(gs[1, 2]); curve(ax, lambda r: float(np.median([x["ca_err_A"] for x in r])), "(c) median error of the denoised C-alpha (A)", "A", (0, 20))
ax = f.add_subplot(gs[2, 0]); curve(ax, lambda r: 100 * np.mean([x["atoms_ok_dd"] for x in r]), "(d) decoder emits exactly the native atoms (%)", "%", (0, 105), lambda r: 100 * np.mean([x["atoms_ok_oracle"] for x in r]))
ax = f.add_subplot(gs[2, 1]); curve(ax, lambda r: 100 * np.mean([x["chir_ok_dd"] for x in r if x["chir_ok_dd"] is not None]), "(e) C-alpha handedness right (%)", "%", (40, 105), lambda r: 100 * np.mean([x["chir_ok_oracle"] for x in r if x["chir_ok_oracle"] is not None]))
ax = f.add_subplot(gs[2, 2]); tt = TS[-1]; G = ["D-forms", "general codes", "other non-canonical"]; w = 0.2
for j, (k, lab, col) in enumerate(DN):
    vals = [float(np.median([x["rmsd_dd_A"] for x in sel(k, tt, g)])) if sel(k, tt, g) else 0 for g in G]; xs = np.arange(len(G)) + (j - 1.5) * w
    ax.bar(xs, vals, w * 0.95, color=col, edgecolor="#333", linewidth=0.5, label=lab)
    for x, v in zip(xs, vals): ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.axhline(float(np.median([x["rmsd_oracle_A"] for x in sel("control", tt)])), color="#c1121f", ls="--", lw=0.8)
ax.set_xticks(range(len(G))); ax.set_xticklabels([f"{g}\n(n={len(sel('control', tt, g))})" for g in G], fontsize=8); ax.set_ylabel("A"); ax.set_title(f"(f) median RMSD at t = {tt} by residue group (dashed = oracle)", fontsize=9.2, loc="left", fontweight="bold")
c9 = sel("control", TS[-1]); g3 = sel("G3_bal", TS[-1]); c7 = sel("control", 0.7) if 0.7 in TS else []
f.text(0.01, 0.005, "Result: from a mildly noised input (t = 0.9) denoiser + decoder rebuild the NCAA residue to a median %.2f A (G3 %.2f A; decoder alone %.2f A); the error grows as the input gets noisier and from near-pure noise the model generates a new pose instead of reproducing the native one. The fine-tuned denoisers differ little from the untuned one." % (float(np.median([x["rmsd_dd_A"] for x in c9])), float(np.median([x["rmsd_dd_A"] for x in g3])), float(np.median([x["rmsd_oracle_A"] for x in c9]))), fontsize=8.2, fontweight="bold")
f.savefig(f"{D}/figures/fig67_denoise_decode.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig67", {k: len(v) for k, v in rows.items()})
