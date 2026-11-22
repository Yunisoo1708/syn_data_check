"""Figure: the written synthetic complexes (pilot_out): peptide lengths, non-canonical fraction, re-check of the written files."""
import json, os, matplotlib.pyplot as plt, numpy as np
D = "C:/Research/RFD4_fine_tune"
def fig(OUT):
    M = json.load(open(f"{D}/data/pilot_out/manifest.json"))
    f, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(wspace=0.32))
    a = ax[0]; L = [m["n_res"] for m in M]; a.hist(L, bins=range(8, 32, 2), color="#2a6f97", edgecolor="#555"); a.set_xlabel("peptide length (residues)"); a.set_ylabel("complexes")
    a.set_title(f"(a) {len(M)} converted complexes, peptide length", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[1]; fr = [100 * m["n_noncanonical"] / m["n_res"] for m in M]; b.hist(fr, bins=np.arange(0, 105, 10), color="#c9a227", edgecolor="#555")
    b.set_xlabel("share of the peptide that is non-canonical (%)"); b.set_ylabel("complexes"); b.set_title(f"(b) Non-canonical share (mean {np.mean(fr):.0f}%)", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[2]; chk = sum(m["checked"] for m in M); ps = sum(m["passed"] for m in M); ok = sum(m["passed"] == m["checked"] for m in M)
    c.bar([0, 1], [100 * ps / chk, 100 * ok / len(M)], color=["#2a6f97", "#8d99ae"], edgecolor="#555")
    for i, (v, t) in enumerate([(100 * ps / chk, f"{ps}/{chk}"), (100 * ok / len(M), f"{ok}/{len(M)}")]): c.text(i, v + 1, t, ha="center", fontsize=10, fontweight="bold")
    c.set_xticks([0, 1]); c.set_xticklabels(["non-canonical residues\npassing", "complexes with every\nresidue passing"]); c.set_ylim(0, 112); c.set_ylabel("%")
    c.set_title("(c) Strict test re-run on the written CIF files", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.04, "All 264 complexes are train entries (SplitGuard v2). The strict test is the pool's own test (not independent of the model's training signal). One complex (4WVI) has two peptide-link failures and is flagged.", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig31_pilot_cifs.png"), bbox_inches="tight"); plt.close(f)
if __name__ == "__main__": fig(f"{D}/figures")
