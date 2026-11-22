"""Figure: PyRosetta REF2015 relaxed interface energy, natural peptide vs converted peptide (pilot complexes)."""
import os, json, glob, collections
import numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune"
def load():
    M = {m["entry"]: m for m in json.load(open(f"{D}/data/pilot_out/manifest.json"))}
    rows = []
    for f in glob.glob(f"{D}/data/natural_complexes/energy_pilot_conv_inputs_*.json"):
        r = json.load(open(f)); m = M[r["item"]]
        r.update(n_res=m["n_res"], nnc=m["n_noncanonical"], swaps=m["swaps"], codes=[s["code"] for s in m["swaps"]])
        r["valid"] = ("before_dG" in r and "after_dG" in r and r.get("before_pep_residues") == m["n_res"] and r.get("after_pep_residues") == m["n_res"] and r["before_dG"] < 0 and r["after_dG"] < 0)
        rows.append(r)
    return rows
def fig(OUT):
    R = load(); V = [r for r in R if r["valid"]]
    d = np.array([r["after_dG"] - r["before_dG"] for r in V]); b = np.array([r["before_dG"] for r in V]); a = np.array([r["after_dG"] for r in V])
    f, ax = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw=dict(wspace=0.32))
    x = ax[0]; x.scatter(b, a, s=16, color="#2a6f97", alpha=0.75, edgecolor="none"); lim = [min(b.min(), a.min()) - 3, 0]; x.plot(lim, lim, color="#222", lw=1); x.set_xlim(lim); x.set_ylim(lim)
    x.set_xlabel("natural peptide dG_separated (REU)"); x.set_ylabel("converted peptide (REU)"); x.set_title(f"(a) Paired interface energies (n={len(V)})", fontsize=10.5, loc="left", fontweight="bold")
    y = ax[1]; y.hist(np.clip(d, -40, 60), bins=np.arange(-40, 62, 4), color="#c9a227", edgecolor="#555"); y.axvline(0, color="#222", lw=1); y.axvspan(-5, 5, color="#8d99ae", alpha=0.25)
    y.set_xlabel("energy change, converted minus natural (REU; positive = worse)"); y.set_ylabel("complexes"); y.set_title(f"(b) Change (median {np.median(d):+.1f}; grey = +-5 REU noise)", fontsize=10.5, loc="left", fontweight="bold")
    z = ax[2]; fr = np.array([r["nnc"] / r["n_res"] for r in V]); z.scatter(100 * fr, d, s=16, color="#b23a48", alpha=0.7, edgecolor="none"); z.axhline(0, color="#222", lw=1)
    rho = np.corrcoef(np.argsort(np.argsort(fr)), np.argsort(np.argsort(d)))[0, 1]
    z.set_xlabel("share of the peptide that is non-canonical (%)"); z.set_ylabel("energy change (REU)"); z.set_ylim(-40, 80); z.set_title(f"(c) Change vs share converted (rank r = {rho:.2f})", fontsize=10.5, loc="left", fontweight="bold")
    w = ax[3]; cnt = collections.Counter(); tot = collections.Counter()
    for r in R:
        for k in ("before", "after"):
            tot[k] += 1; cnt[k + "_ok"] += int(k + "_dG" in r and r.get(k + "_pep_residues") == r["n_res"])
    lab = ["valid pairs", "natural:\nall kept", "converted:\nall kept"]; vals = [len(V), cnt["before_ok"], cnt["after_ok"]]
    w.bar(range(3), [100 * v / len(R) for v in vals], color=["#2a6f97", "#8d99ae", "#c9a227"], edgecolor="#555")
    for i, v in enumerate(vals): w.text(i, 100 * v / len(R) + 1.5, f"{v}/{len(R)}", ha="center", fontsize=9)
    w.set_xticks(range(3)); w.set_xticklabels(lab, fontsize=8.5); w.set_ylim(0, 112); w.set_ylabel("complexes (%)"); w.set_title("(d) How many runs are usable", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.03, "PyRosetta FastRelax with coordinate restraints, REF2015, InterfaceAnalyzer dG_separated; a pair is valid when Rosetta kept every peptide residue and both energies are negative. REF2015 is calibrated on canonical residues, so shifts for non-canonical residues are partly scoring artefact.", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig32_pilot_rosetta.png"), bbox_inches="tight"); plt.close(f)
    return R, V, d
if __name__ == "__main__":
    R, V, d = fig(f"{D}/figures")
    print("runs", len(R), "valid", len(V), "median change", np.median(d), "mean", d.mean(), "within +-5", int((abs(d) <= 5).sum()), "worse>5", int((d > 5).sum()), "better<-5", int((d < -5).sum()))
