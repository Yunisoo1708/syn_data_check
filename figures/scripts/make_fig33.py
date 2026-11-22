"""Per-swap Rosetta interface energy change by residue family/code (states_out from energy_states.py)."""
import os, json, glob, collections, re
import numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune/data/natural_complexes"
RENAME = {"DSN": "DSE", "DGL": "DGU", "DCY": "DCS", "DPN": "DPH", "MED": "DME", "DSG": "DAN"}
FAM = {"N-alkyl": "N-alkyl (backbone N-methyl etc.)", "alpha,alpha-disubstituted": "alpha,alpha-disubstituted (backbone)", "D-alpha": "D-alpha", "D-alpha (new code)": "D-alpha", "alpha, side-chain modified": "side-chain modified"}
COL = {"N-alkyl (backbone N-methyl etc.)": "#b23a48", "alpha,alpha-disubstituted (backbone)": "#c9a227", "D-alpha": "#2a6f97", "side-chain modified": "#6c757d"}
def load():
    idx = json.load(open(f"{D}/states_in/index.json")); st = {}
    for f in glob.glob(f"{D}/states_out/w*.jsonl"):
        for l in open(f):
            try: r = json.loads(l); st[(r["item"], r["state"])] = r
            except Exception: pass
    rows = []
    for s in idx:
        r = st.get((s["item"], s["state"]))
        ok = False
        if r and "dG" in r and r["dG"] < 0:
            nums = {}
            for l in open(f"{D}/states_in/{s['item']}_s{s['state']}.pdb"):
                if l.startswith(("ATOM", "HETATM")) and l[21] == "P": nums[int(l[22:26])] = l[17:20].strip()
            exp = {n: RENAME.get(c, c) for n, c in nums.items()}
            got = {int(n): c for n, c in r["pep"]}
            ok = got == exp
        rows.append(dict(s, dG=(r or {}).get("dG"), valid=ok, done=r is not None))
    return rows
def swaps(rows):
    by = {(r["item"], r["state"]): r for r in rows}; out = []
    for r in rows:
        if r["state"] == 0: continue
        p = by.get((r["item"], r["state"] - 1))
        if p and r["valid"] and p["valid"]:
            out.append(dict(item=r["item"], code=r["code"], parent=r["parent"], fam=FAM.get(r["family"], r["family"]), d=r["dG"] - p["dG"], mirrored=r["mirrored"]))
    return out
def fig(OUT):
    rows = load(); S = swaps(rows)
    f, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.25, 1]))
    a = ax[0]; fams = sorted({s["fam"] for s in S}, key=lambda k: -len([s for s in S if s["fam"] == k]))
    for i, k in enumerate(fams):
        v = np.array([s["d"] for s in S if s["fam"] == k]); m = np.mean(v); se = np.std(v) / np.sqrt(len(v))
        a.bar(i, m, yerr=se, color=COL.get(k, "#999"), edgecolor="#555", capsize=4); a.text(i, m + (se + 0.25 if m >= 0 else -se - 0.8), f"{m:+.2f}\nn={len(v)}", ha="center", fontsize=8.5)
    a.axhline(0, color="#222", lw=1); a.set_xticks(range(len(fams))); a.set_xticklabels([k.replace(" (", "\n(") for k in fams], fontsize=8); a.set_ylabel("mean change of interface energy per swap (REU; negative = binds better)")
    a.set_title("(a) Per-swap energy change by family (mean +- s.e.)", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[1]; codes = collections.defaultdict(list)
    for s in S: codes[s["code"]].append(s["d"])
    cs = sorted([(np.median(v), c, len(v)) for c, v in codes.items() if len(v) >= 12])
    fam_of = {s["code"]: s["fam"] for s in S}
    b.barh(range(len(cs)), [m for m, _, _ in cs], color=[COL.get(fam_of[c], "#999") for _, c, _ in cs], edgecolor="#555")
    b.set_yticks(range(len(cs))); b.set_yticklabels([f"{c} (n={n})" for _, c, n in cs], fontsize=8); b.axvline(0, color="#222", lw=1); b.set_xlabel("median energy change of the swap (REU)")
    b.set_title("(b) Residue codes with at least 12 swaps", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[2]
    for k in fams:
        v = np.array([s["d"] for s in S if s["fam"] == k]); c.hist(np.clip(v, -20, 30), bins=np.arange(-20, 32, 2), histtype="step", lw=2, color=COL.get(k, "#999"), density=True, label=k.split(" (")[0])
    c.axvline(0, color="#222", lw=1); c.set_xlabel("energy change of one swap (REU)"); c.set_ylabel("density"); c.legend(frameon=False, fontsize=8); c.set_title("(c) Distribution of per-swap changes", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.03, "Each swap = Rosetta relax of the peptide-receptor complex before and after the swap, same seed; only pairs where Rosetta kept every residue under the right name. REF2015 is calibrated on canonical residues, so absolute values for non-canonical residues are uncertain.", fontsize=9, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig33_pilot_per_swap.png"), bbox_inches="tight"); plt.close(f)
    return rows, S
if __name__ == "__main__":
    rows, S = fig("C:/Research/RFD4_fine_tune/figures")
    print("states", len(rows), "done", sum(r["done"] for r in rows), "valid", sum(r["valid"] for r in rows), "swaps with valid pair", len(S))
    g = collections.defaultdict(list)
    for s in S: g[s["fam"]].append(s["d"])
    for k, v in sorted(g.items(), key=lambda t: -len(t[1])): print(f"{k:40s} n={len(v):4d} mean {np.mean(v):+.2f} median {np.median(v):+.2f} se {np.std(v)/np.sqrt(len(v)):.2f} better<-2: {sum(x<-2 for x in v)} worse>2: {sum(x>2 for x in v)}")
