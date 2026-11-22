"""Paired Rosetta interface dG, one NCAA position at a time (rosetta_score/paired_dg.py) on the real test complexes (test only).
Per slot: dG with the native NCAA vs dG after swapping ONLY that residue to its L parent / ALA / GLY (interface repack, separate+repack, 3 repeats). gain = dG_ref - dG_native (> 0: the NCAA is better than the canonical residue)."""
import json, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; JS = sys.argv[1] if len(sys.argv) > 1 else "real_paired_dg.json"; OUTF = sys.argv[2] if len(sys.argv) > 2 else "fig71_rosetta_paired_dg.png"; TAG = sys.argv[3] if len(sys.argv) > 3 else "unrelaxed structures, 3 repeats"
R = json.load(open(f"{D}/rosetta_score/{JS}"))
rows = [x for r in R for x in r["rows"] if "refs" in x]; skipped = [x for r in R for x in r["rows"] if "refs" not in x]
DF = {"DAL", "DAR", "DAS", "DCS", "DGU", "DGN", "DHI", "DIL", "DLE", "DLY", "DPH", "DPR", "DAN", "DSE", "DTH", "DTR", "DTY", "DVA"}
isD = lambda x: x["ncaa"] in DF
f = plt.figure(figsize=(19, 8.5)); gs = f.add_gridspec(2, 3, height_ratios=[0.28, 1], hspace=0.35, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, f"Paired Rosetta score, ONE POSITION AT A TIME: for each NCAA slot only that residue is swapped (to its L parent, to ALA, to GLY); all other residues stay native.\nBoth variants: repack the interface, E(complex) - E(peptide) - E(receptor) after repacking each (dG, lower = binds better), mean of 3 to 5 repeats (see title).   gain = dG(swap) - dG(native):  > 0 means the NCAA beats the canonical residue.\nTest set: {len({r["file"] for r in R})} real complexes, {len(rows)} NCAA slots scored ({len(skipped)} skipped: no parent known or Rosetta cannot build the residue).", ha="center", va="center", fontsize=8.6)
s.set_title("Rosetta interface-energy gain of a non-canonical residue over the canonical one (" + TAG + ")", fontsize=10.5, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 0]); sd = [x["sd_native"] for x in rows]; ax.hist(sd, bins=30, color="#2a9d8f", edgecolor="#333", lw=0.4)
ax.axvline(np.median(sd), color="#c1121f", ls="--"); ax.text(np.median(sd), ax.get_ylim()[1] * 0.9, f" median {np.median(sd):.2f}", color="#c1121f", fontsize=8)
ax.set_xlabel("sd of dG over the repeats of one slot (Rosetta units)"); ax.set_ylabel("NCAA slots"); ax.set_title("(a) repeat noise of one slot", fontsize=9.2, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 1]); groups = [("D-forms", [x for x in rows if isD(x)]), ("other NCAAs", [x for x in rows if not isD(x)])]; w = 0.26
for j, (ref, col) in enumerate([("ALA", "#e76f51"), ("GLY", "#6a4c93"), ("parent", "#264653")]):
    vals = []
    for g, xs in groups:
        v = [(x["refs"].get(ref) or (x["refs"].get(x["parent"]) if ref == "parent" else None)) for x in xs]; v = [y["gain"] for y in v if y and (ref != "parent" or not y["backbone_clash"])]
        vals.append((np.median(v) if v else 0, len(v)))
    xs_ = np.arange(2) + (j - 1) * w; ax.bar(xs_, [v[0] for v in vals], w * 0.95, color=col, edgecolor="#333", lw=0.5, label={"ALA": "vs ALA", "GLY": "vs GLY", "parent": "vs L parent (no backbone clash)"}[ref])
    for x, v in zip(xs_, vals): ax.text(x, v[0], f"n={v[1]}", ha="center", va="bottom", fontsize=7)
ax.axhline(0, color="#333", lw=0.6); ax.set_xticks(range(2)); ax.set_xticklabels([g for g, _ in groups]); ax.set_ylabel("median gain (Rosetta units)"); ax.legend(fontsize=7.5, frameon=False)
ax.set_title("(b) gain of the real NCAA over a canonical residue", fontsize=9.2, loc="left", fontweight="bold")
ax = f.add_subplot(gs[1, 2]); labs = []; vals = []
for g, xs in groups:
    for ref in ("ALA", "GLY"):
        v = [x["refs"][ref]["gain"] for x in xs if ref in x["refs"]]; labs.append(f"{g}\nvs {ref}"); vals.append(100 * np.mean([y > 0 for y in v]) if v else 0)
pp = [x["refs"][x["parent"]] for x in rows if not isD(x) and x["parent"] in x["refs"] and x["parent"] not in ("ALA", "GLY")]
labs.append("other NCAAs\nvs L parent"); vals.append(100 * np.mean([y["gain"] > 0 for y in pp if not y["backbone_clash"]]) if pp else 0)
ax.bar(range(len(vals)), vals, color="#e9c46a", edgecolor="#333", lw=0.5)
for x, v in enumerate(vals): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.axhline(50, color="#c1121f", ls="--", lw=0.8); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=7.5); ax.set_ylim(0, 105); ax.set_ylabel("% of slots where the NCAA is better"); ax.set_title("(c) validity check: real NCAAs should beat the canonical swap > 50%", fontsize=9.2, loc="left", fontweight="bold")
dcl = [x["refs"][x["parent"]]["backbone_clash"] for x in rows if isD(x) and x["parent"] in x["refs"]]
allv = {g: {r: [x["refs"][r]["gain"] for x in xs if r in x["refs"]] for r in ("ALA", "GLY")} for g, xs in groups}
f.text(0.01, -0.02, "Reading: real NCAAs beat ALA in %.0f%% of D-form slots (n=%d) and %.0f%% of other slots (n=%d); GLY %.0f%% / %.0f%%; the L parent only %.0f%% (n=%d, small). Median gains are 0.1 to %.1f units, to compare with the repeat noise (median sd %.1f, tail up to %.0f): one slot is not reliable, the average over many slots is.\nD-forms swapped to the L parent clash with the D backbone in %.0f%% of slots (flagged, excluded from the L-parent bars): for D-forms ALA/GLY are the references. Structures: %s. REF2015 units, not kcal/mol." % (
    100 * np.mean(np.array(allv["D-forms"]["ALA"]) > 0), len(allv["D-forms"]["ALA"]), 100 * np.mean(np.array(allv["other NCAAs"]["ALA"]) > 0), len(allv["other NCAAs"]["ALA"]),
    100 * np.mean(np.array(allv["D-forms"]["GLY"]) > 0), 100 * np.mean(np.array(allv["other NCAAs"]["GLY"]) > 0), vals[-1], len(pp),
    max(np.median(allv["D-forms"]["ALA"]), 0), np.median(sd), max(sd), 100 * np.mean(dcl) if dcl else 0, TAG), fontsize=8.6, fontweight="bold")
f.savefig(f"{D}/figures/{OUTF}", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig71", len(rows), "slots")
