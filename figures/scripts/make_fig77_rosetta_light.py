"""Light paired Rosetta score (rosetta_score/paired_dg_light.py): the globally relaxed native, then an identical LOCAL protocol (swap, repack, chi minimisation, backbone fixed) for the native and every swap, ONE position at a time.
Scores: E_slot = interaction energy of the slot with the receptor, E_int = whole peptide-receptor interaction energy (two-body REF2015 energies, lower = binds better). gain = E(swap) - E(native); > 0: the native residue is better.
Control: canonical interface residues of the same peptides against ALA (an alanine scan, known to be mostly > 0)."""
import json, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; R = json.load(open(f"{D}/rosetta_score/real_paired_light.json"))
rows = [x for r in R for x in r["rows"] if "refs" in x]
DF = {"DAL", "DAR", "DAS", "DCS", "DGU", "DGN", "DHI", "DIL", "DLE", "DLY", "DPH", "DPR", "DAN", "DSE", "DTH", "DTR", "DTY", "DVA"}
grp = lambda x: "canonical control" if x["kind"] == "control" else ("D-forms" if x["ncaa"] in DF else "other NCAAs")
G = ["D-forms", "other NCAAs", "canonical control"]; COL = {"D-forms": "#e76f51", "other NCAAs": "#264653", "canonical control": "#999999"}
def gain(x, ref, k=1): h = x["refs"][ref]["half"]; return float(np.mean([h[0][k], h[1][k]]))
f = plt.figure(figsize=(19, 10.5)); gs = f.add_gridspec(2, 3, height_ratios=[0.34, 1], hspace=0.5, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
n = collections.Counter(grp(x) for x in rows); spread = np.median([abs(x["refs"][r]["half"][0][1] - x["refs"][r]["half"][1][1]) for x in rows for r in x["refs"]])
s.text(50, 5, f"Light protocol, ONE POSITION AT A TIME: the complex is relaxed once; then the native residue and each swap (L parent, ALA, GLY) get the SAME local steps: swap, repack the neighbourhood, side-chain minimisation, backbone fixed. 12 repacks per variant.\nScore = interaction energy of the slot with the receptor (E_slot) or of the whole peptide with the receptor (E_int); gain = E(swap) - E(native), > 0: the native residue is better. Slots: D-forms {n['D-forms']}, other NCAAs {n['other NCAAs']}, canonical control {n['canonical control']} (interface residues against ALA, an alanine scan).\nRepeat spread of the two independent halves: median {spread:.2f} units (matched-relax run: 7.5).", ha="center", va="center", fontsize=8.6)
s.set_title("Rosetta interaction-energy gain of an NCAA over a canonical residue: light protocol", fontsize=10.5, loc="left", fontweight="bold")
# (a) gain vs ALA by group
ax = f.add_subplot(gs[1, 0]); rng = np.random.default_rng(0)
for j, g in enumerate(G):
    v = np.array([gain(x, "ALA") for x in rows if grp(x) == g]); ax.boxplot([v], positions=[j], widths=0.5, showfliers=False, medianprops=dict(color="k"))
    ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=COL[g], alpha=0.7, zorder=3); ax.text(j, ax.get_ylim()[0], f"n={len(v)}", ha="center", fontsize=8)
ax.axhline(0, color="#c1121f", lw=0.9, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels(G); ax.set_ylabel("E_slot gain vs ALA (units, > 0: native better)"); ax.set_title("(a) gain over ALA at each slot", fontsize=9.4, loc="left", fontweight="bold")
# (b) share > 0
ax = f.add_subplot(gs[1, 1]); labs, vals = [], []
for g in G:
    for ref in (["ALA", "GLY"] if g != "canonical control" else ["ALA"]):
        xs = [x for x in rows if grp(x) == g and ref in x["refs"]]
        for k, kn in ((1, "E_slot"), (0, "E_int")): labs.append(f"{g}\nvs {ref}\n{kn}"); vals.append((100 * np.mean([gain(x, ref, k) > 0 for x in xs]), g))
ax.bar(range(len(vals)), [v[0] for v in vals], color=[COL[v[1]] for v in vals], edgecolor="#333", lw=0.5)
for i, v in enumerate(vals): ax.text(i, v[0], f"{v[0]:.0f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(50, color="#c1121f", ls="--", lw=0.8); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=7.4, rotation=70, ha="right"); ax.set_ylim(0, 105); ax.set_ylabel("% of slots with gain > 0"); ax.set_title("(b) share of slots where the native residue wins", fontsize=9.4, loc="left", fontweight="bold")
# (c) per code
ax = f.add_subplot(gs[1, 2]); by = collections.defaultdict(list)
for x in rows:
    if x["kind"] == "ncaa": by[x["ncaa"]].append(gain(x, "ALA"))
    else: by["(canon) " + x["ncaa"]].append(gain(x, "ALA"))
items = sorted([(np.median(v), k, len(v)) for k, v in by.items() if len(v) >= 3]); y = range(len(items))
ax.barh(y, [m for m, _, _ in items], color=["#999999" if k.startswith("(canon)") else "#e76f51" if k in DF else "#264653" for _, k, _ in items], edgecolor="#333", lw=0.4)
ax.set_yticks(list(y)); ax.set_yticklabels([f"{k} (n={n_})" for _, k, n_ in items], fontsize=7.5); ax.axvline(0, color="#c1121f", ls="--", lw=0.8); ax.set_xlabel("median E_slot gain vs ALA"); ax.set_title("(c) per residue code (at least 3 slots)", fontsize=9.4, loc="left", fontweight="bold")
nd = [x for x in rows if x["kind"] == "ncaa" and x["ncaa"] not in DF and x["parent"] in x["refs"] and x["parent"] not in ("ALA", "GLY")]
dp = [gain(x, x["parent"]) for x in rows if x["kind"] == "ncaa" and x["ncaa"] in DF and x["parent"] in x["refs"]]
f.text(0.01, -0.05, "Reading: see Results.MD 80. Other NCAAs against their L parent: %d slots, native better in %.0f%% (median %.1f); D-forms against the L parent: median %.0f\n(the L parent on a D backbone clashes, so ALA and GLY are the references for D-forms).\nBackbone is fixed, so the native residue keeps the advantage of the relaxed pocket; the control shows how large that baseline advantage is for canonical residues." % (len(nd), 100 * np.mean([gain(x, x["parent"]) > 0 for x in nd]) if nd else 0, np.median([gain(x, x["parent"]) for x in nd]) if nd else 0, np.median(dp) if dp else 0), fontsize=8.4, fontweight="bold")
f.savefig(f"{D}/figures/fig77_rosetta_light.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig77", len(rows))
