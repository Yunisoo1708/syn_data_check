"""Scoring of the FREE-CHOICE designs (Results.MD 82): (a) family of the chosen NCAA vs native, (b) receptor contacts of the NCAA slot, (c-e) light Rosetta score (relax once, identical local protocol, one position at a time; rosetta_score/paired_score_designs.py).
Native = the 11 native pockets (gen_test/inputs); designs = 3 denoisers x 88 (decoder run 8). Rosetta cannot swap N-methylated and some side-chain NCAAs (SAR, MAA, 4FB, 04Q, 23F, XCP): the gain over ALA covers only the swappable slots."""
import json, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; CS = json.load(open(f"{D}/gen_test/diag/choice_scores.json")); R = json.load(open(f"{D}/rosetta_score/real_design_scores.json"))
GR = [("native", "native\npocket", "#2a9d8f"), ("control", "control", "#999999"), ("G3", "G3", "#6a4c93"), ("S1", "S1", "#e76f51")]
def g(x, ref, k=1):
    d = x["refs"].get(ref); return None if not d or "half" not in d else float(np.mean([d["half"][0][k], d["half"][1][k]]))
own = lambda x, k=1: float(np.mean([x["own"][0][k], x["own"][1][k]]))
rows = {gr: [x for r in R if r["info"]["group"] == gr for x in r["rows"]] for gr, *_ in GR}
f = plt.figure(figsize=(19, 11)); gs = f.add_gridspec(2, 3, height_ratios=[0.3, 1], hspace=0.55, wspace=0.28)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
nswap = {gr: sum(1 for x in rows[gr] if x["kind"] == "ncaa" and g(x, "ALA") is not None) for gr, *_ in GR}; nall = {gr: sum(1 for x in rows[gr] if x["kind"] == "ncaa") for gr, *_ in GR}
s.text(50, 5, "Do the NCAAs the model chose (identity hidden, Results.MD 81) look like good choices? Three views, 88 designs per denoiser (176 NCAA slots), against the native NCAAs of the same 11 pockets.\n(a) family of the chosen code against the native family, (b) heavy-atom contacts of the slot residue with the receptor, (c-e) light Rosetta score (relax once, identical local protocol, one position at a time).\nRosetta cannot swap N-methylated and some side-chain NCAAs: the gain over ALA covers " + ", ".join(f"{nswap[gr]} of {nall[gr]}" for gr, *_ in GR) + " NCAA slots (native, control, G3, S1).", ha="center", va="center", fontsize=8.6)
s.set_title("Scoring the free-choice designs", fontsize=10.5, loc="left", fontweight="bold")
# (a) families
ax = f.add_subplot(gs[1, 0]); fams = ["D-alpha", "N-alkyl", "side-chain modified", "PTM", "beta/gamma", "alpha,alpha-disubst."]; cols = ["#e76f51", "#264653", "#e9c46a", "#2a9d8f", "#8ab17d", "#bbbbbb"]
fn = collections.Counter(r["fam_native"] for r in CS["control"]); comp = {"native": {k: v / sum(fn.values()) for k, v in fn.items()}}
for k in ("control", "G3", "S1"):
    c = collections.Counter(r["fam_designed"] for r in CS[k]); comp[k] = {a: b / sum(c.values()) for a, b in c.items()}
agree = {k: np.mean([r["fam_native"] == r["fam_designed"] for r in CS[k]]) for k in CS}; chance = {k: sum(comp["native"].get(a, 0) * comp[k].get(a, 0) for a in fams) for k in CS}
for j, (gr, lab, _) in enumerate(GR):
    bot = 0
    for fm, c in zip(fams, cols):
        v = 100 * comp[gr].get(fm, 0); ax.bar(j, v, 0.7, bottom=bot, color=c, edgecolor="#333", lw=0.4, label=fm if j == 0 else None); bot += v
for j, k in enumerate(["control", "G3", "S1"], 1): ax.text(j, 102, f"agree {100 * agree[k]:.0f}%\n(chance {100 * chance[k]:.0f}%)", ha="center", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in GR]); ax.set_ylim(0, 118); ax.set_ylabel("% of NCAA slots"); ax.legend(fontsize=6.8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
ax.set_title("(a) family of the chosen NCAA", fontsize=9.4, loc="left", fontweight="bold")
# (b) contacts
ax = f.add_subplot(gs[1, 1]); data = [[r["native_contacts"] for r in CS["control"]]] + [[r["contacts"] for r in CS[k]] for k in ("control", "G3", "S1")]
bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
for b, (_, _, c) in zip(bp["boxes"], GR): b.set_facecolor(c); b.set_alpha(0.7)
for j, d in enumerate(data, 1): ax.text(j, np.percentile(d, 75) + 1.5, f"median {np.median(d):.0f}", ha="center", fontsize=8)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([l for _, l, _ in GR]); ax.set_ylabel("receptor contacts (< 4.5 A) of the slot residue"); ax.set_title("(b) how much the NCAA touches the receptor", fontsize=9.4, loc="left", fontweight="bold")
# (c) own E_slot
ax = f.add_subplot(gs[1, 2]); data = [[own(x) for x in rows[gr] if x["kind"] == "ncaa"] for gr, *_ in GR]
bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
for b, (_, _, c) in zip(bp["boxes"], GR): b.set_facecolor(c); b.set_alpha(0.7)
cm = np.median([own(x) for gr, *_ in GR for x in rows[gr] if x["kind"] == "control"]); ax.axhline(cm, color="#c1121f", ls="--", lw=0.9); ax.text(4.45, cm + 0.4, f"canonical interface residues: {cm:.1f}", ha="right", fontsize=7.8, color="#c1121f")
for j, d in enumerate(data, 1): ax.text(j, np.percentile(d, 25) - 1.2, f"{np.median(d):.1f}", ha="center", va="top", fontsize=8)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([l for _, l, _ in GR]); ax.set_ylabel("interaction energy of the slot with the receptor (lower = better)"); ax.set_title("(c) Rosetta E_slot of the chosen NCAA", fontsize=9.4, loc="left", fontweight="bold")
f.text(0.01, -0.1, "Reading: the chosen NCAAs are no better matched to the native family than chance, and the Rosetta interaction energy of the chosen NCAA is weaker than that of the native NCAAs and of canonical residues (mostly small SAR / MAA residues); G3 and S1 NCAAs touch the receptor more than the native ones (about 20 contacts against 8).\nGain over ALA at the swappable slots (D-forms, SEP, TPO): native 5.3 (74% > 0), control 1.5, G3 1.9, S1 3.0 (88% > 0), against 4 to 5 for canonical interface residues; E_int of the whole peptide per design: native -41, control -37, G3 -40, S1 -47 REU. Light protocol caveats of Results.MD 80 apply.", fontsize=8.4, fontweight="bold")
f.savefig(f"{D}/figures/fig79_choice_scores.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig79")
