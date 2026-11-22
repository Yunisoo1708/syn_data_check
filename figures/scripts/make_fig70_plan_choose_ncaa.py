"""Visual plan (20 Nov 2026): what the denoiser must learn to choose NCAAs itself, with at most 5 NCAAs per design.
Panel 1 = the problem today (example + measured numbers from Results.MD 71, 72), panel 2 = why, panel 3 = target behaviour (cap 5), panel 4 = training plan.
Numbers: free-sequence test 100% native code copied (section 71); S1 readout at real NCAA tokens top-1 7.3% vs control 9.5% vs always-ALY 10%, canonical guesses 28% vs 12% (section 72)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D = "C:/Research/RFD4_fine_tune"
CAN, NC, GREY, RED, GRN, BLU = "#d9d9d9", "#f4a261", "#999999", "#c1121f", "#2a9d8f", "#264653"
f = plt.figure(figsize=(19, 12.5)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, txt, fc, fs=9, bold=False, ec="#555", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.5", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc)

def arrow(x0, y0, x1, y1, col="#444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.6))

def strip(x, y, res, w=3.6, h=4.2, mark=None):
    for i, r in enumerate(res):
        nc = r in mark if mark else False
        box(x + i * (w + 0.5), y, w, h, r, NC if nc else CAN, fs=9, bold=nc)

def head(y, n, t):
    ax.text(1.5, y, n, fontsize=15, fontweight="bold", color=BLU); ax.text(4.2, y, t, fontsize=13, fontweight="bold")

# ---------- 1 PROBLEM ----------
head(96.5, "1", "Problem today: the model never chooses an NCAA, it is handed one")
ax.text(2, 91.5, "Example: designing an 8-residue peptide for a pocket", fontsize=9.5, color="#444")
ax.text(2, 87.5, "Spec we give", fontsize=9, fontweight="bold"); strip(13, 86, ["T", "F", "ALY", "L", "D", "DAR", "P", "E"], mark=["ALY", "DAR"])
ax.text(13, 82.6, "identity of the 2 NCAAs is written in the input", fontsize=8.3, color=RED)
arrow(47, 88.1, 55, 88.1); box(55.5, 85.9, 12, 4.4, "model builds\nthe 3D structure", "#cfe8ef", fs=8.8)
arrow(68, 88.1, 72, 88.1); ax.text(72.5, 88.1, "looks right, but the\nNCAA choice was copied", fontsize=8.8, va="center", color=RED, fontweight="bold")
ax.text(2, 77, "Identity removed", fontsize=9, fontweight="bold"); strip(13, 75.2, ["T", "F", "?", "L", "D", "?", "P", "E"], mark=["?"])
arrow(47, 77.3, 55, 77.3); box(55.5, 75.1, 12, 4.4, "pipeline fills it\nfrom the input file", "#f7d6d6", fs=8.8, ec=RED)
arrow(68, 77.3, 72, 77.3); ax.text(72.5, 77.3, "native NCAA code returned\nin 100% of 176 slots", fontsize=8.8, va="center", color=RED, fontweight="bold")
ax.text(2, 70.2, "Ask the denoiser's readout to guess the hidden NCAA (1,065 real NCAA tokens):", fontsize=9, fontweight="bold")
bars = [("always guess\nthe commonest (ALY)", 10, GREY), ("untuned\ncontrol", 9.5, GREY), ("our S1\n(trained readout)", 7.3, "#e76f51")]
a1 = f.add_axes([0.07, 0.575, 0.2, 0.105]); a1.barh(range(3), [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="#333", lw=0.5)
a1.set_yticks(range(3)); a1.set_yticklabels([b[0] for b in bars], fontsize=8); a1.invert_yaxis(); a1.set_xlim(0, 40)
for i, b in enumerate(bars): a1.text(b[1] + 0.8, i, f"{b[1]}%", va="center", fontsize=9)
a1.set_title("right code in the top guess (%)", fontsize=9, loc="left", fontweight="bold"); a1.spines[["top", "right"]].set_visible(False)
bars2 = [("untuned control", 12, GREY), ("our S1", 28, "#e76f51")]
a2 = f.add_axes([0.40, 0.575, 0.2, 0.105]); a2.barh(range(2), [b[1] for b in bars2], color=[b[2] for b in bars2], edgecolor="#333", lw=0.5)
a2.set_yticks(range(2)); a2.set_yticklabels([b[0] for b in bars2], fontsize=8); a2.invert_yaxis(); a2.set_xlim(0, 40)
for i, b in enumerate(bars2): a2.text(b[1] + 0.8, i, f"{b[1]}%", va="center", fontsize=9)
a2.set_title("guesses that are a plain canonical residue (%)", fontsize=9, loc="left", fontweight="bold"); a2.spines[["top", "right"]].set_visible(False)
box(65, 58.5, 33, 11, "Net: the model has no idea which NCAA\nbelongs where, and leans to canonical.\nIt only learned: \"given this NCAA, build it\".", "#fff3cd", fs=9.5, bold=True)

# ---------- 2 WHY ----------
head(51.8, "2", "Why: four causes")
why = [("Identity is an input", "readout never saw\na hidden NCAA\nduring training"), ("Data has one rule", "synthetic NCAA =\nD-form of the parent;\nnot chosen to bind better"),
       ("No signal for 'better'", "nothing tells the model\nwhy ALY beats LYS\nat this pocket"), ("Pretrained on canonical", "20 amino acids only;\nNCAA classes ~0%\nin the readout")]
for i, (t, d) in enumerate(why):
    x = 2 + i * 24.5; box(x, 42.5, 22.5, 8, t, "#f7d6d6", fs=10, bold=True, ec=RED); ax.text(x + 11.25, 39.5, d, ha="center", va="top", fontsize=8.8)

# ---------- 3 TARGET ----------
head(31.5, "3", "Target: at most 5 NCAAs per design, each chosen because it beats the canonical residue")
ax.text(2, 26.3, "Designer: pocket + max 5 NCAAs", fontsize=9, fontweight="bold")
res = ["T", "F", "K", "L", "D", "R", "P", "E", "W", "S", "V", "Y"]
strip(2, 21.5, ["T", "F", "ALY", "L", "D", "DAR", "P", "E", "W", "MeS", "V", "Y"], w=3.6, h=3.8, mark=["ALY", "DAR", "MeS"])
ax.text(2, 19.6, "model picks the positions AND the NCAAs (here 3 of 12; never more than 5)", fontsize=8.5, color=GRN, fontweight="bold")
ax.text(2, 16.3, "Position 3, pocket wants a charge-neutral, buried contact:", fontsize=8.8, fontweight="bold")
for i, (n, s, c) in enumerate([("LYS", "score 0.0 (reference)", CAN), ("DLY", "-0.1  worse fold", CAN), ("ALY", "+0.6  better contact", NC)]):
    box(2 + i * 15, 11.2, 5, 3.6, n, c, fs=9, bold=(n == "ALY")); ax.text(2 + i * 15 + 5.4, 13, s, fontsize=8, va="center")
ax.text(2, 9.3, "The model must learn the difference between the NCAA and the canonical residue at the same slot (a score gain),", fontsize=8.8)
ax.text(2, 7.4, "not copy the generator's rule. Rule of the cap: keep the 5 slots with the biggest gain.", fontsize=8.8)

# ---------- 4 PLAN ----------
ax.text(52, 31.5, "4", fontsize=15, fontweight="bold", color=BLU); ax.text(54.7, 31.5, "Training plan", fontsize=13, fontweight="bold")
steps = [("A  Open the door", "patch pipeline: identity hidden,\ntoken type = 'designable'"), ("B  Pair data", "same pocket + position:\ncanonical vs NCAA variant + score gain"),
         ("C  Train the choice", "readout picks canonical or NCAA;\nloss on the gain; cap 5 per design"), ("D  Check (real, test only)", "gain over canonical, # NCAAs <= 5,\nstructure quality unchanged")]
for i, (t, d) in enumerate(steps):
    y = 24.5 - i * 5.9; box(52, y - 1.2, 15, 4.6, t, "#d8f0e8", fs=9, bold=True, ec=GRN); ax.text(68, y + 1.1, d, fontsize=8.8, va="center")
    if i < 3: arrow(59.5, y - 1.3, 59.5, y - 2.8, GRN)
ax.text(52, 1.5, "Score for B/C/D: Rosetta interface dG (PyRosetta, repack + relax), NCAA variant minus canonical. Runs free on the laptop (WSL), no GPU. ipTM dropped: not an affinity measure.", fontsize=8.4, color="#444")
f.savefig(f"{D}/figures/fig70_plan_choose_ncaa.png", dpi=130, bbox_inches="tight"); plt.close(f); print("saved fig70")
