"""Decoder journey: from the untouched RFD4 decoder to run 8 (default) and run 9, with the accuracy reached at each step.
All numbers are copied from Results.MD (no new run): sections 4 (176-class naming), 49 (189-class naming, mirror D-peptides), 58 (blind decoder),
59-62 and 65 (generation test, 11 real pockets, 88 designs per denoiser; mean of denoisers G, G2, G3), 69 (per position, 1,066 real NCAA positions, true latents),
70 (denoise + decode). Real complexes are test only."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D = "C:/Research/RFD4_fine_tune"
GRN, GLD, RED, GRY, BLU, ORG = "#2a9d8f", "#e9c46a", "#e76f51", "#bdbdbd", "#264653", "#f4a261"
f = plt.figure(figsize=(20, 16.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.62, wspace=0.26, left=0.05, right=0.985, top=0.95, bottom=0.2)

# ---------- timeline strip ----------
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 20)
st = [("Start", "untouched RFD4\ndecoder, 147 classes", GRY), ("Runs 1-2 (176)", "+29 NCAA classes\nresidue tables only\n43k synthetic peptides", "#cfe8ef"), ("189 classes", "whole decoder\n+13 D-forms", "#cfe8ef"),
      ("Run 2 (189)", "+1,429 synthetic\ncomplexes", "#cfe8ef"), ("Run 3", "+7,534 mirror-image\nD-peptides", "#b7e4c7"), ("Run 4 (blind)", "identity hidden\nnot used for design", GRY),
      ("Runs 5, 5b", "+ atom-coordinate\nloss (atoms, not\nonly names)", "#b7e4c7"), ("Runs 6, 7", "+ chirality loss\nhandedness vs\nbonds trade-off", "#ffe8b0"), ("Run 8  DEFAULT", "+ peptide-bond\nloss", GLD), ("Run 9", "+ clash loss\nno gain, dropped", "#f7d6d6")]
w = 8.9; gap = 1.0
for i, (t, d, c) in enumerate(st):
    x = 0.4 + i * (w + gap); s.add_patch(FancyBboxPatch((x, 3), w, 13, boxstyle="round,pad=0.1,rounding_size=0.6", fc=c, ec="#333", lw=1.6 if "DEFAULT" in t else 0.8))
    s.text(x + w / 2, 14, t, ha="center", va="center", fontsize=8.6, fontweight="bold"); s.text(x + w / 2, 8.2, d, ha="center", va="center", fontsize=7.6)
    if i < len(st) - 1: s.add_patch(FancyArrowPatch((x + w + 0.05, 9.5), (x + w + gap - 0.05, 9.5), arrowstyle="-|>", mutation_scale=9, color="#444", lw=1))
s.text(0.4, 0.8, "Each step starts from the previous good one. Costs: 0.2 to 1.4 USD per run on the Tillicum H200 (runs 3 to 7 on a MIG slice, 8 and 9 on a full H200, about 37 min). Green = kept, gold = default, grey = not used, red = no gain.", fontsize=8, color="#444")
s.set_title("How the decoder was fine-tuned, step by step", fontsize=12, loc="left", fontweight="bold")

# ---------- (a) naming ----------
ax = f.add_subplot(gs[1, 0])
x1 = ["untouched", "run 1\n(20 ep)", "run 2\n(90 ep)"]; v1 = [25.1, 90.1, 99.6]
x2 = ["untouched", "run 1", "run 2", "run 3\n(mirror)"]; v2 = [17.9, 68.8, 64.3, 98.2]
ax.plot(range(3), v1, "-o", color=BLU, lw=2, label="176 classes: 32 trained NCAAs (446 real positions)")
ax.plot([3.5 + i for i in range(4)], v2, "-s", color=ORG, lw=2, label="189 classes: 13 new D-forms (112 real positions)")
for i, v in enumerate(v1): ax.text(i, v + 3, f"{v:.0f}%", ha="center", fontsize=8.5)
for i, v in enumerate(v2): ax.text(3.5 + i, v + 3, f"{v:.0f}%", ha="center", fontsize=8.5)
ax.set_xticks(list(range(3)) + [3.5 + i for i in range(4)]); ax.set_xticklabels(x1 + x2, fontsize=7.5); ax.set_ylim(0, 115); ax.set_ylabel("residue named correctly (%)")
ax.axvline(2.75, color="#999", ls=":"); ax.legend(fontsize=7.2, frameon=False, loc="lower right")
ax.set_title("(a) Naming a real NCAA (the model sees the peptide atoms)", fontsize=9.4, loc="left", fontweight="bold")

# ---------- (b) rebuild per position ----------
ax = f.add_subplot(gs[1, 1]); names = ["untouched", "run 3", "run 5b", "run 8"]; cols = [GRY, "#8fb8de", GRN, GLD]
rm = [3.90, 3.65, 0.64, 0.65]; ax.bar(range(4), rm, color=cols, edgecolor="#333", lw=0.6)
for i, v in enumerate(rm): ax.text(i, v + 0.08, f"{v:.2f} A", ha="center", fontsize=9)
ax.axhline(1.0, color=RED, ls="--", lw=0.8); ax.text(3.45, 1.07, "1 A", color=RED, fontsize=8, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylim(0, 4.6); ax.set_ylabel("median all-atom RMSD of the NCAA residue (A)")
ax.set_title("(b) Rebuilding the NCAA residue (1,066 real positions, true latents)", fontsize=9.4, loc="left", fontweight="bold")

# ---------- (c) rebuild quality shares ----------
ax = f.add_subplot(gs[1, 2]); M = [("within 1 A of native", [0, 0, 82, 82]), ("exactly the native atoms", [8, 25, 95, 97]), ("handedness right", [54, 59, 92, 89])]; w = 0.2
for j, (lab, vals) in enumerate(M):
    xs = np.arange(3) * 1.0 + 0
for j, nm in enumerate(names):
    vals = [m[1][j] for m in M]; xs = np.arange(3) + (j - 1.5) * 0.2; ax.bar(xs, vals, 0.19, color=cols[j], edgecolor="#333", lw=0.5, label=nm)
    for x, v in zip(xs, vals): ax.text(x, v + 1, f"{v}", ha="center", fontsize=7.2)
ax.set_xticks(range(3)); ax.set_xticklabels([m[0] for m in M], fontsize=8.5); ax.set_ylim(0, 115); ax.set_ylabel("% of NCAA positions"); ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper center")
ax.set_title("(c) Share of positions rebuilt well (same test)", fontsize=9.4, loc="left", fontweight="bold")

# ---------- (d) generation test across runs ----------
ax = f.add_subplot(gs[2, 0]); runs = ["3", "4*", "5", "5b", "6", "7", "8", "9"]
hand = [38.7, 30.0, 89.3, 95.3, 98.7, 98.0, 96.3, 97.4]; bond = [22.3, 24.0, 67.0, 70.7, 23.0, 51.7, 85.7, 82.6]; clos = [66.3, 80.3, 25.3, 24.3, 31.0, 26.7, 23.0, 26.5]
ax.plot(range(8), hand, "-o", color=GRN, lw=2, label="D-forms with the right handedness (higher = better)")
ax.plot(range(8), bond, "-s", color=BLU, lw=2, label="peptide bonds 1.2-1.5 A (higher = better)")
ax.plot(range(8), clos, "-^", color=RED, lw=2, label="designs with an atom pair < 2 A to receptor (lower = better)")
for xs_, vs, c in [(range(8), hand, GRN), (range(8), bond, BLU)]:
    for i, v in zip(xs_, vs): ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=7, color=c)
ax.axvspan(5.6, 6.4, color=GLD, alpha=0.3); ax.set_xticks(range(8)); ax.set_xticklabels(["run " + r for r in runs]); ax.set_ylim(0, 112); ax.set_ylabel("% of designs / bonds")
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.24)); ax.set_title("(d) Generated designs (11 real pockets, mean of 3 denoisers)", fontsize=9.4, loc="left", fontweight="bold")

# ---------- (e) trade-off ----------
ax = f.add_subplot(gs[2, 1]); lab = ["3", "5", "5b", "6", "7", "8", "9"]; bx = [22.3, 67.0, 70.7, 23.0, 51.7, 85.7, 82.6]; hy = [38.7, 89.3, 95.3, 98.7, 98.0, 96.3, 97.4]
ax.scatter(bx, hy, s=[260 if l == "8" else 110 for l in lab], c=[GLD if l == "8" else (RED if l == "9" else GRN) for l in lab], edgecolor="#333", zorder=3)
for l, x, y in zip(lab, bx, hy):
    off = (-8, 10) if l == "9" else ((8, -12) if l == "8" else (8, -3))
    ax.annotate("run " + l, (x, y), textcoords="offset points", xytext=off, fontsize=8.5, fontweight="bold" if l == "8" else "normal", ha="right" if l == "9" else "left")
ax.annotate("", xy=(97, 101), xytext=(75, 80), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5)); ax.text(88, 79, "better", fontsize=9, color="#555", rotation=0)
ax.set_xlim(10, 100); ax.set_ylim(30, 104); ax.set_xlabel("peptide bonds of sensible length (%)"); ax.set_ylabel("D-forms with the right handedness (%)")
ax.set_title("(e) The trade-off the bond loss removed", fontsize=9.4, loc="left", fontweight="bold"); ax.grid(alpha=0.25)

# ---------- (f) denoise + decode ----------
ax = f.add_subplot(gs[2, 2]); T = [0.9, 0.7, 0.5, 0.3, 0.1]
M2 = {"control denoiser": [1.03, 2.00, 4.26, 7.40, 9.35], "G": [1.05, 2.08, 4.20, 6.99, 8.97], "G2": [1.05, 2.04, 3.98, 6.64, 8.81], "G3": [1.03, 2.04, 4.01, 6.81, 8.98]}
for (k, v), c in zip(M2.items(), [GRY, GRN, BLU, ORG]): ax.plot(T, v, "-o", color=c, label=k + " + decoder run 8", ms=4)
ax.axhline(0.66, color=GLD, lw=2.2); ax.text(0.1, 1.0, "decoder run 8 alone on the true latent: 0.66 A (84% within 1 A)", fontsize=8, color="#8a6d00", ha="right")
ax.axhline(4.0, color=GRY, ls=":", lw=1); ax.text(0.1, 4.25, "untouched decoder alone: 4.0 A", fontsize=8, color="#666", ha="right")
ax.set_xlabel("noise level t of the generated tokens (1 = clean, 0 = pure noise)"); ax.set_ylabel("median all-atom RMSD of the NCAA (A)"); ax.set_ylim(0, 10); ax.invert_xaxis(); ax.legend(fontsize=7.2, frameon=False, loc="upper left")
ax.set_title("(f) Denoiser + decoder together (1,065 real NCAA positions)", fontsize=9.4, loc="left", fontweight="bold")

f.text(0.05, 0.075, "Where it stands: given the NCAA identity, run 8 rebuilds the residue to a median 0.65 A (82% within 1 A, native atoms 97%, handedness 89 to 92%) against 3.9 A / 0% / 8% / 54% for the untouched decoder; in designs D-forms are right in 92 to 100%, bonds sensible in 83 to 87%,\nand a best-of-4 filter removes the close contacts (98 to 100% clash-free for most denoisers). Not solved: naming an NCAA from the latent alone (the identity must be given), and the denoiser adds about 0.4 A (46% within 1 A at t = 0.9).", fontsize=8.6, fontweight="bold")
f.text(0.05, 0.04, "Caveats: (a) 189-class naming probably had the true identity as input (Results.MD 58); (d,e) 88 designs per denoiser, 11 pockets, G/G2/G3 mean, run 4* = blind decoder, one run per setting, raw unrelaxed designs; (b,c,f) real complexes are test only; run 9 = clash loss.", fontsize=7.6, color="#444")
f.savefig(f"{D}/figures/fig72_decoder_journey.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig72")
