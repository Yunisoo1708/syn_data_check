"""How every denoiser was obtained (lineage of the LoRA trials). All numbers from Results.MD 43-46, 50, 53, 55, 66, 68, 72 and denoiser/PRESETS.md (no new run).
Scores: paired change of the flow loss against the untuned control (negative = better): 163 held-out synthetic complexes (all positions) and the 26 real complexes that contain one of the 13 new D-forms. Generation: median CA distance to the native peptide (decoder run 8)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D = "C:/Research/RFD4_fine_tune"
C1, C2, C3, C4, C5, C6, CG = "#cfe8ef", "#f7d6d6", "#fff0c2", "#b7e4c7", "#b7e4c7", "#e3e3e3", "#e9c46a"
f = plt.figure(figsize=(21, 15)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, lines, fc, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.5", fc=fc, ec="#333", lw=lw))
    ax.text(x + w / 2, y + h - 1.3, lines[0], ha="center", va="top", fontsize=9.6, fontweight="bold")
    ax.text(x + w / 2, y + h - 3.6, "\n".join(lines[1:]), ha="center", va="top", fontsize=7.7, linespacing=1.35)
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="#444", lw=1.5))
ax.text(1.2, 97.8, "How each denoiser was obtained", fontsize=16, fontweight="bold")
ax.text(1.2, 95.3, "Every trial = a small LoRA adapter (rank 16 unless noted) on the frozen pretrained 189-class RFD4 flow model, trained on synthetic complexes. Scores = change of the flow loss vs the untuned control (negative = better):\nfirst number = 163 held-out synthetic complexes, second = the 26 real complexes that hold a new D-form (test only).", fontsize=9, color="#444", va="top")
W, GAP, X0 = 12.6, 1.6, 1.2; H = 11.5; cols = [X0 + i * (W + GAP) for i in range(7)]
heads = ["start", "1  learn the basics\nLoRA on attention + 2 heads", "2  more capacity\n(did not help)", "3  more data\n(gain only on synthetic)", "4  change the loss\n(best so far)", "5  bigger / balanced\ndata", "6  extra signals\n(no gain)"]
for c, h in zip(cols, heads): ax.text(c + W / 2, 90, h, ha="center", va="center", fontsize=9.4, fontweight="bold", color="#264653")
Y = [74, 60, 46]
box(cols[0], Y[1], W, H, ["Control", "untuned RFD4 flow", "model (189 classes)", "reference of every test"], "#dddddd", 1.4)
box(cols[1], Y[0], W, H, ["A", "1,266 complexes, batch 1", "lr 1e-4, best step 500", "-0.41  |  -0.94"], C1)
box(cols[1], Y[1], W, H, ["B", "first batch only (228)", "best step 800", "-0.35  |  -1.05"], C1)
box(cols[1], Y[2], W, H, ["C", "A run longer (stop 1,143)", "best step 600", "-0.47  |  -1.04"], C1)
box(cols[2], Y[0], W, H, ["D", "rank 64 (4x bigger)", "best step 1,050", "-0.14  |  -0.46  worse"], C2)
box(cols[2], Y[1], W, H, ["E", "rank 64 + feed-forward", "layers + gate, step 150", "+0.22  |  -0.04  worst"], C2)
box(cols[3], Y[0], W, H, ["F  (box and H200)", "C recipe, 6,457-complex", "pool (batch 4, lr 2e-4)", "-0.30  |  -0.47  (box -0.37 | -0.83)"], C3)
box(cols[3], Y[1], W, H, ["F2", "F continued, lr 5e-5", "best step 300", "-0.54  |  -0.61"], C3)
box(cols[4], Y[1], W, H, ["G", "C + 10x loss on NCAA", "positions, noise weight", "capped, lr halved; step 2,700", "-0.68  |  -1.00"], C4, 1.6)
box(cols[5], Y[0], W, H, ["G2", "G continued on the", "8,947-complex pool", "best step 1,200", "-0.66  |  -1.14"], C5)
box(cols[5], Y[1], W, H, ["G3  DEFAULT", "G continued on a class-", "balanced subset (2,232", "complexes, 25 codes x 280)", "-0.69  |  -1.12"], CG, 2.0)
box(cols[6], Y[0], W, H, ["G4", "G3 + clash loss through", "the frozen decoder", "(600 steps)  no gain"], C6)
box(cols[6], Y[1], W, H, ["S1", "G3 + readout trained", "on hidden NCAA identity", "no transfer to real NCAAs"], C6)
box(cols[6], Y[2], W, H, ["H  (skipped)", "G3 + mirror-image", "complexes", "not needed any more"], "#f0f0f0")
# arrows between groups
arr(cols[0] + W, Y[1] + H / 2, cols[1], Y[1] + H / 2); arr(cols[1] + W, Y[1] + H / 2, cols[2], Y[1] + H / 2); arr(cols[2] + W, Y[1] + H / 2, cols[3], Y[1] + H / 2)
arr(cols[3] + W, Y[1] + H / 2, cols[4], Y[1] + H / 2); arr(cols[4] + W, Y[1] + H / 2, cols[5], Y[1] + H / 2); arr(cols[5] + W, Y[1] + H / 2, cols[6], Y[1] + H / 2)
ax.text(50, 43.2, "Arrows = the next idea tried after the previous result. Solid starts: F2 from F, G2 and G3 from G, G4 and S1 from G3 (they continue the earlier weights); the others were trained from the control with the recipe written in the box.", fontsize=8.2, color="#444", ha="center")

# ---------- charts ----------
names = ["A", "B", "C", "D", "E", "F box", "F H200", "F2", "G", "G2", "G3"]
held = [-0.41, -0.35, -0.47, -0.14, 0.22, -0.37, -0.30, -0.54, -0.68, -0.66, -0.69]
dform = [-0.94, -1.05, -1.04, -0.46, -0.04, -0.83, -0.47, -0.61, -1.00, -1.14, -1.12]
col = [C1[:] if False else "#8fb8de"] * 3 + ["#e76f51"] * 2 + ["#e9c46a"] * 3 + ["#2a9d8f"] * 3; col[-1] = "#c9a227"
def bars(rect, vals, title, ylim, fmt="%.2f"):
    a = f.add_axes(rect); a.bar(range(len(vals)), vals, color=col, edgecolor="#333", lw=0.5)
    for i, v in enumerate(vals): a.text(i, v + (0.02 if v >= 0 else -0.02), fmt % v, ha="center", va="bottom" if v >= 0 else "top", fontsize=7.6)
    a.axhline(0, color="#333", lw=0.8); a.set_xticks(range(len(vals))); a.set_xticklabels(names, fontsize=8, rotation=35, ha="right"); a.set_ylim(*ylim); a.set_title(title, fontsize=9.6, loc="left", fontweight="bold"); a.spines[["top", "right"]].set_visible(False); return a
bars([0.05, 0.15, 0.27, 0.19], held, "(a) held-out synthetic complexes (163), all positions", (-0.85, 0.4))
bars([0.37, 0.15, 0.27, 0.19], dform, "(b) real complexes with a new D-form (26)", (-1.4, 0.3))
a = f.add_axes([0.69, 0.15, 0.27, 0.19]); gn = ["control", "G", "G2", "G3", "G4"]; gv = [5.91, 4.91, 5.20, 5.23, 5.14]; gc = ["#999999", "#2a9d8f", "#2a9d8f", "#c9a227", "#e76f51"]
a.bar(range(5), gv, color=gc, edgecolor="#333", lw=0.5)
for i, v in enumerate(gv): a.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
a.set_xticks(range(5)); a.set_xticklabels(gn); a.set_ylim(0, 7); a.set_ylabel("A"); a.set_title("(c) generated designs: median CA distance to native (A)", fontsize=9.6, loc="left", fontweight="bold"); a.spines[["top", "right"]].set_visible(False)
f.text(0.05, 0.065, "Reading: the biggest step was the change of the loss in G (-0.68 on synthetic, -1.00 on real D-form complexes); more capacity (D, E) hurt, more data alone (F, F2) helped only the synthetic set, and G2/G3 kept G's gains with more or balanced data.\nIn generation the fine-tuned denoisers place the peptide 0.7 to 1.0 A closer to the native one than the control; G4 (clash loss) and S1 (readout) gave no gain. One run per trial; flow-loss tests, not generated structures, except panel (c).", fontsize=8.6, fontweight="bold")
f.text(0.05, 0.03, "Weights: denoiser_trial/saved/ (box_out/A_all ... H200_G3/dn_G3_H200_495885/best, H200_G4, H200_S1). Table of every trial: denoiser/PRESETS.md. Sections: Results.MD 43-46, 50, 53, 55, 66, 72.", fontsize=7.8, color="#444")
f.savefig(f"{D}/figures/fig75_denoiser_lineage.png", dpi=130, bbox_inches="tight"); plt.close(f); print("saved fig75")
