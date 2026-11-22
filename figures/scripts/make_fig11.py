"""Figure: how the NCFlow reward fine-tune is set up (for a non-specialist)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


def fig(OUT, C):
    f = plt.figure(figsize=(15, 10))
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.3, 9.6, "Reward fine-tune of NCFlow: teach the model to draw residues with correct chemistry", fontsize=14, fontweight="bold")
    ax.text(0.3, 9.2, "Reward = a penalty for wrong bond lengths, wrong angles and a wrong bond to the neighbour residue. Lower penalty = better.", fontsize=10.5)

    def box(x, y, w, h, title, body, fc, ec="#495057"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.75, body, ha="center", va="top", fontsize=9.2)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#495057"))

    # row 1: the loop
    box(0.3, 6.3, 2.7, 2.3, "1. Start", "the model after the\nlarger fine-tune (ft2):\nit already draws a missing\nresidue, often with bad\nchemistry", "#dbe9f4")
    box(3.4, 6.3, 2.9, 2.3, "2. Take an example", "a peptide with one residue\nhidden. We know the right\nanswer (a real residue or one\nof our synthetic residues)", "#e9ecef")
    box(6.7, 6.3, 2.7, 2.3, "3. Blur the answer", "add random noise to the\nright answer, then ask the\nmodel for its best guess of\nthe clean residue", "#fff3cd", "#c9a227")
    box(9.8, 6.3, 2.5, 2.3, "4. Score the guess", "two scores (below)", "#f8d7da", "#b23a48")
    box(12.7, 6.3, 2.0, 2.3, "5. Update", "nudge the model\nso both scores get\nbetter. Repeat\n6,000 times", "#d8f0dc", "#3a7d44")
    for x0, x1 in ((3.0, 3.4), (6.3, 6.7), (9.4, 9.8), (12.3, 12.7)):
        arrow(x0, 7.45, x1, 7.45)

    # row 2: the two scores
    ax.add_patch(FancyBboxPatch((0.3, 3.0), 6.0, 2.9, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#f1f3f5", ec="#495057", lw=1.3))
    ax.text(3.3, 5.6, "Score A: copy the right answer (the usual training loss)", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(3.3, 5.1, "How far is the guess from the known correct residue?\nThis is what the earlier fine-tunes used.", ha="center", va="top", fontsize=9.5)
    for k, (x, c) in enumerate(((1.6, "#2a6f97"), (3.0, "#b23a48"))):
        ax.add_patch(Circle((x, 3.75), 0.2, fc=c, ec="#222"))
    ax.plot([1.6, 3.0], [3.75, 3.75], color="#495057", ls=":")
    ax.text(2.3, 3.45, "guess vs answer", ha="center", fontsize=8.5)
    ax.text(5.0, 3.8, "A = distance\nbetween the two", ha="center", fontsize=9.5)

    ax.add_patch(FancyBboxPatch((6.7, 3.0), 8.0, 2.9, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fdecef", ec="#b23a48", lw=1.6))
    ax.text(10.7, 5.6, "Score B: the chemistry reward (new)", ha="center", fontsize=10.5, fontweight="bold")
    # three mini panels
    panels = [(7.2, "Bond length", "each bond should be\nclose to its textbook\nlength (about 1.5 A)"),
              (9.8, "Bond angle", "each angle should be\nclose to its textbook\nvalue (about 110 deg)"),
              (12.4, "Link to the neighbour", "the peptide bond to the\nnext residue should\nbe 1.33 A")]
    for x, t, b in panels:
        ax.text(x + 0.9, 5.25, t, ha="center", fontsize=9.8, fontweight="bold")
        ax.text(x + 0.9, 3.3, b, ha="center", va="bottom", fontsize=8.6)
    # bond length drawing
    ax.add_patch(Circle((7.6, 4.45), 0.17, fc="#2a6f97", ec="#222")); ax.add_patch(Circle((8.5, 4.45), 0.17, fc="#2a6f97", ec="#222"))
    ax.plot([7.6, 8.5], [4.45, 4.45], color="#222"); ax.text(8.05, 4.7, "too long?", ha="center", fontsize=8.5, color="#b23a48")
    # angle drawing
    ax.plot([10.2, 10.7, 11.2], [4.1, 4.75, 4.1], color="#222")
    for x, y in ((10.2, 4.1), (10.7, 4.75), (11.2, 4.1)):
        ax.add_patch(Circle((x, y), 0.15, fc="#2a6f97", ec="#222"))
    ax.text(10.7, 4.3, "angle?", ha="center", fontsize=8.5, color="#b23a48")
    # link drawing
    ax.add_patch(Circle((12.8, 4.4), 0.17, fc="#8d99ae", ec="#222")); ax.add_patch(Circle((13.7, 4.4), 0.17, fc="#2a6f97", ec="#222"))
    ax.plot([12.8, 13.7], [4.4, 4.4], color="#222"); ax.text(13.25, 4.7, "1.33 A?", ha="center", fontsize=8.5, color="#b23a48")
    ax.text(12.6, 4.05, "neighbour", ha="center", fontsize=7.8, color="#555"); ax.text(13.9, 4.05, "new", ha="center", fontsize=7.8, color="#555")
    ax.text(10.7, 3.05, "", fontsize=1)

    arrow(10.9, 6.3, 10.9, 5.95)
    arrow(3.3, 6.3, 3.3, 5.95)

    # row 3: formula and timeline
    ax.add_patch(FancyBboxPatch((0.3, 1.2), 14.4, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fff8e1", ec="#c9a227", lw=1.3))
    ax.text(7.5, 2.4, "Total score to minimise  =  A (copy the answer)  +  1.0 x trust x B (chemistry penalty)", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(7.5, 1.95, "'trust' = (1 - noise)^2: when the answer was hardly blurred the model's guess is reliable, so the penalty counts fully;\nwhen it was heavily blurred the guess is rough, so the penalty counts little (and each penalty is capped so one bad guess cannot dominate).", ha="center", va="top", fontsize=9.3)

    ax.text(0.3, 0.75, "Versions we compare, all tested on the same held-out residues:", fontsize=10, fontweight="bold")
    chips = [(5.2, "raw", "#f8d7da"), (6.4, "ft1", "#fff3cd"), (7.6, "ft2", "#fff3cd"), (8.8, "reward (ft2 + B)", "#d8f0dc")]
    for x, t, fc in chips:
        w = 2.0 if "reward" in t else 0.9
        ax.add_patch(FancyBboxPatch((x, 0.4), w, 0.55, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#495057"))
        ax.text(x + w / 2, 0.67, t, ha="center", va="center", fontsize=9)
    ax.text(0.3, 0.2, "This is 'reward fine-tuning': the model is trained directly to lower the penalty. It is a simple form of reinforcement learning (the reward is measured on the model's own guess, no trial-and-error sampling loop).", fontsize=8.8, color="#444")
    f.savefig(os.path.join(OUT, "fig11_reward_finetune_setup.png"), bbox_inches="tight")
    plt.close(f)
