"""Trial S1 (G3 + sequence readout trained on NCAAs with the identity hidden) against G3 and the untuned control: the readout (189 classes) at the generated NCAA tokens of the real complexes (103 real + 365 extra real with an NCAA; test only), identity HIDDEN by hand (gen_test/eval_readout.py), generated tokens noised to a fixed time t (CA and latent). Training curve from the training log. Baseline = always guess the most common NCAA code of the set."""
import json, os, re, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; MD = [("control", "control (untuned)", "#999999"), ("G3_bal", "G3", "#6a4c93"), ("S1_readout", "S1 (G3 + trained readout)", "#e76f51")]
rows = {}
for k, *_ in MD:
    r = []
    for st in ("real", "ham"):
        p = f"{D}/gen_test/diag/readout_{k}_{st}.json"
        if os.path.exists(p): r += json.load(open(p))["rows"]
    rows[k] = r
TS = sorted({x["t"] for x in rows["control"]});
def sel(k, t, g=None): return [x for x in rows[k] if x["t"] == t and (g is None or x["group"] == g)]
cnt = collections.Counter(x["code"] for x in sel("control", TS[-1])); maj_code, maj_n = cnt.most_common(1)[0]; maj = maj_n / sum(cnt.values())
f = plt.figure(figsize=(19, 11)); gs = f.add_gridspec(3, 3, height_ratios=[0.4, 1, 1], hspace=0.55, wspace=0.27)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
s.add_patch(FancyBboxPatch((1, 0.8), 98, 8.4, boxstyle="round,pad=0.2", fc="#ffd6a5", ec="#555"))
s.text(50, 5, f"Trial S1: the sequence readout (189 classes) of G3 is trained on NCAAs with the identity hidden (readout attached to the trunk, weight 3, 600 steps, balanced data, 1 H200, about 18 min).\nTest: {len(sel('control', TS[-1]))} generated NCAA tokens of real complexes (test only), identity hidden by hand, noise time t fixed (0 = pure noise, 1 = clean). Baseline = always guess the most common code ({maj_code}, {100 * maj:.0f}% of the tokens).", ha="center", va="center", fontsize=8.4)
s.set_title("Can the denoiser's readout name the right non-canonical residue?", fontsize=10.5, loc="left", fontweight="bold")
def curve(ax, fn, title, ylabel, base=None, ylim=None):
    for k, lab, col in MD:
        ax.plot(TS, [fn(sel(k, t)) for t in TS], "-o", color=col, label=lab, ms=4)
    if base is not None: ax.axhline(base, color="#c1121f", ls="--", lw=0.8, label=f"always {maj_code}")
    ax.set_xlabel("noise time t (0 = pure noise, 1 = clean)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9.2, loc="left", fontweight="bold")
    if ylim: ax.set_ylim(*ylim)
ax = f.add_subplot(gs[1, 0]); curve(ax, lambda r: 100 * np.mean([x["top1"] for x in r]), "(a) top-1: exact NCAA code (%)", "%", 100 * maj, (0, 40)); ax.legend(fontsize=7, frameon=False)
ax = f.add_subplot(gs[1, 1]); curve(ax, lambda r: 100 * np.mean([x["top5"] for x in r]), "(b) top-5 contains the code (%)", "%", None, (0, 80))
ax = f.add_subplot(gs[1, 2]); curve(ax, lambda r: 100 * np.mean([x["guess_canonical"] for x in r]), "(c) guess is a CANONICAL class (%, the canonical bias; lower = better)", "%", None, (0, 100))
ax = f.add_subplot(gs[2, 0]); curve(ax, lambda r: 100 * np.mean([x["guess_D"] for x in r]), "(d) guess is a D-form code (%)", "%", None, (0, 100))
G = ["D-forms", "general codes", "other non-canonical"]; tt = TS[-1]; ax = f.add_subplot(gs[2, 1]); w = 0.26
for j, (k, lab, col) in enumerate(MD):
    vals = [100 * np.mean([x["top1"] for x in sel(k, tt, g)]) if sel(k, tt, g) else 0 for g in G]; xs = np.arange(3) + (j - 1) * w
    ax.bar(xs, vals, w * 0.95, color=col, edgecolor="#333", linewidth=0.5, label=lab)
    for x, v in zip(xs, vals): ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{g}\n(n={len(sel('control', tt, g))})" for g in G], fontsize=8); ax.set_ylabel("%"); ax.set_title(f"(e) top-1 at t = {tt} by residue group", fontsize=9.2, loc="left", fontweight="bold")
ax = f.add_subplot(gs[2, 2]); logp = f"{D}/denoiser_trial/saved/H200_S1/dn_s1_H200_502194.train.log"; st, t1, t5, cg = [0], [], [], []
if os.path.exists(logp):
    pend = None
    for ln in open(logp, errors="ignore"):
        m = re.match(r"\s+held-out readout at NCAA tokens \(identity hidden\): n (\d+) top-1 (\S+) top-5 (\S+) \| guess is a CANONICAL class in (\S+) of", ln)
        if m: pend = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        m = re.match(r"eval step (\d+) val", ln)
        if m and pend: st.append(int(m.group(1))); t1.append(pend[0] * 100); t5.append(pend[1] * 100); cg.append(pend[2] * 100); pend = None
    st = st[1:] if len(st) > len(t1) else st
    ax.plot(st, t1, "-o", color="#e76f51", label="top-1"); ax.plot(st, t5, "-o", color="#2a9d8f", label="top-5"); ax.plot(st, cg, "-o", color="#6a4c93", label="guess canonical")
ax.set_xlabel("training step"); ax.set_ylabel("% of 82 held-out NCAA tokens"); ax.legend(fontsize=7, frameon=False); ax.set_title("(f) S1 training: held-out readout (identity hidden, synthetic held-out set)", fontsize=9.2, loc="left", fontweight="bold")
b1, s1 = 100 * np.mean([x["top1"] for x in sel("G3_bal", tt)]), 100 * np.mean([x["top1"] for x in sel("S1_readout", tt)]); b5, s5 = 100 * np.mean([x["top5"] for x in sel("G3_bal", tt)]), 100 * np.mean([x["top5"] for x in sel("S1_readout", tt)])
c3, cs = 100 * np.mean([x["guess_canonical"] for x in sel("G3_bal", tt)]), 100 * np.mean([x["guess_canonical"] for x in sel("S1_readout", tt)])
f.text(0.01, 0.005, "Result at t = %.1f: top-1 G3 %.1f%% -> S1 %.1f%% (always-%s baseline %.0f%%); top-5 %.1f%% -> %.1f%%; guesses that are canonical %.1f%% -> %.1f%%." % (tt, b1, s1, maj_code, 100 * maj, b5, s5, c3, cs), fontsize=8.6, fontweight="bold")
f.savefig(f"{D}/figures/fig69_readout.png", bbox_inches="tight", dpi=130); plt.close(f); print("saved fig69", {k: len(v) for k, v in rows.items()})
