"""Denoiser trial G4 (G3 + clash loss through the frozen run 8 decoder) vs G3: generation test, 11 real pockets, 88 designs each, decoder run 8: (a) designs with an atom pair < 2 A to the receptor, (b) peptide bonds of sensible length, (c) D-forms with the right handedness, (d) median CA distance to the native peptide, (e) contacts relative to native, (f) held-out check during training (weighted clash part)."""
import json, re, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json"))
LOG = f"{D}/denoiser_trial/saved/H200_G4/dn_v4_H200_501522.train.log"
RUNS = [("G3_bal_v10", "G3 (start)", "#6a4c93"), ("G4_clash_v10", "G4 = G3 + clash loss (weight 2)", "#e76f51")]
f = plt.figure(figsize=(19, 11.5)); gs = f.add_gridspec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.5, wspace=0.25)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Clash loss in the denoiser", "predicted clean latent + CA of the denoiser -> frozen run 8 decoder\n-> decoded peptide atoms vs TRUE receptor atoms: max(0, 2.5 A - d)^2,\nweight 2, only samples with both noise levels >= 0.6 (about 17%)", "#ffd6a5"),
       ("Training", "start = G3 best (LoRA), same class-balanced data, 600 steps, batch 4,\n5 inner updates, lr 3e-5, 1 full H200 (about 19 min); peptide = chains\nholding a non-canonical residue (chain_iid)", "#ade8f4"),
       ("Test", "generation: 11 real pockets, identity of the non-canonical residues given,\n88 designs per run, decoder run 8, raw outputs, same seeds as G3", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 0.5), 31, 9, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 8.1, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 4.1, d, ha="center", va="center", fontsize=8.2)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: a clash loss on the decoded atoms inside the denoiser training", fontsize=10.5, loc="left", fontweight="bold")
M = [("clash", "(a) designs with an atom pair < 2 A to the receptor (%, lower = better)", 100, "{:.1f}"), ("bond_ok", "(b) peptide bonds of sensible length (%)", 100, "{:.1f}"), ("chir_D", "(c) D-forms with the right handedness (%)", 100, "{:.1f}"),
     ("ca_rmsd_med", "(d) median CA distance to the native peptide (A, lower = better)", 1, "{:.2f}"), ("contacts_rel_med", "(e) contacts relative to the native peptide", 1, "{:.2f}")]
for i, (k, title, mult, fmt) in enumerate(M):
    ax = f.add_subplot(gs[1, i] if i < 3 else gs[2, i - 3])
    vals = [S[r]["agg"][k] * mult for r, _, _ in RUNS]; ax.bar(range(len(RUNS)), vals, color=[c for *_, c in RUNS], edgecolor="#333", linewidth=0.5)
    for x, v in enumerate(vals): ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(RUNS))); ax.set_xticklabels(["G3", "G4"]); ax.set_title(title, fontsize=9.3, loc="left", fontweight="bold")
    if mult == 100: ax.set_ylim(0, 112)
    else: ax.set_ylim(0, max(vals) * 1.25)
# training curves from the log
steps, flow, cl = [], [], []
ev_s, ev_v, ev_c = [], [], []
for ln in open(LOG, errors="ignore"):
    m = re.match(r"step (\d+) loss \S+ \d+s \| flow (\S+) clash (\S+) x w (\S+) pairs", ln)
    if m and float(m.group(2)) < 1000: steps.append(int(m.group(1))); flow.append(float(m.group(2))); cl.append(float(m.group(3)) * float(m.group(4)))
    m = re.match(r"eval step (\d+) val (\S+) terms .*'clash_w': ([\d.]+)", ln)
    if m: ev_s.append(int(m.group(1))); ev_v.append(float(m.group(2))); ev_c.append(float(m.group(3)))
def sm(v, w=30): v = np.array(v); return np.convolve(v, np.ones(w) / w, mode="valid")
ax = f.add_subplot(gs[2, 2]); ax.plot(steps[29:], sm(flow), color="#6a4c93", label="training flow loss (30-step mean)")
ax2 = ax.twinx(); ax2.plot(steps[29:], sm(cl), color="#e76f51", label="training clash term x weight (30-step mean)"); ax2.plot(ev_s, ev_c, "o--", color="#c1121f", label="held-out clash term x weight")
ax.set_xlabel("step"); ax.set_ylabel("flow loss"); ax2.set_ylabel("clash term x weight"); ax.set_title("(f) training: flow loss and clash term", fontsize=9.3, loc="left", fontweight="bold")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.8, frameon=False, loc="upper center")
f.legend(handles=[Patch(fc=c, ec="#333", label=l) for _, l, c in RUNS], frameon=False, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.025))
g3, g4 = S["G3_bal_v10"]["agg"], S["G4_clash_v10"]["agg"]
f.text(0.01, 0.005, "Generation: raw outputs, 88 designs per run, unrelaxed; one design = 1.1 points. G3 -> G4: close contacts %.0f -> %.0f%%, bonds %.0f -> %.0f%%, D right %.0f -> %.0f%%, CA distance %.2f -> %.2f A." % (100 * g3["clash"], 100 * g4["clash"], 100 * g3["bond_ok"], 100 * g4["bond_ok"], 100 * g3["chir_D"], 100 * g4["chir_D"], g3["ca_rmsd_med"], g4["ca_rmsd_med"]), fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig63_denoiser_clash.png", bbox_inches="tight", dpi=130); plt.close(f)
print({r: {k: round(S[r]["agg"][k], 3) for k in ("clash", "bond_ok", "chir_D", "ca_rmsd_med", "contacts_rel_med", "nc_kept", "free_rec", "nc_atoms")} for r, _, _ in RUNS})
