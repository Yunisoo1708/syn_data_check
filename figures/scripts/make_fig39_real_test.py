"""Figure: untuned control vs trials A-E scored on the REAL peptide-protein complexes of the real benchmark (TEST ONLY, never trained on).
Flow loss, same fixed noise per complex for every model (paired). Split by whether the peptide contains one of the 13 new D-forms (the residues the original decoder was never trained on) or not."""
import os, json, csv, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; E = f"{D}/denoiser_trial/saved/eval_real"
ITEMS = "C:/Research/Dataset_ncaa/real_benchmark/complex_test/complex_items.tsv"
NEWD = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split())
MODELS = [("control", "Untuned control", "#8d99ae"), ("A_all", "Trial A", "#2a6f97"), ("B_first", "Trial B", "#c9a227"), ("C_long", "Trial C", "#b23a48"), ("D_rank64", "Trial D", "#3a7d44"), ("E_rank64_ff", "Trial E", "#7b5ea7"), ("F_pool2", "Trial F (box)", "#e07a1f"), ("F_H200", "Trial F (H200)", "#17a2b8"), ("G_ncw10", "Trial G", "#d62768")]
R = {n: json.load(open(f"{E}/{n}.json"))["results"] for n, _, _ in MODELS if os.path.exists(f"{E}/{n}.json")}
MODELS = [m for m in MODELS if m[0] in R]
files = sorted(set.intersection(*[{f for f, v in R[n].items() if "total" in v} for n in R]))
seq = {r["item_id"]: r["ccd_sequence"].split("-") for r in csv.DictReader(open(ITEMS), delimiter="\t")}
iid = [f.replace("_real.cif", "") for f in files]
hasD = np.array([bool(NEWD & set(seq.get(i, []))) for i in iid])
TERMS = ["bb_ca", "local_latents", "sequence_prediction"]
def arr(n, k="total"): return np.array([R[n][f][k] for f in files])
ctrl = arr("control"); cut = np.percentile(ctrl, 95); keep = ctrl <= cut
rng = np.random.default_rng(0)
def boot(d, n=2000):
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return np.percentile(m, [2.5, 97.5])
rows = {}
print(f"real complexes scored in all models: {len(files)}; trimmed (control loss <= p95): {int(keep.sum())}; with a new D-form: {int((hasD & keep).sum())}; without: {int((~hasD & keep).sum())}")
for n, lab, col in MODELS:
    a = arr(n); d = (a - ctrl)[keep]
    rows[n] = dict(mean=a[keep].mean(), diff=d.mean(), ci=boot(d), win=float((d < 0).mean()),
                   d_dform=(a - ctrl)[keep & hasD].mean() if (keep & hasD).any() else np.nan, d_other=(a - ctrl)[keep & ~hasD].mean(),
                   terms={t: arr(n, t)[keep].mean() for t in TERMS}, seqterm_dform=arr(n, "sequence_prediction")[keep & hasD].mean(), seqterm_other=arr(n, "sequence_prediction")[keep & ~hasD].mean())
    print(f"{lab:16s} mean {rows[n]['mean']:7.3f} | change vs control {rows[n]['diff']:+.3f} (95% CI {rows[n]['ci'][0]:+.3f} to {rows[n]['ci'][1]:+.3f}) | lower on {100 * rows[n]['win']:.0f}% | with new D-form {rows[n]['d_dform']:+.3f} without {rows[n]['d_other']:+.3f} | residue-type term: D-form {rows[n]['seqterm_dform']:.3f} other {rows[n]['seqterm_other']:.3f}")
json.dump({n: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for n, r in rows.items()}, open(f"{E}/summary_real.json", "w"), indent=1, default=float)

f = plt.figure(figsize=(19, 10.2)); gs = f.add_gridspec(2, 3, height_ratios=[0.8, 2.2], hspace=0.34, wspace=0.3)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Real test set", f"{len(files)} real PDB peptide-protein complexes\n(non-canonical peptide + receptor pocket 12 A)\nnever used for training; {int(hasD.sum())} contain a new D-form", "#ffd6a5"),
       ("Models", "untuned control + trials A to F\n(best weights of each, no further training)", "#ade8f4"),
       ("Score", "flow loss of each model on each real complex,\nsame noise per complex for every model\n(paired; lower is better)", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 1), 30, 8, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15, 7.4, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15, 3.7, d, ha="center", va="center", fontsize=8.5)
for x0, x1 in [(31.5, 33.5), (64.5, 66.5)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: the same models scored on real structures", fontsize=10.5, loc="left", fontweight="bold")
a1 = f.add_subplot(gs[1, 0]); x = np.arange(len(MODELS)); vals = [rows[n]["mean"] for n, _, _ in MODELS]
a1.bar(x, vals, color=[c for _, _, c in MODELS], edgecolor="#555")
for i, v in enumerate(vals): a1.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
a1.set_ylim(min(vals) - 1.2, max(vals) + 1.0); a1.set_xticks(x); a1.set_xticklabels([l.replace("Untuned ", "Untuned\n") for _, l, _ in MODELS], fontsize=8.5); a1.set_ylabel("flow loss on real complexes (lower is better)")
a1.set_title("(a) Mean loss (axis cut)", fontsize=10.5, loc="left", fontweight="bold")
a2 = f.add_subplot(gs[1, 1]); mm = [m for m in MODELS if m[0] != "control"]; y = np.arange(len(mm))
for i, (n, l, c) in enumerate(mm):
    r = rows[n]; a2.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]], fmt="o", color=c, capsize=4, ms=9)
    a2.text(max(r["ci"][1], 0) + 0.03, i, f"better on {100 * r['win']:.0f}%", va="center", fontsize=8.5)
a2.axvline(0, color="#555", lw=1); a2.set_yticks(y); a2.set_yticklabels([l for _, l, _ in mm]); a2.invert_yaxis(); a2.set_xlabel("change vs untuned control, per complex (negative = better)")
a2.set_title("(b) Paired change with 95% interval", fontsize=10.5, loc="left", fontweight="bold")
a3 = f.add_subplot(gs[1, 2]); w = 0.8 / max(1, len(mm)); xs = np.arange(2)
for i, (n, l, c) in enumerate(mm):
    a3.bar(xs + (i - (len(mm) - 1) / 2) * w, [rows[n]["d_dform"], rows[n]["d_other"]], w, color=c, edgecolor="#555", label=l)
a3.axhline(0, color="#555", lw=1); a3.set_xticks(xs); a3.set_xticklabels([f"peptide has a new D-form\n(n={int((hasD & keep).sum())})", f"no new D-form\n(n={int((~hasD & keep).sum())})"], fontsize=9)
a3.set_ylabel("change vs control (negative = better)"); a3.legend(frameon=False, fontsize=8, ncol=2)
a3.set_title("(c) By peptide content", fontsize=10.5, loc="left", fontweight="bold")
f.text(0.01, 0.005, f"Real structures are test only. Flow loss, not generated structures; one run per model; {len(files)} complexes ({int(keep.sum())} after trimming the 5% with the most extreme noise draw for every model alike). The decoder is not in this score (the frozen encoder gives the targets), so the control is the same with any decoder.", fontsize=8.3, fontweight="bold")
f.savefig(f"{D}/figures/fig39_real_test_all_trials.png", bbox_inches="tight", dpi=130); plt.close(f)
