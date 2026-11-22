"""Comprehensive benchmark: untuned control and every fine-tuned version (A to F2) on (1) the flow loss of all positions on 163 held-out synthetic complexes and 103 real complexes,
(2) the flow loss on NON-CANONICAL positions only (residues outside the 189-class vocabulary masked) on both sets, by residue group, (3) the flow model's residue-type readout at those positions.
Every model sees the same noise on each complex (paired). Change vs control: negative = better."""
import os, json, csv, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; S = f"{D}/denoiser_trial/saved"
MODELS = [("control", "Untuned control", "#8d99ae"), ("A_all", "A", "#2a6f97"), ("B_first", "B", "#c9a227"), ("C_long", "C", "#b23a48"), ("D_rank64", "D", "#3a7d44"), ("E_rank64_ff", "E", "#7b5ea7"),
          ("F_pool2", "F box", "#e07a1f"), ("F_H200", "F H200", "#17a2b8"), ("F2_H200", "F2", "#0b6e4f"), ("G_ncw10", "G", "#d62768")]
DF13 = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split()); GEN12 = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
rng = np.random.default_rng(0)
def boot(d, n=2000):
    d = np.asarray(d); m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return np.percentile(m, [2.5, 97.5])
def loadj(p): return json.load(open(p))["results"] if os.path.exists(p) else None

# ---- (1) flow loss of all positions
def flow(setdir):
    R = {n: loadj(f"{S}/{setdir}/{n}.json") for n, _, _ in MODELS}; R = {n: r for n, r in R.items() if r}
    fs = sorted(set.intersection(*[{f for f, v in r.items() if "total" in v} for r in R.values()]))
    a = {n: np.array([R[n][f]["total"] for f in fs]) for n in R}; keep = a["control"] <= np.percentile(a["control"], 95)
    out = {}
    for n in R:
        d = (a[n] - a["control"])[keep]; out[n] = (d.mean(), boot(d)) if n != "control" else (0.0, np.zeros(2))
    return out, fs, keep
held, hfs, hkeep = flow("eval"); real, rfs, rkeep = flow("eval_real")
seq = {r["item_id"]: r["ccd_sequence"].split("-") for r in csv.DictReader(open("C:/Research/Dataset_ncaa/real_benchmark/complex_test/complex_items.tsv"), delimiter="\t")}
hasD = np.array([bool(DF13 & set(seq[f.replace("_real.cif", "")])) for f in rfs])
Rr = {n: loadj(f"{S}/eval_real/{n}.json") for n, _, _ in MODELS}; Rr = {n: r for n, r in Rr.items() if r}
ar = {n: np.array([Rr[n][f]["total"] for f in rfs]) for n in Rr}
real_d = {n: ((ar[n] - ar["control"])[hasD].mean(), boot((ar[n] - ar["control"])[hasD])) for n in Rr if n != "control"}
real_o = {n: ((ar[n] - ar["control"])[~hasD].mean(), boot((ar[n] - ar["control"])[~hasD])) for n in Rr if n != "control"}
real_all_untrim = {n: ((ar[n] - ar["control"]).mean(), boot(ar[n] - ar["control"])) for n in Rr if n != "control"}

# ---- (2) non-canonical positions only
def nc(setname):
    R = {n: loadj(f"{S}/eval_nc/{setname}_{n}.json") for n, _, _ in MODELS}; R = {n: r for n, r in R.items() if r}
    fs = sorted(set.intersection(*[{f for f, v in r.items() if "codes" in v} for r in R.values()])); return R, fs
def grp_vals(R, fs, grp, key):
    """per complex token-weighted mean of `key` over the codes in group grp (None = every non-canonical code); NaN if none"""
    out = {}
    for n, r in R.items():
        v = []
        for f in fs:
            cs = {c: e for c, e in r[f]["codes"].items() if not c.startswith("_") and (grp is None or c in grp)}
            tn = sum(e["n"] for e in cs.values()); v.append(sum(e["n"] * e[key] for e in cs.values()) / tn if tn else np.nan)
        out[n] = np.array(v)
    return out
def rel(vals, n):
    ok = ~np.isnan(vals["control"]); d = (vals[n] - vals["control"])[ok]; c = vals["control"][ok]
    pct = 100 * d.mean() / c.mean(); idx = np.arange(len(d)); bs = [100 * d[i].mean() / c[i].mean() for i in (rng.choice(idx, len(idx)) for _ in range(1500))]
    return pct, np.percentile(bs, [2.5, 97.5]), int(ok.sum())
NC = {}
for setname in ("real", "heldout"):
    R, fs = nc(setname)
    if not R: continue
    NC[setname] = dict(R=R, fs=fs, groups={}, ntok={})
    for gname, g in (("all non-canonical", None), ("13 D-forms", DF13), ("12 general codes", GEN12), ("other vocab codes", "OTHER")):
        gg = g
        if g == "OTHER":
            gg = set(c for r in R.values() for f in fs for c in r[f]["codes"] if not c.startswith("_")) - DF13 - GEN12
        vals = {k: grp_vals(R, fs, gg, k) for k in ("local_latents_raw", "bb_ca_raw", "ce", "acc")}
        ok = ~np.isnan(vals["local_latents_raw"]["control"])
        ntok = int(sum(e["n"] for f in np.array(fs)[ok] for c, e in R["control"][f]["codes"].items() if not c.startswith("_") and (gg is None or c in gg)))
        NC[setname]["groups"][gname] = {n: dict(lat=rel(vals["local_latents_raw"], n), ca=rel(vals["bb_ca_raw"], n), ce=float(np.nanmean(vals["ce"][n])), acc=float(np.nanmean(vals["acc"][n]))) for n in R if n != "control"}
        NC[setname]["groups"][gname]["_control"] = dict(ce=float(np.nanmean(vals["ce"]["control"])), acc=float(np.nanmean(vals["acc"]["control"])), n_complex=int(ok.sum()), n_tok=ntok)

# ---- report
lab = {n: l for n, l, _ in MODELS}
print("HELD-OUT flow (163, trimmed 154):  ", {lab[n]: f"{v[0]:+.3f} [{v[1][0]:+.2f},{v[1][1]:+.2f}]" for n, v in held.items() if n != "control"})
print("REAL flow trimmed (97):            ", {lab[n]: f"{v[0]:+.3f} [{v[1][0]:+.2f},{v[1][1]:+.2f}]" for n, v in real.items() if n != "control"})
print("REAL flow untrimmed (103):         ", {lab[n]: f"{v[0]:+.3f} [{v[1][0]:+.2f},{v[1][1]:+.2f}]" for n, v in real_all_untrim.items()})
print(f"REAL with new D-form (n={int(hasD.sum())}):", {lab[n]: f"{v[0]:+.3f} [{v[1][0]:+.2f},{v[1][1]:+.2f}]" for n, v in real_d.items()})
print(f"REAL without (n={int((~hasD).sum())}):     ", {lab[n]: f"{v[0]:+.3f} [{v[1][0]:+.2f},{v[1][1]:+.2f}]" for n, v in real_o.items()})
for setname, dd in NC.items():
    for g, gv in dd["groups"].items():
        c = gv["_control"]; print(f"NC {setname:7s} | {g:18s} | complexes {c['n_complex']:3d} tokens {c['n_tok']:5d} | control readout CE {c['ce']:.2f} acc {c['acc']:.3f}")
        print("   latent % change:", {lab[n]: f"{v['lat'][0]:+.2f}% [{v['lat'][1][0]:+.1f},{v['lat'][1][1]:+.1f}]" for n, v in gv.items() if n != "_control"})
        print("   CA     % change:", {lab[n]: f"{v['ca'][0]:+.1f}% [{v['ca'][1][0]:+.0f},{v['ca'][1][1]:+.0f}]" for n, v in gv.items() if n != "_control"})
        print("   readout CE/acc :", {lab[n]: f"{v['ce']:.2f}/{v['acc']:.3f}" for n, v in gv.items() if n != "_control"})

# ---- figure
names = [n for n, _, _ in MODELS if n != "control" and n in held]; cols = {n: c for n, _, c in MODELS}
f = plt.figure(figsize=(20, 11.6)); gs = f.add_gridspec(2, 3, height_ratios=[0.5, 2.5], hspace=0.38, wspace=0.34)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Models", "untuned control + 8 fine-tuned versions (A to F2)\nrank-16 LoRA on attention (A, B, C, F, F2), rank 64 (D), + feed-forward (E)", "#ade8f4"),
       ("Test sets (never trained on)", f"{len(hfs)} held-out synthetic complexes (protein families never trained on)\n{len(rfs)} real PDB complexes ({int(hasD.sum())} with a new D-form)", "#ffd6a5"),
       ("Scores", "flow loss, all positions | non-canonical positions only (residues\noutside the 189 vocabulary masked) | residue-type readout", "#dde5b6")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 1), 31, 8, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15.5, 7.4, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15.5, 3.7, d, ha="center", va="center", fontsize=8.3)
for x0, x1 in [(32.3, 34.0), (65.3, 67.0)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: same noise on every complex for every model (paired). Negative = better than the untuned control", fontsize=10.5, loc="left", fontweight="bold")
def forest(ax, vals, title, xlabel, pct=False):
    y = np.arange(len(names))
    for i, n in enumerate(names):
        if n not in vals: continue
        v = vals[n]; m, ci = (v[0], v[1]); ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=cols[n], capsize=3, ms=7)
    ax.axvline(0, color="#555", lw=1); ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in names]); ax.invert_yaxis(); ax.set_xlabel(xlabel, fontsize=9); ax.set_title(title, fontsize=10.2, loc="left", fontweight="bold")
forest(f.add_subplot(gs[1, 0]), held, "(a) All positions, 163 held-out synthetic complexes", "change in flow loss vs control")
ax = f.add_subplot(gs[1, 1]); y = np.arange(len(names)); h = 0.27
for j, (vals, lb, mk) in enumerate([(real, "all 103 (trimmed 97)", "o"), (real_d, f"with new D-form (n={int(hasD.sum())})", "s"), (real_o, f"without (n={int((~hasD).sum())})", "^")]):
    for i, n in enumerate(names):
        if n in vals: m, ci = vals[n]; ax.errorbar(m, i + (j - 1) * h, xerr=[[m - ci[0]], [ci[1] - m]], fmt=mk, color=cols[n], capsize=2, ms=6, alpha=[1, 0.8, 0.5][j], label=lb if i == 0 else None)
ax.axvline(0, color="#555", lw=1); ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in names]); ax.invert_yaxis(); ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3); ax.set_xlabel("change in flow loss vs control", fontsize=9)
ax.set_title("(b) All positions, 103 real complexes", fontsize=10.2, loc="left", fontweight="bold")
for k, setname in enumerate(("real", "heldout")):
    if setname not in NC: continue
    a = f.add_subplot(gs[1, 2]) if k == 0 else a
    gv = NC[setname]["groups"]["all non-canonical"]; vals = {n: gv[n]["lat"] for n in names if n in gv}
    y = np.arange(len(names))
    for i, n in enumerate(names):
        if n in vals: m, ci, _ = vals[n]; a.errorbar(m, i + (k - 0.5) * 0.3, xerr=[[m - ci[0]], [ci[1] - m]], fmt=["o", "D"][k], color=cols[n], capsize=2, ms=6, alpha=[1, 0.55][k], label=["real complexes", "held-out synthetic"][k] if i == 0 else None)
a.axvline(0, color="#555", lw=1); a.set_yticks(y); a.set_yticklabels([lab[n] for n in names]); a.invert_yaxis(); a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2); a.set_xlabel("latent loss change vs control (% of control)", fontsize=9)
a.set_title("(c) Non-canonical positions only", fontsize=10.2, loc="left", fontweight="bold")
f.text(0.01, 0.005, "Flow loss, not generated structures; one run per model. Non-canonical positions = the 142 non-canonical residue types of the 189-class vocabulary; residues outside the vocabulary are masked. Latent loss = the 32-number residue code, weight 1/((1-t)^2+1e-5) divided out.", fontsize=8.3, fontweight="bold")
f.savefig(f"{D}/figures/fig41_benchmark_all_models.png", bbox_inches="tight", dpi=130); plt.close(f)

# ---- second figure: by residue group, real and held-out
f2, axs = plt.subplots(2, 3, figsize=(20, 9.6), gridspec_kw=dict(hspace=0.42, wspace=0.32))
for r_i, setname in enumerate(("real", "heldout")):
    if setname not in NC: continue
    for c_i, g in enumerate(("13 D-forms", "12 general codes", "other vocab codes")):
        ax = axs[r_i, c_i]; gv = NC[setname]["groups"][g]; c = gv["_control"]
        vals = [(n, gv[n]["lat"]) for n in names if n in gv]
        for i, (n, v) in enumerate(vals): m, ci, _ = v; ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=cols[n], capsize=3, ms=7)
        ax.axvline(0, color="#555", lw=1); ax.set_yticks(range(len(vals))); ax.set_yticklabels([lab[n] for n, _ in vals]); ax.invert_yaxis()
        ax.set_title(f"{'real' if setname == 'real' else 'held-out synthetic'}: {g}\n({c['n_complex']} complexes, {c['n_tok']} positions)", fontsize=10, loc="left", fontweight="bold"); ax.set_xlabel("latent loss change vs control (% of control)", fontsize=8.5)
f2.suptitle("Non-canonical positions only, by residue group (negative = better than the untuned control)", fontsize=12, fontweight="bold", y=0.995)
f2.savefig(f"{D}/figures/fig42_noncanonical_by_group.png", bbox_inches="tight", dpi=130); plt.close(f2)
json.dump({"held": {n: [float(v[0]), [float(x) for x in v[1]]] for n, v in held.items()}, "real_trim": {n: [float(v[0]), [float(x) for x in v[1]]] for n, v in real.items()},
           "nc": {s: {g: {n: ({k: (list(map(float, v[k][:1])) + [list(map(float, v[k][1]))]) for k in ("lat", "ca")} | dict(ce=v["ce"], acc=v["acc"])) if n != "_control" else v for n, v in gv.items()} for g, gv in d["groups"].items()} for s, d in NC.items()}}, open(f"{S}/eval_nc/benchmark_summary.json", "w"), indent=1)

# ---- per residue code table (latent loss, % change vs control; codes with at least 6 positions in the set)
import collections
with open(f"{S}/eval_nc/per_code_latent_change.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["set", "code", "group", "positions", "complexes", "control_latent_raw"] + [lab[n] for n in names])
    for setname, dd in NC.items():
        R, fs = dd["R"], dd["fs"]; codes = collections.Counter()
        for f in fs:
            for c, e in R["control"][f]["codes"].items():
                if not c.startswith("_"): codes[c] += e["n"]
        for c, ntok in codes.most_common():
            if ntok < 6: continue
            row = [setname, c, "D-form" if c in DF13 else ("general" if c in GEN12 else "other"), ntok, sum(1 for f in fs if c in R["control"][f]["codes"])]
            fl = [f for f in fs if c in R["control"][f]["codes"]]; ctrl = np.array([R["control"][f]["codes"][c]["local_latents_raw"] for f in fl]); wt = np.array([R["control"][f]["codes"][c]["n"] for f in fl])
            cm = np.average(ctrl, weights=wt); row.append(round(float(cm), 3))
            for n in names:
                if n not in R: row.append(""); continue
                v = np.array([R[n][f]["codes"][c]["local_latents_raw"] for f in fl]); row.append(round(100 * (np.average(v, weights=wt) - cm) / cm, 1))
            w.writerow(row)
print("wrote per-code table")
