"""Extra real benchmark (884 Hamdani non-natural complexes of split v4 test, never trained on) + the first 103 real complexes: non-canonical positions only, by residue group and code, untuned control vs every fine-tuned version."""
import os, json, csv, collections, numpy as np, matplotlib.pyplot as plt
D = "C:/Research/RFD4_fine_tune"; S = f"{D}/denoiser_trial/saved/eval_nc"
MODELS = [("control", "Untuned control", "#8d99ae"), ("A_all", "A", "#2a6f97"), ("B_first", "B", "#c9a227"), ("C_long", "C", "#b23a48"), ("D_rank64", "D", "#3a7d44"), ("E_rank64_ff", "E", "#7b5ea7"),
          ("F_pool2", "F box", "#e07a1f"), ("F_H200", "F H200", "#17a2b8"), ("F2_H200", "F2", "#0b6e4f"), ("G_ncw10", "G", "#d62768"), ("G2_pool", "G2", "#6a4c93")]
DF13 = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split()); GEN12 = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
rng = np.random.default_rng(0); names = [n for n, _, _ in MODELS if n != "control"]; lab = {n: l for n, l, _ in MODELS}; col = {n: c for n, _, c in MODELS}
def load(st): return {n: json.load(open(f"{S}/{st}_{n}.json"))["results"] for n, _, _ in MODELS if os.path.exists(f"{S}/{st}_{n}.json")}
sets = {"ham": load("ham"), "real103": load("real")}; sets["all987"] = None
def merged():
    out = {}
    for n in sets["ham"]:
        if n in sets["real103"]:
            r = {}; r.update({"h_" + k: v for k, v in sets["ham"][n].items()}); r.update({"r_" + k: v for k, v in sets["real103"][n].items()}); out[n] = r
    return out
sets["all987"] = merged()
def groupvals(R, fs, grp, key):
    out = {}
    for n, r in R.items():
        v = []
        for f in fs:
            cs = {c: e for c, e in r[f]["codes"].items() if not c.startswith("_") and (grp is None or c in grp)}
            tn = sum(e["n"] for e in cs.values()); v.append(sum(e["n"] * e[key] for e in cs.values()) / tn if tn else np.nan)
        out[n] = np.array(v)
    return out
def rel(vals, n):
    ok = ~np.isnan(vals["control"]); d = (vals[n] - vals["control"])[ok]; c = vals["control"][ok]; idx = np.arange(len(d))
    pct = 100 * d.mean() / c.mean(); bs = [100 * d[i].mean() / c[i].mean() for i in (rng.choice(idx, len(idx)) for _ in range(1500))]
    return pct, np.percentile(bs, [2.5, 97.5]), int(ok.sum())
res = {}
for st in ("ham", "real103", "all987"):
    R = sets[st]; fs = sorted(set.intersection(*[{f for f, v in r.items() if "codes" in v} for r in R.values()]))
    other = set(c for r in R.values() for f in fs for c in r[f]["codes"] if not c.startswith("_")) - DF13 - GEN12
    res[st] = {}
    for g, gg in (("all non-canonical", None), ("13 D-forms", DF13), ("12 general codes", GEN12), ("other vocab codes", other), ("canonical (reference)", "CAN")):
        if gg == "CAN":
            gv = {k: groupvals(R, fs, {"_CANONICAL"}, k) for k in ("local_latents_raw",)}
            # canonical is stored under the key _CANONICAL: handle separately
            vals = {n: np.array([r[f]["codes"].get("_CANONICAL", {}).get("local_latents_raw", np.nan) for f in fs]) for n, r in R.items()}
        else:
            vals = groupvals(R, fs, gg, "local_latents_raw")
        ok = ~np.isnan(vals["control"]); ntok = int(sum(e["n"] for f in np.array(fs)[ok] for c, e in R["control"][f]["codes"].items() if (c == "_CANONICAL" if gg == "CAN" else (not c.startswith("_") and (gg is None or c in gg)))))
        res[st][g] = dict(n_complex=int(ok.sum()), n_tok=ntok, change={n: rel(vals, n) for n in names if n in R})
print("complexes scored: ham", len(sets["ham"]["control"]), "real103", len(sets["real103"]["control"]))
for st in res:
    for g, v in res[st].items():
        print(f"{st:8s} | {g:22s} | {v['n_complex']:4d} complexes {v['n_tok']:5d} positions | " + " ".join(f"{lab[n]} {c[0]:+.1f}%[{c[1][0]:+.0f},{c[1][1]:+.0f}]" for n, c in v["change"].items()))
# per code table on the 987 real complexes (latent loss, % change vs control)
R = sets["all987"]; fs = sorted(set.intersection(*[{f for f, v in r.items() if "codes" in v} for r in R.values()])); codes = collections.Counter()
for f in fs:
    for c, e in R["control"][f]["codes"].items():
        if not c.startswith("_"): codes[c] += e["n"]
with open(f"{S}/per_code_real987.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["code", "group", "positions", "complexes"] + [lab[n] for n in names])
    for c, nt in codes.most_common():
        if nt < 8: continue
        fl = [f for f in fs if c in R["control"][f]["codes"]]; wt = np.array([R["control"][f]["codes"][c]["n"] for f in fl]); ctrl = np.array([R["control"][f]["codes"][c]["local_latents_raw"] for f in fl]); cm = np.average(ctrl, weights=wt)
        row = [c, "D-form" if c in DF13 else ("general" if c in GEN12 else "other"), nt, len(fl)]
        for n in names: row.append(round(100 * (np.average([R[n][f]["codes"][c]["local_latents_raw"] for f in fl], weights=wt) - cm) / cm, 1) if n in R else "")
        w.writerow(row)
# figure
f, axs = plt.subplots(2, 2, figsize=(17, 10.5), gridspec_kw=dict(hspace=0.42, wspace=0.28))
for ax, (st, g, ttl) in zip(axs.ravel(), [("ham", "all non-canonical", "884 extra real complexes: all non-canonical"), ("all987", "12 general codes", "all 987 real complexes: 12 general codes"), ("all987", "13 D-forms", "all 987 real complexes: 13 D-forms"), ("all987", "canonical (reference)", "all 987 real complexes: canonical positions")]):
    v = res[st][g]; y = np.arange(len(names))
    for i, n in enumerate(names):
        m, ci, _ = v["change"][n]; ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=col[n], capsize=3, ms=7)
    ax.axvline(0, color="#555", lw=1); ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in names]); ax.invert_yaxis(); ax.set_xlabel("latent loss change vs control (% of control)", fontsize=9)
    ax.set_title(f"{ttl}\n({v['n_complex']} complexes, {v['n_tok']} positions)", fontsize=10.2, loc="left", fontweight="bold")
f.suptitle("Larger real benchmark (test only): untuned control vs every fine-tuned version, latent loss at the scored positions (negative = better)", fontsize=12, fontweight="bold", y=0.995)
f.savefig(f"{D}/figures/fig50_real_benchmark_extended_G2.png", bbox_inches="tight", dpi=130); plt.close(f)
