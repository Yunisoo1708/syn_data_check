"""ipTM filter analysis (Results.MD 83). usage: python analyze_af3.py <collected.json> <out_summary.json>
1 separation: ipTM by group (native, designs, shuffled, random-NCAA, ALA, wrong receptor), AUC native vs each negative; designs vs random-NCAA/shuffled (within pocket).
2 agreement: within-pocket Spearman of design ipTM against Rosetta E_int (whole peptide), E_slot of the chosen NCAAs, receptor contacts, AF3-vs-design pose RMSD.
3 filter: keep the top 25% / 50% of the designs of each pocket by ipTM (all 3 denoisers pooled); compare Rosetta E_int / E_slot / contacts of the kept set with all designs and with 2000 random keeps of the same size."""
import json, sys, numpy as np, collections
def _rank(x):
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in np.unique(x):
        m = np.asarray(x) == v
        if m.sum() > 1: o[m] = o[m].mean()
    return o
def spearmanr(a, b, n_perm=5000):
    ra, rb = _rank(np.array(a)), _rank(np.array(b)); rho = float(np.corrcoef(ra, rb)[0, 1]); g = np.random.default_rng(1)
    perm = [abs(np.corrcoef(ra, g.permutation(rb))[0, 1]) for _ in range(n_perm)]; return rho, float((1 + sum(x >= abs(rho) for x in perm)) / (n_perm + 1))
R = json.load(open(sys.argv[1])); G = r"C:\Research\RFD4_fine_tune"
DES = ("control", "G3", "S1")
def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    if not len(pos) or not len(neg): return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
by = collections.defaultdict(list)
for r in R: by[r["group"]].append(r)
out = {"n": {g: len(v) for g, v in by.items()}, "iptm_mean": {g: float(np.mean([r["iptm"] for r in v])) for g, v in by.items()},
       "iptm_median": {g: float(np.median([r["iptm"] for r in v])) for g, v in by.items()}, "pep_plddt": {g: float(np.mean([r["pep_plddt"] for r in v])) for g, v in by.items()}}
nat = {r["ex"]: r["iptm"] for r in by["native"]}
out["auc_native_vs"] = {g: auc([r["iptm"] for r in by["native"]], [r["iptm"] for r in by[g]]) for g in ("decoy", "shuf", "rnd", "ala", *DES) if g in by}
# within pocket: designs against the random-NCAA / shuffled of the same pocket (paired difference)
def pocket(g): return collections.defaultdict(list, {}) if g not in by else None
def mean_by_ex(g):
    d = collections.defaultdict(list)
    for r in by.get(g, []): d[r["ex"]].append(r["iptm"])
    return {k: float(np.mean(v)) for k, v in d.items()}
M = {g: mean_by_ex(g) for g in ("native", "rnd", "shuf", "ala", "decoy", *DES)}
out["per_pocket_mean_iptm"] = M
out["design_minus_rnd"] = {g: float(np.mean([M[g][k] - M["rnd"][k] for k in M[g] if k in M["rnd"]])) for g in DES}
out["design_minus_shuf"] = {g: float(np.mean([M[g][k] - M["shuf"][k] for k in M[g] if k in M["shuf"]])) for g in DES}
out["design_minus_native"] = {g: float(np.mean([M[g][k] - M["native"][k] for k in M[g] if k in M["native"]])) for g in DES}
# join with Rosetta + contacts
RS = {}
for x in json.load(open(rf"{G}\rosetta_score\real_design_scores.json")):
    i = x["info"]
    if "error" in i or i["group"] not in DES: continue
    sl = [np.mean([r["own"][0][1], r["own"][1][1]]) for r in x["rows"] if r["kind"] == "ncaa"]
    RS[(i["group"], i["ex"])] = dict(e_int=i.get("e_int"), e_slot=float(np.mean(sl)) if sl else None)
CS = json.load(open(rf"{G}\gen_test\diag\choice_scores.json")); CT = {}
for g in DES:
    d = collections.defaultdict(list)
    for r in CS[g]: d[r["ex"]].append(r["contacts"])
    for k, v in d.items(): CT[(g, k)] = float(np.mean(v))
D = []
for r in R:
    if r["group"] not in DES: continue
    k = (r["group"], f"{r['ex']}_{r['rep']}"); s = RS.get(k, {})
    D.append(dict(g=r["group"], ex=r["ex"], iptm=r["iptm"], pep_plddt=r["pep_plddt"], rmsd=r.get("pep_rmsd"), e_int=s.get("e_int"), e_slot=s.get("e_slot"), contacts=CT.get(k)))
def zp(vals, key):  # within-pocket rank (0..1)
    out_ = {}
    for ex in set(d["ex"] for d in D):
        v = [d for d in D if d["ex"] == ex and d[key] is not None]
        for j, d in enumerate(sorted(v, key=lambda d: d[key])): out_[id(d)] = j / max(1, len(v) - 1)
    return out_
rk = zp(D, "iptm"); out["n_designs_joined"] = len(D)
out["spearman_within_pocket"] = {}
for key in ("e_int", "e_slot", "contacts", "rmsd", "pep_plddt"):
    xs = [(rk[id(d)], d[key]) for d in D if d[key] is not None]
    if len(xs) > 10:
        # demean target within pocket so pockets do not dominate
        ys = []; xx = []
        for ex in set(d["ex"] for d in D):
            v = [d for d in D if d["ex"] == ex and d[key] is not None]
            if len(v) < 4: continue
            m = np.mean([d[key] for d in v]); ys += [d[key] - m for d in v]; xx += [rk[id(d)] for d in v]
        rho, p = spearmanr(xx, ys); out["spearman_within_pocket"][key] = dict(rho=float(rho), p=float(p), n=len(xx))
rng = np.random.default_rng(0); out["filter"] = {}
for frac in (0.25, 0.5):
    kept = [d for d in D if rk[id(d)] >= 1 - frac]
    row = {}
    for key in ("e_int", "e_slot", "contacts", "rmsd", "pep_plddt"):
        allv = [d[key] for d in D if d[key] is not None]; kv = [d[key] for d in kept if d[key] is not None]
        if not kv: continue
        # random keeps of the same size within each pocket
        sims = []
        for _ in range(2000):
            pick = []
            for ex in set(d["ex"] for d in D):
                v = [d for d in D if d["ex"] == ex]; n = int(round(len(v) * frac)); idx = rng.choice(len(v), n, replace=False); pick += [v[i][key] for i in idx if v[i][key] is not None]
            sims.append(np.mean(pick))
        row[key] = dict(all=float(np.mean(allv)), kept=float(np.mean(kv)), rand_lo=float(np.percentile(sims, 2.5)), rand_hi=float(np.percentile(sims, 97.5)), n_kept=len(kept))
    out["filter"][str(frac)] = row
    out["filter"][str(frac)]["by_denoiser"] = {g: float(np.mean([d["g"] == g for d in kept])) for g in DES}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float); json.dump(D, open(sys.argv[2].replace(".json", "_designs.json"), "w"), default=float)
print(json.dumps({k: out[k] for k in ("n", "iptm_mean", "auc_native_vs", "design_minus_rnd", "design_minus_shuf", "design_minus_native", "spearman_within_pocket", "filter")}, indent=1, default=float))
