"""Best-of-N filter at generation time: per pocket draw N of the 8 designs (without replacement, Monte Carlo over the draws) and keep one design WITHOUT an atom pair closer than 2 A to the receptor (rule A) or the clash-free one with the most sensible peptide bonds (rule B); if none is clash-free the pocket counts as a failure and a random draw is kept.
Reports, per run: pockets with at least one clash-free design among the N, the clash rate of the kept designs, and the quality of the kept designs (bonds, D handedness, CA distance, contacts) against the unfiltered mean.
usage: best_of_n_filter.py RUN_KEY... (keys of gen_test/gen_summary.json, e.g. G3_bal_v10); writes gen_test/diag/best_of_n.json"""
import sys, json, collections, numpy as np
D = "C:/Research/RFD4_fine_tune"; S = json.load(open(f"{D}/gen_test/gen_summary.json")); rng = np.random.default_rng(0); out = {}
KEYS = ("bond_ok", "chir_D", "ca_rmsd", "contacts_rel", "free_rec")
def mean(v): v = [x for x in v if x == x]; return float(np.mean(v)) if v else float("nan")
for key in sys.argv[1:]:
    rows = S[key]["rows"]; by = collections.defaultdict(list)
    for r in rows: by[r["name"]].append(r)
    base = {k: mean([r[k] for r in rows]) for k in KEYS}; base["clash"] = mean([r["clash"] for r in rows]); out[key] = dict(n_designs=len(rows), pockets=len(by), unfiltered=base, N={})
    for N in (1, 2, 4, 8):
        for rule in ("A", "B"):
            succ = []; kept = {k: [] for k in KEYS}; ca = []; ct = []
            for _ in range(4000 if N < 8 else 1):
                for nm, rs in by.items():
                    idx = rng.choice(len(rs), size=min(N, len(rs)), replace=False); dr = [rs[i] for i in idx]; free = [r for r in dr if r["clash"] == 0]
                    if free:
                        pick = free[rng.integers(len(free))] if rule == "A" else max(free, key=lambda r: (r["bond_ok"] if r["bond_ok"] == r["bond_ok"] else -1))
                        succ.append(1)
                    else: pick = dr[rng.integers(len(dr))]; succ.append(0)
                    for k in KEYS: kept[k].append(pick[k])
                    ca.append(pick["ca_rmsd"]); ct.append(pick["contacts_rel"])
            out[key]["N"][f"{N}{rule}"] = dict(clash_free_share=mean(succ), bond_ok=mean(kept["bond_ok"]), chir_D=mean(kept["chir_D"]), ca_rmsd_med=float(np.median(ca)), contacts_rel_med=float(np.median(ct)), free_rec=mean(kept["free_rec"]))
    print(key, "unfiltered clash %.3f bonds %.3f D %.3f CA(med) %.2f" % (base["clash"], base["bond_ok"], base["chir_D"], float(np.median([r["ca_rmsd"] for r in rows]))))
    for k, v in out[key]["N"].items(): print("  best-of-%s: clash-free share %.3f | kept bonds %.3f D %.3f CA(med) %.2f contacts(med) %.2f free_rec %.3f" % (k, v["clash_free_share"], v["bond_ok"], v["chir_D"], v["ca_rmsd_med"], v["contacts_rel_med"], v["free_rec"]))
json.dump(out, open(f"{D}/gen_test/diag/best_of_n.json", "w"), indent=1)
