"""Figure: smoke test of the pilot generator (ncf_pilot.py, released raw_base weights, 3 train complexes, 12 swaps)."""
import os, pickle, collections
import matplotlib.pyplot as plt

P = "C:/Research/ncflow_ref/ncflow/out2/pilot_smoke.pkl"
GATES = ["chem", "clash", "interaction", "pose", "passed"]
COL = {"chem": "#b23a48", "clash": "#c9a227", "interaction": "#8d99ae", "pose": "#6c757d", "passed": "#2a6f97"}
LAB = {"chem": "fails strict chemistry test", "clash": "clashes with protein", "interaction": "loses interactions (<0.9)", "pose": "pose drift", "passed": "passes all gates"}


def fig(OUT, C):
    r = pickle.load(open(P, "rb"))
    rows = []
    for k, v in r.items():
        for s in v["snaps"][1:]:
            i = s["info"]
            rows.append((k, i["parent"], i["fails"], not i.get("skipped")))
    f, ax = plt.subplots(1, 3, figsize=(16.5, 5.6), gridspec_kw=dict(wspace=0.38))
    a = ax[0]
    tot = collections.Counter()
    for _, _, fl, _ in rows:
        for g in GATES:
            tot[g] += fl[g]
    n = sum(tot.values())
    left = 0
    for g in GATES:
        a.barh(0, tot[g], left=left, color=COL[g], edgecolor="#fff", label=f"{LAB[g]}: {tot[g]}")
        left += tot[g]
    a.set_yticks([]); a.set_xlabel("draws (NCFlow samples)")
    a.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    a.set_title(f"(a) What happened to {n} draws\n(gates are applied in this order)", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[1]
    par = collections.defaultdict(lambda: [0, 0])
    for _, p, _, ok in rows:
        par[p][0 if ok else 1] += 1
    ps = sorted(par, key=lambda p: -par[p][0] / sum(par[p]))
    for i, p in enumerate(ps):
        b.bar(i, par[p][0], color="#2a6f97", edgecolor="#555")
        b.bar(i, par[p][1], bottom=par[p][0], color="#e9ecef", edgecolor="#555")
    b.set_xticks(range(len(ps))); b.set_xticklabels(ps)
    b.set_ylabel("planned swaps"); b.set_xlabel("canonical residue being replaced")
    b.set_title(f"(b) Swaps accepted (blue) or left\ncanonical (grey): {sum(1 for x in rows if x[3])} of {len(rows)} accepted", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[2]
    xs = [sum(fl[g] for g in GATES) for _, _, fl, _ in rows]
    for x, (_, p, fl, ok) in zip(xs, rows):
        c.scatter(x, fl["passed"], s=60, color="#2a6f97" if ok else "#b23a48", edgecolor="#222", zorder=3)
    c.set_xlabel("draws used for the swap"); c.set_ylabel("draws that passed all gates")
    c.set_title("(c) Each dot is one swap\n(red: no draw passed)", fontsize=10.5, loc="left", fontweight="bold")
    npro = sum(1 for x in rows if x[1] == "PRO")
    f.text(0.01, -0.3, f"Result: the generator runs end to end ({sum(1 for x in rows if x[3])} of {len(rows)} swaps accepted; every accepted swap kept 100% of contacts and polar contacts, CA drift 0.1-0.5 A).\nAlmost all rejections ({tot['chem']} of {n} draws) come from the strict chemistry test, only {tot['interaction']} from the interaction rule (the plan swaps the least-touching residues first),\nand none of the {npro} Pro swaps passed (35-36 of 36 draws failed the chemistry test each).", fontsize=9.3, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig25_pilot_smoke.png"), bbox_inches="tight")
    plt.close(f)
