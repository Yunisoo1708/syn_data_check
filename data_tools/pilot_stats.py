"""Summary of pilot generator shards (ncf_pilot.py output pickles). usage: pilot_stats.py OUT.json shard1.pkl [shard2.pkl ...]"""
import sys, pickle, json, collections
import numpy as np
out, files = sys.argv[1], sys.argv[2:]
rows = []; cx = 0
for f in files:
    r = pickle.load(open(f, "rb"))
    for k, v in r.items():
        cx += 1
        for s in v["snaps"][1:]:
            i = s["info"]
            rows.append(dict(cx=k, parent=i["parent"], code=i.get("code", i.get("new")), skipped=bool(i.get("skipped")), fails=i["fails"], keys=sorted(i.keys())))
print("complexes", cx, "swaps", len(rows), "accepted", sum(not r["skipped"] for r in rows))
print("info keys:", rows[0]["keys"])
fails = collections.Counter()
for r in rows:
    for g, n in r["fails"].items(): fails[g] += n
print("draw outcomes", dict(fails))
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    by[r["parent"]][0] += 1; by[r["parent"]][1] += (not r["skipped"])
print({k: f"{v[1]}/{v[0]}" for k, v in sorted(by.items())})
nc = collections.Counter(r["code"] for r in rows if not r["skipped"])
print(len(nc), "codes", nc.most_common(12))
json.dump(dict(complexes=cx, swaps=len(rows), accepted=sum(not r["skipped"] for r in rows), draws=dict(fails), by_parent={k: v for k, v in by.items()}, codes=dict(nc)), open(out, "w"))
