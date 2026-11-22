"""Extract per-swap metrics from pilot shards to JSON (WSL). usage: pilot_metrics.py OUT.json shard.pkl ..."""
import sys, pickle, json
out, files = sys.argv[1], sys.argv[2:]
rows = []
for f in files:
    for k, v in pickle.load(open(f, "rb")).items():
        n = 0
        for s in v["snaps"][1:]:
            i = s["info"]
            m = i.get("metrics") or {}
            rows.append(dict(cx=k, parent=i["parent"], code=i.get("code"), family=i.get("family"), ok=not i.get("skipped"), fails=i["fails"], attempts=i.get("attempts"), mirrored=bool(i.get("mirrored")),
                             ret_c=m.get("ret_c"), ret_p=m.get("ret_p"), atoms=m.get("atoms"), drift=m.get("drift"), drift_max=m.get("drift_max"), plddt=i.get("plddt")))
json.dump(rows, open(out, "w"))
print(len(rows))
