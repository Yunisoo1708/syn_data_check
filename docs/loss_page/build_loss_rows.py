"""Turn a decoder training log (decoder_fine_tune_v3/v4.py output) into loss rows, one per epoch.
usage: build_loss_rows.py LOG RUN_NAME OUT.json   (rows: run, epoch, train_pool, train_complex, val_pool, val_complex, val_complex_acc, new_class_acc, seconds, best)"""
import sys, re, json, collections
log, run, out = sys.argv[1:4]
steps = collections.defaultdict(lambda: [[], []])
rows = {}
best = None; saved = set()
last_ep = None
for l in open(log, errors="ignore"):
    m = re.match(r"ep (\d+) step (\d+) pool ([\d.]+) complex ([\d.]+) (\d+)s", l)
    if m:
        e = int(m.group(1)); steps[e][0].append(float(m.group(3))); steps[e][1].append(float(m.group(4)))
    m = re.match(r"epoch (\d+): val pool loss ([\d.]+) acc ([\d.]+) \| val complexes loss ([\d.]+) acc ([\d.]+) \| new-class tokens (\d+) acc ([\d.]+) \| (\d+)s", l)
    if m:
        e = int(m.group(1)); last_ep = e
        rows[e] = dict(run=run, epoch=e, val_pool=float(m.group(2)), val_complex=float(m.group(4)), val_complex_acc=float(m.group(5)), new_class_acc=float(m.group(7)), seconds=int(m.group(8)))
    if l.startswith("saved best") and last_ep is not None:
        saved.add(last_ep)
for e, r in sorted(rows.items()):
    p, c = steps.get(e, [[], []])
    r["train_pool"] = round(sum(p) / len(p), 5) if p else None
    r["train_complex"] = round(sum(c) / len(c), 5) if c else None
    r["best"] = 1 if e in saved else 0
json.dump({"rows": [rows[e] for e in sorted(rows)], "meta": [{"label": "Log", "value": log.split("/")[-1].split("\\")[-1]}, {"label": "Run", "value": run}]}, open(out, "w"))
print(len(rows), "epochs;", "saved at", sorted(saved))
