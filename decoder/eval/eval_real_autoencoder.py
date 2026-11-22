"""Sequence recovery of the 176-class decoder on the REAL benchmark peptides (non-redundant set).
Input to the model = the real peptide CIF (coordinates + atoms); the model must name each residue. Truth = benchmark_residues.tsv class_idx.
Variants: 'bundle' (fresh 176 bundle = new class rows untrained), plus any weight files given as name=path."""
import local_vocab  # must be first
import rfd4_proteina as rfd
import os, sys, csv, collections, torch
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
B = os.environ.get("BENCH", "/gpfs/projects/stf/legia8/NCAA_dataset/real_benchmark"); BUNDLE = os.environ.get("RFD_BUNDLE")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = rfd.Proteina.from_pretrained(BUNDLE, device=device); model.eval()
encoder, decoder = model.autoencoder.encoder, model.autoencoder.decoder
variants = [("bundle", None)] + [tuple(a.split("=", 1)) for a in sys.argv[1:]]
items = {r["item_id"]: r for r in csv.DictReader(open(B + "/benchmark_items.tsv"), delimiter="\t") if r["nr_representative"] == "1"}
res = collections.defaultdict(list)
for r in csv.DictReader(open(B + "/benchmark_residues.tsv"), delimiter="\t"):
    if r["item_id"] in items: res[r["item_id"]].append(r)
cat = {r["code"]: r for r in csv.DictReader(open(B + "/trained_ncaa_catalog.csv"))}
idx2code = {int(r["class_idx"]): c for c, r in cat.items()}
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None)
SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
# build all batches once (one item per batch element; items that fail to build are skipped and counted)
batches, skipped = [], []
for item in items:
    f = f"{B}/structures_fixed/{item}.cif"
    try:
        ds = DesignProblemDataset([Path(f)], transform=pipe); batches.append((item, collate_batch([ds[0]], schema=SCHEMA)))
    except Exception as e:
        skipped.append((item, (type(e).__name__ + ': ' + str(e))[:300]))
print("items", len(items), "built", len(batches), "skipped", len(skipped), flush=True)
for it, e in skipped[:3]: print("  skip", it, e, flush=True)
@torch.no_grad()
def predict(batch):
    b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
    b["z_latent"] = encoder(b)["z_latent"]; b["ca_coors_nm"] = b["coords_nm"][..., 1, :]
    out = decoder(b); emit = out["emit_mask"][0]
    return out["seq_logits"][0][emit].argmax(-1).cpu().tolist(), b["residue_type"][0][emit].long().cpu().tolist()
summary = {}
for name, path in variants:
    if path:
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
    rows = []; skipped_len = 0
    for item, batch in batches:
        pred, tru = predict(batch)
        rr = res[item]
        # pair the model's tokens with the benchmark residues by their true class ids (residues outside the vocabulary are not
        # tokens the decoder emits, so a plain index match fails for peptides that contain them)
        import difflib
        bt = [int(r["class_idx"]) for r in rr]
        sm = difflib.SequenceMatcher(None, tru, bt, autojunk=False)
        pairs = [(i + k, j + k) for i, j, n in sm.get_matching_blocks() for k in range(n)]
        if len(pairs) == 0: skipped_len += 1; continue
        for i, j in pairs:
            row = rr[j]
            if row["scored"] != "1": continue
            rows.append((item, j, row["ccd"], row["category"], int(row["class_idx"]), pred[i], int(pred[i] == int(row["class_idx"])), items[item]["tier"]))
    with open(f"{B}/../ae_eval_{name}.tsv" if False else f"ae_eval_{name}.tsv", "w", newline="") as fo:
        w = csv.writer(fo, delimiter="\t"); w.writerow(["item", "pos", "ccd", "category", "true_idx", "pred_idx", "correct", "tier"]); w.writerows(rows)
    for tier in ("A", "B", "ALL"):
        sub = [r for r in rows if tier == "ALL" or r[7] == tier]
        c = collections.defaultdict(lambda: [0, 0])
        for _, _, ccd, ct, t, p, ok, _t in sub: c[ct][0] += ok; c[ct][1] += 1; c["ALL"][0] += ok; c["ALL"][1] += 1
        print(f"== {name} | tier {tier}: positions {len(sub)}, items with no alignable token {skipped_len}", flush=True)
        for k in sorted(c): print(f"   {k:16s} {c[k][0]:5d}/{c[k][1]:5d} = {100*c[k][0]/max(1,c[k][1]):.1f}%", flush=True)
        per = collections.defaultdict(lambda: [0, 0])
        for _, _, ccd, ct, t, p, ok, _t in sub:
            if ct == "trained_ncaa": per[ccd][0] += ok; per[ccd][1] += 1
        print("   per-NCAA recall:", ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items())), flush=True)
        print("   codes covered:", len(per), "of 32", flush=True)
