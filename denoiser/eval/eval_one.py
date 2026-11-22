"""Score ONE model (untuned control, or the base model + a saved LoRA) on EVERY held-out complex, one complex at a time.
Each complex gets its own fixed seed (from its file name) for the data pipeline AND the noise, so every model sees identical noise on each complex: paired, repeatable.
usage: eval_one.py NAME ADAPTER_DIR_or_none OUT.json     (env as for ncaa_fine_tuning_v2.py)"""
import os, sys, json, glob, hashlib, random, time
import numpy as np
import _common
import torch, peft as pt
from pathlib import Path
from functools import partial
import rfd4_proteina as rfd
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
name, adapter, out = sys.argv[1:4]
T = os.path.expanduser("~/rfd4_trial")
files = sorted(glob.glob(f"{T}/train_all_cif_clean/*.cif"))
G = json.load(open(f"{T}/complex_groups.json"))
def gkey(f): e = os.path.basename(f).split("_")[0].upper(); return G.get(e, e)
val_f = [f for f in files if int(hashlib.sha1(("fmval" + gkey(f)).encode()).hexdigest(), 16) % 100 < 12]
dev = "cuda"
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.train()
if adapter != "none": model.nn = pt.PeftModel.from_pretrained(model.nn, adapter, is_trainable=False)
model.nn.eval()
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
res = {}; t0 = time.time()
with torch.no_grad():
    for k, f in enumerate(val_f):
        seed = int(hashlib.sha1(os.path.basename(f).encode()).hexdigest()[:8], 16)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {kk: v.to(dev) for kk, v in tb.items() if isinstance(v, torch.Tensor)}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b); model.fm.corrupt_batch(b)
            o = model.call_nn(b, n_recycle=1); d = model.fm.optimization_terms(model.fm.compute_loss(b, o))
            r = {kk: float(v.mean()) for kk, v in d.items()}; r["total"] = sum(r.values()); r["t_ca"] = float(b["t"]["bb_ca"].mean()) if "t" in b else None
            res[os.path.basename(f)] = r
        except Exception as e:
            res[os.path.basename(f)] = {"error": str(e)[:120]}
        if k % 40 == 0: print(name, k, "of", len(val_f), f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(name=name, adapter=adapter, n=len(res), results=res), open(out, "w"))
ok = [v["total"] for v in res.values() if "total" in v]
print(name, "done", len(ok), "of", len(val_f), "mean total", sum(ok) / max(1, len(ok)), flush=True)
