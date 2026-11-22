"""Score ONE model on the NON-CANONICAL residue positions only, per residue type, on a fixed set of complexes (same per-complex seeds as eval_one.py).
Positions of residues outside the 189-class vocabulary (the pipeline gives them the unknown token) are not scored; canonical positions are scored separately for reference.
usage: eval_one_nc.py NAME ADAPTER_DIR_or_none OUT.json SET   (SET = heldout | real; env as for eval_one.py)
Per complex and per residue code: n supervised tokens, flow terms on those tokens only (bb_ca, local_latents) as the training loss computes them (weighted by 1/((1-t)^2+1e-5)) and the raw
version (weight divided out, stable), residue-type cross-entropy and top-1 accuracy of the flow model's own sequence readout."""
import os, sys, json, glob, hashlib, random, time
import numpy as np
import _common
import torch, peft as pt
import torch.nn.functional as F
from pathlib import Path
import rfd4_proteina as rfd
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as ENC
name, adapter, out, which = sys.argv[1:5]
T = os.path.expanduser("~/rfd4_trial")
if which == "real": files = sorted(glob.glob(f"{T}/real_test_cif/*.cif"))
elif which == "ham": files = sorted(glob.glob(f"{T}/real_test_ham/*.cif"))  # 884 extra real complexes (Hamdani non-natural set, test split v4)
else:
    allf = sorted(glob.glob(f"{T}/train_all_cif_clean/*.cif")); G = json.load(open(f"{T}/complex_groups.json"))
    def gkey(f): e = os.path.basename(f).split("_")[0].upper(); return G.get(e, e)
    files = [f for f in allf if int(hashlib.sha1(("fmval" + gkey(f)).encode()).hexdigest(), 16) % 100 < 12]
dev = "cuda"
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.train()
if adapter != "none": model.nn = pt.PeftModel.from_pretrained(model.nn, adapter, is_trainable=False)
model.nn.eval()
t2i = {str(k): int(v) for k, v in ENC.token_to_idx.items()}; i2t = {v: k for k, v in t2i.items()}
import importlib
mod = sys.modules[model.fm.__class__.__module__]; SUPM = mod.compute_flow_supervision_masks
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
try:
    import rfproteina_encoding_ext as X
    ncset = set(map(str, list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"])))
    t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}; i2t = {v: k for k, v in t2i.items()}  # the 189-class vocabulary the model was trained with
except Exception as e:
    ncset = None; print("ext encoding not importable:", e)
print("vocab size", len(t2i), "has AIB", "AIB" in t2i, "has DTY", "DTY" in t2i, "UNK idx", t2i.get("UNK"), "NC codes", None if ncset is None else len(ncset), flush=True)
if ncset is None: ncset = {k for k in t2i if k not in CAN and k not in ("<M>", "UNK", "<A>", "<G>", "N", "DN", "A", "C", "G", "U", "DA", "DC", "DG", "DT")}
nc_idx = {t2i[k] for k in ncset if k in t2i}; can_idx = {t2i[k] for k in CAN if k in t2i}
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
res = {}; t0 = time.time()
with torch.no_grad():
    for k, f in enumerate(files):
        seed = int(hashlib.sha1(os.path.basename(f).encode()).hexdigest()[:8], 16)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {kk: v.to(dev) if isinstance(v, torch.Tensor) else v for kk, v in tb.items()}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b); model.fm.corrupt_batch(b)
            o = model.call_nn(b, n_recycle=1)
            sup = SUPM(b)["local_latents"]; rt = b["residue_type"].long(); nres = sup.sum(-1).clamp(min=1).float()
            r = {"_t": {m: float(b["t"][m].mean()) for m in ("bb_ca", "local_latents")}, "_nres": int(nres.item()), "codes": {}}
            logits = o.get("sequence", {}).get("logits")
            seqc = b.get("sequence") if "sequence" in b else None
            lm = sup.clone()
            if logits is not None:
                ce = F.cross_entropy(logits[0], rt[0], reduction="none"); acc = (logits[0].argmax(-1) == rt[0]).float()
            groups = {}
            for c in sorted(set(rt[0][sup[0]].tolist())):
                nm = i2t.get(c, str(c))
                if c in nc_idx: groups[nm] = (rt == c)
            groups["_NC_ALL"] = torch.zeros_like(sup)
            for c in nc_idx: groups["_NC_ALL"] |= (rt == c)
            groups["_CANONICAL"] = torch.zeros_like(sup)
            for c in can_idx: groups["_CANONICAL"] |= (rt == c)
            for nm, m in groups.items():
                m = m & sup; n_c = int(m.sum().item())
                if n_c == 0: continue
                b["loss_weight"] = m.float()
                d = model.fm.compute_loss(b, o)
                e = {"n": n_c}
                for mode in ("bb_ca", "local_latents"):
                    tt = b["t"][mode]; w = float(((1 - tt) ** 2 + 1e-5).mean())
                    wtd = float(d[mode].mean()) * float(nres.item()) / n_c
                    e[mode] = wtd; e[mode + "_raw"] = wtd * w
                if logits is not None:
                    e["ce"] = float(ce[m[0]].mean()); e["acc"] = float(acc[m[0]].mean())
                r["codes"][nm] = e
            b.pop("loss_weight", None)
            res[os.path.basename(f)] = r
        except Exception as ex:
            res[os.path.basename(f)] = {"error": str(ex)[:200]}
        if k % 40 == 0: print(name, which, k, "of", len(files), f"{time.time() - t0:.0f}s", flush=True)
json.dump({"name": name, "adapter": adapter, "set": which, "results": res}, open(out, "w"))
nerr = sum(1 for v in res.values() if "error" in v)
print(name, which, "done", len(files), "complexes;", nerr, "errors", flush=True)
