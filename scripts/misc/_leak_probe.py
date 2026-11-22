import os, sys, glob, hashlib, random, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/rfd4_trial/patch")); sys.path.insert(0, os.path.expanduser("~/rfd4_trial/code"))
import _common, rfd4_proteina as rfd
from pathlib import Path
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
import rfproteina_encoding_ext as X
T = os.path.expanduser("~/rfd4_trial"); files = sorted(glob.glob(f"{T}/real_test_cif/*.cif"))[:6]
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device="cuda"); model.train()
mod = sys.modules[model.fm.__class__.__module__]
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}
nc = {t2i[k] for k in list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"]) if k in t2i}
for f in files:
    random.seed(1); torch.manual_seed(1)
    tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
    b = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in tb.items()}
    sup = mod.compute_flow_supervision_masks(b)["local_latents"][0]; emit = mod.compute_flow_emit_masks(b)["local_latents"][0]; rt = b["residue_type"][0].long()
    isnc = torch.tensor([int(x) in nc for x in rt.tolist()], device="cuda")
    keys = [k for k in b.keys() if "seq" in k.lower() or "token_type" in k.lower() or "cond" in k.lower()]
    print(os.path.basename(f), "tokens", len(rt), "supervised", int(sup.sum()), "NC tokens", int(isnc.sum()), "NC supervised", int((isnc & sup).sum()), "NC emitted", int((isnc & emit).sum()), "| keys:", keys[:8])
    for k in keys:
        v = b[k]
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == len(rt) and v.dtype in (torch.long, torch.int, torch.int64, torch.bool):
            vv = v[0].reshape(len(rt), -1)[:, 0]
            print("   ", k, "NC tokens:", vv[isnc].tolist()[:8], "| canonical sample:", vv[~isnc][:8].tolist())
