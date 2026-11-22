"""Inspect what a denoiser training batch looks like: which tokens are context (given), which are flow-emitted, which chains exist, where the non-canonical residues are. usage: inspect_batch_v4.py <bundle_189> <cif_dir> [n_batches]"""
import sys, glob, os, torch, _common
from functools import partial
from torch.utils.data import DataLoader
import rfd4_proteina as rfd
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
from rfproteina.model_masks import compute_flow_emit_masks
files = sorted(glob.glob(os.path.join(sys.argv[2], "*.cif"))); nb = int(sys.argv[3]) if len(sys.argv) > 3 else 6
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
dl = DataLoader(DesignProblemDataset(files[:200], transform=pipe), batch_size=2, shuffle=False, collate_fn=partial(collate_batch, schema=SCH))
for k, tb in enumerate(dl):
    if k >= nb: break
    print("=== batch", k, "keys with tensors:", {n: tuple(v.shape) for n, v in tb.items() if isinstance(v, torch.Tensor)}, flush=True)
    for i in range(tb["coords"].shape[0]):
        pm = tb["padding_mask"][i].bool(); n = int(pm.sum()); print("  sample", i, "tokens", n)
        for key in ("is_in_context_conditioning", "supports_local_latent", "chain_iid", "chain_id", "token_type", "is_motif_token_all", "is_motif_token_with_fixed_coord"):
            if key in tb:
                v = tb[key][i][pm]; print("   ", key, "unique", torch.unique(v).tolist()[:12], "counts", [int((v == u).sum()) for u in torch.unique(v)][:12] if v.dtype != torch.float32 else "")
        if "chain_iid" in tb:
            ci = tb["chain_iid"][i][pm]; rt = tb["residue_type"][i][pm].long() if "residue_type" in tb else None
            for c in torch.unique(ci).tolist(): print("    chain", c, "tokens", int((ci == c).sum()))
    try:
        fm = compute_flow_emit_masks({**tb, "padding_mask": tb["padding_mask"].bool()}); print("  flow emit masks (tokens per sample):", {m: [int(x) for x in v.flatten(1).sum(-1)] for m, v in fm.items()})
    except Exception as e: print("  flow emit masks failed:", repr(e)[:200])
print("DONE")
