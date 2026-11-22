#!/usr/bin/env python3
"""STAGE 3 - flow-model LoRA fine-tuning on the NCAA dataset. Same training logic as your fine_tuning.py, with:
  * the 176-class bundle and the extended vocabulary (via _common),
  * optional adapted decoder from stage 2,
  * cleaned/masked CIFs as input (is_resolved_mask handles bad residues),
  * the batch loop restructured so each batch is noised once and trained `inner` times exactly like your loop.
usage: ncaa_fine_tuning.py <bundle_176> <cif_dir> <out_dir> [--decoder decoder_adapted.pt] [--steps N] [--bs 3] [--inner 5] [--lr 1e-4] [--device cuda]
"""
import argparse, glob, os, torch
import _common
import peft as pt
from functools import partial
from torch.utils.data import DataLoader
import rfd4_proteina as rfd
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
p = argparse.ArgumentParser(); p.add_argument("bundle"); p.add_argument("cifs"); p.add_argument("out")
p.add_argument("--decoder"); p.add_argument("--steps", type=int, default=1000); p.add_argument("--bs", type=int, default=3)
p.add_argument("--inner", type=int, default=5); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = p.parse_args()
model = rfd.Proteina.from_pretrained(a.bundle, device=a.device); model.train()
if a.decoder: print("decoder:", model.autoencoder.decoder.load_state_dict(torch.load(a.decoder, map_location=a.device), strict=True))
targets = ["to_qkv", "to_out_node"]
modules = [f"transformer_layers.{i}.mhba.mha.{q}" for i in range(14) for q in targets]
cfg = pt.LoraConfig(r=16, lora_alpha=32, target_modules=modules, modules_to_save=["model.nn.ca_linear", "model.nn.local_latents_linear"], bias="lora_only")
model.nn = pt.get_peft_model(model.nn, cfg)
files = sorted(glob.glob(os.path.join(a.cifs, "*.cif"))); CH = 2000
ds = torch.utils.data.ConcatDataset([DesignProblemDataset(files[i:i + CH], transform=build_design_train_pipeline(crop_size=None, crop_size_range=None)) for i in range(0, len(files), CH)])
loader = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=partial(collate_batch, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample"))
opt = torch.optim.AdamW([q for q in model.nn.parameters() if q.requires_grad], lr=a.lr); os.makedirs(a.out, exist_ok=True); step = 0
while step < a.steps:
    for training_batch in loader:
        b = {k: v.to(a.device) for k, v in training_batch.items() if isinstance(v, torch.Tensor)}
        b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
        with torch.no_grad():
            model.autoencoder.encoder(b); model.add_clean_samples(b)
        model.fm.corrupt_batch(b)
        for _ in range(a.inner):
            opt.zero_grad(); out = model.call_nn(b, n_recycle=1)
            d = model.fm.optimization_terms(model.fm.compute_loss(b, out)); loss = sum(v for v in d.values()).mean()
            loss.backward(); opt.step()
        step += 1; print(f"step {step} loss {loss.item():.4f}", flush=True)
        if step % 100 == 0 or step >= a.steps: model.nn.save_pretrained(os.path.join(a.out, f"lora_step{step}"))
        if step >= a.steps: break
print("done")
