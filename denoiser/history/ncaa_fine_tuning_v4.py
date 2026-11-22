#!/usr/bin/env python3
"""v3 (18 Nov 2026) = v2 + three options: --nc-weight W (per-token loss weight W on non-canonical positions of the 189-class vocabulary, 1 elsewhere; the held-out check is unchanged), --wcap C (cap the noise-level weight 1/((1-t)^2+1e-5) of the flow terms at C: a sample with a larger weight is scaled down to C), --plateau F (after --plateau-evals held-out checks without a new best, reload the best LoRA and multiply the learning rate by F, up to --max-decays times; early stop only after the last decay).
Denoiser (flow-model LoRA) fine-tuning TRIAL on the synthetic NCAA complexes, with a held-out check. Same training logic as ncaa_fine_tuning.py (noise a batch once, train `inner` times), plus:
  * held-out complexes split by PROTEIN FAMILY (COMPLEX_GROUPS json: entry -> family key; hash split, VAL_PCT) so no family is on both sides,
  * a held-out flow loss (fixed noise and time seed) at step 0 (the untuned model = baseline) and every --eval-every steps, each loss term printed,
  * early stopping (--patience evals without a better held-out loss), LoRA of the best eval saved in OUT/best, last in OUT/last,
  * --filter SUBSTR keeps only the training/held-out files whose name contains it (dataset-size ablation: e.g. _ncf = first batch only).
usage: ncaa_fine_tuning_v2.py <bundle_189> <cif_dir> <out_dir> [--decoder decoder.pt] [--groups complex_groups.json] [--steps N] [--bs 2] [--inner 5] [--lr 1e-4] [--eval-every 100] [--eval-batches 12] [--patience 4] [--filter _ncf] [--max-seconds S]
Saves: OUT/best (best held-out), OUT/ckpt (weights + optimizer + step at every eval), OUT/last (at exit, also on SIGTERM); continue later with --resume OUT/ckpt (or OUT/best). Safeguards: batches with loss > 1e4 or non-finite are skipped, gradient norm clipped at 10. Log lines: 'step S loss X' ; 'eval step S val X terms {...} baseline X0 delta D best B' ; 'EARLY STOP' ; 'done'."""
import argparse, glob, os, json, time, hashlib, torch
import _common
import peft as pt
from functools import partial
from itertools import islice
from torch.utils.data import DataLoader
import rfd4_proteina as rfd
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
p = argparse.ArgumentParser(); p.add_argument("bundle"); p.add_argument("cifs"); p.add_argument("out")
p.add_argument("--decoder"); p.add_argument("--groups"); p.add_argument("--steps", type=int, default=1200); p.add_argument("--epochs", type=float, default=0, help="if > 0: steps = epochs x ceil(train complexes / bs) (overrides --steps)"); p.add_argument("--bs", type=int, default=2)
p.add_argument("--inner", type=int, default=5); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--eval-batches", type=int, default=12); p.add_argument("--patience", type=int, default=4); p.add_argument("--filter", default="")
p.add_argument("--val-pct", type=int, default=12); p.add_argument("--max-seconds", type=float, default=1e9)
p.add_argument("--rank", type=int, default=16, help="LoRA rank (default 16 = the trials so far)"); p.add_argument("--alpha", type=int, default=32, help="LoRA alpha (default 32)")
p.add_argument("--extra", default="", help="comma list of extra module names INSIDE each of the 14 transformer blocks to adapt, e.g. transition.transition.swish_linear,transition.transition.linear_out,mhba.mha.to_g,mhba.adaln.to_gamma,mhba.adaln.to_beta")
p.add_argument("--resume", default="", help="a ckpt or best dir saved by an earlier run: continue from its LoRA weights (and optimizer/step if the ckpt has them)")
p.add_argument("--eval-clip", type=float, default=0.0, help="clip each held-out complex's loss at this value (0 = off)"); p.add_argument("--eval-wcap", type=float, default=0.0, help="cap the noise-level weight in the held-out check too (0 = off)"); p.add_argument("--eval-list", default="", help="text file of held-out complex file names to score first (fixed held-out set across runs)"); p.add_argument("--nc-weight", type=float, default=1.0); p.add_argument("--wcap", type=float, default=0.0); p.add_argument("--plateau", type=float, default=0.0); p.add_argument("--plateau-evals", type=int, default=3); p.add_argument("--max-decays", type=int, default=2)
p.add_argument("--clash-w", type=float, default=0.0, help="v4: weight of the clash loss through the FROZEN decoder (0 = off = v3)"); p.add_argument("--clash-d0", type=float, default=2.5); p.add_argument("--clash-tmin", type=float, default=0.6, help="clash term only for samples whose noise levels (bb_ca and local_latents) are both >= this (the clean prediction is sharp)"); p.add_argument("--max-steps-debug", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = p.parse_args()
model = rfd.Proteina.from_pretrained(a.bundle, device=a.device); model.train()
if a.decoder: print("decoder:", model.autoencoder.decoder.load_state_dict(torch.load(a.decoder, map_location=a.device), strict=True), flush=True)
if a.clash_w > 0:  # v4: the decoder is a frozen part of the loss (eval mode, no weight gradients), gradients only flow back to the LoRA through the predicted clean latent and CA
    model.autoencoder.eval()
    for _q in model.autoencoder.parameters(): _q.requires_grad_(False)
from rfproteina.model_masks import compute_flow_emit_masks
import rfproteina_encoding_ext as XE
_t2i = {str(k): int(v) for k, v in XE.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}
NC_IDX = torch.tensor(sorted({_t2i[k] for k in map(str, list(XE.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(XE.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"])) if k in _t2i}), device=a.device)
print("non-canonical classes of the vocabulary:", len(NC_IDX), "| nc-weight", a.nc_weight, "| wcap", a.wcap, "| plateau", a.plateau, flush=True)
targets = ["to_qkv", "to_out_node"]
modules = [f"transformer_layers.{i}.mhba.mha.{q}" for i in range(14) for q in targets] + [f"transformer_layers.{i}.{q.strip()}" for i in range(14) for q in a.extra.split(",") if q.strip()]
print(f"LoRA rank {a.rank}, alpha {a.alpha}, {len(modules)} adapted layers", flush=True)
cfg = pt.LoraConfig(r=a.rank, lora_alpha=a.alpha, target_modules=modules, modules_to_save=["model.nn.ca_linear", "model.nn.local_latents_linear"], bias="lora_only")
model.nn = pt.PeftModel.from_pretrained(model.nn, a.resume, is_trainable=True) if a.resume else pt.get_peft_model(model.nn, cfg)
import atexit, signal
def _save_last(): model.nn.save_pretrained(os.path.join(a.out, "last")); print("saved last", flush=True)
atexit.register(_save_last); signal.signal(signal.SIGTERM, lambda *x: (_ for _ in ()).throw(SystemExit(0)))
files = sorted(f for f in glob.glob(os.path.join(a.cifs, "*.cif")) if a.filter in os.path.basename(f))
G = json.load(open(a.groups)) if a.groups else {}
def gkey(f): e = os.path.basename(f).split("_")[0].upper(); return G.get(e, e)
def is_val(f): return int(hashlib.sha1(("fmval" + gkey(f)).encode()).hexdigest(), 16) % 100 < a.val_pct
val_f = [f for f in files if is_val(f)]; train_f = [f for f in files if not is_val(f)]
if a.eval_list:  # put these held-out complexes first, so --eval-batches scores the SAME complexes (same noise draws) as earlier runs, whatever the pool size
    _nm = [l.strip() for l in open(a.eval_list) if l.strip()]; _by = {os.path.basename(f): f for f in val_f}; _first = [_by[n] for n in _nm if n in _by]
    val_f = _first + [f for f in val_f if f not in set(_first)]; print(f"eval list: {len(_first)} of {len(_nm)} listed complexes are held-out and scored first", flush=True)
assert not ({gkey(f) for f in val_f} & {gkey(f) for f in train_f}), "family on both sides"
print(f"complexes: train {len(train_f)} (families {len({gkey(f) for f in train_f})}) held-out {len(val_f)} (families {len({gkey(f) for f in val_f})})", flush=True)
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
def mk(fl, shuffle, bs):
    return DataLoader(torch.utils.data.ConcatDataset([DesignProblemDataset(fl[i:i + 1000], transform=pipe) for i in range(0, len(fl), 1000)]), batch_size=bs, shuffle=shuffle, collate_fn=partial(collate_batch, schema=SCH))
loader, vloader = mk(train_f, True, a.bs), mk(val_f, False, 1)  # held-out is always scored one complex at a time (same complexes and noise for every run, whatever the training batch size); --eval-batches = number of held-out complexes
if a.epochs > 0: a.steps = int(a.epochs * -(-len(train_f) // a.bs)); print(f"{a.epochs} epochs x {len(train_f)} complexes / batch {a.bs} = {a.steps} steps", flush=True)
opt = torch.optim.AdamW([q for q in model.nn.parameters() if q.requires_grad], lr=a.lr); os.makedirs(a.out, exist_ok=True)


def prep(tb):
    b = {k: v.to(a.device) for k, v in tb.items() if isinstance(v, torch.Tensor)}
    b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
    with torch.no_grad():
        model.autoencoder.encoder(b); model.add_clean_samples(b)
    model.fm.corrupt_batch(b)
    return b


CS = dict(pairs=0, used=0, samples=0); decoder_dbg = [0]
def clash_term(b, out):
    """v4: clash loss of the DECODED clean prediction. x_1 = clean prediction of the denoiser (CA and local latent) -> the same decode as generation (frozen decoder) -> atoms of the generated tokens (flow emit mask of bb_ca) against the TRUE atoms of the context tokens (receptor, detached): sum of max(0, d0 - d)^2 in A^2 per sample, mean over the samples with t >= clash-tmin. Returns (loss, pairs below 2 A, samples used)."""
    zero = torch.zeros((), device=b["coords"].device)
    t = torch.minimum(b["t"]["bb_ca"], b["t"]["local_latents"]).reshape(-1)
    use = t >= a.clash_tmin
    if not bool(use.any()): return zero, 0, 0
    x1 = model.fm.nn_out_to_clean_sample_prediction(batch=b, nn_out=out)
    fm = model._format_sample_local_latents(x=x1, ret_mode="coors37_n_aatype", batch=b)
    # peptide tokens = every token of the chain(s) that hold a non-canonical residue (all training complexes are real receptors with a swapped non-canonical peptide residue); in these batches every token is noised and is_in_context_conditioning is all False, so chain_iid + residue_type is the only split
    pad = b["padding_mask"].bool(); rt = b["residue_type"].long(); ci = b["chain_iid"].long(); isnc = torch.isin(rt, NC_IDX) & pad
    gen = torch.zeros_like(pad)
    for _i in range(pad.shape[0]):
        _pc = torch.unique(ci[_i][isnc[_i]])
        if _pc.numel(): gen[_i] = torch.isin(ci[_i], _pc) & pad[_i]
    P, pm = fm["coors"], fm["atom_mask"].bool() & gen[..., None]
    R, rm = b["coords"].detach(), b["atom37_mask"].bool() & ~gen[..., None] & pad[..., None]
    if a.max_steps_debug and decoder_dbg[0] < 40: decoder_dbg[0] += 1; print("   clash_term dbg: t_bb", [round(float(v), 2) for v in b["t"]["bb_ca"].reshape(-1)], "t_ll", [round(float(v), 2) for v in b["t"]["local_latents"].reshape(-1)], "| tokens", [int(x) for x in b["padding_mask"].sum(-1)], "PEPTIDE tokens", [int(x) for x in gen.sum(-1)], "peptide atoms", [int(x) for x in pm.flatten(1).sum(-1)], "receptor atoms", [int(x) for x in rm.flatten(1).sum(-1)], flush=True)
    tot = zero; n = 0; pairs = 0
    for i in torch.nonzero(use).flatten().tolist():
        if not bool(pm[i].any()) or not bool(rm[i].any()): continue
        d = torch.cdist(P[i][pm[i]], R[i][rm[i]]); tot = tot + (torch.relu(a.clash_d0 - d) ** 2).sum(); pairs += int((d < 2.0).sum()); n += 1
    CS["pairs"] += pairs; CS["used"] += n; CS["samples"] += int(t.numel())
    return (tot / max(n, 1)), pairs, n


@torch.no_grad()
def val_loss():
    import random as _r, numpy as _np
    _r.seed(1234); _np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)  # python, numpy and torch generators: the data pipeline's random rotation / design-task sampling must repeat exactly, or the held-out number drifts between evaluations
    model.nn.eval(); tot = 0.0; n = 0; terms = {}; _per = []
    for tb in islice(vloader, a.eval_batches):
        b = prep(tb); out = model.call_nn(b, n_recycle=1); d = model.fm.optimization_terms(model.fm.compute_loss(b, out))
        if a.eval_wcap > 0:  # robust held-out check: an extreme low-noise draw (weight up to 1e5) can make one complex weigh 1e11; cap the weight as in training
            for _m in ("bb_ca", "local_latents"):
                _w = 1.0 / ((1.0 - b["t"][_m]) ** 2 + 1e-5); d[_m] = d[_m] * (a.eval_wcap / _w).clamp(max=1.0)
        if a.clash_w > 0:  # the clash term is part of the held-out score (baseline included), weighted like in training
            _c, _, _ = clash_term(b, out); d["clash_w"] = a.clash_w * _c.reshape(1)
        _l = sum(v for v in d.values()).mean().item(); _per.append(round(_l, 2))
        if a.eval_clip > 0: _l = min(_l, a.eval_clip)  # a complex whose noise draw makes the loss explode (seen: 6.6e12 next to normal 40 to 66) counts as eval_clip, not as its value
        tot += _l; n += 1
        for k, v in d.items(): terms[k] = terms.get(k, 0.0) + float(v.mean())
    model.nn.train(); print("held-out per-complex losses (first 30, max %.4g):" % max(_per), _per, flush=True)
    return tot / max(n, 1), {k: round(v / max(n, 1), 4) for k, v in terms.items()}


base, bterms = val_loss(); best = base; bad = 0; step = 0; t0 = time.time(); decays = 0
if a.resume and os.path.exists(os.path.join(a.resume, "state.json")):
    st = json.load(open(os.path.join(a.resume, "state.json"))); base = st["baseline"]; best = st["best"]; step = st["step"]; bad = 0
    if os.path.exists(os.path.join(a.resume, "optimizer.pt")): opt.load_state_dict(torch.load(os.path.join(a.resume, "optimizer.pt"), map_location=a.device))
    print(f"resumed from {a.resume}: step {step}, best {best:.4f}, original baseline {base:.4f}", flush=True)
os.makedirs(a.out, exist_ok=True); model.nn.save_pretrained(os.path.join(a.out, "best"))  # the untuned LoRA (B = 0) is the first "best", so a plateau reload always has a target
print(f"eval step 0 val {base:.4f} terms {bterms} baseline {base:.4f} (untuned model)", flush=True)
stop = False
while step < a.steps and not stop:
    for tb in loader:
        b = prep(tb)
        if a.nc_weight != 1.0: b["loss_weight"] = 1.0 + (a.nc_weight - 1.0) * torch.isin(b["residue_type"].long(), NC_IDX).float()  # more weight on the non-canonical positions (training only)
        for _ in range(a.inner):
            opt.zero_grad(); out = model.call_nn(b, n_recycle=1)
            d = model.fm.optimization_terms(model.fm.compute_loss(b, out))
            if a.wcap > 0:
                for _m in ("bb_ca", "local_latents"):
                    _w = 1.0 / ((1.0 - b["t"][_m]) ** 2 + 1e-5); d[_m] = d[_m] * (a.wcap / _w).clamp(max=1.0)  # cap the noise-level weight
            flow = sum(v for v in d.values()).mean(); loss = flow; cl_ = None
            if a.clash_w > 0:
                cl_, cpairs, cused = clash_term(b, out); loss = flow + a.clash_w * cl_
            if not torch.isfinite(loss) or loss.item() > 1e4:  # an outlier batch (seen: 4.7e8) would wreck the LoRA weights: skip it
                print(f"skipped a batch with loss {loss.item():.3g}", flush=True); opt.zero_grad(); break
            loss.backward(); torch.nn.utils.clip_grad_norm_([q for q in model.nn.parameters() if q.requires_grad], 10.0); opt.step()
        step += 1; print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s" + (f" | flow {flow.item():.3f} clash {float(cl_):.4f} x w {a.clash_w:g} pairs<2A {cpairs} used samples {cused} of {b['t']['bb_ca'].numel()} t_min {float(torch.minimum(b['t']['bb_ca'], b['t']['local_latents']).min()):.2f}" if a.clash_w > 0 else ""), flush=True)
        if a.max_steps_debug and step >= a.max_steps_debug: stop = True; break
        if step % a.eval_every == 0 or step >= a.steps:
            v, terms = val_loss()
            if v < best - 1e-4:
                best = v; bad = 0; model.nn.save_pretrained(os.path.join(a.out, "best"))
            else:
                bad += 1
                if a.plateau > 0 and bad >= a.plateau_evals and decays < a.max_decays:
                    import safetensors.torch as _st
                    from peft import set_peft_model_state_dict as _setp
                    _setp(model.nn, _st.load_file(os.path.join(a.out, "best", "adapter_model.safetensors"), device=a.device))
                    for _g in opt.param_groups: _g["lr"] *= a.plateau
                    decays += 1; bad = 0; print(f"PLATEAU: reloaded the best LoRA, learning rate -> {opt.param_groups[0]['lr']:.2e} (decay {decays}/{a.max_decays})", flush=True)
            ck = os.path.join(a.out, "ckpt"); model.nn.save_pretrained(ck); torch.save(opt.state_dict(), os.path.join(ck, "optimizer.pt"))
            json.dump(dict(step=step, best=best, baseline=base, last_val=v), open(os.path.join(ck, "state.json"), "w"))
            print(f"eval step {step} val {v:.4f} terms {terms} baseline {base:.4f} delta {v - base:+.4f} best {best:.4f} no-improve {bad}/{a.patience}", flush=True)
            if bad >= a.patience and (a.plateau <= 0 or decays >= a.max_decays): print("EARLY STOP: held-out flow loss stopped improving", flush=True); stop = True; break
        if time.time() - t0 > a.max_seconds: print("STOP: time limit", flush=True); stop = True; break
        if step >= a.steps: break
model.nn.save_pretrained(os.path.join(a.out, "last")); print("done", flush=True)
