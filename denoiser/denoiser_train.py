#!/usr/bin/env python3
"""denoiser_train.py - fine-tune the RFD4 DENOISER (flow model that generates new structures) with a small LoRA adapter on synthetic NCAA complexes.

WHAT THIS SCRIPT IS
  One script for every denoiser trial of this project (trials A to G3 and the clash trial G4 in Results.MD). The older scripts
  (ncaa_fine_tuning.py, _v2, _v3, _v4, kept in denoiser/history/) were successive versions of this file; each added options that are all
  OFF by default here, so a plain call reproduces the first recipe and the options switch on the later ones (see TRIAL PRESETS).

THE MODEL IN ONE PARAGRAPH
  RFD4 generates a peptide-protein complex in two coupled parts: C-alpha coordinates and a small latent vector per residue (flow matching:
  noise is removed step by step). A decoder (see decoder/) turns the latents into atoms. Here the pretrained denoiser is frozen and only a
  LoRA adapter is trained (rank 16: 1.08 M of 178 M numbers; attention to_qkv and to_out_node of the 14 transformer blocks + the two output
  heads for CA and latents). The autoencoder (encoder + decoder) is frozen too.

TRAINING LOOP
  For every batch: encode the complex, add noise at a random level t (each of the two parts has its own t), predict the clean complex, take the
  flow loss, and repeat the update --inner times on the same noised batch (5 in all trials). Gradients are clipped at norm 10; a batch with a
  loss above 1e4 or non-finite is skipped (single outlier batches of 1e7 to 1e16 occur at very low noise and are expected).

HELD-OUT CHECK
  Complexes are split by PROTEIN FAMILY (--groups json, hash split, --val-pct 12) so no family is on both sides. Every --eval-every steps the
  held-out flow loss is computed with fixed noise (same complexes and noise for every run, so runs are comparable) and the LoRA of the best
  check is saved in OUT/best; OUT/ckpt (with optimizer + step, for --resume) and OUT/last are also saved. Early stop after --patience checks
  without a better value.

OPTIONS ADDED BY LATER VERSIONS (all off by default)
  v3  --nc-weight W    loss weight W on non-canonical positions (1 elsewhere);  --wcap C  cap the noise-level weight 1/((1-t)^2+1e-5) at C;
      --plateau F      after --plateau-evals checks without a new best, reload the best LoRA and multiply the learning rate by F (up to --max-decays times);
      --eval-clip / --eval-wcap / --eval-list   robust, comparable held-out check (a fixed list of complexes; clip one complex's loss that explodes)
  v2  --rank/--alpha/--extra (bigger adapters), --resume, --filter SUBSTR (use only files whose name contains it)
  v4  --clash-w W      CLASH LOSS THROUGH THE FROZEN DECODER (trial G4, Results.MD section 66): the denoiser's predicted clean latent and CA go through the
                       frozen decoder (--decoder, run 8) and the decoded peptide atoms are penalised for coming closer than --clash-d0 (2.5 A) to the TRUE
                       receptor atoms, only for samples whose two noise levels are both >= --clash-tmin (0.6). Peptide = every chain that holds a
                       non-canonical residue (training batches have all tokens noised and no flagged context). Result: no improvement over G3.
      --max-steps-debug N   stop after N steps and print per-sample diagnostics (used for the three debug jobs)

USAGE
  python denoiser_train.py BUNDLE_189 CIF_DIR OUT_DIR [--decoder DECODER.pt] [--groups complex_groups.json] [--steps N | --epochs E] [--bs 4] [--inner 5]
         [--lr 1e-4] [--eval-every 100] [--eval-batches 30] [--patience 5] [--resume DIR] [options above] [--max-seconds S]
  Needs the RFD4 python environment with peft and the patch folder (Dataset_ncaa/multigpu/rfd4_ncaa_patch: _common.py, rfproteina_encoding_ext.py) on PYTHONPATH.
  Log lines: 'step S loss X' ; 'eval step S val X terms {...} baseline X0 delta D best B' ; 'EARLY STOP' ; 'PLATEAU' ; 'done'.

TRIAL PRESETS (data folders are on the cluster; see denoiser/PRESETS.md for results)
  A  CIF_DIR=train_all_cif_clean (1,266 training complexes)  --bs 2 --steps 1000 --eval-batches 14 --patience 4
  B  same, --filter _ncf (first batch only, 228 complexes)
  C  --bs 1 --steps 3000 --eval-every 150 --eval-batches 30 --patience 8 --lr 1e-4
  D  C + --rank 64 --alpha 128                         E  D + --extra transition.transition.swish_linear,transition.transition.linear_out,mhba.mha.to_g
  F  C on train_all_cif_clean2 (6,457 complexes); F H200: --bs 4 --lr 2e-4 --epochs 3;  F2: --resume F/best --lr 5e-5 --epochs 1
  G  C + --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --patience 3 (box, batch 1)
  G2 --resume G/best, 8,947-complex pool, --bs 4 --lr 5e-5 --steps 3000 --patience 5 --eval-wcap 100 --eval-clip 150 --eval-list heldout30_list.txt  (default denoiser candidate)
  G3 G2 recipe on the class-balanced subset (train_all_cif_bal280: 2,232 training complexes)  <- BEST, default denoiser
  G4 G3 start (--resume G3/best) + --clash-w 2 --decoder decoder_v9_499602.pt --lr 3e-5 --steps 600  (no improvement)
"""
import argparse, glob, os, sys, json, time, hashlib, torch
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
p.add_argument("--train-readout", action="store_true", help="v5: also train the sequence readout head (189 classes) on top of the LoRA (it was frozen in all earlier trials)"); p.add_argument("--readout-attach", action="store_true", help="v5: let the readout loss flow into the transformer (config default: the readout is a detached probe)"); p.add_argument("--seq-w", type=float, default=1.0, help="v5: weight of the sequence-readout CE (config default 1.0)"); p.add_argument("--nc-hide-p", type=float, default=0.0, help="v5: probability that the identity of a non-canonical token is HIDDEN (conditioning set to <M>) in a batch, so the readout must infer the NCAA from the structure; 0 = never (all earlier trials)")
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
if a.readout_attach:
    for _m in model.nn.modules():
        if hasattr(_m, "sequence_readout_detach"): _m.sequence_readout_detach = False; print("sequence readout attached to the trunk", flush=True)
if a.seq_w != 1.0: model.fm.cfg_exp.loss.sequence_prediction["weight"] = a.seq_w
targets = ["to_qkv", "to_out_node"]
modules = [f"transformer_layers.{i}.mhba.mha.{q}" for i in range(14) for q in targets] + [f"transformer_layers.{i}.{q.strip()}" for i in range(14) for q in a.extra.split(",") if q.strip()]
print(f"LoRA rank {a.rank}, alpha {a.alpha}, {len(modules)} adapted layers", flush=True)
cfg = pt.LoraConfig(r=a.rank, lora_alpha=a.alpha, target_modules=modules, modules_to_save=["model.nn.ca_linear", "model.nn.local_latents_linear"] + (["model.nn.sequence_readout"] if a.train_readout else []), bias="lora_only")
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


from atomworks.ml.conditions import C_SEQ
M_IDX = int(XE.UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]); CAN_IDX = torch.tensor(sorted(_t2i[k] for k in "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()), device=a.device)
_mod = sys.modules[model.fm.__class__.__module__]; SUPM = _mod.compute_flow_supervision_masks
RS = dict(n=0, top1=0, top5=0, canon=0, given=0, hidden=0)   # readout statistics at non-canonical tokens
def prep(tb):
    b = {k: v.to(a.device) for k, v in tb.items() if isinstance(v, torch.Tensor)}
    b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
    with torch.no_grad():
        model.autoencoder.encoder(b); model.add_clean_samples(b)
    if C_SEQ.full_name in b:   # v5: hide the identity of non-canonical tokens with probability --nc-hide-p (the readout must then infer the NCAA from the structure)
        nc = torch.isin(b["residue_type"].long(), NC_IDX)
        RS["given"] += int((nc & (b[C_SEQ.full_name] != M_IDX)).sum()); RS["hidden"] += int((nc & (b[C_SEQ.full_name] == M_IDX)).sum())   # counted BEFORE hiding: how often the pipeline hands the NCAA identity to the model
        if a.nc_hide_p > 0: b[C_SEQ.full_name] = torch.where(nc & (torch.rand(nc.shape, device=nc.device) < a.nc_hide_p), torch.full_like(b[C_SEQ.full_name], M_IDX), b[C_SEQ.full_name])
    model.fm.corrupt_batch(b)
    return b


def nc_readout(b, out):
    """top-1 / top-5 accuracy of the sequence readout at generated non-canonical tokens whose identity is hidden, and how often its guess is a CANONICAL class (the canonical bias)."""
    lg = out.get("sequence", {}).get("logits") if isinstance(out, dict) else None
    if lg is None or C_SEQ.full_name not in b: return
    rt = b["residue_type"].long(); m = SUPM(b)["local_latents"] & (b[C_SEQ.full_name] == M_IDX) & torch.isin(rt, NC_IDX)
    n = int(m.sum())
    if n == 0: return
    pr = lg.argmax(-1)[m]; RS["n"] += n; RS["top1"] += int((pr == rt[m]).sum()); RS["top5"] += int((lg.topk(5, -1).indices[m] == rt[m][:, None]).any(-1).sum()); RS["canon"] += int(torch.isin(pr, CAN_IDX).sum())


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
    model.nn.eval(); tot = 0.0; n = 0; terms = {}; _per = []; _hp = a.nc_hide_p
    if a.train_readout or a.nc_hide_p > 0: a.nc_hide_p = 1.0   # held-out readout check: identity of every NCAA hidden
    for k_ in RS: RS[k_] = 0
    for tb in islice(vloader, a.eval_batches):
        b = prep(tb); out = model.call_nn(b, n_recycle=1); nc_readout(b, out); d = model.fm.optimization_terms(model.fm.compute_loss(b, out))
        if a.eval_wcap > 0:  # robust held-out check: an extreme low-noise draw (weight up to 1e5) can make one complex weigh 1e11; cap the weight as in training
            for _m in ("bb_ca", "local_latents"):
                _w = 1.0 / ((1.0 - b["t"][_m]) ** 2 + 1e-5); d[_m] = d[_m] * (a.eval_wcap / _w).clamp(max=1.0)
        if a.clash_w > 0:  # the clash term is part of the held-out score (baseline included), weighted like in training
            _c, _, _ = clash_term(b, out); d["clash_w"] = a.clash_w * _c.reshape(1)
        _l = sum(v for v in d.values()).mean().item(); _per.append(round(_l, 2))
        if a.eval_clip > 0: _l = min(_l, a.eval_clip)  # a complex whose noise draw makes the loss explode (seen: 6.6e12 next to normal 40 to 66) counts as eval_clip, not as its value
        tot += _l; n += 1
        for k, v in d.items(): terms[k] = terms.get(k, 0.0) + float(v.mean())
    a.nc_hide_p = _hp
    if RS["n"]: print(f"   held-out readout at NCAA tokens (identity hidden): n {RS['n']} top-1 {RS['top1'] / RS['n']:.3f} top-5 {RS['top5'] / RS['n']:.3f} | guess is a CANONICAL class in {RS['canon'] / RS['n']:.3f} of the tokens", flush=True)
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
        if step % 50 == 0 and (a.train_readout or a.nc_hide_p > 0): print(f"   NCAA tokens seen so far (before hiding): identity given {RS['given']} hidden {RS['hidden']}", flush=True)
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
