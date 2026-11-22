"""decoder_train.py - fine-tune the RFD4 DECODER (latent -> all-atom structure) on non-canonical amino acids (NCAAs).

WHAT THIS SCRIPT IS
  One script that contains every decoder training recipe used in this project (runs 1 to 9 of Results.MD). The older scripts
  (decoder_fine_tune_v2 ... v10, kept in decoder/history/) were successive versions of this file; each version only ADDED a loss term,
  so the whole history is reproduced here by switching the loss weights on or off with environment variables (see RUN PRESETS).

THE MODEL IN ONE PARAGRAPH
  RFD4 turns a protein structure into a small "latent" vector per residue (encoder, frozen here) and back into atoms (decoder, trained here).
  The denoiser (see denoiser/) generates new latents; the decoder must read them as atoms and residue names. Here the whole decoder
  (about 128 M numbers) is trained on 189 residue classes (147 original + 29 NCAAs + 13 D-amino-acid forms); the encoder stays frozen.

DATA (all local-file inputs, no real test complex is ever used)
  POOL_DIR     46k synthetic NCAA peptides that passed the strict geometry test (80/10/10 split by file, seed 0)
  MIRROR_LIST  optional list of mirror-image D-peptides added to the pool training set (gives real D-geometry)
  PILOT_DIR    synthetic peptide-protein complexes (CIF files made by the NCFlow pipeline, see data_tools/); the new D-forms live only here
  COMPLEX_GROUPS  json: complex -> protein-family key; the held-out 12% (VAL_PCT) are whole families, so no family is on both sides
  Every step uses one pool batch and one complex batch: loss = loss(pool) + LAMBDA * loss(complexes).

LOSS (per batch) = residue-type cross-entropy
                 + STRUCT_W * squared error of the decoded atoms against the true atoms (A^2; non-canonical tokens count NCW times)
                 + CHI_W    * chirality loss (signed volume of N, C, CB around CA keeps its true sign, so D-forms stay D)
                 + BOND_W   * peptide-bond loss: flat-bottom penalty on the decoded C(i)-N(i+1) distance, 1.33 +- BOND_TOL A (bond_loss.py)
                 + CLASH_W  * clash loss on peptide-receptor atom pairs below CLASH_D0 (bond_loss.py), true latent and (NOISE_SIGMA > 0) a noised latent
  A weight of 0 switches the term off. BLIND=1 hides the residue identity from the decoder (conditioning input set to <M>); BLIND=0 gives it.

ENVIRONMENT VARIABLES (name = default)
  paths     RFD_BUNDLE (189-class RFD4 bundle, required)  RFD4_VOCAB_YAML (default PATCH_DIR/aa_rna_dna_glycan_ncaa_ext_v3.yaml)  PATCH_DIR (folder with _common.py; default = this folder)
            POOL_DIR  POOL_LIST  MIRROR_LIST  PILOT_DIR  COMPLEX_GROUPS  INIT (decoder state dict to start from; none = bundle decoder)  SAVE_PATH (output .pt)
  schedule  BATCH (pool batch)  PBATCH (complex batch)  EPOCHS=3  LAMBDA=1.0  LR_BODY=3e-5  LR_HEAD=1e-4  MAX_STEPS=0  MAX_SECONDS  PATIENCE=3  VAL_PCT=12  NUM_WORKERS=4  CLIP=5 (gradient-norm clip)
  losses    STRUCT_W=0.5  NCW=2  CHI_W=0.15  BOND_W=10  BOND_TOL=0.04  BOND_NOISE=0 (bond loss on a noised decode; 0.1 diverged)
            CLASH_W=0  CLASH_D0=2.5  NOISE_SIGMA=0  CA_JIT=0  NOISY_BOND_W=0   (the defaults are the RUN 8 recipe = the best decoder; the clash options are the failed run 9)
  other     BLIND=0  DEBUG_TERMS=0 (1 = print every loss term every 20 steps)
  Upload this file together with bond_loss.py and _common.py (Dataset_ncaa/multigpu/rfd4_ncaa_patch) to the same folder on the cluster.

RUN PRESETS (see decoder/PRESETS.md for the table with weights and job numbers)
  run 3   STRUCT_W=0 CHI_W=0 BOND_W=0 MIRROR_LIST=...   (residue-type loss + mirror D-peptides)
  run 4   BLIND=1 INIT=run 3 (not used further)
  run 5   STRUCT_W=0.25 CHI_W=0 BOND_W=0 NCW=1 INIT=run 3
  run 5b  same as run 5, INIT=run 5 (default for generation until run 8)
  run 6   STRUCT_W=0.25 NCW=3 CHI_W=0.5 BOND_W=0 INIT=run 5b (not used)
  run 7   STRUCT_W=0.5 NCW=2 CHI_W=0.15 BOND_W=0 INIT=run 5b (handedness-first alternative)
  run 8   defaults above, INIT=run 5b  -> decoder_v9_499602.pt, THE DEFAULT DECODER
  run 9   run 8 + CLASH_W=5 NOISE_SIGMA=0.2 INIT=run 8 -> decoder_v10_500687.pt (did NOT help, kept for the record)

OUTPUT   SAVE_PATH = state dict of the best epoch (judged on pool-val + complex-val loss). Log lines: 'ep E step S pool X complex Y', per-epoch validation with
         D-form / L-residue handedness, peptide-bond accuracy, clash counts, accuracy of the new classes, and per-category (canonical / non-canonical) losses."""
import os, sys, time, random, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.environ.get("PATCH_DIR", HERE), HERE):
    sys.path.insert(0, d)
os.environ.setdefault("RFD4_VOCAB_YAML", os.path.join(os.environ.get("PATCH_DIR", HERE), "aa_rna_dna_glycan_ncaa_ext_v3.yaml"))
import _common  # puts the 189-class vocabulary in before rfproteina is imported
import rfd4_proteina as rfd
import torch, torch.nn.functional as F
from pathlib import Path
from functools import partial
from torch.utils.data import ConcatDataset, DataLoader
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = rfd.Proteina.from_pretrained(os.environ["RFD_BUNDLE"], device=device)
encoder, decoder = model.autoencoder.encoder, model.autoencoder.decoder
if os.environ.get('INIT'): print('init from', os.environ['INIT'], decoder.load_state_dict(torch.load(os.environ['INIT'], map_location=device), strict=True), flush=True)
for p in model.parameters(): p.requires_grad_(False)
head = {id(decoder.init_repr_factory.linear_out.weight), id(decoder.logit_linear[1].weight)}
body = [p for p in decoder.parameters() if id(p) not in head]
heads = [p for p in decoder.parameters() if id(p) in head]
for p in body + heads: p.requires_grad_(True)
LR_BODY, LR_HEAD = float(os.environ.get("LR_BODY", 3e-5)), float(os.environ.get("LR_HEAD", 1e-4))
opt = torch.optim.AdamW([{"params": body, "lr": LR_BODY}, {"params": heads, "lr": LR_HEAD}], weight_decay=0.0)
print(f"trainable: body {sum(p.numel() for p in body)}, heads {sum(p.numel() for p in heads)}", flush=True)

pool_dir = Path(os.environ["POOL_DIR"])
if os.environ.get("POOL_LIST"):
    files = [pool_dir / l.strip() for l in open(os.environ["POOL_LIST"]) if l.strip()]
else:
    files = sorted(pool_dir.glob("*.cif"))
random.Random(0).shuffle(files)
n_val = max(1, int(0.1 * len(files))); n_test = max(1, int(0.1 * len(files)))
val_f, test_f, train_f = files[:n_val], files[n_val:n_val + n_test], files[n_val + n_test:]
if os.environ.get("MIRROR_LIST"):  # v5: mirror-image D-peptides (made from TRAIN-split pool peptides only) are added to the training side AFTER the split, so validation and test are exactly those of run 2
    _mir = [Path(l.strip()) for l in open(os.environ["MIRROR_LIST"]) if l.strip()]; train_f = train_f + _mir; print("mirror-image D-peptides added to the pool training set:", len(_mir), flush=True)
import json, hashlib
pil = sorted(Path(os.environ["PILOT_DIR"]).glob("*.cif"))
GROUPS = json.load(open(os.environ["COMPLEX_GROUPS"])) if os.environ.get("COMPLEX_GROUPS") else {}
def gkey(p): e = p.name.split("_")[0].upper(); return GROUPS.get(e, e)
def is_val(p): return int(hashlib.sha1(("v4val" + gkey(p)).encode()).hexdigest(), 16) % 100 < int(os.environ.get("VAL_PCT", 12))
pil_val = [p for p in pil if is_val(p)]; pil_train = [p for p in pil if not is_val(p)]
assert not ({gkey(p) for p in pil_val} & {gkey(p) for p in pil_train}), "protein family on both sides"
print(f"pool: train {len(train_f)} val {len(val_f)} test {len(test_f)} | complexes: train {len(pil_train)} val {len(pil_val)}", flush=True)
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None)
SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
CHUNK = 250
def loader(fl, bs, shuffle, workers):
    ds = ConcatDataset([DesignProblemDataset(fl[i:i + CHUNK], transform=pipe) for i in range(0, len(fl), CHUNK)])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=partial(collate_batch, schema=SCHEMA), num_workers=workers, persistent_workers=workers > 0, pin_memory=False)
NW = int(os.environ.get("NUM_WORKERS", 4)); BATCH = int(os.environ.get("BATCH", 512)); PB = int(os.environ.get("PBATCH", 16))
tr_pool, va_pool = loader(train_f, BATCH, True, NW), loader(val_f, BATCH, False, 2)
tr_pil, va_pil = loader(pil_train, PB, True, 2), loader(pil_val, PB, False, 0)
FIRST_NEW = 176
import yaml
_AA20 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TWP TYR VAL".replace("TWP", "TRP").split())
_names = []
for _sec, _d in yaml.safe_load(open(os.environ["RFD4_VOCAB_YAML"]))["residues"].items(): _names += list(_d)
CANON = torch.zeros(len(_names) + 8, dtype=torch.bool, device=device)
for _i, _n in enumerate(_names):
    if _n in _AA20: CANON[_i] = True
print("canonical classes:", int(CANON.sum()), "of", len(_names), flush=True)
from atomworks.ml.conditions import C_SEQ
import rfproteina_encoding_ext as _X
from bond_loss import peptide_bond_loss, clash_loss
BOND_W = float(os.environ.get('BOND_W', 10.0)); BOND_TOL = float(os.environ.get('BOND_TOL', 0.04)); BOND_NOISE = float(os.environ.get('BOND_NOISE', 0.0))
CLASH_W = float(os.environ.get('CLASH_W', 0.0)); CLASH_D0 = float(os.environ.get('CLASH_D0', 2.5)); NOISE_SIGMA = float(os.environ.get('NOISE_SIGMA', 0.0)); CA_JIT = float(os.environ.get('CA_JIT', 0.0)); NOISY_BOND_W = float(os.environ.get('NOISY_BOND_W', 0.0)); CLIP = float(os.environ.get('CLIP', 5.0))
M_IDX = int(_X.UNIFIED_ATOM37_ENCODING.token_to_idx['<M>']); BLIND = os.environ.get('BLIND', '0') == '1'; STRUCT_W = float(os.environ.get('STRUCT_W', 0.5)); NCW = float(os.environ.get('NCW', 2.0)); CHI_W = float(os.environ.get('CHI_W', 0.15))
print('BLIND decoder training:', BLIND, 'mask index', M_IDX, flush=True)
def new_stats(): return dict(can_n=0, can_loss=0.0, can_ok=0, nc_n=0, nc_loss=0.0, nc_ok=0)
def add_stats(a, b):
    for k in a: a[k] += b[k]
def fmt_stats(prefix, st):
    return (f"   cats {prefix}: canonical loss {st['can_loss'] / max(st['can_n'], 1):.4f} acc {st['can_ok'] / max(st['can_n'], 1):.4f} n {st['can_n']} | "
            f"non-canonical loss {st['nc_loss'] / max(st['nc_n'], 1):.4f} acc {st['nc_ok'] / max(st['nc_n'], 1):.4f} n {st['nc_n']}")

_DF = [int(_X.UNIFIED_ATOM37_ENCODING.token_to_idx[k]) for k in 'DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN'.split()]
_CL = [int(_X.UNIFIED_ATOM37_ENCODING.token_to_idx[k]) for k in 'ALA ARG ASN ASP CYS GLN GLU HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()]
DBG = []
HC = dict(D_n=0, D_ok=0, L_n=0, L_ok=0, S=0.0, S_n=0, B_n=0, B_ok=0, BL=0.0, BL_n=0, BN_n=0, BN_ok=0, CL=0.0, CL_n=0, CP2=0, CC=0)
def _sv(c): return ((torch.cross(c[..., 0, :] - c[..., 1, :], c[..., 2, :] - c[..., 1, :], dim=-1)) * (c[..., 3, :] - c[..., 1, :])).sum(-1)
def batch_loss(tb, split_new=False):
    b = {k: v.to(device) for k, v in tb.items() if isinstance(v, torch.Tensor)}
    if BLIND and C_SEQ.full_name in b: b[C_SEQ.full_name] = torch.full_like(b[C_SEQ.full_name], M_IDX)   # identity withheld from the decoder
    b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
    with torch.no_grad(): b["z_latent"] = encoder(b)["z_latent"]
    b["ca_coors_nm"] = b["coords_nm"][..., 1, :]
    out = decoder(b); emit = out["emit_mask"]
    if emit.sum() == 0: return None
    lg = out["seq_logits"][emit]; tg = b["residue_type"][emit].long()
    ce_tok = F.cross_entropy(lg, tg, reduction="none"); ce = ce_tok.mean()
    am = emit[..., None] & b["atom37_mask"].bool()                      # supervised atoms
    sq = (((out["coors_nm"] - b["coords_nm"]) * 10.0) ** 2).sum(-1)     # squared error per atom, A^2
    tokw = 1.0 + (NCW - 1.0) * (~CANON[b["residue_type"].long()]).float()          # non-canonical tokens count NCW times
    wam = am.float() * tokw[..., None]
    struct = (sq * wam).sum() / wam.sum().clamp(min=1e-6)
    hasCB = emit & b["atom37_mask"][..., 3].bool()
    svp_, svt_ = _sv(out["coors_nm"]), _sv(b["coords_nm"])
    chi_tok = F.softplus(-torch.sign(svt_) * svp_ / 0.0025)                       # native |signed volume| is about 0.0025 nm^3
    chi = (chi_tok * tokw * hasCB.float()).sum() / (tokw * hasCB.float()).sum().clamp(min=1e-6)
    with torch.no_grad():
        has = emit & b["atom37_mask"][..., 3].bool(); sp, sn = _sv(out["coors_nm"]), _sv(b["coords_nm"]); rt = b["residue_type"].long()
        for nm_, ids in (("D", _DF), ("L", _CL)):
            mk = has & torch.isin(rt, torch.tensor(ids, device=rt.device)); HC[nm_ + "_n"] += int(mk.sum()); HC[nm_ + "_ok"] += int((torch.sign(sp[mk]) == torch.sign(sn[mk])).sum())
        HC["S"] += float(struct); HC["S_n"] += 1
    bl, bn, bok = peptide_bond_loss(out["coors_nm"], b["coords_nm"], am, tol=BOND_TOL)
    HC["B_n"] += bn; HC["B_ok"] += bok; HC["BL"] += float(bl); HC["BL_n"] += 1
    bl2 = 0.0
    if BOND_NOISE > 0 and decoder.training:
        b2 = dict(b); zl = b["z_latent"]; b2["z_latent"] = zl + BOND_NOISE * zl.std() * torch.randn_like(zl)
        out2 = decoder(b2); bl2, bn2, bok2 = peptide_bond_loss(out2["coors_nm"], b["coords_nm"], am, tol=BOND_TOL)
        HC["BN_n"] += bn2; HC["BN_ok"] += bok2
    cl, cp2, cc = clash_loss(out["coors_nm"], b["coords_nm"], b["atom37_mask"].bool(), emit, b["chain_iid"], d0=CLASH_D0)
    HC["CL"] += float(cl); HC["CL_n"] += 1; HC["CP2"] += cp2; HC["CC"] += cc
    nl = 0.0; cl2 = bl_n = torch.zeros(())
    if NOISE_SIGMA > 0 and decoder.training:
        b2 = dict(b); zl = b["z_latent"]; b2["z_latent"] = zl + NOISE_SIGMA * zl.std() * torch.randn_like(zl)
        b2["ca_coors_nm"] = b["ca_coors_nm"] + CA_JIT * torch.randn_like(b["ca_coors_nm"]) * emit[..., None].float()
        out2 = decoder(b2); cl2, _, _ = clash_loss(out2["coors_nm"], b["coords_nm"], b["atom37_mask"].bool(), out2["emit_mask"], b["chain_iid"], d0=CLASH_D0)
        bl_n, _, _ = peptide_bond_loss(out2["coors_nm"], b["coords_nm"], am, tol=BOND_TOL); nl = CLASH_W * cl2 + NOISY_BOND_W * bl_n
    DBG.append(dict(ce=float(ce), struct=float(struct), chi=float(chi), bond=float(bl), clash=float(cl), n_clash_pairs=cp2, noisy_clash=float(cl2), noisy_bond=float(bl_n), emit=int(emit.sum()), tokens=int(b['atom37_mask'].any(-1).sum()), ctrue_chain_n=int(b['chain_iid'].max())))
    ce_only = ce; ce = ce + STRUCT_W * struct + CHI_W * chi + BOND_W * (bl + bl2) + CLASH_W * cl + nl
    pred = lg.argmax(-1); can = CANON[tg]; nc = ~can; ok = pred == tg
    st = dict(can_n=int(can.sum()), can_loss=float(ce_tok[can].sum()), can_ok=int(ok[can].sum()), nc_n=int(nc.sum()), nc_loss=float(ce_tok[nc].sum()), nc_ok=int(ok[nc].sum()))
    if split_new:
        new = tg >= FIRST_NEW
        return ce, int(new.sum()), int(ok[new].sum()), int(ok.sum()), int(len(tg)), st
    return ce, st

@torch.no_grad()
def evaluate(ld, complexes=False):
    model.eval(); tot = n = 0; nn = nc = ac = at = 0; ST = new_stats()
    for tb in ld:
        r = batch_loss(tb, split_new=True)
        if r is None: continue
        ce, a, b_, c_, d_, st = r; tot += ce.item(); n += 1; nn += a; nc += b_; ac += c_; at += d_; add_stats(ST, st)
    return tot / max(n, 1), nc / max(nn, 1), nn, ac / max(at, 1), ST

LAM = float(os.environ.get("LAMBDA", 1.0)); EPOCHS = int(os.environ.get("EPOCHS", 3)); MAXS = int(os.environ.get("MAX_STEPS", 0))
SAVE = os.environ.get("SAVE_PATH", os.path.join(HERE, "best_decoder_v3.pt"))
pil_iter = itertools.cycle(tr_pil); best = float("inf"); step = 0; t0 = time.time(); PAT = int(os.environ.get("PATIENCE", 3)); bad = 0; MAXSEC = float(os.environ.get("MAX_SECONDS", 1e9))
for ep in range(EPOCHS):
    model.train(); encoder.eval(); s1 = s2 = nb = 0; TP = new_stats(); TC = new_stats()
    for tb in tr_pool:
        r1 = batch_loss(tb); r2 = batch_loss(next(pil_iter))
        if r1 is None or r2 is None: continue
        l1, st1 = r1; l2, st2 = r2; add_stats(TP, st1); add_stats(TC, st2)
        loss = l1 + LAM * l2
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(body + heads, CLIP); opt.step(); step += 1; s1 += l1.item(); s2 += l2.item(); nb += 1
        if step % 20 == 0 and len(DBG) >= 2 and os.environ.get('DEBUG_TERMS') == '1': print("   terms pool", {k: round(v, 4) for k, v in DBG[-2].items()}, "| complex", {k: round(v, 4) for k, v in DBG[-1].items()}, flush=True)
        DBG.clear()
        if step % 20 == 0: print(f"ep {ep} step {step} pool {l1.item():.4f} complex {l2.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if MAXS and step >= MAXS: break
    for _k in HC: HC[_k] = 0 if _k != "S" else 0.0
    vp, _, _, accp, VP = evaluate(va_pool); vc, new_acc, n_new, accc, VC = evaluate(va_pil)
    print(f"   val handedness (decoded vs true C-alpha sign): D-forms {HC['D_ok']}/{HC['D_n']} = {HC['D_ok'] / max(HC['D_n'], 1):.3f} | L-residues {HC['L_ok']}/{HC['L_n']} = {HC['L_ok'] / max(HC['L_n'], 1):.3f} | mean atom sq error (A^2) {HC['S'] / max(HC['S_n'], 1):.3f}", flush=True)
    print(f"   val peptide bonds (decoded atoms of true latents): {HC['B_ok']}/{HC['B_n']} in 1.2-1.5 A = {HC['B_ok'] / max(HC['B_n'], 1):.4f} | mean bond loss {HC['BL'] / max(HC['BL_n'], 1):.4f}", flush=True)
    print(f"   val peptide-receptor clash (true latents): pairs < 2 A {HC['CP2']} in {HC['CC']} complexes | mean clash loss {HC['CL'] / max(HC['CL_n'], 1):.4f}", flush=True)
    print(f"epoch {ep}: val pool loss {vp:.4f} acc {accp:.4f} | val complexes loss {vc:.4f} acc {accc:.4f} | new-class tokens {n_new} acc {new_acc:.4f} | {time.time() - t0:.0f}s", flush=True)
    score = vp + vc
    print(fmt_stats("train pool", TP), flush=True); print(fmt_stats("train complex", TC), flush=True)
    print(fmt_stats("val pool", VP), flush=True); print(fmt_stats("val complex", VC), flush=True)
    print(f"   train means: pool {s1 / max(nb, 1):.4f} complex {s2 / max(nb, 1):.4f} | gap val-train complex {vc - s2 / max(nb, 1):+.4f}", flush=True)
    if score < best - 1e-4:
        best = score; bad = 0; torch.save({k: v.detach().cpu() for k, v in decoder.state_dict().items()}, SAVE); print(f"saved best ({score:.4f})", flush=True)
    else:
        bad += 1; print(f"no improvement {bad}/{PAT}", flush=True)
        if bad >= PAT: print("EARLY STOP: held-out loss stopped improving", flush=True); break
    if time.time() - t0 > MAXSEC: print("STOP: time limit", flush=True); break
    if MAXS and step >= MAXS: break
print("done", flush=True)
