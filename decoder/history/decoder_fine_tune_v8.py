"""COORDINATE+CHIRALITY decoder (run 6): v7 + the coordinate loss of non-canonical tokens weighted NCW times and a chirality loss (softplus on the signed volume of N, C, CB around CA against the true sign, env CHI_W) on every token, so D-forms keep their handedness. Env INIT = decoder to start from.
COORDINATE decoder (run 5): v6 + an atom-coordinate loss (decoded atoms against the true atoms, A^2, no alignment; env STRUCT_W weight) added to the residue-type CE, so the atom output is trained on D-forms too; env BLIND=0 keeps the identity given (as run 3). Tracks the C-alpha handedness of decoded D-forms and L-residues in validation.
BLIND decoder (run 4): v5 + the residue identity is HIDDEN from the decoder (env BLIND=1: the conditioning input C_SEQ / cond_residue_type is set to <M> for every token, in training and validation), so the decoder must read the residue type from the latent code; env INIT = decoder state dict to start from (run 3).
Whole-decoder fine-tune on the 189-class vocabulary (176 + 13 D-forms), encoder frozen.
Data: (1) the 46k strict-passed peptide pool (80/10/10 split by file, seed 0, as in decoder_fine_tune_v2.py), (2) the synthetic peptide-protein complexes (pilot_train_189, the 13 new D-forms live only here).
Each step = one pool batch + one complex batch; loss = CE(pool) + LAMBDA * CE(complexes), so the rare new classes are not drowned (13 classes are about 0.1% of the tokens otherwise).
Env: RFD_BUNDLE (189 bundle), RFD4_VOCAB_YAML (v3), POOL_DIR, POOL_LIST (optional file with names), PILOT_DIR, BATCH (pool), PBATCH (complexes), EPOCHS, LAMBDA, LR_BODY, LR_HEAD, MAX_STEPS, SAVE_PATH, NUM_WORKERS.
Saves the best decoder state dict (whole decoder, 189 rows) judged on pool-val + complex-val loss."""
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
M_IDX = int(_X.UNIFIED_ATOM37_ENCODING.token_to_idx['<M>']); BLIND = os.environ.get('BLIND', '0') == '1'; STRUCT_W = float(os.environ.get('STRUCT_W', 0.25)); NCW = float(os.environ.get('NCW', 3.0)); CHI_W = float(os.environ.get('CHI_W', 0.5))
print('BLIND decoder training:', BLIND, 'mask index', M_IDX, flush=True)
def new_stats(): return dict(can_n=0, can_loss=0.0, can_ok=0, nc_n=0, nc_loss=0.0, nc_ok=0)
def add_stats(a, b):
    for k in a: a[k] += b[k]
def fmt_stats(prefix, st):
    return (f"   cats {prefix}: canonical loss {st['can_loss'] / max(st['can_n'], 1):.4f} acc {st['can_ok'] / max(st['can_n'], 1):.4f} n {st['can_n']} | "
            f"non-canonical loss {st['nc_loss'] / max(st['nc_n'], 1):.4f} acc {st['nc_ok'] / max(st['nc_n'], 1):.4f} n {st['nc_n']}")

_DF = [int(_X.UNIFIED_ATOM37_ENCODING.token_to_idx[k]) for k in 'DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN'.split()]
_CL = [int(_X.UNIFIED_ATOM37_ENCODING.token_to_idx[k]) for k in 'ALA ARG ASN ASP CYS GLN GLU HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()]
HC = dict(D_n=0, D_ok=0, L_n=0, L_ok=0, S=0.0, S_n=0)
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
    ce_only = ce; ce = ce + STRUCT_W * struct + CHI_W * chi
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
        opt.zero_grad(); loss.backward(); opt.step(); step += 1; s1 += l1.item(); s2 += l2.item(); nb += 1
        if step % 20 == 0: print(f"ep {ep} step {step} pool {l1.item():.4f} complex {l2.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if MAXS and step >= MAXS: break
    for _k in HC: HC[_k] = 0 if _k != "S" else 0.0
    vp, _, _, accp, VP = evaluate(va_pool); vc, new_acc, n_new, accc, VC = evaluate(va_pil)
    print(f"   val handedness (decoded vs true C-alpha sign): D-forms {HC['D_ok']}/{HC['D_n']} = {HC['D_ok'] / max(HC['D_n'], 1):.3f} | L-residues {HC['L_ok']}/{HC['L_n']} = {HC['L_ok'] / max(HC['L_n'], 1):.3f} | mean atom sq error (A^2) {HC['S'] / max(HC['S_n'], 1):.3f}", flush=True)
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
