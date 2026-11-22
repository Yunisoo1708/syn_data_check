"""Whole-decoder fine-tune on the 189-class vocabulary (176 + 13 D-forms), encoder frozen.
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
pil = sorted(Path(os.environ["PILOT_DIR"]).glob("*.cif")); random.Random(1).shuffle(pil)
pv = max(1, int(0.1 * len(pil))); pil_val, pil_train = pil[:pv], pil[pv:]
print(f"pool: train {len(train_f)} val {len(val_f)} test {len(test_f)} | complexes: train {len(pil_train)} val {len(pil_val)}", flush=True)
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None)
SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
CHUNK = 250
def loader(fl, bs, shuffle, workers):
    ds = ConcatDataset([DesignProblemDataset(fl[i:i + CHUNK], transform=pipe) for i in range(0, len(fl), CHUNK)])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=partial(collate_batch, schema=SCHEMA), num_workers=workers, persistent_workers=workers > 0, pin_memory=device == "cuda")
NW = int(os.environ.get("NUM_WORKERS", 4)); BATCH = int(os.environ.get("BATCH", 512)); PB = int(os.environ.get("PBATCH", 16))
tr_pool, va_pool = loader(train_f, BATCH, True, NW), loader(val_f, BATCH, False, 2)
tr_pil, va_pil = loader(pil_train, PB, True, 2), loader(pil_val, PB, False, 0)
FIRST_NEW = 176

def batch_loss(tb, split_new=False):
    b = {k: v.to(device) for k, v in tb.items() if isinstance(v, torch.Tensor)}
    b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
    with torch.no_grad(): b["z_latent"] = encoder(b)["z_latent"]
    b["ca_coors_nm"] = b["coords_nm"][..., 1, :]
    out = decoder(b); emit = out["emit_mask"]
    if emit.sum() == 0: return None
    lg = out["seq_logits"][emit]; tg = b["residue_type"][emit].long()
    ce = F.cross_entropy(lg, tg)
    if split_new:
        new = tg >= FIRST_NEW
        return ce, int(new.sum()), int((lg.argmax(-1)[new] == tg[new]).sum()), int((lg.argmax(-1) == tg).sum()), int(len(tg))
    return ce

@torch.no_grad()
def evaluate(ld, complexes=False):
    model.eval(); tot = n = 0; nn = nc = ac = at = 0
    for tb in ld:
        r = batch_loss(tb, split_new=True)
        if r is None: continue
        ce, a, b_, c_, d_ = r; tot += ce.item(); n += 1; nn += a; nc += b_; ac += c_; at += d_
    return tot / max(n, 1), nc / max(nn, 1), nn, ac / max(at, 1)

LAM = float(os.environ.get("LAMBDA", 1.0)); EPOCHS = int(os.environ.get("EPOCHS", 3)); MAXS = int(os.environ.get("MAX_STEPS", 0))
SAVE = os.environ.get("SAVE_PATH", os.path.join(HERE, "best_decoder_v3.pt"))
pil_iter = itertools.cycle(tr_pil); best = float("inf"); step = 0; t0 = time.time()
for ep in range(EPOCHS):
    model.train(); encoder.eval()
    for tb in tr_pool:
        l1 = batch_loss(tb); l2 = batch_loss(next(pil_iter))
        if l1 is None or l2 is None: continue
        loss = l1 + LAM * l2
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 20 == 0: print(f"ep {ep} step {step} pool {l1.item():.4f} complex {l2.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if MAXS and step >= MAXS: break
    vp, _, _, accp = evaluate(va_pool); vc, new_acc, n_new, accc = evaluate(va_pil)
    print(f"epoch {ep}: val pool loss {vp:.4f} acc {accp:.4f} | val complexes loss {vc:.4f} acc {accc:.4f} | new-class tokens {n_new} acc {new_acc:.4f} | {time.time() - t0:.0f}s", flush=True)
    score = vp + vc
    if score < best:
        best = score; torch.save({k: v.detach().cpu() for k, v in decoder.state_dict().items()}, SAVE); print(f"saved best ({score:.4f})", flush=True)
    if MAXS and step >= MAXS: break
print("done", flush=True)
