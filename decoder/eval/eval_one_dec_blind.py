"""BLIND variant (env MODE=v1|v2): the identity of the generated residues is WITHHELD. v1: the denoiser and decoder get <M> instead of the residue identity at the generated tokens (token type kept, so the decoder only chooses among non-canonical classes); v2: the decoder's allowed-class mask is switched off (it may name any of the 189 classes, L-forms included; the token type is still given). Extra outputs: share of D-form positions decoded as their L-form partner, share decoded as any non-canonical class.
Denoise-then-decode test with ONE fixed decoder for every model: the denoiser (control = intact original RFD4 flow network, or the base + a saved LoRA) predicts the clean latents and CA positions from a noisy copy of a complex
(same per-complex seed and noise for every model), the SAME retrained decoder (--decoder, 189 classes) reads them back into residue types. Reported per residue code (non-canonical codes of the 189-class vocabulary; residues outside the vocabulary
are masked) and for all canonical positions: top-1 accuracy and cross-entropy of the decoder's residue naming, plus the same decode of the TRUE latents (the oracle, independent of the denoiser) and the latent/CA error of the prediction.
usage: eval_one_dec.py NAME ADAPTER_DIR_or_none OUT.json SET DECODER.pt   (SET = heldout | real | ham; SMOKE=3 limits to 3 complexes)"""
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
name, adapter, out, which, dec_path = sys.argv[1:6]
T = os.path.expanduser("~/rfd4_trial")
if which == "real": files = sorted(glob.glob(f"{T}/real_test_cif/*.cif"))
elif which == "ham": files = sorted(glob.glob(f"{T}/real_test_ham/*.cif"))
else:
    allf = sorted(glob.glob(f"{T}/train_all_cif_clean/*.cif")); G = json.load(open(f"{T}/complex_groups.json"))
    def gkey(f): e = os.path.basename(f).split("_")[0].upper(); return G.get(e, e)
    files = [f for f in allf if int(hashlib.sha1(("fmval" + gkey(f)).encode()).hexdigest(), 16) % 100 < 12]
if os.environ.get("NCLIST"): keep = set(json.load(open(os.environ["NCLIST"]))); files = [f for f in files if os.path.basename(f) in keep]  # complexes with at least one scored non-canonical position
if os.environ.get("SMOKE"): files = files[:int(os.environ["SMOKE"])]
dev = "cuda"
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.train()
print("decoder:", model.autoencoder.decoder.load_state_dict(torch.load(dec_path, map_location=dev), strict=True), flush=True)  # the SAME retrained decoder for every model
if adapter != "none": model.nn = pt.PeftModel.from_pretrained(model.nn, adapter, is_trainable=False)
model.nn.eval(); model.autoencoder.eval()
if os.environ.get("MODE", "v2") == "v2": model.autoencoder.decoder.allowed_residue_types.fill_(True)   # v2: the decoder may name ANY of the 189 classes (canonical L-forms included), not only the classes of its token type
import rfproteina_encoding_ext as X
from atomworks.ml.conditions import C_SEQ
from rfproteina.datapipes.schema.proteina import TokenType
MODE = os.environ.get('MODE', 'v2')
LPAIR = {}
for _p in 'DAL=ALA DAR=ARG DSG=ASN DAS=ASP DCY=CYS DGN=GLN DGL=GLU DHI=HIS DIL=ILE DLE=LEU DLY=LYS MED=MET DPN=PHE DPR=PRO DSN=SER DTH=THR DTR=TRP DTY=TYR DVA=VAL'.split(): LPAIR[_p.split('=')[0]] = _p.split('=')[1]
t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}; i2t = {v: k for k, v in t2i.items()}
ncset = set(map(str, list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"])))
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
nc_idx = {t2i[k] for k in ncset if k in t2i}; can_idx = {t2i[k] for k in CAN if k in t2i}
mod = sys.modules[model.fm.__class__.__module__]; SUPM = mod.compute_flow_supervision_masks
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
            if os.environ.get("SMOKE") and k == 0:
                for m in ("bb_ca", "local_latents"):
                    tt = b["t"][m].view(-1, *([1] * (b["x_1"][m].dim() - 1))); xi = (1 - tt) * b["x_0"][m] + tt * b["x_1"][m]
                    print("interp check", m, "t", float(b["t"][m].mean()), "max|x_t - ((1-t)x_0 + t x_1)| =", float((b["x_t"][m] - xi).abs().max()), "(emit-masked tokens only matter)", flush=True)
            emit_m = mod.compute_flow_emit_masks(b)
            sup0 = SUPM(b)["local_latents"].clone()
            _M = t2i["<M>"]; _key = C_SEQ.full_name
            if _key in b: b[_key] = torch.where(sup0, torch.full_like(b[_key], _M), b[_key])   # identity withheld at the generated tokens
            tok_true = b["token_type"].clone() if "token_type" in b else None
            for T_FIX in ([None] if not os.environ.get("TS") else [float(x) for x in os.environ["TS"].split(",")]):
              if T_FIX is not None:
                for m in ("local_latents",):  # the latent time is fixed (x_t is exactly (1-t) x_0 + t x_1 there); the CA keeps its own seeded draw, identical for every model
                    tt = torch.full_like(b["t"][m], T_FIX); b["t"][m] = tt; tv = tt.view(-1, *([1] * (b["x_1"][m].dim() - 1)))
                    b["x_t"][m] = torch.where(emit_m[m][..., None], (1 - tv) * b["x_0"][m] + tv * b["x_1"][m], b["x_t"][m])
              o = model.call_nn(b, n_recycle=1); o = model.fm.nn_out_add_clean_sample_prediction(batch=b, nn_out=o)
              zh = o["local_latents"]["x_1"]; ch = o["bb_ca"]["x_1"]                      # the denoiser's guess of the clean latents and CA positions (nm)
              sup = sup0; rt = b["residue_type"].long()
              zt = b["x_1"]["local_latents"]; ct = b["x_1"]["bb_ca"]                        # the true clean latents / CA (encoder output)
              def decode(z, c):
                  d = dict(b); d["z_latent"] = z; d["ca_coors_nm"] = c
                  # v2: token type UNKNOWN at the generated tokens = the decoder may choose among all 189 classes; v1: true token type (non-canonical classes only)
                  dout = model.autoencoder.decoder(d); return dout["seq_logits"], dout["emit_mask"]
              z_in = torch.where(sup[..., None], zh, zt); c_in = torch.where(sup[..., None], ch, ct)   # only the generated tokens use the guess, given tokens keep their true values
              lg, em = decode(z_in, c_in); lg0, em0 = decode(zt, ct)
              r = {"_t": {m: float(b["t"][m].mean()) for m in ("bb_ca", "local_latents")}, "codes": {}}
              ok = (sup & em)[0]; ok0 = (sup & em0)[0]
              ce = F.cross_entropy(lg[0], rt[0], reduction="none"); ce0 = F.cross_entropy(lg0[0], rt[0], reduction="none")
              pr = lg[0].argmax(-1); pr0 = lg0[0].argmax(-1)
              lat_err = ((zh - zt) ** 2).sum(-1)[0]; ca_err = ((ch - ct) ** 2).sum(-1)[0]
              groups = {}
              for c in sorted(set(rt[0][ok].tolist())):
                  if c in nc_idx: groups[i2t.get(c, str(c))] = (rt[0] == c)
              groups["_NC_ALL"] = torch.zeros_like(ok)
              for c in nc_idx: groups["_NC_ALL"] |= (rt[0] == c)
              groups["_CANONICAL"] = torch.zeros_like(ok)
              for c in can_idx: groups["_CANONICAL"] |= (rt[0] == c)
              for nm, m in groups.items():
                  m1 = m & ok; m0 = m & ok0; n_c = int(m1.sum().item())
                  if n_c == 0: continue
                  r["codes"][nm] = {"n": n_c, "acc": float((pr[m1] == rt[0][m1]).float().mean()), "ce": float(ce[m1].mean()), "acc_oracle": float((pr0[m1] == rt[0][m1]).float().mean()), "ce_oracle": float(ce0[m1].mean()),
                                    "lat_err": float(lat_err[m1].mean()), "ca_err": float(ca_err[m1].mean()),
                                    "pred_Lpair": (float((pr[m1] == t2i[LPAIR[nm]]).float().mean()) if nm in LPAIR and LPAIR[nm] in t2i else None), "pred_nc": float(torch.isin(pr[m1], torch.tensor(sorted(nc_idx), device=pr.device)).float().mean()),
                                    "pred_Lpair_oracle": (float((pr0[m1] == t2i[LPAIR[nm]]).float().mean()) if nm in LPAIR and LPAIR[nm] in t2i else None)}
              res[os.path.basename(f) + ("" if T_FIX is None else f"@{T_FIX}")] = r
              if os.environ.get("SMOKE"): print(os.path.basename(f), {kk: (v if kk != "codes" else {c: {a: (round(x, 3) if x is not None else None) for a, x in e.items()} for c, e in list(v.items())[:3]}) for kk, v in r.items()}, flush=True)
        except Exception as ex:
            res[os.path.basename(f)] = {"error": str(ex)[:300]}
            if os.environ.get("SMOKE"): import traceback; traceback.print_exc()
        if k % 40 == 0: print(name, which, k, "of", len(files), f"{time.time() - t0:.0f}s", flush=True)
json.dump({"name": name, "adapter": adapter, "set": which, "decoder": dec_path, "results": res}, open(out, "w"))
print(name, which, "done", len(files), "complexes;", sum(1 for v in res.values() if "error" in v), "errors", flush=True)
