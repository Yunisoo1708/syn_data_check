"""DENOISE-THEN-DECODE, ALL-ATOM, PER NON-CANONICAL POSITION: does denoiser + decoder rebuild an NCAA residue accurately? For every real complex that holds an NCAA the latent is noised at a fixed time t (same seeded noise for every model), the denoiser (control = untuned, or + a LoRA adapter) predicts the clean latents and CA positions of the generated tokens (given tokens keep their true values, receptor and NCAA identities given as in the generation test), the decoder (run 8) turns them into atoms, and every generated non-canonical residue is compared with the native one:
  rmsd_A all-atom RMSD to the native residue | ca_err_A distance between the denoised and the true C-alpha | atoms_ok exactly the native atoms | chir_ok C-alpha handedness | bond_dev_A bond-length error inside the residue;  and the same for the decode of the TRUE latents (oracle: decoder alone).
usage: eval_dd_atoms.py NAME ADAPTER_or_none DECODER.pt OUT.json SET     (SET = real | ham; env TS=0.1,0.3,0.5,0.7,0.9 (t = 0 pure noise, 1 clean; BOTH the CA and the latent noise levels are fixed), NCLIST=ham list)"""
import os, sys, json, glob, hashlib, random, time
import numpy as np
T = os.path.expanduser("~/rfd4_trial"); sys.path.insert(0, f"{T}/patch"); sys.path.insert(0, f"{T}/code")
import _common
import torch, peft as pt
from pathlib import Path
import rfd4_proteina as rfd
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
name, adapter, dec_path, out, which = sys.argv[1:6]
files = sorted(glob.glob(f"{T}/real_test_cif/*.cif" if which == "real" else f"{T}/real_test_ham/*.cif"))
if os.environ.get("NCLIST"): keep = set(json.load(open(os.environ["NCLIST"]))); files = [f for f in files if os.path.basename(f) in keep]
if os.environ.get("SMOKE"): files = files[:int(os.environ["SMOKE"])]
TS = [float(x) for x in os.environ.get("TS", "0.1,0.3,0.5,0.7,0.9").split(",")]
dev = "cuda"; model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.train()
print("decoder:", model.autoencoder.decoder.load_state_dict(torch.load(dec_path, map_location=dev), strict=True), flush=True)
if adapter != "none": model.nn = pt.PeftModel.from_pretrained(model.nn, adapter, is_trainable=False)
model.nn.eval(); model.autoencoder.eval()
import rfproteina_encoding_ext as X
t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}; i2t = {v: k for k, v in t2i.items()}
NCs = set(map(str, list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"]))); NCt = torch.tensor(sorted(t2i[k] for k in NCs if k in t2i), device=dev)
DFn = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split()); GENn = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
mod = sys.modules[model.fm.__class__.__module__]; SUPM = mod.compute_flow_supervision_masks
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
def sv(c): n, ca, cc, cb = c[:, 0], c[:, 1], c[:, 2], c[:, 3]; return ((torch.cross(n - ca, cc - ca, dim=-1)) * (cb - ca)).sum(-1)
rows = []; t0 = time.time(); nerr = 0
with torch.no_grad():
    for k, f in enumerate(files):
        seed = int(hashlib.sha1(os.path.basename(f).encode()).hexdigest()[:8], 16)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {kk: v.to(dev) if isinstance(v, torch.Tensor) else v for kk, v in tb.items()}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b); model.fm.corrupt_batch(b)
            emit_m = mod.compute_flow_emit_masks(b); rt = b["residue_type"][0].long(); nat = b["coords_nm"][0]; am = b["atom37_mask"][0].bool()
            sup = SUPM(b)["local_latents"][0]; sel = torch.nonzero(sup & torch.isin(rt, NCt)).flatten().tolist()
            if not sel: continue
            zt = b["x_1"]["local_latents"]; ct = b["x_1"]["bb_ca"]
            for T_FIX in TS:
                for mm_ in ("bb_ca", "local_latents"):   # BOTH noise levels fixed to T_FIX (t = 0 pure noise, t = 1 clean): x_t = (1-t) x_0 + t x_1 on the generated tokens
                    tt = torch.full_like(b["t"][mm_], T_FIX); b["t"][mm_] = tt; tv = tt.view(-1, *([1] * (b["x_1"][mm_].dim() - 1)))
                    b["x_t"][mm_] = torch.where(emit_m[mm_][..., None], (1 - tv) * b["x_0"][mm_] + tv * b["x_1"][mm_], b["x_t"][mm_])
                o = model.call_nn(b, n_recycle=1); o = model.fm.nn_out_add_clean_sample_prediction(batch=b, nn_out=o)
                zh = o["local_latents"]["x_1"]; ch = o["bb_ca"]["x_1"]; supm = SUPM(b)["local_latents"]
                z_in = torch.where(supm[..., None], zh, zt); c_in = torch.where(supm[..., None], ch, ct)
                outs = {}
                for tag, (z, c) in (("dd", (z_in, c_in)), ("oracle", (zt, ct))):
                    d = dict(b); d["z_latent"] = z; d["ca_coors_nm"] = c; outs[tag] = model.autoencoder.decoder(d)
                for i in sel:
                    mk = am[i]; code = i2t.get(int(rt[i]), str(int(rt[i]))); row = dict(set=which, f=os.path.basename(f), pos=i, t=T_FIX, code=code, group="D-forms" if code in DFn else "general codes" if code in GENn else "other non-canonical", n_atoms=int(mk.sum()),
                                                                                    ca_err_A=float(torch.norm((ch[0, i] - ct[0, i]) * 10.0)))
                    for tag in ("dd", "oracle"):
                        dg = outs[tag]["coors_nm"][0][i]; row[f"rmsd_{tag}_A"] = float(torch.sqrt((((dg[mk] - nat[i][mk]) * 10.0) ** 2).sum(-1).mean()))
                        row[f"atoms_ok_{tag}"] = bool(int(outs[tag]["atom_mask"][0][i].sum()) == int(mk.sum()))
                        row[f"chir_ok_{tag}"] = bool(torch.sign(sv(dg[None])[0]) == torch.sign(sv(nat[i][None])[0])) if bool(mk[3]) else None
                        P = nat[i][mk] * 10.0; Q = dg[mk] * 10.0; dn = torch.cdist(P, P); dd = torch.cdist(Q, Q); bm = (dn < 1.9) & (dn > 0.5)
                        row[f"bond_dev_{tag}_A"] = float((dd - dn).abs()[bm].mean()) if bool(bm.any()) else None
                    rows.append(row)
        except Exception as ex:
            nerr += 1; print("ERR", os.path.basename(f), str(ex)[:120], flush=True)
        if k % 40 == 0: print(name, which, k, "of", len(files), f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(name=name, adapter=adapter, decoder=dec_path, set=which, ts=TS, errors=nerr, rows=rows), open(out, "w"))
for T_FIX in TS:
    r = [x for x in rows if x["t"] == T_FIX]
    if r: print("DONE", name, which, "t", T_FIX, "positions", len(r), "median rmsd dd %.2f oracle %.2f | ca_err median %.2f" % (np.median([x["rmsd_dd_A"] for x in r]), np.median([x["rmsd_oracle_A"] for x in r]), np.median([x["ca_err_A"] for x in r])), flush=True)
