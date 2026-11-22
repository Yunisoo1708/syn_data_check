"""ATOM version: also the all-atom RMSD (A) of the decoded atoms against the native atoms per residue group (D-forms, 12 general codes, other non-canonical, canonical). Original: Diagnostic: does the DECODER reproduce D-form handedness from the TRUE latents? For the real complexes: encode the native complex (encoder, frozen), decode the true latents with a chosen decoder, and measure the C-alpha handedness (signed volume of N, C, CB around CA, atom37 slots 0, 2, 3 around 1) of the decoded atoms at D-form and canonical L positions, against the native coordinates.
usage: MODE=given|hidden diag_decode_chir.py DECODER.pt OUT.json SET    (SET = real | ham)"""
import os, sys, json, glob, hashlib, random
import numpy as np
T = os.path.expanduser("~/rfd4_trial"); sys.path.insert(0, f"{T}/patch"); sys.path.insert(0, f"{T}/code")
import _common
import torch, rfd4_proteina as rfd
from pathlib import Path
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch
from atomworks.ml.conditions import C_SEQ
import rfproteina_encoding_ext as X
dec_path, out, which = sys.argv[1:4]; MODE = os.environ.get("MODE", "given")
files = sorted(glob.glob(f"{T}/real_test_cif/*.cif" if which == "real" else f"{T}/real_test_ham/*.cif"))
if which == "ham" and os.path.exists(f"{T}/out/eval_dec/ham_nclist.json"): keep = set(json.load(open(f"{T}/out/eval_dec/ham_nclist.json"))); files = [f for f in files if os.path.basename(f) in keep]
dev = "cuda"; model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.eval()
print("decoder:", model.autoencoder.decoder.load_state_dict(torch.load(dec_path, map_location=dev), strict=True), flush=True); model.autoencoder.eval()
t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}
DF = [t2i[k] for k in "DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split() if k in t2i]
CANL = [t2i[k] for k in "ALA ARG ASN ASP CYS GLN GLU HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()]
DFt, CANt = torch.tensor(DF, device=dev), torch.tensor(CANL, device=dev)
GEN = [t2i[k] for k in "23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split() if k in t2i]; GENt = torch.tensor(GEN, device=dev)
NCs = set(map(str, list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"]))); NCt = torch.tensor(sorted(t2i[k] for k in NCs if k in t2i), device=dev)
ACC = {g: [0.0, 0] for g in ("D-forms", "general codes", "other non-canonical", "canonical")}
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
def sv(c):   # signed volume N, C, CB around CA; c = [n, 37, 3]
    n, ca, cc, cb = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return ((torch.cross(n - ca, cc - ca, dim=-1)) * (cb - ca)).sum(-1)
rows = []
with torch.no_grad():
    for f in files:
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tb.items()}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b)
            if MODE == "hidden" and C_SEQ.full_name in b: b[C_SEQ.full_name] = torch.full_like(b[C_SEQ.full_name], t2i["<M>"])
            d = dict(b); d["z_latent"] = b["x_1"]["local_latents"]; d["ca_coors_nm"] = b["x_1"]["bb_ca"]
            dout = model.autoencoder.decoder(d); em = dout["emit_mask"][0]; rt = b["residue_type"][0].long()
            dec = dout["coors_nm"][0]; nat = b["coords_nm"][0]
            sd, sn = sv(dec), sv(nat)
            Lm = em & torch.isin(rt, CANt) & b["atom37_mask"][0][:, 3].bool(); Dm = em & torch.isin(rt, DFt) & b["atom37_mask"][0][:, 3].bool()
            if Dm.sum() == 0 or Lm.sum() < 3: continue
            ref = torch.sign(sn[Lm].median())                     # native L reference sign
            sq_ = (((dec - nat) * 10.0) ** 2).sum(-1); am_ = b["atom37_mask"][0].bool() & em[:, None]
            for g, mk in (("D-forms", torch.isin(rt, DFt)), ("general codes", torch.isin(rt, GENt)), ("other non-canonical", torch.isin(rt, NCt) & ~torch.isin(rt, DFt) & ~torch.isin(rt, GENt)), ("canonical", torch.isin(rt, CANt))):
                a_ = am_ & mk[:, None]; ACC[g][0] += float((sq_ * a_).sum()); ACC[g][1] += int(a_.sum())
            rows.append(dict(f=os.path.basename(f), nD=int(Dm.sum()), D_dec_ok=int((torch.sign(sd[Dm]) == -ref).sum()), D_nat_ok=int((torch.sign(sn[Dm]) == -ref).sum()), nL=int(Lm.sum()), L_dec_ok=int((torch.sign(sd[Lm]) == ref).sum()), D_dec_mag=float(sd[Dm].abs().mean())))
        except Exception as ex:
            rows.append(dict(f=os.path.basename(f), error=str(ex)[:150]))
ok = [r for r in rows if "error" not in r]
nD = sum(r["nD"] for r in ok); nL = sum(r["nL"] for r in ok)
summ = dict(decoder=dec_path, mode=MODE, set=which, complexes=len(ok), errors=len(rows) - len(ok), nD=nD, D_decoded_right=sum(r["D_dec_ok"] for r in ok) / max(nD, 1), D_native_right=sum(r["D_nat_ok"] for r in ok) / max(nD, 1), nL=nL, L_decoded_right=sum(r["L_dec_ok"] for r in ok) / max(nL, 1), D_decoded_mag=float(np.mean([r["D_dec_mag"] for r in ok])) if ok else None)
summ['atom_rmsd_A'] = {g: (float(np.sqrt(v[0] / v[1])) if v[1] else None) for g, v in ACC.items()}; summ['atom_counts'] = {g: v[1] for g, v in ACC.items()}
json.dump(dict(summary=summ, rows=rows), open(out, "w"), indent=1); print("SUMMARY", json.dumps(summ), flush=True)
