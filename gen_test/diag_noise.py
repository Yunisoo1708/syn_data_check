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
dec_path, out, which = sys.argv[1:4]; MODE = "given"; SIGMAS = [float(x) for x in os.environ.get("SIGMAS", "0,0.02,0.05,0.1,0.2").split(",")]; CAJ = float(os.environ.get("CAJ", 0.0))
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
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
torch.manual_seed(0)
R = {s: dict(bonds=0, bok=0, clash_designs=0, n=0, sq=0.0, na=0, pairs_near=0) for s in SIGMAS}; info = []
with torch.no_grad():
    for f in files:
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tb.items()}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b)
            z0 = b["x_1"]["local_latents"]; ca0 = b["x_1"]["bb_ca"]; nat = b["coords_nm"][0]; am = b["atom37_mask"][0].bool()
            for sg in SIGMAS:
                d = dict(b); d["z_latent"] = z0 + sg * z0.std() * torch.randn_like(z0); d["ca_coors_nm"] = ca0 + CAJ * torch.randn_like(ca0)
                dout = model.autoencoder.decoder(d); em = dout["emit_mask"][0]; dec = dout["coors_nm"][0]
                if sg == SIGMAS[0]: info.append((os.path.basename(f), int(em.sum()), int(am.any(-1).sum())))
                ch = b["chain_iid"][0].long(); tok = torch.arange(len(ch), device=dev)[:, None].expand(-1, 37); chm = ch[tok]; pe = em[:, None] & am
                P = dec[pe] * 10; Rr = nat[am] * 10; diff = chm[pe][:, None] != chm[am][None]
                if len(P) and bool(diff.any()):
                    dd_ = torch.cdist(P, Rr); dm = torch.where(diff, dd_, torch.full_like(dd_, 99.0)).min(); R[sg]["clash_designs"] += int(dm < 2.0); R[sg]["n"] += 1; R[sg]["pairs_near"] += int(((dd_ < 2.5) & diff).sum())
                C_ = dec[:-1, 2] * 10; N_ = dec[1:, 0] * 10; dn_ = (nat[1:, 0] - nat[:-1, 2]).norm(dim=-1) * 10; mk = em[:-1] & em[1:] & am[:-1, 2] & am[1:, 0] & (dn_ > 1.0) & (dn_ < 1.8)
                dd = (N_ - C_).norm(dim=-1)[mk]; R[sg]["bonds"] += int(mk.sum()); R[sg]["bok"] += int(((dd >= 1.2) & (dd <= 1.5)).sum())
                sq = (((dec - nat) * 10.0) ** 2).sum(-1); a2 = am & em[:, None]; R[sg]["sq"] += float((sq * a2).sum()); R[sg]["na"] += int(a2.sum())
        except Exception as ex:
            print("ERR", os.path.basename(f), str(ex)[:120])
print("emit tokens / all tokens (first 5 complexes):", info[:5])
out_ = {str(s): dict(sigma=s, ca_jitter_nm=CAJ, complexes=v["n"], clash_designs=v["clash_designs"] / max(v["n"], 1), pairs_below_2p5A=v["pairs_near"], bond_ok=v["bok"] / max(v["bonds"], 1), atom_rmsd_A=(v["sq"] / max(v["na"], 1)) ** 0.5) for s, v in R.items()}
json.dump(out_, open(out, "w"), indent=1)
for k, v in out_.items(): print("SIGMA", k, {a: round(b, 3) for a, b in v.items() if a not in ("sigma",)})
