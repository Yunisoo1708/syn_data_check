"""ATOM version: also the all-atom RMSD (A) of the decoded atoms against the native atoms per residue group (D-forms, 12 general codes, other non-canonical, canonical). Original: Diagnostic: does the DECODER reproduce D-form handedness from the TRUE latents? For the real complexes: encode the native complex (encoder, frozen), decode the true latents with a chosen decoder, and measure the C-alpha handedness (signed volume of N, C, CB around CA, atom37 slots 0, 2, 3 around 1) of the decoded atoms at D-form and canonical L positions, against the native coordinates.
usage: MODE=given|hidden diag_decode_chir.py DECODER.pt OUT.json SET    (SET = real | ham)"""
import os, sys, json, glob, hashlib, random
import numpy as np
T = os.path.expanduser("~/rfd4_trial"); sys.path.insert(0, f"{T}/patch"); sys.path.insert(0, f"{T}/code")
sys.path.insert(0, os.path.expanduser("~/rfd4_trial/code_extra")); import _common
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
from bond_loss import peptide_bond_loss, clash_loss
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
dec = model.autoencoder.decoder; torch.manual_seed(0)
acc = {}
for f in files[:24]:
    tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
    b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tb.items()}
    b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
    with torch.no_grad(): b["z_latent"] = model.autoencoder.encoder(b)["z_latent"]
    b["ca_coors_nm"] = b["coords_nm"][..., 1, :]; am0 = b["atom37_mask"].bool()
    for mode in ("eval", "train"):
        dec.train(mode == "train")
        for sg in (0.0, 0.15):
            with torch.no_grad():
                b2 = dict(b); zl = b["z_latent"]; b2["z_latent"] = zl + sg * zl.std() * torch.randn_like(zl)
                out = dec(b2); em = out["emit_mask"]; am = em[..., None] & am0
                bl, _, _ = peptide_bond_loss(out["coors_nm"], b["coords_nm"], am); cl, n2, _ = clash_loss(out["coors_nm"], b["coords_nm"], am0, em, b["chain_iid"])
                sq = ((((out["coors_nm"] - b["coords_nm"]) * 10.0) ** 2).sum(-1) * am).sum() / am.sum()
            k = (mode, sg); a = acc.setdefault(k, [0, 0, 0, 0]); a[0] += float(bl); a[1] += float(cl); a[2] += float(sq); a[3] += 1
for k, a in acc.items(): print("TERMS", k, "bond %.4f clash %.4f atom_sq %.3f" % (a[0] / a[3], a[1] / a[3], a[2] / a[3]))
