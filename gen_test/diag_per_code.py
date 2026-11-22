"""PER NON-CANONICAL POSITION test of a DECODER on the TRUE latents of the real complexes (no denoiser): for every non-canonical residue of every real complex (103 real + the 365 extra real complexes that hold one), encode the native complex (frozen encoder), decode the true latent, and record per position:
  rmsd_A      all-atom RMSD of the decoded residue against the native residue (A, same frame, atoms present in the native residue)
  bond_dev_A  mean |decoded - native| length of the covalent bonds inside the residue (native heavy-atom pairs closer than 1.9 A): chemical sanity of the decoded residue
  atoms_ok    the decoder emits the native number of heavy atoms for the residue (IDENTITY GIVEN)
  chir_ok     for residues with a C-beta: handedness (signed volume N, C, CB around CA) equals the native sign
  name_ok     residue identity HIDDEN (conditioning set to <M>): the decoder names the residue correctly (argmax)
usage: diag_per_code.py DECODER.pt|bundle OUT.json     (bundle = the untouched bundle_189 decoder = vanilla)"""
import os, sys, json, glob
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
dec_path, out = sys.argv[1:3]
real = sorted(glob.glob(f"{T}/real_test_cif/*.cif")); ham = sorted(glob.glob(f"{T}/real_test_ham/*.cif"))
if os.path.exists(f"{T}/out/eval_dec/ham_nclist.json"):
    keep = set(json.load(open(f"{T}/out/eval_dec/ham_nclist.json"))); ham = [f for f in ham if os.path.basename(f) in keep]
files = [("real", f) for f in real] + [("ham", f) for f in ham]
dev = "cuda"; model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.eval()
print("decoder:", "BUNDLE (vanilla)" if dec_path == "bundle" else model.autoencoder.decoder.load_state_dict(torch.load(dec_path, map_location=dev), strict=True), flush=True); model.autoencoder.eval()
t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}; i2t = {v: k for k, v in t2i.items()}
DFn = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split()); GENn = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
NCs = set(map(str, list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"]))); NCt = torch.tensor(sorted(t2i[k] for k in NCs if k in t2i), device=dev)
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
def sv(c):
    n, ca, cc, cb = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return ((torch.cross(n - ca, cc - ca, dim=-1)) * (cb - ca)).sum(-1)
rows = []; nerr = 0
with torch.no_grad():
    for which, f in files:
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tb.items()}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b)
            rt = b["residue_type"][0].long(); sel = torch.nonzero(torch.isin(rt, NCt)).flatten().tolist()
            if not sel: continue
            res = {}
            for mode in ("given", "hidden"):
                bb = dict(b)
                if mode == "hidden" and C_SEQ.full_name in bb: bb[C_SEQ.full_name] = torch.full_like(bb[C_SEQ.full_name], t2i["<M>"])
                d = dict(bb); d["z_latent"] = b["x_1"]["local_latents"]; d["ca_coors_nm"] = b["x_1"]["bb_ca"]; res[mode] = model.autoencoder.decoder(d)
            nat = b["coords_nm"][0]; am = b["atom37_mask"][0].bool()
            for i in sel:
                if not bool(res["given"]["emit_mask"][0][i]): continue
                code = i2t.get(int(rt[i]), str(int(rt[i]))); dg = res["given"]["coors_nm"][0][i]; mk = am[i]
                diff = ((dg[mk] - nat[i][mk]) * 10.0); rmsd = float(torch.sqrt((diff ** 2).sum(-1).mean()))
                P = nat[i][mk] * 10.0; Q = dg[mk] * 10.0; dn = torch.cdist(P, P); dd = torch.cdist(Q, Q); bm = (dn < 1.9) & (dn > 0.5)
                bond_dev = float((dd - dn).abs()[bm].mean()) if bool(bm.any()) else float("nan")
                atoms_ok = int(res["given"]["atom_mask"][0][i].sum()) == int(mk.sum())
                chir = None
                if bool(mk[3]): chir = bool(torch.sign(sv(dg[None])[0]) == torch.sign(sv(nat[i][None])[0]))
                name_ok = int(res["hidden"]["aatype_max"][0][i]) == int(rt[i])
                rows.append(dict(set=which, f=os.path.basename(f), pos=i, code=code, group="D-forms" if code in DFn else "general codes" if code in GENn else "other non-canonical",
                                 n_atoms=int(mk.sum()), rmsd_A=rmsd, bond_dev_A=bond_dev, atoms_ok=bool(atoms_ok), chir_ok=chir, name_ok=bool(name_ok)))
        except Exception as ex:
            nerr += 1; print("ERR", os.path.basename(f), str(ex)[:120], flush=True)
json.dump(dict(decoder=dec_path, n_files=len(files), errors=nerr, rows=rows), open(out, "w"))
a = np.array([r["rmsd_A"] for r in rows])
print("DONE", dec_path, "positions", len(rows), "median rmsd %.2f" % np.median(a), "name_ok %.3f" % np.mean([r["name_ok"] for r in rows]), "atoms_ok %.3f" % np.mean([r["atoms_ok"] for r in rows]), flush=True)
