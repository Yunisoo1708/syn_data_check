"""SEQUENCE READOUT at NON-CANONICAL positions of real complexes (test only), identity HIDDEN: for every real complex that holds an NCAA, hide the identity of its NCAA tokens (conditioning <M>), noise the generated tokens at a fixed time t (both CA and latent; t = 0 pure noise, 1 clean), run the denoiser (control, or + a LoRA adapter incl. a trained readout head) and read the sequence readout (189 classes) at the generated NCAA tokens: top-1, top-5, share of guesses that are CANONICAL classes (canonical bias), plus group-level (D-form / general / other) and parent-level agreement for D-forms.
usage: eval_readout.py NAME ADAPTER_or_none OUT.json SET      (SET = real | ham; env TS=0.3,0.5,0.7,0.9, NCLIST for ham)"""
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
from atomworks.ml.conditions import C_SEQ
name, adapter, out, which = sys.argv[1:5]
files = sorted(glob.glob(f"{T}/real_test_cif/*.cif" if which == "real" else f"{T}/real_test_ham/*.cif"))
if os.environ.get("NCLIST"): keep = set(json.load(open(os.environ["NCLIST"]))); files = [f for f in files if os.path.basename(f) in keep]
if os.environ.get("SMOKE"): files = files[:int(os.environ["SMOKE"])]
TS = [float(x) for x in os.environ.get("TS", "0.3,0.5,0.7,0.9").split(",")]
dev = "cuda"; model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device=dev); model.train()
if adapter != "none": model.nn = pt.PeftModel.from_pretrained(model.nn, adapter, is_trainable=False)
model.nn.eval(); model.autoencoder.eval()
import rfproteina_encoding_ext as X
t2i = {str(k): int(v) for k, v in X.UNIFIED_ATOM37_ENCODING.token_to_idx.items()}; i2t = {v: k for k, v in t2i.items()}; M_IDX = t2i["<M>"]
NCs = set(map(str, list(X.RESIDUES_BY_TOKEN_TYPE["ncaa"]) + list(X.RESIDUES_BY_TOKEN_TYPE["ncaa_ext"]))); NCt = torch.tensor(sorted(t2i[k] for k in NCs if k in t2i), device=dev)
CANt = torch.tensor(sorted(t2i[k] for k in "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()), device=dev)
DFn = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split()); GENn = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
PARENT = dict(DAL="ALA", DAR="ARG", DAS="ASP", DCY="CYS", DGL="GLU", DGN="GLN", DHI="HIS", DIL="ILE", DLE="LEU", DLY="LYS", DPN="PHE", DPR="PRO", DSG="ASN", DSN="SER", DTH="THR", DTR="TRP", DTY="TYR", DVA="VAL", MED="MET")
mod = sys.modules[model.fm.__class__.__module__]; SUPM = mod.compute_flow_supervision_masks
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None); SCH = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
rows = []; t0 = time.time(); nerr = 0
with torch.no_grad():
    for k, f in enumerate(files):
        seed = int(hashlib.sha1(os.path.basename(f).encode()).hexdigest()[:8], 16)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        try:
            tb = collate_batch([DesignProblemDataset([Path(f)], transform=pipe)[0]], schema=SCH)
            b = {kk: v.to(dev) if isinstance(v, torch.Tensor) else v for kk, v in tb.items()}
            b["coords_nm"] = b["coords"] / 10; b["coord_mask"] = b["atom37_mask"]
            model.autoencoder.encoder(b); model.add_clean_samples(b)
            rt = b["residue_type"][0].long(); nc = torch.isin(rt, NCt)
            if not bool(nc.any()): continue
            if C_SEQ.full_name in b: b[C_SEQ.full_name] = torch.where(nc[None], torch.full_like(b[C_SEQ.full_name], M_IDX), b[C_SEQ.full_name])
            model.fm.corrupt_batch(b); emit_m = mod.compute_flow_emit_masks(b)
            sup = SUPM(b)["local_latents"][0] & nc; sel = torch.nonzero(sup).flatten().tolist()
            if not sel: continue
            for T_FIX in TS:
                for mm_ in ("bb_ca", "local_latents"):
                    tt = torch.full_like(b["t"][mm_], T_FIX); b["t"][mm_] = tt; tv = tt.view(-1, *([1] * (b["x_1"][mm_].dim() - 1)))
                    b["x_t"][mm_] = torch.where(emit_m[mm_][..., None], (1 - tv) * b["x_0"][mm_] + tv * b["x_1"][mm_], b["x_t"][mm_])
                o = model.call_nn(b, n_recycle=1); lg = o["sequence"]["logits"][0]
                for i in sel:
                    code = i2t.get(int(rt[i]), str(int(rt[i]))); pr = int(lg[i].argmax()); top5 = lg[i].topk(5).indices.tolist(); pc = i2t.get(pr, str(pr))
                    rows.append(dict(set=which, f=os.path.basename(f), pos=i, t=T_FIX, code=code, group="D-forms" if code in DFn else "general codes" if code in GENn else "other non-canonical",
                                     top1=bool(pr == int(rt[i])), top5=bool(int(rt[i]) in top5), guess=pc, guess_canonical=bool(pr in CANt.tolist()), guess_nc=bool(pr in NCt.tolist()),
                                     parent_ok=bool(code in PARENT and PARENT.get(pc, pc) == PARENT[code]), guess_D=bool(pc in DFn)))
        except Exception as ex:
            nerr += 1; print("ERR", os.path.basename(f), str(ex)[:120], flush=True)
        if k % 40 == 0: print(name, which, k, "of", len(files), f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(name=name, adapter=adapter, set=which, ts=TS, errors=nerr, rows=rows), open(out, "w"))
for T_FIX in TS:
    r = [x for x in rows if x["t"] == T_FIX]
    if r: print("DONE", name, which, "t", T_FIX, "n", len(r), "top1 %.3f top5 %.3f canonical-guess %.3f parent(D) %.3f" % (np.mean([x["top1"] for x in r]), np.mean([x["top5"] for x in r]), np.mean([x["guess_canonical"] for x in r]), np.mean([x["parent_ok"] for x in r])), flush=True)
