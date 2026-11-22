"""Class-level and contact-level scoring of the FREE-CHOICE designs (Results.MD 82). usage: python score_choice_designs.py  (Windows python + gemmi; writes diag/choice_scores.json)
Per NCAA slot: family of the designed code vs the family of the native code at that slot; heavy-atom contacts (< 4.5 A) of the designed slot residue with the receptor, minimum distance, close contact (< 2.0 A);
the native residue's own numbers come from the native pocket files (gen_test/inputs). Receptor chains are fixed (C_CRD), so designs and natives share one frame, no superposition needed."""
import glob, os, re, json, collections, numpy as np, gemmi
G = r"C:\Research\RFD4_fine_tune\gen_test"
FAM = {}
for fam, codes in {"D-alpha": "DAL DAR DAS DCY DGL DGN DHI DIL DLE DLY DPN DPR DSG DSN DTH DTR DTY DVA MED", "N-alkyl": "SAR 04Q MAA MLE MVA IML MEA NMM", "alpha,alpha-disubst.": "AIB 2ML DIV 02K 1AC 192 0XL AC5",
                   "beta/gamma": "B3L B3A BAL B3K B3Q XCP STA", "side-chain modified": "23F 4FB FP9 HYP ORN DBU", "PTM": "SEP TPO PTR M3L MLY ALY CGU"}.items():
    for c in codes.split(): FAM[c] = fam
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()); WAT = {"HOH", "WAT"}
fam = lambda c: "canonical" if c in CAN else FAM.get(c, "other")
def contacts(res, rec_xyz):
    p = np.array([a.pos.tolist() for a in res if a.element.name != "H"]); d = np.linalg.norm(p[:, None, :] - rec_xyz[None], axis=-1); return int((d < 4.5).sum()), float(d.min())
def slots_of(f):
    m = gemmi.read_structure(f)[0]; P = [c for c in m if c.name == "P"]
    if not P: return None
    pep = [r for r in P[0] if r.name not in WAT]; rec = np.array([a.pos.tolist() for c in m if c.name != "P" for r in c if r.name not in WAT for a in r if a.element.name != "H"])
    return pep, rec
nat = {}
for f in sorted(glob.glob(G + r"\inputs\*.cif")):
    r = slots_of(f)
    if r: nat[os.path.basename(f)[:-4]] = r
res = {}
for run, d in (("control", "control_choice_run8"), ("G3", "G3_bal_choice_run8"), ("S1", "S1_readout_choice_run8")):
    rows = []
    for f in sorted(glob.glob(G + rf"\out_local\{d}\inference_samples\*\*_generated.cif")):
        ex = os.path.basename(f).replace("_generated.cif", ""); name = re.sub(r"_\d+$", "", ex)
        if name not in nat: continue
        r = slots_of(f)
        if not r: continue
        pep, rec = r; npep, nrec = nat[name]
        if len(pep) != len(npep): continue
        for i, (dres, nres) in enumerate(zip(pep, npep)):
            if nres.name in CAN: continue
            dc, dm = contacts(dres, rec); nc, nm = contacts(nres, nrec)
            rows.append(dict(ex=ex, pos=i + 1, native=nres.name, designed=dres.name, fam_native=fam(nres.name), fam_designed=fam(dres.name), contacts=dc, dmin=dm, native_contacts=nc, native_dmin=nm))
    res[run] = rows
json.dump(res, open(G + r"\diag\choice_scores.json", "w"))
for run, rows in res.items():
    fn = collections.Counter(r["fam_native"] for r in rows); fd = collections.Counter(r["fam_designed"] for r in rows); n = len(rows)
    agree = np.mean([r["fam_native"] == r["fam_designed"] for r in rows]); chance = sum((fn[k] / n) * (fd[k] / n) for k in fn)
    print(run, "n", n, "family agreement %.2f (chance %.2f)" % (agree, chance), "| designed families", {k: round(v / n, 2) for k, v in fd.items()}, "| native", {k: round(v / n, 2) for k, v in fn.items()},
          "| contacts median designed %.0f native %.0f | close(<2A) %.2f native %.2f | >=1 contact %.2f" % (np.median([r["contacts"] for r in rows]), np.median([r["native_contacts"] for r in rows]), np.mean([r["dmin"] < 2.0 for r in rows]), np.mean([r["native_dmin"] < 2.0 for r in rows]), np.mean([r["contacts"] > 0 for r in rows])))
