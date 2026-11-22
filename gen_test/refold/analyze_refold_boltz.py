"""(Boltz-2 version of analyze_refold2.py: three samples per job, best = highest confidence_score) Compare refolds with the designs and the native peptide (control). usage: analyze_refold2.py REFOLD_OUT_DIR MANIFEST DESIGN_RUN_DIR OUT_JSON
Frames: the AF3 model is superposed on the NATIVE receptor in two ways: all receptor CA ('global') and only the pocket CA (receptor residues with a CA within 10 A of a native peptide CA, 'pocket'); peptide CA RMSD to the native peptide and to the design (the design is superposed on the native receptor with all receptor CA, as in analyze_gen.py) is measured in the pocket frame.
D handedness uses the sign of the receptor L residues of the same model as reference."""
import sys, os, json, glob, numpy as np, gemmi
OUT, MAN, RUN, OJ = sys.argv[1:5]; R = "/mnt/c/Research/RFD4_fine_tune"; man = json.load(open(MAN))
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()); WAT = {"HOH", "WAT", "DOD"}
DF = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split())
def xyz(r, n):
    a = r.find_atom(n, "*"); return np.array(a.pos.tolist()) if a else None
def chir(r):
    n, ca, c, cb = xyz(r, "N"), xyz(r, "CA"), xyz(r, "C"), xyz(r, "CB")
    return None if any(v is None for v in (n, ca, c, cb)) else float(np.dot(np.cross(n - ca, c - ca), cb - ca))
def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); R_ = Vt.T @ np.diag([1, 1, d]) @ U.T; return R_, qc - R_ @ pc
def rms(a, b): return float(np.sqrt(((a - b) ** 2).sum(1).mean()))
def sgnL(rr):
    v = [chir(r) for r in rr if r.name in CAN and r.name != "GLY"]; v = [x for x in v if x is not None]; return float(np.sign(np.median(v)))
res = {}
for name, info in man.items():
    d = f"{OUT}/{name}"; cifs = sorted(glob.glob(f"{d}/{name}_model_*.cif"))
    if not cifs: res[name] = dict(error="no output"); continue
    sc = [json.load(open(f"{d}/confidence_{name}_model_{c.split('_model_')[1][:-4]}.json")) for c in cifs]; bi = int(np.argmax([x["confidence_score"] for x in sc])); S = dict(sc[bi]); S["ranking_score"] = S["confidence_score"]; cj = None; cif = [cifs[bi]]
    m = gemmi.read_structure(cif[0])[0]; pr = list([c for c in m if c.name == "B"][0]); rr = list([c for c in m if c.name == "A"][0])
    nat = gemmi.read_structure(f"{R}/data/real_test_cif/{info['complex']}_real.cif")[0]
    nrr = [r for r in [c for c in nat if c.name != "P" and sum(1 for r in c if r.name not in WAT) > 20][0] if r.name not in WAT]
    npep = [r for r in [c for c in nat if c.name == "P"][0] if r.name not in WAT]
    A = np.array([xyz(r, "CA") for r in rr]); B = np.array([xyz(r, "CA") for r in nrr]); NP = np.array([xyz(r, "CA") for r in npep])
    if len(A) != len(B): res[name] = dict(error="receptor length mismatch"); continue
    ok = ~np.isnan(A).any(1) & ~np.isnan(B).any(1)
    dmin = np.linalg.norm(B[:, None, :] - NP[None], axis=-1).min(1); pocket = ok & (dmin < 10.0)
    Rg, tg = kabsch(A[ok], B[ok]); Rp, tp = kabsch(A[pocket], B[pocket])
    out = dict(kind=info["kind"], complex=info["complex"], iptm=S.get("iptm"), ptm=S.get("ptm"), ranking=S.get("ranking_score"), n_pocket=int(pocket.sum()),
               rec_rmsd_global=rms(A[ok] @ Rg.T + tg, B[ok]), rec_rmsd_pocket=rms(A[pocket] @ Rp.T + tp, B[pocket]))
    out["complex_plddt"] = S.get("complex_plddt"); out["n_samples"] = len(cifs)
    PA = np.array([xyz(r, "CA") for r in pr]) @ Rp.T + tp
    if len(npep) == len(pr): out["pep_ca_rmsd_native"] = rms(PA, NP)
    nc = [i for i, nm in enumerate(info["peptide"]) if nm not in CAN]
    out["nc_kept"] = float(np.mean([pr[i].name == info["peptide"][i] for i in nc])) if nc else None
    sL = sgnL(rr); dl = [chir(r) for r in pr if r.name in DF]; dl = [v for v in dl if v is not None]
    out["D_right"] = float(np.mean([np.sign(v) == -sL for v in dl])) if dl else None
    bd = [np.linalg.norm(xyz(a, "C") - xyz(b, "N")) for a, b in zip(pr[:-1], pr[1:]) if xyz(a, "C") is not None and xyz(b, "N") is not None]
    out["bond_ok"] = float(np.mean([1.2 <= x <= 1.5 for x in bd])) if bd else None
    if info["kind"] == "design":
        dd = glob.glob(f"{RUN}/inference_samples/{info['complex']}_{name.split('-d')[1]}/*_generated.cif")
        if dd:
            dm = gemmi.read_structure(dd[0])[0]; dpep = [r for r in [c for c in dm if c.name == "P"][0] if r.name not in WAT]; drec = [r for c in dm if c.name != "P" for r in c if r.name not in WAT]
            if len(drec) == len(nrr) and len(dpep) == len(pr):
                DA = np.array([xyz(r, "CA") for r in drec]); Rd, td = kabsch(DA, B); DP = np.array([xyz(r, "CA") for r in dpep]) @ Rd.T + td
                out["pep_ca_rmsd_design"] = rms(PA, DP); out["design_ca_rmsd_native"] = rms(DP, NP) if len(NP) == len(DP) else None
    res[name] = out
json.dump(res, open(OJ, "w"), indent=1)
for k, v in res.items(): print(k.ljust(15), {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a not in ("complex", "kind", "ptm", "ranking")})
