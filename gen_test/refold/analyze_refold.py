"""Compare AF3 refolds with the designs (and the native peptide as control). usage: analyze_refold.py REFOLD_OUT_DIR MANIFEST DESIGN_RUN_DIR OUT_JSON
Per job: AF3 ranking/iptm/ptm, peptide mean pLDDT, then in the receptor frame (Kabsch on receptor CA): peptide CA RMSD to the DESIGN and to the NATIVE peptide (same length only), share of non-canonical residues whose name AF3 kept, C-alpha handedness of the D-forms, peptide bond lengths 1.2-1.5, receptor CA RMSD to the template."""
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
def chains(st):
    m = st[0]; pep = max(m, key=lambda c: -len(c)) if False else None
    cs = sorted(m, key=lambda c: len(c)); return cs
res = {}
for name, info in man.items():
    d = f"{OUT}/{name}"; d = d if os.path.isdir(d) else f"{OUT}/{name.lower()}"
    cif = glob.glob(f"{d}/*_model.cif"); sj = glob.glob(f"{d}/*_summary_confidences.json")
    if not cif or not sj: res[name] = dict(error="no output"); continue
    S = json.load(open(sj[0])); cj = glob.glob(f"{d}/*_confidences.json")
    st = gemmi.read_structure(cif[0]); m = st[0]; pc = [c for c in m if c.name == "B"][0]; rc = [c for c in m if c.name == "A"][0]
    pr = [r for r in pc]; rr = [r for r in rc]
    # native frame
    nat = gemmi.read_structure(f"{R}/data/real_test_cif/{info['complex']}_real.cif")[0]; nrc = [c for c in nat if c.name != "P" and sum(1 for r in c if r.name not in WAT) > 20][0]; nrr = [r for r in nrc if r.name not in WAT]
    npep = [r for r in [c for c in nat if c.name == "P"][0] if r.name not in WAT]
    A = np.array([xyz(r, "CA") for r in rr]); B = np.array([xyz(r, "CA") for r in nrr]); ok = ~np.isnan(A).any(1) & ~np.isnan(B).any(1) if len(A) == len(B) else None
    if ok is None: res[name] = dict(error="receptor length mismatch"); continue
    Rm, t = kabsch(A[ok], B[ok]); rec_rmsd = float(np.sqrt(((A[ok] @ Rm.T + t - B[ok]) ** 2).sum(1).mean()))
    PA = np.array([xyz(r, "CA") for r in pr]) @ Rm.T + t
    out = dict(kind=info["kind"], complex=info["complex"], iptm=S.get("iptm"), ptm=S.get("ptm"), ranking=S.get("ranking_score"), rec_rmsd_to_native=rec_rmsd)
    if cj:
        pl = json.load(open(cj[0])).get("atom_plddts"); 
        if pl: 
            n_rec = sum(len(r) for r in rr); out["pep_plddt"] = float(np.mean(pl[n_rec:]))
    out["nc_kept"] = float(np.mean([r.name == nm for r, nm in zip(pr, info["peptide"]) if nm not in CAN])) if any(nm not in CAN for nm in info["peptide"]) else None
    dl = [chir(r) for r in pr if r.name in DF]; out["D_right"] = float(np.mean([v is not None and v < 0 for v in dl])) if dl else None
    bd = []
    for a, b in zip(pr[:-1], pr[1:]):
        c, n = xyz(a, "C"), xyz(b, "N")
        if c is not None and n is not None: bd.append(np.linalg.norm(c - n))
    out["bond_ok"] = float(np.mean([1.2 <= x <= 1.5 for x in bd])) if bd else None
    if len(npep) == len(pr): out["ca_rmsd_to_native"] = float(np.sqrt(((PA - np.array([xyz(r, "CA") for r in npep])) ** 2).sum(1).mean()))
    if info["kind"] == "design":
        dd = glob.glob(f"{RUN}/inference_samples/{info['complex']}_{name.split('-d')[1]}/*_generated.cif")
        if dd:
            dp = [r for r in [c for c in gemmi.read_structure(dd[0])[0] if c.name == "P"][0] if r.name not in WAT]
            if len(dp) == len(pr): out["ca_rmsd_to_design"] = float(np.sqrt(((PA - np.array([xyz(r, "CA") for r in dp])) ** 2).sum(1).mean()))
    res[name] = out
json.dump(res, open(OJ, "w"), indent=1)
for k, v in res.items(): print(k.ljust(16), {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a not in ("complex",)})
