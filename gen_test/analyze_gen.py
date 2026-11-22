"""Analyse the generation test (WSL, gemmi): for every design of every run (denoiser x decoder):
 - nc_kept: the given non-canonical identities are present at their positions
 - free_rec: sequence identity to the native peptide at the free (designed) positions
 - ca_rmsd: CA RMSD to the native peptide in the receptor frame (no superposition: the pocket is fixed)
 - chir_D / chir_L: share of D-form / L-amino-acid residues whose C-alpha handedness is right (signed volume N,C,CB around CA, reference sign from the native receptor L residues)
 - nc_atoms: share of non-canonical residues whose heavy-atom count equals the native residue's
 - bond_ok: share of peptide bonds (C-N) between 1.2 and 1.5 A; clash: share of designs with a peptide-receptor heavy-atom distance below 2.0 A
 - contacts: peptide-receptor heavy-atom pairs within 4.5 A, relative to the native peptide
usage: analyze_gen.py RUN_DIR... (each = .../inference_samples/*/*_generated.cif); prints a table and writes summary json next to each run dir"""
import sys, os, glob, json, re, collections
import numpy as np, gemmi
D = "/mnt/c/Research/RFD4_fine_tune"; CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
DF = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split()); WAT = {"HOH", "WAT", "DOD"}
def xyz(r, n):
    a = r.find_atom(n, "*"); return np.array(a.pos.tolist()) if a else None
def chir(r):
    n, ca, c, cb = xyz(r, "N"), xyz(r, "CA"), xyz(r, "C"), xyz(r, "CB")
    if any(v is None for v in (n, ca, c, cb)): return None
    return float(np.dot(np.cross(n - ca, c - ca), cb - ca))
def native(name):
    st = gemmi.read_structure(f"{D}/data/real_test_cif/{name}_real.cif"); m = st[0]
    pep = [r for r in [c for c in m if c.name == "P"][0] if r.name not in WAT]
    rec = [r for c in m if c.name != "P" for r in c if r.name not in WAT]
    ls = [chir(r) for r in rec if r.name in CAN and r.name != "GLY"]; ls = [v for v in ls if v is not None]
    return pep, rec, float(np.sign(np.median(ls)))
NAT = {}
def kabsch(P, Q):
    """rotation+translation taking P onto Q (rows = points)"""
    pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T; return R, qc - R @ pc
def contacts(pep, rec_xyz):
    p = np.array([a.pos.tolist() for r in pep for a in r if a.element.name != "H"])
    d = np.linalg.norm(p[:, None, :] - rec_xyz[None], axis=-1); return int((d < 4.5).sum()), float(d.min())
def analyse(run):
    rows = []
    for f in sorted(glob.glob(f"{run}/inference_samples/*/*_generated.cif")):
        ex = os.path.basename(f).replace("_generated.cif", ""); name = re.sub(r"_\d+$", "", ex)
        if name not in NAT: NAT[name] = native(name)
        npep, nrec, sgnL = NAT[name]
        st = gemmi.read_structure(f); m = st[0]; P = [c for c in m if c.name == "P"]
        if not P: continue
        pep = [r for r in P[0] if r.name not in WAT]
        if len(pep) != len(npep): continue
        grec = [r for c in m if c.name != "P" for r in c if r.name not in WAT]
        if len(grec) != len(nrec): continue
        R, t = kabsch(np.array([xyz(r, "CA") for r in grec]), np.array([xyz(r, "CA") for r in nrec]))   # superpose the designed complex on the native receptor
        for r in pep:
            for a in r: a.pos = gemmi.Position(*(R @ np.array(a.pos.tolist()) + t))
        rec_xyz = np.array([a.pos.tolist() for r in nrec for a in r if a.element.name != "H"])
        nc_pos = [i for i, r in enumerate(npep) if r.name not in CAN]; free = [i for i in range(len(npep)) if i not in nc_pos]
        kept = float(np.mean([pep[i].name == npep[i].name for i in nc_pos])) if nc_pos else np.nan
        rec_ = float(np.mean([pep[i].name == npep[i].name for i in free])) if free else np.nan
        _c = lambda r: (lambda v: np.full(3, np.nan) if v is None else v)(xyz(r, "CA"))   # a designed residue class without a C-alpha (cap, ligand-like class) gives NaN
        ca = np.array([_c(r) for r in pep]); nca = np.array([_c(r) for r in npep])
        rmsd = float(np.sqrt(np.nanmean(((ca - nca) ** 2).sum(-1))))
        cD = [np.sign(chir(r)) == -sgnL for r in pep if r.name in DF and chir(r) is not None]; cL = [np.sign(chir(r)) == sgnL for r in pep if r.name in CAN and r.name != "GLY" and chir(r) is not None]
        nat_atoms = [sum(1 for a in r if a.element.name != "H") for r in npep]; gen_atoms = [sum(1 for a in r if a.element.name != "H") for r in pep]
        ncat = float(np.mean([gen_atoms[i] == nat_atoms[i] for i in nc_pos])) if nc_pos else np.nan
        bonds = []
        for a, b in zip(pep[:-1], pep[1:]):
            c, n = xyz(a, "C"), xyz(b, "N")
            if c is not None and n is not None: bonds.append(1.2 <= np.linalg.norm(c - n) <= 1.5)
        ncont, dmin = contacts(pep, rec_xyz); nn, _ = contacts(npep, rec_xyz)
        rows.append(dict(ex=ex, name=name, nc_kept=kept, free_rec=rec_, ca_rmsd=rmsd, chir_D=float(np.mean(cD)) if cD else np.nan, chir_L=float(np.mean(cL)) if cL else np.nan,
                         nc_atoms=ncat, bond_ok=float(np.mean(bonds)) if bonds else np.nan, clash=float(dmin < 2.0), contacts_rel=ncont / max(nn, 1), seq="".join(r.name + "-" for r in pep)))
    return rows
if __name__ == "__main__":
    out = {}
    for run in sys.argv[1:]:
        rows = analyse(run); key = os.path.basename(run.rstrip("/"))
        if not rows: print(key, "no designs"); continue
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in ("nc_kept", "free_rec", "chir_D", "chir_L", "nc_atoms", "bond_ok", "clash")}
        agg["ca_rmsd_med"] = float(np.nanmedian([r["ca_rmsd"] for r in rows])); agg["contacts_rel_med"] = float(np.nanmedian([r["contacts_rel"] for r in rows]))
        agg["n"] = len(rows); out[key] = dict(agg=agg, rows=rows)
        print(key.ljust(22), " ".join(f"{k} {v:.3f}" for k, v in agg.items()))
    json.dump(out, open(f"{D}/gen_test/gen_summary.json", "w"), indent=1)
