"""PER NON-CANONICAL POSITION analysis of generated designs (identity of the NCAAs given, 11 real pockets): for every NCAA position of every design, compare the designed residue with the native residue of the same pocket. Metrics per position:
  ca_dist_A     distance between the designed and the native C-alpha (receptor superposed): did the denoiser put the residue where the native one is?
  res_rmsd_A    all-atom RMSD of the designed residue against the native residue (pose + conformation, atoms matched by name)
  local_rmsd_A  the same after superposing the residue on the native one (internal geometry only: decoder quality)
  bond_dev_A    mean |designed - native| length of the covalent bonds inside the residue (native pairs closer than 1.9 A)
  atoms_ok      designed residue has exactly the native atom names
  chir_ok       C-alpha handedness of the residue equals the native one (residues with a C-beta)
  clash         any atom of the residue closer than 2.0 A to a receptor atom
  contacts_rel  receptor contacts (< 4.5 A) of the residue relative to the native residue
usage: analyze_positions.py RUN_KEY=RUN_DIR ...   (RUN_DIR as for analyze_gen.py); writes diag/positions.json"""
import sys, os, glob, json, re, collections
import numpy as np, gemmi
import analyze_gen as A
def heavy(r): return {a.name: np.array(a.pos.tolist()) for a in r if a.element.name != "H"}
def kabsch_rmsd(P, Q):
    R, t = A.kabsch(P, Q); return float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(-1).mean()))
def analyse(rdir):
    rows = []
    for f in sorted(glob.glob(f"{rdir}/inference_samples/*/*_generated.cif")):
        ex = os.path.basename(f).replace("_generated.cif", ""); name = re.sub(r"_\d+$", "", ex)
        if name not in A.NAT: A.NAT[name] = A.native(name)
        npep, nrec, sgnL = A.NAT[name]
        m = gemmi.read_structure(f)[0]; P = [c for c in m if c.name == "P"]
        if not P: continue
        pep = [r for r in P[0] if r.name not in A.WAT]
        if len(pep) != len(npep): continue
        grec = [r for c in m if c.name != "P" for r in c if r.name not in A.WAT]
        if len(grec) != len(nrec): continue
        R, t = A.kabsch(np.array([A.xyz(r, "CA") for r in grec]), np.array([A.xyz(r, "CA") for r in nrec]))
        for r in pep:
            for a in r: a.pos = gemmi.Position(*(R @ np.array(a.pos.tolist()) + t))
        rec = np.array([a.pos.tolist() for r in nrec for a in r if a.element.name != "H"])
        for i, nr in enumerate(npep):
            if nr.name in A.CAN: continue
            dr = pep[i]; hd, hn = heavy(dr), heavy(nr); common = [k for k in hn if k in hd]
            if len(common) < 3 or "CA" not in hd or "CA" not in hn: continue
            Pd = np.array([hd[k] for k in common]); Pn = np.array([hn[k] for k in common])
            dn = np.linalg.norm(Pn[:, None] - Pn[None], axis=-1); dd = np.linalg.norm(Pd[:, None] - Pd[None], axis=-1); bm = (dn < 1.9) & (dn > 0.5)
            ch = None
            if A.xyz(dr, "CB") is not None and A.xyz(nr, "CB") is not None:
                cd, cn = A.chir(dr), A.chir(nr); ch = bool(np.sign(cd) == np.sign(cn)) if cd is not None and cn is not None else None
            dmin = float(np.linalg.norm(np.array(list(hd.values()))[:, None] - rec[None], axis=-1).min())
            cnt = lambda X: int((np.linalg.norm(X[:, None] - rec[None], axis=-1) < 4.5).sum())
            rows.append(dict(ex=ex, name=name, pos=i, code=nr.name, group="D-forms" if nr.name in A.DF else "general codes" if nr.name in "23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split() else "other non-canonical",
                             ca_dist_A=float(np.linalg.norm(hd["CA"] - hn["CA"])), res_rmsd_A=float(np.sqrt(((Pd - Pn) ** 2).sum(-1).mean())), local_rmsd_A=kabsch_rmsd(Pd, Pn),
                             bond_dev_A=float(np.abs(dd - dn)[bm].mean()) if bm.any() else float("nan"), atoms_ok=bool(set(hd) == set(hn)), chir_ok=ch, clash=bool(dmin < 2.0), contacts_rel=cnt(Pd) / max(cnt(Pn), 1)))
    return rows
if __name__ == "__main__":
    out = {}
    for a in sys.argv[1:]:
        key, rd = a.split("="); rows = analyse(rd); out[key] = rows
        g = collections.defaultdict(list)
        for r in rows: g[r["group"]].append(r)
        print(key.ljust(18), "positions", len(rows), "| ca_dist med %.2f | res_rmsd med %.2f | local_rmsd med %.2f | bond_dev %.3f | atoms_ok %.3f | chir_ok %.3f | clash %.3f | contacts_rel med %.2f" % (
            np.median([r["ca_dist_A"] for r in rows]), np.median([r["res_rmsd_A"] for r in rows]), np.median([r["local_rmsd_A"] for r in rows]), np.nanmean([r["bond_dev_A"] for r in rows]),
            np.mean([r["atoms_ok"] for r in rows]), np.mean([r["chir_ok"] for r in rows if r["chir_ok"] is not None]), np.mean([r["clash"] for r in rows]), np.median([r["contacts_rel"] for r in rows])))
    json.dump(out, open(f"{A.D}/gen_test/diag/positions.json", "w"))
