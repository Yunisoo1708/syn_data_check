"""Score the FREE-CHOICE designs (and the native pockets) with the light Rosetta protocol, one position at a time.
Per structure: relax once (interface residues free, whole complex restrained to the start coordinates); then for every NCAA slot of the peptide (caps excluded) the residue that is THERE (designed, or native for group 'native') against ALA, GLY and its L parent (only that slot changes; identical local protocol: swap, repack, chi minimisation, 12 repacks, two halves);
control slots: up to 3 canonical interface residues per structure against ALA (an alanine scan). Also E_int of the whole peptide with the receptor.
usage: python3 paired_score_designs.py <design_in dir> <out.json> [NREP=12] [NPROC=28]    (env ONLY=a.pdb,b.pdb)"""
import sys, os, json, time
import numpy as np
import pyrosetta
from multiprocessing import Pool
import paired_dg as P, paired_dg_matched as M, paired_dg_light as L
PAR = dict(P.PARENT); PAR.update({"4FB": "PHE", "04Q": "PHE", "23F": "PHE", "FP9": "PRO", "1AC": "ALA", "MAA": "ALA", "SAR": "GLY", "BAL": "ALA"})
CAPS = P.CAPS; CANON = P.CANON

def swap(pose, slot, ref):
    """Replace ONLY residue `slot` by the canonical `ref`; MutateResidue first, else (on-the-fly PDB residue types such as pdb_SAR) copy the backbone into the canonical type."""
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.rosetta.core.pose import replace_pose_residue_copying_existing_coordinates
    mut = pose.clone()
    try:
        MutateResidue(slot, ref).apply(mut); return mut
    except Exception:
        mut = pose.clone(); rt = mut.residue_type_set_for_pose().name_map(ref); replace_pose_residue_copying_existing_coordinates(mut, slot, rt); return mut

def task(a):
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    d, m, nrep = a; t0 = time.time(); rows = []; info = dict(file=m["file"], group=m["group"], ex=m["ex"])
    try:
        pose = pyrosetta.pose_from_file(os.path.join(d, m["file"])); pdi = pose.pdb_info()
        pep = [i for i in range(1, pose.size() + 1) if pdi.chain(i) == m["pep"]]; rec = [i for i in range(1, pose.size() + 1) if i not in set(pep)]
        codes = m["codes"]; cname = {i: codes[pdi.number(i) - 1] for i in pep if 1 <= pdi.number(i) <= len(codes)}   # residue names from the FILE by residue number (Rosetta reports N-methylated residues under the parent name)
        pep = [i for i in pep if i in cname]
        iface = P.interface(pose, pep); slots = [i for i in pep if cname[i] not in CANON and cname[i] not in CAPS]
        pose, drift = M.relax(pose, set(iface) | set(slots)); info["relax_ca_rmsd"] = drift
        xyz = {i: np.array(list(pose.residue(i).nbr_atom_xyz())) for i in range(1, pose.size() + 1)}
        ctrl = sorted([(sum(1 for j in rec if np.linalg.norm(xyz[i] - xyz[j]) < 8), i) for i in pep if cname[i] in CANON and cname[i] not in ("GLY", "ALA", "PRO")], reverse=True)
        ctrl = [(n, i) for n, i in ctrl if n >= 3][:3]
        jobs = [("ncaa", i) for i in slots] + [("control", i) for _, i in ctrl]
        for kind, slot in jobs:
            nm = cname[slot]; near = [j for j in range(1, pose.size() + 1) if np.linalg.norm(xyz[j] - xyz[slot]) < 10]
            nat = L.local_variant(pose, slot, pep, rec, near, nrep); row = dict(kind=kind, slot=slot, pos_in_pep=int(pdi.number(slot)), code=nm, own=nat, refs={})
            refs = ["ALA"] if kind == "control" else list(dict.fromkeys(["ALA", "GLY"] + ([PAR[nm]] if nm in PAR else [])))
            for ref in refs:
                if ref == nm: continue
                try:
                    mut = swap(pose, slot, ref); b = L.local_variant(mut, slot, pep, rec, near, nrep)
                    row["refs"][ref] = dict(half=[[b[h][k] - nat[h][k] for k in (0, 1)] for h in (0, 1)])
                except Exception as e: row["refs"][ref] = dict(error=str(e)[:60])
            rows.append(row)
        if rows: info["e_int"] = float(np.mean([np.mean([h[0] for h in r["own"]]) for r in rows]))
    except Exception as e:
        info["error"] = str(e)[:100]
    return dict(info=info, rows=rows, sec=time.time() - t0)

if __name__ == "__main__":
    d, outp = sys.argv[1], sys.argv[2]; nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 12; nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 28
    man = json.load(open(os.path.join(d, "manifest.json"))); only = os.environ.get("ONLY")
    if only: man = [m for m in man if m["file"] in only.split(",")]
    res = []; t0 = time.time()
    with Pool(nproc, initializer=P.init) as pool:
        for r in pool.imap_unordered(task, [(d, m, nrep) for m in man]):
            res.append(r); print(r["info"]["file"], r["info"].get("error", ""), "rows", len(r["rows"]), "%.0fs" % r["sec"], "(%d/%d)" % (len(res), len(man)), flush=True); json.dump(res, open(outp, "w"))
    print("done %.0fs" % (time.time() - t0))
