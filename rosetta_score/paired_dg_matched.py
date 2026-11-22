"""Paired Rosetta dG with MATCHED relax: the native complex and every swapped variant go through exactly the same protocol, so a relax that favours the native pocket cannot create a fake gain.
For each NCAA slot (slot list read from an earlier result json, numbering = the unrelaxed ros_in PDBs) and for each variant (native, L parent, ALA, GLY; only that ONE slot differs):
  start from the ORIGINAL (unrelaxed) complex -> apply the swap -> FastRelax (3 repeats, bb and chi free only for the interface residues, whole complex restrained to the start coordinates) -> dG (interface repack, E_complex - E_peptide - E_receptor, NREP repacks)
  done NREL times independently; dG of a variant = mean over relaxes and repacks.   gain = dG(swap) - dG(native)  (> 0: the NCAA is better).
usage: python3 paired_dg_matched.py <ros_in dir> <slots.json (earlier result)> <out.json> [NREL] [NREP] [NPROC]     env ONLY=file1.pdb,file2.pdb to restrict; DONE=a.json,b.json skips slots already scored; REVERSE=1 walks the list from the end; SLICE=a:b takes a part (after DONE/REVERSE)."""
import sys, os, json, time
import numpy as np
import pyrosetta
from multiprocessing import Pool
import paired_dg as P

def relax(pose, free):
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    p = pose.clone(); fr = FastRelax(P.SF, 3); fr.constrain_relax_to_start_coords(True); fr.ramp_down_constraints(False)
    mm = MoveMap(); mm.set_bb(False); mm.set_chi(False)
    for i in free: mm.set_bb(i, True); mm.set_chi(i, True)
    fr.set_movemap(mm); fr.apply(p); return p, float(CA_rmsd(pose, p))

def variant_dG(pose, pep, iface, slot, nrel, nrep):
    vals, drift = [], []
    for _ in range(nrel):
        p, d = relax(pose, set(iface) | {slot}); drift.append(d)
        vals += P.dG(p, pep, iface, [slot], nrep)[2]
    return float(np.mean(vals)), float(np.std(vals)), drift

def task(a):
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    d, rowin, nrel, nrep = a; t0 = time.time(); slot = rowin["slot"]
    try:
        pose = pyrosetta.pose_from_file(os.path.join(d, rowin["file"])); pdi = pose.pdb_info()
        pep = [i for i in range(1, pose.size() + 1) if pdi.chain(i) == rowin["pep"]]; iface = P.interface(pose, pep)
        nat = variant_dG(pose, pep, iface, slot, nrel, nrep); row = dict(file=rowin["file"], slot=slot, pos_in_pep=rowin.get("pos_in_pep"), pep_len=rowin.get("pep_len"), ncaa=rowin["ncaa"], parent=rowin["parent"], dG_native=nat[0], sd_native=nat[1], native_relax_ca_rmsd=float(np.median(nat[2])), refs={})
        for ref in dict.fromkeys([rowin["parent"], "ALA", "GLY"]):
            mut = pose.clone(); MutateResidue(slot, ref).apply(mut); b = variant_dG(mut, pep, iface, slot, nrel, nrep)
            row["refs"][ref] = dict(dG=b[0], sd=b[1], gain=b[0] - nat[0], backbone_clash=bool(b[0] - nat[0] > 20), relax_ca_rmsd=float(np.median(b[2])))
        row["sec"] = time.time() - t0; return row
    except Exception as e:
        return dict(file=rowin["file"], slot=slot, ncaa=rowin["ncaa"], skip="error " + str(e)[:80])

if __name__ == "__main__":
    d, slots_json, out = sys.argv[1], sys.argv[2], sys.argv[3]; nrel = int(sys.argv[4]) if len(sys.argv) > 4 else 2; nrep = int(sys.argv[5]) if len(sys.argv) > 5 else 3; nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 28
    man = {m["file"]: m for m in json.load(open(os.path.join(d, "manifest.json")))}
    rows = [dict(x, pep=man[x["file"]]["pep"]) for r in json.load(open(slots_json)) for x in r["rows"] if "refs" in x]
    only = os.environ.get("ONLY")
    if only: rows = [x for x in rows if x["file"] in only.split(",")]
    for dj in filter(None, os.environ.get("DONE", "").split(",")):            # skip (file, slot) pairs already scored in these result files
        done = {(r["file"], x["slot"]) for r in json.load(open(dj)) for x in r["rows"] if "refs" in x}; rows = [x for x in rows if (x["file"], x["slot"]) not in done]
    if os.environ.get("REVERSE") == "1": rows = rows[::-1]                      # a second machine takes the slots from the END of the list
    if os.environ.get("SLICE"): a, b = os.environ["SLICE"].split(":"); rows = rows[int(a or 0):int(b) if b else None]
    res = []; t0 = time.time()
    with Pool(nproc, initializer=P.init) as pool:
        for r in pool.imap_unordered(task, [(d, x, nrel, nrep) for x in rows]):
            res.append(dict(file=r["file"], rows=[r])); print(r["file"], r["slot"], r["ncaa"], "ok" if "refs" in r else r["skip"], "%.0fs" % r.get("sec", 0), "(%d/%d)" % (len(res), len(rows)), flush=True)
            json.dump(res, open(out, "w"))
    print("done %.0fs" % (time.time() - t0))
