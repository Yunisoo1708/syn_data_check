"""LIGHT paired Rosetta score: relax ONCE (the globally relaxed native from ros_in_relaxed/), then every variant (native, L parent, ALA, GLY; ONLY the one slot differs) gets exactly the same LOCAL protocol:
swap -> repack the neighbourhood of the slot (extra rotamers) -> chi-only minimisation; backbone fixed, no independent global relax (that was the noise source of paired_dg_matched.py).
Per variant NREP repacks, split in two independent halves A and B; the score of a half = mean over its 3 lowest-total-energy repacks.
Scores (two-body energies from the energy graph, REF2015 weights, lower = binds better):
  E_int  = sum over peptide-receptor residue pairs        (interface interaction energy)
  E_slot = sum over pairs (slot, receptor residue)        (what the slot itself contributes to the interface)
gain = E(swap) - E(native); > 0: the NCAA (or native residue) is better than the swap.
CONTROL (known answer): canonical interface residues of the same peptides, native residue against ALA: hot-spot-like residues should mostly show gain > 0 (an alanine scan).
Reliability = correlation of the gains of half A and half B across slots.
usage: python3 paired_dg_light.py <ros_in dir> <slots.json> <out.json> [NREP=12] [NPROC=28]   (reads <ros_in>_relaxed/<file>)"""
import sys, os, json, time
import numpy as np
import pyrosetta
from multiprocessing import Pool
import paired_dg as P
CANON = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())

def edge_sum(pose, i, js):
    eg = pose.energies().energy_graph(); w = P.SF.weights(); tot = 0.0
    for j in js:
        if j == i: continue
        e = eg.find_energy_edge(min(i, j), max(i, j))
        if e is not None: tot += e.dot(w)
    return tot

def local_variant(pose, slot, pep, rec, near, nrep):
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    free = set(near) | {slot}; rows = []
    for _ in range(nrep):
        p = pose.clone(); P.pack(p, free)
        mm = MoveMap(); mm.set_bb(False); mm.set_chi(False)
        for i in free: mm.set_chi(i, True)
        MinMover(mm, P.SF, "lbfgs_armijo_nonmonotone", 0.01, True).apply(p)
        et = P.SF(p)
        e_int = sum(edge_sum(p, i, rec) for i in pep); e_slot = edge_sum(p, slot, rec)
        rows.append((et, e_int, e_slot))
    rows = np.array(rows); half = nrep // 2; out = []
    for a in (rows[:half], rows[half:]):
        k = np.argsort(a[:, 0])[:3]; out.append((float(a[k, 1].mean()), float(a[k, 2].mean())))
    return out   # [(E_int, E_slot) half A, half B]

def task(a):
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    d, relaxed_d, file, pepchain, slots, nrep = a; t0 = time.time(); out = []
    try:
        pose = pyrosetta.pose_from_file(os.path.join(relaxed_d, file)); pdi = pose.pdb_info()
        pep = [i for i in range(1, pose.size() + 1) if pdi.chain(i) == pepchain]; rec = [i for i in range(1, pose.size() + 1) if i not in set(pep)]
        xyz = {i: np.array(list(pose.residue(i).nbr_atom_xyz())) for i in range(1, pose.size() + 1)}
        # canonical control slots: peptide residues with receptor contact, big side chain, at most 3 per complex
        ctrl = []
        for i in pep:
            nm = pose.residue(i).name3().strip()
            if nm in CANON and nm not in ("GLY", "ALA", "PRO"):
                nc = sum(1 for j in rec if np.linalg.norm(xyz[i] - xyz[j]) < 8)
                if nc >= 3: ctrl.append((nc, i, nm))
        ctrl = sorted(ctrl, reverse=True)[:3]
        jobs = [dict(kind="ncaa", slot=s["slot"], ncaa=s["ncaa"], parent=s["parent"], refs=list(dict.fromkeys([s["parent"], "ALA", "GLY"]))) for s in slots] + [dict(kind="control", slot=i, ncaa=nm, parent=nm, refs=["ALA"]) for _, i, nm in ctrl]
        for jb in jobs:
            slot = jb["slot"]; near = [j for j in range(1, pose.size() + 1) if np.linalg.norm(xyz[j] - xyz[slot]) < 10]
            nat = local_variant(pose, slot, pep, rec, near, nrep); row = dict(file=file, slot=slot, kind=jb["kind"], ncaa=jb["ncaa"], parent=jb["parent"], native=nat, refs={})
            for ref in jb["refs"]:
                mut = pose.clone(); MutateResidue(slot, ref).apply(mut); b = local_variant(mut, slot, pep, rec, near, nrep)
                row["refs"][ref] = dict(half=[[b[h][k] - nat[h][k] for k in (0, 1)] for h in (0, 1)])   # [half][E_int gain, E_slot gain]
            out.append(row)
    except Exception as e:
        out.append(dict(file=file, error=str(e)[:100]))
    return dict(file=file, rows=out, sec=time.time() - t0)

if __name__ == "__main__":
    d, slots_json, outp = sys.argv[1], sys.argv[2], sys.argv[3]; nrep = int(sys.argv[4]) if len(sys.argv) > 4 else 12; nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 28
    man = {m["file"]: m for m in json.load(open(os.path.join(d, "manifest.json")))}; by = {}
    for r in json.load(open(slots_json)):
        for x in r["rows"]:
            if "refs" in x: by.setdefault(x["file"], []).append(dict(slot=x["slot"], ncaa=x["ncaa"], parent=x["parent"]))
    only = os.environ.get("ONLY"); files = [f for f in by if (not only or f in only.split(","))]
    res = []; t0 = time.time()
    with Pool(nproc, initializer=P.init) as pool:
        for r in pool.imap_unordered(task, [(d, d.rstrip("/") + "_relaxed", f, man[f]["pep"], by[f], nrep) for f in files]):
            res.append(r); print(r["file"], "rows", len(r["rows"]), [x.get("error") for x in r["rows"] if x.get("error")][:1], "%.0fs" % r["sec"], "(%d/%d)" % (len(res), len(files)), flush=True); json.dump(res, open(outp, "w"))
    print("done %.0fs" % (time.time() - t0))
