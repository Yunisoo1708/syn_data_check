"""Step 2 (WSL, PyRosetta): PAIRED Rosetta interface dG, ONE POSITION AT A TIME.
For every non-canonical residue slot i of the peptide (caps ACE/NH2 excluded) of a complex:
  variant NATIVE : the structure as given
  variant PARENT : ONLY residue i swapped to its canonical L parent (e.g. DAL->ALA, ALY->LYS); all other residues,
                   including other NCAAs, stay untouched
For each variant: repack the interface (extra rotamers), E_complex; split peptide / receptor, repack both again,
E_peptide, E_receptor;  dG = E_complex - E_peptide - E_receptor  (Rosetta units, lower = binds better).
Repeated NREP times (random repacks), mean and sd reported.   gain = dG_ref - dG_native for ref = L parent, ALA, GLY (> 0: the NCAA is better than that canonical residue at this slot). An L parent swapped onto a D backbone clashes (gain > 20, flagged backbone_clash): for D-forms use the ALA/GLY references.
usage: [RELAX=0|1 (default 1)] [ONLY=a.pdb,b.pdb] python3 paired_dg.py <pdb_dir_with_manifest.json> <out.json> [NREP] [NPROC]
RELAX=1: every complex is first relaxed once (FastRelax, 3 repeats, constrained to the start coordinates); relaxed PDBs go to <pdb_dir>_relaxed/.
NOTE: do not import gemmi here (segfaults together with PyRosetta); InterfaceAnalyzerMover is not used (segfaults on split-chain NCAA peptides)."""
import sys, os, json, time, traceback
import numpy as np
import pyrosetta
from multiprocessing import Pool
PARENT = dict(DAL="ALA", DAR="ARG", DAS="ASP", DCS="CYS", DGU="GLU", DGN="GLN", DHI="HIS", DIL="ILE", DLE="LEU", DLY="LYS", DPH="PHE", DPR="PRO", DAN="ASN", DSE="SER", DTH="THR", DTR="TRP", DTY="TYR", DVA="VAL", MED="MET",
              ALY="LYS", M3L="LYS", MLY="LYS", MLE="LEU", MVA="VAL", IML="ILE", HYP="PRO", PTR="TYR", SEP="SER", TPO="THR", CGU="GLU", NLE="LEU", AIB="ALA", SAR="GLY", ORN="LYS", MEA="PHE", MAA="ALA", ABA="ALA", DAB="LYS", NMM="ARG")
CAPS = {"ACE", "NH2", "ETA"}
CANON = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
SF = None

def init():
    global SF
    pyrosetta.init("-mute all -ex1 -ex2", silent=True); SF = pyrosetta.get_fa_scorefxn()

def pack(pose, free):
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    t = pyrosetta.standard_packer_task(pose); t.restrict_to_repacking()
    for i in range(1, pose.size() + 1):
        if i not in free: t.nonconst_residue_task(i).prevent_repacking()
    PackRotamersMover(SF, t).apply(pose)

def interface(pose, pep):
    xyz = {i: np.array(list(pose.residue(i).nbr_atom_xyz())) for i in range(1, pose.size() + 1)}
    rec = [i for i in xyz if i not in pep]; P = np.array([xyz[i] for i in pep]); R = np.array([xyz[i] for i in rec])
    d = np.linalg.norm(P[:, None, :] - R[None, :, :], axis=2)
    return set(np.array(pep)[(d < 10).any(1)].tolist()) | set(np.array(rec)[(d < 10).any(0)].tolist())

def part(pose, keep):
    p = pose.clone()
    for i in range(pose.size(), 0, -1):
        if i not in keep: p.delete_residue_slow(i)
    return p

def dG_once(pose0, pep, iface, extra):
    free = set(iface) | set(extra); c = pose0.clone(); pack(c, free); e_c = SF(c)
    pk = sorted(pep); rk = [i for i in range(1, c.size() + 1) if i not in set(pep)]
    out = []
    for keep in (pk, rk):
        p = part(c, set(keep)); m = {old: new + 1 for new, old in enumerate(keep)}; pack(p, {m[i] for i in free if i in m}); out.append(SF(p))
    return e_c - out[0] - out[1]

def dG(pose0, pep, iface, extra, nrep):
    v = [dG_once(pose0, pep, iface, extra) for _ in range(nrep)]; return float(np.mean(v)), float(np.std(v)), v

def one(args):
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    d, m, nrep = args; rows = []; t0 = time.time()
    try:
        pose = pyrosetta.pose_from_file(os.path.join(d, m["file"]))
    except Exception as e:
        return dict(file=m["file"], error="load: " + str(e)[:80], rows=[])
    relax_info = None
    if os.environ.get("RELAX", "1") == "1":                 # one coordinate-constrained FastRelax of the whole complex, then every slot is scored on the relaxed pose
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import CA_rmsd
        t1 = time.time(); start = pose.clone()
        try:
            fr = FastRelax(SF, 3); fr.constrain_relax_to_start_coords(True); fr.ramp_down_constraints(False); e0 = SF(pose); fr.apply(pose)
            relax_info = dict(e_before=e0, e_after=SF(pose), ca_rmsd=float(CA_rmsd(start, pose)), sec=time.time() - t1)
            os.makedirs(d.rstrip("/") + "_relaxed", exist_ok=True); pose.dump_pdb(os.path.join(d.rstrip("/") + "_relaxed", m["file"]))
        except Exception as e:
            relax_info = dict(error=str(e)[:80]); pose = start
    pdi = pose.pdb_info(); pep = [i for i in range(1, pose.size() + 1) if pdi.chain(i) == m["pep"]]
    if not pep or len(pep) == pose.size(): return dict(file=m["file"], error="no peptide/receptor split", rows=[])
    iface = interface(pose, pep)
    slots = [i for i in pep if pose.residue(i).name3().strip() not in CANON and pose.residue(i).name3().strip() not in CAPS and pose.residue(i).is_protein()]
    for i in slots:                                      # ONE POSITION AT A TIME: only slot i is changed
        nm = pose.residue(i).name3().strip(); par = PARENT.get(nm)
        if par is None: rows.append(dict(file=m["file"], slot=i, ncaa=nm, skip="no parent known")); continue
        try:
            a = dG(pose, pep, iface, [i], nrep); row = dict(file=m["file"], slot=i, pos_in_pep=pep.index(i) + 1, pep_len=len(pep), ncaa=nm, parent=par, dG_native=a[0], sd_native=a[1], refs={})
            for ref in dict.fromkeys([par, "ALA", "GLY"]):      # canonical references, each swap changes ONLY slot i
                mut = pose.clone(); MutateResidue(i, ref).apply(mut); b = dG(mut, pep, iface, [i], nrep)
                row["refs"][ref] = dict(dG=b[0], sd=b[1], gain=b[0] - a[0], backbone_clash=bool(b[0] - a[0] > 20))
            rows.append(row)
        except Exception as e:
            rows.append(dict(file=m["file"], slot=i, ncaa=nm, skip="error " + str(e)[:80]))
    return dict(file=m["file"], rows=rows, sec=time.time() - t0, relax=relax_info)

if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]; nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 3; nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    man = json.load(open(os.path.join(d, "manifest.json"))); only = os.environ.get("ONLY")
    if only: man = [m for m in man if m["file"] in only.split(",")]
    res = []; t0 = time.time()
    with Pool(nproc, initializer=init) as P:
        for r in P.imap_unordered(one, [(d, m, nrep) for m in man]):
            res.append(r); n = sum(1 for x in r["rows"] if "refs" in x)
            print(r["file"], r.get("error", ""), "slots scored", n, "%.0fs" % r.get("sec", 0), flush=True)
            json.dump(res, open(out, "w"))
    print("done %.0fs" % (time.time() - t0))
