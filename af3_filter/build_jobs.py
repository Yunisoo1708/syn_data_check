"""AF3 inputs for the ipTM filter test (Results.MD 83). Single-sequence mode, full receptor chain (the chain that touches the native peptide in Dataset_ncaa/real_peptides),
peptide = chain B with NCAAs as CCD modifications (same encoding as the NCAA structure dataset). Groups per pocket:
  native, control/G3/S1 designs (free choice, 8 each), shuf (native residues shuffled, canonical letters), rnd (native with every NCAA slot replaced by a random NCAA), ala (NCAA->ALA), decoy (native peptide on the receptor of another pocket).
usage: python build_jobs.py <out_dir> [TEMPLATE=0|1 via env]"""
import gemmi, glob, os, sys, json, random, re, numpy as np
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); TEMPL = os.environ.get("TEMPLATE", "0") == "1"
G = r"C:\Research\RFD4_fine_tune\gen_test"; RP = r"C:\Research\Dataset_ncaa\real_peptides"
CAN = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
SKIP = {"9B4J_A", "9B4K_A", "9B4L_E"}; WAT = {"HOH", "WAT"}
NC_POOL = "SAR MAA DIL 4FB 04Q 23F DAL SEP DTY DPR DHI 1AC XCP TPO DSG DAS DVA DPN DAR DGL BAL".split()
rng = random.Random(7)
def recv(ex):
    e, pc = ex.split("_"); st = gemmi.read_structure(rf"{RP}\{e}.cif"); st.setup_entities(); m = st[0]
    pep = [c for c in m if c.name == pc][0]; pxyz = np.array([a.pos.tolist() for r in pep for a in r if a.element.name != "H"]); best = None
    for c in m:
        if c.name == pc: continue
        xyz = np.array([a.pos.tolist() for r in c if r.name not in WAT for a in r if a.element.name != "H"])
        if not len(xyz): continue
        n = int((np.linalg.norm(pxyz[:, None] - xyz[None], axis=-1) < 4.5).any(1).sum())
        if best is None or n > best[0]: best = (n, c)
    c = best[1]; seq = ""; rs = []
    for r in c:
        if r.name in WAT or r.entity_type != gemmi.EntityType.Polymer: continue
        t = gemmi.find_tabulated_residue(r.name); l = CAN.get(r.name) or (t.one_letter_code.upper() if t and t.one_letter_code and t.one_letter_code != "?" else "X")
        seq += l; rs.append(r)
    return c.name, seq, c, st
def pep_entry(codes):
    seq = ""; mods = []
    for i, c in enumerate(codes, 1):
        if c in CAN: seq += CAN[c]
        else: seq += "A"; mods.append({"ptmType": c, "ptmPosition": i})
    d = {"id": "B", "sequence": seq, "unpairedMsa": "", "pairedMsa": "", "templates": []}
    if mods: d["modifications"] = mods
    return {"protein": d}
def rec_entry(seq, c, st):
    d = {"id": "A", "sequence": seq, "unpairedMsa": "", "pairedMsa": "", "templates": []}
    if TEMPL:
        s2 = gemmi.Structure(); s2.cell = st.cell; s2.spacegroup_hm = st.spacegroup_hm; mm = gemmi.Model("1"); ch = gemmi.Chain("A")
        k = 0
        for r in c:
            if r.name in WAT or r.entity_type != gemmi.EntityType.Polymer: continue
            r2 = r.clone(); r2.seqid = gemmi.SeqId(k + 1, " "); ch.add_residue(r2); k += 1
        mm.add_chain(ch); s2.add_model(mm); s2.setup_entities()
        REV = "\nloop_\n_pdbx_audit_revision_history.ordinal\n_pdbx_audit_revision_history.data_content_type\n_pdbx_audit_revision_history.major_revision\n_pdbx_audit_revision_history.minor_revision\n_pdbx_audit_revision_history.revision_date\n1 'Structure model' 1 0 2000-01-01\n"
        d["templates"] = [{"mmcif": s2.make_mmcif_document().as_string() + REV, "queryIndices": list(range(k)), "templateIndices": list(range(k))}]
    return {"protein": d}
def names(grp):
    return {"control": "control_choice_run8", "G3": "G3_bal_choice_run8", "S1": "S1_readout_choice_run8"}[grp]
R = {}; native = {}
for f in sorted(glob.glob(G + r"\inputs\*.cif")):
    ex = os.path.basename(f)[:-4]
    if ex in SKIP: continue
    R[ex] = recv(ex); m = gemmi.read_structure(f)[0]; native[ex] = [r.name for r in [c for c in m if c.name == "P"][0] if r.name not in WAT]
man = []
def add(grp, ex, k, codes, rex=None):
    rex = rex or ex; cname, seq, c, st = R[rex]; name = f"{grp}_{ex.lower()}_{k}"
    job = {"name": name, "modelSeeds": [1], "sequences": [rec_entry(seq, c, st), pep_entry(codes)], "dialect": "alphafold3", "version": 1}
    json.dump(job, open(os.path.join(OUT, name + ".json"), "w")); man.append(dict(name=name, group=grp, ex=ex, rep=k, rec_ex=rex, codes=codes, rec_len=len(seq)))
for ex in R:
    add("native", ex, 0, native[ex]); cn = native[ex]
    for k in range(2):
        s = [x for x in cn]; rng.shuffle(s); add("shuf", ex, k, [x if x in CAN else "ALA" for x in s])
        add("rnd", ex, k, [x if x in CAN else rng.choice(NC_POOL) for x in cn])
    add("ala", ex, 0, [x if x in CAN else "ALA" for x in cn])
ks = list(R)
for i, ex in enumerate(ks): add("decoy", ex, 0, native[ex], rex=ks[(i + 3) % len(ks)])
for grp in ("control", "G3", "S1"):
    for f in sorted(glob.glob(G + rf"\out_local\{names(grp)}\inference_samples\*\*_generated.cif")):
        e = os.path.basename(f).replace("_generated.cif", ""); ex = re.sub(r"_\d+$", "", e); k = int(e.rsplit("_", 1)[1])
        if ex not in R: continue
        m = gemmi.read_structure(f)[0]; P = [c for c in m if c.name == "P"]
        if not P: continue
        add(grp, ex, k, [r.name for r in P[0] if r.name not in WAT])
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"))
import collections; print(len(man), "jobs", dict(collections.Counter(m["group"] for m in man)), "template" if TEMPL else "no template", "| receptor lengths", {k: len(v[1]) for k, v in R.items()}, "| X in receptors", {k: v[1].count("X") for k, v in R.items() if "X" in v[1]})
