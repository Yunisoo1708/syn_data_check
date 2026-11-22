"""Build design inputs for the generation test from the REAL test complexes (test only): receptor pocket kept as fixed context, peptide = designed chain whose non-canonical residues keep their (given) identity; one YAML spec for all complexes.
usage (WSL, gemmi): python prep_inputs.py N"""
import gemmi, glob, os, sys, json, collections
D = "/mnt/c/Research/RFD4_fine_tune"; OUT = f"{D}/gen_test/inputs"; N = int(sys.argv[1])
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
DF19 = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split()); GEN12 = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split()); OK = CAN | DF19 | GEN12
WAT = {"HOH", "WAT", "DOD"}
sel = []
for f in sorted(glob.glob(f"{D}/data/real_test_cif/*.cif")):
    st = gemmi.read_structure(f); m = st[0]
    P = [c for c in m if c.name == "P"]
    if not P: continue
    pep = [r for r in P[0] if r.name not in WAT]
    names = [r.name for r in pep]
    nc = [i for i, n in enumerate(names) if n not in CAN]
    if not (6 <= len(pep) <= 16) or not nc or any(n not in OK for n in names): continue
    rec = [c for c in m if c.name != "P"]
    nrec = sum(len([r for r in c if r.name not in WAT]) for c in rec)
    if nrec > 140: continue
    sel.append((f, names, nc, nrec))
print("candidates", len(sel))
sel = sel[:N]; spec = {}; info = {}
for f, names, nc, nrec in sel:
    st = gemmi.read_structure(f); m = st[0]; name = os.path.basename(f).replace("_real.cif", "")
    new = gemmi.Structure(); new.spacegroup_hm = "P 1"; new.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90); mm = gemmi.Model("1")
    segs = []; POOL = iter('ABCDEFGHIJKLMNOQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    for c in m:
        if c.name == "P":
            ch = gemmi.Chain("P"); k = 0
            for r in c:
                if r.name in WAT: continue
                k += 1; r2 = gemmi.Residue(); r2.name = r.name; r2.seqid = gemmi.SeqId(k, " "); r2.het_flag = "A" if r.name in CAN else "H"; r2.subchain = "P"
                for a in r: r2.add_atom(a)
                ch.add_residue(r2)
            mm.add_chain(ch)
        else:
            # every run of consecutive residue numbers (pocket fragment) becomes its own chain, so no bond is implied between fragments
            runs = []; cur = []; prev = None
            for r in c:
                if r.name in WAT: continue
                brk = prev is not None and r.seqid.num != prev + 1
                if cur and not brk:   # consecutive numbers but no real peptide bond (C to N farther than 2 A): also a fragment break
                    c0, n1 = cur[-1].find_atom("C", "*"), r.find_atom("N", "*")
                    brk = c0 is None or n1 is None or c0.pos.dist(n1.pos) > 2.0
                if cur and brk: runs.append(cur); cur = []
                cur.append(r); prev = r.seqid.num
            if cur: runs.append(cur)
            for run in runs:
                cid = next(POOL); ch = gemmi.Chain(cid)
                for r in run: r.subchain = cid; ch.add_residue(r)
                mm.add_chain(ch)
    new.add_model(mm); new.setup_entities(); new.assign_label_seq_id(True)
    path = f"{OUT}/{name}.cif"; new.make_mmcif_document().write_file(path)
    info[name] = dict(peptide=names, nc_positions=[i + 1 for i in nc], receptor_residues=nrec)
print(len(info), "input CIFs written")
