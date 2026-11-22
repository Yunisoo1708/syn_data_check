"""Extra real test complexes (TEST ONLY): the Hamdani non-natural-amino-acid peptide complexes whose entry is in the frozen test split v4 (and not in the first 103-complex benchmark).
Peptide chain = the chain in the file name (renamed P, all its residues kept as ATOM records so the non-canonical residues stay in the polymer); receptor = every other protein chain, cropped to the residues with an atom within POCKET A of the peptide.
usage: crop_real_ham.py OUT_DIR   -> OUT_DIR/{ENTRY}_{CHAIN}_real.cif and manifest_real_ham.json (peptide codes per file)"""
import os, sys, json, collections, gemmi
OUT = sys.argv[1]; POCKET = 12.0
RR = "/mnt/c/Research" if os.name != "nt" else "C:/Research"
D = RR + "/RFD4_fine_tune/splits"; P = D + "/hamdani_dataset/Peptide_protein_2026/Non_natural_amino_acids"
sp = json.load(open(D + "/split_v4.json")); soe = sp["split_of_entry"]
bench = {f.split("_")[0].upper() for f in os.listdir(RR + "/RFD4_fine_tune/data/real_test_ok")}
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
WAT = {"HOH", "WAT", "DOD"}
os.makedirs(OUT, exist_ok=True); man = {}; rep = collections.Counter()
for f in sorted(os.listdir(P)):
    if not f.endswith(".pdb"): continue
    ent, ch = f[:-4].split("_", 1); ent = ent.upper(); ch = ch[0]
    if soe.get(ent) != "test" or ent in bench: rep["skipped (not test or already in benchmark)"] += 1; continue
    lines = []
    for ln in open(f"{P}/{f}", errors="ignore"):
        if ln[:6] in ("ATOM  ", "HETATM"):
            if ln[17:20].strip() in WAT: continue
            if ln[21] == ch: ln = "ATOM  " + ln[6:21] + "P" + ln[22:]
            elif ln[:6] == "HETATM": continue  # receptor hetero groups (ligands, ions) are not part of the benchmark
            lines.append(ln.rstrip("\n"))
        elif ln[:3] == "TER": lines.append(ln.rstrip("\n"))
    try:
        st = gemmi.read_pdb_string("\n".join(lines) + "\nEND\n"); st.setup_entities(); m = st[0]
        pep = [c for c in m if c.name == "P"]
        if not pep or len(pep[0]) < 2: rep["no peptide chain"] += 1; continue
        ns = gemmi.NeighborSearch(m, st.cell, 6).populate(); keep = set()
        for r in pep[0]:
            for a in r:
                for mk in ns.find_atoms(a.pos, "\0", radius=POCKET):
                    cra = mk.to_cra(m); keep.add((cra.chain.name, str(cra.residue.seqid)))
        new = gemmi.Structure(); new.cell = st.cell; new.spacegroup_hm = st.spacegroup_hm; new.name = f[:-4]; nm = gemmi.Model("1")
        for c in m:
            if c.name == "P": nm.add_chain(c); continue
            nc = gemmi.Chain(c.name)
            for r in c:
                if (c.name, str(r.seqid)) in keep: nc.add_residue(r)
            if len(nc): nm.add_chain(nc)
        if len([c for c in nm if c.name != "P"]) == 0: rep["no receptor in contact"] += 1; continue
        new.add_model(nm); new.setup_entities(); new.assign_label_seq_id(True)
        name = f"{ent}_{ch}_real"; new.make_mmcif_document().write_file(f"{OUT}/{name}.cif")
        man[name] = dict(entry=ent, chain=ch, peptide=[r.name for r in pep[0]], noncanonical=sorted({r.name for r in pep[0] if r.name not in CAN}), rec_res=sum(len(c) for c in nm if c.name != "P"))
        rep["written"] += 1
    except Exception as e:
        rep["error: " + str(e)[:60]] += 1
json.dump(man, open(f"{OUT}/manifest_real_ham.json", "w"))
print(dict(rep))
