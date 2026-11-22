"""Real peptide-protein complexes of the real benchmark (test only) -> pocket-cropped CIFs in the same layout as the training CIFs
(peptide chain P whole + receptor residues with any atom within POCKET A of the peptide). usage: crop_real_test.py IN_DIR OUT_DIR items.tsv"""
import sys, os, csv, gemmi
IN, OUT, ITEMS = sys.argv[1:4]; POCKET = 12.0
ok = [r["item_id"] for r in csv.DictReader(open(ITEMS), delimiter="\t") if r["status"] == "ok"]
os.makedirs(OUT, exist_ok=True); n = 0; rep = []
for it in ok:
    p = f"{IN}/{it}.cif"
    if not os.path.exists(p): rep.append((it, "missing")); continue
    st = gemmi.read_structure(p); st.setup_entities(); m = st[0]
    pep = [c for c in m if c.name == "P"]
    if not pep: rep.append((it, "no chain P")); continue
    ns = gemmi.NeighborSearch(m, st.cell, 6).populate()
    keep = set()
    for r in pep[0]:
        for a in r:
            for mk in ns.find_atoms(a.pos, "\0", radius=POCKET):
                cra = mk.to_cra(m); keep.add((cra.chain.name, str(cra.residue.seqid)))
    new = gemmi.Structure(); new.cell = st.cell; new.spacegroup_hm = st.spacegroup_hm; new.name = it; nm = gemmi.Model("1")
    for c in m:
        if c.name == "P": nm.add_chain(c); continue
        nc = gemmi.Chain(c.name)
        for r in c:
            if (c.name, str(r.seqid)) in keep and r.het_flag != "H": nc.add_residue(r)
        if len(nc): nm.add_chain(nc)
    new.add_model(nm); new.setup_entities(); new.assign_label_seq_id() if hasattr(new, "assign_label_seq_id") else None
    new.make_mmcif_document().write_file(f"{OUT}/{it}_real.cif"); n += 1
    rep.append((it, f"ok pep {len(pep[0])} rec {sum(len(c) for c in nm if c.name != 'P')}"))
print("written", n, "of", len(ok)); print("\n".join(f"{a} {b}" for a, b in rep[:5])); print([x for x in rep if not x[1].startswith("ok")])
