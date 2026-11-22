"""Step 1 (Windows python + gemmi): turn complexes (CIF) into Rosetta-ready PDBs + manifest.json.
usage: python prep_pdb.py <in_dir_with_cif> <out_dir>
Rosetta wants its own names for some D-forms (DPN->DPH, DCY->DCS, DSG->DAN, DSN->DSE, DGL->DGU); waters, ligands,
hydrogens and alternate conformations are removed. Peptide chain = 'P' if present (our convention), else the shortest chain
that holds a non-canonical residue. gemmi and PyRosetta crash when imported in the same process, so this is a separate step."""
import glob, os, sys, json, gemmi
MAP = dict(DPN="DPH", DCY="DCS", DSG="DAN", DSN="DSE", DGL="DGU")
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
inp, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); man = []
for f in sorted(glob.glob(os.path.join(inp, "*.cif"))):
    st = gemmi.read_structure(f); st.setup_entities(); st.remove_ligands_and_waters(); st.remove_hydrogens(); st.remove_alternative_conformations()
    ch = {c.name: [MAP.get(r.name, r.name) for r in c] for c in st[0]}
    if "P" in ch: pep = "P"
    else:
        cand = [k for k, v in ch.items() if any(x not in CAN for x in v) and 3 <= len(v) <= 30]
        if not cand: continue
        pep = cand[0]
    rec = [k for k in ch if k != pep and len(ch[k]) > 30]
    if not rec: continue
    for c in st[0]:
        for r in c: r.name = MAP.get(r.name, r.name)
    name = os.path.basename(f).replace(".cif", ".pdb"); st.write_pdb(os.path.join(out, name))
    man.append(dict(file=name, pep=pep, rec=rec))
json.dump(man, open(os.path.join(out, "manifest.json"), "w")); print(len(man), "complexes written to", out)
