"""Windows python + gemmi: Rosetta-ready PDBs of the free-choice designs (3 denoisers x 88) and of the 11 NATIVE pockets (gen_test/inputs), plus manifest.json (group, example, peptide chain, designed codes).
D-forms are renamed to Rosetta's names (DPN->DPH, DCY->DCS, DSG->DAN, DSN->DSE, DGL->DGU). usage: python prep_designs.py <out_dir>"""
import glob, os, sys, json, re, gemmi
MAP = dict(DPN="DPH", DCY="DCS", DSG="DAN", DSN="DSE", DGL="DGU")
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
G = r"C:\Research\RFD4_fine_tune\gen_test"; out = sys.argv[1]; os.makedirs(out, exist_ok=True); man = []
SRC = [("control", glob.glob(G + r"\out_local\control_choice_run8\inference_samples\*\*_generated.cif")), ("G3", glob.glob(G + r"\out_local\G3_bal_choice_run8\inference_samples\*\*_generated.cif")),
       ("S1", glob.glob(G + r"\out_local\S1_readout_choice_run8\inference_samples\*\*_generated.cif")), ("native", glob.glob(G + r"\inputs\*.cif"))]
for grp, files in SRC:
    for f in sorted(files):
        ex = os.path.basename(f).replace("_generated.cif", "").replace(".cif", "")
        st = gemmi.read_structure(f); st.setup_entities(); st.remove_ligands_and_waters(); st.remove_hydrogens(); st.remove_alternative_conformations()
        names = {c.name: [MAP.get(r.name, r.name) for r in c] for c in st[0]}
        if "P" not in names: continue
        for c in st[0]:
            for r in c: r.name = MAP.get(r.name, r.name)
        name = f"{grp}__{ex}.pdb"; st.write_pdb(os.path.join(out, name))
        man.append(dict(file=name, group=grp, ex=ex, pep="P", rec=[k for k in names if k != "P"], codes=names["P"]))
json.dump(man, open(os.path.join(out, "manifest.json"), "w")); print(len(man), "structures", {g: sum(1 for m in man if m["group"] == g) for g in ("control", "G3", "S1", "native")})
