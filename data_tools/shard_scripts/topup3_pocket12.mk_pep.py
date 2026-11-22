"""Peptide-only (chain P) CIFs of the converted complexes, in the pool format (one chain 'A'), for the single-chain Stage 0/1 tools. usage: make_pep_cifs.py SRC_DIR OUT_DIR"""
import sys, os, glob
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True); n = 0
for f in sorted(glob.glob(src + "/*_tu3.cif")):
    atoms = []; seq = {}
    for l in open(f):
        if l.startswith("ATOM"):
            t = l.split()
            if t[6] == "P":
                atoms.append(t)
    if not atoms: continue
    resorder = []
    for t in atoms:
        k = int(t[8])
        if k not in seq: seq[k] = t[5]; resorder.append(k)
    L = ["data_model", "_entry.id model", "", "_cell.entry_id model", "_cell.length_a 1", "_cell.length_b 1", "_cell.length_c 1", "_cell.angle_alpha 90", "_cell.angle_beta 90", "_cell.angle_gamma 90", "", "_symmetry.entry_id model", "_symmetry.space_group_name_H-M ''", "",
         "loop_", "_entity.id", "_entity.type", "1 polymer", "", "loop_", "_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_strand_id", "_entity_poly.pdbx_seq_one_letter_code", "1 polypeptide(L) A " + "".join(f"({seq[k]})" for k in resorder), "",
         "loop_", "_struct_asym.id", "_struct_asym.entity_id", "A 1", "", "loop_", "_entity_poly_seq.entity_id", "_entity_poly_seq.num", "_entity_poly_seq.mon_id", "_entity_poly_seq.hetero"]
    L += [f"1 {i + 1} {seq[k]} n" for i, k in enumerate(resorder)]
    L += ["", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
          "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv", "_atom_site.pdbx_formal_charge", "_atom_site.auth_seq_id", "_atom_site.auth_asym_id", "_atom_site.pdbx_PDB_model_num"]
    for i, t in enumerate(atoms):
        L.append(f"ATOM {i + 1} {t[2]} {t[3]} . {t[5]} A 1 {resorder.index(int(t[8])) + 1} ? {t[10]} {t[11]} {t[12]} 1 20.00 ? {resorder.index(int(t[8])) + 1} A 1")
    open(os.path.join(out, os.path.basename(f).replace("_tu3.cif", "_pep.cif")), "w").write(chr(10).join(L) + chr(10)); n += 1
print("written", n)
