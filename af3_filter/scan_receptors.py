"""Which full receptor chains touch each native peptide (Dataset_ncaa/real_peptides)? usage: python scan_receptors.py"""
import gemmi, numpy as np, glob, os
IN = glob.glob(r"C:\Research\RFD4_fine_tune\gen_test\inputs\*.cif")
for f in sorted(IN):
    ex = os.path.basename(f)[:-4]; e, pc = ex.split("_")
    st = gemmi.read_structure(rf"C:\Research\Dataset_ncaa\real_peptides\{e}.cif"); st.setup_entities(); m = st[0]
    pep = [c for c in m if c.name == pc]
    if not pep: print(ex, "no peptide chain"); continue
    pep = pep[0]; pxyz = np.array([a.pos.tolist() for r in pep for a in r if a.element.name != "H"])
    out = []
    for c in m:
        if c.name == pc: continue
        xyz = np.array([a.pos.tolist() for r in c if r.name not in ("HOH",) for a in r if a.element.name != "H"])
        if not len(xyz): continue
        n = int((np.linalg.norm(pxyz[:, None] - xyz[None], axis=-1) < 4.5).any(1).sum())
        if n: out.append((c.name, len([r for r in c if r.name != "HOH"]), n))
    print(ex, "pep", len(pep), [r.name for r in pep], "touching chains (name, len, n pep atoms<4.5):", out)
