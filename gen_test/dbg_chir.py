import sys; sys.path.insert(0, "/mnt/c/Research/RFD4_fine_tune/gen_test")
import analyze_gen as A, numpy as np, gemmi, glob
for name in ["7XAU_F", "7PIU_P", "8X8N_C"]:
    pep, rec, sg = A.native(name)
    print(name, "L-reference sign", sg, [(r.name, round(A.chir(r), 2) if A.chir(r) is not None else None) for r in pep])
for f in sorted(glob.glob("/mnt/c/Research/RFD4_fine_tune/gen_test/out_local/smoke/inference_samples/*/*_generated.cif")):
    m = gemmi.read_structure(f)[0]; P = [c for c in m if c.name == "P"][0]
    print(f.split("/")[-1], [(r.name, round(A.chir(r), 2) if A.chir(r) is not None else None) for r in P])
