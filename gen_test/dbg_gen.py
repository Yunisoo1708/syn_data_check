import gemmi, numpy as np, glob
f = sorted(glob.glob("/mnt/c/Research/RFD4_fine_tune/gen_test/out_local/smoke/inference_samples/*/*_generated.cif"))[0]
st = gemmi.read_structure(f); m = st[0]; P = [c for c in m if c.name == "P"][0]
def xyz(r, n):
    a = r.find_atom(n, "*"); return np.array(a.pos.tolist()) if a else None
for r in P: print(r.name, [a.name for a in r][:12], len(r))
for a, b in zip(list(P)[:-1], list(P)[1:]):
    c, n = xyz(a, "C"), xyz(b, "N"); ca1, ca2 = xyz(a, "CA"), xyz(b, "CA")
    print(a.name, b.name, "C-N", None if c is None or n is None else round(float(np.linalg.norm(c - n)), 2), "CA-CA", round(float(np.linalg.norm(ca1 - ca2)), 2))
