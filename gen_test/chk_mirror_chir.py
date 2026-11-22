import gemmi, glob, random, numpy as np
CAN = set("ALA ARG ASN ASP CYS GLN GLU HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
DF = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split())
def xyz(r, n):
    a = r.find_atom(n, "*"); return np.array(a.pos.tolist()) if a else None
def chir(r):
    n, ca, c, cb = (xyz(r, k) for k in ("N", "CA", "C", "CB"))
    if any(v is None for v in (n, ca, c, cb)): return None
    return float(np.dot(np.cross(n - ca, c - ca), cb - ca))
fs = sorted(glob.glob("/mnt/c/Research/RFD4_fine_tune/data/mirror_complexes/mirc_*.cif")); random.Random(0).shuffle(fs)
okD = nD = okL = nL = 0; nf = 0; peps = []
for f in fs[:150]:
    try: m = gemmi.read_structure(f)[0]
    except Exception as e: print("parse error", f, e); continue
    nf += 1
    for c in m:
        for r in c:
            v = chir(r)
            if v is None: continue
            if r.name in DF: nD += 1; okD += v < 0
            elif r.name in CAN: nL += 1; okL += v > 0
print("files", nf, "| residues with D code and D handedness", okD, "/", nD, "| residues with L code and L handedness", okL, "/", nL)
m = gemmi.read_structure(fs[0])[0]; print(fs[0].split("/")[-1], [(c.name, len(c), c[0].name) for c in m][:6])
