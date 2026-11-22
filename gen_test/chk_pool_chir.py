import gemmi, glob, random, numpy as np, collections
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
DF = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR DAR DPR DAL DVA DLE DPN".split())
def xyz(r, n):
    a = r.find_atom(n, "*"); return np.array(a.pos.tolist()) if a else None
def chir(r):
    n, ca, c, cb = (xyz(r, k) for k in ("N", "CA", "C", "CB"))
    if any(v is None for v in (n, ca, c, cb)): return None
    return float(np.dot(np.cross(n - ca, c - ca), cb - ca))
files = sorted(glob.glob("/mnt/c/Research/RFD4_fine_tune/data/train_all_cif_clean2/*.cif")); random.Random(0).shuffle(files)
tot = collections.Counter(); ok = collections.Counter(); mag = []
for f in files[:1500]:
    try: m = gemmi.read_structure(f)[0]
    except Exception: continue
    Ls = [chir(r) for c in m for r in c if r.name in CAN and r.name != "GLY"]; Ls = [v for v in Ls if v is not None]
    if len(Ls) < 5: continue
    sg = np.sign(np.median(Ls))
    for c in m:
        for r in c:
            if r.name in DF:
                v = chir(r)
                if v is None: continue
                tot[r.name] += 1; ok[r.name] += (np.sign(v) == -sg); mag.append(abs(v))
T = sum(tot.values()); O = sum(ok.values())
print("D-residues checked", T, "with the right C-alpha handedness", O, f"({100*O/T:.1f}%)", "| median |signed volume|", round(float(np.median(mag)), 2), "(native ~2.5)")
print({k: f"{ok[k]}/{tot[k]}" for k in sorted(tot, key=lambda k: -tot[k])[:10]})
