"""Reference CA chirality sign of each CCD code from the chemical definition (RDKit mol of the Boltz cache), as in multigpu/chir_sign.py. WSL python: python3 ref_sign.py codes_used.txt out.json"""
import pickle, sys, json, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
D = "/mnt/c/Research/Dataset_ncaa"
def sgn(n, ca, c, x): return float(np.sign(np.dot(np.cross(n - ca, c - ca), x - ca)))
out = {}
for code in open(sys.argv[1]).read().split():
    try:
        m = pickle.load(open(f"{D}/boltz_env/boltz_cache/mols/{code}.pkl", "rb")); nm = [a.GetProp("name") for a in m.GetAtoms()]; mh = Chem.AddHs(m)
        if AllChem.EmbedMolecule(mh, randomSeed=7) != 0: out[code] = None; continue
        pos = mh.GetConformer().GetPositions(); idx = {n: i for i, n in enumerate(nm)}
        out[code] = sgn(pos[idx["N"]], pos[idx["CA"]], pos[idx["C"]], pos[idx["CB"]]) if all(k in idx for k in ("N", "CA", "C", "CB")) else None
    except Exception as e: out[code] = None
json.dump(out, open(sys.argv[2], "w")); print(len(out), "codes,", sum(v is not None for v in out.values()), "with a reference sign")
