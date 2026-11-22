"""Mirror-image augmentation for the decoder (18 Nov 2026), text version that keeps the pool files' own CIF layout: the mirror image of an L-peptide is a valid D-peptide.
Take pool peptides of the TRAIN split of the decoder's 80/10/10 pool split (so no mirrored copy of a validation/test peptide is trained on), keep those whose residues are all canonical or D-alpha (19 L/D pairs; GLY stays),
negate x of every atom and swap every residue code (atom rows, entity_poly sequence, chem_comp, struct_conn) for its mirror counterpart. The 13 new D-forms thereby get real D-peptide geometry (backbone included).
usage: mirror_pool.py POOL_DIR OUT_DIR N_MAX   -> OUT_DIR/mir_<name>.cif and OUT_DIR/mirror_list.txt (absolute paths)"""
import sys, re, random
from pathlib import Path
pool, out, nmax = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
PAIRS = {"ALA": "DAL", "ARG": "DAR", "ASN": "DSG", "ASP": "DAS", "CYS": "DCY", "GLN": "DGN", "GLU": "DGL", "HIS": "DHI", "ILE": "DIL", "LEU": "DLE", "LYS": "DLY", "MET": "MED", "PHE": "DPN", "PRO": "DPR",
         "SER": "DSN", "THR": "DTH", "TRP": "DTR", "TYR": "DTY", "VAL": "DVA"}
MAP = dict(PAIRS); MAP.update({v: k for k, v in PAIRS.items()}); MAP["GLY"] = "GLY"
ONE = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU", G="GLY", H="HIS", I="ILE", L="LEU", K="LYS", M="MET", F="PHE", P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")
THREE2ONE = {v: k for k, v in ONE.items()}
files = sorted(pool.glob("*.cif")); random.Random(0).shuffle(files)          # the same shuffle as decoder_fine_tune_v4.py
n_val = max(1, int(0.1 * len(files))); n_test = max(1, int(0.1 * len(files))); train = files[n_val + n_test:]
random.Random(1).shuffle(train); written = 0; skipped = 0; lst = []
tok = re.compile(r"\(([A-Z0-9]{1,3})\)|([A-Z])")
def seq_field(s):
    names = [m.group(1) or ONE[m.group(2)] for m in tok.finditer(s)]
    return "".join(THREE2ONE[MAP[n]] if MAP[n] in THREE2ONE else f"({MAP[n]})" for n in names)
for f in train:
    if written >= nmax: break
    try:
        lines = f.read_text().split("\n"); res = set(); bad = False
        for ln in lines:
            if ln.startswith(("ATOM", "HETATM")): res.add(ln.split()[5])
        if not res <= set(MAP): skipped += 1; continue
        new = []
        for ln in lines:
            if ln.startswith(("ATOM", "HETATM")):
                p = ln.split(); p[5] = MAP[p[5]]; p[10] = repr(-float(p[10])); new.append(" ".join(p))
            elif ln[:1].isdigit() and "polypeptide(" in ln:
                p = ln.split(None, 3); p[3] = seq_field(p[3]); new.append(" ".join(p))
            elif ln[:1] in ("_", "l", "d", "#") or not ln.strip(): new.append(ln)
            else: new.append(re.sub(r"\b([A-Z0-9]{3})\b", lambda m: MAP.get(m.group(1), m.group(1)), ln))
        o = out / ("mir_" + f.name); o.write_text("\n".join(new)); lst.append(str(o.resolve())); written += 1
    except Exception as e:
        skipped += 1
open(out / "mirror_list.txt", "w").write("\n".join(lst))
print("pool", len(files), "train split", len(train), "mirrored", written, "skipped (non-mirrorable residues or errors)", skipped)
