"""Step 2 (box, atomworks): build spec_gen.yaml from the parsed input CIFs (chain ids and residue numbers exactly as the generator sees them)."""
import sys, os, glob, json, yaml, warnings
import numpy as np
warnings.filterwarnings("ignore")
from atomworks.io.parser import parse
G = os.path.expanduser("~/rfd4_trial/gen_test"); CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
spec = {}; info = {}
for f in sorted(glob.glob(f"{G}/inputs/*.cif")):
    name = os.path.basename(f)[:-4]; aa = parse(f)["assemblies"]["1"][0]
    chains = [str(c) for c in dict.fromkeys(aa.chain_id.tolist())]; pcs = [c for c in chains if c.startswith("P")]
    if len(pcs) != 1: print("skip", name, chains); continue
    pc = pcs[0]; rec = [c for c in chains if c != pc]
    rids = sorted(set(aa.res_id[aa.chain_id == pc].tolist())); L = len(rids)
    nrec = sum(len(set(aa.res_id[aa.chain_id == c].tolist())) for c in rec)
    if nrec < 20 or rids != list(range(1, L + 1)): print("skip", name, nrec, rids[:5]); continue
    segs = []
    for c in rec:
        ids = sorted(set(aa.res_id[aa.chain_id == c].tolist())); runs = []; s0 = prev = ids[0]
        for v in ids[1:] + [None]:
            if v is None or v != prev + 1: runs.append((s0, prev)); s0 = v
            prev = v
        segs.append(",".join(f"{c}{a}-{b}" if a != b else f"{c}{a}" for a, b in runs))
    names = [str(aa.res_name[(aa.chain_id == pc) & (aa.res_id == r)][0]) for r in rids]
    nc = [r for r, n in zip(rids, names) if n not in CAN]
    sel_rec = " or ".join(f"chain_id == '{c}'" for c in rec)
    # hotspots = receptor residues with a heavy atom within 4.5 A of the native peptide (where the peptide should bind)
    pm = aa.chain_id == pc; pxyz = aa.coord[pm]; rm = ~pm
    rxyz = aa.coord[rm]; pxyz = pxyz[~np.isnan(pxyz).any(1)]
    dd = np.linalg.norm(rxyz[:, None, :] - pxyz[None], axis=-1); dd = np.where(np.isnan(dd), 1e9, dd); dmin = dd.min(1)
    hs = sorted({(str(c), int(r)) for c, r, d in zip(aa.chain_id[rm], aa.res_id[rm], dmin) if d < 4.5})
    sel_hot = " or ".join(f"(chain_id == '{c}' and res_id == {r})" for c, r in hs)
    spec[name] = {"input": f, "contig": ",/0,".join(segs) + f",/0,{pc}1-{L}",
                  "conditions": {"C_CRD": {"true": [{"select": sel_rec}]}, "C_SEQ": {"true": [{"select": sel_rec + " or " + " or ".join(f"(chain_id == '{pc}' and res_id == {r})" for r in nc)}]}, "hotspot": {"true": [{"select": sel_hot}]}}}
    info[name] = dict(peptide=names, nc_positions=nc, peptide_chain=pc, receptor_residues=nrec)
yaml.safe_dump(spec, open(f"{G}/spec_gen.yaml", "w"), sort_keys=False, width=100000); json.dump(info, open(f"{G}/info.json", "w"), indent=1)
print(len(spec), "examples:", {k: (v["peptide_chain"], len(v["peptide"]), v["nc_positions"]) for k, v in info.items()})
