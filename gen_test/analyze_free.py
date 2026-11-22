"""Analyse FREE-SEQUENCE designs (spec_free.yaml: no residue identity given in the peptide; NCAA slots keep their token type): does the model choose the right NCAA?
Per run: at the NCAA slots (native residue non-canonical): share of designs that put ANY non-canonical residue, the EXACT native code, the native code's parent amino acid (D-forms), a D-form where the native is a D-form; number of distinct codes and the share of the most common one (mode collapse); at the canonical slots: exact recovery and composition; baselines: most-frequent-native-code and uniform random over the allowed set; structure measures (bonds, close contacts, handedness) from analyze_gen.
usage: analyze_free.py RUN_KEY=RUN_DIR ...  -> diag/free.json"""
import sys, os, re, glob, json, collections
import numpy as np, gemmi
import analyze_gen as A
PARENT = dict(DAL="ALA", DAR="ARG", DAS="ASP", DCY="CYS", DGL="GLU", DGN="GLN", DHI="HIS", DIL="ILE", DLE="LEU", DLY="LYS", DPN="PHE", DPR="PRO", DSG="ASN", DSN="SER", DTH="THR", DTR="TRP", DTY="TYR", DVA="VAL", MED="MET")
def analyse(rdir):
    rec = dict(nc=[], can=[]); names_nc = []
    for f in sorted(glob.glob(f"{rdir}/inference_samples/*/*_generated.cif")):
        ex = os.path.basename(f).replace("_generated.cif", ""); name = re.sub(r"_\d+$", "", ex)
        if name not in A.NAT: A.NAT[name] = A.native(name)
        npep = A.NAT[name][0]; m = gemmi.read_structure(f)[0]; P = [c for c in m if c.name == "P"]
        if not P: continue
        pep = [r for r in P[0] if r.name not in A.WAT]
        if len(pep) != len(npep): continue
        for i, (d, n) in enumerate(zip(pep, npep)):
            row = dict(pocket=name, native=n.name, designed=d.name)
            (rec["nc"] if n.name not in A.CAN else rec["can"]).append(row)
    return rec
def summarize(rec, rows):
    nc, can = rec["nc"], rec["can"]
    if not nc or not can: return None
    out = dict(n_nc_slots=len(nc), n_can_slots=len(can))
    out["any_nc"] = float(np.mean([r["designed"] not in A.CAN for r in nc])); out["exact"] = float(np.mean([r["designed"] == r["native"] for r in nc]))
    dn = [r for r in nc if r["native"] in PARENT]
    out["native_D_slots"] = len(dn); out["D_designed_at_D_slot"] = float(np.mean([r["designed"] in A.DF for r in dn])) if dn else None
    out["parent_match"] = float(np.mean([PARENT.get(r["designed"], r["designed"]) == PARENT[r["native"]] for r in dn])) if dn else None
    cnt = collections.Counter(r["designed"] for r in nc); out["distinct_codes"] = len(cnt); out["top_code"], out["top_code_share"] = cnt.most_common(1)[0][0], cnt.most_common(1)[0][1] / len(nc)
    nat = collections.Counter(r["native"] for r in nc); out["majority_baseline"] = nat.most_common(1)[0][1] / len(nc); out["majority_code"] = nat.most_common(1)[0][0]
    out["canonical_slot_exact"] = float(np.mean([r["designed"] == r["native"] for r in can])); out["canonical_slot_any_nc"] = float(np.mean([r["designed"] not in A.CAN for r in can]))
    cc = collections.Counter(r["designed"] for r in can); out["canonical_top"] = cc.most_common(5)
    out["designed_nc_codes"] = cnt.most_common(8)
    out["bond_ok"] = float(np.nanmean([r["bond_ok"] for r in rows])); out["clash"] = float(np.mean([r["clash"] for r in rows])); out["chir_D"] = float(np.nanmean([r["chir_D"] for r in rows]))
    return out
if __name__ == "__main__":
    res = {}
    for a in sys.argv[1:]:
        key, rd = a.split("="); rec = analyse(rd); rows = A.analyse(rd); s = summarize(rec, rows)
        if s is None: print(key, "no designs"); continue
        res[key] = s
        print(key.ljust(24), "NCAA slots %d: any-NC %.2f exact %.3f parent %s | distinct %d top %s %.2f (majority baseline %s %.2f) | canonical slots exact %.3f any-NC %.2f | bonds %.2f clash %.2f D %.2f" % (
            s["n_nc_slots"], s["any_nc"], s["exact"], "%.2f" % s["parent_match"] if s["parent_match"] is not None else "-", s["distinct_codes"], s["top_code"], s["top_code_share"], s["majority_code"], s["majority_baseline"], s["canonical_slot_exact"], s["canonical_slot_any_nc"], s["bond_ok"], s["clash"], s["chir_D"]))
    json.dump(res, open(f"{A.D}/gen_test/diag/free.json", "w"), indent=1)
