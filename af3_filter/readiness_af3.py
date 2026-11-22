"""Readiness of AF3 for D-amino acids and canonical-like NCAAs, measured on the 310 ipTM-test models (Results.MD 84). usage: python readiness_af3.py <jobs_dir> <out_dir ...> -> readiness.json
Per residue of the peptide (chain B), per CCD code: (1) CA chirality against the reference sign of the chemical definition (ref_sign.json, from the RDKit mol of the code); (2) peptide-bond sanity (C(i)-N(i+1) in 1.20-1.50 A, both neighbours present);
(3) mean pLDDT of the residue's atoms; (4) whether the residue is bonded into the chain at all. Families as in gen_test/score_choice_designs.py."""
import sys, json, glob, os, collections, numpy as np, gemmi
J = sys.argv[1]; OUTS = sys.argv[2:]; man = {m["name"].lower(): m for m in json.load(open(os.path.join(J, "manifest.json")))}
REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_sign.json")))
FAM = {}
for fam, codes in {"D-alpha": "DAL DAR DAS DCY DGL DGN DHI DIL DLE DLY DPN DPR DSG DSN DTH DTR DTY DVA MED", "N-alkyl": "SAR 04Q MAA MLE MVA IML MEA NMM", "alpha,alpha-disubst.": "AIB 2ML DIV 02K 1AC 192 0XL AC5",
                   "beta/gamma": "B3L B3A BAL B3K B3Q XCP STA", "side-chain modified": "23F 4FB FP9 HYP ORN DBU", "PTM": "SEP TPO PTR M3L MLY ALY CGU"}.items():
    for c in codes.split(): FAM[c] = fam
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
fam = lambda c: "canonical L" if c in CAN else FAM.get(c, "other")
def sgn(n, ca, c, x): return float(np.sign(np.dot(np.cross(n - ca, c - ca), x - ca)))
rows = []
for O in OUTS:
    for name, m in man.items():
        d = [x for x in glob.glob(os.path.join(O, name)) if os.path.isdir(x)]
        if not d: continue
        d = d[0]; mf = glob.glob(os.path.join(d, "*_model.cif")); cf = glob.glob(os.path.join(d, "*_confidences.json"))
        if not mf or not cf: continue
        st = gemmi.read_structure(mf[0]); B = st[0]["B"]; cj = json.load(open(cf[0])); ch = np.array(cj["atom_chain_ids"]); pl = np.array(cj["atom_plddts"])[ch == "B"]
        k = 0; res = []
        for r in B:
            a = {x.name: np.array(x.pos.tolist()) for x in r}; res.append((r.name, a, float(np.mean(pl[k:k + len(r)])))); k += len(r)
        for i, (nm, a, pp) in enumerate(res):
            row = dict(group=m["group"], job=name, code=nm, fam=fam(nm), plddt=pp, chir=None, bond_next=None)
            ref = REF.get(nm)
            if ref is not None and all(x in a for x in ("N", "CA", "C", "CB")): row["chir"] = bool(sgn(a["N"], a["CA"], a["C"], a["CB"]) == ref)
            if i + 1 < len(res) and "C" in a and "N" in res[i + 1][1]: row["bond_next"] = bool(1.20 <= np.linalg.norm(a["C"] - res[i + 1][1]["N"]) <= 1.50)
            rows.append(row)
json.dump(rows, open("readiness_rows.json", "w"))
def summ(sel):
    c = [r["chir"] for r in sel if r["chir"] is not None]; b = [r["bond_next"] for r in sel if r["bond_next"] is not None]
    return dict(n=len(sel), chir_ok=float(np.mean(c)) if c else None, n_chir=len(c), bond_ok=float(np.mean(b)) if b else None, n_bond=len(b), plddt=float(np.mean([r["plddt"] for r in sel])))
out = {"by_family": {f: summ([r for r in rows if r["fam"] == f]) for f in sorted(set(r["fam"] for r in rows))}, "by_code": {c: summ([r for r in rows if r["code"] == c]) for c in sorted(set(r["code"] for r in rows))}}
# plddt of a residue relative to the canonical residues of the same peptide
rel = collections.defaultdict(list)
for job in set(r["job"] for r in rows):
    sel = [r for r in rows if r["job"] == job]; can = [r["plddt"] for r in sel if r["fam"] == "canonical L"]
    if not can: continue
    for r in sel:
        if r["fam"] != "canonical L": rel[r["fam"]].append(r["plddt"] - np.mean(can))
out["plddt_minus_canonical_same_peptide"] = {f: float(np.mean(v)) for f, v in rel.items()}
json.dump(out, open("readiness.json", "w"), indent=1)
for f, v in out["by_family"].items(): print("%-22s n %4d | chirality ok %s (n %d) | bond ok %s (n %d) | pLDDT %.0f" % (f, v["n"], "%.3f" % v["chir_ok"] if v["chir_ok"] is not None else "  -  ", v["n_chir"], "%.3f" % v["bond_ok"] if v["bond_ok"] is not None else " - ", v["n_bond"], v["plddt"]))
print("pLDDT minus canonical residues of the same peptide:", {k: round(v, 1) for k, v in out["plddt_minus_canonical_same_peptide"].items()})
print("worst chirality codes:", sorted([(c, round(v["chir_ok"], 2), v["n_chir"]) for c, v in out["by_code"].items() if v["chir_ok"] is not None and v["n_chir"] >= 5 and v["chir_ok"] < 0.95], key=lambda x: x[1]))
print("worst bond codes:", sorted([(c, round(v["bond_ok"], 2), v["n_bond"]) for c, v in out["by_code"].items() if v["bond_ok"] is not None and v["n_bond"] >= 5 and v["bond_ok"] < 0.95], key=lambda x: x[1]))
