"""Training copies of the pilot complexes: residues of the peptide chain P that the 176-class vocabulary cannot represent (13 D-forms) or that failed Stage 0 get occupancy 0
(RFD4 treats them as context only: no loss, bonds not validated). A complex is kept if at least 50% of its peptide residues stay resolved and it is in usable.json.
usage: mask_pilot_cifs.py SRC_CIF_DIR OUT_DIR USABLE.json MANIFEST.json VERDICTS_PEP.tsv VOCAB_YAML"""
import sys, os, re, json, csv, collections
src, out, usable, manifest, verd, yaml = sys.argv[1:7]
os.makedirs(out + "/cif", exist_ok=True)
vocab = set(re.findall(r"\b([A-Z0-9]{3})\b", open(yaml).read()))
M = {m["entry"]: m for m in json.load(open(manifest))}
U = set(json.load(open(usable))["usable"])
bad = collections.defaultdict(set)   # entry -> set of peptide residue index (1-based) that failed stage 0
for r in csv.DictReader(open(verd), delimiter="\t"):
    if r["verdict"] != "ok":
        bad[r["file"].replace("_pep.cif", "")].add(int(r["resnum"]))
kept = dropped = 0; res_tot = res_masked = 0; rows = []
for iid in sorted(U):
    seq = M[iid]["sequence"]
    mask = {k + 1 for k, c in enumerate(seq) if c not in vocab and c not in set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())} | bad.get(iid, set())
    frac = 1 - len(mask) / len(seq)
    rows.append((iid, len(seq), len(mask), int(frac >= 0.5)))
    if frac < 0.5:
        dropped += 1; continue
    lines = []
    for l in open(f"{src}/{iid}_bl2.cif"):
        if l.startswith("ATOM"):
            t = l.split()
            if t[6] == "P" and int(t[8]) in mask:
                t[13] = "0"; l = " ".join(t) + "\n"
        lines.append(l)
    open(f"{out}/cif/{iid}_bl2.cif", "w").write("".join(lines)); kept += 1; res_tot += len(seq); res_masked += len(mask)
open(out + "/masked_report.tsv", "w").write("entry\tn_peptide_res\tn_masked\tkept\n" + "\n".join("\t".join(map(str, r)) for r in rows) + "\n")
print(f"kept {kept} dropped {dropped}; masked {res_masked} of {res_tot} peptide residues in the kept complexes")
