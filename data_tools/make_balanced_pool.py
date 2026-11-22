"""Build a class-balanced training subset of the pool train_all_cif_clean2: every non-canonical residue type that was swapped in (25 codes: 12 general + 13 D-forms) gets a similar number of peptide positions.
Greedy: complexes are visited from the rarest codes up; a complex is taken only if no code of it would exceed the cap. The held-out split (family hash split, same as the trainers) is copied unchanged so the same held-out complexes are scored.
usage: python make_balanced_pool.py CAP   -> train_all_cif_balanced{CAP}/ (+ balanced_list_{CAP}.txt, balanced_{CAP}_report.json)"""
import os, sys, json, hashlib, random, shutil, collections
D = "C:/Research/RFD4_fine_tune"; CAP = int(sys.argv[1]); POOL = f"{D}/data/train_all_cif_clean2"
CANON = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
DF13 = "DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split(); GEN12 = "23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split(); BAL = set(DF13 + GEN12)
MAN = ["pilot_pocket12", "topup_pocket12", "topup2_pocket12", "topup3_pocket12", "gen_pocket12", "gen_pocket12b", "gen_pocket12c", "gen_pocket12d", "gen_pocket12e", "top_pocket12a", "top_pocket12b", "top_pocket12c", "top_pocket12d", "bal_pocket12a", "bal_pocket12b"]
rec = {}
for d in MAN:
    for m in json.load(open(f"{D}/data/{d}/manifest.json")): rec[os.path.basename(m["cif"])] = m["sequence"]
G = json.load(open(f"{D}/data/lists/complex_groups.json"))
gk = lambda f: G.get(f.split("_")[0].upper(), f.split("_")[0].upper())
isval = lambda f: int(hashlib.sha1(("fmval" + gk(f)).encode()).hexdigest(), 16) % 100 < 12
files = sorted(f for f in os.listdir(POOL) if f.endswith(".cif")); val = [f for f in files if isval(f)]; train = [f for f in files if not isval(f)]
cnt = lambda f: collections.Counter(c for c in rec[f] if c in BAL)
tr_tot = collections.Counter(); [tr_tot.update(cnt(f)) for f in train]
print("train complexes", len(train), "| per-code positions in the train split (min %d %s, max %d %s)" % (min(tr_tot.values()), min(tr_tot, key=tr_tot.get), max(tr_tot.values()), max(tr_tot, key=tr_tot.get)))
rng = random.Random(0); order = train[:]; rng.shuffle(order)
def rarity(f): c = cnt(f); return min((tr_tot[k] for k in c), default=10**9)
order.sort(key=rarity)  # complexes carrying the rarest codes first, ties random
got = collections.Counter(); keep = []
for f in order:
    c = cnt(f)
    if not c: continue
    if all(got[k] + n <= CAP for k, n in c.items()): keep.append(f); got.update(c)
print(f"CAP {CAP}: kept {len(keep)} of {len(train)} train complexes; per-code positions min {min(got.values())} max {max(got.values())} (ratio {max(got.values()) / min(got.values()):.2f})")
for k in DF13 + GEN12: print(f"  {k}: {tr_tot[k]:5d} -> {got[k]:4d}")
out = f"{D}/train_all_cif_balanced{CAP}"; DRY = os.environ.get("DRY"); os.makedirs(out, exist_ok=True) if not DRY else None
for f in ([] if DRY else keep + val): shutil.copy(f"{POOL}/{f}", f"{out}/{f}")
open(f"{D}/balanced_list_{CAP}.txt", "w").write("\n".join(keep) + "\n")
other = collections.Counter(c for f in keep for c in rec[f] if c not in CANON and c not in BAL)
json.dump(dict(cap=CAP, kept=len(keep), val=len(val), before={k: tr_tot[k] for k in BAL}, after={k: got[k] for k in BAL}, other_codes_kept=dict(other)), open(f"{D}/balanced_{CAP}_report.json", "w"), indent=1)
print("complexes in the new folder:", len(os.listdir(out)), "(train subset + all", len(val), "held-out)", "| other D-forms kept:", dict(other))
