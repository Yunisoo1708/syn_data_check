"""Collect AF3 results of the ipTM filter test into one json (Results.MD 83).
usage: python collect_af3.py <jobs_dir with manifest.json> <af3_out_dir> <out.json>
Per job: ipTM / pTM / ranking score (best sample and mean over samples), chain-pair ipTM, peptide mean pLDDT, clash flag, and the pose of the peptide against the reference
(native peptide for native/shuf/rnd/ala, the designed pose for designs): backbone CA RMSD after superposing the AF3 receptor on the native receptor (the designs share the native frame)."""
import json, glob, os, sys, re, numpy as np, gemmi
J, O, OUT = sys.argv[1:4]; man = {m["name"].lower(): m for m in json.load(open(os.path.join(J, "manifest.json")))}
G = r"C:\Research\RFD4_fine_tune\gen_test"; RP = r"C:\Research\Dataset_ncaa\real_peptides"; WAT = {"HOH", "WAT"}
DES = {"control": "control_choice_run8", "G3": "G3_bal_choice_run8", "S1": "S1_readout_choice_run8"}
def native_rec(ex):
    e, pc = ex.split("_"); st = gemmi.read_structure(rf"{RP}\{e}.cif"); st.setup_entities(); m = st[0]; pep = [c for c in m if c.name == pc][0]
    pxyz = np.array([a.pos.tolist() for r in pep for a in r if a.element.name != "H"]); best = None
    for c in m:
        if c.name == pc: continue
        xyz = np.array([a.pos.tolist() for r in c if r.name not in WAT for a in r if a.element.name != "H"])
        if not len(xyz): continue
        n = int((np.linalg.norm(pxyz[:, None] - xyz[None], axis=-1) < 4.5).any(1).sum())
        if best is None or n > best[0]: best = (n, c)
    return [r for r in best[1] if r.name not in WAT and r.entity_type == gemmi.EntityType.Polymer]
def real_to_gen(ex):
    """rigid transform from the real_peptides frame to the generation frame (superpose the native peptide CA)"""
    e, pc = ex.split("_"); st = gemmi.read_structure(rf"{RP}\{e}.cif"); rp = [r for r in st[0][pc] if r.name not in WAT]
    gp = [r for r in gemmi.read_structure(rf"{G}\inputs\{ex}.cif")[0]["P"] if r.name not in WAT]
    assert len(rp) == len(gp), (ex, len(rp), len(gp))
    A = [ca(r) for r in rp]; B = [ca(r) for r in gp]; return gemmi.superpose_positions(B, A).transform
def in_to_design(ex, f):
    """inputs frame -> frame of a generated design (generation recentres the complex; receptor atoms are identical)"""
    a = gemmi.read_structure(rf"{G}\inputs\{ex}.cif")[0]; b = gemmi.read_structure(f)[0]
    A = [ca(r) for c in a if c.name != "P" for r in c if r.name not in WAT]; B = [ca(r) for c in b if c.name != "P" for r in c if r.name not in WAT]
    k = [(x, y) for x, y in zip(A, B) if x is not None and y is not None]; return gemmi.superpose_positions([y for x, y in k], [x for x, y in k]).transform
def ca(res):
    for a in res:
        if a.name == "CA": return a.pos
    return None
NREC = {}; TG = {}
def ref_pep(m):
    """peptide reference in the generation (inputs) frame, plus the transform into the design frame if it is a design"""
    if m["group"] in DES:
        f = glob.glob(G + rf"\out_local\{DES[m['group']]}\inference_samples\{m['ex']}_{m['rep']}\*_generated.cif")[0]
        return [r for r in gemmi.read_structure(f)[0]["P"] if r.name not in WAT], in_to_design(m["ex"], f)
    return [r for r in gemmi.read_structure(rf"{G}\inputs\{m['ex']}.cif")[0]["P"] if r.name not in WAT], None
rows = []
for name, m in man.items():
    d = glob.glob(os.path.join(O, "**", name), recursive=True)
    d = [x for x in d if os.path.isdir(x)]
    if not d: continue
    d = d[0]; s = [f for f in glob.glob(os.path.join(d, "*_summary_confidences.json"))]
    if not s: continue
    summ = json.load(open(s[0])); cp = summ.get("chain_pair_iptm")
    samp = []
    for sf in glob.glob(os.path.join(d, "seed-*_sample-*", "summary_confidences.json")):
        x = json.load(open(sf)); samp.append(x["iptm"])
    cj = json.load(open(glob.glob(os.path.join(d, "*_confidences.json"))[0])); ch = np.array(cj["atom_chain_ids"]); pl = np.array(cj["atom_plddts"])
    row = dict(m, iptm=summ["iptm"], ptm=summ["ptm"], rank=summ["ranking_score"], iptm_mean=float(np.mean(samp)) if samp else summ["iptm"], iptm_samples=samp,
               cp_iptm=(cp[0][1] if cp else None), pep_plddt=float(pl[ch == "B"].mean()), rec_plddt=float(pl[ch == "A"].mean()), clash=summ.get("has_clash"), pae_int=None)
    try:
        mod = gemmi.read_structure(glob.glob(os.path.join(d, "*_model.cif"))[0])[0]; a_, b_ = mod["A"], mod["B"]
        ex = m["rec_ex"]; nat = NREC.setdefault(ex, native_rec(ex)); pairs = [(ca(x), ca(y)) for x, y in zip(a_, nat)]
        P = [x[0] for x in pairs if x[0] is not None]; Q = [x[1] for x in pairs if x[1] is not None]
        sup = gemmi.superpose_positions(Q, P)    # moves AF3 receptor onto the native receptor
        row["rec_rmsd"] = sup.rmsd
        if m["group"] != "decoy":
            ref, t2 = ref_pep(m); pairs = [(ca(x), ca(y)) for x, y in zip(b_, ref) if ca(x) is not None and ca(y) is not None]
            tg = TG.setdefault(ex, real_to_gen(ex)); inv = t2.inverse() if t2 is not None else None
            d2 = [np.linalg.norm(np.array(tg.apply(sup.transform.apply(x)).tolist()) - np.array((inv.apply(y) if inv is not None else y).tolist())) ** 2 for x, y in pairs]; row["pep_rmsd"] = float(np.sqrt(np.mean(d2))) if d2 else None
    except Exception as e: row["pose_err"] = str(e)[:80]
    rows.append(row)
json.dump(rows, open(OUT, "w")); print(len(rows), "jobs collected of", len(man))
