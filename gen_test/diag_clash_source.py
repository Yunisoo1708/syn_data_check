"""Where do the close contacts of the generated designs come from? For every design with a peptide-receptor heavy-atom pair < 2.0 A (same definition as analyze_gen.py), list the clashing peptide atoms: backbone (N CA C O) or side chain, canonical or non-canonical residue, given (non-canonical, identity specified) or free position; also the closest peptide-CA to receptor-CA distance of every design (a backbone-level measure the denoiser controls) and the same for the native peptide.
usage: diag_clash_source.py RUN_DIR... ; prints one block per run and writes diag/clash_source.json"""
import sys, os, glob, json, re, collections
import numpy as np, gemmi
import analyze_gen as A
BB = {"N", "CA", "C", "O", "OXT"}
def run(rdir):
    res = dict(designs=0, clash_designs=0, atoms_bb=0, atoms_sc=0, atoms_nc=0, atoms_can=0, atoms_pos_nc_given=0, atoms_pos_free=0, designs_bb_only=0, designs_sc_only=0, designs_both=0, ca_ca_min=[], ca_any_min=[], native_ca_ca_min=[], clash_dmin=[])
    for f in sorted(glob.glob(f"{rdir}/inference_samples/*/*_generated.cif")):
        ex = os.path.basename(f).replace("_generated.cif", ""); name = re.sub(r"_\d+$", "", ex)
        if name not in A.NAT: A.NAT[name] = A.native(name)
        npep, nrec, sgnL = A.NAT[name]
        st = gemmi.read_structure(f); m = st[0]; P = [c for c in m if c.name == "P"]
        if not P: continue
        pep = [r for r in P[0] if r.name not in A.WAT]
        if len(pep) != len(npep): continue
        grec = [r for c in m if c.name != "P" for r in c if r.name not in A.WAT]
        if len(grec) != len(nrec): continue
        R, t = A.kabsch(np.array([A.xyz(r, "CA") for r in grec]), np.array([A.xyz(r, "CA") for r in nrec]))
        for r in pep:
            for a in r: a.pos = gemmi.Position(*(R @ np.array(a.pos.tolist()) + t))
        recat = [(a, r) for r in nrec for a in r if a.element.name != "H"]; rec_xyz = np.array([a.pos.tolist() for a, _ in recat]); rec_ca = np.array([a.pos.tolist() for a, _ in recat if a.name == "CA"])
        nc_pos = {i for i, r in enumerate(npep) if r.name not in A.CAN}
        res["designs"] += 1
        pa = [(a, i, r) for i, r in enumerate(pep) for a in r if a.element.name != "H"]; pxyz = np.array([a.pos.tolist() for a, _, _ in pa])
        d = np.linalg.norm(pxyz[:, None, :] - rec_xyz[None], axis=-1)
        res["ca_any_min"].append(float(min(d[k].min() for k, (a, _, _) in enumerate(pa) if a.name == "CA")))
        pca = np.array([a.pos.tolist() for a, _, _ in pa if a.name == "CA"]); res["ca_ca_min"].append(float(np.linalg.norm(pca[:, None] - rec_ca[None], axis=-1).min()))
        nxyz = np.array([a.pos.tolist() for r in npep for a in r if a.name == "CA"]); res["native_ca_ca_min"].append(float(np.linalg.norm(nxyz[:, None] - rec_ca[None], axis=-1).min()))
        hit = np.argwhere(d < 2.0)
        res["clash_dmin"].append(float(d.min()))
        if len(hit) == 0: continue
        res["clash_designs"] += 1; kinds = set()
        for k in sorted({int(h[0]) for h in hit}):
            a, i, r = pa[k]; bb = a.name in BB; kinds.add("bb" if bb else "sc")
            res["atoms_bb" if bb else "atoms_sc"] += 1; res["atoms_nc" if r.name not in A.CAN else "atoms_can"] += 1
            res["atoms_pos_nc_given" if i in nc_pos else "atoms_pos_free"] += 1
        res["designs_bb_only" if kinds == {"bb"} else "designs_sc_only" if kinds == {"sc"} else "designs_both"] += 1
    for k in ("ca_ca_min", "ca_any_min", "native_ca_ca_min", "clash_dmin"):
        v = np.array(res[k]); res[k] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), min=float(v.min()), frac_below=float((v < {"ca_ca_min": 3.5, "ca_any_min": 2.8, "native_ca_ca_min": 3.5, "clash_dmin": 2.0}[k]).mean()))
    return res
if __name__ == "__main__":
    out = {}
    for rd in sys.argv[1:]:
        key = os.path.basename(rd.rstrip("/")); out[key] = run(rd); r = out[key]
        print(key.ljust(14), "designs", r["designs"], "clash", r["clash_designs"], "| clashing atoms: backbone", r["atoms_bb"], "side chain", r["atoms_sc"], "| non-canonical res", r["atoms_nc"], "canonical", r["atoms_can"], "| at given-NC positions", r["atoms_pos_nc_given"], "free", r["atoms_pos_free"], "| designs bb-only/sc-only/both", r["designs_bb_only"], r["designs_sc_only"], r["designs_both"],
              "| min CA-CA to receptor: median %.2f p10 %.2f below 3.5A %.2f (native median %.2f below 3.5A %.2f)" % (r["ca_ca_min"]["median"], r["ca_ca_min"]["p10"], r["ca_ca_min"]["frac_below"], r["native_ca_ca_min"]["median"], r["native_ca_ca_min"]["frac_below"]))
    json.dump(out, open(f"{A.D}/gen_test/diag/clash_source.json", "w"), indent=1)
