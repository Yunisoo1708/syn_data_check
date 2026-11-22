"""Build AF3 inputs (single-sequence, receptor given as a TEMPLATE from the native structure) to refold a small subset of designs.
usage: build_refold_inputs.py OUT_DIR RUN_DIR(G_ncw10_v10) N_PER_COMPLEX
For each real test complex: receptor = full native receptor chain (sequence + template), peptide = designed sequence with the non-canonical residues as CCD modifications (A + ptmType); also the NATIVE peptide as a control (only if all its residues are in the 189-class vocabulary).
Output: OUT_DIR/*.json + OUT_DIR/manifest.json"""
import sys, os, json, glob, yaml, gemmi
OUT, RUN, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = "/mnt/c/Research/RFD4_fine_tune"
os.makedirs(OUT, exist_ok=True)
vocab = set()
for sec, d in yaml.safe_load(open("/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml"))["residues"].items(): vocab |= set(d)
CAN = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()); WAT = {"HOH", "WAT", "DOD"}
def one(n):
    t = gemmi.find_tabulated_residue(n); c = t.one_letter_code.upper() if t else "X"
    return c if c.isalpha() and c != "X" else "A"
def pep_entry(names):
    seq = ""; mods = []
    for i, n in enumerate(names):
        if n in CAN: seq += one(n)
        else: seq += "A"; mods.append({"ptmType": n, "ptmPosition": i + 1})
    e = {"id": "B", "sequence": seq, "unpairedMsa": "", "pairedMsa": "", "templates": []}
    if mods: e["modifications"] = mods
    return {"protein": e}
man = {}
for f in sorted(glob.glob(f"{R}/data/real_test_cif/*_real.cif")):
    cid = os.path.basename(f)[:-9]
    designs = sorted(glob.glob(f"{RUN}/inference_samples/{cid}_*/*_generated.cif"))
    if not designs: continue
    st = gemmi.read_structure(f); m = st[0]
    rec = [c for c in m if c.name != "P" and sum(1 for r in c if r.name not in WAT) > 20]
    ntp = [c for c in m if c.name == "P"]
    if len(rec) != 1 or not ntp: print("skip", cid, [(c.name, len(c)) for c in m]); continue
    rc = rec[0]; rres = [r for r in rc if r.name not in WAT]
    rseq = "".join(one(r.name) for r in rres)
    ns = gemmi.Structure(); ns.cell = st.cell; ns.spacegroup_hm = st.spacegroup_hm; nm = gemmi.Model("1"); nc = gemmi.Chain("A")
    for k, r in enumerate(rres):
        r2 = gemmi.Residue(); r2.name = r.name; r2.seqid = gemmi.SeqId(k + 1, " "); r2.het_flag = "A"; r2.subchain = "A"; r2.label_seq = k + 1; r2.entity_type = gemmi.EntityType.Polymer
        for a in r: r2.add_atom(a)
        nc.add_residue(r2)
    nm.add_chain(nc); ns.add_model(nm); ns.setup_entities()
    if not ns.entities:
        e = gemmi.Entity("1"); e.entity_type = gemmi.EntityType.Polymer; e.polymer_type = gemmi.PolymerType.PeptideL; e.subchains = ["A"]; e.full_sequence = [r.name for r in rres]; ns.entities.append(e)
    ns.assign_label_seq_id(True); ns.name = cid
    doc = ns.make_mmcif_document(); tmpl = doc.as_string()
    if "_entity_poly_seq" not in tmpl:
        eid = ns.entities[0].name
        NL = chr(10)
        tmpl += NL + "loop_" + NL + "_entity_poly_seq.entity_id" + NL + "_entity_poly_seq.num" + NL + "_entity_poly_seq.mon_id" + NL + "_entity_poly_seq.hetero" + NL + "".join(f"{eid} {k + 1} {r.name} n" + NL for k, r in enumerate(rres))
    NL = chr(10)
    tmpl += NL + "_pdbx_audit_revision_history.ordinal 1" + NL + "_pdbx_audit_revision_history.data_content_type 'Structure model'" + NL + "_pdbx_audit_revision_history.major_revision 1" + NL + "_pdbx_audit_revision_history.minor_revision 0" + NL + "_pdbx_audit_revision_history.revision_date 1990-01-01" + NL
    idx = list(range(len(rres)))
    rent = {"protein": {"id": "A", "sequence": rseq, "unpairedMsa": "", "pairedMsa": "", "templates": [{"mmcif": tmpl, "queryIndices": idx, "templateIndices": idx}]}}
    def job(name, pnames):
        j = {"name": name, "modelSeeds": [1], "sequences": [rent, pep_entry(pnames)], "dialect": "alphafold3", "version": 1}
        json.dump(j, open(f"{OUT}/{name}.json", "w")); man[name] = dict(complex=cid, kind="design" if "-d" in name else "native", peptide=pnames, receptor_len=len(rres))
    for dpath in designs[:N]:
        k = dpath.split("/")[-2].rsplit("_", 1)[1]; dst = gemmi.read_structure(dpath); pc = [c for c in dst[0] if c.name == "P"][0]
        job(f"{cid}-d{k}", [r.name for r in pc if r.name not in WAT])
    nat = [r.name for r in ntp[0] if r.name not in WAT]
    if all(n in vocab or n in CAN for n in nat): job(f"{cid}-native", nat)
    else: print("native skipped (residue not in vocabulary):", cid, [n for n in nat if n not in vocab and n not in CAN])
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
print(len(man), "jobs:", sum(v["kind"] == "design" for v in man.values()), "designs,", sum(v["kind"] == "native" for v in man.values()), "natives")
