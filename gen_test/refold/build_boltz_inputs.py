"""Turn the AF3 inputs (inputs/*.json) into Boltz-2 yaml with the receptor held by a FORCED template (native receptor structure).
usage: build_boltz_inputs.py IN_DIR OUT_DIR   (OUT_DIR/*.yaml, OUT_DIR/tmpl/<job>_rec.cif; template paths are relative to the box location given by TMPL_ROOT env)"""
import sys, os, json, glob
IN, OUT = sys.argv[1], sys.argv[2]; ROOT = os.environ.get("TMPL_ROOT", "/home/yunisoo/rfd4_trial/refold/boltz_in/tmpl")
os.makedirs(f"{OUT}/tmpl", exist_ok=True)
for f in sorted(glob.glob(f"{IN}/*.json")):
    if f.endswith("manifest.json"): continue
    j = json.load(open(f)); name = j["name"]; rec = j["sequences"][0]["protein"]; pep = j["sequences"][1]["protein"]
    cid = name.split("-")[0]; tf = f"{OUT}/tmpl/{cid}_rec.cif"
    if not os.path.exists(tf): open(tf, "w").write(rec["templates"][0]["mmcif"])
    mods = "".join(f"\n        - position: {m['ptmPosition']}\n          ccd: {m['ptmType']}" for m in pep.get("modifications", []))
    y = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {rec['sequence']}
      msa: empty
  - protein:
      id: B
      sequence: {pep['sequence']}
      msa: empty{chr(10) + '      modifications:' + mods if mods else ''}
templates:
  - cif: {ROOT}/{cid}_rec.cif
    chain_id: A
    force: true
    threshold: 2
"""
    open(f"{OUT}/{name}.yaml", "w").write(y)
print(len(glob.glob(f"{OUT}/*.yaml")), "yaml")
