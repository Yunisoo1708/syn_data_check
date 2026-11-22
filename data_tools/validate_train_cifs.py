"""Run every training CIF through the exact data pipeline of decoder_fine_tune_v4.py and list the files the pipeline refuses (the trainer would crash on the first one).
env: RFD4_VOCAB_YAML, PATCH_DIR, CIF_DIR, OUT (json list of failures). usage: python validate_train_cifs.py"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.environ.get("PATCH_DIR", HERE), HERE):
    sys.path.insert(0, d)
os.environ.setdefault("RFD4_VOCAB_YAML", os.path.join(os.environ.get("PATCH_DIR", HERE), "aa_rna_dna_glycan_ncaa_ext_v3.yaml"))
import _common  # noqa: F401  (puts the 189-class vocabulary in before rfproteina is imported)
from pathlib import Path
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None)
files = sorted(Path(os.environ["CIF_DIR"]).glob("*.cif"))
bad = []; t0 = time.time()
for i, f in enumerate(files):
    try:
        ds = DesignProblemDataset([f], transform=pipe); ds[0]
    except Exception as e:
        bad.append(dict(file=f.name, error=(type(e).__name__ + ": " + str(e))[:300]))
    if i % 200 == 0: print(i, "of", len(files), "bad so far", len(bad), f"{time.time() - t0:.0f}s", flush=True)
json.dump(bad, open(os.environ["OUT"], "w"), indent=1)
print("done", len(files), "files,", len(bad), "refused by the pipeline", flush=True)
