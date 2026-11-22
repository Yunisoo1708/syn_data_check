# reference/ - upstream source, read-only

`rfd4_source/` holds copies of upstream RFD4 files (`rfproteina_*.py`, `pipeline.py`, `inference_datasets.py`) that were downloaded to read how the model works (training pipeline, flow matcher, autoencoder, feature factory). They are **not** the code that runs on the cluster and are not imported from here; to run any script of this repo, use the RFD4 python environment (which has the real package). Do not edit them.
