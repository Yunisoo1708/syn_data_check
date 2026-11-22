# data_tools/ - scripts that build and check the datasets

Run order of the data pipeline (details in Results.MD sections 10 to 42):
1. **Make synthetic complexes** (NCFlow swaps a residue; runs on the GPU box / clusters, scripts live in `splits/` and `Dataset_ncaa/`): `ncf_cpu_tillicum.slurm` is the CPU job file.
2. **Convert the results to training CIFs:** `convert_batch.sh` (writes CIFs with a 12 A pocket crop, runs the geometry check and the split guard), `run_convert_*.sh` (the launchers used per batch), `shard_scripts/` (the generated per-shard `mk_mask` / `mk_pep` helper scripts), `merge_pilot.sh` (merge shard outputs).
3. **Check:** `validate_train_cifs.py` (runs every CIF through the exact training data pipeline and lists the files it refuses), `pilot_metrics.py`, `pilot_stats.py`.
4. **Balance:** `make_balanced_pool.py CAP` builds the class-balanced subset `data/train_all_cif_balancedCAP`.
5. **Mirror-image data:** `mirror_pool.py` (mirror the peptide pool into real D-geometry), `mirror_complex.py` (mirror whole complexes).
6. **Real test sets:** `crop_real_test.py`, `crop_real_ham.py` (cut real complexes to peptide + pocket).
7. **Model files:** `make_bundle_176.py` (expand the bundle to 176 classes), `local_vocab.py` (load the vocabulary without importing the whole package).

The split rule that these scripts obey is in `../splits/` (`split_guard.py` refuses any test entry or relative of one).
