# docs/ - notes, plans and the clean-up record

| Item | What it is |
|---|---|
| `model_guide.md` | a short guide to the models (which weights are which) as of 18 Nov 2026 |
| `notes/` | the working notes of the project: `STICKY_NOTE.md` (running notes), `morning_report_17nov.md` to `_19nov.md` (what happened overnight), `night_plan*.md` (what was planned overnight, with decision rules) |
| `ideas/` | parked ideas (e.g. `Ideas_backbone_swap.md`) |
| `loss_page/` | a small HTML page that tabulates loss values (`index.html`) |
| `cleanup_moves_20Nov2026.tsv` | **every file move of the 20 Nov 2026 clean-up** (old path, new path) |

## The 20 Nov 2026 clean-up
The root of the repository had 317 loose items (scripts of 11 decoder versions, 60 figure scripts, weights, logs, 40 data folders). They were sorted into the folders described in the top-level README. Nothing was deleted, except empty folders (`runs/`) and Python cache folders (`__pycache__`). Two merged trainers were created: `decoder/decoder_train.py` and `denoiser/denoiser_train.py` (the originals are kept in the `history/` folders).

* Absolute paths that pointed at moved items (in scripts, Results.MD and the notes) were rewritten; relative mentions in prose (for example "`decoder_fine_tune_v9.py`") are *not*: use the table below or the `.tsv`.
* A backup of Results.MD before the clean-up is in `../archive/before_cleanup/`.

| Old location (repo root) | New location |
|---|---|
| `decoder_fine_tune*.py`, `bond_loss.py` | `decoder/history/`, `decoder/` (merged script: `decoder/decoder_train.py`) |
| `decoder_*.slurm`, `dec_v*.log` | `decoder/slurm/`, `decoder/logs/` |
| `*.pt` decoder weights | `weights/decoder/` |
| `eval_one_dec*.py`, `eval_real_autoencoder.py`, `run_eval_dec*.sh`, `run_eval_blind*.sh` | `decoder/eval/` |
| `denoiser_*.slurm`, `run_denoiser_trial_*.sh`, `eval_one.py`, `eval_one_nc.py`, other `run_eval_*.sh` | `denoiser/slurm/`, `denoiser/launchers/`, `denoiser/eval/` (merged script: `denoiser/denoiser_train.py`) |
| `make_fig*.py` | `figures/scripts/` |
| `*.mk_mask.py`, `*.mk_pep.py`, `convert_*.sh`, `make_balanced_pool.py`, `mirror_*.py`, `crop_real_*.py`, `validate_train_cifs.py` | `data_tools/` (shard helpers in `data_tools/shard_scripts/`) |
| data folders (`train_all_cif*`, `*_pocket12*`, `real_test_*`, `natural_complexes`, `pilot_*`, ...) | `data/` |
| `complex_groups.json`, `balanced_*`, `heldout30_list.txt`, `pilot_*.json`, ... | `data/lists/` |
| `*.yaml` vocabularies | `data/vocab/` |
| `RFD4_bundle*` | `models/` |
| `rfproteina_*.py`, `pipeline.py`, `inference_datasets.py` | `reference/rfd4_source/` |
| `model_guide.md`, `STICKY_NOTE.md`, reports, plans | `docs/` |
| probes and one-off shell scripts (`_*.py`, `_*.sh`, `list_*.py`, `run.sh`) | `scripts/misc/` |
| `*.write.log`, `convert_*.log` | `logs/data_generation/` |

**Undo:** `python docs/undo_cleanup.py --apply` moves every item back to the repository root (it refuses to overwrite). The merged trainers and the README files are new files and stay.
