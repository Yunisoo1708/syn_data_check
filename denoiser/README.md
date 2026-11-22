# denoiser/ - training and testing the RFD4 denoiser (LoRA)

**What the denoiser does:** RFD4 is a flow-matching model. Starting from noise it generates the backbone (C-alpha positions) and a latent vector for each residue of a new peptide around a fixed protein pocket. The decoder (see `../decoder/`) then turns this into atoms. We fine-tune the denoiser with a **LoRA adapter** (1% of its numbers) on synthetic NCAA-peptide/protein complexes so that it learns to place NCAA residues.

| File / folder | Purpose |
|---|---|
| **`denoiser_train.py`** | **the one training script** for every trial (options off by default; the header explains the loop, the held-out check, every option and each trial's settings) |
| `PRESETS.md` | **one row per trial** (A to G4): data, settings, weights, result; plus the generation-test comparison table |
| `slurm/` | cluster job files (`denoiser_*_h200.slurm`, the G4 debug jobs, `inspect_v4_mig.slurm`) |
| `launchers/` | shell scripts that ran the first trials on the 2-GPU box (`run_denoiser_trial_*.sh`) |
| `eval/` | scripts that score a trained adapter: held-out synthetic complexes, real complexes (`eval_one.py`, `eval_one_nc.py`) and their launchers |
| `tools/` | `inspect_batch_v4.py` (prints what a training batch looks like; written while building the G4 clash loss) |
| `history/` | earlier versions of the trainer (`fine_tuning.py`, `ncaa_fine_tuning.py`, `_v2`, `_v3`, `_v4`). Use `denoiser_train.py` instead |

Trained adapters are not in this folder: they are in [`../denoiser_trial/saved/`](../denoiser_trial/saved/) (about 4 MB each). **Default denoiser = G3** (`H200_G3/dn_G3_H200_495885/best`).

## How a trial is run (G3 as the example)
```
python denoiser_train.py BUNDLE_189 train_all_cif_bal280 OUT_DIR --groups complex_groups.json --resume G_best \
   --steps 3000 --bs 4 --inner 5 --lr 5e-5 --eval-every 100 --eval-batches 30 --patience 5 \
   --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 \
   --eval-wcap 100 --eval-clip 150 --eval-list heldout30_list.txt --max-seconds 6000
```
What the numbers mean: `--bs` complexes per batch, `--inner` repeated updates per noised batch, `--nc-weight` loss weight on non-canonical positions, `--wcap` cap on the noise-level weight, `--plateau` learning-rate decay when the held-out loss stalls, `--eval-*` the fixed held-out check (same complexes and noise for every run, so trials are comparable).

## Things worth knowing
* In training batches **every** token is noised and a complex has many small chains; `is_in_context_conditioning` is all False. At generation time the receptor is fixed, so training and generation differ.
* The noise level of the C-alpha part and of the latent part are sampled **independently** (Beta(1.9,1) and Beta(1,1.5)).
* Single batches with a loss of 1e7 to 1e16 occur at very low noise; the trainer skips anything above 1e4.
* The decoder gradient path (used by `--clash-w`) needs a full H200; it runs out of memory on a MIG slice.
