# Decoder runs: one row per run

All runs use [`decoder_train.py`](decoder_train.py) (settings = environment variables, see its header). Weights are in [`../weights/decoder/`](../weights/decoder/), the training logs in [`logs/`](logs/), the job files in [`slurm/`](slurm/), the older script that produced each run in [`history/`](history/). "Section" = section of [`../Results.MD`](../Results.MD).

The test used for the rightmost columns is the **generation test** (11 real pockets, 88 designs per denoiser; Results.MD 59 to 62). Numbers are for the fine-tuned denoisers G / G2 / G3 unless stated. "Handedness" = share of generated D-forms whose C-alpha has the right handedness; "bonds" = share of peptide C-N bonds of 1.2 to 1.5 A; "close" = share of designs with an atom pair under 2 A to the receptor.

| Run | Weights file | Start from | What changed (env settings for `decoder_train.py`) | Job | Section | Result |
|---|---|---|---|---|---|---|
| 1, 2 (176 classes) | `best_decoder_run1.pt`, `best_decoder_run2.pt` | bundle | only the residue-type tables trained (321,792 numbers); history script `decoder_fine_tune.py` | 468963 | 3, 4 | real PDB peptides: 99.6% of trained NCAAs named correctly |
| whole decoder, 13 new D-forms | `decoder_v3_485816.pt` | bundle | whole decoder, residue-type loss; `STRUCT_W=0 CHI_W=0 BOND_W=0` | 485816 | 38 | first 189-class decoder |
| "run 2" | `decoder_v4_488524.pt` | bundle | + 1,429 synthetic complexes, category logging | 488524 | 41 | weak D-forms still poor (30% pooled) |
| 3 | `decoder_v5_493747.pt` | bundle | + 7,534 mirror-image D-peptides (`MIRROR_LIST`); `STRUCT_W=0 CHI_W=0 BOND_W=0` | 493747 | 49 | real D-forms 110 of 112 named; but generated D-forms right only 35-42%, bonds 22%, close 62-72% |
| 4 (blind) | `decoder_v6_496342.pt` | run 3 | `BLIND=1` (residue identity hidden from the decoder) | 496342 | 58 | not better for generation; not used |
| 5 | `decoder_v7_498188.pt` | run 3 | + atom-coordinate loss: `STRUCT_W=0.25 NCW=1 CHI_W=0 BOND_W=0` | 498188 | 60 | handedness 83-95%, bonds 67%, close 19-34% |
| 5b | `decoder_v7b_498314.pt` | run 5 | same as run 5, one more epoch | 498314 | 61 | handedness 94-96%, bonds 67-74%, close 17-33% |
| 6 | `decoder_v8_498707.pt` | run 5b | + chirality loss: `STRUCT_W=0.25 NCW=3 CHI_W=0.5 BOND_W=0` | 498707 | 61 | handedness 98-99% but bonds back to 22%; not used |
| 7 | `decoder_v8_499080.pt` | run 5b | `STRUCT_W=0.5 NCW=2 CHI_W=0.15 BOND_W=0` | 499080 | 61 | handedness 97-99%, bonds 50-54% (handedness-first alternative) |
| **8 (default)** | `decoder_v9_499602.pt` | run 5b | + peptide-bond loss: `STRUCT_W=0.5 NCW=2 CHI_W=0.15 BOND_W=10 BOND_TOL=0.04 BOND_NOISE=0` = **the defaults of `decoder_train.py`** | 499602 | 62 | handedness 92-100%, **bonds 83-87%**, close 19-30% |
| 9 | `decoder_v10_500687.pt` | run 8 | + clash loss: `CLASH_W=5 NOISE_SIGMA=0.2 NOISY_BOND_W=0` | 500687 | 65 | **no improvement**: close contacts 20-30% unchanged, bonds and canonical atoms worse; not used |

Typical cost: 0.4 USD on a MIG slice (runs 3 to 7), 0.6 USD on a full H200 (runs 8, 9: 4 epochs in about 37 minutes).

## Failed or abandoned attempts (kept for the record)
* Bond loss on a second decode of *noised* latents (`BOND_NOISE=0.1`): the training loss exploded (pool loss 64); switched off in run 8. A noised-latent bond term in the first run 9 attempt also gave a loss of 14 (job cancelled); removed.
* Run 9 clash loss: see Results.MD 65 and its addendum for why it could not work (the remaining close contacts are mostly side-chain atoms, and the decoder never sees the receptor).

## Re-creating a run
Upload `decoder_train.py`, `bond_loss.py` and the patch folder files to the cluster, set the environment variables of the row (`RFD_BUNDLE`, `POOL_DIR`, `PILOT_DIR`, `COMPLEX_GROUPS`, `MIRROR_LIST`, `INIT`, `SAVE_PATH` plus the loss weights), and submit `slurm/decoder_train_h200.slurm` (its defaults are the run 8 recipe; override with `sbatch --export=ALL,...`). Example for run 8: `INIT=decoder_v7b_498314.pt STRUCT_W=0.5 NCW=2 CHI_W=0.15 BOND_W=10 BATCH=128 PBATCH=16 EPOCHS=4`.
