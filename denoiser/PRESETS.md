# Denoiser trials: one row per trial

All trials use [`denoiser_train.py`](denoiser_train.py) (command-line options, see its header). Each trains a small LoRA adapter (rank 16 unless noted) on top of the frozen pretrained 189-class RFD4 denoiser. Outputs and logs are in [`../denoiser_trial/`](../denoiser_trial/) (`saved/` has the best LoRA of each trial); job files in [`slurm/`](slurm/), box launchers in [`launchers/`](launchers/). "Section" = section of [`../Results.MD`](../Results.MD).

"Control" = the untuned denoiser. Test sets: 163 held-out synthetic complexes (families never trained on), 103 real PDB peptide-protein complexes, and the generation test (11 real pockets, 88 designs per denoiser, decoder run 8).

| Trial | Data (synthetic complexes) | Recipe | Weights (under `../denoiser_trial/saved/`) | Section | Result |
|---|---|---|---|---|---|
| A | 1,266 training complexes (`train_all_cif_clean`) | rank 16, attention layers, batch 2, lr 1e-4, best step 500 | `box_out/A_all/best` | 43 | held-out loss -0.41 vs control; non-canonical positions -6% (real) |
| B | first batch only (228) | same as A | `box_out/B_first/best` | 43 | best on real complexes with a new D-form |
| C | as A, longer | batch 1, 3,000 steps, stopped at 1,143, best 600 | `box_out/C_long/best` | 43 | non-canonical -6.6% real, -7.4% held-out |
| D | as A | rank 64, alpha 128 | `box_out/D_rank64/best` | 43a | worse on all-position real score (+0.32) |
| E | as A | rank 64 + feed-forward layers + gate | `box_out/E_rank64_ff/best` | 43b | worst on real complexes (+0.55 all positions); no gain |
| F | 6,457-complex pool (`train_all_cif_clean2`) | recipe C on the larger pool; H200: batch 4, lr 2e-4, early stop 1,600 | `box_out/F_pool2/best`, `H200_F/.../best` | 45 | helps synthetic general codes, **not real ones** |
| F2 | same pool | F continued, lr 5e-5 (job 493216) | `H200_F2/.../best` | 46 | best held-out synthetic (-0.54), no real gain |
| G | 1,266 complexes | recipe C + v3 options: `--nc-weight 10 --wcap 100 --plateau 0.5`, 2,700 steps | `box_out/G_ncw10/best` | 50 | best on held-out synthetic (-0.68 vs control); closest backbone in generation |
| G2 | 8,947-complex pool | G continued (job 495390), batch 4, lr 5e-5, `--eval-list heldout30_list.txt` | `H200_G2/dn_G2_H200_495390/best` | 53 | best on real general codes (-8.8% latent loss) |
| **G3** | class-balanced subset (2,232 training complexes, cap 280 per swapped code) | G2 recipe (job 495885, early stop, best step 300) | `H200_G3/dn_G3_H200_495885/best` | 55 | **default denoiser for later tests** |
| H | G3 recipe + 3,500 mirror-image complexes | (slurm written, **not run**: no longer needed once the decoder fixed handedness) | - | 61 | skipped |
| G4 | as G3 | G3 start + `--clash-w 2` (clash loss through the frozen run 8 decoder), 600 steps, job 501522 | `H200_G4/dn_v4_H200_501522/best` | 66 | **no improvement** over G3 |

## Generation test (decoder run 8, 88 designs per denoiser; Results.MD 59-62, 66)
| Denoiser | D-forms right | Sensible bonds | Close contact (< 2 A) | Median CA distance to native | Contacts vs native | Free-position identity |
|---|---|---|---|---|---|---|
| control | 92% | 87% | 22% | 5.91 A | 1.29 | 8% |
| G | 100% | 83% | 30% | 4.78 A | 1.50 | 15% |
| G2 | 92% | 87% | 19% | 5.20 A | 1.49 | 14% |
| G3 | 97% | 87% | 20% | 5.35 A | 1.49 | 12% |
| G4 | 94% | 78% | 19% | 5.14 A | 1.42 | 14% |

What the fine-tuned denoisers add over the control: a closer pose (0.6 to 1.1 A), more contacts, and more native-like sequence at the designed positions. With decoder run 8 all denoisers have similar handedness, bonds and close contacts. These are differences between single runs of 88 designs; no significance tests.

## Caveats
* Benchmarks in Results.MD 43 to 56 are *flow-loss* tests, not tests of generated structures.
* One run per setting; the real non-canonical sets are small (257 positions in 102 complexes, 47 D-form positions).
* The sequence read-out head was never trained, so residue identity at generation time comes from the given identities and the frozen decoder.
