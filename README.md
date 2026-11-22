# RFD4 for non-canonical amino acids (NCAAs): fine-tuning, generation test, results

**Start here.** This folder holds everything for one project: teaching the protein-design model RFD4 to place **non-canonical amino acids** (amino acids that are not among the 20 natural ones, for example D-amino acids, N-methylated or alpha-methylated residues) in a peptide that binds a protein. The full lab notebook with every number is [`Results.MD`](Results.MD) (66 numbered sections, long; this README is the short version).

## 1. The goal in plain words
* **Small goal (what we do now):** a designer chooses the protein pocket, the position in the peptide and *which* NCAA goes there. The model generates a peptide around that residue, and the result should be a plausible 3D structure.
* **Big goal (parked):** the model also chooses *which* NCAA to use.
* **Hard rule:** real complexes that contain NCAAs are used for **testing only**. Training data are synthetic (made by us) plus natural complexes with ordinary amino acids. No test protein or close relative of one is ever used for training (see `splits/`).

## 2. How the model works (two parts)
```
 structure  --encoder-->  latent (a few numbers per residue)  --decoder-->  atoms + residue names
                               ^
        DENOISER (flow model) generates new latents + backbone positions, starting from noise
```
* **Encoder** (frozen): structure to latent. **Decoder** (trained here, [`decoder/`](decoder/)): latent to all atoms and residue type. **Denoiser** (trained here with a small LoRA adapter, [`denoiser/`](denoiser/)): invents new latents and backbone positions.
* At generation time the denoiser makes the peptide, the decoder turns it into atoms. A design is only as good as both parts, so we test them together in [`gen_test/`](gen_test/).

## 3. What was done, in order
1. **Data** ([`data_tools/`](data_tools/), [`data/`](data/)): 125,000 synthetic NCAA peptides folded with RF3 (43,000 passed a strict geometry test); then **synthetic complexes**: a natural peptide-protein complex with one peptide residue swapped for an NCAA by the NCFlow generator and kept only if chemistry and contacts pass (about 9,000 to 11,000 complexes, class-balanced).
2. **Decoder** (runs 1 to 9, [`decoder/PRESETS.md`](decoder/PRESETS.md)): first learned to *name* NCAAs, then to place their atoms correctly (D-form handedness), then to keep peptide bonds the right length. A clash loss (run 9) did not help.
3. **Denoiser** (trials A to G4, [`denoiser/PRESETS.md`](denoiser/PRESETS.md)): LoRA fine-tunes on the synthetic complexes. The best trials are G, G2 and G3. A clash loss inside the denoiser (G4) did not help.
4. **Generation test** (11 real pockets, 88 designs per run) and refolding checks with Boltz-2 and AF3.

## 4. Results at a glance
| Question | Answer | Where |
|---|---|---|
| Can the decoder name NCAAs in real PDB peptides? | Yes: 99.6% of 446 trained-NCAA positions, canonical 98.4% (189 real peptides) | Results.MD 4 |
| Which predictor places an NCAA peptide on its pocket best? | Boltz-2 (75% within 5 A of the real centre, 40 complexes) over DiffDock (50%) and RF3 (18%) | Results.MD 7 |
| Does fine-tuning the denoiser help? | At NCAA positions the latent loss falls 4 to 15%; the big synthetic pool does not transfer to real complexes | Results.MD 46 |
| D-form handedness in generated designs | 35-42% (run 3 decoder) to 92-100% (run 8): the decoder, not the denoiser, was the cap | Results.MD 59-61 |
| Peptide bonds of sensible length | 22% (run 3) to 67-74% (run 5b) to **83-87% (run 8, default)** | Results.MD 62 |
| Designs with an atom pair closer than 2 A to the receptor | about 20% for every denoiser incl. the untuned one | Results.MD 62b |
| Does a clash loss fix it? | No: decoder (run 9) and denoiser (G4) both left it unchanged and cost bonds or handedness | Results.MD 65-66 |
| What does help? | Drawing 4 designs per pocket and keeping a clash-free one: works for 91-99% of pockets (first batch and a fresh seed), bonds and pose unchanged | Results.MD 65 addendum, 67 |

**Defaults to use:** decoder run 8 (`weights/decoder/decoder_v9_499602.pt`) and denoiser G3 (`denoiser_trial/saved/H200_G3/.../best`).
**Limits:** 11 pockets, 88 designs per run, one seed, designs not relaxed, no energy check; the NCAA identity is given, not chosen by the model.

## 5. Folder guide
| Folder | What is in it |
|---|---|
| [`decoder/`](decoder/) | **`decoder_train.py`** (the one decoder trainer, all runs), `bond_loss.py`, Slurm job files, training logs, evaluation scripts, `PRESETS.md` (one row per run), `history/` (the 11 older script versions) |
| [`denoiser/`](denoiser/) | **`denoiser_train.py`** (the one denoiser trainer, all trials), Slurm and launcher files, evaluation scripts, `PRESETS.md` (one row per trial), `history/` |
| [`denoiser_trial/`](denoiser_trial/) | outputs of the denoiser trials (LoRA weights in `saved/`, logs, benchmark tables) |
| [`gen_test/`](gen_test/) | the generation test: prepare pockets, generate designs on the GPU box, analyse, diagnostics (`best_of_n_filter.py`, `diag_*.py`), refolding (`refold/`) |
| [`data/`](data/) | all datasets: synthetic complexes, training sets, real test complexes, lists, vocabulary files (see `data/README.md`) |
| [`data_tools/`](data_tools/) | scripts that build, convert, balance and check the datasets |
| [`splits/`](splits/) | the frozen train/test split and the scripts that built it (the anti-leakage rule) |
| [`weights/`](weights/) | the 13 decoder weight files (`weights/README.md` says which is which) |
| [`models/`](models/) | the RFD4 model bundles (176 and 189 classes) |
| [`figures/`](figures/) | 66 figures + `scripts/` that draw them + an index |
| [`docs/`](docs/) | model guide, old notes and plans, ideas, the move log of the 20 Nov 2026 clean-up |
| [`reference/`](reference/) | read-only copies of upstream RFD4 source files (to read the code; not meant to be run from here) |
| [`scripts/`](scripts/), [`logs/`](logs/), [`archive/`](archive/) | one-off probes and helpers; logs; old material |

## 6. How to run things
* **Train a decoder:** `decoder/decoder_train.py`, configured with environment variables; the table in `decoder/PRESETS.md` gives the exact settings of every run. Run on a GPU cluster (we used UW Tillicum, one H200 for about 40 minutes per run).
* **Train a denoiser:** `denoiser/denoiser_train.py` with command-line options; `denoiser/PRESETS.md` lists each trial.
* **Test generation:** `gen_test/README.md` (generate on a GPU, then analyse on any machine with Python + gemmi).
* **Redraw a figure:** `python figures/scripts/make_figNN_*.py`.
* The scripts need the RFD4 python environment on the GPU machine (not included here) and the patch folder `Dataset_ncaa/multigpu/rfd4_ncaa_patch` (`_common.py`, `rfproteina_encoding_ext.py`).

## 7. Glossary
* **NCAA**: non-canonical amino acid. **D-form**: mirror-image version of a natural (L) amino acid; its C-alpha has the opposite handedness.
* **Latent**: the short vector the model uses to describe one residue. **Flow matching / denoiser**: generate by removing noise step by step.
* **LoRA**: a small add-on trained instead of the whole network (1% of the numbers).
* **NCFlow**: the generator that swaps one residue of a real complex for an NCAA. **Pocket**: the receptor residues around the peptide.
* **Handedness right**: the decoded D-residue has its C-beta on the correct side. **Sensible bond**: peptide C-N distance of 1.2 to 1.5 A. **Close contact**: a peptide atom within 2 A of a receptor atom.
* **Tillicum / Klone**: UW GPU clusters. **The box**: our 2-GPU workstation used for generation tests.

## 8. Where files moved
On 20 Nov 2026 the loose files of this folder were sorted into the layout above. Nothing was deleted. Results.MD text and scripts that name old paths were updated where they used absolute paths; `docs/cleanup_moves_20Nov2026.tsv` lists every move (old path to new path), and `docs/README.md` explains how to undo it.
