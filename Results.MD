# Results: RFD4 autoencoder extended to 32 non-canonical residues (176 classes)

Date of the runs: 14 Nov 2026. All numbers below were measured in this project; the files they come from are listed at the end.

> **Reading guide (added 20 Nov 2026).** This is the full lab notebook, in the order the work was done (66 numbered sections). For a short overview start with [`README.md`](README.md); for the decoder runs see [`decoder/PRESETS.md`](decoder/PRESETS.md), for the denoiser trials [`denoiser/PRESETS.md`](denoiser/PRESETS.md), for the figures [`figures/README.md`](figures/README.md). The repository was reorganised on 20 Nov 2026: file names written in the text without a folder (for example `decoder_fine_tune_v9.py`, `make_fig59_bond_loss.py`) are now inside `decoder/history/`, `figures/scripts/` and so on; the old-to-new table is in [`docs/README.md`](docs/README.md) and every move is in `docs/cleanup_moves_20Nov2026.tsv`. Absolute paths in this file were updated.

## 1. What was done

1. Built a synthetic structure library of non-canonical amino acid (NCAA) peptides: 125,000 sequences, 5,000 for each length from 6 to 30 residues, made from 32 real CCD monomers.
2. Predicted every structure with RF3 (from an RDKit-built starting conformer), repaired wrongly handed stereocentres, and kept only structures that pass a strict per-residue geometry test.
3. Extended the RFD4 vocabulary from 147 to 176 classes (29 new classes) and fine-tuned the residue-type decoder on the structures that passed.
4. Tested the fine-tuned decoder on real peptides from the PDB that contain these residues.

## 2. Training data

| Item | Value |
|---|---|
| Library | 125,000 peptides, L06 to L30, all computed |
| Structure source | RF3 (AF3 outputs were never used; checked: 0 AF3 files in the pool) |
| Strict test (every residue must pass) | bond lengths within 0.20 A, bond angles, no clashes (2.0 / 2.2 A), peptide C-N link 1.33 +/- 0.20 A, handedness matches the residue definition (L residues L, D residues D) |
| Structures that passed | 43,465 used for training (about a third of those attempted) |
| Vocabulary | `merged/aa_rna_dna_glycan_ncaa_ext_v2.yaml`, 176 classes |
| Split | 80 / 10 / 10 by file (train 34,773, validation 4,346, test 4,346), fixed seed |

## 3. Training

Only two tables of the decoder were trained (321,792 numbers); the encoder and everything else were frozen. Loss: cross-entropy over the 176 residue classes.

| Run | Settings | Result |
|---|---|---|
| Run 1 | batch 1024, 20 epochs, lr 1e-4, one H200, 87 min | validation loss 0.381, test loss 0.380 |
| Run 2 | continued from run 1, +70 epochs, same settings, one H200, 5 h 10 min | best validation loss 0.0130, test loss 0.0123 |

The training and validation loss stayed close together through run 2 (0.0123 and 0.0131 at the last epoch), so there was no sign of overfitting.

## 4. Evaluation on real PDB structures

### 4.1 Test set

Real peptide chains cut from experimental PDB entries downloaded from RCSB. Each chain contains at least one of the 32 residues. Identical sequences were collapsed to the best-resolution entry (the non-redundant set).

| Tier | Source | Non-redundant items | Scored positions |
|---|---|---|---|
| A | 274 cryo-EM entries | 104 (103 evaluated) | 1,314 |
| B | 89 X-ray / NMR entries added to cover residues that cryo-EM lacks | 88 (86 evaluated) | 1,338 |
| All | | 192 (189 evaluated) | 2,652 |

- All 32 residues have at least one real example (Tier A alone covered 18 of 32).
- 3 items could not be evaluated because the data pipeline rejected them for a bond length above its limit: 22ES_A (Tier A), 9BI3_A and 9OXP_A (Tier B).
- Scored positions are canonical residues, the 32 trained NCAAs and other vocabulary NCAAs. Residues outside the 176 classes are not scored.

### 4.2 Task

The model sees the real peptide's coordinates and atoms, never its residue names. It must name every residue (argmax over 176 classes). The ground truth is the CCD code of each residue in the PDB entry.

### 4.3 Overall accuracy (all tiers, 2,652 positions)

| Decoder version | All positions | Canonical (2,156) | Trained NCAAs (446) | Other vocabulary NCAAs (50) |
|---|---|---|---|---|
| Fresh model (new classes untrained) | 86.2% (2,286) | 98.5% | 25.1% (112) | 100% |
| Run 1 (20 epochs) | 96.6% (2,563) | 98.1% | 90.1% (402) | 94.0% |
| **Run 2 (90 epochs, final)** | **98.5% (2,611)** | **98.4%** (2,121) | **99.6%** (444) | **92.0%** (46) |

The fresh model's 25% on the trained NCAAs comes only from AIB, HYP and ORN, which the original 147-class model already knew. Every new class scored 0 before training. Canonical accuracy did not drop (98.5% before, 98.4% after), so the earlier classes were not forgotten.

### 4.4 Run 2 by tier

| Tier | All positions | Canonical | Trained NCAAs | Other vocabulary NCAAs |
|---|---|---|---|---|
| A (cryo-EM) | 98.9% (1,300 of 1,314) | 99.0% (1,091 of 1,102) | 99.5% (187 of 188) | 91.7% (22 of 24) |
| B (X-ray / NMR) | 98.0% (1,311 of 1,338) | 97.7% (1,030 of 1,054) | 99.6% (257 of 258) | 92.3% (24 of 26) |

### 4.5 Recall of each of the 32 trained residues (run 2, all tiers)

The count is correct / real examples.

| Family | Residue: recall |
|---|---|
| D-alpha | DAL 38/39, DAR 20/20, DPN 43/43, DVA 20/20, DLE 12/12, DPR 9/9 |
| N-alkyl | SAR 12/12, 04Q 2/2, MAA 7/7, MLE 18/18, MVA 8/8 |
| Alpha,alpha-disubstituted | AIB 72/72, 2ML 12/12, DIV 11/11, 02K 2/2, 1AC 2/2, 192 4/4, 0XL 1/1, AC5 11/11 |
| Beta | B3L 6/6, B3A 10/10, BAL 16/16, B3K 10/10, B3Q 5/5, XCP 19/19 |
| Gamma | STA 7/7 |
| Side-chain modified | DBU 6/7, 23F 9/9, 4FB 3/3, FP9 9/9, HYP 33/33, ORN 7/7 |

Only 2 of the 446 trained-NCAA positions were wrong: one DAL and one DBU.

## 5. Limitations (state these when reporting)

- **Small counts per residue.** Several residues have 1 to 4 real examples (for example 04Q 2, 02K 2, 1AC 2, 0XL 1, MAA 7, MVA 8). Their 100% recall is anecdotal and not a reliable rate. AIB (72), DPN (43), DAL (39), HYP (33) and XCP (19) have more.
- **Prior exposure.** The original 147-class model may have seen some of these PDB entries in pretraining, and canonical residues, AIB, HYP and ORN were already in its vocabulary. The 29 new classes were not.
- **Real peptides are in a bound state.** The chains were extracted without their partner protein, and this test does not measure whether the model handles the binding interface.
- **Mixed evaluation population.** Canonical residues are 81% of the scored positions, so the overall 98.5% is dominated by them. The trained-NCAA figure (99.6%) is the number that answers whether the new classes were learned.
- **Other vocabulary NCAAs** (50 positions) fell from 100% to 92.0% after training. Four positions were affected, so this is too small to judge, but it is worth a check on a larger set.
- **Three items** were not evaluated (bond-length validation).
- **Test peptides are real and the training peptides were generated.** The 99.6% shows transfer to real structures, but both sets only contain peptides of 3 to 50 residues for the real data and 6 to 30 for training.

## 6. Where the results are

- Final weights: `C:\Research\RFD4_fine_tune\weights\decoder\best_decoder_run2.pt` (run 1: `best_decoder_run1.pt`).
- Training log: `C:\Research\RFD4_fine_tune\slurm-468963.out` (run 2).
- Per-position evaluation tables: `C:\Research\Dataset_ncaa\real_benchmark\ae_eval\ae_eval_{bundle,run1,run2}.tsv` and the full log `slurm-470425.out` in the same folder.
- Benchmark and builder: `C:\Research\Dataset_ncaa\real_benchmark\` (`benchmark_items.tsv`, `benchmark_residues.tsv`, `scripts/build_benchmark.py`, `tierB_supplement_list.csv`).
- Evaluation script: `C:\Research\RFD4_fine_tune\decoder\eval\eval_real_autoencoder.py`.

## 7. Which model best predicts a non-canonical peptide bound to a protein? (Boltz-2 vs RF3 vs DiffDock)

Status on 14 Nov 2026: **all three models are complete.**

### 7.1 Test set

83 real peptide-protein complexes from the PDB (same real entries as section 4; the non-redundant peptides whose receptor chains in contact have 20 to 800 residues). The real complex is the ground truth. Every peptide contains at least one non-canonical residue (given by its CCD code).

### 7.2 What each model was given

- **Boltz-2 and RF3:** the receptor sequence and the peptide with its real CCD codes. Single-sequence mode (no multiple sequence alignment) for both, so the receptor is folded from sequence alone. One structure per complex, one seed. RF3 was run one complex at a time.
- **DiffDock (v1.1):** the real receptor structure plus the peptide as a ligand (a chemical structure built from the CCD definitions). Blind docking over the whole receptor, 10 sampled poses per complex, and the pose DiffDock ranks first is scored. Only peptides of 12 residues or less, and only residues that can be turned into a chemical structure (40 of the 83 complexes). This gives DiffDock an advantage, since it is handed the real receptor.

### 7.3 Measures

- Peptide CA RMSD after superposing the receptor CA atoms (lower is better; under 4 A counts as a success).
- Centroid distance: distance between the predicted and real peptide centers, in the real receptor frame (under 5 A means the right site).
- Interface recall and precision: share of the real peptide-contact receptor residues (within 5 A) that the predicted peptide also touches, and the share of the predicted contacts that are real.

### 7.4 Results on all complexes each model could run

| Model | Complexes scored | Median peptide CA RMSD (A) | CA RMSD < 4 A | Median centroid distance (A) | Centroid < 5 A | Median interface recall / precision |
|---|---|---|---|---|---|---|
| Boltz-2 | 72 of 83 | 7.1 | 23 (32%) | 3.6 | 42 (58%) | 0.68 / 0.70 |
| RF3 | 73 of 83 | 17.2 | 1 (1%) | 13.9 | 9 (12%) | 0.32 / 0.27 |

On the 72 complexes both models ran, the numbers are the same within rounding (Boltz-2 7.1 A, 32% under 4 A; RF3 17.4 A, 1% under 4 A).

**Conclusion:** Boltz-2 puts the non-canonical peptide at the right binding site in most cases (median 3.6 A from the real center, 58% within 5 A). RF3 usually puts the peptide away from the real binding site (median 14 A, 12% within 5 A).

### 7.5 Same 40 complexes for all three models

The 40 complexes DiffDock could take (peptides of up to 12 residues), where Boltz-2 and RF3 also produced a structure. DiffDock docks the peptide as a rigid-chemistry ligand, so its pose cannot be matched atom by atom to the real peptide; for it only the site measures (centroid and interface) are available.

| Model | Complexes | Median peptide CA RMSD (A) | CA RMSD < 4 A | Median centroid distance (A) | Centroid < 5 A | Median interface recall / precision |
|---|---|---|---|---|---|---|
| **Boltz-2** | 40 | 5.6 | 15 (38%) | **2.0** | **30 (75%)** | **0.73 / 0.78** |
| DiffDock | 40 | not available | not available | 4.9 | 20 (50%) | 0.56 / 0.59 |
| RF3 | 40 | 14.4 | 0 (0%) | 12.1 | 7 (18%) | 0.39 / 0.33 |

**Conclusion: Boltz-2 is the best of the three** at placing a non-canonical peptide on its real binding site, even though DiffDock was given the real receptor structure and Boltz-2 had to fold the receptor from sequence. Ranking on the shared 40: Boltz-2 (median 2.0 A from the real peptide center, 75% within 5 A), then DiffDock (4.9 A, 50%), then RF3 (12.1 A, 18%).

### 7.6 Limitations

- Boltz-2 produced no structure for 11 of the 83 complexes and RF3 for 10. Both failed on residue codes missing from their built-in dictionaries (for example A1L9X, A1MF6). The same 10 failed in both models, and Boltz-2 failed on one more (9IVC_E). These complexes are not counted as model failures.
- Single-sequence mode, one structure per complex, one seed: a known handicap for receptors, and the relative order is more reliable than the absolute RMSD values.
- The peptides are in their real bound conformation, so a model that docks but folds the peptide wrongly still scores a high RMSD.
- DiffDock covers only the 40 complexes with peptides of 12 residues or less, which are the easier cases. Its poses are scored on the top-ranked pose only (the other 9 sampled poses were not used), and only on peptide position (centroid and interface), because its output cannot be matched atom by atom to the real peptide. A centroid within 5 A is a lenient test of a correct pose.
- DiffDock was given the real receptor structure, while Boltz-2 and RF3 folded the receptor from its sequence, so the comparison favours DiffDock.

## 8. Can Boltz-2 be improved on the complex test? (variants, 14 Nov 2026)

Same 72 complexes Boltz-2 could run (section 7.1). Each variant changes one thing and is compared with the baseline run on the same complexes. Status: `s5`, `bonds` and `tmpl` are complete; `steps` and `all` were still running when this was written, so their rows are partial.

| Variant | What changed |
|---|---|
| baseline | single-sequence receptor (no alignment), 1 sample, default steps |
| s5 | 5 samples; the sample with Boltz-2's highest confidence score is kept (top-1). "Best of 5" is an oracle that knows the real answer, shown for reference only |
| steps | recycling 10 (default 3), sampling steps 500 (default 200), potentials (steering) on |
| bonds | adds the cyclic flag and covalent bond constraints (disulfides and others) found in the real peptide; only the complexes that have any |
| tmpl | the real receptor structure is given as a template |
| all | bonds + template + 5 samples + the steps settings |

| Variant | Complexes | Median receptor CA RMSD (A) | Median peptide CA RMSD (A) | Peptide < 4 A | Median centroid distance (A) | Centroid < 5 A | Median interface recall |
|---|---|---|---|---|---|---|---|
| baseline | 72 | 3.8 | 7.1 | 23 (32%) | 3.6 | 42 (58%) | 0.68 |
| s5 top-1 by confidence | 72 | 3.9 | 6.7 | 22 (31%) | 3.5 | 43 (60%) | 0.62 |
| s5 best of 5 (oracle) | 72 | 3.9 | 5.6 | 30 (42%) | 2.8 | 47 (65%) | 0.69 |
| baseline | 65 | 4.2 | 7.1 | 22 (34%) | 3.7 | 37 (57%) | 0.68 |
| steps (partial) | 65 | 4.9 | 7.4 | 21 (32%) | 3.5 | 36 (55%) | 0.60 |
| baseline | 18 | 7.8 | 12.0 | 6 (33%) | 9.5 | 9 (50%) | 0.53 |
| bonds | 18 | 7.8 | 9.2 | 5 (28%) | 5.8 | 9 (50%) | 0.61 |
| baseline | 72 | 3.8 | 7.1 | 23 (32%) | 3.6 | 42 (58%) | 0.68 |
| **tmpl** | 72 | **1.1** | **2.6** | **41 (57%)** | **1.2** | **62 (86%)** | **0.82** |
| baseline (partial) | 19 | 4.0 | 4.5 | 9 (47%) | 2.4 | 12 (63%) | 0.75 |
| all (partial) | 19 | 1.0 | 1.8 | 13 (68%) | 0.9 | 17 (89%) | 0.90 |

**Findings**
- **The receptor template is the only large improvement:** the share of peptides within 4 A rose from 32% to 57%, and the share within 5 A of the real binding site from 58% to 86%. The median receptor error fell from 3.8 to 1.1 A. This matches the baseline analysis in which complexes with a badly folded receptor (5 A or more) placed the peptide correctly only 3% of the time.
- **More samples did not help when ranked by Boltz-2's own confidence** (31% vs 32%): the confidence score does not pick the best sample. Taking the best of 5 would reach 42%, but that needs the answer.
- **More recycling, more sampling steps and steering gave no gain** (65 complexes scored so far: 32% vs 34%).
- **Covalent bonds and cyclization helped the peptide placement a little in the median** (peptide RMSD 12.0 to 9.2 A, centroid 9.5 to 5.8 A, interface recall 0.53 to 0.61), but the count within 4 A did not rise (5 vs 6 of 18). Eighteen complexes is too few to be sure.

**Caveat for the template result:** the template is the receptor taken from the same PDB entry, in the conformation it has in the complex. It is an upper bound that assumes a good receptor structure is already known; for a real design the template would be an apo or homologous structure, and the gain would be smaller.

## 9. RF3 with the receptor given as a template (14 to 15 Nov 2026)

Question: if RF3 is given the receptor structure, does it place a non-canonical peptide as well as Boltz-2? RF3 alone failed because it could not fold the receptor from sequence (median receptor error 17 A).

**Setup.** One input file per complex: the real receptor chains (real coordinates, marked as templates), plus the peptide chain built by RDKit from its residue definitions (the conformer our synthetic library uses), moved 300 A away and not marked as a template, so the real pose cannot leak in. One structure per complex. Only the 32 complexes for which the RDKit peptide could be built were run (the other 51 contain residue codes the builder cannot handle). The comparison below uses the same 32 complexes for every model.

| Model | Complexes | Median receptor CA RMSD (A) | Median peptide CA RMSD (A) | Peptide < 4 A | Median centroid distance (A) | Centroid < 5 A | Median interface recall | Peptide passes strict geometry test |
|---|---|---|---|---|---|---|---|---|
| RF3 + receptor template | 32 | 0.1 | 2.0 | 19 (59%) | 1.2 | 27 (84%) | 0.83 | **16 (50%)** |
| RF3, no template | 32 | 18.7 | 23.2 | 0 (0%) | 17.0 | 3 (9%) | 0.24 | 6 (19%) |
| Boltz-2, no template | 32 | 4.2 | 4.7 | 15 (47%) | 2.7 | 19 (59%) | 0.71 | 3 (9%) |
| Boltz-2 + receptor template | 32 | 1.1 | 1.5 | 23 (72%) | 1.0 | 27 (84%) | 0.86 | 6 (19%) |

(The strict geometry test is the one used for the training pool; real PDB peptides pass it 55% of the time, section 7 control.)

**Findings**
- The template fixes RF3 completely: the receptor error falls from 18.7 to 0.1 A and the peptide lands within 4 A in 59% of complexes instead of 0%.
- With a template, Boltz-2 places the peptide slightly better (72% vs 59% within 4 A, 1.5 vs 2.0 A median), but both find the binding site equally often (84% within 5 A).
- RF3 + template produces chemically valid peptides far more often than Boltz-2 + template: 50% pass the strict geometry test, against 19%. This is close to the 55% of real PDB peptides. RF3 with a template is therefore the better engine for synthetic training complexes.

**Caveats:** 32 complexes only (those with a buildable peptide); the template is the receptor from the same PDB entry in its bound conformation (best case; real use would have an apo or homologous structure); one sample and one seed per complex.

## 10. Pilot: converting canonical peptides into non-canonical ones in rounds (15 Nov 2026)

Idea: start from real peptide-protein complexes and convert residues into non-canonical ones round by round, keeping the real receptor and peptide backbone. Round 1 of this idea was tested in two ways on the real complexes of section 4 (non-redundant items; scripts in `real_benchmark/pilot`).

**Method for one conversion.** Keep the real N, CA and C atoms of the residue. Build the new residue from its chemical definition with RDKit (up to 100 conformers), align each conformer on N-CA-C, and rank by how well it packs in the real environment (a Lennard-Jones-type term plus a clash penalty). Residues that change the backbone (beta, gamma) are not covered. 04Q (N-benzylglycine) was excluded because its atom names are non-standard.

**10.1 Fidelity test, run backwards on real data.** For 376 real positions holding one of the 32 residues with an ordinary backbone, the real new residue was removed and rebuilt, and compared with the real one (RMSD of its non-backbone heavy atoms).

| Side-chain size | Residues | Median RMSD of best-packing pick (A) | Within 1.5 A | Random conformer (median, A) | Best of 100 (median, A) |
|---|---|---|---|---|---|
| All | 376 | 0.40 | 85% | 1.23 | 0.29 |
| 1 to 2 atoms (AIB, D-Ala, ...) | 142 | 0.08 | 99% | 0.08 | 0.08 |
| 3 to 5 atoms | 168 | 0.66 | 87% | 1.31 | 0.49 |
| 6 or more atoms | 66 | 1.56 | 48% | 4.21 | 1.13 |

By family (median, within 1.5 A): D-alpha 0.58 A (74%), N-alkyl 0.63 A (87%), side-chain-modified 0.79 A (88%), alpha,alpha-disubstituted 0.11 A (95%). The median recovery of the real receptor contacts was 100% (precision 100%). Weakest residues: D-Arg 2.7 A (n=20), ORN 2.1 A (13), 2ML 1.6 A (12). The overall figure is helped by the many trivial residues; the 3 to 5 and 6+ atom rows are the fairer measure.

**10.2 Three forward rounds.** One more canonical residue per round converted (1, then 2, then 3 per complex) into a related non-canonical residue: round 1 side-chain-modified and alpha,alpha-disubstituted, round 2 adds N-alkyl, round 3 adds D-alpha. A conversion is accepted if no new side-chain atom is closer than 2.2 A to the environment and the new residue passes the strict per-residue geometry test.

| Round | Residues converted per complex | Complexes with a canonical position available | Accepted |
|---|---|---|---|
| 1 | 1 | 145 | 113 (78%) |
| 2 | 2 | 98 | 79 (81%) |
| 3 | 3 | 69 | 60 (87%) |

47 of the 192 complexes had no canonical residue to convert in round 1 (many benchmark peptides already consist mostly of non-canonical residues). Rejections were split roughly evenly between clashes and the geometry test. Converted peptides are saved in `real_benchmark/pilot/forward`, and per-round results in `forward_results.tsv`.

**Caveats:** a conversion that is geometrically valid is not proven to bind: N-methylation removes a hydrogen-bond donor and a D-residue inverts the side-chain direction. The packing score is a crude surrogate and the backbone stays fixed, so only residues with an ordinary alpha backbone were handled. The D-residues are placed on a backbone that belongs to an L-residue, which can be strained. Acceptance rates measure whether a plausible structure can be built, not binding.

## 11. Related work read for this project

- **NCFlow** (Lee and Kim, bioRxiv 2025): predicts the atoms of one non-canonical residue in a fixed protein pocket with the backbone fixed (atom-level flow matching, three training stages: 13.8M PubChem conformers, 293,591 Plinder protein-ligand complexes, 38,618 PDB non-canonical environments). Reported mean RMSD 1.43 to 1.58 A on its test set, against about 3.3 A for Boltz-1 and Boltz-2 (not a like-for-like comparison, as the paper notes). Single substitutions only. Checkpoints downloaded to the folder `ncflow_ref` under `C:/Research` (not yet tested).
- **Neopep** (Jagannathan and Kim, bioRxiv 2026): fine-tunes a Boltz-2-style model on synthetic non-canonical peptides with a machine-learned force-field energy as the reward (no real structures needed). Minimum CA RMSD over 25 samples on 34 non-canonical peptides: 1.57 A against 4.45 A for Boltz-2. Single peptides only, no receptor; no code release found.
- Both were read through partial text; their later results sections were available only as summaries.

## 12. NCFlow check, with attention to the backbone (15 Nov 2026)

Aim: find out whether NCFlow (Lee and Kim 2025) can substitute a non-canonical residue into a real structure and give a reasonable result, and what it does to the backbone. NCFlow checkpoint `base.pth` plus its confidence model, 4 samples per case, the sample with the highest pLDDT kept. 160 cases (100 reverse, 60 forward), all ran. 04Q (N-benzylglycine) was excluded from the scoring because its definition has non-standard atom names.

**12.1 Backbone: what NCFlow can and cannot do.**
- Its pool builder accepts a residue only if the SMILES contains exactly one N-C-C(=O)-O alpha-amino-acid backbone. Result for the 32 residues: alpha 12 of 12 accepted, N-alkyl 5 of 5, alpha,alpha-disubstituted 8 of 8, **beta 0 of 6 and gamma 0 of 1 rejected** (B3L, B3A, BAL, B3K, B3Q, XCP, STA).
- In the code (`extract_features`), the first four atoms of the replaced residue (N, CA, C, O) are copied from the original residue and held fixed; only the other atoms are predicted. NCFlow therefore cannot change the backbone, and it assumes an L-residue backbone for D-forms.

**12.2 Reverse test (a real non-canonical residue is removed and predicted again; 25 real cases per family).** Side-chain RMSD to the real residue, compared with the RDKit placement of section 10 on the same cases:

| Family | Cases | NCFlow median (A) | NCFlow < 1.5 A | RDKit median (A) | RDKit < 1.5 A | NCFlow passes strict test | RDKit passes strict test |
|---|---|---|---|---|---|---|---|
| D-alpha | 25 | 0.30 | 80% | 0.18 | 76% | 80% | 96% |
| N-alkyl | 25 | 0.31 | 92% | 0.84 | 88% | 64% | 72% |
| Side-chain modified | 25 | 0.16 | 100% | 0.89 | 88% | 80% | 88% |
| Alpha,alpha-disubstituted | 25 | 0.08 | 92% | 0.14 | 96% | 96% | 96% |
| **All** | 100 | **0.17** | **91%** | 0.52 | 87% | 80% | 88% |

By side-chain size: residues with 6 or more side-chain atoms (10 cases): NCFlow 0.69 A, 80% within 1.5 A, against 1.84 A and 40% for RDKit. Split by data age: recent cryo-EM entries (Tier A, 58 cases) 0.29 A, older X-ray and NMR entries (Tier B, 42 cases) 0.14 A. NCFlow's training snapshot is dated July 2023, so the Tier B figure may be flattered by overlap with its training data; Tier A is the cleaner test, and NCFlow still beats RDKit there (0.29 against 0.45 A).

**12.3 Forward test (a non-canonical residue is placed on a real canonical position; no answer exists, so the test is whether the result is reasonable).** A result is accepted if it is clash-free (nearest atom 2.2 A or more) and passes the strict per-residue geometry test, the same criteria as section 10.2. 57 cases:

| Family | Cases | NCFlow accepted | NCFlow clash-free | NCFlow geometry-valid | RDKit placement accepted |
|---|---|---|---|---|---|
| D-alpha | 21 | **0%** | 100% | **0%** | 76% |
| N-alkyl | 21 | 38% | 38% | 52% | 95% |
| Side-chain modified | 4 | 50% | 100% | 50% | 50% |
| Alpha,alpha-disubstituted | 11 | 100% | 100% | 100% | 100% |
| **All** | 57 | **37%** | 77% | 42% | **86%** |

**Findings**
- Where the backbone is already compatible (the real environment of that very residue), NCFlow places the new atoms very well, better than the RDKit placement, especially for larger side chains.
- Where the backbone must change to suit the new residue, it fails. D-residues placed on a canonical L backbone never pass the geometry test (wrong handedness), and N-alkyl residues clash in 62% of cases. This follows from the fixed N-CA-C-O and the L-backbone assumption.
- The two methods are complementary for side-chain work, but neither handles backbone-changing residues. NCFlow rejects beta and gamma residues outright, and the RDKit pilot also covers only alpha backbones.

**Caveats:** 100 reverse and 57 forward cases only; the reverse test is NCFlow's own task and its training data overlaps part of the real set; the strict geometry test checks the new residue in the peptide only, not the interface; code is MIT-licensed, one file (`utils/chirality.py`) is derived from AlphaFold 3 source code under CC BY-NC-SA 4.0.

## 13. Backbone test: can residues with a changed backbone be placed by rebuilding the gap? (15 Nov 2026)

Aim: the model has to understand non-canonical backbones first, so this tests residues whose backbone differs from alpha-L (beta, gamma, D, N-alkyl), which neither NCFlow (section 12) nor the pilot of section 10 can handle, because both keep the old residue's N, CA and C fixed.

**Method (`real_benchmark/pilot/pilot_backbone.py`).** The residue is removed completely, backbone included. The two real neighbours and the real receptor stay. The new residue is rebuilt from its chemical definition with RDKit, capped with an acetyl group at its N end and an N-methylamide at its C end. The five atoms of the caps are pinned (coordinate map) to C-alpha(i-1), C(i-1), O(i-1), N(i+1) and C-alpha(i+1) of the real neighbours, so the new backbone has to bridge the same gap. Between 40 and 60 conformers are generated and the best-packing one is kept (Lennard-Jones-type packing term in the real environment). Checks: peptide-bond lengths to both neighbours (1.33 +- 0.2 A), the strict per-residue geometry test of the training pool, and no atom closer than 2.2 A to the environment.

**13.1 Test A: rebuild a real residue in its own gap** (non-redundant real complexes; 56 beta, 6 gamma, 115 D-alpha, 35 N-alkyl, 114 alpha,alpha-disubstituted, 53 side-chain-modified; 119 canonical residues as control, sampled).

| Residue class | Built | All-atom RMSD, median (A) | Backbone-path RMSD, median (A) | All-atom < 1.5 A | Peptide bonds to neighbours valid | Strict geometry passes | Clash-free |
|---|---|---|---|---|---|---|---|
| Canonical (control) | 113 of 119 | 1.22 | 0.23 | 65% | 100% | 98% | 96% |
| Side-chain modified | 52 of 53 | 0.81 | 0.22 | 87% | 98% | 96% | 96% |
| Alpha,alpha-disubstituted | 114 of 114 | 0.51 | 0.25 | 96% | 100% | 96% | 96% |
| N-alkyl | 34 of 35 | 0.65 | 0.20 | 74% | 100% | 100% | 91% |
| D-alpha | 113 of 115 | 0.86 | 0.22 | 77% | 100% | 94% | 88% |
| **Beta** | 55 of 56 | 0.59 | 0.25 | 78% | 100% | 95% | 96% |
| **Gamma** | 6 of 6 | 1.40 | 0.70 | 67% | 100% | 100% | 100% |

The beta backbone is reproduced as accurately as the canonical control (0.25 A against 0.23 A). Weakest: gamma (STA, only 6 cases) and the long beta lysine B3K (about 2.1 A all-atom). This test is the easy direction: the real neighbours already sit where that residue needs them.

**13.2 Test B: swap a residue into a real canonical position, neighbours unmoved.** For 1,103 real canonical positions, each non-canonical residue that has that canonical residue as its parent was rebuilt in the alpha gap (2,984 attempts). A residue "fits" if a conformer bridges the gap with both peptide bonds valid. "Accepted" also requires the strict geometry test and no clash.

| Residue class | Attempts | Embedding failed | Gap mismatch | Fits the gap | Accepted (of all) |
|---|---|---|---|---|---|
| Alpha,alpha-disubstituted | 495 | 3% | 0% | 96% | 91% |
| **Beta** | 755 | 0% | 0% | 100% | **83%** |
| D-alpha | 707 | 3% | 1% | 96% | 79% |
| Side-chain modified | 322 | 1% | 1% | 99% | 75% |
| **N-alkyl** | 705 | 1% | 0% | 99% | **22%** |

By residue (accepted / tried): beta B3A 92%, B3K 82%, B3L 79%, B3Q 81%, BAL 82%; D-alpha DAL 93%, DPR 89%, DLE 87%, DVA 79%, DAR 71%, DPN 39%; N-alkyl SAR 35%, MAA 24%, MVA 22%, 04Q 18%, MLE 13%; side-chain modified 23F 34%, others above 92%. Gamma (STA) and XCP were not tested here because the catalog gives them no canonical parent.

**Findings**
- A beta residue can usually be rebuilt in the gap left by an alpha residue without moving the neighbours (all beta attempts fit, 83% pass the geometry and clash tests): the extra backbone carbon is absorbed by changed torsion angles.
- N-alkyl residues are the exception: only 22% are accepted, mostly because the N-substituent clashes with its surroundings. This agrees with the NCFlow result (38%).
- D residues fit the gap in most cases (79%), except the bulky D-Phe (39%).

**Caveats.** "Accepted" means valid bond lengths, angles, handedness, peptide bonds and no clash. It does not measure strain in the new backbone torsions, whether backbone hydrogen bonds to the receptor are kept, or whether the peptide still binds. Swapping a residue usually changes more than the residue itself (neighbours, hydrogen-bond pattern). Gamma is covered only in Test A (6 cases). Candidate residues were limited to the 32 residues of this project.

## 14. Figures (one per test)

Made by `make_figures.py` (this folder); the numbers are the ones in the tables above.

| Figure | Test | Section |
|---|---|---|
| ![Backbone tests](figures/fig1_backbone_tests.png) | Backbone test: rebuild a residue between its real neighbours, and swap residues into a canonical gap | 13 |
| ![NCFlow check](figures/fig2_ncflow_check.png) | NCFlow check: side-chain placement on a fixed backbone, compared with RDKit placement | 12 |
| ![Complex pose](figures/fig3_complex_pose.png) | Boltz-2 and RF3 on real peptide-protein complexes, with and without a receptor template | 7, 9 |
| ![Autoencoder](figures/fig4_autoencoder.png) | Autoencoder recovery of residue names on real PDB peptides | 4 |
| ![Conversion pilot](figures/fig5_conversion_pilot.png) | Round-by-round conversion of canonical peptides | 10 |
| ![Boltz-2 variants](figures/fig6_boltz2_variants.png) | Ways to improve Boltz-2 on the complex test | 8 |
| ![RF3 peptide template](figures/fig7_rf3_peptide_template.png) | RF3 with the peptide also given as a template | 15 |
| ![RF3 shape hint](figures/fig8_rf3_shape_hint.png) | RF3 with the peptide shape given but not its position | 16 |
| ![NCFlow fine-tune](figures/fig9_ncflow_finetune.png) | NCFlow with the backbone released: raw model and a short fine-tune | 17 |
| ![RF3 pose substitution](figures/fig10_pose_substitution.png) | RF3 pose substitution: our residues in the real peptide pose | 18 |
| ![Reward fine-tune setup](figures/fig11_reward_finetune_setup.png) | NCFlow reward fine-tune: how it is set up (results to follow) | 19 |
| ![NCFlow strict test](figures/fig12_ncflow_strict_test.png) | NCFlow strict test on held-out residues and iterative conversion | 20 |
| ![NCFlow handedness](figures/fig13_handedness.png) | NCFlow handedness: coin flip, handedness input, best of 4 by the check | 21 |
| ![Sample and check](figures/fig14_sample_and_check.png) | Draw several samples and keep one that passes the check (pass at k) | 21 |
| ![Conversion with check](figures/fig15_conversion_with_check.png) | Iterative conversion of 31 real complexes: top sample / check + retries / + mirror repair | 22 |
| ![Natural complex list](figures/fig16_natural_complex_list.png) | 31 natural peptide-protein complexes to start from | 23 |
| ![Backbone options](figures/fig17_backbone_options.png) | Backbone changes and what they do to the side-chain positions | 23 |
| ![Natural peptide map](figures/fig18_natural_peptide_map.png) | Per-residue map of 30 natural peptides: interface and where a backbone change fits | 23 |
| ![Rewrite rule](figures/fig19_rewrite_protocol.png) | The rewrite rule (interaction map kept at 0.9) and how often today's generator meets it | 24 |
| ![Split v1](figures/fig22_split_v1.png) | Frozen train / dev-test / blind-test split of the targets (v1) | 25 |
| ![Split v2](figures/fig23_split_v2.png) | Split v2: all real non-canonical complexes are test only | 26 |
| ![Energy test](figures/fig24_energy_test.png) | Relaxed interface energy: alanine scan, repair control, conversions | 27 |
| ![Pilot smoke test](figures/fig25_pilot_smoke.png) | Pilot generator smoke test: gate outcomes, accepted swaps by residue | 28 |
| ![Pool docking](figures/fig26_pool_docking.png) | Docking and design of pool peptides into 264 train pockets: self-test, gate funnel, score vs natural, residue usage | 29 |
| ![NCFlow synthetic only](figures/fig27_ncflow_synthetic_only.png) | NCFlow raw vs synthetic-only fine-tunes: strict pass, bond and link defects, proline analogues | 30 |
| ![Synthetic only vs real](figures/fig28_syn_only_vs_real.png) | Synthetic-only NCFlow against ft2 and rl (trained with real data), same held-out cases | 31 |
| ![Pilot laptop](figures/fig29_pilot_laptop.png) | Pilot generator with rl_syn, laptop share (40 complexes): draw outcomes, accepted swaps by residue, contacts kept, residues produced | 32 |
| ![Pilot all](figures/fig30_pilot_all.png) | Pilot generator, all 264 train complexes: draw outcomes, accepted swaps by residue, contacts kept, residues produced | 33 |
| ![Pilot CIFs](figures/fig31_pilot_cifs.png) | Synthetic complexes written as CIF: length, non-canonical share, re-check of the files | 34 |
| ![Rosetta pilot](figures/fig32_pilot_rosetta.png) | Rosetta interface energy, natural vs converted peptide, 211 valid pairs | 35 |
| ![Training check](figures/fig34_training_check.png) | Training recipe check on the synthetic complexes: Stage 0, vocabulary coverage, smoke tests | 36 |
| ![Real pose test](figures/fig35_real_pose_synthetic_head.png) | Decoder head trained on masked synthetic complexes, tested on 189 real peptides: accuracy, per code, vs training examples | 37 |
| ![Real pose 189](figures/fig36_real_pose_189_decoder.png) | Whole-decoder 189-class model vs run2 overlay on the relabelled real benchmark: accuracy by group, the 13 new D-forms, recall vs synthetic examples | 38 |
| ![Decoder run 2](figures/fig37_decoder_run2_real_pose.png) | Decoder retrain on 1,429 synthetic complexes (run 2) vs run 1 vs untrained rows on 189 real peptides; six weak D-forms; held-out loss | 41 |
| ![Trials vs control](figures/fig38_trials_heldout_test.png) | Denoiser fine-tune trials A, B, C vs the untuned control on all 163 held-out complexes (paired, same noise): mean loss, paired change with interval, by loss term | 43 |

Rule for later work: every new test gets a figure (method schematic plus result chart) added here.

## 15. RF3 with the peptide's own structure also given as a template (15 Nov 2026)

![RF3 peptide template](figures/fig7_rf3_peptide_template.png)

Question: does it help RF3 to mark the RDKit-built peptide structure as a template, in addition to the receptor template of section 9? The peptide structure is placed 300 A from the receptor (so the real pose cannot leak) and the peptide chain was added to the template list. Same 32 complexes and measures as section 9, one H200 job.

| Input | Peptide < 4 A | Centre < 5 A | Median peptide CA RMSD (A) | Interface recall | Passes strict geometry test |
|---|---|---|---|---|---|
| Receptor template only (section 9) | 19 (59%) | 27 (84%) | 2.0 | 0.83 | 16 (50%) |
| Receptor and peptide as templates | 0 (0%) | 17 (53%) | 13.3 | 0.52 | 8 (25%) |
| No template | 0 (0%) | 3 (9%) | 23.2 | 0.24 | 6 (19%) |

**Finding:** marking the far-away peptide as a template made RF3 worse on every measure, including the chemistry of the predicted peptide. A likely cause is that RF3 reads the 300 A separation between peptide and receptor as part of the template, but this was not tested separately. Giving RF3 only the peptide's internal geometry, without a position (for example through its ground-truth-conformer option or custom residue definitions), remains to be tested.

## 16. RF3 with the peptide's shape given but not its position (15 Nov 2026)

![RF3 shape hint](figures/fig8_rf3_shape_hint.png)

Follow-up to section 15. RF3's ground-truth-conformer option gives the model a per-residue reference shape that RF3 re-centres on each residue's own origin, so the hint carries no position. The receptor is still a template; the peptide is not a template. Same 32 complexes, one H200 job (481188, 47 predictions). Inputs are built by `build_rf3g.py`; scoring by `score_complex.py` (models rf3g, rf3r) and the strict geometry test.

| Input | Complexes | Peptide < 4 A | Peptide < 2 A | Centre < 5 A | Median peptide CA RMSD (A) | Median interface recall | Strict geometry pass |
|---|---|---|---|---|---|---|---|
| Baseline: receptor template only | 32 | 19 (59%) | 16 | 27 | 2.00 | 0.83 | 16 (50%) |
| + our RDKit peptide shape | 32 | 20 (62%) | 15 | 27 | 2.48 | 0.85 | 12 (38%) |
| Baseline, 15-complex subset | 15 | 10 | 9 | 12 | 1.68 | 0.83 | 7 |
| + our RDKit peptide shape, same 15 | 15 | 11 | 8 | 13 | 1.78 | 0.90 | 8 |
| + the REAL peptide shape, same 15 (upper bound) | 15 | 10 | 9 | 12 | 1.87 | 0.87 | 6 |

**Finding:** giving RF3 the peptide shape does not improve the result. Across the 32 complexes, 5 flip across the 4 A line, in both directions (7AUE_C and 7LOK_G become wrong; 7F55_L, 7PIU_P and 7VAB_P become right), which is single-sample noise. The strict chemistry pass rate moves from 50% to 38% (16 to 12 of 32), which is within what 32 complexes can show but gives no sign of a gain. Even the real conformation of the peptide, the best possible hint, gives no gain (10 of 15 both times). RF3 therefore already builds the residue geometry correctly from its own chemistry; what limits it is the pose and the stability of the whole peptide, not the starting shape of each residue.

Limits: one sample per complex, 32 (or 15) complexes, so differences of one or two complexes are not meaningful. The real-shape test covers only the 15 complexes whose real peptide matches the built peptide atom by atom. The 11 complexes with residue codes missing from the dictionaries still cannot be run, because RF3 needs a definition for those residues and the option only supplies a hint, not a definition.

## 17. NCFlow, backbone released: raw model and a short fine-tune (15 Nov 2026)

![NCFlow fine-tune](figures/fig9_ncflow_finetune.png)

Question: can NCFlow generate a whole residue, backbone included, between its two real neighbours, and can it be fine-tuned on our data for that? Cases: real residues of every backbone class from the benchmark (alpha, side-chain modified, alpha,alpha-disubstituted, N-alkyl, D-alpha, beta, gamma, plus 3% canonical as a control) with both neighbours present: 342 cases, split by PDB entry into 252 training and 90 held-out. The residue is removed, NCFlow gets the real complex without it and must generate all atoms (backbone atoms are not copied from the real residue, unlike NCFlow's normal mode). Each case is scored on 2 samples against the real residue. Evaluation set: 90 held-out cases plus 40 training cases (129 of 130 scored per model).

Models: (a) released base checkpoint (`base.pth`), (b) released plinder checkpoint (`plinder.pth`), (c) `base.pth` fine-tuned by us for 1,500 steps (about 6 passes over 251 training examples, batch accumulation 4, learning rate 3e-5, flow-matching loss, 4 minutes on an H200). The first run (job 481120, 16 GB slice) evaluated (a) and (b) and ran out of memory in training; job 481363 on a full H200 reused (a) and (b) and ran (c).

**Held-out cases (89 scored):**

| Model | Median all-atom error (A) | Median backbone error (A) | < 1.5 A | Cases with a peptide-link length > 0.2 A off | Cases with a bad bond |
|---|---|---|---|---|---|
| Raw base | 0.73 | 0.50 | 89% | 73% | 31% |
| Raw plinder | about 2 to 3 | about 1.4 to 2.5 | 0 to 21% (by family) | about 100% | 41 to 73% (by family) |
| Fine-tuned base | 0.36 | 0.21 | 91% | 16% | 1% |

Training cases show the same direction (all-atom 0.52 to 0.31 A, backbone 0.52 to 0.21 A), so the gain on held-out entries is not memorisation.

**By family, held-out, raw base to fine-tuned base (median backbone error in A; cases with a bad bond):**

| Family | Cases | Backbone error | Bad bond |
|---|---|---|---|
| Canonical (control) | 11 | 0.57 to 0.17 | 45% to 0% |
| Alpha, modified (side-chain modified and alpha,alpha-disubstituted) | 36 | 0.42 to 0.21 | 14% to 0% |
| D-alpha | 19 | 0.52 to 0.19 | 21% to 0% |
| N-alkyl | 11 | 0.94 to 0.57 | 64% to 9% |
| Beta and gamma | 12 | 0.68 to 0.23 | 58% to 0% |

**Findings:**
- The raw base model already puts the residue in about the right place (0.7 A all-atom, backbone 0.5 A), for beta and D residues too. Its weakness is bond geometry: about 73% of cases have a peptide link more than 0.2 A off the normal 1.33 A, and 31% have a bad bond. The raw plinder checkpoint is much worse and should not be used for this.
- A 4-minute fine-tune on 251 of our cases removes nearly all of that: backbone error 0.50 to 0.21 A, bad bonds 31% to 1%, link problems 73% to 16%. Beta/gamma end at 0.23 A backbone error, 0% bad bonds, which is on par with my gap-rebuild method on beta (0.25 A backbone, 0.59 A all-atom, 95% strict pass; different case sets, so not an exact comparison). N-alkyl is the weakest class (0.57 A, 9% bad bonds).
- The canonical control improves just as much, so a large part of the gain is the model adapting to the task format (generate the backbone too and join it to the neighbours), not NCAA-specific learning.

**Limits:** small data (12 beta/gamma and 11 N-alkyl held-out cases, only 6 gamma overall), one fine-tune run, one seed, 2 samples per case. The pass criteria here are bond lengths and error, not the full strict per-residue test of the pool (angles, clashes, handedness) and not binding quality. Next steps would be to run the pool's strict test on the fine-tuned outputs, test more steps and more data (our synthetic structures), and test inside the receptor-bound setting with the chemistry-aware swap of section 13.

## 18. RF3 pose substitution: our residues in the real peptide's pose (15 Nov 2026)

![RF3 pose substitution](figures/fig10_pose_substitution.png)

Idea: take a real peptide-protein complex, keep the receptor and the real binding pose as templates, change residues of the peptide to non-canonical ones, and let RF3 refold (interface and geometry). Level 1 is a control (same residues, receptor and real peptide both marked as templates, at the real position). Level 2 swaps 1, 2 or 3 residues for non-canonical residues (the accepted conversions of the round-by-round pilot, section 10; the converted residue keeps the real backbone atoms, other atoms are built by RDKit). Inputs by `build_pose_sub.py`, one H200 job (481402, 68 predictions), scored with `score_complex.py` (models l1, l2r1, l2r2, l2r3) and the strict geometry test. The baseline is RF3 with the receptor template only on the same complexes.

| Input | Complexes | Peptide < 2 A | Median peptide CA RMSD (A) | Median contact recovery | Median interface recall | Whole peptide passes strict test | Same, baseline |
|---|---|---|---|---|---|---|---|
| Level 1 control (same residues) | 32 | 26 (81%) | 0.56 | 0.87 | 0.94 | 15 (47%) | 15 (47%), pose 16 (50%) < 2 A, median 2.00 A |
| Level 2, 1 residue swapped | 20 | 17 (85%) | 0.54 | 0.88 | 0.94 | 10 (50%) | 8 (40%), pose 11 (55%), median 1.55 A |
| Level 2, 2 residues swapped | 10 | 8 (80%) | 0.41 | 0.91 | 0.97 | 3 (30%) | 6 (60%), pose 3 (30%), median 6.98 A |
| Level 2, 3 residues swapped | 5 | 4 (80%) | 0.38 | 0.94 | 0.97 | 0 (0%) | 4 (80%), pose 1 (20%), median 3.30 A |

Where the strict test fails (residue level): swapped residues fail in 5 of 20 (1 swap), 5 of 20 (2 swaps) and 5 of 15 (3 swaps), mostly bond length and peptide-link length errors; the other residues of the peptide fail mainly by clashes (the control has the same pattern, 33 residues with at least one failure over 32 complexes).

**Findings:**
- The pose transplant works for the pose: the peptide stays 0.4 to 0.55 A from the real pose and the interface recall is 0.94 to 0.97, against 2.0 A and 0.83 for the receptor-template-only baseline. This is expected, since the real pose is given as a template. It is not a prediction of the pose, but it shows RF3 accepts a template on a peptide with swapped non-canonical residues and keeps it.
- It does not improve chemistry. Swapped residues pass the per-residue test about 67 to 75% of the time, and the whole-peptide pass rate falls with more swaps (50%, 30%, 0%, against 40%, 60%, 80% for the baseline). RF3 follows the template and does not repair the geometry of the swapped residues. A plausible reason is that RF3 copies the RDKit-built atoms given in the template, but this was not tested separately.
- Small numbers: 10 complexes at 2 swaps and 5 at 3 swaps. The trend for the strict pass is a hint, not a result.

**Limits:** the starting poses are real, so the pose numbers are not a pose-prediction accuracy. One sample per complex. The strict test is for the whole peptide (all residues must pass), which is a harsh test for 14 to 36-residue peptides.

## 20. NCFlow strict test, larger fine-tune and iterative conversion (15 Nov 2026)

![NCFlow strict test](figures/fig12_ncflow_strict_test.png)

Test: hide one residue of a peptide, let NCFlow redraw all its atoms (backbone included; 4 samples, the one with the best NCFlow confidence is kept), and run the strict per-residue test of the training pool (bond lengths, angles, clashes, peptide link to the neighbours, handedness; all must pass). Three held-out sets: 89 real residues of PDB entries not used in training; 39 residues of RF3-predicted peptide-receptor complexes (synthetic complexes); 180 residues of our synthetic pool peptides (30 per residue type, peptide only, files not used in training). Models: raw (released checkpoint, backbone released), ft1 (252 real residues, 1,500 steps), ft2 (the same plus about 3,950 synthetic residues, 8,000 steps). Code: `ncf_lib.py`, `ncf_run2.py`, `ncf_strict_lib.py`, `ncf_strict_run.py` in `ncflow_ref` and `real_benchmark/pilot`.

| Set | Raw | ft1 | ft2 | The original residue (best possible) |
|---|---|---|---|---|
| Real (89) | 29% | 72% | 72% | 99% |
| RF3 complexes (39) | 26% | 64% | 67% | 82% |
| Synthetic (180) | 10% | 46% | 43% | 100% |

On the synthetic set the share of residues failing each check (raw, ft1, ft2): link to the next residue 82, 34, 29%; wrong handedness 38, 29, 34%; bond length 59, 13, 7%; bond angle 44, 7, 5%; atoms too close 14, 6, 7%. Best of the 4 samples would pass in 81% (ft2), against 43% for the top-confidence sample.

**Iterative conversion:** 11 real peptide-receptor complexes (from the 32 complexes of section 9), 88 swaps in total. One canonical residue per round (random order) is replaced by a non-canonical one until none is left (GLY, PHE, PRO, ALA, LEU, VAL, ARG use residues of the 32-code vocabulary; the other 13 canonical residues are replaced by their D-forms, which were not in the training vocabulary). NCFlow redraws each swapped residue from its context, including the previously swapped residues. Swapped residues that fail the strict test: raw 97%, ft1 69%, ft2 74%. Average bond-length error of swapped residues: 0.148, 0.057, 0.045 A. Average bond-angle error: 11.8, 5.2, 5.0 deg. The peptide stays in place (contacts with the receptor kept: 95%, 95%, 97%; the CA atoms move 0.52, 0.26, 0.30 A).

**Findings:**
- Training helps a lot (10 to 46% on synthetic, 29 to 72% on real), but more data did not help: ft2 is the same as ft1 within the noise of the test sets (180 and 89 residues).
- Bond lengths and angles are mostly solved. The two main failures are wrong handedness (about a third of synthetic residues, 21 of 30 gamma residues) and a wrong bond to the neighbouring residue (29%). The best-of-4 rate (81%) shows the model can produce a passing residue, but it does not rank it first.
- In the iterative conversion the pose is kept but about 70% of swapped residues still fail the strict test, mostly for the same two reasons.
- The first version of the reward fine-tune (penalty on bond lengths, angles and link only) was cancelled before it started, because it would have left handedness untouched. The version that was run adds a handedness term.

**Reward fine-tune (rl):** starts from ft2, 6,000 steps on the same real plus synthetic residues, with the loss = flow matching + 1.0 x (1 - t)^2 x [bond-length strain + bond-angle strain + peptide-link strain + handedness penalty], measured on the model's one-step estimate of the clean residue (`ncf_rl.py`; job 482117, 13 minutes of training). Results on the same held-out sets and the same iterative conversion:

| Measure | ft2 | rl |
|---|---|---|
| Pass rate, real (89) | 72% | 74% |
| Pass rate, RF3 complexes (39) | 67% | 64% |
| Pass rate, synthetic (180) | 43% | 51% |
| Synthetic residues failing bond length | 7% | 0% |
| Synthetic residues failing bond angle | 5% | 0% |
| Synthetic residues failing link to next residue | 29% | 25% |
| Synthetic residues failing handedness | 34% | 38% |
| Iterative conversion: swapped residues failing | 74% | 74 to 76% |
| Bond-length error of swapped residues (A) | 0.045 | 0.029 |
| Bond-angle error of swapped residues (deg) | 5.0 | 3.7 |

**Finding:** the reward removes all bond-length and bond-angle failures (0% on all three sets) and makes the swapped residues' strain the lowest of all models. It does not fix handedness (38% of synthetic residues, 31% of RF3 complexes, 16% of real) and only slightly the link to the next residue (25%), so the overall pass rate improves only on the synthetic set (43 to 51%) and the iterative conversion is unchanged. A likely reason is that the model has no input that says whether the residue should be the L or the D form (the input holds atom types, bonds and the context, and mirror images of a residue have identical bonds); it can only guess from the neighbours' geometry. A reward cannot teach an answer the model is not given the question for. This is a diagnosis from the pattern of failures (D-alpha, beta and gamma fail most), not yet tested. The next step would be to add a handedness input (a flag on the stereocentre atoms), or to mirror the environment for D residues, and then re-train.

## 21. NCFlow handedness: a handedness input does not help; checking the samples does (15 Nov 2026)

![NCFlow handedness](figures/fig13_handedness.png)

After section 20 the main remaining failure was wrong handedness (a residue drawn as its mirror image). Two checks. First, among residues that have a stereocentre, the share drawn with the wrong handedness (signed volume of three neighbours around the stereocentre, compared with the reference conformer of the definition; this agrees with the strict test's handedness verdict on all 125 synthetic residues): raw, ft1, ft2, reward, reward + handedness input = 52, 42, 49, 54, 50% on synthetic peptides (125 residues); 33, 14, 22, 22, 25% on real residues (64); 39, 50, 56, 67, 67% on RF3-predicted complexes (18). A model that does not know the answer scores 50%, so on synthetic peptides, where the context holds no handedness cue, none of the models has learned it.

Second, the handedness input (`ncf_lib.ChiLinear`, `build_chi`): 4 extra input channels, the required handedness on each stereocentre atom (+1 or -1 from the definition) and the order of its three neighbours; new weights start at zero and train with step size 1e-3; model trained 8,000 steps from the reward model (job 482255, loss = flow matching + 1.0 x (1 - t)^2 x (bond, angle, link, handedness penalties)). Result: pass rate 54% on synthetic (reward model 51%), 74% real, 64% RF3 complexes; wrong handedness 50% on synthetic (reward model 54%). Diagnostic (job 482394, 125 residues, 2 samples): with the input set to zero 46% are wrong, with the input flipped 46%, as given 49% (4 samples). The model therefore does not use the input in any useful way: the learned input weights are small (mean 0.026, against 0.37 for the atom-type weights).

**What works:** picking, among the 4 samples of a residue, the first one that passes the strict check. On synthetic peptides this gives 81% passing for ft2, the reward model and the handedness model (19% raw, 70% ft1), against 43 to 54% for the sample NCFlow ranks first; on real residues 91 to 93%, on RF3 complexes 85 to 92%. Handedness is close to a coin flip per sample, so more samples raise it quickly (4 samples: about 94% right at 50% each).

**Conclusion:** use NCFlow with sample-and-check (draw 4 or more, keep a sample that passes the strict test) for generating residues; the reward fine-tune removes bond and angle failures and is worth keeping, the handedness input is not. Not tested: a stronger input (larger channel scale, or a handedness input to the pair features), longer training, mirroring the surroundings for D residues.

**Sample-and-check, in detail** (`passk.py`; figure below). Share of residues for which at least one of the first k samples passes the strict test, k = 1, 2, 3, 4. Synthetic peptides (180): ft2 46, 64, 74, 81%; reward model 52, 68, 77, 81%; handedness model 55, 70, 77, 81%. Real residues (89, reward model): 81, 91, 92, 93%. RF3 complexes (39, reward model): 67, 77, 87, 87%. If handedness were the only problem and each sample were a coin flip, four samples would pass 96% of synthetic residues; the observed 81% is lower because some residues fail for another reason in every sample (mostly the bond to the neighbouring residue, and gamma residues, 53% for ft2). The check uses only the residue's definition, not the true answer, so it can be applied when designing. Eight samples were not measured. The iterative conversion of section 20 used the top-ranked sample of each swap, not sample-and-check; it has not been repeated with the check.

![Sample and check](figures/fig14_sample_and_check.png)

## 22. Iterative conversion of real peptides with the check, retries and a mirror repair (15 Nov 2026)

![Conversion with check](figures/fig15_conversion_with_check.png)

Test: 31 real peptide-receptor complexes of the benchmark (peptide of 6 to 25 residues with at least 60% canonical residues; the real peptide is chain P of `truth_complex`, the receptor chains in contact), 227 planned swaps. One canonical residue per round, in random order, is replaced by a non-canonical residue of the same parent (vocabulary residues, or the D-form for the 13 parents without one; DME of the first run was the wrong code, dimethoxyethane, and was replaced by MED, so the first run of this section was repeated). NCFlow (the reward fine-tune of section 20) redraws the residue in the real pocket, 4 samples per draw. Three variants: top-confidence sample with no check (`top1`); in-loop strict check (`ncf_check.py`, identical to the pool's test on all 308 saved samples), up to 3 rounds of 4 draws for the planned residue then up to 2 alternative residues, a swap with no passing sample leaves the residue canonical (`check`); the same plus a mirror repair (a draw that fails only on handedness is mirrored through the plane N-CA-C and checked again; `repair`). Code `ncf_run2.py iter2`, analysis `ncf_iter2_analyze.py`; real-interaction measures per round against the starting complex.

| Measure | Top sample only | Check + retries | Check + retries + mirror repair |
|---|---|---|---|
| Swaps accepted (of 227) | 227 | 156 | 198 |
| Accepted swaps that pass the check | 65 | 156 | 198 |
| Swaps left canonical | 0 | 71 | 29 |
| Residues failing the strict test in the finished peptides (start: 21 of 271) | 162 of 271 | 5 | 8 |
| Finished peptides passing the whole strict test (start: 19 of 31) | 1 | 28 | 25 |
| Peptides with every planned swap done | 31 | 2 | 15 |
| Draws per planned swap | 4.0 | 8.3 | 6.1 |
| Residue contacts with the protein kept (5 A, median) | 92% | 93% | 88% |
| Polar contacts (N/O to N/O within 3.5 A) kept | 86% | 89% | 83% |
| Heavy-atom contacts within 4.5 A, relative to the start | 101% | 102% | 104% |
| Peptides with an atom within 2.2 A of the protein | 7 | 0 | 0 |
| CA movement (median) | 0.34 A | 0.47 A | 0.48 A |

**Findings:**
- Without the check, 71% of the swapped residues are chemically wrong (162 of 227), 7 of 31 peptides have an atom touching the protein, and only 1 finished peptide passes the strict test. With the check, every accepted swap passes, no peptide touches the protein, and 25 to 28 of 31 finished peptides pass the whole strict test (19 of 31 did at the start, because some original residues were replaced).
- The price is residues left canonical: 71 of 227 with check + retries (31%), 29 (13%) with the mirror repair. Fully converted peptides: 2 and 15 of 31. The mirror repair is what makes complete conversion possible.
- The swaps left canonical fail almost always on handedness (820 failed draws for check + retries, 306 with the repair), then the bond to the neighbouring residue (about 90), atoms too close (55); bond lengths and angles almost never. The D-form is drawn as the L-form because the surroundings are an L peptide (12 of 12 draws wrong in many cases); retries cannot fix a systematic error, the mirror repair can for residues with a single stereocentre. ILE and THR (two stereocentres), CYS, LYS and TRP are the parents most often left canonical.
- The interaction with the protein is kept: about 90% of residue contacts and polar contacts, the number of atom contacts unchanged, peptide movement under 0.5 A. This is a geometric measure of the interaction (contacts kept, no clash); it is not a binding energy, and real D-residue substitutions may change binding.

**Limits:** swaps are random choices of residue per parent, not designed; one run per variant; the starting peptides are real PDB peptides (mostly canonical residues with some non-canonical ones), not designed binders. A re-docking test (RF3 or Boltz-2 on the converted sequences) was not run.

## 23. Parked ideas and lists

- **Natural peptide-protein complexes to start from** (31 complexes, 2.0 A or better, canonical peptides, one per protein family): `natural_complexes/natural_complexes.md`; figure below. The structures were downloaded (12 MB) and screened: secondary structure, interface, backbone contacts, where an N-methyl or alpha-methyl fits (figure 18).
- **Change only the backbone, keep the side-chain interactions** (options, recipe, how it fits the denoiser fine-tuning, open questions): `Ideas_backbone_swap.md`.

![Natural complex list](figures/fig16_natural_complex_list.png)
![Backbone options](figures/fig17_backbone_options.png)
![Natural peptide map](figures/fig18_natural_peptide_map.png)

## 24. The rewrite rule: keep at least 0.9 of the interaction map (defined 16 Nov 2026, not yet implemented)

![Rewrite rule](figures/fig19_rewrite_protocol.png)

Definition. The interaction map of a peptide-protein complex has three layers, measured on the natural complex (the reference) and on every candidate in the same receptor frame: (a) residue pairs, peptide residue with protein residue, with any heavy atoms within 5 A; (b) polar pairs, N or O to N or O within 3.5 A, at residue-pair level; (c) heavy-atom contacts within 4.5 A (a count). Retention of (a) and (b) is the share of the reference pairs still present. A residue rewrite is accepted only if a candidate passes all gates: (1) the strict chemistry test (bonds, angles, clashes, link, handedness); (2) no atom within 2.2 A of the protein; (3) retention of (a) and of (b) at least 0.9 against the natural complex, and one swap may lose at most 0.05; (4) pose: CA RMS movement at most 1.0 A (no single CA more than 1.5 A) and the contact count (c) between 0.9 and 1.3 of the start. Among the candidates that pass, the one with the highest retention is kept (ties by NCFlow confidence); swaps are tried from the least interacting position to the most interacting one. If nothing passes, the fallback ladder runs: more draws (up to 36), mirror repair, alternative residue (up to 2), a geometric builder (to be built), then the residue stays canonical and is reported as not rewritable under the rule. A finished peptide counts as converted if all five criteria hold (the four above for the whole peptide and the strict test for every residue).

Feasibility on the conversion run of section 22 (31 real complexes; the rule applied afterwards to the finished peptides, not used to choose swaps):

| Criterion | Top sample | Check + retries | Check + retries + mirror repair |
|---|---|---|---|
| Residue contacts >= 0.9 | 21 of 31 | 23 | 13 |
| Polar contacts >= 0.9 | 10 | 13 | 10 |
| CA drift <= 1.0 A | 31 | 30 | 31 |
| No protein clash | 24 | 31 | 31 |
| Strict test on every residue | 1 | 28 | 25 |
| All five | 0 | 10 | 8 |

One accepted swap loses a median of 0 of the contact map (90th percentile 0.05 with check + retries, 0.053 with the repair) and 0 of the polar map (90th percentile 0.111 and 0.125); the median retention after 6 or more swaps falls to 0.85-0.9 for contacts and 0.82-0.89 for polar contacts. So the 0.9 rule is demanding: polar contacts are the bottleneck, and the rule must be used inside the choice of each swap (choose the candidate with the highest retention, skip positions that cannot meet it) instead of being checked at the end. Limits: the interaction map is geometric (contacts and distances), not a binding energy; D residues and other backbone changes that move a side chain cannot keep its contacts, and the rule will correctly leave those positions unchanged. An independent pose check (re-docking the converted sequence with RF3 or Boltz-2 and comparing the site and the interface) is planned.

## 25. Frozen train / test split of the targets, v1 (18 Nov 2026)

![Split v1](figures/fig22_split_v1.png)

Rules (the user's requirement: the model must not know the pose beforehand, and the denoiser is never trained on a target it is tested on): (1) at test time only the receptor structure and the peptide spec are given, never peptide coordinates or a pose template; (2) the unit is a group of complexes linked by receptor family (RCSB 30 % sequence-identity cluster) or by a similar peptide (similarity at least 0.75), so no homolog or similar peptide sits on both sides; (3) test groups are every group whose entries were all released on or after 2025-01-01 (tier post_cutoff, the models' training cutoffs are not verified) plus about 20 % of the older groups chosen by a salted hash (tier older, at least 3 with non-canonical benchmark complexes); groups with 8 or more entries (large receptor families such as the 24-entry GPCR group) stay in train; (4) test groups alternate between dev test and blind test, the blind set is not inspected during development; (5) no test receptor or homolog may be used for training, as a conversion start, as an RF3 template for data generation, or in any training example; (6) NCFlow models were fine-tuned on residues of 113 benchmark entries, 13 of which fall in test families, so NCFlow must be retrained without them before it is used to generate training data. Metadata from RCSB only; file `splits/split_v1.json` with its sha256 in `splits/split_v1.sha256.txt`; code `splits/freeze_split.py` and `splits/split_guard.py` (refuses test entries and receptors similar to a test receptor; checks the hash).

| Split | Groups | Entries | Non-canonical benchmark complexes | Natural complexes |
|---|---|---|---|---|
| Train | 124 | 188 | 63 | 25 of the 31 listed in section 23 |
| Dev test | 21 (8 post-cutoff, 13 older) | 24 | 7 | 3 (9F7W, 7RKE, 1CKA) |
| Blind test | 21 (10 post-cutoff, 11 older) | 30 | 13 | 3 (6XZZ, 1D4T, 9MXD) |

The 83 non-canonical benchmark complexes are only 23 receptor families, so the test families number few; more targets are needed for a robust test (the RCSB search can be widened beyond 2.0 A and to cryo-EM). Dev and blind are not balanced (7 and 13 benchmark entries). The natural complexes in the test split (6 of the 31) must not be used for conversions or training; the 63 benchmark entries and 25 natural complexes in train can.

## 26. Split v2: all real non-canonical complexes are test only (18 Nov 2026)

![Split v2](figures/fig23_split_v2.png)

The user's rule (18 Nov 2026): keep all real non-canonical complexes as the testing subject; train and validation data must come from synthetic data made with NCFlow and RF3, "to see how good they are at first". Split v2 supersedes v1 (v1 was never used for training; its natural test entries are released). Rules: (1) test = every entry of the benchmark of real non-canonical complexes, 278 entries; they are never used for training, validation, conversion starts, RF3 templates or data generation; (2) train and validation receptors come from canonical natural peptide-protein complexes found by a wider RCSB search (X-ray and cryo-EM, resolution 3.5 A or better, one peptide of 8 to 30 residues, canonical residues only; 7,314 entries found, 3,839 kept after the filters) and from the 159 natural families of section 23; (3) a group is a receptor family (RCSB 30 % sequence-identity cluster of any protein entity of the entry) or a similar peptide (shared 7-mer and similarity at least 0.75); any group that contains a test entry is closed to training and validation; (4) the rest is split by a salted hash into train and validation (about 15 % of the groups; groups of 8 or more entries stay in train); (5) any model that generates data (NCFlow, RF3 use) must be trained without a test entry; at test time only the receptor and the peptide spec are given. Files `splits/split_v2.json` (sha256 in `split_v2.sha256.txt`), `freeze_split_v2.py`, `search_wide.py`, and the guard `split_guard.py` (use `SplitGuard("v2")`: refuses test entries, entries of test families, and receptors similar to a test receptor).

| Role | Groups | Entries |
|---|---|---|
| Test: real non-canonical complexes | 71 test families | 278 |
| Canonical complexes related to a test entry (not used) | in the same 71 groups | 1,524 |
| Train (receptors for synthetic data) | 585 | 2,139 |
| Validation (synthetic) | 95 | 176 |

Consequences: (a) the NCFlow models of sections 17 to 22 were fine-tuned on residues of real benchmark complexes (and then on synthetic residues), so they cannot be used to generate training data; NCFlow is retrained on synthetic pool residues only (jobs 483602 and 483603, retrained models `ft_syn` and `rl_syn`, evaluated on all 342 real residue cases as the test and on held-out synthetic residues as validation); (b) the real-complex conversions and repair tests of sections 18, 21, 22 were development experiments on test entries: they stay as evidence about the method, but their outputs cannot enter training; (c) a large share of canonical complexes (1,524 entries) is closed because it shares a family with a test complex, so the usable training receptors are the 2,139 entries of 585 groups.

**Pilot train receptors downloaded (18 Nov 2026):** 300 structure files from RCSB (`splits/train_structures/`, 154 MB, resolution 0.91 to 1.99 A), one entry per train group (best resolution, receptor 40 to 600 residues), each checked with `SplitGuard('v2')` before the download; list in `splits/train_structures_list.json`.

## 27. Relaxed interface energy as a second check (PyRosetta REF2015; 18 Nov 2026)

![Energy test](figures/fig24_energy_test.png)

Method: the complex (peptide chain vs protein chains) is relaxed with PyRosetta FastRelax with restraints to the start coordinates and the interface energy dG_separated is computed (REU, more negative is more favourable); scripts `natural_complexes/energy_scan.py`, `energy_converted.py`, `prep_pdb.py`, `prep_converted.py`, `prep_repair.py`; PyRosetta runs on the laptop and now also on Tillicum (`venv_pyrosetta`, a copy of the laptop installation; CPU-only jobs are free there). Three tests. (a) Alanine scan on 3 natural complexes (3RM1, 7RKE, 8KDX; 9MXD not finished): 29 mutations of peptide residues to Ala; residues with side chains touching the protein cost a median of 3.0 REU, others 0.1 (rank correlation with the number of contact atoms 0.50). One outlier without contacts (C-terminal Asp of 7RKE, 14.5 REU) is probably a terminal-charge effect. (b) Control: the 32 RF3 predictions before and after the NCFlow repair (residue types unchanged): 27 valid pairs (5 failed or gave no interface), median change +2.3 REU, 17 of 27 within +-5 REU; this is the noise level of the pipeline. (c) Conversions: the 31 real complexes converted by the repair variant of section 22, natural peptide against converted peptide: 25 valid pairs (Rosetta stopped on 3 complexes: a disulfide energy error and "too many tries in fill_missing"; 3 more gave no interface), median change +14.1 REU, 18 of 25 worse than +5 REU, none better than -5; rank correlation with the geometric contact retention -0.26 (weak).

Findings: the energy score sees the interface on canonical residues; the NCFlow repair itself costs about nothing (median +2.3, within noise); converted peptides are typically about 14 REU less favourable than the natural ones even when about 90% of the contacts are kept, so geometric retention does not imply energetic retention. Two caveats: REF2015 is calibrated on canonical residues, and for D residues and other non-canonical residues part of the shift may be a scoring artefact (atom-name mismatches rebuilt by Rosetta; terminal and cap handling) instead of real loss; the control cannot separate these. A fairer test needs conversions of canonical residues to their canonical neighbours and a calibration set with measured affinity changes. Planned: the alanine scan on 29 natural complexes on Tillicum (job 483522, still queued) to calibrate the score.

## 28. Pilot generator, smoke test (15 Nov 2026)

![Pilot smoke test](figures/fig25_pilot_smoke.png)

Setup: `ncf_pilot.py` (gated conversion, one residue per round, rule of section 24) with the released `raw_base` weights, 3 of the 264 prepared train complexes (4XOJ, 4RTZ, 4LN2), at most 4 swaps each, 4 draws per round, up to 3 rounds per candidate residue and 3 candidate residues per swap. Laptop RTX 4060, 27 minutes for 12 swaps (about 5 s per draw). Not a result about quality: it only shows that the generator runs end to end.

Findings: 4 of 12 swaps were accepted (CYS to DCY, ARG to DAR, GLU to DGL, LEU to 2ML); all 4 kept 100% of residue contacts and polar contacts, CA drift 0.1 to 0.5 A (largest single CA 1.3 A, under the 1.5 A limit). Of 300 draws, 292 failed the strict chemistry test, 2 the pose gate, 1 the interaction gate, 5 passed. No draw clashed with the protein. All 6 Pro swaps failed (35 to 36 of 36 draws each, over the primary residue and both alternatives), so proline analogues (DPR, FP9, HYP) are the main hole; the smoke test cannot yet say whether this is the released weights or a ring-handling problem in the chemistry check.

Caveats: the plan swaps the least-touching residues first, so the interaction gate was barely tested here (it will matter on later swaps and on hot spots). Raw weights, not the synthetic-only models. At 5 s per draw the laptop would need about 70 hours for the full pilot (about 2,100 swaps); the H200 draws in about 1 s.


## 29. Docking the 46k pool into the train pockets (16 Nov 2026)

![Pool docking](figures/fig26_pool_docking.png)

Setup: the 46,379 strict-passed pool peptides (Boltz-2 and RF3 sources, lengths 6 to 30) are matched, window by window, on the CA trace of the natural peptide in each of the 264 train pockets (pilot_items.json; no test receptor used). The 3,000 best CA matches are scored, the best 100 are refined by rigid-body minimisation. Gates: no receptor clash (closest contact at least 2.2 A), at least 60% of the natural pocket residues still contacted, then score against the natural peptide. Score is an AutoDock-Vina-style interface proxy computed from elements and distances; it works for any residue but it is a proxy for affinity, not a measurement. Residue-level design then substitutes pool residue conformers one position at a time (superposed on N-CA-C), kept only if the residue passes the strict per-residue test in place, no receptor clash, score improves by at least 0.05 and pocket contact recall stays at least 0.6. Memory-mapped pool, 4 workers (docking 74 min, design 10 min).

Findings: (a) self-test, natural peptides re-docked from perturbed starts: 89% within 1 A when started 1.5 A / 11 deg away, 55% from 2.5 A / 20 deg (263 complexes), so the optimiser and score recover the real pose when the pose is close. (b) 263 of 264 pockets gave a result. Pockets with a gated candidate (no clash, contacts kept): 84% for the best-matching windows against 63% for random windows. Candidates at least as good as the natural binder: 6% rigid (against 3% random), 43% after design (114 of 222 gated pockets). (c) the median best score goes from about -10 (rigid) to -16 (after design), against -17 for the natural peptides. (d) designs use 45 different non-canonical codes (DAR, 23F, DPN, 2ML, 4FB most often). Independent geometry re-check of the final designs with the strict test (the same used for the pool): 222 of 222 peptides and 2,946 of 2,946 checked residues pass (bond, angle, clash, link, handedness, receptor clash all 0 failures); 69 residues of non-alpha type could not be checked.

Caveats: (1) the design step optimises the same proxy score it is judged on, so the 43% is an upper bound on real binding, not evidence of it; an independent check (Rosetta, RF3 re-docking) is still needed. (2) the greedy design replaces most residues (mean 14 substitutions, 99% non-canonical), so the result is a mostly new peptide on the docked backbone, with the natural contact pattern only partly kept (mean recall 0.81, gate 0.6, not the 0.9 rule of section 24). (3) rigid placement of the whole peptide, no backbone relaxation, so a placed peptide is only as good as the chosen window. (4) train pockets only; no test or validation receptor was touched.


## 30. NCFlow trained on synthetic data only (16 Nov 2026)

![NCFlow synthetic only](figures/fig27_ncflow_synthetic_only.png)

Setup: ft_syn (flow-matching fine-tune) and rl_syn (reward fine-tune on top) trained only on the 3,952 synthetic cases (Tillicum job 483602, 1 h 20 min). Evaluated with the strict per-residue test (ncf_strict_run.py, top-pLDDT sample out of 4) on the held-out real residues (real_test, n=89, and real_train, n=251, neither used for training) and on held-out synthetic cases (syn_test, n=180). Raw = released weights. Files in C:\Research\ncflow_ref\out2; tables in Dataset_ncaa\real_benchmark\pilot\ncf2\strict_real_new.tsv and strict_syn_new.tsv.

Findings: strict pass on real_test goes from 27% (raw) to 75% (ft_syn) and 74% (rl_syn); real_train 30% to 76% and 78%; any-of-4 samples 49% to 90% and 91%. Bad bonds drop from 36% to 3% (ft_syn) and 1% (rl_syn), bad angles from 26% to 0%, peptide link off 63% to 16 to 17%, backbone error 0.51 to 0.27 A. Synthetic held-out cases are harder: 48% (ft_syn) and 52% (rl_syn) strict, 80 to 82% any-of-4. Proline analogues: HYP 3/23 to 14/25 (ft) and 11/25 (rl), FP9 3/8 to 7/13 and 6/13, natural PRO 2/2, DPR 0/4 (still fails). Weak families for rl_syn (all sets): gamma 36%, side-chain modified alpha 58%; best alpha,alpha-disubstituted 81%.

Caveats: rl_syn is not clearly better than ft_syn on strict pass (74 vs 75% on real_test, within noise), only on bond and angle strain. Link defects (16%) and the gamma family remain. Raw was not rerun on syn_test in this table. The pilot generator still has to be run to know the pass rate inside the gated, one-residue-at-a-time loop.


## 31. Synthetic-only NCFlow against the models trained with real data (16 Nov 2026)

![Synthetic only vs real](figures/fig28_syn_only_vs_real.png)

Same held-out cases (89 real residues, 180 synthetic residues), same strict test, top-pLDDT sample. ft2 and rl were trained with 252 real residues plus about 3,950 synthetic ones; ft_syn and rl_syn only on the 3,952 synthetic cases.

| Measure | ft2 | ft_syn | rl | rl_syn |
|---|---|---|---|---|
| Strict pass, real (89) | 72% | 75% | 74% | 74% |
| Strict pass, synthetic (180) | 43% | 48% | 51% | 52% |
| Best of 4 samples, real | 91% | 90% | 93% | 91% |
| Best of 4 samples, synthetic | 81% | 80% | 81% | 82% |
| Link off, real | 17% | 17% | 15% | 16% |
| Link off, synthetic | 29% | 28% | 25% | 26% |
| Backbone error, real (A) | 0.26 | 0.27 | 0.30 | 0.28 |

Findings: removing the real residues from training costs nothing measurable: differences are 0 to 5 points, inside the noise of 89 and 180 cases (one real residue is 1.1 points). The synthetic-only models are even slightly higher on the synthetic set (48 against 43%, 52 against 51%), as expected since they saw more of that distribution. So the synthetic data alone are enough to teach valid geometry, and the real residues can all stay in the test set. Caveats: the real training residues were few (252), so the test says synthetic data are as good as a small real set, not as good as a large one; the strict test is the pool's own test, not an independent measure of how close the draws are to the true structure (backbone error 0.26 to 0.30 A is, and it is the same).


## 32. Pilot generator with the synthetic-only model, laptop share (16 Nov 2026)

![Pilot laptop](figures/fig29_pilot_laptop.png)

Setup: `ncf_pilot.py` with rl_syn (trained only on synthetic data), gates of section 24 (strict chemistry, no receptor clash under 2.2 A, 90% of residue and polar contacts kept with a loss of at most 0.05 per swap, pose drift at most 1.0 A, no CA more than 1.5 A), 4 draws per round, up to 3 rounds per candidate, at most 8 swaps per complex, one residue per round. The 264 train complexes are split into 48 shards: shards 40 to 47 (40 complexes, 320 planned swaps) ran on the laptop RTX 4060 (about 4 minutes per complex); shards 0 to 39 were submitted to Klone ckpt (A40 and A100 arrays 43290022 and 43290023), not yet read.

Findings: 284 of 320 swaps accepted (89%), against 4 of 12 in the smoke test with the released weights. Of 1,904 draws: 667 passed all gates, 685 failed the strict chemistry test, 236 clashed with the protein, 258 lost interactions, 58 drifted. Proline swaps work now: 31 of 32 accepted (released weights: 0 of 6). Hardest parents: GLU 19 of 28, ILE 8 of 13, ARG 8 of 12, ASP 19 of 23. Median contacts kept is 1.00 (residue and polar), all accepted swaps at 0.9 or more by the gate. 31 different non-canonical residues are produced (most often SAR 30, DSN 22, DGL 19, DAS 19, DTH 17).

Caveats: each swap passed the pool's own strict test, the same test used to select the pool and train rl_syn, so it is not independent of the model's training signal (an independent check such as Rosetta energy is still to do). The retention values are against the natural complex, which is by design; the converted complex keeps the natural pose, so it is a conservative variant of the real complex, not a new binding mode. Train complexes only.


## 33. Pilot generator, all 264 train complexes (16 Nov 2026)

![Pilot all](figures/fig30_pilot_all.png)

Setup: as section 32, all 48 shards (laptop 40 to 47 and 25 to 29, Klone ckpt A40, A100, RTX 6000 and 2080 Ti for the rest; no cost). One shard (24) was first run on a P100 that this PyTorch build does not support: it wrote empty results (0 draws) and was rerun, the bad file is kept as BAD_p100_s24.pkl on Klone and not used. Merged with `merge_pilot.sh`; per-swap table `pilot_all_swaps.json`, summary `pilot_all_stats.json`. Raw shards: `ncflow_ref\ncflow\out2` (laptop) and `out2\klone`.

Findings: 264 complexes, 2,110 planned swaps (at most 8 per complex), 1,852 accepted (88%) from 12,472 draws (4,490 passed all gates, 4,023 failed the strict chemistry test, 1,915 clashed with the protein, 1,759 lost interactions, 285 drifted). 262 of 264 complexes got at least one swap, 121 got all planned swaps, mean 7.0 accepted per complex. Every accepted swap kept at least 90% of residue and polar contacts (median 1.00), median CA drift 0.19 A, largest single CA 1.48 A. Proline: 198 of 202 accepted. Weakest parents: ILE 48/71, ARG 89/131, TRP 12/18, TYR 34/44, GLU 151/185. 31 non-canonical codes appear (DSN 159, DGL 151, SAR 140, DTH 106, DAS 104).

Caveats: (1) diversity is narrow: by family, D-alpha forms are 1,235 of the 1,852 swaps (968 of them a D-form not in the training vocabulary), N-alkyl 316, side-chain modified 160, alpha,alpha-disubstituted 141, no beta or gamma residue (the plan maps each canonical residue to analogues of the same type; the pool's 46k peptides have beta and gamma residues that this path does not use). (2) 932 of the accepted swaps needed the mirror repair (handedness is a coin flip per draw). (3) These are pickles of atoms per round, not yet written as structure files in the training format; they have not gone through the independent re-check (Rosetta) or the split guard. (4) Natural pose kept, so these are conservative variants of real complexes, not new binding modes; the 46k-pool docking of section 29 is the other route. Train complexes only.


## 34. Synthetic complexes written as CIF files (16 Nov 2026)

![Pilot CIFs](figures/fig31_pilot_cifs.png)

Setup: `splits/write_pilot_cifs.py` writes, for each of the 264 converted train complexes, one mmCIF in the style of the pool files (receptor chains from `splits/pilot_in/<ID>.cif`, peptide chain P replaced by the final converted peptide of the shard, `ATOM` records, polymer entities with the non-canonical codes in the sequence). Every converted residue is re-checked after reading the written file (strict per-residue test with the previous and next residue, receptor clash), and every entry is checked against SplitGuard v2 (train only). Output folder `RFD4_fine_tune/data/pilot_out/` (41 MB): `cif/<ID>_ncf.cif` (264 files), `manifest.json` (one row per complex: sequence, swaps with parent, code, retention, check results), `usable.json`.

Findings: 264 complexes, all in the train split (SplitGuard v2); 1,852 non-canonical residues, 1,850 pass the re-check from the files (99.9%); the two failures are peptide-link distances in one complex (4WVI, adjacent glycine-to-sarcosine swaps, residues 190 to 192, the link of one swap changed when its neighbour was swapped later). 261 complexes are usable (at least one non-canonical residue and every residue passing); 3 are flagged: 4WVI (link failures) and 3S1T and 2ZD7 (no swap accepted, still fully canonical). Peptide length 8 to 30; on average 54% of the residues of a peptide are non-canonical (range 0 to 100%).

Caveats: the check is the pool's own strict test, so it confirms the files were written correctly and that later swaps did not break earlier ones (one did), not that the structures are independent of the model's training signal; an independent check (Rosetta or RF3 re-prediction) is still needed before training. The peptide is in the natural pose, the diversity is narrow (31 codes, section 33), no hydrogens, B-factors and occupancy are placeholders. The test and validation entries were never touched.


## 35. Independent energy check of the synthetic complexes (PyRosetta REF2015; 16 Nov 2026)

![Rosetta pilot](figures/fig32_pilot_rosetta.png)

Method: as section 27 (`natural_complexes/energy_converted.py`, inputs by `prep_pilot_energy.py`): for each of the 261 usable pilot complexes, the natural peptide (`splits/pilot_in`) and the converted peptide (`pilot_out`) are relaxed with PyRosetta FastRelax (coordinate-restrained, REF2015, `-ignore_unrecognized_res`) and the interface energy dG_separated is computed (REU; lower is better; about 1 REU is roughly 1 kcal/mol but it is a score, not a measurement). 12 workers on the laptop, 15 to 30 s per relax. A pair is valid when Rosetta kept every peptide residue in both files and both energies are negative. 258 of 261 complexes finished (3 relaxes hung for 20 to 44 minutes and were stopped).

Findings: 211 valid pairs. 45 complexes were dropped because Rosetta did not recognise a converted residue and deleted it (most often SAR 35, MAA 17, AIB 10, MVA 9, also some D-forms in those peptides), 2 had no interface. Median interface energy: natural -50.1 REU, converted -41.8 REU, median change +6.8 REU (mean +8.6): 84 of 211 pairs within +-5 REU, 122 worse than +5, 5 better than -5. Peptides with only D-form swaps (23 pairs) change by a median +2.8 REU (13 within +-5); mixed conversions (188 pairs) by +7.1. The change grows slightly with the share of the peptide converted (rank correlation 0.34). Section 27's older conversions cost +14 REU (31 complexes); the pilot costs about half.

Caveats: (1) the valid set excludes complexes with residues Rosetta cannot read (N-methyl, alpha-methyl and others), so it over-represents D-forms and the tested residues are the ones Rosetta knows; for those the score is still calibrated on canonical chemistry. (2) Part of the +6.8 REU may be a scoring or relax artefact, not real binding loss; the control of section 27 (repair on unchanged residues, +2.3 REU, noise about +-5) is the only reference. (3) A systematic positive shift means converted peptides are on average somewhat less favourable than the natural ones, as expected when 54% of the residues are replaced while the pose is kept; it does not mean they do not bind. (4) Single relax per structure, no repeats; energies of pairs from different relaxations differ by noise.


## 36. Can the synthetic complexes be used by the RFD4 training recipe? (16 Nov 2026)

![Training check](figures/fig34_training_check.png)

Test: the 264 complexes of section 34 against `rfd4_ncaa_patch/WORKFLOW.md` (Stage 0 geometry check, Stage 1 masking, Stage 3 LoRA fine-tune, 2 steps, batch 2, laptop RTX 4060 8 GB; bundle `Dataset_ncaa/rfd4_bundle_176`).

Findings: (1) Stage 0 and 1 tools read only the first chain, so they must be run on peptide-only files (`splits/make_pep_cifs.py` -> `pilot_out/pep_cif`, masked copies `pilot_out/clean_pep_cif`). Of the 884 residues in the 176-class vocabulary, 878 pass the independent geo_check (4 chirality, 2 link failures); Stage 1 masks 6 residues and keeps 258 files (6 files hold only out-of-vocabulary codes and were not covered). (2) 884 of the 1,852 swaps (18 codes) are in the vocabulary; 968 swaps (13 D-forms: DSN, DGL, DTH, DAS, DLY, DGN, DSG, DIL, DHI, DCY, DTY, MED, DTR) are not, so the model has no class for them. (3) Training smoke tests: peptide-only files run; full complexes first failed RFD4's validator (my writer numbered residues continuously across gaps in the protein chains, giving a 6.57 A "bond"); after splitting protein chains at gaps (`write_pilot_cifs.py`, rewritten files in `pilot_out/cif`) the validator passes but the 8 GB card runs out of memory with whole proteins; with a 12 A pocket crop (`POCKET=12`, files in `pilot_pocket12/cif`, 16 MB) 2 steps run (the card is at its limit; an A40 or H200 has room).

Caveats: 2 steps only show that the files go through, not that the model learns from them. The 13 out-of-vocabulary D-forms either need new classes (vocabulary extension and expanded checkpoint) or have to be masked to context; I have not tested what the pipeline does with such residues in the loss. Stage 0 and 1 do not check the protein or the contacts; the pilot's own gates did. Pocket radius 12 A is my choice, not tested against other radii.


## 37. Training on the masked synthetic complexes, tested on real poses (16 Nov 2026)

![Real pose test](figures/fig35_real_pose_synthetic_head.png)

Setup: Stage 3 of the RFD4 NCAA recipe (`ae_adapt.py`: frozen encoder, only `decoder.logit_linear` trained) on the 254 masked synthetic complexes (`pilot_train_masked/cif`: pocket crop 12 A, protein chains split at gaps, the 13 out-of-vocabulary D-forms and 6 residues failing Stage 0 set to occupancy 0, 7 complexes with more than half of the peptide masked dropped; 932 of 3,703 peptide residues masked), 600 steps, batch 4, lr 1e-3, laptop RTX 4060, final seq_rec about 0.98 on the synthetic batches. Test: the real benchmark of section 4 (189 real peptide chains from PDB, 2,659 scored positions; the model sees coordinates and atoms and must name every residue), same script (`eval_real_adapted.py`, which loads decoder-only weights; the old script's `strict=False` load would have loaded nothing). Real complexes were never used for training (split v2).

Findings: overall 86.7% (fresh 176-class model) to 91.4%. Canonical residues unchanged (98.6%). The 32 trained non-canonical residues: 28.4% (127 of 447) to 56.2% (251 of 447). Codes that go from about 0 to high recall: SAR 2/12 to 12/12, MAA 0/7 to 7/7, MVA 0/8 to 8/8, DAR 4/20 to 19/20, MLE 1/18 to 13/18, DAL 0/39 to 27/39, 2ML 0/12 to 8/12, DPR 0/9 to 7/9, FP9 0/9 to 6/9, 4FB 0/3 to 3/3, 1AC 0/2 to 2/2. Still near zero: the beta and gamma codes (B3A, B3K, B3L, B3Q, BAL, XCP, STA, AC5, DIV), 02K, 04Q, 0XL, 192, 23F (1/9), which have no or very few synthetic examples; they are mostly named as the nearest alpha class (BAL as SAR, XCP as DVA, AC5 as MVA). Weak despite training examples: DPN 9/43 (12 synthetic swaps; 27 of the misses named 23F), DVA 8/20 (9 named MVA), DLE 3/12, DAL 12 misses, 9 of them named MAA: D-form and N-methyl classes are confused. The earlier run-2 decoder (section 4, trained for 90 epochs on other data) reached 99.6% on these 447 positions, so this synthetic-only 600-step head is about half-way.

Caveats: (1) only 18 of the 32 codes have synthetic examples; the model cannot learn the rest from this data, so the result says the synthetic data teach the codes they contain, not that they are enough. (2) Head only, 600 steps, one run, no repeats; more steps or the whole decoder may do better. (3) The test names residues from coordinates; it does not test the flow model or design quality (Stage 4 is not run). (4) Residues outside the 176 classes are not scored. (5) The earlier decoders may have seen different data; the numbers compare model versions, not data sources alone.

## 38. Whole-decoder retrain with 13 new D-forms, tested on real poses (16 Nov 2026)

![Real pose 189](figures/fig36_real_pose_189_decoder.png)

Setup: vocabulary 176 to 189 (13 D-forms: DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR, same atom layouts as the L-forms; bundle `Dataset_ncaa/rfd4_bundle_189`, start weights = `best_decoder_run2.pt`). Whole decoder trained (128M numbers, encoder frozen, `decoder_fine_tune_v3.py`) on the 46,379-peptide pool (80/10/10 split) plus 261 pocket-cropped synthetic complexes (235 train / 26 val, 968 D-form swaps); each step = 1 pool batch of 256 + 1 complex batch of 16. Tillicum job 485816 (H200), cancelled after 22 of 30 epochs (1 h 34 min, about 1.4 USD, peak RAM 25 GB). Saved weights = epoch 0, the best on held-out pool + complex loss; after epoch 0 the held-out complex loss rose (0.07 to 0.09-0.17) while the training complex loss went to about 0.0003 (overfitting to 235 complexes). Weights: `decoder_v3_485816.pt`, log `dec_v3_485816.log`. Test: `Dataset_ncaa/real_benchmark_189` (real PDB peptides, test only; 2,775 scored positions, 112 of them new D-forms), script `pilot_train_189/eval_real_189_v3.py`, outputs `pilot_train_189/eval_v3/`.

Findings (run2 overlay, then whole decoder): overall 95.0% to 97.4%. Canonical 98.3% to 98.6%. 32 trained non-canonical 98.4% to 99.3%. Other in-vocabulary non-canonical 84.6% to 92.3%. The 13 new D-forms 20.5% (23 of 112) to 68.8% (77 of 112). Per D-form after training: DCY 12/12, DTH 11/11, MED 3/3, DSN 11/13, DTR 9/11, DGL 6/7, DLY 15/21, DSG 2/3, DAS 1/3, DGN 1/3, DTY 4/11, DIL 2/9, DHI 0/5. Misses are mostly named as another non-canonical residue (XCP, DLE, DAL, B3K, DPN) rather than as the L-form.

Caveats: (1) DHI (3 of 5 to 0 of 5) and DIL (6 of 9 to 2 of 9) got worse than the baseline; DTY (4 of 11) is weak. (2) Positions per code are small (3 to 21), so single-code numbers are noisy. (3) One run, no repeats; the saved epoch is the first one, so the decoder changed only a little from run2 and further training did not help on this validation set. (4) Test names residues from coordinates; it does not test flow-model or design quality (Stage 4 not run). (5) Only 26 validation complexes and 97 new-class tokens were used to pick the epoch.

## 39. More natural complexes and a larger real test set: split v3 and v4 (16 Nov 2026)

Source: curated PepBDB set of Hamdani, Delgado & Serrano, Protein Science 35(10) e70779 (https://doi.org/10.1002/pro.70779; Zenodo 10.5281/zenodo.19223613; user-provided; CC-BY 4.0), downloaded by the user to `splits/hamdani_dataset`. The PEPBI database (Torres et al., Scientific Data 12:1455, https://www.nature.com/articles/s41597-025-05754-7, user-provided) was read and not used (329 predicted complexes from 22 structures).

Split v3 (`splits/freeze_split_v3.py`, v2 copied unchanged): 4,038 entries of the folder No_Bridge/peptide_protein, 2,194 not in v2; 1,453 qualified (one peptide of 5 to 50 standard parent residues, receptors of 40 to 1000 residues). Linking to v2 groups by receptor 30 % cluster, similar peptide (7-mer, 0.75) and receptor 5-mer overlap with test receptors: 899 train (635 join a train group, 264 new), 77 val, 466 closed (test family), 11 ambiguous (links to both train and val). Train pool 2,139 to 3,038 entries. Downloaded from RCSB into `splits/train_structures_v3` and `val_structures_v3`.

Split v4 (`splits/freeze_split_v4.py`): the folder Non_natural_amino_acids has 1,603 real non-canonical complexes (35 already test). Of the 1,350 classifiable ones, 870 touch no train or val entry and were added as test (test 278 to 1,148); 480 would close train/val data and are left out of every split (user decision). An estimate before the 5-mer check said about 35 % of the train pool would close if all were test (1,070 of 3,038). Caveat: 218 entries could not be classified (no usable metadata, peptide or receptor); the 870 still need benchmark tables (items, residues) before they can score a model.

## 40. Limitations of the NCFlow conversion approach (17 Nov 2026)

What the method is: take a real natural peptide-protein complex, keep the natural pose, replace one peptide residue at a time with a non-canonical residue drawn by NCFlow, keep a draw only if it passes every gate (strict chemistry test, no receptor clash below 2.2 A, at least 0.9 of residue and polar contacts kept, CA drift at most 1.0 A, atom-contact ratio 0.9 to 1.3 against the original natural complex).

1. **Chemical coverage is narrow.** NCFlow builds only four families: side-chain modified, alpha,alpha-disubstituted, N-alkyl and D-alpha. That is 12 non-D-form codes (23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA) plus 6 D-alpha codes and the 13 added D-forms. Beta and gamma amino acids, ornithine, statine and the other codes of the 32-code vocabulary cannot be built this way (known data gap). Each code replaces only its parent residue (SAR only Gly, MVA only Val, MAA/AIB/1AC only Ala, MLE/2ML only Leu, 4FB/FP9/HYP only Pro, 23F/04Q only Phe), so the supply of sites limits the balance per code.
2. **The pose is the natural pose.** The backbone and contacts are held close to the natural complex (CA drift at most 1.0 A, contacts kept), so the synthetic complexes show how a non-canonical residue sits in a pose that a natural peptide chose. They do not show new binding modes that real non-canonical binders may use. The data teach residue recognition and local geometry more than design.
3. **No independent energy evidence.** The gates are geometric. The Rosetta energy check was started and parked (about 1,670 of 2,116 states done); whether a swap keeps or improves binding stability is not shown.
4. **Gates reject a lot, unevenly.** Roughly 25 to 70 percent of attempts pass per code (23F about 31 %, 04Q about 36 %, DIL and DTY about 27 %, AIB about 70 %). Accepted swaps are probably biased to easy positions; the plan includes residues with zero contacts (peptide ends). How many accepted swaps sit at the interface has not been measured.
5. **Handedness is unreliable per draw.** NCFlow draws come out mirrored about half the time (coin flip), which needs a mirror repair step and the strict check; best-of-4 draws helps but costs speed.
6. **Sources are limited and redundant.** Only natural complexes with canonical peptides of 5 to 30 residues, from the frozen train split (3,038 train entries in v3, about 700 protein families). Peptides with internal modified residues were dropped (182), and complexes with insertion-code or odd receptor atoms were refused by the training pipeline (35 of 1,464).
7. **Cost.** Roughly 19 s per swap site on a Quadro RTX 6000 and 90 min per 24-complex shard on the laptop 4060; the CPU-only run on Tillicum was slow (one complex took more than 5 minutes on 16 CPUs in the test). The general batch (21,370 sites) is about 110 RTX 6000 hours. Klone is also capped by the stf group job limit.
8. **The training set is skewed unless balanced.** The first batch and the top-up are 87 % D-forms (4,797 swaps). The balanced general batch (12 non-D codes, about 970 accepted each) is meant to fix that; until it runs, evaluation on other families (N-methyl and the rest) should not be trusted.
9. **What was and was not tested.** Tested: naming residues from coordinates on real PDB peptides (decoder). Not tested: the flow model or design quality (Stage 4 never run on this data), and whether the synthetic complexes predict anything about real non-canonical binding.
10. **Generator provenance.** NCFlow itself was trained on synthetic data only (rl_syn), so its draws are bounded by that synthetic quality; the strict gates, not the model, are what guarantee the chemistry.


## 41. Decoder run 2: retrain on 1,429 synthetic complexes, tested on real poses (17 Nov 2026)

![Decoder run 2](figures/fig37_decoder_run2_real_pose.png)

Setup: whole decoder (128M numbers, 189 classes, encoder frozen) on the 46,379-peptide pool plus 1,429 clean synthetic complexes (261 first batch + 1,203 top-up minus 35 files the RFD4 pipeline refused, mostly receptors with odd atoms or bond-length violations; 8 had insertion-code residues merged by an old writer bug, fixed), 12 % of protein families held out for validation, early stopping (3 epochs), Tillicum H200 MIG slice (job 488524, 18 GB, batch 64/8, about 16 min per epoch). Weights: `decoder_v4_488524.pt` (epoch 0). Trainer `decoder_fine_tune_v4.py` also reports canonical and non-canonical residues separately.

Training: held-out complex loss 0.052 (epoch 0), 0.062, 0.061, 0.082 while the training complex loss fell (gap +0.056 at epoch 3); held-out pool loss fell from 0.0113 to 0.0058. Early stop at epoch 3, best = epoch 0. Held-out synthetic complexes, non-canonical residues: 518 tokens, 99.8 to 100 % accuracy (not a real-structure result).

Real-pose test (`real_benchmark_189`, 2,765 positions, same session for all three weights): overall 94.9 % (untrained new rows) to 97.3 % (run 1) to 97.5 % (run 2); canonical 98.3, 98.5, 98.5; 32 trained non-canonical 98.0, 99.3, 99.3; 13 new D-forms 23/112 (20.5 %), 74/112 (66.1 %), 79/112 (70.5 %). Six weak D-forms pooled: 6/34 (17.6 %), 9/34 (26.5 %), 10/34 (29.4 %): DHI 3/5, 0/5, 1/5; DIL 3/9, 3/9, 1/9; DTY 0/11, 4/11, 7/11; DAS 0/3 in all; DGN 0/3, 1/3, 1/3; DSG 0/3, 1/3, 0/3. Misses are not the L-form (0 of 15 in run 2): they are named as another non-canonical code (XCP 9, DLE 7, DAL 3, ...).

Decision rule (night plan): improved = weak pooled up by at least 10 points and overall not down by more than 1 point. Result: weak pooled +2.9 points over run 1 (+11.8 over untrained), overall +0.2: NOT improved; the 95 % stop line is far away. So the extra top-up complexes were converted into more decoder data (see section 42) and no further decoder run was started.

Caveats: the weak codes have only 3 to 11 real positions each (34 in total); the same weights scored 77/112 and 74/112 new D-forms in two evaluations, so differences of about 3 positions are noise; one training run per setting; the held-out synthetic loss rose after epoch 0, so more synthetic complexes of the same kind may not help recognition of real positions; what separates real from synthetic poses (the misses are other non-canonical codes, not L-forms) is open.

## 42. Extra top-up complexes converted into more decoder training data (17 Nov 2026)

Because section 41 was not an improvement, the top-up shards finished after the first snapshot were converted with the same pipeline (`convert_batch.sh`: write CIFs with a 12 A pocket crop and the v4 split guard, peptide-only CIFs, Stage 0 geometry check, masking, `validate_train_cifs.py`). Source: shards finished on the box (main list 59, new-complex list 9 to 15 and 24 to 27) and the laptop (main list 48 and 49): 291 complexes, 780 swaps; all 780 residues passed the geometry check, none was masked, the training pipeline refused none. Combined set `train_all_cif_clean2` = 1,429 + 291 = 1,720 complexes (not yet trained on). Weak D-form swaps (first batch -> + top-up -> + this batch): DHI 40 -> 285 -> 360, DIL 48 -> 577 -> 757, DTY 34 -> 379 -> 486, DAS 104 -> 951 -> 1,159, DGN 89 -> 602 -> 691, DSG 62 -> 528 -> 649.

Not included: the Klone shards that finished or were still queued when its login lapsed (main list 50 to 57, and the shards moved there), and the balanced general batch (section 43 when merged). Section 41 suggests that more synthetic complexes of this kind may not be what limits real-pose recognition of these D-forms (held-out synthetic loss rose after epoch 0; misses are named as other non-canonical codes), so a further decoder run on this set was not started.

Addendum (17 Nov 2026, later): Klone's last main-list shards (50 to 57) were pulled after its login came back and converted the same way (`topup3_pocket12`): 192 complexes, 520 swaps, all passed the geometry check, 1 complex refused by the training pipeline (4C5I, a bond over 2 A), 191 kept. `train_all_cif_clean2` is now 1,911 complexes. With this, all 64 main-list shards and all 32 new-complex shards of the top-up are finished and converted.


## 43. Denoiser (flow-model LoRA) fine-tuning trials A, B, C vs the untuned control, on all held-out complexes (17 Nov 2026)

![Trials vs control](figures/fig38_trials_heldout_test.png)

Setup: `ncaa_fine_tuning_v2.py` (LoRA rank 16, alpha 32, on to_qkv and to_out_node of the 14 transformer blocks + the CA and latent output heads: 1.08M trainable numbers of 178M; lr 1e-4, batch 1, 5 inner updates; autoencoder frozen), data `train_all_cif_clean` (1,429 clean synthetic complexes, 12 % of protein families held out = 163 complexes). Trial A: all 1,266 training complexes, 900 steps (early stop), best = step 500. Trial B: first-batch complexes only (228), 1,000 steps, best = step 800. Trial C: all 1,266 training complexes, stopped by hand at step 1,143 (plateau), best = step 600. Control: the 189-class RFD4 flow model with no LoRA.

Test: every model scored on all 163 held-out complexes one at a time, each complex with its own fixed seed for the data pipeline and the noise, so every model sees identical noise on each complex (paired comparison; `eval_one.py`, results in `denoiser_trial/saved/eval`). The 9 complexes (5 %) whose noise draw is at the extreme low-noise end dominate the mean for every model alike and are left out of the means (the loss weight is 1/((1-t)^2 + 1e-5)). Mean flow loss over the other 154: control 49.34, A 48.93, B 48.99, C 48.87. Paired change vs control: A -0.41 (95 % interval -0.50 to -0.33; lower on 79 % of complexes), B -0.35 (-0.45 to -0.26; 66 %), C -0.47 (-0.55 to -0.38; 81 %), i.e. about 0.7 to 0.9 %. Between models: C minus A -0.05 (-0.12 to +0.01), A minus B -0.06 (-0.14 to +0.04), C minus B -0.11 (-0.18 to -0.05): only C vs B is separable. By term (change vs control): backbone CA A -3.9 %, B -2.6 %, C -6.3 %; latent code -0.7 %, -0.7 %, -0.8 %; residue type -1.2 %, -0.3 %, -0.4 %. By source: on the 33 held-out first-batch complexes B improves most (-0.83 vs A -0.59, C -0.72); on the 121 top-up complexes B improves least (-0.25 vs A -0.38, C -0.41).

Corrections to earlier readings in this session: (1) the held-out loss printed during training was not repeatable (the same untuned model gave 53.47 and 53.86 in two launches; the data pipeline's random rotation / design-task sampling was not seeded), so reading-to-reading swings of about 0.5 were partly measurement noise; evaluation is now seeded (two runs gave identical 55.2396). (2) The earlier statement that trial B's best weights were worse than the untuned model on trial A's held-out set (56.84 vs 55.81, 14 complexes) came from that noisy measurement and is not supported by this test: B is 0.35 better than the control on all 163 complexes. (3) The step-by-step "overfitting" reading of trial B was premature.

Caveats: held-out flow loss on synthetic complexes, not generated structures or real non-canonical complexes; one run per model; 154 to 163 complexes; the gain is small (under 1 %) and the settings are gentle (LoRA rank 16 on attention only, at most about 1,100 steps at batch 1, i.e. about one pass over the data, against 5,000 steps at batch 3 in the original recipe); the models were chosen by their own noisy in-training readings, which makes their best checkpoints slightly optimistic.

### 43a. Addendum: trial D (LoRA rank 64 instead of 16), 18 Nov 2026

Setup: the same data (all 1,266 training complexes), the same attention layers (`to_qkv`, `to_out_node` in 14 blocks) and the same held-out check as trial C; only the LoRA rank changes, 16 to 64 (alpha 32 to 128; trainable numbers about 4 times larger). Batch 1, eval every 150 steps on 30 held-out complexes (the held-out loss is now repeatable; untuned baseline 53.93). The early stop (8 readings without a better held-out loss) ended it at step 2,250; best step 1,050 (53.62, change −0.31 against the untuned model on those 30 complexes). Over the run the held-out readings stayed between −0.31 and +0.33 of the baseline.

Test on all 163 held-out complexes (same per-complex seeds as A, B, C, 154 after the trimming rule): paired change against the untuned control −0.138 (95% interval −0.240 to −0.030; lower than control on 57% of complexes). Trial C (rank 16) was −0.465. Paired D minus C: +0.327 (interval +0.249 to +0.405; D lower than C on 24% of complexes). Figure: `figures/fig38_trials_heldout_test.png` (D added).

Reading: a four times larger LoRA on the same layers did not lower the held-out flow loss further; it did worse than rank 16 (the larger adapter overfits earlier, step 1,050 against 600 for C, and the held-out loss then rises). One run each, so the size of the difference is only indicative. The attention layers alone are therefore not limited by LoRA rank; trial E tests the other frozen parts (feed-forward layers and attention gate, about 99M of the 157M transformer numbers). Weights: `denoiser_trial/saved/box_out/D_rank64/{best,ckpt,last}` (continue with `ncaa_fine_tuning_v2.py ... --resume <dir>/ckpt`).

### 43b. Addendum: trial E (rank 64 on attention + feed-forward layers + attention gate), 18 Nov 2026

Setup: as D (rank 64, alpha 128, batch 1, eval every 150 steps on 30 held-out complexes, patience 8) but 70 adapted layers per model: attention `to_qkv`, `to_out_node`, gate `to_g`, feed-forward `transition.transition.swish_linear.0` and `linear_out` in all 14 blocks (the first launch failed because `swish_linear` is a container; the Linear inside is `swish_linear.0`). The early stop ended it at step 1,350. Held-out readings (untuned baseline 53.93): step 150 53.91 (best), 300 54.59, 450 54.77, 600 55.39, 750 55.58, 900 54.81, 1,050 55.12, 1,200 55.06, 1,350 54.95: above the baseline at every reading after step 150.

Test on all 163 held-out complexes (best weights = step 150): paired change against the untuned control +0.222 (95% interval +0.109 to +0.328; lower than control on 41% of complexes); against C +0.687 (+0.596 to +0.775); against D +0.360 (+0.274 to +0.449). Adapting the feed-forward layers at this learning rate makes the held-out flow loss worse than not tuning at all. Weights: `denoiser_trial/saved/box_out/E_rank64_ff/{best,ckpt,last}`.

## 44. Real structures: untuned control vs trials A to E (18 Nov 2026)

Test set (TEST ONLY, never trained on): the 104 real PDB peptide-protein complexes of the real benchmark (`real_benchmark/complex_test/truth_complex`: real peptide chain P with a non-canonical residue + the real receptor chains in contact), cropped to the peptide plus every receptor residue with an atom within 12 A (`crop_real_test.py`, same layout as the training CIFs). 103 pass the trainer's pipeline check (`validate_train_cifs.py`; 22ES refused: a 5.05 A bond between DLY and an unknown residue A1L9X). Each model (best weights of each trial, no further training) scores the flow loss on each complex with a fixed per-complex seed (`eval_one_real.py`), so every model sees the same noise on each complex. Split: 26 complexes whose peptide contains one of the 13 new D-forms (the residues the original decoder was never trained on) and 77 without. Figure: `figures/fig39_real_test_all_trials.png` (trimmed set: the 5% of complexes with the most extreme noise draw left out for every model, 97 complexes).

Findings (paired change against the untuned control; negative = better):
- All 103 complexes, untrimmed: A -0.14 (95% interval -0.48 to +0.12), B -0.08 (-0.50 to +0.24), C -0.13 (-0.51 to +0.17), D +0.32 (-0.01 to +0.58), E +0.55 (+0.25 to +0.79). Trimmed (97): A +0.01, B +0.11, C +0.04, D +0.47 (+0.27 to +0.66), E +0.68 (+0.44 to +0.86). A, B and C are not distinguishable from the control on real complexes as a whole; D and E are worse.
- Complexes whose peptide has a new D-form (n=26, untrimmed): A -0.94 (-2.06 to -0.15), B -1.05 (-2.46 to -0.11), C -1.04 (-2.18 to -0.17); D -0.46 (-1.63 to +0.42), E -0.04 (-1.02 to +0.66). With the trimming rule (n=24) the same direction but smaller: A -0.38, B -0.33, C -0.43.
- Complexes without a new D-form (n=77): A +0.13, B +0.25, C +0.18 (slightly worse than the control), D +0.58 (+0.40 to +0.72), E +0.74 (+0.48 to +0.92).

Reading: the rank-16 attention-only fine-tunes (A, B, C) help on the real complexes that contain the new D-forms the synthetic data was built around, and are neutral to slightly worse elsewhere; the larger adapters (D, E) are worse on real structures, matching the synthetic held-out result. The D-form subset is small and was split after the scoring, and its size depends on the trimming, so it is a lead, not a result. Decoder note: the flow-loss score does not use the decoder (the frozen encoder gives the targets), so a control with a fully retrained decoder scores the same; the residue-type term (flow model's own readout) is reported per group in `denoiser_trial/saved/eval_real/summary_real.json`.
Caveats: flow loss, not generated structures; one run per model; real complexes are cropped to a 12 A pocket like the training files; no real complex or receptor family of the test set was used for training (train split of the frozen split v4).

## 45. Trial F: the 6,457-complex pool (first batch, top-ups, balanced general batch) against trials A to E (18 Nov 2026)

Setup: the settings of trial C (LoRA rank 16 on attention `to_qkv` and `to_out_node`, 14 blocks, 1.08M trainable numbers) trained on `train_all_cif_clean2` (6,457 complexes from 2,458 PDB entries, 735 protein families; 5,734 for training, 723 held out by family; the 163 held-out complexes of A to E stay held out because the family split is a hash of the family key). Two runs: **F (box)**, RTX 5060 Ti, batch 1, lr 1e-4, stopped by hand at step 981 (best step 750, held-out 51.77 against 52.33 untuned on its own 30-complex check); **F (H200)**, Tillicum job 493150, full H200 (peak 115 of 144 GB), batch 4, lr 2e-4, early stop at step 1,600 (best step 800, 51.72 against 52.33; the held-out loss bounced between 51.72 and 52.25 from step 100 on; 73 of 713 early batches were skipped for loss above 1e4). Figures: `figures/fig38_trials_heldout_test.png` and `fig39_real_test_all_trials.png` (all models). Data diversity of the pool: `fig40_training_data_diversity.png`.

Held-out (163 complexes, 154 after trimming; paired change against the untuned control, negative = better): F (box) -0.368 (95% interval -0.454 to -0.281), F (H200) -0.303 (-0.395 to -0.190); trial C -0.465, A -0.412, B -0.353, D -0.138, E +0.222. F minus C: box +0.098 (+0.045 to +0.152), H200 +0.162 (+0.091 to +0.239). So the 4.5 times larger pool did not beat trial C; the difference is small and C is still best.

Real structures (103 real complexes, untrimmed, against the control): F (box) -0.06 (-0.45 to +0.24), F (H200) +0.25 (-0.08 to +0.50). Complexes with a new D-form (n=26): F (box) -0.83 (-2.06 to -0.03), F (H200) -0.47 (-1.35 to +0.24); A -0.94, C -1.04. Complexes without a new D-form: F (box) +0.19, F (H200) +0.50 (+0.20 to +0.71), against A +0.13 and C +0.18.

Reading: more general (non-D-form) synthetic data gave no gain over the smaller D-form-heavy set on either test. The D-form gain on real complexes shrank (-0.8 and -0.5 against -0.9 to -1.0), which fits the share of D-form positions falling from 87% to 29% of the pool; the H200 run (higher learning rate, batch 4) is worse than the box run. Caveats: one run each; the box run was stopped at step 981; learning rates and batch sizes differ between the two F runs; the continuation F2 (job 493216, lr 5e-5 from the H200 best weights) is queued. Weights: `denoiser_trial/saved/box_out/F_pool2/{best,ckpt,last}` and `denoiser_trial/saved/H200_F/dn_F_H200_493150/{best,ckpt,last}`.

## 46. Comprehensive benchmark: untuned control and eight fine-tuned versions (18 Nov 2026)

![Benchmark](figures/fig41_benchmark_all_models.png)
![By residue group](figures/fig42_noncanonical_by_group.png)

Models (all best weights, none trained further for the test): **A** rank-16 attention LoRA on 1,266 complexes; **B** same on the 228 first-batch complexes; **C** same as A, longer; **D** rank 64; **E** rank 64 plus feed-forward layers and gate; **F box** and **F H200** the C recipe on the 6,457-complex pool (box: batch 1, lr 1e-4, stopped at step 981; H200: batch 4, lr 2e-4, early stop at 1,600, job 493150); **F2** continuation of F H200 from its best weights at lr 5e-5 (job 493216, early stop at step 800, best step 300, 26 minutes). Tillicum cost of the three H200 jobs: about 1.3 USD. Weights: `denoiser_trial/saved/box_out/{A_all,B_first,C_long,D_rank64,E_rank64_ff,F_pool2,F2_H200}/best` and `denoiser_trial/saved/H200_F/dn_F_H200_493150/best`, `denoiser_trial/saved/H200_F2/dn_F2_H200_493216/best` (continue with `ncaa_fine_tuning_v2.py ... --resume <dir>/ckpt`).

Tests (never trained on, same noise on each complex for every model, paired; `eval_one.py`, `eval_one_real.py`, `eval_one_nc.py`): 163 held-out synthetic complexes (families never trained on); 103 real PDB peptide-protein complexes (26 with one of the 13 new D-forms). Non-canonical-only score: only the positions whose residue is one of the 142 non-canonical types of the 189-class vocabulary are scored; residues outside the vocabulary are masked (they come through as the unknown token); the latent loss is shown with the noise weight 1/((1-t)^2+1e-5) divided out (stable). Scripts and tables: `make_fig41_benchmark.py`, `denoiser_trial/saved/eval_nc/{benchmark_summary.json,per_code_latent_change.csv}`.

Change against the untuned control (negative = better; flow loss in loss units, non-canonical numbers in % of the control's latent loss):

| Model | Held-out, all positions (163) | Real, all positions (103) | Real, new D-form (26) | Real, no new D-form (77) | Non-canonical only, real | Non-canonical only, held-out | Held-out D-forms | Held-out general codes |
|---|---|---|---|---|---|---|---|---|
| A | -0.41 | -0.14 | -0.94 | +0.13 | -6.1% | -7.0% | -13.1% | -11.1% |
| B | -0.35 | -0.08 | -1.05 | +0.25 | -3.7% | -5.9% | -9.7% | -17.3% |
| C | -0.47 | -0.13 | -1.04 | +0.18 | -6.6% | -7.4% | -13.6% | -13.4% |
| D | -0.14 | +0.32 | -0.46 | +0.58 | -5.5% | -7.8% | -14.9% | -11.0% |
| E | +0.22 | +0.55 | -0.04 | +0.74 | -1.5% | -7.0% | -12.8% | -11.4% |
| F box | -0.37 | -0.06 | -0.83 | +0.19 | -3.3% | -5.5% | -9.1% | -15.9% |
| F H200 | -0.30 | +0.25 | -0.47 | +0.50 | -4.1% | -6.3% | -9.9% | -19.9% |
| F2 | **-0.54** | +0.10 | -0.61 | +0.33 | -0.3% | -7.0% | -10.6% | **-21.8%** |

(Real untrimmed; held-out trimmed to 154 complexes. Intervals are in `benchmark_summary.json`; the real non-canonical sets are small: 257 positions in 102 complexes, 47 D-form positions in 26 complexes, 83 general-code positions in 53 complexes. Held-out: 584 positions in 155 complexes, 473 of them D-forms, 71 general-code positions in 31 complexes.)

Findings:
1. **All-position scores hide most of the effect.** At non-canonical positions every version lowers the latent loss by 4 to 8% on both test sets (held-out D-forms by 9 to 15%), against 1% for the all-position score. The training changes mainly the non-canonical positions.
2. **Best by test.** Held-out synthetic, all positions: F2 (-0.54), then C, A, F box. Real complexes, all positions: A, C, B and F box are about 0 (not distinguishable from the control); D, E and F H200 are worse. Real complexes with a new D-form: B, C and A are best (about -1.0). Non-canonical only, real: C (-6.6%), A (-6.1%), D (-5.5%); the weakest are E and F2 (about -1 and -0.3%). Held-out non-canonical: D, C, A, E about -7 to -8%.
3. **The larger pool (F) helps synthetic general codes, not real ones.** Held-out general codes: F2 -21.8%, F H200 -19.9%, F box -15.9%, against -11 to -13% for A, C, D. On real general codes: A -7.8%, C -6.4%, D -6.2%, but F box -0.2%, F H200 -2.8%, F2 +2.4% (83 positions, wide intervals). More synthetic data of the same generator improves the synthetic distribution and does not transfer to real complexes.
4. **Cost at canonical positions.** On the real complexes every fine-tuned version raises the latent loss at canonical positions slightly (+0.5% for A to +1.3% for E; D +1.0, F H200 +1.1, F2 +0.8), while it lowers it at non-canonical positions: the tuning trades a little canonical accuracy for non-canonical accuracy on real structures. On the held-out synthetic complexes canonical positions improve by about 0.5% (F2 -0.74%) except E (+0.14%).
5. **Backbone (CA) at non-canonical positions:** no consistent gain on real complexes (A to F box +7 to +17%, intervals mostly include 0; F H200 -4%, F2 -10%, not significant); on held-out synthetic F H200 (-14%) and F2 (-13%) are significantly better. Few positions per complex make this noisy.
6. **Residue-type readout:** top-1 accuracy at non-canonical positions is 3 to 5% for every model including the control (real) and 3 to 9% (held-out). The readout head is not trained in any of the trials (LoRA covers attention and the two output heads of the latents and the CA), and its cross-entropy at non-canonical positions rises a little (control 7.6, fine-tuned 8.0 to 8.8 on real).

Reading: the fine-tunes do learn the non-canonical residues (latent loss at their positions falls 4 to 15%), the small rank-16 attention-only fine-tunes on D-form-heavy data (A, B, C) transfer best to real structures, and the larger general-code pool and the best synthetic model (F2) do not carry their synthetic gains to real complexes. Caveats: flow loss, not generated structures; one run per model; real non-canonical sets are small (confidence intervals in the table notes); only the held-out and real sets of this project were used, never the train split; the sequence readout was never trained, so residue identity at generation time still depends on the frozen decoder and readout.

## 47. Extended real benchmark: 884 more real complexes, residue groups and codes (18 Nov 2026)

![Extended real benchmark](figures/fig43_real_benchmark_extended.png)

New test set (TEST ONLY): the 942 peptide complexes of the Hamdani non-natural-amino-acid folder (`splits/hamdani_dataset/.../Non_natural_amino_acids`) whose entry is in the frozen test split v4 and not in the first 103-complex benchmark, cropped to the peptide plus every receptor residue within 12 A (`crop_real_ham.py`; peptide chain renamed P, label_seq_id assigned); 884 pass the trainer's pipeline check (46 refused for atoms that do not fit the atom37 format, 11 for batch invariants, 1 for residue order). 88 of them have at least one of the 25 target codes in the peptide (MLE 26, MVA 24, SAR 17, DLY 16, MAA 12, DTY 11, HYP 6, AIB 5, DTR 5, DCY 5, 1AC 4, 4FB 4, DGL 3, FP9 2, DHI 2, DSG 2, DSN 1, DAS 1, DIL 1, MED 1; none with 23F or 04Q). The untuned control and all eight fine-tuned versions were scored with `eval_one_nc.py` (non-canonical positions only, residues outside the 189-class vocabulary masked, same noise per complex). The first 103 real complexes (section 44) and the 884 are combined as "987 real complexes" (467 complexes have at least one scored non-canonical position, 1,065 positions).

Change in latent loss against the untuned control, % of the control (negative = better; 95% bootstrap interval over complexes in brackets):

| Group (987 real complexes) | Positions | A | B | C | D | E | F box | F H200 | F2 |
|---|---|---|---|---|---|---|---|---|---|
| 13 D-forms (56 complexes) | 140 | -7.6 | -13.3 | -8.9 | -8.3 | -5.2 | -8.9 | -8.2 | -8.2 |
| 12 general codes (115 complexes) | 254 | -6.9 | -2.9 | -6.7 | -5.7 | -1.7 | -1.1 | -5.5 | -3.3 |
| other vocabulary codes (400 complexes) | 671 | -1.0 | -1.0 | -1.5 | 0.0 | +1.4 | +1.4 | -1.9 | +1.5 |
| all non-canonical (467 complexes) | 1,065 | -2.3 | -2.2 | -2.8 | -1.4 | +0.5 | +0.6 | -2.9 | +0.5 |
| canonical positions (979 complexes) | 100,570 | +0.2 | +0.3 | +0.3 | +0.7 | +0.9 | +0.3 | +0.2 | +0.1 |

The 884-complex set alone (not in the first benchmark): 12 general codes (62 complexes, 171 positions): A -6.2, B -6.2, C -7.1, D -5.3, E -2.3, F box -1.9, F H200 -7.7, F2 -7.9 (all intervals but E and F box exclude 0); 13 D-forms (30 complexes, 93 positions): -6 to -16 for every version (B -16.1); other vocabulary codes about 0 (-1.1 to +2.7). Per code on the 987 complexes (`denoiser_trial/saved/eval_nc/per_code_real987.csv`, codes with at least 8 positions): MLE (93 positions) -1.4 to -9.3 (F H200 -9.3, F2 -8.4, A -7.3); DLY (59) -5 to -17 (B -16.9); MAA (18) -4 to -9 (A, C, D, F H200 about -9); AIB (39) -3 to +5; MVA (37) -3.5 to +5.5; SAR (25) A -5.5, C -3.9, D -5.4 but F H200 +7.2 and F2 +13.6; HYP (23) -5.4 to +3.2; DTY (20) -5.5 to -11; DTR (18) -1 to -5; DCY (12) -2 to -6; FP9 (8 positions in 2 complexes) about -22 to -37.

Findings: (1) with 2.4 times more real general-code positions the earlier gap of the larger pool is not confirmed: on the new 884 complexes F H200 (-7.7%) and F2 (-7.9%) are as good as A and C (-6.2 and -7.1%), while on the first 103 complexes they were weak (-2.8, +2.4, intervals very wide); in the combined set A (-6.9) and C (-6.7) are best, F H200 -5.5, F2 -3.3, F box -1.1 and E -1.7 are weaker; the difference between A/C and F2 is inside the intervals. (2) D-forms improve by 5 to 13% for every version (B the most, -13.3%, from only 228 complexes). (3) The other 142-code vocabulary members that are not targeted (400 complexes, 671 positions) are unchanged, so the gain is specific to the trained codes. (4) Canonical positions lose 0.1 to 0.9% for every version (smallest for F2 and F H200). (5) Weak spots: SAR (F H200, F2 worse than the control), AIB, MVA and HYP (little or no gain for any version); 23F and 04Q cannot be tested on real structures, and several D-forms have 1 to 5 real complexes. Caveats: flow loss, not generated structures; one run per model; positions per code are small; Hamdani complexes may repeat receptor families among themselves.

## 48. Same decoder for every model: denoise-then-decode benchmark (18 Nov 2026)

![Decode benchmark](figures/fig44_decode_same_decoder.png)

Condition: the untuned control (intact original RFD4 flow network) and the eight fine-tuned denoisers (A, B, C, D, E, F box, F H200, F2) all use the SAME retrained 189-class decoder (`decoder_v4_488524.pt`, strict load); only the denoiser weights differ. The residue-type readout head of the denoiser is not used. Procedure (`eval_one_dec.py`): per complex a fixed seed gives every model the same noise; the latent time is fixed at t = 0.05, 0.15 and 0.3 (CA keeps its seeded draw; the extra real set at t = 0.1); the denoiser predicts the clean latents and CA positions of the generated tokens (conditioned tokens keep their true values); the decoder names each residue from them; top-1 accuracy at non-canonical positions (residues outside the 189-class vocabulary masked), by group. "Oracle" = the decoder on the TRUE latents (independent of the denoiser). Scripts: `make_fig44_decode.py`; data in `denoiser_trial/saved/eval_dec/`.

Change in accuracy against the control, percentage points (positive = better); control accuracy and oracle in brackets:

| Test | Positions | Control (oracle) | A | B | C | D | E | F box | F H200 | F2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Held-out, t=0.05, all non-canonical | 584 | 75.7% (99.7%) | +3.8 | +6.3 | +4.3 | **+11.1** | +8.7 | +6.3 | +0.2 | +6.0 |
| Held-out, t=0.05, 13 D-forms | 473 | 70.2% (99.6%) | +4.4 | +7.8 | +5.1 | **+13.5** | +10.6 | +7.6 | +0.2 | +7.2 |
| Held-out, t=0.15, all | 584 | 80.7% | +3.4 | +4.6 | +4.6 | **+9.4** | +8.6 | +3.1 | 0.0 | +5.0 |
| Held-out, t=0.3, all | 584 | 89.9% | +3.6 | +3.9 | +4.8 | +5.5 | +5.0 | +2.4 | +0.2 | +2.4 |
| Real 103, t=0.05, all | 257 | 90.7% (96.5%) | 0.0 | +0.8 | -0.4 | +2.3 | +2.3 | 0.0 | 0.0 | +1.6 |
| Real 103, t=0.05, 13 D-forms | 47 | 55.3% (85.1%) | -2.1 | 0.0 | 0.0 | +8.5 | +8.5 | -2.1 | -2.1 | +4.3 |
| Extra real (365 complexes), t=0.1, all | 808 | 93.3% (95.8%) | +0.7 | +0.9 | +0.4 | +1.0 | -1.0 | +0.1 | +0.5 | +0.5 |
| Extra real, t=0.1, 13 D-forms | 93 | 50.5% (72.0%) | +5.4 | +5.4 | +1.1 | +6.5 | -8.6 | +1.1 | +5.4 | +6.5 |

Findings and caveats:
1. **What it measures, and a limit I found.** In this design pipeline the identity of every non-canonical residue is given to the model as input (the sequence-conditioning feature holds the true class for those tokens, and the decoder limits each token to the classes of its token type), whereas canonical residues are not given (their accuracy from noisy latents is 7 to 12%, near chance, against 96 to 99% from the true latents). So the non-canonical accuracy is high for every model (general codes 99 to 100%, no headroom) and the informative part is separating a D-form from its L-form, which share the same atoms: control 50 to 70%, true-latent decode 72 to 100%.
2. On the held-out synthetic complexes every fine-tuned denoiser decodes D-forms better than the control by 4 to 14 points at t = 0.05 (D best +13.5, E +10.6, B +7.8, F box +7.6, F2 +7.2, C +5.1, A +4.4; F H200 +0.2), and the gain shrinks as the latent gets cleaner (t = 0.3: +2 to +7).
3. On real complexes the gains are small and mostly inside the noise (47 and 93 D-form positions; one position is 2.1 and 1.1 points): the D-form change at t = 0.05 on the first 103 real complexes is D and E +8.5, F2 +4.3, A and F -2.1; on the 884 extra real complexes A, B, F H200 +5.4, D and F2 +6.5, but E -8.6.
4. Canonical positions lose 0.1 to 1.9 points for every version (E most), the same small cost seen in the loss benchmarks.
5. Not tested: a variant where the identity of non-canonical residues is withheld from the denoiser, all-atom reconstruction error of the decoded structure, and generated structures from pure noise. The residue-type readout head is not used here and was never trained.

## 49. Decoder run 3: mirror-image D-peptides fix the real D-form recognition (18 Nov 2026)

![Decoder run 3](figures/fig45_decoder_run3_mirror.png)

Hypothesis: the weak real D-forms (DHI, DIL, DTY, DAS, DGN, DSG; baseline run 1 10/34, run 2 7 to 10/34) fail because the synthetic complexes put a D side chain on an L backbone, whereas real D-residues have a D backbone. The mirror image of an L-peptide is a valid D-peptide, so 7,534 pool peptides of the TRAIN split of the decoder's 80/10/10 pool split (those made only of canonical or D-alpha residues, 19 L/D pairs, GLY unchanged) were mirrored (`mirror_pool.py`: x negated, residue codes swapped in the atom rows, the entity sequence, chem_comp and struct_conn; the pool files' own CIF layout kept). They were added to the training side AFTER the split (`decoder_fine_tune_v5.py`, env MIRROR_LIST), so validation and test are exactly those of run 2.

Run 3 (Tillicum job 493747, H200 MIG slice, 1 h 14 min, about 0.16 USD): the settings of run 2 (bundle start, 1,429 synthetic complexes, batch 64/8, 4 epochs, patience 3) + the mirrored peptides. Synthetic validation is unchanged (epoch 0 to 3: pool accuracy 99.89 to 99.94%, complex accuracy 98.86 to 98.88%, new-class tokens 100%); the best held-out loss was at epoch 1 (saved: `decoder_v5_493747.pt`; epochs 2 and 3 had a higher held-out complex loss). Real-pose test (`eval_real_189_v3.py`, the real benchmark peptides with 189-class labels, 2,750 positions; all four variants re-scored in one job 493899, outputs `pilot_train_189/eval_v5/`):

| Variant | All | Canonical | 13 new D-forms | Weak six pooled | Other vocabulary non-canonical | 32 trained non-canonical |
|---|---|---|---|---|---|---|
| Untrained rows (bundle) | 95.0% | 97.9% | 20/112 (17.9%) | 7/34 | 87.5% | 98.4% |
| Run 1 | 97.0% | 98.3% | 77/112 (68.8%) | 10/34 | 95.8% | 99.8% |
| Run 2 | 97.3% | 98.5% | 72/112 (64.3%) | 7/34 | 100% | 99.8% |
| **Run 3 (mirror)** | **98.7%** | **98.6%** | **110/112 (98.2%)** | **34/34** | 93.8% (45/48) | 99.8% |

Per code, run 2 to run 3 (real positions): DHI 1/5 to 5/5, DIL 1/9 to 9/9, DTY 5/11 to 11/11, DAS 0/3 to 3/3, DGN 0/3 to 3/3, DSG 0/3 to 3/3, DTR 8/11 to 11/11, DCY 10/12 to 12/12, DLY 15/21 to 19/21, DGL 6/7 to 7/7, MED 2/3 to 3/3, DSN and DTH stay at 100%.

Decision rule (pooled weak recall up at least 10 points and overall accuracy not down by more than 1 point): IMPROVED (weak six 20.6% to 100%, overall +1.4 points). Findings and caveats: (1) real D-form recognition is solved at this test's size (110 of 112, the two misses are DLY); the earlier failure was the backbone mismatch, not the amount of data. (2) Other-vocabulary non-canonical positions fell from 48/48 to 45/48 (3 misses); this group has 48 positions. (3) One run each; positions per code are 1 to 21. (4) The real benchmark is test only; the pool is synthetic (generated peptides, not real structures); the mirrored files come from the train split only, so no validation or test peptide or its mirror image was trained on; the separate check that no real benchmark sequence is among the pool peptides was done for runs 1 and 2 and was not repeated for the mirrored set (the mirror of a pool peptide has the opposite handedness, so the same sequences). (5) The new decoder file is not yet used by any denoiser result: the flow-loss scores never run the decoder; the denoise-then-decode test (section 48) used the run 2 decoder and is re-run with run 3 below if time allows.

## 50. Trial G: heavier loss on non-canonical positions, capped noise weight, learning-rate decay (18 Nov 2026)

Setup (`ncaa_fine_tuning_v3.py`, `run_denoiser_trial_G.sh`; box GPU 0): the recipe of trial C (rank-16 LoRA on attention, the 1,266 training complexes, batch 1, lr 1e-4, 28 adapted layers) with three changes aimed at the problems found in the earlier benchmarks: (1) per-token loss weight 10 on the positions whose residue is one of the 142 non-canonical classes of the 189-class vocabulary (they are about 2% of the positions, so the loss was diluted; training only, the held-out check is unchanged); (2) the noise-level weight 1/((1-t)^2+1e-5) of the flow terms capped at 100 (a sample with a larger weight is scaled down; the earlier runs wasted about 10% of their batches on spikes); (3) the learning rate halved twice (to 5e-5, 2.5e-5) after 3 held-out checks without a new best, reloading the best LoRA at each halving. Held-out checks every 150 steps on 30 complexes (untuned baseline 53.93): 53.46 (step 150), 53.33, 53.27 (450), then the first halving, 53.11 (1,350), 53.03 (2,250), 53.01 (2,400), best 52.97 at step 2,700 (change -0.96); the run was stopped by its 3-hour cap while still improving. Weights: `denoiser_trial/saved/box_out/G_ncw10/{best,ckpt,last}` (continue with `--resume <dir>/ckpt`).

Results against the untuned control (negative = better; the same tests as section 46):
- 163 held-out synthetic complexes, all positions: **-0.681** (95% interval -0.776 to -0.590; better than the control on 92% of complexes). Against trial C (-0.465): -0.216 (-0.270 to -0.157); against F2 (-0.541): -0.141 (-0.193 to -0.089). G is the best denoiser on this test.
- 103 real complexes, all positions: -0.022 trimmed (-0.23 to +0.14), -0.224 untrimmed (-0.60 to +0.06); the best of the nine models, but not distinguishable from the control. Real complexes with a new D-form (26): -1.00 (-2.34 to -0.14), the same as A, B, C (-0.94 to -1.05); without a new D-form (77): +0.04 (-0.20 to +0.21), the smallest loss of the fine-tuned models (A +0.13, C +0.18).
- Non-canonical positions only (latent loss, % of the control's): held-out synthetic all -8.9% (-19.6 to -4.0), 13 D-forms -16.7% (-20.5 to -13.2; the best of all models, C -13.6, D -14.9), 12 general codes -15.5%; the 103 real complexes: all -5.3%, 13 D-forms -7.6% (-11.9 to -3.0; the best, C -7.1), 12 general codes -4.8% (-10.2 to +0.3; A -7.8, C -6.35). The 987 real complexes: 13 D-forms -11.4% (best; B -13.3 from only 228 complexes), 12 general codes -5.5% (A -6.9, C -6.7), all non-canonical -1.1%; canonical positions +0.1% (the smallest canonical cost with F2 and F H200).
Reading: the heavier weight on non-canonical positions and the learning-rate decay produce the best denoiser so far on the synthetic held-out set and the best D-form scores on both test sets, at no extra canonical cost; on real general codes it is equal to A and C within the intervals (not better), so the remaining gap for general codes on real structures is not removed by this change. Caveats: one run; stopped at the time cap (still improving); flow loss, not generated structures; the real sets are small.

## 51. Denoise-then-decode with the run 3 decoder (mirror-image D-peptides) for every model, and trial G with the run 2 decoder (18 Nov 2026)

![Decode benchmark, run 3 decoder](figures/fig46_decode_run3_decoder.png)

Same procedure and caveats as section 48 (`eval_one_dec.py`, same noise per complex, latent time fixed at t = 0.05 / 0.15 / 0.3, extra real set at t = 0.1; the identity of non-canonical residues is given to the model as input, so the informative part is the D-form against L-form separation), now with the SAME decoder `decoder_v5_493747.pt` (run 3, section 49) for the untuned control and all nine trained denoisers (A, B, C, D, E, F box, F H200, F2, G). Outputs `denoiser_trial/saved/eval_dec5/`, figure `fig46_decode_run3_decoder.png`; section 48's run 2 decoder results now also include G (`eval_dec/`, `fig44_decode_same_decoder.png`).

Accuracy at non-canonical positions; control accuracy, then the change of each denoiser in points (run 3 decoder):

| Test | Control | A | B | C | D | E | F box | F H200 | F2 | G |
|---|---|---|---|---|---|---|---|---|---|---|
| Held-out t=0.05, all (584) | 97.3% | +1.5 | +0.2 | +1.4 | +1.9 | +1.2 | +0.7 | +0.9 | +1.5 | +1.9 |
| Held-out t=0.05, 13 D-forms (473) | 96.8% | +1.9 | 0.0 | +1.7 | +2.1 | +1.5 | +0.6 | +0.8 | +1.7 | +2.1 |
| Real 103 t=0.05, 13 D-forms (47) | 97.9% | 0.0 | 0.0 | +2.1 | 0.0 | 0.0 | 0.0 | 0.0 | -2.1 | -2.1 |
| Extra real t=0.1, 13 D-forms (93) | 91.4% | +5.4 | +5.4 | +1.1 | +5.4 | +4.3 | +2.2 | +2.2 | +3.2 | +1.1 |

The same control with the run 2 decoder scored 70.2% (held-out D-forms), 55.3% (real 103) and 50.5% (extra real); G with the run 2 decoder: held-out +11.6 (all) and +14.2 (D-forms), real 103 D-forms +2.1, extra real D-forms +2.2.

Findings: (1) the better decoder lifts the untuned control's D-form decoding by 27 to 42 points (held-out 70.2 to 96.8%, real 103 55.3 to 97.9%, extra real 50.5 to 91.4%), more than any denoiser fine-tune did with the run 2 decoder (at most +14); the earlier gains of the denoisers in this test were largely compensating for the decoder's weakness. (2) With the run 3 decoder the denoisers add 0.2 to 2.1 points on the held-out set (D and G best, +2.1 on D-forms) and 0 to +5 on the extra real set (positions per code are small: one position is 1.1 points), and nothing measurable on the first 103 real complexes. (3) Canonical positions (identity not given, 7 to 9% accuracy from noisy latents against 96 to 99% from true latents) lose 0 to 0.7 points for the fine-tuned versions, as in the loss benchmarks. (4) Not tested: withholding the non-canonical identity from the denoiser, all-atom reconstruction, generation from noise.

## 52. Diversity of the current training pool (8,006 complexes) and trial G2 launched (18 Nov 2026)

![Pool diversity](figures/fig47_pool_diversity_8006.png)

Pool `train_all_cif_clean2` after the top-up batches (script `make_fig47_pool_diversity.py`): 8,006 complexes from 2,508 PDB entries and 747 protein families; median peptide length 13 (5th to 95th percentile 6 to 26); median 2 non-canonical residues per complex (1.6% of the residues of the cropped complex). Sources: first batch 261, top-up 3,043, general 4,702. 31 non-canonical codes, 19,991 positions: the 12 general codes 13,432 positions (each 1,000 to 1,400 complexes, the seven low codes now 1,041 to 1,158 positions), the 13 D-forms 6,292 positions (DAS 1,281, DIL 839, DGN 776, DSG 719, DTY 537, DHI 400, DCY 295, DSN 287, MED 268, DLY 261, DGL 256, DTH 246, DTR 127), 267 positions of six other D-forms kept from the natural entry (DAR 89, DPR 52, DAL 45, DVA 35, DLE 34, DPN 12). 2,250 complexes (28%) carry at least one D-form.

Weak points of the diversity: (1) family concentration: the largest family holds 7.8% of the complexes (623), the top 10 families 22%, the top 50 families 44%; 178 families have 10 or more complexes and 142 have only one or two. (2) Same receptor reused: median 3, maximum 8 complexes per PDB entry. (3) Only one non-canonical position per complex for most (median 2), peptide lengths concentrated at 8 to 30. (4) The six D-forms that were not swapped (DAR DPR DAL DVA DLE DPN) are rare and come only from natural entries.

Split check for the continuation (script run on the laptop): hash split by family with `complex_groups.json` (all 2,508 entries covered) gives train 7,117 / held-out 889 complexes; the 163 benchmark held-out complexes are all in the held-out part and no training complex shares a family with them.

Trial G2 (Tillicum job 495125, H200, submitted 18 Nov 2026, `denoiser_G2_h200.slurm`): continues trial G from its best weights (box run, step 2,700) on the 7,117 training complexes with the v3 recipe (10x non-canonical weight, noise weight capped at 100, learning rate halved on plateau, twice), batch 4, learning rate 5e-5 (G had decayed to 2.5e-5 at batch 1), up to 3,000 steps (about 1.7 epochs), 4 h training cap, held-out check every 100 steps on 30 held-out complexes. Cost at most 3.9 USD. Results will be added when the run ends.

The Klone checkpoint array finished (96 of 96 shards, 37 MB), the laptop (12 shards) and the box (8 shards) are finished too; all are pulled to the laptop (`ncflow_ref/ncflow/out2_klone_top`, `out2_box_top`, `out2/toploc_L12_s*`) and not yet converted into the pool.

### 52b. Merge of the Klone, laptop and box top-up shards (18 Nov 2026)

![Pool diversity after the merge](figures/fig47_pool_diversity_8947.png)

Of the 1,946 items in the finished shards (Klone 1,390, laptop 334, box 222), about 1,000 were already in the pool (same peptide sequence as an entry of `top_pocket12a/b/c`); the other 946 (Klone 698, laptop 138, box 110; 7 Klone items share an id with a pool entry but have a different sequence and were kept) were converted with `convert_batch.sh` under the new suffix `tp4` into `top_pocket12d` (all 946 in the train split of split v4 by the guard; 2,311 non-canonical residues checked, 2,310 passed; 5 refused by the trainer pipeline; 941 kept). They were copied into `train_all_cif_clean2`, which now holds 8,947 complexes (was 8,006).

Pool now: 2,530 PDB entries, 748 families, 22,289 non-canonical positions. The seven low general codes are at 1,205 to 1,308 positions (SAR 1,308, 4FB 1,215, 1AC 1,214, HYP 1,213, AIB 1,206, FP9 1,206, MAA 1,205); D-forms 7,553 positions (MED 538, DCY 528, DLY 441, DSN 422, DGL 400, DTH 398, DTR 274 after the top-up); 2,881 complexes (32%) carry a D-form. Largest family 7.0% of the complexes, top 10 families 21%, families with only 1 or 2 complexes 91 (was 142), with 10 or more 210 (was 178). Split check (same hash split by family, `complex_groups.json` covers all entries): train 7,938 / held-out 1,009; the 163 benchmark held-out complexes are all held out and no training complex shares a family with them. Trial G2 (job 495125) trains on the 8,006 snapshot, not on these new complexes.

### 52c. Class balance of the pool: balanced subset and a new generation plan (18 Nov 2026)

Counts in the pool differ a lot between the 25 swapped non-canonical codes (train split, positions): 12 general codes 891 to 1,222 each; 13 D-forms 234 (DTR) to 1,129 (DAS); six unswapped D-forms 4 to 38 (kept from natural entries).

(1) Balanced subset (`make_balanced_pool.py 260`): greedy selection of training complexes, rarest codes first, a complex is taken only if no code would exceed 260 positions. Result: 2,010 training complexes, every one of the 25 codes between 234 and 260 positions (ratio 1.11; DTR 234 is all there is). Folder `train_all_cif_balanced260` = these 2,010 + all 1,009 held-out complexes (3,019 files; same held-out as every run), list `balanced_list_260.txt`, report `balanced_260_report.json`. Caps 220 / 300 / 400 give 1,713 / 2,312 / 3,023 complexes with ratios 1.00 / 1.28 / 1.71.
(2) New complexes instead of cutting data (`splits/build_balance_plan.py`): raise the nine D-forms below 650 positions (DTR 274, DTH 398, DGL 400, DHI 400, DSN 422, DLY 441, DCY 528, MED 538, DTY 537) to about 650 with the same water-filling and family cap (150) as the earlier top-ups: 2,402 complexes, 6,379 swap attempts (acceptance assumed 0.3), items `pilot_items_bal.json`, CIF suffix `bal`, Klone script `splits/pilot_klone_bal.slurm` (96 shards, outputs `out2/bal_n96_s*.pkl`), submit helper `klone_submit_bal.sh`. Not yet submitted: the Klone login has expired.
(3) Six rarest D-forms (DAR DPR DAL DVA DLE DPN, 12 to 89 positions, only kept from natural entries): plan to about 1,000 positions each (`splits/build_rare6_plan.py`, `pilot_items_rare6.json`: 4,929 complexes, 19,114 swap attempts, acceptance assumed 0.3 and not yet measured for these codes). Split by speed and complexity into `pilot_items_rare6_laptop.json` (1,479 complexes, 60 shards, `run_laptop_r6.sh`, outputs `ncflow_ref/ncflow/out2/r6loc_s*.pkl`), `..._box0.json` (2,465 complexes, 96 shards, box RTX 5060 Ti, `run_box_r6.sh`, outputs `~/ncflow_box/ncflow/out2/r6box0_s*.pkl`) and `..._box1.json` (985 complexes, 32 shards, box GTX 1660 Super, `r6box1_s*.pkl`). Started 18 Nov 2026; a finished shard can be converted on its own (suffix `r6`).
Laptop test of the six codes (6 complexes, 24 swap attempts): 18 accepted (75%; DAR 2/3, DAL 4/4, DPN 3/4, DPR 2/3, DLE 5/8, DVA 2/2), about 25 to 30 s per attempt. The plan assumed 0.3, so about 40% of it (about 8,000 attempts) should reach 1,000 positions per code; shards are processed in order and the run is stopped when the pool counts reach the target.
Klone, nine-D-form balance run: job 43352313 (all 96 shards in sequence, 2080 Ti, started 10:16) plus job 43353994 (array task 1 of NJ=2, odd shards, accepted 11:00 when one submit slot opened); 2 shards finished and pulled to `ncflow_ref/ncflow/out2_klone_bal` (not yet converted).

## 53. Trial G2: trial G continued on the 8,947-complex pool (18 Nov 2026)

![G2 benchmark](figures/fig48_benchmark_with_G2.png)
![G2 by residue group](figures/fig49_noncanonical_by_group_G2.png)
![G2 extended real benchmark](figures/fig50_real_benchmark_extended_G2.png)

Setup (Tillicum job 495390, H200, `denoiser_G2_h200.slurm`, `ncaa_fine_tuning_v3.py`): start = trial G best weights (box run, step 2,700), data = `train_all_cif_clean2` as of 18 Nov (8,947 complexes: train 7,938, held-out 1,009, same family hash split; the 163 benchmark held-out complexes are all held out and no training complex shares a family with them), G recipe (10x loss weight on non-canonical positions, noise weight capped at 100, learning rate halved on plateau, twice), batch 4, learning rate 5e-5, held-out check every 100 steps on the first 30 benchmark held-out complexes. Early stop at step 2,000 (1 h 4 min, about 1.0 USD); best = step 1,200. Weights `denoiser_trial/saved/H200_G2/dn_G2_H200_495390/best`.
Problem on the way and fix: the held-out check exploded (2.2e11) on three launches (jobs 495125, 495233, 495281) because on the H200 two of the 30 held-out complexes (`1FYN_tu`, `2XUM_tu`) give a loss of 5.6e3 and 6.6e12 while the other 28 are 39 to 66; the weight cap did not remove it (the squared error itself is huge), so each complex's check loss is now clipped at 150 (`--eval-clip 150 --eval-wcap 100`) and the held-out list is fixed (`--eval-list heldout30_list.txt`). During training the check moved from 57.955 (G's weights) to 57.81 (about 0.4% on the 28 normal complexes; G's own run gained about 1.8%). The cause of the two exploding complexes on the H200 was not found.

Scores (same scripts, same noise per complex, paired change vs the untuned control, negative = better), G2 against G:
- Held-out synthetic, all positions (163, trimmed 154): G2 -0.658 (-0.75 to -0.57), G -0.681 (-0.78 to -0.59): same.
- Real 103, all positions (trimmed 97): G2 -0.086 (-0.30 to +0.10), G -0.023 (-0.22 to +0.14); untrimmed 103: G2 -0.281 (-0.67 to 0.00), G -0.224; with a new D-form (n=26): G2 -1.14, G -1.00; without (n=77): G2 +0.009, G +0.038. G2 is the first version whose real all-position score is not above the control for the complexes without a new D-form.
- Non-canonical positions only, latent loss change (% of control): real 103: all G2 -7.0% (G -5.3%); 13 D-forms G2 -4.6% (G -7.6%); 12 general codes G2 -9.3% (-14.8 to -4.9) (G -4.8%); held-out: all G2 -8.1% (G -8.9%), D-forms G2 -13.4% (G -16.7%), general codes G2 -22.8% (G -15.5%).
- 987 real complexes (103 + 884 extra), non-canonical latent loss: all non-canonical G2 -4.1% (-7 to -2) (G -1.1%, best of all versions); 13 D-forms G2 -9.7% (G -11.4%, B -13.3%); 12 general codes G2 -8.8% (-12 to -6) (G -5.5%, A -6.9%, C -6.7%): the best of all versions and the first to clearly improve on the real general codes; other vocabulary codes G2 -2.9% (G +0.5%); canonical positions G2 -0.0% (G +0.1%), i.e. no loss at the canonical residues.

Findings: (1) the 8,947-complex pool did not improve the synthetic held-out score (G2 equals G), as with trial F; (2) it improved the real general codes (-8.8% on 987 complexes against -5.5% for G) and the overall non-canonical score, and removed the small loss at canonical positions, the first signs that more diverse synthetic data transfers to real structures when combined with G's recipe; (3) D-form scores are a little worse than G's (-9.7% against -11.4%), consistent with the D-forms being a smaller share of the pool (the pool was built for the general codes first); this is what the balanced set and the new D-form generation (section 52c) target. One run per model, intervals overlap for several of these differences. Denoise-then-decode with the run 3 decoder for G2 was still running when this was written.

### 52d. Klone balance shards merged (18 Nov 2026)
The first 8 finished shards of the nine-D-form balance run (Klone jobs 43352313 / 43353994) were converted (`bal_pocket12a`, suffix `bal`: 167 complexes, 299 non-canonical residues, 299 passed the geometry checks, 166 kept after the trainer pipeline) and copied into `train_all_cif_clean2`: 9,113 complexes (was 8,947; 2,532 entries, 749 families, train 8,092 / held-out 1,021; no training complex shares a family with the 163 benchmark held-out complexes). D-form positions after the merge: DTR 313 (274), DGL 431 (400), DTH 435 (398), DHI 442 (400), DSN 480 (422), DLY 485 (441), DCY 545 (528), DTY 549 (537), MED 556 (538). The six rarest D-forms are unchanged (DAR 89, DPR 52, DAL 45, DVA 35, DLE 34, DPN 12; their laptop and box shards are not yet converted).

## 54. Model guide and trial G3 launched (18 Nov 2026)
What is unique about each model (control, A, B, C, D, E, F box, F H200, F2, G, G2, G3): see `model_guide.md`. Trial G3 (Tillicum job 495885, H200, `denoiser_G3_h200.slurm`, at most 1.8 USD): G2's setup on the class-balanced subset (`make_balanced_pool.py 280`, folder `train_all_cif_balanced280`: 2,232 training complexes, each of the 25 swapped residue types between 273 and 280 positions, plus the 1,021 held-out complexes of the 9,113-complex pool); start = trial G best weights, so only the data differs from G2. Results will be added when it ends.

## 55. Trial G3: G on the class-balanced subset of the pool (18 Nov 2026)

![G3 benchmark](figures/fig51_benchmark_with_G3.png)
![G3 extended real benchmark](figures/fig52_real_benchmark_extended_G3.png)
![G3 by residue group](figures/fig53_noncanonical_by_group_G3.png)

Setup (Tillicum job 495885, H200, `denoiser_G3_h200.slurm`): identical to G2 (section 53: start = trial G best weights, G recipe, batch 4, learning rate 5e-5, same clipped held-out check on the same 30 complexes) except the data: the class-balanced subset of the 9,113-complex pool (`make_balanced_pool.py 280`, folder `train_all_cif_balanced280`): 2,232 training complexes in which each of the 25 swapped residue types has 273 to 280 positions (ratio 1.03; the limit is DTR with 273 in the pool's training split), plus the pool's 1,021 held-out complexes. Early stop at step 1,900 (58 min, about 0.9 USD), best = step 1,100 (held-out check 57.77; G2 57.81). Weights `denoiser_trial/saved/H200_G3/dn_G3_H200_495885/best`.

Scores, paired change vs the untuned control (G / G2 / G3):
- Held-out synthetic, all positions (163, trimmed 154): -0.681 / -0.658 / -0.690: the same.
- Real 103, all positions (trimmed 97): -0.023 / -0.086 / +0.026; with a new D-form (n=26): -1.00 / -1.14 / -1.12; without (n=77): +0.038 / +0.009 / +0.125 (G3 is a little worse than G2 here; intervals overlap).
- Non-canonical latent loss, % change vs control, 987 real complexes: all non-canonical -1.1 / -4.1 / -5.1% (G3 best of all versions); 13 D-forms -11.4 / -9.7 / -13.3% (B -13.3%); 12 general codes -5.5 / -8.8 / -7.8%; other vocabulary codes +0.5 / -2.9 / -4.1%; canonical positions +0.1 / -0.0 / +0.1%.
- Real 103 only: all non-canonical -5.3 / -7.0 / -7.9%; D-forms -7.6 / -4.6 / -6.1%; general codes -4.8 / -9.3 / -7.8%; other codes -3.7 / -6.2 / -8.1%; canonical +0.4 / +0.4 / +0.6%.
- Held-out non-canonical: all -8.9 / -8.1 / -8.9%; D-forms -16.7 / -13.4 / -15.2%; general codes -15.5 / -22.8 / -22.5%.

Findings: (1) balancing the residue counts brought the D-form score back up (987 real: -13.3% against G2's -9.7%) and improved the overall non-canonical score (-5.1%, best of all versions) while using only 2,232 training complexes, a quarter of G2's data, and with the same synthetic held-out score; (2) the general codes stay about where G2 left them (-7.8% against -8.8%, intervals overlap), so the balance did not cost the general codes much even though each is now trained on 280 positions instead of about 1,000; (3) the price is small: canonical positions are not improved (+0.1%) and the all-position score on real complexes without a D-form is a little worse than G2's (+0.125 against +0.009, intervals overlap); (4) the six rarest D-forms (DAR DPR DAL DVA DLE DPN) are not in the balanced target (12 to 89 positions each in the pool); their generation is running (section 52c). One run per model; several differences are inside the intervals.
Klone retry 18 Nov, afternoon: job 43361084 (array task 1 of NJ=2, odd shards, pending; it duplicates the order of 43353994, so the two may collide on the same shard until one passes the other); 22 balance shards finished and pulled (`out2_klone_bal`), 8 merged into the pool, 14 not yet converted.

## 56. Denoise-then-decode with the same run 3 decoder for every model including G2 and G3 (18 Nov 2026)

![Decode benchmark, G2 and G3](figures/fig54_decode_run3_decoder_G2_G3.png)

Same procedure as section 51 (`eval_one_dec.py`, same noise per complex, latent time t = 0.05 / 0.15 / 0.3, extra real set at t = 0.1; the identity of non-canonical residues is given to the denoiser as input, so the informative part is D-form against L-form separation). ONE decoder for all twelve models: `decoder_v5_493747.pt` (run 3; every result file records this path; md5 ae541e8b1622ca0f7f736cf231505bda on the box). Models: untuned control, A, B, C, D, E, F box, F H200, F2, G, G2, G3. Outputs `denoiser_trial/saved/eval_dec5/`, figure `fig54_decode_run3_decoder_G2_G3.png`, script `make_fig54_decode_G2_G3.py`.

Accuracy change at non-canonical positions vs the control with the same decoder (points; control accuracy in brackets), G / G2 / G3:
- Held-out t=0.05, 13 D-forms (473 positions, control 96.8%): +2.1 / +2.1 / +1.7 (A +1.9, C +1.7, D +2.1); t=0.15: +1.9 / +2.7 / +2.3 (control 96.6%); t=0.3: +1.3 / +1.5 / +1.7 (control 97.9%).
- Held-out all non-canonical t=0.05 (584, control 97.3%): +1.9 / +1.9 / +1.4.
- Extra real set (884 complexes, t=0.1), 13 D-forms (93 positions, control 91.4%): +1.1 / +2.2 / +4.3 (A, B, D +5.4); all non-canonical (808 positions, control 96.4%): +0.7 / +1.0 / +1.2 (D +1.9).
- Real 103, 13 D-forms (47 positions, control 97.9%, one position = 2.1 points): -2.1 / 0.0 / +2.1 at t=0.05: no measurable difference.
- 12 general codes: control already at 98.6 to 100%; no model changes them by more than 1.4 points.
- Canonical positions (identity not given, 8.6 to 12% from noisy latents against 99% from true latents): all models within -0.7 to +0.4 points.

Findings: (1) with a fixed, strong decoder the control already names the D-forms at 91 to 98%, and every fine-tuned denoiser adds at most about 2 to 5 points; the differences between G, G2 and G3 are inside one or two positions per code and are not significant; (2) G3 (balanced data) is the highest on the extra real D-forms (+4.3 points) and G2 and G are higher than G3 on the held-out D-forms by 0.4 points at most: no clear winner; (3) this decode test therefore does not separate the denoiser versions beyond what section 51 showed; the loss benchmarks (sections 53 and 55) remain the discriminating tests, and a withheld-identity test or a generation test is still needed to see a real difference in naming. Not tested: withholding the non-canonical identity, all-atom reconstruction, generation from noise.

## 57. Decode with the residue identity WITHHELD: the decoder cannot name non-canonical residues without it (18 Nov 2026, interim)

Why: in every decode test so far (sections 48, 51, 56) the pipeline gave the true identity of each non-canonical residue to the model as an input (`C_SEQ`), and the decoder's own input features contain `cond_residue_type` (bundle config `feats_seq`: `ca_coors_nm`, `z_latent_seq`, `token_type`, `cond_residue_type`), so the decoder could copy the identity. Script `eval_one_dec_blind.py` (runner `run_eval_blind.sh`, outputs `~/rfd4_trial/out/eval_blind/{v1,v2}_{set}_{model}.json` on the box): `MODE=v1` replaces the identity of the generated tokens by <M> for both the denoiser and the decoder (token type kept, so the decoder chooses among the non-canonical classes only); the decode of the TRUE latents (oracle) gives the decoder's own ability, the decode of the denoiser's predicted latents the denoiser plus decoder. Same decoder (`decoder_v5_493747.pt`, run 3), latent t = 0.15 (extra real set 0.1), same noise per complex. `MODE=v2` (decoder allowed to name any of the 189 classes) is INVALID: the decoder was trained only with its class list restricted by token type, so outside logits are untrained (canonical oracle 99% -> 42%); not used.

Untuned control, top-1 accuracy at non-canonical positions, identity given (sections 51, 56) against identity withheld:
- Held-out synthetic (584 positions): identity given 100% (true latents) / 97.3% (predicted latents); withheld: true latents 34.6%, predicted latents 3.9%.
- By group, withheld, true latents (oracle) / predicted latents: held-out D-forms (473) 39.3% / 4.2%; general codes (71) 4.2% / 0.0%; other vocabulary codes (40) 32.5% / 7.5%. Real 103: D-forms (47) 38.3% / 0.0%; general codes (83) 0.0% / 0.0%; other (127) 15.7% / 8.7%. Extra real 884: D-forms (93) 38.7% / 0.0%; general codes (171) 1.8% / 0.0%; other (544) 6.4% / 4.4%.
- Canonical positions (identity never given there): true latents 98.8%, predicted latents 9% (unchanged).

Findings: (1) the run 3 decoder reads D-forms from the true latent code of a real peptide only about 38% of the time and the general codes (AIB, HYP, SAR ...) about 0 to 4% when the identity is not given; the 91 to 98% in sections 51 and 56 and the 110 of 112 in section 49 are largely the decoder copying the given identity (the real-pose test uses the same pipeline; whether its identity was given is NOT verified). (2) With the denoiser in the loop the control names 0 to 4% of the non-canonical residues blind (chance among the 142 classes is below 1%). (3) The denoiser comparison in sections 51 and 56 therefore says little about naming. (4) For design (where the identity is not given) the decoder has to be retrained with the identity hidden; the denoiser versions can be compared blind only up to the decoder's own ceiling (the oracle above). The blind scores of all twelve models are being computed (run in progress).
Klone retry 18 Nov, evening: job 43362246 (array tasks 1 and 2 of NJ=3, shards 1,4,7,... and 2,5,8,..., pending); 31 balance shards finished and pulled (`out2_klone_bal`), 8 merged into the pool, 23 not yet converted. Running: 43352313, 43353994, 43361084.
Klone retry 18 Nov, late evening: job 43364065 (array task 1 of NJ=2, pending); running: 43353994, 43361084, 43362246 (tasks 1, 2); 38 balance shards finished and pulled (`out2_klone_bal`), 8 merged, 30 not yet converted.
Klone retry 18 Nov, night: job 43365631 (array tasks 1-3 of NJ=4, pending); running: 43361084, 43362246 (tasks 1, 2), 43364065; 50 balance shards finished and pulled (`out2_klone_bal`), 8 merged, 42 not yet converted.
Note 18 Nov (evening): the NCFlow job of the six rare D-forms on box GPU 0 (`r6box0`, shard 13, pid 493971) was PAUSED (SIGSTOP) to give the generation test and the blind decode scoring the card; resume with `kill -CONT 493971` on the box (the GPU 1 job `r6box1` and the laptop job keep running).
Klone retry 19 Nov, early morning: job 43368782 (array task 1, pending; accepted by the first of two calls); running: 43361084, 43362246 (tasks 1, 2), 43364065, 43365631 (tasks 1-3); 75 balance shards finished and pulled (`out2_klone_bal`), 8 merged, 67 not yet converted.

## 58. Blind decoder (run 4): trained with the residue identity hidden (18 to 19 Nov 2026)

![Blind decode, run 3 against run 4 decoder](figures/fig55_blind_decoder.png)

Correction to earlier sections: the decode tests of sections 48, 51 and 56, and very probably the real-pose test of section 49 (same pipeline, not verified), GAVE the true identity of the non-canonical residues to the model (`C_SEQ`); the decoder takes it as input (`cond_residue_type`, bundle config) and was trained with it, so those accuracies (91 to 98% on D-forms) are largely the decoder copying the identity. The loss benchmarks (sections 46, 47, 50, 53, 55) use no decoder and are not affected.

Decoder run 4: `decoder_fine_tune_v6.py`, Tillicum job 496342 (MIG slice, 75 min, about 0.2 USD): start = run 3 decoder, same data (pool + 1,429 synthetic complexes + 7,534 mirror-image D-peptides, same split), but the identity (C_SEQ) is set to <M> for every token in training and validation (`BLIND=1`), so the decoder must read the residue from the latent code. Weights `decoder_v6_496342.pt` (laptop and box `decoder_v6_BLIND.pt`, md5 60a1a051e7cb88a3d48fbe871f1fa37f). Synthetic validation with the identity hidden after epoch 4: non-canonical accuracy 93.8% (run 3: about 35 to 39% in the same blind setting), canonical 98.8% unchanged.
Blind decode test (`eval_one_dec_blind.py`, `run_eval_blind.sh`, `run_eval_blind6.sh`; identity hidden for denoiser and decoder, token type kept, latent t = 0.15, extra real 0.1; box outputs `out/eval_blind` = run 3 decoder, 12 models; `out/eval_blind6` = run 4 decoder, 7 models; laptop `denoiser_trial/saved/eval_blind*`), top-1 accuracy at non-canonical positions: decoder alone on the TRUE latents, run 3 / run 4:
- 13 D-forms: held-out synthetic 39.3 / 94.3%; real 103 38.3 / 53.2%; extra real 884 38.7 / 61.3%.
- 12 general codes: 4.2 / 87.3%; 0.0 / 85.5%; 1.8 / 89.5%.
- other vocabulary codes: 32.5 / 82.5%; 15.7 / 59.1%; 6.4 / 24.3%.
Denoiser + decoder (control / G / G2 / G3), run 3 decoder: D-forms 0 to 8%, general codes 0 to 1%, other codes 4 to 12%; run 4 decoder: D-forms 0 to 7%, general codes 1 to 7%, other codes 1 to 5%, i.e. no model names the hidden residues from its own predicted latents.

Findings: (1) blind training works: the decoder reads general codes from true latents at 86 to 90% (run 3: 0 to 4%) and D-forms at 53 to 61% on real structures (94% on synthetic), so the identity is encoded in the latent code and the earlier 91 to 98% was copying; the remaining gap between synthetic (94%) and real (53 to 61%) D-forms is the synthetic/real gap; other codes are weak on real complexes (24 to 59%). (2) With the denoiser in the loop every model, including the untuned control and all fine-tunes, stays at 0 to 12%: the latents the denoiser predicts from a noisy complex carry almost no usable identity unless the identity is given as input; the fine-tunes (A to G3) do not change this. (3) For design the identity therefore has to be specified by the designer (as in the generation test, section 59), or the denoiser would have to be trained with the identity hidden, which no trial did. (4) Not tested: the run 4 decoder on the 5 remaining denoisers (D, E, F box, F H200, F2).

## 59. Generation test with real pockets, four denoisers x two decoders (19 Nov 2026)

![Generation test](figures/fig56_generation_test.png)

Setup (`gen_test/`: `prep_inputs.py`, `make_spec.py`, `gen_run.py`, `run_gen_all.sh`, `analyze_gen.py`, figure `make_fig56_generation.py`; run on the box GPU 0 with `rfd4_proteina.generate(model, overrides)` and a model built from `bundle_189` + the denoiser LoRA merged + the chosen decoder): 11 real peptide-protein complexes of the real test set (peptides of 7 to 10 residues, all non-canonical residues in the 189-class vocabulary, receptor pocket of at least 20 residues; test only, never trained on). The receptor pocket is fixed (C_CRD and C_SEQ on all receptor tokens, every run of consecutive residues of the pocket is its own chain), hotspots = receptor residues within 4.5 A of the native peptide, the peptide is generated (60 steps) with the identity of its non-canonical residues GIVEN at their native positions (C_SEQ at those positions) and the other residues designed. 8 designs per complex (88 per run), same seed, 4 denoisers (untuned control, G, G2, G3) x 2 decoders (run 3 `decoder_v5_493747.pt`, run 4 blind `decoder_v6_496342.pt`) = 8 runs. Raw outputs, no relaxation, no refolding with a structure predictor. The designed complex is superposed on the native receptor (receptor C-alpha RMSD 0.06 A) before comparing the peptide. (A first pass produced 48 designs per run because one complex failed the geometry check on a receptor fragment with a 4 A gap; fragments are now also split at C-N distances above 2 A and the five missing complexes were added.)

Results (88 designs per run; control / G / G2 / G3 with decoder run 3, then run 4):
- Given non-canonical residues kept: 100% in every run.
- D-forms with the right handedness (C-alpha signed volume, reference from native receptor L-residues; native D-residues are all right): run 3 29 / 35 / 39 / 42%; run 4 19 / 31 / 32 / 27%. Canonical L-residues right: 99 to 100% in all runs.
- Non-canonical residues with the native heavy-atom count: run 3 98.7% in all four; run 4 94.0 / 95.8 / 98.5 / 95.7%.
- Median CA distance to the native peptide (A): run 3 5.9 / 5.1 / 5.5 / 6.1; run 4 7.6 / 5.1 / 5.5 / 6.1 (the backbone does not depend on the decoder, so G, G2 and G3 place the peptide about 0.5 to 0.8 A closer than the control, G best).
- Median peptide-receptor contacts relative to native: run 3 1.2 / 1.3 / 1.4 / 1.4; run 4 1.4 / 1.5 / 1.7 / 1.7 (the designs pack tighter than the native peptide).
- Sequence identity to native at the free positions: 9.7 / 13.1 / 12.7 / 12.4% (same for both decoders; G, G2, G3 above the control).
- Peptide bonds of sensible length (1.2 to 1.5 A): 22 to 26% in all runs; designs with an atom pair closer than 2 A to the receptor: run 3 53 / 63 / 65 / 72%; run 4 73 / 73 / 82 / 86%.

Findings: (1) the pipeline works with the fine-tuned denoisers and keeps the specified residues, and the pose is plausible (peptide centre within about 1 A of the native one in the example checked, median backbone distance 5 to 6 A for G and G2). (2) The main failure is handedness: designed D-forms have the right C-alpha handedness in only 19 to 42% of cases (a coin flip would give 50%), for every denoiser and both decoders, while L-residues are right; this agrees with the earlier finding that the synthetic D-forms were L-geometry with a D label and says the denoisers/decoders do not produce real D-geometry in generation. (3) The blind decoder (run 4) is not better for generation: handedness is lower (27 to 32% for G, G2, G3 against 35 to 42%), atom counts and clashes are worse; run 3 is the better decoder when the identity is given. (4) Among denoisers with decoder run 3, G3 (balanced data) has the highest D-form handedness (42%) and G has the closest backbone (5.1 A); the differences are small (88 designs each, 11 complexes) and not tested for significance. (5) Raw bond geometry is poor in every run (only about a quarter of peptide bonds have a sensible length) and most designs have close contacts, so the designs would need relaxation, and a structure-predictor (RF3 / AF3) check, before any conclusion about foldability; not done. (6) Not tested: more denoising steps or recycles, relaxation, structure-predictor refolding, canonical-only controls.
Klone retry 19 Nov: job 43370282 (array tasks 1-4 of NJ=5, pending; they mostly re-check finished shards); running: 43362246 (task 1), 43364065, 43365631 (tasks 2, 3); 90 of the 96 balance shards are finished and pulled (`out2_klone_bal`), 8 merged into the pool, 82 not yet converted.

### 59b. Correction and check on the D-form handedness explanation (19 Nov 2026)
Finding (2) of section 59 linked the wrong handedness of designed D-forms to the synthetic D-forms being "L-geometry with a D label". That was not measured and is partly wrong. Check (`gen_test/chk_pool_chir.py`, 1,500 random complexes of `train_all_cif_clean2`, same signed-volume measure as the generation analysis): the NCFlow D-residues in the training pool have the right C-alpha handedness in 1,308 of 1,308 cases (100%; median |signed volume| 2.41, native 2.5). So the data do not contain wrongly handed D-residues at the C-alpha. What the data cannot show (not measured) is whether the backbone torsions (phi, psi) of those D-residues are the mirrored ones (NCFlow keeps the natural L peptide backbone and mirrors the side chain, `mirrored: True` in the manifests), and why the generated D-forms come out wrong (denoiser latents or decoder reading) is not yet separated: the proposed diagnostic is to decode the TRUE latents of native D-residues with the run 3 decoder and measure the handedness of the decoded atoms.

### 59c. Diagnostic: the decoder cannot reproduce D-form handedness even from the TRUE latents (19 Nov 2026)
`gen_test/diag_decode_chir.py`, `run_diag_chir.sh` (results `gen_test/diag/*.json`): the real complexes are encoded with the frozen encoder and the TRUE latents are decoded (no denoiser); handedness = C-alpha signed volume (atom37 slots N 0, CA 1, C 2, CB 3) of the decoded atoms at D-form positions, reference sign from the native L residues of the same complex; positions with a CB only. Native D-residues: 100% right in both sets. Decoded D-residues right (real 103: 104 positions in 55 complexes; extra real: 178 positions in 72 complexes): run 3 decoder, identity given 49% / 54%; run 3 decoder, identity hidden 52% / 61%; run 4 decoder (blind), identity hidden 38% / 31%. Decoded canonical L-residues right: 99.9 to 100% in every setting.
Finding: with perfect latents the decoder places the C-beta of a D-residue on the right face only about half of the time (a coin flip), so the decoder alone caps D-form handedness in generation at about 50%, whatever the denoiser does; the denoiser result (19 to 42%) is therefore not explained by the denoiser alone, and the mirror-complex data for the denoiser cannot fix it by itself. Likely cause (not yet tested): the decoder fine-tunes (runs 1 to 4, `decoder_fine_tune_v3` to `v6`) train only the residue-type cross-entropy (`loss = CE(pool) + LAMBDA * CE(complexes)`), so the atom-coordinate output was never trained on D-forms; the identity and the C-alpha are right but the atom layout of a D-residue comes from the original canonical-L decoder. Proposed fix: fine-tune the decoder with the autoencoder's coordinate (reconstruction) loss on D-residues (mirror peptides, NCFlow complexes, mirror complexes), then repeat this diagnostic and the generation test.
Klone 19 Nov: 95 of the 96 balance shards are finished and pulled (`out2_klone_bal`); the submitted job 43372557 was cancelled again (nothing left to do), the last shard is still running in the earlier jobs. 87 shards are not yet converted into the pool (8 are).
Klone 19 Nov (night): the hourly Klone retry job was stopped; 95 of 96 balance shards are pulled and all converted into the pool (`bal_pocket12a`, `bal_pocket12b`; the missing shard was not generated).

## 60. Decoder run 5 (atom-coordinate loss): the decoder was the cap on D-form handedness (19 Nov 2026)

![Generation test, decoder run 3 against run 5](figures/fig57_generation_decoder_run5.png)

Decoder run 5: `decoder_fine_tune_v7.py`, Tillicum job 498188 (MIG slice, stopped by a host-memory error in the data loader after epoch 0, 28 min; the epoch-0 weights `decoder_v7_498188.pt` were saved and used; laptop and box `decoder_v7_COORD.pt`, md5 0b13ebd744beb1124c56953db5976549): start = run 3 decoder, identity given, same data as run 3 (46k peptide pool + 7,534 mirror-image D-peptides) with the 8,947-complex synthetic pool as the complex source, loss = residue-type CE + 0.25 x squared error of the decoded atoms against the true atoms (A^2, no alignment), validation prints the C-alpha handedness of decoded D-forms and L-residues. Epoch 0 (validation, mostly synthetic and mirror data): D-forms right 93.2% (19,235 of 20,647), L-residues 100.0%, atom squared error 0.357 A^2, naming of the new D-form classes 100%. Continuation `dec_v7b` (job 498314, one data worker, start = epoch-0 weights): epoch 0 of the continuation D-forms 96.7% (19,964 of 20,647), L 100%, atom squared error 0.269 A^2, best saved (`decoder_v7b_498314.pt` on Tillicum); it also stopped in epoch 1 with the same host-memory error.

True-latent decode of the real complexes (`gen_test/diag_decode_atoms.py`, identity given; atom RMSD in A against the native atoms; run 3 -> run 5): real 103: D-forms 3.92 -> 0.81, general codes 5.38 -> 0.76, other non-canonical 3.26 -> 0.97, canonical 0.52 -> 0.19; extra real: D-forms 3.88 -> 0.80, general 5.80 -> 0.90, other 3.41 -> 0.82, canonical 0.55 -> 0.19. D-forms decoded with the right handedness: real 103 51% -> 83%, extra real 47% -> 88% (native 100%); L-residues 99.9 to 100% in both. The run 3 decoder placed the atoms of non-canonical residues 3 to 6 A from the truth even from perfect latents, because decoder runs 1 to 4 had trained only the residue-type cross-entropy.

Generation test (same setup and 88 designs per run as section 59, only the decoder changes; denoiser: control / G / G2 / G3; decoder run 3 then run 5):
- D-forms with the right handedness: 29 / 35 / 39 / 42% -> 67 / 95 / 83 / 90%; canonical L-residues 100% in all.
- Peptide bonds of sensible length: 26 / 23 / 22 / 22% -> 67 / 67 / 66 / 68%.
- Designs with an atom pair closer than 2 A to the receptor: 53 / 63 / 65 / 72% -> 18 / 34 / 19 / 23%.
- Non-canonical residues with the native atom count: 98.7% in all (same as run 3; the given identities are kept in 100%).
- Median CA distance to the native peptide (A): 5.9 / 5.1 / 5.5 / 6.1 -> 5.9 / 4.8 / 5.2 / 5.3; sequence identity to native at free positions 9.7 / 13.1 / 12.7 / 12.4% -> 9.2 / 14.9 / 14.4 / 11.6%; median contacts relative to native 1.2 to 1.5 (unchanged).
Findings: (1) the decoder, not the denoiser, was the main cap on D-form handedness (section 59c); with a decoder whose atom output was trained on D-forms the designed D-residues have the right handedness in 83 to 95% of the designs of the fine-tuned denoisers, and bond lengths and clashes also improve a lot. (2) The denoiser fine-tunes now matter: the untuned control reaches 67%, G 95%, G3 90%, G2 83% (the differences between the fine-tunes are small with 88 designs each; not tested for significance). (3) Raw designs are still not relaxed or refolded (not done); about a third of the peptide bonds are still outside 1.2 to 1.5 A. (4) Decision (night plan rule 1): the mean over G, G2, G3 is 89% (above 70%), so the mirror-complex denoiser trial H is skipped tonight; the next step is a stronger decoder (run 6) and refolding.

## 61. Decoder runs 5b, 6 and 7: handedness against bond geometry (19 Nov 2026, overnight)

![All decoder runs](figures/fig58_decoder_runs.png)

Runs (all: identity given, start from the previous best, same 46k peptide pool + 7,534 mirror-image D-peptides as the peptide source; complex source `train_all_cif3` for run 5/5b and `train_all_cif4` = 11,291 complexes (with the new D-form complexes of the Klone balance run) for runs 6 and 7; `decoder_fine_tune_v7.py` / `v8.py`; one epoch each because the Tillicum data loader runs out of host memory when the second epoch starts, the best epoch-0 weights were saved; Tillicum MIG slice, about 0.4 USD each): run 5b = `dec_v7b` job 498314 (run 5 continued, `decoder_v7b_498314.pt`); run 6 = job 498707 (`decoder_v8_498707.pt`: coordinate weight of non-canonical tokens x3, chirality loss 0.5); run 7 = job 499080 (`decoder_v8_499080.pt`: x2, chirality 0.15, coordinate weight 0.5). Box copies `decoder_v7b_COORD2.pt`, `decoder_v8_CHI.pt`, `decoder_v9_BAL.pt`; all on the laptop in `C:\Research\RFD4_fine_tune`.

Validation (identity given): D-forms right run 5 93.2% / 5b 96.7% / 6 99.3% / 7 98.7%, L-residues 100%, atom squared error 0.357 / 0.269 / 0.357 (different weights) / 0.230 A^2.
Decoder alone on the TRUE latents of the real 103 complexes (`diag_decode_atoms.py`; run 3 / 5 / 5b / 6 / 7): D-forms decoded with the right handedness 51.0 / 82.7 / 95.2 / 98.1 / 99.0% (extra real 884: 46.6 / 88.2 / 93.3 / 97.2 / 99.4%); atom RMSD (A) D-forms 3.92 / 0.81 / 0.56 / 0.67 / 0.53, general codes 5.38 / 0.76 / 0.67 / 0.75 / 0.76, other non-canonical 3.26 / 0.97 / 0.92 / 0.83 / 0.88, canonical 0.52 / 0.19 / 0.18 / 0.40 / 0.21.
Generation test (G / G2 / G3, 88 designs each, same setup as section 59): D-forms right: run 3 35 / 39 / 42%, run 5 95 / 83 / 90%, run 5b 96 / 94 / 96%, run 6 98 / 99 / 99%, run 7 98 / 97 / 99%; peptide bonds of sensible length: 23 / 22 / 22%, 67 / 66 / 68%, 67 / 74 / 71%, 26 / 22 / 21%, 51 / 50 / 54%; designs with an atom pair closer than 2 A: 62 / 65 / 72%, 34 / 19 / 23%, 33 / 17 / 23%, 38 / 22 / 33%, 35 / 19 / 26%; given identities kept 100% and native atom count 98.7% in all runs; backbone distance and sequence identity depend on the denoiser only.
Findings: (1) training the decoder's atom output on D-forms removed the cap on D-form handedness: 35 to 42% (run 3) to 94 to 99% with runs 5b to 7, and the atom RMSD of non-canonical residues fell from 3 to 5 A to 0.5 to 0.9 A (general codes such as AIB, HYP, SAR included). (2) There is a trade-off: the stronger the handedness loss, the worse the backbone atoms (run 6: canonical RMSD 0.40 A, peptide bonds back to about 22%); run 7 (lighter loss) keeps 97 to 99% handedness with 50 to 54% good bonds and canonical RMSD 0.21 A; run 5b has the best bonds (67 to 74%) with 94 to 96% handedness. (3) Decision: by the night-plan rule (D handedness >= 95%, bonds not worse than 5b, canonical RMSD <= 0.25) run 5b stays the default decoder (`decoder_v7b_498314.pt`) and run 7 (`decoder_v8_499080.pt`) is the handedness-first alternative; run 6 is not used. (4) Unchanged: raw designs are not relaxed or refolded; about a third to a half of the peptide bonds are still outside 1.2 to 1.5 A; the denoiser fine-tunes (G, G2, G3) differ little with 88 designs (not tested for significance); the mirror-complex denoiser trial H was not run (not needed for handedness any more).


## 62. Decoder run 8: a loss on the peptide-bond length (20 Nov 2026)

![fig59](figures/fig59_bond_loss.png)
Figure: `figures/fig59_bond_loss.png` (method strip; generation test with real pockets, G / G2 / G3, 88 designs each; decoder alone on true latents of the real 103 complexes). Script `make_fig59_bond_loss.py`.

Question: can a loss on the decoded C(i)-N(i+1) distance fix the bond lengths (run 5b: only 67 to 74% of designed peptide bonds in 1.2 to 1.5 A) without losing D-form handedness?
Loss (`bond_loss.py`): for each pair of neighbours that is bonded in the TRUE structure (true C-N between 1.0 and 1.8 A; chain breaks, gaps and padding are skipped) the penalty is max(0, |d - 1.33 A| - 0.04 A)^2 on the decoded atoms; weight 10 (`BOND_W`), added to the run 7 loss (CE + 0.5 x atom coordinates with non-canonical x2 + 0.15 x chirality). Unit test on a native complex: native bonds cost about 0 (139 of 139 sensible), loss grows with atom noise (0.0088 at 0.1 A, 0.062 at 0.2 A) and gradients flow. Script `decoder_fine_tune_v9.py`, slurm `decoder_v9_h200.slurm`.
Training: one full H200 on Tillicum (8 CPUs, 120 GB host memory, batch 128 + 16, 4 workers), start = run 5b, same data as run 7 (46k peptides, 7,534 mirror D-peptides, 11,291-complex pool). First try (job 499572) also put the bond loss on a second decode of NOISED latents (0.1 x latent std): the loss exploded (pool loss 64, complex loss up to 60), job cancelled after 10 min (about 0.15 USD). Second try (job 499602, noised term off) trained 4 epochs in 37 min (about 0.6 USD, EXIT 0, no host-memory crash with the larger memory); val on true latents per epoch: sensible bonds 80.6 / 80.7 / 84.5 / 85.5%, D-form handedness 92.6 / 93.7 / 94.6 / 95.4%, atom sq error 0.303 / 0.310 / 0.304 / 0.294 A^2. Weights `decoder_v9_499602.pt` (box `decoder_v10_BOND.pt`), log `dec_v9_499602.log`.
Generation test (G / G2 / G3, 88 designs each, same seed and setup as section 59; box runs `*_v10`):
- peptide bonds of sensible length: run 3 23 / 22 / 22%, run 5b 67 / 74 / 71%, run 7 51 / 50 / 54%, **run 8 83 / 87 / 87%**.
- D-forms with the right handedness: 35 / 39 / 42%, 96 / 93 / 96%, 98 / 97 / 99%, **run 8 100 / 92 / 97%**.
- designs with an atom pair closer than 2 A: 62 / 65 / 72%, 33 / 17 / 23%, 35 / 19 / 26%, **run 8 30 / 19 / 20%**.
- given identities kept 100% and native atom count 98.5 to 98.7% in all runs.
Decoder alone on true latents of the real 103 complexes: D-forms right 51.0 / 95.2 / 99.0 / 91.3%; atom RMSD of D-forms 3.92 / 0.56 / 0.53 / 0.64 A; general codes 5.38 / 0.67 / 0.76 / 0.71 A; other non-canonical 3.26 / 0.92 / 0.88 / 0.82 A; canonical 0.52 / 0.18 / 0.21 / 0.24 A (run 3 / 5b / 7 / 8).
Findings: (1) the bond loss lifts sensible peptide bonds in designs from 67 to 74% to 83 to 87% while keeping D-form handedness at 92 to 100% and not increasing close contacts, so it removes most of the trade-off seen in sections 60 and 61 (handedness against bonds). (2) On true latents run 8 is a little weaker than run 5b and run 7 (D right 91.3%, D-form RMSD 0.64 A, canonical 0.24 A), so the gain shows up where it matters, in decoding the denoiser's imperfect latents; G2 handedness is 92% (run 5b 93%). (3) Decision: run 8 (`decoder_v9_499602.pt`) is the new default decoder for generation by the rule bonds not worse and handedness >= 92% (canonical RMSD 0.24 <= 0.25); run 5b stays as the fallback, run 7 for handedness first. (4) Not done: the noised-latent bond term needs a much smaller noise scale; raw designs are still not relaxed or refolded with RF3 or AF3; 88 designs per run, no significance test; bonds are still 13 to 17% outside 1.2 to 1.5 A.


## 63. AF3 refold of a small subset of run 8 designs (20 Nov 2026)

![fig60](figures/fig60_af3_refold.png)
Figure: `figures/fig60_af3_refold.png` (method strip; receptor gate; valid refolds; ipTM). Scripts `gen_test/refold/build_refold_inputs.py`, `run_refold.sh`, `analyze_refold2.py`, `make_fig60_refold.py`; outputs `gen_test/refold/out_all`, summary `refold_summary.json`.

Question: do the designs (run 8 decoder, denoiser G) keep their pose and specified residues when an independent predictor refolds them?
Method: AlphaFold 3 on the box (GPU 0, single sequence without MSA, 3 diffusion samples, 10 recycles). Subset: designs 0 and 1 of each of 10 real complexes (9B4L_E skipped: two receptor chains) = 20 designs, plus the NATIVE peptide of each complex as a control (10; all residues in the 189-class vocabulary). Receptor = full native chain, sequence and native structure given as a template (single-chain mmCIF with entity_poly_seq and a release date; a first try failed on those two fields); peptide = designed sequence with every non-canonical residue given as a CCD modification. Comparison: AF3 model superposed on the native receptor (all CA and pocket CA, pocket = receptor CA within 10 A of a native peptide CA); designs superposed on the native receptor as in section 59.
Results: (1) the receptor was NOT kept in most jobs: pocket RMSD to native above 4 A in 14 of 20 designs and 8 of 10 natives (up to 17 A), so the template was not followed (single-sequence AF3 folds these receptors poorly and the template did not hold them); a pose comparison is only valid in 6 designs and 2 natives (pocket RMSD below 4 A). (2) In the valid cases: native control 8I95_D peptide CA RMSD 1.6 A, 9UMJ_D 6.1 A; designs: refold vs design 3.9 A (9UMX_D design 1), 6.8 and 7.0 A (9UMJ_D), 12.0 and 17.1 A (8I95_D), 13.9 A (9Q0F_B design 1). So where the receptor is right, AF3 agrees with the design in one case out of six (3.9 A) and the native control itself is recovered in one of two. (3) Specified non-canonical residues are present in 100% of refolds, but this is forced by the CCD input, not a finding; D-form handedness of AF3 outputs by my sign test is also mixed on the native controls (0 to 100%), so it is not informative here. (4) AF3 ipTM: designs median 0.42, natives median 0.50 (n 20 and 10); peptide pLDDT 47 to 81. Peptide bond lengths in AF3 outputs 1.2 to 1.5 A in 80 to 100% of bonds per peptide.
Findings: the test is inconclusive on pose, not a pass or a fail: the independent predictor cannot rebuild even the native complexes in this setting (single sequence, template not enforced), so a mismatch between refold and design cannot be blamed on the design. Not done: a refold in which the receptor is held fixed (template that AF3 obeys, or a predictor that takes pocket or structure constraints, e.g. Boltz-2 with templates or RF3), MSA for the receptor, more designs, the canonical-only comparison.


## 64. Boltz-2 refold with the receptor held (20 Nov 2026)

![fig61](figures/fig61_boltz_refold.png)
Figure: `figures/fig61_boltz_refold.png` (method strip; receptor gate AF3 vs Boltz-2; peptide pose per complex; ipTM). Scripts `gen_test/refold/build_boltz_inputs.py`, `run_boltz_refold.sh`, `analyze_refold_boltz.py`, `make_fig61_boltz_refold.py`; outputs `gen_test/refold/boltz_out`, `boltz_summary.json`.

Why: the AF3 refold of section 63 did not keep the receptor (pocket RMSD above 4 A in most jobs, even for the native controls), so it could not judge the designs. Method: same 30 jobs (20 run 8 designs of denoiser G, 2 per complex, 10 complexes; plus the 10 native peptides), now with Boltz-2 (2.2.1) and the native receptor structure as a FORCED template (`force: true`, threshold 2 A, `--use_potentials`), single sequence, 3 diffusion samples on the box 5060 Ti, best sample by Boltz confidence. Compared in the receptor frame as before (peptide CA RMSD to the design and to the native peptide, designs superposed on the native receptor).
Results: (1) the receptor was kept in 30 of 30 jobs (median pocket RMSD 1.09 A, all below 1.4 A; AF3: 14 of 30 above 4 A), so every pose comparison is valid. (2) Refold vs design: CA RMSD below 3 A in 4 of 20 designs (9KXS_D designs 0 and 1: 2.7 and 2.6 A, 9V38_C designs 0 and 1: 2.5 and 2.4 A), below 5 A in 11 of 20; median 4.8 A. (3) The control: the NATIVE peptide refolded with the same receptor is recovered below 3 A in only 4 of 10 complexes (7XAU_F 0.9, 9Q0F_B 0.9, 8I95_D 1.4, 9KXS_D 2.8 A), below 5 A in 5 of 10, median 5.4 A; the designs themselves are within 3 A of the native peptide in 3 of 20 and within 5 A in 12 of 20 (median 4.6 A), so a design agrees with its refold about as often as the native peptide agrees with its own refold. (4) Boltz ipTM: designs median 0.72, natives 0.70 (AF3: 0.42 and 0.50). (5) Specified non-canonical residues are present in 100% of refolds (forced by the CCD input, not a finding); D-form handedness (sign relative to the receptor L residues) is right in every refold that has a D-form, native controls included, so Boltz places the CCD D-forms correctly whatever the pose; peptide bonds in 1.2 to 1.5 A in 92% of bonds on average (60 to 100% per peptide).
Findings: with the receptor held, the independent predictor does not contradict the designs: it gives designs the same confidence as native peptides and recovers the design pose about as well as it recovers the native pose, but that is a weak test because Boltz itself recovers the native peptide below 3 A in only 4 of 10 (peptides of 5 to 10 residues, no MSA). The four designs that refold within 3 A of the design (two complexes) are the ones with the strongest agreement. Not done: more designs and the other denoisers, the canonical-only control designs, relaxation of the designs, multiple seeds, structure-based scoring (interface energy).
Unresolved: 7 complexes (all but 9KXS_D, 9V38_C, 9Q0F_B for pose) have a design-versus-refold CA RMSD of 4 to 17 A; whether this is the design or the predictor cannot be separated with this subset.

### 62b. Control (vanilla) denoiser with decoder run 8 (20 Nov 2026)
Generation test, same setup as section 59 (88 designs; box run `control_v10`, denoiser = the untuned base model, decoder run 8): D-forms right 92%, peptide bonds of sensible length 87%, designs with an atom pair closer than 2 A 22%, median CA distance to the native peptide 5.91 A, contacts relative to native 1.29, free-position sequence identity 0.083. Fine-tuned denoisers with the same decoder (G / G2 / G3): D-forms 100 / 92 / 97%, bonds 83 / 87 / 87%, close contacts 30 / 19 / 20%, CA distance 4.78 / 5.20 / 5.35 A, contacts 1.50 / 1.49 / 1.49, free identity 0.149 / 0.144 / 0.116. So with decoder run 8 the decoder fixes handedness and bonds for every denoiser (the control had 67% handedness with run 5, section 60); what the fine-tuned denoisers add is the pose (CA distance 0.6 to 1.1 A closer, contacts 1.49 against 1.29) and the sequence identity at the designed positions (0.12 to 0.15 against 0.08). Close contacts are about the same for the control (22%) and G2 / G3 (19 / 20%); G is higher (30%). With decoder run 5 the control had fewer (18%) because its peptides sit farther from the pocket. Anatomy of the clashes (G3, run 8): 19 of 88 designs, 13 clashing side-chain atoms and 8 backbone atoms, no CA atom closer than 2 A, median closest atom pair 1.72 A (G: 29 designs).

## 65. Decoder run 9: a clash loss on peptide-receptor atom distances (20 Nov 2026)

![fig62](figures/fig62_clash_loss.png)
Figure: `figures/fig62_clash_loss.png` (method strip; generation bonds, D handedness and close contacts per denoiser; decoder alone on true latents). Script `make_fig62_clash_loss.py`; code `bond_loss.py` (`clash_loss`), `decoder_fine_tune_v10.py`, `decoder_v10_h200.slurm`; weights `decoder_v10_500687.pt` (box `~/rfd4_trial/decoder_v11_CLASH.pt`); box runs `gen_test/out_local/*_v11`, diag `gen_test/diag/atoms_run9_*.json`.

Method: start = run 8, same loss (CE + atom coordinates 0.5 + chirality 0.15 + peptide-bond loss 10) plus a clash loss, weight 5: the DECODED atoms of the peptide against the TRUE atoms (detached) of the other chains of the same complex (chains from `chain_iid`), penalty max(0, 2.5 A - distance)^2, on the true latent and on a second decode of a noised latent (sigma 0.2 x std of the latent, training only). The noised-pass bond term was set to 0. 1 full H200, 4 epochs, job 500687 (about 35 min, about 0.6 USD); gradient clip 5.
The first attempt (job 500305, noised bond term weight 2) was cancelled: pool loss 14 to 18. A debug job (500426, MIG slice) found the cause: the noised-latent bond term read 6 to 8 on every pool batch (and 6.5 on some complex batches) while the clash terms were small (0.001 to 0.35); that term was removed in run 9 (it was not needed: run 8 holds bonds under noise, 93 to 89% at sigma 0.2 in the calibration).
Training: healthy, pool loss 0.8 to 1.6, complex loss 0.3 to 3.1 (the clash terms dominate the complex loss). Validation (epoch 3): pool loss 1.21, complexes 0.77; D-forms right 97.0%, L 99.7%; bonds in 1.2 to 1.5 A 85.8%; mean clash loss on true latents 0.067 (60,522 pairs below 2 A in 1,303 complexes).

Generation test (same setup as section 59: 11 real pockets, 88 designs per denoiser, raw outputs, unrelaxed), run 8 -> run 9:
- designs with an atom pair closer than 2 A to the receptor: G 29.5 -> 35.2%, G2 19.3 -> 20.5%, G3 20.5 -> 23.9%, control 21.6 -> 21.6%. NO reduction.
- peptide bonds of sensible length: G 82.9 -> 82.9%, G2 86.7 -> 83.9%, G3 86.9 -> 80.9%, control 86.6 -> 77.8% (lower).
- D-forms with the right handedness: G 99.6 -> 97.9%, G2 92.3 -> 96.8%, G3 97.2 -> 97.5%, control 92.1 -> 94.7%.
- CA distance to the native peptide and contacts are unchanged within rounding (G 4.78 -> 4.80 A, G3 5.35 -> 5.36 A, control 5.91 -> 5.91 A; contacts 1.50 -> 1.44 for G, 1.49 -> 1.48 for G3), as expected: the decoder does not move the backbone placement made by the denoiser.
Decoder alone on the true latents of the real complexes (55 with a D-form): D-forms right 96.2% (run 8: 91.3%; run 5b: 95.2%), D-form atom RMSD 0.65 A (0.64; 0.56), canonical residues 0.43 A (0.24; 0.18), general codes 1.19 A (0.71; 0.67). So the clash loss made canonical residues clearly less accurate.

Findings: the clash loss did not lower the share of designs with a close contact for any denoiser (G, G3 and G2 slightly higher, control equal; with 88 designs per run one design is 1.1 points, so these are within noise, but no improvement is seen), and it cost sensible bonds (3 to 9 points for G2, G3, control) and canonical atom accuracy. Run 8 stays the default decoder. Likely reasons (not tested): the close contacts of the generated designs come mainly from the denoiser placing the peptide backbone too close to the pocket (a decoder can move side chains but not the backbone, section 62b), and a noised latent with sigma 0.2 is probably not the same kind of error the denoiser makes. Not done: a smaller or larger clash weight, a different noise level, noising only the side-chain part, a clash term inside the denoiser training (reward or loss on the generated backbone), relaxation of the designs; one training run per setting.

Addendum to section 65 (20 Nov 2026): where the close contacts of the run 8 designs come from (`gen_test/diag_clash_source.py`, `gen_test/diag/clash_source.json`). The earlier guess that the denoiser backbone causes them is NOT supported: in none of the 352 designs (4 denoisers x 88) is a peptide CA closer than 3.5 A to a receptor CA (median closest CA-CA 4.3 to 4.6 A; native peptides 4.9 A). The clashing atoms are mostly side-chain atoms: backbone / side chain atoms of the clashing designs G 8 / 38, G2 8 / 10, G3 8 / 11, control 9 / 12; designs with backbone-only clashes 7 to 8 per run, side-chain-only 9 to 18 per run. For G, 34 of 46 clashing atoms sit at the given non-canonical positions; for G2, G3 and the control most sit at canonical (free) positions (12 to 13 of 18 to 21). Clash-free designs per pocket (8 designs each): at least one clash-free design in 10 of 11 pockets (G) and 11 of 11 (G2, G3, control); drawing 4 designs and keeping a clash-free one would succeed for about 90 to 98% of pockets (expected value from the per-pocket clash rates). Reading: the decoder never sees the receptor, so the side-chain placement can only come from the latent that the denoiser makes; this is why a decoder-only clash loss (run 9) could not help, and a filter at generation time already handles most pockets.

## 66. Denoiser trial G4: a clash loss through the frozen decoder inside the denoiser training (20 Nov 2026)

![fig63](figures/fig63_denoiser_clash.png)
Figure: `figures/fig63_denoiser_clash.png` (method strip; G3 vs G4: close contacts, bonds, D handedness, CA distance, contacts; training curve). Trainer `rfd4_ncaa_patch/ncaa_fine_tuning_v4.py` (v3 + `--clash-w`, `--clash-d0`, `--clash-tmin`), `inspect_batch_v4.py`, slurm `denoiser_v4_h200.slurm` (job 501522), `denoiser_v4_h200_dbg.slurm`; weights `denoiser_trial/saved/H200_G4/dn_v4_H200_501522/best` (box `~/rfd4_trial/out/G4_clash/best`); generation `gen_test/out_local/G4_clash_v10`; script `make_fig63_denoiser_clash.py`.

Why: section 65 addendum showed that the close contacts of the designs are mostly side-chain atoms (no CA closer than 3.5 A to a receptor CA) and that the decoder never sees the receptor; the only place that can put receptor-aware side-chain information into the latent is the denoiser.
Method: start = G3 best (LoRA), same class-balanced data (`train_all_cif_bal280`), v3 recipe (10x non-canonical weight, wcap 100), batch 4, 5 inner updates, lr 3e-5, 600 steps, one full H200 (job 501522, about 19 min, about 0.3 USD). Added loss: clean prediction of the denoiser (CA + local latent) -> the generation decode (`_format_sample_local_latents`, frozen run 8 decoder, eval mode, gradients only to the LoRA) -> decoded atoms of the peptide against the TRUE atoms (detached) of everything else: sum of max(0, 2.5 A - d)^2 per sample, mean over the samples whose two noise levels (sampled independently) are both >= 0.6 (about 17% of the samples), weight 2 (about 5 to 10% of the flow loss at that noise range). Peptide = all tokens of the chains that hold a non-canonical residue: in the training batches EVERY token is noised, `is_in_context_conditioning` is all False and each complex has many small chains (inspection job 501471), so this was the only split available. Three debug jobs were needed (MIG slice out of memory in the decoder gradient path; empty receptor set because all tokens are flow-emitted; then peptide tokens 5 to 30 and receptor atoms 400 to 1,500 per sample, healthy). The held-out check includes the weighted clash term.
Training: healthy, flow loss 50 to 80 (single outlier batches of 1e7 to 1e16 skipped, 16 of 600; one of 4,398 was used, clipped), clash x weight 0 to 4. Held-out score 59.11 (step 0), 59.01, 59.43, 58.89 (best, step 300), 59.03, 59.52, 60.10; weighted clash part 1.35, 1.17, 1.56, 1.00, 1.1, 1.6, 2.2: no trend; plateau decay at step 600. Best = step 300.
Generation test (G3 + decoder run 8 vs G4 + decoder run 8, 88 designs each, same setup): designs with an atom pair closer than 2 A 20.5 -> 19.3% (18 -> 17 designs), peptide bonds of sensible length 86.9 -> 78.3%, D-forms with the right handedness 97.2 -> 94.0%, median CA distance to native 5.35 -> 5.14 A, contacts relative to native 1.49 -> 1.42, non-canonical identities kept 100% for both, free-position identity 0.116 -> 0.137. Clash anatomy (`diag_clash_source.py`): G3 8 backbone + 11 side-chain atoms (6 at given non-canonical positions), G4 10 + 12 atoms (4 at given positions); no CA closer than 3.5 A to a receptor CA in either.
Findings: NO real change in close contacts (one design of 88, within noise) and a loss of sensible bonds (8.6 points) and a small drop in D handedness (3 points, 3 designs) at the same time; the other differences are small. So adding the clash loss to the denoiser at weight 2 did not reduce clashes. Possible reasons (not tested): the loss only acts on about 17% of the samples at the right noise level and is weighted at about 5 to 10% of the flow loss, so it may be too weak; the 11 pockets and 8 designs per pocket are too few to see a change of a few points; the training samples are noised complexes with all chains generated, whereas generation holds the receptor fixed; a latent that moves to avoid clashes may leave the region where decoder run 8 gives good bonds (bonds fell). Not done: a larger weight, a higher noise threshold, the same loss with a decoder that also gets bond loss through the denoiser, more designs per pocket, a design-time filter (section 65 addendum: 4 draws and keeping a clash-free design would work for about 90 to 98% of pockets), relaxation. G3 and decoder run 8 stay the defaults.

## 67. Best-of-N clash filter at generation time (20 Nov 2026)

![fig64](figures/fig64_clash_filter.png)
Figure: `figures/fig64_clash_filter.png` (method strip; clash-free share against N; quality of the kept designs; per-pocket counts). Scripts `gen_test/best_of_n_filter.py` (writes `gen_test/diag/best_of_n.json`), `figures/scripts/make_fig64_clash_filter.py`; fresh designs `gen_test/out_local/{G3_bal_v10_s777,G_ncw10_v10_s777}` (launcher `gen_test/run_seed777.sh`, `gen_run.py` now reads the seed from `GEN_SEED`).

Why: sections 65 and 66 showed that neither a decoder loss nor a denoiser loss lowers the share of designs with an atom pair closer than 2 A to the receptor (about 20% for every denoiser). Method: per pocket, draw N of the 8 designs (without replacement, 4,000 Monte Carlo draws per pocket) and keep one without such a pair (rule A: random among the clash-free ones; rule B: the clash-free one with the most sensible peptide bonds); a pocket with no clash-free design among the N counts as a failure. No training, decoder run 8, 88 designs per run (11 real pockets x 8). First batch seed 1234 (G, G2, G3, control); a FRESH batch with seed 777 (G3 and G, 88 designs each) generated afterwards on the box to check that the numbers are not specific to one seed.
Results (share of pockets with a clash-free design among N = 1 / 2 / 4 draws, rule A): seed 1234: G 70 / 86 / 91%, G2 81 / 94 / 98%, G3 79 / 93 / 99%, control 78 / 93 / 98%; fresh seed 777: G3 78 / 94 / 99.7%, G 76 / 89 / 95%. Unfiltered close-contact share of the fresh batch: G3 21.6% (seed 1234: 20.5%), G 23.9% (29.5%).
Quality of the kept designs (N = 4, G3, seed 1234; all designs -> kept, rule A): sensible bonds 86.9 -> 86.8%, D handedness 97.2 -> 96.8%, median CA distance 5.35 -> 5.26 A, contacts 1.49 -> 1.52; fresh seed: bonds 84.1 -> 85.5%, D 93.1 -> 92.2%, CA 5.13 -> 5.06 A. Rule B (clash-free and best bonds) lifts the sensible-bond share to 94.7% (G3, seed 1234), 91.4% (G3, seed 777), 88% (G), 93% (G2), 91% (control), with D handedness 95 to 100% and the pose unchanged.
Findings: (1) drawing 4 designs per pocket and keeping a clash-free one works for 98 to 99.7% of pockets for G2, G3 and the control and for 91 to 95% for G (one pocket of G had no clash-free design among its 8), and it does so without changing bonds, handedness or pose; it is the only measure of sections 62 to 67 that removed the close contacts. (2) The fresh seed gives the same picture (G3 99.7%), so the numbers are not specific to the first batch. (3) Cost: N designs per pocket instead of one (about 8 min for 88 designs on the box GPU, so about 4 min for 4 draws per pocket). Caveats: 11 pockets, 88 designs per run, Monte Carlo draws from 8 designs per pocket (the draws are not independent of the 8); the filter only tests heavy-atom distances below 2 A: clash-free designs are not relaxed and may still be strained (no energy check, no refolding of the filtered set); the contacts of the kept designs were not re-checked against the receptor chemistry. Not done: a filter on the 2.2 A threshold or on interface energy, refolding of the filtered designs.

## 68. Summary: decoder run 8 with each denoiser (20 Nov 2026)

![fig65](figures/fig65_denoiser_comparison.png)
Figure: `figures/fig65_denoiser_comparison.png` (script `figures/scripts/make_fig65_denoiser_comparison.py`); numbers from sections 62b, 66 and 67 (no new run). Generation test, 11 real pockets, decoder run 8; G and G3 pooled over two seeds (176 designs), control, G2, G4 88 designs.
Without the filter (close contacts / sensible bonds / D handedness / median CA distance / contacts): control 21.6% / 86.6% / 92.1% / 5.91 A / 1.29; G 26.7% / 83.5% / 99.1% / 4.91 A / 1.50; G2 19.3% / 86.7% / 92.3% / 5.20 A / 1.49; G3 21.0% / 85.5% / 95.2% / 5.23 A / 1.51; G4 19.3% / 78.3% / 94.0% / 5.14 A / 1.42. Identity to native at the designed positions: control 8%, G 14%, G2 14%, G3 12%, G4 14%. With the best-of-4 filter, rule B (clash-free share / bonds / D handedness): control 98 / 91 / 92%, G 91-95 / 88-89 / 98-100%, G2 98 / 93 / 96%, G3 99-100 / 91-95 / 93-95%.
Reading: no denoiser wins every measure. G has the closest pose (4.9 A) and the best handedness but the most close contacts and the fewest sensible bonds; G2 and G3 have fewer close contacts and more bonds; G4 (clash loss) is not better than G3; the untuned control has the farthest pose (5.9 A), the fewest contacts and the lowest sequence identity but is otherwise similar. Differences of a few points are within one standard error (about 2 to 4 points); the pose difference between the fine-tuned denoisers (4.9 to 5.2 A) is not tested for significance. With decoder run 8 plus the clash filter, G3 is the best all-rounder, G the choice when pose and handedness matter most.

## 69. Per non-canonical position: decoder alone and denoisers (20 Nov 2026)

![fig66](figures/fig66_per_code_decoder.png)
Figure: `figures/fig66_per_code_decoder.png` (script `figures/scripts/make_fig66_per_code_decoder.py`). Scripts `gen_test/diag_per_code.py` (decoder alone, box), `gen_test/run_percode.sh`, `gen_test/analyze_positions.py` (generated designs), outputs `gen_test/diag/percode_{vanilla,run3,run5b,run8}.json`, `gen_test/diag/positions.json`.

Question: how accurately is each individual non-canonical residue rebuilt, and do the fine-tuned models do better than the vanilla one at those positions?
(A) Decoder alone on the TRUE latents: all 468 real complexes that hold an NCAA (103 real + 365 of the 884 extra; test only), every NCAA position (1,066), identity GIVEN; vanilla = the untouched bundle_189 decoder (new classes untrained). Median all-atom RMSD to the native residue (vanilla / run 3 / run 5b / run 8): D-forms 4.00 / 3.65 / 0.52 / 0.54 A, general codes 4.53 / 4.84 / 0.69 / 0.70, other vocabulary codes 3.41 / 3.15 / 0.75 / 0.74, all 3.90 / 3.65 / 0.64 / 0.65 A. Residues within 1 A of the native: 0 / 0 / 82 / 82%. Decoder emits exactly the native atoms: 8 / 25 / 95 / 97%. Bond-length error inside the residue: 2.44 / 2.20 / 0.25 / 0.27 A. C-alpha handedness right (residues with a C-beta): 54 / 59 / 92 / 89% (D-forms 62 / 53 / 95 / 94%). Per code (run 8, codes with at least 12 positions): median RMSD 0.29 to 0.92 A for 21 of 26 codes (AIB 0.29, DAL 0.38, HYP 0.35, DPN 0.43, MLE 0.75, ALY 0.66); weakest CGU 1.32 A (51 positions, handedness 65%), STA 1.15, PTR 1.13 (handedness 35%, 17 positions), SEP 0.99; with at least 5 positions also TBG, SLL, B3Q, KCR (1.3 to 1.4 A). With the identity HIDDEN (decode from the latent alone) the residue is named correctly in 8 / 12 / 13 / 16% of the positions (D-forms up to 49% for run 8, general codes 1%): the decoder cannot name an NCAA from the latent, the specification must be given (agrees with section 57; the 99.6% of section 4 is a different setup in which the model is given the peptide's atoms).
(B) Generated designs (decoder run 8, 11 real pockets, 88 designs per denoiser, 176 NCAA positions each, G and G3 pooled over two seeds = 352; 152 to 304 of them D-forms, no 'other' codes): distance between the designed and the native C-alpha of the NCAA residue (receptor superposed), median: control 5.85 A, G 6.89, G2 6.06, G3 6.22, G4 5.47; fine-tuned (G, G2, G3) minus control +0.61 A, 95% bootstrap interval [-1.60, +1.47]: no detectable change. Internal geometry of the designed residue (superposed on the native residue) 1.0 to 1.1 A for every denoiser (set by the decoder). C-alpha handedness right at the NCAA positions: control 90.5%, G 98.5%, G2 88.1%, G3 89.9%, G4 91.7% (G clearly higher). Residue with an atom pair under 2 A to the receptor: control 5%, G 10%, G2 3%, G3 5%, G4 2%. Contacts relative to the native residue: median control 0.92, G 2.0, G2 1.7, G3 2.2, G4 1.5 (the fine-tuned designs pack the NCAA tighter).
Findings: (1) given the specification, the fine-tuned decoder rebuilds the residue accurately: 82% within 1 A, native atoms in 97%, bond errors 0.27 A, handedness 89 to 92%, against 3.9 A, 8%, 2.4 A and 54% for the vanilla decoder: this is the large jump of the project. (2) The decoder is accurate even for codes it was never specifically trained on (ALY, MK8, MP8, NLE, ABA, all under 0.7 A), with a few exceptions (CGU, PTR, STA, SEP). (3) The denoiser fine-tunes do not place the NCAA residue closer to the native residue in generation (about 5.5 to 7 A for every model; the native pose is not given, so different binding modes are legitimate) and do not change the internal geometry; their measurable gains are in the latent loss at NCAA positions (-5 to -13% for D-forms and general codes, sections 46 and 47), in the sequence identity at the designed positions (8% to 12-15%), in contacts (1.29 to about 1.5) and, for G, in handedness. Caveats: the generation sample is 11 pockets with 2 NCAA positions each, almost all D-forms: no per-code conclusion for designs; the decoder test starts from true latents; no relaxation.

## 70. Denoiser + decoder run 8 rebuild of NCAA residues (denoise-then-decode, all atoms) (20 Nov 2026)

![fig67](figures/fig67_denoise_decode.png)
Figure: `figures/fig67_denoise_decode.png` (script `figures/scripts/make_fig67_denoise_decode.py`). Scripts `gen_test/eval_dd_atoms.py`, `gen_test/run_dd.sh`; outputs `gen_test/diag/dd_{control,G_ncw10,G2_pool,G3_bal}_{real,ham}.json`.

Question: do the denoiser and the decoder TOGETHER rebuild a non-canonical residue accurately (the earlier denoise-then-decode tests, sections 48, 51, 56, scored only the naming accuracy with the identity given)? Method: the 103 real complexes plus the 365 extra real complexes that hold an NCAA (test only; 1,065 generated NCAA positions per noise level); the CA and latent of the generated tokens are noised to a fixed time t (0 = pure noise, 1 = clean; both fixed, same seeded noise for every model), the denoiser (untuned control, G, G2, G3) predicts the clean complex (given tokens keep their true values; the receptor and the NCAA identity are given, as in the generation test), decoder run 8 builds the atoms, and each NCAA residue is compared with the native one (all-atom RMSD). Oracle = decoder run 8 on the TRUE latent (no denoiser): median 0.66 A, 84% within 1 A, native atoms 97%, handedness 89%.
Median all-atom RMSD of the NCAA residue (control / G / G2 / G3, A): t = 0.9: 1.03 / 1.05 / 1.05 / 1.03; t = 0.7: 2.00 / 2.08 / 2.04 / 2.04; t = 0.5: 4.26 / 4.20 / 3.98 / 4.01; t = 0.3: 7.40 / 6.99 / 6.64 / 6.81; t = 0.1: 9.35 / 8.97 / 8.81 / 8.98. Residues within 1 A of the native: t = 0.9 about 46% for every model (oracle 84%), t = 0.7 5 to 6%, lower t 0%. Median error of the denoised C-alpha: t = 0.9 0.64 to 0.69 A, t = 0.7 1.7, t = 0.5 3.5 to 3.9, t = 0.3 6.2 to 7.0. Decoder emits exactly the native atoms: 97% at t = 0.9 and 95 to 96% at t = 0.7 (oracle 97%), 82 to 92% at lower t. C-alpha handedness right: t = 0.9 88 to 89% (oracle 89%), t = 0.7 81 to 84%, t = 0.5 72 to 79%, t = 0.1 64 to 67%. By group at t = 0.9 (median RMSD, control / G / G2 / G3): D-forms (336) 1.11 / 1.12 / 1.11 / 1.09 A, general codes (254) 0.99 / 0.99 / 0.99 / 0.98 A, other vocabulary codes (475) 1.01 / 1.05 / 1.04 / 1.02 A.
Findings: (1) from a mildly noised input (t = 0.9) denoiser + decoder rebuild a non-canonical residue to a median 1.0 A with the native atoms in 97% and handedness as good as the decoder alone (88 to 89%); the denoiser adds about 0.4 A to the decoder's own 0.66 A, and only 46% of the residues stay within 1 A (decoder alone 84%). (2) The error grows quickly with the noise: 2 A at t = 0.7, 4 A at t = 0.5, 7 A at t = 0.3; from near-pure noise the model generates a new pose and does not reproduce the native one (this is generation, not reconstruction). (3) The fine-tuned denoisers (G, G2, G3) are not distinguishable from the untuned control in the RMSD at t = 0.9 and 0.7 (differences of 0.02 to 0.08 A); at lower t the fine-tuned ones are a little better (0.3 to 0.8 A at t = 0.3 to 0.5, C-alpha error 0.1 to 0.8 A lower), with G2 the best; handedness is slightly worse than the control at t = 0.3 to 0.7 for G. (4) Caveats: the identity of the NCAA and the receptor are given; the noise is applied to the generated tokens only; these are real complexes of the test set; no relaxation; the C-alpha noise level is tied to the latent noise level, which generation does not do exactly.

## 71. Free-sequence generation: can the model choose the NCAA? (20 Nov 2026)

![fig68](figures/fig68_free_sequence.png)
Figure: `figures/fig68_free_sequence.png` (script `figures/scripts/make_fig68_free_sequence.py`). Scripts `gen_test/spec_free.yaml`, `gen_test/run_free.sh`, `gen_test/run_free_missing.sh`, `gen_test/analyze_free.py`; designs `gen_test/out_local/{control,G_ncw10,G3_bal}_free_{run8,blind}`; table `gen_test/diag/free.json`.

Method: the 11 real pockets of the generation test, 8 designs per pocket (88 designs, 176 NCAA slots per run), receptor fixed, the sequence condition (C_SEQ) of the WHOLE peptide removed (the NCAA identities are no longer listed in the spec); denoisers control, G, G3 x decoder run 8 and the blind decoder (run 4, trained with the identity hidden).
Result: the NCAA slots carry the NATIVE code in 100% of the 176 slots of every one of the six runs (control, G, G3; run 8 and blind decoder), with the same ten codes (DAL 32%, DPN 14%, DTR 14%, DIL 9%, ...) as in the runs with the identity given, and the canonical-slot recovery is identical to the earlier runs (control 8.6%, G 13.4%, G3 10.8%). So the removal of C_SEQ from the spec changed nothing: in this design pipeline the identity of a non-canonical residue is handed to the model (and written into the output) whatever the spec says (consistent with the finding of section 48: the sequence-conditioning feature holds the true class of every non-canonical token). The test therefore cannot measure whether the model can choose an NCAA, and nothing about the choice can be concluded from it; a real test needs the pipeline (the step that fills the sequence condition for non-canonical residues) or the input CIF changed so that the identity is not present at the NCAA slots.
What the run does show: (1) free canonical positions are recovered only 8.6% (control), 13.4% (G), 10.8% (G3) of the time (uniform guess over 20 residues: 5%); (2) the blind decoder (run 4) gives much worse structures in generation than decoder run 8: sensible bonds 26% against 85% (mean of the three denoisers), D-forms with the right handedness 19% against 96%, designs without a close contact 21% against 76%; it is no route to choosing the NCAA by decoding the latent. Consequence for the plan (`docs/ideas/ncaa_sequence_denoiser_plan.md`): the pipeline step that supplies the NCAA identity has to be changed before a free-choice generation test (and before the readout of trial S1 can be used in generation); the held-out readout tests of trial S1 (`gen_test/eval_readout.py`) hide the identity by hand and are not affected.

## 72. Trial S1: a sequence readout trained on non-canonical residues (20 Nov 2026)

![fig69](figures/fig69_readout.png)
Figure: `figures/fig69_readout.png` (script `figures/scripts/make_fig69_readout.py`). Trainer `denoiser/denoiser_train.py` (options `--train-readout --readout-attach --seq-w --nc-hide-p`), job files `denoiser/slurm/denoiser_s1_{mig_dbg,h200}.slurm`; Tillicum jobs 502169 (debug) and 502194 (S1); weights `denoiser_trial/saved/H200_S1/dn_s1_H200_502194/best` (box `~/rfd4_trial/out/S1_readout/best`); readout test `gen_test/eval_readout.py`, `run_readout.sh`, tables `gen_test/diag/readout_{control,G3_bal,S1_readout}_{real,ham}.json`; plan `docs/ideas/ncaa_sequence_denoiser_plan.md`.

Why: the denoiser's sequence readout (189 classes, a detached probe in the bundle) was never trained on NCAAs and in every earlier trial the identity of the NCAA tokens was an INPUT, so the readout loss never saw an NCAA (3 to 5% top-1 for every denoiser, section 46). The training log of S1 confirms the second point: counted before hiding, the pipeline gave the NCAA identity to the model for 100% of the NCAA tokens of the batches (identity given 88, 771, ... , hidden 0 at every check).
Method: start = G3 best, same class-balanced data (`train_all_cif_bal280`), the identity of every NCAA token hidden in training (`--nc-hide-p 1`), the readout head trainable and attached to the transformer, readout CE weight 3, batch 4, 5 inner updates, lr 5e-5, 600 steps, one full H200 (job 502194): 18 minutes, about 0.3 USD (estimate 1.35 USD at the cap). Training healthy: flow loss 50 to 80 per step (many single outlier batches of 1e4 to 1e17 were skipped, as in earlier trials), held-out score 64.63 against the G3 baseline 65.16.
Held-out check during training (82 NCAA tokens of the synthetic held-out complexes, identity hidden), step 0 -> 600: top-1 7.3 -> 9.8% (noisy: 2.4 to 9.8% in between), top-5 15.9 -> 31.7%, guesses that are a canonical class 43.9 -> 24.4%.
Test on real complexes (103 real + 365 extra real with an NCAA, 1,065 generated NCAA tokens per noise level, test only, identity hidden by hand, noise time t fixed for CA and latent; control / G3 / S1): top-1 at t = 0.9: 9.5 / 6.0 / 7.3%, t = 0.7: 13.0 / 4.3 / 6.0%, t = 0.5: 7.7 / 2.9 / 3.8%, t = 0.3: 2.4 / 1.2 / 1.9%; top-5 at t = 0.9: 26.4 / 22.1 / 25.5%; always guessing the most common code (ALY, 10% of the tokens) gives 10%. Share of guesses that are a canonical class at t = 0.9: 11.7 / 19.2 / 28.2%. At t = 0.9 by group (top-1 / top-5, control / G3 / S1): D-forms (336) 6.2/43.5, 7.1/46.1, 7.7/53.0%; general codes (254) 19.3/32.7, 15.0/26.0, 20.1/33.5%; other vocabulary codes (475, never targeted in training) 6.5/10.9, 0.4/2.9, 0.2/1.9%. The most frequent guesses of S1 are SAR, ILE, 1AC, MAA, DIL, DSG.
Findings: (1) NEGATIVE on real complexes: the trained readout does not name the real NCAAs better than the untuned control (top-1 7.3% against 9.5% at t = 0.9, below the 10% of always guessing ALY), and its guesses are canonical more often (28% against 12%); only the D-form top-5 rises (53% against 44%). (2) The gains of the held-out synthetic check (canonical guesses 44 -> 24%, top-5 16 -> 32%) do not transfer to the real complexes: the synthetic NCAAs are chosen by a generator rule, mostly the D-form or N-methyl variant of the residue that was there, the real ones are different (ALY, MLE, CGU, PTR, ...); the 'other vocabulary' codes were never in the training data and stay at about 0%. (3) What worked: the machinery (hiding the identity, training the 189-class readout with the loss attached to the trunk) runs and moves the synthetic held-out readout in the right direction in 18 minutes for 0.3 USD; what did not: 600 steps at weight 3 on the generator's distribution of NCAAs give no usable readout of real NCAAs. Not tested: whether attaching the readout changed structure quality (bonds, handedness, close contacts of generated designs; NOT re-tested), longer training, a larger readout weight, other data (real-like NCAA distributions), and the readout inside generation (the pipeline supplies the identity, section 71). Caveats: the synthetic held-out check has only 82 tokens; one training run.

## 73. Plan: let the denoiser choose NCAAs, at most 5 per design (20 Nov 2026)
Refined goal (user): the model must not turn every position into an NCAA; a design carries **at most 5 NCAAs**, and the model has to learn what separates an NCAA from the canonical residue at the same slot (a score gain), not the generator's rule. Visual plan with the example of the current problem (identity handed in, 100% copied; readout 7% vs 10% baseline, 28% canonical guesses): ![plan](figures/fig70_plan_choose_ncaa.png) (script `figures/scripts/make_fig70_plan_choose_ncaa.py`). Steps: A patch the pipeline (identity hidden, token type 'designable'); B paired data (same pocket and position, canonical vs NCAA variant, score gain = Rosetta interface dG (PyRosetta), NCAA variant minus canonical; Boltz-2/AF3 ipTM dropped (not an affinity measure)); C train the choice (readout picks canonical or NCAA, loss on the gain, cap of 5 by keeping the 5 slots with the biggest gain at generation); D check on real complexes (test only): gain over canonical, number of NCAAs <= 5, structure quality unchanged. Nothing of this is run yet.
Update (20 Nov 2026, Rosetta feasibility check): **PyRosetta 2025.44 is installed in WSL** (system python3, `~/.local`). Of the 132 non-canonical codes in the real test set plus common PTMs, **93 load as Rosetta residue types** (all D-forms exist under Rosetta's names: DPN=DPH, DCY=DCS, DSG=DAN, DSN=DSE; ALY, M3L, MLY, PTR, SEP, TPO, HYP, SAR, AIB, NLE and N-methyls are in); **39 exotic codes (A1xxx, KZ0, T09, ...) have no params** and would need `molfile_to_params`. On the 84 real complexes with a peptide chain: **62 load and score, 19 skipped (unsupported code), 3 fail to load** ("too many tries in fill_missing_atoms"). Unrelaxed total energy is 2-6 REU per residue for most (sane), 10-57 for some (relax first). `InterfaceAnalyzerMover` segfaults on these poses (the peptide with NCAAs is split into several chains/jumps), so the interface dG has to be computed by hand: translate the peptide away, repack the interface, take complex minus separated; paired NCAA-vs-canonical dG gains need several repack repeats because single values are noisy. Boltz-2/AF3 ipTM is dropped as the score (not an affinity measure). Scripts: scratchpad ros_prep.py (Windows, gemmi writes Rosetta PDBs) + ros_batch.py (WSL PyRosetta). Gotchas: gemmi and PyRosetta crash when loaded in the same process, so keep them in separate scripts.

## 74. Paired Rosetta interface dG, one NCAA position at a time (20 Nov 2026)

![fig71](figures/fig71_rosetta_paired_dg.png)
Figure: `figures/fig71_rosetta_paired_dg.png` (script `figures/scripts/make_fig71_rosetta_dg.py`). Code and README: `rosetta_score/` (`prep_pdb.py` Windows + gemmi, `paired_dg.py` PyRosetta in WSL, results `rosetta_score/real_paired_dg.json`, log `run_real.log`). Free (laptop, 28 cores, 330 s).

Method (user request: one position at a time): for every NCAA slot of the peptide (caps excluded) ONLY that residue is swapped to its L parent, to ALA, to GLY; every other residue, other NCAAs included, stays native. Each variant: repack the interface (extra rotamers), E(complex) minus E(peptide) minus E(receptor) with both parts repacked (dG, REF2015 units, lower = binds better), mean of 3 repeats; gain = dG(swap) minus dG(native), positive = the NCAA beats the canonical residue. An L parent on a D backbone clashes (flag gain > 20, 45% of D-form slots): excluded from the parent bars, ALA/GLY are the references for D-forms.
Results on the real test complexes (test only, unrelaxed): 95 complexes, 109 NCAA slots scored (20 complexes failed to load in Rosetta, 45 slots skipped: unknown parent or the residue cannot be built). Repeat noise: median sd 0.7 units, tail up to 26. Real NCAA better than ALA: D-forms 72% (n=65, median gain 3.3), other NCAAs 59% (n=44, 0.6); better than GLY 57% / 55%; better than the L parent only 30% (n=11 without backbone clash, median -0.3). Reading: the score has the right sign on average (a real NCAA beats ALA in most D-form slots) but the effect is small against the noise, the L-parent test for non-D NCAAs fails on a small sample, and unrelaxed crystal structures limit it. It can rank NCAA against canonical on average over many slots; a single slot is not reliable. Next: relax the complexes first, or use more repeats, before it is used as a training signal. Not tested: designed (generated) complexes, other canonical residues than ALA/GLY/parent.

## 75. Decoder journey figure (20 Nov 2026)

![fig72](figures/fig72_decoder_journey.png)
Figure: `figures/fig72_decoder_journey.png` (script `figures/scripts/make_fig72_decoder_journey.py`; no new run, numbers from sections 4, 49, 58 to 62, 65, 69, 70). Timeline of every decoder step (untouched, runs 1-2 residue tables, 189 classes, synthetic complexes, mirror-image D-peptides, blind run, atom loss, chirality loss, bond loss = run 8 default, clash loss = run 9 no gain) with (a) naming accuracy of real NCAAs, (b, c) rebuild of 1,066 real NCAA positions from true latents (RMSD 3.90 -> 0.65 A, within 1 A 0 -> 82%, native atoms 8 -> 97%, handedness 54 -> 89-92%), (d, e) generated designs per run (D handedness, sensible bonds, close contacts; the bond loss removed the handedness-versus-bonds trade-off), (f) denoiser + decoder error against noise.

## 76. How the synthetic data were made (figure, 20 Nov 2026)

![fig74](figures/fig74_synthetic_data.png)
Figure: `figures/fig74_synthetic_data.png` (script `figures/scripts/make_fig74_synthetic_data.py`; no new run, numbers from sections 1-2, 10, 28-34, 36, 40, 49, 52). Two synthetic sources: (A) free peptides (125,000 RDKit/RF3 sequences, repair of wrong-handed stereocentres, strict per-residue test, 46,379 pool peptides, plus 7,534 mirror-image D-peptides) and (B) NCAA residues swapped one at a time into natural complexes of the train split with NCFlow (4 draws per round, mirror repair, five gates, at most 8 swaps per complex, pocket-crop CIFs; pilot: 1,852 of 2,110 planned swaps accepted; pool 8,947 complexes, 22,289 NCAA positions, 31 codes; balanced subset 2,010 complexes). The figure also lists the limits (narrow chemistry, natural pose, geometric gates only, D-form skew, identity chosen by the generator's rule).


Example structure (Fig 80, `figures/fig80_synthetic_example.png`): one complex of the balanced subset, 3RSF (peptide ALA-ALA-[D-TRP]-LEU-PHE-GLU-ALA), residue 3 swapped from L-Trp to D-Trp by NCFlow (mirrored; 96% of the residue contacts and 100% of the polar contacts kept). Rendered with PyMOL (`figures/scripts/render_fig80_synthetic_example.py`, pymol-oss python).

## 77. Paired Rosetta dG after relaxing the complexes (20 Nov 2026)

![fig73](figures/fig73_rosetta_paired_dg_relaxed.png)
Figure: `figures/fig73_rosetta_paired_dg_relaxed.png` (script `figures/scripts/make_fig71_rosetta_dg.py real_paired_dg_relaxed.json ...`); results `rosetta_score/real_paired_dg_relaxed.json`, log `run_real_relaxed.log`, relaxed PDBs `rosetta_score/ros_in_relaxed/`. Same one-position-at-a-time protocol as section 74, but every complex is first relaxed once (FastRelax, 3 repeats, constrained to the start coordinates) and every slot is scored on the relaxed pose with 5 repeats; 95 complexes in 1,037 s on 28 cores (free).
Results (109 slots, same as section 74; 20 complexes still fail to load in Rosetta, 45 slots skipped): relax lowers the median complex energy from +1,206 to -200 REU, all 75 relaxes that ran finished (the 20 complexes that do not load are not relaxed). Repeat noise falls from a median sd of 0.68 (unrelaxed, 3 repeats) to 0.35 (relaxed, 5 repeats; 90th percentile 1.85, a few slots up to 48). Real NCAA better than ALA: D-forms 77% (n=65, median gain 5.6), other NCAAs 70% (n=44, 0.9); better than GLY 82% / 80% (median 3.1 / 2.6); better than the L parent 55% (n=11 without backbone clash, median +0.05). Before relaxing: 72% / 59% (ALA), 57% / 55% (GLY), 30% (L parent).
Reading: relaxing strengthened the signal: the real NCAA now beats the canonical residue in 70 to 82% of slots against ALA and GLY, with gains larger than the repeat noise for D-forms; against the L parent of the non-D NCAAs it is a coin flip (11 slots). The score therefore separates "NCAA" from "small canonical residue", which is a usable but modest signal for ranking; it does not yet say whether an NCAA is better than the best canonical residue. Caveats: the relax moved the C-alpha atoms by a median 1.9 A (maximum 12 A, so the restraint was loose for some complexes; relaxing the native also relaxes the NCAA into its own pocket while the swapped variants are only repacked, which favours the native: a possible bias towards gain > 0); 20 of 95 complexes and 45 of 154 slots are missing (N-alkyl residues and loading failures); REF2015 units, not kcal/mol. Next check before using it as a training signal: relax both variants with the same protocol, and score designs, not only natural complexes.

## 78. How each denoiser was obtained (lineage figure, 20 Nov 2026)

![fig75](figures/fig75_denoiser_lineage.png)
Figure: `figures/fig75_denoiser_lineage.png` (script `figures/scripts/make_fig75_denoiser_lineage.py`; no new run, numbers from sections 43-46, 50, 53, 55, 66, 68, 72 and `denoiser/PRESETS.md`). Family tree of the LoRA trials: control; 1 basics (A, B, C: rank 16 on attention); 2 more capacity (D rank 64, E + feed-forward: worse); 3 more data (F, F2: gain only on synthetic); 4 change the loss (G: 10x weight on NCAA positions, capped noise weight, lr decay: best so far); 5 bigger / balanced data (G2 on the 8,947 pool, G3 on the class-balanced subset = default); 6 extra signals (G4 clash loss, S1 readout: no gain; H skipped). Bars: held-out synthetic change, real new-D-form change, generation CA distance.

## 79. Paired Rosetta dG with MATCHED relax: the earlier signal does not hold (20 Nov 2026)

![fig76](figures/fig76_rosetta_paired_dg_matched.png)
Figure: `figures/fig76_rosetta_paired_dg_matched.png` (same script, `real_paired_dg_matched.json`); code `rosetta_score/paired_dg_matched.py`, log `run_real_matched.log` (4,057 s, 28 cores, free). Purpose: remove the bias noted in section 77 (the native complex had been relaxed into its pocket while the swapped variants were only repacked). Protocol: for each of the 109 slots and for each variant (native, L parent, ALA, GLY; one position changed) start from the ORIGINAL complex, apply the swap, FastRelax (3 repeats, backbone and chi free only for the interface residues, whole complex restrained to the start coordinates), then dG (interface repack, 3 repacks), done 2 independent times per variant; gain = dG(swap) minus dG(native).
Results: all 109 slots ran. The relax drift is smaller (median C-alpha shift 1.0 A, maximum 2.2 A) but the repeat noise is large: median sd of one slot 7.5 units (0.35 in section 77), 90th percentile 147, maximum 871; only 7% of slots have sd below 1, 23% below 3, 54% below 10. Real NCAA better than ALA: D-forms 60% (n=65, median gain 2.5), other NCAAs 52% (n=44, 0.1); better than GLY 58% / 45%; better than the L parent 38%. On the 36 slots with sd below 10 (27 D-form, 9 other): D-forms beat ALA 63% and GLY 74%, others 67% / 56%. The ALA gains of section 77 and of this run are uncorrelated (r = -0.02, n = 109).
Reading: with the bias removed the validity signal drops to 45-60% (coin flip level for the non-D NCAAs), and independent relaxes of the same structure differ by several to hundreds of REU, so a gain of a few units cannot be told from relax noise. The 70-82% of section 77 was therefore mostly the bias plus noise, not evidence that Rosetta can tell an NCAA from a canonical residue. Not a usable training signal in this form. Next options (not run): (1) relax the complex once, then give all variants an identical local protocol (swap, repack, side-chain minimisation, backbone fixed; no independent global relax); (2) many more relaxes and use the median; (3) a more stable score (per-residue interaction energy of the slot with the receptor instead of the full interface dG); (4) first test the score on a case with a known answer (for example known hot-spot substitutions) before using it. Caveats: 2 relaxes per variant; relaxes with the NCAA types in REF2015 have known weaknesses; 20 of 95 complexes could not be loaded, so this is the same 109-slot subset as section 74.

## 80. Light paired Rosetta score: relax once, identical local protocol for every variant (20 Nov 2026)

![fig77](figures/fig77_rosetta_light.png)
Figure: `figures/fig77_rosetta_light.png` (script `figures/scripts/make_fig77_rosetta_light.py`); code `rosetta_score/paired_dg_light.py`, results `rosetta_score/real_paired_light.json`, log `run_real_light.log` (398 s, 28 cores, free). Purpose: remove the relax noise that made section 79 unusable and keep the one-position-at-a-time rule.
Protocol: start from the globally relaxed native (`ros_in_relaxed/`, section 77); for the native residue and each swap (L parent, ALA, GLY; ONLY the slot differs) apply exactly the same local steps: swap, repack the residues within 10 A of the slot (extra rotamers), side-chain minimisation, backbone fixed; 12 repacks per variant, two independent halves, each half scored as the mean of its 3 lowest-energy repacks. Scores from the energy graph (REF2015 two-body energies): E_slot = interaction of the slot with the receptor, E_int = whole peptide-receptor interaction; gain = E(swap) minus E(native), positive = the native residue is better. Control with a known answer: 160 canonical interface residues of the same peptides against ALA (an alanine scan).
Results (109 NCAA slots in 65 complexes, 2 complexes failed): repeat spread between the halves is 0.00 units (median; maximum 11.9), against 7.5 for the matched relax of section 79, so the score is stable. Native residue better than ALA: D-forms 91% (E_slot) / 94% (E_int), n=65, median gain 5.4 / 5.7; other NCAAs 86% / 95%, n=44, median 1.1 / 1.3; against GLY D-forms 92% / 91%, others 91% / 95%. The canonical control (alanine scan) 93% / 93% with median gain 4.4. Other NCAAs against their L parent: native better in 55% of 11 slots, median 0.0. Per code (median E_slot gain over ALA): DTR 9.9, DPH 7.6, DTY 7.2, DAR 6.5, DCS 4.9, DAL 2.6, DTH 1.7, AIB 1.5, HYP 0.0, DVA -0.1, with the canonical residues in the same order of side-chain size (TRP 8.9, TYR 7.4, PHE 6.6 ... SER 0.4).
Reading: the score is stable and passes the known-answer check (the canonical residues beat ALA in 93% of slots, as an alanine scan should), but the NCAAs behave like canonical residues of the same size: the gain follows the side-chain size (large aromatic or charged residues high, small ones near zero) and the real NCAAs are not better than the canonical control (median 2.2 for all NCAA slots against 4.4 for the control). It therefore measures "how much does this side chain contact the receptor" and cannot say that an NCAA is better than the best canonical residue. Use: a quiet, cheap score for ranking variants at one slot (for example two candidate NCAAs against each other, or a design against the same design with a different residue), not a proof of NCAA benefit. Caveats: the backbone is fixed and relaxed with the native residue, so the native keeps an advantage (the control shows the size of this baseline); D-form outliers up to 120 units probably come from clashes of the ALA/GLY swap with the D backbone; the L-parent comparison has only 11 non-D slots; no test on designs yet; REF2015 units, not kcal/mol. Not done: ranking two NCAAs at the same slot against known binding data.

## 81. Identity step patched: free choice of the non-canonical residue (20 Nov 2026)

![fig78](figures/fig78_free_choice.png)
Figure: `figures/fig78_free_choice.png` (script `figures/scripts/make_fig78_free_choice.py`). Code: `gen_test/probe_identity.py` + `run_probe_identity.sh` (probe), `patch_hide_nc.py` (patch), `run_free_choice.sh` (generation), `pull_and_analyze_choice.sh` / `analyze_choice_only.sh`, `analyze_free.py` (made robust to residues without C-alpha); results `gen_test/diag/free_choice.json`, designs `gen_test/out_local/*_choice_run8`; memory note `identity-leak-and-patch`.
Cause of section 71 (found by the probe, box GPU 0): the library's `rfproteina/datapipes/pipeline.py` (the file the generator imports) treats NCAA tokens (token type 6) as sequence-conditioned by default: the C_SEQ feature and the `element` feature keep the input residue even when the spec has no C_SEQ condition for them (probe on 7PIU_P: C_SEQ = DAL and DPN at the denoiser input; canonical peptide tokens were correctly `<M>`). The canonical positions were always free, the NCAA positions never were.
Patch (box venv file, backup kept, default behaviour unchanged): with env `HIDE_NC_IDENTITY_CHAIN_PREFIX=P` the NCAA tokens of the peptide chain get C_SEQ = `<M>` and no element feature; the token type stays 6, so only NCAA classes are allowed. The identity then comes from the denoiser's own sequence readout (`condition_decoder_on_sequence_prediction` is on by default) restricted to NCAA classes, and decoder run 8 builds the atoms. Probe after the patch: C_SEQ = `<M>` at the NCAA tokens, readout picks SAR and SAR (native DAL, DPN).
Test (box, free): 11 real pockets x 8 designs per denoiser, no identity in the spec, decoder run 8; denoisers control, G3, S1 (readout trained on hidden NCAAs, section 72). 176 NCAA slots per run, 152 of them D-forms in the native peptide.
Results (control / G3 / S1): exact native code 4.0 / 0.6 / 2.3% (always-DAL baseline 32%; before the patch 100%); a D-form chosen where the native is a D-form 20 / 20 / 41% (chance: 19 of the 142 NCAA classes are D-forms = 13%); native parent amino acid of the D-form slots matched 0 / 0 / 0%; distinct codes 17 / 14 / 14, commonest code SAR 36% / MAA 22% / DIL 24% (the three models pick mostly SAR, MAA, DIL, 4FB); canonical free positions recovered 8.6 / 10.8 / 10.3% (unchanged, chance 5%); designs: sensible bonds 87 / 87 / 84%, close contact 17 / 19 / 24%, D-form handedness 72 / 82 / 91% (against the pick, not the native).
Reading: this is the first valid free-choice test. With the identity really hidden none of the models picks the native NCAA (0.6 to 4%, far below the always-DAL baseline), and all collapse to a handful of codes. The one measurable effect of S1 (readout trained on hidden NCAAs) is on the class: it picks a D-form at the D-form slots in 41% of cases against 20% for control and G3 (chance 13%) and gives the highest D handedness (91%); it does not pick the native parent amino acid. So the readout training moved the model toward the right kind of NCAA, not toward the right NCAA, in line with section 72 (offline: no transfer to real codes). Caveats: 11 pockets, 88 designs per run, one seed; only decoder run 8; the identity enters only through the readout argmax (no sampling over the readout distribution); the patch is on the box only (Tillicum untouched); the exact-code test is harsh (several NCAAs may fit a slot), a class- or contact-level score and the Rosetta light score (section 80) are the next checks.

## 82. Scoring the free-choice designs: family, contacts and the light Rosetta score (20 Nov 2026)

![fig79](figures/fig79_choice_scores.png)
Figure: `figures/fig79_choice_scores.png` (script `figures/scripts/make_fig79_choice_scores.py`). Code: `gen_test/score_choice_designs.py` (family and contacts, `diag/choice_scores.json`), `rosetta_score/prep_designs.py` + `paired_score_designs.py` (light Rosetta score of the designs and of the native pockets; results `real_design_scores.json`, log `run_design_scores.log`, 1,463 s on 28 cores, free). Designs: the three free-choice runs of section 81 (control, G3, S1; 88 designs each); native = the native pockets of `gen_test/inputs`.
Family of the chosen NCAA against the native family at the same slot (native: 86% D-alpha, 14% N-alkyl): agreement control 23% (chance 26%), G3 23% (25%), S1 41% (42%); designed families control 62% N-alkyl / 20% D-alpha / 12% PTM, G3 45% N-alkyl / 31% side-chain modified / 22% D-alpha, S1 44% D-alpha / 32% N-alkyl / 18% side-chain modified: S1 shifts the overall frequency toward D-forms, but the choice is not slot-specific (agreement at chance level).
Contacts of the NCAA residue with the receptor (heavy-atom pairs < 4.5 A): median native 8, control 8, G3 21, S1 18; slots with an atom pair < 2 A: native 0%, control 2%, G3 3%, S1 6%.
Light Rosetta score (relax once, identical local protocol, one position at a time): 266 of 277 structures scored. E_slot of the NCAA slot (median; lower = better): native NCAAs -5.7, control -1.0, G3 -1.6, S1 -3.1, canonical interface residues -7.1: the chosen NCAAs interact much less than the native NCAAs and the canonical residues (the choices are mostly small SAR / MAA residues). Rosetta can swap only some codes: gain over ALA covers 19 of 21 native slots and 57 / 35 / 75 of 153 / 144 / 133 designed slots (D-forms, SEP, TPO; the N-methylated SAR, MAA and 4FB, 04Q, 23F, 1AC, XCP cannot be swapped). On the swappable slots the gain over ALA is native 5.3 (74% > 0), control 1.5 (75%), G3 1.9 (66%), S1 3.0 (88%), against 3.8 to 4.8 (88-97%) for canonical interface residues. E_int of the whole peptide with the receptor per structure (median): native -41, control -37, G3 -40, S1 -47 REU.
Reading: with the identity hidden the models choose NCAAs that are not better matched to the native family than chance, that interact weaker with the receptor than the native NCAAs (E_slot) although they touch it more (18 to 21 contacts against 8), and S1 is the best of the three (D-form share, E_slot, E_int) but still below the native NCAAs on E_slot. The score separates "small NCAA with few interactions" from "residue that makes interactions", a useful signal for training (the first one the model could learn from), but it still does not say which NCAA is the best choice. Caveats: the swap-based gain covers only 25 to 55% of the designed slots and is biased towards D-forms; one seed, 88 designs per run; designed peptides differ in length from the native, so E_int is only a rough comparison; the native pockets are cropped pocket files (11 structures); 11 structures failed to load; the light protocol caveats of section 80 apply.

## 83. AF3 ipTM as a plausibility filter (21 Nov 2026)

Question (user): use AF3 as a filter layer, with ipTM as the measure of how likely a complex is (not as an affinity score). Setup (`af3_filter/`): 310 AF3 runs on the 2-GPU box (GPU 0, single-sequence, 2 diffusion samples, 10 recycles, ~23 s per job), 10 GPCR-like pockets (the three fibril entries 9B4J/9B4K/9B4L have no real receptor and were skipped). Receptor = the full chain that touches the native peptide (`Dataset_ncaa/real_peptides`), peptide = chain B with the NCAAs as CCD modifications. Groups per pocket: native peptide; native with NCAAs set to ALA; native with 2 random-NCAA versions; 2 shuffled versions; native peptide on the receptor of another pocket; and the free-choice designs of Results.MD 81 (control, G3, S1: 80 each). Scripts: `build_jobs.py`, `run_af3_iptm.sh`, `collect_af3.py`, `analyze_af3.py`; figure `figures/fig81_af3_iptm_filter.png` (`scripts/make_fig81_af3_iptm.py`).

Two setup findings. (1) Without a receptor template AF3 misfolds the receptor (about 8 A off) and ipTM does not discriminate (native peptide 0.28, other receptor 0.33, designs 0.2-0.3): not usable. (2) With the native receptor given as a template (the template mmCIF needs a release date) native peptides reach 0.88; all numbers below are from this mode.

Results. Mean ipTM: native 0.88, native NCAA->ALA 0.71, native + random NCAAs 0.73, other receptor 0.66, shuffled 0.57, designs control 0.64, G3 0.64, S1 0.67. AUC native against shuffled 0.995, random NCAAs 0.78, ALA 0.76, other receptor 0.76. Designs reach ipTM >= 0.7 in 28% (control), 30% (G3) and 40% (S1) of the runs (native 100%, shuffled 15%). Designs are 0.21-0.25 below native and 0.07-0.10 above shuffled; they are 0.06-0.09 below native + random NCAAs, which keeps the native context and is not like-for-like. AF3 puts the peptide 4.3-5.0 A (median) from the designed pose; the native peptide is 2.1 A from its own pose.

Within-pocket agreement of design ipTM with the other scores (Spearman, pockets demeaned): AF3-to-design pose RMSD -0.26 (p < 0.001), Rosetta slot energy -0.19 (p = 0.003), whole-peptide E_int -0.08 (n.s.), receptor contacts 0.04 (n.s.); peptide pLDDT 0.85 (ipTM is mostly peptide confidence). Filter test (keep the top 25% / 50% by ipTM inside each pocket against 2,000 random keeps of the same size): slot energy -3.7 / -3.1 against -0.9 for all designs (at the edge of the random range), pose RMSD 4.7 / 4.6 against 5.4 A (edge of the range), E_int -30.6 / -36.6 against -35.9 (no gain), contacts 22.8 / 21.5 against 21.2 (no gain). S1 is over-represented in the kept set (47% of the top 25% against 33%).

Reading. ipTM with a receptor template is a good coarse gate for "a peptide-shaped sequence that fits this pocket" (it rejects scrambled peptides) but it is saturated for anything that sits in the pocket: it cannot tell ALA or random NCAAs from the real ones (0.76-0.78), so it does not measure whether the NCAA is a good choice. Among our designs it is a weak ranker. A useful gate would combine ipTM >= 0.7 with an AF3 pose within 3 A of the designed one: 12-17% of the designs pass both (native 70%, shuffled 0%).

Caveats: 10 pockets (about 5 sit in one GPCR family, so the "other receptor" negative is often a related pocket and weaker than a true wrong receptor), one seed, two diffusion samples, AF3 may have seen some native complexes in training (native ipTM may be inflated), the random-NCAA control keeps the native residues, N-methyl and other uncommon NCAAs are modelled by AF3 as given CCD ligands-in-chain with no check of how well it knows them.

## 84. Readiness of AF3 for D-amino acids and canonical-like NCAAs (21 Nov 2026)

Question (user): is AF3 ready for D-amino acids and for NCAAs with a backbone like the canonical one? Test on the 310 AF3 models of section 83 (best-ranked sample of each; 2,573 peptide residues, 42 CCD codes; scripts `af3_filter/ref_sign.py`, `readiness_af3.py`; figure `figures/fig82_af3_readiness.png`, `scripts/make_fig82_af3_readiness.py`). Per residue: (1) hand of the C-alpha (sign of (N-CA) x (C-CA) . (CB-CA)) against the sign of the chemical definition of the code (RDKit mol of the Boltz cache; check: the experimental D-residues of the test pockets match it 100%: DAL 7/7, DIL 4/4, DPN 3/3, DTR 5/5, DVA 3/3, DAR 1/1); (2) peptide bond to the next residue present and 1.20-1.50 A; (3) pLDDT of the residue against the canonical residues of the same peptide. AF3 input as in the NCAA structure dataset: sequence letter plus a CCD modification at the position.

Results. (a) AF3 accepts every code (no job rejected) and bonds the residues into the chain. (b) Hand: canonical L 100% (2,138 residues), phosphorylated SEP/TPO 24 of 24, N-methyl-Leu (MLE) 4 of 4, but D-alpha residues have the right hand in only **17%** (167 residues): DIL 9% (7 of 80), DHI 0 of 13, DTY 2 of 15, DAL 3 of 14, DPR 2 of 7, DAS 3 of 7, DPN 5 of 11, DSG 3 of 5, DAR 0 of 4; N-methyl-Ala (MAA) 6 of 11; 23F 5 of 23 (reference sign from the code definition). In the native-group models the D-residues of the real peptides are also wrong (DAL 2 of 7, DTR 0 of 2). (c) Peptide bonds are intact for 98% of the canonical residues, 95% of the D-forms, 92% of the N-alkyl and PTM residues (MAA 73%, 04Q 88%, SEP 89%). (d) pLDDT of D-form residues is 4.6 above the canonical residues of the same peptide (side-chain modified +6.5, N-alkyl +2.7): AF3 is confident about the wrong hand.

Reading. AF3 is ready for codes in the sense of accepting them and for L-like and phosphorylated residues, but it does not enforce the hand of D-residues given as CCD modifications (it is close to a coin flip at best and mostly wrong), so an AF3 ipTM or pose for a peptide with D-forms is computed for a molecule with the wrong chirality at most D positions. This matches the handedness problem of RF3/Boltz (the mirror repair of Results.MD 40ff). For the ipTM filter of section 83 this means: apply it only after a chirality check, or fold the mirror-image problem (L-peptide on the mirror target) where AF3 is reliable. Caveats: best-ranked sample only (the second diffusion sample was not checked); 10 pockets; reference sign missing for 04Q, 4FB, GLY, SAR; codes with few residues (DAR, DGL, DVA, DSG) are anecdotal; a different encoding (D-residue as a bonded ligand, or a different sequence letter) was not tried.

Addendum to 84 (21 Nov 2026): second samples and another input encoding (`figures/fig83_af3_encoding.png`, `figures/scripts/make_fig83_af3_encoding.py`, jobs `af3_filter/jobs_lig`). (1) The second diffusion sample is no better than the first (D-residues right 16% and 21%); only 12% of the best-ranked models with D-forms have all their D-residues right (native 22%, control 9%, G3 7%, S1 12%), so drawing more samples does not help. (2) A mirror-image formulation does not help either: the mirror of a D-peptide on an L receptor is an L-peptide on a D receptor, which moves the problem to the receptor. (3) Encoding the peptide as one bonded ligand of CCD residues (C-N bonds) instead of modifications in a protein chain: 30 jobs (10 natives, 20 S1 designs with D-forms). D-residue hand 61% in the natives (9 jobs, was 33%) but 0% in the 20 designs (12 of their 21 D-residues are D-isoleucine); the hand of canonical L residues falls to 91% and intact peptide bonds to 80% (from 100% and 98%). The ligand encoding is not a fix. Small pilot, one seed.
