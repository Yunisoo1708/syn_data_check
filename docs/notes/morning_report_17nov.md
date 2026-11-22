# Report: synthetic non-canonical data and denoiser fine-tuning (16 to 18 Nov 2026)

## What ran
- **Decoder (whole decoder, 189 classes):** run 2 on 1,429 synthetic complexes plus the 46,379-peptide pool (Tillicum job 488524, MIG slice, 57 min). Not "improved" by the decision rule (weak six D-forms 10 of 34 on real poses); Results.MD section 41.
- **Data:** split v3 (+1,453 entries) and v4 (+870 test entries), frozen; the 480 close entries are left out of every split. Synthetic pool `train_all_cif_clean2` now has **6,613 complexes** (2,458 PDB entries, 735 protein families): first batch 261, top-ups 1,650, balanced general batch about 4,700. Accepted swaps per non-D code over all converted general shards (4,702 complexes): 23F 1,360, 04Q 1,178, 2ML 1,144, MLE 1,077, MVA 957, 1AC 898, 4FB 898, SAR 889, FP9 878, HYP 872, MAA 846, AIB 840 (target 1,236 each; spread still 1.6x, the lower seven need a top-up). Diversity: `figures/fig40_training_data_diversity.png`. NCFlow generates the backbone too, within 1 A CA drift of the natural pose; only four chemical families are buildable (section 40).
- **Denoiser (flow model, LoRA) trials A to F2:** see section 43, 43a, 43b, 44, 45, 46. Best weights saved under `denoiser_trial/saved/`; resume notes in the memory note `denoiser-lora-weights-and-benchmark`.

## Main results (section 46, `figures/fig41_benchmark_all_models.png`, `fig42_noncanonical_by_group.png`)
- Every fine-tuned version lowers the latent loss at **non-canonical positions** by 4 to 8% (held-out D-forms 9 to 15%); the all-position score moves only about 1% because non-canonical residues are about 2% of each complex.
- Best on synthetic held-out: F2 (-0.54), then C (-0.47). Best on real complexes: the small rank-16 attention fine-tunes A, B, C (real complexes with a new D-form about -1.0; non-canonical positions -6 to -7%).
- The larger pool (F, F2) improves synthetic general codes (-16 to -22%) but not real ones (-0.2 to +2.4% against -6 to -8% for A, C, D). Rank 64 (D) and feed-forward adapters (E) are worse on real structures; E is also worse on synthetic.
- Cost: every version loses 0.5 to 1.3% at canonical positions on real complexes.
- Not tested: generated structures and design quality; the residue-type readout was never trained (accuracy 3 to 5% for all models).

## Decisions made without asking
- Held-out evaluation seeded (deterministic) after the first trials showed a drifting baseline; trials A, B, C compared again on all 163 held-out complexes.
- Full H200 run: batch 8 ran out of memory (117 GB at step 80), so batch 4; it peaked at 115 of 144 GB and ran at 97 to 100% utilisation. The learning-rate continuation F2 (5e-5, from the best weights) was queued behind it.
- Box used for scoring only after the training was stopped on request.
- Extra checks added: real-structure test (103 real complexes, test only), non-canonical-only score with out-of-vocabulary residues masked.

## Costs
Tillicum: H200 jobs 491259 (6.5 min, crashed out of memory), 493150 (50 min), 493216 (26 min) = about 1.3 USD; decoder slice job 488524 about 0.1 USD. Total budget used so far about 112 of 250 USD (about 138 left).

## Problems
- Login to Tillicum lapsed during the day (needs the user's Duo); fixed by the user.
- First trial E launch failed on a module name (`swish_linear` is a container; the linear layer is `swish_linear.0`).
- Many early batches (about 10% at batch 4) are skipped by the safeguard because the loss weight explodes near t = 1.
- The scorer on the slow second box GPU stalled; moved to the fast GPU.

## Next steps (proposals)
1. Cap the loss weight (limit t to 0.99) and test it as a separate trial; add "reload best weights and halve the learning rate" to the training script.
2. Top up the seven under-represented general codes (AIB, MAA, HYP, FP9, SAR, 1AC, 4FB) and the rare D-forms (DTR, MED, DCY, DLY, DTH).
3. Train the sequence readout (or the decoder path) so residue identity improves at generation time; the current LoRA does not touch it.
4. Real-structure generation test (design quality), the step that has not run on this data yet.
