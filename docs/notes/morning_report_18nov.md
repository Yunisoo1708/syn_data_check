# Overnight report, 18 Nov 2026 (second night)

## What ran
- **Trial G (denoiser, box):** trial C's recipe + 10x loss weight on non-canonical positions + noise-level weight capped at 100 + learning rate halved twice on plateau (`ncaa_fine_tuning_v3.py`). Stopped by its 3-hour cap at step 2,700 while still improving (held-out check -0.96 vs the untuned 53.93). Results section 50.
- **Decoder run 3 (Tillicum slice, 1 h 14 min, about 0.16 USD):** run 2's settings + 7,534 mirror-image D-peptides from the train split of the pool (`mirror_pool.py`, `decoder_fine_tune_v5.py`); real-pose test on 2,750 positions (job 493899). Results section 49.
- **Same-decoder tests:** denoise-then-decode for the control and all nine denoisers with the run 3 decoder; trial G with the run 2 decoder. Section 51.
- **Top-up data (rare D-forms + the seven low general codes):** laptop, box 1660 Super, Tillicum slices (shards 0-15, about 2.1 USD), Klone checkpoint array. Training pool now 8,006 complexes (was 6,613). No new training on it tonight (as instructed).
- The extended real benchmark (987 real complexes) and the non-canonical-only scores were done earlier (sections 46-48).

## Results
1. **Decoder run 3: IMPROVED by the decision rule, strongly.** Real D-forms 72/112 -> 110/112 (64% -> 98%); the weak six (DHI DIL DTY DAS DGN DSG) 7/34 -> 34/34; overall 97.3% -> 98.7%; canonical unchanged (98.6%). Cost: other-vocabulary non-canonical 48/48 -> 45/48. The failure was the D side chain on an L backbone in the synthetic complexes; mirror images give real D backbone geometry.
2. **Trial G is the best denoiser on the synthetic held-out set:** -0.68 vs the control (interval -0.78 to -0.59), 0.22 better than C (-0.27 to -0.16); best D-form scores on both test sets (non-canonical latent loss: held-out D-forms -16.7%, real D-forms -7.6%, and -11.4% on the 987 real complexes); real all-position score equal to the control within the interval; real general codes about equal to A and C (-5.5% vs -6.9/-6.7%), i.e. not improved.
3. **With the run 3 decoder the control already decodes D-forms at 91-98%** (was 50-70%); the denoiser fine-tunes then add only 0-2 points (held-out) and up to +5 (extra real set). The decoder was the main bottleneck for D-form identity.
4. **Top-up status** (accepted swaps per code; goals 1,250 general / 300 D-forms): AIB 992, MAA 996, HYP 1,026, FP9 1,013, SAR 1,021, 1AC 1,030, 4FB 1,028 (general); D-forms added by the top-up so far: DCY 259, MED 237, DLY 165, DTH 140, DSN 128, DTR 115, DGL 105 (on top of the earlier 12-159 each). Klone is at 52 of 96 shards; laptop 7 of 12 shards, box 4 of 8; their later shards are NOT yet pulled or converted.

## Decisions taken without asking
- Cancelled the unstarted Tillicum slice shards (16-23) as agreed; moved their share to Klone's list.
- Chose mirror-image augmentation for the decoder (instead of more synthetic D-form complexes) and used the same validation/test as run 2; trial G uses C's data (1,266 complexes) to stay comparable.
- Fixed the latent time (t = 0.05 / 0.15 / 0.3) in the decode test because at moderate noise the decoder already read the identities perfectly.
- Tillicum spend tonight stayed under the 4 USD cap (about 2.4 USD: slices about 2.1, decoder 0.16, tests about 0.1).

## Problems and caveats
- Mirrored CIF files: the first two versions (gemmi rewrite) did not keep the pool files' layout; the third (text transform) matches the original line for line.
- The decode test gives the identity of non-canonical residues to the denoiser (pipeline conditioning), so its accuracy is high for every model; only the D-form vs L-form part is informative. A version that withholds the identity is not done.
- The run 3 decoder was chosen by the best held-out synthetic loss (epoch 1); synthetic validation is saturated, so the real-pose test is the only informative check. The no-leakage check of the pool against the real benchmark sequences was done for runs 1 and 2 and not repeated for the mirrored set (same sequences, opposite handedness).
- Trial G was cut by the 3-hour cap while still improving: longer training may help.

## Next steps (proposals)
1. Continue trial G from its checkpoint (`--resume`), and try G's recipe on the larger pool (8,006 complexes) with the new D-form data.
2. Use the run 3 decoder in a generation test (design a binder around a fixed non-canonical monomer, check with a structure predictor): the first test of design quality.
3. Run the decode test with the non-canonical identity withheld from the denoiser.
4. Pull and convert the remaining top-up shards (Klone, laptop, box) only if the general-code gap on real structures still needs data; the benchmarks suggest more of the same synthetic data helps little.
5. Decide the data-policy question (use a family-split part of the real non-canonical complexes for training) if real general codes stay behind.

## Where things are
Weights: `denoiser_trial/saved/box_out/{A_all,B_first,C_long,D_rank64,E_rank64_ff,F_pool2,F2_H200,G_ncw10}/best`, `denoiser_trial/saved/H200_F*/`; decoders `decoder_v4_488524.pt` and `decoder_v5_493747.pt` (run 3, 513 MB, also on the box). Figures: fig38 (held-out, with G), fig39/fig41/fig43 (real, non-canonical), fig44/fig46 (decode), fig45 (decoder run 3), fig40 (training data). Results.MD sections 43-51.
