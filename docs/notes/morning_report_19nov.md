# Overnight report, 19 Nov 2026

## What ran
- **Generation test with real pockets** (Results.MD sections 59 to 61): 11 real peptide-protein complexes, receptor pocket fixed, hotspots at the native contact residues, the non-canonical residues of the peptide given at their positions, the rest designed; 88 designs per run, raw outputs.
- **Blind decode test** (section 57 to 58): without the given residue identity the run 3 decoder reads D-forms at about 38% and general codes at 0 to 4% even from true latents; the blind-trained decoder (run 4) reads general codes at 86 to 90% and D-forms at 53 to 61% on real structures, but no denoiser's own latents carry the identity (0 to 12%).
- **Diagnostic** (59c): decoding the TRUE latents of real complexes put D-form side chains on the right face only about half of the time; the NCFlow D-residues in the training pool are 100% right at the C-alpha, so the data were fine; decoder runs 1 to 4 had trained only the residue-type loss, never the atom positions.
- **Decoder runs 5, 5b, 6, 7** (sections 60, 61, fig57, fig58): atom-coordinate loss (plus a handedness loss in runs 6 and 7) on the 46k peptide pool + 7,534 mirror D-peptides + the synthetic complexes (pool now 11,291 complexes: the 2,178 new D-form complexes from the Klone balance run were merged first).
- Pool and data: Klone balance run 95 of 96 shards converted (pool 11,291; all main D-forms >= 719 positions); mirror-image complexes built for the denoiser (3,500, handedness verified) and staged with a balanced set on Tillicum (`train_all_cif_H`, `denoiser_H_h200.slurm`) but NOT trained.

## Results (numbers)
- D-forms with the right handedness in generation (G / G2 / G3): decoder run 3 35 / 39 / 42% -> run 5 95 / 83 / 90% -> **run 5b 96 / 94 / 96%** -> run 6 98 / 99 / 99% -> run 7 98 / 97 / 99%.
- Peptide bonds of sensible length: 23 / 22 / 22% -> 67 / 66 / 68% -> **67 / 74 / 71%** -> 26 / 22 / 21% -> 51 / 50 / 54%. Designs with an atom pair closer than 2 A: 62 / 65 / 72% -> 34 / 19 / 23% -> 33 / 17 / 23% -> 38 / 22 / 33% -> 35 / 19 / 26%.
- Decoder alone on true latents of the real 103: D-forms right 51 -> 83 -> 95 -> 98 -> 99%; atom RMSD of D-forms 3.92 -> 0.81 -> 0.56 -> 0.67 -> 0.53 A; general codes 5.38 -> 0.76 -> 0.67 -> 0.75 -> 0.76; canonical 0.52 -> 0.19 -> 0.18 -> 0.40 -> 0.21.
- The given non-canonical residues are kept in 100% of designs; the untuned control reaches only 67% handedness with decoder run 5, the fine-tuned G 95%, G3 90%: the denoiser fine-tunes matter once the decoder is fixed.

## Decisions taken without asking
- Default decoder = **run 5b** (`decoder_v7b_498314.pt`: best balance of handedness and bond geometry); run 7 (`decoder_v8_499080.pt`) is the handedness-first alternative; run 6 not used (canonical atoms got worse).
- Denoiser trial H (mirror complexes) skipped: the mean handedness of G, G2, G3 with decoder run 5 was 89% (rule: run it only below 70%).
- One epoch per decoder job: the Tillicum host-memory error in the data loader appears when the second epoch starts (with 1 or 0 workers too); every job saved its best epoch-0 weights first.
- Spend tonight about 1.6 USD (four MIG-slice decoder jobs at about 0.4 each, estimates); well under the 8 USD cap; no H200 job.
- The paused NCFlow rare-D-form job on the box GPU 0 was resumed.

## Problems and caveats
- Raw designs: still about a third to a half of the peptide bonds are outside 1.2 to 1.5 A and many designs have close contacts; no relaxation, no refolding with RF3 or AF3 yet, so nothing is known about foldability.
- 88 designs per run on 11 complexes: the differences between G, G2 and G3 are small and untested for significance.
- Earlier decode accuracies of 91 to 98% (sections 48, 51, 56) were largely the decoder copying the given identity; section 49 probably too (not verified).
- The first generation pass failed on one receptor fragment and was repaired (fragments are now also split at C-N distances above 2 A).
- The Tillicum host-memory error (data loader) is not fixed; a fix would let the decoder train for several epochs.

## Next steps (proposals)
1. Relax the designs and refold them with RF3 or AF3 (the box has the NCAA wrapper) and check whether the specified residue and the pose survive; compare with a canonical-only design on the same pockets.
2. Fix the data-loader memory problem, then train the default decoder for several epochs with the lighter handedness loss (run 7 recipe) to get both handedness and bonds.
3. The bigger goal (the model chooses the non-canonical residue): denoiser readout head + identity-hidden training, decoder on predicted latents (parked, sections 57 and 58).
4. Run the mirror-complex denoiser trial H (staged) to see whether real D backbone geometry helps further.

## Where things are
Results.MD sections 52 to 61; figures fig47 (pool), fig48 to fig53 (G2, G3 benchmarks), fig54 to fig58 (decode, blind decoder, generation, decoder runs); `gen_test/` (inputs, spec, designs in `out_local/`, diagnostics in `diag/`, `analyze_gen.py`); decoders `decoder_v5_493747.pt` (run 3), `decoder_v6_496342.pt` (blind), `decoder_v7_498188.pt` (run 5), `decoder_v7b_498314.pt` (run 5b, default), `decoder_v8_498707.pt` (run 6), `decoder_v8_499080.pt` (run 7) in `C:\Research\RFD4_fine_tune`; `night_plan_19nov.md`; model guide `model_guide.md`.
