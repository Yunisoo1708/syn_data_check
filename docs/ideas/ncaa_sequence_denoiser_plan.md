# Plan: teaching the denoiser to choose non-canonical residues itself (the big goal)

Written 20 Nov 2026. Status of the route is in Results.MD sections 66-70 and `README.md`.

## Why the current model cannot do it (the canonical bias, in four layers)
1. **The model was pretrained on canonical sequences.** RFD4's sequence readout (a 189-class head on the denoiser's hidden state, `nn.sequence_readout`) exists in the bundle but was learned on the 20 natural amino acids. It was a frozen *detached probe* in every trial so far: our LoRA only trained attention and the two output heads (CA, latents). Top-1 accuracy at NCAA positions is 3-5% for every denoiser, the untuned one included.
2. **The NCAA identity was always handed to the model.** In every training batch the identity of a non-canonical residue is an *input* (the sequence-conditioning feature), so the readout loss (which only counts tokens whose identity is hidden) never saw an NCAA. The denoiser learned "given this NCAA, make a compatible structure", never "which NCAA belongs here".
3. **The architecture restricts the choice.** The token type (0 protein, 6 NCAA/PTM) is an input and the decoder only allows the residue classes of that type, so a protein-type token can never become an NCAA and the model cannot choose *where* NCAAs go.
4. **The data carry little signal about *which* NCAA.** A synthetic complex is a natural complex with one residue swapped by the NCFlow generator by a rule (D-form of the parent, N-methyl of the parent, ...), not chosen because it binds better. NCAA tokens are about 10% of the tokens, and most synthetic examples are D-forms of the residue that was there. A perfect model of this data learns "NCAA = chemical variant of the natural residue that sat here", which is a useful prior but not a design principle.

## Step S1 (implemented in `denoiser/denoiser_train.py`, Tillicum debug job 502169): train the readout on NCAAs
* `--train-readout` adds `sequence_readout` to the trainable modules; `--readout-attach` lets its loss flow into the transformer (so the hidden state must *carry* the identity, not only expose it); `--nc-hide-p 1` hides the identity of every NCAA token in training, so the readout must infer it from the pocket, the backbone and the neighbours; `--seq-w` weights the readout CE (3).
* The loss is the library's own cross-entropy over the 189 classes on generated tokens whose identity is hidden: canonical tokens keep teaching the canonical prior, NCAA tokens now teach the NCAAs. Class-balanced data (`train_all_cif_bal280`) counters the prior over codes.
* Held-out check printed each evaluation: top-1 / top-5 at hidden NCAA tokens and the share of guesses that are canonical classes (a direct measure of the canonical bias).
* How the choice reaches the structure: at generation the readout's argmax (restricted to the token type's allowed classes) becomes the identity input of the decoder (`condition_decoder_on_sequence_prediction` option of RFD4), so decoder run 8 (identity-conditioned) builds the atoms. No retraining of the decoder is needed for this route.

## Step S2: let the model choose the token type (where NCAAs go)
Replace the input token type of designable peptide positions by a single "designable" type that allows canonical and NCAA classes, and train the readout to choose between them; add a prior on the fraction of NCAAs so the model neither never nor always chooses one. Needs a change of the allowed-class mask in the decoder call and training examples in which the same pocket appears with canonical and NCAA versions.

## Step S3: train as it is used (receptor fixed)
Training batches noise every token; generation fixes the receptor. Make the receptor (C_CRD and C_SEQ) conditioning in training and put the loss only on the peptide tokens, so the sequence readout learns the receptor-conditioned distribution.

## Step S4: a signal for *which* NCAA is better
Without a binding or stability signal the model can only reproduce the generator's prior. Options in rising cost: Rosetta interface energy of designs (PyRosetta on the laptop, relaxed), Boltz-2 / AF3 ipTM of the refolded design with the receptor held by a template (works, Results.MD 64), and reward fine-tuning (as done for NCFlow earlier): sample N designs, score, train toward the best.

## How we will know it works (all on real complexes, test only)
* Readout accuracy at NCAA positions (top-1, top-5, share of canonical guesses) after denoising at t = 0.9 ... 0.1, against the untuned control (3-5%) and a majority-code baseline.
* Free-sequence generation (`gen_test/run_free.sh`, `analyze_free.py`): share of NCAA slots that get the native code, the native parent amino acid, the right class (D-form), and the diversity of codes; structure measures as before.
* Not enough on its own: exact recovery of the native NCAA is a harsh test (many NCAAs are interchangeable); report class-level and parent-level agreement too.

## Risks
The readout may learn only the generator's rule (parent -> D-form); the synthetic prior may transfer poorly to real complexes (as the larger pool did); attaching the readout to the trunk may cost structure quality (check bonds, handedness, close contacts after S1 with the generation test).

## Finding of 20 Nov (Results.MD 71): the generation pipeline always supplies the NCAA identity
Removing the sequence condition of the peptide from the spec did not change the output: the NCAA slots came out with the native code in 100% of the designs (control and blind decoder included). The pipeline fills the sequence condition for non-canonical residues from the input structure. Before a free-choice generation test (or any use of the trained readout in generation) the step that supplies the identity must be patched (or the NCAA slots in the input CIF replaced by a placeholder residue of the NCAA token type). Trial S1's readout is tested offline with the identity hidden by hand (`gen_test/eval_readout.py`).

## Refinement of 20 Nov (user): cap of 5 NCAAs per design, learn the gain over the canonical residue
The model should not choose NCAAs exclusively: at most 5 NCAAs per design. What it must learn is what an NCAA adds compared with the canonical residue at the same slot (score gain), not the generator's parent -> D-form rule. Consequences for the steps: S2 (token type 'designable') gets a hard cap (keep the 5 slots with the largest NCAA-minus-canonical margin at generation, plus a penalty on the expected NCAA count in training); S4 (score) becomes paired: same pocket and position, canonical vs NCAA variant, label = score gain. Visual plan: figures/fig70_plan_choose_ncaa.png, Results.MD section 73.

## Score for the paired gain: Rosetta, not ipTM (20 Nov, user)
Boltz-2/AF3 ipTM is not an affinity measure, so S4 uses Rosetta interface dG (PyRosetta in WSL, free, no GPU). Feasibility (Results.MD 73): 93 of 132 NCAA codes have Rosetta residue types (all D-forms under Rosetta's names, ALY/M3L/PTR/SEP/TPO/HYP/SAR/AIB/NLE and N-methyls); 39 exotic codes lack params; 62 of 84 real complexes load and score. InterfaceAnalyzerMover segfaults on poses where the NCAA peptide is split into several chains: compute dG by hand (separate, repack interface, complex minus separated) and average several repacks. Paired gain = dG(NCAA variant) - dG(canonical parent) at the same slot after the same repack protocol.
