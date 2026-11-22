# gen_test/ - the generation test (does the whole pipeline produce good designs?)

**The test:** take 11 real peptide-protein complexes (test only, peptides of 7 to 10 residues, all with NCAAs). Keep the receptor pocket fixed, give the denoiser the *identity* of the NCAAs at their native positions, let it design the rest of the peptide (8 designs per pocket = 88 per run), decode with a chosen decoder, and measure the result against the native peptide. No relaxation, no refolding: raw outputs.

## Measures (computed by `analyze_gen.py`)
`nc_kept` the given NCAAs are present | `chir_D` D-forms with the right C-alpha handedness | `bond_ok` peptide bonds of 1.2-1.5 A | `clash` share of designs with a peptide-receptor atom pair < 2 A | `ca_rmsd_med` median CA distance to the native peptide in the receptor frame | `contacts_rel_med` peptide-receptor contacts relative to native | `free_rec` sequence identity to native at the designed positions | `nc_atoms` NCAAs with the native atom count.

## Workflow
1. `prep_inputs.py`, `make_spec.py` build the generation inputs from the real complexes (`inputs/`, `spec_gen.yaml`).
2. **On the GPU box:** `gen_run.py NAME ADAPTER DECODER OUT_DIR "" 8` generates the designs (denoiser adapter + decoder; the seed is `GEN_SEED`, default 1234). The `run_*.sh` files are the launchers of each experiment (e.g. `run_v10_all.sh` = decoder run 8 with G, G2, G3; `run_g4.sh` = trial G4; `run_seed777.sh` = a second seed). They set the environment of the RFD4 python on the box.
3. **Anywhere with Python + gemmi:** `python analyze_gen.py out_local/RUN_DIR ...` (always pass all run directories; it rewrites `gen_summary.json`).
4. Diagnostics: `diag_decode_atoms.py` (decode the TRUE latents of real complexes: tests the decoder alone), `diag_decode_chir.py`, `diag_emit.py`, `diag_noise.py`, `diag_terms.py`, `diag_clash_source.py` (which atoms clash), `best_of_n_filter.py` (keep one clash-free design out of N draws).
5. `refold/`: re-predict selected designs with AF3 or Boltz-2 (receptor held by a template) as an independent check (Results.MD 63, 64).

## Folders
`out_local/` the generated designs (one folder per denoiser x decoder, e.g. `G3_bal_v10` = denoiser G3 with decoder run 8; `v5` = run 3, `v7b` = run 5b, `v9` = run 7, `v10` = run 8, `v11` = run 9), `diag/` diagnostic tables, `inputs*/` generation inputs, `refold/` refolding.

Results: Results.MD sections 59 to 66, figures 56 to 63.
