# rosetta_score: paired Rosetta interface energy, one NCAA position at a time

Why: Boltz-2 / AF3 ipTM is not an affinity measure. The score used to teach the denoiser *why an NCAA beats the canonical residue*
(plan: Results.MD section 73, `docs/ideas/ncaa_sequence_denoiser_plan.md`) is a Rosetta interface dG.

## What it computes
For every non-canonical residue slot `i` of the peptide (caps ACE/NH2 excluded), **only that slot is changed** (all other residues,
other NCAAs included, stay as they are):

| variant | meaning |
|---|---|
| native | structure as given |
| parent | slot i swapped to its canonical L parent (DAL->ALA, ALY->LYS, ...) |
| ALA, GLY | slot i swapped to ALA / GLY (side-chain-free references) |

Per variant: repack the interface (extra rotamers), `E_complex`; cut peptide / receptor, repack both again; `dG = E_complex - E_peptide - E_receptor`
(Rosetta units, lower = binds better), mean and sd of 3 repeats. `gain = dG(swap) - dG(native)`; `gain > 0` means the NCAA beats the canonical residue.

Caveat: an L parent placed on a D-backbone clashes (gain > 20, flag `backbone_clash`); for D-forms use ALA/GLY as the reference.

## Run
```
python prep_pdb.py <dir_with_cif> ros_in                       # Windows python + gemmi -> Rosetta PDBs + manifest.json
wsl: python3 paired_dg.py ros_in out.json 3 28                  # PyRosetta (WSL system python3), 3 repeats, 28 processes
```
Never import gemmi in the PyRosetta process (segfault). `InterfaceAnalyzerMover` is not used (segfaults on split-chain NCAA peptides).
Rosetta supports 93 of the 132 NCAA codes of the real test set (all D-forms under Rosetta's names); other codes are skipped.

## Files
`prep_pdb.py`, `paired_dg.py`, `real_paired_dg.json` (results on the real test complexes), `run_real.log`, `ros_in/` (generated PDBs, can be deleted).
Figure: `figures/fig71_rosetta_paired_dg.png` (`figures/scripts/make_fig71_rosetta_dg.py`).
