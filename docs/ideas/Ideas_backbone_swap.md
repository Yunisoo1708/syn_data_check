# Idea (parked): change only the backbone, keep the side-chain interactions

Saved 15 Nov 2026 for the time we fine-tune the denoiser. ![Backbone options](figures/fig17_backbone_options.png)

## Goal
Start from a natural peptide-protein complex (list: `natural_complexes/natural_complexes.md`) and turn the peptide into a fully non-canonical one: the side chains stay as they are, so the contacts with the protein are kept, and only the backbone is changed (N-methyl, alpha-methyl, amide isosteres, beta or gamma residues, D residues, retro-inverso, macrocycles). This follows the priority "first a model that understands non-canonical backbones, then the side chains".

## Which backbone changes keep the side chains where they are
(see the figure; design judgement from the chemistry, not a measurement)
- **Same atoms in the same places, only the amide changes:** N-methylation, alpha-methylation (Aib type), amide isosteres (ester, thioamide, reduced amide, E-olefin, triazole). The side-chain atoms do not move; check the hydrogen-bond donor or acceptor that is lost and the extra methyl against the protein.
- **Length-changing backbones:** beta3-homologation (same side chain, one extra CH2) and gamma residues move the neighbouring side chains apart. In helices an alpha/beta pattern (alpha on the binding face, beta on the other face) keeps the binding-face side chains in place; elsewhere the backbone must be re-routed between side-chain anchors that stay fixed.
- **Mirror-type changes:** a single D residue puts the side chain on the opposite face (changes the interaction, except at turns); retro-inverso (reversed sequence, all D) keeps the side-chain topology of helices and strands; peptoids move the side chain from CA to N (one bond shift).
- **Macrocycles and staples:** use side-chain to side-chain links on the face that does not touch the protein.

## Recipe (side-chain-anchored backbone redesign)
1. For each peptide residue define its anchor: the CA position, the CA-to-CB direction and the side-chain atoms with their chi angles. Mark the residues whose side chains touch the protein (contacts within 4.5 A).
2. From the structure find which backbone NH and CO atoms hydrogen-bond to the protein or inside the peptide. Rules: keep an NH that bonds the protein (no N-methyl there); N-methyl or alpha-methyl are allowed where the NH is solvent-exposed or only bonds inside the peptide; length-changing residues only where a neighbouring anchor can move or on the non-binding face.
3. Build the new backbone round by round (one position per round, as in the conversion pilot): same-length changes by direct substitution on the existing backbone atoms; length-changing ones by generating the backbone between the fixed anchors (NCFlow-style local generation with the side chains and the protein as fixed context, or RFD4 motif scaffolding with the side-chain atoms as the fixed motif).
4. Keep a sample only if it passes the strict test (bonds, angles, clashes, link, handedness) and keeps the interaction: side-chain atom RMSD and chi angles against the natural complex, residue contacts and polar contacts kept, no clash with the protein (the measures of Results.MD sections 18 and 22).
5. Optionally score the binding with a short relaxation or a docking re-run of the converted sequence (not done yet).

## Where this meets the denoiser fine-tuning
- Task for RFD4: receptor + fixed side-chain atoms of the interface residues (the motif) + a backbone-class token per position (the 176-class vocabulary) -> generate the new backbone and the rest of the peptide. The pairs (natural complex, converted peptide) come from our pipeline: conversion with NCFlow or RF3 pose substitution, kept only if the strict test passes.
- What the experiments so far say about it (Results.MD sections 17 to 22): NCFlow redraws residues with correct bond lengths and angles after the reward fine-tune but draws the wrong handedness about half of the time on synthetic peptides, so every generated residue must go through the strict check (4 draws are enough for about 81 %); the mirror repair makes complete D conversions possible; RF3 keeps a given pose but does not repair chemistry; the interaction with the protein is kept (about 90 % of contacts) in the real-complex conversions.

## Open questions
- (Answered in a first screen, `natural_complexes/natural_complexes.md`, fig18: 36% of the residues have a backbone N or O near the protein, 21% of positions take an N-methyl with a free NH, 8% an alpha-methyl; helical peptides are the most open. To be confirmed with water-mediated bonds and an energy check.)
- For beta homologation, how much can the neighbouring side chains be re-placed before the interaction is lost (anchor displacement budget)?
- A suitable score for binding after the change (side-chain contacts only, or a relaxation energy)?
- Amide isosteres need new residue definitions (ester, thioamide) in the geometry test and in the generators.
