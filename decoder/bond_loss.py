"""Peptide-bond length loss for the decoder.

Atom37 layout: N 0, CA 1, C 2, CB 3, O 4. The peptide bond of residue i and i+1 is the distance C(i) - N(i+1).
Which neighbours are really bonded is taken from the TRUE structure (true C-N distance between BOND_MIN and BOND_MAX Angstrom),
so chain breaks, gaps, padding and cropped ends are skipped without needing chain ids.
Loss = flat-bottom square: relu(|d - target| - tol)^2 (A^2), mean over real bonds. No penalty inside target +- tol, so the
decoder is not pushed away from the native spread; a bond of 1.5 A (the edge of the sensible range) costs (0.17-tol)^2.
Coordinates are in nm (as the decoder emits); distances are converted to Angstrom here.
"""
import torch


def peptide_bond_loss(pred_nm, true_nm, atom_mask, target=1.33, tol=0.04, bond_min=1.0, bond_max=1.8):
    """pred_nm, true_nm: [B, L, 37, 3]; atom_mask: [B, L, 37] bool (supervised atoms, already restricted to emitted tokens).
    Returns (loss, n_bonds, n_sensible); n_sensible = bonds with 1.2 <= d <= 1.5 A (the generation-test criterion)."""
    d_pred = (pred_nm[:, 1:, 0, :] - pred_nm[:, :-1, 2, :]).norm(dim=-1) * 10.0
    d_true = (true_nm[:, 1:, 0, :] - true_nm[:, :-1, 2, :]).norm(dim=-1) * 10.0
    m = atom_mask[:, 1:, 0].bool() & atom_mask[:, :-1, 2].bool() & (d_true > bond_min) & (d_true < bond_max)
    if int(m.sum()) == 0:
        return pred_nm.sum() * 0.0, 0, 0
    dev = (d_pred - target).abs()
    loss = (torch.relu(dev - tol) ** 2)[m].mean()
    ok = int(((d_pred >= 1.2) & (d_pred <= 1.5) & m).sum())
    return loss, int(m.sum()), ok


def clash_loss(pred_nm, true_nm, atom_mask, emit, chain, d0=2.5):
    """Inter-chain clash loss (the peptide against the receptor, found through the chain ids, not through the emit mask, which also covers receptor tokens).
    For every decoded atom of an emitted token and every TRUE atom (no gradient) of a token of a DIFFERENT chain closer than d0 Angstrom: penalty (d0 - d)^2, summed per complex and divided by the number of decoded atoms, mean over complexes with at least one such pair of chains.
    d0 = 2.5 A is below the shortest native hydrogen-bond distances (about 2.6 A), so native contacts are not penalised; the generation test counts a clash below 2.0 A.
    pred_nm, true_nm: [B, L, 37, 3]; atom_mask [B, L, 37] bool; emit [B, L] bool; chain [B, L] int. Returns (loss, n_pairs_below_2A, n_complexes_used)."""
    tot, used, n2 = pred_nm.sum() * 0.0, 0, 0
    for i in range(pred_nm.shape[0]):
        am = atom_mask[i].bool(); pm = emit[i][:, None] & am
        if int(pm.sum()) == 0: continue
        tok = torch.arange(am.shape[0], device=am.device)[:, None].expand_as(am); ch = chain[i].long()[tok]
        P = pred_nm[i][pm] * 10.0; cp = ch[pm]; R = (true_nm[i][am] * 10.0).detach(); cr = ch[am]
        diff = cp[:, None] != cr[None]
        if not bool(diff.any()): continue
        d = torch.cdist(P, R); pen = (torch.relu(d0 - d) ** 2) * diff
        tot = tot + pen.sum() / P.shape[0]; used += 1; n2 += int(((d < 2.0) & diff).sum())
    return tot / max(used, 1), n2, used
