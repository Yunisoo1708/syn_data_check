from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.vectors import calc_angle
from itertools import combinations
import numpy as np

# Ideal bond lengths in Angstrom. Keys are frozensets so the order of the two atoms doesn't matter:
# BACKBONE_BONDS[frozenset(('CA','N'))] and BACKBONE_BONDS[frozenset(('N','CA'))] give the same answer.
BACKBONE_BONDS = {
    frozenset(('N',  'CA')): 1.46,   # N-CA, inside one residue
    frozenset(('CA', 'C')):  1.52,   # CA-C, inside one residue
    frozenset(('C',  'O')):  1.23,   # carbonyl C=O (shorter because it is a double bond)
    frozenset(('CA', 'CB')): 1.53,   # CA to the first side-chain atom (Ala 1.52, others up to ~1.54)
}
PEPTIDE_BOND = 1.33                  # C of residue i to N of residue i+1: BETWEEN two residues, partial double bond

# Rough values for side chains, keyed by the two ELEMENTS (atom.element, e.g. 'C', 'N').
SIDECHAIN_BONDS = {
    frozenset(('C', 'C')): 1.53,     # single C-C (aromatic ring C-C is shorter: ~1.39)
    frozenset(('C', 'N')): 1.46,     # single C-N (amide/guanidinium C-N is shorter: ~1.33)
    frozenset(('C', 'O')): 1.43,     # single C-O, as in Ser/Thr hydroxyl (carboxylate C-O is ~1.25)
    frozenset(('C', 'S')): 1.81,     # C-S in Cys/Met
    frozenset(('S', 'S')): 2.03,     # disulfide
}

TOL_STRICT = 0.05   # for experimental or well-refined structures
TOL_LOOSE  = 0.10   # for predicted or generated structures

# (ideal angle, typical spread) in degrees. Keys are "end1-middle-end2": the MIDDLE atom is the vertex of the angle.
# "N+1" = N of the NEXT residue, "C-1" = C of the PREVIOUS residue (those bonds cross a peptide bond).
BACKBONE_ANGLES = {
    'N-CA-C':     (111.0, 2.7),   # the main backbone angle (varies with phi/psi, so the spread is big)
    'CA-C-O':     (120.1, 2.1),   # carbonyl
    'CA-C-N+1':   (117.2, 2.2),   # across the peptide bond
    'O-C-N+1':    (122.7, 1.6),   # carbonyl O to the next N
    'C-1-N-CA':   (121.7, 1.8),   # previous C, this N, this CA
    'N-CA-CB':    (110.5, 1.7),   # backbone to side chain
    'C-CA-CB':    (110.1, 1.9),
}

# Residues that differ from the generic values above.
SPECIAL_ANGLES = {
    'GLY': {'N-CA-C': (113.1, 2.5)},                                   # no CB, more open
    'PRO': {'N-CA-C': (112.1, 2.6), 'N-CA-CB': (103.0, 1.1), 'C-1-N-CA': (122.6, 5.0)},  # ring squeezes N-CA-CB
}

# Rough defaults by geometry, for side-chain angles you have no table for.
SIDECHAIN_DEFAULTS = {'sp3 carbon': (109.5, 3.0),   # CB-CG-CD, etc. (real values range ~108-116)
                      'aromatic ring': (120.0, 2.0)}  # inside Phe/Tyr/Trp rings

TOL_SIGMA = 3.0   # an angle fails if it is more than this many spreads from the ideal

Backbone_atoms = ['N', 'CA', 'C', 'O', 'CB'] #PROBLEM: OXT (C-terminal oxygen) is not in here so it counts as side chain, and the C-OXT bond is never checked (it crosses the 2 groups).

def parse_structure(file):
    parser = MMCIFParser() #optional: MMCIFParser(QUIET=True) hides the long warning list.
    structure = parser.get_structure('structure', file)
    chains = list(structure.get_chains())
    residues = []
    for chain in chains:
        residue = chain.get_residues() #get_residues() returns all residues of a chain. 
        residues += residue
    return (chain, residues)

def bond_length_check(file):
    _, residues = parse_structure(file)
    distances = {}
    for res in residues: 
        atoms = list(res.get_atoms())
        side_chain_atoms = [a for a in atoms if a.get_id() not in Backbone_atoms]
        backbone_atoms = [a for a in atoms if a.get_id() in Backbone_atoms] 
        assert len(backbone_atoms) != 0, f'No backbone atom found for {res.get_id()}'
        ns_bb = NeighborSearch(backbone_atoms) 
        bonded_pairs_backbone = ns_bb.search_all(radius=2.1, level='A') 
        for a_1, a_2 in bonded_pairs_backbone: 
            ideal_bond_length = BACKBONE_BONDS.get(frozenset((f'{a_1.get_id()}', f'{a_2.get_id()}')))
            distances[(f'{a_1.get_id()}_{res.id[1]}', f'{a_2.get_id()}_{res.id[1]}')] =  bool(abs(np.sqrt(np.sum((a_1.get_coord() - a_2.get_coord())**2)) - ideal_bond_length) < TOL_STRICT) 

        if len(side_chain_atoms) != 0:
            ns_sd = NeighborSearch(side_chain_atoms)   
            bonded_side_chain_pairs = ns_sd.search_all(radius=2.1, level='A') 
            for a_1, a_2 in bonded_side_chain_pairs: 
                ideal_bond_length = SIDECHAIN_BONDS.get(frozenset((f'{a_1.element}', f'{a_2.element}')))
                distances[f'{a_1.get_id()}_{res.id[1]}', f'{a_2.get_id()}_{res.id[1]}'] = bool(abs(np.sqrt(np.sum((a_1.get_coord() - a_2.get_coord())**2)) - ideal_bond_length) < TOL_STRICT) 
        else:
            pass 

    return distances #NOTE: only bonds INSIDE one residue are found, so the peptide bond C(i)-N(i+1) (PEPTIDE_BOND) is never checked. 

def angle_check(file):
    _, residues = parse_structure(file)
    angle = {} #PROBLEM: same name as the number `angle` below, so this dict is overwritten at the first triple. Rename one of them.
    for res in residues: 
        atoms = list(res.get_atoms())
        side_chain_atoms = [a for a in atoms if a.get_id() not in Backbone_atoms]
        backbone_atoms = [a for a in atoms if a.get_id() in Backbone_atoms]
        backbone_combinations = combinations(backbone_atoms, 3) #NOTE 1: only atoms of ONE residue, so the table entries with N+1 / C-1 can never be reached. NOTE 2: beta amino acids (B3A, B3K, B3L, B3Q, BAL) and 04Q have no normal N-CA-C backbone, skip them or they give false failures. NOTE 3: Gly / Pro should use SPECIAL_ANGLES[res.get_resname()].
        side_chain_combinations = combinations(side_chain_atoms, 3)
        for group in [backbone_combinations, side_chain_combinations]:
            for minor in group: 
                a1, a2, a3 = minor #PROBLEM: combinations gives the atoms in FILE order (N, CA, C, O, CB), so a2 is not always the corner of the angle. C-CA-CB is never found this way. Try each of the 3 atoms as the corner (a2). 
                v1 = a1.get_vector()
                v2 = a2.get_vector()
                v3 = a3.get_vector()
                angle  = calc_angle(v1,v2,v3) #PROBLEM: calc_angle returns RADIANS but the table is in DEGREES -> np.degrees(...). (v2 = the corner is correctly in the middle.)
                standard_angle, deviation = BACKBONE_ANGLES.values(f'{a1.get_id()}-{a2.get_id()}-{a3.get_id()}') #PROBLEM 1: dict.values() takes no argument, look up with BACKBONE_ANGLES[key]. PROBLEM 2: that raises KeyError when the key is not in the table (every side chain triple, and e.g. N-C-O). Check `key in` first and `continue`. PROBLEM 3: the key can also be written backwards (C-CA-N), try both.
                angle[minor] = (abs(angle - standard_angle) < TOL_SIGMA* deviation) #PROBLEM: `angle` is a number here (see the dict on top), so item assignment fails. Also the key `minor` is a tuple of Atom objects, hard to read; use the names + residue number. #Toll_sigma = Idea of how spread do you want the thing to be.

    return angle #PROBLEM: returns the last NUMBER, not the dict of results (rename as above).

print(bond_length_check('data/test_cif/rf3_L07-C0310.cif'))


