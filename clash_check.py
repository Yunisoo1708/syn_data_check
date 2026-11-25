from pymol import cmd
import argparse
import numpy as np
from Bio import PDB
from Bio.PDB import MMCIFParser, NeighborSearch

atom_radii = {
    "H": 1.20, 
    "C": 1.70, 
    "N": 1.55, 
    "O": 1.52,
    "S": 1.80,
    "F": 1.47, 
    "P": 1.80, 
    "CL": 1.75, 
    "MG": 1.73,
}

def VdW_check(file, clash_cutoff = 1):
    parser = MMCIFParser() 
    structure = parser.get_structure('label_protein',file)
    chains = list(structure.get_chains())
    clash_count = {}
    clash_atoms = []

    for chain in chains: 
        residues = list(chain.get_residues()) 
        atoms = [a for res in residues for a in res.get_atoms()]
        search = NeighborSearch(atoms)
        nearby_atom = search.search_all(2, level='A')
        for a1, a2 in nearby_atom: 
            a1_radius = atom_radii.get(str(a1.element))
            a2_radius = atom_radii.get(str(a2.element))
            ideal_dist = a1_radius + a2_radius 
            a1_coordinate = np.array([a1.get_coord()])
            a2_coordinate = np.array([a2.get_coord()])
            dist = np.sqrt(np.sum((a1_coordinate - a2_coordinate)**2, axis=-1)) #Real distance
            if dist < clash_cutoff*ideal_dist:
                clash_count[f'{a1.id}_{a1.get_parent().id[1]}_{chain.id}-{a2.id}_{a2.get_parent().id[1]}_{chain.id}'] = (dist, clash_cutoff*ideal_dist)
                clash_atoms.append((a1, a2, dist))
            else: 
                continue

    return clash_count, clash_atoms

def pymol_visualization(file):
    cmd.load(file, 'struc')
    cmd.show('cartoon','struc')
    _, clash_atoms = VdW_check(file)
    for k, (a1, a2, dist) in enumerate(clash_atoms):
        s1 = cmd.select('a1', f'struc and id {a1.get_serial_number()}') #a1 = name of selection, 2nd slot: the command. struc = name of structure and id (= serial number. The order that atoms appear in the structure)
        s2 = cmd.select('a2', f'struct and id {a2.get_serial_number()}')
        new_dist = cmd.distance(f'clash_{k}', s1, s2)
        assert abs(dist - new_dist) < 0.1, f'Mismatch between Pymol {new_dist} and Biopython distance {dist}'

    return None

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure')
    arg = parser.parse_args() #Read what input from parser has been filled
    structure = arg.structure #extract it here 
    clash_count, _ = VdW_check(structure)
    if clash_count.items is None: 
        print('No clash found')
    else:
        for k, d in clash_count.items(): 
            measured_distance, cut_off_distance = d 
            print(f"Clash found at: {k}. With measured distance: {measured_distance}. Cut-off distance: {cut_off_distance}")
        
    pymol_visualization(structure)