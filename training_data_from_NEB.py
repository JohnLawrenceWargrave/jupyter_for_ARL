import os
import warnings

import numpy as np
from ase.io import read as ASEread

from Editing_NEB_files_to_anchor_method import end_structure


def test_for_outcars(abs_path, no_of_steps):
    """
    Check for the existence of OUTCAR files in a specified directory structure.

    This function verifies the presence of OUTCAR files in the initial, final, and transition steps
    of a given directory structure. It raises an AttributeError if any of the vital OUTCAR files
    (initial or final) are missing and issues a warning if any transition OUTCAR files are missing.

    Parameters:
    abs_path (str): The absolute path to the <.../run_final*>-directory containing the OUTCAR files.
    no_of_steps (int): The number of transition steps to check for.

    Returns:
    list: A list of paths to the OUTCAR files that exist.

    Raises:
    AttributeError: If the initial or final OUTCAR files are missing.

    Warnings:
    UserWarning: If any transition OUTCAR files are missing.

    Example:
    >>> test_for_outcars('/path/to/directory', no_of_steps=5)
    ['/path/to/directory/00/OUTCAR', '/path/to/directory/01/OUTCAR', ...]
    """
    abs_path_without_run_final = '/'.join(abs_path.split('/')[0:-1])
    outcar_paths = []

    # creation of expected paths
    outcar_paths.append(abs_path_without_run_final + '/OUTCAR_initial_image')
    outcar_paths.append(abs_path_without_run_final + '/OUTCAR_final_image')
    for i in range(0, no_of_steps + 2):
        if i < 10:
            outcar_paths[i] = f"{abs_path}/0{i}/OUTCAR"
        else:
            outcar_paths[i] = f"{abs_path}/{i}/OUTCAR"

    # testing for existance ot those paths
    existing_paths = []
    missing = []
    for i in range len(outcar_paths):
        if os.path.exists(outcar_paths[i]):
            existing_paths.append(outcar_paths[i])
        else:
            missing.append(outcar_paths[i].replace(abs_path_without_run_final, ''))
            if i < 2:
                raise AttributeError(f'{outcar_paths[i]} - does not exist but is vital \t PROCESS STOPPED')
    if len(missing) > 0:
        warnings.warn(f'In folder <{abs_path_without_run_final}> the data of {missing} is missing {no_of_steps} transition points were expected', UserWarning)
    return (outcar_paths)


def identify_jumping_atom(atoms_init, atoms_final):
    """
    Identify the index of the atom that jumps between the initial and final structures.

    This function calculates the displacement of each atom between the initial and final structures
    using the minimum image convention. It identifies the atom that has moved the most, which is
    assumed to be the jumping atom.

    Parameters:
    atoms_init (ase.Atoms): The initial structure.
    atoms_final (ase.Atoms): The final structure.

    Returns:
    int: The index of the jumping atom.

    Raises:
    Exception: If more than one atom is found to have moved more than 2 Angstroms.
    """
    # apply minimum image convention to get the displacement vector in relative coordinates for every atom
    init_pos  = atoms_init.get_scaled_positions()
    final_pos = atoms_final.get_scaled_positions()
    for i, (pos_init, pos_fin) in enumerate(zip(init_pos, final_pos)):
        for xyz in [0,1,2]:
            if pos_fin[xyz] - pos_init[xyz] > 0.5:
                final_pos[i][xyz] = final_pos[i][xyz] - 1
            elif pos_fin[xyz] - pos_init[xyz] < -0.5:
                final_pos[i][xyz] = final_pos[i][xyz] + 1
    rel_displacement_vector = final_pos - init_pos

    # get cell and compute the displacement vector in cartesian form
    (a,b,c) = atoms_init.get_cell()
    displacement_vector = []
    for (x, y, z) in rel_displacement_vector:
        displacement_vector.append(x*a+y*b+z*c)
    displacement_vector = np.array(displacement_vector)

    # get displacement magnitude and check for the ones with displacement larger than 2
    # which should be just the Li that jumps. Get its index
    displacement = np.sqrt(np.sum(displacement_vector ** 2, axis = 1))
    count = 0
    for i, displ in enumerate(displacement):
        if displ > 2:
            count += 1
            Li_index = i

    # Check that there are not >1 atoms moving
    if count != 1:
        raise Exception(f"WARNING: There are {count} ions that move more than 2 Ang during the NEB! Check NEB path again!")
    else:
        return Li_index


def create_anchor_structure(atoms_init, atoms_final):
    """
    Create an anchor structure by identifying the jumping atom and modifying the initial and final structures.

    This function identifies the atom that jumps between the initial and final structures, renames it to 'Ti' in both
    structures, and writes the modified initial structure with the jumping atom from the final structure appended to it
    as a new file in the current directory.

    Parameters:
    start_structure (ase.Atoms): The initial structure.
    end_structure (ase.Atoms): The final structure.

    Returns:
    None

    Raises:
    Exception: If more than one atom is found to have moved more than 2 Angstroms.
    """
    i = identify_jumping_atom(atoms_init, atoms_final)
    atoms_anchor = atoms_init.copy()
    atoms_anchor[i].symbol = 'Ti'
    atoms_final[i].symbol = 'Ti'
    atoms_anchor.append(atoms_final[i])
    return atoms_anchor

def get_energy(atoms):
    # Compute heat of mixing and per atom
    li_count = atoms[-1].get_chemical_symbols().count("Li")
    o_count  = atoms[-1].get_chemical_symbols().count("O")
    heat_of_mixing    = atoms[-1].get_potential_energy() - Li_count * E_ref_LiNiO2_per_O2 - (O_count/2 - Li_count) * E_ref_NiO2_per_O2

    # total number of lattice sites:  (Oxygen) O_count + (Lithium)O_count/2 + (Nickel) O_count/2 for Ni-Sites = 2*O_count --> Lattice Sites
    H_o_M_for_training_NEB_initial_and_final_images.append( H_o_M / (2*O_count) )

def create_CE_training_data_point(abs_path, no_of_steps=5):
    true_paths = test_for_outcars(abs_path, no_of_steps)
    abs_path_without_run_final = '/'.join(abs_path.split('/')[0:-1])
    atoms_init = ASEread(abs_path_without_run_final + '/OUTCAR_initial_image')
    atoms_final = ASEread(abs_path_without_run_final + '/OUTCAR_final_image')
    atoms_anchor = create_anchor_structure(atoms_init, atoms_final)
    hom_


