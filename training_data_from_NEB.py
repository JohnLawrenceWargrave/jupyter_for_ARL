import os
import warnings
import glob
import numpy as np
from ase import Atoms
from ase.io import read as ase_read


class TrainingDataCreator:
    def __init__(self,
                 file_li_ni_o2: str ="/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/CE_database_Marcel/02_enumerate_P21c_0-4fu/0001_finished_approved/run_final_approved/OUTCAR",
                 file_ni_o2: str = "/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/CE_database_Marcel/02_enumerate_P21c_0-4fu/0003_finished_approved/run_final_approved/OUTCAR"):
        self.li_ni_o2 = ase_read(file_li_ni_o2)
        self.ni_o2 = ase_read(file_ni_o2)
        self.e_ref_LiNiO2_per_O2 = (
                self.li_ni_o2.get_potential_energy() / self.li_ni_o2.get_chemical_symbols().count('O') * 2)
        self.e_ref_NiO2_per_O2 = self.ni_o2.get_potential_energy() / self.ni_o2.get_chemical_symbols().count('O') * 2

    def create_CE_datapoint_list_withoutTS(self, folder: str):
        datapoint_list = [[]]

    @staticmethod
    def find_outcar_files_without_TS():
        """
        Warning: HARDCODED

        Returns:
        list: A list of all paths for OUTCAR files deemed good in Marcel's database.
        """
        path_outcars_without_TS = [
            # own enumerated structures based on P21/c
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/CE_database_Marcel/02_enumerate_P21c_0-4fu/0*_finished_approved/run_final_approved/OUTCAR"),
            # Markus low Energy data

            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0250/image*/02_scan/*final/OUTCAR"),
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0500/image*/02_scan/*final/OUTCAR"),
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0750/image*/02_scan/*final/OUTCAR"),
            # and the ones from the random structures
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0*random*/01_initial_structure/02_scan/*final/OUTCAR"),
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0*random*/02_odh/image*/02_scan/*final/OUTCAR"),
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0*random*/03_tsh/image*/02_scan/*final/OUTCAR"),
            glob.glob("/nfshome/sadowski/work/LiNiO2_data_base_Sabrina/DFT_database/NEBs_Marcel/0*random*/04_double_tsh/image*/02_scan/*final/OUTCAR"),
            ] #todo markus strukturen und strukturen von neb start und final
        return path_outcars_without_TS

    @staticmethod
    def find_outcar_files_transition_states() -> list[str]:
        """
        Find all files named OUTCAR* in the given folder and its subfolders.

        Parameters:
        base_folder (str): The path to the base folder where the search will begin.

        Returns:
        list: A list of paths to the found OUTCAR* files.
        """
        path_outcar_files = []
        for root, dirs, files in os.walk(base_folder):
            path_outcar_files.extend(glob.glob(os.path.join(root, "OUTCAR")))
        # todo: everything

    def create_CE_training_datapoint_withoutTS(self, path: str, path_original_structure: str = __path__) -> tuple[Atoms, float]:
        atoms = ase_read(path, index=-1)
        e_ref, o_count = self.reference_state_energy(atoms)
        h_o_m_per_site = (e_ref + atoms[-1].get_potential_energy()) / (2 * o_count)
        return atoms, h_o_m_per_site

    def create_CE_training_datapoint_transition_state(self, path_final: str, no_of_steps: int=6) -> tuple[Atoms, float]: 
        """
        Create a CE training data point for a transition state.
        This function identifies the jumping atom in the transition state, creates an anchor structure,
        and calculates the heat of mixing per site.
        Parameters:
        path_final (str): The path to the final OUTCAR file.
        no_of_steps (int): The number of transition steps.
        Returns:
        tuple: A tuple containing the anchor structure and the heat of mixing per site.
        """      
        path_without_run_final = '/'.join(path_final.split('/')[0:-1])
        atoms_init = ase_read(path_without_run_final + '/OUTCAR_initial_image')
        atoms_final = ase_read(path_without_run_final + '/OUTCAR_final_image')
        atoms_anchor = self.create_anchor_structure(atoms_init, atoms_final)
        transition_energy = self.get_energy_barrier(self.test_for_outcars(path_final, no_of_steps))
        e_ref, o_count = self.reference_state_energy(atoms_init)
        h_o_m_per_site = (e_ref + transition_energy) / (2 * o_count)
        return atoms_anchor, h_o_m_per_site

    def reference_state_energy(self, atoms: Atoms) -> tuple[float, int]:
        """
        Calculate the reference state energy (structural heat of mixing) for a given atomic structure.
        This function computes the reference state energy based on the number of Li and O atoms in the structure.
        It uses the reference energies of LiNiO2 and NiO2 per O2 unit to calculate the energy.
        Parameters:
        atoms (ase.Atoms): The atomic structure for which to calculate the reference state energy.
        Returns:
        tuple: A tuple containing the reference state energy and the number of O atoms in the structure.
        """
        li_count = atoms.get_chemical_symbols().count("Li") # +1 for the jumping Li
        o_count  = atoms.get_chemical_symbols().count("O")
        e_ref = -li_count * self.e_ref_LiNiO2_per_O2 - (o_count/2-li_count) * self.e_ref_NiO2_per_O2
        return e_ref, o_count

#         h_o_m    = max(energies) - Li_count * E_ref_LiNiO2_per_O2 - (O_count/2-Li_count) * E_ref_NiO2_per_O2
#         # total number of lattice sites: no(Oxygen) + [no(Li)=no(O)/2] + [no(Ni)=no(O)/2] = 2*no(O) --> Lattice Sites
#         h_o_m_per_site = h_o_m / (2 * o_count)

    @staticmethod
    def get_energy_barrier(paths_outcar: list[str]) -> float:
        """
        Compute the transition energy from a list of OUTCAR file paths.

        Parameters:
        paths_outcar (list[str]): List of paths to OUTCAR files.

        Returns:
        float: The maximum energy found among the OUTCAR files.

        Raises:
        IOError: If half or more of the OUTCAR files could not be read.
        ValueError: If the highest energy is found at the initial or final image.
        """
        energies = []
        paths_not_readable = []
        for path in paths_outcar:
            try:
                energies.append(ase_read(path).get_potential_energy())
            except:
                paths_not_readable.append(path)
        if len(paths_not_readable) >= len(paths_outcar) / 2:
            raise IOError(TrainingDataCreator.red_string_output(f"Ignored {'/'.join(paths_not_readable[0].split('/')[0:-1])}\t  ---> could not read half of the OUTCARs!"))
        # For "Proper" paths, there should be maximum in energy !between! initial and final paths... ignore those where this is not the case
        index_highest_energy = energies.index(max(energies))
        if index_highest_energy == 0 or index_highest_energy == len(paths_outcar) - 1:
            raise ValueError(TrainingDataCreator.red_string_output(f"Ignore {'/'.join(paths_outcar[0].split('/')[0:-1])}\n  ---> image {index_highest_energy} has highest energy!"))
        return max(energies)

    #             #using one structure to calc heat of mixing
    #             atoms_initial = ASEread(paths_outcar[0])
    #             # Compute heat of mixing per atom and append to list
    #             li_count = ideal_TS_atoms.get_chemical_symbols().count("Li") + 1 # +1 for the jumping Li
    #             o_count  = ideal_TS_atoms.get_chemical_symbols().count("O")
    #             h_o_m    = max(energies) - Li_count * E_ref_LiNiO2_per_O2 - (O_count/2-Li_count) * E_ref_NiO2_per_O2
    #             # total number of lattice sites: no(Oxygen) + [no(Li)=no(O)/2] + [no(Ni)=no(O)/2] = 2*no(O) --> Lattice Sites


    @staticmethod
    def test_for_outcars(path_without_run_final: str, no_of_steps: int) -> list[str]:
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
        paths_to_try = []
        # creation of expected paths
        paths_to_try.append(path_without_run_final + '/OUTCAR_initial_image')
        for i in range(1, no_of_steps): #todo: umschreiben zu glob.glob ?
            if i < 10:
                paths_to_try[i] = f"{abs_path}/0{i}/OUTCAR"
            else:
                paths_to_try[i] = f"{abs_path}/{i}/OUTCAR"
        paths_to_try.append(path_without_run_final + '/OUTCAR_final_image')

        # testing for existance ot those paths
        paths_outcar = []
        paths_missing = []
        for i in range(len(paths_to_try)):
            if os.path.exists(paths_to_try[i]):
                paths_outcar.append(paths_to_try[i])
            else:
                paths_missing.append(paths_to_try[i].replace(path_without_run_final, ''))
                if i == 1 or i == len(paths_to_try)-1:
                    raise AttributeError(f'{paths_to_try[i]} - does not exist but is vital \t PROCESS STOPPED \n')
        paths_missing = [s.replace(path_without_run_final, '') for s in paths_missing]
        if len(paths_missing) > 0:
            warnings.warn(
                f'In folder <{path_without_run_final}> the data of {paths_missing} is missing, {no_of_steps} transition points were expected \n',
                UserWarning)
        return (paths_outcar)


    @staticmethod
    def identify_jumping_atom(atoms_init: Atoms, atoms_final: Atoms, minimal_jump_distance: float = 2) -> int:
        """
        Identify the index of the atom that has moved the most between the initial and final atomic structures.

        This function applies the minimum image convention to calculate the displacement vector in relative coordinates
        for each atom. It then converts these displacements to Cartesian coordinates and identifies the atom with the
        largest displacement, which is assumed to be the jumping atom.

        Parameters:
        atoms_init (Atoms): The initial atomic structure.
        atoms_final (Atoms): The final atomic structure.
        minimal_jump_distance (float): The minimum distance in Ang an atom must move to be considered jumping. Default is 2 Ang.

        Returns:
        int: The index of the atom that has moved the most.

        Raises:
        Exception: If more than one atom has moved more than the minimal_jump_distance, indicating an issue with the NEB path.
        """
        # apply minimum image convention to get the displacement vector in relative coordinates for every atom
        positions_init  = atoms_init.get_scaled_positions()
        positions_final = atoms_final.get_scaled_positions()
        for i, (pos_init, pos_fin) in enumerate(zip(positions_init, positions_final)):
            for xyz in [0,1,2]:
                if pos_fin[xyz] - pos_init[xyz] > 0.5:
                    positions_final[i][xyz] = positions_final[i][xyz] - 1
                elif pos_fin[xyz] - pos_init[xyz] < -0.5:
                    positions_final[i][xyz] = positions_final[i][xyz] + 1
        rel_displacement_vector = positions_final - positions_init
        # get cell and compute the displacement vector in cartesian form
        (a,b,c) = atoms_init.get_cell()
        displacement_vector = []
        for (x, y, z) in rel_displacement_vector:
            displacement_vector.append(x*a+y*b+z*c)
        displacement_vector = np.array(displacement_vector)
        # get displacement magnitude and check for the ones with displacement larger than 2
        # which should be just the Li that jumps. Get its index
        displacements = np.sqrt(np.sum(displacement_vector ** 2, axis = 1))
        count = 0
        for i, displ in enumerate(displacements):
            if displ > minimal_jump_distance:
                count += 1
                jumping_atom_index = i
        # Check that there are not >1 atoms moving
        if count != 1:
            raise Exception(f"WARNING: There are {count} ions that move more than 2 Ang during the NEB! Check NEB path again!", UserWarning)
        else:
            return jumping_atom_index

    @staticmethod
    def create_anchor_structure(atoms_init: Atoms, atoms_final: Atoms) -> Atoms:
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

    @staticmethod
    def red_string_output(string: str) -> str:
        """
        Make the string red in the terminal output.
        Parameters:
        string (str): The string to be colored.
        Returns:
        str: The colored string.
        """
        return f"\33[91m{string}\33[0m"
    
    @property
    def e_ref_LiNiO2_per_O2(self):
        return self._e_ref_LiNiO2_per_O2
    @property
    def e_ref_NiO2_per_O2(self):
        return self._e_ref_NiO2_per_O2

