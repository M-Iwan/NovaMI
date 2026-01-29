""" File with classes and functions related to computational-chemistry related tasks """
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import (ETKDGv3, EmbedMolecule, MMFFOptimizeMolecule,
                                MMFFGetMoleculeProperties, MMFFGetMoleculeForceField)
from rdkit.Chem.MolStandardize import rdMolStandardize
import prolif as plf
import os
import glob
import re
import pandas as pd
import MDAnalysis as mda


class Molecule:

    def __init__(self, smiles: str, mol_id: str):

        self.smiles = smiles
        self.mol_id = mol_id
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.dock_params = {}
        self.dock_score = None
        self.protein_path = None
        self.docked_path = None
        self.original_path = None
        self.hydrogen_path = None
        self.split_paths = []
        self.pli_df = None
        self.pli = False
        self.tautomers = None

    def __str__(self):
        return f'Molecule object of < {self.smiles} >'

    def __repr__(self):
        return f'Molecule object of < {self.smiles} > with id < {self.mol_id} >'

    def embed(self, attempts: int = 10):
        """
        Attempt to find lowest-energy 3D coordinates from 2D molecule.
        """
        self.mol = Chem.AddHs(self.mol)
        params = ETKDGv3()
        mols = []
        energies = []

        def optimize_molecule(mol_, conf_id_):
            ex_code = MMFFOptimizeMolecule(mol_, maxIters=500, confId=conf_id_)
            match ex_code:
                case -1:  # optimization not possible, return whatever molecule was embedded
                    print('Could not setup the force field for provided molecule')
                    return mol_, np.nan
                case 0:  # optimization converged, proceed normally
                    prop = MMFFGetMoleculeProperties(mol_)
                    ff = MMFFGetMoleculeForceField(mol_, prop)
                    energy_ = ff.CalcEnergy()
                    return mol_, energy_
                case 1:  # force field setup correct, but convergence not reached
                    _ = MMFFOptimizeMolecule(mol_, maxIters=5000, confId=conf_id_)
                    prop = MMFFGetMoleculeProperties(mol_)
                    ff = MMFFGetMoleculeForceField(mol_, prop)
                    energy_ = ff.CalcEnergy()
                    return mol_, energy_

        for attempt in range(attempts):
            mol = copy.deepcopy(self.mol)
            conf_id = EmbedMolecule(mol, params)
            if conf_id >= 0:  # in case of correct embedding the returned number is not negative
                mol, energy = optimize_molecule(mol, conf_id)
                mols.append(mol)
                energies.append(energy)
            else:
                conf_id = EmbedMolecule(mol, maxAttempts=1000, useRandomCoords=True)
                if isinstance(conf_id, int):  # check if other embedding works
                    mol, energy = optimize_molecule(mol, conf_id)
                    mols.append(mol)
                    energies.append(energy)
                else:
                    return mol  # return unoptimized mol

        if not all([x is np.nan for x in energies]):  # if there is at least one correct energy
            min_id = np.nanargmin(energies)
            return mols[min_id]  # return mol with the lowest energy
        else:
            print(f'Optimization not possible for < {self.smiles} >')
            mol = copy.deepcopy(self.mol)
            EmbedMolecule(mol)
            return mol  # return original molecule

    def validate_mol(self):
        raise NotImplementedError

    def show(self):
        return Draw.MolToImage(self.mol, size=(512, 512))

    def dock(self, dock_params: dict, overwrite: bool = False):

        output = dock_params['output_dir'].rstrip('/')

        os.makedirs(output, exist_ok=True)  # make dir if it's not there yet

        mol_in = f'{output}/mol_{self.mol_id}.mol'
        mol_out = f'{output}/mol_{self.mol_id}.mol2'

        if glob.glob(mol_in) and not overwrite:
            print(f'Molecule file mol_{self.mol_id}.mol already exists.')
        else:
            self.mol = self.embed()
            Chem.MolToMolFile(self.mol, mol_in)
            self.original_path = mol_in
            convert_command = f'obabel -imol {mol_in} -omol2 -O {mol_out}'
            os.system(convert_command)

            smina = dock_params['smina_path']
            protein = dock_params['protein_path']
            mol2_in = f'{output}/mol_{self.mol_id}.mol2'
            mol2_out = f'{output}/mol_{self.mol_id}_out.mol2'
            log_out = f'{output}/mol_{self.mol_id}_out.log'

            x_coord = dock_params['x_coord']
            x_size = dock_params['x_size']

            y_coord = dock_params['y_coord']
            y_size = dock_params['y_size']

            z_coord = dock_params['z_coord']
            z_size = dock_params['z_size']

            exh = dock_params['exhaustiveness']

            dock_command = (f'{smina} -r {protein} -l {mol2_in} --center_x {x_coord} --center_y {y_coord} --center_z {z_coord} '
                            f'--size_x {x_size} --size_y {y_size} --size_z {z_size} --exhaustiveness {exh} --out {mol2_out};')

            os.system(dock_command)

            output_command = f'{smina} -r {protein} -l {mol2_out} --score_only --log {log_out}'

            os.system(output_command)

            output = open(log_out, 'r').read()
            score = np.min([float(score) for score in re.findall(r'Affinity:\s*([\-?+\d.]+)\s+\(kcal/mol\)', output)])

            with open(log_out, 'a') as file:
                file.write(f'--- Docking parameters ---\n')
                for name, value in dock_params.items():
                    file.write(f'< {name} > : < {value} >\n')

            if isinstance(score, float):
                print(f'Docking successful')
                self.dock_params = dock_params
                self.dock_score = score
                self.protein_path = protein
                self.docked_path = mol2_out

    def read_dock(self, output_dir: str):

        log_file = glob.glob(f"{output_dir.rstrip('/')}/mol_{self.mol_id}_out.log")

        if len(log_file) == 0:
            print('Molecule not yet docked')
            self.dock_score = np.nan
        else:
            output = open(log_file[0], 'r').read()
            self.dock_score = np.min([float(score) for score in re.findall(r'Affinity:\s*([\-?+\d.]+)\s+\(kcal/mol\)', output)])
            self.docked_path = glob.glob(f"{output_dir.rstrip('/')}/mol_{self.mol_id}_out.mol2")[0]
            self.original_path = glob.glob(f"{output_dir.rstrip('/')}/mol_{self.mol_id}.mol")[0]
            output_list = output.split('\n')
            for line in output_list:
                if (matches := re.search(r'\A<\s(.+)\s>\s:\s<\s(.+)\s>\Z', line)) is not None:  # new!
                    self.dock_params[matches.group(1)] = matches.group(2)
            self.protein_path = self.dock_params['protein_path']

    def calculate_pli(self, count: bool = False, file_path: str = None):
        """
        Calculate protein-ligand interaction fingerprint using prolif library.

        Parameters
        ----------
        count: bool
            Whether to return count fingerprint or bit fingerprint. Defaults to False
        file_path: str
            If not None, use provided path to file instead of self.docked_path
        """

        def split_mol_file(path_):
            """
            plf has issues accepting conformers from RDKit so the original file is split into separate .mol files,
            and processed individually
            """
            split_indices_ = []
            file_ = open(path_, 'r').readlines()
            for i_, line_ in enumerate(file_):
                if '*****' in line_:  # used to define a new conformation
                    split_indices_.append(i_)
            split_indices_.append(len(file_))
            for j_ in range(len(split_indices_)-1):
                new_path_ = path_.rstrip('.mol') + f'_{j_}.mol'
                new_file_ = file_[split_indices_[j_]: split_indices_[j_+1]]
                with open(new_path_, 'w') as f:
                    for line_ in new_file_:
                        f.write(line_)
                self.split_paths.append(new_path_)

        if not all([self.dock_score is not None, self.protein_path is not None, self.docked_path is not None]):
            print(f'Molecule not docked successfully')
        else:
            assert self.protein_path.split('.')[-1] == 'pdb'  # check protein file extension
            assert self.docked_path.split('.')[-1] == 'mol2'  # check molecule file extension
            """
            # alternative solution using rdkit
            rdkit_protein = Chem.MolFromPDBFile(self.protein_path, removeHs=False)  # read protein file
            plf_protein = plf.Molecule(rdkit_protein)  # potentially change back to plf.Molecule(rdkit_protein)
            """
            u = mda.Universe(self.protein_path)
            plf_protein = plf.Molecule.from_mda(u)

            if file_path is None:  # proceed as usual:
                # OpenBabel sometimes would not add hydrogen atoms to .mol2 file, so it's first converted to .mol file
                conversion_path = self.docked_path.rstrip('2')
                self.hydrogen_path = self.docked_path.rstrip('.mol2') + '_H.mol'
                convert_command_1 = f"obabel -imol2 {self.docked_path} -omol -O {conversion_path}"
                convert_command_2 = f'obabel -imol {conversion_path} -omol -O {self.hydrogen_path} -p 7.4'
                os.system(convert_command_1)
                os.system(convert_command_2)
                os.system(f'rm {conversion_path}')
            else:  # use manually prepared file; expected to be .mol file with hydrogen atoms
                self.hydrogen_path = file_path

            split_mol_file(self.hydrogen_path)

            for i, path in enumerate(self.split_paths):
                try:
                    ifp = plf.Fingerprint()
                    rdkit_mol = Chem.MolFromMolFile(path, removeHs=False)  # read .mol file
                    # noinspection PyTypeChecker
                    ifp.run_from_iterable([plf.Molecule.from_rdkit(rdkit_mol)], plf_protein)  # calculate
                    ifp_df = ifp.to_dataframe(index_col='Pose', drop_empty=False, count=count)  # pass to dataframe
                    ifp_df = pd.melt(ifp_df)  # remove multi-indexing
                    ifp_df['dock_number'] = i

                    if self.pli_df is None:
                        self.pli_df = ifp_df
                    else:
                        self.pli_df = pd.concat([self.pli_df, ifp_df], axis=0, ignore_index=True)

                    os.system(f'rm {path}')

                except AttributeError as e:
                    print(f'Could not calculate IFP due to:\n{e}')

        if self.pli_df is not None:  # combine and clean dfs
            self.pli_df = self.pli_df.drop(columns=['ligand'])
            self.pli_df['Mol_ID'] = self.mol_id
            self.pli_df['SMILES'] = self.smiles
            self.pli_df = self.pli_df.rename(columns={'protein': 'residue'})
            self.pli = True

    def enumerate_tautomers(self):
        enumerator = rdMolStandardize.TautomerEnumerator()
        canon = enumerator.canonicalize(self.mol)
        csmi = Chem.MolToSmiles(canon)
        res = [canon]
        tautomers = enumerator.Enumerate(self.mol)
        smis = [Chem.MolToSmiles(x) for x in tautomers]
        stpl = sorted((x, y) for x,y in zip(smis, tautomers) if x != csmi)
        res += [y for x, y in stpl]
        self.tautomers = res

    def enumerate_conformers(self):
        raise NotImplementedError
