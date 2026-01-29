"""
version = 0.1.0
date = 25.06.2024
author = M-Iwan
"""

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import ETKDGv3, EmbedMolecule, MMFFOptimizeMolecule, MMFFGetMoleculeProperties, MMFFGetMoleculeForceField
import os
import re
import glob
import json
import copy

parser = argparse.ArgumentParser(prog='Dock',
                                 description='Command line implementation of SMINA for docking')

parser.add_argument('-i', '--input', help='Absolute path to an input file')
parser.add_argument('-o', '--output', help='Absolute path to an output file')
parser.add_argument('-c', '--conf', help="Absolute path to .json configuration file")
parser.add_argument('-s', '--smiles_col', help="Name of the column holding SMILES.")
parser.add_argument('-n', '--mol_id', help='Name of the column with Mol_ID')
parser.add_argument('-d', '--dock_col', help='Name of the column where to save the results')

parsed_args = parser.parse_args()


def main(args):
    """ File with classes and functions related to computational-chemistry related tasks """

    class Molecule:

        def __init__(self, smiles: str, mol_id: str):

            self.smiles = smiles
            self.mol_id = mol_id
            self.mol = Chem.MolFromSmiles(self.smiles)
            self.dock_score = None

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

        def show(self):
            return Draw.MolToImage(self.mol, size=(512, 512))

        def dock(self, dock_params: dict):

            output = dock_params['output_dir'].rstrip('/')

            os.makedirs(output, exist_ok=True)  # make dir if it's not there yet

            mol_in = f'{output}/mol_{self.mol_id}.mol'
            mol_out = f'{output}/mol_{self.mol_id}.mol2'

            if glob.glob(mol_in):
                print(f'Molecule file mol_{self.mol_id}.mol already exists.')
            else:
                self.mol = self.embed()
                Chem.MolToMolFile(self.mol, mol_in)
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

                self.dock_score = score

        def read_dock(self, output_dir: str):

            log_file = glob.glob(f"{output_dir.rstrip('/')}/mol_{self.mol_id}_out.log")

            if len(log_file) == 0:
                print('Molecule not yet docked')
                self.dock_score = np.nan
            else:
                output = open(log_file[0], 'r').read()
                self.dock_score = np.min([float(score) for score in re.findall(r'Affinity:\s*([\-?+\d.]+)\s+\(kcal/mol\)', output)])

    def read_df(path):
        ext = path.split('.')[-1]
        if ext == 'xlsx':
            df = pd.read_excel(path)
        elif ext == 'csv':
            df = pd.read_csv(path)
        elif ext == 'parquet':
            df = pd.read_parquet(path)
        elif ext == 'pkl':
            df = pd.read_pickle(path)
        else:
            raise ValueError('Incorrect extension')

        return df

    def write_df(df, path):
        ext = path.split('.')[-1]
        if len(df) >= 1:
            if ext == 'xlsx':
                df.to_excel(path, index=False)
            elif ext == 'csv':
                df.to_csv(path, index=False, header=True)
            elif ext == 'parquet':
                df.to_parquet(path)
            elif ext == 'pkl':
                df.to_pickle(path)
            else:
                raise ValueError('Incorrect extension')

    def dock_score(row, smiles_col_, mol_id_col_, dock_params_):
        smiles = row[smiles_col_]
        mol_id = row[mol_id_col_]
        mol = Molecule(smiles,  mol_id)
        mol.dock(dock_params_)
        score = mol.dock_score
        return score

    input_path = args.input
    output_path = args.output
    config_path = args.conf
    smiles_col = args.smiles_col
    mol_id_col = args.mol_id
    dock_col = args.dock_col

    data = read_df(input_path)

    dock_params = json.load(open(config_path, 'r'))

    if mol_id_col not in data.columns:
        data[mol_id_col] = np.arange(0, len(data))

    data[dock_col] = data.apply(dock_score, smiles_col_=smiles_col, mol_id_col_=mol_id_col, dock_params_=dock_params, axis=1)

    write_df(data, output_path)


if __name__ == '__main__':
    main(parsed_args)
