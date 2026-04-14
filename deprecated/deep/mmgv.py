"""
Legacy multi-modal graph vectorizer used by MMMTGNN.

Prefer :class:`novami.deep.vectorizer.GraphVectorizer` for new code with
:class:`novami.deep.dataset.MMDataset` / :class:`novami.deep.loader.MMLoader`.
"""

import copy
from typing import List

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import (ETKDGv3, EmbedMolecule, MMFFOptimizeMolecule,
                                MMFFGetMoleculeProperties, MMFFGetMoleculeForceField)
import torch
from torch_geometric.data import Data as Graph


class MMGV:
    """
    Transform RDKit molecules into ``torch_geometric`` graphs (legacy API).

    Parameters
    ----------
    atom_enc : dict, optional
        Mapping element name to index.
    bond_enc : dict, optional
        Mapping bond type name to index.
    suppress : bool, optional
        If True, suppress RDKit logs. Default is True.
    mode : str, optional
        ``'train'`` or ``'eval'``. Default is ``'train'``.
    """

    def __init__(self, atom_enc: dict = None, bond_enc: dict = None, suppress: bool = True, mode: str = 'train'):

        if atom_enc is None:
            self.atom_encoding = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'P': 5, 'Cl': 6, 'Mg': 7,
                                  'Na': 8, 'Br': 9, 'Fe': 10, 'Ca': 11, 'Cu': 12, 'Mc': 13, 'Pd': 14,
                                  'Pb': 15, 'K': 16, 'I': 17, 'Al': 18, 'Ni': 19, 'Mn': 20}
        else:
            self.atom_encoding = atom_enc

        self.groups = {
            0: ['H', 'C', 'N', 'O', 'P', 'S'],  # non_metals
            1: ['Li', 'Na', 'K', 'Rb', ' Cs', 'Fr'],  # alkaline metals
            2: ['Be', 'Mg', 'Ca', 'Sr', ' Ba', 'Ra'],  # alkaline earth metals
            3: ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Age', 'Cd',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Rf', 'Db', 'Sg', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'],  # transition metals
            4: ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Nh', 'Fl', 'Mc', 'Lv'],  # metals
            5: ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'Po'],  # metalloids
            6: ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],  # halogens
            7: ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'],  # noble gases
            8: ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],  # lanthanide
            9: ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']  # actinides
        }

        self.group_encoding = {}
        for group, elements in self.groups.items():
            for element in elements:
                self.group_encoding[element] = group

        self.atom_encoding_size = len(self.atom_encoding) + 1
        self.group_encoding_size = len(self.groups) + 1

        if bond_enc is None:
            self.bond_encoding = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
        else:
            self.bond_encoding = bond_enc

        self.bond_encoding_size = len(self.bond_encoding) + 1

        if suppress:
            RDLogger.DisableLog('rdApp.*')
        self.mode = mode
        self.embed_params = ETKDGv3()

    def encode_atom(self, atom):

        type_enc = np.zeros(shape=(self.atom_encoding_size,))
        type_enc[self.atom_encoding.get(atom.GetSymbol(), -1)] = 1

        group_enc = np.zeros(shape=(self.group_encoding_size,))
        group_enc[self.group_encoding.get(atom.GetSymbol(), -1)] = 1

        prop_enc = np.array([atom.GetFormalCharge(), atom.GetHybridization().real, atom.GetIsAromatic(),
                             atom.GetNumExplicitHs(), atom.GetDegree(), atom.IsInRing()])

        return np.hstack((type_enc, group_enc, prop_enc))

    def encode_mol_atoms(self, mol: Chem.rdchem.Mol) -> np.ndarray:

        atom_list = [self.encode_atom(atom) for atom in mol.GetAtoms()]
        atom_list.append(np.zeros(atom_list[0].shape))  # add a fake atom to the list

        atom_array = np.vstack(atom_list).astype(np.float64)

        return atom_array

    def encode_bond(self, bond):

        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_edges = [[start_idx, end_idx], [end_idx, start_idx]]  # connectivity

        bond_type = str(bond.GetBondType())
        type_enc = np.zeros(shape=(self.bond_encoding_size,))
        type_enc[self.bond_encoding.get(bond_type, -1)] = 1

        prop_enc = np.array([bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()])
        bond_enc = np.hstack((type_enc, prop_enc))  # properties

        return bond_edges, bond_enc

    def encode_mol_bonds(self, mol: Chem.rdchem.Mol):

        if len(mol.GetBonds()) == 0:
            return np.array([0, 0]).reshape(2, 1), np.zeros(shape=(1, self.bond_encoding_size + 3))

        edges = []  # of shape [2, num_edges]
        encoding = []  # of shape [num_edges, encoding_size]

        for bond in mol.GetBonds():
            bond_edges, bond_enc = self.encode_bond(bond)

            edges.extend(bond_edges)
            encoding.extend([bond_enc, bond_enc])

        virtual_encoding = np.zeros(encoding[0].shape)

        for i in range(num_atoms := len(mol.GetAtoms())):
            edges.extend([[i, num_atoms], [num_atoms, i]])
            encoding.extend([virtual_encoding, virtual_encoding])

        edge_array = np.array(edges).T.astype(np.float64)
        bond_array = np.vstack(encoding).astype(np.float64)

        return edge_array, bond_array

    def from_smiles(self, smiles: str, label: np.ndarray, weight: np.ndarray = None, kwargs: dict = None):
        """
        Build a single ``Data`` graph from SMILES and labels.

        Parameters
        ----------
        smiles : str
            SMILES string.
        label : np.ndarray
            Target values.
        weight : np.ndarray, optional
            Per-target weights. Default is uniform ones if omitted and ``kwargs`` allows.
        kwargs : dict, optional
            Extra 1D features merged into the graph batch.
        """

        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        atoms_encoding_array = self.encode_mol_atoms(mol)
        edges_array, bonds_encoding_array = self.encode_mol_bonds(mol)

        graph_data = {
            'x': torch.FloatTensor(atoms_encoding_array),
            'edge_index': torch.LongTensor(edges_array),
            'edge_attr': torch.FloatTensor(bonds_encoding_array),
            'y': torch.FloatTensor(label.reshape(1, -1)),
            'weights': torch.FloatTensor(weight.reshape(1, -1))
        }

        if kwargs is not None:
            for key, value in kwargs.items():
                graph_data[key] = torch.FloatTensor(value).reshape(1, -1)  # required by torch.cat

        return Graph(**graph_data)

    def from_lists(self, descriptor_list: List[dict], label_list: List[np.ndarray], weight_list: List[np.ndarray] = None):
        """
        Batch-convert records to graphs.

        Parameters
        ----------
        descriptor_list : list of dict
            Each dict must contain ``'SMILES'``; other keys are passed as ``kwargs``.
        label_list : list of np.ndarray
            Labels aligned with ``descriptor_list``.
        weight_list : list of np.ndarray, optional
            Weights aligned with labels; defaults to ones if omitted.
        """

        smiles_list = [entry['SMILES'] for entry in descriptor_list]
        metadata_list = [{k: v for k, v in entry.items() if k != 'SMILES'} for entry in descriptor_list]

        if weight_list is None:
            weight_list = [np.ones_like(label) for label in label_list]

        return [self.from_smiles(smiles=smi, label=lab, weight=wgt, kwargs=meta) for smi, lab, wgt, meta
                in zip(smiles_list, label_list, weight_list, metadata_list)]

    @staticmethod
    def validate_smiles(smiles: str):
        """
        Validate a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string to be validated.

        Returns
        -------
        bool
            True if the SMILES string is valid, False otherwise.
        """
        if not isinstance(smiles, str):
            return False

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is not None:
                return True
            else:
                return False
        except Exception as e:
            print(f'The following string is invalid: < {smiles} > due to {e}')
            return False

    def embed_molecule(self, mol, attempts: int = 10):
        """
        Attempt to find lowest-energy 3D coordinates from 2D molecule.
        """
        mol = Chem.AddHs(mol)
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
            mol = copy.deepcopy(mol)
            conf_id = EmbedMolecule(mol, self.embed_params)
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
            print(f'Optimization not possible for < {Chem.MolToSmiles(mol)} >')
            mol = copy.deepcopy(mol)
            EmbedMolecule(mol)
            return mol
