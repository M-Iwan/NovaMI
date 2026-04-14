"""
SMILES / SELFIES / DeepSMILES encoders and graph featurization for ``novami.deep``.

Active classes: :class:`GraphVectorizer`, :class:`StringVectorizer`. Legacy
:class:`MMGV` is loaded on demand via :func:`__getattr__` (see ``deprecated.deep.mmgv``).
"""
from collections import Counter
from typing import Union, Iterable, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import (ETKDGv3, EmbedMolecule, MMFFOptimizeMolecule,
                                MMFFGetMoleculeProperties, MMFFGetMoleculeForceField)
import re
import torch
import selfies as sf
import deepsmiles as ds
from torch_geometric.data import Data as Graph


class GraphVectorizer:
    """
    Encode molecules as ``torch_geometric.data.Data`` graphs from SMILES.

    Lighter-weight alternative to the legacy :class:`deprecated.deep.mmgv.MMGV`
    API; pairs with :class:`novami.deep.dataset.MMDataset` modality ``'graph'``.

    Parameters
    ----------
    atom_encoding : dict, optional
        Map element symbol to one-hot index; default covers common organic elements.
    bond_encoding : dict, optional
        Map RDKit bond type string to index.
    suppress : bool, optional
        If True, disable RDKit logs. Default is True.
    """

    def __init__(self, atom_encoding: dict = None, bond_encoding: dict = None,
                 suppress: bool = True):

        if atom_encoding is None:
            self.atom_encoding = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'P': 5, 'Cl': 6, 'Mg': 7,
                                  'Na': 8, 'Br': 9, 'Fe': 10, 'Ca': 11, 'Cu': 12, 'Mc': 13, 'Pd': 14,
                                  'Pb': 15, 'K': 16, 'I': 17, 'Al': 18, 'Ni': 19, 'Mn': 20}
        else:
            self.atom_encoding = atom_encoding

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

        if bond_encoding is None:
            self.bond_encoding = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
        else:
            self.bond_encoding = bond_encoding

        self.bond_encoding_size = len(self.bond_encoding) + 1

        if suppress:
            RDLogger.DisableLog('rdApp.*')
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

    def encode_mol_bonds(self, mol: Chem.rdchem.Mol) -> Tuple[np.ndarray, np.ndarray]:

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

    def from_smiles(self, smiles: str):
        """
        Change to work internally on np.ndarray. Missing values are expected to be nan
        """

        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        atoms_encoding_array = self.encode_mol_atoms(mol)
        edges_array, bonds_encoding_array = self.encode_mol_bonds(mol)

        graph_data = {
            'x': torch.FloatTensor(atoms_encoding_array),
            'edge_index': torch.LongTensor(edges_array),
            'edge_attr': torch.FloatTensor(bonds_encoding_array),
        }

        return Graph(**graph_data)

    def encode(self, smiles: Union[str, Iterable[str]]):
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]
        else:
            if hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
                return [self.from_smiles(item) for item in smiles]
            else:
                raise ValueError("Unsupported datatype passed. Expected smiles to be either string"
                                 "or iterable of strings")


class StringVectorizer:
    def __init__(self, alphabet: tuple = None, alphabet_type: str = 'smiles', max_length: int = None,
                 padding: bool = True, suppress: bool = True):

        self.alphabet = alphabet
        self.alphabet_type = alphabet_type
        if self.alphabet_type not in ['smiles', 'deepsmiles', 'selfies']:
            raise ValueError('Allowed options for alphabet are: smiles, deepsmiles, selfies')
        self.max_length = max_length
        self.padding = padding
        self.ds_converter = ds.Converter(branches=True, rings=True)
        if suppress:
            RDLogger.DisableLog('rdApp.*')

        r_atoms = r"Cl|Br|Si|Se|Na|Ca|Li|Mg|Zn|Fe|Cu|Mn|Hg|Sn|As|Bi|Cd|se|Cr|Sb"

        self.regex_patterns = {
            'smiles': re.compile(rf"(\[|]|{r_atoms}|[A-Z]|[a-z]|[=#/\\().+\-:]|\d)"),
            'deepsmiles': re.compile(rf"(\[|]|{r_atoms}|[A-Z]|[a-z]|[=#/\\().+\-:]|\)+|\(+|\d)"),
            'selfies': re.compile(r"\[.*?]")
        }
        self.char2idx = {char: idx for idx, char in enumerate(self.alphabet)} if self.alphabet is not None else None
        self.idx2char = {idx: char for idx, char in enumerate(self.alphabet)} if self.alphabet is not None else None

    def from_smiles(self, smiles: str):
        if self.char2idx is None:
            raise RuntimeError("Alphabet not initialized. Call prepare_alphabet to obtain it.")

        string = self.convert(smiles)
        string, length = self.split(string)

        if self.padding:
            string = self.pad(string, length)

        unk_idx = self.char2idx.get('<unk>', -1)
        array = np.array([self.char2idx.get(token, unk_idx) for token in string])
        tensor = torch.from_numpy(array).to(torch.int32).reshape(-1)
        return tensor, length

    def encode(self, smiles: Union[str, Iterable[str]]):
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]
        else:
            if hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
                return [self.from_smiles(item) for item in smiles]
            else:
                raise ValueError("Unsupported datatype passed. Expected smiles to be either string"
                                 "or iterable of strings")

    def decode(self, indices: List[int]):
        return ''.join(self.idx2char.get(i, '<unk>') for i in indices)

    def convert(self, smiles: str):
        if self.alphabet_type == 'smiles':
            return smiles
        elif self.alphabet_type == 'deepsmiles':
            return self.ds_converter.encode(smiles)
        elif self.alphabet_type == 'selfies':
            return sf.encoder(smiles)
        else:
            raise ValueError(f"Unsupported alphabet type: {self.alphabet_type}")

    def split(self, string):
        split_string = self.regex_patterns[self.alphabet_type].findall(string)
        length = len(split_string)

        if (self.max_length is not None) and (length > self.max_length):
            raise ValueError(f'Number of tokens in < {string} > [{len(split_string)}] exceeds allowed.')

        return split_string, length

    def pad(self, string, length):
        return string + ['<pad>'] * (self.max_length - length)

    def prepare_alphabet(self, smiles: List[str]):

        token_counter = Counter()

        for smi in smiles:
            string = self.convert(smi)
            tokens, _ = self.split(string)
            token_counter.update(tokens)

        alphabet = [token for token, _ in token_counter.most_common()] + ['<unk>']

        if self.padding:
            alphabet = ['<pad>'] + alphabet

        return alphabet


def __getattr__(name):
    """Re-export legacy :class:`MMGV` from ``deprecated.deep.mmgv``."""
    import warnings
    if name == "MMGV":
        warnings.warn(
            "MMGV has moved to deprecated.deep.mmgv; prefer GraphVectorizer for new code.",
            DeprecationWarning,
            stacklevel=2,
        )
        from deprecated.deep.mmgv import MMGV as _MMGV
        return _MMGV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")