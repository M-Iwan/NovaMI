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
    Turn SMILES into torch_geometric Data (x, edge_index, edge_attr).

    Simpler than the legacy MMGV class in deprecated.deep.mmgv; use with MMDataset
    modality graph.

    Parameters
    ----------
    atom_encoding : dict, optional
        Element symbol to index; default covers common organic elements.
    bond_encoding : dict, optional
        Bond type string to index; default SINGLE, DOUBLE, TRIPLE, AROMATIC.
    suppress : bool, optional
        If True, silence RDKit logs. Default is True.
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
        Build one unlabeled graph Data object from a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES parsed with RDKit sanitize=True.

        Returns
        -------
        data : torch_geometric.data.Data
            Node features, edge_index, edge_attr.
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
        """
        Vectorize one SMILES or many; returns a list of Data graphs.

        Parameters
        ----------
        smiles : str or iterable of str

        Returns
        -------
        list
            torch_geometric Data instances in input order.
        """
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]
        else:
            if hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
                return [self.from_smiles(item) for item in smiles]
            else:
                raise ValueError("Unsupported datatype passed. Expected smiles to be either string"
                                 "or iterable of strings")


class StringVectorizer:
    """
    Tokenize SMILES, DeepSMILES, or SELFIES into integer indices for embedding layers.

    Call prepare_alphabet on a corpus before from_smiles if alphabet is None.

    Parameters
    ----------
    alphabet : tuple, optional
        Token strings in vocabulary order; if None, set later via prepare_alphabet.
    alphabet_type : str, optional
        One of smiles, deepsmiles, selfies. Default is smiles.
    max_length : int, optional
        Maximum token count after split; longer strings raise ValueError.
    padding : bool, optional
        If True, pad sequences to max_length with pad token when alphabet is built.
    suppress : bool, optional
        If True, silence RDKit logs. Default is True.
    """
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
        """
        Encode one string to (int32 token tensor, int length).

        Parameters
        ----------
        smiles : str
            Input in the format given by alphabet_type.

        Returns
        -------
        tensor : torch.Tensor
            int32, shape (seq_len,) or (max_length,) if padding.
        length : int
            True token count before padding.
        """
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
        """
        Same as from_smiles but accepts a str or iterable of str; always returns a list.

        Parameters
        ----------
        smiles : str or iterable of str

        Returns
        -------
        list [tensor, length]
        """
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]
        else:
            if hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
                return [self.from_smiles(item) for item in smiles]
            else:
                raise ValueError("Unsupported datatype passed. Expected smiles to be either string"
                                 "or iterable of strings")

    def decode(self, indices: List[int]):
        """
        Map token indices back to a string (best-effort for SMILES-like alphabets).

        Parameters
        ----------
        indices : list of int
            Token indices.

        Returns
        -------
        str
            Concatenated characters; unknown indices become unk.
        """
        return ''.join(self.idx2char.get(i, '<unk>') for i in indices)

    def convert(self, smiles: str):
        """
        Normalize input to the token string form used by split (SMILES as-is, etc.).

        Parameters
        ----------
        smiles : str

        Returns
        -------
        str
            SMILES, DeepSMILES, or SELFIES string depending on alphabet_type.
        """
        if self.alphabet_type == 'smiles':
            return smiles
        elif self.alphabet_type == 'deepsmiles':
            return self.ds_converter.encode(smiles)
        elif self.alphabet_type == 'selfies':
            return sf.encoder(smiles)
        else:
            raise ValueError(f"Unsupported alphabet type: {self.alphabet_type}")

    def split(self, string):
        """
        Regex tokenization and length; enforces max_length when set.

        Parameters
        ----------
        string : str
            Already converted (e.g. via convert).

        Returns
        -------
        tokens : list
            Token strings.
        length : int
            len(tokens).
        """
        split_string = self.regex_patterns[self.alphabet_type].findall(string)
        length = len(split_string)

        if (self.max_length is not None) and (length > self.max_length):
            raise ValueError(f'Number of tokens in < {string} > [{len(split_string)}] exceeds allowed.')

        return split_string, length

    def pad(self, string, length):
        """
        Append pad tokens so token list length equals max_length.

        Parameters
        ----------
        string : list
            Token list after split.
        length : int
            Original length before padding.

        Returns
        -------
        list
            Padded token list.
        """
        return string + ['<pad>'] * (self.max_length - length)

    def prepare_alphabet(self, smiles: List[str]):
        """
        Build vocabulary from corpus token frequency (most common first).

        Parameters
        ----------
        smiles : list of str
            Corpus of SMILES (or converted strings) for counting.

        Returns
        -------
        alphabet : list
            Token list with unk and optional pad prepended when padding is True.
        """
        token_counter = Counter()

        for smi in smiles:
            string = self.convert(smi)
            tokens, _ = self.split(string)
            token_counter.update(tokens)

        alphabet = [token for token, _ in token_counter.most_common()] + ['<unk>']

        if self.padding:
            alphabet = ['<pad>'] + alphabet

        return alphabet
