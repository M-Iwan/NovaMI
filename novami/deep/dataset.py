"""
TODO: Force BF16 precision
"""

from functools import partial
from typing import Union, List
from collections import defaultdict


import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch_geometric
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Batch


class StringDataset(TorchDataset):
    def __init__(self, strings: List[str], labels: Union[List, Tensor], lengths: Union[List[int], Tensor]):
        """
        TODO: update docstring, add collate_fn to calculate lengths; move RecurrentVectorizer from model to here
        TODO: include string processing within __getitem__ method
        """
        super(StringDataset).__init__()
        self.strings = strings
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.strings)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the vectorized SMILES string and its corresponding label.
        """

        string = self.strings[idx]
        label = self.labels[idx]
        length = self.lengths[idx]

        return string, label, length


class GraphDataset(GeometricDataset):
    def __init__(self, data: List[torch_geometric.data.Data]):
        """
        Raw version, but working
        """
        super(GraphDataset).__init__()
        self.data = data

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        """
        return self.data[idx]


class NumpyDataset(TorchDataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        TODO: update docstring
        Dataset class for handling numpy input data.
        """

        super(NumpyDataset).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index as a torch tensor.
        """

        x_tens = torch.FloatTensor(self.x[idx, :])
        y_tens = torch.FloatTensor(self.y[idx, :])

        return x_tens, y_tens

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class DeepDataset(TorchDataset):
    def __init__(self, dataframe, descriptor_cols: Union[List[str], str], label_col: str, weight_col: str = None,
                 signature_col: str = None, ttype: torch.DataType = torch.bfloat16):
        """
        descriptors: List[Dict[str, Any]]
        labels: np.ndarray or torch.Tensor
        sample_weights: np.ndarray or torch.Tensor
        """

        if isinstance(descriptor_cols, str):
            descriptor_cols = [descriptor_cols]

        for col in descriptor_cols:
            value = dataframe[col].iloc[0]
            if isinstance(value, np.ndarray):
                dataframe[col] = dataframe[col].apply(lambda array: torch.from_numpy(array).to(ttype).reshape(-1))

        self.descriptors = dataframe[descriptor_cols].to_dict(orient='records')  # List[dict]
        label_array = np.vstack(dataframe[label_col].to_numpy())

        self.labels = torch.from_numpy(label_array).to(ttype)

        if weight_col is not None:
            weights_array = np.vstack(dataframe[weight_col].to_numpy())
            self.weights = torch.from_numpy(weights_array).to(ttype)
        else:
            self.weights = torch.ones_like(self.labels).to(ttype)

        if signature_col is not None:
            self.signatures = dataframe[signature_col].tolist()
        else:
            self.signatures = [None] * len(dataframe)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.descriptors[idx].copy()
        sample['Label'] = self.labels[idx, :]
        sample['Weight'] = self.weights[idx, :]
        sample['Signature'] = self.signatures[idx]
        return AttrDict(sample)


def deep_collate(batch, sign_names: List[str]):
    collated = AttrDict({
        'Graph': [],
        'String_tokens': [],
        'String_lengths': [],
        'Label': [],
        'Weight': [],
        'Signature': defaultdict(list)
    })

    present_keys = batch[0].keys()
    other_fields = defaultdict(list)

    for sample in batch:
        if 'Graph' in present_keys:
            collated['Graph'].append(sample['Graph'])

        if 'String' in present_keys:
            string_tensor, token_len = sample['String']  # already tensor and int
            collated['String_tokens'].append(string_tensor)
            collated['String_lengths'].append(token_len)

        collated['Label'].append(sample['Label'])
        collated['Weight'].append(sample['Weight'])

        for sign, name in zip(sample['Signature'], sign_names):
            collated.Signature[name].append(sign)

        for key, value in sample.items():
            if key not in ['Graph', 'String', 'Label', 'Weight', 'Signature']:
                other_fields[key].append(value)

    if 'Graph' in present_keys:
        collated['Graph'] = Batch.from_data_list(collated['Graph'])

    if 'String' in present_keys:
        string_tensor = torch.stack(collated['String_tokens'])  # (B, max_seq_len)
        string_lengths = torch.tensor(collated['String_lengths'])  # (B,)
        collated['String'] = (string_tensor, string_lengths)

    del collated['String_tokens']
    del collated['String_lengths']

    collated['Label'] = torch.stack(collated['Label'])
    collated['Weight'] = torch.stack(collated['Weight'])

    for key, values in other_fields.items():
        collated[key] = torch.stack(values)

    return collated


class DeepLoader(DataLoader):
    def __init__(self, dataframe, descriptor_cols: Union[List[str], str], label_col: str,
                 weight_col: str = None, signature_col: str = None, signature_names: List[str] = None,
                 batch_size: int = 64, shuffle: bool = True, **kwargs):
        dataset = DeepDataset(
            dataframe=dataframe,
            descriptor_cols=descriptor_cols,
            label_col=label_col,
            weight_col=weight_col,
            signature_col=signature_col,
        )

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=partial(deep_collate, sign_names=signature_names), **kwargs)