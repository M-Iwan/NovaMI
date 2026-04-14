"""
TODO: Force BF16 precision
"""

from functools import partial
from typing import Union, List, Dict, Any
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data as Graph
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.data import Batch
import polars as pl


class StringDataset(TorchDataset):
    def __init__(self, strings: List[str], labels: Union[List, Tensor], lengths: Union[List[int], Tensor]):
        """
        TODO: update docstring, add collate_fn to calculate lengths; move RecurrentVectorizer from model to here
        TODO: include string processing within __getitem__ method
        """
        super().__init__()
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


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class GraphDataset(TorchDataset):
    def __init__(self, df: pl.DataFrame, device: str = 'cpu', graph_col: str = 'Graph', label_col: str = 'pIC50',
                 weights_col: str = None, n_tasks: int = 1, ttype: torch.dtype = torch.bfloat16):
        """
        Dataset class for handling graph-based data.

        Parameters
        ----------
        df: pl.DataFrame
            A Polars DataFrame with torch_geometric Graphs
        device: str
            'cpu' or 'cuda'
        graph_col: str
            Name of the column with Graphs
        label_col: str
            Name of the column with target values
        weights_col: str
            Name of the column with sample weights
        n_tasks: int
            Number of tasks
        ttype
            Tensor type to use. Default is bfloat16
        """
        super().__init__()

        if graph_col not in df.columns:
            raise KeyError(f"Graph column '{graph_col}' not found in dataframe")

        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in dataframe")

        if weights_col is not None:
            if weights_col not in df.columns:
                raise KeyError(f"Weight column '{weights_col}' not found in dataframe")

        self.device = device
        self.n_tasks = n_tasks
        self.ttype = ttype
        self.graphs = df[graph_col].to_list()

        sample_graph = self.graphs[0]
        if not isinstance(sample_graph, Graph):
            raise TypeError(f"Expected torch_geometric.data.Data objects, got {type(sample_graph)}")

        labels = np.vstack(df[label_col].to_numpy()).reshape(-1, self.n_tasks)
        self.labels = torch.from_numpy(labels).to(dtype=self.ttype, device=self.device)

        if weights_col is not None:
            weights = np.vstack(df[weights_col].to_numpy()).reshape(-1, self.n_tasks)
            self.weights = torch.from_numpy(weights).to(dtype=self.ttype, device=self.device)
        else:
            self.weights = torch.ones_like(self.labels).to(ttype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[Graph, torch.Tensor]]:
        return {
            "Graph": self.graphs[idx],
            "Label": self.labels[idx, :],
            "Weight": self.weights[idx, :]
        }


class GraphLoader(DataLoader):
    def __init__(self, dataset: GraphDataset, batch_size: int = 64, shuffle: bool = False,
                 num_workers: int = 0, pin_memory: bool = False, **kwargs):
        """
        DataLoader for GraphDataset.

        Parameters
        ----------
        dataset: GraphDataset
            The dataset to load from
        batch_size: int
            Number of samples per batch
        shuffle: bool
            Whether to shuffle the data
        num_workers: int
            Number of subprocesses for data loading
        pin_memory: bool
            Whether to pin memory for faster GPU transfer
        **kwargs
            Additional arguments passed to torch DataLoader
        """

        # Custom collate function for handling PyTorch Geometric graphs
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
            **kwargs
        )

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to batch PyTorch Geometric graphs and stack tensors.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of sample dictionaries from the dataset

        Returns
        -------
        Dict[str, torch.Tensor]
            Batched dictionary with:
            - Graph data batched using PyTorch Geometric's Batch
            - Labels and weights stacked as tensors
        """

        graphs = [item["Graph"] for item in batch]
        labels = [item["Label"] for item in batch]
        weights = [item["Weight"] for item in batch]

        batched_graph = Batch.from_data_list(graphs)

        batched_labels = torch.stack(labels, dim=0)
        batched_weights = torch.stack(weights, dim=0)

        return {
            "Graph": batched_graph,
            "Label": batched_labels,
            "Weight": batched_weights
        }


class DeepDataset(TorchDataset):
    def __init__(self, dataframe, descriptor_cols: Union[List[str], str], label_col: str, weight_col: str = None,
                 signature_col: str = None, ttype: torch.dtype = torch.bfloat16):
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