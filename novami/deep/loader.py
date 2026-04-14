import warnings
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GeometricBatch

from novami.deep.dataset import MMDataset, MMBatch


class MMLoader:
    """
    Wraps torch.utils.data.DataLoader with a collate function that stacks or
    batches each modality (strings with masks, graphs via PyG Batch, tensors, groups).

    Parameters
    ----------
    dataset : MMDataset
        Source dataset.
    batch_size : int, optional
        Batch size. Default is 32.
    shuffle : bool, optional
        Whether to shuffle indices each epoch. Default is False.
    num_workers : int, optional
        Worker processes for the DataLoader. Default is 0.
    pin_memory : bool, optional
        Passed through to DataLoader. Default is False.
    drop_last : bool, optional
        Whether to drop the last incomplete batch. Default is False.
    **kwargs
        Any other keyword arguments forwarded to torch.utils.data.DataLoader.
    """

    def __init__(
            self,
            dataset: MMDataset,
            batch_size: int = 32,
            shuffle: bool = False,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = dataset.config
        self.modality_info = dataset.get_modality_info()

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
            **kwargs
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> MMBatch:
        """
        Stack list-of-dict samples into one MMBatch.

        Parameters
        ----------
        batch : list of dict
            Output of MMDataset.__getitem__ for each index in the batch.

        Returns
        -------
        out : MMBatch
            Collated tensors and structures; empty dict yields empty MMBatch.
        """
        if not batch:
            return MMBatch({}, self.config)

        collated_data = {}

        for col_name, modality in self.config.items():
            if col_name not in batch[0]:
                continue

            values = [sample[col_name] for sample in batch]

            if modality == 'string':
                tokens_list = [tokens for tokens, length in values]
                lengths_list = [length for tokens, length in values]

                stacked_tokens = torch.stack(tokens_list, dim=0)
                lengths_tensor = torch.tensor(lengths_list, dtype=torch.long)
                attention_mask = self._create_attention_mask(tokens_list, lengths_list)

                collated_data[col_name] = (stacked_tokens, lengths_tensor, attention_mask)

            elif modality == 'graph':
                collated_data[col_name] = self._collate_graphs(values)

            elif modality == 'target':
                collated_data['y_true'] = torch.stack(values, dim=0)

            elif modality in {'descriptor', 'image'}:
                collated_data[col_name] = torch.stack(values, dim=0)

            elif modality == 'sample_weight':
                collated_data['y_wgts'] = torch.stack(values, dim=0)

            elif modality == 'group':
                collated_data['group'] = values

        if 'y_true' in collated_data and 'y_wgts' not in collated_data:
            yt = collated_data['y_true']
            collated_data['y_wgts'] = torch.ones_like(yt, dtype=torch.float32)

        return MMBatch(collated_data, self.config)

    @staticmethod
    def _create_attention_mask(tokens_list: List[torch.Tensor], lengths_list: List[int]) -> torch.Tensor:
        """
        Boolean mask [batch, max_len] with True for valid token positions.

        Parameters
        ----------
        tokens_list : list of Tensor
            One 1D int tensor per row (variable length).
        lengths_list : list of int
            True length of each sequence (before padding).

        Returns
        -------
        mask : Tensor
            dtype bool, shape (batch_size, max_len).
        """
        batch_size = len(tokens_list)
        max_len = max(seq.size(0) for seq in tokens_list)

        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, length in enumerate(lengths_list):
            mask[i, :length] = True

        return mask

    @staticmethod
    def _collate_graphs(graphs: List) -> Any:
        """
        Batch a list of torch_geometric Data objects when possible.

        Parameters
        ----------
        graphs : list
            PyG Data instances.

        Returns
        -------
        batched : Batch or list
            Batch.from_data_list(graphs) on success; on failure, the original list
            and a warning.
        """
        try:
            return GeometricBatch.from_data_list(graphs)
        except Exception as e:
            warnings.warn(f"Failed to batch graphs with torch_geometric: {e}. Returning as list.")
            return graphs

    def __iter__(self):
        """Yield MMBatch instances from the underlying DataLoader."""
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        """Number of batches per epoch."""
        return len(self.dataloader)
