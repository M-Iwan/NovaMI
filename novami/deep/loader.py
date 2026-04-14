import warnings
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GeometricBatch

from novami.deep.dataset import MMDataset, MMBatch


class MMLoader:
    """
    Multi-Modal DataLoader that efficiently batches different modality types.

    Parameters
    ----------
    dataset: MMDataset
        The multi-modal dataset to load from
    batch_size: int, default=32
        Number of samples per batch
    shuffle: bool, default=False
        Whether to shuffle the data
    num_workers: int, default=0
        Number of worker processes for data loading
    pin_memory: bool, default=False
        Whether to use pinned memory for faster GPU transfer
    drop_last: bool, default=False
        Whether to drop the last incomplete batch
    **kwargs
        Additional arguments passed to PyTorch DataLoader
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

        # Create the underlying PyTorch DataLoader
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
        Custom collate function that handles different modality types efficiently.
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

        return MMBatch(collated_data, self.config)

    @staticmethod
    def _create_attention_mask(tokens_list: List[torch.Tensor], lengths_list: List[int]) -> torch.Tensor:
        """
        Create attention masks based on sequence lengths.
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
        Batch graph data using torch_geometric's batching.
        """
        try:
            # Use torch_geometric's batching
            return GeometricBatch.from_data_list(graphs)
        except Exception as e:
            warnings.warn(f"Failed to batch graphs with torch_geometric: {e}. Returning as list.")
            return graphs

    def __iter__(self):
        """Iterate over batches."""
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        return len(self.dataloader)