"""
Multi-modal Polars-backed :class:`torch.utils.data.Dataset` and batch container.
"""
import torch
from torch.utils.data import Dataset
import polars as pl
from typing import Dict, Any, List, Optional, Union


class MMDataset(Dataset):
    """
    Multi-modal dataset for strings, graphs, descriptors, images, targets, and weights.

    Parameters
    ----------
    df: pl.DataFrame
        A Polars DataFrame containing at least one feature type and target values.
    config: dict    
        A dictionary mapping column names to supported modality types:
            - 'string': Tuple[torch.Tensor (tokens), int (length)] from StringVectorizer
            - 'graph': torch_geometric.Data from GraphVectorizer
            - 'descriptor': 1D torch.Tensor [desc_size]
            - 'image': 2D/3D torch.Tensor
            - 'target': 1D torch.Tensor of shape [num_task]
            - 'sample_weight': 1D torch.Tensor; same shape as 'target'
            - 'group': String labels per row (e.g. ``numpy.array(['Small', 'Acid'])``).
              The source column is renamed to ``group`` in the dataframe; batch key is ``'group'``.
    """

    SUPPORTED_MODALITIES = {
        'string', 'graph', 'descriptor', 'image', 'target', 'sample_weight', 'group'
    }

    def __init__(self, df: pl.DataFrame, config: Dict[str, str]):
        self.df = df
        self.config: Dict[str, str] = {}
        self.feature_columns: List[str] = []
        self.target_column: Optional[str] = None
        self.weight_column: Optional[str] = None
        self.group_column: Optional[str] = None

        missing_cols = set(config.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        unsupported = set(config.values()) - self.SUPPORTED_MODALITIES
        if unsupported:
            raise ValueError(f"Unsupported modalities: {unsupported}")

        modality_counts: Dict[str, int] = {}
        for modality in config.values():
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        multi_allowed = {'descriptor', 'string', 'graph', 'image'}
        for modality, count in modality_counts.items():
            if count > 1 and modality not in multi_allowed:
                raise ValueError(f"Modality '{modality}' can only be assigned to one column")

        if modality_counts.get('target', 0) != 1:
            raise ValueError("Exactly one column must have modality 'target'")
        if not any(m in {'string', 'graph', 'descriptor', 'image'} for m in config.values()):
            raise ValueError("At least one feature modality is required")

        for col_name, modality in config.items():
            if modality in {'string', 'graph', 'descriptor', 'image'}:
                self.feature_columns.append(col_name)
                self.config[col_name] = modality
            elif modality == 'target':
                self.target_column = col_name
                self.config['y_true'] = modality
                self.df = self.df.rename({col_name: 'y_true'})
            elif modality == 'sample_weight':
                self.weight_column = col_name
                self.config['y_wgts'] = modality
                self.df = self.df.rename({col_name: 'y_wgts'})
            elif modality == 'group':
                self.group_column = col_name
                self.df = self.df.rename({col_name: 'group'})
                self.config['group'] = modality

        self._validate_config()

    def _validate_config(self):
        """Validate internal ``self.config`` after construction."""

        unsupported = set(self.config.values()) - self.SUPPORTED_MODALITIES
        if unsupported:
            raise ValueError(f"Unsupported modalities: {unsupported}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one sample as a dict keyed like ``self.config`` / standard batch keys.

        Returns
        -------
        dict
            Feature keys match ``self.config``; ``'y_true'`` / ``'y_wgts'`` / ``'group'``
            when present.
        """
        row = self.df.row(idx, named=True)
        sample: Dict[str, Any] = {}

        for col_name, modality in self.config.items():
            data = row[col_name]

            if modality == 'string':
                sample[col_name] = data
            elif modality in {'graph', 'descriptor', 'image'}:
                sample[col_name] = data
            elif modality == 'target':
                sample['y_true'] = data
            elif modality == 'sample_weight':
                sample['y_wgts'] = data
            elif modality == 'group':
                sample['group'] = data

        return sample

    def get_feature_columns(self) -> List[str]:
        """Return a copy of feature column names."""
        return self.feature_columns.copy()

    def get_target_column(self) -> Optional[str]:
        """Original target column name before rename to ``y_true``."""
        return self.target_column

    def get_group_column(self) -> Optional[str]:
        """Original group column name before rename to ``group``."""
        return self.group_column

    def get_modality_info(self) -> Dict[str, List[str]]:
        """Map modality strings to lists of column / batch keys."""
        modality_info: Dict[str, List[str]] = {}
        for col_name, modality in self.config.items():
            modality_info.setdefault(modality, []).append(col_name)
        return modality_info

    def has_sample_weights(self) -> bool:
        """True if a sample-weight column was provided."""
        return self.weight_column is not None

    def has_groups(self) -> bool:
        """True if a group column was provided."""
        return self.group_column is not None


class MMBatch:
    """
    Dict-like batch produced by :class:`novami.deep.loader.MMLoader` collate.

    Parameters
    ----------
    data : dict
        Batched tensors and other values.
    config : dict
        Same modality mapping as the source :class:`MMDataset` 
    """

    def __init__(self, data: Dict[str, Any], config: Dict[str, str]):
        self.data = data
        self.config = config

    def __getitem__(self, key: str):
        return self.data[key]

    def __contains__(self, key: str):
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def get_modality(self, key: str) -> Optional[str]:
        """Modality string for ``key``, if known."""
        return self.config.get(key)

    def get_by_modality(self, modality: str) -> Dict[str, Any]:
        """All batch entries whose modality equals ``modality``."""
        return {key: self.data[key] for key, mod in self.config.items()
                if mod == modality and key in self.data}

    def to(self, device: Union[str, torch.device], non_blocking: bool = False):
        """
        Move tensors (and ``torch_geometric`` objects) to ``device``.

        Parameters
        ----------
        device : str or torch.device
            Target device.
        non_blocking : bool, optional
            Passed to ``tensor.to(...)`` where applicable.

        Returns
        -------
        MMBatch
            New batch sharing the same ``config``.
        """
        new_data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.to(device, non_blocking=non_blocking)
            elif isinstance(value, tuple) and len(value) in (2, 3):
                new_data[key] = tuple(
                    item.to(device, non_blocking=non_blocking)
                    if isinstance(item, torch.Tensor) else item
                    for item in value
                )
            elif hasattr(value, 'to'):
                new_data[key] = value.to(device)
            else:
                new_data[key] = value
        return MMBatch(new_data, self.config)

    def __repr__(self):
        return f"MMBatch({list(self.data.keys())})"
