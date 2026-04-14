import torch
from torch.utils.data import Dataset
import polars as pl
from typing import Dict, Any, List, Optional, Union


class MMDataset(Dataset):
    """
    Multi-Modal Dataset that handles strings, graphs, descriptors, images, etc.
    
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
            - 'group': Any type for grouping
    """

    SUPPORTED_MODALITIES = {
        'string', 'graph', 'descriptor', 'image', 'target', 'sample_weight', 'group'
    }

    def __init__(self, df: pl.DataFrame, config: Dict[str, str]):
        self.df = df
        self.config = {}

        self._validate_config()

        self.feature_columns = []
        self.target_column = None
        self.weight_column = None
        self.group_column = None

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
                self.config['group'] = modality

        # Validate minimum requirements
        if not self.feature_columns:
            raise ValueError("At least one feature modality is required")
        if not self.target_column:
            raise ValueError("At least one target is required")

        # Validate all columns exist in DataFrame
        missing_cols = set(config.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    def _validate_config(self):
        """Validate the configuration dictionary"""
        
        unsupported = set(self.config.values()) - self.SUPPORTED_MODALITIES
        if unsupported:
            raise ValueError(f"Unsupported modalities: {unsupported}")

        modality_counts = {}
        for modality in self.config.values():
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        # Multiple instances allowed for these modalities
        multi_allowed = {'descriptor', 'string', 'graph', 'image'}
        for modality, count in modality_counts.items():
            if count > 1 and modality not in multi_allowed:
                raise ValueError(f"Modality '{modality}' can only be assigned to one column")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        """
        row = self.df.row(idx, named=True)
        sample = {}

        for col_name, modality in self.config.items():
            data = row[col_name]

            if modality == 'string':
                sample[col_name] = data
            if modality == 'target':
                sample['y_true'] = data
            elif modality in {'graph', 'descriptor', 'image'}:
                sample[col_name] = data
            elif modality == 'sample_weight':
                sample['y_wgts'] = data
            elif modality == 'group':
                sample['group'] = data

        return sample

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return self.feature_columns.copy()

    def get_target_column(self) -> str:
        """Get target column name"""
        return self.target_column

    def get_modality_info(self) -> Dict[str, List[str]]:
        """Get columns organized by modality type"""
        modality_info = {}
        for col_name, modality in self.config.items():
            if modality not in modality_info:
                modality_info[modality] = []
            modality_info[modality].append(col_name)
        return modality_info

    def has_sample_weights(self) -> bool:
        """Check if dataset has sample weights"""
        return self.weight_column is not None

    def has_groups(self) -> bool:
        """Check if dataset has group information"""
        return self.group_column is not None


class MMBatch:
    """
    Multi-modal batch container with dictionary-like access and automatic device transfer.
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
        """Get the modality type for a given key"""
        return self.config.get(key)
    
    def get_by_modality(self, modality: str) -> Dict[str, Any]:
        """Get all data of a specific modality type"""
        return {key: self.data[key] for key, mod in self.config.items() 
                if mod == modality and key in self.data}
    
    def to(self, device: Union[str, torch.device], non_blocking: bool = False):
        """
        Move all tensors in the batch to the specified device
        """
        new_data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.to(device, non_blocking=non_blocking)
            elif isinstance(value, tuple) and len(value) == 3:
                new_data[key] = tuple(
                    item.to(device, non_blocking=non_blocking) 
                    if isinstance(item, torch.Tensor) else item 
                    for item in value
                )
            elif hasattr(value, 'to'):  # torch_geometric.Data objects
                new_data[key] = value.to(device)
            else:
                new_data[key] = value
        return MMBatch(new_data, self.config)
    
    def __repr__(self):
        return f"MMBatch({list(self.data.keys())})"
