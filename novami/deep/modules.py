"""
Reusable GNN / CNN / RNN blocks and layer factories for ``novami.deep`` models.
"""
import inspect
from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GATv2Conv, EdgeConv

from novami.deep.utils import get_activation_fn


class GNNModule(nn.Module):
    """
    One message-passing block: conv → optional norm → activation → dropout.

    Parameters
    ----------
    graph_layer : torch.nn.Module
        Typically a PyG conv layer.
    batch_norm : torch.nn.Module or None
    activation : torch.nn.Module or None
    dropout : torch.nn.Module or None
    """
    def __init__(self, graph_layer, batch_norm, activation, dropout):
        super().__init__()
        self.graph_layer = graph_layer
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.accepts_edge_attr = 'edge_attr' in inspect.signature(self.graph_layer.forward).parameters

    def forward(self, graph_batch):

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = getattr(graph_batch, 'edge_attr', None)

        if self.accepts_edge_attr and edge_attr is not None:
            x = self.graph_layer(x, edge_index, edge_attr)
        else:
            x = self.graph_layer(x, edge_index)

        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)

        return x


def build_graph_layers(gnn_params):
    """
    Stack :class:`GNNModule` layers from a parameter dict.

    Parameters
    ----------
    gnn_params : dict
        Must contain ``layer``, ``layer_type``, ``sizes``, ``input_dim``, and
        per-layer ``args`` (list of dicts). Optional: ``activation``, ``dropout``,
        ``batch_norm``, ``heads`` (attention).

    Returns
    -------
    layers : nn.ModuleList
    out_dim : int
        Output channel width after the last layer.
    """
    layer_class = gnn_params['layer']
    layer_type = gnn_params.get('layer_type')  # default
    sizes = gnn_params['sizes']
    input_dim = gnn_params['input_dim']
    layer_args = gnn_params.get('args', [{} for _ in sizes])
    activation = gnn_params.get('activation', 'relu')
    dropout = gnn_params.get('dropout', 0.0)
    batch_norm = gnn_params.get('batch_norm', True)

    gnn_blocks = []
    in_dim = input_dim

    for i, (out_dim, args) in enumerate(zip(sizes, layer_args)):

        if layer_type == 'attention':
            heads = gnn_params.get('heads', 1)
            next_in = out_dim * heads
            graph_layer = layer_class(in_dim, out_dim, heads=heads, **args)
        elif layer_type == 'convolutional':
            next_in = out_dim
            graph_layer = layer_class(in_dim, out_dim, **args)
        elif layer_type == 'edge':
            nn_layer = nn.Linear(in_dim * 2, out_dim)
            next_in = out_dim
            graph_layer = layer_class(nn_layer, **args)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        batch_norm_layer = nn.BatchNorm1d(next_in) if batch_norm else None
        activation_layer = get_activation_fn(activation) if activation else None
        dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

        gnn_blocks.append(
            GNNModule(
                graph_layer=graph_layer,
                batch_norm=batch_norm_layer,
                activation=activation_layer,
                dropout=dropout_layer
            )
        )
        in_dim = next_in

    return nn.ModuleList(gnn_blocks), in_dim


def build_gnn_config():
    def suggest_gnn_layers():
        return {
            'convolutional': [GCNConv, SAGEConv, GINConv],
            'attention': [GATConv, GATv2Conv],
            'edge': [EdgeConv]
        }

    print("=== GNN Layer Configuration ===")
    layer_types = ['convolutional', 'attention', 'edge']

    while True:
        print(f"Available layer types: {layer_types}")
        layer_type = input("Select layer type: ").strip().lower()
        if layer_type in layer_types:
            break
        print("Invalid layer type. Please try again.")

    suggestions = suggest_gnn_layers()[layer_type]
    layer_class_names = [cls.__name__ for cls in suggestions]
    while True:
        print(f"Suggested layers for '{layer_type}': {layer_class_names}")
        layer_name = input("Enter the layer class name: ").strip()
        layer_class = next((cls for cls in suggestions if cls.__name__ == layer_name), None)
        if layer_class:
            break
        print("Invalid layer name. Please choose from the suggestions.")

    sizes = input("Enter output sizes (comma-separated, e.g. 64,64,32): ")
    sizes = list(map(int, sizes.split(',')))
    assert all(size > 0 for size in sizes)

    input_dim = int(input("Enter input dimension: "))
    assert input_dim > 0

    layer_args = [{} for _ in sizes]
    edge_dim = None
    heads = 1

    if layer_type == 'attention':
        heads = int(input("Enter number of attention heads [1]: ") or 1)
        edge_dim = int(input("Enter edge feature dimension (edge_dim): "))
        assert heads > 1
        assert edge_dim > 0

    if input("Do you want to enter custom args for each layer? (y/n): ").strip().lower() == 'y':
        for i in range(len(sizes)):
            arg_str = input(f"Layer {i} args (as Python dict): ")
            layer_args[i] = eval(arg_str)

    if layer_type == 'attention' and edge_dim is not None:
        for args in layer_args:
            args['edge_dim'] = edge_dim

    available_activations = ['relu', 'leaky_relu', 'gelu', 'tanh']
    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or 'relu'
        if activation in available_activations:
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout = input("Enter dropout rate [0.1]: ").strip()
            dropout = float(dropout) if dropout else 0.1
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    while True:
        try:
            batch_norm = input("Enter batch_norm [Y/N]: ").strip()
            if batch_norm in ['Y', 'N']:
                batch_norm = {"Y": True, "N": False}.get(batch_norm)
                break
            else:
                print(f"Batch norm must be Y/N. Try again.")
        except ValueError:
            "Invalid answer. Try again."

    gnn_params = {
        'layer': layer_class,
        'layer_type': layer_type,
        'sizes': sizes,
        'input_dim': input_dim,
        'args': layer_args,
        'activation': activation,
        'dropout': dropout,
        'batch_norm': batch_norm
    }

    if layer_type == 'attention':
        gnn_params['heads'] = heads

    return gnn_params


class CNNModule(nn.Module):
    """
    One 1D convolution block with optional norm, activation, pooling, dropout.

    Parameters
    ----------
    conv_layer : torch.nn.Module
    batch_norm, activation, max_pool, dropout : torch.nn.Module or None
    kernel_size, stride, pool_kernel_size : int
        Shape hyperparameters recorded for debugging or padding logic.
    """
    def __init__(self, conv_layer, batch_norm, activation, max_pool, dropout, kernel_size, stride, pool_kernel_size):
        super().__init__()
        self.conv_layer = conv_layer
        self.batch_norm = batch_norm
        self.activation = activation
        self.max_pool = max_pool
        self.dropout = dropout

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, lengths):

        x = self.conv_layer(x)

        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        if self.max_pool:
            x = self.max_pool(x)
        if self.dropout:
            x = self.dropout(x)

        lengths = ((lengths - self.kernel_size) // self.stride) + 1
        lengths = lengths // self.pool_kernel_size
        lengths = torch.clamp(lengths, min=1)

        return x, lengths


def build_conv_layers(cnn_params):
    alphabet_len = cnn_params['alphabet_len']
    embedding_dim = cnn_params['embedding_dim']  # i.e. in_channels for the first layer
    padding_idx = cnn_params.get('padding_idx', 0)

    cnn_embedding = nn.Embedding(
        num_embeddings=alphabet_len,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx
    )

    sizes = cnn_params.get('sizes', [256])
    kernel_size = cnn_params.get('kernel_size', 5)
    stride = cnn_params.get('stride', 1)
    dropout = cnn_params.get('dropout', 0.1)
    activation_fn = get_activation_fn(cnn_params.get('activation', 'relu'))
    pool_kernel_size = cnn_params.get('pool_size', 2)
    batch_norm = cnn_params.get('batch_norm', True)

    in_channels = embedding_dim
    cnn_blocks = []

    for out_channels in sizes:
        conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        activation = activation_fn if activation_fn else None
        max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size) if pool_kernel_size > 1 else None
        dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

        cnn_blocks.append(
            CNNModule(
                conv_layer=conv_layer,
                batch_norm=batch_norm,
                activation=activation,
                dropout=dropout_layer,
                max_pool=max_pool,
                kernel_size=kernel_size,
                stride=stride,
                pool_kernel_size=pool_kernel_size
            )
        )
        in_channels = out_channels

    return nn.ModuleList(cnn_blocks), cnn_embedding, in_channels


def build_cnn_config():
    print("=== CNN Layer Configuration ===")

    while True:
        try:
            alphabet_len = int(input("Enter alphabet length (num embeddings): "))
            if alphabet_len > 0:
                break
            else:
                print("Must be a positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            embedding_dim = int(input("Enter embedding dimension (input channels): "))
            if embedding_dim > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    padding_idx_input = input("Enter padding index [0]: ").strip()
    padding_idx = int(padding_idx_input) if padding_idx_input else 0

    while True:
        sizes_input = input("Enter conv layer output sizes (comma-separated, e.g. 256,128): ")
        try:
            sizes = list(map(int, sizes_input.split(',')))
            if all(s > 0 for s in sizes):
                break
            else:
                print("All sizes must be positive integers.")
        except Exception:
            print("Invalid input. Please enter comma-separated positive integers.")

    while True:
        try:
            kernel_size = int(input("Enter kernel size [5]: ") or 5)
            if kernel_size > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            stride = int(input("Enter stride [1]: ") or 1)
            if stride > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            pool_size = int(input("Enter max pooling kernel size [2]: ") or 2)
            if pool_size >= 1:
                break
            else:
                print("Must be integer >= 1.")
        except ValueError:
            print("Invalid integer. Try again.")

    available_activations = ['relu', 'leaky_relu', 'gelu', 'tanh']
    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or 'relu'
        if activation in available_activations:
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout = input("Enter dropout rate [0.1]: ").strip()
            dropout = float(dropout) if dropout else 0.1
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    while True:
        try:
            batch_norm = input("Enter batch_norm [Y/N]: ").strip()
            if batch_norm in ['Y', 'N']:
                batch_norm = {"Y": True, "N": False}.get(batch_norm)
                break
            else:
                print(f"Batch norm must be Y/N. Try again.")
        except ValueError:
            "Invalid answer. Try again."


    cnn_params = {
        'alphabet_len': alphabet_len,
        'embedding_dim': embedding_dim,
        'padding_idx': padding_idx,
        'sizes': sizes,
        'kernel_size': kernel_size,
        'stride': stride,
        'pool_size': pool_size,
        'activation': activation,
        'dropout': dropout,
        'batch_norm': batch_norm
    }

    print("\nCNN layer config complete.")
    return cnn_params


class RNNModule(nn.Module):
    """
    Pack-padded recurrent cell with fixed ``total_length`` when unpacking.

    Parameters
    ----------
    recurrent_layer : torch.nn.Module
        ``nn.LSTM``, ``nn.GRU``, or ``nn.RNN`` with ``batch_first=True``.
    max_len : int
        ``total_length`` passed to :func:`torch.nn.utils.rnn.pad_packed_sequence`.
    """
    def __init__(self, recurrent_layer, max_len: int):
        super().__init__()
        self.recurrent_layer = recurrent_layer
        self.max_len = max_len

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.recurrent_layer(packed_x)
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.max_len)

        return unpacked_out, lengths


def build_recurrent_layers(rnn_params: dict):
    """
    Build an embedding plus recurrent stack for sequence inputs.

    Parameters
    ----------
    rnn_params : dict
        Must include ``alphabet_len``, ``embedding_dim``, ``hidden_size``,
        ``max_len``, and ``layer`` (one of ``'lstm'``, ``'gru'``, ``'rnn'``).
        Optional keys: ``padding_idx`` (default 0).

    Returns
    -------
    rnn_blocks : nn.ModuleList
        Wrapped RNN modules.
    rnn_embedding : nn.Embedding
        Token embedding layer.
    output_dim : int
        Hidden size of the recurrent layer.
    """
    alphabet_len = rnn_params['alphabet_len']
    embedding_dim = rnn_params['embedding_dim']  # i.e. in_channels for the first layer
    padding_idx = rnn_params.get('padding_idx', 0)

    rnn_embedding = nn.Embedding(
        num_embeddings=alphabet_len,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx
    )

    rnn_type = rnn_params.get('layer', 'gru').lower()
    hidden_size = rnn_params.get('hidden_size')
    rnn_blocks = []  # for potential extension to chained recurrent layers

    # Build RNN layer
    if rnn_type == 'lstm':
        rnn_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    elif rnn_type == 'gru':
        rnn_layer = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    elif rnn_type == 'rnn':
        rnn_layer = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    else:
        raise ValueError(f"Unsupported RNN layer type: {rnn_type}")

    max_len = rnn_params['max_len']
    rnn_blocks.append(
        RNNModule(
            recurrent_layer=rnn_layer,
            max_len=max_len
        )
    )
    output_dim = hidden_size

    return nn.ModuleList(rnn_blocks), rnn_embedding, output_dim


def build_rnn_config():
    print("=== RNN Layer Configuration ===")

    while True:
        try:
            alphabet_len = int(input("Enter alphabet length (num embeddings): "))
            if alphabet_len > 0:
                break
            else:
                print("Must be a positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            embedding_dim = int(input("Enter embedding dimension (input channels): "))
            if embedding_dim > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    padding_idx_input = input("Enter padding index [0]: ").strip()
    padding_idx = int(padding_idx_input) if padding_idx_input else 0

    rnn_types = ['lstm', 'gru', 'rnn']
    while True:
        print(f"Available RNN types: {rnn_types}")
        rnn_type = input("Select RNN type [gru]: ").strip().lower() or 'gru'
        if rnn_type in rnn_types:
            break
        print("Invalid RNN type. Try again.")

    while True:
        try:
            hidden_size = int(input("Enter hidden size: "))
            if hidden_size > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            max_len = int(input("Enter max sequence length: "))
            if max_len > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    rnn_params = {
        'alphabet_len': alphabet_len,
        'embedding_dim': embedding_dim,
        'padding_idx': padding_idx,
        'layer': rnn_type,
        'hidden_size': hidden_size,
        'max_len': max_len,
    }

    print("\nRNN layer config complete.")
    return rnn_params


def build_linear_layers(sizes: List[int], batch_norm: bool = True,
                        activation: str = 'relu', dropout: float = 0.0):
    def init_linear(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.1, std=0.025)

    linear_layers = []

    linear_sizes = sizes.copy()
    lin_out_size = linear_sizes[-1]
    in_features = linear_sizes.pop(0)

    for out_features in linear_sizes:
        linear = nn.Linear(in_features, out_features)
        init_linear(linear)
        linear_layers.append(linear)

        if batch_norm:
            linear_layers.append(nn.BatchNorm1d(out_features))
        if activation is not None:
            linear_layers.append(get_activation_fn(activation))
        if dropout > 0:
            linear_layers.append(nn.Dropout(p=dropout))

        in_features = out_features

    return nn.Sequential(*linear_layers), lin_out_size


def build_lin_config():
    print("=== Linear Layer Configuration ===")

    while True:
        sizes_input = input("Enter linear layer sizes (comma-separated, include input and output sizes, e.g. 128,64,32): ")
        try:
            sizes = list(map(int, sizes_input.split(',')))
            if len(sizes) >= 2 and all(s > 0 for s in sizes):
                break
            else:
                print("Provide at least two positive integers.")
        except Exception:
            print("Invalid input. Try again.")

    while True:
        try:
            batch_norm = input("Enter batch_norm [Y/N]: ").strip()
            if batch_norm in ['Y', 'N']:
                batch_norm = {"Y": True, "N": False}.get(batch_norm)
                break
            else:
                print(f"Batch norm must be Y/N. Try again.")
        except ValueError:
            "Invalid answer. Try again."

    available_activations = ['relu', 'leaky_relu', 'gelu', 'tanh', 'none']
    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or 'relu'
        if activation in available_activations:
            if activation == 'none':
                activation = None
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout_input = input("Dropout rate [0.0]: ").strip()
            dropout = float(dropout_input) if dropout_input else 0.0
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    linear_params = {
        'sizes': sizes,
        'batch_norm': batch_norm,
        'activation': activation,
        'dropout': dropout
    }

    print("\nLinear layer config complete.")
    return linear_params
