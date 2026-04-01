import inspect
from typing import List

import torch
from torch import nn

from novami.deep.utils import get_activation_fn


class GNNModule(nn.Module):
    """

    """
    def __init__(self, graph_layer, batch_norm, activation, dropout):
        super().__init__()
        self.graph_layer = graph_layer
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.accepts_edge_attr = 'edge_attr' in inspect.signature(self.graph_layer.forward).parameters

    def forward(self, x, edge_index, edge_attr=None):

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
            nn_layer = nn.Linear(in_dim, out_dim)
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


class CNNModule(nn.Module):
    """

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


class RNNModule(nn.Module):
    """

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


def build_recurrent_layers(self):

    alphabet_len = self.rnn_params['alphabet_len']
    embedding_dim = self.rnn_params['embedding_dim']  # i.e. in_channels for the first layer
    padding_idx = self.rnn_params.get('padding_idx', 0)

    rnn_embedding = nn.Embedding(
        num_embeddings=alphabet_len,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx
    )

    rnn_type = self.rnn_params.get('layer', 'gru').lower()
    hidden_size = self.rnn_params.get('hidden_size')
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

    max_len = self.rnn_params['max_len']
    rnn_blocks.append(
        RNNModule(
            recurrent_layer=rnn_layer,
            max_len=max_len
        )
    )
    output_dim = hidden_size

    return nn.ModuleList(rnn_blocks), rnn_embedding, output_dim


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