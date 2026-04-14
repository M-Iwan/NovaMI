import inspect
from copy import deepcopy
from functools import reduce
from collections import defaultdict
from typing import List, Iterable, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             accuracy_score, recall_score)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch_geometric
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool

from novami.deep.dataset import StringDataset, GraphDataset
from novami.deep.vectorizer import MMGV
from novami.deep.modules import build_linear_layers, build_graph_layers, build_conv_layers, build_recurrent_layers


class GNNLayerBlock(nn.Module):

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


class GNNBackbone(nn.Module):
    def __init__(self, graph_layers, projection_layer, attention_layer, device):
        super().__init__()
        self.layers = graph_layers
        self.projection = projection_layer
        self.attention = attention_layer
        self.device = device

    def forward(self, graph, query_vector):
        graph = graph.to(self.device)
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x, mask = to_dense_batch(x, batch)
        x = self.projection(x)
        x, attn_weights = self.attention(query_vector, x, x, key_padding_mask=~mask)
        x = x.mean(dim=1)
        return x, attn_weights


class CNNLayerBlock(nn.Module):

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


class CNNBackbone(nn.Module):
    def __init__(self, conv_layers, embedding_layer, projection_layer, attention_layer, device):
        super().__init__()
        self.layers = conv_layers
        self.embedding = embedding_layer
        self.projection = projection_layer
        self.attention = attention_layer
        self.device = device

    def forward(self, tokens, lengths, query_vector):
        x = self.embedding(tokens)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x, lengths = layer(x=x, lengths=lengths)
        x = x.permute(0, 2, 1)
        mask = self.build_mask(lengths, max_len=x.size(1)).to(self.device)
        x = self.projection(x)
        x, attn_weights = self.attention(query_vector, x, x, key_padding_mask=~mask)
        x = x.mean(dim=1)
        return x, attn_weights

    @staticmethod
    def build_mask(lengths, max_len=None):
        if max_len is None:
            max_len = lengths.max()
        range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        mask = range_row < lengths.unsqueeze(1)
        return mask  # shape: (B, T)


class RNNLayerBlock(nn.Module):

    def __init__(self, recurrent_layer, max_len: int):
        super().__init__()
        self.recurrent_layer = recurrent_layer
        self.max_len = max_len

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.recurrent_layer(packed_x)
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.max_len)

        return unpacked_out, lengths


class RNNBackbone(nn.Module):
    def __init__(self, recurrent_layers, embedding_layer, projection_layer, attention_layer, device):
        super().__init__()
        self.layers = recurrent_layers
        self.embedding = embedding_layer
        self.projection = projection_layer
        self.attention = attention_layer
        self.device = device

    def forward(self, tokens, lengths, query_vector):
        x = self.embedding(tokens)
        for layer in self.layers:
            x, _ = layer(x=x, lengths=lengths)
        mask = self.build_mask(lengths, max_len=x.size(1)).to(self.device)
        x = self.projection(x)
        x, attn_weights = self.attention(query_vector, x, x, key_padding_mask=~mask)
        x = x.mean(dim=1)
        return x, attn_weights

    @staticmethod
    def build_mask(lengths, max_len=None):
        batch_size = lengths.size(0)
        max_len = max_len or lengths.max().item()
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        mask = mask >= lengths.unsqueeze(1)
        return mask  # shape: (batch, max_len), dtype: bool


class DescBackbone(nn.Module):
    def __init__(self, linear_layers, projection_layer, attention_layer, device):
        super().__init__()
        self.layers = linear_layers
        self.projection = projection_layer
        self.attention = attention_layer
        self.device = device

    def forward(self, x, query_vector):
        for layer in self.layers:
            x = layer(x)
        x = self.projection(x)
        x = x.unsqueeze(1)
        x, attn_weights = self.attention(query_vector, x, x)
        x = x.squeeze(1)
        return x, attn_weights


class MMMTGNN(nn.Module):
    """
    Multi-Modal Multi-Task Generalized Neural Network.
    The model accepts input in a form of dataframe.to_dict(orient='records').

    def __init__(self, device: str = 'cpu', task: str = 'classification', num_task: int = 1,
        label_name: str = None, weight_name: str = None, signature_name: str = None,
        gnn_params: dict = None, cnn_params: dict = None, rnn_params: dict = None,
        des_params: dict = None, att_params: dict = None, lin_params: dict = None,
        max_norm: float = 1.0, query_desc: str = 'Demo'):

    The gnn_params must specify the following:
        - 'layer': class from PyTorch Geometric NN module
        - 'layer_type':
            * 'convolutional': GCNConv, GraphConv, GINConv
            * 'attention': GAT, GATv2, TransformerConv
            * 'edge': NNConv, GINEConv
        - 'sizes': sizes of each layer, also decides the number of layers
        - 'input_dim': number of atom features, currently 39
        - 'activation': name of activation function
        - 'dropout': float of probability
        - 'use_edge_attr': bool whether the layer uses edges

    The cnn_params must specify the following:
        - 'alphabet_len': number of unique tokens in the alphabet
        - 'embedding_dim': per-token embedding size
        - 'padding_idx': idx of token for padding
        - 'sizes': sizes of each layer
        - 'kernel_size': convolution size
        - 'stride': convolution stride
        - 'dropout': linear dropout after convolutions
        - 'activation': activation function after convolutions
        - 'pool_kernel_size': pooling size

    The rnn_params must specify the following:
        - 'alphabet_len': number of unique tokens in the alphabet
        - 'embedding_dim': per-token embedding size
        - 'padding_idx': idx of token for padding
        - 'layer': one of 'lstm', 'gru', 'rnn'
        - 'hidden_size': hidden size of the recurrent layer

    The des_params must be in form of: desc_name: dict, (e.g. 'CDDD': {}).
    Each internal dictionary must specify:
        - 'sizes': sizes of linear layers, the first value must be descriptor size
        - 'batch_norm': bool
        - 'activation': name of the activation function
        - 'dropout': float, 0.2 by default
    """
    def __init__(self, device: str = 'cpu', task: str = 'classification', num_task: int = 1,
                 label_name: str = None, weight_name: str = None, signature_name: str = None,
                 gnn_params: dict = None, cnn_params: dict = None, rnn_params: dict = None,
                 des_params: dict = None, att_params: dict = None, lin_params: dict = None,
                 max_norm: float = 1.0, query_desc: str = 'Demo'):

        super(MMMTGNN, self).__init__()

        self.set_seed(42)
        self.hparams = self.get_hyperparameters()
        self.device = device
        self.task = task
        self.metrics_function = {'classification': self.score_classification, 'regression': self.score_regression}.get(self.task, None)
        self.num_task = num_task
        self.label_name = label_name
        self.weight_name = weight_name
        self.signature_name = signature_name
        self.gnn_params = deepcopy(gnn_params)
        self.cnn_params = deepcopy(cnn_params)
        self.rnn_params = deepcopy(rnn_params)
        self.des_params = deepcopy(des_params)
        self.att_params = deepcopy(att_params)
        self.lin_params = deepcopy(lin_params)
        self.max_norm = max_norm
        self.query_desc = query_desc
        self.query_params = self.des_params.pop(self.query_desc)

        self.backbones = nn.ModuleDict()
        self.desc_names = list()
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.logs = defaultdict(list)
        self.epoch = 1
        self.best_state_dict = None
        self.best_epoch = None

        if self.gnn_params is not None:  # i.e. if we want to have a Graph module
            self.gnn_input_name = self.gnn_params.get('input_name', 'Graph')

            gnn_layers, gnn_out_size = build_graph_layers(self.gnn_params)
            gnn_projection, gnn_attention = self.build_attention_layers(input_size=gnn_out_size, **self.att_params)

            gnn_backbone = GNNBackbone(
                graph_layers=gnn_layers,
                projection_layer=gnn_projection,
                attention_layer=gnn_attention,
                device=self.device
            )
            self.backbones['GNN'] = gnn_backbone

        # Convolutional layers
        if self.cnn_params is not None:
            self.cnn_input_name = self.cnn_params.get('input_name', 'String')
            cnn_layers, cnn_embedding, cnn_out_size = self.build_conv_layers()
            cnn_projection, cnn_attention = self.build_attention_layers(input_size=cnn_out_size, **self.att_params)

            cnn_backbone = CNNBackbone(
                conv_layers=cnn_layers,
                embedding_layer=cnn_embedding,
                projection_layer=cnn_projection,
                attention_layer=cnn_attention,
                device=self.device
            )
            self.backbones['CNN'] = cnn_backbone

        # Recurrent layers
        if self.rnn_params is not None:
            self.rnn_input_name = self.rnn_params.get('input_name', 'String')
            self.rnn_name = 'RNN'
            rnn_layer, rnn_embedding, rnn_out_size = build_recurrent_layers(self.rnn_params)
            rnn_projection, rnn_attention = self.build_attention_layers(input_size=rnn_out_size, **self.att_params)

            rnn_backbone = RNNBackbone(
                recurrent_layers=rnn_layer,
                embedding_layer=rnn_embedding,
                projection_layer=rnn_projection,
                attention_layer=rnn_attention,
                device=self.device
            )
            self.backbones['RNN'] = rnn_backbone

        # Descriptor layers
        if self.des_params:
            for desc_name, params in self.des_params.items():

                desc_layers, desc_out_size = build_linear_layers(**params)
                desc_projection, desc_attention = self.build_attention_layers(input_size=desc_out_size, **self.att_params)

                desc_backbone = DescBackbone(
                    linear_layers=desc_layers,
                    projection_layer=desc_projection,
                    attention_layer=desc_attention,
                    device=self.device
                )
                self.backbones[desc_name] = desc_backbone
                self.desc_names.append(desc_name)

        # Query layers
        attn_size = self.att_params.get('attn_size')
        self.query_layers, query_out_size = build_linear_layers(self.query_params)
        self.query_layers.append(nn.Linear(in_features=query_out_size, out_features=attn_size))

        # Linear layers and output
        in_features = len(self.backbones) * attn_size  # number of modalities * att_output
        self.lin_params['sizes'] = [in_features] + self.lin_params['sizes'] + [self.num_task]
        self.lin_layers, _ = self.build_linear_layers(**self.lin_params)

    def forward(self, batch_input: dict):
        outputs = []
        attention_weights = {}

        query = batch_input[self.query_desc].to(self.device)
        query = self.query_layers(query)
        query = query.unsqueeze(1)

        if self.gnn_params is not None:
            graph = batch_input[self.gnn_input_name]
            x_gnn, gnn_weights = self.backbones['GNN'](graph=graph, query_vector=query)
            outputs.append(x_gnn)
            attention_weights['GNN'] = gnn_weights

        if self.cnn_params is not None:
            string = batch_input[self.cnn_input_name]  # is a tuple
            tokens, lengths = string[0].to(self.device), string[1].to('cpu')
            x_cnn, cnn_weights = self.backbones['CNN'](tokens=tokens, lengths=lengths, query_vector=query)
            outputs.append(x_cnn)
            attention_weights['CNN'] = cnn_weights

        if self.rnn_params is not None:
            string = batch_input[self.rnn_input_name]  # is a tuple
            tokens, lengths = string[0].to(self.device), string[1].to('cpu')
            x_rnn, rnn_weights = self.backbones['RNN'](tokens=tokens, lengths=lengths, query_vector=query)
            outputs.append(x_rnn)
            attention_weights['RNN'] = rnn_weights

        if self.desc_names:
            for name in self.desc_names:
                desc = batch_input[name].to(self.device)
                x_desc, desc_weights = self.backbones[name](x=desc, query_vector=query)
                outputs.append(x_desc)
                attention_weights[name] = desc_weights

        x_out = self.lin_layers(torch.cat(outputs, dim=1))

        return {'predictions': x_out, 'attention_weights': attention_weights, 'outputs': outputs}

    def fit_epoch(self, dataloader):
        self.train()

        epoch_losses = []
        epoch_data = {
            'y_pred': [],
            'y_true': [],
            'y_wgts': [],
            'y_sign': defaultdict(list)
        }
        epoch_grad_norms = []

        for batch in dataloader:
            self.optimizer.zero_grad()

            out = self(batch)
            y_pred = out['predictions']
            y_true = batch[self.label_name].to(self.device)
            y_wgts = batch[self.weight_name].to(self.device)
            y_sign = batch[self.signature_name] if self.signature_name is not None else None

            epoch_data['y_pred'].append(y_pred)
            epoch_data['y_true'].append(y_true)
            epoch_data['y_wgts'].append(y_wgts)

            for sign_type, sign_values in y_sign.items():
                epoch_data['y_sign'][sign_type].extend(sign_values)

            batch_loss = self.mws_loss(y_pred=y_pred, y_true=y_true, y_wgts=y_wgts, y_sign=y_sign)

            loss, loss_wgt = batch_loss.get('Total')  # this one already includes masking and weights
            loss.backward()

            batch_grad_norm = self.grad_norm() / (loss_wgt + 1e-6)
            epoch_grad_norms.append(batch_grad_norm)

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            epoch_losses.append(batch_loss)
        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = reduce(self.combine_mws_losses, epoch_losses)
        epoch_loss = self.normalize_mws_loss(epoch_loss)

        epoch_metrics = self.mws_metrics(**self.pack_epoch_data(epoch_data))

        return epoch_loss, epoch_metrics, epoch_grad_norms

    def eval_epoch(self, dataloader):
        self.eval()

        epoch_losses = []
        epoch_data = {
            'y_pred': [],
            'y_true': [],
            'y_wgts': [],
            'y_sign': defaultdict(list)
        }

        with torch.no_grad():
            for batch in dataloader:

                out = self(batch)
                y_pred = out['predictions']
                y_true = batch[self.label_name].to(self.device)
                y_wgts = batch[self.weight_name].to(self.device)
                y_sign = batch[self.signature_name] if self.signature_name is not None else None

                epoch_data['y_pred'].append(y_pred)
                epoch_data['y_true'].append(y_true)
                epoch_data['y_wgts'].append(y_wgts)

                for sign_type, sign_values in y_sign.items():
                    epoch_data['y_sign'][sign_type].extend(sign_values)

                batch_loss = self.mws_loss(y_pred=y_pred, y_true=y_true, y_wgts=y_wgts, y_sign=y_sign)

                epoch_losses.append(batch_loss)

        epoch_loss = reduce(self.combine_mws_losses, epoch_losses)
        epoch_loss = self.normalize_mws_loss(epoch_loss)

        epoch_metrics = self.mws_metrics(**self.pack_epoch_data(epoch_data))

        return epoch_loss, epoch_metrics

    def fit(self, dataloader, n_epochs: int = 4, early_stop: int = 8,
            save_freq: int = 8, save_dir: str = None, verbose: int = 0):

        if self.loss_fn is None or self.optimizer is None:
            raise ValueError(f'Please set the loss function and optimizer before attempting to fit the model.')

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(1, n_epochs + 1):
            train_loss, train_metrics, train_grad_norms = self.fit_epoch(dataloader)
            self.logs['train_loss'].append((epoch, train_loss))
            self.logs['train_metrics'].append((epoch, train_metrics))
            self.logs['train_grad_norms'].append((epoch, train_grad_norms))
            self.epoch += 1

            current_loss = train_loss.get('Total')

            if verbose > 0:
                print(f"Epoch {epoch} train loss: {current_loss:.5f}")

            if early_stop is not None:
                if current_loss < min_loss:
                    min_loss = current_loss
                    patience = deepcopy(early_stop)
                    self.set_best_model()
                    if save_dir is not None:
                        save_path = save_dir.rstrip('/') + f'/MMMTGNN_min.pth'
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    self.get_best_model()
                    self.epoch = deepcopy(self.best_epoch)
                    break

            if (save_freq is not None) and (save_dir is not None) and (epoch % save_freq == 0):
                save_path = save_dir.rstrip('/') + f'/MMMTGNN_{epoch}.pth'
                self.save(save_path)

    def fit_eval(self, train_dataloader, eval_dataloader, n_epochs: int = 4, early_stop: int = 4,
                 save_freq: int = 4, save_dir: str = None, verbose: int = 0):

        if self.loss_fn is None or self.optimizer is None:
            raise ValueError(f'Please set the loss function and optimizer before attempting to fit the model.')

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(1, n_epochs + 1):
            train_loss, train_metrics, train_grad_norms = self.fit_epoch(train_dataloader)
            eval_loss, eval_metrics = self.eval_epoch(eval_dataloader)

            self.logs['train_loss'].append((epoch, train_loss))
            self.logs['train_metrics'].append((epoch, train_metrics))
            self.logs['train_grad_norms'].append((epoch, train_grad_norms))

            self.logs['eval_loss'].append((epoch, eval_loss))
            self.logs['eval_metrics'].append((epoch, eval_metrics))
            self.epoch += 1

            current_loss = eval_loss['Total']

            if verbose > 0:
                print(f"Epoch {epoch} train loss: {train_loss['Total']:.5f}")
                print(f"Epoch {epoch} eval loss: {eval_loss['Total']:.5f}")

            if early_stop is not None:
                if current_loss < min_loss:
                    min_loss = current_loss
                    patience = deepcopy(early_stop)
                    self.set_best_model()
                    if save_dir is not None:
                        save_path = save_dir.rstrip('/') + f'/MMMTGNN_min.pth'
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    self.get_best_model()
                    self.epoch = deepcopy(self.best_epoch)
                    break

            if (save_freq is not None) and (save_dir is not None) and (epoch % save_freq == 0):
                save_path = save_dir.rstrip('/') + f'/MMMTGNN_{epoch}.pth'
                self.save(save_path)

    def predict(self, dataloader):

        self.eval()

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                preds = self(batch)['predictions']
                predictions.append(preds)

        return torch.cat(predictions, dim=0).detach().cpu().numpy()

    def mws_loss(self, y_pred, y_true, y_wgts, y_sign: dict):
        """
        Masked Weighted Signature Loss (MWS Loss) with hierarchical signatures.
        """

        mask = ~torch.isnan(y_true)
        y_true = torch.nan_to_num(y_true, nan=0.0)  # potentially remove

        loss = self.loss_fn(y_pred, y_true)  # shape: (batch_size, num_tasks)

        m_wgts = y_wgts * mask
        mw_loss = loss * m_wgts  # masked weighted total loss

        total_loss = mw_loss.sum()
        total_loss_wgt = m_wgts.sum()  # needed later for normalization

        per_task_loss = mw_loss.sum(dim=0)
        per_task_loss_wgt = (y_wgts * mask).sum(dim=0)

        per_sign_loss = defaultdict(lambda: [torch.zeros(self.num_task, device=self.device),
                                             torch.zeros(self.num_task, device=self.device)])

        for sign_type, sign_values in y_sign.items():  # e.g. Age: [List]
            for s_idx in range(y_true.shape[0]):
                sign_name = sign_values[s_idx]
                for t_idx in range(y_true.shape[1]):
                    key = (sign_type, sign_name)
                    per_sign_loss[key][0][t_idx] += mw_loss[s_idx, t_idx]
                    per_sign_loss[key][1][t_idx] += m_wgts[s_idx, t_idx]

        batch_loss = {
            'Total': (total_loss, total_loss_wgt),
            'Task': (per_task_loss, per_task_loss_wgt),
            'Sign': per_sign_loss
        }

        return batch_loss

    def combine_mws_losses(self, loss_1, loss_2):
        """
        Combine two MWS loss output dictionaries into one aggregated loss.
        Assumes both have the format returned by `mws_loss`.
        """

        total_loss = loss_1['Total'][0] + loss_2['Total'][0]
        total_wgt = loss_1['Total'][1] + loss_2['Total'][1]

        per_task_loss = loss_1['Task'][0] + loss_2['Task'][0]
        per_task_wgt = loss_1['Task'][1] + loss_2['Task'][1]

        per_sign_loss = defaultdict(lambda: [torch.zeros(self.num_task, device=self.device),
                                             torch.zeros(self.num_task, device=self.device)])

        def combine(sign_loss):
            for key, (sub_loss, sub_wgts) in sign_loss.items():
                per_sign_loss[key][0] += sub_loss
                per_sign_loss[key][1] += sub_wgts

        combine(loss_1['Sign'])
        combine(loss_2['Sign'])

        return {
            'Total': (total_loss, total_wgt),
            'Task': (per_task_loss, per_task_wgt),
            'Sign': per_sign_loss
        }

    def normalize_mws_loss(self, mws_loss):

        per_sample_loss = mws_loss['Total'][0] / mws_loss['Total'][1]
        per_sample_loss = np.round(per_sample_loss.detach().cpu().numpy(), 5)
        per_task_loss = mws_loss['Task'][0] / mws_loss['Task'][1]
        per_task_loss = np.round(per_task_loss.detach().cpu().numpy(), 5)
        per_sign_loss = defaultdict(lambda: torch.zeros(self.num_task, device=self.device))

        for key, (sub_loss, sub_wgts) in mws_loss['Sign'].items():
            sub_array = sub_loss / sub_wgts
            per_sign_loss[key] = np.round(sub_array.detach().cpu().numpy(), 5)

        return {
            'Total': per_sample_loss,
            'Task': per_task_loss,
            'Sign': per_sign_loss
        }

    def mws_metrics(self, y_true, y_pred, y_wgts, y_sign: dict):
        """
        Aggregate predictions based on different criteria. Intended to be used
        with classical ML metrics, purely on predictions
        """
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_wgts = y_wgts.detach().cpu().numpy()

        # Calculate Overall metrics
        total_metrics = self.metrics_function(y_true=y_true.flatten(),
                                              y_pred=y_pred.flatten(),
                                              y_wgts=y_wgts.flatten())

        # Calculate per-task metrics
        per_task_metrics = defaultdict(dict)
        for t_idx in range(self.num_task):
            per_task_metrics[f"Task_{t_idx}"] = self.metrics_function(y_true=y_true[:, t_idx].flatten(),
                                                                      y_pred=y_pred[:, t_idx].flatten(),
                                                                      y_wgts=y_wgts[:, t_idx].flatten())

        # Calculate per-sign metrics
        per_sign_metrics = defaultdict(dict)  # i.e. sign_type: task: sign_name: metrics | horrible
        for sign_type, sign_values in y_sign.items():
            sign_values = np.array(sign_values)
            unique_signs = set(sign_values)
            for t_idx in range(self.num_task):
                y_true_task = y_true[:, t_idx].flatten()
                y_pred_task = y_pred[:, t_idx].flatten()
                y_wgts_task = y_wgts[:, t_idx].flatten()
                for sign_name in unique_signs:
                    key = (sign_type, f"Task_{t_idx}", sign_name)
                    sign_idx = np.where(sign_values == sign_name)[0]
                    per_sign_metrics[key] = self.metrics_function(y_true=y_true_task[sign_idx], y_pred=y_pred_task[sign_idx],
                                                                  y_wgts=y_wgts_task[sign_idx]) if len(sign_idx) > 0 else {}
        return {
            'Total': total_metrics,
            'Task': per_task_metrics,
            'Sign': per_sign_metrics
        }

    @staticmethod
    def pack_epoch_data(epoch_data):
        packed_data = {
            'y_true': torch.cat(epoch_data['y_true'], dim=0),
            'y_pred': torch.cat(epoch_data['y_pred'], dim=0),
            'y_wgts': torch.cat(epoch_data['y_wgts'], dim=0),
            'y_sign': dict(epoch_data['y_sign'])
        }
        return packed_data

    def grad_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    @staticmethod
    def score_classification(y_true: np.ndarray, y_pred: np.ndarray, y_wgts: np.ndarray, threshold: float = 0.5):
        """
        To be used as an inner function with 1D numpy arrays only.
        """

        def safe_div(numerator, denominator, default=0.0):
            return numerator / denominator if denominator != 0 else default

        y_pred_bin = (y_pred >= threshold).astype(int)

        mask = ~np.isnan(y_true)
        y_true = y_true[mask].astype(int)
        y_pred = y_pred[mask]
        y_pred_bin = y_pred_bin[mask]
        y_wgts = y_wgts[mask]

        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred_bin, sample_weight=y_wgts)
        if conf_mat.size != 4:
            raise AttributeError(f'Unexpected conf_mat shape: {conf_mat.shape}')
        tn, fp, fn, tp = conf_mat.ravel()

        rec = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)

        metrics = {
            'TP': np.round(tp, 5),
            'FP': np.round(fp, 5),
            'FN': np.round(fn, 5),
            'TN': np.round(tn, 5),
            'Accuracy': np.round(safe_div(tp + tn, tp + fp + fn + tn), 5),
            'Recall': np.round(rec, 5),
            'Specificity': np.round(spec, 5),
            'Balanced Accuracy': np.round((rec + spec) / 2, 5),
            'Precision': np.round(safe_div(tp, tp + fp), 5),
            'F1 Score': np.round(safe_div(2 * tp, 2 * tp + fp + fn), 5),
            'ROC AUC': np.round(roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=y_wgts), 5),
            'MCC': np.round(safe_div((tp * tn) - (fp * fn), np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))), 5)
        }

        return metrics

    @staticmethod
    def score_regression(y_true: np.ndarray, y_pred: np.ndarray, y_wgts: np.ndarray):

        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        y_wgts = y_wgts[mask]

        metrics = {
            'R2': np.round(r2_score(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5),
            'MAE': np.round(mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5),
            'RMSE': np.round(mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False, sample_weight=y_wgts), 5)
        }

        return metrics

    def build_attention_layers(self, input_size, attn_size, num_heads, dropout):

        projection = nn.Linear(in_features=input_size, out_features=attn_size)
        self.init_linear(projection)
        attn_layer = nn.MultiheadAttention(embed_dim=attn_size, num_heads=num_heads,
                                           batch_first=True, dropout=dropout)

        return projection, attn_layer

    @staticmethod
    def init_linear(layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.normal_(layer.bias, mean=0.1, std=0.025)

    @staticmethod
    def get_activation_fn(name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Activation function not supported: {name}")

    def set_loss_function(self, loss_fn=torch.nn.BCEWithLogitsLoss, loss_params: dict = None):
        if callable(loss_fn):
            self.loss_fn = loss_fn(**loss_params) if loss_params else loss_fn()
        else:
            raise ValueError("Loss function must be callable")
        if getattr(self.loss_fn, 'reduction', None) != 'none':
            raise ValueError('Loss function must be initialized with reduction="none"')

    def set_optimizer(self, optim=torch.optim.AdamW, optim_params: dict = None):
        if callable(optim):
            self.optimizer = optim(self.parameters(), **optim_params) if optim_params else optim(self.parameters())
        else:
            raise ValueError("Optimizer must be callable")

    def set_scheduler(self, scheduler=torch.optim.lr_scheduler.LRScheduler, params: dict = None):
        if callable(scheduler) and (self.optimizer is not None):
            self.scheduler = scheduler(self.optimizer, **params) if params else scheduler(self.optimizer)
        else:
            raise ValueError("Scheduler must be callable and the optimizer must be set")

    @staticmethod
    def get_hyperparameters():
        frame = inspect.currentframe()
        try:
            _, _, _, local_vars = inspect.getargvalues(frame.f_back)
            hparams = {
                k: v for k, v in local_vars.items() if k != 'self'
            }
        finally:
            del frame

        return hparams

    def set_best_model(self):
        self.best_state_dict = deepcopy(self.state_dict())
        self.best_epoch = deepcopy(self.epoch)

    def get_best_model(self):
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)
        else:
            raise RuntimeError("Best model not saved yet.")

    def save(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'hyperparameters': self.get_hyperparameters()
        }
        torch.save(checkpoint, path)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @staticmethod
    def set_seed(seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
