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


class CRN(nn.Module):
    """
    Convolutional Recurrent Network
    """

    def __init__(self, vectorizer, device: str = 'cpu', batch_norm: bool = True, max_norm: float = 1.0, embedding_dim: int = 256,
                 conv_size: List[int] = None, conv_kernel_size: int = 3, conv_stride: int = 1, conv_dropout: float = 0.25,
                 conv_activation: torch.nn.Module = torch.nn.ReLU, conv_max_pool: int = 2,
                 recurrent_layer: str = 'lstm', recurrent_dim: int = 256, recurrent_aggr: str = 'mean',
                 linear_size: List[int] = None, linear_activation: torch.nn.Module = torch.nn.ReLU, linear_dropout: float = 0.25,
                 output_size: int = 1, task: str = 'classification'):

        super(CRN, self).__init__()

        # Initial setup
        self.vectorizer = vectorizer
        self.task = task

        if self.vectorizer.padding:
            if self.vectorizer.char2idx.get('nop') is None:
                raise ValueError(f'Padding character index < nop > is missing in vectorizer dictionary')

        self.batch_norm = batch_norm
        self.max_norm = max_norm

        # Embedding layer
        self.embedding_dim = embedding_dim
        self.embedding_layer = torch.nn.Embedding(num_embeddings=self.vectorizer.alphabet_len, embedding_dim=self.embedding_dim,
                                                  padding_idx=self.vectorizer.char2idx.get('nop'))

        # Convolutional layer
        self.conv_size = conv_size if conv_size is not None else [256]
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_dropout = conv_dropout
        self.conv_activation = conv_activation
        self.conv_max_pool = conv_max_pool

        conv_layers = []

        in_channels = self.embedding_dim
        for out_channels in self.conv_size:
            if self.batch_norm:
                conv_layers.append(nn.BatchNorm1d(in_channels))
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=self.conv_kernel_size,
                                   stride=self.conv_stride, padding=0, dilation=1)
            conv_layers.append(conv_layer)
            conv_layers.append(self.conv_activation())
            conv_layers.append(nn.MaxPool1d(kernel_size=self.conv_max_pool))
            if self.conv_dropout > 0:
                conv_layers.append(nn.Dropout(p=self.conv_dropout))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Recurrent layer
        self.recurrent_layer = recurrent_layer
        self.recurrent_dim = recurrent_dim
        self.recurrent_aggr = recurrent_aggr

        if self.recurrent_aggr == 'min':
            self.aggr_pad = float('inf')
            self.aggr_func = torch.amin
        elif self.recurrent_aggr == 'mean':
            self.aggr_pad = float('nan')
            self.aggr_func = torch.nanmean
        elif self.recurrent_aggr == 'max':
            self.aggr_pad = float('-inf')
            self.aggr_func = torch.amax
        else:
            raise ValueError(f'Allowed options for < recurrent_aggr > are: min, mean, max')

        if recurrent_layer == 'lstm':
            self.recurrent = torch.nn.LSTM(input_size=self.conv_size[-1], hidden_size=recurrent_dim, batch_first=True)
        elif recurrent_layer == 'gru':
            self.recurrent = torch.nn.GRU(input_size=self.conv_size[-1], hidden_size=recurrent_dim, batch_first=True)
        elif recurrent_layer == 'rnn':
            self.recurrent = torch.nn.RNN(input_size=self.conv_size[-1], hidden_size=recurrent_dim, batch_first=True)
        else:
            raise ValueError(f'Allowed options for < recurrent_layer > are: lstm, gru, rnn')

        # Linear layers
        self.linear_size = linear_size if linear_size is not None else [512]
        self.linear_activation = linear_activation
        self.linear_dropout = linear_dropout

        linear_layers = []

        in_features = self.recurrent_dim
        for out_features in self.linear_size:
            if batch_norm:
                linear_layers.append(torch.nn.BatchNorm1d(in_features))
            linear_layers.append(torch.nn.Linear(in_features, out_features))
            linear_layers.append(self.linear_activation())
            if self.linear_dropout > 0:
                linear_layers.append(torch.nn.Dropout(p=self.linear_dropout))
            in_features = out_features

        self.linear_aggregate_layers = nn.Sequential(*linear_layers)
        self.linear_hidden_layers = nn.Sequential(*linear_layers)

        # Output
        self.output_size = output_size

        self.concat_layer = torch.nn.Sequential(*[torch.nn.Linear(2 * in_features, in_features), self.linear_activation()])

        if self.task == 'classification':
            self.output_layer = torch.nn.Sequential(*[torch.nn.Linear(in_features, self.output_size), torch.nn.Sigmoid()])
        elif self.task == 'regression':
            self.output_layer = torch.nn.Linear(in_features, self.output_size)
        else:
            raise ValueError('Allowed options for < task > are: regression, classification')

        # Other components
        self.loss_fn = None
        self.optimizer = None
        self.device = device
        self.to(self.device)
        self.recurrent.flatten_parameters()  # Call after moving to device
        self.init_hidden()
        self.train_loss = []
        self.eval_loss = []
        self.verbose = 0

    def forward(self, x, lengths):
        # x: torch.Tensor [batch_size x vectorizer.max_length]
        # lengths: torch.Tensor [batch_size]

        x = self.embedding_layer(x)
        # x: torch.Tensor [batch_size x max_length x embedding_dimension]

        # experimental - set all values corresponding to padding to zeros to decrease the effect during convolutions
        mask = torch.arange(x.size(1)).unsqueeze(0).to(self.device) < lengths.unsqueeze(1).to(self.device)
        x = x * mask.unsqueeze(2)

        x = x.permute(0, 2, 1)
        # x: torch.Tensor [batch_size x embedding_dim x max_length]

        x = self.conv_layers(x)

        for i in range(len(self.conv_size)):  # correct lengths: take the last one where at least one token was not-padding
            lengths = ((lengths - self.conv_kernel_size) // self.conv_stride) + 1
            lengths = lengths // self.conv_max_pool  # consider max_pooling which divides the length by kernel_size
            lengths = torch.clamp(lengths, min=1)
            # ((input_length - kernel_size) // stride + 1) // conv_max_pool

        x = x.permute(0, 2, 1)
        # lengths: torch.Tensor [batch_size]
        # x: torch.Tensor [batch_size, new_conv_length x out_channels]

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x_aggr, (x_hidden, _) = self.recurrent(x)  # x_aggr - all hidden states, x_hidden - last not padded state

        # Aggregation branch
        x_aggr, _ = pad_packed_sequence(x_aggr, batch_first=True, padding_value=self.aggr_pad)
        # batch_size x max(lengths) x embedding_dim

        x_aggr = self.aggr_func(x_aggr, dim=1)  # experimental

        # Hidden branch
        x_hidden = x_hidden.permute(1, 0, 2).squeeze(1)
        # change to [batch_size x 1 x recurrent_dim] and then to [batch_size x recurrent_dim]

        x = torch.cat((self.linear_aggregate_layers(x_aggr), self.linear_hidden_layers(x_hidden)), dim=1)

        x_enc = self.concat_layer(x)
        x_out = self.output_layer(x_enc)

        return {'value': x_out.reshape(-1, self.output_size), 'linear_embedding': x_enc, 'recurrent_embedding': x_hidden}

    def fit_epoch(self, train_dataloader):
        """
        TODO: add per-class MCC analysis of predictions
        """

        self.train()

        epoch_loss = 0.0

        for data_ in train_dataloader:
            self.optimizer.zero_grad()

            strings, labels, lengths = (item.to(self.device) for item in data_)
            lengths = lengths.cpu()

            predictions = self(strings, lengths)['value']

            mask = ~torch.isnan(labels)
            norm = torch.sum(mask)

            masked_preds = predictions[mask]
            masked_targets = labels[mask]

            if norm > 0:
                loss = self.loss_fn(masked_preds, masked_targets)
                loss.backward()
                epoch_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

        self.train_loss.append(epoch_loss)

    def eval_epoch(self, eval_dataloader):
        """
        TODO: add per-class MCC analysis of predictions
        """
        self.eval()

        epoch_loss = 0.0

        with torch.no_grad():
            for data_ in eval_dataloader:

                strings, labels, lengths = (item.to(self.device) for item in data_)
                lengths = lengths.cpu()
                predictions = self(strings, lengths)['value']

                mask = ~torch.isnan(labels)
                norm = torch.sum(mask)

                masked_preds = predictions[mask]
                masked_targets = labels[mask]

                if norm > 0:
                    loss = self.loss_fn(masked_preds, masked_targets)
                    epoch_loss += loss.item()

        self.eval_loss.append(epoch_loss)

    def fit(self, strings: Iterable[str], labels: Union[List, np.ndarray], n_epochs: int = 64,
            early_stop: int = None, early_dir: str = None, save_freq: int = None, save_dir: str = None,
            batch_size: int = 256, verbose: int = 0):
        """
        Train the model on the provided dataset of SMILES strings and corresponding labels, with optional
        early stopping and periodic model saving.

        Parameters
        ----------
        strings : Iterable[str]
            List of SMILES representing valid chemical structures.
        labels : Union[List, np.ndarray]
            Target values associated with each SMILES. Accepts a list, 1D np.ndarray, or
            2D np.ndarray. Labels should have the shape (num_samples, num_targets).
        n_epochs : int, optional
            The number of epochs for training. Default is 64.
        early_stop : int, optional
            The patience threshold for early stopping. If provided, training will halt if no improvement
            in the loss is observed for this number of epochs. Default is None (no early stopping).
        early_dir : str, optional
            Path to the directory where the model with the lowest recorded loss will be saved, if early
            stopping is active. Default is None.
        save_freq : int, optional
            Frequency (in epochs) for saving the model during training. If provided, the model will be
            saved every `save_freq` epochs. Default is None (no periodic saving).
        save_dir : str, optional
            Path to the directory where periodic model checkpoints will be saved, if `save_freq` is
            specified. Default is None.
        batch_size : int, optional
            Batch size for the DataLoader during training. Default is 256.
        verbose : int, optional
            Verbosity level. If greater than 0, the method will print the loss at each epoch. Default is 0.
        """

        if isinstance(labels, list):  # list are only allowed when there is only 1 target value for each string
            labels = np.array(labels, dtype=np.float64).reshape(-1, 1)  # making tensor from lists is apparently slow

        if isinstance(labels, np.ndarray):  # should be a num_samples x num_targets array
            if len(labels.shape) == 1:  # assume only one target value for string
                labels = labels.reshape(-1, 1)

        processed_results = [self.vectorizer.process_smiles(string, how='embedding', as_tensor=True) for string in strings]
        processed_strings, lengths = zip(*processed_results)
        processed_lengths = torch.LongTensor(lengths).cpu()

        processed_labels = torch.from_numpy(labels).float()

        train_dataset = StringDataset(processed_strings, processed_labels, processed_lengths)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(1, n_epochs + 1):

            self.fit_epoch(train_dataloader)

            current_loss = self.train_loss[-1]
            if verbose > 0:
                print(f'Epoch {epoch} - current loss: {current_loss:.5f}')

            if early_stop is not None:
                if current_loss < min_loss:
                    min_loss = current_loss
                    patience = deepcopy(early_stop)
                    if early_dir is not None:
                        save_path = early_dir.rstrip('/') + f'/CRN_min.pth'
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    break

            if save_freq is not None and epoch % save_freq == 0:  # save the model every save_freq epochs
                save_path = save_dir.rstrip('/') + f'/CRN_{epoch}.pth'
                self.save(save_path)

    def fit_eval(self, train_strings: Iterable[str], eval_strings: Iterable[str], train_labels: Union[List, np.ndarray],
                 eval_labels: Union[List, np.ndarray], n_epochs: int = 64, batch_size: int = 256, verbose: int = 0):
        """
        TODO: add automatic saving and early-stopping to fit_eval as well
        """

        if isinstance(train_labels, list) or isinstance(eval_labels, list):  # assuming both to be of same type
            train_labels = np.array(train_labels, dtype=np.float64).reshape(-1, 1)
            eval_labels = np.array(eval_labels, dtype=np.float64).reshape(-1, 1)

        if isinstance(train_labels, np.ndarray) or isinstance(eval_labels, np.ndarray):  # should be a num_samples x num_targets array

            if len(train_labels.shape) == 1:
                train_labels = train_labels.reshape(-1, 1)

            if len(eval_labels.shape) == 1:
                eval_labels = eval_labels.reshape(-1, 1)

        processed_train_results = [self.vectorizer.process_smiles(string, how='embedding', as_tensor=True) for string in
                                   train_strings]
        processed_train_strings, train_lengths = zip(*processed_train_results)

        processed_train_labels = torch.FloatTensor(train_labels)
        processed_train_lengths = torch.LongTensor(train_lengths)

        train_dataset = StringDataset(processed_train_strings, processed_train_labels, processed_train_lengths)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        processed_eval_results = [self.vectorizer.process_smiles(string, how='embedding', as_tensor=True) for string in
                                  eval_strings]
        processed_eval_strings, eval_lengths = zip(*processed_eval_results)

        processed_eval_labels = torch.FloatTensor(eval_labels)
        processed_eval_lengths = torch.LongTensor(eval_lengths)

        eval_dataset = StringDataset(processed_eval_strings, processed_eval_labels, processed_eval_lengths)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            self.fit_epoch(train_dataloader)
            self.eval_epoch(eval_dataloader)

            if verbose > 0:
                print(f'Current train loss: {self.train_loss[-1]:.5f}')
                print(f'Current eval loss : {self.eval_loss[-1]:.5f}')

    def embed(self, strings: List[str], kind: str = 'linear', batch_size: int = 256):
        """
        Retrieve embeddings from SMILES
        """

        self.eval()

        processed_results = [self.vectorizer.process_smiles(string, how='embedding', as_tensor=True) for string in
                             strings]
        processed_strings, lengths = zip(*processed_results)

        processed_lengths = torch.FloatTensor(lengths)

        embedding_dataset = StringDataset(processed_strings, [np.nan for _ in range(len(strings))], processed_lengths)
        embedding_dataloader = torch.utils.data.DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for data_ in embedding_dataloader:

                x, _, lengths = (item.to(self.device) for item in data_)
                lengths = lengths.cpu()

                if kind == 'linear':
                    data_embedding = self(x, lengths)['linear_embedding']
                elif kind == 'recurrent':
                    data_embedding = self(x, lengths)['recurrent_embedding']
                else:
                    raise ValueError(f'Allowed options for < type > are: linear, recurrent')

                embeddings.append(data_embedding)
        return torch.cat(embeddings, dim=0)

    def predict(self, strings: List[str], batch_size: int = 256):

        self.eval()

        processed_results = [self.vectorizer.process_smiles(string, how='embedding', as_tensor=True) for string in
                             strings]
        processed_strings, lengths = zip(*processed_results)

        processed_lengths = torch.FloatTensor(lengths)

        predict_dataset = StringDataset(processed_strings, [np.nan for _ in range(len(strings))], processed_lengths)
        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for data_ in predict_dataloader:
                x, _, lengths = (item.to(self.device) for item in data_)
                lengths = lengths.cpu()

                data_predictions = self(x, lengths)['value']
                predictions.append(data_predictions)

        predictions = torch.cat(predictions, dim=0)
        return predictions.detach().to('cpu')

    def set_loss_function(self, loss_fn=torch.nn.MSELoss, loss_params: dict = None):
        if callable(loss_fn):
            self.loss_fn = loss_fn(**loss_params) if loss_params else loss_fn()
        else:
            raise ValueError("Loss function must be callable")

    def set_optimizer(self, optim=torch.optim.AdamW, optim_params: dict = None):
        if callable(optim):
            self.optimizer = optim(self.parameters(), **optim_params) if optim_params else optim(self.parameters())
        else:
            raise ValueError("Optimizer must be callable")

    @staticmethod
    def init_linear(layer):
        # TODO: add to __init__
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def init_hidden(self):
        # TODO: add to __init__
        for name, param in self.recurrent.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)

    def save(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def plot_loss(self):
        sns.set_style('whitegrid')
        sns.set_context('notebook')
        if not self.train_loss:
            print('Please train your model first')
            return

        if not self.eval_loss:
            loss_df = pd.DataFrame({'Epoch': np.arange(1, len(self.train_loss) + 1), 'Loss': self.train_loss})
            g = sns.relplot(loss_df, x='Epoch', y='Loss', kind='line')

        else:
            loss_df = pd.DataFrame(
                {'Epoch': np.arange(1, len(self.train_loss) + 1), 'Train': self.train_loss, 'Eval': self.eval_loss})
            loss_df = pd.melt(loss_df, id_vars='Epoch', value_vars=['Train', 'Eval'], var_name='Dataset',
                              value_name='Loss')
            g = sns.relplot(loss_df, x='Epoch', y='Loss', col='Dataset', kind='line')

        return g

    def calculate_mcc(self):
        raise NotImplementedError


class MMGNN(torch.nn.Module):
    """
    Multimodal GNN architecture
    """

    def __init__(self, vectorizer: MMGV, device: str = 'cpu', batch_norm: bool = True, max_norm: float = 1.0, descriptor_size: int = None,
                 descriptor_names: List[str] = None,
                 graph_layer=torch_geometric.nn.GATv2Conv, graph_size: List = None, graph_heads: int = 1, graph_dropout: float = 0.25,
                 linear_size: List = None, linear_activation: torch.nn.Module = torch.nn.ReLU, linear_dropout: float = 0.25,
                 output_size: int = 1, task: str = 'classification'):

        """
        The architecture is as follows:
        - input: torch_geometric.Graph with x, edge_index, edge_attr, y, and optional other parameters (e.g. RDKit descriptors)
        - smiles string is transformed into graph using vectorizer
        - by default GATv2Conv is used for message passing
        - max and mean pool are applied as a read-out functions and processed by linear_layers
        - outputs from pooling are concatenated with optional descriptors and processed by concat_layers
        - the final output is obtained
        TODO: Add sample weighting option (Huge P.I.T.A. inc.) based on label uncertainty
        TODO: Add attention layer
        TODO: Add individual processing of inputs (i.e. small NN for each descriptor type)
        TODO: Implement ordinal classification (apparently using torch-ordinal or Corn ordinal regression)
        """

        super(MMGNN, self).__init__()

        # Initial setup
        self.vectorizer = vectorizer
        self.task = task
        self.batch_norm = batch_norm
        self.max_norm = max_norm

        if self.task == 'classification':
            self.scoring_function = self.score_classification
        elif self.task == 'regression':
            self.scoring_function = self.score_regression
        else:
            raise ValueError('Task must be either classification or regression')

        # Graph layers
        self.graph_layer = graph_layer
        self.graph_size = graph_size if graph_size is not None else [32]
        self.graph_heads = graph_heads
        self.graph_dropout = graph_dropout

        graph_layers = []

        in_channels = 39  # size of atom embedding = 39, size of edge embedding = 8

        for out_channels in self.graph_size:
            graph_layers.append(self.graph_layer(in_channels, out_channels, heads=self.graph_heads, dropout=self.graph_dropout, edge_dim=8))
            in_channels = out_channels * self.graph_heads

        self.graph_layers = torch.nn.ModuleList(graph_layers)

        # Linear layers
        self.linear_size = linear_size if linear_size is not None else [512]
        self.linear_activation = linear_activation
        self.linear_dropout = linear_dropout

        self.descriptor_size = descriptor_size if descriptor_size is not None else 0
        self.descriptor_names = descriptor_names if descriptor_names is not None else []

        linear_layers = []
        in_features = self.graph_size[-1] * self.graph_heads * 2  # we have both mean and max read-out

        if self.descriptor_size is not None:
            in_features += self.descriptor_size

        for out_features in self.linear_size:
            if self.batch_norm:
                linear_layers.append(torch.nn.BatchNorm1d(in_features))
            linear_layers.append(torch.nn.Linear(in_features, out_features))
            linear_layers.append(self.linear_activation())
            if self.linear_dropout > 0:
                linear_layers.append(torch.nn.Dropout(p=self.linear_dropout))
            in_features = out_features

        self.linear_layers = torch.nn.Sequential(*linear_layers)

        # Output layer

        self.output_size = output_size
        self.output_layer = torch.nn.Linear(in_features, self.output_size)

        # Other
        self.loss_fn = None
        self.optimizer = None
        self.device = device
        self.to(self.device)
        self.train_loss = []
        self.eval_loss = []
        self.train_scores = []
        self.eval_scores = []
        self.verbose = 0

    def forward(self, x, edge_index, edge_attr, batch, meta_data):
        """
        TODO: Add learnable attention layer after/before concatenation
        """
        for layer in self.graph_layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)  # graph layer must implement edge_attr

        x_mean = global_mean_pool(x, batch)  # batch_size x n_heads * graph_size[-1]
        x_max = global_max_pool(x, batch)  # batch_size x n_heads * graph_size[-1]

        x = torch.cat((x_mean, x_max), dim=1)  # batch_size x n_heads * graph_size[-1] * 2

        if meta_data:
            tensors = torch.cat(meta_data, dim=1)
            x = torch.cat((x, tensors), dim=1)

        x_enc = self.linear_layers(x)
        x_out = self.output_layer(x_enc)

        return {'value': x_out.reshape(-1, self.output_size), 'linear_embedding': x_enc}

    def fit_epoch(self, train_dataloader):

        self.train()

        epoch_loss = 0.0
        num_entries = 0

        y_pred = []
        y_true = []

        for data_ in train_dataloader:
            self.optimizer.zero_grad()

            data_.to(self.device)

            x, edge_index, edge_attr, y, batch = data_.x, data_.edge_index, data_.edge_attr, data_.y, data_.batch

            # sorted should prevent the change of order; meta_data is a list of tensors
            meta_data = [value for key, value in sorted(data_.items()) if key in self.descriptor_names]
            predictions = self(x, edge_index, edge_attr, batch, meta_data)['value']

            mask = ~torch.isnan(y)
            norm = torch.sum(mask)

            masked_preds = predictions[mask]
            masked_targets = y[mask]

            if norm > 0:
                loss = self.loss_fn(masked_preds, masked_targets)
                loss.backward()
                epoch_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

                y_pred.append(masked_preds)
                y_true.append(masked_targets)

                num_entries += masked_targets.shape[0]

        y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
        y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()

        scores = self.scoring_function(y_true=y_true, y_pred=y_pred)

        self.train_loss.append(epoch_loss / num_entries)
        self.train_scores.append(scores)

    def eval_epoch(self, eval_dataloader):

        self.eval()

        epoch_loss = 0.0
        num_entries = 0

        y_pred = []
        y_true = []

        with torch.no_grad():
            for data_ in eval_dataloader:

                data_.to(self.device)
                x, edge_index, edge_attr, y, batch = data_.x, data_.edge_index, data_.edge_attr, data_.y, data_.batch

                meta_data = [value for key, value in sorted(data_.items()) if key in self.descriptor_names]
                predictions = self(x, edge_index, edge_attr, batch, meta_data)['value']

                mask = ~torch.isnan(y)
                norm = torch.sum(mask)

                masked_preds = predictions[mask]
                masked_targets = y[mask]

                if norm > 0:
                    loss = self.loss_fn(masked_preds, masked_targets)
                    epoch_loss += loss.item()

                    y_pred.append(masked_preds)
                    y_true.append(masked_targets)

                    num_entries += masked_targets.shape[0]

        y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
        y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()

        scores = self.scoring_function(y_true=y_true, y_pred=y_pred)

        self.eval_loss.append(epoch_loss / num_entries)
        self.eval_scores.append(scores)

    def check_descriptors(self, descriptors):
        if not isinstance(descriptors, list):
            raise TypeError(f'Expected descriptors to be of type < list >, got < {type(descriptors)} > instead')

        if not isinstance(descriptors[0], dict):
            raise TypeError(f'Expected items in the list to be < dict >, got < type{descriptors[0]} > instead')

        desc_expected = set(['SMILES'] + self.descriptor_names)
        desc_received = set(descriptors[0].keys())

        missing_desc = desc_expected.difference(desc_received)

        if missing_desc:
            raise AttributeError(f'Expected the following keys to be present: < {desc_expected} >, got < {desc_received} > instead')

    def check_and_convert_labels(self, labels):
        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError(f'Expected labels to be of type < list, np.ndarray >, got < {type(labels)} > instead')

        if isinstance(labels, list):
            entry = labels[0]
            if isinstance(entry, (int, float)):
                return np.array(labels, dtype=np.float64).reshape(-1, 1)
            if isinstance(entry, np.ndarray):
                if self.output_size != entry.shape[1]:
                    raise ValueError(f'Expected the number of tasks to be < {self.output_size} >, got < {entry.shape[1]} > instead')
                return np.vstack(labels).astype(np.float64)

        if isinstance(labels, np.ndarray):
            if len(labels.shape) == 1 and self.output_size == 1:
                return labels.astype(np.float64).reshape(-1, 1)

            if len(labels.shape) == 2:
                if labels.shape[1] != self.output_size:
                    raise ValueError(f'Expected the number of tasks to be < {self.output_size} >, got < {labels.shape[1]} > instead')
                return labels.astype(np.float64)

    @staticmethod
    def check_and_convert_weights(weights):
        if not isinstance(weights, (list, np.ndarray)):
            raise TypeError(f'Expected labels to be of type < list, np.ndarray >, got < {type(weights)} > instead')

        if isinstance(weights, list):
            return np.array(weights, dtype=np.float64).reshape(-1, 1)

        if isinstance(weights, np.ndarray):
            return weights.astype(np.float64).reshape(-1, 1)

    def fit(self, descriptors: List[dict], labels: Union[List, np.ndarray], weights: Union[List, np.ndarray] = None,
            n_epochs: int = 64, early_stop: int = 8, early_dir: str = None, save_freq: int = 8,
            save_dir: str = None, batch_size: int = 256, verbose: int = 0):

        """
        TODO: add sample weights
        Accepted formats of data:
        - descriptors should be a List of Dictionaries obtained through df[[columns]].to_dict(orient='records').
        - labels can be a List[int, float] OR 1D numpy ndarray
        - sample_weight should be a List[int, float] OR 1D numpy ndarray
        """

        if self.loss_fn is None or self.optimizer is None:
            raise ValueError(f'Please set the loss function and optimizer before attempting to fit the model.')

        self.check_descriptors(descriptors)
        labels = self.check_and_convert_labels(labels)
        weights = self.check_and_convert_weights(weights)

        if len(descriptors) < batch_size:
            batch_size = len(descriptors)

        train_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, labels))
        train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(1, n_epochs + 1):
            self.fit_epoch(train_dataloader)

            current_loss = self.train_loss[-1]
            if verbose > 0:
                print(f'Epoch {epoch} - current loss: {current_loss:.5f}')

            if early_stop is not None:
                if current_loss < min_loss:
                    min_loss = current_loss
                    patience = deepcopy(early_stop)
                    if early_dir is not None:
                        save_path = early_dir.rstrip('/') + f'/MMGNN_min.pth'
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    break

            if (save_freq is not None) and (save_dir is not None) and (epoch % save_freq == 0):
                save_path = save_dir.rstrip('/') + f'/MMGNN_{epoch}.pth'
                self.save(save_path)

    def fit_eval(self, train_descriptors: List[dict], eval_descriptors: List[dict],
                 train_labels: Union[List, np.ndarray], eval_labels: Union[List, np.ndarray],
                 n_epochs: int = 64, batch_size: int = 256, verbose: int = 0):
        """

            def fit(self, descriptors: List[dict], labels: Union[List, np.ndarray], n_epochs: int = 64,
            early_stop: int = 8, early_dir: str = None, save_freq: int = 8, save_dir: str = None,
            batch_size: int = 256, verbose: int = 0):
        TODO: add early stopping
        """
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError(f'Please set the loss function and optimizer before attempting to fit the model.')

        self.check_descriptors(train_descriptors)
        self.check_descriptors(eval_descriptors)

        train_labels = self.check_and_convert_labels(train_labels)
        eval_labels = self.check_and_convert_labels(eval_labels)

        min_samples = np.min(len(train_descriptors), len(eval_descriptors))
        if min_samples < batch_size:
            batch_size = min_samples

        train_dataset = GraphDataset(self.vectorizer.from_lists(train_descriptors, train_labels))
        train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        eval_dataset = GraphDataset(self.vectorizer.from_lists(eval_descriptors, eval_labels))
        eval_dataloader = torch_geometric.loader.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(n_epochs):
            self.fit_epoch(train_dataloader)
            self.eval_epoch(eval_dataloader)
            if verbose > 0:
                print(f'Current train loss: {self.train_loss[-1]:.5f}')
                print(f'Current eval loss: {self.eval_loss[-1]:.5f}')

    def embed(self, descriptors: List[dict], batch_size: int = 256):

        self.eval()
        self.vectorizer.mode = 'eval'

        self.check_descriptors(descriptors)

        embedding_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, np.zeros(shape=(len(descriptors), 1))))
        embedding_dataloader = torch_geometric.loader.DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for data_ in embedding_dataloader:
                data_.to(self.device)
                x, edge_index, edge_attr, batch = data_.x, data_.edge_index, data_.edge_attr, data_.batch
                meta_data = [value for key, value in sorted(data_.items()) if key in self.descriptor_names]

                embeddings = self(x, edge_index, edge_attr, batch, meta_data)['linear_embedding']
                embeddings.append(embeddings)

        return torch.cat(embeddings, dim=0).detach().cpu().numpy()

    def predict(self, descriptors: List[dict], batch_size: int = 256):

        self.eval()
        self.vectorizer.mode = 'eval'

        self.check_descriptors(descriptors)

        predict_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, np.zeros(shape=(len(descriptors), 1))))
        predict_dataloader = torch_geometric.loader.DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for data_ in predict_dataloader:
                data_.to(self.device)
                x, edge_index, edge_attr, batch = data_.x, data_.edge_index, data_.edge_attr, data_.batch
                meta_data = [value for key, value in sorted(data_.items()) if key in self.descriptor_names]

                logits = self(x, edge_index, edge_attr, batch, meta_data)['value']

                if self.task == 'classification':
                    predictions.append(torch.sigmoid(logits))
                else:
                    predictions.append(logits)

        return torch.cat(predictions, dim=0).detach().cpu().numpy()

    def set_loss_function(self, loss_fn=torch.nn.MSELoss, loss_params: dict = None):
        if callable(loss_fn):
            self.loss_fn = loss_fn(**loss_params) if loss_params else loss_fn()
        else:
            raise ValueError("Loss function must be callable")

    def set_optimizer(self, optim=torch.optim.AdamW, optim_params: dict = None):
        if callable(optim):
            self.optimizer = optim(self.parameters(), **optim_params) if optim_params else optim(self.parameters())
        else:
            raise ValueError("Optimizer must be callable")

    @staticmethod
    def init_linear(layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def save(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def score_classification(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):

        y_pred_bin = (y_pred >= threshold).astype(int)

        if self.output_size == 1:
            scores = {
                'Accuracy': accuracy_score(y_true, y_pred_bin),
                'Recall': recall_score(y_true, y_pred_bin),
                'Precision': precision_score(y_true, y_pred_bin, zero_division=0),
                'F1': f1_score(y_true, y_pred_bin, zero_division=0),
                'MCC': matthews_corrcoef(y_true, y_pred_bin),
                'ROC AUC': roc_auc_score(y_true, y_pred)
            }
        else:
            scores = {}
            for i in range(self.output_size):
                task_score = {
                    'Accuracy': accuracy_score(y_true[:, i], y_pred_bin[:, i]),
                    'Recall': recall_score(y_true[:, i], y_pred_bin[:, i]),
                    'Precision': precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
                    'F1': f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
                    'MCC': matthews_corrcoef(y_true[:, i], y_pred_bin[:, i]),
                    'ROC AUC': roc_auc_score(y_true[:, i], y_pred[:, i])
                }
                scores[f'Task {i}'] = task_score
        return scores

    def score_regression(self, y_true: np.ndarray, y_pred: np.ndarray):

        if self.output_size == 1:
            scores = {
                'R2': r2_score(y_true, y_pred),
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': mean_squared_error(y_true, y_pred, squared=False)
            }
        else:
            scores = {}
            for i in range(self.output_size):
                task_score = {
                    'R2': r2_score(y_true[:, i], y_pred[:, i]),
                    'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                    'RMSE': mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
                }
                scores[f'Task {i}'] = task_score
        return scores

    def plot_loss(self):
        sns.set_style('whitegrid')
        sns.set_context('notebook')
        if not self.train_loss:
            print('Please train your model first')
            return

        if not self.eval_loss:
            loss_df = pd.DataFrame({'Epoch': np.arange(1, len(self.train_loss) + 1), 'Loss': self.train_loss})
            g = sns.relplot(loss_df, x='Epoch', y='Loss', kind='line')

        else:
            loss_df = pd.DataFrame({'Epoch': np.arange(1, len(self.train_loss) + 1), 'Train': self.train_loss, 'Eval': self.eval_loss})
            loss_df = pd.melt(loss_df, id_vars='Epoch', value_vars=['Train', 'Eval'], var_name='Dataset', value_name='Loss')
            g = sns.relplot(loss_df, x='Epoch', y='Loss', col='Dataset', kind='line')

        return g

    def plot_scores(self):
        sns.set_style('whitegrid')
        sns.set_context('notebook')
        if not self.train_scores:
            print('Please train your model first')
            return

        if not self.eval_loss:
            scores_df = pd.DataFrame(self.train_scores)
            scores_df['Epoch'] = scores_df.index
            scores_df = pd.melt(scores_df, id_vars='Epoch', var_name='Metric', value_name='Value')
            g = sns.relplot(scores_df, x='Epoch', y='Value', hue='Metric', kind='line')

        else:
            train_scores_df = pd.DataFrame(self.train_scores)
            train_scores_df['Epoch'] = train_scores_df.index
            train_scores_df = pd.melt(train_scores_df, id_vars='Epoch', var_name='Metric', value_name='Train')

            eval_scores_df = pd.DataFrame(self.eval_scores)
            eval_scores_df['Epoch'] = eval_scores_df.index
            eval_scores_df = pd.melt(eval_scores_df, id_vars='Epoch', var_name='Metric', value_name='Eval')

            scores_df = train_scores_df.merge(eval_scores_df, on=['Epoch', 'Metric'])
            scores_df = pd.melt(scores_df, id_vars=['Epoch', 'Metric'], value_vars=['Train', 'Eval'], var_name='Dataset', value_name='Value')
            g = sns.relplot(scores_df, x='Epoch', y='Value', hue='Metric', kind='line', col='Dataset')


class MMWGNN(torch.nn.Module):
    """
    Multimodal Weighted Graph Neural Network
    """

    def __init__(self, vectorizer: MMGV, device: str = 'cpu', batch_norm: bool = True, max_norm: float = 1.0,
                 descriptor_params: dict = None, graph_layer=None,
                 graph_size: List[int] = None, graph_heads: int = 1, graph_dropout: float = 0.25,
                 attn_dim: int = 128, attn_heads: int = 1, attn_dropout: float = 0.0, query_name: str = 'Demo_FP',
                 linear_sizes: List[int] = None, linear_activation: torch.nn.Module = torch.nn.ReLU, linear_dropout: float = 0.25,
                 output_size: int = 1, task: str = 'classification'):

        """
        The architecture is as follows:
        - input: torch_geometric.Graph with x, edge_index, edge_attr, y, and optional other parameters (e.g. RDKit descriptors)
        - smiles string is transformed into graph using vectorizer
        - by default GATv2Conv is used for message passing
        - max and mean pool are applied as a read-out functions and processed by linear_layers
        - outputs from pooling are concatenated with optional descriptors and processed by concat_layers
        - the final output is obtained
        TODO: Correct sample weighting schemes for classical metrics, i.e. somehow factor in the fact that we're not really sure about some of them
        TODO: Implement ordinal classification (apparently using torch-ordinal or Corn ordinal regression)
        TODO: Add explicit modelling of two parameters: Class and Confidence
        """

        super(MMWGNN, self).__init__()

        # Initial setup
        self.vectorizer = vectorizer
        self.batch_norm = batch_norm
        self.max_norm = max_norm
        self.task = task

        if self.task == 'classification':
            self.scoring_function = self.score_classification
        elif self.task == 'regression':
            self.scoring_function = self.score_regression
        else:
            raise ValueError('Task must be either classification or regression')

        # Graph layers
        self.graph_layer = graph_layer if graph_layer is not None else torch_geometric.nn.GATv2Conv
        self.use_edge_attr = 'edge_attr' in inspect.signature(self.graph_layer.forward).parameters
        self.graph_size = graph_size if graph_size is not None else [32]
        self.graph_heads = graph_heads
        self.graph_dropout = graph_dropout

        graph_layers = []

        in_channels = 39  # size of atom embedding = 39, size of edge embedding = 8

        for out_channels in self.graph_size:
            graph_layers.append(self.graph_layer(in_channels, out_channels, heads=self.graph_heads,
                                                 dropout=self.graph_dropout, edge_dim=8))
            in_channels = out_channels * self.graph_heads

        self.graph_layers = torch.nn.ModuleList(graph_layers)

        # Descriptor layers and attention
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.query_name = query_name

        self.linear_activation = linear_activation
        self.linear_dropout = linear_dropout

        self.descriptor_params = descriptor_params if descriptor_params is not None else {}
        self.descriptor_params['Graph'] = [self.graph_size[-1] * self.graph_heads * 2, 128]

        self.branch_layers = torch.nn.ModuleDict({
            name: self.make_linear_layers(sizes)
            for name, sizes in self.descriptor_params.items()
        })

        self.projection_layers = torch.nn.ModuleDict({
            name: torch.nn.Linear(sizes[-1], self.attn_dim)
            for name, sizes in self.descriptor_params.items()
        })

        self.attn_layer = torch.nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=self.attn_heads,
                                                      batch_first=True, dropout=self.attn_dropout)

        # Post-attention layers
        self.linear_sizes = linear_sizes if linear_sizes is not None else [128]
        self.linear_layers = self.make_linear_layers([self.attn_dim] + self.linear_sizes)  # we explicitly add Demo_FP again

        # Output layer
        self.output_size = output_size
        self.output_layer = torch.nn.Linear(self.linear_sizes[-1], self.output_size)

        # Other
        self.loss_fn = None
        self.optimizer = None
        self.device = device
        self.to(self.device)
        self.train_loss = []
        self.eval_loss = []
        self.train_scores = []
        self.eval_scores = []
        self.attention_weights = []
        self.verbose = 0
        self.testing = None

    def forward(self, x, edge_index, edge_attr, batch, meta_data):

        # Process graph layers
        for layer in self.graph_layers:
            if self.use_edge_attr:
                x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                x = layer(x=x, edge_index=edge_index)

        x_mean = global_mean_pool(x, batch)  # batch_size * n_heads * graph_size[-1]
        x_max = global_max_pool(x, batch)  # batch_size * n_heads * graph_size[-1]

        meta_data['Graph'] = torch.cat((x_mean, x_max), dim=1)

        projections = {}

        # Process descriptor branches
        for name, tensor in meta_data.items():
            embedding = self.branch_layers[name](tensor)
            projections[name] = self.projection_layers[name](embedding)

        query = projections.pop(self.query_name)

        desc_keys = sorted(projections.keys())

        kv = torch.cat([projections[k] for k in desc_keys], dim=1)
        kv = kv.view(kv.size(0), -1, query.size(1))

        attn_out, attn_weights = self.attn_layer(
            query=query.unsqueeze(1),
            key=kv,
            value=kv,
            need_weights=True
        )

        attn_weights = attn_weights.squeeze(1)
        attn_weights = {
            'weights': attn_weights,
            'keys': desc_keys
        }

        x = attn_out.squeeze(1)  # batch, attn_dim

        x_enc = self.linear_layers(x)
        x_out = self.output_layer(x_enc)

        return {'value': x_out.reshape(-1, self.output_size), 'attention_embedding': attn_out, 'linear_embedding': x_enc,
                'attention_weights': attn_weights}

    def fit_epoch(self, train_dataloader):

        self.train()

        loss_per_task = torch.zeros(self.output_size, device=self.device)
        weight_per_task = torch.zeros(self.output_size, device=self.device)

        epoch_loss = 0.0
        epoch_weight = 0.0

        y_pred = []
        y_true = []
        y_wgts = []

        attn_weights_all = []
        attn_keys = None

        for data_ in train_dataloader:
            self.optimizer.zero_grad()

            data_.to(self.device)

            x, edge_index, edge_attr, y, batch, weights = data_.x, data_.edge_index, data_.edge_attr, data_.y, data_.batch, data_.weights
            meta_data = {key: value for key, value in data_.items() if key in self.descriptor_params.keys()}

            out = self(x, edge_index, edge_attr, batch, meta_data)
            predictions = out['value']

            mask = ~torch.isnan(y)

            masked_preds = predictions[mask]
            masked_targets = y[mask]
            masked_weights = weights[mask]

            if mask.sum() > 0:

                loss = self.loss_fn(masked_preds, masked_targets)
                weighted_loss = (loss * masked_weights).sum() / masked_weights.sum()
                weighted_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

                batch_weights = masked_weights.sum().item()

                epoch_loss += weighted_loss.item() * batch_weights
                epoch_weight += batch_weights

                y_pred.append(masked_preds)
                y_true.append(masked_targets)
                y_wgts.append(masked_weights)

            attention_weights = out['attention_weights']

            if attention_weights is not None:
                attn_weights_all.append(attention_weights['weights'].cpu())
                if attn_keys is None:
                    attn_keys = attention_weights['keys']

        y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
        y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
        y_wgts = torch.cat(y_wgts, dim=0).detach().cpu().numpy()

        scores = self.scoring_function(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts)

        self.train_loss.append(np.round(epoch_loss / epoch_weight, 5))
        self.train_scores.append(scores)

        if attn_weights_all:
            attn_tensor = torch.cat(attn_weights_all, dim=0)
            attn_tensor = attn_tensor / attn_tensor.sum(dim=1, keepdim=True)

            attn_mean = attn_tensor.mean(dim=0)
            attn_std = attn_tensor.std(dim=0)

            attn_mean = attn_mean / attn_mean.sum()

            self.attention_weights.append({
                'weights': attn_mean.detach().numpy(),
                'std': attn_std.detach().numpy(),
                'keys': attn_keys
            })

    def eval_epoch(self, eval_dataloader):

        self.eval()

        epoch_loss = 0.0
        epoch_weight = 0.0

        y_pred = []
        y_true = []
        y_wgts = []

        with torch.no_grad():
            for data_ in eval_dataloader:

                data_.to(self.device)
                x, edge_index, edge_attr, y, batch, weights = data_.x, data_.edge_index, data_.edge_attr, data_.y, data_.batch, data_.weights

                meta_data = {key: value for key, value in data_.items() if key in self.descriptor_params.keys()}
                predictions = self(x, edge_index, edge_attr, batch, meta_data)['value']

                mask = ~torch.isnan(y)

                masked_preds = predictions[mask]
                masked_targets = y[mask]
                masked_weights = weights[mask]

                if mask.sum() > 0:

                    loss = self.loss_fn(masked_preds, masked_targets)
                    weighted_loss = (loss * masked_weights).sum() / masked_weights.sum()

                    y_pred.append(masked_preds)
                    y_true.append(masked_targets)
                    y_wgts.append(masked_weights)

                    batch_weights = masked_weights.sum().item()

                    epoch_loss += weighted_loss.item() * batch_weights
                    epoch_weight += batch_weights

        y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
        y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
        y_wgts = torch.cat(y_wgts, dim=0).detach().cpu().numpy()

        scores = self.scoring_function(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts)

        self.eval_loss.append(np.round(epoch_loss / epoch_weight, 5))
        self.eval_scores.append(scores)

    @staticmethod
    def check_descriptors(descriptors):
        if not isinstance(descriptors, list):
            raise TypeError(f'Expected descriptors to be of type < list >, got < {type(descriptors)} > instead')

        if not isinstance(descriptors[0], dict):
            raise TypeError(f'Expected items in the list to be < dict >, got < type{descriptors[0]} > instead')

    def check_and_convert_labels(self, labels):
        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError(f'Expected labels to be of type < list, np.ndarray >, got < {type(labels)} > instead')

        if isinstance(labels, list):
            entry = labels[0]
            if isinstance(entry, (int, float)):
                return np.array(labels, dtype=np.float64).reshape(-1, 1)
            if isinstance(entry, np.ndarray):
                if self.output_size != entry.shape[1]:
                    raise ValueError(f'Expected the number of tasks to be < {self.output_size} >, got < {entry.shape[1]} > instead')
                return np.vstack(labels).astype(np.float64)

        if isinstance(labels, np.ndarray):
            if len(labels.shape) == 1 and self.output_size == 1:
                return labels.astype(np.float64).reshape(-1, 1)

            if len(labels.shape) == 2:
                if labels.shape[1] != self.output_size:
                    raise ValueError(f'Expected the number of tasks to be < {self.output_size} >, got < {labels.shape[1]} > instead')
                return labels.astype(np.float64)

    @staticmethod
    def check_and_convert_weights(weights):
        if not isinstance(weights, (list, np.ndarray)):
            raise TypeError(f'Expected weights to be of type < list, np.ndarray >, got < {type(weights)} > instead')

        if isinstance(weights, list):
            return np.array(weights, dtype=np.float64).reshape(-1, 1)

        if isinstance(weights, np.ndarray):
            return weights.astype(np.float64).reshape(-1, 1)

    def fit(self, descriptors: List[dict], labels: Union[List, np.ndarray], weights: Union[List, np.ndarray] = None,
            n_epochs: int = 64, early_stop: int = 8, early_dir: str = None, save_freq: int = 8,
            save_dir: str = None, batch_size: int = 256):

        """
        Accepted formats of data:
        - descriptors should be a List of Dictionaries obtained through df[[columns]].to_dict(orient='records').
        - labels can be a List[int, float] OR 1D numpy ndarray
        - sample_weight should be a List[int, float] OR List[1D np ndarray]
        """

        if self.loss_fn is None or self.optimizer is None:
            raise ValueError(f'Please set the loss function and optimizer before attempting to fit the model.')

        self.check_descriptors(descriptors)
        labels = self.check_and_convert_labels(labels)

        if weights is not None:
            weights = self.check_and_convert_weights(weights)
            train_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, labels, weights))
        else:
            train_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, labels))

        train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(1, n_epochs + 1):
            self.fit_epoch(train_dataloader)

            current_loss = self.train_loss[-1]
            if self.verbose > 0:
                print(f'Epoch {epoch} - current loss: {current_loss:.5f}')

            if early_stop is not None:
                if current_loss < min_loss:
                    min_loss = current_loss
                    patience = deepcopy(early_stop)
                    if early_dir is not None:
                        save_path = early_dir.rstrip('/') + f'/MMGNN_min.pth'
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    break

            if (save_freq is not None) and (save_dir is not None) and (epoch % save_freq == 0):
                save_path = save_dir.rstrip('/') + f'/MMGNN_{epoch}.pth'
                self.save(save_path)

    def fit_eval(self, train_descriptors: List[dict], eval_descriptors: List[dict],
                 train_labels: Union[List, np.ndarray], eval_labels: Union[List, np.ndarray],
                 train_weights: Union[List, np.ndarray] = None, eval_weights: Union[List, np.ndarray] = None,
                 n_epochs: int = 64, batch_size: int = 256):
        """

            def fit(self, descriptors: List[dict], labels: Union[List, np.ndarray], n_epochs: int = 64,
            early_stop: int = 8, early_dir: str = None, save_freq: int = 8, save_dir: str = None,
            batch_size: int = 256, verbose: int = 0):
        TODO: add early stopping
        """
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError(f'Please set the loss function and optimizer before attempting to fit the model.')

        self.check_descriptors(train_descriptors)
        self.check_descriptors(eval_descriptors)

        train_labels = self.check_and_convert_labels(train_labels)
        eval_labels = self.check_and_convert_labels(eval_labels)

        min_samples = np.min([train_labels.shape[0], eval_labels.shape[0]])
        if min_samples < batch_size:
            batch_size = min_samples

        if any([train_weights is None, eval_weights is None]):
            train_dataset = GraphDataset(self.vectorizer.from_lists(train_descriptors, train_labels))
            eval_dataset = GraphDataset(self.vectorizer.from_lists(eval_descriptors, eval_labels))
        else:
            train_weights = self.check_and_convert_weights(train_weights)
            eval_weights = self.check_and_convert_weights(eval_weights)
            train_dataset = GraphDataset(self.vectorizer.from_lists(train_descriptors, train_labels, train_weights))
            eval_dataset = GraphDataset(self.vectorizer.from_lists(eval_descriptors, eval_labels, eval_weights))

        train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        eval_dataloader = torch_geometric.loader.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(n_epochs):
            self.fit_epoch(train_dataloader)
            self.eval_epoch(eval_dataloader)
            if self.verbose > 0:
                print(f'Train loss: {self.train_loss[-1]:.5f}')
                print(f'Eval loss: {self.eval_loss[-1]:.5f}')

    def embed(self, descriptors: List[dict], batch_size: int = 256):

        self.eval()
        self.vectorizer.mode = 'eval'

        self.check_descriptors(descriptors)

        embedding_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, np.zeros(shape=(len(descriptors), 1))))
        embedding_dataloader = torch_geometric.loader.DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for data_ in embedding_dataloader:
                data_.to(self.device)
                x, edge_index, edge_attr, batch = data_.x, data_.edge_index, data_.edge_attr, data_.batch
                meta_data = {key: value for key, value in data_.items() if key in self.descriptor_params.keys()}

                embeddings = self(x, edge_index, edge_attr, batch, meta_data)['linear_embedding']
                embeddings.append(embeddings)

        return torch.cat(embeddings, dim=0).detach().cpu().numpy()

    def predict(self, descriptors: List[dict], batch_size: int = 256):

        self.eval()
        self.vectorizer.mode = 'eval'

        self.check_descriptors(descriptors)

        predict_dataset = GraphDataset(self.vectorizer.from_lists(descriptors, np.zeros(shape=(len(descriptors), 1))))
        predict_dataloader = torch_geometric.loader.DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for data_ in predict_dataloader:
                data_.to(self.device)
                x, edge_index, edge_attr, batch = data_.x, data_.edge_index, data_.edge_attr, data_.batch
                meta_data = {key: value for key, value in data_.items() if key in self.descriptor_params.keys()}

                logits = self(x, edge_index, edge_attr, batch, meta_data)['value']

                if self.task == 'classification':
                    predictions.append(torch.sigmoid(logits))
                else:
                    predictions.append(logits)

        return torch.cat(predictions, dim=0).detach().cpu().numpy()

    def set_loss_function(self, loss_fn=torch.nn.MSELoss, loss_params: dict = None):
        if callable(loss_fn):
            self.loss_fn = loss_fn(**loss_params) if loss_params else loss_fn()
        else:
            raise ValueError("Loss function must be callable")

    def set_optimizer(self, optim=torch.optim.AdamW, optim_params: dict = None):
        if callable(optim):
            self.optimizer = optim(self.parameters(), **optim_params) if optim_params else optim(self.parameters())
        else:
            raise ValueError("Optimizer must be callable")

    @staticmethod
    def init_linear(layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def make_linear_layers(self, linear_sizes: List[int]):

        linear_layers = []

        sizes = linear_sizes.copy()
        in_features = sizes.pop(0)

        for out_features in sizes:
            if self.batch_norm:
                linear_layers.append(torch.nn.BatchNorm1d(in_features))

            linear = torch.nn.Linear(in_features, out_features)
            self.init_linear(linear)
            linear_layers.append(linear)

            linear_layers.append(self.linear_activation())
            if self.linear_dropout > 0:
                linear_layers.append(torch.nn.Dropout(p=self.linear_dropout))
            in_features = out_features

        return torch.nn.Sequential(*linear_layers)

    def save(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @staticmethod
    def score_classification(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray, threshold: float = 0.5):
        """
        Score one task (!)
        """
        y_pred_bin = (y_pred >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred_bin, sample_weight=sample_weight).ravel()

        scores = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Accuracy': (tp + tn) / (tp + fp + fn + tn),
            'Recall': tp / (tp + fn),
            'Specificity': tn / (tn + fp),
            'Precision': tp / (tp + fp),
            'F1 score': (2 * tp) / (2 * tp + fp + fn),
            'ROC AUC': roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight),
            'MCC': matthews_corrcoef(y_true=y_true, y_pred=y_pred_bin, sample_weight=sample_weight)
        }
        return scores

    @staticmethod
    def score_regression(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray):
        """
        Score one task (!)
        """

        scores = {
            'R2': r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
            'MAE': mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
            'RMSE': mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, squared=False)
        }
        return scores

    def plot_loss(self, save_path: str = None):
        sns.set_style('whitegrid')
        sns.set_context('notebook')
        if not self.train_loss:
            print('Please train your model first')
            return

        if not self.eval_loss:
            loss_df = pd.DataFrame({'Epoch': np.arange(1, len(self.train_loss) + 1), 'Loss': self.train_loss})
            g = sns.relplot(loss_df, x='Epoch', y='Loss', kind='line')

        else:
            loss_df = pd.DataFrame({'Epoch': np.arange(1, len(self.train_loss) + 1), 'Train': self.train_loss, 'Eval': self.eval_loss})
            loss_df = pd.melt(loss_df, id_vars='Epoch', value_vars=['Train', 'Eval'], var_name='Dataset', value_name='Loss')
            g = sns.relplot(loss_df, x='Epoch', y='Loss', col='Dataset', kind='line')

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_scores(self, save_path: str = None):
        sns.set_style('whitegrid')
        sns.set_context('notebook')
        if not self.train_scores:
            print('Please train your model first')
            return

        if not self.eval_loss:
            scores_df = pd.DataFrame(self.train_scores).drop(columns=['TP', 'FP', 'FN', 'TN'])
            scores_df['Epoch'] = scores_df.index
            scores_df = pd.melt(scores_df, id_vars='Epoch', var_name='Metric', value_name='Value')
            g = sns.relplot(scores_df, x='Epoch', y='Value', hue='Metric', kind='line')

        else:
            train_scores_df = pd.DataFrame(self.train_scores).drop(columns=['TP', 'FP', 'FN', 'TN'])
            train_scores_df['Epoch'] = train_scores_df.index
            train_scores_df = pd.melt(train_scores_df, id_vars='Epoch', var_name='Metric', value_name='Train')

            eval_scores_df = pd.DataFrame(self.eval_scores).drop(columns=['TP', 'FP', 'FN', 'TN'])
            eval_scores_df['Epoch'] = eval_scores_df.index
            eval_scores_df = pd.melt(eval_scores_df, id_vars='Epoch', var_name='Metric', value_name='Eval')

            scores_df = train_scores_df.merge(eval_scores_df, on=['Epoch', 'Metric'])
            scores_df = pd.melt(scores_df, id_vars=['Epoch', 'Metric'], value_vars=['Train', 'Eval'], var_name='Dataset', value_name='Value')
            g = sns.relplot(scores_df, x='Epoch', y='Value', hue='Metric', kind='line', col='Dataset')

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_attention(self, save_path: str = None):
        sns.set_style('whitegrid')
        sns.set_context('notebook')
        if not self.attention_weights:
            print('Please train your model first')
            return

        records = []

        for epoch_idx, attn_record in enumerate(self.attention_weights):
            weights = attn_record["weights"]
            stds = attn_record["std"]
            keys = attn_record["keys"]

            for idx, key in enumerate(keys):
                weight_val = weights[idx]
                weight_std = stds[idx]
                records.append({
                    "Epoch": epoch_idx + 1,
                    "Descriptor": key,
                    "Attention Weight": weight_val,
                    "Lower": weight_val - weight_std,
                    "Upper": weight_val + weight_std
                })

        df = pd.DataFrame.from_records(records)

        plt.figure(figsize=(10, 6))

        # Plot mean lines
        sns.lineplot(data=df, x="Epoch", y="Attention Weight", hue="Descriptor", marker="o", legend=True)

        # Add shaded bands for std
        for descriptor in df["Descriptor"].unique():
            sub_df = df[df["Descriptor"] == descriptor]
            plt.fill_between(
                sub_df["Epoch"],
                sub_df["Lower"],
                sub_df["Upper"],
                alpha=0.2,
            )

        plt.title("Descriptor Attention Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Attention Weight")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()



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

            gnn_layers, gnn_out_size = self.build_graph_layers()
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
            rnn_layer, rnn_embedding, rnn_out_size = self.build_recurrent_layers()
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

                desc_layers, desc_out_size = self.build_linear_layers(**params)
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
        self.query_layers, query_out_size = self.build_linear_layers(**self.query_params)
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

            current_loss = train_loss['Total']

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

    def build_graph_layers(self):
        layer_class = self.gnn_params['layer']
        layer_type = self.gnn_params.get('layer_type')  # default
        sizes = self.gnn_params['sizes']
        input_dim = self.gnn_params['input_dim']
        layer_args = self.gnn_params.get('args', [{} for _ in sizes])
        activation = self.gnn_params.get('activation', 'relu')
        dropout = self.gnn_params.get('dropout', 0.0)

        gnn_blocks = []
        in_dim = input_dim

        for i, (out_dim, args) in enumerate(zip(sizes, layer_args)):

            if layer_type == 'attention':
                heads = self.gnn_params.get('heads', 1)
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

            batch_norm_layer = nn.BatchNorm1d(next_in) if self.batch_norm else None
            activation_layer = self.get_activation_fn(activation) if activation else None
            dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

            gnn_blocks.append(
                GNNLayerBlock(
                    graph_layer=graph_layer,
                    batch_norm=batch_norm_layer,
                    activation=activation_layer,
                    dropout=dropout_layer
                )
            )
            in_dim = next_in

        return nn.ModuleList(gnn_blocks), in_dim

    def build_conv_layers(self):
        alphabet_len = self.cnn_params['alphabet_len']
        embedding_dim = self.cnn_params['embedding_dim']  # i.e. in_channels for the first layer
        padding_idx = self.cnn_params.get('padding_idx', 0)

        cnn_embedding = nn.Embedding(
            num_embeddings=alphabet_len,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        sizes = self.cnn_params.get('sizes', [256])
        kernel_size = self.cnn_params.get('kernel_size', 5)
        stride = self.cnn_params.get('stride', 1)
        dropout = self.cnn_params.get('dropout', 0.1)
        activation_fn = self.get_activation_fn(self.cnn_params.get('activation', 'relu'))
        pool_kernel_size = self.cnn_params.get('pool_size', 2)

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
            batch_norm = nn.BatchNorm1d(out_channels) if self.batch_norm else None
            activation = activation_fn if activation_fn else None
            max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size) if pool_kernel_size > 1 else None
            dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

            cnn_blocks.append(
                CNNLayerBlock(
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
            RNNLayerBlock(
                recurrent_layer=rnn_layer,
                max_len=max_len
            )
        )
        output_dim = hidden_size

        return nn.ModuleList(rnn_blocks), rnn_embedding, output_dim

    def build_linear_layers(self, sizes: List[int], batch_norm: bool = True,
                            activation: str = 'relu', dropout: float = 0.0):

        linear_layers = []

        linear_sizes = sizes.copy()
        lin_out_size = linear_sizes[-1]
        in_features = linear_sizes.pop(0)

        for out_features in linear_sizes:
            linear = torch.nn.Linear(in_features, out_features)
            self.init_linear(linear)
            linear_layers.append(linear)

            if batch_norm:
                linear_layers.append(torch.nn.BatchNorm1d(out_features))
            if activation is not None:
                linear_layers.append(self.get_activation_fn(activation))
            if dropout > 0:
                linear_layers.append(torch.nn.Dropout(p=dropout))

            in_features = out_features

        return torch.nn.Sequential(*linear_layers), lin_out_size

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
            raise ValueError(f"Unsupported activation function: {name}")

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


class BetaMMMTGNN(nn.Module):
    """
    Multi-Modal Multi-Task Generalized Neural Network.
    The model accepts input in a form of dataframe.to_dict(orient='records').

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

    def __init__(self, device: str = 'cpu', task: str = 'classification',
                 label_name: str = None, weight_name: str = None, signature_name: str = None,
                 gnn_params: dict = None, cnn_params: dict = None, rnn_params: dict = None,
                 des_params: dict = None, att_params: dict = None, lin_params: dict = None,
                 batch_norm: bool = True, max_norm: float = 1.0, query_desc: str = 'Demo'):

        super(BetaMMMTGNN, self).__init__()

        self.set_seed(42)
        self.hparams = self.get_hyperparameters()
        self.device = device
        self.task = task
        self.metrics_function = {'classification': self.score_classification, 'regression': self.score_regression}.get(self.task, None)
        self.label_name = label_name
        self.weight_name = weight_name
        self.signature_name = signature_name
        self.gnn_params = deepcopy(gnn_params)
        self.cnn_params = deepcopy(cnn_params)
        self.rnn_params = deepcopy(rnn_params)
        self.des_params = deepcopy(des_params)
        self.att_params = deepcopy(att_params)
        self.lin_params = deepcopy(lin_params)
        self.batch_norm = batch_norm
        self.max_norm = max_norm
        self.query_desc = query_desc
        self.query_params = self.des_params.pop(self.query_desc)

        self.backbones = nn.ModuleDict()
        self.desc_names = list()
        self.loss_fn = self.BetaNLLLoss
        self.optimizer = None
        self.scheduler = None
        self.logs = defaultdict(list)
        self.epoch = 1
        self.best_state_dict = None
        self.best_epoch = None

        # Graph layers
        if self.gnn_params is not None:  # i.e. if we want to have a Graph module
            self.gnn_input_name = self.gnn_params.get('input_name', 'Graph')

            gnn_layers, gnn_out_size = self.build_graph_layers()
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
            rnn_layer, rnn_embedding, rnn_out_size = self.build_recurrent_layers()
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

                desc_layers, desc_out_size = self.build_linear_layers(**params)
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
        self.query_layers, query_out_size = self.build_linear_layers(**self.query_params)
        self.query_layers.append(nn.Linear(in_features=query_out_size, out_features=attn_size))

        # Linear layers and output
        in_features = len(self.backbones) * attn_size  # number of modalities * att_output
        self.lin_params['sizes'] = [in_features] + self.lin_params['sizes']
        self.lin_layers, _ = self.build_linear_layers(**self.lin_params)
        self.lin_layers.append(nn.Linear(in_features=self.lin_params['sizes'][-1], out_features=2))

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

        x_out = self.lin_layers(torch.cat(outputs, dim=1))  # batch_size x 2
        alpha_beta = nn.functional.softplus(x_out) + 1e-6
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        ab_sum = alpha + beta
        mu = alpha / ab_sum
        var = (alpha * beta) / (ab_sum**2 * (ab_sum + 1))

        return {'alpha': alpha, 'beta': beta,
                'mu': mu, 'var': var,
                'attention_weights': attention_weights}

    def fit_epoch(self, dataloader):
        self.train()

        epoch_losses = []
        epoch_data = {
            'mu': [],
            'y_true': [],
            'y_wgts': [],
            'y_sign': defaultdict(list)
        }
        epoch_grad_norms = []

        for batch in dataloader:
            self.optimizer.zero_grad()

            out = self(batch)
            alpha, beta, mu = out['alpha'], out['beta'], out['mu']
            y_true = batch[self.label_name].to(self.device).squeeze(1)
            y_wgts = batch[self.weight_name].to(self.device).squeeze(1)
            y_sign = batch[self.signature_name] if self.signature_name is not None else None

            epoch_data['mu'].append(mu)
            epoch_data['y_true'].append(y_true)
            epoch_data['y_wgts'].append(y_wgts)

            for sign_type, sign_values in y_sign.items():
                epoch_data['y_sign'][sign_type].extend(sign_values)

            batch_loss = self.mws_loss(alpha=alpha, beta=beta, y_true=y_true, y_wgts=y_wgts, y_sign=y_sign)

            loss, loss_wgt = batch_loss.get('Total')  # this one already includes masking and weights
            loss.backward()

            batch_grad_norm = self.grad_norm() / (loss_wgt + 1e-6)
            epoch_grad_norms.append(batch_grad_norm)

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            epoch_losses.append(batch_loss)

        epoch_loss = reduce(self.combine_mws_losses, epoch_losses)
        epoch_loss = self.normalize_mws_loss(epoch_loss)

        epoch_metrics = self.mws_metrics(**self.pack_epoch_data(epoch_data))

        return epoch_loss, epoch_metrics, epoch_grad_norms

    def eval_epoch(self, dataloader):
        self.eval()

        epoch_losses = []
        epoch_data = {
            'mu': [],
            'y_true': [],
            'y_wgts': [],
            'y_sign': defaultdict(list)
        }

        with torch.no_grad():
            for batch in dataloader:

                out = self(batch)
                alpha, beta, mu = out['alpha'], out['beta'], out['mu']
                y_true = batch[self.label_name].to(self.device)
                y_wgts = batch[self.weight_name].to(self.device)
                y_sign = batch[self.signature_name] if self.signature_name is not None else None

                epoch_data['mu'].append(mu)
                epoch_data['y_true'].append(y_true)
                epoch_data['y_wgts'].append(y_wgts)

                for sign_type, sign_values in y_sign.items():
                    epoch_data['y_sign'][sign_type].extend(sign_values)

                batch_loss = self.mws_loss(alpha=alpha, beta=beta, y_true=y_true, y_wgts=y_wgts, y_sign=y_sign)

                epoch_losses.append(batch_loss)

        epoch_loss = reduce(self.combine_mws_losses, epoch_losses)
        epoch_loss = self.normalize_mws_loss(epoch_loss)

        epoch_metrics = self.mws_metrics(**self.pack_epoch_data(epoch_data))

        return epoch_loss, epoch_metrics

    def fit(self, dataloader, n_epochs: int = 4, early_stop: int = 8,
            save_freq: int = 8, save_dir: str = None, verbose: int = 0):

        if self.optimizer is None:
            raise ValueError(f'Please set the optimizer before attempting to fit the model.')

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(self.epoch, self.epoch + n_epochs):
            train_loss, train_metrics, train_grad_norms = self.fit_epoch(dataloader)
            self.logs['train_loss'].append((epoch, train_loss))
            self.logs['train_metrics'].append((epoch, train_metrics))
            self.logs['train_grad_norms'].append((epoch, train_grad_norms))
            self.epoch += 1

            current_loss = train_loss['Total']

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

        if self.optimizer is None:
            raise ValueError(f'Set the optimizer before fitting the model.')

        min_loss = float('inf')
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch in range(self.epoch, self.epoch + n_epochs):
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

        pred_mu = []
        pred_var = []

        with torch.no_grad():
            for batch in dataloader:
                out = self(batch)
                mu, var = out['mu'], out['var']
                pred_mu.append(mu)
                pred_var.append(var)

        pred_mu = torch.cat(pred_mu, dim=0).detach().cpu().numpy()
        pred_var = torch.cat(pred_var, dim=0).detach().cpu().numpy()

        return pred_mu, pred_var

    def mws_loss(self, alpha, beta, y_true, y_wgts, y_sign: dict):
        """
        Masked Weighted Signature Loss (MWS Loss) with hierarchical signatures.
        """

        mask = ~torch.isnan(y_true)
        y_true = torch.nan_to_num(y_true, nan=0.0)  # potentially remove

        loss = self.loss_fn(alpha=alpha, beta=beta, y_true=y_true)  # shape: (batch_size, 2)

        m_wgts = y_wgts * mask
        mw_loss = loss * m_wgts  # masked weighted total loss

        total_loss = mw_loss.sum()
        total_loss_wgt = m_wgts.sum()  # needed later for normalization

        per_sign_loss = defaultdict(lambda: torch.zeros(2, device=self.device))

        for sign_type, sign_values in y_sign.items():  # e.g. Age: [List]
            for s_idx in range(y_true.shape[0]):
                sign_name = sign_values[s_idx]
                key = (sign_type, sign_name)
                per_sign_loss[key][0] += mw_loss[s_idx]
                per_sign_loss[key][1] += m_wgts[s_idx]

        batch_loss = {
            'Total': (total_loss, total_loss_wgt),
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

        per_sign_loss = defaultdict(lambda: torch.zeros(2, device=self.device))

        def combine(sign_loss):
            for key, sub_loss in sign_loss.items():
                per_sign_loss[key] += sub_loss

        combine(loss_1['Sign'])
        combine(loss_2['Sign'])

        return {
            'Total': (total_loss, total_wgt),
            'Sign': per_sign_loss
        }

    def normalize_mws_loss(self, mws_loss):

        per_sample_loss = mws_loss['Total'][0] / mws_loss['Total'][1]
        per_sample_loss = np.round(per_sample_loss.detach().cpu().numpy(), 5)

        per_sign_loss = defaultdict(lambda: torch.zeros(2, device=self.device))

        for key, sub_loss in mws_loss['Sign'].items():
            sub_array = sub_loss[0] / sub_loss[1]
            per_sign_loss[key] = np.round(sub_array.detach().cpu().numpy(), 5)

        return {
            'Total': per_sample_loss,
            'Sign': per_sign_loss
        }

    def mws_metrics(self, y_true, mu, y_wgts, y_sign: dict):
        """
        Aggregate predictions based on different criteria. Intended to be used
        with classical ML metrics, purely on predictions
        """
        y_true = y_true.detach().cpu().numpy()
        mu = mu.detach().cpu().numpy()
        y_wgts = y_wgts.detach().cpu().numpy()

        # Calculate Overall metrics
        total_metrics = self.metrics_function(y_true=y_true, alpha=mu, y_wgts=y_wgts)

        # Calculate per-sign metrics
        per_sign_metrics = defaultdict(dict)  # i.e. sign_type: task: sign_name: metrics | horrible
        for sign_type, sign_values in y_sign.items():
            sign_values = np.array(sign_values)
            unique_signs = set(sign_values)
            for sign_name in unique_signs:
                key = (sign_type, sign_name)
                sign_idx = np.where(sign_values == sign_name)[0]
                sign_values = np.array(sign_values)
                per_sign_metrics[key] = self.metrics_function(y_true=y_true[sign_idx], mu=mu[sign_idx],
                                                              y_wgts=y_wgts[sign_idx]) if len(sign_idx) > 0 else {}
        return {
            'Total': total_metrics,
            'Sign': per_sign_metrics
        }

    @staticmethod
    def score_classification(y_true: np.ndarray, mu: np.ndarray, y_wgts: np.ndarray, threshold: float = 0.5):
        """
        To be used as an inner function with 1D numpy arrays only.
        """

        def safe_div(numerator, denominator, default=0.0):
            return numerator / denominator if denominator != 0 else default

        alpha_bin = (mu >= threshold).astype(int)

        mask = ~np.isnan(y_true)
        y_true = y_true[mask].astype(int)
        mu = mu[mask]
        alpha_bin = alpha_bin[mask]
        y_wgts = y_wgts[mask]

        conf_mat = confusion_matrix(y_true=y_true, y_pred=alpha_bin, sample_weight=y_wgts)
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
            'ROC AUC': np.round(roc_auc_score(y_true=y_true, y_score=mu, sample_weight=y_wgts), 5),
            'MCC': np.round(safe_div((tp * tn) - (fp * fn), np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))), 5)
        }

        return metrics

    @staticmethod
    def score_regression(y_true: np.ndarray, mu: np.ndarray, y_wgts: np.ndarray):

        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        mu = mu[mask]
        y_wgts = y_wgts[mask]

        metrics = {
            'R2': np.round(r2_score(y_true=y_true, y_pred=mu, sample_weight=y_wgts), 5),
            'MAE': np.round(mean_absolute_error(y_true=y_true, y_pred=mu, sample_weight=y_wgts), 5),
            'RMSE': np.round(mean_squared_error(y_true=y_true, y_pred=mu, squared=False, sample_weight=y_wgts), 5)
        }

        return metrics

    @staticmethod
    def pack_epoch_data(epoch_data):
        packed_data = {
            'mu': torch.cat(epoch_data['mu'], dim=0),
            'y_true': torch.cat(epoch_data['y_true'], dim=0),
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

    def build_graph_layers(self):
        layer_class = self.gnn_params['layer']
        layer_type = self.gnn_params.get('layer_type')  # default
        sizes = self.gnn_params['sizes']
        input_dim = self.gnn_params['input_dim']
        layer_args = self.gnn_params.get('args', [{} for _ in sizes])
        activation = self.gnn_params.get('activation', 'relu')
        dropout = self.gnn_params.get('dropout', 0.0)

        gnn_blocks = []
        in_dim = input_dim

        for i, (out_dim, args) in enumerate(zip(sizes, layer_args)):

            if layer_type == 'attention':
                heads = self.gnn_params.get('heads', 1)
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

            batch_norm_layer = nn.BatchNorm1d(next_in) if self.batch_norm else None
            activation_layer = self.get_activation_fn(activation) if activation else None
            dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

            gnn_blocks.append(
                GNNLayerBlock(
                    graph_layer=graph_layer,
                    batch_norm=batch_norm_layer,
                    activation=activation_layer,
                    dropout=dropout_layer
                )
            )
            in_dim = next_in

        return nn.ModuleList(gnn_blocks), in_dim

    def build_conv_layers(self):
        alphabet_len = self.cnn_params['alphabet_len']
        embedding_dim = self.cnn_params['embedding_dim']  # i.e. in_channels for the first layer
        padding_idx = self.cnn_params.get('padding_idx', 0)

        cnn_embedding = nn.Embedding(
            num_embeddings=alphabet_len,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        sizes = self.cnn_params.get('sizes', [256])
        kernel_size = self.cnn_params.get('kernel_size', 5)
        stride = self.cnn_params.get('stride', 1)
        dropout = self.cnn_params.get('dropout', 0.1)
        activation_fn = self.get_activation_fn(self.cnn_params.get('activation', 'relu'))
        pool_kernel_size = self.cnn_params.get('pool_size', 2)

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
            batch_norm = nn.BatchNorm1d(out_channels) if self.batch_norm else None
            activation = activation_fn if activation_fn else None
            max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size) if pool_kernel_size > 1 else None
            dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

            cnn_blocks.append(
                CNNLayerBlock(
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
        rnn_block = RNNLayerBlock(
            recurrent_layer=rnn_layer,
            max_len=max_len)
        output_dim = hidden_size

        return rnn_block, rnn_embedding, output_dim

    def build_linear_layers(self, sizes: List[int], batch_norm: bool = True,
                            activation: str = 'relu', dropout: float = 0.0):

        linear_layers = []

        linear_sizes = sizes.copy()
        lin_out_size = linear_sizes[-1]
        in_features = linear_sizes.pop(0)

        for out_features in linear_sizes:
            linear = torch.nn.Linear(in_features, out_features)
            self.init_linear(linear)
            linear_layers.append(linear)

            if batch_norm:
                linear_layers.append(torch.nn.BatchNorm1d(out_features))
            if activation is not None:
                linear_layers.append(self.get_activation_fn(activation))
            if dropout > 0:
                linear_layers.append(torch.nn.Dropout(p=dropout))

            in_features = out_features

        return torch.nn.Sequential(*linear_layers), lin_out_size

    def build_attention_layers(self, input_size, attn_size, num_heads, dropout):

        projection = nn.Linear(in_features=input_size, out_features=attn_size)
        self.init_linear(projection)
        attn_layer = nn.MultiheadAttention(embed_dim=attn_size, num_heads=num_heads,
                                           batch_first=True, dropout=dropout)

        return projection, attn_layer

    @staticmethod
    def init_linear(layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)
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
            raise ValueError(f"Unsupported activation function: {name}")

    def set_optimizer(self, optimizer=torch.optim.AdamW, params: dict = None):
        if callable(optimizer):
            self.optimizer = optimizer(self.parameters(), **params) if params else optimizer(self.parameters())
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

    def load(self, path, map_location:str = None):
        ext = path.split('.')[-1]
        if ext not in ['pt', 'pth']:
            raise ValueError(f'Unsupported file extension: {ext}')
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @staticmethod
    def BetaNLLLoss(alpha: Tensor, beta: Tensor, y_true: Tensor, reduction: str = 'none') -> Tensor:
        """
        TODO: correct shapes
        """
        eps = 1e-6
        y_true = y_true.clamp(min=eps, max=1 - eps)
        beta_loss = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        target_loss = (alpha - 1) * torch.log(y_true) + (beta - 1) * torch.log(1 - y_true)
        loss = -(beta_loss + target_loss)
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum(dim=1)
        elif reduction == 'mean':
            return loss.mean(dim=1)
        else:
            raise ValueError('Unrecognized reduction')

    @staticmethod
    def set_seed(seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)