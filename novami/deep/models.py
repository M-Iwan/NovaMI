import os
import inspect
from copy import deepcopy
from functools import reduce
from collections import defaultdict
from typing import Any, Dict, List, Iterable, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

import torch
from torch import nn, Tensor

from novami.deep.modules import build_linear_layers, build_graph_layers, build_conv_layers, build_recurrent_layers
from torch_geometric.nn import global_mean_pool


class MMTUnit(ABC, nn.Module):
    """
    Abstract base class for deep learning models.

    Parameters
    ----------
    device : str, optional
        Device used for calculations. Default is cpu.
    num_task : int, optional
        Number of tasks (output width for regression). Default is 1.
    model_name : str, optional
        Used in default checkpoint filenames; also stored as name and model_name.
        Default is Unit.
    """
    def __init__(self, device: str = 'cpu', num_task: int = 1, model_name: str = 'Unit'):
        super(MMTUnit, self).__init__()
        self.device = device
        self.num_task = num_task
        self.name = model_name
        self.model_name = model_name  # alias for checkpoints / save paths

        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.logs = defaultdict(list)
        self.epoch = 1
        self.best_state_dict = None
        self.best_epoch = None

    @abstractmethod
    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch : dict
            Mapping of batch keys to tensors (dict-like MMBatch is fine).
        """

    @staticmethod
    @abstractmethod
    def score(y_true: np.ndarray, y_pred: np.ndarray, y_wgts: np.ndarray):
        """
        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        y_wgts : np.ndarray
        """

    def grad_norm(self):
        """L2 norm of stacked parameter gradients (for logging)."""
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def fit_epoch(self, dataloader):
        self.train()

        epoch_losses = []
        epoch_data = {
            'y_pred': [],
            'y_true': [],
            'y_wgts': [],
        }

        epoch_grad_norms = []

        for batch in dataloader:
            self.optimizer.zero_grad()

            y_pred = self(batch)
            y_true = batch['y_true'].to(self.device)
            y_wgts = batch['y_wgts'].to(self.device)

            epoch_data['y_pred'].append(y_pred)
            epoch_data['y_true'].append(y_true)
            epoch_data['y_wgts'].append(y_wgts)

            group = batch.get('group') if hasattr(batch, 'get') else None
            if group is None and isinstance(batch, dict):
                group = batch.get('group')

            batch_loss = self.loss(
                y_pred=y_pred,
                y_true=y_true,
                y_wgts=y_wgts,
                group=group,
            )

            loss, loss_wgt = batch_loss.get('Total')
            loss.backward()

            batch_grad_norm = self.grad_norm() / (loss_wgt + 1e-6)
            epoch_grad_norms.append(batch_grad_norm)

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            epoch_losses.append(batch_loss)
        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = reduce(self.combine_loss, epoch_losses)
        epoch_loss = self.normalize_loss(epoch_loss)

        epoch_metrics = self.mw_metrics(**self.pack_epoch_data(epoch_data))

        return epoch_loss, epoch_metrics, epoch_grad_norms

    def eval_epoch(self, dataloader):
        self.eval()

        epoch_losses = []
        epoch_data = {
            'y_pred': [],
            'y_true': [],
            'y_wgts': [],
        }

        with torch.no_grad():
            for batch in dataloader:

                y_pred = self(batch)
                y_true = batch['y_true'].to(self.device)
                y_wgts = batch['y_wgts'].to(self.device)

                epoch_data['y_pred'].append(y_pred)
                epoch_data['y_true'].append(y_true)
                epoch_data['y_wgts'].append(y_wgts)

                group = batch.get('group') if hasattr(batch, 'get') else None
                if group is None and isinstance(batch, dict):
                    group = batch.get('group')

                batch_loss = self.loss(
                    y_pred=y_pred,
                    y_true=y_true,
                    y_wgts=y_wgts,
                    group=group,
                )

                epoch_losses.append(batch_loss)

        epoch_loss = reduce(self.combine_loss, epoch_losses)
        epoch_loss = self.normalize_loss(epoch_loss)

        epoch_metrics = self.mw_metrics(**self.pack_epoch_data(epoch_data))

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
                        save_path = os.path.join(save_dir, f'{self.model_name}_min.pth')
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    self.get_best_model()
                    self.epoch = deepcopy(self.best_epoch)
                    break

            if (save_freq is not None) and (save_dir is not None) and (epoch % save_freq == 0):
                save_path = os.path.join(save_dir, f'{self.model_name}_{epoch}.pth')
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

            current_loss = eval_loss.get('Total')

            if verbose > 0:
                print(f"Epoch {epoch} train loss: {train_loss['Total']:.5f}")
                print(f"Epoch {epoch} eval loss: {eval_loss['Total']:.5f}")

            if early_stop is not None:
                if current_loss < min_loss:
                    min_loss = current_loss
                    patience = deepcopy(early_stop)
                    self.set_best_model()
                    if save_dir is not None:
                        save_path = os.path.join(save_dir, f'{self.model_name}_min.pth')
                        self.save(save_path)
                else:
                    patience -= 1

                if patience == 0:
                    print(f'Early stopping at epoch {epoch} with minimum loss {min_loss:.5f}')
                    self.get_best_model()
                    self.epoch = deepcopy(self.best_epoch)
                    break

            if (save_freq is not None) and (save_dir is not None) and (epoch % save_freq == 0):
                save_path = os.path.join(save_dir, f'{self.model_name}_{epoch}.pth')
                self.save(save_path)

    def predict(self, dataloader):
        self.eval()

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                predictions.append(self(batch))

        return torch.cat(predictions, dim=0).detach().cpu().numpy()

    def loss(self, y_pred, y_true, y_wgts, group: Optional[List[Any]] = None):
        """
        Masked, weighted loss.

        Parameters
        ----------
        y_pred : torch.Tensor
        y_true : torch.Tensor
        y_wgts : torch.Tensor
        group : list, optional
            Length B; each element iterable of tags for that row.

        Returns
        -------
        batch_loss : dict
            Keys Total and Task map to (scalar loss sum, scalar weight sum).
            Optionally Group maps each tag string to the same tuple shape.
        """

        mask = ~torch.isnan(y_true)
        y_true = torch.nan_to_num(y_true, nan=0.0)

        loss = self.loss_fn(y_pred, y_true)

        m_wgts = y_wgts * mask
        mw_loss = loss * m_wgts

        total_loss = mw_loss.sum()
        total_loss_wgt = m_wgts.sum()

        per_task_loss = mw_loss.sum(dim=0)
        per_task_loss_wgt = (y_wgts * mask).sum(dim=0)

        batch_loss: Dict[str, Any] = {
            'Total': (total_loss, total_loss_wgt),
            'Task': (per_task_loss, per_task_loss_wgt),
        }

        if group is not None and len(group) == y_pred.shape[0]:
            unique_labels = set()
            for row in group:
                arr = np.asarray(row, dtype=object).ravel()
                for x in arr:
                    if x is None:
                        continue
                    s = str(x).strip()
                    if not s or s.lower() == 'nan':
                        continue
                    unique_labels.add(s)

            if unique_labels:
                device = y_pred.device
                dtype = mw_loss.dtype
                b = y_pred.shape[0]
                row_axes = (1,) * max(0, mw_loss.ndim - 1)
                group_losses: Dict[str, tuple] = {}
                for label in unique_labels:
                    row_mask = np.zeros(b, dtype=np.float64)
                    for i, row_g in enumerate(group):
                        arr = np.asarray(row_g, dtype=object).ravel()
                        tags = {
                            str(x).strip()
                            for x in arr
                            if x is not None and str(x).strip() and str(x).strip().lower() != 'nan'
                        }
                        if label in tags:
                            row_mask[i] = 1.0
                    rm = torch.as_tensor(row_mask, device=device, dtype=dtype).view(b, *row_axes)
                    partial = mw_loss * rm
                    gl = partial.sum()
                    gw = (m_wgts * rm).sum()
                    if float(gw.detach().cpu().item()) > 0.0:
                        group_losses[label] = (gl, gw)
                if group_losses:
                    batch_loss['Group'] = group_losses

        return batch_loss

    @staticmethod
    def combine_loss(loss_1, loss_2):
        total_loss = loss_1['Total'][0] + loss_2['Total'][0]
        total_wgt = loss_1['Total'][1] + loss_2['Total'][1]

        per_task_loss = loss_1['Task'][0] + loss_2['Task'][0]
        per_task_wgt = loss_1['Task'][1] + loss_2['Task'][1]

        out: Dict[str, Any] = {
            'Total': (total_loss, total_wgt),
            'Task': (per_task_loss, per_task_wgt),
        }

        g1: Dict[str, tuple] = loss_1.get('Group') or {}
        g2: Dict[str, tuple] = loss_2.get('Group') or {}
        if g1 or g2:
            dev = total_loss.device
            dt = total_loss.dtype
            merged: Dict[str, tuple] = {}
            for key in set(g1) | set(g2):
                l_sum = torch.zeros((), device=dev, dtype=dt)
                w_sum = torch.zeros((), device=dev, dtype=dt)
                if key in g1:
                    l_sum = l_sum + g1[key][0]
                    w_sum = w_sum + g1[key][1]
                if key in g2:
                    l_sum = l_sum + g2[key][0]
                    w_sum = w_sum + g2[key][1]
                merged[key] = (l_sum, w_sum)
            out['Group'] = merged

        return out

    @staticmethod
    def normalize_loss(mw_loss):
        per_sample_loss = mw_loss['Total'][0] / mw_loss['Total'][1]
        per_sample_loss = np.round(per_sample_loss.detach().cpu().numpy(), 5)
        per_task_loss = mw_loss['Task'][0] / mw_loss['Task'][1]
        per_task_loss = np.round(per_task_loss.detach().cpu().numpy(), 5)

        out: Dict[str, Any] = {
            'Total': per_sample_loss,
            'Task': per_task_loss,
        }

        grp = mw_loss.get('Group')
        if grp:
            out['Group'] = {}
            for label, (lv, wv) in grp.items():
                w = float(wv.detach().cpu().item())
                if w <= 0.0:
                    out['Group'][label] = np.float64(0.0)
                else:
                    out['Group'][label] = np.round((lv / wv).detach().cpu().numpy(), 5)

        return out

    def mw_metrics(self, y_true, y_pred, y_wgts):
        """
        Concatenate epoch tensors and call score() for overall and per-task views.

        Parameters
        ----------
        y_true, y_pred, y_wgts : torch.Tensor
            Stacked batch outputs (same leading dimension).

        Returns
        -------
        out : dict
            Total: dict from score on flattened arrays; Task: dict keyed Task_0, ...
        """
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_wgts = y_wgts.detach().cpu().numpy()

        # Calculate Overall metrics
        total_metrics = self.score(y_true=y_true.flatten(),
                                   y_pred=y_pred.flatten(),
                                   y_wgts=y_wgts.flatten())

        # Calculate per-task metrics
        per_task_metrics = defaultdict(dict)
        for t_idx in range(self.num_task):
            per_task_metrics[f"Task_{t_idx}"] = self.score(y_true=y_true[:, t_idx].flatten(),
                                                           y_pred=y_pred[:, t_idx].flatten(),
                                                           y_wgts=y_wgts[:, t_idx].flatten())
        return {
            'Total': total_metrics,
            'Task': per_task_metrics,
        }

    @staticmethod
    def pack_epoch_data(epoch_data):
        packed_data = {
            'y_true': torch.cat(epoch_data['y_true'], dim=0),
            'y_pred': torch.cat(epoch_data['y_pred'], dim=0),
            'y_wgts': torch.cat(epoch_data['y_wgts'], dim=0),
        }
        return packed_data

    def set_loss_function(self, loss_fn=torch.nn.BCEWithLogitsLoss, loss_params: dict = None):
        """
        Instantiate the per-element loss used in mw_loss.

        Parameters
        ----------
        loss_fn : callable
            Class or factory returning a torch loss module.
        loss_params : dict, optional
            Keyword arguments for the loss constructor.

        Notes
        -----
        The loss must use reduction='none' so mw_loss can weight per element.
        """
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


class TestModel(MMTUnit):
    """
    Example MMTUnit: PyG GNN stack, global mean pool, then a linear MLP head.

    Parameters
    ----------
    device : str, optional
        Passed to MMTUnit. Default is cpu.
    num_task : int, optional
        Number of regression outputs. Default is 1.
    model_name : str, optional
        Passed to MMTUnit. Default is Unit.
    gnn_params : dict, optional
        Dict for build_graph_layers: layer, layer_type (convolutional, attention,
        edge), sizes, input_dim, optional input_name (batch key for the graph;
        default Graph).
    lin_params : dict, optional
        Keyword args for build_linear_layers; the first segment of sizes is
        filled from the GNN output width and the last from num_task.
    max_norm : float, optional
        Gradient clipping max norm in fit_epoch. Default is 1.0.
    """
    def __init__(self, device: str = 'cpu', num_task: int = 1, model_name: str = 'Unit',
                 gnn_params: dict=None, lin_params: dict=None, max_norm: float = 1.0):
        super(TestModel, self).__init__(device=device, num_task=num_task, model_name=model_name)

        self.gnn_params = deepcopy(gnn_params)
        self.lin_params = deepcopy(lin_params)
        self.max_norm = max_norm

        self.gnn_input_name = self.gnn_params.get('input_name', "Graph")
        self.gnn_layer, self.gnn_out_size = build_graph_layers(self.gnn_params)
        self.lin_params['sizes'] = [self.gnn_out_size] + self.lin_params.get('sizes') + [self.num_task]
        self.lin_layers, _ = build_linear_layers(**self.lin_params)

    def forward(self, batch_input: dict):
        """
        Message passing on the graph batch, then global mean pool and lin_layers.

        Parameters
        ----------
        batch_input : dict
            Must contain the batched graph under gnn_params['input_name']
            (default key Graph).

        Returns
        -------
        x_out : torch.Tensor
            Shape (batch, num_task).
        """
        graph_batch = batch_input[self.gnn_input_name]

        for layer in self.gnn_layer:
            x = layer(graph_batch)
            graph_batch.x = x

        x_out = global_mean_pool(graph_batch.x, graph_batch.batch)
        x_out = self.lin_layers(x_out)

        return x_out

    @staticmethod
    def score(y_true: np.ndarray, y_pred: np.ndarray, y_wgts: np.ndarray):
        """
        R2, MAE, RMSE from sklearn after dropping NaN y_true and applying weights.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        y_wgts : np.ndarray

        Returns
        -------
        metrics : dict
            Keys R2, MAE, RMSE with values rounded to 5 decimals.
        """
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        y_wgts = y_wgts[mask]

        metrics = {
            'R2': np.round(r2_score(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5),
            'MAE': np.round(mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5),
            'RMSE': np.round(root_mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5)
        }

        return metrics
