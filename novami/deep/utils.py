from torch import nn


def get_activation_fn(name: str) -> nn.Module:
    """
    Return a built-in activation module by name (no learnable parameters).

    Parameters
    ----------
    name : str
        One of relu, leaky_relu, gelu, tanh. Matching is case-insensitive.

    Returns
    -------
    act : nn.Module
        A fresh activation instance (e.g. nn.ReLU()).
    """
    name = name.lower()
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
