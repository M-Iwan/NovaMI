from torch import nn


def get_activation_fn(name: str) -> nn.Module:
    """
    Return a non-parametric activation module by name.

    Parameters
    ----------
    name : str
        One of ``'relu'``, ``'leaky_relu'``, ``'gelu'``, ``'tanh'`` (case-insensitive).

    Returns
    -------
    torch.nn.Module
        Instantiated activation.

    Raises
    ------
    ValueError
        If ``name`` is not supported.
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
