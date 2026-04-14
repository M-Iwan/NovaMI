from torch import nn


def get_activation_fn(name: str) -> nn.Module:
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
