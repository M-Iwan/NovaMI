from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GATv2Conv, EdgeConv


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

    dropout = float(input("Dropout rate [0.0]: ") or 0.0)

    gnn_params = {
        'layer': layer_class,
        'layer_type': layer_type,
        'sizes': sizes,
        'input_dim': input_dim,
        'args': layer_args,
        'activation': activation,
        'dropout': dropout
    }

    if layer_type == 'attention':
        gnn_params['heads'] = heads

    return gnn_params


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

    cnn_params = {
        'alphabet_len': alphabet_len,
        'embedding_dim': embedding_dim,
        'padding_idx': padding_idx,
        'sizes': sizes,
        'kernel_size': kernel_size,
        'stride': stride,
        'pool_size': pool_size,
        'activation': activation,
        'dropout': dropout
    }

    print("\nCNN layer config complete.")
    return cnn_params


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
        'max_len': max_len
    }

    print("\nRNN layer config complete.")
    return rnn_params


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
        batch_norm_input = input("Use batch normalization? (y/n) [y]: ").strip().lower() or 'y'
        if batch_norm_input in ('y', 'n'):
            batch_norm = batch_norm_input == 'y'
            break
        print("Enter 'y' or 'n'.")

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


def build_des_config():
    print("=== Descriptors Layer Configuration ===")
    configs = {}

    print("Enter new descriptor names for linear layer configs.")

    while True:
        key = input("Enter descriptor name: ").strip()
        if not key:
            break

        print(f"\nConfiguring linear layers for '{key}':")
        config = build_lin_config()
        configs[key] = config
        print(f"Config for '{key}' saved.\n")

    print("All linear layer configs collected.")
    return configs


def build_att_config():
    print("=== Attention Layer Configuration ===")

    # Attention size
    while True:
        try:
            attn_size = int(input("Enter attention output size (attn_size): "))
            if attn_size > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    # Number of heads
    while True:
        try:
            num_heads = int(input("Enter number of attention heads: "))
            if num_heads > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    # Dropout rate
    while True:
        try:
            dropout = input("Enter dropout rate [0.0]: ").strip()
            dropout = float(dropout) if dropout else 0.0
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    print("\nAttention layer config complete.")
    return {
        'attn_size': attn_size,
        'num_heads': num_heads,
        'dropout': dropout
    }


def config_model():
    print("=== Model General Configuration ===")
    device = input("Device cpu/cuda [cpu]: ").strip() or 'cpu'
    task = input("Task (classification/regression) [classification]: ").strip() or 'classification'

    while True:
        try:
            num_task = int(input("Number of tasks [1]: ").strip() or '1')
            if num_task > 0:
                break
            else:
                print("Must be positive integer.")
        except ValueError:
            print("Invalid integer.")

    label_name = input("Label name [None]: ").strip() or None
    weight_name = input("Weight name [None]: ").strip() or None
    signature_name = input("Signature name [None]: ").strip() or None

    while True:
        try:
            max_norm = float(input("Max norm [1.0]: ").strip() or '1.0')
            if max_norm > 0:
                break
            else:
                print("Must be positive number.")
        except ValueError:
            print("Invalid float.")

    query_desc = input("Query description [Demo]: ").strip() or 'Demo'

    gnn_params = build_gnn_config() if input("Add GNN layers? y/n") == "y" else None
    cnn_params = build_cnn_config() if input("Add CNN layers? y/n") == "y" else None
    rnn_params = build_rnn_config() if input("Add RNN layers? y/n") == "y" else None
    des_params = build_des_config() if input("Add Desc layers? y/n") == "y" else None

    att_params = build_att_config()
    lin_params = build_lin_config()

    final_config = {
        'device': device,
        'task': task,
        'num_task': num_task,
        'label_name': label_name,
        'weight_name': weight_name,
        'signature_name': signature_name,
        'gnn_params': gnn_params,
        'cnn_params': cnn_params,
        'rnn_params': rnn_params,
        'des_params': des_params,
        'att_params': att_params,
        'lin_params': lin_params,
        'max_norm': max_norm,
        'query_desc': query_desc
    }

    print("\nFinal configuration ready.")
    return final_config