# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains the helper functions for the model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from omnisafe.typing import Activation, InitFunction


def initialize_layer(init_function: InitFunction, layer: nn.Linear) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif init_function in ['glorot', 'xavier_uniform']:
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')


def get_activation(
    activation: Activation,
) -> type[nn.Identity | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Tanh]:
    """Get the activation function.

    The ``activation`` can be chosen from: ``identity``, ``relu``, ``sigmoid``, ``softplus``,
    ``tanh``.

    Args:
        activation (Activation): The activation function.

    Returns:
        The activation function, ranging from ``nn.Identity``, ``nn.ReLU``, ``nn.Sigmoid``,
        ``nn.Softplus`` to ``nn.Tanh``.
    """
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
    }
    assert activation in activations
    return activations[activation]


def build_mlp_network(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
) -> nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn if j < len(sizes) - 2 else output_activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act_fn()]
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    def __init__(self, input_channels, conv_filters, activation):
        super(ConvEncoder, self).__init__()
        layers = []
        for out_channels, kernel_size, stride in conv_filters:
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride))
            layers.append(get_activation(activation)())
            input_channels = out_channels
        self.conv = nn.Sequential(*layers)

    def output_size(self, input_shape):
        with torch.no_grad():
            input_tensor = torch.zeros(*input_shape)
            output = self.conv(input_tensor)
            return output.view(1, -1).size(1)

    def forward(self, x):
        return self.conv(x)


# Define the combined CNN+MLP Network
class CNN_MLP_Network(nn.Module):
    def __init__(self, cnn_cfg, mlp_sizes, mlp_activation, mlp_weight_init: InitFunction):
        super(CNN_MLP_Network, self).__init__()
        self.cnn = ConvEncoder(cnn_cfg['input_channels'], cnn_cfg['conv_filters'], cnn_cfg['activation'])
        cnn_output_size = self.cnn.output_size(cnn_cfg['input_shape'])
        mlp_input_size = cnn_output_size
        sizes = [mlp_input_size] + mlp_sizes
        self.mlp = build_mlp_network(sizes, mlp_activation, weight_initialization_mode=mlp_weight_init)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

# def calc_num_action_parameters(action_space: ActionSpace) -> int:
#     """Returns the number of paramaters required to represent the given action space."""
#     if isinstance(action_space, gym.spaces.Discrete):
#         return action_space.n
#     elif isinstance(action_space, gym.spaces.Tuple):
#         return sum([calc_num_action_parameters(a) for a in action_space])
#     elif isinstance(action_space, gym.spaces.Box):
#         # one mean and one standard deviation for every action
#         return np.prod(action_space.shape) * 2
#     else:
#         raise NotImplementedError(f"Action space type {type(action_space)} not supported!")

# num_action_outputs = calc_num_action_parameters(action_space)
# self.distribution_linear = nn.Linear(core_out_size, num_action_outputs)

# def sample_actions_log_probs(self):
#     list_of_action_batches = [d.sample() for d in self.distributions]
#     batch_of_action_tuples = self._flatten_actions(list_of_action_batches)
#     log_probs = self._calc_log_probs(list_of_action_batches)
#     return batch_of_action_tuples, log_probs
