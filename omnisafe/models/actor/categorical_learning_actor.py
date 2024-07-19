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
"""Implementation of GaussianLearningActor."""

from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical

from omnisafe.models.base import Actor
from omnisafe.typing import OmnisafeSpace, Activation, InitFunction
from omnisafe.utils.model import CNN_MLP_Network


# pylint: disable-next=too-many-instance-attributes
class MultiDiscreteLearningActor(Actor):
    """Implementation of MultiDiscreteLearningActor.

    MultiDiscreteLearningActor is a categorical actor for multi-discrete action spaces. It is used in
    on-policy algorithms such as ``PPO``, ``TRPO`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        cnn_cfg (dict): Configuration for the CNN part.
        hidden_sizes (list of int): List of hidden layer sizes for the MLP part.
        activation (str, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (str, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        cnn_cfg: dict,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`MultiDiscreteLearningActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self._hidden_sizes = hidden_sizes
        self._total_act_dim = sum(self._act_dims)  # Total number of logits needed
        self.cnn_mlp_network = CNN_MLP_Network(
            cnn_cfg,
            hidden_sizes + [self._total_act_dim],
            activation,
            weight_initialization_mode,
        )

    def _distribution(self, obs: torch.Tensor) -> List[Categorical]:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            A list of categorical distributions for each action dimension.
        """
        logits = self.cnn_mlp_network(obs)
        split_logits = torch.split(logits, self._act_dims, dim=-1)
        return [Categorical(logits=logit) for logit in split_logits]

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mode of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            A tensor of predicted actions.
        """
        self._current_dists = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return torch.stack([dist.probs.argmax(dim=-1) for dist in self._current_dists], dim=-1)
        return torch.stack([dist.sample() for dist in self._current_dists], dim=-1)

    def forward(self, obs: torch.Tensor) -> List[Categorical]:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            A list of categorical distributions for each action dimension.
        """
        self._current_dists = self._distribution(obs)
        self._after_inference = True
        return self._current_dists

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward`.

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return torch.stack([dist.log_prob(a) for dist, a in zip(self._current_dists, act.T)], dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Compute the entropy of the current distribution.

        Returns:
            The entropy of the current distribution.
        """
        assert self._after_inference, 'entropy() should be called after predict() or forward()'
        return torch.stack([dist.entropy() for dist in self._current_dists], dim=-1).sum(dim=-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Generate actions and corresponding log probabilities.

        Args:
            obs (torch.Tensor): Observations from the environment.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            A dictionary containing the actions, log probabilities, and the entropy of the actions.
        """
        actions = self.predict(obs, deterministic)
        log_probs = self.log_prob(actions)
        entropies = self.entropy()
        return {
            'actions': actions,
            'log_probs': log_probs,
            'entropies': entropies
        }
