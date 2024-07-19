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
"""Implementation of the PPO algorithm."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient


@registry.register
class PPO(PolicyGradient):
    """The Proximal Policy Optimization (PPO) algorithm.

    References:
        - Title: Proximal Policy Optimization Algorithms
        - Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
        - URL: `PPO <https://arxiv.org/abs/1707.06347>`_
    """

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Proximal Policy Optimization, the loss is defined as:

        .. math::

            L^{CLIP} = \underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                \min ( r_t A^{R}_{\pi_{\theta}} (s_t, a_t) , \text{clip} (r_t, 1 - \epsilon, 1 + \epsilon)
                A^{R}_{\pi_{\theta}} (s_t, a_t)
            \right]

        where :math:`r_t = \frac{\pi_{\theta}^{'} (a_t|s_t)}{\pi_{\theta} (a_t|s_t)}`,
        :math:`\epsilon` is the clip parameter, and :math:`A^{R}_{\pi_{\theta}} (s_t, a_t)` is the
        advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        # Get the distribution of actions for the given observations
        distribution = self._actor_critic.actor(obs)

        # Compute the log probabilities of the actions under the current policy
        if isinstance(distribution, torch.distributions.Categorical):
            logp_ = distribution.log_prob(act)
        else:
            # If the distribution is a list of Categorical distributions, handle each separately
            logp_ = torch.stack([dist.log_prob(a) for dist, a in zip(distribution, act.T)], dim=-1).sum(dim=-1)

        # Compute the ratio of the new and old action probabilities
        ratio = torch.exp(logp_ - logp)

        # Clip the ratio to avoid large policy updates
        ratio_clipped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )

        # Calculate the PPO loss
        loss = -torch.min(ratio * adv, ratio_clipped * adv).mean()

        # Subtract the entropy bonus from the loss to encourage exploration
        if isinstance(distribution, torch.distributions.Categorical):
            entropy = distribution.entropy().mean()
        else:
            entropy = torch.stack([dist.entropy().mean() for dist in distribution]).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * entropy

        # useful extra info for logging
        self._logger.store(
            {
                'Train/Entropy': entropy.item(),
                'Train/PolicyRatio': ratio.mean().item(),
                'Loss/Loss_pi': loss.item(),
            },
        )
        return loss
