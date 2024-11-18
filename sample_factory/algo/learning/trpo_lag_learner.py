from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.trpo_learner import TRPOLearner
from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import log


class TRPOLagLearner(TRPOLearner):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.lambda_lagr = None
        self.safety_bound = self.cfg.safety_bound

    def init(self) -> InitModelData:
        init_res = super().init()
        self.lambda_lagr = self.cfg.lambda_lagr
        return init_res

    def _calculate_losses(
            self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor, Tensor, Dict]:
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            valids = mb.valids

        # Calculate policy head outputs
        with self.timing.add_time("forward_head"):
            head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = head_outputs.size(0)

        # Build RNN inputs
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()
                head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                    head_outputs,
                    done_or_invalid,
                    mb.rnn_states,
                    recurrence,
                )
            else:
                rnn_states = mb.rnn_states[::recurrence]

        # Forward pass through RNN
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    core_output_seq, _ = self.actor_critic.forward_core(head_output_seq, rnn_states)
                core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                del core_output_seq
            else:
                core_outputs, _ = self.actor_critic.forward_core(head_outputs, rnn_states)
            del head_outputs

        num_trajectories = minibatch_size // recurrence
        assert core_outputs.shape[0] == minibatch_size

        # Calculate action distributions and values
        with self.timing.add_time("tail"):
            result = self.actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)

            values = result["values"].squeeze()
            cost_values = result["cost_values"].squeeze()
            del core_outputs

        # Update Lagrange multiplier
        mean_cost = mb["costs"].mean()
        with self.timing.add_time("lagrange_update"):
            cost_violation = self._update_lagrange(mean_cost)

        # Compute advantages
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            adv = mb.advantages
            targets = mb.returns
            cost_adv = mb.cost_advantages
            cost_targets = mb.cost_returns

            # Compute surrogate advantage
            adv = self._compute_adv_surrogate(adv, cost_adv)
            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # Normalize advantage

        # Compute losses
        with self.timing.add_time("losses"):
            # Surrogate loss for TRPO
            surrogate_loss = -torch.mean(masked_select(log_prob_actions * adv, valids, num_invalids))

            # Compute KL divergence between old and new policies
            old_action_distribution = get_action_distribution(
                self.actor_critic.action_space,
                mb.action_logits,
            )
            kl_div = old_action_distribution.kl_divergence(action_distribution)
            kl_div = masked_select(kl_div, valids, num_invalids).mean()

            # Value function losses
            value_loss = self._value_loss(values, mb["values"], targets, self.cfg.ppo_clip_value, valids, num_invalids)
            cost_value_loss = self._value_loss(
                cost_values, mb["cost_values"], cost_targets, self.cfg.ppo_clip_value, valids, num_invalids
            )

        loss_summaries = dict(
            values=result["values"],
            cost_values=result["cost_values"],
            avg_cost=mean_cost,
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            cost_violation=cost_violation,
            lagrange_multiplier=self._get_lagrange_multiplier(),
            kl_divergence=kl_div,
        )

        return action_distribution, surrogate_loss, value_loss, cost_value_loss, kl_div, loss_summaries

    def _update_lagrange(self, mean_cost):
        # Update lambda_lagr based on the cost constraint violation
        cost_violation = (mean_cost - self.safety_bound).detach()
        delta_lambda_lagr = cost_violation * self.cfg.lagrangian_coef_rate
        new_lambda_lagr = self.lambda_lagr + delta_lambda_lagr
        new_lambda_lagr = torch.nn.functional.relu(new_lambda_lagr)
        self.lambda_lagr = new_lambda_lagr
        return cost_violation

    def _get_lagrange_multiplier(self):
        return self.lambda_lagr

    def _compute_adv_surrogate(self, adv, cost_adv):
        return adv - self.lambda_lagr * cost_adv  # Adjusted advantage with cost

    def _train(
            self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_loss = 1e9
            epoch_losses = [0] * self.cfg.num_batches_per_epoch

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]
                    mb = self._get_minibatch(gpu_buffer, indices)
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        surrogate_loss,
                        value_loss,
                        cost_value_loss,
                        kl_div,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    total_value_loss = value_loss + cost_value_loss
                    epoch_losses[batch_num] = float(surrogate_loss + total_value_loss)

                # Perform TRPO update
                with timing.add_time("update"):
                    self._trpo_step(mb, surrogate_loss, kl_div, num_invalids, mb.valids)
                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    synchronize(self.cfg, self.device)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            new_epoch_loss = float(np.mean(epoch_losses))
            loss_delta_abs = abs(prev_epoch_loss - new_epoch_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                    )
                break

            prev_epoch_loss = new_epoch_loss

        return stats_and_summaries

    def _trpo_step(self, mb, surrogate_loss, kl_div, num_invalids, valids):
        # Implement TRPO update using conjugate gradient and line search

        # Compute policy gradients
        policy_params = [p for p in self.actor_critic.parameters() if p.requires_grad]
        loss = surrogate_loss
        self.optimizer.zero_grad()
        loss.backward()

        # Flatten gradients
        grads = torch.cat([p.grad.view(-1) for p in policy_params]).detach()

        # Fisher vector product function
        def Fvp(v):
            kl = self._compute_kl(mb, valids, num_invalids)
            kl = kl.mean()

            kl_grad = torch.autograd.grad(kl, policy_params, create_graph=True)
            flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in kl_grad])

            kl_grad_v = (flat_kl_grad * v).sum()
            kl_hessian = torch.autograd.grad(kl_grad_v, policy_params)
            flat_kl_hessian = torch.cat([g.contiguous().view(-1) for g in kl_hessian]).detach()

            return flat_kl_hessian + self.cfg.damping_coeff * v

        # Compute step direction using Conjugate Gradient
        step_dir = self._conjugate_gradient(Fvp, grads)

        # Compute step size
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum(0, keepdim=True)
        max_step = torch.sqrt(self.cfg.max_kl / shs)[0]
        full_step = -step_dir * max_step

        # Line search to enforce KL constraint
        prev_params = self._get_flat_params_from(policy_params)
        success, new_params = self._line_search(mb, prev_params, full_step, surrogate_loss, kl_div, valids, num_invalids)

        if success:
            self._set_flat_params_to(policy_params, new_params)
        else:
            log.warning("Line search failed. No parameter update performed.")

    def _compute_kl(self, mb, valids, num_invalids):
        action_distribution = self.actor_critic.action_distribution()
        old_action_distribution = get_action_distribution(
            self.actor_critic.action_space,
            mb.action_logits,
        )
        kl = old_action_distribution.kl_divergence(action_distribution)
        kl = masked_select(kl, valids, num_invalids)
        return kl

    def _conjugate_gradient(self, Avp_func, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = r.dot(r)

        for _ in range(nsteps):
            Avp = Avp_func(p)
            alpha = rdotr / (p.dot(Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def _line_search(self, mb, prev_params, full_step, prev_loss, prev_kl, valids, num_invalids):
        max_backtracks = 10
        accept_ratio = 0.1
        stepfrac = 1.0
        params = [p for p in self.actor_critic.parameters() if p.requires_grad]

        for _ in range(max_backtracks):
            new_params = prev_params + stepfrac * full_step
            self._set_flat_params_to(params, new_params)

            with torch.no_grad():
                # Recompute losses and KL divergence
                action_distribution = self.actor_critic.action_distribution()
                log_prob_actions = action_distribution.log_prob(mb.actions)
                adv = self._compute_adv_surrogate(mb.advantages, mb.cost_advantages)
                adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)

                surrogate_loss = -torch.mean(masked_select(log_prob_actions * adv, valids, num_invalids))
                kl_div = self._compute_kl(mb, valids, num_invalids).mean()

            # Check improvement and KL constraint
            loss_improve = prev_loss - surrogate_loss
            if loss_improve.item() > 0 and kl_div.item() <= self.cfg.max_kl:
                return True, new_params
            stepfrac *= 0.5

        # If line search fails, revert to previous parameters
        self._set_flat_params_to(params, prev_params)
        return False, prev_params

    def _get_flat_params_from(self, params):
        return torch.cat([p.data.view(-1) for p in params])

    def _set_flat_params_to(self, params, flat_params):
        prev_ind = 0
        for p in params:
            flat_size = int(np.prod(list(p.size())))
            p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
            prev_ind += flat_size

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars

        stats = super()._record_summaries(train_loop_vars)

        stats.cost_value_loss = var.cost_value_loss
        stats.cost_values = var.cost_values.mean()
        stats.cost_violation = var.cost_violation
        stats.avg_cost = var.avg_cost
        stats.lagrange_multiplier = var.lagrange_multiplier
        stats.kl_divergence = var.kl_divergence.item()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        # Similar to PPOLagLearner's _prepare_batch method
        with torch.no_grad():
            buff = shallow_recursive_copy(batch)
            valids: Tensor = buff["policy_id"] == self.policy_id
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            buff["valids"][:, -1] = buff["valids"][:, -2]

            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]

            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)
            buff["values"][:, -1] = next_values["values"]
            buff["cost_values"][:, -1] = next_values["cost_values"]

            if self.cfg.normalize_returns:
                denormalized_values = buff["values"].clone()
                denormalized_cost_values = buff["cost_values"].clone()
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
                self.actor_critic.costs_normalizer(denormalized_cost_values, denormalize=True)
            else:
                denormalized_values = buff["values"]
                denormalized_cost_values = buff["cost_values"]

            if self.cfg.value_bootstrap:
                buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])

            buff["advantages"] = gae_advantages(
                buff["rewards"],
                buff["dones"],
                denormalized_values,
                buff["valids"],
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )
            buff["cost_advantages"] = gae_advantages(
                buff["costs"],
                buff["dones"],
                denormalized_cost_values,
                buff["valids"],
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )

            buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]
            buff["cost_returns"] = buff["cost_advantages"] + buff["valids"][:, :-1] * denormalized_cost_values[:, :-1]

            for key in ["normalized_obs", "rnn_states", "values", "cost_values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])
                self.actor_critic.costs_normalizer(buff["cost_returns"])

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                buff["log_prob_actions"][invalid_indices] = -1

            return buff, dataset_size, num_invalids
