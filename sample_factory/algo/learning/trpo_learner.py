from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod
from os.path import join
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import gae_advantages, prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir, log


class TRPOLearner(Configurable):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        Configurable.__init__(self, cfg)

        self.timing = Timing(name=f"Learner {policy_id} profile")

        self.policy_id = policy_id

        self.env_info = env_info

        self.device = None
        self.actor_critic: Optional[ActorCritic] = None

        self.curr_lr: Optional[float] = None

        self.train_step: int = 0  # total number of optimization steps
        self.env_steps: int = 0  # total number of environment steps consumed by the learner

        self.best_performance = -1e9

        self.new_cfg: Optional[Dict] = None

        self.policy_to_load: Optional[PolicyID] = None

        self.summary_rate_decay_seconds = LinearDecay([(0, 2), (100000, 60), (1000000, 120)])
        self.last_summary_time = 0
        self.last_milestone_time = 0

        self.policy_versions_tensor: Tensor = policy_versions_tensor

        self.param_server: ParameterServer = param_server

        self.is_initialized = False

    def init(self) -> InitModelData:
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.device = policy_device(self.cfg, self.policy_id)

        log.debug("Initializing actor-critic model on device %s", self.device)

        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)

        def share_mem(t):
            if t is not None and not t.is_cuda:
                return t.share_memory_()
            return t

        # noinspection PyProtectedMember
        self.actor_critic._apply(share_mem)
        self.actor_critic.train()

        # TRPO does not use a standard optimizer like Adam; instead, updates are computed via conjugate gradient

        # self.load_from_checkpoint(self.policy_id)
        self.param_server.init(self.actor_critic, self.train_step, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        self.is_initialized = True

        return model_initialization_data(self.cfg, self.policy_id, self.actor_critic, self.train_step, self.device)

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimization step."""
        self.train_step += 1

    def _get_checkpoint_dict(self):
        checkpoint = {
            "train_step": self.train_step,
            "env_steps": self.env_steps,
            "best_performance": self.best_performance,
            "model": self.actor_critic.state_dict(),
            "curr_lr": self.curr_lr,
        }
        return checkpoint

    def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True) -> bool:
        if not self.is_initialized:
            return False

        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, f"{name_prefix}_temp")
        checkpoint_name = f"{name_prefix}_{self.train_step:09d}_{self.env_steps}{name_suffix}.pth"
        filepath = join(checkpoint_dir, checkpoint_name)
        if verbose:
            log.info("Saving %s...", filepath)

        torch.save(checkpoint, tmp_filepath)
        os.rename(tmp_filepath, filepath)

        while len(checkpoints := self.get_checkpoints(checkpoint_dir, f"{name_prefix}_*")) > keep_checkpoints:
            oldest_checkpoint = checkpoints[0]
            if os.path.isfile(oldest_checkpoint):
                if verbose:
                    log.debug("Removing %s", oldest_checkpoint)
                os.remove(oldest_checkpoint)

        return True

    def save(self) -> bool:
        return self._save_impl("checkpoint", "", self.cfg.keep_checkpoints)

    def save_milestone(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None
        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        checkpoint_name = f"checkpoint_{self.train_step:09d}_{self.env_steps}.pth"

        milestones_dir = ensure_dir_exists(join(checkpoint_dir, "milestones"))
        milestone_path = join(milestones_dir, f"{checkpoint_name}")
        log.info("Saving a milestone %s", milestone_path)
        torch.save(checkpoint, milestone_path)

    def save_best(self, policy_id, metric, metric_value) -> bool:
        if policy_id != self.policy_id:
            return False
        p = 3  # precision, number of significant digits
        if metric_value - self.best_performance > 1 / 10**p:
            log.info(f"Saving new best policy, {metric}={metric_value:.{p}f}!")
            self.best_performance = metric_value
            name_suffix = f"_{metric}_{metric_value:.{p}f}"
            return self._save_impl("best", name_suffix, 1, verbose=False)

        return False

    def set_new_cfg(self, new_cfg: Dict) -> None:
        self.new_cfg = new_cfg

    def set_policy_to_load(self, policy_to_load: PolicyID) -> None:
        self.policy_to_load = policy_to_load

    def _maybe_update_cfg(self) -> None:
        if self.new_cfg is not None:
            for key, value in self.new_cfg.items():
                if getattr(self.cfg, key) != value:
                    log.debug("Learner %d replacing cfg parameter %r with new value %r", self.policy_id, key, value)
                    setattr(self.cfg, key, value)
            self.new_cfg = None

    def _maybe_load_policy(self) -> None:
        cfg = self.cfg
        if self.policy_to_load is not None:
            with self.param_server.policy_lock:
                self.load_from_checkpoint(self.policy_to_load, load_progress=False)
            synchronize(cfg, self.device)
            self.train_step += cfg.max_policy_lag + 1
            self.policy_versions_tensor[self.policy_id] = self.train_step
            self.policy_to_load = None

    def _calculate_losses(
            self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor, Tensor, Dict]:
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            valids = mb.valids

        with self.timing.add_time("forward_head"):
            head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = head_outputs.size(0)

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

        with self.timing.add_time("tail"):
            result = self.actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            values = result["values"].squeeze()
            del core_outputs

        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            adv = mb.advantages
            targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)

        with self.timing.add_time("losses"):
            surrogate_loss = -torch.mean(masked_select(log_prob_actions * adv, valids, num_invalids))

            old_action_distribution = get_action_distribution(
                self.actor_critic.action_space,
                mb.action_logits,
            )
            kl_div = old_action_distribution.kl_divergence(action_distribution)
            kl_div = masked_select(kl_div, valids, num_invalids).mean()

            value_loss = self._value_loss(values, mb["values"], targets, self.cfg.ppo_clip_value, valids, num_invalids)

        loss_summaries = dict(
            values=result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            kl_divergence=kl_div,
        )

        return action_distribution, surrogate_loss, value_loss, kl_div, loss_summaries

    def _value_loss(
            self,
            new_values: Tensor,
            old_values: Tensor,
            target: Tensor,
            clip_value: float,
            valids: Tensor,
            num_invalids: int,
    ) -> Tensor:
        value_loss = (new_values - target).pow(2)
        value_loss = masked_select(value_loss, valids, num_invalids)
        value_loss = value_loss.mean()
        value_loss *= self.cfg.value_loss_coeff
        return value_loss

    def _train(
            self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_loss = 1e9
            epoch_losses = [0] * self.cfg.num_batches_per_epoch

            num_optimization_steps = 0
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
                        kl_div,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    total_loss = surrogate_loss + value_loss
                    epoch_losses[batch_num] = float(total_loss)

                with timing.add_time("update"):
                    self._trpo_step(mb, surrogate_loss, kl_div, num_invalids, mb.valids)
                    num_optimization_steps += 1

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
                    "Early stopping after %d epochs (%d optimization steps), loss delta %.7f",
                    epoch + 1,
                    num_optimization_steps,
                    loss_delta_abs,
                    )
                break

            prev_epoch_loss = new_epoch_loss

        return stats_and_summaries

    def _trpo_step(self, mb, surrogate_loss, kl_div, num_invalids, valids):
        # Implement TRPO update using conjugate gradient and line search
        policy_params = [p for p in self.actor_critic.parameters() if p.requires_grad]

        # Compute policy gradients
        loss = surrogate_loss
        for p in policy_params:
            p.grad = None
        loss.backward(retain_graph=True)

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
        success, new_params = self._line_search(
            mb, prev_params, full_step, surrogate_loss, kl_div, valids, num_invalids
        )

        if success:
            self._set_flat_params_to(policy_params, new_params)
        else:
            log.warning("Line search failed. No parameter update performed.")

        # Update value function using standard gradient descent
        value_params = [p for p in self.actor_critic.value_function_parameters()]
        value_loss = self._value_loss(
            self.actor_critic.values, mb["values"], mb.returns, self.cfg.ppo_clip_value, valids, num_invalids
        )
        for p in value_params:
            p.grad = None
        value_loss.backward()
        # Update value parameters using a simple optimizer, e.g., Adam
        self._update_value_params(value_params)

    def _update_value_params(self, value_params):
        lr = self.cfg.value_learning_rate
        for p in value_params:
            if p.grad is not None:
                p.data -= lr * p.grad

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
                adv = mb.advantages
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

        self.last_summary_time = time.time()
        stats = AttrDict()

        stats.lr = self.curr_lr

        stats.update(self.actor_critic.summaries())

        stats.valids_fraction = var.mb.valids.float().mean()
        stats.same_policy_fraction = (var.mb.policy_id == self.policy_id).float().mean()

        grad_norm = (
                sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters() if p.grad is not None) ** 0.5
        )
        stats.grad_norm = grad_norm
        stats.loss = var.surrogate_loss + var.value_loss
        stats.value = var.values.mean()
        stats.policy_loss = var.surrogate_loss
        stats.value_loss = var.value_loss

        stats.act_min = var.mb.actions.min()
        stats.act_max = var.mb.actions.max()

        if "adv_mean" in stats:
            stats.adv_min = var.mb.advantages.min()
            stats.adv_max = var.mb.advantages.max()
            stats.adv_std = var.adv_std
            stats.adv_mean = var.adv_mean

        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        if hasattr(var.action_distribution, "summaries"):
            stats.update(var.action_distribution.summaries())

        stats.kl_divergence = var.kl_divergence.item()
        stats.num_optimization_steps = var.num_optimization_steps

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _prepare_and_normalize_obs(self, obs: TensorDict) -> TensorDict:
        og_shape = dict()

        for key, x in obs.items():
            og_shape[key] = x.shape
            obs[key] = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])

        with self.param_server.policy_lock:
            normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)

        for key, x in normalized_obs.items():
            normalized_obs[key] = x.view(og_shape[key])

        return normalized_obs

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
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
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)["values"]
            buff["values"][:, -1] = next_values

            if self.cfg.normalize_returns:
                denormalized_values = buff["values"].clone()
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
            else:
                denormalized_values = buff["values"]

            if self.cfg.value_bootstrap:
                buff["rewards"].add_(
                    self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"]
                )

            buff["advantages"] = gae_advantages(
                buff["rewards"],
                buff["dones"],
                denormalized_values,
                buff["valids"],
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )

            buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]

            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                buff["log_prob_actions"][invalid_indices] = -1

            return buff, dataset_size, num_invalids

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                train_stats = self._train(buff, self.cfg.batch_size, experience_size, num_invalids)

            if self.cfg.summaries_use_frameskip:
                self.env_steps += experience_size * self.env_info.frameskip
            else:
                self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f"checkpoint_p{policy_id}")
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        checkpoints = glob.glob(join(checkpoints_dir, pattern))
        return sorted(checkpoints)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning("No checkpoints found")
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f"Could not load from checkpoint, attempt {attempt}")

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict["train_step"]
            self.env_steps = checkpoint_dict["env_steps"]
            self.best_performance = checkpoint_dict.get("best_performance", self.best_performance)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.curr_lr = checkpoint_dict.get("curr_lr", self.cfg.learning_rate)

        log.info(f"Loaded experiment state at {self.train_step=}, {self.env_steps=}")

    def load_from_checkpoint(self, policy_id: PolicyID, load_progress: bool = True) -> None:
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f"{name_prefix}_*")
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug("Did not load from checkpoint, starting from scratch!")
        else:
            log.debug("Loading model from checkpoint")
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def _get_minibatches(self, batch_size, experience_size):
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % batch_size == 0, f"experience size: {experience_size}, batch size: {batch_size}"
        minibatches_per_epoch = self.cfg.num_batches_per_epoch

        if minibatches_per_epoch == 1:
            return [None]

        if self.cfg.shuffle_minibatches:
            indices = np.arange(0, experience_size, self.cfg.recurrence)
            indices = np.random.permutation(indices)
            indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
            indices = np.concatenate(indices)
            assert len(indices) == experience_size
            num_minibatches = experience_size // batch_size
            minibatches = np.split(indices, num_minibatches)
        else:
            minibatches = list(slice(i * batch_size, (i + 1) * batch_size) for i in range(0, minibatches_per_epoch))
        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            return buffer
        mb = buffer[indices]
        return mb
