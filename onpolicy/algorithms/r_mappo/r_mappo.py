import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_counterfactual_credit = getattr(args, "use_counterfactual_credit", False)
        self._use_cf_advantage = getattr(args, "use_cf_advantage", False)
        self._use_cf_intervention_loss = getattr(args, "use_cf_intervention_loss", False)
        self._cf_shuffle_labels = getattr(args, "cf_shuffle_labels", False)
        self.cf_ordinary_loss_coef = getattr(args, "cf_ordinary_loss_coef", 1.0)
        self.cf_intervention_loss_coef = getattr(args, "cf_intervention_loss_coef", 1.0)
        self.cf_epoch = getattr(args, "cf_epoch", 1)
        self.cf_action_dim = getattr(self.policy.act_space, "n", 0)
        self.cf_num_agents = int(getattr(args, "num_agents", getattr(args, "pso_particles", 1)))
        self.cf_shuffle_label_pool_size = int(getattr(args, "cf_shuffle_label_pool_size", 256))
        self.cf_shuffle_delta_pool = []
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def _counterfactual_arrays(self, buffer):
        actions = buffer.actions[..., 0].astype(np.int64)
        episode_length, n_threads, num_agents = actions.shape
        share_obs = buffer.share_obs[:-1].reshape(episode_length, n_threads, num_agents, -1)
        share_obs = share_obs.reshape(-1, share_obs.shape[-1]).astype(np.float32)
        joint_context = np.repeat(actions.reshape(episode_length * n_threads, num_agents), num_agents, axis=0)
        agent_ids = np.tile(np.arange(num_agents, dtype=np.int64), episode_length * n_threads)
        actual_actions = actions.reshape(-1).astype(np.int64)
        returns = buffer.returns[:-1].reshape(-1, 1).astype(np.float32)
        active_masks = buffer.active_masks[:-1].reshape(-1, 1).astype(np.float32)
        return share_obs, joint_context, agent_ids, actual_actions, returns, active_masks

    def _shuffle_intervention_delta(self, delta):
        if not self._cf_shuffle_labels:
            return delta, 0.0, float(len(self.cf_shuffle_delta_pool))

        raw = delta.reshape(-1).copy()
        shuffled = raw.copy()
        changed = np.zeros(len(raw), dtype=np.float32)
        pool_before = float(len(self.cf_shuffle_delta_pool))
        if len(raw) > 1:
            perm = np.random.permutation(len(raw))
            if np.any(perm == np.arange(len(raw))):
                perm = np.roll(np.arange(len(raw)), 1)
            shuffled = raw[perm]
            changed = (perm != np.arange(len(raw))).astype(np.float32)
        elif len(raw) == 1 and len(self.cf_shuffle_delta_pool) > 0:
            shuffled[0] = float(np.random.choice(self.cf_shuffle_delta_pool))
            changed[0] = 1.0

        self.cf_shuffle_delta_pool.extend(float(x) for x in raw)
        if len(self.cf_shuffle_delta_pool) > self.cf_shuffle_label_pool_size:
            self.cf_shuffle_delta_pool = self.cf_shuffle_delta_pool[-self.cf_shuffle_label_pool_size:]

        return shuffled.reshape(-1, 1).astype(np.float32), float(np.mean(changed)), pool_before

    def train_counterfactual_critic(self, buffer):
        if self.policy.cf_critic is None:
            return {}

        share_obs, joint_context, agent_ids, actual_actions, returns, active_masks = self._counterfactual_arrays(buffer)
        interventions = getattr(buffer, "intervention_batch", [])
        infos = {
            "cf_ordinary_loss": 0.0,
            "cf_intervention_loss": 0.0,
            "cf_total_loss": 0.0,
            "cf_intervention_count": float(len(interventions)),
            "cf_shuffle_label_changed": 0.0,
            "cf_shuffle_label_pool_size": float(len(self.cf_shuffle_delta_pool)),
        }

        intervention_data = None
        if self._use_cf_intervention_loss and len(interventions) > 0:
            int_share_obs = np.stack([x["share_obs"] for x in interventions]).astype(np.float32)
            actual_joint = np.stack([x["joint_actions"] for x in interventions]).astype(np.int64)
            alt_joint = actual_joint.copy()
            int_agent_ids = np.asarray([x["agent_id"] for x in interventions], dtype=np.int64)
            actual_action = np.asarray([x["actual_action"] for x in interventions], dtype=np.int64)
            alt_action = np.asarray([x["alternative_action"] for x in interventions], dtype=np.int64)
            alt_joint[np.arange(len(interventions)), int_agent_ids] = alt_action
            delta = np.asarray([x["delta"] for x in interventions], dtype=np.float32).reshape(-1, 1)
            delta, changed, pool_before = self._shuffle_intervention_delta(delta)
            infos["cf_shuffle_label_changed"] = changed
            infos["cf_shuffle_label_pool_size"] = pool_before
            intervention_data = (
                int_share_obs,
                actual_joint,
                alt_joint,
                int_agent_ids,
                actual_action,
                alt_action,
                delta,
            )

        for _ in range(self.cf_epoch):
            q_actual = self.policy.cf_critic(share_obs, joint_context, agent_ids, actual_actions)
            target = torch.from_numpy(returns).to(**self.tpdv)
            active = torch.from_numpy(active_masks).to(**self.tpdv)
            ordinary_loss = (((q_actual - target) ** 2) * active).sum() / active.sum().clamp(min=1.0)

            intervention_loss = torch.zeros((), **self.tpdv)
            if intervention_data is not None:
                int_share_obs, actual_joint, alt_joint, int_agent_ids, actual_action, alt_action, delta = intervention_data
                q_real = self.policy.cf_critic(int_share_obs, actual_joint, int_agent_ids, actual_action)
                q_alt = self.policy.cf_critic(int_share_obs, alt_joint, int_agent_ids, alt_action)
                delta_t = torch.from_numpy(delta).to(**self.tpdv)
                intervention_loss = torch.mean((q_real - q_alt - delta_t) ** 2)

            total_loss = (
                self.cf_ordinary_loss_coef * ordinary_loss
                + self.cf_intervention_loss_coef * intervention_loss
            )
            self.policy.cf_critic_optimizer.zero_grad()
            total_loss.backward()
            if self._use_max_grad_norm:
                nn.utils.clip_grad_norm_(self.policy.cf_critic.parameters(), self.max_grad_norm)
            self.policy.cf_critic_optimizer.step()

            infos["cf_ordinary_loss"] += ordinary_loss.item()
            infos["cf_intervention_loss"] += intervention_loss.item()
            infos["cf_total_loss"] += total_loss.item()

        infos["cf_ordinary_loss"] /= max(1, self.cf_epoch)
        infos["cf_intervention_loss"] /= max(1, self.cf_epoch)
        infos["cf_total_loss"] /= max(1, self.cf_epoch)
        return infos

    @torch.no_grad()
    def compute_counterfactual_advantages(self, buffer):
        share_obs, joint_context, agent_ids, actual_actions, _, _ = self._counterfactual_arrays(buffer)
        obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[3:]).astype(np.float32)
        rnn_states = buffer.rnn_states[:-1].reshape(-1, *buffer.rnn_states.shape[3:]).astype(np.float32)
        masks = buffer.masks[:-1].reshape(-1, 1).astype(np.float32)
        probs = self.policy.get_action_probs(obs, rnn_states, masks)
        probs = probs.detach().cpu().numpy().astype(np.float32)

        q_values = []
        rows = np.arange(joint_context.shape[0])
        for candidate in range(self.cf_action_dim):
            candidate_joint = joint_context.copy()
            candidate_joint[rows, agent_ids] = candidate
            candidate_actions = np.full(joint_context.shape[0], candidate, dtype=np.int64)
            q_candidate = self.policy.cf_critic(share_obs, candidate_joint, agent_ids, candidate_actions)
            q_values.append(q_candidate.detach().cpu().numpy())
        q_values = np.concatenate(q_values, axis=1)
        q_actual = q_values[rows, actual_actions]
        baseline = np.sum(probs * q_values, axis=1)
        advantages = (q_actual - baseline).reshape(buffer.rewards.shape)
        return advantages.astype(np.float32)

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        cf_infos = {}
        if self._use_counterfactual_credit:
            cf_infos = self.train_counterfactual_critic(buffer)

        if self._use_counterfactual_credit and self._use_cf_advantage:
            advantages = self.compute_counterfactual_advantages(buffer)
        elif self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info.update(cf_infos)
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        if self.policy.cf_critic is not None:
            self.policy.cf_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        if self.policy.cf_critic is not None:
            self.policy.cf_critic.eval()
