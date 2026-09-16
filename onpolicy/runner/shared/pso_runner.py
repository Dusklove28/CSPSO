import time

import numpy as np
import torch

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class PSORunner(Runner):
    """Runner for sparse-intervention PSO credit-assignment experiments."""

    def __init__(self, config):
        super(PSORunner, self).__init__(config)
        self.intervention_rng = np.random.default_rng(self.all_args.seed + 7919)
        if self.all_args.use_cf_intervention_loss:
            if self.n_rollout_threads != 1:
                raise ValueError("Sparse PSO interventions currently require --n_rollout_threads 1.")
            if self.all_args.use_recurrent_policy or self.all_args.use_naive_recurrent_policy:
                raise ValueError("Sparse PSO interventions currently require non-recurrent MAPPO.")

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = max(1, episodes)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            intervention_plan = self._sample_intervention_plan(episode)
            pending_interventions = []
            last_infos = None

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                if intervention_plan is not None and step == intervention_plan["step"]:
                    pending_interventions = self._capture_interventions(step, actions, intervention_plan)

                obs, rewards, dones, infos = self.envs.step(actions_env)
                last_infos = infos

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            actual_final_best = self._final_best_from_infos(last_infos)
            intervention_batch, intervention_infos = self._finalize_interventions(
                pending_interventions,
                actual_final_best,
            )
            self.buffer.intervention_batch = intervention_batch

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / max(end - start, 1e-6))
                print(
                    "\n PSO objective {} credit {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.pso_objective,
                        self.all_args.credit_mode,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        fps,
                    )
                )
                env_infos = self._episode_env_infos(last_infos, intervention_infos)
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["fps"] = fps
                print(
                    "final global best is {}, intervention count is {}".format(
                        np.mean(env_infos["final_global_best"]),
                        len(intervention_batch),
                    )
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions.astype(np.int64)], 2)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    def _sample_intervention_plan(self, episode):
        if not self.all_args.use_cf_intervention_loss:
            return None
        if episode < self.all_args.pso_intervention_warmup_episodes:
            return None
        interval = max(1, int(self.all_args.pso_intervention_interval))
        if episode % interval != 0:
            return None
        return {
            "step": int(self.intervention_rng.integers(0, self.episode_length)),
            "count": int(self.all_args.pso_interventions_per_event),
        }

    def _capture_interventions(self, step, actions, plan):
        env = self.envs.envs[0]
        count = max(1, plan["count"])
        count = min(count, self.num_agents)
        particle_ids = self.intervention_rng.choice(self.num_agents, size=count, replace=False)
        joint_actions = actions[0, :, 0].astype(np.int64)
        records = []
        for particle_id in particle_ids:
            actual = int(joint_actions[particle_id])
            alternatives = [x for x in range(self.envs.action_space[0].n) if x != actual]
            alternative = int(self.intervention_rng.choice(alternatives))
            alt_joint = joint_actions.copy()
            alt_joint[particle_id] = alternative
            records.append(
                {
                    "snapshot": env.get_snapshot(),
                    "share_obs": self.buffer.share_obs[step, 0, particle_id].copy(),
                    "joint_actions": joint_actions.copy(),
                    "alternative_joint_actions": alt_joint,
                    "agent_id": int(particle_id),
                    "actual_action": actual,
                    "alternative_action": alternative,
                    "reward_scale": float(env.reward_scale),
                    "torch_rng_state": self._capture_torch_rng_state(),
                }
            )
        return records

    def _finalize_interventions(self, records, actual_final_best):
        if len(records) == 0:
            return [], {"intervention_delta": [], "intervention_extra_evaluations": []}

        env = self.envs.envs[0]
        batch = []
        deltas = []
        branch_evals = []
        for record in records:
            current_env_snapshot = env.get_snapshot()
            current_torch_rng = self._capture_torch_rng_state()
            self._restore_torch_rng_state(record["torch_rng_state"])
            alt_final, evals_used = env.rollout_alternative_from_snapshot(
                record["snapshot"],
                record["alternative_joint_actions"],
                self._branch_policy,
            )
            self._restore_torch_rng_state(current_torch_rng)
            env.restore_snapshot(current_env_snapshot)
            env.extra_evaluations += evals_used

            delta = (float(alt_final) - float(actual_final_best[0])) / record["reward_scale"]
            item = {
                "share_obs": record["share_obs"],
                "joint_actions": record["joint_actions"],
                "agent_id": record["agent_id"],
                "actual_action": record["actual_action"],
                "alternative_action": record["alternative_action"],
                "delta": float(delta),
                "alternative_final_best": float(alt_final),
                "actual_final_best": float(actual_final_best[0]),
                "extra_evaluations": int(evals_used),
            }
            batch.append(item)
            deltas.append(float(delta))
            branch_evals.append(int(evals_used))
        return batch, {"intervention_delta": deltas, "intervention_extra_evaluations": branch_evals}

    @torch.no_grad()
    def _branch_policy(self, obs):
        self.trainer.prep_rollout()
        rnn_states = np.zeros(
            (self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.num_agents, 1), dtype=np.float32)
        action, _ = self.trainer.policy.act(
            obs.astype(np.float32),
            rnn_states,
            masks,
            deterministic=self.all_args.pso_cf_branch_deterministic,
        )
        return _t2n(action).reshape(self.num_agents).astype(np.int64)

    def _final_best_from_infos(self, infos):
        final_best = []
        for env_info in infos:
            final_best.append(float(env_info[0]["global_best"]))
        return final_best

    def _episode_env_infos(self, infos, intervention_infos):
        final_best = self._final_best_from_infos(infos)
        evals = [float(env_info[0]["evaluations"]) for env_info in infos]
        extra_evals = []
        for env_id, env_info in enumerate(infos):
            env_extra = float(env_info[0]["extra_evaluations"])
            if hasattr(self.envs, "envs"):
                env_extra = float(self.envs.envs[env_id].extra_evaluations)
            extra_evals.append(env_extra)
        env_infos = {
            "final_global_best": final_best,
            "function_evaluations": evals,
            "extra_function_evaluations": extra_evals,
            "extra_eval_ratio": [x / max(1.0, e) for x, e in zip(extra_evals, evals)],
        }
        env_infos.update(intervention_infos)
        return env_infos

    def _capture_torch_rng_state(self):
        state = {"cpu": torch.get_rng_state().clone(), "cuda": None}
        if torch.cuda.is_available():
            state["cuda"] = [x.clone() for x in torch.cuda.get_rng_state_all()]
        return state

    def _restore_torch_rng_state(self, state):
        torch.set_rng_state(state["cpu"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_best = []
        for _ in range(self.all_args.eval_episodes):
            obs = self.eval_envs.reset()
            eval_rnn_states = np.zeros(
                (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            infos = None
            for _ in range(self.episode_length):
                self.trainer.prep_rollout()
                action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_eval_rollout_threads))
                actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[actions.astype(np.int64)], 2)
                obs, _, dones, infos = self.eval_envs.step(actions_env)
                eval_rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            eval_episode_best.extend(self._final_best_from_infos(infos))

        eval_env_infos = {"eval_final_global_best": eval_episode_best}
        print("eval final global best: " + str(np.mean(eval_episode_best)))
        self.log_env(eval_env_infos, total_num_steps)
