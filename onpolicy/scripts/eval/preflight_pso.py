#!/usr/bin/env python
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config
from onpolicy.envs.pso import RingTopologyPSOEnv
from onpolicy.scripts.train.train_pso import (
    apply_pso_defaults,
    main as train_main,
    make_eval_env,
    make_train_env,
    parse_args,
)


def _training_args(results_dir):
    return [
        "--credit_mode", "cf_intervention",
        "--experiment_name", "smoke_preflight",
        "--results_dir", str(results_dir),
        "--seed", "1",
        "--pso_objective", "sphere",
        "--pso_particles", "4",
        "--pso_dim", "2",
        "--pso_generations", "5",
        "--num_env_steps", "10",
        "--n_rollout_threads", "1",
        "--n_eval_rollout_threads", "1",
        "--ppo_epoch", "1",
        "--num_mini_batch", "1",
        "--use_eval",
        "--eval_interval", "1",
        "--eval_episodes", "2",
        "--log_interval", "1",
        "--save_interval", "10",
        "--use_wandb",
        "--cuda",
    ]


def _parsed_args(results_dir):
    args = parse_args(_training_args(results_dir), get_config())
    apply_pso_defaults(args)
    return args


def _assert_equal(left, right, label="value"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), label
        for key in left:
            _assert_equal(left[key], right[key], "{}.{}".format(label, key))
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=label)
    elif isinstance(left, (float, np.floating)):
        assert left == right, "{}: {} != {}".format(label, left, right)
    else:
        assert left == right, "{}: {} != {}".format(label, left, right)


def _obs_policy(obs):
    candidate_scores = np.asarray(obs)[:, [4, 8, 12]]
    return np.argmax(candidate_scores, axis=1).astype(np.int64)


def _rollout_trace(env, snapshot, initial_actions):
    env.restore_snapshot(snapshot)
    obs, _, dones, _ = env.step(initial_actions)
    trace = [(np.asarray(initial_actions).copy(), env.get_snapshot())]
    while not np.all(dones):
        actions = _obs_policy(obs)
        obs, _, dones, _ = env.step(actions)
        trace.append((actions.copy(), env.get_snapshot()))
    return trace


def verify_environment(args):
    env = RingTopologyPSOEnv(args)
    env.seed(args.seed)
    env.reset()
    for _ in range(2):
        env.step(env.ring_best_actions())

    snapshot = env.get_snapshot()
    actual = env.ring_best_actions()
    first_trace = _rollout_trace(env, snapshot, actual)
    second_trace = _rollout_trace(env, snapshot, actual)
    assert len(first_trace) == len(second_trace)
    for index, ((actions_a, state_a), (actions_b, state_b)) in enumerate(zip(first_trace, second_trace)):
        _assert_equal(actions_a, actions_b, "trace[{}].actions".format(index))
        _assert_equal(state_a, state_b, "trace[{}].state".format(index))

    main_endpoint = first_trace[-1][1]
    env.restore_snapshot(main_endpoint)
    endpoint_before_branch = env.get_snapshot()
    alternative = actual.copy()
    alternative[0] = (alternative[0] + 1) % 3
    alternative_trace = _rollout_trace(env, snapshot, alternative)
    assert not np.array_equal(
        first_trace[0][1]["positions"],
        alternative_trace[0][1]["positions"],
    )
    for branch_name, trace in (("actual", first_trace), ("alternative", alternative_trace)):
        for index in range(1, len(trace)):
            env.restore_snapshot(trace[index - 1][1])
            expected_actions = _obs_policy(env._get_obs())
            _assert_equal(trace[index][0], expected_actions, "{}.actions[{}]".format(branch_name, index))

    env.restore_snapshot(endpoint_before_branch)
    _assert_equal(endpoint_before_branch, env.get_snapshot(), "branch_restoration")

    expected_main = args.pso_particles * (1 + args.pso_generations)
    assert main_endpoint["total_evaluations"] == expected_main
    expected_branch = args.pso_particles * (args.pso_generations - snapshot["generation"])
    env.restore_snapshot(endpoint_before_branch)
    _, branch_evals = env.rollout_alternative_from_snapshot(snapshot, alternative, _obs_policy)
    assert branch_evals == expected_branch
    print("PASS environment snapshot, paired replay, branch restoration, and evaluation counts")


def _state_dict_equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _clone_module(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class CandidateProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, cent_obs, joint_actions, agent_ids, candidate_actions):
        joint = torch.as_tensor(joint_actions)
        agents = torch.as_tensor(agent_ids).long().view(-1)
        candidates = torch.as_tensor(candidate_actions).long().view(-1)
        rows = torch.arange(joint.shape[0])
        assert torch.equal(joint[rows, agents].cpu(), candidates.cpu())
        self.calls += 1
        return torch.zeros((joint.shape[0], 1), dtype=torch.float32)


def verify_training(results_dir):
    train_main(_training_args(results_dir))
    run_dir = results_dir / "PSO" / "sphere" / "cf_intervention" / "smoke_preflight" / "run1"
    with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"

    with open(run_dir / "episode_metrics.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    per_episode_main = 4 * (1 + 5)
    per_eval_event = 2 * per_episode_main
    for index, record in enumerate(records, start=1):
        assert record["cumulative_main_function_evaluations"] == index * per_episode_main
        assert record["cumulative_eval_function_evaluations"] == index * per_eval_event
        assert record["cumulative_train_function_evaluations"] == (
            record["cumulative_main_function_evaluations"]
            + record["cumulative_intervention_function_evaluations"]
        )
        assert record["cumulative_total_function_evaluations"] == (
            record["cumulative_train_function_evaluations"]
            + record["cumulative_eval_function_evaluations"]
        )
    assert records[-1]["total_num_steps"] == 10
    assert records[-1]["eval_final_global_best"] != ""
    assert len(records[-1]["eval_final_global_best_values"]) == 2

    args = _parsed_args(results_dir)
    env = RingTopologyPSOEnv(args)
    torch.manual_seed(args.seed)
    initial_policy = R_MAPPOPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=torch.device("cpu"),
    )
    saved_actor = torch.load(run_dir / "models" / "actor.pt", map_location="cpu")
    assert not _state_dict_equal(_clone_module(initial_policy.actor), saved_actor)

    reload_args = _parsed_args(results_dir)
    reload_args.model_dir = str(run_dir / "models")
    envs = make_train_env(reload_args)
    eval_envs = make_eval_env(reload_args)
    reload_dir = results_dir / "reload_check"
    reload_dir.mkdir(parents=True, exist_ok=True)
    from onpolicy.runner.shared.pso_runner import PSORunner

    runner = PSORunner({
        "all_args": reload_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": reload_args.num_agents,
        "device": torch.device("cpu"),
        "run_dir": reload_dir,
    })
    try:
        assert _state_dict_equal(_clone_module(runner.policy.actor), saved_actor)
        saved_critic = torch.load(run_dir / "models" / "critic.pt", map_location="cpu")
        saved_cf = torch.load(run_dir / "models" / "cf_critic.pt", map_location="cpu")
        assert _state_dict_equal(_clone_module(runner.policy.critic), saved_critic)
        assert _state_dict_equal(_clone_module(runner.policy.cf_critic), saved_cf)

        frozen = {
            "actor": _clone_module(runner.policy.actor),
            "critic": _clone_module(runner.policy.critic),
            "cf": _clone_module(runner.policy.cf_critic),
        }
        runner.eval(total_num_steps=10)
        assert _state_dict_equal(frozen["actor"], _clone_module(runner.policy.actor))
        assert _state_dict_equal(frozen["critic"], _clone_module(runner.policy.critic))
        assert _state_dict_equal(frozen["cf"], _clone_module(runner.policy.cf_critic))

        original_cf = runner.trainer.policy.cf_critic
        probe = CandidateProbe()
        runner.trainer.policy.cf_critic = probe
        runner.trainer.compute_counterfactual_advantages(runner.buffer)
        assert probe.calls == 3
        runner.trainer.policy.cf_critic = original_cf
    finally:
        runner.writter.flush()
        runner.writter.close()
        envs.close()
        eval_envs.close()

    print("PASS training update, final evaluation, accounting, save/load, frozen eval, and candidate replacement")


def main():
    with tempfile.TemporaryDirectory(prefix="cspso-preflight-") as temp_dir:
        results_dir = Path(temp_dir)
        args = _parsed_args(results_dir)
        verify_environment(args)
        verify_training(results_dir)
    print("PSO preflight passed")


if __name__ == "__main__":
    main()
