#!/usr/bin/env python
import argparse
import copy

import numpy as np

from onpolicy.envs.pso import RingTopologyPSOEnv


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pso_particles", type=int, default=20)
    parser.add_argument("--pso_dim", type=int, default=10)
    parser.add_argument("--pso_generations", type=int, default=100)
    parser.add_argument("--pso_objective", choices=["sphere", "rastrigin", "rosenbrock", "ackley"], default="rastrigin")
    parser.add_argument("--pso_lower_bound", type=float, default=None)
    parser.add_argument("--pso_upper_bound", type=float, default=None)
    parser.add_argument("--pso_chi", type=float, default=0.72984)
    parser.add_argument("--pso_c1", type=float, default=2.05)
    parser.add_argument("--pso_c2", type=float, default=2.05)
    parser.add_argument("--pso_boundary", choices=["clip", "clip_zero_velocity", "none"], default="clip")
    parser.add_argument("--pso_initial_velocity_scale", type=float, default=0.1)
    parser.add_argument("--pso_scale_floor", type=float, default=1.0)
    parser.add_argument("--probe_generation", type=int, default=10)
    parser.add_argument("--particle", type=int, default=0)
    return parser.parse_args()


def continue_with_ring(env, initial_actions):
    _, _, dones, _ = env.step(initial_actions)
    while not dones.all():
        _, _, dones, _ = env.step(env.ring_best_actions())
    return env.global_best_value


def paired_branch(env, snapshot, initial_actions):
    env.restore_snapshot(snapshot)
    value = continue_with_ring(env, initial_actions)
    return value


def main():
    args = build_args()
    env = RingTopologyPSOEnv(args)
    env.seed(args.seed)
    env.reset()
    for _ in range(args.probe_generation):
        env.step(env.ring_best_actions())

    snapshot = env.get_snapshot()
    actual_actions = env.ring_best_actions()
    same_actions = actual_actions.copy()
    alt_actions = actual_actions.copy()
    particle = args.particle % args.pso_particles
    alt_actions[particle] = (alt_actions[particle] + 1) % 3

    original_rng_state = copy.deepcopy(env.rng.bit_generator.state)
    same_value = paired_branch(env, snapshot, same_actions)
    env.rng.bit_generator.state = copy.deepcopy(original_rng_state)
    actual_value = paired_branch(env, snapshot, actual_actions)
    null_delta = same_value - actual_value

    social_args = copy.copy(args)
    social_args.pso_c2 = 0.0
    social_env = RingTopologyPSOEnv(social_args)
    social_env.seed(args.seed)
    social_env.reset()
    for _ in range(args.probe_generation):
        social_env.step(social_env.ring_best_actions())
    social_snapshot = social_env.get_snapshot()
    social_actual = social_env.ring_best_actions()
    social_alt = social_actual.copy()
    social_alt[particle] = (social_alt[particle] + 1) % 3
    social_rng_state = copy.deepcopy(social_env.rng.bit_generator.state)
    social_actual_value = paired_branch(social_env, social_snapshot, social_actual)
    social_env.rng.bit_generator.state = copy.deepcopy(social_rng_state)
    social_alt_value = paired_branch(social_env, social_snapshot, social_alt)
    social_delta = social_alt_value - social_actual_value

    print("same-action paired delta: {:.12g}".format(null_delta))
    print("c2=0 leader-change delta: {:.12g}".format(social_delta))
    if np.isclose(null_delta, 0.0) and np.isclose(social_delta, 0.0):
        print("diagnosis passed")
    else:
        print("diagnosis failed")


if __name__ == "__main__":
    main()
