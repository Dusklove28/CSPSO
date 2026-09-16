#!/usr/bin/env python
import argparse
import statistics

from onpolicy.envs.pso import RingTopologyPSOEnv


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=["ring_best", "random"], default="ring_best")
    parser.add_argument("--seeds", type=int, default=5)
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
    return parser.parse_args()


def run_once(args, seed):
    env = RingTopologyPSOEnv(args)
    env.seed(seed)
    env.reset()
    done = False
    while not done:
        if args.rule == "ring_best":
            actions = env.ring_best_actions()
        else:
            actions = env.random_actions()
        _, _, dones, _ = env.step(actions)
        done = bool(dones.all())
    return env.global_best_value, env.total_evaluations


def main():
    args = build_args()
    values = []
    evals = []
    for offset in range(args.seeds):
        value, evaluations = run_once(args, args.seed + offset)
        values.append(value)
        evals.append(evaluations)
        print("seed={} final_best={:.10g} evaluations={}".format(args.seed + offset, value, evaluations))
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    print("mean_final_best={:.10g} std={:.10g} mean_evaluations={:.1f}".format(
        statistics.mean(values),
        std,
        statistics.mean(evals),
    ))


if __name__ == "__main__":
    main()
