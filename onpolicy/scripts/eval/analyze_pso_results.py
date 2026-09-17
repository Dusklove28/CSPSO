#!/usr/bin/env python
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


OBJECTIVES = ("sphere", "rastrigin", "rosenbrock", "ackley")
MODES = ("mappo", "cf_no_intervention", "cf_intervention", "cf_intervention_shuffled")
SEEDS = (1, 2, 3, 4, 5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("runs/2026-09-17/PSO"))
    return parser.parse_args()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def average_ranks(values_by_mode):
    ordered = sorted(values_by_mode.items(), key=lambda item: item[1])
    ranks = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def main():
    args = parse_args()
    runs = {}
    errors = []
    commits = set()
    dirty_values = set()

    for manifest_path in args.results_dir.rglob("run_manifest.json"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        config = manifest["config"]
        key = (config["pso_objective"], config["credit_mode"], int(config["seed"]))
        if key in runs:
            errors.append("duplicate run {}".format(key))
            continue
        metrics_path = manifest_path.parent / "episode_metrics.jsonl"
        records = read_jsonl(metrics_path) if metrics_path.exists() else []
        runs[key] = {"manifest": manifest, "records": records, "path": manifest_path.parent}
        commits.add(manifest["git"]["commit"])
        dirty_values.add(manifest["git"]["dirty"])

    expected = {(objective, mode, seed) for objective in OBJECTIVES for mode in MODES for seed in SEEDS}
    missing = sorted(expected - set(runs))
    unexpected = sorted(set(runs) - expected)
    if missing:
        errors.append("missing combinations: {}".format(missing))
    if unexpected:
        errors.append("unexpected combinations: {}".format(unexpected))

    results = {}
    cost_by_mode = defaultdict(list)
    intervention_counts = defaultdict(list)
    shuffle_changes = []
    for key, run in sorted(runs.items()):
        objective, mode, seed = key
        manifest = run["manifest"]
        config = manifest["config"]
        records = run["records"]
        if manifest["status"] != "completed":
            errors.append("{} status={}".format(key, manifest["status"]))
        expected_config = {
            "pso_particles": 20,
            "pso_dim": 10,
            "pso_generations": 100,
            "num_env_steps": 20000,
            "n_rollout_threads": 1,
            "eval_interval": 10,
            "eval_episodes": 10,
        }
        for name, expected_value in expected_config.items():
            if config.get(name) != expected_value:
                errors.append("{} {}={} expected {}".format(key, name, config.get(name), expected_value))
        if len(records) != 200:
            errors.append("{} has {} episode records".format(key, len(records)))
            continue

        eval_records = [record for record in records if record.get("eval_final_global_best", "") != ""]
        eval_episodes = [record["episode"] + 1 for record in eval_records]
        if eval_episodes != list(range(10, 201, 10)):
            errors.append("{} evaluation episodes={}".format(key, eval_episodes))
        final = records[-1]
        final_values = final.get("eval_final_global_best_values", [])
        if len(final_values) != 10:
            errors.append("{} final eval count={}".format(key, len(final_values)))
            continue
        final_mean = statistics.mean(final_values)
        if not math.isclose(final_mean, float(final["eval_final_global_best"]), rel_tol=1e-12, abs_tol=1e-12):
            errors.append("{} final eval mean mismatch".format(key))

        main_evals = int(final["cumulative_main_function_evaluations"])
        intervention_evals = int(final["cumulative_intervention_function_evaluations"])
        eval_evals = int(final["cumulative_eval_function_evaluations"])
        train_evals = int(final["cumulative_train_function_evaluations"])
        total_evals = int(final["cumulative_total_function_evaluations"])
        if main_evals != 404000:
            errors.append("{} main evaluations={}".format(key, main_evals))
        if eval_evals != 404000:
            errors.append("{} eval evaluations={}".format(key, eval_evals))
        if train_evals != main_evals + intervention_evals or total_evals != train_evals + eval_evals:
            errors.append("{} evaluation accounting mismatch".format(key))
        if mode in ("mappo", "cf_no_intervention") and intervention_evals != 0:
            errors.append("{} unexpected intervention evaluations={}".format(key, intervention_evals))

        count = sum(int(record.get("intervention_count", 0)) for record in records)
        if mode.startswith("cf_intervention") and count != 50:
            errors.append("{} intervention count={}".format(key, count))
        intervention_counts[mode].append(count)
        if mode == "cf_intervention_shuffled":
            shuffle_changes.append(sum(float(record.get("cf_shuffle_label_changed", 0.0)) for record in records))

        cost_by_mode[mode].append(train_evals)
        results[key] = final_mean

    print("INTEGRITY")
    print("runs={} commits={} dirty_values={} errors={}".format(
        len(runs), sorted(commits), sorted(dirty_values, key=str), len(errors)
    ))
    for error in errors:
        print("ERROR " + error)

    print("\nFINAL EVALUATION BY OBJECTIVE (mean +/- sample SD across five seed means; median)")
    for objective in OBJECTIVES:
        print(objective)
        for mode in MODES:
            values = [results[(objective, mode, seed)] for seed in SEEDS]
            mean, sd = mean_sd(values)
            print("  {:28s} mean={:.10g} sd={:.10g} median={:.10g}".format(
                mode, mean, sd, statistics.median(values)
            ))

    print("\nPAIRED CF_INTERVENTION COMPARISONS (lower is better)")
    comparators = ("mappo", "cf_no_intervention", "cf_intervention_shuffled")
    overall = {mode: {"wins": 0, "losses": 0, "ties": 0, "ratios": []} for mode in comparators}
    for objective in OBJECTIVES:
        print(objective)
        target = [results[(objective, "cf_intervention", seed)] for seed in SEEDS]
        for comparator in comparators:
            baseline = [results[(objective, comparator, seed)] for seed in SEEDS]
            wins = sum(t < b for t, b in zip(target, baseline))
            losses = sum(t > b for t, b in zip(target, baseline))
            ties = len(target) - wins - losses
            ratios = [t / b for t, b in zip(target, baseline) if b > 0]
            overall[comparator]["wins"] += wins
            overall[comparator]["losses"] += losses
            overall[comparator]["ties"] += ties
            overall[comparator]["ratios"].extend(ratios)
            print("  vs {:25s} W-L-T={}-{}-{} median_ratio={:.6g} sign_p={:.6g}".format(
                comparator, wins, losses, ties, statistics.median(ratios), sign_test_p(wins, losses)
            ))

    print("overall across 20 function-seed blocks")
    for comparator in comparators:
        item = overall[comparator]
        print("  vs {:25s} W-L-T={}-{}-{} median_ratio={:.6g} sign_p={:.6g}".format(
            comparator,
            item["wins"],
            item["losses"],
            item["ties"],
            statistics.median(item["ratios"]),
            sign_test_p(item["wins"], item["losses"]),
        ))

    rank_totals = defaultdict(float)
    rank_count = 0
    for objective in OBJECTIVES:
        for seed in SEEDS:
            ranks = average_ranks({mode: results[(objective, mode, seed)] for mode in MODES})
            for mode, rank in ranks.items():
                rank_totals[mode] += rank
            rank_count += 1
    print("\nAVERAGE RANK ACROSS 20 FUNCTION-SEED BLOCKS")
    for mode in sorted(MODES, key=lambda item: rank_totals[item]):
        print("  {:28s} {:.4f}".format(mode, rank_totals[mode] / rank_count))

    print("\nTRAINING FUNCTION EVALUATION COST")
    for mode in MODES:
        values = cost_by_mode[mode]
        mean, sd = mean_sd(values)
        print("  {:28s} mean={:.1f} sd={:.1f} min={} max={}".format(
            mode, mean, sd, min(values), max(values)
        ))
    if shuffle_changes:
        print("  shuffled label changes per run: min={:.0f} median={:.0f} max={:.0f}".format(
            min(shuffle_changes), statistics.median(shuffle_changes), max(shuffle_changes)
        ))

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
