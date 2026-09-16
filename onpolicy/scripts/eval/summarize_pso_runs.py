#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="onpolicy/scripts/results/PSO")
    parser.add_argument("--output", type=str, default="pso_summary.csv")
    return parser.parse_args()


def scalar_series(summary, key):
    rows = summary.get(key, [])
    values = []
    for row in rows:
        payload = row.get("values", {})
        if key in payload:
            values.append((row.get("step"), payload[key]))
    return values


def last_value(summary, key):
    values = scalar_series(summary, key)
    if not values:
        return ""
    return values[-1][1]


def best_value(summary, key):
    values = [x[1] for x in scalar_series(summary, key)]
    if not values:
        return ""
    return min(values)


def summarize_run(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        print("skip invalid or incomplete summary: {}".format(path))
        return None

    parts = path.parts
    try:
        pso_index = parts.index("PSO")
        objective = parts[pso_index + 1]
        credit_mode = parts[pso_index + 2]
        experiment = parts[pso_index + 3]
        run = parts[pso_index + 4]
    except (ValueError, IndexError):
        objective = credit_mode = experiment = run = ""

    return {
        "summary_path": str(path),
        "objective": objective,
        "credit_mode": credit_mode,
        "experiment": experiment,
        "run": run,
        "last_final_global_best": last_value(summary, "final_global_best"),
        "best_logged_global_best": best_value(summary, "final_global_best"),
        "last_extra_eval_ratio": last_value(summary, "extra_eval_ratio"),
        "last_function_evaluations": last_value(summary, "function_evaluations"),
        "last_extra_function_evaluations": last_value(summary, "extra_function_evaluations"),
        "last_cf_ordinary_loss": last_value(summary, "cf_ordinary_loss"),
        "last_cf_intervention_loss": last_value(summary, "cf_intervention_loss"),
        "last_cf_intervention_count": last_value(summary, "cf_intervention_count"),
        "last_fps": last_value(summary, "fps"),
    }


def main():
    args = parse_args()
    root = Path(args.results_dir)
    rows = [summarize_run(path) for path in root.rglob("summary.json")]
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda x: (x["objective"], x["credit_mode"], x["experiment"], x["run"]))

    fieldnames = [
        "objective",
        "credit_mode",
        "experiment",
        "run",
        "last_final_global_best",
        "best_logged_global_best",
        "last_extra_eval_ratio",
        "last_function_evaluations",
        "last_extra_function_evaluations",
        "last_cf_ordinary_loss",
        "last_cf_intervention_loss",
        "last_cf_intervention_count",
        "last_fps",
        "summary_path",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} runs to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
