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


def run_metadata(path):
    parts = path.parts
    try:
        pso_index = parts.index("PSO")
        return {
            "objective": parts[pso_index + 1],
            "credit_mode": parts[pso_index + 2],
            "experiment": parts[pso_index + 3],
            "run": parts[pso_index + 4],
        }
    except (ValueError, IndexError):
        return {"objective": "", "credit_mode": "", "experiment": "", "run": ""}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tag_matches(tag, key):
    tag = str(tag).replace("\\", "/")
    return tag == key or tag.endswith("/" + key) or tag.endswith("/" + key + "/" + key)


def scalar_series(summary, key):
    values = []
    for tag, rows in summary.items():
        if not tag_matches(tag, key):
            continue
        for row in rows:
            if isinstance(row, dict):
                payload = row.get("values", {})
                if key in payload:
                    values.append((row.get("step"), payload[key]))
            elif isinstance(row, (list, tuple)) and len(row) >= 3:
                values.append((row[1], row[2]))
    return values


def last_series_value(summary, key):
    values = scalar_series(summary, key)
    if not values:
        return ""
    return values[-1][1]


def best_series_value(summary, key):
    values = [to_float(x[1]) for x in scalar_series(summary, key)]
    values = [x for x in values if x is not None]
    if not values:
        return ""
    return min(values)


def max_series_value(summary, key):
    values = [to_float(x[1]) for x in scalar_series(summary, key)]
    values = [x for x in values if x is not None]
    if not values:
        return ""
    return max(values)


def sum_series_value(summary, key):
    values = [to_float(x[1]) for x in scalar_series(summary, key)]
    values = [x for x in values if x is not None]
    if not values:
        return ""
    return sum(values)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print("skip invalid metrics line in {}".format(path))
    return rows


def last_record_value(records, key):
    for row in reversed(records):
        value = row.get(key, "")
        if value != "":
            return value
    return ""


def best_record_value(records, key):
    values = [to_float(row.get(key, "")) for row in records]
    values = [x for x in values if x is not None]
    if not values:
        return ""
    return min(values)


def max_record_value(records, key):
    values = [to_float(row.get(key, "")) for row in records]
    values = [x for x in values if x is not None]
    if not values:
        return ""
    return max(values)


def sum_record_value(records, key):
    values = [to_float(row.get(key, "")) for row in records]
    values = [x for x in values if x is not None]
    if not values:
        return ""
    return sum(values)


def summarize_jsonl(path):
    records = read_jsonl(path)
    if not records:
        return None
    row = run_metadata(path)
    row.update(
        {
            "summary_path": str(path),
            "last_final_global_best": last_record_value(records, "final_global_best"),
            "best_logged_global_best": best_record_value(records, "final_global_best"),
            "last_eval_final_global_best": last_record_value(records, "eval_final_global_best"),
            "last_eval_final_global_best_values": json.dumps(
                last_record_value(records, "eval_final_global_best_values"),
                ensure_ascii=True,
            ),
            "last_extra_eval_ratio": last_record_value(records, "extra_eval_ratio"),
            "last_cumulative_extra_eval_ratio": last_record_value(records, "cumulative_extra_eval_ratio"),
            "last_function_evaluations": last_record_value(records, "function_evaluations"),
            "last_extra_function_evaluations": last_record_value(records, "extra_function_evaluations"),
            "last_cumulative_main_function_evaluations": last_record_value(
                records, "cumulative_main_function_evaluations"
            ),
            "last_cumulative_intervention_function_evaluations": last_record_value(
                records, "cumulative_intervention_function_evaluations"
            ),
            "last_cumulative_eval_function_evaluations": last_record_value(
                records, "cumulative_eval_function_evaluations"
            ),
            "last_cumulative_train_function_evaluations": last_record_value(
                records, "cumulative_train_function_evaluations"
            ),
            "last_cumulative_total_function_evaluations": last_record_value(
                records, "cumulative_total_function_evaluations"
            ),
            "last_cf_ordinary_loss": last_record_value(records, "cf_ordinary_loss"),
            "last_cf_intervention_loss": last_record_value(records, "cf_intervention_loss"),
            "last_cf_intervention_count": last_record_value(records, "cf_intervention_count"),
            "total_intervention_count": sum_record_value(records, "intervention_count"),
            "last_cf_shuffle_label_changed": last_record_value(records, "cf_shuffle_label_changed"),
            "max_cf_shuffle_label_changed": max_record_value(records, "cf_shuffle_label_changed"),
            "last_fps": last_record_value(records, "fps"),
        }
    )
    return row


def summarize_summary_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        print("skip invalid or incomplete summary: {}".format(path))
        return None

    row = run_metadata(path)
    row.update(
        {
            "summary_path": str(path),
            "last_final_global_best": last_series_value(summary, "final_global_best"),
            "best_logged_global_best": best_series_value(summary, "final_global_best"),
            "last_eval_final_global_best": last_series_value(summary, "eval_final_global_best"),
            "last_eval_final_global_best_values": "",
            "last_extra_eval_ratio": last_series_value(summary, "extra_eval_ratio"),
            "last_cumulative_extra_eval_ratio": last_series_value(summary, "cumulative_extra_eval_ratio"),
            "last_function_evaluations": last_series_value(summary, "function_evaluations"),
            "last_extra_function_evaluations": last_series_value(summary, "extra_function_evaluations"),
            "last_cumulative_main_function_evaluations": last_series_value(
                summary, "cumulative_main_function_evaluations"
            ),
            "last_cumulative_intervention_function_evaluations": last_series_value(
                summary, "cumulative_intervention_function_evaluations"
            ),
            "last_cumulative_eval_function_evaluations": last_series_value(
                summary, "cumulative_eval_function_evaluations"
            ),
            "last_cumulative_train_function_evaluations": last_series_value(
                summary, "cumulative_train_function_evaluations"
            ),
            "last_cumulative_total_function_evaluations": last_series_value(
                summary, "cumulative_total_function_evaluations"
            ),
            "last_cf_ordinary_loss": last_series_value(summary, "cf_ordinary_loss"),
            "last_cf_intervention_loss": last_series_value(summary, "cf_intervention_loss"),
            "last_cf_intervention_count": last_series_value(summary, "cf_intervention_count"),
            "total_intervention_count": sum_series_value(summary, "cf_intervention_count"),
            "last_cf_shuffle_label_changed": last_series_value(summary, "cf_shuffle_label_changed"),
            "max_cf_shuffle_label_changed": max_series_value(summary, "cf_shuffle_label_changed"),
            "last_fps": last_series_value(summary, "fps"),
        }
    )
    return row


def main():
    args = parse_args()
    root = Path(args.results_dir)
    metric_paths = list(root.rglob("episode_metrics.jsonl"))
    metric_run_dirs = {path.parent for path in metric_paths}

    rows = [summarize_jsonl(path) for path in metric_paths]
    for path in root.rglob("summary.json"):
        run_dir = path.parent.parent if path.parent.name == "logs" else path.parent
        if run_dir in metric_run_dirs:
            continue
        rows.append(summarize_summary_json(path))

    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda x: (x["objective"], x["credit_mode"], x["experiment"], x["run"]))

    fieldnames = [
        "objective",
        "credit_mode",
        "experiment",
        "run",
        "last_final_global_best",
        "best_logged_global_best",
        "last_eval_final_global_best",
        "last_eval_final_global_best_values",
        "last_extra_eval_ratio",
        "last_cumulative_extra_eval_ratio",
        "last_function_evaluations",
        "last_extra_function_evaluations",
        "last_cumulative_main_function_evaluations",
        "last_cumulative_intervention_function_evaluations",
        "last_cumulative_eval_function_evaluations",
        "last_cumulative_train_function_evaluations",
        "last_cumulative_total_function_evaluations",
        "last_cf_ordinary_loss",
        "last_cf_intervention_loss",
        "last_cf_intervention_count",
        "total_intervention_count",
        "last_cf_shuffle_label_changed",
        "max_cf_shuffle_label_changed",
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
