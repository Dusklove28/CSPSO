# CSPSO

Sparse-intervention credit assignment for particle swarm optimization, built on
a compact shared-parameter MAPPO training loop.

## Scope

This repository is focused on Credit Assignment PSO. The original MAPPO game
environments and unrelated algorithms were removed.

Kept components:

- constricted ring-neighborhood PSO environment;
- shared Actor MAPPO controller for choosing each particle's learning target;
- centralized value critic;
- counterfactual critic `Q_i(S, a_-i, k)`;
- sparse intervention branch runner;
- PSO rule baselines and result-summary utilities.

## Environment

```powershell
conda env create -f environment.yml
conda activate cspso
pip install -e .
```

If `cspso` already exists:

```powershell
conda activate cspso
pip install -r requirements.txt
pip install -e .
```

## Diagnostics

```powershell
python -m onpolicy.scripts.eval.diagnose_pso_interventions
```

Expected checks:

- same-action paired delta is zero;
- when `c2=0`, changing the leader has zero effect.

## Pilot Runs

The inherited flags `--use_wandb` and `--cuda` are reverse switches in this
codebase. In the commands below, they mean "disable wandb" and "use CPU".

```powershell
python -m onpolicy.scripts.train.train_pso --credit_mode mappo --num_env_steps 20000 --n_rollout_threads 1 --use_wandb --cuda
python -m onpolicy.scripts.train.train_pso --credit_mode cf_no_intervention --num_env_steps 20000 --n_rollout_threads 1 --use_wandb --cuda
python -m onpolicy.scripts.train.train_pso --credit_mode cf_intervention --num_env_steps 20000 --n_rollout_threads 1 --use_wandb --cuda
python -m onpolicy.scripts.train.train_pso --credit_mode cf_intervention_shuffled --num_env_steps 20000 --n_rollout_threads 1 --use_wandb --cuda
```

## Rule Baselines

```powershell
python -m onpolicy.scripts.eval.eval_pso_rules --rule ring_best
python -m onpolicy.scripts.eval.eval_pso_rules --rule random
```

## Result Summary

Local summaries are stored under:

```text
onpolicy/scripts/results/PSO/<objective>/<credit_mode>/<experiment_name>/run*/episode_metrics.jsonl
onpolicy/scripts/results/PSO/<objective>/<credit_mode>/<experiment_name>/run*/logs/summary.json
```

`episode_metrics.jsonl` is written every training episode and is the preferred
source for formal analysis. TensorBoard's `summary.json` is kept for scalar
visualization and as a fallback for older runs.

Summarize runs into CSV:

```powershell
python -m onpolicy.scripts.eval.summarize_pso_runs --results_dir onpolicy/scripts/results/PSO --output pso_summary.csv
```

Main analysis:

- final solution quality: lower `final_global_best` is better;
- training cost: compare `cumulative_extra_eval_ratio` and
  `cumulative_train_function_evaluations`;
- learning stability: compare median and interquartile range across seeds;
- credit mechanism value: compare `cf_intervention` against
  `cf_no_intervention` and `cf_intervention_shuffled`;
- shuffled-label sanity: in `cf_intervention_shuffled`,
  `max_cf_shuffle_label_changed` should become positive after the first stored
  intervention label is available;
- speed only: `fps` is not a research metric and only helps judge whether a run
  is unexpectedly slow.
