# Credit Assignment PSO

This fork adds a PSO environment and sparse counterfactual intervention
supervision on top of the original shared-parameter MAPPO code.

## Implemented First Version

- PSO backbone: constricted ring-neighborhood PSO.
- Particle count: default 20.
- Dimension: default 10.
- Episode length: default 100 generations.
- Update parameters: `chi=0.72984`, `c1=c2=2.05`.
- Action: each particle chooses `{left, self, right}` as the social-learning source.
- Reward: team improvement in the global best value,
  `(b_t - b_{t+1}) / s`.
- Training return: fixed budget with `gamma=1` and no GAE, so the return is
  the normalized final improvement.
- Actor: shared MAPPO actor, default hidden size 64.
- Critic: centralized value critic, default hidden size 128.
- Counterfactual critic: `Q_i(S_t, a_-i, k)`.

The intervention changes one particle's leader choice at one saved generation.
The real branch reuses the collected trajectory. The alternative branch restores
the saved PSO state, changes only that one action, then continues with the same
frozen policy and paired random states until the same budget endpoint.

## Training Modes

Run from the repository root:

```bash
python -m onpolicy.scripts.train.train_pso --credit_mode mappo --use_wandb
```

Available `credit_mode` values:

- `mappo`: ordinary MAPPO with team reward.
- `cf_no_intervention`: same counterfactual critic and advantage form, but only
  ordinary return supervision.
- `cf_intervention`: sparse intervention difference supervision.
- `cf_intervention_shuffled`: same as intervention mode, but shuffled
  intervention labels as a falsification control.

The flag name `--use_wandb` is inherited from the original repository and
actually disables wandb logging because it uses `store_false`.

## Small Smoke Run

This is only for checking that the pipeline works:

```bash
python -m onpolicy.scripts.train.train_pso ^
  --credit_mode cf_intervention ^
  --pso_particles 4 ^
  --pso_dim 2 ^
  --pso_generations 5 ^
  --num_env_steps 10 ^
  --n_rollout_threads 1 ^
  --ppo_epoch 1 ^
  --num_mini_batch 1 ^
  --use_wandb
```

## Required Controls

Rule baselines:

```bash
python -m onpolicy.scripts.eval.eval_pso_rules --rule ring_best
python -m onpolicy.scripts.eval.eval_pso_rules --rule random
```

MAPPO controls:

```bash
python -m onpolicy.scripts.train.train_pso --credit_mode mappo --use_wandb
python -m onpolicy.scripts.train.train_pso --credit_mode cf_no_intervention --use_wandb
python -m onpolicy.scripts.train.train_pso --credit_mode cf_intervention --use_wandb
python -m onpolicy.scripts.train.train_pso --credit_mode cf_intervention_shuffled --use_wandb
```

Same-total-budget controls should be run by increasing `--num_env_steps` for
`mappo` or `cf_no_intervention` to match the extra function evaluations logged
by `extra_function_evaluations` in intervention mode.

## Intervention Diagnostics

Before long training, check that snapshot and paired-branch mechanics are sane:

```bash
python -m onpolicy.scripts.eval.diagnose_pso_interventions
```

Expected output:

- same-action paired delta is zero;
- with `c2=0`, changing the leader has zero effect under the deterministic
  ring-best continuation.

These checks do not prove the research claim; they only validate the
intervention toolchain.

## Logged Quantities

Training logs include:

- `final_global_best`;
- `function_evaluations`;
- `extra_function_evaluations`;
- `extra_eval_ratio`;
- `intervention_delta`;
- `cf_ordinary_loss`;
- `cf_intervention_loss`;
- ordinary MAPPO losses.

The main claim should be evaluated by both search quality and attribution
accuracy. High predicted contribution decisions should produce larger losses
when replaced in independent repeated interventions.
