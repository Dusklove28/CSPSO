param(
    [string]$RunDate = (Get-Date -Format "yyyy-MM-dd"),
    [int[]]$Seeds = (1..5),
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ($env:CONDA_DEFAULT_ENV -ne "cspso") {
    throw "Activate the cspso conda environment before starting the formal batch."
}

$GitChanges = @(git status --porcelain --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read Git status."
}
if ($GitChanges.Count -gt 0) {
    throw "Formal experiments require a clean Git worktree. Commit or remove pending changes first."
}

$Commit = (git rev-parse HEAD).Trim()
$ResultsRoot = Join-Path $RepoRoot (Join-Path "runs" $RunDate)
New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null

$Protocol = [ordered]@{
    schema_version = 1
    run_date = $RunDate
    git_commit = $Commit
    objectives = @("sphere", "rastrigin", "rosenbrock", "ackley")
    credit_modes = @("mappo", "cf_no_intervention", "cf_intervention", "cf_intervention_shuffled")
    seeds = $Seeds
    particles = 20
    dimensions = 10
    generations_per_episode = 100
    environment_steps = 20000
    training_episodes = 200
    evaluation_interval_episodes = 10
    evaluation_episodes_per_checkpoint = 10
    model_selection = "final_checkpoint"
    device = "cpu"
}
$Protocol | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $ResultsRoot "protocol.json")

if (-not $SkipPreflight) {
    & python -m onpolicy.scripts.eval.preflight_pso
    if ($LASTEXITCODE -ne 0) {
        throw "PSO preflight failed; formal experiments were not started."
    }
}

$Objectives = @("sphere", "rastrigin", "rosenbrock", "ackley")
$Modes = @("mappo", "cf_no_intervention", "cf_intervention", "cf_intervention_shuffled")

foreach ($Objective in $Objectives) {
    foreach ($Mode in $Modes) {
        foreach ($Seed in $Seeds) {
            $Experiment = "formal_seed$Seed"
            $ExperimentDir = Join-Path $ResultsRoot (Join-Path "PSO" (Join-Path $Objective (Join-Path $Mode $Experiment)))
            $Completed = $false
            if (Test-Path $ExperimentDir) {
                foreach ($ManifestFile in Get-ChildItem -Path $ExperimentDir -Filter "run_manifest.json" -Recurse) {
                    $Manifest = Get-Content -Raw $ManifestFile.FullName | ConvertFrom-Json
                    if ($Manifest.status -eq "completed" -and $Manifest.git.commit -eq $Commit) {
                        $Completed = $true
                        break
                    }
                }
            }
            if ($Completed) {
                Write-Host "SKIP completed: objective=$Objective mode=$Mode seed=$Seed"
                continue
            }

            Write-Host "START objective=$Objective mode=$Mode seed=$Seed"
            $TrainArgs = @(
                "-m", "onpolicy.scripts.train.train_pso",
                "--results_dir", $ResultsRoot,
                "--experiment_name", $Experiment,
                "--pso_objective", $Objective,
                "--credit_mode", $Mode,
                "--seed", $Seed,
                "--pso_particles", "20",
                "--pso_dim", "10",
                "--pso_generations", "100",
                "--pso_chi", "0.72984",
                "--pso_c1", "2.05",
                "--pso_c2", "2.05",
                "--pso_boundary", "clip",
                "--pso_intervention_interval", "4",
                "--pso_interventions_per_event", "1",
                "--num_env_steps", "20000",
                "--n_rollout_threads", "1",
                "--n_eval_rollout_threads", "1",
                "--n_training_threads", "1",
                "--ppo_epoch", "15",
                "--num_mini_batch", "1",
                "--cf_epoch", "1",
                "--use_eval",
                "--eval_interval", "10",
                "--eval_episodes", "10",
                "--save_interval", "200",
                "--log_interval", "10",
                "--use_wandb",
                "--cuda"
            )
            & python @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed: objective=$Objective mode=$Mode seed=$Seed"
            }
        }
    }
}

$SummaryPath = Join-Path $ResultsRoot "pso_summary.csv"
& python -m onpolicy.scripts.eval.summarize_pso_runs `
    --results_dir (Join-Path $ResultsRoot "PSO") `
    --output $SummaryPath
if ($LASTEXITCODE -ne 0) {
    throw "Summary generation failed."
}

Write-Host "Formal primary batch completed."
Write-Host "Results: $ResultsRoot"
Write-Host "Summary: $SummaryPath"
