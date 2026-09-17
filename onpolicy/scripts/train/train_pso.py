#!/usr/bin/env python
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from onpolicy.envs.pso import RingTopologyPSOEnv

try:
    import setproctitle
except ImportError:
    setproctitle = None

try:
    import wandb
except ImportError:
    wandb = None


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_metadata(repo_root):
    def run_git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _write_manifest(path, all_args, device, status, started_at, error=None):
    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "updated_at": _timestamp(),
        "command": [sys.executable, "-m", "onpolicy.scripts.train.train_pso", *sys.argv[1:]],
        "config": vars(all_args),
        "git": _git_metadata(repo_root),
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if error is not None:
        manifest["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = RingTopologyPSOEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = RingTopologyPSOEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    all_args.env_name = "PSO"
    all_args.scenario_name = all_args.pso_objective
    all_args.num_agents = all_args.pso_particles
    all_args.episode_length = all_args.pso_generations
    return all_args


def apply_pso_defaults(all_args):
    if all_args.algorithm_name != "mappo":
        raise NotImplementedError("PSO experiments use shared-parameter MAPPO.")
    all_args.use_recurrent_policy = False
    all_args.use_naive_recurrent_policy = False
    all_args.use_centralized_V = True
    all_args.share_policy = True
    all_args.gamma = 1.0
    all_args.use_gae = False
    all_args.lr = 3e-4
    all_args.critic_lr = 3e-4
    all_args.clip_param = 0.2
    all_args.hidden_size = 64
    if all_args.critic_hidden_size is None:
        all_args.critic_hidden_size = 128

    all_args.use_counterfactual_credit = False
    all_args.use_cf_advantage = False
    all_args.use_cf_intervention_loss = False
    all_args.cf_shuffle_labels = False
    if all_args.credit_mode == "cf_no_intervention":
        all_args.use_counterfactual_credit = True
        all_args.use_cf_advantage = True
    elif all_args.credit_mode == "cf_intervention":
        all_args.use_counterfactual_credit = True
        all_args.use_cf_advantage = True
        all_args.use_cf_intervention_loss = True
    elif all_args.credit_mode == "cf_intervention_shuffled":
        all_args.use_counterfactual_credit = True
        all_args.use_cf_advantage = True
        all_args.use_cf_intervention_loss = True
        all_args.cf_shuffle_labels = True

    if all_args.use_cf_intervention_loss and all_args.n_rollout_threads != 1:
        raise ValueError("Use --n_rollout_threads 1 for sparse intervention training.")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    apply_pso_defaults(all_args)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    results_root = (
        Path(all_args.results_dir).expanduser()
        if all_args.results_dir is not None
        else Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
    )
    run_dir = (
        results_root
        / all_args.env_name
        / all_args.pso_objective
        / all_args.credit_mode
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Disable wandb logging with the inherited --use_wandb flag.")
        run = wandb.init(
            config=all_args,
            project="Credit-Assignment-PSO",
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name="{}_{}_seed{}".format(all_args.credit_mode, all_args.experiment_name, all_args.seed),
            group=all_args.pso_objective,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            curr_run = "run%i" % (max(exst_run_nums) + 1) if len(exst_run_nums) > 0 else "run1"
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    artifact_dir = Path(wandb.run.dir) if all_args.use_wandb else run_dir
    manifest_path = artifact_dir / "run_manifest.json"
    started_at = _timestamp()
    _write_manifest(manifest_path, all_args, device, "running", started_at)

    if setproctitle is not None:
        setproctitle.setproctitle(
            "{}-{}-{}@{}".format(
                all_args.algorithm_name,
                all_args.env_name,
                all_args.experiment_name,
                all_args.user_name,
            )
        )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from onpolicy.runner.shared.pso_runner import PSORunner as Runner

    runner = None
    try:
        runner = Runner(config)
        runner.run()
    except Exception as exc:
        _write_manifest(manifest_path, all_args, device, "failed", started_at, repr(exc))
        raise
    else:
        _write_manifest(manifest_path, all_args, device, "completed", started_at)
    finally:
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()

        if all_args.use_wandb:
            run.finish()
        elif runner is not None:
            runner.writter.flush()
            runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
