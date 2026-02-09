#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GR00T Open-Loop Inference Evaluation Script

Orchestrates open-loop inference evaluation of GR00T policies by executing
inference inside the groot-thor Docker container, collecting MSE/MAE metrics,
and computing statistics.

Two-layer design:
  - Host-side orchestrator (this script) - Manages Docker execution, collects results
  - Docker-side inference - Python code executed via docker exec

Usage:
    python scripts/real_robot/eval_groot_openloop.py \
        --checkpoint /home/zechenli/hf_checkpoint/checkpoint-5000_bu \
        --dataset /home/zechenli/my_lerobot50 \
        --num-runs 10 \
        --num-trajectories 5 \
        --seed 42 \
        --output-dir ~/Desktop/ResultsTill/eval_run1 \
        --start-step 200 \
        --end-step 800
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class InferenceConfig:
    """Configuration for a single inference run."""
    checkpoint: str
    dataset: str
    trajectory_id: int
    run_id: int
    start_step: int
    end_step: int
    action_horizon: int
    seed: Optional[int]
    save_plot: bool
    plot_save_path: Optional[str]


@dataclass
class InferenceResult:
    """Result from a single inference run."""
    trajectory_id: int
    run_id: int
    mse: float
    mae: float
    success: bool
    timestamp: str
    error_message: Optional[str] = None


@dataclass
class TrajectoryStats:
    """Statistics for a single trajectory."""
    trajectory_id: int
    num_runs: int
    mse_mean: float
    mse_median: float
    mse_std: float
    mae_mean: float
    mae_median: float
    mae_std: float


class DockerInferenceRunner:
    """Runs GR00T inference inside Docker container."""

    def __init__(self, container: str = "groot-thor", timeout: int = 300, max_retries: int = 2, verbose: bool = False):
        self.container = container
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose

    def check_container_running(self) -> bool:
        """Verify Docker container is active."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def run_single_inference(self, config: InferenceConfig) -> InferenceResult:
        """Execute one inference run via docker exec."""
        timestamp = datetime.now().isoformat()

        for attempt in range(self.max_retries + 1):
            try:
                code = self._build_inference_code(config)
                cmd = [
                    "docker", "exec", "-i", self.container,
                    "/bin/bash", "-c",
                    f'cd /workspace/Isaac-GR00T && /opt/venv/bin/python3 -c "{code}"'
                ]

                if self.verbose:
                    print(f"  [VERBOSE] Running docker exec command...")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                if self.verbose:
                    print(f"  [VERBOSE] Return code: {result.returncode}")
                    if result.stdout:
                        print(f"  [VERBOSE] STDOUT:\n{result.stdout}")
                    if result.stderr:
                        print(f"  [VERBOSE] STDERR:\n{result.stderr}")

                if result.returncode != 0:
                    if attempt < self.max_retries:
                        print(f"  [RETRY] Attempt {attempt + 1} failed, retrying...")
                        continue
                    # Combine stdout and stderr for better error reporting
                    error_parts = []
                    if result.stderr:
                        error_parts.append(f"STDERR:\n{result.stderr}")
                    if result.stdout:
                        error_parts.append(f"STDOUT:\n{result.stdout}")
                    error_msg = "\n".join(error_parts) if error_parts else "Unknown error"
                    return InferenceResult(
                        trajectory_id=config.trajectory_id,
                        run_id=config.run_id,
                        mse=0.0,
                        mae=0.0,
                        success=False,
                        timestamp=timestamp,
                        error_message=error_msg,
                    )

                mse, mae = self._parse_output(result.stdout)
                return InferenceResult(
                    trajectory_id=config.trajectory_id,
                    run_id=config.run_id,
                    mse=mse,
                    mae=mae,
                    success=True,
                    timestamp=timestamp,
                )

            except subprocess.TimeoutExpired:
                if attempt < self.max_retries:
                    print(f"  [RETRY] Attempt {attempt + 1} timed out, retrying...")
                    continue
                return InferenceResult(
                    trajectory_id=config.trajectory_id,
                    run_id=config.run_id,
                    mse=0.0,
                    mae=0.0,
                    success=False,
                    timestamp=timestamp,
                    error_message=f"Timeout after {self.timeout}s",
                )
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  [RETRY] Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                return InferenceResult(
                    trajectory_id=config.trajectory_id,
                    run_id=config.run_id,
                    mse=0.0,
                    mae=0.0,
                    success=False,
                    timestamp=timestamp,
                    error_message=str(e)[:500],
                )

        # Should not reach here, but just in case
        return InferenceResult(
            trajectory_id=config.trajectory_id,
            run_id=config.run_id,
            mse=0.0,
            mae=0.0,
            success=False,
            timestamp=timestamp,
            error_message="Max retries exceeded",
        )

    def _build_inference_code(self, config: InferenceConfig) -> str:
        """Generate Python code for Docker execution."""
        # Escape single quotes for bash
        seed_setup = ""
        if config.seed is not None:
            seed_setup = f"""
import random
import torch
np.random.seed({config.seed} + {config.trajectory_id} * 1000 + {config.run_id})
random.seed({config.seed} + {config.trajectory_id} * 1000 + {config.run_id})
torch.manual_seed({config.seed} + {config.trajectory_id} * 1000 + {config.run_id})
if torch.cuda.is_available():
    torch.cuda.manual_seed_all({config.seed} + {config.trajectory_id} * 1000 + {config.run_id})
"""

        plot_code = ""
        if config.save_plot and config.plot_save_path:
            plot_code = f"""
# Plot
action_dim = gt_action_across_time.shape[1]
fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(12, 3 * action_dim))
if action_dim == 1:
    axes = [axes]

fig.suptitle(f'Trajectory {config.trajectory_id} - Steps {config.start_step}-{{actual_end}} | MSE: {{mse:.4f}} | MAE: {{mae:.4f}}', fontsize=14)

x_range = np.arange({config.start_step}, {config.start_step} + len(gt_action_across_time))

keyz=['left_base_joint', 'left_shoulder_joint', 'left_elbow_joint', 'left_wrist1_joint', 'left_wrist2_joint', 'left_wrist3_joint','left_gripper', 'left_gripper_force', 'left_current_limit', 'right_base_joint', 'right_shoulder_joint', 'right_elbow_joint', 'right_wrist1_joint', 'right_wrist2_joint', 'right_wrist3_joint', 'right_gripper', 'right_gripper_force', 'right_current_limit']

for idx in range(action_dim):
    ax = axes[idx]
    ax.plot(x_range, gt_action_across_time[:, idx], label='gt action', linewidth=1.5)
    ax.plot(x_range, pred_action_across_time[:, idx], label='pred action', linewidth=1.5, alpha=0.8)

    # Mark inference points
    for j in range({config.start_step}, {config.start_step} + len(gt_action_across_time), {config.action_horizon}):
        rel_idx = j - {config.start_step}
        if 0 <= rel_idx < len(gt_action_across_time):
            ax.plot(j, gt_action_across_time[rel_idx, idx], 'ro', markersize=4, label='inference point' if j == {config.start_step} else '')

    ax.set_title(f'{{keyz[idx]}}' if idx < len(keyz) else f'Action {{idx}}')
    ax.legend(loc='upper right')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('{config.plot_save_path}', dpi=150)
plt.close()
print(f'Plot saved to {config.plot_save_path}')
"""

        code = f"""
import numpy as np
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
{seed_setup}
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

# Config
dataset_path = '{config.dataset}'
model_path = '{config.checkpoint}'
traj_id = {config.trajectory_id}
action_horizon = {config.action_horizon}
start_step = {config.start_step}
end_step = {config.end_step}

print('Loading model...')
policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path=model_path,
    device='cuda'
)

modality = policy.get_modality_config()
dataset = LeRobotEpisodeLoader(
    dataset_path=dataset_path,
    modality_configs=modality,
    video_backend='torchcodec',
    video_backend_kwargs=None,
)

print(f'Dataset length: {{len(dataset)}}')
traj = dataset[traj_id]
traj_length = len(traj)
actual_end = min(end_step, traj_length)
print(f'Running steps {{start_step}}-{{actual_end}} (traj length: {{traj_length}})')

action_keys = modality['action'].modality_keys
print(action_keys)

modality_configs = deepcopy(modality)
modality_configs.pop('action')

pred_action_across_time = []

# Run inference only from start_step to end_step
for step_count in range(start_step, actual_end, action_horizon):
    print(f'Inferencing at step: {{step_count}}')
    data_point = extract_step_data(traj, step_count, modality_configs, EmbodimentTag.NEW_EMBODIMENT)
    obs = {{}}
    for k, v in data_point.states.items():
        obs[f'state.{{k}}'] = v
    for k, v in data_point.images.items():
        obs[f'video.{{k}}'] = np.array(v)
    for language_key in modality['language'].modality_keys:
        obs[language_key] = data_point.text

    new_obs = {{}}
    for mod in ['video', 'state', 'language']:
        new_obs[mod] = {{}}
        for key in modality[mod].modality_keys:
            if mod == 'language':
                parsed_key = key
            else:
                parsed_key = f'{{mod}}.{{key}}'
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[mod][key] = [[arr]]
            else:
                new_obs[mod][key] = arr[None, :]

    _action_chunk, _ = policy.get_action(new_obs)
    action_chunk = {{f'action.{{key}}': _action_chunk[key][0] for key in _action_chunk}}

    for j in range(action_horizon):
        if step_count + j < actual_end:
            concat_pred_action = np.concatenate(
                [np.atleast_1d(np.atleast_1d(action_chunk[f'action.{{key}}'])[j]) for key in action_keys],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

# Extract ground truth for the range
def extract_state_joints(traj, columns):
    np_dict = {{}}
    for column in columns:
        np_dict[column] = np.vstack([arr for arr in traj[column]])
    return np.concatenate([np_dict[column] for column in columns], axis=-1)

gt_action_full = extract_state_joints(traj, [f'action.{{key}}' for key in action_keys])
gt_action_across_time = gt_action_full[start_step:actual_end]
pred_action_across_time = np.array(pred_action_across_time)[:len(gt_action_across_time)]

# Calculate metrics
mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
print(f'RESULT:MSE={{mse}},MAE={{mae}}')
{plot_code}
"""
        # Escape for bash -c "..."
        code = code.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
        return code

    def _parse_output(self, stdout: str) -> tuple:
        """Extract MSE/MAE from stdout."""
        # Look for RESULT:MSE=...,MAE=...
        match = re.search(r'RESULT:MSE=([0-9.e+-]+),MAE=([0-9.e+-]+)', stdout)
        if match:
            mse = float(match.group(1))
            mae = float(match.group(2))
            return mse, mae
        raise ValueError(f"Could not parse MSE/MAE from output: {stdout[-500:]}")


class ResultsCollector:
    """Collects results and computes statistics."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "results.csv"
        self.results: List[InferenceResult] = []

        # Initialize CSV file with header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trajectory_id', 'run_id', 'mse', 'mae', 'success', 'timestamp'])

    def add_result(self, result: InferenceResult):
        """Append result to CSV immediately (crash-safe)."""
        self.results.append(result)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.trajectory_id,
                result.run_id,
                result.mse,
                result.mae,
                result.success,
                result.timestamp,
            ])

    def compute_statistics(self) -> tuple:
        """Calculate mean/median/std per trajectory and overall."""
        import numpy as np

        # Group by trajectory
        traj_results: Dict[int, List[InferenceResult]] = {}
        for r in self.results:
            if r.success:
                if r.trajectory_id not in traj_results:
                    traj_results[r.trajectory_id] = []
                traj_results[r.trajectory_id].append(r)

        # Per-trajectory stats
        traj_stats: List[TrajectoryStats] = []
        for traj_id in sorted(traj_results.keys()):
            results = traj_results[traj_id]
            mse_values = np.array([r.mse for r in results])
            mae_values = np.array([r.mae for r in results])

            traj_stats.append(TrajectoryStats(
                trajectory_id=traj_id,
                num_runs=len(results),
                mse_mean=float(np.mean(mse_values)),
                mse_median=float(np.median(mse_values)),
                mse_std=float(np.std(mse_values)),
                mae_mean=float(np.mean(mae_values)),
                mae_median=float(np.median(mae_values)),
                mae_std=float(np.std(mae_values)),
            ))

        # Overall stats
        all_successful = [r for r in self.results if r.success]
        all_failed = [r for r in self.results if not r.success]

        if all_successful:
            all_mse = np.array([r.mse for r in all_successful])
            all_mae = np.array([r.mae for r in all_successful])
            overall_stats = {
                'total_successful': len(all_successful),
                'total_failed': len(all_failed),
                'mse_mean': float(np.mean(all_mse)),
                'mse_median': float(np.median(all_mse)),
                'mse_std': float(np.std(all_mse)),
                'mae_mean': float(np.mean(all_mae)),
                'mae_median': float(np.median(all_mae)),
                'mae_std': float(np.std(all_mae)),
            }
        else:
            overall_stats = {
                'total_successful': 0,
                'total_failed': len(all_failed),
                'mse_mean': 0.0,
                'mse_median': 0.0,
                'mse_std': 0.0,
                'mae_mean': 0.0,
                'mae_median': 0.0,
                'mae_std': 0.0,
            }

        return traj_stats, overall_stats

    def print_summary(self, traj_stats: List[TrajectoryStats], overall_stats: dict):
        """Format and print summary to terminal."""
        print()
        print("=" * 70)
        print("GR00T Open-Loop Inference Evaluation Results")
        print("=" * 70)
        print()
        print("Per-Trajectory Statistics:")
        print("-" * 70)
        print(f"{'Traj':<6}{'Runs':<6}{'MSE Mean':<12}{'MSE Median':<12}{'MSE Std':<12}{'MAE Mean':<12}{'MAE Std':<10}")
        print("-" * 70)

        for ts in traj_stats:
            print(f"{ts.trajectory_id:<6}{ts.num_runs:<6}{ts.mse_mean:<12.6f}{ts.mse_median:<12.6f}"
                  f"{ts.mse_std:<12.6f}{ts.mae_mean:<12.6f}{ts.mae_std:<10.6f}")

        print()
        print("=" * 70)
        print("Overall Statistics:")
        print(f"  Total successful runs: {overall_stats['total_successful']}")
        print(f"  Total failed runs:     {overall_stats['total_failed']}")
        print()
        print(f"  MSE - Mean: {overall_stats['mse_mean']:.6f}, "
              f"Median: {overall_stats['mse_median']:.6f}, "
              f"Std: {overall_stats['mse_std']:.6f}")
        print(f"  MAE - Mean: {overall_stats['mae_mean']:.6f}, "
              f"Median: {overall_stats['mae_median']:.6f}, "
              f"Std: {overall_stats['mae_std']:.6f}")
        print("=" * 70)

    def save_statistics(self, traj_stats: List[TrajectoryStats], overall_stats: dict, args: argparse.Namespace):
        """Save statistics to JSON file."""
        stats_data = {
            'config': {
                'checkpoint': args.checkpoint,
                'dataset': args.dataset,
                'num_runs': args.num_runs,
                'num_trajectories': args.num_trajectories,
                'seed': args.seed,
                'start_step': args.start_step,
                'end_step': args.end_step,
                'action_horizon': args.action_horizon,
            },
            'per_trajectory': [
                {
                    'trajectory_id': ts.trajectory_id,
                    'num_runs': ts.num_runs,
                    'mse_mean': ts.mse_mean,
                    'mse_median': ts.mse_median,
                    'mse_std': ts.mse_std,
                    'mae_mean': ts.mae_mean,
                    'mae_median': ts.mae_median,
                    'mae_std': ts.mae_std,
                }
                for ts in traj_stats
            ],
            'overall': overall_stats,
            'timestamp': datetime.now().isoformat(),
        }

        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GR00T Open-Loop Inference Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path (inside Docker)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset path (inside Docker)")

    # Evaluation parameters
    parser.add_argument("--num-runs", "-X", type=int, default=10,
                        help="Inference runs per trajectory")
    parser.add_argument("--num-trajectories", "-Y", type=int, default=5,
                        help="Number of trajectories to evaluate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./groot_eval_results",
                        help="Results directory (host-side)")
    parser.add_argument("--docker-output-dir", type=str, default="/home/zechenli/ResultsTill",
                        help="Plot save directory (inside Docker)")

    # Inference parameters
    parser.add_argument("--start-step", type=int, default=200,
                        help="Starting step for inference")
    parser.add_argument("--end-step", type=int, default=800,
                        help="Ending step for inference")
    parser.add_argument("--action-horizon", type=int, default=16,
                        help="Action prediction horizon")

    # Docker configuration
    parser.add_argument("--container", type=str, default="groot-thor",
                        help="Docker container name")

    # Debug options
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output including docker command results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Expand user paths
    args.output_dir = os.path.expanduser(args.output_dir)

    # Banner
    print("=" * 70)
    print("GR00T Open-Loop Inference Evaluation")
    print("=" * 70)
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num runs (X):     {args.num_runs}")
    print(f"Num trajectories: {args.num_trajectories}")
    print(f"Seed:             {args.seed}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Docker output:    {args.docker_output_dir}")
    print(f"Steps:            {args.start_step} - {args.end_step}")
    print(f"Action horizon:   {args.action_horizon}")
    print(f"Container:        {args.container}")
    print(f"Verbose:          {args.verbose}")
    print("=" * 70)
    print()

    # Warn about relative paths (they won't work inside Docker)
    if not args.checkpoint.startswith('/'):
        print(f"[WARNING] Checkpoint path '{args.checkpoint}' is relative.")
        print("          Relative paths are resolved from Docker's working directory")
        print("          (/workspace/Isaac-GR00T), not from the host's current directory.")
        print("          Consider using an absolute path inside the container.")
        print()
    if not args.dataset.startswith('/'):
        print(f"[WARNING] Dataset path '{args.dataset}' is relative.")
        print("          Relative paths are resolved from Docker's working directory")
        print("          (/workspace/Isaac-GR00T), not from the host's current directory.")
        print("          Consider using an absolute path inside the container.")
        print()

    # Initialize components
    runner = DockerInferenceRunner(container=args.container, verbose=args.verbose)
    collector = ResultsCollector(output_dir=args.output_dir)

    # Check Docker container
    print("Checking Docker container...")
    if not runner.check_container_running():
        print(f"[ERROR] Docker container '{args.container}' is not running.")
        print("Please start the container with: docker start " + args.container)
        sys.exit(1)
    print(f"[OK] Container '{args.container}' is running")
    print()

    # Main evaluation loop
    total_runs = args.num_trajectories * args.num_runs
    completed = 0

    for traj_id in range(args.num_trajectories):
        print(f"\n{'='*50}")
        print(f"Trajectory {traj_id}")
        print(f"{'='*50}")

        for run_id in range(args.num_runs):
            completed += 1
            print(f"\n[{completed}/{total_runs}] Trajectory {traj_id}, Run {run_id}")

            # Build configuration
            save_plot = (run_id == 0)  # Only save plot on first run
            plot_path = None
            if save_plot:
                plot_path = f"{args.docker_output_dir}/traj_{traj_id}_run_0.png"

            config = InferenceConfig(
                checkpoint=args.checkpoint,
                dataset=args.dataset,
                trajectory_id=traj_id,
                run_id=run_id,
                start_step=args.start_step,
                end_step=args.end_step,
                action_horizon=args.action_horizon,
                seed=args.seed,
                save_plot=save_plot,
                plot_save_path=plot_path,
            )

            # Run inference
            result = runner.run_single_inference(config)

            # Record result
            collector.add_result(result)

            if result.success:
                print(f"  MSE: {result.mse:.6f}, MAE: {result.mae:.6f}")
                if save_plot:
                    print(f"  Plot saved to: {plot_path}")
            else:
                print(f"  [FAILED]")
                # Print full error with indentation for readability
                if result.error_message:
                    for line in result.error_message.split('\n'):
                        print(f"    {line}")

    # Compute and display statistics
    traj_stats, overall_stats = collector.compute_statistics()
    collector.print_summary(traj_stats, overall_stats)
    collector.save_statistics(traj_stats, overall_stats, args)

    print(f"\nResults CSV: {collector.csv_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
