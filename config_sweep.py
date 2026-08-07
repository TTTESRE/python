#!/usr/bin/env python3
"""
config_sweep.py
Run multiple config.yaml variations using WalkerTrainer directly.
Each config runs for 150 episodes; best avg50 is recorded.
Max 2 parallel runs using threads.
"""

import os
import sys
import time
import json
import shutil
import random
import threading
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from text_trainer import load_config, WalkerTrainer

RESULTS_FILE = BASE_DIR / "sweep_results.json"
SWEEP_DIR = BASE_DIR / "sweep_runs"
TOTAL_CONFIGS = 100
EPISODES_PER_CONFIG = 150
PARALLEL = 2


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_config(path, cfg):
    import yaml
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def sample_configs(base, n=100):
    configs = []
    lrs = [1e-4, 3e-4, 5e-4, 1e-3]
    hiddens = [64, 128, 256]
    update_intervals = [4096, 8192, 16384]
    mini_batch_sizes = [512, 1024, 2048]
    epochs_per_updates = [3, 5, 8]
    gammas = [0.99, 0.995, 0.999]
    lams = [0.9, 0.95, 0.97]
    clips = [0.1, 0.2, 0.3]
    entropy_coefs = [0.0, 0.01, 0.02]
    vf_coefs = [0.5, 0.8, 1.0]
    opencl_mins = [2048, 4096, 8192]
    num_envs_list = [8, 16, 32]

    keys = [
        "lr", "hidden", "update_interval", "mini_batch_size", "epochs_per_update",
        "gamma", "lam", "clip", "entropy_coef", "vf_coef", "opencl_min_elements", "num_envs"
    ]
    value_lists = [
        lrs, hiddens, update_intervals, mini_batch_sizes, epochs_per_updates,
        gammas, lams, clips, entropy_coefs, vf_coefs, opencl_mins, num_envs_list
    ]

    seen = set()
    attempts = 0
    while len(configs) < n and attempts < n * 10:
        attempts += 1
        cfg = dict(base)
        cfg["max_episodes"] = EPISODES_PER_CONFIG
        cfg["max_steps"] = 2000
        cfg["laser_activate_after"] = 999999
        for key, vals in zip(keys, value_lists):
            cfg[key] = random.choice(vals)
        cfg["num_envs"] = min(cfg["num_envs"], PARALLEL)
        cfg["mini_batch_size"] = min(cfg["mini_batch_size"], cfg["update_interval"])
        key_data = []
        for k, v in sorted(cfg.items()):
            if isinstance(v, dict):
                key_data.append((k, tuple(sorted((ik, iv) for ik, iv in v.items()))))
            elif isinstance(v, list):
                key_data.append((k, tuple(v)))
            else:
                key_data.append((k, v))
        try:
            key_tuple = tuple(key_data)
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
        except TypeError:
            pass
        configs.append(cfg)
    return configs


def run_single_config(cfg, run_dir, result_holder, idx):
    config_path = run_dir / "config.yaml"
    save_config(config_path, cfg)
    log_path = run_dir / "walker_log.csv"
    checkpoint_path = run_dir / "walker_checkpoint.pt"

    best_reward = -float("inf")
    final_avg50 = None
    episodes = 0
    runtime = 0.0
    timeout = False
    error = None

    try:
        start = time.time()
        trainer = WalkerTrainer(
            mode="train",
            video_path=None,
            container="webm",
            config_path=str(config_path),
        )
        trainer.log_path = str(log_path)
        trainer.checkpoint_path = str(checkpoint_path)
        trainer.max_episodes = EPISODES_PER_CONFIG
        trainer.train()
        runtime = time.time() - start
    except Exception as e:
        runtime = time.time() - start
        error = str(e)
        timeout = True

    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            if len(lines) > 1:
                last = lines[-1].strip().split(",")
                if len(last) >= 2:
                    try:
                        final_avg50 = float(last[1])
                        episodes = int(last[0])
                    except ValueError:
                        pass
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            r = float(parts[1])
                            if r > best_reward:
                                best_reward = r
                        except ValueError:
                            pass
        except Exception:
            pass

    if checkpoint_path.exists():
        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            br = ckpt.get("best_reward", -float("inf"))
            if isinstance(br, (int, float)) and br > best_reward:
                best_reward = br
        except Exception:
            pass

    result_holder[idx] = {
        "best_reward": best_reward,
        "final_avg50": final_avg50,
        "episodes": episodes,
        "runtime_sec": round(runtime, 2),
        "timeout": timeout,
        "error": error,
    }


def main():
    random.seed(42)
    np.random.seed(42)
    ensure_dir(SWEEP_DIR)
    base = load_config()
    configs = sample_configs(base, TOTAL_CONFIGS)
    print(f"Generated {len(configs)} configs for sweep")
    print(f"Running max {PARALLEL} parallel, {EPISODES_PER_CONFIG} episodes each")

    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    else:
        results = {}

    pending = []
    for i, cfg in enumerate(configs):
        run_name = f"config_{i:03d}"
        if run_name in results:
            continue
        pending.append((i, cfg))

    print(f"Pending runs: {len(pending)}")
    if not pending:
        print("All configs already completed.")
        return

    batches = []
    for i in range(0, len(pending), PARALLEL):
        batches.append(pending[i:i + PARALLEL])

    for batch_idx, batch in enumerate(batches):
        print(f"\n=== Batch {batch_idx + 1}/{len(batches)} ({len(batch)} configs) ===")
        threads = []
        results_batch = [None] * len(batch)
        for idx, (orig_i, cfg) in enumerate(batch):
            run_name = f"config_{orig_i:03d}"
            run_dir = SWEEP_DIR / run_name
            ensure_dir(run_dir)
            print(f"  Starting {run_name}...")
            t = threading.Thread(
                target=run_single_config,
                args=(cfg, run_dir, results_batch, idx),
                daemon=True,
            )
            threads.append((run_name, t, time.time()))
            t.start()

        for (run_name, t, t0) in threads:
            t.join()
            runtime = time.time() - t0
            idx = [i for i, (rn, th, _) in enumerate(threads) if rn == run_name][0]
            res = results_batch[idx]
            if res is None:
                res = {
                    "best_reward": -float("inf"),
                    "final_avg50": None,
                    "episodes": 0,
                    "runtime_sec": round(runtime, 2),
                    "timeout": True,
                    "error": "thread crash",
                }
            res["runtime_sec"] = round(runtime, 2)
            results[run_name] = res
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            status = "TIMEOUT" if res.get("timeout") else "done"
            print(f"  {run_name}: {status} best={res['best_reward']:.2f} avg50={res.get('final_avg50')} episodes={res['episodes']} time={res['runtime_sec']:.1f}s")

    print("\n=== SWEEP COMPLETE ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("best_reward", -float("inf")), reverse=True)
    print(f"{'Rank':<5} {'Config':<15} {'Best Reward':>12} {'Avg50':>10} {'Episodes':>10} {'Time (s)':>10}")
    print("-" * 65)
    for rank, (name, res) in enumerate(sorted_results[:20], 1):
        avg50 = res.get("final_avg50")
        avg50_str = f"{avg50:.2f}" if avg50 is not None else "N/A"
        print(f"{rank:<5} {name:<15} {res.get('best_reward', -float('inf')):>12.2f} {avg50_str:>10} {res.get('episodes', 0):>10} {res.get('runtime_sec', 0):>10.1f}")
    best_name = sorted_results[0][0]
    best_cfg_path = SWEEP_DIR / best_name / "config.yaml"
    if best_cfg_path.exists():
        print(f"\nBest config saved to: {best_cfg_path}")
        print("To use it:")
        print(f"  cp {best_cfg_path} {BASE_DIR}/config.yaml")


if __name__ == "__main__":
    main()
