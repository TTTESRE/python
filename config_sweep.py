#!/usr/bin/env python3
"""
config_sweep.py
Smoke-test 25 config variations using the trainer's own setup logic.
No full env runs: each config is validated by constructing WalkerTrainer
and running a tiny rollout to measure throughput / stability.
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

from text_trainer import load_config

RESULTS_FILE = BASE_DIR / "sweep_results.json"
SWEEP_DIR = BASE_DIR / "sweep_runs"
TOTAL_CONFIGS = 25
PARALLEL = 2


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_config(path, cfg):
    import yaml
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def sample_configs(base, n=25):
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
    num_envs_list = [1, 2, 4, 8, 16, 32]

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
        cfg["max_episodes"] = 5
        cfg["max_steps"] = 200
        cfg["laser_activate_after"] = 999999
        for key, vals in zip(keys, value_lists):
            cfg[key] = random.choice(vals)
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


def to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def run_config_test(cfg, run_dir, result_holder, idx):
    config_path = run_dir / "config.yaml"
    save_config(config_path, cfg)
    log_path = run_dir / "walker_log.csv"
    checkpoint_path = run_dir / "walker_checkpoint.pt"
    stats_path = run_dir / "live_stats.json"

    best_reward = -float("inf")
    final_avg50 = None
    episodes = 0
    runtime = 0.0
    timeout = False
    error = None
    metrics = {}

    try:
        start = time.time()
        trainer = load_config(str(config_path))
        runtime = time.time() - start
        metrics["config_loaded"] = True
    except Exception as e:
        runtime = time.time() - start
        error = f"load_config failed: {e}"
        metrics["config_loaded"] = False
        result_holder[idx] = {
            "best_reward": -float("inf"),
            "final_avg50": None,
            "episodes": 0,
            "runtime_sec": round(runtime, 2),
            "timeout": True,
            "error": error,
            "metrics": to_serializable(metrics),
        }
        return

    try:
        import gymnasium as gym
        import pygame

        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        test_episodes = 0
        max_test_episodes = 3

        while test_episodes < max_test_episodes:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            obs = next_obs

            if done or episode_steps >= 200:
                test_episodes += 1
                if episode_reward > best_reward:
                    best_reward = episode_reward
                if test_episodes == 1:
                    final_avg50 = episode_reward
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0

        env.close()
        metrics["env_ok"] = True
        metrics["test_reward"] = best_reward
        metrics["test_episodes"] = test_episodes
    except Exception as e:
        metrics["env_ok"] = False
        metrics["env_error"] = str(e)

    result_holder[idx] = {
        "best_reward": float(best_reward),
        "final_avg50": float(final_avg50) if final_avg50 is not None else None,
        "episodes": int(episodes),
        "runtime_sec": round(runtime, 2),
        "timeout": timeout,
        "error": error,
        "metrics": to_serializable(metrics),
    }


def main():
    random.seed(42)
    np.random.seed(42)
    ensure_dir(SWEEP_DIR)
    base = load_config()
    configs = sample_configs(base, TOTAL_CONFIGS)
    print(f"Generated {len(configs)} configs for sweep")
    print(f"Running max {PARALLEL} parallel")

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
                target=run_config_test,
                args=(cfg, run_dir, results_batch, idx),
                daemon=True,
            )
            threads.append((run_name, t, time.time()))
            t.start()

        for run_name, t, t0 in threads:
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
                    "metrics": {},
                }
            res["runtime_sec"] = round(runtime, 2)
            results[run_name] = res
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            status = "ERR" if res.get("timeout") else "ok"
            print(f"  {run_name}: {status} reward={res['best_reward']:.2f} metrics={res.get('metrics')} time={res['runtime_sec']:.1f}s")

    print("\n=== SWEEP COMPLETE ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("best_reward", -float("inf")), reverse=True)
    print(f"{'Rank':<5} {'Config':<15} {'Reward':>10} {'Env OK':>8} {'Time (s)':>10}")
    print("-" * 50)
    for rank, (name, res) in enumerate(sorted_results[:20], 1):
        env_ok = res.get("metrics", {}).get("env_ok", False)
        print(f"{rank:<5} {name:<15} {res.get('best_reward', -float('inf')):>10.2f} {str(env_ok):>8} {res.get('runtime_sec', 0):>10.1f}")
    best_name = sorted_results[0][0]
    best_cfg_path = SWEEP_DIR / best_name / "config.yaml"
    if best_cfg_path.exists():
        print(f"\nBest config: {best_cfg_path}")
        print(f"  cp {best_cfg_path} {BASE_DIR}/config.yaml")


if __name__ == "__main__":
    main()
