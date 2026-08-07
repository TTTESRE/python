# Project Info

## Overview

BipedalWalker-v3 PPO trainer with:
- Custom laser hazard wrapper
- OpenCL acceleration for network forward and backward via libtorch tensors
- Optional pygame rendering + ffmpeg video recording
- External fixers: `fixer` (force pygame flips) and `fixer_opencl` (validate GEMM, force CPU fallback on mismatch)
- Config-driven hyperparameters (`config.yaml`)
- Optional SyncVectorEnv parallel environments
- CSV + TensorBoard logging
- Live web dashboard (`dashboard_server.py` + `templates/dashboard.html`) via Flask + SSE

## Key Files

| File | Purpose |
|------|---------|
| `text_trainer.py` | Main training / inference loop, PPO agent, LaserHazardWrapper, OpenCL dispatch, atomic live stats writer |
| `dashboard_server.py` | Flask + SSE server that reads `runs/live_stats.json` and pushes to browser |
| `templates/dashboard.html` | Single-page dashboard UI (training, performance, OpenCL, kernel, system) |
| `opencl_ocl.cc` | OpenCL kernels (tiled SGEMM, fused Linear+ReLU/Tanh, ReLU/Tanh, backward) + pybind11 |
| `setup_ocl.py` | Builds the OpenCL extension with CMake + libtorch |
| `CMakeLists.txt` | CMake config (Torch, torch_python, OpenCL, pybind11) |
| `config.yaml` | Hyperparameters (episodes, lr, laser, hidden, num_envs, compute_chain, dashboard…) |
| `fixer.cc` | Forces pygame `display.flip()` + `event.pump()` via GDB every 33ms |
| `fixer_opencl.cc` | Validates OpenCL vs CPU every 1s; calls `cleanup()` on mismatch |
| `best_walker.pt` / `walker_checkpoint.pt` | Saved policy/value checkpoints |
| `training_log.csv` | Per-episode reward / steps / laser_speed / best |
| `runs/live_stats.json` | Atomic live stats snapshot consumed by dashboard |

## Network Architecture

- **PolicyNet**: obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(act) → Tanh
- **ValueNet**: obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(1)
- Defaults: obs=24, act=4, hidden from config (default 256)
- Action distribution: Independent Normal (mean from net, learned log_std)

## OpenCL Status

- Uses real `torch::Tensor` API (no numpy round-trips)
- Tiled SGEMM (TILE=16, local memory) + fused bias + optional ReLU/Tanh
- Weight/bias buffer cache keyed by data pointer
- Size gate: only use OpenCL for `x.numel() * N >= opencl_min_elements` (default 8192)
- Compute chain from config: `opencl,aten` (falls through to ATen/CPU)
- Backward kernels hooked into Python autograd via OpenCL backward functions
- Tiled GEMM adapted from dlprimitives (MIT, Artyom Beilis) with attribution comment in source
- Instrumented with `time.perf_counter()` kernel timing and module-level call/sample counters
- Cache limits configurable via `opencl:` section in `config.yaml` (`max_param_cache_mb`, `max_buf_cache_mb`)
- C++ API: `set_cache_limits(max_param_mb, max_buf_mb)`, `drop_param_cache()`, `drop_scratch_cache()`, `cache_stats()`
- Backward paths use `alloc_buffer`/`release_buffer` for activations/gradients; only weights/bias use param cache
- Per-update memory logging: `[mem] rss_mb=... ocl_param_entries=... ocl_param_mb=... ocl_buf_entries=... ocl_buf_mb=...`

## Laser Hazard

- Activates after `step_count > laser_activate_after` (from config)
- Starts at `laser_pos = -5`, base speed from config `laser_speed`
- Accelerates each step; terminates on hit (distance < 0.5) with -10 reward
- All laser settings (`laser_speed`, `laser_range`, `laser_activate_after`) are now wired from config

## Live Dashboard

- Separate process: `python dashboard_server.py`
- Reads `runs/live_stats.json` written atomically by trainer
- SSE endpoint `/events` pushes snapshots at `board_fps` (default 5 Hz)
- `/api/stats` returns latest JSON; `/` serves single-page UI
- Config under `dashboard:` in `config.yaml` (`host`, `port`, `board_fps`, `stats_path`)
- Stale detection: UI shows stale/offline if file age > 2s
- Trainer has no Flask imports and does not block on network I/O

## Build

```bash
python setup_ocl.py
```