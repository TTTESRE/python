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

## Key Files

| File | Purpose |
|------|---------|
| `text_trainer.py` | Main training / inference loop, PPO agent, LaserHazardWrapper, OpenCL dispatch |
| `opencl_ocl.cc` | OpenCL kernels (tiled SGEMM, fused Linear+ReLU/Tanh, ReLU/Tanh, backward) + pybind11 |
| `setup_ocl.py` | Builds the OpenCL extension with CMake + libtorch |
| `CMakeLists.txt` | CMake config (Torch, torch_python, OpenCL, pybind11) |
| `config.yaml` | Hyperparameters (episodes, lr, laser, hidden, num_envs, compute_chain, …) |
| `fixer.cc` | Forces pygame `display.flip()` + `event.pump()` via GDB every 33ms |
| `fixer_opencl.cc` | Validates OpenCL vs CPU every 1s; calls `cleanup()` on mismatch |
| `best_walker.pt` / `walker_checkpoint.pt` | Saved policy/value checkpoints |
| `training_log.csv` | Per-episode reward / steps / laser_speed / best |

## Network Architecture

- **PolicyNet**: obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(act) → Tanh
- **ValueNet**: obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(1)
- Defaults: obs=24, act=4, hidden from config (default 256)
- Action distribution: Independent Normal (mean from net, learned log_std)

## OpenCL Status

- Uses real `torch::Tensor` API (no numpy round-trips)
- Tiled SGEMM (TILE=16, local memory) + fused bias + optional ReLU/Tanh
- Weight/bias buffer cache keyed by data pointer
- Size gate: only OpenCL when `x.numel() * N >= opencl_min_elements` (default 8192)
- Compute chain from config: `opencl,aten` (falls through to ATen/CPU)
- Backward kernels hooked into Python autograd via OpenCL backward functions
- Tiled GEMM adapted from dlprimitives (MIT, Artyom Beilis) with attribution comment in source

## Laser Hazard

- Activates after `step_count > laser_activate_after` (from config)
- Starts at `laser_pos = -5`, base speed from config `laser_speed`
- Accelerates each step; terminates on hit (distance < 0.5) with -10 reward
- All laser settings (`laser_speed`, `laser_range`, `laser_activate_after`) are now wired from config

## Build

```bash
python setup_ocl.py