# Project Info

## Overview

BipedalWalker-v3 PPO trainer with:
- Custom laser hazard wrapper
- Experimental OpenCL acceleration for network forward pass
- Optional pygame rendering + ffmpeg video recording
- External "fixer" process that forces display updates via GDB

## Key Files

| File | Purpose |
|------|---------|
| `text_trainer.py` | Main training / inference loop, PPO agent, LaserHazardWrapper |
| `opencl_ocl.cc` | Custom OpenCL kernels (Linear, ReLU, Tanh) + pybind11 bindings |
| `setup_ocl.py` | Builds the OpenCL extension with CMake + libtorch |
| `CMakeLists.txt` | CMake config (finds Torch, OpenCL, pybind11) |
| `fixer.cc` | Forces pygame flips by attaching GDB to the Python process |
| `best_walker.pt` | Saved best policy/value checkpoint |
| `generate.py` | Unrelated leftover (CharCNN text generation) |

## Network Architecture

- **PolicyNet**: obs(24) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(4) → Tanh
- **ValueNet**: obs(24) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(1)
- Action distribution: Independent Normal (mean from network, learned log_std)

## OpenCL Status

- Currently only implements forward kernels
- Links against libtorch but does **not** call any ATen/c10 APIs yet
- Heavy numpy ↔ OpenCL copies every layer → usually slower than pure PyTorch CPU
- Automatic fallback chain:
  1. Custom OpenCL kernel
  2. Attempt `torch.device("ocl:0")` (almost always fails)
  3. Normal PyTorch CPU

## Laser Hazard

- Activates after `step_count > 100`
- Starts at `laser_pos = 0.0`, moves and accelerates
- Terminates episode on hit (distance < 0.5) with -10 reward
- User has already adjusted spawn / speed

## Build

```bash
python setup_ocl.py
```

Requires: libtorch (or PyTorch with cmake files), OpenCL, pybind11, CMake.

## Run Modes

```bash
python text_trainer.py --train          # headless training
python text_trainer.py --run            # inference + video
python text_trainer.py --runtrain       # training with GUI
python text_trainer.py --fixtrain       # training + fixer process
```
