# Done

## Core Project

- [x] Basic PPO agent (policy + value networks)
- [x] BipedalWalker-v3 environment integration (Gymnasium)
- [x] LaserHazardWrapper with moving laser and collision
- [x] Checkpoint saving / loading (`walker_checkpoint.pt`, `best_walker.pt`)
- [x] Multiple run modes (train, run, runtrain, fix*)
- [x] pygame rendering path
- [x] Optional ffmpeg video recording (headless `--run` produces webm/mp4)
- [x] Signal handler for clean exit + save
- [x] Fixed PPO GAE return/advantage calculation (proper lambda-GAE with terminal handling)
- [x] Stored value estimates per step and compute GAE at update time
- [x] Basic logging + avg reward tracking + TensorBoard SummaryWriter

## OpenCL Extension

- [x] Custom OpenCL kernels for Linear, ReLU, Tanh (torch::Tensor API, no numpy round-trips)
- [x] pybind11 module (`opencl_ocl`) with `torch::Tensor` inputs/outputs
- [x] Python autograd.Function wrappers (`OpenCLLinear`, `OpenCLReLU`, `OpenCLTanh`, `OpenCLLinearReLU`, `OpenCLLinearTanh`)
- [x] Device buffer caching for weights/bias keyed by data pointer (reuses cl_mem, re-uploads contents)
- [x] Fused kernels: `linear_relu_forward`, `linear_tanh_forward`
- [x] 2D tiled SGEMM kernel (TILE=16, local memory) with fused bias+activation
- [x] Automatic fallback when OpenCL is unavailable or fails
- [x] Size-based dispatch: only use OpenCL for `x.numel() * N >= 8192`
- [x] CMake + setup_ocl.py build system (links libtorch + torch_python + OpenCL)
- [x] `is_available()` check

## Fixers

- [x] `fixer.cc`: attaches GDB to running Python process, forces `pygame.display.flip()` + `pygame.event.pump()` every 33ms
- [x] `fixer_opencl.cc`: attaches GDB, validates OpenCL GEMM output vs CPU every 1s, calls `opencl_ocl.cleanup()` on mismatch (force CPU fallback)
- [x] `--fixrun`, `--fixtrain`, `--fixruntrain` flags launch both fixers

## Verification

- [x] Kernel correctness vs PyTorch reference: max abs error < 1e-5 across multiple shapes including large values
- [x] Training smoke test under gdb: completed 3 episodes with no crash
- [x] Batched update smoke test under gdb (M=64 through OpenCL path): no crash, grads finite
- [x] Integrated network forward (OpenCL vs CPU): maxerr ~1e-7

## Fixes Already Applied by User

- [x] Laser no longer spawns directly on top of the walker
- [x] Laser speed / behavior adjusted

## Known Working Behaviour

- Training continues even if OpenCL path completely fails (falls back to ATen CPU)
- Best model is saved when a new high reward is reached
- Inference mode loads `best_walker.pt` and can record video to webm/mp4 via ffmpeg
- fixer.cc + fixer_opencl run as subprocesses in fix modes
