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
- [x] Atomic `runs/live_stats.json` writer for dashboard integration
- [x] Live web dashboard via Flask + SSE (`dashboard_server.py` + `templates/dashboard.html`)
- [x] OpenCL kernel timing instrumentation (`time.perf_counter()` in forward/backward)
- [x] Module-level OpenCL/ATen call and sample counters
- [x] Dashboard config section in `config.yaml` (`dashboard:` keys)
- [x] Dashboard server reads config for `host`, `port`, `board_fps`, `stats_path`

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
- [x] OpenCL backward kernels hooked into Python autograd
- [x] dlprimitives attribution comment added to opencl_ocl.cc
- [x] Kernel timing instrumentation for dashboard performance metrics

## Fixers

- [x] `fixer.cc`: attaches GDB to running Python process, forces `pygame.display.flip()` + `pygame.event.pump()` every 33ms
- [x] `fixer_opencl.cc`: attaches GDB, validates OpenCL GEMM output vs CPU every 1s, calls `opencl_ocl.cleanup()` on mismatch (force CPU fallback)
- [x] `--fixrun`, `--fixtrain`, `--fixruntrain` flags launch both fixers

## Live Dashboard

- [x] Trainer writes `runs/live_stats.json` atomically (write temp + rename)
- [x] Stats written on episode end via `_publish_live_stats()`
- [x] Dashboard server reads config for host/port/board_fps/stats_path
- [x] SSE `/events` endpoint broadcasts stats to connected clients
- [x] `/api/stats` endpoint returns latest JSON snapshot
- [x] `GET /` serves single-page HTML dashboard
- [x] Stale detection: status dot switches to stale/offline if file mtime > 2s
- [x] UI sections: training, performance, OpenCL, kernel, system
- [x] Verified trainer runs standalone without dashboard process
- [x] Verified dashboard shows waiting state when trainer is offline
- [x] Title + connection status (live / stale / offline)
- [x] Training card: episode, reward, avg50, best, steps, laser_speed
- [x] Performance card: fwd/s, bwd/s, samples/s, env steps/s
- [x] OpenCL card: enabled, chain, threshold, calls, ATen calls, fallbacks
- [x] Kernel card: avg kernel time, total kernel time
- [x] System card: RAM, CPU, elapsed
- [x] Clean layout with simple HTML + CSS
- [x] Graceful shutdown on Ctrl+C
- [x] `laser_speed` in UI is non-zero when laser is active

## Verification

- [x] Kernel correctness vs PyTorch reference: max abs error < 1e-5 across multiple shapes including large values
- [x] Training smoke test under gdb: completed 3 episodes with no crash
- [x] Batched update smoke test under gdb (M=64 through OpenCL path): no crash, grads finite
- [x] Integrated network forward (OpenCL vs CPU): maxerr ~1e-7
- [x] Dashboard smoke test: trainer writes stats, server reads and serves via SSE
- [x] Dashboard offline state: shows waiting/stale when trainer not running

## Config Wiring

- [x] Wired `config.yaml` laser settings (`laser_speed`, `laser_range`, `laser_activate_after`) into `LaserHazardWrapper`
- [x] Wired `config.yaml` `hidden` parameter into `PolicyNet` and `ValueNet`
- [x] Fixed laser_speed logging in training_log.csv
- [x] Added `dashboard:` config section with `enabled`, `host`, `port`, `board_fps`, `stats_path`
- [x] Dashboard defaults included in `load_config()`

## Fixes Already Applied by User

- [x] Laser no longer spawns directly on top of the walker
- [x] Laser speed / behavior adjusted

## Known Working Behaviour

- Training continues even if OpenCL path completely fails (falls back to ATen CPU)
- Best model is saved when a new high reward is reached
- Inference mode loads `best_walker.pt` and can record video to webm/mp4 via ffmpeg
- fixer.cc + fixer_opencl run as subprocesses in fix modes
- Vectorized training no longer resets all envs after every update
- Live dashboard updates ~5×/sec without blocking training
