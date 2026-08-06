# TODO

## High Priority

- [x] Rewrite `opencl_ocl.cc` to use real libtorch tensors (`torch::Tensor`) instead of numpy round-trips
- [x] Cache weight/bias buffers on the OpenCL device (avoid re-uploading every forward pass)
- [x] Add fused kernels: `linear_relu` and `linear_tanh`
- [x] Replace naive Linear kernel with tiled version (local memory)
- [x] Fix PPO return / advantage calculation (current GAE logic is broken)
- [x] Properly handle episode terminals when computing returns
- [x] Add TensorBoard (optional, via SummaryWriter)

## Medium Priority

- [x] Make OpenCL path automatically fall back to fast ATen (`at::linear`, `at::relu`, `at::tanh`) when OpenCL is slower or unavailable
- [x] Add size-based decision: only use OpenCL for matrix sizes where it actually wins
- [x] Clean up laser spawn position and speed (already partially done by user)
- [x] Remove or isolate unrelated `generate.py` (CharCNN leftover)
- [x] Add basic logging / average reward tracking

## Low Priority / Later

- [ ] Implement OpenCL backward kernels (optional, high effort)
- [ ] Hyperparameter config file instead of hardcoded values
- [ ] Support multiple parallel environments
- [ ] Strip unused libtorch dependency **or** fully embrace it (current choice: embrace it)

## Notes

- Network ops that matter: Linear, ReLU, Tanh only
- Keep distribution sampling, log_prob, entropy, Adam on PyTorch
- Current OpenCL path is usually slower than pure CPU because of host↔device copies
- Use a x to mark that something is done , exp. [x] 
- GEMM kernel adapted from dlprimitives (MIT, Artyom Beilis) with attribution in opencl_ocl.cc
