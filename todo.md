```markdown
# TODO

## High Priority — training correctness

- [ ] Fix laser speed not resetting on episode end
  - `LaserHazardWrapper.reset()` does not restore `laser_speed`
  - Speed keeps accelerating across episodes
  - Store base at init: `self._base_laser_speed = laser_speed`
  - In `reset()`: `self.laser_speed = self._base_laser_speed`
  - Keep existing resets for `laser_pos`, direction, active, step_count

- [ ] Fix vectorized training full-reset after every PPO update
  - `_train_vectorized` still does:
    ```python
    trajectories = [[] for _ in range(self.num_envs)]
    obs, _ = self.env.reset()
    obs, _ = self.env.reset()  # duplicate — remove
    ```
  - Wipes partial episodes and discards ongoing experience
  - Desired behavior:
    - Clear trajectory buffers after GAE/update
    - Do **not** call `env.reset()` after updates
    - Let Gym auto-reset individual envs on `done`
    - Keep current vector `obs` as-is

## High Priority — OpenCL caching

- [ ] Stop re-uploading unchanged weights every forward
  - `get_param_buffer()` always does `clEnqueueWriteBuffer` even on cache hit
  - Track version / dirty flag / storage version
  - Upload only when weight/bias data actually changed (e.g. after optimizer step)

- [ ] Reduce host↔device traffic per layer
  - Today: upload X → kernel → download Y every call
  - At minimum: skip weight re-upload when unchanged
  - Optional later: keep intermediates on device across layers

- [ ] Fix `is_available()` semantics
  - On OpenCL init failure, code sets `ocr_initialized = true` so `is_available()` returns true
  - Should return false when there is no real device
  - Or split: module loaded vs device ready

## Medium Priority

- [ ] Tune `opencl_min_elements` after cache fix
  - Re-profile M=64, hidden=128/256 with real weight caching
  - Only use OpenCL when it beats ATen

- [ ] Optional: explicit param cache invalidation after `optimizer.step()`
  - e.g. `opencl_ocl.invalidate_params()` once per update epoch

- [ ] Inference run count
  - `_run_inference` uses `max_episodes` (5000) — use a small fixed default (e.g. 10)

- [ ] Cleanup
  - Remove duplicate `env.reset()` 
  - Drop unused imports in `text_trainer.py` (`Dataset`, `DataLoader`, `math`)
  - Align `done.md` claim about vector reset with actual code

## Low Priority / Later

- [ ] AsyncVectorEnv instead of SyncVectorEnv
- [ ] Keep intermediate activations on device across fused layers (bigger redesign)
- [ ] Strip any remaining dead code paths

## Already done (rounds 1–2)

- [x] Wire `config.yaml` laser settings into `LaserHazardWrapper`
- [x] Pass `hidden` from config into `PolicyNet` / `ValueNet`
- [x] Fix GAE `done` indexing in `compute_returns`
- [x] Hook OpenCL backward into Python autograd
- [x] Make `LaserHazardWrapper.__init__` use the `laser_speed` argument
- [x] Fix laser_speed logging path
- [x] Add dlprimitives attribution in `opencl_ocl.cc`
- [x] Update `info.md`

## Notes

- Network ops that matter: Linear, ReLU, Tanh (fused Linear+ReLU / Linear+Tanh preferred)
- Keep distribution sampling, log_prob, entropy, Adam on PyTorch
- Tiled SGEMM adapted from **dlprimitives** (MIT License, Artyom Beilis)
  - https://github.com/artyom-beilis/dlprimitives
- Highest impact left: laser speed reset, vector-env wipe, weight cache re-upload
```