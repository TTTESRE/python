# TODO — Beta blockers

## Critical (training will not run)

- [x] Fix `IndentationError` in `text_trainer.py` around line 783
  - Error: `avg = self._log_episode(self.episode, episode_reward, episode_steps, laser_speed)` unexpected indent
  - Fixed indent of `laser_info`, `laser_speed`, `avg`, `print(...)`, best-model save, and `obs, _ = self.env.reset()` to align with the surrounding `if done` block
  - Verified: `python -m py_compile text_trainer.py` passes

- [x] Export `invalidate_params` in `opencl_ocl.cc`
  - Python calls `opencl_ocl.invalidate_params()` after `optimizer.step()`
  - Moved `invalidate_param_cache()` outside the anonymous namespace and added `m.def("invalidate_params", ...)` to `PYBIND11_MODULE`
  - Rebuilt the `.so` (`python setup_ocl.py`) — verified export works

## High Priority — OpenCL weight cache

- [x] Stop force-uploading weights every forward
  - Changed all `get_param_buffer(..., true)` calls to `get_param_buffer(..., false)` in `linear_kernel_dispatch`, `std_kernel_dispatch`, and `relu_backward`
  - With `force_upload=false`, cache hit skips `clEnqueueWriteBuffer` unless version was cleared
  - `invalidate_params()` after `optimizer.step()` clears versions → next forward uploads once

- [x] Verify cache behavior after fix
  - Verified forward/backward/invalidate_params all work correctly
  - Cache hit path skips `clEnqueueWriteBuffer` on subsequent forwards with unchanged weights

## After the three fixes

```bash
python setup_ocl.py          # rebuild extension — DONE
python -m py_compile text_trainer.py    # syntax test — DONE
python text_trainer.py --train      # check if the thing works — READY
```

## Notes

- All three beta blockers are resolved — training is unblocked
- Tiled SGEMM adapted from dlprimitives (MIT, Artyom Beilis)