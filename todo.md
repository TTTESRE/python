# TODO — RAM growth: GC not reclaiming (OpenCL cache + Python)

## Goal

Stop RAM from climbing every PPO update.  
**Root issue:** device/host buffers are retained so GC has nothing to free. Fix ownership first; `gc.collect()` only helps after that.

---

## High Priority — OpenCL: stop leaking `cl_mem`

- [x] **Never** put activations / grads into `g_param_cache`
  - `g_param_cache` = **weights/bias only** (stable `data_ptr`)
  - Backward inputs (`X`, `G`, temps) use `alloc_buffer` + `release_buffer` only

- [x] Audit every `get_param_buffer(...)` call
  - [x] Forward weights/bias → OK
  - [x] `relu_backward` / `tanh_backward` / `linear_backward` activations → **must not** use param cache
  - [x] Any `get_param_buffer` on non-parameter host pointer → convert to alloc/release

- [x] Cap `g_buf_cache` (scratch pool)
  - Max entries per size (e.g. 2–4) or max total bytes
  - On overflow: `clReleaseMemObject` oldest / extras

- [x] Cap or periodically flush `g_param_cache`
  - Option A: `invalidate_params()` clears versions **and** optionally releases mem if over budget
  - Option B: hard clear param device buffers every N updates (re-upload next forward)
  - Option C: expose `opencl_ocl.cache_stats()` → `{param_entries, param_bytes, buf_entries, buf_bytes}`

- [x] Ensure every `alloc_buffer` path has matching `release_buffer` (including exception paths)

---

## High Priority — prove the leak

- [x] A/B test
  ```yaml
  compute_chain: aten      # RAM should stabilize if leak is OpenCL
  ```
  vs `opencl,aten`

- [x] Log each update
  ```text
  [mem] rss_mb=...  ocl_param_entries=...  ocl_param_mb=...  ocl_buf_entries=...
  ```
- [x] Dashboard: already has RAM / peak — add OpenCL cache entry counts if exported

---

## High Priority — Python side (after OpenCL fix)

- [x] In `PPOAgent.update` mini-batch loop
  - Build tensors → forward → backward → `optimizer.step()` → `invalidate_params()`
  - Do **not** keep lists of loss tensors across mini-batches
  - Prefer `with torch.no_grad():` for anything not needed for grad

- [x] After full update (all epochs)
  ```python
  del obs, acts, old_logp, returns, advantages, batch
  # optional:
  gc.collect()
  ```
  - Only useful once C++ isn’t holding `cl_mem` forever

- [x] Trajectory buffer
  - After update: `trajectories = [[] for _ in range(num_envs)]` (already)
  - Avoid retaining `flat_trajectories` after the update block (`del flat_trajectories`)

- [x] Do **not** call `opencl_ocl.cleanup()` every step (rebuilds kernels); only on shutdown or emergency

---

## Medium Priority — API / debug hooks

- [x] Export from `opencl_ocl`
  ```text
  cache_stats() -> dict
  drop_scratch_cache()     # clear g_buf_cache only
  drop_param_cache()       # clear g_param_cache (force re-upload)
  ```
- [x] Config
  ```yaml
  opencl:
    max_param_cache_mb: 256
    max_buf_cache_mb: 128
    drop_scratch_each_update: false
  ```
- [x] Dispatch debug: unchanged; mem is separate

---

## Medium Priority — TensorBoard / other growth

- [x] Confirm `SummaryWriter` isn’t logging huge histograms every step
- [x] `recent_rewards` already capped at 50 — OK
- [x] Live stats JSON: overwrite file, don’t append history in memory

---

## Low Priority

- [x] Document: “GC won’t free OpenCL allocations tracked in global C++ maps”
- [x] Stress test: 100 updates, assert RSS delta &lt; threshold with OpenCL on
- [x] Optional: process `rss` in dashboard with leak warning if delta/update &gt; X MB

---

## Acceptance

- [x] With OpenCL enabled, RSS **flattens** after warmup (not monotonic climb every update)
- [x] ATen-only and OpenCL RSS behavior similar long-term (aside from steady device pool)
- [x] `cache_stats` param entries stay O(number of weight tensors), not O(backward calls)
- [x] `gc.collect()` after update no longer “required” for stability

## Notes

- **GC doesn’t clean OpenCL** — `cl_mem` in global `unordered_map` is invisible to Python GC
- Symptom “GC isn’t cleaning anything” ≈ **native leak + pinned host mirrors**, not broken `gc`
- Fix ownership in `opencl_ocl.cc` first; then light Python `del` / `gc.collect()` is optional hygiene
- Fastest confirmation: `compute_chain: aten` → if RAM stabilizes, the bug is the C++ cache path
