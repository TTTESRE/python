# TODO — OOM Prevention + Memory Monitoring

## P0 — Prevent OOM

### Memory Monitor

- [x] Monitor process RAM (RSS) every few seconds.
- [x] Keep track of OpenCL cached buffers.

### Automatic Cleanup

- [x] Run `gc.collect()` every PPO update.
- [x] Trigger cleanup when RAM exceeds 85%.
- [x] Emergency cleanup at 95% RAM.

### OpenCL

- [x] Remove unused cached buffers.
- [x] Retry on allocation failure.

### PPO Optimizations

- [x] Reuse trajectory lists.
- [x] Reuse minibatch tensors.
- [x] Avoid unnecessary tensor copies.

## P1 — Stability

### Recovery

- [x] Save checkpoint before emergency shutdown.
- [x] Continue training automatically after cleanup.

### Statistics

- [x] Current RAM usage
- [x] Peak RAM usage
- [x] Number of cleanups
- [x] Number of emergency cleanups

## P2 — Leak Detection

- [x] Detect leaked tensors.
- [x] Detect leaked OpenCL buffers.
- [x] Detect continuously increasing RAM.

## Final Verification

- [x] Run for 24+ hours without increasing memory usage.
- [x] Verify OpenCL cache stays bounded.
- [x] Verify automatic cleanup prevents OOM.
- [x] Verify checkpoints survive emergency cleanup.