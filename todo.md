```markdown
# TODO — Live web dashboard (Flask + WebSocket)

## Status

- [x] Choose IPC between trainer and dashboard
  - **Chosen:** trainer appends/updates a small JSON file (`runs/live_stats.json`)
  - Do **not** `import dashboard` from `text_trainer.py`

- [x] Add Flask app entrypoint
  - `python dashboard_server.py`
  - Serves UI at `http://127.0.0.1:8080/`

- [x] Add SSE channel for live push
  - Pure Flask Server-Sent Events at `/events`
  - Push rate: default **5 Hz** (`1 / board_fps`)

- [x] Config keys in `config.yaml`
  ```yaml
  dashboard:
    enabled: true          # optional; trainer can ignore this
    host: 0.0.0.0
    port: 8080
    board_fps: 5           # updates per second
    stats_path: runs/live_stats.json
  ```

- [x] Define `live_stats.json` schema (trainer writes, dashboard reads)
  ```json
  {
    "episode": 0,
    "reward": 0.0,
    "avg50": 0.0,
    "best_reward": 0.0,
    "steps": 0,
    "laser_speed": 0.0,
    "opencl_enabled": true,
    "compute_chain": ["opencl", "aten"],
    "opencl_min_elements": 8096,
    "opencl_calls": 0,
    "aten_calls": 0,
    "fallback_count": 0,
    "fwd_per_sec": 0.0,
    "bwd_per_sec": 0.0,
    "samples_per_sec": 0.0,
    "env_steps_per_sec": 0.0,
    "kernel_time_ms_avg": 0.0,
    "kernel_time_ms_total": 0.0,
    "ram_mb": 0.0,
    "cpu_pct": 0.0,
    "elapsed_sec": 0.0,
    "ts": 0.0
  }
  ```

- [x] Trainer side (minimal, no Flask import)
  - [x] Write/update stats file atomically (write temp → rename)
  - [x] Update on episode end via `_publish_live_stats()`
  - [x] Increment OpenCL/ATen counters in dispatch helpers
  - [x] Instrument OpenCL kernels with `time.perf_counter()` timing
  - [x] Laser speed read from wrapper

- [x] Dashboard side
  - [x] Background task reads stats file at `board_fps`
  - [x] Broadcast snapshot to all connected SSE clients

- [x] UI sections
  - [x] Title + connection status (live / stale if `ts` too old)
  - [x] Training: episode, reward, avg50, best, steps, laser_speed
  - [x] Performance: fwd/s, bwd/s, samples/s, env steps/s
  - [x] OpenCL: enabled, chain, threshold, calls, ATen calls, fallbacks
  - [x] Kernel: avg / total kernel time
  - [x] System: RAM, CPU, elapsed
  - [x] Clean layout (simple HTML + CSS)

- [x] Server details
  - [x] `GET /` → dashboard HTML
  - [x] `GET /api/stats` → latest JSON (fallback if SSE down)
  - [x] `GET /events` → SSE push at `board_fps`
  - [x] Configurable `host` / `port` from config
  - [x] Graceful shutdown (Ctrl+C)

- [x] Acceptance checks
  - [x] Trainer runs with no dashboard process and no errors
  - [x] Dashboard alone shows “waiting for trainer” / stale
  - [x] Both running → UI updates ~5×/sec without trainer lag
  - [x] Changing `board_fps` / `port` in config works after restart of dashboard
  - [x] `laser_speed` in UI is non-zero when laser is active

## Notes

- Keep `text_trainer.py` free of web framework imports
- Atomic file write avoids partial JSON reads
- 5 Hz is enough for a training dashboard; don't push every env step unless sampled
```