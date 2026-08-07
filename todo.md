```markdown
# TODO — Live web dashboard (Flask + WebSocket)

## Goal

Separate process (not imported by `text_trainer.py`):

- HTTP server on **port 8080** (configurable)
- Live updates **5 times per second** (configurable)
- Config via `config.yaml` (or `config.yml`)
- Training process only **publishes metrics**; dashboard only **serves + displays**

---

## High Priority — architecture

- [ ] Choose IPC between trainer and dashboard (pick one)
  - **Recommended:** trainer appends/updates a small JSON file or Redis-less shared status file (e.g. `runs/live_stats.json`)
  - Alternative: UDP/TCP localhost socket, or multiprocessing Queue (harder if separate processes)
  - Do **not** `import dashboard` from `text_trainer.py`

- [ ] Add Flask app entrypoint
  - e.g. `python dashboard_server.py` or `python -m dashboard`
  - Serves UI at `http://127.0.0.1:8080/`

- [ ] Add WebSocket (or SSE) channel for live push
  - Prefer **Flask-SocketIO** or **fastapi + websockets**; if staying pure Flask: Server-Sent Events are simpler than raw WS
  - Push rate: default **5 Hz** (`1 / board_fps`)

- [ ] Config keys in `config.yaml`
  ```yaml
  dashboard:
    enabled: true          # optional; trainer can ignore this
    host: 0.0.0.0
    port: 8080
    board_fps: 5           # updates per second
    stats_path: runs/live_stats.json
  ```

---

## High Priority — data contract

- [ ] Define `live_stats.json` schema (trainer writes, dashboard reads)
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

- [ ] Trainer side (minimal, no Flask import)
  - [ ] Write/update stats file atomically (write temp → rename)
  - [ ] Update on episode end + optionally every N steps
  - [ ] Increment OpenCL/ATen counters in dispatch helpers
  - [ ] Fix laser_speed read: use wrapper `.laser_speed`, not `.env.laser_speed`

- [ ] Dashboard side
  - [ ] Background task reads stats file at `board_fps`
  - [ ] Broadcast snapshot to all connected WS/SSE clients

---

## Medium Priority — UI sections

- [ ] Title + connection status (live / stale if `ts` too old)
- [ ] Training: episode, reward, avg50, best, steps, laser_speed
- [ ] Performance: fwd/s, bwd/s, samples/s, env steps/s
- [ ] OpenCL: enabled, chain, threshold, calls, ATen calls, fallbacks
- [ ] Kernel: avg / total kernel time
- [ ] System: RAM, CPU, elapsed
- [ ] Clean layout (simple HTML + CSS; no heavy frontend framework required)

---

## Medium Priority — server details

- [ ] `GET /` → dashboard HTML
- [ ] `GET /api/stats` → latest JSON (fallback if WS down)
- [ ] `WS /ws` or Socket.IO `/` → push at `board_fps`
- [ ] Configurable `host` / `port` from config
- [ ] Graceful shutdown (Ctrl+C)
- [ ] Dependency list: `flask`, `flask-socketio` (or SSE-only to avoid extra deps)

---

## Low Priority

- [ ] Stale detection (e.g. no file update for >2s → show “trainer offline”)
- [ ] Optional auth bind to `127.0.0.1` only by default for safety
- [ ] Chart of last N rewards (client-side from pushed points)
- [ ] Document in README:
  ```bash
  # terminal 1
  python text_trainer.py --train
  # terminal 2
  python dashboard_server.py
  # browser
  http://127.0.0.1:8080
  ```

---

## Out of scope

- Importing Flask/dashboard inside `text_trainer.py`
- Blocking the training loop on network I/O
- Replacing TensorBoard (dashboard is a live glance, not full experiment tracking)

---

## Acceptance checks

- [ ] Trainer runs with no dashboard process and no errors
- [ ] Dashboard alone shows “waiting for trainer” / stale
- [ ] Both running → UI updates ~5×/sec without trainer lag
- [ ] Changing `board_fps` / `port` in config works after restart of dashboard
- [ ] `laser_speed` in UI is non-zero when laser is active

## Notes

- Keep `text_trainer.py` free of web framework imports
- Atomic file write avoids partial JSON reads
- 5 Hz is enough for a training dashboard; don’t push every env step unless sampled
```