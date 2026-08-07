#!/usr/bin/env python3
"""
dashboard_server.py
Flask + SSE live dashboard for the OpenCL PyTorch trainer.
Reads runs/live_stats.json and pushes updates to connected browsers.
"""

import json
import os
import time
import threading
import queue
from pathlib import Path

import yaml
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

stats_path = "runs/live_stats.json"
board_fps = 5
clients = set()
lock = threading.Lock()
last_stats = None
last_mtime = 0.0
trainer_online = False


def load_config():
    global board_fps, stats_path
    path = Path(__file__).with_name("config.yaml")
    if path.exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        dash = data.get("dashboard", {})
        board_fps = int(dash.get("board_fps", 5))
        stats_path = dash.get("stats_path", "runs/live_stats.json")


def read_stats():
    global last_stats, last_mtime, trainer_online
    try:
        p = Path(stats_path)
        if not p.exists():
            trainer_online = False
            return None
        mtime = p.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            with open(p, "r") as f:
                last_stats = json.load(f)
            trainer_online = True
        elif time.time() - mtime > 2.0:
            trainer_online = False
    except Exception:
        trainer_online = False
    return last_stats


def broadcaster():
    interval = 1.0 / max(board_fps, 1)
    while True:
        stats = read_stats()
        payload = json.dumps(stats) if stats else "null"
        msg = f"data: {payload}\n\n"
        dead = []
        with lock:
            for q in list(clients):
                try:
                    q.put_nowait(msg)
                except Exception:
                    dead.append(q)
            for q in dead:
                clients.discard(q)
        time.sleep(interval)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/stats")
def api_stats():
    stats = read_stats()
    if stats is None:
        return jsonify({"trainer_online": False})
    return jsonify(stats)


@app.route("/events")
def events():
    q = queue.Queue(maxsize=10)
    with lock:
        clients.add(q)
    def gen():
        try:
            while True:
                yield q.get()
        finally:
            with lock:
                clients.discard(q)
    return Response(gen(), mimetype="text/event-stream")


def main():
    load_config()
    t = threading.Thread(target=broadcaster, daemon=True)
    t.start()
    host = "0.0.0.0"
    port = 8080
    try:
        with open(Path(__file__).with_name("config.yaml"), "r") as f:
            data = yaml.safe_load(f) or {}
        dash = data.get("dashboard", {})
        host = dash.get("host", host)
        port = int(dash.get("port", port))
    except Exception:
        pass
    print(f"[dashboard] http://{host}:{port}/")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
