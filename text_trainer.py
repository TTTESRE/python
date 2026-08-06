#!/usr/bin/env python3
"""
2D Bipedal Walker Trainer with OpenCL-accelerized Policy Network
- Uses BipedalWalker-v3 environment
- PPO (Proximal Policy Optimization) algorithm
- Custom OpenCL kernel via opencl_ocl extension compiled against libtorch
"""

import argparse
import sys
import os
import time
import signal
import subprocess
import yaml
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import gymnasium as gym
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import opencl_ocl


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    defaults = {
        "env": "BipedalWalker-v3",
        "max_episodes": 5000,
        "max_steps": 1600,
        "update_interval": 2048,
        "mini_batch_size": 64,
        "epochs_per_update": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip": 0.2,
        "entropy_coef": 0.01,
        "vf_coef": 0.5,
        "laser_range": 2.0,
        "laser_speed": 0.5,
        "laser_activate_after": 100,
        "hidden": 128,
        "opencl_min_elements": 8192,
        "compute_chain": ["opencl", "aten"],
        "fps": 30,
        "record_video_container": "webm",
    }
    if os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if k == "compute_chain" and isinstance(v, str):
                defaults[k] = [s.strip() for s in v.split(",") if s.strip()]
            elif v is not None:
                defaults[k] = v
    return defaults


class OpenCLLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        out = opencl_ocl.linear_forward(weight.detach(), bias.detach(), input.detach())
        ctx.save_for_backward(input, weight, bias)
        return out.to(device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = opencl_ocl.linear_backward(grad_output.detach(), weight.detach(), input.detach())
        return grad_input.to(device=input.device, dtype=input.dtype), grad_weight.to(device=weight.device, dtype=weight.dtype), grad_bias.to(device=bias.device, dtype=bias.dtype)


class OpenCLReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = opencl_ocl.relu_forward(input.detach())
        ctx.save_for_backward(input)
        return out.to(device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = opencl_ocl.relu_backward(grad_output.detach(), input.detach())
        return grad_input.to(device=input.device, dtype=input.dtype)


class OpenCLTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = opencl_ocl.tanh_forward(input.detach())
        ctx.save_for_backward(input, out)
        return out.to(device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = opencl_ocl.tanh_backward(grad_output.detach(), output.detach())
        return grad_input.to(device=input.device, dtype=input.dtype)


class OpenCLLinearReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        out = opencl_ocl.linear_relu_forward(weight.detach(), bias.detach(), input.detach())
        ctx.save_for_backward(input, weight, bias)
        return out.to(device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        pre = input @ weight.t() + bias
        mask = (pre > 0).float()
        g = grad_output * mask
        grad_input, grad_weight, grad_bias = opencl_ocl.linear_backward(g.detach(), weight.detach(), input.detach())
        return grad_input.to(device=input.device, dtype=input.dtype), grad_weight.to(device=weight.device, dtype=weight.dtype), grad_bias.to(device=bias.device, dtype=bias.dtype)


class OpenCLLinearTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        out = opencl_ocl.linear_tanh_forward(weight.detach(), bias.detach(), input.detach())
        ctx.save_for_backward(input, weight, bias, out)
        return out.to(device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        d = grad_output * (1 - output ** 2)
        grad_input, grad_weight, grad_bias = opencl_ocl.linear_backward(d.detach(), weight.detach(), input.detach())
        return grad_input.to(device=input.device, dtype=input.dtype), grad_weight.to(device=weight.device, dtype=weight.dtype), grad_bias.to(device=bias.device, dtype=bias.dtype)


# Only dispatch to OpenCL when the op is big enough to amortise the
# host<->device copies. Otherwise fall straight through to ATen (CPU), which is
# faster for small matrices.
_OCL_MIN_ELEMENTS = 8192
_compute_chain = ["opencl", "aten"]


def set_compute_chain(chain):
    global _compute_chain
    if isinstance(chain, str):
        chain = [s.strip() for s in chain.split(",") if s.strip()]
    _compute_chain = list(chain)


def _try_opencl_linear(x, w, b):
    if opencl_ocl.is_available() and x.numel() * w.shape[0] >= _OCL_MIN_ELEMENTS:
        return OpenCLLinear.apply(x, w, b)
    return None


def _try_opencl_linear_relu(x, w, b):
    if opencl_ocl.is_available() and x.numel() * w.shape[0] >= _OCL_MIN_ELEMENTS:
        return OpenCLLinearReLU.apply(x, w, b)
    return None


def _try_opencl_linear_tanh(x, w, b):
    if opencl_ocl.is_available() and x.numel() * w.shape[0] >= _OCL_MIN_ELEMENTS:
        return OpenCLLinearTanh.apply(x, w, b)
    return None


def _try_opencl_relu(x):
    if opencl_ocl.is_available() and x.numel() >= _OCL_MIN_ELEMENTS:
        return OpenCLReLU.apply(x)
    return None


def _try_opencl_tanh(x):
    if opencl_ocl.is_available() and x.numel() >= _OCL_MIN_ELEMENTS:
        return OpenCLTanh.apply(x)
    return None


def ocl_linear(x, w, b):
    for backend in _compute_chain:
        if backend == "opencl":
            r = _try_opencl_linear(x, w, b)
            if r is not None:
                return r
        elif backend in ("aten", "cpu"):
            return torch.nn.functional.linear(x, w, b)
    return torch.nn.functional.linear(x, w, b)


def ocl_relu(x):
    for backend in _compute_chain:
        if backend == "opencl":
            r = _try_opencl_relu(x)
            if r is not None:
                return r
        elif backend in ("aten", "cpu"):
            return torch.nn.functional.relu(x)
    return torch.nn.functional.relu(x)


def ocl_tanh(x):
    for backend in _compute_chain:
        if backend == "opencl":
            r = _try_opencl_tanh(x)
            if r is not None:
                return r
        elif backend in ("aten", "cpu"):
            return torch.tanh(x)
    return torch.tanh(x)


def ocl_linear_relu(x, w, b):
    for backend in _compute_chain:
        if backend == "opencl":
            r = _try_opencl_linear_relu(x, w, b)
            if r is not None:
                return r
        elif backend in ("aten", "cpu"):
            return torch.nn.functional.relu(torch.nn.functional.linear(x, w, b))
    return torch.nn.functional.relu(torch.nn.functional.linear(x, w, b))


def ocl_linear_tanh(x, w, b):
    for backend in _compute_chain:
        if backend == "opencl":
            r = _try_opencl_linear_tanh(x, w, b)
            if r is not None:
                return r
        elif backend in ("aten", "cpu"):
            return torch.tanh(torch.nn.functional.linear(x, w, b))
    return torch.tanh(torch.nn.functional.linear(x, w, b))


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, act_dim)
        self.fc_logstd = nn.Parameter(torch.zeros(act_dim))
        self.act_dim = act_dim

    def forward(self, x):
        x = ocl_linear_relu(x, self.fc1.weight, self.fc1.bias)
        x = ocl_linear_relu(x, self.fc2.weight, self.fc2.bias)
        mean = ocl_linear_tanh(x, self.fc_mean.weight, self.fc_mean.bias)
        return mean


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = ocl_linear_relu(x, self.fc1.weight, self.fc1.bias)
        x = ocl_linear_relu(x, self.fc2.weight, self.fc2.bias)
        return ocl_linear(x, self.fc3.weight, self.fc3.bias)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, hidden=128, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, entropy_coef=0.01, vf_coef=0.5):
        self.device = torch.device("cpu")
        self.policy = PolicyNet(obs_dim, act_dim, hidden=hidden).to(self.device)
        self.value = ValueNet(obs_dim, hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            mean = self.policy(obs)
            std = torch.exp(self.policy.fc_logstd)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            logp = dist.log_prob(action).sum(dim=-1)
            value = self.value(obs).squeeze(-1)
        action_np = action.cpu().numpy()
        logp_np = logp.cpu().numpy()
        value_np = value.cpu().numpy()
        if action_np.shape[0] == 1:
            return action_np[0], logp_np[0], value_np[0]
        return action_np, logp_np, value_np

    def compute_returns(self, trajectories, last_value):
        """Generalized Advantage Estimation (GAE-lambda).

        `trajectories` is a list of dicts each containing 'rew', 'done' and 'val'
        (the value estimate at that step). `last_value` is the bootstrap value of
        the state following the last step (0.0 if the last step terminated).

        Returns (returns, advantages) lists aligned with `trajectories`.
        Episode boundaries are respected via the `done` flags so advantages and
        returns never leak across episodes.
        """
        T = len(trajectories)
        returns = [0.0] * T
        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
                next_done = trajectories[t]["done"]
            else:
                next_val = trajectories[t + 1]["val"]
                next_done = trajectories[t]["done"]
            nonterminal = 0.0 if next_done else 1.0
            delta = trajectories[t]["rew"] + self.gamma * next_val * nonterminal - trajectories[t]["val"]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae
            returns[t] = gae + trajectories[t]["val"]
        return returns, advantages

    def update(self, trajectories):
        if len(trajectories) == 0:
            return 0.0, 0.0, 0.0
        obs = torch.as_tensor(np.stack([t["obs"] for t in trajectories]), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(np.stack([t["act"] for t in trajectories]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([t["logp"] for t in trajectories], dtype=torch.float32, device=self.device).unsqueeze(1)
        returns = torch.as_tensor([t["ret"] for t in trajectories], dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages = torch.as_tensor([t["adv"] for t in trajectories], dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean = self.policy(obs)
        std = torch.exp(self.policy.fc_logstd)
        dist = torch.distributions.Normal(mean, std)
        new_logp = dist.log_prob(acts).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (returns - self.value(obs)).pow(2).mean()

        loss = policy_loss - self.entropy_coef * entropy + self.vf_coef * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.optimizer.step()
        opencl_ocl.invalidate_params()
        return policy_loss.item(), value_loss.item(), entropy.item()


class LaserHazardWrapper(gym.Wrapper):
    def __init__(self, env, laser_speed=0.5, laser_range=2.0, laser_activate_after=100):
        super().__init__(env)
        self.laser_pos = -5
        self._base_laser_speed = laser_speed
        self.laser_speed = laser_speed
        self.laser_range = laser_range
        self.laser_activate_after = laser_activate_after
        self.laser_direction = 1.0
        self.laser_active = False
        self.step_count = 0
        self.current_score = 0.0
        self.high_score = -float("inf")
        self._render_window_created = False
        self._clock = pygame.time.Clock()
        self.headless = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.laser_pos = -5
        self.laser_speed = self._base_laser_speed
        self.laser_direction = 1.0
        self.laser_active = False
        self.step_count = 0
        self.current_score = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.current_score += reward

        if self.step_count > self.laser_activate_after:
            self.laser_active = True

        if self.laser_active:
            self.laser_pos += self.laser_speed * self.laser_direction
            if abs(self.laser_pos) > self.laser_range:
                self.laser_direction *= -1.0
                self.laser_pos = np.clip(self.laser_pos, -self.laser_range, self.laser_range)

            self.laser_speed += 0.002

            walker_x = obs[0] if len(obs) > 0 else 0.0
            distance = abs(walker_x - self.laser_pos)

            if distance < 0.5:
                reward -= 10.0
                terminated = True
                info["laser_hit"] = True
            else:
                info["laser_hit"] = False
                reward -= 0.01 * self.laser_speed

        info["laser_pos"] = self.laser_pos
        info["laser_speed"] = self.laser_speed
        info["laser_active"] = self.laser_active

        if terminated or truncated:
            if self.current_score > self.high_score:
                self.high_score = self.current_score

        return obs, reward, terminated, truncated, info

    def render(self):
        result = self.env.render()
        if result is None:
            return result
        frame = result.copy()
        if self.laser_active:
            try:
                h, w = frame.shape[:2]
                laser_x = int((self.laser_pos / self.laser_range + 1.0) * 0.5 * w)
                laser_x = int(np.clip(laser_x, 0, w - 1))
                y_coords = np.arange(h)
                frame[y_coords, laser_x, 0] = 0
                frame[y_coords, laser_x, 1] = 0
                frame[y_coords, laser_x, 2] = 255
                if laser_x > 0:
                    frame[y_coords, laser_x - 1, 2] = 200
                if laser_x < w - 1:
                    frame[y_coords, laser_x + 1, 2] = 200
            except Exception:
                pass
        if self.headless:
            return frame
        try:
            h, w = frame.shape[:2]
            if not self._render_window_created:
                pygame.init()
                self._screen = pygame.display.set_mode((w, h))
                self._render_window_created = True
            pygame.display.set_caption("BipedalWalker - Laser Training")
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    sys.exit(0)
        except Exception:
            pass
        return frame


class WalkerTrainer:
    def __init__(self, mode="runtrain", video_path=None, container="webm", config_path=None):
        cfg = load_config(config_path)
        self.env_name = cfg.get("env", "BipedalWalker-v3")
        self.max_episodes = int(cfg.get("max_episodes", 5000))
        self.max_steps = int(cfg.get("max_steps", 1600))
        self.update_interval = int(cfg.get("update_interval", 2048))
        self.mini_batch_size = int(cfg.get("mini_batch_size", 64))
        self.epochs_per_update = int(cfg.get("epochs_per_update", 10))
        self.mode = mode
        self.fix_mode = mode.startswith("fix")
        self.headless = mode in ("run", "fixrun", "train", "fixtrain")
        self.record_video = mode in ("run", "fixrun")
        self.num_envs = int(cfg.get("num_envs", 1))
        if self.record_video:
            self.num_envs = 1
        compute_chain = cfg.get("compute_chain", ["opencl", "aten"])
        if isinstance(compute_chain, str):
            compute_chain = [s.strip() for s in compute_chain.split(",") if s.strip()]
        self.compute_chain = compute_chain
        set_compute_chain(compute_chain)
        self.render = mode in ("run", "runtrain", "fixrun", "fixruntrain", "fixtrain")
        self.video_path = video_path or ("run_inference.webm" if container == "webm" else "run_inference.mp4")
        self.container = container
        self.ffmpeg_proc = None
        self.fixer_proc = None
        self.fixer_opencl_proc = None

        def make_env():
            e = gym.make(self.env_name, render_mode="rgb_array")
            e = LaserHazardWrapper(
                e,
                laser_speed=float(cfg.get("laser_speed", 0.5)),
                laser_range=float(cfg.get("laser_range", 2.0)),
                laser_activate_after=int(cfg.get("laser_activate_after", 100))
            )
            e.headless = self.headless
            return e

        if self.num_envs > 1:
            try:
                self.env = gym.vector.SyncVectorEnv([make_env for _ in range(self.num_envs)])
            except Exception as e:
                print(f"[vec] SyncVectorEnv failed ({e}), falling back to single env")
                self.num_envs = 1
                self.env = make_env()
        else:
            self.env = make_env()

        obs_space = self.env.observation_space.shape
        act_space = self.env.action_space.shape
        self.obs_dim = obs_space[-1]
        self.act_dim = act_space[-1]

        hidden = int(cfg.get("hidden", 128))
        self.agent = PPOAgent(
            self.obs_dim,
            self.act_dim,
            hidden=hidden,
            lr=float(cfg.get("lr", 3e-4)),
            gamma=float(cfg.get("gamma", 0.99)),
            lam=float(cfg.get("lam", 0.95)),
            clip=float(cfg.get("clip", 0.2)),
            entropy_coef=float(cfg.get("entropy_coef", 0.01)),
            vf_coef=float(cfg.get("vf_coef", 0.5)),
        )
        global _OCL_MIN_ELEMENTS
        _OCL_MIN_ELEMENTS = int(cfg.get("opencl_min_elements", 8192))

        self.best_reward = -float("inf")
        self.episode = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        self.checkpoint_path = "walker_checkpoint.pt"

        self._init_logging()

        if self.fix_mode:
            self._launch_fixer()

    def _init_logging(self):
        self.recent_rewards = []
        self.log_path = "training_log.csv"
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir="runs/walker")
            print("[log] TensorBoard enabled at runs/walker")
        except Exception:
            self.writer = None
        try:
            with open(self.log_path, "w") as f:
                f.write("episode,reward,steps,laser_speed,best_reward\n")
        except Exception:
            pass

    def _log_episode(self, episode, reward, steps, laser_speed):
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)
        avg = sum(self.recent_rewards) / len(self.recent_rewards)
        try:
            with open(self.log_path, "a") as f:
                f.write(f"{episode},{reward:.2f},{steps},{laser_speed:.2f},{self.best_reward:.2f}\n")
        except Exception:
            pass
        if self.writer is not None:
            try:
                self.writer.add_scalar("reward/episode", reward, episode)
                self.writer.add_scalar("reward/avg50", avg, episode)
                self.writer.add_scalar("laser/speed", laser_speed, episode)
            except Exception:
                pass
        return avg

    def _stop_logging(self):
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None

    def _launch_fixer(self):
        here = os.path.dirname(os.path.abspath(__file__))
        fixer_bin = os.path.join(here, "fixer")
        if not os.path.exists(fixer_bin):
            print(f"[fixer] binary not found at {fixer_bin}, skipping fix mode")
        else:
            try:
                self.fixer_proc = subprocess.Popen(
                    [fixer_bin, self.mode, str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[fixer] launched fixer.cc (PID {self.fixer_proc.pid}) attached to {os.getpid()}")
            except Exception as e:
                print(f"[fixer] failed to launch: {e}")

        fixer_opencl_bin = os.path.join(here, "fixer_opencl")
        if not os.path.exists(fixer_opencl_bin):
            print(f"[fixer-opencl] binary not found at {fixer_opencl_bin}, skipping")
        else:
            try:
                self.fixer_opencl_proc = subprocess.Popen(
                    [fixer_opencl_bin, self.mode, str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[fixer-opencl] launched (PID {self.fixer_opencl_proc.pid}) attached to {os.getpid()}")
            except Exception as e:
                print(f"[fixer-opencl] failed to launch: {e}")

    def _start_video(self, frame):
        if not self.record_video or self.ffmpeg_proc is not None:
            return
        h, w = frame.shape[:2]
        codec = "libvpx-vp9" if self.container == "webm" else "libx264"
        extra = ["-b:v", "1M"] if self.container == "webm" else ["-pix_fmt", "yuv420p"]
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", "30",
            "-i", "-",
            "-an", "-c:v", codec, "-crf", "18",
        ] + extra + [self.video_path]
        try:
            self.ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            print(f"[video] recording to {self.video_path} ({w}x{h})")
        except Exception as e:
            print(f"[video] failed to start ffmpeg: {e}")
            self.ffmpeg_proc = None

    def _write_frame(self, frame):
        if self.ffmpeg_proc is not None and frame is not None:
            try:
                self.ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())
            except Exception:
                pass

    def _stop_video(self):
        if self.ffmpeg_proc is not None:
            try:
                self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.wait(timeout=30)
            except Exception:
                try:
                    self.ffmpeg_proc.kill()
                except Exception:
                    pass
            self.ffmpeg_proc = None
            print(f"[video] saved {self.video_path}")

    def _stop_fixer(self):
        if self.fixer_proc is not None:
            try:
                self.fixer_proc.terminate()
                self.fixer_proc.wait(timeout=5)
            except Exception:
                try:
                    self.fixer_proc.kill()
                except Exception:
                    pass
            self.fixer_proc = None
        if self.fixer_opencl_proc is not None:
            try:
                self.fixer_opencl_proc.terminate()
                self.fixer_opencl_proc.wait(timeout=5)
            except Exception:
                try:
                    self.fixer_opencl_proc.kill()
                except Exception:
                    pass
            self.fixer_opencl_proc = None

    def _signal_handler(self, signum, frame):
        self._stop_video()
        self._stop_fixer()
        self._stop_logging()
        self.save()
        print("\nSaved checkpoint and exiting.")
        sys.exit(0)

    def save(self):
        torch.save({
            "policy": self.agent.policy.state_dict(),
            "value": self.agent.value.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "episode": self.episode,
            "best_reward": self.best_reward,
        }, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load(self):
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.value.load_state_dict(ckpt["value"])
            self.agent.optimizer.load_state_dict(ckpt["optimizer"])
            self.episode = ckpt.get("episode", 0)
            self.best_reward = ckpt.get("best_reward", -float("inf"))
            print(f"Loaded checkpoint from episode {self.episode}, best reward: {self.best_reward:.2f}")

    def train(self):
        if self.mode == "run":
            self._run_inference()
            return
        self.load()
        obs, _ = self.env.reset()

        print(f"OpenCL available: {opencl_ocl.is_available()}")
        print(f"Environment: {self.env_name}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.act_dim}")
        print(f"Num envs: {self.num_envs}")
        print(f"Starting training from episode {self.episode}...")

        try:
            if self.num_envs <= 1:
                self._train_single(obs)
            else:
                self._train_vectorized(obs)
        except KeyboardInterrupt:
            pass

        self._stop_video()
        self._stop_fixer()
        self._stop_logging()
        self.save()
        print("Training finished.")

    def _train_single(self, obs):
        episode_reward = 0.0
        episode_steps = 0
        trajectories = []
        total_steps = 0

        while self.episode < self.max_episodes:
            action, logp, value = self.agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if self.render:
                frame = self.env.render()
                if self.record_video and frame is not None:
                    if self.ffmpeg_proc is None:
                        self._start_video(np.asarray(frame))
                    self._write_frame(np.asarray(frame))

            trajectories.append({
                "obs": obs.astype(np.float32),
                "act": action.astype(np.float32),
                "logp": float(logp),
                "rew": float(reward),
                "val": float(value),
                "done": done,
            })

            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            total_steps += 1

            if done or episode_steps >= self.max_steps:
                self.episode += 1
                laser_info = ""
                laser_speed = 0.0
                if hasattr(self.env, 'env') and hasattr(self.env.env, 'laser_speed'):
                    laser_speed = self.env.env.laser_speed
                    laser_info = f" | Laser spd={laser_speed:.2f}"
                avg = self._log_episode(self.episode, episode_reward, episode_steps, laser_speed)
                print(f"Episode {self.episode:4d} | Reward: {episode_reward:8.2f} | Avg50: {avg:7.2f} | Steps: {episode_steps:4d}{laser_info}")

                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    torch.save({
                        "policy": self.agent.policy.state_dict(),
                        "value": self.agent.value.state_dict(),
                        "optimizer": self.agent.optimizer.state_dict(),
                        "episode": self.episode,
                        "best_reward": self.best_reward,
                    }, "best_walker.pt")
                    print(f"  New best reward! Saved to best_walker.pt")

                if episode_reward >= 300.0:
                    print(f"Solved in episode {self.episode}!")
                    break

                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_steps = 0

            if len(trajectories) >= self.update_interval:
                last_value = 0.0
                if not done:
                    _, _, last_value = self.agent.get_action(obs)
                rets, advs = self.agent.compute_returns(trajectories, last_value)
                for t, r, a in zip(trajectories, rets, advs):
                    t["ret"] = r
                    t["adv"] = a

                t0 = time.perf_counter()
                for _ in range(self.epochs_per_update):
                    np.random.shuffle(trajectories)
                    for start in range(0, len(trajectories), self.mini_batch_size):
                        batch = trajectories[start:start + self.mini_batch_size]
                        self.agent.update(batch)
                dt = time.perf_counter() - t0
                print(f"  [update] {self.epochs_per_update} epochs over {len(trajectories)} steps in {dt:.2f}s")
                trajectories.clear()

    def _train_vectorized(self, obs):
        ep_rewards = np.zeros(self.num_envs)
        ep_steps = np.zeros(self.num_envs, dtype=int)
        trajectories = [[] for _ in range(self.num_envs)]
        total_steps = 0

        while self.episode < self.max_episodes:
            actions, logps, values = self.agent.get_action(obs)
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            if self.render:
                frame = self.env.render()
                if self.record_video and frame is not None:
                    if self.ffmpeg_proc is None:
                        self._start_video(np.asarray(frame))
                    self._write_frame(np.asarray(frame))

            for i in range(self.num_envs):
                trajectories[i].append({
                    "obs": obs[i].astype(np.float32),
                    "act": actions[i].astype(np.float32),
                    "logp": float(logps[i]),
                    "rew": float(rewards[i]),
                    "val": float(values[i]),
                    "done": bool(dones[i]),
                })

            ep_rewards += rewards
            ep_steps += 1
            total_steps += self.num_envs
            obs = next_obs

            for i in range(self.num_envs):
                if dones[i]:
                    self.episode += 1
                    laser_info = ""
                    laser_speed = 0.0
                    if hasattr(self.env, 'envs') and i < len(self.env.envs):
                        env_i = self.env.envs[i]
                        if hasattr(env_i, 'env') and hasattr(env_i.env, 'laser_speed'):
                            laser_speed = env_i.env.laser_speed
                            laser_info = f" | Laser spd={laser_speed:.2f}"
                    avg = self._log_episode(self.episode, float(ep_rewards[i]), int(ep_steps[i]), laser_speed)
                    print(f"Episode {self.episode:4d} | Reward: {ep_rewards[i]:8.2f} | Avg50: {avg:7.2f} | Steps: {ep_steps[i]:4d}{laser_info}")

                    if ep_rewards[i] > self.best_reward:
                        self.best_reward = ep_rewards[i]
                        torch.save({
                            "policy": self.agent.policy.state_dict(),
                            "value": self.agent.value.state_dict(),
                            "optimizer": self.agent.optimizer.state_dict(),
                            "episode": self.episode,
                            "best_reward": self.best_reward,
                        }, "best_walker.pt")
                        print(f"  New best reward! Saved to best_walker.pt")

                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0

            total_in_buffer = sum(len(t) for t in trajectories)
            if total_in_buffer >= self.update_interval:
                flat_trajectories = []
                for traj in trajectories:
                    flat_trajectories.extend(traj)

                last_values = np.zeros(self.num_envs)
                for i in range(self.num_envs):
                    if trajectories[i] and not trajectories[i][-1]["done"]:
                        _, _, lv = self.agent.get_action(obs[i])
                        last_values[i] = float(lv)

                for i, traj in enumerate(trajectories):
                    if len(traj) == 0:
                        continue
                    rets, advs = self.agent.compute_returns(traj, last_values[i])
                    for t, r, a in zip(traj, rets, advs):
                        t["ret"] = r
                        t["adv"] = a

                t0 = time.perf_counter()
                for _ in range(self.epochs_per_update):
                    np.random.shuffle(flat_trajectories)
                    for start in range(0, len(flat_trajectories), self.mini_batch_size):
                        batch = flat_trajectories[start:start + self.mini_batch_size]
                        self.agent.update(batch)
                dt = time.perf_counter() - t0
                print(f"  [update] {self.epochs_per_update} epochs over {len(flat_trajectories)} steps in {dt:.2f}s")

                trajectories = [[] for _ in range(self.num_envs)]

    def _run_inference(self):
        ckpt_path = "best_walker.pt"
        if not os.path.exists(ckpt_path):
            print("No checkpoint found at best_walker.pt. Run --train first.")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.value.load_state_dict(ckpt["value"])
        except RuntimeError as e:
            print(f"[run] Architecture changed ({e}); cannot run inference with mismatched model")
            return
        print(f"Loaded best checkpoint (best reward: {ckpt.get('best_reward', 'N/A')})")

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode = 0
        max_inference_episodes = 10

        print(f"Running inference for {max_inference_episodes} episodes...")
        try:
            while episode < max_inference_episodes:
                action, logp, value = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                frame = self.env.render()
                if self.record_video and frame is not None:
                    if self.ffmpeg_proc is None:
                        self._start_video(np.asarray(frame))
                    self._write_frame(np.asarray(frame))

                episode_reward += reward
                episode_steps += 1
                obs = next_obs

                if done or episode_steps >= self.max_steps:
                    episode += 1
                    print(f"Episode {episode:4d} | Reward: {episode_reward:8.2f} | Steps: {episode_steps:4d}")
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
        except KeyboardInterrupt:
            pass
        self._stop_video()
        self._stop_fixer()
        print("Inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BipedalWalker PPO Trainer")
    parser.add_argument("--run", action="store_true", help="Headless run: record ffmpeg video (webm) of inference")
    parser.add_argument("--train", action="store_true", help="Headless training without rendering")
    parser.add_argument("--runtrain", action="store_true", help="Trial and error with GUI rendering")
    parser.add_argument("--fixrun", action="store_true", help="Headless run with fixer.cc forcing render + video")
    parser.add_argument("--fixtrain", action="store_true", help="Headless training with fixer.cc forcing render")
    parser.add_argument("--fixruntrain", action="store_true", help="GUI training with fixer.cc forcing every frame")
    parser.add_argument("--video", type=str, default=None, help="Output video path for --run/--fixrun")
    parser.add_argument("--mp4", action="store_true", help="Use mp4 container instead of webm for video output")
    args = parser.parse_args()

    if args.fixrun:
        mode = "fixrun"
    elif args.fixtrain:
        mode = "fixtrain"
    elif args.fixruntrain:
        mode = "fixruntrain"
    elif args.train:
        mode = "train"
    elif args.run:
        mode = "run"
    else:
        mode = "runtrain"

    container = "mp4" if args.mp4 else "webm"
    trainer = WalkerTrainer(mode=mode, video_path=args.video, container=container)
    trainer.train()