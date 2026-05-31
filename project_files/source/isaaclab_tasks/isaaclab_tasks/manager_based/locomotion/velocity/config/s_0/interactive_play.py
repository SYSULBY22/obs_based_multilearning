from __future__ import annotations

"""Interactive play script for S0 multitask policy.

Run policy inference and switch task commands online from terminal:
  - forward
  - right
  - left
  - random
  - quit
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Interactive play for S0 multitask locomotion.")
parser.add_argument("--task", type=str, required=True, help="Gym task id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument(
    "--command-term",
    type=str,
    default="base_velocity",
    help="Command term name in command manager (default: base_velocity).",
)
# App launcher args (device/headless/livestream/enable_cameras/etc.)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Runtime imports after app startup
# -----------------------------------------------------------------------------
import gymnasium as gym
import torch
import importlib.metadata as metadata
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# -----------------------------------------------------------------------------
# LLM command extraction
# -----------------------------------------------------------------------------
# Local default key (DEEPSEEK_API_KEY env var has higher priority)
DEFAULT_DEEPSEEK_API_KEY = "your key"
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"


def _extract_cmd_from_text(text: str) -> str | None:
    valid_cmds = {"forward", "right", "left", "random", "quit"}
    t = text.strip().lower()
    if not t:
        return None
    if t in valid_cmds:
        return t

    # Rule-based fallback (fast path for Chinese instructions)
    rule_map = [
        (("退出", "结束", "quit"), "quit"),
        (("随机", "随便", "都可以"), "random"),
        (("右转", "向右", "往右", "右拐"), "right"),
        (("左转", "向左", "往左", "左拐"), "left"),
        (("前进", "向前", "往前", "全部向前"), "forward"),
    ]
    for keys, cmd in rule_map:
        if any(k in text for k in keys):
            return cmd

    # LLM extraction path
    api_key = os.environ.get("DEEPSEEK_API_KEY", DEFAULT_DEEPSEEK_API_KEY).strip()
    if not api_key:
        return None

    system_prompt = (
        "你是机器人控制指令解析器。"
        "把用户输入映射到唯一命令之一：forward/right/left/random/quit。"
        "只输出JSON：{\"command\":\"<one_of_5>\"}。不要输出其他内容。"
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    req = urllib.request.Request(
        url=f"{DEEPSEEK_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(body)
            content = obj["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
        return None

    # Prefer JSON answer
    try:
        content_obj = json.loads(content)
        cmd = str(content_obj.get("command", "")).strip().lower()
        if cmd in valid_cmds:
            return cmd
    except json.JSONDecodeError:
        pass

    # Fallback: parse plain text token
    m = re.search(r"\b(forward|right|left|random|quit)\b", content.lower())
    if m:
        return m.group(1)
    return None


def _input_worker(cmd_queue: queue.Queue[str], stop_event: threading.Event):
    prompt = (
        "\n[interactive] 输入命令并回车（支持英文或自然语言）:\n"
        "  forward | right | left | random | quit\n"
        "  例如：全部向前移动！ / 往右转 / 向左\n> "
    )
    print(prompt, end="", flush=True)
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            cmd_queue.put("quit")
            return
        raw = line.strip()
        if raw:
            cmd_queue.put(raw)
        if raw.lower() == "quit":
            return
        print("> ", end="", flush=True)


def _apply_skill_command(raw_env, command_term_name: str, command: str):
    """Update command term mode and force immediate resample for all envs."""
    term = raw_env.command_manager.get_term(command_term_name)
    if not hasattr(term, "cfg") or not hasattr(term.cfg, "fixed_task"):
        raise RuntimeError(
            f"Command term '{command_term_name}' does not expose cfg.fixed_task. "
            "Use this script with S0 flat_multitask command term."
        )
    term.cfg.fixed_task = command
    # Force command update now instead of waiting for next scheduled resample.
    env_ids = list(range(raw_env.num_envs))
    term._resample(env_ids)  # noqa: SLF001 (intentional: interactive control hook)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Basic overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Force non-random at startup for deterministic first command.
    # You can switch later from terminal.
    # This expects your multitask cfg to expose: env.commands.base_velocity.fixed_task
    env_cfg.commands.base_velocity.fixed_task = "forward"

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped

    # Wrap for RSL-RL and load checkpoint
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=raw_env.device)
    if version.parse(installed_version) >= version.parse("4.0.0"):
        policy_reset_fn = policy.reset
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic
        policy_reset_fn = policy_nn.reset

    # Interactive terminal thread
    cmd_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(target=_input_worker, args=(cmd_queue, stop_event), daemon=True)
    input_thread.start()

    valid_cmds = {"forward", "right", "left", "random", "quit"}
    dt = raw_env.step_dt

    # Initial reset
    obs = env.get_observations()
    _apply_skill_command(raw_env, args_cli.command_term, "forward")
    print("[interactive] Initial command: forward")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            # Apply latest command from queue (drain all pending; keep newest)
            latest_cmd = None
            while not cmd_queue.empty():
                latest_cmd = cmd_queue.get_nowait()
            if latest_cmd is not None:
                parsed_cmd = _extract_cmd_from_text(latest_cmd)
                if parsed_cmd is None:
                    print(
                        f"[interactive] 无法解析输入: '{latest_cmd}'。"
                        "可输入 forward/right/left/random/quit 或中文自然语言。"
                    )
                elif parsed_cmd not in valid_cmds:
                    print(f"[interactive] 解析结果非法: {parsed_cmd}")
                elif parsed_cmd == "quit":
                    print("[interactive] Quit requested.")
                    break
                else:
                    _apply_skill_command(raw_env, args_cli.command_term, parsed_cmd)
                    print(f"[interactive] 已执行输入: '{latest_cmd}'")

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_reset_fn(dones)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
