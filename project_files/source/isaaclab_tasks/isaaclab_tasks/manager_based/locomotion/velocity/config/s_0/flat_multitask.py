from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import S0FlatEnvCfg


class MultiSkillVelocityCommand(CommandTerm):
    """Sample one of three locomotion skills and output SE(2) velocity command."""

    cfg: "MultiSkillVelocityCommandCfg"

    def __init__(self, cfg: "MultiSkillVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_onehot = torch.zeros(self.num_envs, self.cfg.task_obs_dim, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max(max_command_time / self._env.step_dt, 1.0)
        robot = self._env.scene[self.cfg.asset_name]
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        task_to_idx = {"forward": 0, "right": 1, "left": 2}
        if self.cfg.fixed_task == "random":
            sampled_task = torch.randint(0, 3, (len(env_ids_t),), device=self.device)
        else:
            if self.cfg.fixed_task not in task_to_idx:
                raise ValueError(
                    f"Invalid fixed_task='{self.cfg.fixed_task}'. "
                    "Expected one of ['random', 'forward', 'right', 'left']."
                )
            sampled_task = torch.full(
                (len(env_ids_t),), task_to_idx[self.cfg.fixed_task], dtype=torch.long, device=self.device
            )

        command_bank = torch.tensor(
            [
                self.cfg.forward_command,
                self.cfg.right_turn_command,
                self.cfg.left_turn_command,
            ],
            dtype=torch.float,
            device=self.device,
        )

        self.vel_command_b[env_ids_t] = command_bank[sampled_task]
        self.task_ids[env_ids_t] = sampled_task
        self.task_onehot[env_ids_t] = 0.0
        self.task_onehot[env_ids_t, sampled_task] = 1.0

    def _update_command(self):
        # Commands are directly sampled from the skill bank.
        return


@configclass
class MultiSkillVelocityCommandCfg(CommandTermCfg):
    """Config for 3-skill locomotion commands with one-hot task embedding."""

    class_type: type = MultiSkillVelocityCommand
    asset_name: str = MISSING

    # Keep one command per episode by default (episode length is 20s in this task family).
    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    debug_vis: bool = False

    # Output one-hot task embedding length.
    task_obs_dim: int = 10
    # Command selection mode:
    # - "random": default multitask sampling (1/3 each)
    # - "forward" / "right" / "left": force a single skill (useful for play)
    fixed_task: str = "random"

    # (lin_x [m/s], lin_y [m/s], ang_z [rad/s])
    forward_command: tuple[float, float, float] = (0.50, 0.0, 0.0)
    right_turn_command: tuple[float, float, float] = (0.20, 0.0, -0.60)
    left_turn_command: tuple[float, float, float] = (0.20, 0.0, 0.60)


def task_instruction_obs(env, command_name: str = "base_velocity", task_obs_dim: int = 10) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "task_onehot"):
        return term.task_onehot[:, :task_obs_dim]
    return torch.zeros((env.num_envs, task_obs_dim), device=env.device)


def _task_mask(env, task_idx: int, command_name: str = "base_velocity") -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "task_ids"):
        return (term.task_ids == task_idx).float()
    return torch.zeros(env.num_envs, device=env.device)


def task_track_lin_vel_xy_exp(
    env,
    task_idx: int,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _task_mask(env, task_idx, command_name) * mdp.track_lin_vel_xy_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )


def task_track_ang_vel_z_exp(
    env,
    task_idx: int,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _task_mask(env, task_idx, command_name) * mdp.track_ang_vel_z_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )


def task_low_speed_penalty(
    env,
    task_idx: int,
    speed_threshold: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    speed_xy = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    penalty = torch.clamp(speed_threshold - speed_xy, min=0.0)
    return _task_mask(env, task_idx, command_name) * penalty


@configclass
class S0FlatMultiTaskEnvCfg(S0FlatEnvCfg):
    """Multi-task flat locomotion: forward, right-turn, left-turn (1/3 each)."""

    def __post_init__(self):
        super().__post_init__()

        # Replace velocity command generator with discrete skill sampler.
        self.commands.base_velocity = MultiSkillVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            task_obs_dim=10,
            forward_command=(0.50, 0.0, 0.0),
            right_turn_command=(0.20, 0.0, -0.60),
            left_turn_command=(0.20, 0.0, 0.60),
            debug_vis=False,
        )

        # Append one-hot task instruction to policy observations.
        self.observations.policy.task_instruction = ObsTerm(
            func=task_instruction_obs,
            params={"command_name": "base_velocity", "task_obs_dim": 10},
        )

        # Disable single-task tracking rewards and use task-conditioned reward clusters.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.low_speed_penalty.weight = 0.0

        # Cluster A: forward.
        self.rewards.forward_track_lin = RewTerm(
            func=task_track_lin_vel_xy_exp,
            weight=4.0,
            params={"task_idx": 0, "std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.forward_track_ang = RewTerm(
            func=task_track_ang_vel_z_exp,
            weight=1.0,
            params={"task_idx": 0, "std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.forward_low_speed = RewTerm(
            func=task_low_speed_penalty,
            weight=-0.8,
            params={"task_idx": 0, "speed_threshold": 0.20, "command_name": "base_velocity"},
        )

        # Cluster B: right turn.
        self.rewards.right_track_ang = RewTerm(
            func=task_track_ang_vel_z_exp,
            weight=4.5,
            params={"task_idx": 1, "std": 0.45, "command_name": "base_velocity"},
        )
        self.rewards.right_track_lin = RewTerm(
            func=task_track_lin_vel_xy_exp,
            weight=1.2,
            params={"task_idx": 1, "std": 0.6, "command_name": "base_velocity"},
        )

        # Cluster C: left turn.
        self.rewards.left_track_ang = RewTerm(
            func=task_track_ang_vel_z_exp,
            weight=4.5,
            params={"task_idx": 2, "std": 0.45, "command_name": "base_velocity"},
        )
        self.rewards.left_track_lin = RewTerm(
            func=task_track_lin_vel_xy_exp,
            weight=1.2,
            params={"task_idx": 2, "std": 0.6, "command_name": "base_velocity"},
        )


@configclass
class S0FlatMultiTaskEnvCfg_PLAY(S0FlatMultiTaskEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
