# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets import S0_CFG  # isort: skip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def low_speed_penalty(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize near-zero planar speed to avoid lying still."""
    asset: Articulation = env.scene[asset_cfg.name]
    speed_xy = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return torch.clamp(speed_threshold - speed_xy, min=0.0)


def low_base_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = 0.16,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height near ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.clamp(min_height - base_height, min=0.0)


@configclass
class S0RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = S0_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # lightweight robot: slightly milder terrain
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Action: map policy output to joint limits directly.
        self.actions.joint_pos = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["joint_leg_.*", "joint_arm_.*", "joint_jaw_.*"],
            scale=0.25,
            rescale_to_limits=True,
        )

        # Events
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # Rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "jaw_.*"
        self.rewards.feet_air_time.weight = 0.08
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_torques_l2.weight = -4.0e-4
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["joint_leg_.*", "joint_arm_.*", "joint_jaw_.*"],
        )
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 0.2
        self.rewards.dof_acc_l2.weight = -1.5e-6
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["joint_leg_.*", "joint_arm_.*", "joint_jaw_.*"],
        )
        # Keep a soft joint-limit penalty, but avoid over-penalizing near-limit motion.
        self.rewards.dof_pos_limits.weight = -0.2
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["joint_leg_.*", "joint_arm_.*", "joint_jaw_.*"],
        )
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.low_speed_penalty = RewTerm(
            func=low_speed_penalty,
            weight=-0.5,
            params={"speed_threshold": 0.20, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.low_base_height_penalty = RewTerm(
            func=low_base_height_penalty,
            weight=-1.2,
            params={"min_height": 0.16, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Commands: train only stable forward locomotion.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Terminations
        # S_0 geometry is currently very close to the ground in the imported asset.
        # Keeping base-contact termination here causes immediate reset loops.
        # Re-enable once default pose / collision geometry are refined.
        self.terminations.base_contact = None
        # Do not terminate on joint-limit crossing during exploration.
        # Physical limits are already enforced in USD; this avoids reset spikes and PPO instability.
        self.terminations.joint_pos_manual_limits = None


@configclass
class S0RoughEnvCfg_PLAY(S0RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
