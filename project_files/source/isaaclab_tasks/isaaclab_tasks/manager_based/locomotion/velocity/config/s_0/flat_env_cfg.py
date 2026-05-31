# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import S0RoughEnvCfg


@configclass
class S0FlatEnvCfg(S0RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards for flat terrain
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.08
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 0.2
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_acc_l2.weight = -1.5e-6
        self.rewards.dof_torques_l2.weight = -4.0e-4
        self.rewards.low_speed_penalty.weight = -0.5
        self.rewards.low_base_height_penalty.weight = -1.2

        # pure forward curriculum: no strafing, no turning
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class S0FlatEnvCfg_PLAY(S0FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
