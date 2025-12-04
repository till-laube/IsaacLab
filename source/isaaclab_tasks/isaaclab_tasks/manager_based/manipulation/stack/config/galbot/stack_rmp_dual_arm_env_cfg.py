# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase, DeviceCfg, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3RelRetargeterCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.vive import Se3ViveControllerCfg, DualArmViveRetargeterCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab.controllers.config.rmp_flow import (  # isort: skip
    GALBOT_LEFT_ARM_RMPFLOW_CFG,
    GALBOT_RIGHT_ARM_RMPFLOW_CFG,
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


##
# RmpFlow Controller for Galbot Dual-Arm Cube Stack Task
##
@configclass
class RmpFlowGalbotDualArmCubeStackEnvCfg(stack_joint_pos_env_cfg.GalbotLeftArmCubeStackEnvCfg):
    """Dual-arm configuration for Galbot with 16 DOF support (7 per arm + 1 gripper each)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # read use_relative_mode from environment variable
        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for LEFT ARM
        self.actions.left_arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="left_gripper_tcp_link",
            controller=GALBOT_LEFT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Set actions for RIGHT ARM
        self.actions.right_arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_gripper_tcp_link",
            controller=GALBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Left gripper action
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_gripper_.*_joint"],
            open_command_expr={"left_gripper_.*_joint": 0.035},
            close_command_expr={"left_gripper_.*_joint": 0.0},
        )

        # Right gripper action
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_gripper_.*_joint"],
            open_command_expr={"right_gripper_.*_joint": 0.035},
            close_command_expr={"right_gripper_.*_joint": 0.0},
        )

        # Remove single-arm action if it exists
        if hasattr(self.actions, "arm_action"):
            delattr(self.actions, "arm_action")
        if hasattr(self.actions, "gripper_action"):
            delattr(self.actions, "gripper_action")

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0

        # Add frame transformer for right arm end effector
        marker_right_ee_cfg = FRAME_MARKER_CFG.copy()
        marker_right_ee_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_right_ee_cfg.prim_path = "/Visuals/FrameTransformerRightEE"

        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_right_ee_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_tcp_link",
                    name="right_end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # Configure teleop devices for dual-arm control
        self.teleop_devices = DevicesCfg(
            devices={
                "vive": Se3ViveControllerCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                    control_mode="dual_arm",
                    retargeters=[
                        DualArmViveRetargeterCfg(arm_side="left"),
                        DualArmViveRetargeterCfg(arm_side="right"),
                    ],
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        # Left hand controls left arm
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT, sim_device=self.sim.device
                        ),
                        # Right hand controls right arm
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


##
# Dual-Arm Observations
##
@configclass
class ObservationDualArmCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for dual-arm policy."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        object = ObsTerm(
            func=mdp.object_abs_obs_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        cube_positions = ObsTerm(
            func=mdp.cube_poses_in_base_frame, params={"robot_cfg": SceneEntityCfg("robot"), "return_key": "pos"}
        )
        cube_orientations = ObsTerm(
            func=mdp.cube_poses_in_base_frame, params={"robot_cfg": SceneEntityCfg("robot"), "return_key": "quat"}
        )

        # Left arm end effector
        left_eef_pos = ObsTerm(
            func=mdp.ee_frame_pose_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "return_key": "pos",
            },
        )
        left_eef_quat = ObsTerm(
            func=mdp.ee_frame_pose_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "return_key": "quat",
            },
        )

        # Right arm end effector
        right_eef_pos = ObsTerm(
            func=mdp.ee_frame_pose_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "return_key": "pos",
            },
        )
        right_eef_quat = ObsTerm(
            func=mdp.ee_frame_pose_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "return_key": "quat",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
