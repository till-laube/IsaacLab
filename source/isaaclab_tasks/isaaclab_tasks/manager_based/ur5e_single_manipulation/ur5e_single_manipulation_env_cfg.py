# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import ViveControllerSe3RetargeterCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaacsim.core.utils.extensions import get_extension_path_from_name

import isaaclab.envs.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# RMPFlow configuration for UR5e
##

_RMP_CONFIG_DIR = os.path.join(
    get_extension_path_from_name("isaacsim.robot_motion.motion_generation"), "motion_policy_configs"
)

# Path to custom URDF with Robotiq gripper and tcp_link
_CUSTOM_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "isaaclab_assets", "data", "ur5e_dual_setup", "ur5e_robotiq_2f_140.urdf"
)

# Path to custom RMPFlow configs for UR5e + Robotiq 2F-140
_CUSTOM_RMPFLOW_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "isaaclab_assets", "data", "ur5e_dual_setup", "rmpflow"
)

# Path to custom asset data directory for UR5e dual setup
_ASSET_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "isaaclab_assets", "data", "ur5e_dual_setup"
)

UR5E_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(_CUSTOM_RMPFLOW_DIR, "ur5e_robotiq140_rmpflow_config.yaml"),
    urdf_file=_CUSTOM_URDF_PATH,  # Using custom URDF with Robotiq gripper and tcp_link
    collision_file=os.path.join(_CUSTOM_RMPFLOW_DIR, "ur5e_robotiq140_robot_description.yaml"),
    frame_name="tcp_link",  # Matches the USD tcp_link frame
    evaluations_per_frame=5,
    ignore_robot_state_updates=True,  # CRITICAL: Same as Galbot for proper relative mode
)
"""Configuration of RMPFlow for UR5e arm with Robotiq 2F-140 gripper (custom configs with gripper collision)."""

##
# Scene definition
##


@configclass
class Ur5eSingleManipulationSceneCfg(InteractiveSceneCfg):
    """Configuration for a single UR5e arm manipulation scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Simple table using a cube (0.8m x 0.8m x 0.05m)
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.1),  # Brown wood color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.825),  # Half height of table (0.05/2) + 0.8m elevation
            rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion for RPY (180°, 45°, 90°) - matches robot
        ),
    )

    # Robot - Single UR5e with Robotiq 2F-140 gripper
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSET_DATA_DIR, "ur5e_robotiq_2f_140.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85),  # On top of table (table height = 0.05m) + 0.8m elevation
            rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion for RPY (180°, 45°, 90°) - matches dual arm right
            joint_pos={
                # Arm joints - neutral ready position (matches dual arm right configuration)
                "shoulder_pan_joint": 1.57079,
                "shoulder_lift_joint": -1.57079,
                "elbow_joint": 1.57079,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.57079,
                "wrist_3_joint": 0.78539,
                "finger_joint": 0.0,
                ".*_inner_finger_joint": 0.0,
                ".*_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
            },
        ),
        actuators={
            # Arm actuators (same as UR5E_ROBOTIQ_140_CFG)
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=800.0,
                damping=44.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=400.0,
                damping=22.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=150.0,
                damping=18.0,
            ),
            # Gripper actuators
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=1650.0,
                velocity_limit_sim=10.0,
                stiffness=17.0,
                damping=0.02,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=0.2,
                damping=0.001,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
                effort_limit_sim=1.0,
                velocity_limit_sim=10.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: RMPFlowActionCfg = None  # Will be configured in __post_init__
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.628},  # Closed position for Robotiq 2F-140
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (minimal for teleoperation)."""

    # Dummy reward - not used for teleoperation
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (minimal for teleoperation)."""

    # Time out - only termination condition
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class Ur5eSingleManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the single UR5e manipulation environment."""

    # Scene settings
    scene: Ur5eSingleManipulationSceneCfg = Ur5eSingleManipulationSceneCfg(num_envs=1, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6
        self.decimation = 3
        self.episode_length_s = 30.0

        # Read use_relative_mode from environment variable
        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set up arm action with RMPFlow
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"],
            body_name="tcp_link",  # Using proper TCP from USD
            controller=UR5E_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # XR configuration - standing position
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.8),  # Standing position
            anchor_rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
        )

        # Teleoperation devices configuration
        self.teleop_devices = DevicesCfg(
            devices={
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
                "vive": OpenXRDeviceCfg(
                    retargeters=[
                        ViveControllerSe3RetargeterCfg(
                            hand_side="right",  # Use right controller
                            pos_sensitivity=5.0,
                            rot_sensitivity=5.0,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

        # Add control point visualization
        marker_control_cfg = FRAME_MARKER_CFG.copy()
        marker_control_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Medium markers
        marker_control_cfg.prim_path = "/Visuals/FrameTransformerControl"

        self.scene.control_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link_inertia",
            debug_vis=True,
            visualizer_cfg=marker_control_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/tcp_link",
                    name="control_point",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # Viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
