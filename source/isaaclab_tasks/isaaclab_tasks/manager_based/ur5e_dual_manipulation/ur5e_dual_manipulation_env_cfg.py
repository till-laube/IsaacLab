# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for dual UR5e manipulation environment with VR teleoperation support.

This environment is designed for GR00T teleoperation data collection using:
- Two UR5e arms with Robotiq 2F-140 grippers
- Custom physical setup (metal frame, wooden plate, camera mounts)
- HTC Vive controller teleoperation
- VR headset visualization support
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters import ViveControllerDualArmRetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.extensions import get_extension_path_from_name

import isaaclab.envs.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Import robot configurations
##

from isaaclab_assets.robots.dual_ur5e import UR5E_ROBOTIQ_140_CFG

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

UR5E_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(_RMP_CONFIG_DIR, "universal_robots", "ur5e", "rmpflow", "ur5e_rmpflow_config.yaml"),
    urdf_file=_CUSTOM_URDF_PATH,  # Using custom URDF with Robotiq gripper and tcp_link
    collision_file=os.path.join(_RMP_CONFIG_DIR, "universal_robots", "ur5e", "rmpflow", "ur5e_robot_description.yaml"),
    frame_name="tcp_link",  # Matches the USD tcp_link frame
    evaluations_per_frame=5,
)
"""Configuration of RMPFlow for UR5e arm with Robotiq 2F-140 gripper."""

##
# Scene definition
##


@configclass
class Ur5eDualManipulationSceneCfg(InteractiveSceneCfg):
    """Configuration for dual UR5e manipulation scene.

    This scene contains:
    - Two UR5e arms with Robotiq 2F-140 grippers
    - Metal frame (custom STL)
    - Wooden plate/table (custom STL)
    - Camera mounts (custom STLs - collision meshes only, not functional sensors)
    - Ground plane and lighting
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Lighting - dome light for even illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Metal frame - base structure (using RigidObjectCfg to properly apply RigidBodyAPI)
    # URDF transform: xyz=(0.0, 0.125, 0.0), rpy=(1.5708, 0, 1.5708) = (90°, 0°, 90°)
    metal_frame = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MetalFrame",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/metal_frame.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # Frame is static
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.125, 0.0),
            rot=(0.5, 0.5, 0.5, 0.5),  # Quaternion for RPY (90°, 0°, 90°)
        ),
    )

    # Wooden plate/table - work surface (using RigidObjectCfg to properly apply RigidBodyAPI)
    # URDF transform (relative to metal frame): xyz=(-0.127, 0.724, 0.555), rpy=(1.5708, 0, 0) = (90°, 0°, 0°)
    # Absolute transform (computed from metal frame): pos=(0.555, -0.002, 0.724), rot=(-180°, 0°, 90°)
    wooden_plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WoodenPlate",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/wooden_plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # Table is static
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.555, -0.002, 0.724),  # Absolute position after metal frame transform
            rot=(0.0, 0.707107, 0.707107, 0.0),  # Quaternion for RPY (-180°, 0°, 90°)
        ),
    )

    # Left UR5e arm with Robotiq 2F-140 gripper (custom USD)
    # Original URDF transform: xyz=(0.015, 0.183, 1.264), rpy=(3.1415, -0.7853, 1.5707) = (180°, -45°, 90°)
    left_arm: ArticulationCfg = UR5E_ROBOTIQ_140_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/ur5e_robotiq_2f_140.usd",
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
            pos=(0.015, 0.183, 1.264), 
            rot=(-0.270523, 0.653339, 0.653251, 0.270609), 
            joint_pos={
                # Arm joints - neutral ready position
                "shoulder_pan_joint": -1.57079,
                "shoulder_lift_joint": 3.14159,
                "elbow_joint": 0.0,
                "wrist_1_joint": 3.14149,
                "wrist_2_joint": -1.57079,
                "wrist_3_joint": 0.0,
                # Gripper joints - open position (based on actual joint names in USD)
                "finger_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
            },
        ),
        actuators={
            # Arm actuators (from UR5E_ROBOTIQ_140_CFG)
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
            # Gripper actuators (updated for actual joint names)
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit=1650.0,
                velocity_limit=10.0,
                stiffness=17.0,
                damping=0.02,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                effort_limit=50.0,
                velocity_limit=10.0,
                stiffness=0.2,
                damping=0.001,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
                effort_limit=1.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # Right UR5e arm with Robotiq 2F-140 gripper (custom USD)
    # URDF transform: xyz=(0.015, -0.190, 1.266), rpy=(3.1415, 0.7853, 1.5707) = (180°, 45°, 90°)
    right_arm: ArticulationCfg = UR5E_ROBOTIQ_140_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/ur5e_robotiq_2f_140.usd",
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
            pos=(0.015, -0.190, 1.266),  # Right side, at arm mount height
            rot=(0.270583, 0.653314, 0.653276, -0.270549),  # Quaternion for RPY (180°, 45°, 90°)
            joint_pos={
                # Arm joints - neutral ready position (mirrored)
                "shoulder_pan_joint": 1.57079,
                "shoulder_lift_joint": 0.0,
                "elbow_joint": 0.0,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.57079,
                "wrist_3_joint": 0.0,
                # Gripper joints - open position (based on actual joint names in USD)
                "finger_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
            },
        ),
        actuators={
            # Arm actuators (from UR5E_ROBOTIQ_140_CFG)
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
            # Gripper actuators (updated for actual joint names)
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit=1650.0,
                velocity_limit=10.0,
                stiffness=17.0,
                damping=0.02,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                effort_limit=50.0,
                velocity_limit=10.0,
                stiffness=0.2,
                damping=0.001,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
                effort_limit=1.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # ===== CAMERA MOUNTS (Collision meshes only) =====
    # These are just for collision avoidance during teleoperation
    # Real cameras record outside the simulation
    # TODO: Add cameras to setup

    # Left camera mount
    """camera_mount_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CameraMountLeft",
        spawn=sim_utils.MeshCfg(
            mesh_path="/PATH/TO/camera_mount_left.stl",  # TODO: Update this path
            collision_enabled=True,
            mass_props=sim_utils.MassCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=True,  # Fixed to arm
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),  # Dark gray plastic
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # TODO: Adjust position to match where it's mounted on the left arm
            pos=(-0.19, 0.0, 1.5),  # Placeholder position
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Right camera mount
    camera_mount_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CameraMountRight",
        spawn=sim_utils.MeshCfg(
            mesh_path="/PATH/TO/camera_mount_right.stl",  # TODO: Update this path
            collision_enabled=True,
            mass_props=sim_utils.MassCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=True,  # Fixed to arm
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # TODO: Adjust position to match where it's mounted on the right arm
            pos=(0.19, 0.0, 1.5),  # Placeholder position
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )"""

    left_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/LeftArm/base_link_inertia",  
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/LeftArm/ee_link/robotiq_arg2f_base_link",
                name="left_end_effector",
            ),
        ],
    )

    right_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/RightArm/base_link_inertia",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/RightArm/ee_link/robotiq_arg2f_base_link",
                name="right_end_effector",
            ),
        ],
    )

    # Control point frames for visualization (configured in __post_init__)
    left_control_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/LeftArm/base_link_inertia",
        debug_vis=False,  # Will be enabled in __post_init__
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/LeftArm/ee_link/tcp_link",
                name="left_control_point",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),  # Identity - same as right arm
                ),
            ),
        ],
    )

    right_control_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/RightArm/base_link_inertia",
        debug_vis=False,  # Will be enabled in __post_init__
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/RightArm/ee_link/tcp_link",
                name="right_control_point",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),  # No offset with proper TCP
                ),
            ),
        ],
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    This configuration defines how control commands are sent to the robots:
    - RMPFlow: Cartesian space control for smooth end-effector movements (6 DoF per arm)
    - Binary gripper: Open/close commands for grippers (1 DoF per gripper)
    Total: 14 DoF (6+1 for left arm, 6+1 for right arm)
    """

    # Left arm RMPFlow action (6 DOF: 3 pos + 3 rot) - indices 0-5
    left_arm_action: RMPFlowActionCfg = RMPFlowActionCfg(
        asset_name="left_arm",
        joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"],
        body_name="tcp_link",  # Using proper TCP from USD
        controller=UR5E_RMPFLOW_CFG,
        scale=5.0,
        body_offset=RMPFlowActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # No offset needed with proper TCP
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity - same as right arm
        ),
        articulation_prim_expr="/World/envs/env_.*/LeftArm",
        use_relative_mode=True,  # Use delta movements from VR controller
    )

    # Left gripper binary action (1 DOF: open/close) - index 6
    # NOTE: Order matters! Must come after left_arm to match retargeter output
    left_gripper_action: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},  # Open position
        close_command_expr={"finger_joint": 0.7},  # Close position (adjust based on gripper)
    )

    # Right arm RMPFlow action (6 DOF: 3 pos + 3 rot) - indices 7-12
    right_arm_action: RMPFlowActionCfg = RMPFlowActionCfg(
        asset_name="right_arm",
        joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"],
        body_name="tcp_link",  # Using proper TCP from USD
        controller=UR5E_RMPFLOW_CFG,
        scale=5.0,
        body_offset=RMPFlowActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # No offset needed with proper TCP
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity - adjust if axes are still inverted
        ),
        articulation_prim_expr="/World/envs/env_.*/RightArm",
        use_relative_mode=True,  # Use delta movements from VR controller
    )

    # Right gripper binary action (1 DOF: open/close) - index 13
    right_gripper_action: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},  # Open position
        close_command_expr={"finger_joint": 0.7},  # Close position (adjust based on gripper)
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    For teleoperation data collection, we want to observe:
    - End-effector poses (position + orientation)
    - Joint positions and velocities
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (used for GR00T data collection)."""

        # Left arm observations
        left_ee_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("left_arm", body_names=["robotiq_arg2f_base_link"])},
        )
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"])},
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"])},
        )
        left_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm", joint_names=["finger_joint"])},
        )

        # Right arm observations
        right_ee_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("right_arm", body_names=["robotiq_arg2f_base_link"])},
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"])},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"])},
        )
        right_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm", joint_names=["finger_joint"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events (resets, randomization, etc.)."""

    # Reset both arms to neutral position on episode start
    reset_left_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_arm"),
            "position_range": (0.0, 0.0),  # No randomization for teleoperation
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_right_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_arm"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Note: For teleoperation data collection, rewards are typically not used.
    These are placeholder rewards in case you want to add autonomous training later.
    """

    # Placeholder: constant reward for staying alive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out after episode length
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class Ur5eDualManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for dual UR5e manipulation environment.

    This environment is designed for:
    - VR teleoperation with HTC Vive controllers
    - GR00T data collection
    - Dual-arm manipulation tasks
    """

    # Scene settings
    scene: Ur5eDualManipulationSceneCfg = Ur5eDualManipulationSceneCfg(
        num_envs=1,  # Start with 1 environment for teleoperation
        env_spacing=4.0,
    )

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization - set simulation parameters."""
        # General settings
        self.decimation = 2  # Control decimation (2 = 60Hz control at 120Hz sim)
        self.episode_length_s = 60.0  # 60 second episodes for teleoperation

        # Viewer settings
        self.viewer.eye = (2.0, 2.0, 1.5)  # Camera position for viewing
        self.viewer.lookat = (0.0, 0.0, 1.3)  # Look at the arms

        # Simulation settings
        self.sim.dt = 1 / 120  # 120Hz simulation
        self.sim.render_interval = self.decimation

        # XR/VR settings for headset view
        # This positions the VR camera at a comfortable viewing height
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.8),  # Standing position, looking at workspace
            anchor_rot=(1.0, 0.0, 0.0, 0.0),
        )

        # Teleoperation devices configuration
        # This is required for OpenXR-based VR teleoperation with Vive controllers
        # Must be set in __post_init__ to access self.sim.device and self.xr
        #
        # tracking_mode options:
        #   - "delta": Use relative/delta rotations (default) - controller movements are
        #              converted to incremental changes applied to the current gripper pose
        #   - "absolute": Use absolute orientation tracking - controller orientation is
        #                 directly mapped to gripper orientation
        #
        # The tracking_mode can be overridden via CLI using --tracking_mode argument
        #
        # Controller-to-gripper rotation offsets (for absolute mode):
        #   These quaternions [w,x,y,z] align the controller's coordinate frame with the
        #   gripper's coordinate frame. Adjust these if the gripper orientation doesn't
        #   match how you hold the controller.
        #   - Identity (1,0,0,0): No offset
        #   - 90° around X: (0.707, 0.707, 0, 0)
        #   - 90° around Y: (0.707, 0, 0.707, 0)
        #   - 90° around Z: (0.707, 0, 0, 0.707)
        #   - 180° around X: (0, 1, 0, 0)
        #   - 180° around Y: (0, 0, 1, 0)
        #   - 180° around Z: (0, 0, 0, 1)
        self.teleop_devices = DevicesCfg(
            devices={
                "vive": OpenXRDeviceCfg(
                    retargeters=[
                        ViveControllerDualArmRetargeterCfg(
                            tracking_mode="delta",  # "delta" or "absolute"
                            pos_sensitivity=5.0,  # Position movement sensitivity
                            rot_sensitivity=5.0,  # Rotation sensitivity (used in delta mode only)
                            trigger_threshold=0.5,  # Trigger threshold for gripper close
                            # Arm base rotations for coordinate transformation
                            # Left: (180°, -45°, 90°) in XYZ Euler
                            left_base_quat=(-0.270523, 0.653339, 0.653251, 0.270609),
                            # Right: (180°, 45°, 90°) in XYZ Euler
                            right_base_quat=(0.270583, 0.653314, 0.653276, -0.270549),
                            # Controller-to-gripper rotation offsets (for absolute mode)
                            # Adjust these to align controller axes with gripper axes
                            # Format: [w, x, y, z] quaternion
                            left_controller_to_gripper_rot=(1.0, 0.0, 0.0, 0.0),  # Identity (no offset)
                            right_controller_to_gripper_rot=(1.0, 0.0, 0.0, 0.0),  # Identity (no offset)
                            # Scale compensation for absolute mode: 1.0 / RMPFlowActionCfg.scale
                            # RMPFlowActionCfg.scale = 5.0, so this should be 0.2
                            absolute_mode_rot_scale=0.2,
                        ),
                    ],
                    sim_device=self.sim.device,  # Use same device as simulation
                    xr_cfg=self.xr,  # Pass XR configuration for VR view
                ),
            }
        )

        # Configure control point visualization markers
        marker_left_control_cfg = FRAME_MARKER_CFG.copy()
        marker_left_control_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_left_control_cfg.prim_path = "/Visuals/LeftControlMarker"

        marker_right_control_cfg = FRAME_MARKER_CFG.copy()
        marker_right_control_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_right_control_cfg.prim_path = "/Visuals/RightControlMarker"

        # Enable visualization for control frames
        self.scene.left_control_frame.debug_vis = True
        self.scene.left_control_frame.visualizer_cfg = marker_left_control_cfg

        self.scene.right_control_frame.debug_vis = True
        self.scene.right_control_frame.visualizer_cfg = marker_right_control_cfg
