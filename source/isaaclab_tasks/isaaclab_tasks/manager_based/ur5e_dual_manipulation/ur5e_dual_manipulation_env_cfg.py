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
from isaaclab.envs.mdp.actions.world_frame_rmpflow_actions import WorldFrameRMPFlowActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.extensions import get_extension_path_from_name

import isaaclab.envs.mdp as mdp

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

UR5E_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(_RMP_CONFIG_DIR, "universal_robots", "ur5e", "rmpflow", "ur5e_rmpflow_config.yaml"),
    urdf_file=os.path.join(_RMP_CONFIG_DIR, "universal_robots", "ur5e", "ur5e.urdf"),
    collision_file=os.path.join(_RMP_CONFIG_DIR, "universal_robots", "ur5e", "rmpflow", "ur5e_robot_description.yaml"),
    frame_name="tool0",  # TODO: check, bc it is likely ee_link
    evaluations_per_frame=5,
)
"""Configuration of RMPFlow for UR5e arm."""

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
                "shoulder_pan_joint": 1.0472,
                "shoulder_lift_joint": -1.0472,
                "elbow_joint": 2.0944,
                "wrist_1_joint": 3.6652,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 3.1415,
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
                "shoulder_pan_joint": -1.0472,
                "shoulder_lift_joint": -2.0944,
                "elbow_joint": -2.0944,
                "wrist_1_joint": -0.5236,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 3.1415,
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

    # TODO: write the ActionsCfg



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
            anchor_pos=(0.0, -0.5, 1.6),  # Standing position, looking at workspace
            anchor_rot=(1.0, 0.0, 0.0, 0.0),
        )

        # Teleoperation devices configuration
        # This is required for OpenXR-based VR teleoperation with Vive controllers
        # Must be set in __post_init__ to access self.sim.device and self.xr
        self.teleop_devices = DevicesCfg(
            devices={
                "vive": OpenXRDeviceCfg(
                    retargeters=[
                        ViveControllerDualArmRetargeterCfg(
                            pos_sensitivity=5.0,  # Position movement sensitivity (match Galbot)
                            rot_sensitivity=5.0,  # Rotation sensitivity (match Galbot)
                            trigger_threshold=0.5,  # Trigger threshold for gripper close
                        ),
                    ],
                    sim_device=self.sim.device,  # Use same device as simulation
                    xr_cfg=self.xr,  # Pass XR configuration for VR view
                ),
            }
        )
