# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for dual UR5e robotic arms with Robotiq 2F-140 grippers.

The following configuration parameters are available:

* :obj:`UR5E_CFG`: Single UR5e arm without a gripper.
* :obj:`UR5E_ROBOTIQ_140_CFG`: Single UR5e arm with Robotiq 2F-140 gripper.
* :obj:`DUAL_UR5E_ROBOTIQ_140_CFG`: Dual UR5e arms with Robotiq 2F-140 grippers.

Reference: https://github.com/ros-industrial/universal_robot

UR5e Specifications: 
- Reach: 850 mm
- Payload: 5 kg
- Repeatability: ±0.03 mm
- Weight: 20.6 kg
- 6 revolute joints
- Joint velocity limits: ~180 deg/s (varies by joint)

Robotiq 2F-140 Specifications:
- Stroke: 140 mm
- Grip force: 20-235 N
- Payload: 5 kg
- Weight: 1 kg
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, # TODO: Revisit
            # Solver iteration counts affect simulation stability and accuracy
            # Higher values = more stable but slower
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Initial joint positions (in radians)
        joint_pos={
            "shoulder_pan_joint": 0.0,           # Base rotation
            "shoulder_lift_joint": -1.5708,      # -90 degrees
            "elbow_joint": 1.5708,               # 90 degrees
            "wrist_1_joint": -1.5708,            # -90 degrees
            "wrist_2_joint": -1.5708,            # -90 degrees
            "wrist_3_joint": 0.0,                # 0 degrees
        },
        # Initial pose of the robot base in the world frame
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w, x, y, z)
    ),
    actuators={
        # TODO: TUNE THESE VALUES based on sim behavior:
        # - Increase stiffness if joints are too compliant/slow to respond
        # - Increase damping if there are oscillations
        # - Reduce both if joints are too stiff/jerky

        # Shoulder joints (shoulder_pan_joint, shoulder_lift_joint)
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            # UR5e shoulder motors: smaller than UR10e
            # Estimated stiffness: ~40-60% of UR10e values
            stiffness=800.0,      # TODO: Tune this value (range: 600-1200)
            damping=44.0,         # TODO: Tune this value (range: 30-60)
            friction=0.0, 
            armature=0.0,       
        ),

        # Elbow joint
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            # UR5e elbow motor: intermediate size
            stiffness=400.0,      # TODO: Tune this value (range: 300-600)
            damping=22.0,         # TODO: Tune this value (range: 15-35)
            friction=0.0,
            armature=0.0,
        ),

        # Wrist joints (wrist_1_joint, wrist_2_joint, wrist_3_joint)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=150.0,      # TODO: Tune this value (range: 100-250)
            damping=18.0,         # TODO: Tune this value (range: 10-25)
            friction=0.0,
            armature=0.0,
        ),
    },
)

"""Configuration of UR5e arm using implicit actuator models with estimated PD gains.

Note: The stiffness and damping values are initial estimates and should be tuned based on:
1. Your simulation timestep
2. Desired tracking performance
3. Stability requirements
4. Whether you're using position control, velocity control, or torque control

For more accurate values, consider:
- Testing with a simple position tracking task
- Adjusting gains iteratively to minimize overshoot and oscillation
- Consulting UR5e technical documentation for motor specifications
"""

##
# Configuration with Robotiq 2F-140 Gripper
##

UR5E_ROBOTIQ_140_CFG = UR5E_CFG.copy()
# Configure UR5e with Robotiq 2F-140 gripper variant
UR5E_ROBOTIQ_140_CFG.spawn.variants = {"Gripper": "Robotiq_2f_140"}
UR5E_ROBOTIQ_140_CFG.init_state.joint_pos = {
    # Arm joints
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
    # Gripper joints (open position)
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}

# Add gripper actuators to the existing arm actuators
UR5E_ROBOTIQ_140_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],  # Main gripper drive joint
    effort_limit_sim=1650.0,      # Robotiq 2F-140 max force
    velocity_limit_sim=10.0,
    stiffness=17.0,               # Based on Franka Robotiq config
    damping=0.02,
    friction=0.0,
    armature=0.0,
)

# Gripper finger actuators - enable parallel grasping
UR5E_ROBOTIQ_140_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=50.0,
    velocity_limit_sim=10.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)

# Gripper passive joints - set PD to zero for passive joints in closed-loop gripper
UR5E_ROBOTIQ_140_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=10.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)

"""Configuration of UR5e arm with Robotiq 2F-140 gripper.

The gripper is configured with:
- Main drive joint for opening/closing
- Inner finger joints for adaptive grasping
- Passive joints for mechanical coupling
"""


##
# Dual Arm Configuration with Robotiq 2F-140 Grippers
##

DUAL_UR5E_ROBOTIQ_140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
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
        # Default pose - can be overridden per arm in scene config
        joint_pos={
            # Left arm joints (prefixed with left_)
            "left_shoulder_pan_joint": 0.0,
            "left_shoulder_lift_joint": -1.5708,
            "left_elbow_joint": 1.5708,
            "left_wrist_1_joint": -1.5708,
            "left_wrist_2_joint": -1.5708,
            "left_wrist_3_joint": 0.0,
            # Left gripper joints
            "left_finger_joint": 0.0,
            "left_.*_inner_finger_joint": 0.0,
            "left_.*_inner_finger_knuckle_joint": 0.0,
            "left_.*_outer_.*_joint": 0.0,
            # Right arm joints (prefixed with right_)
            "right_shoulder_pan_joint": 0.0,
            "right_shoulder_lift_joint": -1.5708,
            "right_elbow_joint": 1.5708,
            "right_wrist_1_joint": -1.5708,
            "right_wrist_2_joint": -1.5708,
            "right_wrist_3_joint": 0.0,
            # Right gripper joints
            "right_finger_joint": 0.0,
            "right_.*_inner_finger_joint": 0.0,
            "right_.*_inner_finger_knuckle_joint": 0.0,
            "right_.*_outer_.*_joint": 0.0,
        },
        # Base position - adjust based on your scene layout
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # Left arm actuators
        "left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*"],
            stiffness=800.0,
            damping=44.0,
            friction=0.0,
            armature=0.0,
        ),
        "left_elbow": ImplicitActuatorCfg(
            joint_names_expr=["left_elbow_joint"],
            stiffness=400.0,
            damping=22.0,
            friction=0.0,
            armature=0.0,
        ),
        "left_wrist": ImplicitActuatorCfg(
            joint_names_expr=["left_wrist_.*"],
            stiffness=150.0,
            damping=18.0,
            friction=0.0,
            armature=0.0,
        ),
        # Left gripper actuators
        "left_gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["left_finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.02,
            friction=0.0,
            armature=0.0,
        ),
        "left_gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_.*_inner_finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=0.2,
            damping=0.001,
            friction=0.0,
            armature=0.0,
        ),
        "left_gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["left_.*_inner_finger_knuckle_joint", "left_.*_outer_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
        # Right arm actuators
        "right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*"],
            stiffness=800.0,
            damping=44.0,
            friction=0.0,
            armature=0.0,
        ),
        "right_elbow": ImplicitActuatorCfg(
            joint_names_expr=["right_elbow_joint"],
            stiffness=400.0,
            damping=22.0,
            friction=0.0,
            armature=0.0,
        ),
        "right_wrist": ImplicitActuatorCfg(
            joint_names_expr=["right_wrist_.*"],
            stiffness=150.0,
            damping=18.0,
            friction=0.0,
            armature=0.0,
        ),
        # Right gripper actuators
        "right_gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["right_finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.02,
            friction=0.0,
            armature=0.0,
        ),
        "right_gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_inner_finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=0.2,
            damping=0.001,
            friction=0.0,
            armature=0.0,
        ),
        "right_gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_inner_finger_knuckle_joint", "right_.*_outer_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)

"""Configuration of dual UR5e arms with Robotiq 2F-140 grippers.

This configuration assumes:
- Left arm joints are prefixed with "left_"
- Right arm joints are prefixed with "right_"
- Both arms have Robotiq 2F-140 grippers attached
- Arms are positioned side-by-side (adjust pos in scene config)

Note: You may need to adjust joint name prefixes based on your actual USD file structure.
"""
