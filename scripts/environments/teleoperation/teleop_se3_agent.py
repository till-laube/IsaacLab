# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers).

Real Robot Integration:
    When --real-robot flag is passed, this script will:
    1. Connect to real UR5e robots via RTDE to read initial joint positions
    2. Set simulation robot to match real robot positions
    3. Publish joint states via ZMQ for real robot controller to follow

Usage with real robot:
    ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
        --task Isaac-UR5e-Dual-Manipulation-v0 \
        --real-robot
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# Real robot integration arguments
parser.add_argument(
    "--real-robot",
    action="store_true",
    default=False,
    help="Enable real robot integration (sync initial state and publish joint commands via ZMQ).",
)
parser.add_argument(
    "--left-ip",
    type=str,
    default="100.80.147.160",
    help="IP address of the left UR5e arm (default: 100.80.147.160).",
)
parser.add_argument(
    "--right-ip",
    type=str,
    default="100.80.147.51",
    help="IP address of the right UR5e arm (default: 100.80.147.51).",
)
parser.add_argument(
    "--zmq-port",
    type=int,
    default=5555,
    help="ZMQ port for publishing joint states (default: 5555).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower() or "vive" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import logging
import time
import torch
import numpy as np

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg, Se3ViveController
from isaaclab.devices.vive import Se3ViveControllerCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)

# Real robot integration imports (optional)
REAL_ROBOT_AVAILABLE = False
ZMQ_AVAILABLE = False
RTDEReceiveInterface = None

if args_cli.real_robot:
    print("[REAL ROBOT] --real-robot flag detected, importing dependencies...")
    try:
        from rtde_receive import RTDEReceiveInterface
        REAL_ROBOT_AVAILABLE = True
        print("[REAL ROBOT] ur_rtde imported successfully")
    except ImportError as e:
        print(f"[REAL ROBOT] ERROR: ur_rtde not installed! Error: {e}")
        print("[REAL ROBOT] Install with: pip install ur_rtde")
        print("[REAL ROBOT] Exiting because --real-robot requires ur_rtde")
        simulation_app.close()
        import sys
        sys.exit(1)

    try:
        import zmq
        ZMQ_AVAILABLE = True
        print("[REAL ROBOT] pyzmq imported successfully")
    except ImportError as e:
        print(f"[REAL ROBOT] ERROR: pyzmq not installed! Error: {e}")
        print("[REAL ROBOT] Install with: pip install pyzmq")
        print("[REAL ROBOT] Exiting because --real-robot requires pyzmq")
        simulation_app.close()
        import sys
        sys.exit(1)


class RealRobotFeedback:
    """Maintains RTDE connections for continuous real robot state feedback."""

    def __init__(self, left_ip: str, right_ip: str):
        """Initialize RTDE connections for continuous feedback.

        Args:
            left_ip: IP address of left arm.
            right_ip: IP address of right arm.
        """
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.left_rtde = None
        self.right_rtde = None
        self.connected = False

    def connect(self) -> bool:
        """Establish RTDE connections to both arms.

        Returns:
            True if at least one arm connected successfully.
        """
        if not REAL_ROBOT_AVAILABLE or RTDEReceiveInterface is None:
            print("[FEEDBACK] ur_rtde not available")
            return False

        try:
            print(f"[FEEDBACK] Connecting to LEFT arm at {self.left_ip}...")
            self.left_rtde = RTDEReceiveInterface(self.left_ip)
            print("[FEEDBACK] LEFT arm connected")
        except Exception as e:
            print(f"[FEEDBACK] Failed to connect LEFT arm: {e}")
            self.left_rtde = None

        try:
            print(f"[FEEDBACK] Connecting to RIGHT arm at {self.right_ip}...")
            self.right_rtde = RTDEReceiveInterface(self.right_ip)
            print("[FEEDBACK] RIGHT arm connected")
        except Exception as e:
            print(f"[FEEDBACK] Failed to connect RIGHT arm: {e}")
            self.right_rtde = None

        self.connected = self.left_rtde is not None or self.right_rtde is not None
        if self.connected:
            print("[FEEDBACK] RTDE feedback connections established")
        return self.connected

    def read_positions(self) -> tuple:
        """Read current joint positions from real robots.

        Returns:
            Tuple of (left_joints, right_joints) as lists, or None if unavailable.
        """
        left_joints = None
        right_joints = None

        if self.left_rtde is not None:
            try:
                left_joints = list(self.left_rtde.getActualQ())
            except Exception:
                pass  # Silent fail for performance

        if self.right_rtde is not None:
            try:
                right_joints = list(self.right_rtde.getActualQ())
            except Exception:
                pass  # Silent fail for performance

        return left_joints, right_joints

    def close(self):
        """Close RTDE connections."""
        if self.left_rtde is not None:
            try:
                self.left_rtde.disconnect()
            except Exception:
                pass
            self.left_rtde = None
        if self.right_rtde is not None:
            try:
                self.right_rtde.disconnect()
            except Exception:
                pass
            self.right_rtde = None
        self.connected = False
        print("[FEEDBACK] RTDE connections closed")


class RealRobotPublisher:
    """Publishes joint states to real robot controller via ZMQ."""

    def __init__(self, port: int = 5555):
        """Initialize the ZMQ publisher.

        Args:
            port: ZMQ port to bind to.
        """
        self.port = port
        self.context = None
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        """Initialize ZMQ publisher socket.

        Returns:
            True if successful, False otherwise.
        """
        if not ZMQ_AVAILABLE:
            logger.warning("[ZMQ] pyzmq not available")
            return False

        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.connected = True
            logger.info(f"[ZMQ] Publisher started on port {self.port}")
            # Give subscribers time to connect
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"[ZMQ] Failed to start publisher: {e}")
            return False

    def publish(
        self,
        left_joints: list,
        right_joints: list,
        left_gripper: float,
        right_gripper: float,
        teleop_active: bool = False,
    ):
        """Publish joint states to the real robot controller.

        Args:
            left_joints: Left arm joint positions (6 elements).
            right_joints: Right arm joint positions (6 elements).
            left_gripper: Left gripper position (0-1).
            right_gripper: Right gripper position (0-1).
            teleop_active: Whether teleoperation is currently active (SQUEEZE pressed).
        """
        if not self.connected or self.socket is None:
            return

        message = {
            "timestamp": time.time(),
            "active": teleop_active,  # CRITICAL: Controller only moves when this is True
            "left_arm": left_joints,
            "right_arm": right_joints,
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
        }

        try:
            self.socket.send_json(message)
        except Exception as e:
            logger.warning(f"[ZMQ] Failed to publish: {e}")

    def close(self):
        """Close the ZMQ connection."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        logger.info("[ZMQ] Publisher closed")


def read_gripper_position(ip: str, port: int = 63352, timeout: float = 2.0) -> float:
    """Read current gripper position via Robotiq ASCII socket interface.

    Args:
        ip: IP address of the robot (gripper is connected via robot).
        port: Gripper socket port (default 63352 for Robotiq on UR).
        timeout: Socket timeout in seconds.

    Returns:
        Gripper position as float (0.0 = open, 1.0 = closed), or 0.0 on failure.
    """
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))

        # Send GET POS command
        sock.sendall(b"GET POS\n")

        # Read response
        response = sock.recv(1024).decode().strip()
        sock.close()

        # Parse response - expected format: "POS 128" or just "128"
        if "POS" in response:
            pos_str = response.split()[-1]
        else:
            pos_str = response

        pos_val = int(pos_str)
        # Convert 0-255 range to 0.0-1.0
        gripper_pos = pos_val / 255.0
        print(f"[GRIPPER] Read position from {ip}: {pos_val} -> {gripper_pos:.3f}")
        return gripper_pos

    except Exception as e:
        print(f"[GRIPPER] Failed to read from {ip}:{port}: {e}")
        return 0.0


def read_real_robot_positions(left_ip: str, right_ip: str) -> tuple:
    """Read current joint positions from real UR5e robots.

    Args:
        left_ip: IP address of left arm.
        right_ip: IP address of right arm.

    Returns:
        Tuple of (left_joints, right_joints, left_gripper, right_gripper).
        Returns None values if connection fails.
    """
    print(f"[RTDE] REAL_ROBOT_AVAILABLE = {REAL_ROBOT_AVAILABLE}")
    print(f"[RTDE] RTDEReceiveInterface = {RTDEReceiveInterface}")

    if not REAL_ROBOT_AVAILABLE or RTDEReceiveInterface is None:
        print("[RTDE] ERROR: ur_rtde not available, cannot read real robot positions")
        return None, None, None, None

    left_joints = None
    right_joints = None

    # Read left arm
    try:
        print(f"[RTDE] Connecting to LEFT arm at {left_ip}...")
        left_rtde = RTDEReceiveInterface(left_ip)
        left_joints = list(left_rtde.getActualQ())
        left_rtde.disconnect()
        print(f"[RTDE] LEFT arm joints: {[f'{j:.3f}' for j in left_joints]}")
    except Exception as e:
        print(f"[RTDE] ERROR: Failed to read LEFT arm: {e}")
        import traceback
        traceback.print_exc()

    # Read right arm
    try:
        print(f"[RTDE] Connecting to RIGHT arm at {right_ip}...")
        right_rtde = RTDEReceiveInterface(right_ip)
        right_joints = list(right_rtde.getActualQ())
        right_rtde.disconnect()
        print(f"[RTDE] RIGHT arm joints: {[f'{j:.3f}' for j in right_joints]}")
    except Exception as e:
        print(f"[RTDE] ERROR: Failed to read RIGHT arm: {e}")
        import traceback
        traceback.print_exc()

    # Read gripper positions via Robotiq ASCII socket interface
    print("[GRIPPER] Reading gripper positions...")
    left_gripper = read_gripper_position(left_ip)
    right_gripper = read_gripper_position(right_ip)

    print(f"[RTDE] Returning: left={left_joints is not None}, right={right_joints is not None}")
    print(f"[GRIPPER] Returning: left_gripper={left_gripper:.3f}, right_gripper={right_gripper:.3f}")
    return left_joints, right_joints, left_gripper, right_gripper


def get_joint_positions_from_env(env) -> tuple:
    """Extract joint positions from the environment.

    Args:
        env: The Isaac Lab environment.

    Returns:
        Tuple of (left_joints, right_joints, left_gripper, right_gripper).
    """
    # Get the articulations from the scene
    left_arm = env.scene["left_arm"]
    right_arm = env.scene["right_arm"]

    # Get simulated joint positions
    left_pos = left_arm.data.joint_pos[0].cpu().numpy()
    right_pos = right_arm.data.joint_pos[0].cpu().numpy()

    # Extract arm joints (first 6) and gripper (7th joint - finger_joint)
    left_arm_joints = left_pos[:6].tolist()
    right_arm_joints = right_pos[:6].tolist()

    # Gripper: finger_joint position, convert to 0-1 range (0.7 rad = fully closed)
    left_gripper = float(left_pos[6]) / 0.7 if len(left_pos) > 6 else 0.0
    right_gripper = float(right_pos[6]) / 0.7 if len(right_pos) > 6 else 0.0

    # Clamp gripper values
    left_gripper = max(0.0, min(1.0, left_gripper))
    right_gripper = max(0.0, min(1.0, right_gripper))

    return left_arm_joints, right_arm_joints, left_gripper, right_gripper


def set_robot_joint_positions(env, left_joints: list = None, right_joints: list = None,
                               left_gripper: float = None, right_gripper: float = None):
    """Explicitly set robot joint positions in simulation.

    This forces the simulation robot to match the specified joint positions.

    Args:
        env: The Isaac Lab environment.
        left_joints: Left arm joint positions (6 elements), or None to skip.
        right_joints: Right arm joint positions (6 elements), or None to skip.
        left_gripper: Left gripper position (0.0-1.0), or None to skip.
        right_gripper: Right gripper position (0.0-1.0), or None to skip.
    """
    print(f"[SYNC] set_robot_joint_positions called with left={left_joints is not None}, right={right_joints is not None}")
    print(f"[SYNC] Grippers: left={left_gripper}, right={right_gripper}")

    left_arm = env.scene["left_arm"]
    right_arm = env.scene["right_arm"]

    # Set left arm positions
    if left_joints is not None:
        print(f"[SYNC] Setting LEFT arm to: {[f'{j:.4f}' for j in left_joints]}")
        # Get current joint positions tensor
        joint_pos = left_arm.data.joint_pos.clone()
        joint_vel = left_arm.data.joint_vel.clone()
        # Zero velocity
        joint_vel[:] = 0.0
        # Update first 6 joints (arm joints)
        for i in range(min(6, len(left_joints))):
            joint_pos[0, i] = left_joints[i]
        # Set gripper position if provided (joint index 6 = finger_joint)
        # Gripper: 0.0 = open, 1.0 = closed -> maps to 0.0 to 0.7 rad
        if left_gripper is not None:
            gripper_rad = left_gripper * 0.7  # Convert 0-1 to 0-0.7 rad
            joint_pos[0, 6] = gripper_rad
            print(f"[SYNC] LEFT gripper set to {left_gripper:.3f} -> {gripper_rad:.4f} rad")
        # Write to simulation
        left_arm.write_joint_state_to_sim(joint_pos, joint_vel)
        print(f"[SYNC] LEFT arm joint_pos written to sim")

    # Set right arm positions
    if right_joints is not None:
        print(f"[SYNC] Setting RIGHT arm to: {[f'{j:.4f}' for j in right_joints]}")
        joint_pos = right_arm.data.joint_pos.clone()
        joint_vel = right_arm.data.joint_vel.clone()
        joint_vel[:] = 0.0
        for i in range(min(6, len(right_joints))):
            joint_pos[0, i] = right_joints[i]
        # Set gripper position if provided
        if right_gripper is not None:
            gripper_rad = right_gripper * 0.7
            joint_pos[0, 6] = gripper_rad
            print(f"[SYNC] RIGHT gripper set to {right_gripper:.3f} -> {gripper_rad:.4f} rad")
        right_arm.write_joint_state_to_sim(joint_pos, joint_vel)
        print(f"[SYNC] RIGHT arm joint_pos written to sim")


def sync_sim_to_real_robot(env, feedback: "RealRobotFeedback", verbose: bool = False):
    """Sync simulation joint positions to match real robot state.

    This function reads the current real robot positions and updates the simulation
    to match, preventing drift between sim and real. After writing joint positions,
    it triggers an articulation update to run forward kinematics so that body poses
    (end-effector positions) are immediately available for RMPFlow.

    Args:
        env: The Isaac Lab environment.
        feedback: RealRobotFeedback instance with active RTDE connections.
        verbose: If True, print debug information.
    """
    if feedback is None or not feedback.connected:
        return

    # Read real robot positions
    left_joints, right_joints = feedback.read_positions()

    if left_joints is None and right_joints is None:
        return

    left_arm = env.scene["left_arm"]
    right_arm = env.scene["right_arm"]
    dt = env.sim.cfg.dt

    # Update left arm
    if left_joints is not None:
        joint_pos = left_arm.data.joint_pos.clone()
        joint_vel = left_arm.data.joint_vel.clone()
        # Keep velocity to allow smooth motion, only correct position
        for i in range(min(6, len(left_joints))):
            joint_pos[0, i] = left_joints[i]
        left_arm.write_joint_state_to_sim(joint_pos, joint_vel)
        if verbose:
            print(f"[SYNC] LEFT: {[f'{j:.3f}' for j in left_joints[:3]]}...")

    # Update right arm
    if right_joints is not None:
        joint_pos = right_arm.data.joint_pos.clone()
        joint_vel = right_arm.data.joint_vel.clone()
        for i in range(min(6, len(right_joints))):
            joint_pos[0, i] = right_joints[i]
        right_arm.write_joint_state_to_sim(joint_pos, joint_vel)
        if verbose:
            print(f"[SYNC] RIGHT: {[f'{j:.3f}' for j in right_joints[:3]]}...")

    # Force forward kinematics update so body poses (EE positions) reflect synced joints
    # This is critical: without this, _compute_frame_pose() in RMPFlow reads stale data
    # TODO: Is this really necessary?
    left_arm.update(dt)
    right_arm.update(dt)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # Real robot integration: read initial positions
    real_robot_initial_pos = None
    if args_cli.real_robot:
        print("=" * 60)
        print("Real Robot Integration Enabled")
        print(f"  Left IP:  {args_cli.left_ip}")
        print(f"  Right IP: {args_cli.right_ip}")
        print(f"  ZMQ Port: {args_cli.zmq_port}")
        print("=" * 60)

        left_joints, right_joints, left_grip, right_grip = read_real_robot_positions(
            args_cli.left_ip, args_cli.right_ip
        )

        print(f"[SYNC] Read results: left_joints={left_joints}, right_joints={right_joints}")

        if left_joints is None and right_joints is None:
            print("[SYNC] ERROR: Could not read ANY real robot positions!")
            print("[SYNC] Check that:")
            print("  1. Robot IPs are correct")
            print("  2. Robots are powered on and connected to network")
            print("  3. ur_rtde is properly installed")
            print("[SYNC] Exiting because --real-robot requires valid robot connection")
            simulation_app.close()
            return

        real_robot_initial_pos = {
            "left": left_joints,
            "right": right_joints,
            "left_gripper": left_grip,
            "right_gripper": right_grip,
        }
        print(f"[SYNC] real_robot_initial_pos created: {real_robot_initial_pos}")
        print("[SYNC] Will sync simulation to real robot positions")

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Update initial joint positions from real robot if available
    if real_robot_initial_pos is not None:
        print(f"[SYNC] Updating env_cfg with real robot positions...")
        print(f"[SYNC] env_cfg.scene has left_arm: {hasattr(env_cfg.scene, 'left_arm')}")
        print(f"[SYNC] env_cfg.scene has right_arm: {hasattr(env_cfg.scene, 'right_arm')}")

        if hasattr(env_cfg.scene, "left_arm") and real_robot_initial_pos["left"] is not None:
            # Map real robot joints to simulation joint names
            joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            print(f"[SYNC] Setting LEFT arm joints: {real_robot_initial_pos['left']}")
            for i, name in enumerate(joint_names):
                env_cfg.scene.left_arm.init_state.joint_pos[name] = real_robot_initial_pos["left"][i]
                print(f"[SYNC]   {name} = {real_robot_initial_pos['left'][i]:.4f}")
            print("[SYNC] Updated LEFT arm initial positions in env_cfg")

        if hasattr(env_cfg.scene, "right_arm") and real_robot_initial_pos["right"] is not None:
            joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            print(f"[SYNC] Setting RIGHT arm joints: {real_robot_initial_pos['right']}")
            for i, name in enumerate(joint_names):
                env_cfg.scene.right_arm.init_state.joint_pos[name] = real_robot_initial_pos["right"][i]
                print(f"[SYNC]   {name} = {real_robot_initial_pos['right'][i]:.4f}")
            print("[SYNC] Updated RIGHT arm initial positions in env_cfg")
    else:
        print("[SYNC] WARNING: real_robot_initial_pos is None - this should not happen if --real-robot is set!")

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Initialize ZMQ publisher for real robot control
    zmq_publisher = None
    if args_cli.real_robot:
        zmq_publisher = RealRobotPublisher(port=args_cli.zmq_port)
        if not zmq_publisher.connect():
            logger.warning("[ZMQ] Publisher failed to start, continuing without real robot control")
            zmq_publisher = None

    # Initialize RTDE feedback for continuous sim-to-real synchronization
    real_robot_feedback = None
    if args_cli.real_robot:
        real_robot_feedback = RealRobotFeedback(args_cli.left_ip, args_cli.right_ip)
        if not real_robot_feedback.connect():
            logger.warning("[FEEDBACK] Failed to establish RTDE feedback, sim may drift from real robot")
            real_robot_feedback = None
        else:
            print("[FEEDBACK] Continuous sim-to-real synchronization enabled")

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for XR devices - activate by pressing SQUEEZE button
        teleoperation_active = False
        print("Teleoperation inactive - press SQUEEZE button on controller to activate")
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "vive":
                teleop_interface = Se3ViveController(
                    Se3ViveControllerCfg(pos_sensitivity=0.2 * sensitivity, rot_sensitivity=0.2 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    # After reset, explicitly sync simulation robot to real robot positions
    # This ensures sim matches real robot BEFORE any teleoperation starts
    print(real_robot_initial_pos)
    if real_robot_initial_pos is not None:
        print("[SYNC] Syncing simulation robot to real robot positions...")
        print(f"[SYNC] Left joints to set: {real_robot_initial_pos.get('left')}")
        print(f"[SYNC] Right joints to set: {real_robot_initial_pos.get('right')}")
        print(f"[SYNC] Left gripper to set: {real_robot_initial_pos.get('left_gripper')}")
        print(f"[SYNC] Right gripper to set: {real_robot_initial_pos.get('right_gripper')}")
        
        #TODO: Implement logic that triggers when only one arm is connected: aka
        # real_robot_initial_pos.get = None

        real_left_arm_joints = np.round(real_robot_initial_pos.get("left"), decimals=4)
        real_right_arm_joints = np.round(real_robot_initial_pos.get("right"), decimals=4)

        real_left_gripper = np.round(real_robot_initial_pos.get("left_gripper"), decimals=3)
        real_right_gripper = np.round(real_robot_initial_pos.get("right_gripper"), decimals=3)

        sim_left_arm_joints = np.round(env.scene["left_arm"].data.joint_pos[0,:6].detach().cpu().tolist(), decimals=4)
        sim_right_arm_joints = np.round(env.scene["right_arm"].data.joint_pos[0,:6].detach().cpu().tolist(), decimals = 4)

        print(f"[SYNC] Left SIM joints are: {sim_left_arm_joints}")

        # Set positions multiple times to ensure they stick
        while ((real_left_arm_joints != sim_left_arm_joints).any() or (real_right_arm_joints != sim_right_arm_joints).any()):
            set_robot_joint_positions(
                env,
                left_joints=real_left_arm_joints,
                right_joints=real_right_arm_joints,
                left_gripper=real_left_gripper,
                right_gripper=real_right_gripper,
            )
            # Step simulation to apply the changes
            env.sim.step()
            env.scene.update(env.sim.cfg.dt)

        # Verify the positions were set correctly
        left_arm = env.scene["left_arm"]
        right_arm = env.scene["right_arm"]
        print(f"[SYNC] Verification - LEFT arm actual: {left_arm.data.joint_pos[0, :6].cpu().numpy()}")
        print(f"[SYNC] Verification - RIGHT arm actual: {right_arm.data.joint_pos[0, :6].cpu().numpy()}")
        print(f"[SYNC] Verification - LEFT gripper actual: {left_arm.data.joint_pos[0, 6].cpu().item():.4f} rad")
        print(f"[SYNC] Verification - RIGHT gripper actual: {right_arm.data.joint_pos[0, 6].cpu().item():.4f} rad")
        print("[SYNC] Simulation robot now matches real robot position")

    print("Teleoperation started. Press 'R' to reset the environment.")
    if zmq_publisher is not None:
        print(f"[ZMQ] Publishing joint states on port {args_cli.zmq_port}")
        print("[INFO] Start the dual_ur5e_controller_standalone.py on the robot control PC")

    sync_counter = 0
    SYNC_EVERY_N_FRAMES = 240

    # simulate environment
    try:
        while simulation_app.is_running():
            try:
                # run everything in inference mode
                with torch.inference_mode():
                    # get device command
                    action = teleop_interface.advance()

                    # Only apply teleop commands when active
                    if teleoperation_active:
                        sync_counter += 1
                        if sync_counter >= SYNC_EVERY_N_FRAMES:
                            # Sync simulation to real robot state BEFORE computing RMPFlow
                            # This prevents drift between sim and real robot
                            if real_robot_feedback is not None:
                                sync_sim_to_real_robot(env, real_robot_feedback)
                                sync_counter=0

                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                        # apply actions
                        env.step(actions)
                    else:
                        env.sim.render()

                    # Publish joint states to real robot controller
                    # Always publish (even when frozen) so controller knows connection is alive
                    # Publish joint states to real robot controller
                    # CRITICAL: Pass teleoperation_active flag - controller only moves when True
                    if zmq_publisher is not None:
                        try:
                            left_joints, right_joints, left_grip, right_grip = get_joint_positions_from_env(env)
                            zmq_publisher.publish(
                                left_joints, right_joints, left_grip, right_grip,
                                teleop_active=teleoperation_active
                            )
                        except Exception as e:
                            logger.debug(f"[ZMQ] Failed to get/publish joint positions: {e}")

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break
    finally:
        # Cleanup
        if zmq_publisher is not None:
            zmq_publisher.close()
        if real_robot_feedback is not None:
            real_robot_feedback.close()

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
