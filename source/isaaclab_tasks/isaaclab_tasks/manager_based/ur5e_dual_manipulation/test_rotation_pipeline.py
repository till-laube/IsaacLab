#!/usr/bin/env python3
# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug script to test rotation processing pipeline for UR5e dual manipulation.

This script tests the rotation retargeting by:
1. Injecting clean 90° rotations as controller input (tests full pipeline)
2. Sending clean 90° rotations directly to RMPFlow (bypasses pipeline)
"""

import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation

# Import AppLauncher FIRST, before any isaaclab modules
import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test rotation processing pipeline for UR5e dual manipulation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW import isaaclab modules after simulation app is running
from isaaclab_tasks.manager_based.ur5e_dual_manipulation.ur5e_dual_manipulation_env_cfg import (
    Ur5eDualManipulationEnvCfg,
)
from isaaclab.envs import ManagerBasedRLEnv


class RotationPipelineTester:
    """Test harness for debugging rotation processing pipeline."""

    def __init__(self):
        """Initialize the test environment."""
        # Create environment
        env_cfg = Ur5eDualManipulationEnvCfg()
        env_cfg.scene.num_envs = 1
        self.env = ManagerBasedRLEnv(cfg=env_cfg)

        # Reset environment to initialize controllers
        print("Initializing environment...")
        self.env.reset()
        print("Environment initialized.\n")

        # Create retargeter directly from config
        retargeter_cfg = env_cfg.teleop_devices.devices["vive"].retargeters[0]
        self.retargeter = retargeter_cfg.retargeter_type(retargeter_cfg)

        # Store base rotation for right arm
        self.right_base_quat = retargeter_cfg.right_base_quat

        print("\n" + "="*80)
        print("ROTATION PIPELINE TESTER - RIGHT ARM ONLY")
        print("="*80)
        print(f"Right arm base rotation: {self.right_base_quat}")
        print(f"Right controller offset: {retargeter_cfg.right_controller_offset_quat}")
        print(f"Rotation sensitivity: {retargeter_cfg.rot_sensitivity}")
        print(f"Position sensitivity: {retargeter_cfg.pos_sensitivity}")
        print("="*80 + "\n")

    def create_controller_input(self, axis: str, angle_deg: float) -> dict:
        """Create fake controller input with a clean rotation around specified axis.

        Args:
            axis: 'x', 'y', or 'z'
            angle_deg: Rotation angle in degrees

        Returns:
            Dictionary simulating OpenXR device output
        """
        # Create rotation around specified axis
        if axis == 'x':
            rot = Rotation.from_euler('x', angle_deg, degrees=True)
        elif axis == 'y':
            rot = Rotation.from_euler('y', angle_deg, degrees=True)
        elif axis == 'z':
            rot = Rotation.from_euler('z', angle_deg, degrees=True)
        else:
            raise ValueError(f"Invalid axis: {axis}")

        # Convert to quaternion [x, y, z, w] (scipy format)
        quat_scipy = rot.as_quat()

        # Convert to [w, x, y, z] format for controller data
        quat_wxyz = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

        # Create controller data: [pose_row, input_row]
        # pose_row: [x, y, z, w, x, y, z] - position (identity) + quaternion
        pose_row = np.array([0.0, 0.0, 0.0, quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]])

        # input_row: [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, padding]
        input_row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        controller_data = np.array([pose_row, input_row], dtype=np.float32)

        # Return in OpenXR device output format
        from isaaclab.devices.device_base import DeviceBase
        return {
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT: controller_data,
            DeviceBase.TrackingTarget.CONTROLLER_LEFT: np.array([])  # Empty left controller
        }

    def test_controller_rotation(self, axis: str, angle_deg: float = 90.0):
        """Test rotation through full pipeline (controller input -> retargeter -> RMPFlow).

        Args:
            axis: 'x', 'y', or 'z'
            angle_deg: Rotation angle in degrees
        """
        print(f"\n{'='*80}")
        print(f"TEST: Controller {axis.upper()}-axis rotation ({angle_deg}°) -> Full Pipeline")
        print(f"{'='*80}")

        # Reset the retargeter to clear previous state
        self.retargeter.reset()

        # First frame: identity (establish baseline)
        identity_input = self.create_controller_input('x', 0.0)
        _ = self.retargeter.retarget(identity_input)

        # Second frame: apply rotation
        rotated_input = self.create_controller_input(axis, angle_deg)
        action = self.retargeter.retarget(rotated_input)

        # Retargeter returns (14,) but environment expects (1, 14)
        action = action.unsqueeze(0)

        # Extract right arm action (indices 7-13: pos_delta(3), rot_delta(3), gripper(1))
        right_pos_delta = action[0, 7:10].numpy()
        right_rot_delta = action[0, 10:13].numpy()
        right_gripper = action[0, 13].item()

        print(f"Input: {angle_deg}° rotation around controller {axis.upper()}-axis")
        print(f"Output from retargeter:")
        print(f"  Position delta: {right_pos_delta}")
        print(f"  Rotation delta: {right_rot_delta}")
        print(f"  Rotation magnitude: {np.linalg.norm(right_rot_delta):.4f} rad = {np.rad2deg(np.linalg.norm(right_rot_delta)):.2f}°")
        print(f"  Rotation axis: {right_rot_delta / (np.linalg.norm(right_rot_delta) + 1e-8)}")
        print(f"  Gripper: {right_gripper}")

        # Apply action repeatedly to let robot move
        print(f"\nApplying rotation for 5.0s...")
        apply_steps = int(5.0 / self.env.step_dt)
        for i in range(apply_steps):
            try:
                # RL environment returns: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
            except Exception as e:
                print(f"ERROR during environment step {i}: {e}")
                import traceback
                traceback.print_exc()
                break

        return action

    def test_direct_rmpflow(self, axis: str, angle_deg: float = 90.0):
        """Test rotation by sending directly to RMPFlow (bypass retargeter).

        Args:
            axis: 'x', 'y', or 'z' (in base frame coordinates)
            angle_deg: Rotation angle in degrees
        """
        print(f"\n{'='*80}")
        print(f"TEST: Direct RMPFlow {axis.upper()}-axis rotation ({angle_deg}°) -> Bypass Pipeline")
        print(f"{'='*80}")

        # Get initial EE pose for reference
        right_arm = self.env.scene["right_arm"]
        initial_ee_pos = right_arm.data.body_pos_w[:, right_arm.find_bodies("tcp_link")[0][0]].cpu().numpy()[0]
        initial_ee_quat = right_arm.data.body_quat_w[:, right_arm.find_bodies("tcp_link")[0][0]].cpu().numpy()[0]
        print(f"\nInitial EE state (world frame):")
        print(f"  Position: {initial_ee_pos}")
        print(f"  Quaternion (w,x,y,z): {initial_ee_quat}")

        # Create rotation vector in base frame
        angle_rad = np.deg2rad(angle_deg)
        rot_vec = np.zeros(3)
        if axis == 'x':
            rot_vec[0] = angle_rad
        elif axis == 'y':
            rot_vec[1] = angle_rad
        elif axis == 'z':
            rot_vec[2] = angle_rad
        else:
            raise ValueError(f"Invalid axis: {axis}")

        # Create action: [left_pos(3), left_rot(3), left_grip(1), right_pos(3), right_rot(3), right_grip(1)]
        # Shape must be (num_envs, action_dim) = (1, 14)
        action = torch.zeros(1, 14)
        action[0, 7:10] = torch.tensor([0.0, 0.0, 0.0])  # Right arm position delta = 0
        action[0, 10:13] = torch.tensor(rot_vec)  # Right arm rotation delta
        action[0, 13] = -1.0  # Gripper open

        print(f"\nInput: {angle_deg}° rotation around base frame {axis.upper()}-axis")
        print(f"Direct action to RMPFlow:")
        print(f"  Position delta: {action[0, 7:10].numpy()}")
        print(f"  Rotation delta (axis-angle): {action[0, 10:13].numpy()}")
        print(f"  Rotation magnitude: {np.linalg.norm(rot_vec):.4f} rad = {np.rad2deg(np.linalg.norm(rot_vec)):.2f}°")
        print(f"  Gripper: {action[0, 13].item()}")

        # Apply action repeatedly to let robot move
        print(f"\nApplying rotation for 5.0s...")
        apply_steps = int(5.0 / self.env.step_dt)
        for i in range(apply_steps):
            try:
                # RL environment returns: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
            except Exception as e:
                print(f"ERROR during environment step {i}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Get final EE pose for comparison
        final_ee_pos = right_arm.data.body_pos_w[:, right_arm.find_bodies("tcp_link")[0][0]].cpu().numpy()[0]
        final_ee_quat = right_arm.data.body_quat_w[:, right_arm.find_bodies("tcp_link")[0][0]].cpu().numpy()[0]

        print(f"\nFinal EE state (world frame):")
        print(f"  Position: {final_ee_pos}")
        print(f"  Quaternion (w,x,y,z): {final_ee_quat}")

        # Compute changes
        pos_change = final_ee_pos - initial_ee_pos
        print(f"\nPosition change:")
        print(f"  Delta: {pos_change}")
        print(f"  Magnitude: {np.linalg.norm(pos_change):.4f} m")
        print(f"  WARNING: Position changed even though delta was [0,0,0]!" if np.linalg.norm(pos_change) > 0.001 else "  OK: Position stayed approximately the same")

        return action

    def wait_and_observe(self, duration: float = 2.0):
        """Wait and let the robot move for specified duration.

        Args:
            duration: Wait duration in seconds
        """
        print(f"\nWaiting {duration}s for robot to move...")
        steps = int(duration / self.env.step_dt)
        # Shape must be (num_envs, action_dim) = (1, 14)
        zero_action = torch.zeros(1, 14)
        zero_action[0, 6] = -1.0  # Left gripper open
        zero_action[0, 13] = -1.0  # Right gripper open

        for i in range(steps):
            if not simulation_app.is_running():
                print(f"WARNING: Simulation stopped running at step {i}/{steps}")
                break
            try:
                # RL environment returns: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = self.env.step(zero_action)
            except Exception as e:
                print(f"ERROR during wait step {i}: {e}")
                break

        print("Wait complete.")

    def reset_environment(self):
        """Reset the environment."""
        print("\nResetting environment...")
        self.env.reset()
        # Also reset retargeter to clear previous pose tracking
        self.retargeter.reset()
        print("Reset complete.\n")

    def run_test_sequence(self):
        """Run the complete test sequence."""
        print("\n" + "█"*80)
        print("STARTING TEST SEQUENCE")
        print("█"*80 + "\n")

        # Test 1: Controller X-axis
        self.test_controller_rotation('x', 90.0)
        self.wait_and_observe(2.0)
        self.reset_environment()

        # Test 2: Controller Y-axis
        self.test_controller_rotation('y', 90.0)
        self.wait_and_observe(2.0)
        self.reset_environment()

        # Test 3: Controller Z-axis
        self.test_controller_rotation('z', 90.0)
        self.wait_and_observe(2.0)
        self.reset_environment()

        # Test 4: Direct RMPFlow X-axis
        self.test_direct_rmpflow('x', 90.0)
        self.wait_and_observe(2.0)
        self.reset_environment()

        # Test 5: Direct RMPFlow Y-axis
        self.test_direct_rmpflow('y', 90.0)
        self.wait_and_observe(2.0)
        self.reset_environment()

        # Test 6: Direct RMPFlow Z-axis
        self.test_direct_rmpflow('z', 90.0)
        self.wait_and_observe(2.0)
        self.reset_environment()

        print("\n" + "█"*80)
        print("TEST SEQUENCE COMPLETE")
        print("█"*80 + "\n")

    def close(self):
        """Clean up and close the environment."""
        self.env.close()


def main():
    """Main function to run the rotation pipeline tests."""
    tester = RotationPipelineTester()

    try:
        tester.run_test_sequence()

        # Keep simulator open for final inspection
        print("\nTests complete. Press Ctrl+C to exit.")
        while simulation_app.is_running():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        tester.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
