# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retargeter for Vive motion controllers to manipulator Se3 commands."""

import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase, RetargeterBase, RetargeterCfg
from isaaclab.utils import configclass


class ViveControllerSe3Retargeter(RetargeterBase):
    """Retargeter that converts HTC Vive XR Elite controller OpenXR data (collected from SteamVR
    using ALVR)to Se3 commands.

    Takes OpenXR controller data (pose + inputs) and outputs Se3 format:
    [position(3), rotation_vector(3), gripper(1)] = 7 elements

    The gripper is controlled by the trigger (index finger trigger on Vive controllers).
    """

    cfg: "ViveControllerSe3RetargeterCfg"

    def __init__(self, cfg: "ViveControllerSe3RetargeterCfg"):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)
        self._hand_side = cfg.hand_side
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._trigger_threshold = cfg.trigger_threshold

        # Track previous pose for computing deltas (for relative mode)
        self._prev_position = None
        self._prev_quaternion = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Return required data features for this retargeter."""
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, device_output: dict) -> torch.Tensor:
        """Retarget OpenXR controller data to Se3 command.

        Args:
            device_output: Dictionary with TrackingTarget keys
                          CONTROLLER_LEFT/RIGHT values are 2D arrays: [pose(7), inputs(7+)]

        Returns:
            Tensor: [pos(3), rot_vec(3), gripper(1)] = 7 elements
        """
        # Select the appropriate controller based on hand_side
        if self._hand_side == "left":
            controller_key = DeviceBase.TrackingTarget.CONTROLLER_LEFT
        else:
            controller_key = DeviceBase.TrackingTarget.CONTROLLER_RIGHT

        # Get controller data
        controller_data = device_output.get(controller_key, np.array([]))

        # Default output if no controller data
        default_output = np.zeros(7)
        default_output[6] = -1.0  # Gripper open by default

        if len(controller_data) == 0:
            return torch.tensor(default_output, dtype=torch.float32)

        # Extract pose (row 0)
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return torch.tensor(default_output, dtype=torch.float32)

        pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        if len(pose) < 7:
            return torch.tensor(default_output, dtype=torch.float32)

        # Extract current position and quaternion
        current_position = pose[:3]
        current_quaternion = pose[3:7]  # [qw, qx, qy, qz]

        # Compute deltas for relative mode
        if self._prev_position is None:
            # First frame: initialize with current pose, output zero delta
            self._prev_position = current_position.copy()
            self._prev_quaternion = current_quaternion.copy()
            position_delta = np.zeros(3)
            rotation_delta = np.zeros(3)
        else:
            # Compute position delta
            position_delta_raw = current_position - self._prev_position
            position_delta = position_delta_raw * self._pos_sensitivity

            # Compute rotation delta
            # Convert quaternions to rotations
            quat_prev_scipy = np.array([self._prev_quaternion[1], self._prev_quaternion[2],
                                       self._prev_quaternion[3], self._prev_quaternion[0]])
            quat_curr_scipy = np.array([current_quaternion[1], current_quaternion[2],
                                       current_quaternion[3], current_quaternion[0]])
            rot_prev = Rotation.from_quat(quat_prev_scipy)
            rot_curr = Rotation.from_quat(quat_curr_scipy)

            # Compute relative rotation (delta)
            rot_delta = rot_curr * rot_prev.inv()
            rotation_delta_raw = rot_delta.as_rotvec()
            rotation_delta = rotation_delta_raw * self._rot_sensitivity

            # Update previous pose
            self._prev_position = current_position.copy()
            self._prev_quaternion = current_quaternion.copy()

        position = position_delta
        rotation_vector = rotation_delta

        # Extract gripper state from trigger input
        gripper = -1.0  # Default: open
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value:
                trigger_value = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
                # Convert trigger (0.0-1.0) to gripper command (-1.0 open, 1.0 close)
                gripper = 1.0 if trigger_value > self._trigger_threshold else -1.0

        # Combine into Se3 format
        output = np.concatenate([position, rotation_vector, [gripper]])

        return torch.tensor(output, dtype=torch.float32)


@configclass
class ViveControllerSe3RetargeterCfg(RetargeterCfg):
    """Configuration for Vive controller Se3 retargeter.

    Args:
        hand_side: Which controller to use: "left" or "right"
        pos_sensitivity: Position sensitivity multiplier
        rot_sensitivity: Rotation sensitivity multiplier
        trigger_threshold: Trigger value (0.0-1.0) to consider gripper closed
    """

    retargeter_type: type = ViveControllerSe3Retargeter
    hand_side: str = "left"  # "left" or "right"
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    trigger_threshold: float = 0.5


class ViveControllerDualArmRetargeter(RetargeterBase):
    """Retargeter that converts both HTC Vive Elite XR controllers to dual-arm Se3 commands.

    Outputs 14 DOF: [left_pos(3), left_rot(3), left_grip(1), right_pos(3), right_rot(3), right_grip(1)]

    This retargeter computes DELTA movements for use with relative mode RMPFlow.
    """

    cfg: "ViveControllerDualArmRetargeterCfg"

    def __init__(self, cfg: "ViveControllerDualArmRetargeterCfg"):
        super().__init__(cfg)
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._trigger_threshold = cfg.trigger_threshold

        # Store base rotations for coordinate transformation
        # Convert from [w,x,y,z] to scipy format [x,y,z,w] and create rotation objects
        left_quat_scipy = [cfg.left_base_quat[1], cfg.left_base_quat[2], cfg.left_base_quat[3], cfg.left_base_quat[0]]
        right_quat_scipy = [cfg.right_base_quat[1], cfg.right_base_quat[2], cfg.right_base_quat[3], cfg.right_base_quat[0]]
        self._left_base_rot = Rotation.from_quat(left_quat_scipy)
        self._right_base_rot = Rotation.from_quat(right_quat_scipy)

        # Inverse rotations for world→base transformation
        self._left_world_to_base = self._left_base_rot.inv()
        self._right_world_to_base = self._right_base_rot.inv()

        # Arm-specific controller orientation offsets to align controller frame with gripper frame
        # These account for both the controller-gripper frame difference AND the arm base rotation
        left_controller_offset_scipy = [cfg.left_controller_offset_quat[1], cfg.left_controller_offset_quat[2],
                                        cfg.left_controller_offset_quat[3], cfg.left_controller_offset_quat[0]]
        right_controller_offset_scipy = [cfg.right_controller_offset_quat[1], cfg.right_controller_offset_quat[2],
                                         cfg.right_controller_offset_quat[3], cfg.right_controller_offset_quat[0]]
        self._left_controller_offset_rot = Rotation.from_quat(left_controller_offset_scipy)
        self._right_controller_offset_rot = Rotation.from_quat(right_controller_offset_scipy)

        # Track previous poses for computing deltas (for relative mode)
        self._prev_left_position = None
        self._prev_left_quaternion = None
        self._prev_right_position = None
        self._prev_right_quaternion = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Return required data features for this retargeter."""
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def reset(self):
        """Reset the retargeter state.

        This clears the previous pose tracking, which is important when:
        - XR mode is activated/deactivated
        - The coordinate system changes
        - The user resets the environment
        """
        self._prev_left_position = None
        self._prev_left_quaternion = None
        self._prev_right_position = None
        self._prev_right_quaternion = None
        # print("[DualArmRetargeter] Reset - cleared previous pose tracking")

    def _process_controller(
        self, controller_data: np.ndarray, prev_position, prev_quaternion, is_left: bool,
        world_to_base: Rotation, controller_offset: Rotation
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process single controller data to Se3 DELTA format (7 elements).

        Args:
            controller_data: Raw controller data from OpenXR
            prev_position: Previous position for delta computation (or None for first frame)
            prev_quaternion: Previous quaternion for delta computation (or None for first frame)
            is_left: Whether this is the left controller (for debug output)
            world_to_base: Rotation object to transform deltas from world to arm base frame
            controller_offset: Rotation offset to align controller frame with gripper frame (applied in arm base frame)

        Returns:
            Tuple of (output_array, new_position, new_quaternion)
        """
        default_output = np.zeros(7)
        default_output[6] = -1.0  # Gripper open

        if len(controller_data) == 0 or len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return default_output, prev_position, prev_quaternion

        pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        if len(pose) < 7:
            return default_output, prev_position, prev_quaternion

        # Extract current position and quaternion
        current_position = pose[:3]
        current_quaternion = pose[3:7]  # [qw, qx, qy, qz]

        # Debug: Print actual controller position
        # side = "LEFT" if is_left else "RIGHT"
        # print(f"[DualArmRetargeter {side}] Current position: {current_position}, quat: {current_quaternion}")
        # if prev_position is not None:
        #     print(f"[DualArmRetargeter {side}] Previous position: {prev_position}")

        # Compute deltas for relative mode
        if prev_position is None:
            # First frame: initialize with current pose, output zero delta
            position_delta = np.zeros(3)
            rotation_delta = np.zeros(3)
        else:
            # Compute position delta in world frame
            position_delta_raw = current_position - prev_position
            position_delta = position_delta_raw * self._pos_sensitivity

            # Compute rotation delta in world frame
            # Convert quaternions to rotations (scipy format: [x, y, z, w])
            quat_prev_scipy = np.array(
                [prev_quaternion[1], prev_quaternion[2], prev_quaternion[3], prev_quaternion[0]]
            )
            quat_curr_scipy = np.array(
                [current_quaternion[1], current_quaternion[2], current_quaternion[3], current_quaternion[0]]
            )
            rot_prev = Rotation.from_quat(quat_prev_scipy)
            rot_curr = Rotation.from_quat(quat_curr_scipy)

            # Step 1: Compute relative rotation (delta) in WORLD frame
            rot_delta_world = rot_curr * rot_prev.inv()

            # Step 2: Transform delta from world frame to arm base frame
            rot_delta_base = world_to_base * rot_delta_world * world_to_base.inv()

            # Step 3: Apply controller offset in arm base frame
            # This aligns controller coordinate axes with gripper coordinate axes
            rot_delta_final = rot_delta_base * controller_offset

            # Convert to rotation vector (axis-angle) and apply sensitivity
            rotation_delta = rot_delta_final.as_rotvec() * self._rot_sensitivity

            # Transform position delta from world frame to arm base frame
            position_delta = world_to_base.apply(position_delta)

        position = position_delta
        rotation_vector = rotation_delta

        # Gripper from trigger
        gripper = -1.0
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value:
                trigger_value = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
                gripper = 1.0 if trigger_value > self._trigger_threshold else -1.0

        output = np.concatenate([position, rotation_vector, [gripper]])
        return output, current_position.copy(), current_quaternion.copy()

    def retarget(self, device_output: dict) -> torch.Tensor:
        """Retarget both controllers to dual-arm Se3 DELTA commands.

        Args:
            device_output: Dictionary with CONTROLLER_LEFT and CONTROLLER_RIGHT

        Returns:
            Tensor: [left_delta(7), right_delta(7)] = 14 elements
                   where each 7-element block is [pos_delta(3), rot_delta(3), gripper(1)]
        """
        left_data = device_output.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_data = device_output.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        # Process left controller and update tracking
        # Pass left arm's world_to_base transformation and controller offset
        left_output, new_left_pos, new_left_quat = self._process_controller(
            left_data, self._prev_left_position, self._prev_left_quaternion, is_left=True,
            world_to_base=self._left_world_to_base, controller_offset=self._left_controller_offset_rot
        )
        self._prev_left_position = new_left_pos
        self._prev_left_quaternion = new_left_quat

        # Process right controller and update tracking
        # Pass right arm's world_to_base transformation and controller offset
        right_output, new_right_pos, new_right_quat = self._process_controller(
            right_data, self._prev_right_position, self._prev_right_quaternion, is_left=False,
            world_to_base=self._right_world_to_base, controller_offset=self._right_controller_offset_rot
        )
        self._prev_right_position = new_right_pos
        self._prev_right_quaternion = new_right_quat

        output = np.concatenate([left_output, right_output])
        return torch.tensor(output, dtype=torch.float32)


@configclass
class ViveControllerDualArmRetargeterCfg(RetargeterCfg):
    """Configuration for dual-arm Vive controller retargeter.

    This retargeter transforms controller movements into gripper commands for dual-arm manipulation.
    It handles two key coordinate transformations in sequence:

    **Transformation Pipeline:**
    1. Compute rotation delta in world frame: delta_world = curr_controller * prev_controller^-1
    2. Transform delta from world frame to arm base frame: delta_base = R_world_to_base * delta_world * R_world_to_base^-1
    3. Apply controller offset in arm base frame: delta_final = delta_base * controller_offset
    4. Convert to rotation vector (axis-angle representation)

    Args:
        pos_sensitivity: Multiplier for position deltas (higher = more sensitive)
        rot_sensitivity: Multiplier for rotation deltas (higher = more sensitive)
        trigger_threshold: Trigger value (0.0-1.0) to consider gripper closed

        left_base_quat: Quaternion [w,x,y,z] representing the left arm base orientation in world frame.
                        Used to transform movement deltas from world coordinates to arm base coordinates.
                        Example: For arm mounted at (180°, -45°, 90°), this captures that rotation.

        right_base_quat: Quaternion [w,x,y,z] representing the right arm base orientation in world frame.
                         Used to transform movement deltas from world coordinates to arm base coordinates.
                         Example: For arm mounted at (180°, 45°, 90°), this captures that rotation.

        left_controller_offset_quat: Quaternion [w,x,y,z] to align left controller axes with left gripper axes.
                                     Applied IN ARM BASE FRAME as: delta_final = delta_base * offset
                                     This compensates for controller vs gripper coordinate frame differences.
                                     Identity (1,0,0,0) means controller axes match gripper axes.

        right_controller_offset_quat: Quaternion [w,x,y,z] to align right controller axes with right gripper axes.
                                      Applied IN ARM BASE FRAME as: delta_final = delta_base * offset
                                      This compensates for controller vs gripper coordinate frame differences.
                                      Identity (1,0,0,0) means controller axes match gripper axes.

    """

    retargeter_type: type = ViveControllerDualArmRetargeter
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    trigger_threshold: float = 0.5
    left_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Identity by default
    right_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Identity by default
    left_controller_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Identity by default
    right_controller_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Identity by default
