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

        # Debug: print sensitivity values
        print(f"[ViveRetargeter] Initialized with pos_sensitivity={self._pos_sensitivity}, rot_sensitivity={self._rot_sensitivity}")

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

        # Debug output
        if len(device_output) > 0:
            print(f"[ViveRetargeter] Device output keys: {list(device_output.keys())}")
            if len(controller_data) > 0:
                print(f"[ViveRetargeter] Controller data shape: {controller_data.shape}")
                if len(controller_data) > 0:
                    print(f"[ViveRetargeter] Controller pose: {controller_data[0][:3] if len(controller_data) > 0 else 'N/A'}")
            else:
                print(f"[ViveRetargeter] WARNING: No data for {self._hand_side} controller!")

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

            # Debug output
            print(f"[ViveRetargeter] Raw delta - pos: {position_delta_raw}, sensitivity: {self._pos_sensitivity}, final: {position_delta}")

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

        # Debug output
        print(f"[ViveRetargeter] Output DELTA - pos: {position}, rot: {rotation_vector}, gripper: {gripper}")

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
    """

    cfg: "ViveControllerDualArmRetargeterCfg"

    def __init__(self, cfg: "ViveControllerDualArmRetargeterCfg"):
        super().__init__(cfg)
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._trigger_threshold = cfg.trigger_threshold

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Return required data features for this retargeter."""
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _process_controller(self, controller_data: np.ndarray) -> np.ndarray:
        """Process single controller data to Se3 format (7 elements)."""
        default_output = np.zeros(7)
        default_output[6] = -1.0  # Gripper open

        if len(controller_data) == 0 or len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return default_output

        pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        if len(pose) < 7:
            return default_output

        # Position
        position = pose[:3] * self._pos_sensitivity

        # Quaternion to rotation vector
        quaternion = pose[3:7]
        quat_scipy = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        rot = Rotation.from_quat(quat_scipy)
        rotation_vector = rot.as_rotvec() * self._rot_sensitivity

        # Gripper from trigger
        gripper = -1.0
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value:
                trigger_value = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
                gripper = 1.0 if trigger_value > self._trigger_threshold else -1.0

        return np.concatenate([position, rotation_vector, [gripper]])

    def retarget(self, device_output: dict) -> torch.Tensor:
        """Retarget both controllers to dual-arm Se3 commands.

        Args:
            device_output: Dictionary with CONTROLLER_LEFT and CONTROLLER_RIGHT

        Returns:
            Tensor: [left(7), right(7)] = 14 elements
        """
        left_data = device_output.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_data = device_output.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        left_output = self._process_controller(left_data)
        right_output = self._process_controller(right_data)

        output = np.concatenate([left_output, right_output])
        return torch.tensor(output, dtype=torch.float32)


@configclass
class ViveControllerDualArmRetargeterCfg(RetargeterCfg):
    """Configuration for dual-arm Vive controller retargeter."""

    retargeter_type: type = ViveControllerDualArmRetargeter
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    trigger_threshold: float = 0.5
