# Author: Till Laube
# All rights reserved. 
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm retargeter for HTC Vive XR Elite controllers."""

import torch
from dataclasses import dataclass

from isaaclab.devices.device_base import DeviceBase, RetargeterBase, RetargeterCfg
from isaaclab.utils import configclass


class DualArmViveRetargeter(RetargeterBase):
    """Retargeter that splits 14 DOF Vive controller input into separate arm actions.

    Takes a 14-element tensor from dual Vive controllers:
    [left_pos(3), left_rot(3), left_grip(1), right_pos(3), right_rot(3), right_grip(1)]

    And splits it based on the configuration for each arm.
    """

    cfg: "DualArmViveRetargeterCfg"

    def __init__(self, cfg: "DualArmViveRetargeterCfg"):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)
        self._arm_side = cfg.arm_side

    def retarget(self, device_output: torch.Tensor) -> torch.Tensor:
        """Retarget device output to robot action.

        Args:
            device_output: 14-element tensor from Vive controllers

        Returns:
            7-element tensor for the specified arm [pos(3), rot(3), gripper(1)]
        """
        if device_output.shape[-1] != 14:
            raise ValueError(f"Expected 14 DOF input, got {device_output.shape[-1]}")

        # Split the input based on which arm this retargeter is for
        if self._arm_side == "left":
            # Left arm: elements 0-6
            return device_output[..., :7]
        elif self._arm_side == "right":
            # Right arm: elements 7-13
            return device_output[..., 7:14]
        else:
            raise ValueError(f"Invalid arm_side: {self._arm_side}. Must be 'left' or 'right'")


@configclass
class DualArmViveRetargeterCfg(RetargeterCfg):
    """Configuration for dual-arm Vive retargeter.

    Args:
        arm_side: Which arm this retargeter is for: "left" or "right"
    """

    retargeter_type: type = DualArmViveRetargeter
    arm_side: str = "left"  # "left" or "right"
