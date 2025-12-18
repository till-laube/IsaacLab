# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug wrapper for RMPFlow action to track rotation pipeline issues."""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.utils import configclass


class DebugRMPFlowAction(RMPFlowAction):
    """RMPFlow action with extensive debug output to diagnose rotation issues."""

    def __init__(self, cfg: RMPFlowActionCfg, env):
        super().__init__(cfg, env)
        self.step_count = 0
        self.debug_enabled = True  # Set to False to disable debug output

    def process_actions(self, actions: torch.Tensor):
        """Process actions with debug output."""
        # Store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # Debug: Print input actions every 10 steps
        if self.debug_enabled and self.step_count % 10 == 0:
            print(f"\n[DEBUG Step {self.step_count}] RMPFlowAction.process_actions()")
            print(f"  Raw action input: {self._raw_actions[0].cpu().numpy()}")
            print(f"  Processed action: {self._processed_actions[0].cpu().numpy()}")
            print(f"  Position delta: {self._processed_actions[0, 0:3].cpu().numpy()}")
            print(f"  Rotation delta (axis-angle): {self._processed_actions[0, 3:6].cpu().numpy()}")

        # If use_relative_mode is True, then the controller will apply delta change to the current ee_pose.
        if self.cfg.use_relative_mode:
            # Obtain quantities from simulation
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

            if self.debug_enabled and self.step_count % 10 == 0:
                print(f"\n  Current EE pose (base frame):")
                print(f"    Position: {ee_pos_curr[0].cpu().numpy()}")
                print(f"    Quaternion (w,x,y,z): {ee_quat_curr[0].cpu().numpy()}")

            # Compute ee_pose_targets use_relative_actions
            if ee_pos_curr is None or ee_quat_curr is None:
                raise ValueError(
                    "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                )
            self.ee_pos_des, self.ee_quat_des = math_utils.apply_delta_pose(
                ee_pos_curr, ee_quat_curr, self._processed_actions
            )

            if self.debug_enabled and self.step_count % 10 == 0:
                print(f"\n  Target EE pose after apply_delta_pose (base frame):")
                print(f"    Position: {self.ee_pos_des[0].cpu().numpy()}")
                print(f"    Quaternion (w,x,y,z): {self.ee_quat_des[0].cpu().numpy()}")

                # Compute changes
                pos_change = (self.ee_pos_des[0] - ee_pos_curr[0]).cpu().numpy()
                import numpy as np
                print(f"\n  Computed changes:")
                print(f"    Position change: {pos_change}")
                print(f"    Position change magnitude: {np.linalg.norm(pos_change):.6f} m")

        else:  # If use_relative_mode is False, then the controller will apply absolute ee_pose.
            self.ee_pos_des = self._processed_actions[:, 0:3]
            self.ee_quat_des = self._processed_actions[:, 3:7]

        self.ee_pose_des = torch.cat([self.ee_pos_des, self.ee_quat_des], dim=1)  # shape: [n, 7]

        if self.debug_enabled and self.step_count % 10 == 0:
            print(f"\n  Final command to RMPFlow:")
            print(f"    Position: {self.ee_pose_des[0, 0:3].cpu().numpy()}")
            print(f"    Quaternion (w,x,y,z): {self.ee_pose_des[0, 3:7].cpu().numpy()}")

        # Set command into controller
        self._rmpflow_controller.set_command(self.ee_pose_des)

        self.step_count += 1


@configclass
class DebugRMPFlowActionCfg(RMPFlowActionCfg):
    """Configuration for debug RMPFlow action."""
    class_type: type = DebugRMPFlowAction
