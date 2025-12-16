# Copyright (c) Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom environment for UR5e dual manipulation with controller visualization."""

import torch
import numpy as np
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .ur5e_dual_manipulation_env_cfg import Ur5eDualManipulationEnvCfg


class Ur5eDualManipulationEnv(ManagerBasedRLEnv):
    """Custom environment extending the base RL environment with controller visualization.

    This adds:
    - Visualization of controller coordinate frames at the gripper TCPs
    - Ability to see the controller orientation vs. end-effector orientation

    Args:
        enable_controller_viz: Enable visualization of controller coordinate frames (default: True)
        debug_controller_data: Print controller orientation data for debugging (default: False)
    """

    cfg: Ur5eDualManipulationEnvCfg

    def __init__(
        self,
        cfg: Ur5eDualManipulationEnvCfg,
        render_mode: str | None = None,
        enable_controller_viz: bool = True,
        debug_controller_data: bool = False,
        **kwargs
    ):
        """Initialize the environment with controller visualization.

        Args:
            cfg: Environment configuration
            render_mode: Rendering mode
            enable_controller_viz: Enable controller frame visualization
            debug_controller_data: Print debug info about controller orientation
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Store settings
        self._enable_controller_viz = enable_controller_viz
        self._debug_controller_data = debug_controller_data

        # Frame counter for debug output throttling
        self._debug_frame_count = 0
        self._first_left_viz_update = True
        self._first_right_viz_update = True

        # Create visualization markers for controller coordinate frames
        # These show the controller orientation at the TCP location
        if self._enable_controller_viz:
            self._create_controller_visualizations()

    def _create_controller_visualizations(self):
        """Create visualization markers for controller coordinate frames."""
        # Left controller frame visualization
        try:
            left_controller_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/LeftControllerFrame",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.15, 0.15, 0.15),  # Slightly larger than TCP frame for distinction
                    ),
                }
            )
            self.left_controller_marker = VisualizationMarkers(left_controller_marker_cfg)
            print(f"[Controller Viz] Created LEFT controller visualization marker at {self.left_controller_marker.prim_path}")
        except Exception as e:
            print(f"[Controller Viz] ERROR: Failed to create LEFT controller visualization: {e}")
            self.left_controller_marker = None
            self._enable_controller_viz = False

        # Right controller frame visualization
        try:
            right_controller_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/RightControllerFrame",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.15, 0.15, 0.15),  # Slightly larger than TCP frame for distinction
                    ),
                }
            )
            self.right_controller_marker = VisualizationMarkers(right_controller_marker_cfg)
            print(f"[Controller Viz] Created RIGHT controller visualization marker at {self.right_controller_marker.prim_path}")
        except Exception as e:
            print(f"[Controller Viz] ERROR: Failed to create RIGHT controller visualization: {e}")
            self.right_controller_marker = None
            # Don't disable viz completely if only right fails
            if self.left_controller_marker is None:
                self._enable_controller_viz = False

    def step(self, action: torch.Tensor) -> tuple:
        """Execute environment step and update controller visualizations.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observations, rewards, resets, extras)
        """
        # Execute the normal environment step
        obs, rew, terminated, truncated, extras = super().step(action)

        # Update controller visualizations
        self._update_controller_visualizations()

        return obs, rew, terminated, truncated, extras

    def _update_controller_visualizations(self):
        """Update the controller coordinate frame visualizations for both arms."""
        if not self._enable_controller_viz:
            return

        # Update left controller visualization
        if self.left_controller_marker is not None:
            self._update_single_controller_viz(
                controller_side="left",
                tcp_frame_name="left_control_frame",
                controller_marker=self.left_controller_marker,
                is_first_update=self._first_left_viz_update,
            )
            if self._first_left_viz_update and self._get_left_controller_pose() is not None:
                self._first_left_viz_update = False

        # Update right controller visualization
        if self.right_controller_marker is not None:
            self._update_single_controller_viz(
                controller_side="right",
                tcp_frame_name="right_control_frame",
                controller_marker=self.right_controller_marker,
                is_first_update=self._first_right_viz_update,
            )
            if self._first_right_viz_update and self._get_right_controller_pose() is not None:
                self._first_right_viz_update = False

        self._debug_frame_count += 1

    def _update_single_controller_viz(
        self,
        controller_side: str,
        tcp_frame_name: str,
        controller_marker,
        is_first_update: bool,
    ):
        """Update visualization for a single controller.

        Args:
            controller_side: "left" or "right"
            tcp_frame_name: Name of the TCP frame sensor in the scene
            controller_marker: VisualizationMarkers instance
            is_first_update: Whether this is the first successful update
        """
        try:
            # Get the gripper TCP position (from the scene) for comparison
            tcp_pos_w = self.scene[tcp_frame_name].data.target_pos_w[0, 0, :]  # [3]
            tcp_quat_w = self.scene[tcp_frame_name].data.target_quat_w[0, 0, :]  # [4] as [w, x, y, z]
        except Exception as e:
            if self._debug_frame_count % 120 == 0:
                print(f"[Controller Viz] ERROR: Failed to get {controller_side} TCP data: {e}")
            return

        # Get controller pose (position + orientation)
        if controller_side == "left":
            controller_pose = self._get_left_controller_pose()
        else:
            controller_pose = self._get_right_controller_pose()

        if controller_pose is not None:
            controller_pos, controller_quat = controller_pose  # Unpack position and quaternion

            # Debug output (throttled to every 30 frames = ~0.5 seconds at 60Hz)
            if self._debug_controller_data and self._debug_frame_count % 30 == 0:
                tcp_pos_np = tcp_pos_w.cpu().numpy()
                tcp_quat_np = tcp_quat_w.cpu().numpy()
                print(f"[Controller Viz Debug - {controller_side.upper()}]")
                print(f"  TCP Position: {tcp_pos_np}")
                print(f"  TCP Quat [w,x,y,z]: {tcp_quat_np}")
                print(f"  Controller Position: {controller_pos}")
                print(f"  Controller Quat [w,x,y,z]: {controller_quat}")
                # Compute position difference
                pos_diff = controller_pos - tcp_pos_np
                print(f"  Position Difference (controller - TCP): {pos_diff}")
                print(f"  Position Distance: {np.linalg.norm(pos_diff):.4f} m")
                # Compute rotation difference
                from scipy.spatial.transform import Rotation
                tcp_rot = Rotation.from_quat([tcp_quat_np[1], tcp_quat_np[2], tcp_quat_np[3], tcp_quat_np[0]])
                ctrl_rot = Rotation.from_quat([controller_quat[1], controller_quat[2], controller_quat[3], controller_quat[0]])
                diff_rot = ctrl_rot * tcp_rot.inv()
                diff_euler = diff_rot.as_euler('xyz', degrees=True)
                print(f"  Rotation Difference (euler XYZ degrees): {diff_euler}")

            # Visualize the controller frame at its ACTUAL position in world space
            # marker_indices: which marker prototype to use (0 = "frame")
            # translations: controller's actual position in world space
            # orientations: controller's actual orientation [w, x, y, z]
            controller_marker.visualize(
                marker_indices=[0],
                translations=controller_pos.reshape(1, 3),
                orientations=controller_quat.reshape(1, 4),
            )

            # Log first successful visualization
            if is_first_update:
                print(f"[Controller Viz] ✓ Successfully visualizing {controller_side.upper()} controller pose!")
                print(f"[Controller Viz]   Controller Position: {controller_pos}")
                print(f"[Controller Viz]   You should see a LARGER coordinate frame at the controller's actual location")
                print(f"[Controller Viz]   Compare it with the TCP frame at the {controller_side} gripper")
        else:
            # No controller data - hide the marker by moving it far away
            controller_marker.visualize(
                translations=np.array([[0.0, 0.0, -100.0]]),
            )

    def _get_left_controller_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get the left controller pose (position + orientation) in world space.

        Returns:
            Tuple of (position [x,y,z], quaternion [w,x,y,z]) in world space, or None if not available
        """
        try:
            # Import XRCore to access controller data directly
            from omni.kit.xr.core import XRCore

            xr_core = XRCore.get_singleton()
            if xr_core is None:
                if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                    print("[Controller Viz] XRCore not available")
                return None

            # Get the left controller device
            left_controller = xr_core.get_input_device("/user/hand/left")
            if left_controller is None:
                if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                    print("[Controller Viz] Left controller device not found")
                return None

            # Query the controller pose - use get_virtual_world_pose() which returns a matrix
            # This is the same method used in _query_controller in openxr_device.py
            controller_pose_matrix = left_controller.get_virtual_world_pose()
            if controller_pose_matrix is None:
                if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                    print("[Controller Viz] Left controller pose matrix is None")
                return None

            # Extract position and orientation from the pose matrix
            position = controller_pose_matrix.ExtractTranslation()
            quat = controller_pose_matrix.ExtractRotationQuat()
            quati = quat.GetImaginary()
            quatw = quat.GetReal()

            # Convert to numpy arrays
            controller_pos_world = np.array([position[0], position[1], position[2]], dtype=np.float32)
            controller_quat_world = np.array([quatw, quati[0], quati[1], quati[2]], dtype=np.float32)

            # The pose is already in virtual world space (transformed by the XR anchor)
            # so we can return it directly
            return (controller_pos_world, controller_quat_world)

        except Exception as e:
            # XRCore not available or error accessing controller
            # This is expected when not running in XR mode
            if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                print(f"[Controller Viz] Exception getting LEFT controller pose: {e}")
            return None

    def _get_right_controller_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get the right controller pose (position + orientation) in world space.

        Returns:
            Tuple of (position [x,y,z], quaternion [w,x,y,z]) in world space, or None if not available
        """
        try:
            # Import XRCore to access controller data directly
            from omni.kit.xr.core import XRCore

            xr_core = XRCore.get_singleton()
            if xr_core is None:
                if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                    print("[Controller Viz] XRCore not available")
                return None

            # Get the right controller device
            right_controller = xr_core.get_input_device("/user/hand/right")
            if right_controller is None:
                if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                    print("[Controller Viz] Right controller device not found")
                return None

            # Query the controller pose - use get_virtual_world_pose() which returns a matrix
            # This is the same method used in _query_controller in openxr_device.py
            controller_pose_matrix = right_controller.get_virtual_world_pose()
            if controller_pose_matrix is None:
                if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                    print("[Controller Viz] Right controller pose matrix is None")
                return None

            # Extract position and orientation from the pose matrix
            position = controller_pose_matrix.ExtractTranslation()
            quat = controller_pose_matrix.ExtractRotationQuat()
            quati = quat.GetImaginary()
            quatw = quat.GetReal()

            # Convert to numpy arrays
            controller_pos_world = np.array([position[0], position[1], position[2]], dtype=np.float32)
            controller_quat_world = np.array([quatw, quati[0], quati[1], quati[2]], dtype=np.float32)

            # The pose is already in virtual world space (transformed by the XR anchor)
            # so we can return it directly
            return (controller_pos_world, controller_quat_world)

        except Exception as e:
            # XRCore not available or error accessing controller
            # This is expected when not running in XR mode
            if self._debug_controller_data and self._debug_frame_count % 120 == 0:
                print(f"[Controller Viz] Exception getting RIGHT controller pose: {e}")
            return None
