from isaaclab.devices import DeviceBase, DeviceCfg
import torch
import openvr
from typing import Callable, Any, Literal
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

class Se3ViveController(DeviceBase):
    """Script to enable HTC Vive XR Elite headset and controllers via StreamVR(with ALVR) for SE(3)
    teleoperation in IsaacLab

    Supports two modes:
    - single_arm: Returns 7 DOF (3 pos + 3 rot_vec + 1 gripper) for one controller
    - dual_arm: Returns 14 DOF (3 pos + 3 rot_vec + 1 gripper per controller)
    """

    def __init__(self, cfg: "Se3ViveControllerCfg"):
        super().__init__(retargeters=None)

        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._sim_device = cfg.sim_device
        self._control_mode = cfg.control_mode
        self._primary_hand = cfg.primary_hand

        # Initialize OpenVR (only if not in XR mode)
        # Note: If Isaac Sim XR mode is active, OpenVR will conflict with OpenXR
        try:
            self._vr = openvr.init(openvr.VRApplication_Other)
            self._left_idx = None
            self._right_idx = None
            self._find_controllers()
        except openvr.error_code.InitError as e:
            print(f"[WARNING] Failed to initialize OpenVR: {e}")
            print("[WARNING] If XR mode is active, this is expected. Controller tracking will not work.")
            self._vr = None
            self._left_idx = None
            self._right_idx = None

        # Callbacks for button events
        self._callbacks = {}

        self.reset()

    def _find_controllers(self):
        """Discover connected controllers"""
        #TODO: option to select either controllers or handtracking

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self._vr.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self._vr.getControllerRoleForTrackedDeviceIndex(i)
                if role == openvr.TrackedControllerRole_LeftHand:
                    self._left_idx = i
                elif role == openvr.TrackedControllerRole_RightHand:
                    self._right_idx = i
        
        print(f"Found controllers - Left: {self._left_idx}, Right: {self._right_idx}")

    def reset(self):
        """Reset internal state."""
        # Store as xyz + quat internally, will convert to rot_vec on output
        self._left_pos = torch.zeros(3, device=self._sim_device)
        self._left_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._sim_device)  # w, x, y, z
        self._left_gripper = -1.0  # Open

        self._right_pos = torch.zeros(3, device=self._sim_device)
        self._right_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._sim_device)  # w, x, y, z
        self._right_gripper = -1.0  # Open
        
    def add_callback(self, key: Any, func: Callable):
        """Register callback for button events."""
        self._callbacks[key] = func
    
    def _get_pose_and_buttons(self, device_idx: int):
        """Get pose and button state for a controller."""
        if device_idx is None:
            return None, None, 0.0
        
        # Get pose
        poses = self._vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 
            0, 
            openvr.k_unMaxTrackedDeviceCount
        )
        
        pose = poses[device_idx]
        if not pose.bPoseIsValid:
            return None, None, 0.0
        
        # Extract position and rotation from 3x4 matrix
        m = pose.mDeviceToAbsoluteTracking
        pos = torch.tensor([m[0][3], m[1][3], m[2][3]], device=self._sim_device)

        # Convert rotation matrix to quaternion
        quat = self._matrix_to_quat(m)
        
        # Get controller state (buttons, triggers)
        success, state = self._vr.getControllerState(device_idx)
        trigger_val = state.rAxis[1].x if success else 0.0  # Trigger is usually axis 1
        
        return pos, quat, trigger_val
    
    def _matrix_to_quat(self, m) -> torch.Tensor:
        """Convert 3x4 OpenVR matrix to quaternion (w, x, y, z)."""
        # Extract 3x3 rotation
        trace = m[0][0] + m[1][1] + m[2][2]
        
        if trace > 0:
            s = 0.5 / (trace + 1.0) ** 0.5
            w = 0.25 / s
            x = (m[2][1] - m[1][2]) * s
            y = (m[0][2] - m[2][0]) * s
            z = (m[1][0] - m[0][1]) * s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]) ** 0.5
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s

        return torch.tensor([w, x, y, z], device=self._sim_device)
    
    def _steamvr_to_isaac(self, pos: torch.Tensor, quat: torch.Tensor):
        """Transform from SteamVR coords (Y-up) to Isaac (Z-up)."""
        # SteamVR: X-right, Y-up, Z-back
        # Isaac/USD: X-right, Y-forward, Z-up
        # Rotation: 90° around X-axis
        new_pos = torch.tensor([pos[0], -pos[2], pos[1]], device=self._sim_device)

        # Quaternion rotation for coordinate transform
        # This rotates the orientation accordingly
        # You may need to adjust based on your specific setup
        new_quat = quat  # Simplified - may need rotation

        return new_pos, new_quat
    
    def _quat_to_rotvec(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (w, x, y, z) to rotation vector (rx, ry, rz)."""
        # Convert to numpy, use scipy, convert back to torch
        quat_np = quat.cpu().numpy()
        # scipy expects (x, y, z, w) format
        quat_scipy = np.array([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])
        rot = Rotation.from_quat(quat_scipy)
        rotvec = rot.as_rotvec()
        return torch.tensor(rotvec, dtype=torch.float32, device=self._sim_device)

    def advance(self) -> torch.Tensor:
        """Get current controller states and return command tensor.

        Returns:
            torch.Tensor:
                - single_arm mode: [pos(3), rot_vec(3), gripper(1)] = 7 elements
                - dual_arm mode: [left_pos(3), left_rot_vec(3), left_gripper(1),
                                  right_pos(3), right_rot_vec(3), right_gripper(1)] = 14 elements
        """

        # Get left controller
        left_pos, left_quat, left_trigger = self._get_pose_and_buttons(self._left_idx)
        if left_pos is not None:
            left_pos, left_quat = self._steamvr_to_isaac(left_pos, left_quat)
            self._left_pos = left_pos * self._pos_sensitivity
            self._left_quat = left_quat
            self._left_gripper = 1.0 if left_trigger > 0.5 else -1.0

        # Get right controller
        right_pos, right_quat, right_trigger = self._get_pose_and_buttons(self._right_idx)
        if right_pos is not None:
            right_pos, right_quat = self._steamvr_to_isaac(right_pos, right_quat)
            self._right_pos = right_pos * self._pos_sensitivity
            self._right_quat = right_quat
            self._right_gripper = 1.0 if right_trigger > 0.5 else -1.0

        # Convert quaternions to rotation vectors
        left_rotvec = self._quat_to_rotvec(self._left_quat) * self._rot_sensitivity
        right_rotvec = self._quat_to_rotvec(self._right_quat) * self._rot_sensitivity

        # Return based on control mode
        if self._control_mode == "single_arm":
            # Use primary hand
            if self._primary_hand == "left":
                return torch.cat([
                    self._left_pos,
                    left_rotvec,
                    torch.tensor([self._left_gripper], device=self._sim_device)
                ])
            else:  # right
                return torch.cat([
                    self._right_pos,
                    right_rotvec,
                    torch.tensor([self._right_gripper], device=self._sim_device)
                ])
        else:  # dual_arm
            # Return combined tensor for dual-arm
            # Format: [left_pos(3), left_rot_vec(3), left_grip(1), right_pos(3), right_rot_vec(3), right_grip(1)]
            return torch.cat([
                self._left_pos,
                left_rotvec,
                torch.tensor([self._left_gripper], device=self._sim_device),
                self._right_pos,
                right_rotvec,
                torch.tensor([self._right_gripper], device=self._sim_device)
            ])
    
    def __del__(self):
        """Clean up OpenVR."""
        openvr.shutdown()


@dataclass
class Se3ViveControllerCfg(DeviceCfg):
    """Configuration for HTC Vive XR Elite controllers using OpenVR.

    Args:
        pos_sensitivity: Position sensitivity multiplier
        rot_sensitivity: Rotation sensitivity multiplier
        control_mode: "single_arm" (7 DOF) or "dual_arm" (14 DOF)
        primary_hand: "left" or "right" - which hand to use in single_arm mode
    """

    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    control_mode: Literal["single_arm", "dual_arm"] = "single_arm"
    primary_hand: Literal["left", "right"] = "right"
    retargeters: None = None
    class_type: type[DeviceBase] = Se3ViveController
