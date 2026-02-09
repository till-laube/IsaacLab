#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GR00T N1.6 Policy Evaluation for Dual UR5e Arms

Self-contained evaluation script that runs a GR00T N1.6 policy on the Jetson
Thor to control two UR5e arms with Robotiq 2F-140 grippers. Captures camera
frames locally, queries the GR00T policy via PolicyClient, and sends motor
commands via RTDE.

Architecture: Two-rate control loop
  - Main thread:  policy loop @ 30 Hz (capture → infer → queue)
  - Servo thread: 500 Hz velocity-clamped servoJ
  - Gripper thread: 50 Hz gripper commands with deadband

Usage:
    List connected cameras to find serial numbers:
    python eval_dual_ur5e_groot.py --list-cameras

    Run with camera serial numbers:
    python eval_dual_ur5e_groot.py \\
        --front-cam-serial 123456789 \\
        --left-wrist-cam-serial 987654321 \\
        --right-wrist-cam-serial 555555555 \\
        --language "pick up the red block"

    With custom IPs:
    python eval_dual_ur5e_groot.py \\
        --front-cam-serial 123456789 \\
        --left-wrist-cam-serial 987654321 \\
        --right-wrist-cam-serial 555555555 \\
        --left-ip 100.80.147.160 --right-ip 100.80.147.57 \\
        --policy-host 100.80.147.1 --policy-port 5555 \\
        --language "pick up the red block"

    Dry run (cameras only, no RTDE):
    python eval_dual_ur5e_groot.py \\
        --front-cam-serial 123456789 \\
        --left-wrist-cam-serial 987654321 \\
        --right-wrist-cam-serial 555555555 \\
        --dry-run --language "test task"
"""

import argparse
import collections
import os
import signal
import socket
import sys
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

# RTDE imports (optional for dry-run mode)
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("[WARNING] ur_rtde not installed. Use --dry-run for camera-only mode.")

# RealSense camera (must match collection script backend)
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARNING] pyrealsense2 not installed. Camera capture will not be available.")

# GR00T policy client
try:
    from gr00t.policy.server_client import PolicyClient
    GROOT_AVAILABLE = True
except ImportError:
    GROOT_AVAILABLE = False
    print("[WARNING] gr00t not installed. Policy inference will not be available.")

# =============================================================================
# Configuration defaults
# =============================================================================

LEFT_ARM_IP = "100.80.147.160"
RIGHT_ARM_IP = "100.80.147.57"
GRIPPER_PORT = 63352

# Servo control
SERVO_FREQ = 500
DT = 1.0 / SERVO_FREQ
LOOKAHEAD_TIME = 0.05
SERVO_GAIN = 1000

# Velocity limits (rad/s)
MAX_JOINT_VELOCITY = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

# Watchdog
WATCHDOG_THRESHOLD = 0.1  # radians
WATCHDOG_WINDOW = 50

# Gripper
GRIPPER_SPEED = 150
DEFAULT_GRIPPER_FORCE = 150
GRIPPER_DEADBAND = 13  # ~5% of 0-255 range
CURRENT_POLL_INTERVAL = 0.05
CURRENT_MONITOR_TIMEOUT = 3.0

# Policy
POLICY_FREQ = 30
POLICY_DT = 1.0 / POLICY_FREQ


# =============================================================================
# GR00T helpers
# =============================================================================

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """Recursively add an extra leading dim to arrays or scalars.

    GR00T Policy Server expects obs shaped (batch=1, time=1, ...).
    Call this function twice to get both dims.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


# =============================================================================
# PositionWatchdog
# =============================================================================

class PositionWatchdog:
    """Monitors position divergence between commanded and actual robot positions."""

    def __init__(self, threshold_rad: float = 0.1, window_size: int = 50):
        self.threshold = threshold_rad
        self.window_size = window_size
        self.error_history: List[float] = []
        self._divergence_count = 0

    def check(self, commanded_pos: np.ndarray, actual_pos: np.ndarray) -> bool:
        """Returns True if safe, False if divergence detected."""
        error = np.abs(np.array(commanded_pos) - np.array(actual_pos))
        max_error = float(np.max(error))

        self.error_history.append(max_error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)

        avg_error = np.mean(self.error_history)
        if avg_error > self.threshold:
            self._divergence_count += 1
            return False

        self._divergence_count = 0
        return True

    def get_avg_error(self) -> float:
        return float(np.mean(self.error_history)) if self.error_history else 0.0

    def reset(self):
        self.error_history.clear()
        self._divergence_count = 0


# =============================================================================
# CameraManager
# =============================================================================

class CameraManager:
    """Manages multiple RealSense camera captures with latching on failure.

    Uses pyrealsense2 to match the data collection pipeline (integrated_capture.py).
    Returns BGR frames — same color format saved during data collection.
    """

    def __init__(
        self,
        camera_config: Dict[str, str],
        width: int = 640,
        height: int = 480,
        fps: int = POLICY_FREQ,
    ):
        """Initialize camera manager.

        Args:
            camera_config: Dict mapping camera key names to RealSense serial
                           numbers.  Example: {"front": "123456789",
                           "wristleft": "987654321", "wristright": "555555555"}
            width: Capture width.
            height: Capture height.
            fps: Capture frame rate.
        """
        self.cam_serials = camera_config
        self.width = width
        self.height = height
        self.fps = fps
        self.pipelines: Dict[str, rs.pipeline] = {}
        self.aligns: Dict[str, rs.align] = {}
        self._last_frames: Dict[str, np.ndarray] = {}
        self._opened_successfully: Dict[str, bool] = {}

    @staticmethod
    def discover_cameras() -> List[str]:
        """Find all connected RealSense cameras, return serial numbers."""
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = []
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"[CAMERA] Found: {name} (SN: {serial})")
            serials.append(serial)
        return serials

    def open(self) -> bool:
        """Open all cameras. Returns True if all opened successfully."""
        if not REALSENSE_AVAILABLE:
            print("[CAMERA] ERROR: pyrealsense2 not installed")
            for name in self.cam_serials:
                self._opened_successfully[name] = False
            return False

        discovered = self.discover_cameras()
        all_ok = True

        for name, serial in self.cam_serials.items():
            if serial not in discovered:
                print(f"[CAMERA] WARNING: {name} (SN: {serial}) not found "
                      f"among connected cameras")
                self._opened_successfully[name] = False
                all_ok = False
                continue

            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color, self.width, self.height,
                    rs.format.bgr8, self.fps,
                )
                profile = pipeline.start(config)
                device = profile.get_device()
                cam_name = device.get_info(rs.camera_info.name)
                align = rs.align(rs.stream.color)

                self.pipelines[name] = pipeline
                self.aligns[name] = align
                self._opened_successfully[name] = True
                print(f"[CAMERA] {name} (SN: {serial}) opened: "
                      f"{cam_name} {self.width}x{self.height}@{self.fps}fps")
            except Exception as e:
                print(f"[CAMERA] WARNING: {name} (SN: {serial}) failed: {e}")
                self._opened_successfully[name] = False
                all_ok = False

        return all_ok

    def all_cameras_ok(self) -> bool:
        """Check if all cameras opened successfully."""
        return all(self._opened_successfully.values())

    def get_failed_cameras(self) -> List[str]:
        """Return list of camera names that failed to open."""
        return [name for name, ok in self._opened_successfully.items() if not ok]

    def capture_all(self) -> Dict[str, np.ndarray]:
        """Capture frames from all cameras, returning BGR images.

        BGR format matches the data collection script so the model sees the
        same color space it was trained on.  Latches the last valid frame if
        capture fails.
        """
        frames = {}
        for name, pipeline in self.pipelines.items():
            try:
                rs_frames = pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.aligns[name].process(rs_frames)
                color_frame = aligned.get_color_frame()
                if color_frame:
                    frame_bgr = np.asanyarray(color_frame.get_data())
                    self._last_frames[name] = frame_bgr
                    frames[name] = frame_bgr
                elif name in self._last_frames:
                    frames[name] = self._last_frames[name]
                else:
                    frames[name] = np.zeros(
                        (self.height, self.width, 3), dtype=np.uint8
                    )
            except RuntimeError:
                if name in self._last_frames:
                    frames[name] = self._last_frames[name]
                else:
                    frames[name] = np.zeros(
                        (self.height, self.width, 3), dtype=np.uint8
                    )
        return frames

    def close(self):
        for name, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
            except Exception:
                pass
        self.pipelines.clear()
        print("[CAMERA] All cameras released")


# =============================================================================
# DualUR5eGR00TAdapter
# =============================================================================

class DualUR5eGR00TAdapter:
    """Adapter between raw robot observations and GR00T VLA input/output format.

    Follows the same pattern as the SO100 adapter:
    - obs["video"]    — dict of camera frames
    - obs["state"]    — dict of named state arrays
    - obs["language"] — dict with annotation key
    - B=1, T=1 dims added via recursive_add_extra_dim called twice
    - Action chunk returned as dict of named arrays (B, T, D)
    """

    # State key names — these must match the GR00T model's embodiment config
    STATE_KEYS = [
        "left_arm",            # (6,) joint positions in rad
        "left_gripper",        # (1,) gripper position 0-255
        "left_gripper_force",  # (1,) force setting 0-255
        "left_current_limit",  # (1,) current threshold in Amps
        "right_arm",           # (6,) joint positions in rad
        "right_gripper",       # (1,) gripper position 0-255
        "right_gripper_force", # (1,) force setting 0-255
        "right_current_limit", # (1,) current threshold in Amps
    ]

    def __init__(
        self,
        default_gripper_force: int = DEFAULT_GRIPPER_FORCE,
        default_current_limit: Optional[float] = None,
    ):
        self.default_gripper_force = default_gripper_force
        self.default_current_limit = default_current_limit

    def build_obs(
        self,
        frames: Dict[str, np.ndarray],
        left_joints: np.ndarray,
        right_joints: np.ndarray,
        left_gripper_pos: int,
        right_gripper_pos: int,
        left_gripper_force: int,
        right_gripper_force: int,
        left_current_limit: Optional[float],
        right_current_limit: Optional[float],
        language: str,
    ) -> dict:
        """Build a GR00T VLA observation dictionary.

        After recursive_add_extra_dim x2:
            video.*:    (1, 1, H, W, 3) uint8
            state.*:    (1, 1, D) float32
            language.*: [[str]]

        Camera key names come from the frames dict (set by CameraManager).
        """
        obs = {}

        # (1) Video — dict of camera frames (H, W, 3) uint8
        # Use whatever keys are in frames (configured via CLI)
        obs["video"] = {}
        for cam_name, frame in frames.items():
            if frame is not None:
                obs["video"][cam_name] = frame.astype(np.uint8)

        # (2) State — dict of named arrays (matching SO100 pattern)
        obs["state"] = {
            "left_arm": np.array(left_joints, dtype=np.float32),                    # (6,)
            "left_gripper": np.array([float(left_gripper_pos)], dtype=np.float32),  # (1,)
            "left_gripper_force": np.array(
                [float(left_gripper_force)], dtype=np.float32),                     # (1,)
            "left_current_limit": np.array(
                [float(left_current_limit if left_current_limit is not None
                       else 0.0)], dtype=np.float32),                               # (1,)
            "right_arm": np.array(right_joints, dtype=np.float32),                  # (6,)
            "right_gripper": np.array([float(right_gripper_pos)], dtype=np.float32),# (1,)
            "right_gripper_force": np.array(
                [float(right_gripper_force)], dtype=np.float32),                    # (1,)
            "right_current_limit": np.array(
                [float(right_current_limit if right_current_limit is not None
                       else 0.0)], dtype=np.float32),                               # (1,)
        }

        # (3) Language — dict with annotation key
        obs["language"] = {
            "task": language,
        }

        # (4) Add (B=1, T=1) dims by calling recursive_add_extra_dim twice
        obs = recursive_add_extra_dim(obs)
        obs = recursive_add_extra_dim(obs)

        return obs

    def decode_action_chunk(self, chunk: Dict[str, np.ndarray], t: int) -> dict:
        """Decode a single timestep from an action chunk dict.

        Args:
            chunk: Dict of action arrays, each shaped (B, T, D).
            t: Timestep index within the chunk.

        Returns:
            Dict with left_joints, right_joints, gripper positions, etc.
        """
        d = {}

        # Arm joints — required
        d["left_joints"] = chunk["left_arm"][0][t].copy()      # (6,)
        d["right_joints"] = chunk["right_arm"][0][t].copy()    # (6,)

        # Gripper position — required
        d["left_gripper"] = int(np.clip(chunk["left_gripper"][0][t][0], 0, 255))
        d["right_gripper"] = int(np.clip(chunk["right_gripper"][0][t][0], 0, 255))

        # Gripper force — optional, fall back to defaults
        if "left_gripper_force" in chunk:
            d["left_gripper_force"] = int(np.clip(
                chunk["left_gripper_force"][0][t][0], 0, 255))
        else:
            d["left_gripper_force"] = self.default_gripper_force
        if "right_gripper_force" in chunk:
            d["right_gripper_force"] = int(np.clip(
                chunk["right_gripper_force"][0][t][0], 0, 255))
        else:
            d["right_gripper_force"] = self.default_gripper_force

        # Current limit — optional, fall back to defaults
        if "left_current_limit" in chunk:
            val = float(chunk["left_current_limit"][0][t][0])
            d["left_current_limit"] = val if val > 0 else self.default_current_limit
        else:
            d["left_current_limit"] = self.default_current_limit
        if "right_current_limit" in chunk:
            val = float(chunk["right_current_limit"][0][t][0])
            d["right_current_limit"] = val if val > 0 else self.default_current_limit
        else:
            d["right_current_limit"] = self.default_current_limit

        return d

    def get_action(
        self, policy_client: "PolicyClient", obs: dict
    ) -> List[dict]:
        """Query policy and return list of per-timestep motor command dicts.

        Args:
            policy_client: GR00T PolicyClient instance.
            obs: Observation dict built by build_obs().

        Returns:
            List of dicts, one per action timestep.
        """
        action_chunk, info = policy_client.get_action(obs)

        # Determine horizon from any key: shape is (B, T, D)
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# DualUR5eHardwareInterface
# =============================================================================

class DualUR5eHardwareInterface:
    """Wraps all RTDE + gripper connections for dual UR5e arms."""

    def __init__(
        self,
        left_ip: str = LEFT_ARM_IP,
        right_ip: str = RIGHT_ARM_IP,
        gripper_port: int = GRIPPER_PORT,
    ):
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.gripper_port = gripper_port

        self.left_rtde_c: Optional[RTDEControlInterface] = None
        self.left_rtde_r: Optional[RTDEReceiveInterface] = None
        self.right_rtde_c: Optional[RTDEControlInterface] = None
        self.right_rtde_r: Optional[RTDEReceiveInterface] = None
        self.left_gripper_sock: Optional[socket.socket] = None
        self.right_gripper_sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Connect RTDE and gripper sockets. Returns True if all succeeded."""
        ok = True

        # RTDE connections
        try:
            print(f"[HW] Connecting RTDE to LEFT arm at {self.left_ip}...")
            self.left_rtde_c = RTDEControlInterface(self.left_ip)
            self.left_rtde_r = RTDEReceiveInterface(self.left_ip)
            print("[HW] LEFT arm RTDE connected")
        except Exception as e:
            print(f"[HW] LEFT arm RTDE failed: {e}")
            ok = False

        try:
            print(f"[HW] Connecting RTDE to RIGHT arm at {self.right_ip}...")
            self.right_rtde_c = RTDEControlInterface(self.right_ip)
            self.right_rtde_r = RTDEReceiveInterface(self.right_ip)
            print("[HW] RIGHT arm RTDE connected")
        except Exception as e:
            print(f"[HW] RIGHT arm RTDE failed: {e}")
            ok = False

        # Gripper sockets
        for side, ip in [("LEFT", self.left_ip), ("RIGHT", self.right_ip)]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((ip, self.gripper_port))
                if side == "LEFT":
                    self.left_gripper_sock = sock
                else:
                    self.right_gripper_sock = sock
                print(f"[HW] {side} gripper connected at {ip}:{self.gripper_port}")
            except Exception as e:
                print(f"[HW] {side} gripper connection failed: {e}")

        # Initialize gripper speed/force
        for side, sock in [("LEFT", self.left_gripper_sock),
                           ("RIGHT", self.right_gripper_sock)]:
            if sock:
                self._send_gripper_cmd(sock, f"SET SPE {GRIPPER_SPEED}")
                self._send_gripper_cmd(sock, f"SET FOR {DEFAULT_GRIPPER_FORCE}")
                self._send_gripper_cmd(sock, "SET GTO 1")
                print(f"[HW] {side} gripper initialized: speed={GRIPPER_SPEED}")

        return ok

    def _send_gripper_cmd(self, sock: socket.socket, cmd: str) -> Optional[str]:
        try:
            sock.sendall((cmd.strip() + "\n").encode())
            response = sock.recv(1024).decode().strip()
            return response
        except Exception as e:
            print(f"[HW] Gripper cmd error ({cmd}): {e}")
            return None

    def _read_gripper_position(self, sock: socket.socket) -> Optional[int]:
        response = self._send_gripper_cmd(sock, "GET POS")
        if response is None:
            return None
        try:
            if response.isdigit():
                return int(response)
            parts = response.split()
            if len(parts) >= 2 and parts[0] == "POS":
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return None

    def _read_gripper_current(self, sock: socket.socket) -> Optional[float]:
        response = self._send_gripper_cmd(sock, "GET COU")
        if response is None:
            return None
        try:
            if response.isdigit():
                raw = int(response)
            else:
                parts = response.split()
                if len(parts) >= 2 and parts[0] == "COU":
                    raw = int(parts[1])
                else:
                    return None
            return raw / 255.0 * 1.5
        except (ValueError, IndexError):
            return None

    def read_joint_positions(self) -> tuple:
        """Returns (left_joints, right_joints) as numpy arrays."""
        left = np.array(self.left_rtde_r.getActualQ()) if self.left_rtde_r else np.zeros(6)
        right = np.array(self.right_rtde_r.getActualQ()) if self.right_rtde_r else np.zeros(6)
        return left, right

    def read_gripper_positions(self) -> tuple:
        """Returns (left_pos, right_pos) as ints (0-255)."""
        left = 0
        right = 0
        if self.left_gripper_sock:
            val = self._read_gripper_position(self.left_gripper_sock)
            if val is not None:
                left = val
        if self.right_gripper_sock:
            val = self._read_gripper_position(self.right_gripper_sock)
            if val is not None:
                right = val
        return left, right

    def send_servo(self, left_pos: np.ndarray, right_pos: np.ndarray):
        """Send servoJ commands to both arms."""
        if self.left_rtde_c:
            self.left_rtde_c.servoJ(
                left_pos.tolist(), 0, 0, DT, LOOKAHEAD_TIME, SERVO_GAIN
            )
        if self.right_rtde_c:
            self.right_rtde_c.servoJ(
                right_pos.tolist(), 0, 0, DT, LOOKAHEAD_TIME, SERVO_GAIN
            )

    def send_gripper(self, side: str, position: int, force: int):
        """Send gripper position and force command."""
        sock = self.left_gripper_sock if side == "left" else self.right_gripper_sock
        if sock is None:
            return
        pos_val = max(0, min(255, position))
        force_val = max(0, min(255, force))
        self._send_gripper_cmd(sock, f"SET FOR {force_val}")
        self._send_gripper_cmd(sock, f"SET POS {pos_val}")
        self._send_gripper_cmd(sock, "SET GTO 1")

    def close_with_current_limit(
        self,
        side: str,
        target_pos: int,
        threshold: float,
    ) -> bool:
        """Close gripper while monitoring current; stop if threshold exceeded.

        Returns True if stopped due to current threshold.
        """
        sock = self.left_gripper_sock if side == "left" else self.right_gripper_sock
        if sock is None:
            return False

        self._send_gripper_cmd(sock, f"SET POS {target_pos}")
        self._send_gripper_cmd(sock, "SET GTO 1")

        start_time = time.time()
        while time.time() - start_time < CURRENT_MONITOR_TIMEOUT:
            current = self._read_gripper_current(sock)
            position = self._read_gripper_position(sock)

            if current is not None and current > threshold:
                self._send_gripper_cmd(sock, "SET GTO 0")
                print(f"[HW] {side.upper()} gripper stopped at {current:.4f}A "
                      f"(threshold: {threshold}A, pos: {position})")
                return True

            if position is not None and position >= target_pos - 5:
                break

            time.sleep(CURRENT_POLL_INTERVAL)

        return False

    @staticmethod
    def clamp_velocity(current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Apply velocity limiting for one servo tick."""
        delta = target_pos - current_pos
        max_delta = MAX_JOINT_VELOCITY * DT
        clamped_delta = np.clip(delta, -max_delta, max_delta)
        return current_pos + clamped_delta

    def stop(self):
        """Stop all motion and close connections."""
        for label, rtde_c in [("LEFT", self.left_rtde_c), ("RIGHT", self.right_rtde_c)]:
            if rtde_c:
                try:
                    rtde_c.servoStop()
                except Exception:
                    pass
                try:
                    rtde_c.stopScript()
                except Exception:
                    pass

        for sock in (self.left_gripper_sock, self.right_gripper_sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

        print("[HW] Stopped and cleaned up")


# =============================================================================
# GR00TEvalRunner (orchestrator)
# =============================================================================

class GR00TEvalRunner:
    """Orchestrates GR00T policy evaluation on dual UR5e hardware."""

    def __init__(
        self,
        left_ip: str,
        right_ip: str,
        policy_host: str,
        policy_port: int,
        camera_config: Dict[str, str],
        cam_width: int,
        cam_height: int,
        language: str,
        action_horizon: int,
        use_watchdog: bool = True,
        dry_run: bool = False,
        record_inference: bool = False,
    ):
        """Initialize the evaluation runner.

        Args:
            left_ip: IP address of left UR5e arm.
            right_ip: IP address of right UR5e arm.
            policy_host: GR00T policy server host.
            policy_port: GR00T policy server port.
            camera_config: Dict mapping camera key names to serial numbers.
                           Keys must match training data (e.g., "front", "wristleft").
            cam_width: Camera capture width.
            cam_height: Camera capture height.
            language: Task instruction string.
            action_horizon: Number of actions to execute per policy query.
            use_watchdog: Enable position divergence watchdog.
            dry_run: If True, skip hardware connection (camera-only mode).
            record_inference: If True, record first 10s of camera streams
                              during inference to inference_recording/.
        """
        self.language = language
        self.action_horizon = action_horizon
        self.use_watchdog = use_watchdog
        self.dry_run = dry_run
        self.record_inference = record_inference
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.camera_config = camera_config

        self.policy_host = policy_host
        self.policy_port = policy_port

        # Hardware
        self.hw = DualUR5eHardwareInterface(left_ip, right_ip) if not dry_run else None

        # Cameras — key names must match training data
        self.cameras = CameraManager(
            camera_config=camera_config,
            width=cam_width,
            height=cam_height,
        )

        # GR00T adapter
        self.adapter = DualUR5eGR00TAdapter()

        # Policy client (initialized in connect())
        self.policy_client = None

        # Watchdogs
        self.left_watchdog = PositionWatchdog(WATCHDOG_THRESHOLD, WATCHDOG_WINDOW) if use_watchdog else None
        self.right_watchdog = PositionWatchdog(WATCHDOG_THRESHOLD, WATCHDOG_WINDOW) if use_watchdog else None

        # Thread-safe action queue and gripper targets
        self.lock = threading.Lock()
        self.action_queue = collections.deque()
        self.gripper_target = {
            "left_position": 0,
            "right_position": 0,
            "left_force": DEFAULT_GRIPPER_FORCE,
            "right_force": DEFAULT_GRIPPER_FORCE,
            "left_current_limit": None,
            "right_current_limit": None,
        }

        # Servo state — current interpolated positions
        self.left_servo_pos: Optional[np.ndarray] = None
        self.right_servo_pos: Optional[np.ndarray] = None

        # Current action target for servo interpolation
        self.left_action_target: Optional[np.ndarray] = None
        self.right_action_target: Optional[np.ndarray] = None

        # State read by policy loop
        self.latest_joints_left: Optional[np.ndarray] = None
        self.latest_joints_right: Optional[np.ndarray] = None
        self.latest_gripper_left: int = 0
        self.latest_gripper_right: int = 0

        self._record_writers: Dict[str, cv2.VideoWriter] = {}
        self.running = False

    def connect(self, require_cameras: bool = True):
        """Initialize hardware, cameras, and policy client.

        Args:
            require_cameras: If True, exit if any camera fails to open.
                             This prevents robot commands without valid vision.
        """
        # Cameras — must succeed before connecting to hardware
        print("[EVAL] Opening cameras...")
        self.cameras.open()

        if require_cameras and not self.cameras.all_cameras_ok():
            failed = self.cameras.get_failed_cameras()
            print(f"[ERROR] Required cameras failed to open: {failed}")
            print("[ERROR] Robot connection blocked — fix cameras first.")
            print("[ERROR] Use --no-require-cameras to override (unsafe).")
            self.cameras.close()
            sys.exit(1)

        # Hardware — only connect if cameras are OK (or not required)
        if not self.dry_run:
            if not RTDE_AVAILABLE:
                print("[ERROR] ur_rtde required for real robot mode. Use --dry-run.")
                sys.exit(1)
            print("[EVAL] Connecting hardware...")
            if not self.hw.connect():
                print("[ERROR] Hardware connection failed")
                sys.exit(1)

            # Read initial positions — hold here until first action
            left_j, right_j = self.hw.read_joint_positions()
            self.left_servo_pos = left_j.copy()
            self.right_servo_pos = right_j.copy()
            self.left_action_target = left_j.copy()
            self.right_action_target = right_j.copy()
            self.latest_joints_left = left_j.copy()
            self.latest_joints_right = right_j.copy()
            print(f"[EVAL] LEFT  start: {[f'{j:.3f}' for j in left_j]}")
            print(f"[EVAL] RIGHT start: {[f'{j:.3f}' for j in right_j]}")

            # Read initial gripper positions and seed targets so the gripper
            # thread holds current position until the first policy action arrives
            gl, gr = self.hw.read_gripper_positions()
            self.latest_gripper_left = gl
            self.latest_gripper_right = gr
            with self.lock:
                self.gripper_target["left_position"] = gl
                self.gripper_target["right_position"] = gr
        else:
            # Dry-run defaults
            self.left_servo_pos = np.zeros(6)
            self.right_servo_pos = np.zeros(6)
            self.left_action_target = np.zeros(6)
            self.right_action_target = np.zeros(6)
            self.latest_joints_left = np.zeros(6)
            self.latest_joints_right = np.zeros(6)

        # Policy client
        if GROOT_AVAILABLE:
            print(f"[EVAL] Connecting to GR00T policy at "
                  f"{self.policy_host}:{self.policy_port}...")
            self.policy_client = PolicyClient(
                host=self.policy_host,
                port=self.policy_port,
            )
            print("[EVAL] PolicyClient connected")
        else:
            print("[EVAL] GR00T not available — running without policy (hold position)")

    def run(self):
        """Start all threads and run the evaluation loop."""
        self.running = True

        threads = []

        # Servo thread
        if not self.dry_run:
            t_servo = threading.Thread(target=self._servo_loop, daemon=True, name="servo")
            t_servo.start()
            threads.append(t_servo)

            t_gripper = threading.Thread(target=self._gripper_loop, daemon=True, name="gripper")
            t_gripper.start()
            threads.append(t_gripper)

        try:
            self._policy_loop()
        finally:
            self.stop()
            for t in threads:
                t.join(timeout=1.0)

    def _policy_loop(self):
        """Main policy loop at ~30 Hz."""
        print(f"[POLICY] Starting policy loop at {POLICY_FREQ} Hz")
        print(f"[POLICY] Language instruction: '{self.language}'")
        print(f"[POLICY] Action horizon: {self.action_horizon}")
        print("[POLICY] Press Ctrl+C to stop")

        # Inference recording setup
        self._record_writers: Dict[str, cv2.VideoWriter] = {}
        record_max_frames = int(10.0 * POLICY_FREQ)  # 10 seconds
        record_frame_count = 0
        if self.record_inference:
            out_dir = "inference_recording"
            os.makedirs(out_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for name, serial in self.camera_config.items():
                path = os.path.join(out_dir, f"{name}_{serial}.mp4")
                w = cv2.VideoWriter(
                    path, fourcc, POLICY_FREQ,
                    (self.cam_width, self.cam_height),
                )
                if w.isOpened():
                    self._record_writers[name] = w
                    print(f"[RECORD] Will record {name} -> {path}")
                else:
                    print(f"[RECORD] WARNING: Failed to open writer for {path}")
            print(f"[RECORD] Recording first {record_max_frames} frames "
                  f"({record_max_frames / POLICY_FREQ:.0f}s) during inference")

        loop_count = 0
        last_status_time = time.time()

        while self.running:
            t_start = time.time()

            # 1. Read joint states
            if not self.dry_run:
                left_j, right_j = self.hw.read_joint_positions()
                gl, gr = self.hw.read_gripper_positions()
                self.latest_joints_left = left_j
                self.latest_joints_right = right_j
                self.latest_gripper_left = gl
                self.latest_gripper_right = gr

            # 2. Capture camera frames
            frames = self.cameras.capture_all()

            # Record frames if active
            if self._record_writers and record_frame_count < record_max_frames:
                for name, w in self._record_writers.items():
                    if name in frames:
                        w.write(frames[name])
                record_frame_count += 1
                if record_frame_count >= record_max_frames:
                    for name, w in self._record_writers.items():
                        w.release()
                        print(f"[RECORD] Finished {name} "
                              f"({record_frame_count} frames)")
                    self._record_writers.clear()

            # 3. Build observation
            with self.lock:
                gripper_state = dict(self.gripper_target)

            obs = self.adapter.build_obs(
                frames=frames,
                left_joints=self.latest_joints_left,
                right_joints=self.latest_joints_right,
                left_gripper_pos=self.latest_gripper_left,
                right_gripper_pos=self.latest_gripper_right,
                left_gripper_force=gripper_state["left_force"],
                right_gripper_force=gripper_state["right_force"],
                left_current_limit=gripper_state["left_current_limit"],
                right_current_limit=gripper_state["right_current_limit"],
                language=self.language,
            )

            if self.dry_run:
                # Just log that we built an obs
                if loop_count % POLICY_FREQ == 0:
                    state_keys = list(obs["state"].keys())
                    video_keys = list(obs["video"].keys())
                    print(f"[DRY-RUN] Built obs — state keys: {state_keys}, "
                          f"video keys: {video_keys}")
            elif self.policy_client is not None:
                # 4. Query policy → decode action chunk
                try:
                    actions = self.adapter.get_action(self.policy_client, obs)
                except Exception as e:
                    print(f"[POLICY] Inference error: {e}")
                    loop_count += 1
                    self._sleep_to_freq(t_start, POLICY_DT)
                    continue

                # 5. Push decoded actions into queue
                with self.lock:
                    self.action_queue.clear()
                    for act in actions[:self.action_horizon]:
                        self.action_queue.append(act)

                    # Update gripper targets from first action
                    if actions:
                        a0 = actions[0]
                        self.gripper_target["left_position"] = a0["left_gripper"]
                        self.gripper_target["right_position"] = a0["right_gripper"]
                        self.gripper_target["left_force"] = a0["left_gripper_force"]
                        self.gripper_target["right_force"] = a0["right_gripper_force"]
                        self.gripper_target["left_current_limit"] = a0["left_current_limit"]
                        self.gripper_target["right_current_limit"] = a0["right_current_limit"]
            # else: no policy, hold position

            # Status
            loop_count += 1
            if time.time() - last_status_time > 5.0:
                with self.lock:
                    q_len = len(self.action_queue)
                mode = "DRY-RUN" if self.dry_run else "ACTIVE"
                print(f"[STATUS] {mode} | loop={loop_count} | queue={q_len}")
                last_status_time = time.time()

            self._sleep_to_freq(t_start, POLICY_DT)

    def _servo_loop(self):
        """500 Hz servo loop — pops actions from queue, velocity-clamps, sends servoJ."""
        print(f"[SERVO] Starting servo loop at {SERVO_FREQ} Hz")

        tick_count = 0
        ticks_per_action = int(SERVO_FREQ / POLICY_FREQ)  # ~16-17 ticks per action

        while self.running:
            t_start = time.time()

            # Pop next action target from queue at policy rate
            if tick_count % ticks_per_action == 0:
                with self.lock:
                    if self.action_queue:
                        act = self.action_queue.popleft()
                        self.left_action_target = np.array(act["left_joints"])
                        self.right_action_target = np.array(act["right_joints"])
                        # Update gripper targets
                        self.gripper_target["left_position"] = act["left_gripper"]
                        self.gripper_target["right_position"] = act["right_gripper"]
                        self.gripper_target["left_force"] = act["left_gripper_force"]
                        self.gripper_target["right_force"] = act["right_gripper_force"]
                        self.gripper_target["left_current_limit"] = act["left_current_limit"]
                        self.gripper_target["right_current_limit"] = act["right_current_limit"]

            # Velocity-clamp toward target
            self.left_servo_pos = DualUR5eHardwareInterface.clamp_velocity(
                self.left_servo_pos, self.left_action_target
            )
            self.right_servo_pos = DualUR5eHardwareInterface.clamp_velocity(
                self.right_servo_pos, self.right_action_target
            )

            # Send servo commands
            self.hw.send_servo(self.left_servo_pos, self.right_servo_pos)

            # Watchdog checks (every 500 ticks ≈ 1 sec)
            if self.use_watchdog and tick_count % SERVO_FREQ == 0:
                try:
                    left_actual, right_actual = self.hw.read_joint_positions()
                    if self.left_watchdog:
                        if not self.left_watchdog.check(self.left_servo_pos, left_actual):
                            print(f"[WATCHDOG] LEFT divergence: "
                                  f"{self.left_watchdog.get_avg_error():.3f} rad")
                    if self.right_watchdog:
                        if not self.right_watchdog.check(self.right_servo_pos, right_actual):
                            print(f"[WATCHDOG] RIGHT divergence: "
                                  f"{self.right_watchdog.get_avg_error():.3f} rad")
                except Exception:
                    pass

            tick_count += 1
            self._sleep_to_freq(t_start, DT)

    def _gripper_loop(self):
        """50 Hz gripper control loop with deadband."""
        print("[GRIPPER] Starting gripper loop at 50 Hz")

        last_left_pos = -999
        last_right_pos = -999
        last_left_force = DEFAULT_GRIPPER_FORCE
        last_right_force = DEFAULT_GRIPPER_FORCE

        while self.running:
            with self.lock:
                left_pos = self.gripper_target["left_position"]
                right_pos = self.gripper_target["right_position"]
                left_force = self.gripper_target["left_force"]
                right_force = self.gripper_target["right_force"]
                left_clim = self.gripper_target["left_current_limit"]
                right_clim = self.gripper_target["right_current_limit"]

            # Left gripper
            if left_force != last_left_force and self.hw.left_gripper_sock:
                self.hw._send_gripper_cmd(self.hw.left_gripper_sock, f"SET FOR {left_force}")
                last_left_force = left_force

            if abs(left_pos - last_left_pos) > GRIPPER_DEADBAND:
                is_closing = left_pos > last_left_pos
                if is_closing and left_clim is not None:
                    self.hw.close_with_current_limit("left", left_pos, left_clim)
                else:
                    self.hw.send_gripper("left", left_pos, left_force)
                last_left_pos = left_pos

            # Right gripper
            if right_force != last_right_force and self.hw.right_gripper_sock:
                self.hw._send_gripper_cmd(self.hw.right_gripper_sock, f"SET FOR {right_force}")
                last_right_force = right_force

            if abs(right_pos - last_right_pos) > GRIPPER_DEADBAND:
                is_closing = right_pos > last_right_pos
                if is_closing and right_clim is not None:
                    self.hw.close_with_current_limit("right", right_pos, right_clim)
                else:
                    self.hw.send_gripper("right", right_pos, right_force)
                last_right_pos = right_pos

            time.sleep(0.02)  # 50 Hz

    @staticmethod
    def _sleep_to_freq(t_start: float, dt: float):
        elapsed = time.time() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    def stop(self):
        """Graceful shutdown."""
        self.running = False
        # Release any in-progress inference recording writers
        for name, w in self._record_writers.items():
            w.release()
            print(f"[RECORD] Released {name} (early shutdown)")
        self._record_writers.clear()
        if self.hw:
            self.hw.stop()
        self.cameras.close()
        print("[EVAL] Shutdown complete")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.6 Policy Evaluation for Dual UR5e Arms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Robot IPs
    parser.add_argument("--left-ip", type=str, default=LEFT_ARM_IP,
                        help="IP address of left UR5e arm")
    parser.add_argument("--right-ip", type=str, default=RIGHT_ARM_IP,
                        help="IP address of right UR5e arm")

    # Policy server
    parser.add_argument("--policy-host", type=str, default="localhost",
                        help="GR00T policy server host")
    parser.add_argument("--policy-port", type=int, default=5555,
                        help="GR00T policy server port")

    # Cameras — identified by serial number for deterministic assignment
    parser.add_argument("--front-cam-serial", type=str, default=None,
                        help="RealSense serial number for the front camera")
    parser.add_argument("--left-wrist-cam-serial", type=str, default=None,
                        help="RealSense serial number for the left wrist camera")
    parser.add_argument("--right-wrist-cam-serial", type=str, default=None,
                        help="RealSense serial number for the right wrist camera")
    parser.add_argument("--front-cam-key", type=str, default="front",
                        help="Key name for front camera (must match training data)")
    parser.add_argument("--left-wrist-cam-key", type=str, default="wristleft",
                        help="Key name for left wrist camera (must match training data)")
    parser.add_argument("--right-wrist-cam-key", type=str, default="wristright",
                        help="Key name for right wrist camera (must match training data)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List connected RealSense cameras and exit")
    parser.add_argument("--record-cams", action="store_true",
                        help="Record 10 seconds from each camera and exit "
                             "(useful to verify camera-to-key assignment)")
    parser.add_argument("--cam-width", type=int, default=640,
                        help="Camera capture width")
    parser.add_argument("--cam-height", type=int, default=480,
                        help="Camera capture height")

    # Task
    parser.add_argument("--language", type=str, default=None,
                        help="Natural language task instruction (required for eval)")
    parser.add_argument("--action-horizon", type=int, default=16,
                        help="Number of action timesteps per policy query")

    # Safety / debug
    parser.add_argument("--no-watchdog", action="store_true",
                        help="Disable position watchdog")
    parser.add_argument("--dry-run", action="store_true",
                        help="Camera-only mode (no RTDE connection)")
    parser.add_argument("--no-require-cameras", action="store_true",
                        help="Allow running even if cameras fail to open (unsafe)")
    parser.add_argument("--record-inference", action="store_true",
                        help="Record first 10s of each camera during inference "
                             "to inference_recording/")

    args = parser.parse_args()

    # Signal handling
    runner = None

    def shutdown_handler(signum, frame):
        print("\n[SIGNAL] Shutting down...")
        if runner:
            runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # --list-cameras: discover and exit
    if args.list_cameras:
        if not REALSENSE_AVAILABLE:
            print("[ERROR] pyrealsense2 is not installed")
            sys.exit(1)
        print("Connected RealSense cameras:")
        serials = CameraManager.discover_cameras()
        if not serials:
            print("  (none found)")
        print("\nUse these serial numbers with --front-cam-serial, "
              "--left-wrist-cam-serial, --right-wrist-cam-serial")
        sys.exit(0)

    # Validate that all camera serials are provided
    missing = []
    if args.front_cam_serial is None:
        missing.append("--front-cam-serial")
    if args.left_wrist_cam_serial is None:
        missing.append("--left-wrist-cam-serial")
    if args.right_wrist_cam_serial is None:
        missing.append("--right-wrist-cam-serial")
    if missing:
        print(f"[ERROR] Missing required camera serial numbers: {', '.join(missing)}")
        print("  Run with --list-cameras to see connected cameras and their serial numbers.")
        sys.exit(1)

    # Build camera config dict — key names must match training data
    camera_config = {
        args.front_cam_key: args.front_cam_serial,
        args.left_wrist_cam_key: args.left_wrist_cam_serial,
        args.right_wrist_cam_key: args.right_wrist_cam_serial,
    }

    # --record-cams: open cameras, record 10s each, save and exit
    if args.record_cams:
        record_duration = 10.0
        record_fps = POLICY_FREQ
        cam = CameraManager(
            camera_config=camera_config,
            width=args.cam_width,
            height=args.cam_height,
            fps=record_fps,
        )
        if not cam.open():
            print("[ERROR] Not all cameras opened — check serial numbers")
            cam.close()
            sys.exit(1)

        out_dir = "camera_check"
        os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {}
        for name, serial in camera_config.items():
            path = os.path.join(out_dir, f"{name}_{serial}.mp4")
            w = cv2.VideoWriter(path, fourcc, record_fps,
                                (args.cam_width, args.cam_height))
            if not w.isOpened():
                print(f"[ERROR] Failed to open VideoWriter for {path}")
                cam.close()
                sys.exit(1)
            writers[name] = (w, path)

        print(f"[RECORD] Recording {record_duration}s from {len(writers)} cameras "
              f"at {record_fps} fps...")
        frame_count = 0
        total_frames = int(record_duration * record_fps)
        dt = 1.0 / record_fps
        try:
            while frame_count < total_frames:
                t0 = time.time()
                frames = cam.capture_all()
                for name, frame in frames.items():
                    writers[name][0].write(frame)
                frame_count += 1
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print("\n[RECORD] Interrupted early")
        finally:
            for name, (w, path) in writers.items():
                w.release()
                print(f"[RECORD] Saved {name} -> {path}  ({frame_count} frames)")
            cam.close()
        sys.exit(0)

    # Validate --language is provided for eval mode
    if args.language is None:
        print("[ERROR] --language is required for evaluation mode")
        sys.exit(1)

    # Banner
    print("=" * 70)
    print("GR00T N1.6 Policy Evaluation — Dual UR5e Arms")
    print("=" * 70)
    print(f"LEFT arm IP:     {args.left_ip}")
    print(f"RIGHT arm IP:    {args.right_ip}")
    print(f"Policy server:   {args.policy_host}:{args.policy_port}")
    print(f"Cameras:         {args.front_cam_key}={args.front_cam_serial}, "
          f"{args.left_wrist_cam_key}={args.left_wrist_cam_serial}, "
          f"{args.right_wrist_cam_key}={args.right_wrist_cam_serial}")
    print(f"Resolution:      {args.cam_width}x{args.cam_height}")
    print(f"Language:        {args.language}")
    print(f"Action horizon:  {args.action_horizon}")
    print(f"Watchdog:        {'Disabled' if args.no_watchdog else 'Enabled'}")
    print(f"Require cams:    {not args.no_require_cameras}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 70)

    runner = GR00TEvalRunner(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        camera_config=camera_config,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        language=args.language,
        action_horizon=args.action_horizon,
        use_watchdog=not args.no_watchdog,
        dry_run=args.dry_run,
        record_inference=args.record_inference,
    )

    try:
        runner.connect(require_cameras=not args.no_require_cameras)
        runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if runner:
            runner.stop()


if __name__ == "__main__":
    main()
