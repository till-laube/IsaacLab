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
    python eval_dual_ur5e_groot.py --language "pick up the red block"

    With custom IPs:
    python eval_dual_ur5e_groot.py \\
        --left-ip 100.80.147.160 --right-ip 100.80.147.51 \\
        --policy-host 100.80.147.1 --policy-port 5555 \\
        --language "pick up the red block"

    Dry run (cameras only, no RTDE):
    python eval_dual_ur5e_groot.py --dry-run --language "test task"
"""

import argparse
import collections
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
RIGHT_ARM_IP = "100.80.147.51"
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
    """Manages multiple camera captures with latching on failure."""

    def __init__(
        self,
        front_id: int = 0,
        left_wrist_id: int = 2,
        right_wrist_id: int = 4,
        width: int = 640,
        height: int = 480,
    ):
        self.cam_ids = {
            "front": front_id,
            "left_wrist": left_wrist_id,
            "right_wrist": right_wrist_id,
        }
        self.width = width
        self.height = height
        self.caps: Dict[str, cv2.VideoCapture] = {}
        self._last_frames: Dict[str, np.ndarray] = {}

    def open(self) -> bool:
        """Open all cameras. Returns True if all opened successfully."""
        all_ok = True
        for name, cam_id in self.cam_ids.items():
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.caps[name] = cap
                print(f"[CAMERA] {name} (id={cam_id}) opened: "
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            else:
                print(f"[CAMERA] WARNING: {name} (id={cam_id}) failed to open")
                self.caps[name] = cap  # keep handle for retry
                all_ok = False
        return all_ok

    def capture_all(self) -> Dict[str, np.ndarray]:
        """Capture frames from all cameras, returning RGB images.

        Latches the last valid frame if capture fails.
        """
        frames = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                # BGR → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._last_frames[name] = frame_rgb
                frames[name] = frame_rgb
            elif name in self._last_frames:
                frames[name] = self._last_frames[name]
            else:
                # No frame ever captured — return black
                frames[name] = np.zeros(
                    (self.height, self.width, 3), dtype=np.uint8
                )
        return frames

    def close(self):
        for name, cap in self.caps.items():
            try:
                cap.release()
            except Exception:
                pass
        self.caps.clear()
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

    CAMERA_KEYS = ["front", "left_wrist", "right_wrist"]

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
        """
        obs = {}

        # (1) Video — dict of camera frames (H, W, 3) uint8
        obs["video"] = {}
        for cam_name in self.CAMERA_KEYS:
            frame = frames.get(cam_name)
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
            "annotation.human.action.task_description": language,
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
        front_cam: int,
        left_wrist_cam: int,
        right_wrist_cam: int,
        cam_width: int,
        cam_height: int,
        language: str,
        action_horizon: int,
        use_watchdog: bool = True,
        dry_run: bool = False,
    ):
        self.language = language
        self.action_horizon = action_horizon
        self.use_watchdog = use_watchdog
        self.dry_run = dry_run

        self.policy_host = policy_host
        self.policy_port = policy_port

        # Hardware
        self.hw = DualUR5eHardwareInterface(left_ip, right_ip) if not dry_run else None

        # Cameras
        self.cameras = CameraManager(
            front_id=front_cam,
            left_wrist_id=left_wrist_cam,
            right_wrist_id=right_wrist_cam,
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

        self.running = False

    def connect(self):
        """Initialize hardware, cameras, and policy client."""
        # Cameras
        print("[EVAL] Opening cameras...")
        self.cameras.open()

        # Hardware
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

            # Read initial gripper positions
            gl, gr = self.hw.read_gripper_positions()
            self.latest_gripper_left = gl
            self.latest_gripper_right = gr
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

    # Cameras
    parser.add_argument("--front-cam", type=int, default=0,
                        help="Front camera device ID")
    parser.add_argument("--left-wrist-cam", type=int, default=2,
                        help="Left wrist camera device ID")
    parser.add_argument("--right-wrist-cam", type=int, default=4,
                        help="Right wrist camera device ID")
    parser.add_argument("--cam-width", type=int, default=640,
                        help="Camera capture width")
    parser.add_argument("--cam-height", type=int, default=480,
                        help="Camera capture height")

    # Task
    parser.add_argument("--language", type=str, required=True,
                        help="Natural language task instruction")
    parser.add_argument("--action-horizon", type=int, default=16,
                        help="Number of action timesteps per policy query")

    # Safety / debug
    parser.add_argument("--no-watchdog", action="store_true",
                        help="Disable position watchdog")
    parser.add_argument("--dry-run", action="store_true",
                        help="Camera-only mode (no RTDE connection)")

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

    # Banner
    print("=" * 70)
    print("GR00T N1.6 Policy Evaluation — Dual UR5e Arms")
    print("=" * 70)
    print(f"LEFT arm IP:     {args.left_ip}")
    print(f"RIGHT arm IP:    {args.right_ip}")
    print(f"Policy server:   {args.policy_host}:{args.policy_port}")
    print(f"Cameras:         front={args.front_cam}, "
          f"left_wrist={args.left_wrist_cam}, right_wrist={args.right_wrist_cam}")
    print(f"Resolution:      {args.cam_width}x{args.cam_height}")
    print(f"Language:        {args.language}")
    print(f"Action horizon:  {args.action_horizon}")
    print(f"Watchdog:        {'Disabled' if args.no_watchdog else 'Enabled'}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 70)

    runner = GR00TEvalRunner(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        front_cam=args.front_cam,
        left_wrist_cam=args.left_wrist_cam,
        right_wrist_cam=args.right_wrist_cam,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        language=args.language,
        action_horizon=args.action_horizon,
        use_watchdog=not args.no_watchdog,
        dry_run=args.dry_run,
    )

    try:
        runner.connect()
        runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if runner:
            runner.stop()


if __name__ == "__main__":
    main()
