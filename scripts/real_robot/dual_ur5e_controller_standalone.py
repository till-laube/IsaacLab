#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dual UR5e Robot Controller (Standalone Version)

This script receives joint commands from Isaac Lab via ZMQ and controls
two UR5e robots with Robotiq 2F-140 grippers using RTDE servoJ.

This is a STANDALONE version with no external dependencies beyond:
- ur_rtde (pip install ur_rtde)
- pyzmq (pip install pyzmq)
- numpy

The controller:
1. Connects to both UR5e robots via RTDE
2. Receives joint commands from Isaac Lab via ZMQ
3. Applies velocity limiting for safety
4. Monitors position divergence with optional watchdog
5. Controls Robotiq grippers via socket ASCII interface

Usage:
    python dual_ur5e_controller_standalone.py

    Or with custom IPs:
    python dual_ur5e_controller_standalone.py --isaac-ip 100.80.147.65
"""

import argparse
import os
import socket
import sys
import threading
import time
from typing import List, Optional

import numpy as np

# Try to import ZMQ
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("[ERROR] pyzmq not installed. Install with: pip install pyzmq")
    sys.exit(1)

# Try to import RTDE interfaces
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("[WARNING] ur_rtde not installed. Install with: pip install ur_rtde")
    print("[WARNING] Running in simulation mode (no real robot control).")


# =============================================================================
# Configuration (hardcoded defaults - modify as needed or use CLI args)
# =============================================================================

# Network Configuration
LEFT_ARM_IP = "100.80.147.160"
RIGHT_ARM_IP = "100.80.147.51"
ISAAC_PC_IP = "100.80.147.65"  # IP of the PC running Isaac Lab
ZMQ_PORT = 5555
GRIPPER_PORT = 63352

# Control Parameters
SERVO_FREQ = 500           # Hz, must be 500 for UR e-Series
DT = 1.0 / SERVO_FREQ
LOOKAHEAD_TIME = 0.05      # [0.03, 0.2] seconds
SERVO_GAIN = 1000          # [100, 2000]

# Velocity Limits (rad/s) - conservative values (70% of UR5e max)
# UR5e max: base/shoulder/elbow = 3.14 rad/s, wrist = 6.28 rad/s
MAX_JOINT_VELOCITY = np.array([
    3.14,   # Base (max 3.14)
    3.14,   # Shoulder (max 3.14)
    3.14,   # Elbow (max 3.14)
    6.28,   # Wrist 1 (max 6.28)
    6.28,   # Wrist 2 (max 6.28)
    6.28,   # Wrist 3 (max 6.28)
])

# Watchdog Configuration
WATCHDOG_ENABLED = True
WATCHDOG_THRESHOLD = 0.1   # radians (~5.7 degrees)
WATCHDOG_WINDOW = 50       # samples

# Gripper Control Configuration
GRIPPER_SPEED = 150
DEFAULT_GRIPPER_FORCE = 150  # Fallback if ZMQ message lacks force
CURRENT_MONITOR_ENABLED = True
CURRENT_POLL_INTERVAL = 0.05  # 20Hz current monitoring during close
CURRENT_MONITOR_TIMEOUT = 3.0  # Max seconds to monitor current during close


# =============================================================================
# Position Watchdog (inline implementation)
# =============================================================================

class PositionWatchdog:
    """Monitors position divergence between commanded and actual robot positions."""

    def __init__(self, threshold_rad: float = 0.1, window_size: int = 50):
        self.threshold = threshold_rad
        self.window_size = window_size
        self.error_history: List[float] = []
        self._divergence_count = 0

    def check(self, commanded_pos: np.ndarray, actual_pos: np.ndarray) -> bool:
        """Check if position divergence is within acceptable limits.

        Returns True if safe, False if divergence detected.
        """
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
        """Get current average error."""
        return float(np.mean(self.error_history)) if self.error_history else 0.0

    def reset(self):
        """Reset the watchdog state."""
        self.error_history.clear()
        self._divergence_count = 0


# =============================================================================
# Main Controller Class
# =============================================================================

class DualUR5eController:
    """Controller for dual UR5e robot arms with Robotiq grippers."""

    def __init__(
        self,
        left_ip: str = LEFT_ARM_IP,
        right_ip: str = RIGHT_ARM_IP,
        isaac_ip: str = ISAAC_PC_IP,
        zmq_port: int = ZMQ_PORT,
        current_monitor: bool = CURRENT_MONITOR_ENABLED,
    ):
        """Initialize the dual arm controller.

        Args:
            left_ip: IP address of left UR5e arm
            right_ip: IP address of right UR5e arm
            isaac_ip: IP address of PC running Isaac Lab
            zmq_port: ZMQ port to connect to
            current_monitor: Enable current-based grip stopping
        """
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.isaac_ip = isaac_ip
        self.zmq_port = zmq_port

        self.running = True
        self.current_monitor_enabled = current_monitor

        # Thread-safe data storage
        self.lock = threading.Lock()
        self.latest_data = {
            "left_joints": None,
            "right_joints": None,
            "left_gripper": 0,
            "right_gripper": 0,
            "left_gripper_force": DEFAULT_GRIPPER_FORCE,
            "right_gripper_force": DEFAULT_GRIPPER_FORCE,
            "left_current_limit": None,
            "right_current_limit": None,
            "timestamp": 0.0,
            "updated": False,
            "active": False,  # CRITICAL: Only move when this is True (SQUEEZE pressed)
        }

        # RTDE interfaces
        self.left_rtde_c: Optional[RTDEControlInterface] = None
        self.left_rtde_r: Optional[RTDEReceiveInterface] = None
        self.right_rtde_c: Optional[RTDEControlInterface] = None
        self.right_rtde_r: Optional[RTDEReceiveInterface] = None

        # Gripper sockets
        self.left_gripper_sock: Optional[socket.socket] = None
        self.right_gripper_sock: Optional[socket.socket] = None

        # Keep positions for safety (hold last position if no new data)
        self.left_keep_pos: Optional[np.ndarray] = None
        self.right_keep_pos: Optional[np.ndarray] = None

        # Raw ZMQ targets (unclamped) for continuous interpolation
        self.left_raw_target: Optional[np.ndarray] = None
        self.right_raw_target: Optional[np.ndarray] = None

        # Watchdogs
        self.left_watchdog = PositionWatchdog(WATCHDOG_THRESHOLD, WATCHDOG_WINDOW) if WATCHDOG_ENABLED else None
        self.right_watchdog = PositionWatchdog(WATCHDOG_THRESHOLD, WATCHDOG_WINDOW) if WATCHDOG_ENABLED else None

        # Track gripper state for current monitoring
        self.left_gripper_closing = False
        self.right_gripper_closing = False

    def connect_robots(self) -> bool:
        """Connect to both UR5e robots via RTDE."""
        if not RTDE_AVAILABLE:
            print("[ROBOT] RTDE not available, skipping robot connection")
            return False

        success = True

        # Connect to left arm
        try:
            print(f"[ROBOT] Connecting to LEFT arm at {self.left_ip}...")
            self.left_rtde_c = RTDEControlInterface(self.left_ip)
            self.left_rtde_r = RTDEReceiveInterface(self.left_ip)
            print("[ROBOT] LEFT arm connected")
        except Exception as e:
            print(f"[ROBOT] Failed to connect to LEFT arm: {e}")
            success = False

        # Connect to right arm
        try:
            print(f"[ROBOT] Connecting to RIGHT arm at {self.right_ip}...")
            self.right_rtde_c = RTDEControlInterface(self.right_ip)
            self.right_rtde_r = RTDEReceiveInterface(self.right_ip)
            print("[ROBOT] RIGHT arm connected")
        except Exception as e:
            print(f"[ROBOT] Failed to connect to RIGHT arm: {e}")
            success = False

        return success

    def connect_grippers(self):
        """Connect to Robotiq grippers via socket ASCII interface."""
        # Left gripper
        try:
            self.left_gripper_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.left_gripper_sock.settimeout(5.0)
            self.left_gripper_sock.connect((self.left_ip, GRIPPER_PORT))
            print(f"[GRIPPER] LEFT gripper connected at {self.left_ip}:{GRIPPER_PORT}")
        except Exception as e:
            print(f"[GRIPPER] LEFT gripper connection failed: {e}")
            self.left_gripper_sock = None

        # Right gripper
        try:
            self.right_gripper_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.right_gripper_sock.settimeout(5.0)
            self.right_gripper_sock.connect((self.right_ip, GRIPPER_PORT))
            print(f"[GRIPPER] RIGHT gripper connected at {self.right_ip}:{GRIPPER_PORT}")
        except Exception as e:
            print(f"[GRIPPER] RIGHT gripper connection failed: {e}")
            self.right_gripper_sock = None

    def _send_gripper_cmd(self, sock: socket.socket, cmd: str) -> Optional[str]:
        """Send command to gripper and return response.

        Args:
            sock: Gripper socket connection
            cmd: Command string (without newline)

        Returns:
            Response string or None if error
        """
        try:
            sock.sendall((cmd.strip() + "\n").encode())
            response = sock.recv(1024).decode().strip()
            return response
        except Exception as e:
            print(f"[GRIPPER] Command error ({cmd}): {e}")
            return None

    def _read_gripper_current(self, sock: socket.socket) -> Optional[float]:
        """Read gripper current consumption in Amps.

        Args:
            sock: Gripper socket connection

        Returns:
            Current in Amps or None if error
        """
        response = self._send_gripper_cmd(sock, "GET COU")
        if response is None:
            return None

        try:
            # Parse response: either raw number or "COU <value>"
            if response.isdigit():
                current_raw = int(response)
            else:
                parts = response.split()
                if len(parts) >= 2 and parts[0] == "COU":
                    current_raw = int(parts[1])
                else:
                    return None

            # Convert raw value to amps (Robotiq formula)
            current_amps = current_raw / 255.0 * 1.5
            return current_amps
        except (ValueError, IndexError):
            return None

    def _read_gripper_position(self, sock: socket.socket) -> Optional[int]:
        """Read gripper position (0-255).

        Args:
            sock: Gripper socket connection

        Returns:
            Position 0-255 or None if error
        """
        response = self._send_gripper_cmd(sock, "GET POS")
        if response is None:
            return None

        try:
            if response.isdigit():
                return int(response)
            else:
                parts = response.split()
                if len(parts) >= 2 and parts[0] == "POS":
                    return int(parts[1])
        except (ValueError, IndexError):
            pass
        return None

    def _close_with_current_limit(
        self,
        sock: socket.socket,
        target_pos: int,
        threshold: float,
        name: str,
        gripper_key: str,
    ) -> bool:
        """Close gripper while monitoring current, stop if threshold exceeded.

        This method checks for target changes during monitoring, allowing
        immediate response if the user wants to open the gripper.

        Args:
            sock: Gripper socket connection
            target_pos: Target position (0-255)
            threshold: Current threshold in Amps
            name: Gripper name for logging ("LEFT" or "RIGHT")
            gripper_key: Key in latest_data ("left_gripper" or "right_gripper")

        Returns:
            True if stopped due to current threshold, False otherwise
        """
        # Start closing
        self._send_gripper_cmd(sock, f"SET POS {target_pos}")
        self._send_gripper_cmd(sock, "SET GTO 1")

        start_time = time.time()
        initial_target = target_pos  # Already 0-255 int

        while time.time() - start_time < CURRENT_MONITOR_TIMEOUT:
            # Check if user wants to change gripper position (e.g., open)
            with self.lock:
                current_target = self.latest_data[gripper_key]

            # If target changed significantly (user wants to open or different position)
            if abs(current_target - initial_target) > 13:
                print(f"[GRIPPER] {name} monitoring interrupted - target changed to {current_target}")
                return False  # Exit and let main loop handle new target

            current = self._read_gripper_current(sock)
            position = self._read_gripper_position(sock)

            if current is not None:
                # Check if current exceeds threshold
                if current > threshold:
                    # Stop gripper movement
                    self._send_gripper_cmd(sock, "SET GTO 0")
                    print(f"[GRIPPER] {name} stopped at {current:.4f}A "
                          f"(threshold: {threshold}A, pos: {position})")
                    return True

            # Check if we've reached target position
            if position is not None and position >= target_pos - 5:
                break

            time.sleep(CURRENT_POLL_INTERVAL)

        return False

    def initialize_grippers(self):
        """Configure gripper speed and default force at startup."""
        print("[GRIPPER] Initializing gripper settings...")

        if self.left_gripper_sock:
            self._send_gripper_cmd(self.left_gripper_sock, f"SET SPE {GRIPPER_SPEED}")
            self._send_gripper_cmd(self.left_gripper_sock, f"SET FOR {DEFAULT_GRIPPER_FORCE}")
            self._send_gripper_cmd(self.left_gripper_sock, "SET GTO 1")
            print(f"[GRIPPER] LEFT initialized: speed={GRIPPER_SPEED}, force={DEFAULT_GRIPPER_FORCE}")

        if self.right_gripper_sock:
            self._send_gripper_cmd(self.right_gripper_sock, f"SET SPE {GRIPPER_SPEED}")
            self._send_gripper_cmd(self.right_gripper_sock, f"SET FOR {DEFAULT_GRIPPER_FORCE}")
            self._send_gripper_cmd(self.right_gripper_sock, "SET GTO 1")
            print(f"[GRIPPER] RIGHT initialized: speed={GRIPPER_SPEED}, force={DEFAULT_GRIPPER_FORCE}")

    def clamp_velocity(
        self, current_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        """Apply velocity limiting to prevent sudden movements."""
        delta = np.array(target_pos) - np.array(current_pos)
        max_delta = MAX_JOINT_VELOCITY * DT
        clamped_delta = np.clip(delta, -max_delta, max_delta)
        return current_pos + clamped_delta

    def zmq_receiver_thread(self):
        """Thread for receiving joint commands via ZMQ."""
        context = zmq.Context()
        sock = context.socket(zmq.SUB)

        print(f"[ZMQ] Connecting to {self.isaac_ip}:{self.zmq_port}...")
        sock.connect(f"tcp://{self.isaac_ip}:{self.zmq_port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        print("[ZMQ] Connected and subscribed")

        while self.running:
            try:
                data = sock.recv_json()

                with self.lock:
                    # Parse dual arm message format
                    if isinstance(data, dict):
                        self.latest_data["left_joints"] = data.get("left_arm")
                        self.latest_data["right_joints"] = data.get("right_arm")
                        self.latest_data["left_gripper"] = data.get("left_gripper", 0)
                        self.latest_data["right_gripper"] = data.get("right_gripper", 0)
                        self.latest_data["left_gripper_force"] = data.get("left_gripper_force", DEFAULT_GRIPPER_FORCE)
                        self.latest_data["right_gripper_force"] = data.get("right_gripper_force", DEFAULT_GRIPPER_FORCE)
                        self.latest_data["left_current_limit"] = data.get("left_current_limit", None)
                        self.latest_data["right_current_limit"] = data.get("right_current_limit", None)
                        self.latest_data["timestamp"] = data.get("timestamp", time.time())
                        # CRITICAL: Parse active flag - only move when True
                        self.latest_data["active"] = data.get("active", False)
                    else:
                        # Legacy single-arm format (for testing with isaac_sender.py)
                        self.latest_data["left_joints"] = data[:6] if len(data) >= 6 else None
                        self.latest_data["left_gripper"] = data[6] if len(data) > 6 else 0
                        self.latest_data["active"] = False  # Legacy format = inactive

                    self.latest_data["updated"] = True

            except zmq.ZMQError as e:
                print(f"[ZMQ] Error: {e}")
                time.sleep(0.1)

        sock.close()
        context.term()

    def gripper_control_thread(self):
        """Thread for controlling grippers with current-based stopping.

        Runs at ~50Hz. Gripper values arrive as 0-255 integers from ZMQ.
        Force and current_limit are read from latest_data (set by teleop side).
        """
        update_threshold = 13  # Only send if change exceeds this (~5% of 255)

        last_left_val = -999
        last_right_val = -999
        last_left_force = DEFAULT_GRIPPER_FORCE
        last_right_force = DEFAULT_GRIPPER_FORCE

        while self.running:
            with self.lock:
                is_active = self.latest_data["active"]
                left_target = self.latest_data["left_gripper"]
                right_target = self.latest_data["right_gripper"]
                left_force = self.latest_data["left_gripper_force"]
                right_force = self.latest_data["right_gripper_force"]
                left_current_limit = self.latest_data["left_current_limit"]
                right_current_limit = self.latest_data["right_current_limit"]

            # CRITICAL: Only control grippers when teleop is ACTIVE
            if not is_active:
                time.sleep(0.02)
                continue

            # Left gripper
            # Update force dynamically if it changed
            if self.left_gripper_sock and left_force != last_left_force:
                self._send_gripper_cmd(self.left_gripper_sock, f"SET FOR {left_force}")
                print(f"[GRIPPER] LEFT force updated to {left_force}")
                last_left_force = left_force

            if self.left_gripper_sock and abs(left_target - last_left_val) > update_threshold:
                try:
                    pos_val = max(0, min(255, int(left_target)))

                    is_closing = left_target > last_left_val

                    if is_closing and left_current_limit and self.current_monitor_enabled:
                        stopped = self._close_with_current_limit(
                            self.left_gripper_sock,
                            pos_val,
                            left_current_limit,
                            "LEFT",
                            "left_gripper"
                        )
                        if not stopped:
                            print(f"[GRIPPER] LEFT closed to {pos_val}")
                    else:
                        self._send_gripper_cmd(self.left_gripper_sock, f"SET POS {pos_val}")
                        self._send_gripper_cmd(self.left_gripper_sock, "SET GTO 1")
                        action = "opening" if left_target < last_left_val else "closing"
                        print(f"[GRIPPER] LEFT {action} to {pos_val}")

                    last_left_val = left_target
                except Exception as e:
                    print(f"[GRIPPER] LEFT error: {e}")

            # Right gripper
            # Update force dynamically if it changed
            if self.right_gripper_sock and right_force != last_right_force:
                self._send_gripper_cmd(self.right_gripper_sock, f"SET FOR {right_force}")
                print(f"[GRIPPER] RIGHT force updated to {right_force}")
                last_right_force = right_force

            if self.right_gripper_sock and abs(right_target - last_right_val) > update_threshold:
                try:
                    pos_val = max(0, min(255, int(right_target)))

                    is_closing = right_target > last_right_val

                    if is_closing and right_current_limit and self.current_monitor_enabled:
                        stopped = self._close_with_current_limit(
                            self.right_gripper_sock,
                            pos_val,
                            right_current_limit,
                            "RIGHT",
                            "right_gripper"
                        )
                        if not stopped:
                            print(f"[GRIPPER] RIGHT closed to {pos_val}")
                    else:
                        self._send_gripper_cmd(self.right_gripper_sock, f"SET POS {pos_val}")
                        self._send_gripper_cmd(self.right_gripper_sock, "SET GTO 1")
                        action = "opening" if right_target < last_right_val else "closing"
                        print(f"[GRIPPER] RIGHT {action} to {pos_val}")

                    last_right_val = right_target
                except Exception as e:
                    print(f"[GRIPPER] RIGHT error: {e}")

            time.sleep(0.02)  # 50Hz is sufficient for grippers

    def read_current_robot_positions(self):
        """Read current positions from real robots and use as initial keep positions.

        This ensures the robot doesn't move until teleoperation actually starts.
        """
        print("[ROBOT] Reading current robot positions...")

        # Read left arm current position
        if self.left_rtde_r:
            try:
                self.left_keep_pos = np.array(self.left_rtde_r.getActualQ())
                print(f"[ROBOT] LEFT arm current: {[f'{j:.3f}' for j in self.left_keep_pos]}")
            except Exception as e:
                print(f"[ROBOT] Failed to read LEFT arm position: {e}")

        # Read right arm current position
        if self.right_rtde_r:
            try:
                self.right_keep_pos = np.array(self.right_rtde_r.getActualQ())
                print(f"[ROBOT] RIGHT arm current: {[f'{j:.3f}' for j in self.right_keep_pos]}")
            except Exception as e:
                print(f"[ROBOT] Failed to read RIGHT arm position: {e}")

    def wait_for_first_command(self) -> bool:
        """Wait for the first valid command from Isaac Lab."""
        print("[ROBOT] Waiting for first command from Isaac Lab...")
        print("[ROBOT] (Make sure teleop_se3_agent.py is running with --real-robot flag)")

        while self.running:
            with self.lock:
                if self.latest_data["left_joints"] is not None:
                    return True
            time.sleep(0.1)

        return False

    def initialize_hold_position(self):
        """Initialize servo control by holding current position.

        Unlike moveJ, this does NOT move the robot - it just starts
        servoJ control at the current position. The robot will only
        move when teleoperation sends different positions.
        """
        print("[ROBOT] Initializing servo control (holding current position)...")
        print("[ROBOT] Robot will NOT move until you press SQUEEZE and start teleop.")

        # The keep_pos was already set by read_current_robot_positions()
        # We don't need to do anything else - the servo loop will hold this position

    def run_servo_loop(self):
        """Main 500Hz servo control loop."""
        print(f"[ROBOT] Starting {SERVO_FREQ}Hz servo loop...")
        print("[ROBOT] Waiting for SQUEEZE button press to start movement...")
        print("[ROBOT] Press Ctrl+C to stop")

        loop_count = 0
        last_status_time = time.time()
        was_active = False
        servo_started = False

        try:
            while self.running:
                t_start = time.time()

                # Get latest data including active flag
                with self.lock:
                    updated = self.latest_data["updated"]
                    is_active = self.latest_data["active"]
                    left_target = self.latest_data["left_joints"]
                    right_target = self.latest_data["right_joints"]
                    self.latest_data["updated"] = False

                # Detect activation state changes
                if is_active and not was_active:
                    print("[ROBOT] >>> SQUEEZE PRESSED - Starting robot movement <<<")
                    # Read current position as starting point when activating
                    if self.left_rtde_r:
                        self.left_keep_pos = np.array(self.left_rtde_r.getActualQ())
                    if self.right_rtde_r:
                        self.right_keep_pos = np.array(self.right_rtde_r.getActualQ())
                    servo_started = True
                elif not is_active and was_active:
                    print("[ROBOT] >>> SQUEEZE RELEASED - Stopping robot movement <<<")
                    # Stop servo cleanly
                    if self.left_rtde_c:
                        try:
                            self.left_rtde_c.servoStop()
                        except:
                            pass
                    if self.right_rtde_c:
                        try:
                            self.right_rtde_c.servoStop()
                        except:
                            pass
                    servo_started = False

                was_active = is_active

                # CRITICAL: Only send servo commands when teleop is ACTIVE
                if is_active and servo_started:
                    # Update raw targets when new ZMQ data arrives
                    if updated:
                        if left_target is not None:
                            self.left_raw_target = np.array(left_target)
                        if right_target is not None:
                            self.right_raw_target = np.array(right_target)

                    # Process left arm - interpolate toward target EVERY tick
                    if self.left_rtde_c and self.left_keep_pos is not None and self.left_raw_target is not None:
                        self.left_keep_pos = self.clamp_velocity(self.left_keep_pos, self.left_raw_target)

                        # Watchdog check
                        if self.left_watchdog and self.left_rtde_r:
                            actual = np.array(self.left_rtde_r.getActualQ())
                            if not self.left_watchdog.check(self.left_keep_pos, actual):
                                if loop_count % 500 == 0:
                                    print(f"[WATCHDOG] LEFT arm divergence: {self.left_watchdog.get_avg_error():.3f} rad")

                        # Send servo command
                        self.left_rtde_c.servoJ(
                            self.left_keep_pos.tolist(),
                            0, 0,
                            DT,
                            LOOKAHEAD_TIME,
                            SERVO_GAIN
                        )

                    # Process right arm - interpolate toward target EVERY tick
                    if self.right_rtde_c and self.right_keep_pos is not None and self.right_raw_target is not None:
                        self.right_keep_pos = self.clamp_velocity(self.right_keep_pos, self.right_raw_target)

                        # Watchdog check
                        if self.right_watchdog and self.right_rtde_r:
                            actual = np.array(self.right_rtde_r.getActualQ())
                            if not self.right_watchdog.check(self.right_keep_pos, actual):
                                if loop_count % 500 == 0:
                                    print(f"[WATCHDOG] RIGHT arm divergence: {self.right_watchdog.get_avg_error():.3f} rad")

                        # Send servo command
                        self.right_rtde_c.servoJ(
                            self.right_keep_pos.tolist(),
                            0, 0,
                            DT,
                            LOOKAHEAD_TIME,
                            SERVO_GAIN
                        )

                # Periodic status output
                loop_count += 1
                if time.time() - last_status_time > 5.0:
                    status = "ACTIVE" if is_active else "WAITING (press SQUEEZE to start)"
                    print(f"[STATUS] {status} (loop {loop_count})")
                    last_status_time = time.time()

                # Maintain timing
                t_cycle = time.time() - t_start
                if t_cycle < DT:
                    time.sleep(DT - t_cycle)

        except KeyboardInterrupt:
            print("\n[ROBOT] Stopping...")

    def stop(self):
        """Stop all robot motion and cleanup."""
        self.running = False

        # Stop servo motion
        if self.left_rtde_c:
            try:
                self.left_rtde_c.servoStop()
                self.left_rtde_c.stopScript()
            except Exception:
                pass

        if self.right_rtde_c:
            try:
                self.right_rtde_c.servoStop()
                self.right_rtde_c.stopScript()
            except Exception:
                pass

        # Close gripper sockets
        if self.left_gripper_sock:
            try:
                self.left_gripper_sock.close()
            except Exception:
                pass

        if self.right_gripper_sock:
            try:
                self.right_gripper_sock.close()
            except Exception:
                pass

        print("[ROBOT] Stopped and cleaned up")

    def run(self):
        """Main entry point for running the controller."""
        print("=" * 70)
        print("Dual UR5e Controller (Standalone)")
        print("=" * 70)
        print(f"LEFT arm IP:   {self.left_ip}")
        print(f"RIGHT arm IP:  {self.right_ip}")
        print(f"Isaac Lab IP:  {self.isaac_ip}:{self.zmq_port}")
        print(f"Servo freq:    {SERVO_FREQ} Hz")
        print(f"Watchdog:      {'Enabled' if WATCHDOG_ENABLED else 'Disabled'}")
        print(f"Current mon:   {'Enabled' if self.current_monitor_enabled else 'Disabled'}")
        print("=" * 70)

        # Connect to robots
        if not self.connect_robots():
            if RTDE_AVAILABLE:
                print("[ERROR] Failed to connect to robots. Exiting.")
                return
            else:
                print("[INFO] Running without robot connection (debug mode)")

        # Connect to grippers
        self.connect_grippers()

        # Initialize gripper force/speed settings based on object type
        self.initialize_grippers()

        # Read current robot positions FIRST (before any movement)
        # This ensures we hold the current position until teleop starts
        if RTDE_AVAILABLE:
            self.read_current_robot_positions()

        # Start ZMQ receiver thread
        zmq_thread = threading.Thread(target=self.zmq_receiver_thread, daemon=True)
        zmq_thread.start()

        # Start gripper control thread
        gripper_thread = threading.Thread(target=self.gripper_control_thread, daemon=True)
        gripper_thread.start()

        # Wait for first command from Isaac Lab
        if not self.wait_for_first_command():
            self.stop()
            return

        # Initialize hold position (NO moveJ - robot stays where it is)
        if RTDE_AVAILABLE:
            self.initialize_hold_position()

        # Run servo loop
        try:
            if RTDE_AVAILABLE:
                self.run_servo_loop()
            else:
                # Debug mode: just print received data
                print("[DEBUG] Running in debug mode (printing received data)")
                while self.running:
                    with self.lock:
                        if self.latest_data["updated"]:
                            print(f"[DEBUG] Left:  {self.latest_data['left_joints']}")
                            print(f"[DEBUG] Right: {self.latest_data['right_joints']}")
                            print(f"[DEBUG] Grippers: L={self.latest_data['left_gripper']}, R={self.latest_data['right_gripper']}")
                            self.latest_data["updated"] = False
                    time.sleep(0.1)
        finally:
            self.stop()
            zmq_thread.join(timeout=1.0)
            gripper_thread.join(timeout=1.0)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual UR5e Robot Controller - Receives joint commands via ZMQ and controls robots via RTDE",
    )
    parser.add_argument(
        "--left-ip",
        type=str,
        default=LEFT_ARM_IP,
        help=f"IP address of left UR5e arm (default: {LEFT_ARM_IP})",
    )
    parser.add_argument(
        "--right-ip",
        type=str,
        default=RIGHT_ARM_IP,
        help=f"IP address of right UR5e arm (default: {RIGHT_ARM_IP})",
    )
    parser.add_argument(
        "--isaac-ip",
        type=str,
        default=ISAAC_PC_IP,
        help=f"IP address of PC running Isaac Lab (default: {ISAAC_PC_IP})",
    )
    parser.add_argument(
        "--zmq-port",
        type=int,
        default=ZMQ_PORT,
        help=f"ZMQ port to connect to (default: {ZMQ_PORT})",
    )
    parser.add_argument(
        "--no-current-monitor",
        action="store_true",
        help="Disable current-based grip stopping (always close fully)",
    )
    args = parser.parse_args()

    controller = DualUR5eController(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        isaac_ip=args.isaac_ip,
        zmq_port=args.zmq_port,
        current_monitor=not args.no_current_monitor,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        controller.stop()


if __name__ == "__main__":
    main()
