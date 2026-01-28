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
- pyyaml (optional, for config file)

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
RIGHT_ARM_IP = "100.80.147.78"
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
    2.2,   # Base (max 3.14)
    2.2,   # Shoulder (max 3.14)
    2.2,   # Elbow (max 3.14)
    4.4,   # Wrist 1 (max 6.28)
    4.4,   # Wrist 2 (max 6.28)
    4.4,   # Wrist 3 (max 6.28)
])

# Watchdog Configuration
WATCHDOG_ENABLED = True
WATCHDOG_THRESHOLD = 0.1   # radians (~5.7 degrees)
WATCHDOG_WINDOW = 50       # samples


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
    ):
        """Initialize the dual arm controller."""
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.isaac_ip = isaac_ip
        self.zmq_port = zmq_port

        self.running = True

        # Thread-safe data storage
        self.lock = threading.Lock()
        self.latest_data = {
            "left_joints": None,
            "right_joints": None,
            "left_gripper": 0.0,
            "right_gripper": 0.0,
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

        # Watchdogs
        self.left_watchdog = PositionWatchdog(WATCHDOG_THRESHOLD, WATCHDOG_WINDOW) if WATCHDOG_ENABLED else None
        self.right_watchdog = PositionWatchdog(WATCHDOG_THRESHOLD, WATCHDOG_WINDOW) if WATCHDOG_ENABLED else None

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
                        self.latest_data["left_gripper"] = data.get("left_gripper", 0.0)
                        self.latest_data["right_gripper"] = data.get("right_gripper", 0.0)
                        self.latest_data["timestamp"] = data.get("timestamp", time.time())
                        # CRITICAL: Parse active flag - only move when True
                        self.latest_data["active"] = data.get("active", False)
                    else:
                        # Legacy single-arm format (for testing with isaac_sender.py)
                        self.latest_data["left_joints"] = data[:6] if len(data) >= 6 else None
                        self.latest_data["left_gripper"] = data[6] if len(data) > 6 else 0.0
                        self.latest_data["active"] = False  # Legacy format = inactive

                    self.latest_data["updated"] = True

            except zmq.ZMQError as e:
                print(f"[ZMQ] Error: {e}")
                time.sleep(0.1)

        sock.close()
        context.term()

    def gripper_control_thread(self):
        """Thread for controlling grippers at lower frequency (50Hz)."""
        update_threshold = 0.05  # Only send if change exceeds this

        last_left_val = -999.0
        last_right_val = -999.0

        while self.running:
            with self.lock:
                is_active = self.latest_data["active"]
                left_target = self.latest_data["left_gripper"]
                right_target = self.latest_data["right_gripper"]

            # CRITICAL: Only control grippers when teleop is ACTIVE
            if not is_active:
                time.sleep(0.02)
                continue

            # Left gripper
            if self.left_gripper_sock and abs(left_target - last_left_val) > update_threshold:
                try:
                    pos_val = int(left_target * 255)
                    pos_val = max(0, min(255, pos_val))
                    cmd = f"SET POS {pos_val}\n".encode()
                    self.left_gripper_sock.sendall(cmd)
                    last_left_val = left_target
                    print(f"[GRIPPER] LEFT set to {pos_val} ({left_target:.2f})")
                except Exception as e:
                    print(f"[GRIPPER] LEFT send error: {e}")

            # Right gripper
            if self.right_gripper_sock and abs(right_target - last_right_val) > update_threshold:
                try:
                    pos_val = int(right_target * 255)
                    pos_val = max(0, min(255, pos_val))
                    cmd = f"SET POS {pos_val}\n".encode()
                    self.right_gripper_sock.sendall(cmd)
                    last_right_val = right_target
                    print(f"[GRIPPER] RIGHT set to {pos_val} ({right_target:.2f})")
                except Exception as e:
                    print(f"[GRIPPER] RIGHT send error: {e}")

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
                    # Process left arm
                    if self.left_rtde_c and self.left_keep_pos is not None:
                        if updated and left_target is not None:
                            # Apply velocity limiting
                            current_pos = np.array(self.left_rtde_r.getActualQ()) if self.left_rtde_r else self.left_keep_pos
                            target_limited = self.clamp_velocity(current_pos, np.array(left_target))
                            self.left_keep_pos = target_limited

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

                    # Process right arm
                    if self.right_rtde_c and self.right_keep_pos is not None:
                        if updated and right_target is not None:
                            # Apply velocity limiting
                            current_pos = np.array(self.right_rtde_r.getActualQ()) if self.right_rtde_r else self.right_keep_pos
                            target_limited = self.clamp_velocity(current_pos, np.array(right_target))
                            self.right_keep_pos = target_limited

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
                            print(f"[DEBUG] Grippers: L={self.latest_data['left_gripper']:.2f}, R={self.latest_data['right_gripper']:.2f}")
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
        description="Dual UR5e Robot Controller - Receives joint commands via ZMQ and controls robots via RTDE"
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
    args = parser.parse_args()

    controller = DualUR5eController(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        isaac_ip=args.isaac_ip,
        zmq_port=args.zmq_port,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        controller.stop()


if __name__ == "__main__":
    main()
