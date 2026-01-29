#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Home Dual UR5e Arms

Moves both UR5e arms to their home joint positions using velocity-limited
servoJ control. Adds small random offsets (±1 degree) to each arm joint
to avoid exact repeatability. Grippers are always closed fully.

Usage:
    python home_arms.py
    python home_arms.py --config /path/to/real_robot_config.yaml
    python home_arms.py --left-ip 100.80.147.160 --right-ip 100.80.147.78
"""

import argparse
import math
import os
import socket
import signal
import sys
import time

import numpy as np
import yaml

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("[ERROR] ur_rtde not installed. Install with: pip install ur_rtde")
    sys.exit(1)

# Home positions in radians
LEFT_HOME = np.array([-1.571, -1.745, -2.531, -1.920, 4.712, -0.785])
RIGHT_HOME = np.array([1.571, -1.396, 2.531, -1.222, -4.712, 0.785])

# Max random offset per joint: ±1 degree in radians
MAX_OFFSET_RAD = math.radians(1.0)

# Convergence threshold (rad) — stop when all joints within this of target
CONVERGENCE_THRESH = 0.001

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "real_robot_config.yaml")


def load_config(config_path: str) -> dict:
    """Load config from YAML, falling back to defaults."""
    defaults = {
        "network": {
            "left_arm_ip": "100.80.147.160",
            "right_arm_ip": "100.80.147.51",
            "gripper_port": 63352,
        },
        "control": {
            "servo_frequency": 500,
            "lookahead_time": 0.05,
            "servo_gain": 1000,
        },
        "velocity_limits": {
            "base": 2.2,
            "shoulder": 2.2,
            "elbow": 2.2,
            "wrist_1": 4.4,
            "wrist_2": 4.4,
            "wrist_3": 4.4,
        },
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_cfg = yaml.safe_load(f)
            for key in file_cfg:
                if key in defaults and isinstance(defaults[key], dict):
                    defaults[key].update(file_cfg[key])
                else:
                    defaults[key] = file_cfg[key]
        print(f"[CONFIG] Loaded from {config_path}")
    else:
        print("[CONFIG] Using defaults")

    return defaults


def close_gripper(ip: str, port: int, label: str):
    """Send full-close command to a Robotiq gripper."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ip, port))
        sock.sendall(b"SET POS 255\n")
        print(f"[GRIPPER] {label} gripper → closed (255)")
        time.sleep(0.1)
        sock.close()
    except Exception as e:
        print(f"[GRIPPER] {label} gripper command failed: {e}")


def home_arms(args):
    cfg = load_config(args.config)

    # Allow CLI overrides for IPs
    left_ip = args.left_ip or cfg["network"]["left_arm_ip"]
    right_ip = args.right_ip or cfg["network"]["right_arm_ip"]
    gripper_port = cfg["network"]["gripper_port"]

    ctrl = cfg["control"]
    freq = ctrl["servo_frequency"]
    dt = 1.0 / freq
    lookahead = ctrl["lookahead_time"]
    gain = ctrl["servo_gain"]

    vel_cfg = cfg["velocity_limits"]
    max_velocity = np.array([
        vel_cfg["base"], vel_cfg["shoulder"], vel_cfg["elbow"],
        vel_cfg["wrist_1"], vel_cfg["wrist_2"], vel_cfg["wrist_3"],
    ])

    # Randomize home targets (±1 deg per arm joint)
    rng = np.random.default_rng()
    left_target = LEFT_HOME + rng.uniform(-MAX_OFFSET_RAD, MAX_OFFSET_RAD, size=6)
    right_target = RIGHT_HOME + rng.uniform(-MAX_OFFSET_RAD, MAX_OFFSET_RAD, size=6)

    print("=" * 60)
    print("Home Dual UR5e Arms")
    print("=" * 60)
    print(f"LEFT  arm IP : {left_ip}")
    print(f"RIGHT arm IP : {right_ip}")
    print(f"LEFT  target : {np.array2string(left_target, precision=4, separator=', ')}")
    print(f"RIGHT target : {np.array2string(right_target, precision=4, separator=', ')}")
    print("=" * 60)

    # --- Connect RTDE ---
    print(f"[RTDE] Connecting to LEFT arm at {left_ip}...")
    left_ctrl = RTDEControlInterface(left_ip)
    left_recv = RTDEReceiveInterface(left_ip)
    print("[RTDE] LEFT arm connected")

    print(f"[RTDE] Connecting to RIGHT arm at {right_ip}...")
    right_ctrl = RTDEControlInterface(right_ip)
    right_recv = RTDEReceiveInterface(right_ip)
    print("[RTDE] RIGHT arm connected")

    # Read starting positions
    left_pos = np.array(left_recv.getActualQ())
    right_pos = np.array(right_recv.getActualQ())
    print(f"[START] LEFT  pos: {np.array2string(left_pos, precision=4, separator=', ')}")
    print(f"[START] RIGHT pos: {np.array2string(right_pos, precision=4, separator=', ')}")

    # --- Graceful shutdown on Ctrl+C ---
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # --- Servo loop ---
    print(f"[SERVO] Running at {freq} Hz — moving to home...")
    left_done = False
    right_done = False
    loop_count = 0

    try:
        while running and not (left_done and right_done):
            t0 = time.time()

            # Left arm
            if not left_done:
                cur = np.array(left_recv.getActualQ())
                delta = left_target - cur
                max_delta = max_velocity * dt
                clamped = np.clip(delta, -max_delta, max_delta)
                cmd = cur + clamped
                left_ctrl.servoJ(cmd.tolist(), 0, 0, dt, lookahead, gain)
                if np.all(np.abs(left_target - cur) < CONVERGENCE_THRESH):
                    left_done = True
                    print("[DONE] LEFT arm reached home")

            # Right arm
            if not right_done:
                cur = np.array(right_recv.getActualQ())
                delta = right_target - cur
                max_delta = max_velocity * dt
                clamped = np.clip(delta, -max_delta, max_delta)
                cmd = cur + clamped
                right_ctrl.servoJ(cmd.tolist(), 0, 0, dt, lookahead, gain)
                if np.all(np.abs(right_target - cur) < CONVERGENCE_THRESH):
                    right_done = True
                    print("[DONE] RIGHT arm reached home")

            loop_count += 1
            if loop_count % (freq * 2) == 0:  # every 2 seconds
                left_err = np.max(np.abs(left_target - np.array(left_recv.getActualQ())))
                right_err = np.max(np.abs(right_target - np.array(right_recv.getActualQ())))
                print(f"[STATUS] max err  L={left_err:.4f} R={right_err:.4f} rad")

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        # Always stop cleanly
        print("[STOP] Halting servo control...")
        for ctrl_if in (left_ctrl, right_ctrl):
            try:
                ctrl_if.servoStop()
            except Exception:
                pass
            try:
                ctrl_if.stopScript()
            except Exception:
                pass

    if left_done and right_done:
        print("[OK] Both arms at home position")
    else:
        print("[WARN] Interrupted before both arms converged")

    # --- Close grippers ---
    print("[GRIPPER] Closing both grippers...")
    close_gripper(left_ip, gripper_port, "LEFT")
    close_gripper(right_ip, gripper_port, "RIGHT")

    print("[DONE] Homing complete")


def main():
    parser = argparse.ArgumentParser(description="Home dual UR5e arms to base position")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to real_robot_config.yaml",
    )
    parser.add_argument("--left-ip", type=str, default=None, help="Override left arm IP")
    parser.add_argument("--right-ip", type=str, default=None, help="Override right arm IP")
    args = parser.parse_args()

    home_arms(args)


if __name__ == "__main__":
    main()
