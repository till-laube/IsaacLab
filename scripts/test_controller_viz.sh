#!/bin/bash
# Test script for controller visualization
# This runs the UR5e dual manipulation task with controller visualization enabled

echo "Starting Isaac-Ur5e-Dual-v0 with controller visualization..."
echo "================================================"
echo ""
echo "What to look for:"
echo "  1. '[Controller Viz] Created LEFT/RIGHT controller visualization marker' messages"
echo "  2. '✓ Successfully visualizing LEFT/RIGHT controller frame' messages after first update"
echo "  3. Two coordinate frames at EACH gripper (left and right):"
echo "     - Small frame (0.1 scale) = TCP/end-effector orientation"
echo "     - Large frame (0.15 scale) = Controller orientation"
echo ""
echo "If you don't see the second message, check the error messages below."
echo "================================================"
echo ""

# Run with XR mode enabled
~/IsaacLab/isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Ur5e-Dual-v0 \
    --teleop_device vive \
    --xr \
    "$@"
