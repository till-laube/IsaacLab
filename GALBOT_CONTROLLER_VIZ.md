# Galbot Controller Visualization Guide

## Overview

Controller pose visualization has been added to the **Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0** task. This helps you debug and tune the VR controller mapping for the Galbot left arm.

## What You'll See

When running the task in XR mode with a Vive controller, you'll see **two coordinate frames**:

1. **Small frame (0.05 scale) at the gripper**: The TCP (Tool Center Point) pose - already existed
2. **Large frame (0.1 scale) at controller location**: The controller's actual pose in world space - **NEW!**

The spatial relationship between these frames shows how the controller motion maps to robot motion.

## Quick Start

Run the test script:
```bash
./scripts/test_galbot_controller_viz.sh
```

Or manually:
```bash
export USE_RELATIVE_MODE=True
~/IsaacLab/isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0 \
    --teleop_device vive \
    --xr
```

## What the Visualization Shows

### Position Information
- **Controller position**: Where your VR controller is in the virtual world
- **TCP position**: Where the gripper TCP is
- **Position difference**: The offset/scaling applied by the retargeter

### Orientation Information
- **Controller orientation**: How you're holding the controller
- **TCP orientation**: How the gripper is oriented
- **Rotation difference**: The rotation offset applied

## Debug Output

To enable detailed debug information, modify the environment creation in the teleoperation script:

Edit `scripts/environments/teleoperation/teleop_se3_agent.py` around line 111:

```python
# Change from:
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

# To:
env = gym.make(args_cli.task, cfg=env_cfg, debug_controller_data=True).unwrapped
```

This will print every ~0.5 seconds:
- TCP position and orientation
- Controller position and orientation
- Position difference and distance
- Rotation difference in Euler angles (XYZ degrees)

Example output:
```
[Controller Viz Debug - LEFT ARM]
  TCP Position: [0.123 0.456 0.789]
  TCP Quat [w,x,y,z]: [1.0 0.0 0.0 0.0]
  Controller Position: [0.200 0.500 0.800]
  Controller Quat [w,x,y,z]: [0.98 0.01 0.02 0.03]
  Position Difference (controller - TCP): [0.077 0.044 0.011]
  Position Distance: 0.0891 m
  Rotation Difference (euler XYZ degrees): [2.3 4.6 3.4]
```

## Tuning the Controller Offset

Once you can see both frames:

1. **Move the controller** and observe how the frames relate
2. **Note the position offset** - distance and direction from controller to TCP
3. **Note the rotation offset** - which axes are misaligned (X=red, Y=green, Z=blue)
4. **Adjust the offset** in `stack_rmp_rel_env_cfg.py` around line 61:

```python
body_offset=RMPFlowActionCfg.OffsetCfg(
    pos=[0.0, 0.0, 0.0],           # Position offset [x, y, z]
    rot=[0.7071068, 0, -0.7071068, 0]  # Rotation offset [qw, qx, qy, qz]
)
```

5. **Adjust sensitivity** if needed (line 96):
```python
pos_sensitivity=5.0,  # Position movement sensitivity
rot_sensitivity=5.0,  # Rotation sensitivity
```

6. **Test** and verify the alignment improved

## Troubleshooting

### Frame Not Visible

If you don't see the controller frame:

1. **Check startup messages** - look for "[Controller Viz] Created controller visualization marker"
2. **Verify XR mode** - make sure you're using `--xr` flag
3. **Check controller tracking** - verify the controller is on and tracked in SteamVR
4. **Enable debug output** to see detailed error messages

### Wrong Behavior

If the frame appears but behaves incorrectly:

1. **Frame doesn't move with controller**: Controller tracking issue - check SteamVR
2. **Frame in wrong location**: XR anchor misconfigured - check `xr.anchor_pos` in config
3. **Frame orientation wrong**: Check the body_offset rotation quaternion

### Debug Messages

Key error messages and what they mean:

- `XRCore not available` → Not running in XR mode, use `--xr` flag
- `Left controller device not found` → Controller not detected by SteamVR
- `Controller pose matrix is None` → Controller exists but not actively tracked
- `Failed to get TCP data` → Frame transformer misconfigured

## Files Modified

- **`stack_rmp_rel_env.py`** (NEW): Custom environment with controller visualization
- **`__init__.py`**: Updated to use custom environment for the left arm task
- **`stack_rmp_rel_env_cfg.py`**: No changes needed (configuration remains the same)

## Configuration Details

The Galbot left arm task uses:
- **Robot asset**: "robot" (Galbot)
- **TCP link**: "left_gripper_tcp_link"
- **Controller**: Left Vive controller (`/user/hand/left`)
- **Control mode**: RMPFlow with relative mode (delta movements)
- **Frame sensor**: `left_control_frame` (already existed for TCP visualization)

## Disable Visualization

If you want to disable the controller frame visualization:

```python
env = gym.make(
    "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0",
    cfg=env_cfg,
    enable_controller_viz=False
)
```

## Implementation Notes

The visualization:
1. Queries controller pose directly from XRCore using `get_virtual_world_pose()`
2. Extracts both position and orientation from the pose matrix
3. Creates a VisualizationMarkers instance with a coordinate frame
4. Updates the marker every frame in `step()`
5. Hides the marker (moves to y=-100) when controller data is unavailable

The controller pose is already in world space (transformed by the XR anchor), so no additional transformation is needed.

## Related Documentation

- **General controller visualization**: See `CONTROLLER_VIZ_USAGE.md`
- **Debugging guide**: See `DEBUG_CONTROLLER_VIZ.md`
- **UR5e dual arm example**: See the UR5e dual manipulation task for a dual-arm implementation

## Next Steps

1. Run the task and verify visualization works
2. Enable debug output to see numerical data
3. Tune the controller offset based on the visualization
4. Adjust sensitivity as needed for comfortable control
5. Test with the actual cube stacking task
