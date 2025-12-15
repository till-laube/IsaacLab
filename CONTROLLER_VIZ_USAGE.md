# Controller Visualization - Quick Start Guide

## What Was Added

I've added visualization of the VR controller's coordinate system at the gripper TCP for your Isaac-Ur5e-Dual-v0 task. This helps you see how the controller orientation differs from the end-effector orientation, making it easier to tune the controller offset.

## Visual Changes

When running in XR mode, you'll now see **two coordinate frames** at each gripper (left and right):

1. **Small frame (0.1 scale)**: The actual TCP/end-effector orientation (already existed)
2. **Larger frame (0.15 scale)**: The controller's orientation at the same TCP location (NEW!)

The difference between these two frames shows you the rotation offset being applied by the retargeter.

## How to Use

### Basic Usage

Run your teleoperation as normal:

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Ur5e-Dual-v0 \
    --teleop_device vive \
    --xr
```

The controller frame will automatically appear!

### With Debug Output

To see numerical orientation data (printed every ~0.5 seconds):

```python
# Modify the environment creation in your script to enable debug output
env = gym.make(
    "Isaac-Ur5e-Dual-v0",
    cfg=env_cfg,
    debug_controller_data=True  # Enable debug output
)
```

This will print:
- TCP position and orientation
- Controller orientation
- Rotation difference in Euler angles (XYZ degrees)

### Disable Visualization

If you want to disable the controller frame visualization:

```python
env = gym.make(
    "Isaac-Ur5e-Dual-v0",
    cfg=env_cfg,
    enable_controller_viz=False  # Disable visualization
)
```

## Files Changed

1. **`ur5e_dual_manipulation_env.py`** (NEW): Custom environment with visualization
2. **`__init__.py`**: Updated to use the custom environment class
3. **`ur5e_dual_manipulation_env_cfg.py`**: No changes needed

## Tuning the Controller Offset

Once you can see both frames, follow these steps:

1. **Observe the difference**: Move the controllers and watch how the two frames relate on each arm
2. **Note the axes**: Identify which axes are misaligned (X=red, Y=green, Z=blue)
3. **Adjust the offset**: In `ur5e_dual_manipulation_env_cfg.py`, modify for each arm:
   ```python
   # For left arm
   left_arm_action: RMPFlowActionCfg = RMPFlowActionCfg(
       # ... other params ...
       body_offset=RMPFlowActionCfg.OffsetCfg(
           pos=(0.0, 0.0, 0.0),
           rot=(0.5, -0.5, -0.5, 0.5),  # <-- Adjust this quaternion
       ),
   )

   # For right arm
   right_arm_action: RMPFlowActionCfg = RMPFlowActionCfg(
       # ... other params ...
       body_offset=RMPFlowActionCfg.OffsetCfg(
           pos=(0.0, 0.0, 0.0),
           rot=(0.7071068, 0, -0.7071068, 0),  # <-- Adjust this quaternion
       ),
   )
   ```
4. **Test**: Restart and verify the alignment improved

## Currently Implemented

- ✅ Left controller visualization
- ✅ Right controller visualization
- ✅ Automatic world space transform
- ✅ Debug output option
- ✅ Enable/disable toggle
- ✅ Independent tracking for both controllers

## Troubleshooting

**Frame not visible?**
- Check XR mode is enabled (`--xr` flag)
- Verify controller tracking is working
- Look at the left gripper specifically

**Frame in wrong location?**
- Check that `left_control_frame` sensor is configured correctly
- Verify TCP link exists in your robot USD

**Orientation seems wrong?**
- Check XR anchor configuration in env_cfg.py
- Verify `xr.anchor_pos` and `xr.anchor_rot` are correct

## Implementation Details

The visualization works by:
1. Querying XRCore directly for controller pose (`get_aim_pose()`)
2. Transforming from XR space to world space using the anchor transform
3. Placing the coordinate frame marker at the TCP position with controller orientation
4. Updating every frame in the `step()` method

For more details, see `docs/CONTROLLER_VISUALIZATION.md`
