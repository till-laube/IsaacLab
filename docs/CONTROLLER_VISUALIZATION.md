# Controller Coordinate System Visualization

## Overview

The Isaac-Ur5e-Dual-v0 environment now includes visualization of the VR controller's coordinate system at the gripper TCP (Tool Center Point). This helps you:

1. **See the controller orientation** vs. the end-effector orientation
2. **Tune the retargeter offset** by comparing the two coordinate frames
3. **Debug controller mapping issues** by visualizing the exact controller pose

## What You'll See

When running the environment in XR mode, you'll see two coordinate frames at each gripper:

1. **Small RGB frame** (scale 0.1): The gripper's TCP orientation (existing visualization)
2. **Larger RGB frame** (scale 0.15): The controller's orientation at the TCP location (NEW!)

The controller frame shows the **raw controller orientation** transformed to world space, while the TCP frame shows the **gripper's actual orientation**. The difference between these two frames represents the offset/mapping applied by the retargeter.

## How It Works

The custom environment (`Ur5eDualManipulationEnv`) extends the base environment with:

1. **VisualizationMarkers**: Creates a coordinate frame marker for the controller
2. **XRCore Integration**: Directly queries the controller pose from OpenXR
3. **Transform Pipeline**: Applies the XR anchor transform to convert from controller space to world space
4. **Real-time Update**: Updates the visualization every frame in `step()`

## Technical Details

### Controller Data Flow

```
VR Controller (HTC Vive)
    ↓
OpenXR Runtime (SteamVR via ALVR)
    ↓
XRCore.get_input_device("/user/hand/left").get_aim_pose()
    ↓
XR Anchor Transform (anchor_pos, anchor_rot)
    ↓
World Space Controller Orientation
    ↓
Visualization at TCP Position
```

### Coordinate Frames

- **XR Space**: Controller's raw pose from OpenXR (relative to headset)
- **World Space**: Simulation world coordinates
- **TCP Frame**: The gripper's tool center point orientation

### Current Implementation (Left Controller Only)

The current implementation visualizes only the **left controller** to keep it simple. To add the right controller visualization:

1. Uncomment the right controller marker creation in `_create_controller_visualizations()`
2. Add `_get_right_controller_orientation()` method
3. Update `_update_controller_visualizations()` to handle right controller

## Usage

Run the teleoperation script as usual:

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Ur5e-Dual-v0 \
    --teleop_device vive \
    --xr
```

The controller coordinate frame will automatically appear at the left gripper TCP when:
- XR mode is active
- Left controller is tracked
- Controller data is available

## Tuning the Offset

To tune the controller-to-gripper offset mapping:

1. **Compare the frames**: Look at the difference between the controller frame (large) and TCP frame (small)
2. **Identify the rotation difference**: Note which axes are misaligned
3. **Update the offset**: Modify the `body_offset` quaternion in `ActionsCfg.left_arm_action`
   ```python
   body_offset=RMPFlowActionCfg.OffsetCfg(
       pos=(0.0, 0.0, 0.0),
       rot=(0.5, -0.5, -0.5, 0.5),  # Adjust this quaternion
   )
   ```
4. **Test**: Run again and verify the alignment is better

### Quaternion Tips

The quaternion format is `(qw, qx, qy, qz)`. Common rotations:

- 90° around X: `(0.7071, 0.7071, 0, 0)`
- 90° around Y: `(0.7071, 0, 0.7071, 0)`
- 90° around Z: `(0.7071, 0, 0, 0.7071)`
- 180° around X: `(0, 1, 0, 0)`

## Troubleshooting

### Frame Not Visible

If you don't see the controller frame:

1. Check XR mode is enabled (`--xr` flag)
2. Verify controller is tracked (check console for controller data)
3. Make sure you're looking at the left gripper
4. Check the frame isn't hidden (it moves to y=-100 when no data)

### Frame in Wrong Location

If the frame is not at the TCP:

1. Verify `left_control_frame` sensor is configured correctly
2. Check that the TCP link (`tcp_link`) exists in your robot USD
3. Ensure the sensor is being updated each frame

### Frame Has Wrong Orientation

If the controller frame doesn't match controller movement:

1. Check XR anchor configuration (`xr.anchor_pos`, `xr.anchor_rot`)
2. Verify the anchor transform is correct for your setup
3. Test controller tracking in SteamVR to rule out tracking issues

## Files Modified

- `ur5e_dual_manipulation_env.py`: New custom environment with visualization
- `__init__.py`: Updated to use custom environment class
- Environment config remains unchanged

## Future Enhancements

Possible improvements:

1. **Color coding**: Use different colors for controller vs. TCP frames
2. **Both controllers**: Visualize left and right simultaneously
3. **Delta visualization**: Show the rotation difference as an arrow
4. **Debug overlay**: Display numerical orientation difference in UI
5. **Recording**: Log orientation differences for offline analysis
