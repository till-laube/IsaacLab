# Debugging Controller Visualization

## Quick Test

Run the test script:
```bash
./scripts/test_controller_viz.sh
```

## What to Check

### 1. Marker Creation (On Startup)

Look for these messages when the environment initializes:
```
[Controller Viz] Created LEFT controller visualization marker at /Visuals/LeftControllerFrame_01
[Controller Viz] Created RIGHT controller visualization marker at /Visuals/RightControllerFrame_01
```

**If you don't see these:**
- There was an error creating the visualization markers
- Check for error messages starting with `[Controller Viz] ERROR:`
- The marker creation might have failed due to missing USD file

### 2. First Successful Update (After Controllers Are Tracked)

After the environment starts and controllers are being tracked, you should see:
```
[Controller Viz] ✓ Successfully visualizing LEFT controller pose!
[Controller Viz]   Controller Position: [x, y, z]
[Controller Viz]   You should see a LARGER coordinate frame at the controller's actual location
[Controller Viz]   Compare it with the TCP frame at the left gripper
[Controller Viz] ✓ Successfully visualizing RIGHT controller pose!
[Controller Viz]   Controller Position: [x, y, z]
[Controller Viz]   You should see a LARGER coordinate frame at the controller's actual location
[Controller Viz]   Compare it with the TCP frame at the right gripper
```

**If you don't see this:**
- Controller data is not being received
- Continue to step 3 for detailed debugging

### 3. Enable Debug Output

To see detailed debugging information, modify how you create the environment.

Edit `scripts/environments/teleoperation/teleop_se3_agent.py` and change the environment creation from:
```python
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
```

To:
```python
env = gym.make(args_cli.task, cfg=env_cfg, debug_controller_data=True).unwrapped
```

This will print detailed information every ~0.5 seconds about:
- Controller orientation
- TCP orientation
- Rotation difference

### 4. Debug Messages to Look For

**XRCore not available:**
```
[Controller Viz] XRCore not available
```
→ You're not running in XR mode. Make sure to use `--xr` flag.

**Controller device not found:**
```
[Controller Viz] Left controller device not found
```
→ The XR controller is not being detected by the system.
→ Check that your VR controllers are on and tracked.
→ Verify SteamVR/ALVR is running and controllers are visible there.

**Controller pose matrix is None:**
```
[Controller Viz] Controller pose matrix is None
```
→ The controller exists but isn't returning pose data.
→ Make sure the controller is actively tracked (visible to base stations).

**Failed to get TCP data:**
```
[Controller Viz] ERROR: Failed to get TCP data: [error message]
```
→ The frame transformer for the TCP is not configured correctly.
→ Check that `left_control_frame` exists in the scene configuration.

## Visual Checklist

When working correctly, you should see **two coordinate frames in different locations**:

1. **Small RGB coordinate frame** (scale 0.1) **at each gripper**
   - This is from `left_control_frame` / `right_control_frame` - the existing TCP visualization
   - Red = X axis, Green = Y axis, Blue = Z axis
   - Shows the actual gripper pose (position + orientation)
   - Moves with the robot arm

2. **Larger RGB coordinate frame** (scale 0.15) **at controller location in world space**
   - This is the NEW controller visualization
   - Red = X axis, Green = Y axis, Blue = Z axis
   - Shows the controller's actual pose in the virtual world
   - Should move and rotate when you move/rotate the physical controller
   - Will be in a **different location** from the gripper (not at the same position!)

**The spatial relationship between the two frames** shows:
- **Position offset**: The vector from controller position to TCP position
- **Rotation offset**: The orientation difference between controller and TCP
- **Full transformation**: How the retargeter maps your hand movement to robot movement

You can compare both left and right controllers independently to tune each arm's offset.

## Common Issues

### Visualization is There But Wrong

If the frame appears but doesn't match controller movement:

1. **Check which frame is which:**
   - Smaller frame = TCP (should match gripper)
   - Larger frame = Controller (should match your hand)

2. **Verify controller tracking:**
   - Test in SteamVR to ensure tracking is working
   - Make sure there's no occlusion

3. **Check coordinate transform:**
   - The controller orientation should be in world space
   - If it seems inverted or rotated wrong, the XR anchor might be misconfigured

### Frame Disappears

If the frame was visible but disappears:

1. **Controller lost tracking:**
   - The frame is hidden at y=-100 when no controller data is available
   - Check VR tracking

2. **Environment reset:**
   - Frame should reappear after reset

## Manual Verification

You can manually verify the visualization marker exists in the USD stage:

1. Open Isaac Sim with your scene loaded
2. Look in the Stage panel for `/Visuals/LeftControllerFrame_01` (or similar)
3. The marker should be a PointInstancer with a frame prototype
4. You can manually move it in the viewport to verify it's visible

## Implementation Details

The visualization works through these steps:

1. **Initialization (`__init__`):**
   - Creates `VisualizationMarkers` instance
   - Sets up frame prototype from `frame_prim.usd`

2. **Each Step (`step()`):**
   - Calls `_update_controller_visualizations()`
   - Gets TCP position from `left_control_frame` sensor
   - Gets controller orientation from XRCore
   - Updates marker position and orientation

3. **Controller Data (`_get_left_controller_orientation()`):**
   - Queries `XRCore.get_singleton().get_input_device("/user/hand/left")`
   - Calls `get_virtual_world_pose()` to get the pose matrix
   - Extracts quaternion from the matrix
   - Returns orientation in world space

If any step fails, the marker is hidden by moving it to y=-100.

## Next Steps

Once visualization is working:

1. **Observe the orientation difference** between the two frames
2. **Note which axes are misaligned**
3. **Calculate the rotation offset** needed
4. **Update `body_offset.rot`** in `ur5e_dual_manipulation_env_cfg.py`
5. **Test and iterate** until alignment is good

## Need More Help?

If you're still having issues:

1. Check that you can run the task normally without visualization
2. Verify VR controllers work in other SteamVR applications
3. Make sure you're using `--xr` flag when launching
4. Enable debug output and share the console output
5. Check the Isaac Sim Stage panel to see if the marker prim exists
