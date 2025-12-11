# HTC Vive VR Mode with Isaac Lab (OpenXR Solution)

## How It Works

### Data Flow
```
SteamVR Runtime
    ↓ (OpenXR)
Isaac Sim XR Core
    ↓ (motion controller data)
OpenXRDevice._get_raw_data()
    ↓ (2D array: [pose(7), inputs(7)])
ViveControllerSe3Retargeter
    ↓ (converts to Se3 format)
[pos(3), rot_vec(3), gripper(1)]
    ↓
RMPFlow Action
```

### Controller Data Format

OpenXR provides controller data as a 2D array:
- **Row 0 (POSE)**: `[x, y, z, qw, qx, qy, qz]` - position + quaternion
- **Row 1 (INPUTS)**: `[trigger, squeeze, thumbstick_x, thumbstick_y, ...]` - buttons/axes

The retargeter converts this to Se3 format:
- Position: 3 elements (with sensitivity scaling)
- Rotation: 3 elements (quaternion → rotation vector)
- Gripper: 1 element (trigger > threshold → close)

## Usage

### Single-Arm VR Teleoperation

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0 \
    --teleop_device vive
```

**Result:**
- Environment renders in VR headset
- Left Vive controller controls left arm
- Trigger controls gripper (open/close)

### Dual-Arm VR Teleoperation

The dual-arm task configuration still needs updating to use the OpenXR-based approach. To add dual-arm support:

1. Import the dual-arm retargeter in `stack_rmp_dual_arm_env_cfg.py`:
```python
from isaaclab.devices.openxr.retargeters import ViveControllerDualArmRetargeterCfg
```

2. Replace the Vive device config with:
```python
"vive": OpenXRDeviceCfg(
    retargeters=[
        ViveControllerDualArmRetargeterCfg(
            pos_sensitivity=0.05,
            rot_sensitivity=0.05,
        ),
    ],
    sim_device=self.sim.device,
    xr_cfg=self.xr,
),
```

3. Add a custom retargeter splitter that takes the 14 DOF output and splits it for the two arms.

## Configuration Options

### ViveControllerSe3RetargeterCfg

```python
ViveControllerSe3RetargeterCfg(
    hand_side="left",  # or "right"
    pos_sensitivity=0.05,  # position multiplier
    rot_sensitivity=0.05,  # rotation multiplier
    trigger_threshold=0.5,  # trigger value (0.0-1.0) to close gripper
)
```

### ViveControllerDualArmRetargeterCfg

```python
ViveControllerDualArmRetargeterCfg(
    pos_sensitivity=0.05,
    rot_sensitivity=0.05,
    trigger_threshold=0.5,
)
```

## Requirements

1. **SteamVR** must be running before launching Isaac Sim
2. **OpenXR runtime** set to SteamVR (check `~/.config/openxr/1/active_runtime.json`)
3. **Controllers** paired and tracked in SteamVR
4. **Isaac Sim XR extension** enabled (should be automatic with `--xr` flag)

## Troubleshooting

### No controller data received
- Check SteamVR is running: `ps aux | grep steamvr`
- Verify controllers are tracked in SteamVR overlay
- Check OpenXR runtime: `cat ~/.config/openxr/1/active_runtime.json`

### VR view not showing
- Ensure `--xr` flag is passed (automatic with `--teleop_device vive`)
- Check headset is detected in SteamVR
- Try restarting SteamVR

### Controllers work but no VR view
- XR mode may not be enabled
- Check for XR-related errors in console
- Verify XR extension is loaded

## Files Modified/Created

### New Files:
1. `source/isaaclab/isaaclab/devices/openxr/retargeters/manipulator/vive_controller_retargeter.py`
2. `VIVE_VR_OPENXR_SETUP.md` (this file)

### Modified Files:
1. `source/isaaclab/isaaclab/devices/openxr/retargeters/manipulator/__init__.py`
2. `source/isaaclab/isaaclab/devices/openxr/retargeters/__init__.py`
3. `source/isaaclab_tasks/.../stack/config/galbot/stack_rmp_rel_env_cfg.py`
4. `scripts/environments/teleoperation/teleop_se3_agent.py`

## Next Steps

1. Test single-arm VR teleoperation
2. Adjust sensitivity parameters based on feel
3. Collect demonstration data
4. Extend to dual-arm if needed
