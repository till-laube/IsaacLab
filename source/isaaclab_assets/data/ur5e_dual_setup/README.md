# UR5e Dual Manipulation Setup - STL Files

This directory contains STL files for the dual UR5e setup.

## Existing/Required Files (TODO: add camera mounts and cameras)

1. **metal_frame.stl** - The metal frame base structure
2. **wooden_plate.stl** - The wooden table/work surface
3. **camera_mount_left.stl** - Left camera mount (3D printed) + camera
4. **camera_mount_right.stl** - Right camera mount (3D printed) + camera

### Update path for additional .stls in:

`/home/till/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/ur5e_dual_manipulation/ur5e_dual_manipulation_env_cfg.py`

Search for `"/PATH/TO/"` and replace with:

```python
mesh_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/metal_frame.stl"
mesh_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/wooden_plate.stl"
mesh_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/camera_mount_left.stl"
mesh_path="/home/till/IsaacLab/source/isaaclab_assets/data/ur5e_dual_setup/camera_mount_right.stl"
```

## STL File Requirements

- Files should be in ASCII or binary STL format
- Ensure the scale is correct (meters in Isaac Sim)
- Origin should be at a logical reference point (e.g., bottom center for frame)
- Normals should face outward for proper collision detection

## Testing

After setting up the files, test the environment with:

