import sys
from pxr import Usd, UsdPhysics, UsdGeom

def apply_physics_to_usd(usd_path, prim_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Error: Could not open stage at {usd_path}")
        return

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: Prim not found at path {prim_path}")
        return

    # Apply RigidBodyAPI to the target prim
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        print(f"Applied UsdPhysics.RigidBodyAPI to {prim_path}")
    # Apply CollisionAPI to child meshes if they exist (common for STL conversions)
    meshes_found = False
    for child_prim in prim.GetChildren():
        if child_prim.GetTypeName() == "Mesh":
            if not child_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(child_prim)
                # Optional: set approximation if needed, default is often good
                # UsdPhysics.MeshCollisionAPI.Apply(child_prim).GetApproximationAttr().Set("convexHull")
                print(f"Applied UsdPhysics.CollisionAPI to child mesh {child_prim.GetPath()}")
                meshes_found = True

    if not meshes_found and prim.GetTypeName() == "Mesh":
         if not prim.HasAPI(UsdPhysics.CollisionAPI):
             UsdPhysics.CollisionAPI.Apply(prim)
             print(f"Applied UsdPhysics.CollisionAPI to the mesh prim itself {prim_path}")


    stage.Save()
    print(f"Saved modified USD to {usd_path}")
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python apply_physics.py <path_to_usd_file> <prim_path_to_apply_rigidbody>")
        sys.exit(1)

    usd_file = sys.argv[1]
    target_prim = sys.argv[2]
    apply_physics_to_usd(usd_file, target_prim)

