# Plan: Replace MeshInstance with universal InstanceOf

## Problem
`MeshInstance` conflates "has buffer entity linkage" with "is a mesh instance." Extras (cameras/lights/empties) have `MeshInstance` despite not being mesh objects. Bones use `SubElementOf.Parent` instead. The relationship is universal but the component isn't.

## Solution
One component `InstanceOf { entt::entity Entity; }` on ALL instance entities. Mesh-specific code checks `R.all_of<Mesh>(io.Entity)` instead of gating on a separate component.

## Changes

### src/MeshInstance.h → src/InstanceOf.h (rename + rewrite)
```cpp
struct InstanceOf {
    entt::entity Entity;
};
```

### Includes (4 files)
`#include "MeshInstance.h"` → `#include "InstanceOf.h"` in Scene.cpp, SceneUi.cpp, SceneGltf.cpp, SceneSelection.cpp

### Scene.cpp — emplace sites
- `AddMeshInstance` (1558): `MeshInstance` → `InstanceOf`
- `CreateExtrasObject` (2083): `MeshInstance` → `InstanceOf`
- `CreateBoneInstances` (~1650): **Add** `R.emplace<InstanceOf>(bone_entity, arm_obj_entity)`
- `CreateBoneInstances` (~1703-1717): **Add** `R.emplace<InstanceOf>(head/tail_entity, joint_entity)`
- `CreateSingleBoneInstance` (~1814): **Add** `R.emplace<InstanceOf>(bone_entity, arm_obj_entity)`
- `CreateSingleBoneInstance` (~1838-1850): **Add** `R.emplace<InstanceOf>(head/tail_entity, arm_obj.JointEntity)`
- `DuplicateLinked` (2320): `MeshInstance` → `InstanceOf`

### Scene.cpp — SetVisible show path
`R.try_get<MeshInstance>` + `SubElementOf` fallback → `R.get<InstanceOf>(entity).Entity`

### Scene.cpp — mechanical renames (MeshInstance→InstanceOf, MeshEntity→Entity)
Lines: 324, 385, 401, 424, 901, 947, 950, 1030, 1108, 1458, 2112, 2258, 2271, 2311, 2315, 2379, 2417-2418, 2463-2465, 2518-2519, 2529, 2539, 2554, 2582, 2607, 3130, 3251, 3542

### Scene.cpp — sites needing Mesh gate (used MeshInstance to mean "is a mesh instance")
- `GetMeshEntity` (724): return null if `!R.all_of<Mesh>(io->Entity)`
- `HasSelectedInstance` (424): only count if `R.all_of<Mesh>(io.Entity)`
- Selected tracker (837): only dirty overlays if `R.all_of<Mesh>(io->Entity)`
- `Destroy` (2417): only track mesh_entity for cleanup if `R.all_of<Mesh>(buffer)`
- `ClearMeshes` (2379): only collect top-level instances (`!R.all_of<SubElementOf>(e)`)

### SceneSelection.cpp — renames + Mesh gates
- `ComputePrimaryEditInstances` (105): rename (ObjectKind filter already gates)
- `HasFrozenInstance` (114): rename
- `GetSelectedMeshEntities` (122): rename + add `R.all_of<Mesh>(io.Entity)` gate

### SceneUi.cpp — renames + Mesh gates
- Entity controls (1185): rename, gate vertex/edge/face count on Mesh
- Material panel: already gated by `if (active_mesh_instance)` → `if (io && R.all_of<Mesh>(io->Entity))`
- Visibility toggle (1713): rename, filter out `SubElementOf` entities
- `has_transform_target` (880): rename + Mesh gate for MeshSelectionBitsetRange check

### SceneGltf.cpp — mechanical renames
Lines 305, 355, 397, 445: `R.all_of<MeshInstance>` → `R.all_of<InstanceOf>`
