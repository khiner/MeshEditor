#pragma once

#include "gpu/PBRMaterial.h"
#include "gpu/Transform.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <cstdint>

// Per-mesh offset/count into SelectionBitsetBuffer for the current edit element type.
// Assigned on Edit mode entry; updated on element type switch and mesh topology change.
struct MeshSelectionBitsetRange {
    uint32_t Offset; // Start bit index in SelectionBitsetBuffer
    uint32_t Count; // Element count for current edit mode
};

// Tag to request overlay + element-state buffer refresh after mesh geometry changes.
struct MeshGeometryDirty {};
struct MeshMaterialAssignment {
    uint32_t PrimitiveIndex, MaterialIndex;
};
struct MeshMaterialSlotSelection {
    uint32_t PrimitiveIndex{0};
};

// Generic tag for events that only require command buffer submission (not re-record).
struct SubmitDirty {};
struct LightWireframeDirty {};
struct MaterialDirty {
    uint32_t Index{0};
};

struct LightIndex {
    uint32_t Value{0}; // Index into SceneBuffers::LightBuffer (PunctualLight[])
};

// Tracks pending transform for shader-based preview during Edit mode gizmo manipulation.
// Presence indicates active transform; removal triggers UBO clear.
struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    Transform Delta{};
};