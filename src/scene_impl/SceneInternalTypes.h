#pragma once

#include "gpu/PBRMaterial.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <cstdint>
#include <unordered_set>

struct MeshSelection {
    std::unordered_set<uint32_t> Handles{};
};

// Most recently selected element per mesh (remembered even when not selected).
struct MeshActiveElement {
    uint32_t Handle;
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

struct LightIndex {
    uint32_t Value{0}; // Index into SceneBuffers::LightBuffer (PunctualLight[])
};

// Tracks pending transform for shader-based preview during Edit mode gizmo manipulation.
// Presence indicates active transform; removal triggers UBO clear.
struct PendingTransform {
    vec3 Pivot{};
    quat PivotR{1, 0, 0, 0};
    vec3 P{}; // Translation delta
    quat R{1, 0, 0, 0}; // Rotation delta
    vec3 S{1, 1, 1}; // Scale delta
};

struct MaterialEdit {
    uint32_t Index{0};
    PBRMaterial Value{};
};
