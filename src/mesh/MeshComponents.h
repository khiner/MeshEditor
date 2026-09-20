#pragma once

#include "Range.h"
#include "numeric/vec3.h"

#include "state/Entity.h"

#include <vector>

struct RenderInstance {
    state::Entity Entity;
    uint32_t BufferIndex{0};
    uint32_t MeshletRangeCount{0};
    uint32_t MeshletCount{0};
};
// GPU object IDs are the entity slot plus one. Zero remains the background.
constexpr uint32_t ObjectId(state::Entity e) { return state::Index(e) + 1; }

struct ModelsBuffer {
    Range InstanceRange{};
    uint32_t InstanceCount{0};
};

struct VertexStoreId {
    uint32_t StoreId;
};

struct MeshHandle {
    uint32_t StoreId{~0u};
};

// The record a staged operator produced over the entity's mesh, drawn in its place until the gesture commits or restores.
struct MeshPreview {
    uint32_t StoreId;
};

// Derived from the canonical per-face sharpness store after construction or a shading edit.
struct MeshShadingSummary {
    bool AnySharp{}, AllSharp{};
};

// A new triangle mesh's authored normals in fan order, held until its base normals derive and the custom corner-normal layer encodes them.
struct AuthoredCornerNormals {
    std::vector<vec3> Corners;
};

struct ObjectExtrasTag {};
