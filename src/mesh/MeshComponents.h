#pragma once

#include "Range.h"

#include "state/Entity.h"

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

// Derived from the canonical per-face sharpness store after construction or a shading edit.
struct MeshShadingSummary {
    bool AnySharp{}, AllSharp{};
};

struct ObjectExtrasTag {};
