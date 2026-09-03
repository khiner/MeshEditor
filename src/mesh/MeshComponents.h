#pragma once

#include "Range.h"

#include <entt/entity/fwd.hpp>

struct RenderInstance {
    entt::entity Entity;
    uint32_t BufferIndex{0};
    uint32_t ObjectId{0};
    uint32_t GpuId{InvalidOffset};
    uint32_t MeshletRangeCount{0};
    uint32_t MeshletCount{0};
};

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
