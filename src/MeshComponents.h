#pragma once

#include "entt_fwd.h"
#include "vulkan/Range.h"
#include "vulkan/Slots.h"

// Links a renderable instance to the entity holding its shared GPU buffers (MeshBuffers + ModelsBuffer).
struct RenderInstance {
    entt::entity Entity; // The entity this is an instance of (has MeshBuffers + ModelsBuffer).
    uint32_t BufferIndex{0}; // Global index in shared InstanceArena buffers.
    uint32_t ObjectId{0};
};

// Stored on buffer entities. Lightweight handle into the shared InstanceArena.
struct ModelsBuffer {
    Range InstanceRange{}; // Allocated range in shared InstanceArena.
    uint32_t InstanceCount{0}; // Active instances (≤ InstanceRange.Count).
};

// Cameras, lights, empties.
struct ObjectExtrasTag {};

struct VertexClass {
    uint32_t Offset{InvalidOffset};
};

// Placed on buffer entities. Accumulates instance slots that need GPU erasure.
// Processed and cleared each frame by SyncModelsBuffers.
struct PendingHide {
    std::vector<uint32_t> BufferIndices;
};

// Placed on SceneEntity. Accumulates light buffer indices during Destroy();
// batch-compacted in ProcessComponentEvents.
struct PendingLightRemovals {
    std::vector<uint32_t> Indices;
};

// Placed on extras buffer entities at creation. Consumed by ProcessComponentEvents
// to create edge index buffers, then removed.
struct PendingEdgeIndices {
    std::vector<uint32_t> Indices;
};

// MeshStore vertex buffer ID
struct VertexStoreId {
    uint32_t StoreId;
};
