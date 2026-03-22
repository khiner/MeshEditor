#pragma once

#include "entt_fwd.h"
#include "vulkan/Buffer.h"

// Links a renderable instance to the entity holding its shared GPU buffers (MeshBuffers + ModelsBuffer).
struct RenderInstance {
    entt::entity Entity; // The entity this is an instance of (has MeshBuffers + ModelsBuffer).
    uint32_t BufferIndex{0}; // Slot in GPU model instance buffer
    uint32_t ObjectId{0};
};

// Stored on mesh entities.
// Holds the `WorldTransform` of all instances of the mesh.
struct ModelsBuffer {
    mvk::Buffer Buffer;
    mvk::Buffer ObjectIds; // Per-instance ObjectIds for selection/silhouette rendering.
    mvk::Buffer InstanceStates; // Per-instance selection/active state flags.
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

// MeshStore vertex buffer ID
struct VertexStoreId {
    uint32_t StoreId;
};
