#pragma once

#include "Range.h"

#include <entt/entity/fwd.hpp>

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

struct VertexClass {
    uint32_t Offset{InvalidOffset};
};

// MeshStore vertex buffer ID
struct VertexStoreId {
    uint32_t StoreId;
};

// MeshStore vertex buffer ID into the overlay store. Present on a buffer entity whose geometry is derived, not authored.
struct OverlayVertexStoreId {
    uint32_t StoreId;
};

// The canonical handle to a mesh's vertex data in MeshStore.
struct MeshHandle {
    uint32_t StoreId{~0u};
};

// Cameras, lights, empties
struct ObjectExtrasTag {};

// Presence on a mesh entity: interpolated vertex normals
// Absence: per-face normal computed in the vertex shader
struct SmoothShading {};
