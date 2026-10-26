#pragma once

#include "Range.h"

#include "mesh/MeshSchema.h"
#include "state/Entity.h"

struct RenderInstance {
    state::Entity Entity;
    uint32_t BufferIndex{0};
};
// GPU object IDs are the entity slot plus one.
// Zero remains the background.
constexpr uint32_t ObjectId(state::Entity e) { return state::Index(e) + 1; }
// The entity index an ObjectId names.
constexpr uint32_t ObjectIndex(uint32_t object_id) { return object_id - 1u; }

struct ModelsBuffer {
    Range InstanceRange{};
    uint32_t InstanceCount{0};
};

struct VertexStoreId {
    uint32_t StoreId;
};

// The record a procedure's instances place, held by its owned mesh entity.
struct MeshHandle {
    uint32_t StoreId{~0u};
};
// The record a mesh entity's geometry procedure evaluates to, which its instances draw and its display path edits in place.
struct EvaluatedMesh {
    uint32_t StoreId;
    state::Entity Tip{state::Null}; // The last display-path node whose output the record holds, or null when the next cook evaluates every node
};
// A new mesh's authored normals in polygon-corner order, held until its base normals derive and the custom corner-normal layer encodes them.
struct AuthoredCornerNormals {
    std::vector<vec3> Corners;
};

struct ObjectExtrasTag {};
