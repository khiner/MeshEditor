#pragma once

#include "Range.h"

#include "state/Entity.h"

struct RenderInstance {
    state::Entity Entity;
    uint32_t BufferIndex{0};
};
// GPU object IDs are the entity slot plus one. Zero remains the background.
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

struct MeshHandle {
    uint32_t StoreId{~0u};
};
// The record a mesh entity's geometry procedure evaluates to, which its instances draw.
struct EvaluatedMesh {
    uint32_t StoreId;
    uint32_t Source{InvalidOffset}; // The source record the evaluated record copies, or InvalidOffset for a record made otherwise
};
// Persistent on a mesh entity whose evaluated record copies its source, so the next cook restores only the blocks either record wrote.
// It restores with the record it describes, so a cook after a restore takes the path a replay of the same actions takes.
// The evaluated record and its source share local block indices, so a written local block names both records' blocks.
struct ResyncState {
    std::array<std::vector<uint32_t>, 6> Written{}; // The evaluated record's local blocks the last cook wrote
    std::vector<uint32_t> Attributes{}; // The evaluated record's attributes holding their source attribute's copy
    uint32_t Resyncs{}; // The cooks that restored the record since it was copied
};

// A new mesh's authored normals in polygon-corner order, held until its base normals derive and the custom corner-normal layer encodes them.
struct AuthoredCornerNormals {
    std::vector<vec3> Corners;
};

struct ObjectExtrasTag {};
