#pragma once

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <vector>

enum class RenderRequest : uint8_t {
    None,
    Submit,
    ReRecordSilhouette, // Only silhouette batch + command buffer
    ReRecord, // Full draw list rebuild
};

struct SyncResult {
    std::vector<entt::entity> NewlyInserted; // Entities inserted into GPU buffers — callers must write their WorldTransform before submit.
    std::vector<entt::entity> NewMeshEntities; // Mesh entities needing deferred index buffer creation.
    std::vector<entt::entity> NewExtrasEntities; // Non-mesh buffer entities (extras/bone/joint) needing deferred index creation.
};
