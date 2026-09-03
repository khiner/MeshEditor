#pragma once

#include "Range.h"

#include <entt/entity/entity.hpp>

#include <optional>
#include <unordered_map>
#include <unordered_set>

// Posed-buffer layout for one mesh entity's instance run: base offsets plus per-instance strides.
struct PosedRanges {
    // One instance's derived normals, absent for a mesh with no faces to derive them from.
    struct NormalRanges {
        uint32_t VertexOffset{InvalidOffset}, SeamOffset{InvalidOffset}, FaceOffset{InvalidOffset};
        uint32_t SeamCount{0}, FaceCount{0};
    };

    uint32_t FirstInstance{0};
    bool PerInstance{false};
    uint32_t PositionBase{InvalidOffset};
    uint32_t VertexCount{0};
    uint32_t MeshletBoundsBase{InvalidOffset};
    uint32_t Level0Count{0};
    std::optional<NormalRanges> Normals{};

    uint32_t PositionOffset(uint32_t i) const { return PositionBase + i * VertexCount; }
    uint32_t MeshletBoundsOffset(uint32_t i) const { return MeshletBoundsBase + i * Level0Count; }
    std::optional<NormalRanges> NormalsAt(uint32_t i) const {
        if (!Normals) return std::nullopt;
        return NormalRanges{
            Normals->VertexOffset + i * VertexCount,
            Normals->SeamOffset + i * Normals->SeamCount,
            Normals->FaceOffset + i * Normals->FaceCount,
            Normals->SeamCount,
            Normals->FaceCount,
        };
    }
};

// Host metadata for the persistent GPU scene, refreshed when scene structure or routing changes.
struct GpuSceneState {
    std::unordered_map<entt::entity, PosedRanges> PosedByEntity;
    std::unordered_set<entt::entity> MeshletEditOverlayMeshes;
    bool MeshletEditHasSharpEdges{};
    bool InstanceRecordsStale{true};
    bool InstanceFlagsStale{true};
    uint64_t InstanceRecordInputs{0};
};

// Mark every instance record for a rewrite, for a change the record-input signature does not see.
inline void MarkInstanceRecordsStale(GpuSceneState &scene) {
    scene.InstanceRecordsStale = true;
    scene.InstanceFlagsStale = true;
}
