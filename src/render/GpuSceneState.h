#pragma once

#include "Range.h"
#include "gpu/ElementWork.h"

#include <entt/entity/entity.hpp>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Posed-buffer layout for one mesh entity's instance run: base offsets plus per-instance strides.
struct PosedRanges {
    // One instance's derived normals, absent for a mesh with no faces to derive them from.
    struct NormalRanges {
        uint32_t VertexOffset{InvalidOffset}, SeamOffset{InvalidOffset}, FaceOffset{InvalidOffset};
        uint32_t SeamCount{0}, FaceCount{0};
        bool operator==(const NormalRanges &) const = default;
    };

    uint32_t FirstInstance{0};
    bool PerInstance{false};
    uint32_t PositionBase{InvalidOffset};
    uint32_t VertexCount{0};
    uint32_t MeshletBoundsBase{InvalidOffset};
    uint32_t Level0Count{0};
    std::optional<NormalRanges> Normals{};

    bool operator==(const PosedRanges &) const = default;

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

// Derived work and bounds parents survive gestures until topology, shading layout, or edit mode changes.
struct MeshEditWork {
    uint32_t StoreId{InvalidOffset};
    ElementWork Candidates, Vertices, Faces, Normals, Meshlets, BoundsTiles;
    struct BoundsLevel {
        ElementWork Work;
        Range Values;
    };
    std::vector<BoundsLevel> BoundsLevels;
    bool CandidateReady{}, Modified{}, PreviewActive{}, BoundsInitialized{};
    Range ElementMeshlets;
};

// Host metadata for the persistent GPU scene, refreshed when scene structure or routing changes.
struct GpuSceneState {
    std::unordered_map<entt::entity, PosedRanges> PosedByEntity;
    std::unordered_map<entt::entity, MeshEditWork> EditWork;
    std::unordered_set<entt::entity> MeshletEditOverlayMeshes;
    bool MeshletEditHasSharpEdges{};
    bool EditPreludePending{};
    bool InstanceRecordsStale{true};
    bool InstanceFlagsStale{true};
    uint64_t InstanceRecordInputs{0};
    uint64_t PreludeLayoutInputs{0};
};

// Mark every instance record for a rewrite, for a change the record-input signature does not see.
inline void MarkInstanceRecordsStale(GpuSceneState &scene) {
    scene.InstanceRecordsStale = true;
    scene.InstanceFlagsStale = true;
}
