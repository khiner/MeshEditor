#pragma once

#include "Range.h"
#include "gpu/ElementWork.h"

#include "state/Entity.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Pose namespaces and normal outputs for one mesh entity's instance run.
struct PosedNamespaces {
    struct NormalNamespaces {
        uint32_t Vertex{InvalidOffset}, Sector{InvalidOffset}, Face{InvalidOffset};
        bool operator==(const NormalNamespaces &) const = default;
    };
    uint32_t FirstInstance{};
    bool PerInstance{};
    std::vector<uint32_t> PositionNamespaces, MorphNormalNamespaces, MeshletBoundsNamespaces, VertexBoundsNamespaces;
    std::vector<NormalNamespaces> Normals;
    bool operator==(const PosedNamespaces &) const = default;

    uint32_t PositionNamespace(uint32_t i) const { return PositionNamespaces.at(i); }
    uint32_t MorphNormalNamespace(uint32_t i) const { return MorphNormalNamespaces.empty() ? InvalidOffset : MorphNormalNamespaces.at(i); }
    uint32_t MeshletBoundsNamespace(uint32_t i) const { return MeshletBoundsNamespaces.at(i); }
    uint32_t VertexBoundsNamespace(uint32_t i) const { return VertexBoundsNamespaces.at(i); }
    std::optional<NormalNamespaces> NormalsAt(uint32_t i) const {
        return Normals.empty() ? std::nullopt : std::optional{Normals.at(i)};
    }
};

// Derived work and bounds parents survive gestures until topology, shading layout, or edit mode changes.
struct MeshEditWork {
    uint32_t StoreId{InvalidOffset};
    // Canonical element handles.
    // Bounds work names canonical vertex blocks.
    ElementWork Candidates, Vertices, Faces, Normals, Meshlets, BoundsTiles;
    std::array<ElementWork,3> BoundsLevels;
    // A repeated position refresh has the same local dependency footprint.
    // Keep only its address ranges.
    // The actual work and values remain on GPU.
    std::vector<Range> RefreshRanges;
    bool CandidateReady{}, FootprintReady{}, Modified{}, PreviewActive{}, RequiresPose{};
    Range WorkBudget;
};

// An instance's overlay over its mesh's canonical vertex blocks.
enum class VertexOverlay : uint8_t {
    EditPoints,
    Points,
    SoundPoints,
    Normals,
};
struct VertexOverlayDraw {
    state::Entity MeshEntity;
    uint32_t Instance; // The instance record.
    VertexOverlay Kind;
};

// Host metadata for the persistent GPU scene, refreshed when scene structure or routing changes.
struct GpuSceneState {
    std::unordered_map<state::Entity, PosedNamespaces> PosedByEntity;
    std::unordered_map<state::Entity, MeshEditWork> EditWork;
    // Reconstructed from tracked per-mesh dirty roots after history restore.
    std::unordered_set<state::Entity> PositionDirty, LodDirty;
    // Meshes whose meshlets were built, edited or restored since their cluster hierarchy was last checked.
    std::unordered_set<state::Entity> LodDemand;
    std::unordered_set<state::Entity> MeshletEditOverlayMeshes;
    // Refreshed with the instance flags.
    std::vector<VertexOverlayDraw> VertexOverlays;
    bool MeshletEditHasSharpEdges{};
    bool EditPreludePending{};
    // Element selection bits changed, so edit work candidates reseed.
    bool EditSelectionDirty{};
    bool InstanceRecordsStale{true};
    bool InstanceFlagsStale{true};
    uint64_t InstanceRecordInputs{0};
    uint64_t PreludeLayoutInputs{0};
    uint64_t PreludeWorkInputs{0};
};

// Mark every instance record for a rewrite, for a change the record-input signature does not see.
inline void MarkInstanceRecordsStale(GpuSceneState &scene) {
    scene.InstanceRecordsStale = true;
    scene.InstanceFlagsStale = true;
}
