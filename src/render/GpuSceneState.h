#pragma once

#include "Range.h"
#include "gpu/ElementWork.h"

#include "state/Entity.h"

#include <array>
#include <map>
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
    std::array<ElementWork, 3> BoundsLevels;
    // A repeated position refresh has the same local dependency footprint.
    // Keep only its address ranges.
    // The actual work and values remain on GPU.
    std::vector<Range> RefreshRanges;
    bool CandidateReady{}, FootprintReady{}, Modified{}, PreviewActive{}, RequiresPose{}, TessellationPreview{};
    Range WorkBudget;
};

// An overlay over a mesh's canonical vertex blocks, a bit in its mesh's VertexOverlays mask.
enum class VertexOverlay : uint8_t {
    EditPoints, // On the mesh's primary edit instance.
    Points,
    SoundPoints,
    Normals,
};

// Host metadata for the persistent GPU scene, refreshed when scene structure or routing changes.
struct GpuSceneState {
    std::unordered_map<state::Entity, PosedNamespaces> PosedByEntity;
    std::unordered_map<state::Entity, MeshEditWork> EditWork;
    // Reconstructed from tracked per-mesh dirty roots after history restore.
    std::unordered_set<state::Entity> PositionDirty, LodDirty;
    // Meshes whose meshlets were built, edited or restored since their cluster hierarchy was last checked.
    std::unordered_set<state::Entity> LodDemand;
    // Meshes whose record's display fields rederive at the end of the settle pass, after their render data changed.
    std::unordered_set<state::Entity> DisplayDirty;
    // Each material's required LOD attributes without and with authored tangents, as the live primitives hold them.
    std::vector<std::array<uint32_t, 2>> RequiredMaterialAttributes;
    // The vertex overlays each mesh's instances draw, as masks of VertexOverlay bits, in mesh order.
    std::map<state::Entity, uint8_t> VertexOverlays;
    // Each laid-out mesh's run of bounds entries, which a mesh without per-instance deformation keeps across instance changes.
    struct BoundsRun {
        uint32_t First{}, Count{};
        bool Posed{};
    };
    std::unordered_map<state::Entity, BoundsRun> BoundsRuns;
    // Each bounds entry's tiles in every prelude pass, by entry.
    struct EntryTiles {
        std::array<Range, 4> Bounds{}; // By vertex bounds level.
        Range DeriveFaces{}, DeriveGather{}, MeshletJobs{};
    };
    std::vector<EntryTiles> BoundsEntryTiles;
    // Entries whose bounds the next submit recomputes apart from a full prelude, in any order with repeats.
    std::vector<uint32_t> DirtyBoundsEntries;
    // The entries morph weights pose, and the entries each armature data entity's skins pose.
    std::vector<uint32_t> MorphEntries;
    std::unordered_map<state::Entity, std::vector<uint32_t>> ArmatureEntries;
    // The OR of every mesh's PBR features.
    uint32_t MeshPbrFeatures{0};
    // The display settings the last layout rebuild read.
    uint64_t LayoutDisplayInputs{0};
    bool MeshletEditHasSharpEdges{};
    bool EditPreludePending{};
    // Element selection bits changed, so edit work candidates reseed.
    bool EditSelectionDirty{};
    // Camera lenses, lights, colliders, tets or instance slots changed, so the overlay jobs rebuild before the next record.
    bool OverlayJobsDirty{true};
    uint64_t PreludeLayoutInputs{0};
    uint64_t PreludeWorkInputs{0};
};
