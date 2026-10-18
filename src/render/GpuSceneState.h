#pragma once

#include "Range.h"
#include "gpu/ElementWork.h"
#include "state/Entity.h"

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

// The pose namespaces and deformation of a mesh whose instances lay out posed.
struct PosedLayout {
    PosedNamespaces Namespaces;
    // The deform offsets the record display fields name, InvalidOffset without that deformation.
    uint32_t BoneDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset}, MorphTargetCount{0};
    // The armature data entities whose poses deform the mesh.
    std::vector<state::Entity> Armatures;
};

// A mesh's derived edit work and bounds parents, which survive gestures until topology, shading layout, or edit mode changes.
// Its destroy handler releases its geometry work ranges.
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

// An overlay over a mesh's canonical vertex blocks, a bit in its mesh's MeshVertexOverlays mask.
enum class VertexOverlay : uint8_t {
    EditPoints, // On the mesh's primary edit instance.
    Points,
    SoundPoints,
    Normals,
};

// The vertex overlays a mesh's instances draw, as a nonzero mask of VertexOverlay bits.
struct MeshVertexOverlays {
    uint8_t Mask;
};

// Store bounds, normal-derive, and prelude ranges for one mesh.
// Static runs use one entry and no tile words.
// Posed words contain bounds keys, derive blocks, then finest-node members and meshlet counts.
// Destruction queues range release and frees pose namespaces.
struct BoundsRun {
    Range Entries, DeriveEntries, Tiles;
    std::array<uint32_t, 4> BoundsKeys{}; // By vertex bounds level.
    uint32_t FaceBlocks{}, VertexBlocks{}, MeshletJobs{};
    // A posed run's words contain at least its root key.
    bool Posed() const { return Tiles.Count != 0u; }
};

// A mesh whose meshlets contain position-dirty bits, which the next settle outside edit mode stages.
struct MeshletPositionsDirty {};
// A mesh whose cluster groups contain dirty bits, which the next settle outside edit mode repairs.
struct ClusterLodDirty {};
// A mesh whose meshlets were built, edited or restored since its cluster hierarchy was last checked.
struct ClusterLodDemand {};
// A mesh whose record display fields rederive at the end of the settle pass, after its render data changed.
struct MeshDisplayDirty {};
// A mesh whose bounds entries, prelude tiles and pose namespaces lay out again at the end of the settle pass, after its own inputs changed.
struct MeshLayoutDirty {};
// A laid-out mesh whose bounds entries the next recorded prelude recomputes.
struct MeshBoundsDirty {};

// Host metadata for the persistent GPU scene, refreshed when scene structure or routing changes.
struct GpuSceneState {
    // Required LOD attributes per material, without and with authored tangents.
    std::vector<std::array<uint32_t, 2>> RequiredMaterialAttributes;
    // The OR of every mesh's PBR features.
    uint32_t MeshPbrFeatures{0};
    // The display settings every mesh's record display fields and the overlay jobs read, packed as the last settle pass saw them.
    uint64_t MeshDisplaySettings{0};
    bool EditPreludePending{};
    // Element selection bits changed, so edit work candidates reseed.
    bool EditSelectionDirty{};
    // Camera lenses, lights, colliders, tets or instance slots changed, so the overlay jobs rebuild before the next record.
    bool OverlayJobsDirty{true};
};
inline void Reset(GpuSceneState &s, state::Scene &) { s = {}; }
