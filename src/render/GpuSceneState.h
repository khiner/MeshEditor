#pragma once

#include "gpu/ElementWork.h"
#include "render/RenderSchema.h"

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

struct PosedLayout {
    PosedNamespaces Namespaces;
    // The deform offsets the record display fields hold, or InvalidOffset without that deformation.
    uint32_t BoneDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset}, MorphTargetCount{0};
    std::vector<state::Entity> Armatures;
};

// A mesh's derived edit work and bounds parents, which survive gestures until topology, shading layout, or edit mode changes.
struct MeshEditWork {
    uint32_t StoreId{InvalidOffset};
    // Work items are canonical element handles, and bounds work holds canonical vertex blocks.
    ElementWork Candidates, Vertices, Faces, Normals, Meshlets, BoundsTiles;
    std::array<ElementWork, 3> BoundsLevels;
    std::vector<Range> RefreshRanges;
    bool CandidateReady{}, FootprintReady{}, Modified{}, PreviewActive{};
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

// Static runs use one entry and no tile words.
// Posed words contain bounds keys, derive blocks, then finest-node members and meshlet counts.
struct BoundsRun {
    Range Entries, DeriveEntries, Tiles;
    std::array<uint32_t, 4> BoundsKeys{}; // By vertex bounds level.
    uint32_t FaceBlocks{}, VertexBlocks{}, MeshletJobs{};
    bool Posed() const { return Tiles.Count != 0u; }
};

struct TalliedRecord {
    uint32_t StoreId;
};

// A mesh whose meshlets contain position-dirty bits, which the next settle outside edit mode stages.
struct MeshletPositionsDirty {};
// A mesh whose cluster groups contain dirty bits, which the next settle outside edit mode repairs.
struct ClusterLodDirty {};
struct ClusterLodDemand {};
// A mesh whose record display fields rederive at the end of the settle pass, after its render data changed.
struct MeshDisplayDirty {};
// A mesh whose bounds entries, prelude tiles and pose namespaces lay out again at the end of the settle pass, after its own inputs changed.
struct MeshLayoutDirty {};
// A laid-out mesh whose bounds entries the next recorded prelude recomputes.
struct MeshBoundsDirty {};

// A record takes a route or an overlay when any visible viewport shows it, and each viewport's passes select their own.
struct DisplayUnion {
    bool Overlays{}, Bones{}, VertexNormals{}, FaceNormals{};
    bool BoundingBoxes{};
    bool Wireframe{};
    bool AllMaterialShaded{true};
    uint32_t DebugChannels{}; // One bit per debug channel a visible material-shaded viewport shows.

    bool operator==(const DisplayUnion &) const = default;
};

struct GpuSceneState {
    // Required LOD attributes per material, without and with authored tangents.
    std::vector<std::array<uint32_t, 2>> RequiredMaterialAttributes;
    uint32_t MeshPbrFeatures{0};
    DisplayUnion Displays{};
    bool EditPreludePending{};
    bool EditSelectionDirty{};
    bool OverlayJobsDirty{true};
};
inline void Reset(GpuSceneState &s, state::Registry &) { s = {}; }
