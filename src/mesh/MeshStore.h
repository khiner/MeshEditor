#pragma once

#include "Mesh.h"
#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "Range.h"
#include "SlottedRange.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/MorphTargetVertex.h"

#include <expected>
#include <filesystem>

namespace mvk {
struct BufferContext;
} // namespace mvk

struct ObjPlyMaterial {
    vec4 BaseColorFactor;
    float MetallicFactor, RoughnessFactor;
    std::string Name;

    // OBJ fields
    std::optional<std::filesystem::path> BaseColorTexturePath{}, NormalTexturePath{};
    bool HasAlphaTexture{false};
};

// A freshly created mesh: its store id plus the half-edge connectivity the caller attaches to the entity
// (as a MeshConnectivity component). MeshStore owns no connectivity, so it hands ownership back here.
// MorphTangentDeltas returns the target-major tangent deltas the arena doesn't store, compacted to the welded vertex set.
struct CreatedMesh {
    uint32_t StoreId;
    MeshConnectivity Connectivity;
    std::vector<vec3> MorphTangentDeltas{};
};

struct MeshWithMaterials {
    CreatedMesh Mesh;
    std::vector<ObjPlyMaterial> Materials;
};

struct PrimitiveTriangleRange {
    uint32_t PrimitiveIndex, FirstTriangle, TriangleCount;
};

// Optional per-vertex armature deformation channels.
struct ArmatureDeformData {
    std::vector<uvec4> Joints;
    std::vector<vec4> Weights;
};

struct SharpnessSummary {
    bool Any, All;
};

// Per-source-primitive metadata; all vectors indexed by primitive.
struct MeshPrimitives {
    std::vector<uint32_t> FacePrimitiveIndices{}; // per-face source primitive index
    std::vector<uint32_t> MaterialIndices{};
    std::vector<uint32_t> AttributeFlags{}; // bitmask of MeshAttributeBit_*
    std::vector<uint8_t> HasSourceIndices{}; // 0 = source drew non-indexed
    // Inner size = variant count (empty when primitive has no mappings); nullopt falls back to MaterialIndices.
    std::vector<std::vector<std::optional<uint32_t>>> VariantMappings{};
};

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mvk::BufferContext &);
    ~MeshStore();
    MeshStore(MeshStore &&) noexcept;
    MeshStore &operator=(MeshStore &&) noexcept;

    // Accumulate arena reservations for upcoming mesh operations.
    // Call PlanCreate/PlanClone per mesh, then CommitReserves() once before the actual operations.
    void PlanCreate(const MeshData &, const MeshPrimitives & = {}, bool has_deform = false, uint32_t morph_target_count = 0, const MeshVertexAttributes & = {});
    void PlanClone(const Mesh &);
    // Reserve all arenas for accumulated plans, then reset.
    void CommitReserves();

    // `weld` merges vertices identical in every vertex-domain channel: position, joints/weights, and morph deltas.
    // Welding recovers authored normals as face sharpness on faceted faces and as a custom corner-normal layer where they deviate from derivation.
    CreatedMesh CreateMesh(MeshData &&, MeshVertexAttributes &&, MeshPrimitives &&, bool flat_shaded = false, std::optional<ArmatureDeformData> = {}, std::optional<MorphTargetData> = {}, bool weld = false);
    CreatedMesh CloneMesh(const Mesh &);

    // `weld` merges vertices identical in every vertex-domain channel.
    // UVs and recovered normals keep their per-corner values.
    std::expected<MeshWithMaterials, std::string> LoadMesh(const std::filesystem::path &, bool weld = false);

    // Allocate vertex-only store entry (no topology, no face/edge/primitive/material buffers).
    // Returns {storeId, vertexRange}. Release via Release(storeId).
    std::pair<uint32_t, Range> AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs);

    // Like AllocateVertexBuffer, but in the separate overlay store (its own arena and id space).
    std::pair<uint32_t, Range> AllocateOverlayVertexBuffer(std::span<const vec3> positions);
    SlottedRange GetOverlayVerticesRange(uint32_t id) const;
    void ReleaseOverlay(uint32_t id);

    void SetPositions(const Mesh &, std::span<const vec3>);
    void SetPosition(const Mesh &, uint32_t index, vec3 position); // Single vertex, no normal update

    std::span<const Vertex> GetVertices(uint32_t id) const;
    std::span<Vertex> GetVertices(uint32_t id);
    SlottedRange GetVerticesRange(uint32_t id) const;
    SlottedRange GetBoneDeformRange(uint32_t id) const;
    SlottedRange GetMorphTargetRange(uint32_t id) const;
    uint32_t GetMorphTargetCount(uint32_t id) const { return Entries.at(id).MorphTargetCount; }
    uint32_t GetTriangleCount(uint32_t id) const { return Entries.at(id).TriangleCount; }
    std::span<const float> GetDefaultMorphWeights(uint32_t id) const { return Entries.at(id).DefaultMorphWeights; }

    // Source-form readback used by glTF export. Empty span when the mesh lacks the channel.
    std::span<const BoneDeformVertex> GetBoneDeform(uint32_t id) const;
    std::span<const MorphTargetVertex> GetMorphTargets(uint32_t id) const;

    // Base descriptor slots of the per-mesh GPU buffers (for shader push constants).
    uint32_t GetVertexStateSlot() const;
    uint32_t GetCornerNormalSlot() const;
    uint32_t GetCornerTangentSlot() const;
    uint32_t GetCornerColorSlot() const;
    uint32_t GetCornerUvSlot() const;
    uint32_t GetEdgeSharpnessSlot() const;
    uint32_t GetFacePrimitiveSlot() const;
    uint32_t GetPrimitiveMaterialSlot() const;
    uint32_t GetBoneDeformSlot() const;
    uint32_t GetMorphTargetSlot() const;

    std::span<const uint8_t> GetVertexStates(uint32_t id) const;
    // Canonical per-face and per-edge sharpness: 1 = shading discontinuity (flat face / sharp edge).
    // Callers writing these rederive corner normals afterward.
    std::span<const uint8_t> GetFaceSharpness(uint32_t id) const;
    std::span<uint8_t> GetFaceSharpness(uint32_t id);
    std::span<const uint8_t> GetEdgeSharpness(uint32_t id) const;
    std::span<uint8_t> GetEdgeSharpness(uint32_t id);
    // Any/all summary of the face sharpness bytes.
    SharpnessSummary GetFaceSharpnessSummary(uint32_t id) const;
    std::span<const vec3> GetCornerNormals(uint32_t id) const;
    SlottedRange GetFaceStateRange(uint32_t id) const;
    SlottedRange GetEdgeStateRange(uint32_t id) const;
    Range GetEdgeSharpnessRange(uint32_t id) const;
    Range GetCornerNormalRange(uint32_t id) const;
    // Corner-domain attribute layers (one value per triangulated face corner, fan order).
    // Empty range/span when the mesh lacks the channel.
    Range GetCornerTangentRange(uint32_t id) const;
    Range GetCornerColorRange(uint32_t id) const;
    Range GetCornerUvRange(uint32_t id, uint32_t set) const;
    std::span<const vec4> GetCornerTangents(uint32_t id) const;
    std::span<const vec4> GetCornerColors(uint32_t id) const;
    std::span<const vec2> GetCornerUvs(uint32_t id, uint32_t set) const;

    SlottedRange GetFaceIdRange(uint32_t id) const;
    SlottedRange GetFacePrimitiveRange(uint32_t id) const;
    SlottedRange GetPrimitiveMaterialRange(uint32_t id) const;

    std::span<const uint32_t> GetTriangleFaceIds(uint32_t id) const;
    std::span<const uint32_t> GetFaceFirstTriangles(uint32_t id) const;
    std::span<const uint32_t> GetFacePrimitiveIndices(uint32_t id) const;
    std::span<uint32_t> GetFacePrimitiveIndices(uint32_t id);
    std::span<const uint32_t> GetPrimitiveMaterialIndices(uint32_t id) const;
    std::span<uint32_t> GetPrimitiveMaterialIndices(uint32_t id);

    std::span<const PrimitiveTriangleRange> GetPrimitiveTriangleRanges(uint32_t id) const { return Entries.at(id).PrimitiveTriangleRanges; }

    // Rewrite element states for sound-vertex excitation: the listed vertices are selected,
    // with optional active and excited vertices.
    void UpdateSoundVertexStates(const Mesh &, std::span<const uint32_t> vertices, std::optional<uint32_t> active_vertex = {}, std::optional<uint32_t> excited_vertex = {});
    // Derive the other element domains' states from the current edit element's states.
    void UpdateEdgeStatesFromFaces(const Mesh &, std::optional<uint32_t> active_face);
    void UpdateEdgeStatesFromVertices(const Mesh &);
    void UpdateFaceStatesFromVertices(const Mesh &);
    void UpdateFaceStatesFromEdges(const Mesh &);
    void UpdateVertexStatesFromFaces(const Mesh &, std::optional<uint32_t> active_face = {});
    void UpdateVertexStatesFromEdges(const Mesh &, std::optional<uint32_t> active_edge = {});
    void UpdateVertexNormals(const Mesh &);
    // Write edge sharpness from face dihedral angles: sharp where the angle exceeds `angle` (radians). Boundary edges stay smooth.
    void SetEdgeSharpnessByAngle(const Mesh &, float angle);
    // Derive per-corner shading normals from positions, vertex normals, and sharpness, in triangulated face-fan index order.
    void UpdateCornerNormals(const Mesh &);
    // Derive from `positions` instead of the stored positions, matching what committing them produces.
    void UpdateCornerNormals(const Mesh &, std::span<const vec3> positions);

    void Release(uint32_t id);

    // Reset all arenas + the StoreId table to empty (keeping GPU allocations for reuse). Call only on a full
    // scene clear, where no live entity references a StoreId, so StoreId/offset allocation restarts
    // deterministically — the mesh-arena analog of the entity-allocator reset in ClearScene.
    void Clear();

    // Serialize all mesh arenas and the StoreId->Range entry table to a self-contained blob, and restore from one.
    // Restore writes the bytes back into the existing GPU buffers and re-establishes the entries, keeping every Range/StoreId offset valid.
    std::vector<std::byte> Serialize() const;
    void Deserialize(std::span<const std::byte>);

private:
    struct Buffers;
    std::unique_ptr<Buffers> B; // Owns all GPU buffer storage (vertex/index/state/deform arenas)

    struct Entry {
        Range Vertices;
        Range FaceData; // Per-face range shared by FaceFirstTriangleBuffer and FaceStateBuffer
        Range CornerNormals{}; // One normal per triangulated face corner
        Range CustomCornerNormals{}; // Authored corner normals, vec3(0) = use derived
        Range CornerTangents{}, CornerColors{}; // Corner-domain attribute layers
        std::array<Range, 4> CornerUvs{};
        Range EdgeSharpness{}; // One byte per edge, 1 = sharp
        Range EdgeStates{}, TriangleFaceIds{}, FacePrimitives{}, PrimitiveMaterials{};
        Range BoneDeform{}, MorphTargets{};
        uint32_t MorphTargetCount{0};
        uint32_t TriangleCount{0};
        std::vector<float> DefaultMorphWeights{};
        std::vector<PrimitiveTriangleRange> PrimitiveTriangleRanges{};
        bool Alive{false};
    };

    std::vector<Entry> Entries{};
    std::vector<uint32_t> FreeIds{};

    // Overlay store: vertex range per id.
    std::vector<Range> OverlayEntries{};
    std::vector<uint32_t> OverlayFreeIds{};

    struct PendingReserves {
        uint32_t Vertices{}, Faces{}, Triangles{}, Edges{}, EdgeStates{};
        uint32_t Primitives{};
        uint32_t BoneDeformVertices{}, MorphTargetEntries{};
        uint32_t CornerTangents{}, CornerColors{}, CornerUvs{};
    } Pending{};

    uint32_t AcquireId(Entry &&);
    void ComposeCustomCornerNormals(uint32_t id, std::span<vec3> corners) const;
    Range AllocateVertices(uint32_t count);
    Range AllocateFaces(uint32_t count);
    std::span<uint8_t> GetFaceStates(Range);
    std::span<uint8_t> GetVertexStates(Range);
    std::span<const uint8_t> GetVertexStates(Range) const;
    void ClearElementStates(Range vertices, Range faces, Range edges);
};
