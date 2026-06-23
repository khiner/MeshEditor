#pragma once

#include "Mesh.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "Range.h"
#include "SlottedRange.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/MorphTargetVertex.h"

#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <unordered_set>

namespace mvk {
struct BufferContext;
}

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
struct CreatedMesh {
    uint32_t StoreId;
    MeshConnectivity Connectivity;
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

struct MeshVertexAttributes;

// Per-source-primitive metadata; all vectors indexed by primitive.
struct MeshPrimitives {
    std::vector<uint32_t> FacePrimitiveIndices{}; // per-face source primitive index
    std::vector<uint32_t> MaterialIndices{};
    std::vector<uint32_t> VertexCounts{};
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
    void PlanCreate(const MeshData &, const MeshPrimitives & = {}, bool has_deform = false, uint32_t morph_target_count = 0);
    void PlanClone(const Mesh &);
    // Reserve all arenas for accumulated plans, then reset.
    void CommitReserves();

    CreatedMesh CreateMesh(MeshData &&, MeshVertexAttributes &&, MeshPrimitives &&, std::optional<ArmatureDeformData> = {}, std::optional<MorphTargetData> = {});
    CreatedMesh CloneMesh(const Mesh &);

    std::expected<MeshWithMaterials, std::string> LoadMesh(const std::filesystem::path &);

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
    uint32_t GetFaceFirstTriangleSlot() const;
    uint32_t GetFacePrimitiveSlot() const;
    uint32_t GetPrimitiveMaterialSlot() const;
    uint32_t GetBoneDeformSlot() const;
    uint32_t GetMorphTargetSlot() const;

    std::span<const uint8_t> GetVertexStates(uint32_t id) const;
    SlottedRange GetFaceStateRange(uint32_t id) const;
    SlottedRange GetEdgeStateRange(uint32_t id) const;
    SlottedRange GetFaceFirstTriRange(uint32_t id) const;

    SlottedRange GetFaceIdRange(uint32_t id) const;
    SlottedRange GetFacePrimitiveRange(uint32_t id) const;
    SlottedRange GetPrimitiveMaterialRange(uint32_t id) const;

    std::span<const uint32_t> GetTriangleFaceIds(uint32_t id) const;
    std::span<const uint32_t> GetFacePrimitiveIndices(uint32_t id) const;
    std::span<uint32_t> GetFacePrimitiveIndices(uint32_t id);
    std::span<const uint32_t> GetPrimitiveMaterialIndices(uint32_t id) const;
    std::span<uint32_t> GetPrimitiveMaterialIndices(uint32_t id);

    std::span<const PrimitiveTriangleRange> GetPrimitiveTriangleRanges(uint32_t id) const { return Entries.at(id).PrimitiveTriangleRanges; }

    void UpdateElementStates(
        const Mesh &, Element,
        const std::unordered_set<he::VH> &selected_vertices,
        const std::unordered_set<he::EH> &selected_edges,
        const std::unordered_set<he::EH> &active_edges,
        const std::unordered_set<he::FH> &selected_faces,
        std::optional<uint32_t> active_handle,
        std::optional<uint32_t> excited_handle = {}
    );
    void UpdateEdgeStatesFromFaces(const Mesh &, std::span<const uint32_t> selected_faces, std::optional<uint32_t> active_face);
    void UpdateEdgeStatesFromVertices(const Mesh &);
    void UpdateFaceStatesFromVertices(const Mesh &);
    void UpdateFaceStatesFromEdges(const Mesh &);
    // Writes vertex state buffer from non-vertex element handles (Face/Edge), for the GPU transform preview shader.
    void UpdateVertexStatesFromElements(const Mesh &, std::span<const uint32_t> handles, Element, std::optional<uint32_t> active_handle = {});
    void UpdateNormals(const Mesh &, bool skip_nonzero = false);

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
        uint32_t Vertices{}, Faces{}, Triangles{}, EdgeStates{};
        uint32_t Primitives{};
        uint32_t BoneDeformVertices{}, MorphTargetEntries{};
    } Pending{};

    uint32_t AcquireId(Entry &&);
    Range AllocateVertices(uint32_t count);
    Range AllocateFaces(uint32_t count);
    std::span<uint8_t> GetFaceStates(Range);
    std::span<uint8_t> GetVertexStates(Range);
    std::span<const uint8_t> GetVertexStates(Range) const;
    void ClearElementStates(Range vertices, Range faces, Range edges);
};
