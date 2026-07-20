#pragma once

#include "Mesh.h"
#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "Range.h"
#include "SlottedRange.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/CornerClass.h"
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

// One pose's per-class corner-normal sources, each span entry-relative: the base stores for the rest pose, or the normals derived from a morph target's full-weight pose.
struct CornerNormalSources {
    std::span<const vec3> VertexNormals;
    std::span<const vec3> SeamNormals;
    std::span<const vec3> FaceNormals;
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

    std::span<const Vertex> GetVertices(uint32_t id) const;
    std::span<Vertex> GetVertices(uint32_t id);
    SlottedRange GetVerticesRange(uint32_t id) const;
    SlottedRange GetBoneDeformRange(uint32_t id) const;
    SlottedRange GetMorphTargetRange(uint32_t id) const;
    uint32_t GetMorphTargetCount(uint32_t id) const { return Entries.at(id).MorphTargetCount; }
    uint32_t GetTriangleCount(uint32_t id) const { return Entries.at(id).TriangleCount; }
    std::span<const float> GetDefaultMorphWeights(uint32_t id) const { return Entries.at(id).DefaultMorphWeights; }
    bool MorphTargetsAuthorNormalDeltas(uint32_t id) const;

    // Source-form readback used by glTF export. Empty span when the mesh lacks the channel.
    std::span<const BoneDeformVertex> GetBoneDeform(uint32_t id) const;
    std::span<const MorphTargetVertex> GetMorphTargets(uint32_t id) const;

    // Base descriptor slots of the per-mesh GPU buffers (for shader push constants).
    uint32_t GetVertexStateSlot() const;
    uint32_t GetCornerTangentSlot() const;
    uint32_t GetCornerColorSlot() const;
    uint32_t GetCornerUvSlot() const;
    uint32_t GetEdgeSharpnessSlot() const;
    uint32_t GetFacePrimitiveSlot() const;
    uint32_t GetPrimitiveMaterialSlot() const;
    uint32_t GetBoneDeformSlot() const;
    uint32_t GetMorphTargetSlot() const;
    uint32_t GetAdjacencySlot() const;
    uint32_t GetCornerClassSlot() const;
    uint32_t GetCustomCornerMaskSlot() const;
    uint32_t GetCustomCornerNormalSlot() const;
    uint32_t GetBaseSeamNormalSlot() const;
    uint32_t GetBaseVertexNormalSlot() const;
    uint32_t GetBaseFaceNormalSlot() const;
    uint32_t GetFaceFirstTriangleSlot() const;

    std::span<const uint8_t> GetVertexStates(uint32_t id) const;
    // Canonical per-face and per-edge sharpness: 1 = shading discontinuity (flat face / sharp edge).
    // Callers writing these rederive corner normals afterward.
    std::span<const uint8_t> GetFaceSharpness(uint32_t id) const;
    std::span<uint8_t> GetFaceSharpness(uint32_t id);
    std::span<const uint8_t> GetEdgeSharpness(uint32_t id) const;
    std::span<uint8_t> GetEdgeSharpness(uint32_t id);
    // Any/all summary of the face sharpness bytes.
    SharpnessSummary GetFaceSharpnessSummary(uint32_t id) const;
    // Compose per-corner shading normals from the classification and the base normal stores, in triangulated face-fan order, with authored corner offsets applied where non-identity.
    // Requires current base stores (the derive pass ran since the last position/sharpness write).
    // Returns scratch storage valid until the next call.
    std::span<const vec3> GetCornerNormals(const Mesh &) const;
    // Encode the stashed authored corner normals as offsets from the derived corner normals, filling the custom corner-normal layer.
    // Consumes the stash, so it runs once, after the base normals derive.
    void EncodeAuthoredCornerNormals(const Mesh &);
    // Decide whether the mesh keeps its authored shading normals under morphing.
    // True when any target authors normal deltas, or when any listed full-weight pose derives corner normals beyond the authored match gate from the rest normals it would pin.
    // Requires derived base normals.
    void UpdateMorphShadingAuthored(const Mesh &, std::span<const CornerNormalSources>);
    // CSR vertex-to-edge incidence, edge items in edge order.
    VertexAdjacency GetVertexEdgeAdjacency(uint32_t id) const;
    Range GetVertexFanAdjacencyRange(uint32_t id) const { return Entries.at(id).VertexFanAdjacency; }
    // The class-buffer offset, or a sentinel when the mesh stores none (InvalidOffset = every corner Vertex, UniformFaceOffset = every corner Face).
    uint32_t GetCornerClassOffset(uint32_t id) const;
    Range GetCustomCornerMaskRange(uint32_t id) const { return Entries.at(id).CustomCornerMasks; }
    Range GetCustomCornerNormalRange(uint32_t id) const { return Entries.at(id).CustomCornerNormals; }
    Range GetBaseSeamNormalRange(uint32_t id) const { return Entries.at(id).BaseSeamNormals; }
    bool HasAuthoredNormals(uint32_t id) const { return Entries.at(id).HasAuthoredNormals; }
    bool GetMorphShadingAuthored(uint32_t id) const { return Entries.at(id).MorphShadingAuthored; }
    Range GetSeamFanRange(uint32_t id) const { return Entries.at(id).SeamFans; }
    uint32_t GetSeamCornerCount(uint32_t id) const { return Entries.at(id).SeamCornerCount; }
    Range GetFaceDataRange(uint32_t id) const { return Entries.at(id).FaceData; }
    // Base per-vertex normals at the entry's vertex-arena slots: derived for triangle meshes, authored for face-less meshes.
    std::span<const vec3> GetBaseVertexNormals(uint32_t id) const;
    std::span<vec3> GetBaseVertexNormals(uint32_t id);
    std::span<const vec3> GetBaseFaceNormals(uint32_t id) const;
    std::span<vec3> GetBaseFaceNormals(uint32_t id);
    std::span<const vec3> GetBaseSeamNormals(uint32_t id) const;
    std::span<vec3> GetBaseSeamNormals(uint32_t id);
    // Authored normals of face-less meshes, in vertex order (empty when the mesh has none).
    std::span<const vec3> GetPointNormals(uint32_t id) const;
    SlottedRange GetFaceStateRange(uint32_t id) const;
    SlottedRange GetEdgeStateRange(uint32_t id) const;
    Range GetEdgeSharpnessRange(uint32_t id) const;
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
    // Write edge sharpness from face dihedral angles: sharp where the angle exceeds `angle` (radians). Boundary edges stay smooth.
    void SetEdgeSharpnessByAngle(const Mesh &, float angle);
    // Classify each corner from the sharpness stores: vertex-normal, face-normal, or a seam sector of incident triangles.
    // Call after any sharpness write, then run the base derive pass to refill the base normal stores.
    void UpdateCornerClassification(const Mesh &);

    void Release(uint32_t id);

    // Reset all arenas + the StoreId table to empty (keeping GPU allocations for reuse). Call only on a full
    // scene clear, where no live entity references a StoreId, so StoreId/offset allocation restarts
    // deterministically — the mesh-arena analog of the entity-allocator reset in ClearScene.
    void Clear();

    // Serialize the source mesh arenas and the StoreId->Range entry table to a self-contained blob, and restore from one.
    // Restore writes the bytes back into the existing GPU buffers and re-establishes the entries, keeping every Range/StoreId offset valid.
    // The derived arenas (adjacency CSRs, corner classes, seam normals) rebuild via RebuildDerived after restore.
    std::vector<std::byte> Serialize() const;
    void Deserialize(std::span<const std::byte>);
    // Rebuild the meshes' derived adjacency, corner classification, and seam normals after Deserialize.
    // Runs in store-id order so the arena layout is deterministic (the span is sorted in place).
    void RebuildDerived(std::span<Mesh>);

private:
    struct Buffers;
    std::unique_ptr<Buffers> B; // Owns all GPU buffer storage (vertex/index/state/deform arenas)

    struct Entry {
        Range Vertices;
        Range FaceData; // Per-face range shared by FaceFirstTriangleBuffer and FaceStateBuffer
        Range CornerClasses{}; // One CornerClass value per corner from the sharpness stores, empty when every corner takes UniformCornerClass
        Range CustomCornerMasks{}; // Custom corner-normal presence: a (bitset word, exclusive rank) pair per 32 corners
        Range CustomCornerNormals{}; // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the masked corners
        Range CornerTangents{}, CornerColors{}; // Corner-domain attribute layers
        std::array<Range, 4> CornerUvs{};
        Range EdgeSharpness{}; // One byte per edge, 1 = sharp
        Range EdgeStates{}, TriangleFaceIds{}, FacePrimitives{}, PrimitiveMaterials{};
        // CSR vertex incidence, each range holding (vertex count + 1) offsets followed by the items
        Range VertexFanAdjacency{}, VertexEdgeAdjacency{};
        // Seam-corner sector CSR: (SeamCornerCount + 1) offsets, then fan items (FanItemEncoding)
        Range SeamFans{};
        Range BaseSeamNormals{}; // Composed sector normal per seam corner
        Range PointNormals{}; // Authored normals of face-less meshes, in vertex order
        Range BoneDeform{}, MorphTargets{};
        uint32_t SeamCornerCount{0};
        uint32_t MorphTargetCount{0};
        uint32_t TriangleCount{0};
        CornerClass UniformCornerClass{CornerClass::Vertex}; // Every corner's class while CornerClasses is empty
        // Whether the source authored vertex normals, so shading may stay authored under morphing (glTF semantics).
        bool HasAuthoredNormals{false};
        // The mesh's morph shading keeps its authored normals, because a target authors normal deltas or pins normals that derivation would move.
        bool MorphShadingAuthored{false};
        std::vector<float> DefaultMorphWeights{};
        std::vector<PrimitiveTriangleRange> PrimitiveTriangleRanges{};
        // Authored corner normals held until EncodeAuthoredCornerNormals consumes them once the base normal stores are derived
        std::vector<vec3> AuthoredCornerNormals{};
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
        uint32_t AdjacencyWords{};
    } Pending{};

    uint32_t AcquireId(Entry &&);
    // GetCornerNormals with the mesh's triangulated index stream already at hand.
    std::span<const vec3> GetCornerNormals(const Mesh &, std::span<const uint32_t> indices) const;
    // Fill the base vertex-normal mirror over `vertices`: a face-less mesh's point normals, zero otherwise (triangle meshes rederive the region).
    void FillBaseVertexNormalMirror(Range vertices, Range point_normals);
    void BuildVertexAdjacency(const Mesh &);
    Range AllocateVertices(uint32_t count);
    Range AllocateFaces(uint32_t count);
    std::span<uint8_t> GetFaceStates(Range);
    std::span<uint8_t> GetVertexStates(Range);
    std::span<const uint8_t> GetVertexStates(Range) const;
    void ClearElementStates(Range vertices, Range faces, Range edges);
};
