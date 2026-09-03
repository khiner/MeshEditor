#pragma once

#include "Mesh.h"
#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "Range.h"
#include "SlottedRange.h"
#include "TetBuffers.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/CornerClass.h"
#include "gpu/EditSelectionStorage.h"
#include "gpu/EditSelectionSummary.h"
#include "gpu/MorphTargetVertex.h"
#include "metal/Buffer.h"

#include <expected>
#include <filesystem>

struct ObjPlyMaterial {
    vec4 BaseColorFactor;
    float MetallicFactor, RoughnessFactor;
    std::string Name;

    // OBJ fields
    std::optional<std::filesystem::path> BaseColorTexturePath{}, NormalTexturePath{};
    bool HasAlphaTexture{false};
};

// MorphTangentDeltas returns the target-major tangent deltas the arena doesn't store, compacted to the welded vertex set.
struct CreatedMesh {
    uint32_t StoreId;
    std::vector<vec3> MorphTangentDeltas{};
};

struct PrimitiveTriangleRange {
    uint32_t PrimitiveIndex, FirstTriangle, TriangleCount;
};

struct ArmatureDeformData {
    std::vector<uvec4> Joints;
    std::vector<vec4> Weights;
};

struct SharpnessSummary {
    bool Any, All;
};

// Contains entry-relative corner-normal sources for one pose.
struct CornerNormalSources {
    std::span<const vec3> VertexNormals;
    std::span<const vec3> SeamNormals;
    std::span<const vec3> FaceNormals;
};

// Per-source-primitive metadata, with every vector indexed by primitive.
struct MeshPrimitives {
    std::vector<uint32_t> ElementPrimitiveIndices{}; // source primitive index per drawn element (per face, or per vertex for point/line meshes)
    std::vector<uint32_t> MaterialIndices{};
    std::vector<uint32_t> AttributeFlags{}; // bitmask of MeshAttributeBit_*
    std::vector<uint8_t> HasSourceIndices{}; // 0 = source drew non-indexed
    // Inner size = variant count (empty when primitive has no mappings); nullopt falls back to MaterialIndices.
    std::vector<std::vector<std::optional<uint32_t>>> VariantMappings{};
};

// An OBJ or PLY file read into the arrays a mesh is created from.
struct MeshDataWithMaterials {
    MeshData Mesh;
    MeshVertexAttributes Attrs;
    MeshPrimitives Primitives;
    std::vector<ObjPlyMaterial> Materials;
};

std::expected<MeshDataWithMaterials, std::string> ReadMeshFile(const std::filesystem::path &);

// Contains source-derived data without arena or store ownership.
struct PreparedMesh {
    std::vector<vec4> CornerTangents, CornerColors;
    std::array<std::vector<vec2>, 4> CornerUvs;
    std::vector<vec3> AuthoredCornerNormals;
    // Target-major morph tangent deltas remain host-owned through welding.
    std::vector<vec3> MorphTangentDeltas;
};

// Orders faces by primitive and gathers corner channels while preserving CreateMesh inputs.
// Welding recovers authored normals as face sharpness on faceted faces and as a custom corner-normal layer where they deviate from derivation.
PreparedMesh PrepareMeshSources(MeshData &, MeshVertexAttributes &, MeshPrimitives &);
BuiltConnectivity BuildPreparedConnectivity(const MeshStore &, uint32_t id, const MeshData &, const ConnectivityStorage &);

// Returns true when triangle topology permits GPU vertex-fan construction.
bool BuildsFanAdjacencyOnGpu(const Mesh &);
// Returns true when triangle-manifold topology permits GPU vertex-edge construction.
bool BuildsEdgeAdjacencyOnGpu(const Mesh &);

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mtl::BufferContext &);
    ~MeshStore();
    MeshStore(MeshStore &&) noexcept;
    MeshStore &operator=(MeshStore &&) noexcept;

    // Call once after all PlanCreate and PlanClone calls and before their corresponding operations.
    void PlanCreate(const MeshData &, const MeshPrimitives & = {}, bool has_deform = false, uint32_t morph_target_count = 0, const MeshVertexAttributes & = {});
    void PlanClone(const Mesh &);
    void CommitReserves();

    // Takes source positions and corners into the arenas and returns their store ID.
    uint32_t CreateMeshSource(const MeshData &);
    // Takes skin and morph channels into arenas at the source vertex count for in-place welding.
    void CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &, const std::optional<MorphTargetData> &);
    // Trims all vertex-domain arena ranges to `welded_vertices`.
    void ShrinkMeshSource(uint32_t id, uint32_t welded_vertices);
    MeshConnectivity GetConnectivity(uint32_t id) const;
    // Allocates connectivity storage from source counts in call order.
    void AllocateConnectivity(uint32_t id, uint32_t vertex_count, uint32_t halfedge_count, uint32_t face_count, bool face_starts);
    ConnectivityStorage GetConnectivityStorage(uint32_t id);
    SlottedRange GetConnectivityRange(uint32_t id) const;
    SlottedRange GetConnectivityHalfedgeToEdgeRange(uint32_t id) const;
    SlottedRange GetConnectivityEdgeRange(uint32_t id) const;
    void SetConnectivityEdgeCount(uint32_t id, uint32_t edge_count);
    void PlaceConnectivity(uint32_t id, const BuiltConnectivity &);

    // Completes the store entry created by CreateMeshSource using the prepared source data.
    CreatedMesh CreateMesh(uint32_t id, MeshData &&, MeshVertexAttributes &&, MeshPrimitives &&, PreparedMesh &&, bool flat_shaded = false);
    CreatedMesh CloneMesh(const Mesh &);

    // Returns a vertex-only store ID that must be released with Release.
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs);

    std::span<const Vertex> GetVertices(uint32_t id) const;
    std::span<Vertex> GetMutableVertices(uint32_t id);
    SlottedRange GetVerticesRange(uint32_t id) const;
    SlottedRange GetBoneDeformRange(uint32_t id) const;
    SlottedRange GetMorphTargetRange(uint32_t id) const;
    uint32_t GetMorphTargetCount(uint32_t id) const { return Entries.at(id).MorphTargetCount; }
    uint32_t GetTriangleCount(uint32_t id) const { return Entries.at(id).TriangleCount; }
    std::span<const float> GetDefaultMorphWeights(uint32_t id) const { return Entries.at(id).DefaultMorphWeights; }
    bool MorphTargetsAuthorNormalDeltas(uint32_t id) const;

    // Returns an empty span when the source lacks bone deformation.
    std::span<const BoneDeformVertex> GetBoneDeform(uint32_t id) const;
    std::span<const MorphTargetVertex> GetMorphTargets(uint32_t id) const;

    uint32_t GetCornerTangentSlot() const;
    uint32_t GetCornerColorSlot() const;
    uint32_t GetCornerUvSlot() const;
    uint32_t GetEdgeSharpnessSlot() const;
    uint32_t GetEdgeSharpnessCount() const;
    uint32_t GetElementPrimitiveSlot() const;
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
    uint32_t GetTetPositionSlot() const;
    uint32_t GetTetEdgeIndexSlot() const;
    // Copies tetrahedral wireframe geometry into GPU-only canonical arenas.
    TetBuffers AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices);
    void ReleaseTets(TetBuffers);
    uint32_t GetSoundVertexSlot() const;
    Range AllocateSoundVertices(std::span<const uint32_t>);
    void ReleaseSoundVertices(Range);
    std::span<const uint32_t> GetSoundVertices(Range) const;

    // Canonical per-face and per-edge sharpness: 1 = shading discontinuity (flat face / sharp edge).
    // Callers writing these rederive corner normals afterward.
    std::span<const uint8_t> GetFaceSharpness(uint32_t id) const;
    std::span<uint8_t> GetMutableFaceSharpness(uint32_t id);
    std::span<const uint8_t> GetEdgeSharpness(uint32_t id) const;
    std::span<uint8_t> GetMutableEdgeSharpness(uint32_t id);
    SharpnessSummary GetFaceSharpnessSummary(uint32_t id) const;
    // Returns composed corner normals in triangulated face-fan order until the next call.
    // Requires current base stores (the derive pass ran since the last position/sharpness write).
    // Returns scratch storage valid until the next call.
    std::span<const vec3> GetCornerNormals(const Mesh &) const;
    // Encode the stashed authored corner normals as offsets from the derived corner normals, filling the custom corner-normal layer.
    // Consumes the stash, so it runs once, after the base normals derive.
    void EncodeAuthoredCornerNormals(const Mesh &);
    // Preserves authored shading when targets include normal deltas or materially change derived corner normals.
    // Requires derived base normals.
    void UpdateMorphShadingAuthored(const Mesh &, std::span<const CornerNormalSources>);
    // CSR vertex-to-edge incidence, edge items in edge order.
    VertexAdjacency GetVertexEdgeAdjacency(uint32_t id) const;
    // Rebuild the mesh's CSR tables and report the first entry differing from the stored ones, or empty when they match.
    std::string CheckVertexAdjacency(const Mesh &) const;
    Range GetVertexFanAdjacencyRange(uint32_t id) const { return Entries.at(id).VertexFanAdjacency; }
    Range GetVertexEdgeAdjacencyRange(uint32_t id) const { return Entries.at(id).VertexEdgeAdjacency; }
    // Returns the class-buffer offset or a uniform-class sentinel.
    uint32_t GetCornerClassOffset(uint32_t id) const;
    std::span<const uint32_t> GetCornerClasses(uint32_t id) const;
    Range GetCustomCornerMaskRange(uint32_t id) const { return Entries.at(id).CustomCornerMasks; }
    std::span<const uvec2> GetCustomCornerMasks(uint32_t id) const;
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
    SlottedRange GetBaseFaceNormalRange(uint32_t id) const;
    SlottedRange GetBaseVertexNormalRange(uint32_t id) const;
    SlottedRange GetBaseSeamNormalSlottedRange(uint32_t id) const;
    std::span<const vec3> GetBaseSeamNormals(uint32_t id) const;
    std::span<vec3> GetBaseSeamNormals(uint32_t id);
    // Returns face-less authored normals in vertex order.
    std::span<const vec3> GetPointNormals(uint32_t id) const;
    Range GetEdgeSharpnessRange(uint32_t id) const;
    SlottedRange GetFaceSharpnessRange(uint32_t id) const;
    SlottedRange GetEdgeSharpnessSlottedRange(uint32_t id) const;
    // Corner-domain attribute layers (one value per triangulated face corner, fan order).
    // Empty range/span when the mesh lacks the channel.
    static constexpr uint32_t MaxUvSets{4};
    Range GetCornerTangentRange(uint32_t id) const;
    Range GetCornerColorRange(uint32_t id) const;
    Range GetCornerUvRange(uint32_t id, uint32_t set) const;
    std::span<const vec4> GetCornerTangents(uint32_t id) const;
    std::span<const vec4> GetCornerColors(uint32_t id) const;
    std::span<const vec2> GetCornerUvs(uint32_t id, uint32_t set) const;

    SlottedRange GetFaceIdRange(uint32_t id) const;
    SlottedRange GetElementPrimitiveRange(uint32_t id) const;
    SlottedRange GetPrimitiveMaterialRange(uint32_t id) const;

    std::span<const uint32_t> GetTriangleFaceIds(uint32_t id) const;
    // Returns canonical corner vertex indices shared by connectivity and face drawing.
    std::span<const uint32_t> GetFaceCorners(uint32_t id) const;
    SlottedRange GetFaceCornerRange(uint32_t id) const;
    // Allocates compact masks for every element domain; the GPU derives two from the authoritative domain.
    void EnsureSelectionBits(const Mesh &);
    std::span<const uint32_t> GetSelectionBits(uint32_t id, Element) const;
    uint32_t GetSelectionBitsSlot() const;
    uint32_t GetSelectionBitOffset(uint32_t id, Element) const;
    SlottedRange GetSelectionBitsRange(uint32_t id, Element) const;
    EditSelectionStorage GetEditSelectionStorage(uint32_t id) const;
    SlottedRange GetSelectionBaselineRange(uint32_t id) const;
    SlottedRange GetSelectionSummaryRange(uint32_t id) const;
    const EditSelectionSummary &GetSelectionSummary(uint32_t id) const;
    std::span<const uint32_t> GetFaceFirstTriangles(uint32_t id) const;
    std::span<const uint32_t> GetElementPrimitiveIndices(uint32_t id) const;
    std::span<uint32_t> GetElementPrimitiveIndices(uint32_t id);
    std::span<const uint32_t> GetPrimitiveMaterialIndices(uint32_t id) const;
    std::span<uint32_t> GetPrimitiveMaterialIndices(uint32_t id);

    std::span<const PrimitiveTriangleRange> GetPrimitiveTriangleRanges(uint32_t id) const { return Entries.at(id).PrimitiveTriangleRanges; }

    // Write edge sharpness from face dihedral angles: sharp where the angle exceeds `angle` (radians). Boundary edges stay smooth.
    void SetEdgeSharpnessByAngle(const Mesh &, float angle);
    // Classify each corner from the sharpness stores: vertex-normal, face-normal, or a seam sector of incident triangles.
    // Call after any sharpness write, then run the base derive pass to refill the base normal stores.
    void UpdateCornerClassification(const Mesh &);

    void Release(uint32_t id);

    // Reset all arenas and the StoreId table to empty, keeping GPU allocations for reuse.
    // Requires a full scene clear without live StoreId references so allocation restarts deterministically.
    void Clear();

    // Serialize the source mesh arenas and the StoreId->Range entry table to a self-contained blob, and restore from one.
    // Restore writes the bytes back into the existing GPU buffers and re-establishes the entries, keeping every Range/StoreId offset valid.
    // The derived arenas (adjacency CSRs, corner classes, seam normals) rebuild via RebuildDerived after restore.
    std::vector<std::byte> Serialize() const;
    void Deserialize(std::span<const std::byte>);
    // Rebuilds derived data in store-ID order and sorts the input span in place.
    void RebuildDerived(std::span<Mesh>);

private:
    struct Buffers;
    std::unique_ptr<Buffers> B;

    struct Entry {
        Range Vertices{};
        Range FaceData{}; // Per-face range shared by FaceFirstTriangleBuffer and FaceStateBuffer
        Range CornerClasses{}; // One CornerClass value per corner from the sharpness stores, empty when every corner takes UniformCornerClass
        Range CustomCornerMasks{}; // Custom corner-normal presence: a (bitset word, exclusive rank) pair per 32 corners
        Range CustomCornerNormals{}; // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the masked corners
        Range CornerTangents{}, CornerColors{}; // Corner-domain attribute layers
        std::array<Range, MaxUvSets> CornerUvs{};
        Range EdgeSharpness{}; // One byte per edge, 1 = sharp
        Range TriangleFaceIds{}, ElementPrimitives{}, PrimitiveMaterials{}, FaceCorners{};
        std::array<Range, 3> SelectionBits{}; // vertex, edge, face masks
        Range SelectionBaseline{}, SelectionSummary{};
        // CSR ranges contain vertex-count-plus-one offsets followed by incident items.
        Range VertexFanAdjacency{}, VertexEdgeAdjacency{};
        // The mesh's half-edge connectivity, laid out in the order SliceConnectivity reads it.
        Range Connectivity{}, ConnectivityEdges{}, ConnectivityHalfedgeToEdge{};
        uint32_t ConnectivityVertices{}, ConnectivityHalfedges{}, ConnectivityEdgeCount{}, ConnectivityFaces{};
        bool ConnectivityFaceStarts{false}; // An n-gon mesh stores each face's first halfedge.
        // Seam-fan CSR contains SeamCornerCount-plus-one offsets followed by encoded fan items.
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
        // Retained until base-normal derivation permits offset encoding.
        std::vector<vec3> AuthoredCornerNormals{};
        bool Alive{false};
    };

    std::vector<Entry> Entries{};
    std::vector<uint32_t> FreeIds{};

    struct PendingReserves {
        uint32_t Vertices{}, Faces{}, Triangles{}, Edges{}, FaceCorners{};
        uint32_t Primitives{};
        uint32_t ElementPrimitiveIndices{}; // One per face, or per vertex for point and line meshes.
        uint32_t BoneDeformVertices{}, MorphTargetEntries{};
        uint32_t CornerTangents{}, CornerColors{}, CornerUvs{};
        uint32_t AdjacencyWords{}, ConnectivityWords{};
    } Pending{};

    uint32_t AcquireId(Entry &&);
    // Slices connectivity at access time because later arena growth invalidates prior spans.
    ConnectivityStorage SliceConnectivity(const Entry &) const;
    // GetCornerNormals with the mesh's triangulated index stream already at hand.
    std::span<const vec3> GetCornerNormals(const Mesh &, std::span<const uint32_t> indices) const;
    // Fill the base vertex-normal mirror over `vertices`: a face-less mesh's point normals, zero otherwise (triangle meshes rederive the region).
    void FillBaseVertexNormalMirror(Range vertices, Range point_normals);
    void BuildVertexAdjacency(const Mesh &);
    Range AllocateVertices(uint32_t count);
    Range AllocateFaces(uint32_t count);
};
