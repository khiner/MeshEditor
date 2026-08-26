#pragma once

#include "Mesh.h"
#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "TetBuffers.h"
#include "Range.h"
#include "SlottedRange.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/CornerClass.h"
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

// One pose's per-class corner-normal sources, each span entry-relative: the base stores for the rest pose, or the normals derived from a morph target's full-weight pose.
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

// What creating a mesh derives from the source alone. Deriving it touches no arena and no store
// state, so a batch derives every mesh at once and the arena work stays in source order.
struct PreparedMesh {
    std::vector<vec4> CornerTangents, CornerColors;
    std::array<std::vector<vec2>, 4> CornerUvs;
    std::vector<vec3> AuthoredCornerNormals;
    // Target-major morph tangent deltas, the one vertex-domain channel no arena holds.
    // The weld compares and compacts them alongside the arena channels, and CreateMesh hands them back.
    std::vector<vec3> MorphTangentDeltas;
};

// Order the faces by primitive and gather the corner-domain channels out of `attrs`, leaving the
// vertex domain to the weld. `data`, `attrs` and `primitives` are left in the state CreateMesh expects.
// Welding recovers authored normals as face sharpness on faceted faces and as a custom corner-normal layer where they deviate from derivation.
PreparedMesh PrepareMeshSources(MeshData &, MeshVertexAttributes &, MeshPrimitives &);
BuiltConnectivity BuildPreparedConnectivity(const MeshStore &, uint32_t id, const MeshData &, const ConnectivityStorage &);

// Whether a mesh's vertex-fan CSR fills on the GPU rather than in the store.
// Every face being a triangle is what makes a halfedge's face and loop position arithmetic.
bool BuildsFanAdjacencyOnGpu(const Mesh &);
// Whether a mesh's vertex-edge CSR fills on the GPU rather than in the store.
// It also takes every edge carrying at most two halfedges, which is what ranks an edge index from the edge-first bits.
bool BuildsEdgeAdjacencyOnGpu(const Mesh &);

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mtl::BufferContext &);
    ~MeshStore();
    MeshStore(MeshStore &&) noexcept;
    MeshStore &operator=(MeshStore &&) noexcept;

    // Call PlanCreate or PlanClone per mesh, then CommitReserves once, before the operations themselves.
    void PlanCreate(const MeshData &, const MeshPrimitives & = {}, bool has_deform = false, uint32_t morph_target_count = 0, const MeshVertexAttributes & = {});
    void PlanClone(const Mesh &);
    // Reserve all arenas for accumulated plans, then reset.
    void CommitReserves();

    // Take the mesh's source positions and corner array into the arenas, where every derive reads them,
    // and return the store id the mesh keeps. CreateMesh finishes the same entry.
    // A weld compacts the vertex range to its front, so it ends with ShrinkMeshSource.
    uint32_t CreateMeshSource(const MeshData &);
    // Take the mesh's skin and morph channels into their arenas at the source vertex count, where the
    // weld compares them alongside the positions and compacts them in place.
    void CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &, const std::optional<MorphTargetData> &);
    // Trim every vertex-domain arena run to what a weld left: the positions, the skin channels, and
    // each morph target's deltas at their new stride.
    void ShrinkMeshSource(uint32_t id, uint32_t welded_vertices);
    // The mesh's half-edge connectivity, as views into the arena that holds it.
    MeshConnectivity GetConnectivity(uint32_t id) const;
    // Take the storage a connectivity build fills, sized from the source counts, and place what the
    // build hands back. Both run in source order, with the build itself free to run alongside others.
    void AllocateConnectivity(uint32_t id, uint32_t vertex_count, uint32_t halfedge_count, uint32_t face_count, bool face_starts);
    ConnectivityStorage GetConnectivityStorage(uint32_t id);
    // The arena run holding the mesh's connectivity, which a GPU build fills in place.
    SlottedRange GetConnectivityRange(uint32_t id) const;
    // Record the edge count a GPU build's ranks totalled.
    void SetConnectivityEdgeCount(uint32_t id, uint32_t edge_count);
    void PlaceConnectivity(uint32_t id, const BuiltConnectivity &);

    // Takes what PrepareMeshSources derived, and everything it left in place, into the arenas.
    // `id` comes from CreateMeshSource, which already took the vertex and corner arrays into the arenas.
    CreatedMesh CreateMesh(uint32_t id, MeshData &&, MeshVertexAttributes &&, MeshPrimitives &&, PreparedMesh &&, bool flat_shaded = false);
    CreatedMesh CloneMesh(const Mesh &);

    // Allocate vertex-only store entry (no topology, no face/edge/primitive/material buffers).
    // Returns the store id, released via Release(storeId).
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs);

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

    // Base bindless slots of the per-mesh GPU buffers (for shader push constants).
    uint32_t GetVertexStateSlot() const;
    uint32_t GetCornerTangentSlot() const;
    uint32_t GetCornerColorSlot() const;
    uint32_t GetCornerUvSlot() const;
    uint32_t GetEdgeSharpnessSlot() const;
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
    // Copies tet wireframe geometry into the canonical arenas. Nothing on the host reads it back.
    TetBuffers AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices);
    void ReleaseTets(TetBuffers);
    uint32_t GetSoundVertexSlot() const;
    Range AllocateSoundVertices(std::span<const uint32_t>);
    void ReleaseSoundVertices(Range);
    std::span<const uint32_t> GetSoundVertices(Range) const;

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
    // Rebuild the mesh's CSR tables and report the first entry differing from the stored ones, or empty when they match.
    std::string CheckVertexAdjacency(const Mesh &) const;
    Range GetVertexFanAdjacencyRange(uint32_t id) const { return Entries.at(id).VertexFanAdjacency; }
    Range GetVertexEdgeAdjacencyRange(uint32_t id) const { return Entries.at(id).VertexEdgeAdjacency; }
    // The class-buffer offset, or a sentinel when the mesh stores none (InvalidOffset = every corner Vertex, UniformFaceOffset = every corner Face).
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
    std::span<const vec3> GetBaseSeamNormals(uint32_t id) const;
    std::span<vec3> GetBaseSeamNormals(uint32_t id);
    // Authored normals of face-less meshes, in vertex order (empty when the mesh has none).
    std::span<const vec3> GetPointNormals(uint32_t id) const;
    SlottedRange GetFaceStateRange(uint32_t id) const;
    SlottedRange GetEdgeStateRange(uint32_t id) const;
    Range GetEdgeSharpnessRange(uint32_t id) const;
    // Corner-domain attribute layers (one value per triangulated face corner, fan order).
    // Empty range/span when the mesh lacks the channel.
    static constexpr uint32_t MaxUvSets{4}; // Texture coordinate sets an entry stores, so a higher glTF TEXCOORD_n has nowhere to land.
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
    // The mesh's corner vertex indices, one per halfedge. Canonical: the connectivity and the face
    // index buffer both read this, and a triangle mesh's draws index it directly.
    std::span<const uint32_t> GetFaceCorners(uint32_t id) const;
    SlottedRange GetFaceCornerRange(uint32_t id) const;
    // Allocate and clear a mesh's edge selection states, which only the wireframe and edit overlays read.
    void EnsureEdgeStates(const Mesh &);
    // Take the mesh's element selection bits, one bit per element, sized to its largest element domain
    // so every edit mode indexes the same range. Keeps the bits a mesh already has.
    void EnsureSelectionBits(const Mesh &);
    std::span<const uint32_t> GetSelectionBits(uint32_t id) const;
    std::span<uint32_t> GetSelectionBits(uint32_t id);
    // The selection bits' bindless slot, and a mesh's first bit within it, which the element rasters
    // and the state kernel index by.
    uint32_t GetSelectionBitsSlot() const;
    uint32_t GetSelectionBitOffset(uint32_t id) const;
    // Zero a mesh's vertex, face, and edge states.
    void ClearElementStates(const Mesh &);
    std::span<const uint32_t> GetFaceFirstTriangles(uint32_t id) const;
    std::span<const uint32_t> GetElementPrimitiveIndices(uint32_t id) const;
    std::span<uint32_t> GetElementPrimitiveIndices(uint32_t id);
    std::span<const uint32_t> GetPrimitiveMaterialIndices(uint32_t id) const;
    std::span<uint32_t> GetPrimitiveMaterialIndices(uint32_t id);

    std::span<const PrimitiveTriangleRange> GetPrimitiveTriangleRanges(uint32_t id) const { return Entries.at(id).PrimitiveTriangleRanges; }

    void UpdateEdgeStatesFromFaces(const Mesh &, std::optional<uint32_t> active_face);
    // Mark `vertices` selected, clear every other element state, and derive the edge states from the selection.
    void UpdateSoundVertexStates(const Mesh &, std::span<const uint32_t> vertices);
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

    // Reset all arenas and the StoreId table to empty, keeping GPU allocations for reuse.
    // Call only on a full scene clear, where no live entity references a StoreId, so StoreId and offset
    // allocation restarts deterministically.
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
        Range Vertices{};
        Range FaceData{}; // Per-face range shared by FaceFirstTriangleBuffer and FaceStateBuffer
        Range CornerClasses{}; // One CornerClass value per corner from the sharpness stores, empty when every corner takes UniformCornerClass
        Range CustomCornerMasks{}; // Custom corner-normal presence: a (bitset word, exclusive rank) pair per 32 corners
        Range CustomCornerNormals{}; // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the masked corners
        Range CornerTangents{}, CornerColors{}; // Corner-domain attribute layers
        std::array<Range, MaxUvSets> CornerUvs{};
        Range EdgeSharpness{}; // One byte per edge, 1 = sharp
        Range EdgeStates{}, TriangleFaceIds{}, ElementPrimitives{}, PrimitiveMaterials{}, FaceCorners{};
        Range SelectionBits{}; // Element selection bits, one word per 32 elements of the largest element domain
        // CSR vertex incidence, each range holding (vertex count + 1) offsets followed by the items
        Range VertexFanAdjacency{}, VertexEdgeAdjacency{};
        // The mesh's half-edge connectivity, laid out in the order SliceConnectivity reads it.
        Range Connectivity{}, ConnectivityEdges{}, ConnectivityHalfedgeToEdge{};
        uint32_t ConnectivityVertices{}, ConnectivityHalfedges{}, ConnectivityEdgeCount{}, ConnectivityFaces{};
        bool ConnectivityFaceStarts{false}; // An n-gon mesh stores each face's first halfedge.
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

    struct PendingReserves {
        uint32_t Vertices{}, Faces{}, Triangles{}, Edges{}, EdgeStates{}, FaceCorners{};
        uint32_t Primitives{};
        uint32_t ElementPrimitiveIndices{}; // One per face, or per vertex for point and line meshes.
        uint32_t BoneDeformVertices{}, MorphTargetEntries{};
        uint32_t CornerTangents{}, CornerColors{}, CornerUvs{};
        uint32_t AdjacencyWords{}, ConnectivityWords{};
    } Pending{};

    uint32_t AcquireId(Entry &&);
    // The entry's connectivity sub-spans, in the order its run lays them out. Sliced when a build or
    // a read runs rather than when the range is taken, since a later allocation can grow the arena
    // out from under a span.
    ConnectivityStorage SliceConnectivity(const Entry &) const;
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
