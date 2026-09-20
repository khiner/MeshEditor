#pragma once

#include "numeric/uvec2.h"
#include "numeric/uvec4.h"
#include "numeric/vec2.h"

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
#include "gpu/EditSharpnessOperation.h"
#include "gpu/MorphTargetVertex.h"
#include "metal/BufferArena.h"

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

// Contains record-relative corner-normal sources for one pose.
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
};

// Corner-domain attribute layers in triangulated fan order, empty where the source lacks the channel.
struct CornerLayers {
    std::vector<vec4> Tangents, Colors;
    std::array<std::vector<vec2>, 4> Uvs;
};

// Every mesh arena, one per GPU-readable stream.
// A mirror arena holds one value per element of the arena it mirrors, at the same ranges.
struct MeshArenas {
    explicit MeshArenas(mtl::BufferContext &);

    BufferArena<Vertex> Vertices;
    BufferArena<uint32_t> FaceFirstTriangles; // Per-face index of the face's first triangle in the index buffer
    BufferArena<uint8_t> FaceSharpness; // Mirrors FaceFirstTriangles, 1 = flat-shaded face (canonical sharpness store)
    BufferArena<uint32_t> FaceCorners; // Canonical corner vertex indices shared by connectivity and face drawing
    BufferArena<uint32_t> TriangleFaceIds; // 1-indexed map from face triangles (in mesh face order) to source face ID
    BufferArena<uint32_t> Connectivity; // Each mesh's half-edge connectivity, laid out as the record's sub-ranges describe
    BufferArena<uint32_t> SelectionBits; // Compact edit selection: three domain masks per mesh
    BufferArena<EditSelectionSummary> SelectionSummary; // One summary per mesh
    BufferArena<uint8_t> EdgeSharpness; // One byte per edge, 1 = sharp (canonical sharpness store)
    BufferArena<uvec2> CustomCornerMasks; // Custom corner-normal presence: a (bitset word, exclusive rank) pair per 32 corners
    BufferArena<vec2> CustomCornerNormals; // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the masked corners
    BufferArena<vec3> PointNormals; // Authored normals of face-less meshes, in vertex order
    BufferArena<vec4> CornerTangents; // Corner-domain attribute layers, one value per corner in fan order
    BufferArena<vec4> CornerColors;
    BufferArena<vec2> CornerUvs; // Up to four ranges per mesh, one per UV set
    BufferArena<uint32_t> ElementPrimitives; // Source primitive index per drawn element (per face, or per vertex for point/line meshes)
    BufferArena<uint32_t> PrimitiveMaterials; // Primitive index -> material index
    BufferArena<BoneDeformVertex> BoneDeform;
    BufferArena<MorphTargetVertex> MorphTargets;
    // Canonical tetrahedral wireframe geometry, one range per mesh that carries a modal solve.
    BufferArena<vec3> TetPositions;
    BufferArena<uint32_t> TetEdgeIndices; // Two indices per tet edge
    // Excitable vertex handles per sounding mesh, rebuilt from the sound model after serialization.
    BufferArena<uint32_t> SoundVertices;
    // Transient index lists an operator reads during its run.
    BufferArena<uint32_t> Lists;
    // Stores CSR offsets followed by items for vertex-triangle, vertex-edge, and corner-sector incidence.
    BufferArena<uint32_t> Adjacency;
    BufferArena<uint32_t> CornerClasses; // Per-corner CornerClass values, from the sharpness stores
    BufferArena<vec3> BaseSeamNormals; // Composed sector normal per seam corner
    BufferArena<uint32_t> SelectionBaseline; // Gesture baseline masks, one range per mesh
    BufferArena<vec3> BaseVertexNormals; // Mirrors Vertices: derived smooth normals for triangle meshes, authored normals for face-less meshes
    BufferArena<vec3> BaseFaceNormals; // Mirrors FaceFirstTriangles, one derived face normal per face slot
};

// Bindless slots of the arenas shaders address, fixed for the store's lifetime.
struct MeshSlots {
    uint32_t Vertices, FaceFirstTriangle, FaceSharpness, SelectionBits, EdgeSharpness;
    uint32_t CustomCornerMask, CustomCornerNormal, CornerTangent, CornerColor, CornerUv;
    uint32_t ElementPrimitive, PrimitiveMaterial, BoneDeform, MorphTarget;
    uint32_t TetPosition, TetEdgeIndex, SoundVertex;
    uint32_t Adjacency, CornerClass, BaseSeamNormal, BaseVertexNormal, BaseFaceNormal;
};

// Calls `fn(index)` for each set bit below `count`, in ascending order.
void ForEachSelected(std::span<const uint32_t> bits, uint32_t count, auto &&fn) {
    const uint32_t last_word = (count + 31) / 32;
    for (uint32_t w = 0; w < last_word; ++w) {
        uint32_t word = bits[w];
        while (word) {
            const uint32_t handle = w * 32 + __builtin_ctz(word);
            if (handle < count) fn(handle);
            word &= word - 1;
        }
    }
}

// True when the mesh has faces.
// A face mesh builds its fan and edge adjacency tables on the GPU.
bool BuildsAdjacencyOnGpu(const Mesh &);

// The corner normal a class value selects from the sources: the face normal, a seam sector normal, or the vertex normal.
vec3 ComposeCornerNormal(std::span<const uint32_t> classes, CornerClass uniform_class, uint32_t ci, std::span<const uint32_t> indices, std::span<const uint32_t> face_ids, const CornerNormalSources &);

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
// Reads go through the records and arenas, writes through the explicit mutators, which capture history pages first.
struct MeshStore {
    explicit MeshStore(mtl::BufferContext &);
    ~MeshStore();
    mtl::BufferContext &BufferContext() const { return Buffers.Vertices.Buffer.Ctx; }

    static constexpr uint32_t MaxUvSets{4};

    enum ChangeBits : uint32_t {
        GeometryChanged = 1u << 0,
        TopologyChanged = 1u << 1,
        SelectionChanged = 1u << 2,
        ShadingChanged = 1u << 3,
        AttributesChanged = 1u << 4,
        DeformChanged = 1u << 5,
        EntryChanged = 1u << 6,
    };
    struct Change {
        uint32_t StoreId, Bits;
        std::vector<Range> VertexRanges{};
    };

    // One mesh's persistent arena ranges and counts.
    struct Record {
        Range Vertices{};
        Range FaceData{}; // Per-face range shared by the FaceFirstTriangles, FaceSharpness and BaseFaceNormals arenas
        Range CustomCornerMasks{}, CustomCornerNormals{};
        Range CornerTangents{}, CornerColors{};
        std::array<Range, MaxUvSets> CornerUvs{};
        Range EdgeSharpness{};
        Range TriangleFaceIds{}, ElementPrimitives{}, PrimitiveMaterials{}, FaceCorners{};
        std::array<Range, 3> SelectionBits{}; // vertex, edge, face masks
        Range SelectionSummary{};
        // The mesh's half-edge connectivity, laid out in the order SliceConnectivity reads it.
        Range Connectivity{};
        uint32_t ConnectivityVertices{}, ConnectivityHalfedges{}, ConnectivityEdgeCount{}, ConnectivityFaces{};
        bool ConnectivityFaceStarts{false}; // An n-gon mesh stores each face's first halfedge.
        Range PointNormals{};
        Range BoneDeform{}, MorphTargets{};
        uint32_t MorphTargetCount{0};
        uint32_t TriangleCount{0};
        // Whether the source authored vertex normals, so shading may stay authored under morphing (glTF semantics).
        bool HasAuthoredNormals{false};
        std::vector<float> DefaultMorphWeights{};
        std::vector<PrimitiveTriangleRange> PrimitiveTriangleRanges{};
        bool Alive{false};
    };

    // One mesh's derived arena ranges, rebuilt from connectivity and the sharpness stores after a restore.
    struct DerivedRecord {
        Range CornerClasses{}, SelectionBaseline{};
        // CSR offsets followed by incident items.
        Range VertexFanAdjacency{}, VertexEdgeAdjacency{}, SeamFans{};
        Range BaseSeamNormals{};
        uint32_t SeamCornerCount{};
        CornerClass UniformCornerClass{CornerClass::Vertex};
        bool MorphShadingAuthored{};
    };

    // Output element counts of a topology operator.
    struct TopologyCounts {
        uint32_t Vertices, Halfedges, Faces;
    };
    // Acquires an output record with `source`'s metadata, its ranges allocated at the counts and captured for the GPU writes that fill them.
    // Custom normals are allocated at the source's count until CompleteTopologyOutput trims them.
    uint32_t BeginTopologyOutput(uint32_t source, const TopologyCounts &);
    // Completes the output once its connectivity and edge attributes exist: trims edge and custom normal storage, derives primitive ranges, adjacency, and corner classes.
    void CompleteTopologyOutput(uint32_t id, uint32_t custom_corner_normals);

    void Track(store::History &);
    void FinishRestore();
    std::vector<Change> TakeChanges();
    // Records whose render data is stale, released or changed by a restore, for the render sync to drop.
    std::vector<uint32_t> TakeRenderStale() { return std::exchange(RenderStale, {}); }

    // Capture destination pages before dispatching GPU writes to Persistent mesh data.
    void CaptureVertexEdit(uint32_t id);
    void CaptureSelectionWrite(uint32_t id);
    void CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation);
    void CaptureConnectivityWrite(uint32_t id);
    void CaptureWeldWrite(uint32_t id);

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
    // Allocates connectivity storage from source counts in call order, with the edge list at its halfedge bound.
    // `face_offsets` fills the face starts of a mesh whose faces are not all triangles, and is empty when a GPU pass writes them.
    void AllocateConnectivity(uint32_t id, uint32_t vertex_count, uint32_t halfedge_count, uint32_t face_count, bool face_starts, std::span<const uint32_t> face_offsets = {});
    ConnectivityStorage GetConnectivityStorage(uint32_t id);
    // Completes a build: records the edge count and trims the edge list to it.
    void FinishConnectivity(uint32_t id, uint32_t edge_count);
    MeshConnectivity GetConnectivity(uint32_t id) const;
    // Completes the record created by CreateMeshSource: face tables, corner layers, primitive tables, smooth sharpness stores, and adjacency.
    void CreateMesh(uint32_t id, const MeshData &, const MeshVertexAttributes &, const MeshPrimitives &, const CornerLayers &, bool has_authored_normals);
    // Returns the clone's store ID.
    uint32_t CloneMesh(const Mesh &);
    // Returns a vertex-only store ID that must be released with Release.
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &);
    void Release(uint32_t id);
    // Reset all arenas and the StoreId table to empty, keeping GPU allocations for reuse.
    // Requires a full scene clear without live StoreId references so allocation restarts deterministically.
    void Clear();
    // Rebuilds derived data in store-ID order and sorts the input span in place.
    void RebuildDerived(std::span<Mesh>);

    const Record &Get(uint32_t id) const { return Records.at(id); }
    const DerivedRecord &GetDerived(uint32_t id) const { return DerivedRecords.at(id); }
    const MeshArenas &Arenas() const { return Buffers; }
    const MeshSlots &Slots() const { return SlotTable; }

    // Mutable views over Persistent arena data, capturing the pages they expose.
    std::span<Vertex> EditVertices(uint32_t id);
    std::span<uint32_t> EditPrimitiveMaterials(uint32_t id);
    // Callers writing the sharpness stores rederive corner normals afterward.
    std::span<uint8_t> EditFaceSharpness(uint32_t id);
    std::span<uint8_t> EditEdgeSharpness(uint32_t id);
    // Installs the custom corner-normal layer: one mask pair per 32 corners and the offsets packed to the masked corners.
    void SetCustomCornerNormals(uint32_t id, std::span<const uvec2> masks, std::span<const vec2> packed);
    void SetMorphShadingAuthored(uint32_t id, bool);

    // Copies tetrahedral wireframe geometry into GPU-only canonical arenas.
    TetBuffers AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices);
    void ReleaseTets(TetBuffers);
    Range AllocateSoundVertices(std::span<const uint32_t>);
    void ReleaseSoundVertices(Range);
    Range AllocateList(std::span<const uint32_t>);
    void ReleaseList(Range);

    // Allocates compact masks for every element domain, of which the GPU derives two from the authoritative domain.
    void EnsureSelectionBits(const Mesh &);
    std::span<const uint32_t> GetSelectionBits(uint32_t id, Element) const;
    uint32_t GetSelectionBitOffset(uint32_t id, Element) const;
    SlottedRange GetSelectionBitsRange(uint32_t id, Element) const;
    SlottedRange GetSelectionBaselineRange(uint32_t id) const;
    EditSelectionStorage GetEditSelectionStorage(uint32_t id) const;
    const EditSelectionSummary &GetSelectionSummary(uint32_t id) const;

    SharpnessSummary GetFaceSharpnessSummary(uint32_t id) const;
    // Returns the class-buffer offset or a uniform-class sentinel.
    uint32_t GetCornerClassOffset(uint32_t id) const;
    // Returns composed corner normals in triangulated face-fan order, in scratch storage valid until the next call.
    // Requires current base stores (the derive pass ran since the last position/sharpness write).
    std::span<const vec3> GetCornerNormals(const Mesh &) const;
    // GetCornerNormals with the mesh's triangulated index stream already at hand.
    std::span<const vec3> GetCornerNormals(const Mesh &, std::span<const uint32_t> indices) const;
    // Classify each corner from the sharpness stores: vertex-normal, face-normal, or a seam sector of incident triangles.
    // Call after any sharpness write, then run the base derive pass to refill the base normal stores.
    void UpdateCornerClassification(const Mesh &);
    // CSR vertex-to-edge incidence, edge items in edge order.
    VertexAdjacency GetVertexEdgeAdjacency(uint32_t id) const;

private:
    MeshArenas Buffers;
    MeshSlots SlotTable;
    std::vector<Record> Records{};
    std::vector<DerivedRecord> DerivedRecords{};
    std::vector<uint32_t> FreeIds{};
    std::vector<uint32_t> RenderStale{};

    struct HistoryState;
    std::unique_ptr<HistoryState> Tracked;

    Record &WriteRecord(uint32_t id);
    void ReleaseDerived(uint32_t id);
    uint32_t AcquireId(Record &&);
    // Size every mirror arena to the record's master ranges.
    void SyncMirrors(uint32_t id);
    // Fill the base vertex-normal mirror over `vertices`: a face-less mesh's point normals, zero otherwise (triangle meshes rederive the region).
    void FillBaseVertexNormalMirror(Range vertices, Range point_normals);
    void BuildVertexAdjacency(const Mesh &);
};
