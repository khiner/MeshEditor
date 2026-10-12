#pragma once

#include "Mesh.h"
#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "Range.h"
#include "TetBuffers.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/ClusterGroup.h"
#include "gpu/ClusterGroupLinks.h"
#include "gpu/ConnectivityRef.h"
#include "gpu/CornerClass.h"
#include "gpu/CornerClassMode.h"
#include "gpu/CornerClassificationPushConstants.h"
#include "gpu/CustomNormal.h"
#include "gpu/EditSelectionStorage.h"
#include "gpu/EditSelectionSummary.h"
#include "gpu/EditSharpnessOperation.h"
#include "gpu/Element.h"
#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/LodNode.h"
#include "gpu/MeshRecord.h"
#include "gpu/MeshletPayloads.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshletSpatialNode.h"
#include "gpu/MorphTargetVertex.h"
#include "gpu/NormalSector.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/SelectionAggregate.h"
#include "gpu/SelectionUpdatePushConstants.h"
#include "gpu/SlotOffset.h"
#include "mesh/CornerNormalView.h"
#include "mesh/ElementArena.h"
#include "mesh/ElementAttribute.h"
#include "mesh/ElementAttributeView.h"
#include "mesh/GeometrySelection.h"
#include "mesh/NamedAttribute.h"
#include "mesh/PoseAttributeView.h"
#include "mesh/SelectionQuery.h"
#include "mesh/SelectionView.h"
#include "mesh/VertexFanStore.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"
#include "numeric/uvec2.h"
#include "numeric/uvec4.h"
#include "numeric/vec2.h"
#include "project/store/Hash.h"

#include <expected>
#include <mutex>

namespace mtl {
struct ComputeChain;
}
namespace state {
struct Scene;
}
struct CloneCopies;

struct ArmatureDeformData {
    std::vector<uvec4> Joints;
    std::vector<vec4> Weights;
};

struct SharpnessSummary {
    bool Any, All;
};

// Borrows canonical normal arrays and optional sparse overrides for one pose.
struct CornerNormalSources {
    std::span<const vec3> VertexNormals, FaceNormals;
    PoseAttributeView<vec3> PosedVertexNormals, PosedSectorNormals, PosedFaceNormals;
};

// Per-source-primitive metadata, with every vector indexed by primitive.
struct MeshPrimitives {
    std::vector<uint32_t> ElementPrimitiveIndices{}; // source primitive index per drawn element (per face, or per vertex for point/line meshes)
    std::vector<uint32_t> MaterialIndices{};
    std::vector<uint32_t> AttributeFlags{}; // bitmask of MeshAttributeBit_*
};

// Corner-domain layers in polygon loop order followed by wire endpoint order, empty for absent channels.
struct CornerLayers {
    std::vector<vec4> Tangents, Colors;
    std::array<std::vector<vec2>, 4> Uvs;
};

// The render collections a record owns as block sets: finest and coarse clusters, traversal primitives and nodes, cluster groups, and the leaf member pool.
enum class RenderSet : uint8_t { Meshlets,
                                 Primitives,
                                 Nodes,
                                 Groups,
                                 Members };
constexpr std::array RenderSets{RenderSet::Meshlets, RenderSet::Primitives, RenderSet::Nodes, RenderSet::Groups, RenderSet::Members};
// The payload pools a record owns a range of: its clusters' source elements, vertex corners and local triangles, and its groups' cluster runs.
enum class RenderPayload : uint8_t { TriangleIds,
                                     VertexCorners,
                                     LocalTriangles,
                                     GroupClusters };
constexpr std::array RenderPayloads{RenderPayload::TriangleIds, RenderPayload::VertexCorners, RenderPayload::LocalTriangles, RenderPayload::GroupClusters};
// A record's range of one payload pool: its used values from the range's start, within a capacity that grows as values append.
// Every stored payload offset is relative to the range, so the range moves without a fixup.
struct PayloadRange {
    Range Values{};
    uint32_t Capacity{};
};

// A run of new members of a render set, consecutive in handles and in arena addresses.
struct RenderRun {
    ElementHandle Handle{};
    ElementAddress Address{};
    uint32_t Count{};
};

// The render arenas store records own: clusters and their payloads, the cluster LOD DAG and its traversal nodes, element owners and GPU mesh records.
// Each collection is a block domain with a set per record, so every reference among render values is a handle relative to its record.
struct RenderArenas {
    explicit RenderArenas(mtl::BufferContext &);

    BufferArena<uint32_t> ExtrasFaces, ExtrasEdges; // Bone and joint index data, as each record's vertex handles
    ElementArena<MeshletRecord, RenderBlockSize> Meshlets;
    BufferArena<MeshletSpatialNode> MeshletSpatialNodes; // Mirrors Meshlets
    // Mirrors Meshlets: each cluster's traversal leaf, read only by repairs.
    BufferArena<uint32_t> MeshletLodLeaves;
    // A mask word per block of Meshlets, a bit per slot: finest clusters whose canonical positions changed since coarse repair.
    BufferArena<uint32_t> PositionDirty;
    // Each cluster's payload runs: the element handles a finest cluster draws, its vertex corner handles, and a triangle cluster's local triangles.
    BufferArena<uint32_t> MeshletTriangleIds, MeshletVertexCorners;
    BufferArena<MeshletTriangle> MeshletLocalTriangles;
    // The cluster LOD DAG: one group per simplification step, and the selection forest over them.
    ElementArena<ClusterGroup, RenderBlockSize> ClusterGroups;
    // Mirrors ClusterGroups: a group's inputs and proxies, as runs of GroupClusterIds that the group owns.
    BufferArena<ClusterGroupLinks> GroupLinks;
    // A mask word per block of ClusterGroups, a bit per slot: stale groups, whose coarse clusters rebuild when the mesh leaves edit mode.
    BufferArena<uint32_t> DirtyGroups;
    BufferArena<uint32_t> GroupClusterIds; // Meshlet handles
    ElementArena<LodNode, RenderBlockSize> LodNodes;
    BufferArena<uint32_t> LodParents; // Mirrors LodNodes
    // Each traversal leaf's clusters, as a run of meshlet handles.
    // Headroom of live members holding InvalidOffset follows each run, and the run grows in place over it.
    ElementArena<uint32_t, RenderBlockSize> LodMembers;
    ElementArena<PrimitiveRecord, RenderBlockSize> Primitives;
    // Each render owner's run of render primitive handles, indexed by source primitive and InvalidOffset where absent.
    BufferArena<uint32_t> PrimitiveRoutes;
    BufferArena<MeshRecord> MeshRecords; // One GPU mesh record per store id, rebuilt from the record's current bindings
    // Each record's triangle, loose edge and isolated vertex handles map to the meshlet handles of the clusters drawing them.
    std::array<ElementAttribute<uint32_t>, 3> ElementMeshlets;

    // Adds a run of `count` members to the set, creating it when absent, and sizes the mirrors over any new blocks.
    RenderRun Insert(RenderSet, ElementSetRef &, uint32_t count);
    // Lists the ascending clusters as a new run of the member set with headroom for half as many more, and returns its first handle, or InvalidOffset for none.
    uint32_t ListMembers(ElementSetRef &, std::span<const uint32_t> clusters);
    // Sizes each mirror over its collection's blocks.
    void SyncMirrors();
    // The slots kernels read cluster payloads through.
    MeshletPayloadSlots PayloadSlots() const;
};
// A render collection's per-block mask holds a word per block, with the bit of each slot.
bool MaskHas(const BufferArena<uint32_t> &, ElementAddress);
void MaskSet(BufferArena<uint32_t> &, ElementAddress);
void MaskClear(BufferArena<uint32_t> &, ElementAddress);
// Visits the arena of one render collection.
decltype(auto) WithRenderSet(auto &render, RenderSet set, auto &&fn) {
    switch (set) {
        case RenderSet::Meshlets: return fn(render.Meshlets);
        case RenderSet::Primitives: return fn(render.Primitives);
        case RenderSet::Nodes: return fn(render.LodNodes);
        case RenderSet::Groups: return fn(render.ClusterGroups);
        case RenderSet::Members: return fn(render.LodMembers);
    }
    std::unreachable();
}

// The value blocks of every named attribute of each value type, where each attribute owns one range.
struct AttributeArenas {
    explicit AttributeArenas(mtl::BufferContext &);
    BufferArena<AttributeBlock<float>> Floats;
    BufferArena<AttributeBlock<vec2>> Vec2s;
    BufferArena<AttributeBlock<vec3>> Vec3s;
    BufferArena<AttributeBlock<vec4>> Vec4s;
    BufferArena<AttributeBlock<int32_t>> Ints;
    BufferArena<AttributeBlock<bool>> Bools;
};
template<AttributeScalar T> auto &ValuesOf(auto &arenas) {
    if constexpr (std::same_as<T, float>) return arenas.Floats;
    else if constexpr (std::same_as<T, vec2>) return arenas.Vec2s;
    else if constexpr (std::same_as<T, vec3>) return arenas.Vec3s;
    else if constexpr (std::same_as<T, vec4>) return arenas.Vec4s;
    else if constexpr (std::same_as<T, int32_t>) return arenas.Ints;
    else return arenas.Bools;
}
// Visits the value arena of one value type.
decltype(auto) WithAttributeValues(auto &arenas, AttributeType type, auto &&fn) {
    return WithAttributeScalar(type, [&]<AttributeScalar T>() -> decltype(auto) { return fn(ValuesOf<T>(arenas)); });
}

// Every mesh arena, one per GPU-readable stream.
// Mirrors use canonical element or block indices from their owning domain.
struct MeshArenas {
    explicit MeshArenas(mtl::BufferContext &);

    ElementArena<Vertex> Vertices;
    ElementArena<uint32_t> FaceTriangles; // Canonical first triangle address, and count is the face loop length minus two
    BufferArena<uint8_t> FaceSharpness; // Mirrors FaceTriangles, 1 = flat-shaded face (canonical sharpness store)
    ElementArena<uint32_t> FaceCorners; // Canonical corner vertex indices shared by connectivity and face drawing
    ElementArena<uvec3> Triangles; // Canonical polygon corners for each independently addressable derived triangle
    BufferArena<uint32_t> OutgoingHalfedges; // Mirrors Vertices
    BufferArena<uvec2> VertexCorners; // Mirrors Vertices: first packed fan item and count
    VertexFanStore VertexFans;
    BufferArena<uint32_t> OppositeHalfedges, HalfedgeEdges, HalfedgeFaces; // Mirror FaceCorners
    BufferArena<uvec2> FaceRanges; // Mirrors FaceTriangles
    ElementArena<uint32_t> EdgeHalfedges; // Owns the edge domain, and EdgeSharpness mirrors it
    BufferArena<SelectionBlock> VertexSelection, EdgeSelection, FaceSelection; // Canonical domain block masks
    BufferArena<SelectionBlock> VertexHidden, EdgeHidden, FaceHidden; // Persistent edit-mode visibility masks
    // Derived aggregates mirroring each selectable domain's blocks, rebuilt from masks, positions, sharpness and connectivity.
    BufferArena<SelectionAggregate> VertexAggregates, EdgeAggregates, FaceAggregates;
    SelectionIndex SelectionTree;
    SelectionQuery Query;
    BufferArena<EditSelectionSummary> SelectionSummary; // One summary per mesh
    BufferArena<uint8_t> EdgeSharpness; // One byte per edge, 1 = sharp (canonical sharpness store)
    // Element layers, each holding a range per record that carries it, indexed by the record's handles.
    ElementAttribute<CustomNormal> CustomNormals; // Optional angular offsets of polygon corners
    BufferArena<vec3> PointNormals; // Authored normals of face-less meshes, in vertex order
    ElementAttribute<vec4> CornerTangents, CornerColors, VertexColors;
    std::array<ElementAttribute<vec2>, 4> CornerUvs;
    ElementAttribute<uint32_t> FacePrimitives, VertexPrimitives; // Source primitive assignments
    BufferArena<uint32_t> PrimitiveMaterials; // Primitive index -> material index
    ElementAttribute<BoneDeformVertex> Skin;
    ElementAttribute<MorphTargetVertex> Morph; // One entry per target, so targets are target-major within each vertex block
    // Canonical tetrahedral wireframe geometry, one range per mesh that carries a modal solve.
    BufferArena<vec3> TetPositions;
    BufferArena<uint32_t> TetEdgeIndices; // Two indices per tet edge
    // Excitable vertex handles per sounding mesh, rebuilt from the sound model after serialization.
    BufferArena<uint32_t> SoundVertices;
    ElementAttribute<uint32_t> CornerSectors; // Sector-root corner handles, and InvalidOffset means the vertex normal
    ElementAttribute<NormalSector> NormalSectors; // Base normals at sector roots
    BufferArena<vec3> BaseVertexNormals; // Mirrors Vertices: derived smooth normals for triangle meshes, authored normals for face-less meshes
    BufferArena<vec3> BaseFaceNormals; // Mirrors FaceTriangles, one derived face normal per face slot
    AttributeArenas Named;
    RenderArenas Render;
};

// Bindless slots of the arenas shaders address, fixed for the store's lifetime.
struct MeshSlots {
    uint32_t Vertices, FaceTriangleStart, FaceSharpness, EdgeSharpness;
    uint32_t PrimitiveMaterial;
    uint32_t TetPosition, TetEdgeIndex, SoundVertex;
    uint32_t BaseVertexNormal, BaseFaceNormal;
};

// Composes face, vertex, or sector normals from canonical sharpness and root references.
// The connectivity resolves the record's handles to arena addresses.
vec3 ComposeCornerNormal(const MeshConnectivity &, CornerAttributeView<uint32_t> sectors, ElementAttributeView<NormalSector> normal_sectors, std::span<const uint8_t> sharpness, uint32_t mode, uint32_t ci, TriangleVertexView vertices, TriangleFaceView face_ids, const CornerNormalSources &);

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
// Reads go through the records and arenas, writes through the explicit mutators, which capture history pages first.
struct MeshStore {
    explicit MeshStore(mtl::BufferContext &);
    ~MeshStore();
    mtl::BufferContext &BufferContext() const { return Buffers.Vertices.Buffer.Ctx; }

    static constexpr uint32_t MaxUvSets{4};
    enum class ElementDomain { None,
                               Vertex,
                               Halfedge,
                               Edge,
                               Face,
                               Triangle };
    static constexpr std::array ElementDomains{ElementDomain::Vertex, ElementDomain::Halfedge, ElementDomain::Edge, ElementDomain::Face, ElementDomain::Triangle};

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
        std::vector<Range> VertexRanges{}; // The record's vertex handles whose restored positions changed
        // Ascending vertex, edge, face and halfedge blocks whose restored bytes changed.
        std::array<std::vector<uint32_t>, 4> Blocks{};
    };

    // Each element domain owns a set, whose table resolves the record-relative handles every stored reference holds.
    // Selection masks attach to the domain's blocks, and each element layer the record carries is a range of the layer's blocks indexed by its handles.
    // The render fields name the record's finest clusters, hierarchy, traversal nodes and GPU mesh record in the render arenas.
    struct Record {
        uint32_t StoreId{InvalidOffset}; // The record's own id, and InvalidOffset for a fragment outside the store.
        ElementSetRef Vertices{};
        ElementSetRef FaceData{}; // Shared by FaceRanges, FaceTriangles, FaceSharpness and BaseFaceNormals
        uint32_t CornerAttributes{}, VertexAttributes{};
        ElementSetRef EdgeData{};
        ElementSetRef TriangleData{};
        Range PrimitiveMaterials{};
        bool FacePrimitivesReady{}, VertexPrimitivesReady{};
        ElementSetRef FaceCorners{};
        Range SelectionSummary{};
        bool ConnectivityFaceStarts{false}; // The packed connectivity build input has explicit polygon ranges rather than implicit triangles.
        Range PointNormals{};
        bool SkinBlocksReady{false};
        bool MorphBlocksReady{false};
        uint32_t MorphTargetCount{0};
        uint32_t TriangleCount{0};
        // Whether the source authored vertex normals, so shading may stay authored under morphing (glTF semantics).
        bool HasAuthoredNormals{false};
        std::vector<float> DefaultMorphWeights{};
        // The record's range of each element layer it carries, covering its handle bound in the layer's domain, and empty for a layer it lacks.
        Range FacePrimitives{}, VertexPrimitives{}, Skin{}, Morph{}, CustomNormals{}, CornerTangents{}, CornerColors{}, VertexColors{};
        std::array<Range, 4> CornerUvs{};
        Range CornerSectors{}, NormalSectors{}; // Present while a corner can hold a sector root
        uint32_t Classification{uint32_t(CornerClassMode::UniformVertex)};
        bool MorphShadingAuthored{};
        Range ExtrasFaces{}, ExtrasEdges{}; // Bone and joint index data, present only on an extras record.
        // The record's set in each render collection, whose table resolves the record-relative handles every render value holds.
        ElementSetRef MeshletData{}, PrimitiveData{}, NodeData{}, GroupData{}, MemberData{};
        // The record's range of each payload pool, which its clusters' and groups' payload offsets are relative to.
        PayloadRange MeshletTriangles{}, MeshletVertices{}, MeshletLocalTriangles{}, GroupClusters{};
        // The meshlet handles the last build placed, in the meshlet set of the record's render owner.
        Range Meshlets{};
        uint32_t SpatialRoot{InvalidOffset}; // The meshlet handle at the root of the spatial tree.
        Range PrimitiveRoutes{}; // Three render primitive handles per source material, indexed by topology.
        uint64_t MeshletRevision{};
        uint32_t Level0Count{};
        uint32_t RenderTopologies{}; // Bit mask of published triangle, line and point topology; zero before the first build.
        // The element owner ranges of the record's triangles, loose edges and isolated vertices, and the count of local blocks in each holding an owner, which the range is present for.
        std::array<Range, 3> ElementMeshlets{};
        std::array<uint32_t, 3> ElementMeshletBlockCounts{};
        uint32_t LodDepth{}; // The depth of the record's deepest traversal tree.
        bool Alive{false};
    };

    // Transient interaction storage and pose-cache invalidation. Shading values
    // and their sparse ownership restore with the canonical mesh.
    struct DerivedRecord {
        // The source-domain mask blocks and active handle saved when an additive gesture began.
        std::vector<std::pair<uint32_t, SelectionBlock>> SelectionBaseline{};
        uint32_t SelectionBaselineActive{InvalidOffset};
        Element SelectionBaselineElement{Element::None};
        // One canonical corner per connected smooth normal sector.
        uint64_t NormalRevision{};
    };

    // Output element counts of a topology operator.
    struct TopologyCounts {
        uint32_t Vertices, Halfedges, Faces;
    };
    // Acquires an output record with `source`'s metadata, its ranges allocated at the counts and captured for the GPU writes that fill them.
    uint32_t BeginTopologyOutput(uint32_t source, std::span<const uint32_t> materials);

    // Adds live elements to a mesh domain for a topology emitter to fill, and returns their handles.
    // Without a list the handles form one run, and otherwise a list of nonconsecutive handles goes to a range of `list`, which needs a bindless slot.
    // New elements have their enabled attributes at defaults and no selection, and their value pages are captured.
    // After emitting values, refresh aggregates for affected selectable blocks before publishing the completed edit.
    ElementHandleRange InsertElements(uint32_t id, ElementDomain, uint32_t count, BufferArena<uint32_t> *list);
    // Clears the elements the work names, with their selection, and returns the blocks whose membership changed.
    // The caller refreshes those blocks' aggregates.
    std::vector<uint32_t> EraseElements(uint32_t id, ElementDomain, const BufferArena<uint32_t> &storage, ElementWork);
    // Clears the handles after the first `used` of an insertion made for a bound, listed in `list` when not a run, and shrinks `inserted` to the used ones.
    // The cleared handles never held selection.
    void TrimInsertedElements(uint32_t id, ElementDomain, ElementHandleRange &inserted, const BufferArena<uint32_t> &list, uint32_t used);

    void Track(store::History &);
    void FinishRestore();
    // Indexes the records written since the last index, so a restore maps its changed bytes to records in proportion to the change.
    void IndexHistory();
    // The records a restore changed, whose cached block lists refill on their next read.
    std::vector<Change> TakeChanges();
    // Records the refresh of the selection aggregates of restored blocks and their incident blocks.
    void ReconcileSelection(state::Scene &, mtl::ComputeChain &, std::span<const Change>);

    // Capture destination pages before dispatching GPU writes to Persistent mesh data.
    void CaptureVertexEdit(uint32_t id);
    void CaptureSelectionSummary(uint32_t id);
    void CaptureSelectionBlocks(Element, std::span<const uint32_t> blocks);
    void CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation, const GeometrySelection &);
    void CaptureConnectivityWrite(uint32_t id);
    void CaptureWeldWrite(uint32_t id);

    // Call once after all PlanCreate and PlanClone calls and before their corresponding operations.
    void PlanCreate(const MeshData &, const MeshPrimitives & = {}, bool has_deform = false, uint32_t morph_target_count = 0, const MeshVertexAttributes & = {});
    void PlanClone(const Mesh &);
    void CommitReserves();

    // Takes source positions and corners into the arenas, seeds connectivity storage, and returns their store ID.
    uint32_t CreateMeshSource(const MeshData &);
    // Takes skin and morph channels into arenas at the source vertex count for in-place welding.
    void CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &, const std::optional<MorphTargetData> &);
    // Trims all vertex-domain arena ranges to `welded_vertices`.
    void ShrinkMeshSource(uint32_t id, uint32_t welded_vertices);
    // Allocates connectivity storage from source counts in call order, with the edge list at its halfedge bound.
    // `face_offsets` fills the face starts of a mesh whose faces are not all triangles, and is empty when a GPU pass writes them.
    void AllocateConnectivity(uint32_t id, uint32_t halfedge_count, uint32_t face_count, bool face_starts, std::span<const uint32_t> face_offsets = {}, std::span<const std::array<uint32_t, 2>> wire_edges = {});
    // Completes a build: records the edge count and trims the edge list to it.
    void FinishConnectivity(uint32_t id, uint32_t edge_count);
    MeshConnectivity GetConnectivity(uint32_t id) const;
    ConnectivityRef GetConnectivityRef(uint32_t id) const;
    ElementTables GetElementTables(uint32_t id) const;
    // A mesh domain's blocks in local order and the inclusive live-element prefix over them.
    // It is rebuilt when first read after the domain's membership changes.
    struct BlockList {
        std::span<const uint32_t> Blocks, LivePrefix;
        SlotOffset Gpu; // The blocks, followed by the prefix.
    };
    BlockList GetBlockList(uint32_t id, ElementDomain) const;
    // A submitted frame reads the block lists until it completes, so a list replaced meanwhile keeps its words until then.
    void FrameSubmitted();
    void FrameCompleted();
    // The handle of the record's live element at an arena address, or none when the address names no live element of the record.
    std::optional<ElementHandle> LiveHandleAt(uint32_t id, Element, ElementAddress) const;
    // The arena address of one of the record's handles.
    ElementAddress HandleAddress(uint32_t id, ElementDomain, ElementHandle) const;
    // The arena blocks of the record's local blocks the finished work names.
    std::vector<uint32_t> ArenaBlocks(uint32_t id, ElementDomain, const BufferArena<uint32_t> &storage, ElementWork) const;
    // Packed ordinals count live elements in ascending handle order.
    ElementHandle LiveElementAt(uint32_t id, ElementDomain, uint32_t ordinal) const;
    uint32_t LiveElementOrdinal(uint32_t id, ElementDomain, ElementHandle) const;
    // The fan item runs that vertex root rebuilds free and allocate.
    VertexFanStore &VertexFans() { return Buffers.VertexFans; }
    TriangleCorners GetTriangleCorners(uint32_t id) const;
    // The mesh's live derived triangles and vertices in ordinal order.
    ElementView<uvec3> TriangleView(uint32_t id) const;
    ElementView<Vertex> VertexView(uint32_t id) const;
    // Completes the record created by CreateMeshSource: face tables, corner layers, primitive tables, and smooth sharpness stores.
    void CreateMesh(uint32_t id, const MeshData &, const MeshVertexAttributes &, const MeshPrimitives &, const CornerLayers &, bool has_authored_normals);
    // Each element domain's ascending blocks or local blocks, indexed by ElementDomain.
    using DomainBlocks = std::array<std::vector<uint32_t>, 6>;
    // Clones each source record, queueing its GPU copies on `copies`, and returns the clones' store IDs in source order.
    // A clone's tables have its source's shape over fresh blocks, so its handles name its source's elements at the same handles.
    // `render` copies the sources' render sets the same way.
    // The clones read their copied ranges once the copies record and their chain submits.
    std::vector<uint32_t> CloneMeshes(CloneCopies &, std::span<const uint32_t> source_ids, bool render);
    // Restores `id`, a copy of `source` without render records, to `source` at the ascending local blocks `written` names and wherever their tables differ in shape.
    // The copy's table takes the source's shape, with a fresh block at each local block only the source holds.
    // Its attributes outside the ascending `attributes` take fresh copies of the source's, which `attributes` then lists.
    // The copies record on `copies`, and the returned change names the restored and removed blocks for ReconcileSelection.
    Change Resync(CloneCopies &, uint32_t id, uint32_t source, const DomainBlocks &written, std::span<const uint32_t> written_attributes, std::vector<uint32_t> &attributes);
    // Moves `source`'s elements into `id`, appending its blocks to `id`'s tables, so its references shift by `id`'s block counts and its primitives follow `id`'s in its material table.
    // Element layers and named attributes present on either record hold their defaults on the other's elements, and deform channels follow `id`.
    // Each record's primitives take one table entry at least, so a record without a primitive table reads primitive 0 of its own entries.
    // `source` keeps no elements or render records, and its caller releases it.
    void Join(state::Scene &, uint32_t id, uint32_t source);
    // The element blocks whose canonical values or membership captured writes since the last call, and the named attributes released or renamed since then, each ascending.
    struct Writes {
        DomainBlocks Blocks{};
        std::vector<uint32_t> Attributes{};
    };
    Writes TakeWrites();
    // Returns a vertex-only store ID that must be released with Release.
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &);
    // Stores bone or joint index data for a vertex-only record, as its vertex handles.
    void SetExtrasIndices(uint32_t id, std::span<const uint32_t> faces, std::span<const uint32_t> edges);
    // Releases the records' canonical and render storage.
    void Release(uint32_t id);
    void Release(std::span<const uint32_t> ids);
    // Releases the records' render sets with their element owners and resets the render fields.
    // A fragment's meshlets and their payloads leave the sets of its render owner, the record its store id names.
    void ReleaseRender(std::span<Record *const>);
    // Erases the owner's clusters with their dirty bits and identities, and releases their payload runs.
    void ReleaseMeshletStorage(Record &owner, std::span<const uint32_t> handles);
    // Appends `count` values to the record's range of the pool and returns their offset in it.
    // A full range grows in place when the storage after it is free, and otherwise moves to twice its capacity or to fit, and the record's mesh display takes its new base.
    uint32_t AppendPayload(Record &, RenderPayload, uint32_t count);
    // Releases runs of the record's range of the pool: runs that end its used values shorten them, and other runs stay holes until the next build.
    void ReleasePayloadRuns(Record &, RenderPayload, std::vector<Range> runs);
    // Checks that every live record's table entries name blocks it owns at their local index, that each layer and named attribute range it holds covers its handle bound, and that each reference its element values, vertex fans, sector roots, selection summary and render records hold is null or names a live member of the same record.
    // Returns false with the first violation in `why`.
    bool Audit(std::string &why) const;
    // Hashes the record's values at its live elements and render members in table order with its handles as they are.
    // A clone hashes equal to its source.
    store::Hash128 CanonicalHash(uint32_t id) const;
    // The ids Release retired since the last call, whose render tallies drop.
    std::vector<uint32_t> TakeReleased() { return std::exchange(Released, {}); }
    // Reset all arenas and the StoreId table to empty, keeping GPU allocations for reuse.
    // Requires a full scene clear without live StoreId references so allocation restarts deterministically.
    void Clear();

    const Record &Get(uint32_t id) const { return Records.at(id); }
    // The live record with this id, or null.
    const Record *TryGet(uint32_t id) const { return id < Records.size() && Records[id].Alive ? &Records[id] : nullptr; }
    // Captures the record before a write.
    Record &WriteRecord(uint32_t id);
    const DerivedRecord &GetDerived(uint32_t id) const { return DerivedRecords.at(id); }
    const MeshArenas &Arenas() const { return Buffers; }
    RenderArenas &Render() { return Buffers.Render; }
    const RenderArenas &Render() const { return Buffers.Render; }
    const MeshSlots &Slots() const { return SlotTable; }

    uint32_t MeshletCount(const Record &r) const { return Buffers.Render.Meshlets.Count(r.MeshletData); }
    uint32_t PrimitiveCount(const Record &r) const { return Buffers.Render.Primitives.Count(r.PrimitiveData); }
    uint32_t LodNodeCount(const Record &r) const { return Buffers.Render.LodNodes.Count(r.NodeData); }
    uint32_t ClusterGroupCount(const Record &r) const { return Buffers.Render.ClusterGroups.Count(r.GroupData); }
    // Visits the record's primitives in ascending handle order.
    void ForEachPrimitive(const Record &r, auto &&fn) const {
        const auto &primitives = Buffers.Render.Primitives;
        primitives.ForEach(r.PrimitiveData, [&](ElementHandle handle, ElementAddress address) { fn(handle, primitives.Get({address.Value(), 1u})[0]); });
    }
    // Visits the meshlet handles a traversal leaf of the record lists, in member order.
    void ForEachLodMember(const Record &r, const LodNode &node, auto &&fn) const {
        if (node.Members == InvalidOffset) return;
        const auto &members = Buffers.Render.LodMembers;
        const auto entries = members.Entries(r.MemberData);
        for (uint32_t i = 0u; i < node.MeshletCount; ++i) fn(members.Get({Resolve<RenderBlockSize>(entries, ElementHandle{node.Members + i}).Value(), 1u})[0]);
    }
    // Whether the record holds a finest cluster whose positions changed since coarse repair, and a stale group.
    bool HasDirtyPositions(const Record &) const;
    bool HasDirtyGroups(const Record &) const;
    uint32_t PrimitiveRoute(const Record &r, uint32_t source_primitive, uint32_t topology) const {
        const uint64_t route = 3ull * source_primitive + topology;
        return topology < 3u && route < r.PrimitiveRoutes.Count ? Buffers.Render.PrimitiveRoutes.Get({r.PrimitiveRoutes.Offset + uint32_t(route), 1u})[0] : InvalidOffset;
    }
    // Grows the record's routes to cover `count` source primitives, keeping its routes.
    void ReservePrimitiveRoutes(Record &, uint32_t count);
    // The element domain the finest clusters of a render topology name: triangles, edges or vertices.
    static ElementDomain RenderDomain(uint32_t topology) { return topology == 0u ? ElementDomain::Triangle : topology == 1u ? ElementDomain::Edge :
                                                                                                                              ElementDomain::Vertex; }
    // Visits the arena of the record's elements drawn as `topology` with the record's set in it.
    decltype(auto) WithRenderDomain(const Record &, uint32_t topology, auto &&fn) const;

    // Mutable views over Persistent arena data, capturing the pages they expose.
    std::span<uint32_t> EditPrimitiveMaterials(uint32_t id);
    // Callers writing the sharpness stores rederive corner normals afterward.
    std::span<uint8_t> EditFaceSharpness(uint32_t id);
    std::span<uint8_t> EditEdgeSharpness(uint32_t id);
    // Installs the custom corner-normal layer at the dense corners in handle order, or removes it when no corner holds an offset.
    void SetCustomCornerNormals(uint32_t id, std::span<const CustomNormal> offsets);
    // The record's element layers, as kernels bind them.
    ElementLayers Layers(uint32_t id) const;
    void SetMorphShadingAuthored(uint32_t id, bool);

    // Named attributes hold a value at every handle of a domain of their record, in one range of their type's value blocks with a block per local block.
    // An attribute's id is its slot, and its range grows with the record's table, clones with the record and releases with it.
    enum class AttributeError { InvalidName,
                                Duplicate };
    // Adds an attribute holding `fallback` at every element, which also fills elements an edit creates without a source.
    // A name is unique within its record and fits AttributeName with its terminator.
    std::expected<uint32_t, AttributeError> AddAttribute(uint32_t id, AttributeDomain, AttributeType, std::string_view name, AttributeValue fallback = {});
    // The record's attribute with this name, or InvalidOffset.
    uint32_t FindAttribute(uint32_t id, std::string_view name) const;
    void ReleaseAttribute(uint32_t attribute);
    // Names the attribute anew, under the same rules as AddAttribute.
    std::expected<void, AttributeError> RenameAttribute(uint32_t attribute, std::string_view name);
    // Adds an attribute on the same record and domain holding the attribute's values.
    std::expected<uint32_t, AttributeError> CopyAttribute(uint32_t attribute, std::string_view name);
    const NamedAttribute &GetAttribute(uint32_t attribute) const { return NamedAttributes.at(attribute); }
    // Visits the record's attributes in slot order, a scan of every attribute that authoring, cloning, release and topology setup make.
    void ForEachNamedAttribute(uint32_t id, auto &&visit) const {
        for (uint32_t a = 0u; a < NamedAttributes.size(); ++a)
            if (NamedAttributes[a].Record == id) visit(a, NamedAttributes[a]);
    }
    template<AttributeScalar T> NamedAttributeView<T> AttributeView(uint32_t attribute) const {
        return {ValuesOf<T>(Buffers.Named).Get(TypedAttribute<T>(attribute).Values)};
    }
    // Writes `values` to the consecutive handles from `first`, capturing their pages.
    template<AttributeScalar T> void SetAttribute(uint32_t attribute, uint32_t first, std::span<const T> values) {
        if (values.empty()) return;
        const auto range = TypedAttribute<T>(attribute).Values;
        const auto local = first / MeshElementBlockSize, end = (first + uint32_t(values.size()) - 1u) / MeshElementBlockSize + 1u;
        auto blocks = ValuesOf<T>(Buffers.Named).GetMutable({range.Offset + local, end - local});
        for (uint32_t i = 0u; i < values.size(); ++i) {
            const auto handle = first + i - local * MeshElementBlockSize;
            auto &block = blocks[handle / MeshElementBlockSize];
            const auto slot = handle % MeshElementBlockSize;
            if constexpr (std::same_as<T, bool>) {
                const auto bit = 1u << (slot % 32u);
                auto &mask = block[slot / 32u];
                mask = values[i] ? mask | bit : mask & ~bit;
            } else block[slot] = values[i];
        }
    }
    NamedAttributeRef AttributeRef(uint32_t attribute) const;
    // Captures the values of the record's attributes of `domain` at the arena blocks before a GPU write.
    void CaptureAttributeWrites(uint32_t id, AttributeDomain, std::span<const uint32_t> blocks) const;

    // Copies tetrahedral wireframe geometry into GPU-only canonical arenas.
    TetBuffers AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices);
    void ReleaseTets(TetBuffers);
    Range AllocateSoundVertices(std::span<const uint32_t>);
    void ReleaseSoundVertices(std::vector<Range>);

    // Selection masks mirror canonical element blocks.
    // Only summaries, roots and gesture state are per mesh.
    // Creates cleared masks, a summary and the aggregates of every owned block, once per record.
    // A record without them submits the chain, so their summaries publish before this returns.
    void EnsureSelectionState(state::Scene &, mtl::ComputeChain &, std::span<const uint32_t> ids);
    SelectionView GetSelectedElements(uint32_t id, Element) const;
    SelectionView GetHiddenElements(uint32_t id, Element) const;
    uint32_t GetHiddenSlot(Element) const;
    void EditHiddenBlocks(Element element, std::span<const uint32_t> blocks, auto &&write) {
        auto &bits = element == Element::Vertex ? Buffers.VertexHidden : element == Element::Edge ? Buffers.EdgeHidden :
                                                                                                    Buffers.FaceHidden;
        bits.Buffer.CaptureWriteElements(blocks, sizeof(SelectionBlock));
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            auto words = bits.GetMutable({blocks[first], uint32_t(count)});
            for (size_t j = 0u; j < count; ++j) write(blocks[first + j], words[j]);
        });
    }
    // Empty for a mesh without faces.
    BoundaryEdgeView GetBoundaryEdges(uint32_t id) const;
    const SelectionAggregate &GetSelectionRoot(uint32_t id, Element) const;
    // The vertex root, followed by the edge and face roots.
    SlotOffset GetVertexSelectionRoot(uint32_t id) const;
    // Captures the listed ascending mask blocks and writes each with `write(block, words)`.
    void EditSelectionBlocks(Element element, std::span<const uint32_t> blocks, auto &&write) {
        auto &bits = element == Element::Vertex ? Buffers.VertexSelection : element == Element::Edge ? Buffers.EdgeSelection :
                                                                                                       Buffers.FaceSelection;
        bits.Buffer.CaptureWriteElements(blocks, sizeof(SelectionBlock));
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            auto words = bits.GetMutable({blocks[first], uint32_t(count)});
            for (size_t j = 0u; j < count; ++j) write(blocks[first + j], words[j]);
        });
    }
    // Aggregates of `Blocks` refresh on the GPU, and each seed's incident blocks join them.
    // A valid Source first rewrites the other two domains' words in those blocks from its words, which submits the chain once to capture them.
    // The mesh's roots are then reduced again, and the host reads them once the chain submits.
    struct SelectionUpdate {
        uint32_t StoreId;
        Element Source{Element::None};
        std::array<std::vector<uint32_t>, 3> Blocks{};
        std::vector<SelectionSeed> Seeds{};
    };
    void UpdateSelection(state::Scene &, mtl::ComputeChain &, std::span<const SelectionUpdate>);
    // Records the rewrite of the masks' words in the record's blocks of the two domains other than `source`, derived from the source domain's words the way UpdateSelection derives them.
    // The store's selection, aggregates and hidden masks are unchanged, and hidden elements stay selectable.
    void DeriveSelectionMasks(state::Scene &, mtl::ComputeChain &, uint32_t id, const GeometryMasks &, Element source) const;
    // Clears the record's selection and gives each of its blocks the hidden mask of `source`'s block at the same local block.
    // A record with selection state records the refresh of the changed blocks' aggregates.
    void ResetEditState(state::Scene &, mtl::ComputeChain &, uint32_t id, uint32_t source);
    // Gives the record's selection the masks' bits at each block it owns.
    // A record with selection state records the refresh of the changed blocks' aggregates.
    void SelectMasks(state::Scene &, mtl::ComputeChain &, uint32_t id, const GeometryMasks &);
    // Copies the root counts, sums and sharpness of the summary's mode into the summary.
    void PublishSelectionSummary(uint32_t id);
    EditSelectionSummary &WriteSelectionSummary(uint32_t id);
    void SetSelectionBaseline(uint32_t id, Element, std::vector<std::pair<uint32_t, SelectionBlock>>, uint32_t active);
    // Builds a new clone's index after its GPU block copies have completed.
    void RebuildSelectionIndex(std::span<const uint32_t> ids);
    // Records the gather of the selected handles in ascending order into a range it allocates in `output`, and returns that range.
    Range GatherSelectedElements(state::Scene &, mtl::ComputeChain &, uint32_t id, Element, BufferArena<uint32_t> &output) const;
    bool IsLiveElement(uint32_t id, Element, ElementHandle) const;
    uint32_t GetSelectionSlot(Element) const;
    EditSelectionStorage GetEditSelectionStorage(uint32_t id) const;
    const EditSelectionSummary &GetSelectionSummary(uint32_t id) const;

    SharpnessSummary GetFaceSharpnessSummary(uint32_t id) const;
    SharpnessSummary GetEdgeSharpnessSummary(uint32_t id) const;
    // Returns the class-buffer offset or a uniform-class sentinel.
    uint32_t GetCornerClassMode(uint32_t id) const;
    CornerNormalView GetCornerNormalView(uint32_t id) const;
    // Returns composed corner normals in triangulated face-fan order, in scratch storage valid until the next call.
    // Requires current base stores (the derive pass ran since the last position/sharpness write).
    std::span<const vec3> GetCornerNormals(const Mesh &) const;
    // Classify each corner from the sharpness stores: vertex-normal, face-normal, or a seam sector of incident triangles.
    // Call after any sharpness write, then run the base derive pass to refill the base normal stores.
    // A sparse vertex domain updates only its incoming corners, which `incoming` bounds.
    // The default work descriptor gathers all live vertices through the same GPU path.
    // Encode records the counts and the blocks the roots need.
    // Plan settles the mesh's classification mode once the chain submits, holds sector ranges while the mode is mixed and records the root writes.
    struct CornerClassUpdate {
        uint32_t StoreId{};
        CornerClassificationPushConstants Pc{};
        uint32_t VertexCount{}, Incoming{};
        bool Complete{};
    };
    CornerClassUpdate EncodeCornerClassification(state::Scene &, mtl::ComputeChain &, uint32_t id, ElementWork vertices = {}, uint32_t vertex_count = 0u, uint32_t incoming = 0u, bool complete = false);
    void PlanCornerClassification(state::Scene &, mtl::ComputeChain &, CornerClassUpdate &);
    // Classifies every corner of each listed mesh, through both steps over the chain.
    // It submits the chain once for the plan.
    void UpdateCornerClassification(state::Scene &, mtl::ComputeChain &, std::span<const uint32_t> ids);
    // Enumerates incident edges from canonical corner links, including boundaries.
    VertexEdgeIncidence GetVertexEdgeIncidence(uint32_t id) const;

private:
    MeshArenas Buffers;
    MeshSlots SlotTable;
    std::vector<Record> Records{};
    std::vector<DerivedRecord> DerivedRecords{};
    // Block lists, keyed by record and element domain, with the membership revision they describe.
    // Const reads fill a stale list under the lock, so concurrent reads between membership changes stay safe.
    struct BlockListEntry {
        ElementSetRef Set{};
        uint32_t Revision{};
        Range Words{};
    };
    mutable std::mutex BlockListLock;
    mutable BufferArena<uint32_t> BlockLists;
    mutable std::vector<std::array<BlockListEntry, 5>> BlockListEntries{};
    bool FrameReadsBlockLists{};
    mutable std::vector<Range> RetiredBlockLists{}; // Replaced list words a submitted frame can still read
    uint64_t NextNormalRevision{};
    std::vector<uint32_t> FreeIds{};
    std::vector<uint32_t> Released{};
    std::vector<NamedAttribute> NamedAttributes{};

    struct HistoryState;
    std::unique_ptr<HistoryState> Tracked;

    // Each element domain's pairs of a source block and the block of the record copying it, indexed by ElementDomain.
    using BlockPairs = std::array<std::vector<uvec2>, 6>;
    // The pairs of `source`'s and `destination`'s blocks at each local block both hold.
    BlockPairs PairBlocks(uint32_t source, uint32_t destination) const;

    // Gives each clone its source's render sets over fresh blocks, with every render value copied and each cluster's payloads packed anew.
    // The clones' GPU mesh records are left for RefreshMeshBinding.
    void CloneRenderRecords(CloneCopies &, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids);
    void ReleaseBlockLists(uint32_t id);
    void ReleaseBlockLists(std::span<const uint32_t> ids);
    // Releases replaced list words, or keeps them until the submitted frame completes.
    void RetireBlockListWords(Range) const;
    void FinishEraseElements(uint32_t id, ElementDomain, std::span<const uint32_t> blocks);
    uint32_t AcquireId(Record &&);
    // Size each global mirror to its canonical domain's resident extent.
    void SyncMirrors();
    // Fill the base vertex-normal mirror over `vertices`: a face-less mesh's point normals, zero otherwise (triangle meshes rederive the region).
    void FillBaseVertexNormalMirror(ElementSetRef vertices, Range point_normals);

    template<AttributeScalar T> const NamedAttribute &TypedAttribute(uint32_t attribute) const {
        const auto &entry = NamedAttributes.at(attribute);
        if (entry.Record == InvalidOffset || entry.Type != AttributeTypeOf<T>) throw std::invalid_argument("Attribute type differs from its access.");
        return entry;
    }
    // Captures the attribute before a write.
    NamedAttribute &WriteAttribute(uint32_t attribute);
    // Takes the lowest free slot for the entry.
    uint32_t AcquireAttribute(NamedAttribute &&);
    // Copies each domain's paired blocks of `source` into the blocks of `destination` they pair with, which hold the source blocks' live slots.
    // Values, mirrors, masks and vertex fans copy byte for byte.
    void CopyBlocks(CloneCopies &, uint32_t destination, uint32_t source, const BlockPairs &pairs);
    // Releases every block of the record's domain with the ranges of the domain's layers and named attributes, leaving the record without the domain's set.
    void ReleaseDomain(uint32_t id, ElementDomain);
    // Clears every element of the record's ascending blocks of the domain, which leave it.
    void DetachBlocks(uint32_t id, ElementDomain, std::span<const uint32_t> blocks);
    // A logged buffer names element addresses, element blocks, or the blocks of records' layer ranges.
    enum class WriteIndex : uint8_t { Elements,
                                      Blocks,
                                      Layer };
    struct WriteLog {
        mtl::Buffer *Buffer;
        ElementDomain Domain;
        WriteIndex Index;
        uint32_t Stride;
        std::vector<std::pair<uint64_t, uint64_t>> Extents{};
    };
    std::vector<WriteLog> WriteLogs{}; // Fixed at construction, so each buffer keeps its log's address
    std::vector<uint32_t> WrittenAttributes{};
};

// Visits the arena of one element domain, which every caller resolves before calling.
decltype(auto) WithDomain(auto &arenas, MeshStore::ElementDomain domain, auto &&fn) {
    switch (domain) {
        case MeshStore::ElementDomain::Vertex: return fn(arenas.Vertices);
        case MeshStore::ElementDomain::Halfedge: return fn(arenas.FaceCorners);
        case MeshStore::ElementDomain::Edge: return fn(arenas.EdgeHalfedges);
        case MeshStore::ElementDomain::Face: return fn(arenas.FaceTriangles);
        case MeshStore::ElementDomain::Triangle: return fn(arenas.Triangles);
        case MeshStore::ElementDomain::None: std::unreachable();
    }
}
// The element domain holding an attribute domain's elements.
constexpr MeshStore::ElementDomain ElementDomainOf(AttributeDomain domain) {
    switch (domain) {
        case AttributeDomain::Vertex: return MeshStore::ElementDomain::Vertex;
        case AttributeDomain::Edge: return MeshStore::ElementDomain::Edge;
        case AttributeDomain::Face: return MeshStore::ElementDomain::Face;
        case AttributeDomain::Corner: return MeshStore::ElementDomain::Halfedge;
    }
    std::unreachable();
}
// The record's set in one element domain.
auto &DomainSet(auto &record, MeshStore::ElementDomain domain) {
    switch (domain) {
        case MeshStore::ElementDomain::Vertex: return record.Vertices;
        case MeshStore::ElementDomain::Halfedge: return record.FaceCorners;
        case MeshStore::ElementDomain::Edge: return record.EdgeData;
        case MeshStore::ElementDomain::Face: return record.FaceData;
        case MeshStore::ElementDomain::Triangle: return record.TriangleData;
        case MeshStore::ElementDomain::None: std::unreachable();
    }
}
decltype(auto) MeshStore::WithRenderDomain(const Record &r, uint32_t topology, auto &&fn) const {
    const auto domain = RenderDomain(topology);
    return WithDomain(Buffers, domain, [&](const auto &arena) { return fn(arena, DomainSet(r, domain)); });
}

// The record's set in one render collection.
auto &RenderSetOf(auto &record, RenderSet set) {
    switch (set) {
        case RenderSet::Meshlets: return record.MeshletData;
        case RenderSet::Primitives: return record.PrimitiveData;
        case RenderSet::Nodes: return record.NodeData;
        case RenderSet::Groups: return record.GroupData;
        case RenderSet::Members: return record.MemberData;
    }
    std::unreachable();
}
// Visits the pool of one payload with the record's range of it.
decltype(auto) WithPayload(auto &render, auto &record, RenderPayload payload, auto &&fn) {
    switch (payload) {
        case RenderPayload::TriangleIds: return fn(render.MeshletTriangleIds, record.MeshletTriangles);
        case RenderPayload::VertexCorners: return fn(render.MeshletVertexCorners, record.MeshletVertices);
        case RenderPayload::LocalTriangles: return fn(render.MeshletLocalTriangles, record.MeshletLocalTriangles);
        case RenderPayload::GroupClusters: return fn(render.GroupClusterIds, record.GroupClusters);
    }
    std::unreachable();
}
// The bases of the record's cluster payload ranges in their pools.
inline MeshletPayloadBases PayloadBasesOf(const MeshStore::Record &record) { return {record.MeshletTriangles.Values.Offset, record.MeshletVertices.Values.Offset, record.MeshletLocalTriangles.Values.Offset}; }

// The live record the entity's instances draw, or null.
const MeshStore::Record *TryRecordOf(const state::Scene &, state::Entity);
const MeshStore::Record &RecordOf(const state::Scene &, state::Entity);
// Captures the record the entity's instances draw before a write.
MeshStore::Record &EditRecordOf(state::Scene &, state::Entity);
