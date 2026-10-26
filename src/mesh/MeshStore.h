#pragma once

#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "TetBuffers.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/ClusterGroup.h"
#include "gpu/ClusterGroupLinks.h"
#include "gpu/EditSelectionSummary.h"
#include "gpu/EditSharpnessOperation.h"
#include "gpu/LodNode.h"
#include "gpu/MeshRecord.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshletSpatialNode.h"
#include "gpu/MorphTargetVertex.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/SelectionUpdatePushConstants.h"
#include "mesh/CornerNormalView.h"
#include "mesh/ElementArena.h"
#include "mesh/ElementAttribute.h"
#include "mesh/MeshSchema.h"
#include "mesh/PoseAttributeView.h"
#include "mesh/SelectionQuery.h"
#include "mesh/SelectionView.h"
#include "mesh/VertexFanStore.h"

#include <mutex>

namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state
struct CloneCopies;
struct CornerClassBatch;

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

// Index materials and attribute flags by source primitive; map drawn elements through ElementPrimitiveIndices.
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

// Block sets owned by a record: finest and coarse clusters, traversal primitives and nodes, cluster groups, and leaf members.
enum class RenderSet : uint8_t { Meshlets,
                                 Primitives,
                                 Nodes,
                                 Groups,
                                 Members };
constexpr std::array RenderSets{RenderSet::Meshlets, RenderSet::Primitives, RenderSet::Nodes, RenderSet::Groups, RenderSet::Members};
// Per-record payload ranges: source elements, vertex corners, local triangles, and group cluster runs.
enum class RenderPayload : uint8_t { TriangleIds,
                                     VertexCorners,
                                     LocalTriangles,
                                     GroupClusters };
constexpr std::array RenderPayloads{RenderPayload::TriangleIds, RenderPayload::VertexCorners, RenderPayload::LocalTriangles, RenderPayload::GroupClusters};
// Payload offsets are relative to Values.Offset, allowing relocation without rewriting them.
struct PayloadRange {
    Range Values{};
    uint32_t Capacity{};
};

// Consecutive handles and arena addresses for inserted render members.
struct RenderRun {
    ElementHandle Handle{};
    ElementAddress Address{};
    uint32_t Count{};
};

// Store each record in a block set; render handles are record-relative.
struct RenderArenas {
    explicit RenderArenas(mtl::BufferContext &);

    BufferArena<uint32_t> ExtrasFaces, ExtrasEdges; // Bone and joint index data, as each record's vertex handles
    ElementArena<MeshletRecord, RenderBlockSize> Meshlets;
    BufferArena<MeshletSpatialNode> MeshletSpatialNodes; // Mirrors Meshlets
    // Mirrors Meshlets: each cluster's traversal leaf, read only by repairs.
    BufferArena<uint32_t> MeshletLodLeaves;
    // One bit per meshlet slot marks positions changed since coarse repair.
    BufferArena<uint32_t> PositionDirty;
    // Per-cluster payload ranges: finest source-element handles, vertex-corner handles, and local triangle indices.
    BufferArena<uint32_t> MeshletTriangleIds, MeshletVertexCorners;
    BufferArena<MeshletTriangle> MeshletLocalTriangles;
    // One cluster group per simplification step.
    ElementArena<ClusterGroup, RenderBlockSize> ClusterGroups;
    // Group inputs and proxies reference runs of GroupClusterIds.
    BufferArena<ClusterGroupLinks> GroupLinks;
    // One bit per group slot marks coarse clusters for rebuild on leaving Edit mode.
    BufferArena<uint32_t> DirtyGroups;
    BufferArena<uint32_t> GroupClusterIds; // Meshlet handles
    ElementArena<LodNode, RenderBlockSize> LodNodes;
    BufferArena<uint32_t> LodParents; // Mirrors LodNodes
    // Each traversal leaf references a run of meshlet handles followed by spare InvalidOffset entries.
    ElementArena<uint32_t, RenderBlockSize> LodMembers;
    ElementArena<PrimitiveRecord, RenderBlockSize> Primitives;
    // Map source primitives to render handles; absent entries are InvalidOffset.
    BufferArena<uint32_t> PrimitiveRoutes;
    BufferArena<MeshRecord> MeshRecords; // One GPU mesh record per store id, rebuilt from the record's current bindings
    // Map triangles, loose edges, and isolated vertices to their drawing meshlets.
    std::array<ElementAttribute<uint32_t>, 3> ElementMeshlets;

    // Insert consecutive members, creating the set and sizing mirrors as needed.
    RenderRun Insert(RenderSet, ElementSetRef &, uint32_t count);
    // Return the first member handle for ascending `clusters`, or InvalidOffset for an empty span.
    uint32_t ListMembers(ElementSetRef &, std::span<const uint32_t> clusters);
    void SyncMirrors();
    MeshletPayloadSlots PayloadSlots() const;
};
// Masks use one word per block and one bit per slot.
bool MaskHas(const BufferArena<uint32_t> &, ElementAddress);
void MaskSet(BufferArena<uint32_t> &, ElementAddress);
void MaskClear(BufferArena<uint32_t> &, ElementAddress);
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

// Each named attribute owns one value range, indexed by its domain blocks.
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
decltype(auto) WithAttributeValues(auto &arenas, AttributeType type, auto &&fn) {
    return WithAttributeScalar(type, [&]<AttributeScalar T>() -> decltype(auto) { return fn(ValuesOf<T>(arenas)); });
}

// Allocate one arena per GPU-readable stream.
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
    // Mirror domain blocks with aggregates of selection, positions, sharpness, and connectivity.
    BufferArena<SelectionAggregate> VertexAggregates, EdgeAggregates, FaceAggregates;
    SelectionIndex SelectionTree;
    SelectionQuery Query;
    BufferArena<EditSelectionSummary> SelectionSummary; // One summary per mesh
    BufferArena<uint8_t> EdgeSharpness; // One byte per edge, 1 = sharp (canonical sharpness store)
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
    // Rebuild excitable vertex handles from the sound model after deserialization.
    BufferArena<uint32_t> SoundVertices;
    ElementAttribute<uint32_t> CornerSectors; // Sector-root corner handles, and InvalidOffset means the vertex normal
    ElementAttribute<NormalSector> NormalSectors; // Base normals at sector roots
    BufferArena<vec3> BaseVertexNormals; // Mirrors Vertices: derived smooth normals for triangle meshes, authored normals for face-less meshes
    BufferArena<vec3> BaseFaceNormals; // Mirrors FaceTriangles, one derived face normal per face slot
    AttributeArenas Named;
    RenderArenas Render;
};

// Bindless slots remain fixed for the store lifetime.
struct MeshSlots {
    uint32_t Vertices, FaceTriangleStart, FaceSharpness, EdgeSharpness;
    uint32_t PrimitiveMaterial;
    uint32_t TetPosition, TetEdgeIndex, SoundVertex;
    uint32_t BaseVertexNormal, BaseFaceNormal;
};

// Return a corner normal composed from canonical sharpness and sector references.
vec3 ComposeCornerNormal(const MeshConnectivity &, CornerAttributeView<uint32_t> sectors, ElementAttributeView<NormalSector> normal_sectors, std::span<const uint8_t> sharpness, uint32_t mode, uint32_t ci, TriangleVertexView vertices, TriangleFaceView face_ids, const CornerNormalSources &);

// Own canonical CPU/GPU mesh storage.
// Mutators capture history pages before writing.
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

    // Resolve record-relative handles through each domain set.
    // Attach selection masks to blocks and index attribute ranges by handles.
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
        bool HasAuthoredNormals{false};
        std::vector<float> DefaultMorphWeights{};
        // Attribute ranges cover domain handle bounds; absent layers have empty ranges.
        Range FacePrimitives{}, VertexPrimitives{}, Skin{}, Morph{}, CustomNormals{}, CornerTangents{}, CornerColors{}, VertexColors{};
        std::array<Range, 4> CornerUvs{};
        Range CornerSectors{}, NormalSectors{}; // Present when corners can reference sector roots
        uint32_t Classification{uint32_t(CornerClassMode::UniformVertex)};
        bool MorphShadingAuthored{};
        Range ExtrasFaces{}, ExtrasEdges{}; // Bone and joint index data, present only on an extras record.
        // Resolve record-relative render handles through the block-set table of their collection.
        ElementSetRef MeshletData{}, PrimitiveData{}, NodeData{}, GroupData{}, MemberData{};
        // Cluster and group payload offsets are relative to these per-record ranges.
        PayloadRange MeshletTriangles{}, MeshletVertices{}, MeshletLocalTriangles{}, GroupClusters{};
        // Meshlet handles from the last build, relative to the render owner.
        Range Meshlets{};
        uint32_t SpatialRoot{InvalidOffset}; // The meshlet handle at the root of the spatial tree.
        Range PrimitiveRoutes{}; // Three render primitive handles per source material, indexed by topology.
        uint64_t MeshletRevision{};
        uint32_t Level0Count{};
        uint32_t RenderTopologies{}; // Bit mask of published triangle, line and point topology; zero before the first build.
        // Owner ranges and occupied-block counts for triangles, loose edges, and isolated vertices.
        std::array<Range, 3> ElementMeshlets{};
        std::array<uint32_t, 3> ElementMeshletBlockCounts{};
        uint32_t LodDepth{}; // The depth of the record's deepest traversal tree.
        bool Alive{false};
    };

    // Transient selection gestures and pose-cache invalidation.
    struct DerivedRecord {
        // The source-domain mask blocks and active handle saved when an additive gesture began.
        std::vector<std::pair<uint32_t, SelectionBlock>> SelectionBaseline{};
        uint32_t SelectionBaselineActive{InvalidOffset};
        Element SelectionBaselineElement{Element::None};
        uint64_t NormalRevision{};
    };

    struct TopologyCounts {
        uint32_t Vertices, Halfedges, Faces;
    };
    // Create an output record with source metadata and the supplied materials.
    uint32_t BeginTopologyOutput(uint32_t source, std::span<const uint32_t> materials);

    // Insert unselected elements with default attributes and capture their value pages.
    // With a bindless `list`, return listed handles; otherwise return one consecutive run.
    // Refresh affected selection aggregates before publishing.
    ElementHandleRange InsertElements(uint32_t id, ElementDomain, uint32_t count, BufferArena<uint32_t> *list);
    // Erase elements and selection, returning changed membership blocks.
    // The caller must refresh their aggregates.
    std::vector<uint32_t> EraseElements(uint32_t id, ElementDomain, const BufferArena<uint32_t> &storage, ElementWork);
    // Retain the first `used` inserted handles and release the rest.
    // Released handles must be unselected.
    void TrimInsertedElements(uint32_t id, ElementDomain, ElementHandleRange &inserted, const BufferArena<uint32_t> &list, uint32_t used);

    void Track(store::History &);
    void FinishRestore();
    // Index changed records for sparse history restoration.
    void IndexHistory();
    // Return restored changes and invalidate their cached block lists.
    std::vector<Change> TakeChanges();
    // Refresh aggregates for restored blocks and incident blocks.
    void ReconcileSelection(state::Registry &, mtl::ComputeChain &, std::span<const Change>);

    // Settle the tracked storage and return the hash of its content.
    store::Hash128 StateHash();
    // Settle the tracked storage, pin a version of it as the kept version of its state, and return the state's hash.
    store::Hash128 KeepState();
    // Copy the record `source` as it stood in the state with hash `state` into a new record without render data or selection state.
    // Read a kept version, or else the history's content logs, and return none when neither holds the state.
    std::optional<uint32_t> CloneStored(store::Hash128 state, uint32_t source);
    // Restore the record `id` to its content in the state with hash `state`, keeping its live allocations where they fit, and return the change that reconciles it.
    // Read a kept version, or else the history's content logs, and return none when neither holds the state or the record is not alive in it.
    // Complete CPU and GPU access to mesh storage first.
    std::optional<Change> RestoreRecord(store::Hash128 state, uint32_t id);
    // Restore every track to the kept version of the state with hash `state`, and return whether a kept version held it.
    // Complete CPU and GPU access to mesh storage first, and reconcile the changes TakeChanges then returns.
    bool RestoreKeptState(store::Hash128 state);
    // Record the roots of each listed kept state in the history, persisting the content no recorded state holds.
    void RecordKeptStates(std::span<const store::Hash128> states);
    // End a history step that refers to the listed states: release the kept versions of every other state, recording those the previous step referred to.
    void EndStepStates(std::span<const store::Hash128> states);

    // Capture destination pages before dispatching GPU writes to Persistent mesh data.
    void CaptureSelectionSummary(uint32_t id);
    void CaptureSelectionBlocks(Element, std::span<const uint32_t> blocks);
    void CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation, const GeometrySelection &);
    void CaptureConnectivityWrite(uint32_t id);
    void CaptureWeldWrite(uint32_t id);

    // Call CommitReserves after all PlanCreate and PlanClone calls, before creating or cloning.
    void PlanCreate(const MeshData &, const MeshPrimitives & = {}, bool has_deform = false, uint32_t morph_target_count = 0, const MeshVertexAttributes & = {});
    void PlanClone(const Mesh &);
    void CommitReserves();

    // Copy source positions and corners into arena storage and return the store ID.
    uint32_t CreateMeshSource(const MeshData &);
    // Allocate deform channels at the source vertex count for in-place welding.
    void CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &, const std::optional<MorphTargetData> &);
    void ShrinkMeshSource(uint32_t id, uint32_t welded_vertices);
    // Allocate connectivity in source order, reserving up to one edge per halfedge.
    // Supply face_offsets for polygon starts, or leave empty for GPU writes.
    void AllocateConnectivity(uint32_t id, uint32_t halfedge_count, uint32_t face_count, bool face_starts, std::span<const uint32_t> face_offsets = {}, std::span<const std::array<uint32_t, 2>> wire_edges = {});
    void FinishConnectivity(uint32_t id, uint32_t edge_count);
    MeshConnectivity GetConnectivity(uint32_t id) const;
    ConnectivityRef GetConnectivityRef(uint32_t id) const;
    ElementTables GetElementTables(uint32_t id) const;
    // Blocks in local order with inclusive live-element prefixes.
    // Rebuild on first read after membership changes.
    struct BlockList {
        std::span<const uint32_t> Blocks, LivePrefix;
        SlotOffset Gpu; // The blocks, followed by the prefix.
    };
    BlockList GetBlockList(uint32_t id, ElementDomain) const;
    // Retain replaced block-list storage until the submitted frame completes.
    void FrameSubmitted();
    void FrameCompleted();
    // Return the live handle at an arena address, or nullopt if it is outside the record.
    std::optional<ElementHandle> LiveHandleAt(uint32_t id, Element, ElementAddress) const;
    ElementAddress HandleAddress(uint32_t id, ElementDomain, ElementHandle) const;
    // Return arena blocks addressed by completed element work.
    std::vector<uint32_t> ArenaBlocks(uint32_t id, ElementDomain, const BufferArena<uint32_t> &storage, ElementWork) const;
    // Packed ordinals count live elements in ascending handle order.
    ElementHandle LiveElementAt(uint32_t id, ElementDomain, uint32_t ordinal) const;
    uint32_t LiveElementOrdinal(uint32_t id, ElementDomain, ElementHandle) const;
    VertexFanStore &VertexFans() { return Buffers.VertexFans; }
    TriangleCorners GetTriangleCorners(uint32_t id) const;
    // The mesh's live derived triangles and vertices in ordinal order.
    ElementView<uvec3> TriangleView(uint32_t id) const;
    ElementView<Vertex> VertexView(uint32_t id) const;
    // Complete source metadata and attribute storage; TakeUnshaded lists the record until shading derives.
    void CreateMesh(uint32_t id, const MeshData &, const MeshVertexAttributes &, const MeshPrimitives &, const CornerLayers &, bool has_authored_normals);
    // Each element domain's ascending blocks or local blocks, indexed by ElementDomain.
    using DomainBlocks = std::array<std::vector<uint32_t>, 6>;
    // Return clone IDs in source order, preserving source handles in fresh blocks.
    // Include render sets when `render` is true.
    // Submit queued copies before reading cloned ranges.
    std::vector<uint32_t> CloneMeshes(CloneCopies &, std::span<const uint32_t> source_ids, bool render);
    // Move source elements into id, remapping handles and materials.
    // Use defaults for missing layers and attributes; retain id deformation channels.
    // The caller must release the emptied source record.
    void Join(state::Registry &, uint32_t id, uint32_t source);
    // Returns a vertex-only store ID that must be released with Release.
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &);
    void SetExtrasIndices(uint32_t id, std::span<const uint32_t> faces, std::span<const uint32_t> edges);
    void Release(uint32_t id);
    void Release(std::span<const uint32_t> ids);
    // Release render sets and element owners, then reset render fields.
    // Resolve fragment ownership through StoreId.
    void ReleaseRender(std::span<Record *const>);
    // Release clusters, dirty bits, identities, and payloads.
    void ReleaseMeshletStorage(Record &owner, std::span<const uint32_t> handles);
    // Append payload values and return their record-relative offset.
    uint32_t AppendPayload(Record &, RenderPayload, uint32_t count);
    // Release payload runs, truncating trailing storage and retaining interior gaps until rebuild.
    void ReleasePayloadRuns(Record &, RenderPayload, std::vector<Range> runs);
    // Validate block ownership, attribute bounds, and references to live record members.
    // Return false with the first violation in `why`.
    bool Audit(std::string &why) const;
    // Hash live elements and render members in table order.
    // Clones hash equal to their sources.
    store::Hash128 CanonicalHash(uint32_t id) const;
    // Return IDs retired by Release since the previous call so their render tallies can be removed.
    std::vector<uint32_t> TakeReleased() { return std::exchange(Released, {}); }
    // Return newly created records awaiting corner classification, base normals, and selection state.
    std::vector<uint32_t> TakeUnshaded() { return std::exchange(Unshaded, {}); }
    // Clears records and arenas while retaining GPU allocations.
    // Requires a cleared scene with no live StoreId references.
    friend void Reset(MeshStore &, state::Registry &);

    const Record &Get(uint32_t id) const { return Records.at(id); }
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
    bool HasDirtyPositions(const Record &) const;
    bool HasDirtyGroups(const Record &) const;
    uint32_t PrimitiveRoute(const Record &r, uint32_t source_primitive, uint32_t topology) const {
        const uint64_t route = 3ull * source_primitive + topology;
        return topology < 3u && route < r.PrimitiveRoutes.Count ? Buffers.Render.PrimitiveRoutes.Get({r.PrimitiveRoutes.Offset + uint32_t(route), 1u})[0] : InvalidOffset;
    }
    void ReservePrimitiveRoutes(Record &, uint32_t count);
    // The element domain the finest clusters of a render topology name: triangles, edges or vertices.
    static ElementDomain RenderDomain(uint32_t topology) { return topology == 0u ? ElementDomain::Triangle : topology == 1u ? ElementDomain::Edge :
                                                                                                                              ElementDomain::Vertex; }
    decltype(auto) WithRenderDomain(const Record &, uint32_t topology, auto &&fn) const;

    // Mutable views over Persistent arena data, capturing the pages they expose.
    std::span<uint32_t> EditPrimitiveMaterials(uint32_t id);
    // Callers writing the sharpness stores rederive corner normals afterward.
    std::span<uint8_t> EditFaceSharpness(uint32_t id);
    std::span<uint8_t> EditEdgeSharpness(uint32_t id);
    // Set custom normals in corner-handle order; remove the layer when all offsets are empty.
    void SetCustomCornerNormals(uint32_t id, std::span<const CustomNormal> offsets);
    ElementLayers Layers(uint32_t id) const;
    void SetMorphShadingAuthored(uint32_t id, bool);

    // Named attributes share their record lifetime and grow with its domain table.
    enum class AttributeError { InvalidName,
                                Duplicate };
    // Initialize every element and future source-less elements to `fallback`.
    // Names must be unique within the record and fit AttributeName including the terminator.
    std::expected<uint32_t, AttributeError> AddAttribute(uint32_t id, AttributeDomain, AttributeType, std::string_view name, AttributeValue fallback = {});
    // The record's attribute with this name, or InvalidOffset.
    uint32_t FindAttribute(uint32_t id, std::string_view name) const;
    void ReleaseAttribute(uint32_t attribute);
    // Rename under the same constraints as AddAttribute.
    std::expected<void, AttributeError> RenameAttribute(uint32_t attribute, std::string_view name);
    // Copy values into a new attribute on the same record and domain.
    std::expected<uint32_t, AttributeError> CopyAttribute(uint32_t attribute, std::string_view name);
    const NamedAttribute &GetAttribute(uint32_t attribute) const { return NamedAttributes.at(attribute); }
    void ForEachNamedAttribute(uint32_t id, auto &&visit) const {
        for (uint32_t a = 0u; a < NamedAttributes.size(); ++a) {
            if (NamedAttributes[a].Record == id) visit(a, NamedAttributes[a]);
        }
    }
    template<AttributeScalar T> NamedAttributeView<T> AttributeView(uint32_t attribute) const {
        return {ValuesOf<T>(Buffers.Named).Get(TypedAttribute<T>(attribute).Values)};
    }
    // Writes `values` to the consecutive handles from `first`, capturing their pages.
    template<AttributeScalar T> void SetAttribute(uint32_t attribute, uint32_t first, std::span<const T> values) {
        if (values.empty()) return;
        const auto range = TypedAttribute<T>(attribute).Values;
        const auto local = first / MeshElementBlockSize, end = (first + uint32_t(values.size()) - 1u) / MeshElementBlockSize + 1u;
        const auto blocks = ValuesOf<T>(Buffers.Named).GetMutable({range.Offset + local, end - local});
        for (uint32_t i = 0u; i < values.size(); ++i) {
            const auto handle = first + i - local * MeshElementBlockSize;
            auto &block = blocks[handle / MeshElementBlockSize];
            const auto slot = handle % MeshElementBlockSize;
            if constexpr (std::same_as<T, bool>) {
                const auto bit = 1u << (slot % 32u);
                auto &mask = block[slot / 32u];
                mask = values[i] ? mask | bit : mask & ~bit;
            } else {
                block[slot] = values[i];
            }
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

    // Initialize selection masks and aggregates once per record.
    // Submit the chain before reading published summaries.
    void EnsureSelectionState(state::Registry &, mtl::ComputeChain &, std::span<const uint32_t> ids);
    SelectionView GetSelectedElements(uint32_t id, Element) const;
    SelectionView GetHiddenElements(uint32_t id, Element) const;
    uint32_t GetHiddenSlot(Element) const;
    void EditHiddenBlocks(Element element, std::span<const uint32_t> blocks, auto &&write) {
        auto &bits = element == Element::Vertex ? Buffers.VertexHidden : element == Element::Edge ? Buffers.EdgeHidden :
                                                                                                    Buffers.FaceHidden;
        bits.Buffer.CaptureWriteElements(blocks, sizeof(SelectionBlock));
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            const auto words = bits.GetMutable({blocks[first], uint32_t(count)});
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
            const auto words = bits.GetMutable({blocks[first], uint32_t(count)});
            for (size_t j = 0u; j < count; ++j) write(blocks[first + j], words[j]);
        });
    }
    // Refresh Blocks and blocks incident to Seeds.
    // A valid Source derives the other domains and submits once for capture.
    // Submit before reading the updated roots.
    struct SelectionUpdate {
        uint32_t StoreId;
        Element Source{Element::None};
        std::array<std::vector<uint32_t>, 3> Blocks{};
        std::vector<SelectionSeed> Seeds{};
    };
    void UpdateSelection(state::Registry &, mtl::ComputeChain &, std::span<const SelectionUpdate>);
    // Derive the other two domains in the supplied masks using UpdateSelection rules.
    // Preserve stored selection and aggregates; include hidden elements.
    void DeriveSelectionMasks(state::Registry &, mtl::ComputeChain &, uint32_t id, const GeometryMasks &, Element source) const;
    // Clear selection, copy hidden masks from corresponding source blocks, and refresh changed aggregates.
    void ResetEditState(state::Registry &, mtl::ComputeChain &, uint32_t id, uint32_t source);
    // Apply selection masks and refresh changed aggregates.
    void SelectMasks(state::Registry &, mtl::ComputeChain &, uint32_t id, const GeometryMasks &);
    void PublishSelectionSummary(uint32_t id);
    EditSelectionSummary &WriteSelectionSummary(uint32_t id);
    void SetSelectionBaseline(uint32_t id, Element, std::vector<std::pair<uint32_t, SelectionBlock>>, uint32_t active);
    // Builds a new clone's index after its GPU block copies have completed.
    void RebuildSelectionIndex(std::span<const uint32_t> ids);
    // Gather selected handles in ascending order into a new output range and return it.
    Range GatherSelectedElements(state::Registry &, mtl::ComputeChain &, uint32_t id, Element, BufferArena<uint32_t> &output) const;
    bool IsLiveElement(uint32_t id, Element, ElementHandle) const;
    uint32_t GetSelectionSlot(Element) const;
    EditSelectionStorage GetEditSelectionStorage(uint32_t id) const;
    const EditSelectionSummary &GetSelectionSummary(uint32_t id) const;

    SharpnessSummary GetEdgeSharpnessSummary(uint32_t id) const;
    // Returns the class-buffer offset or a uniform-class sentinel.
    uint32_t GetCornerClassMode(uint32_t id) const;
    CornerNormalView GetCornerNormalView(uint32_t id) const;
    // Return normals in triangulated face-fan order, valid until the next call.
    // Requires base normals derived since the last position or sharpness write.
    std::span<const vec3> GetCornerNormals(const Mesh &) const;
    // Classify incoming corners of the supplied vertices, bounded by Incoming; absent work selects all vertices.
    // Run after sharpness writes and before deriving base normals.
    struct CornerClassRequest {
        uint32_t StoreId;
        ElementWork Vertices{};
        uint32_t VertexCount{}, Incoming{};
        bool Complete{};
    };
    // Encode counts and required root blocks, then submit before Plan.
    // Plan assigns classification modes, allocates mixed-sector ranges, and records root writes.
    CornerClassBatch EncodeCornerClassification(state::Registry &, mtl::ComputeChain &, std::span<const CornerClassRequest>);
    void PlanCornerClassification(state::Registry &, mtl::ComputeChain &, CornerClassBatch &);
    // Classify all corners of the listed meshes, submitting once for planning.
    void UpdateCornerClassification(state::Registry &, mtl::ComputeChain &, std::span<const uint32_t> ids);
    // Enumerates incident edges from canonical corner links, including boundaries.
    VertexEdgeIncidence GetVertexEdgeIncidence(uint32_t id) const;

private:
    MeshArenas Buffers;
    MeshSlots SlotTable;
    std::vector<Record> Records{};
    std::vector<DerivedRecord> DerivedRecords{};
    // Cache block lists by record, domain, and membership revision.
    // Lock stale-list updates to permit concurrent const reads.
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
    std::vector<uint32_t> Unshaded{};
    std::vector<NamedAttribute> NamedAttributes{};

    struct HistoryState;
    std::unique_ptr<HistoryState> Tracked;
    struct TrackOwner {
        store::Pages *Pages{};
        store::Records *Records{};
    };
    store::History *Hist{};
    // Each mesh track's index in the history and its owner, in history registration order.
    std::vector<size_t> TrackIndices;
    std::vector<TrackOwner> Owners;
    // Each owner's position in Owners, by its Pages or Records.
    std::unordered_map<const void *, size_t> OwnerOf;
    // Versions of every tracked mesh page and record, which a restore returns to exactly.
    struct Version {
        std::vector<store::Version> Tracks;
    };
    void Release(Version &);
    struct StoredState;
    // The kept versions stay within this count, since every pinned version slows the captures and restores of the tracks it shares.
    static constexpr size_t MaxKeptStates = 8;
    std::unordered_map<store::Hash128, Version, store::Hash128Hasher> KeptStates;
    // The kept states in the order they were kept.
    std::vector<store::Hash128> KeptOrder;
    // The kept states the last history step referred to.
    std::vector<store::Hash128> StepStates;

    // Pairs of source and destination arena blocks, indexed by ElementDomain.
    using BlockPairs = std::array<std::vector<uvec2>, 6>;
    // Pair blocks at matching local indices.
    BlockPairs PairBlocks(uint32_t source, uint32_t destination) const;

    // Clone render sets and payloads into fresh storage.
    // RefreshMeshBinding must update the clone GPU records.
    void CloneRenderRecords(CloneCopies &, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids);
    void ReleaseBlockLists(uint32_t id);
    void ReleaseBlockLists(std::span<const uint32_t> ids);
    // Retain replaced words until any submitted frame completes.
    void RetireBlockListWords(Range) const;
    void FinishEraseElements(uint32_t id, ElementDomain, std::span<const uint32_t> blocks);
    uint32_t AcquireId(Record &&);
    void SyncMirrors();
    // Copy point normals for face-less meshes; initialize triangle mesh normals to zero for derivation.
    void FillBaseVertexNormalMirror(ElementSetRef vertices, Range point_normals);

    template<AttributeScalar T> const NamedAttribute &TypedAttribute(uint32_t attribute) const {
        const auto &entry = NamedAttributes.at(attribute);
        if (entry.Record == InvalidOffset || entry.Type != AttributeTypeOf<T>) throw std::invalid_argument("Attribute type differs from its access.");
        return entry;
    }
    NamedAttribute &WriteAttribute(uint32_t attribute);
    uint32_t AcquireAttribute(NamedAttribute &&);
    // Copy paired source blocks into destination blocks with matching live slots.
    // Values, mirrors, masks and vertex fans copy byte for byte.
    void CopyBlocks(CloneCopies &, uint32_t destination, uint32_t source, const BlockPairs &pairs);
};

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
constexpr MeshStore::ElementDomain ElementDomainOf(AttributeDomain domain) {
    switch (domain) {
        case AttributeDomain::Vertex: return MeshStore::ElementDomain::Vertex;
        case AttributeDomain::Edge: return MeshStore::ElementDomain::Edge;
        case AttributeDomain::Face: return MeshStore::ElementDomain::Face;
        case AttributeDomain::Corner: return MeshStore::ElementDomain::Halfedge;
    }
    std::unreachable();
}
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
decltype(auto) WithPayload(auto &render, auto &record, RenderPayload payload, auto &&fn) {
    switch (payload) {
        case RenderPayload::TriangleIds: return fn(render.MeshletTriangleIds, record.MeshletTriangles);
        case RenderPayload::VertexCorners: return fn(render.MeshletVertexCorners, record.MeshletVertices);
        case RenderPayload::LocalTriangles: return fn(render.MeshletLocalTriangles, record.MeshletLocalTriangles);
        case RenderPayload::GroupClusters: return fn(render.GroupClusterIds, record.GroupClusters);
    }
    std::unreachable();
}
inline MeshletPayloadBases PayloadBasesOf(const MeshStore::Record &record) { return {record.MeshletTriangles.Values.Offset, record.MeshletVertices.Values.Offset, record.MeshletLocalTriangles.Values.Offset}; }

// The live record the entity's instances draw, or null.
const MeshStore::Record *TryRecordOf(const state::Registry &, state::Entity);
const MeshStore::Record &RecordOf(const state::Registry &, state::Entity);
// Captures the record the entity's instances draw before a write.
MeshStore::Record &EditRecordOf(state::Registry &, state::Entity);
