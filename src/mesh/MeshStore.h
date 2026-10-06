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
#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/LodNode.h"
#include "gpu/MeshRecord.h"
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
#include "mesh/MeshletIndex.h"
#include "mesh/PoseAttributeView.h"
#include "mesh/SelectionQuery.h"
#include "mesh/SelectionView.h"
#include "mesh/VertexFanStore.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"
#include "numeric/uvec2.h"
#include "numeric/uvec4.h"
#include "numeric/vec2.h"

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

// The render arenas store records own: finest clusters and their payloads, the cluster LOD DAG and its traversal nodes, element owners and GPU mesh records.
struct RenderArenas {
    explicit RenderArenas(mtl::BufferContext &);

    BufferArena<uint32_t> ExtrasFaces, ExtrasEdges; // Bone and joint index data, offset by each record's first vertex
    BufferArena<MeshletRecord> Meshlets;
    BufferArena<MeshletSpatialNode> MeshletSpatialNodes; // Mirrors Meshlets, so an edit's new clusters extend it by their own count
    MeshletIndex ActiveMeshlets;
    BufferArena<uint32_t> MeshletTriangleIds, MeshletVertexCorners;
    BufferArena<uint8_t> MeshletLocalTriangles;
    // The cluster LOD DAG: one group per simplification step, and the selection forest over them.
    BufferArena<ClusterGroup> ClusterGroups;
    BufferArena<LodNode> LodNodes;
    // Reverse edit dependencies, read only by repairs. MeshletLodLeaves mirrors Meshlets and LodParents mirrors LodNodes.
    BufferArena<uint32_t> MeshletLodLeaves, LodParents;
    // A group's inputs and proxies occupy packed runs of GroupClusterIds that group addresses own.
    BufferArena<ClusterGroupLinks> GroupLinks;
    BufferArena<uint32_t> GroupClusterIds;
    BufferArena<PrimitiveRecord> Primitives;
    // Each render owner's run of render primitive handles, indexed by source primitive and InvalidOffset where absent.
    BufferArena<uint32_t> PrimitiveRoutes;
    BufferArena<MeshRecord> MeshRecords; // One GPU mesh record per store id, rebuilt from the record's current bindings
    // Canonical triangle, edge, and vertex handles map to global finest meshlet IDs.
    std::array<ElementAttribute<uint32_t>, 3> ElementMeshlets;

    // Allocates `count` cluster records and extends the LOD leaf and spatial mirrors over them.
    Range AllocateMeshlets(uint32_t count);
    // Releases the clusters' payload ranges, which their records name, and their identities.
    void ReleaseMeshletStorage(std::span<const uint32_t> handles);
};

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
    using SelectionBlock = std::array<uint32_t, MeshElementBlockWords>;
    BufferArena<SelectionBlock> VertexSelection, EdgeSelection, FaceSelection; // Canonical domain block masks
    BufferArena<SelectionBlock> VertexHidden, EdgeHidden, FaceHidden; // Persistent edit-mode visibility masks
    // Derived aggregates mirroring each selectable domain's blocks, rebuilt from masks, positions, sharpness and connectivity.
    BufferArena<SelectionAggregate> VertexAggregates, EdgeAggregates, FaceAggregates;
    SelectionIndex SelectionTree;
    SelectionQuery Query;
    BufferArena<EditSelectionSummary> SelectionSummary; // One summary per mesh
    BufferArena<uint8_t> EdgeSharpness; // One byte per edge, 1 = sharp (canonical sharpness store)
    ElementAttribute<CustomNormal> CustomNormals; // Optional angular offsets owned by canonical polygon corners
    BufferArena<vec3> PointNormals; // Authored normals of face-less meshes, in vertex order
    ElementAttribute<vec4> CornerTangents, CornerColors, VertexColors;
    std::array<ElementAttribute<vec2>, 4> CornerUvs;
    ElementAttribute<uint32_t> FacePrimitives, VertexPrimitives; // Optional assignments at canonical element handles
    BufferArena<uint32_t> PrimitiveMaterials; // Primitive index -> material index
    ElementAttribute<BoneDeformVertex> Skin; // Canonical handle-addressed skin data
    ElementAttribute<MorphTargetVertex> Morph; // One payload block per target at each vertex block, so targets are target-major within a block
    // Canonical tetrahedral wireframe geometry, one range per mesh that carries a modal solve.
    BufferArena<vec3> TetPositions;
    BufferArena<uint32_t> TetEdgeIndices; // Two indices per tet edge
    // Excitable vertex handles per sounding mesh, rebuilt from the sound model after serialization.
    BufferArena<uint32_t> SoundVertices;
    ElementAttribute<uint32_t> CornerSectors; // Canonical sector-root handles, and absent/InvalidOffset means the vertex normal
    ElementAttribute<NormalSector> NormalSectors; // Canonical base normals at sector roots
    BufferArena<vec3> BaseVertexNormals; // Mirrors Vertices: derived smooth normals for triangle meshes, authored normals for face-less meshes
    BufferArena<vec3> BaseFaceNormals; // Mirrors FaceTriangles, one derived face normal per face slot
    RenderArenas Render;
};

// Bindless slots of the arenas shaders address, fixed for the store's lifetime.
struct MeshSlots {
    uint32_t Vertices, FaceTriangleStart, FaceSharpness, EdgeSharpness;
    uint32_t PrimitiveMaterial;
    ElementAttributeRef Skin;
    ElementAttributeRef Morph;
    uint32_t TetPosition, TetEdgeIndex, SoundVertex;
    ElementAttributeRef CornerSector, NormalSector;
    uint32_t BaseVertexNormal, BaseFaceNormal;
};

// Composes face, vertex, or sector normals from canonical sharpness and root references.
vec3 ComposeCornerNormal(CornerAttributeView<uint32_t> sectors, ElementAttributeView<NormalSector> normal_sectors, std::span<const uint8_t> sharpness, uint32_t mode, uint32_t ci, TriangleVertexView vertices, TriangleFaceView face_ids, const CornerNormalSources &);

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
        std::vector<Range> VertexRanges{}; // Canonical vertex handles, independent of mesh layout.
        // Ascending canonical vertex, edge, face and halfedge blocks whose restored bytes changed.
        std::array<std::vector<uint32_t>, 4> Blocks{};
    };

    // Canonical domains own stable sets, selection masks and optional attribute blocks.
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
        uint32_t SectorBlockCount{}; // Corner blocks holding sector roots.
        uint32_t Classification{uint32_t(CornerClassMode::UniformVertex)};
        bool MorphShadingAuthored{};
        Range ExtrasFaces{}, ExtrasEdges{}; // Bone and joint index data, present only on an extras record.
        Range Primitives{}, Meshlets{}, MeshletTriangles{}, MeshletVertices{}, MeshletLocalTriangles{};
        // Published sparse roots own cluster/payload and primitive allocations.
        // Before publication, construction ranges own their provisional storage.
        uint32_t MeshletRoot{InvalidOffset}, PrimitiveRoot{InvalidOffset}, SpatialRoot{InvalidOffset};
        Range PrimitiveRoutes{}; // Three render primitive handles per source material, indexed by topology.
        uint64_t MeshletRevision{};
        uint32_t Level0Count{};
        uint32_t RenderTopologies{}; // Bit mask of published triangle, line and point topology; zero before the first build.
        // Element ownership is independent for triangles, loose edges and isolated vertices.
        std::array<uint32_t, 3> ElementMeshletOrigins{InvalidOffset, InvalidOffset, InvalidOffset}, ElementMeshletBlockCounts{};
        // Meshes without coarse geometry use one unpruned span node per primitive.
        // Ranges describe construction placement, and roots own the live allocations.
        Range ClusterGroups{}, LodNodes{}, CoarseVertices{}, CoarseLocalTriangles{};
        uint32_t GroupRoot{InvalidOffset}, NodeRoot{InvalidOffset};
        uint32_t LodDepth{}; // The depth of the record's deepest traversal tree.
        // Finest meshlets whose canonical positions changed since coarse repair.
        uint32_t PositionDirtyRoot{InvalidOffset};
        // Stale LOD groups, whose coarse clusters rebuild when the mesh leaves edit mode.
        uint32_t DirtyGroupRoot{InvalidOffset};
        bool Alive{false};
    };

    // Transient interaction storage and pose-cache invalidation. Shading values
    // and their sparse ownership restore with the canonical mesh.
    struct DerivedRecord {
        // The source-domain mask blocks and active element saved when an additive gesture began.
        std::vector<std::pair<uint32_t, MeshArenas::SelectionBlock>> SelectionBaseline{};
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
    // A mesh domain's owned blocks in ascending order and the inclusive live-element prefix over them.
    // It is rebuilt when first read after the domain's membership changes.
    struct BlockList {
        std::span<const uint32_t> Blocks, LivePrefix;
        SlotOffset Gpu; // The blocks, followed by the prefix.
    };
    BlockList GetBlockList(uint32_t id, ElementDomain) const;
    // A submitted frame reads the block lists until it completes, so a list replaced meanwhile keeps its words until then.
    void FrameSubmitted();
    void FrameCompleted();
    // Packed ordinals count live elements in ascending handle order.
    uint32_t LiveElementAt(uint32_t id, ElementDomain, uint32_t ordinal) const;
    uint32_t LiveElementOrdinal(uint32_t id, ElementDomain, uint32_t handle) const;
    // The fan item runs that vertex root rebuilds free and allocate.
    VertexFanStore &VertexFans() { return Buffers.VertexFans; }
    TriangleCorners GetTriangleCorners(uint32_t id) const;
    // The mesh's live derived triangles and vertices in ordinal order.
    ElementView<uvec3> TriangleView(uint32_t id) const;
    ElementView<Vertex> VertexView(uint32_t id) const;
    // Completes the record created by CreateMeshSource: face tables, corner layers, primitive tables, and smooth sharpness stores.
    void CreateMesh(uint32_t id, const MeshData &, const MeshVertexAttributes &, const MeshPrimitives &, const CornerLayers &, bool has_authored_normals);
    // Clones each source record, queueing its GPU copies on `copies`, and returns the clones' store IDs in source order.
    // The clones read their copied ranges once the copies record and their chain submits.
    std::vector<uint32_t> CloneMeshes(CloneCopies &, std::span<const uint32_t> source_ids);
    // Returns a vertex-only store ID that must be released with Release.
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &);
    // Stores bone or joint index data for a vertex-only record, offset by its first vertex.
    void SetExtrasIndices(uint32_t id, std::span<const uint32_t> faces, std::span<const uint32_t> edges);
    // Releases the records' canonical and render storage.
    void Release(uint32_t id);
    void Release(std::span<const uint32_t> ids);
    // Releases the records' clusters, hierarchy, nodes and owners and resets those fields, for store records and fragments alike.
    void ReleaseRender(std::span<Record *const>);
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

    uint32_t MeshletCount(const Record &r) const { return Buffers.Render.ActiveMeshlets.Count(r.MeshletRoot); }
    uint32_t FirstMeshlet(const Record &r) const { return Buffers.Render.ActiveMeshlets.First(r.MeshletRoot); }
    uint32_t PrimitiveCount(const Record &r) const { return Buffers.Render.ActiveMeshlets.Count(r.PrimitiveRoot); }
    uint32_t ClusterGroupCount(const Record &r) const { return Buffers.Render.ActiveMeshlets.Count(r.GroupRoot); }
    void ForEachPrimitive(const Record &r, auto &&fn) const {
        Buffers.Render.ActiveMeshlets.ForEach(r.PrimitiveRoot, [&](uint32_t id) { fn(id, Buffers.Render.Primitives.Get({id, 1u})[0]); });
    }
    void ForEachLodNode(const Record &r, auto &&fn) const {
        Buffers.Render.ActiveMeshlets.ForEach(r.NodeRoot, [&](uint32_t id) { fn(id, Buffers.Render.LodNodes.Get({id, 1u})[0]); });
    }
    uint32_t PrimitiveRoute(const Record &r, uint32_t source_primitive, uint32_t topology) const {
        const uint64_t route = 3ull * source_primitive + topology;
        return topology < 3u && route < r.PrimitiveRoutes.Count ? Buffers.Render.PrimitiveRoutes.Get({r.PrimitiveRoutes.Offset + uint32_t(route), 1u})[0] : InvalidOffset;
    }
    // Grows the record's routes to cover `count` source primitives, keeping its routes.
    void ReservePrimitiveRoutes(Record &, uint32_t count);
    // The element blocks holding the record's meshlet owner payloads, in ascending order.
    std::vector<uint32_t> MeshletOwnerBlocks(const Record &, uint32_t topology) const;
    // The element domain the finest clusters of a render topology name: triangles, edges or vertices.
    static ElementDomain RenderDomain(uint32_t topology) { return topology == 0u ? ElementDomain::Triangle : topology == 1u ? ElementDomain::Edge :
                                                                                                                              ElementDomain::Vertex; }
    // Visits the arena of the record's elements drawn as `topology` with the record's set in it.
    decltype(auto) WithRenderDomain(const Record &, uint32_t topology, auto &&fn) const;
    // The first element handle of the record's elements drawn as `topology`.
    uint32_t RenderDomainFirst(const Record &, uint32_t topology) const;

    // Mutable views over Persistent arena data, capturing the pages they expose.
    std::span<uint32_t> EditPrimitiveMaterials(uint32_t id);
    // Callers writing the sharpness stores rederive corner normals afterward.
    std::span<uint8_t> EditFaceSharpness(uint32_t id);
    std::span<uint8_t> EditEdgeSharpness(uint32_t id);
    // Installs the custom corner-normal layer: one mask pair per 32 corners and the offsets packed to the masked corners.
    void SetCustomCornerNormals(uint32_t id, std::span<const CustomNormal> offsets);
    void SetMorphShadingAuthored(uint32_t id, bool);

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
        bits.Buffer.CaptureWriteElements(blocks, sizeof(MeshArenas::SelectionBlock));
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
        bits.Buffer.CaptureWriteElements(blocks, sizeof(MeshArenas::SelectionBlock));
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
    // Copies the root counts, sums and sharpness of the summary's mode into the summary.
    void PublishSelectionSummary(uint32_t id);
    EditSelectionSummary &WriteSelectionSummary(uint32_t id);
    void SetSelectionBaseline(uint32_t id, Element, std::vector<std::pair<uint32_t, MeshArenas::SelectionBlock>>, uint32_t active);
    // Builds a new clone's index after its GPU block copies have completed.
    void RebuildSelectionIndex(std::span<const uint32_t> ids);
    // Records the gather of the selected handles in ascending order into a range it allocates in `output`, and returns that range.
    Range GatherSelectedElements(state::Scene &, mtl::ComputeChain &, uint32_t id, Element, BufferArena<uint32_t> &output) const;
    bool IsLiveElement(uint32_t id, Element, uint32_t handle) const;
    uint32_t GetSelectionBitOffset(uint32_t id, Element) const;
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
    // Encode records the counts and the payload plan.
    // Plan attaches payloads once the chain submits, records the writes, and settles the mesh's classification mode.
    // Finish releases emptied payloads once the chain submits again.
    struct CornerClassUpdate {
        uint32_t StoreId{};
        CornerClassificationPushConstants Pc{};
        uint32_t VertexCount{}, Incoming{};
        std::vector<uint32_t> Dirty{};
        bool Complete{};
    };
    CornerClassUpdate EncodeCornerClassification(state::Scene &, mtl::ComputeChain &, uint32_t id, ElementWork vertices = {}, uint32_t vertex_count = 0u, uint32_t incoming = 0u, bool complete = false);
    void PlanCornerClassification(state::Scene &, mtl::ComputeChain &, CornerClassUpdate &);
    void FinishCornerClassification(const mtl::ComputeChain &, const CornerClassUpdate &);
    // Classifies every corner of each listed mesh, through the three steps over the chain.
    // It submits the chain once for the plan, and the finish runs with the chain's next submit.
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

    struct HistoryState;
    std::unique_ptr<HistoryState> Tracked;

    // Copies each source's finest clusters, hierarchy, traversal nodes, spatial tree and element owners onto its clone, with every reference rebased.
    // The clones' GPU mesh records are left for RefreshMeshBinding.
    void CloneRenderRecords(CloneCopies &, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids, std::span<const std::array<Range, 6>> maps);
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

// The live record the entity's instances draw, or null.
const MeshStore::Record *TryRecordOf(const state::Scene &, state::Entity);
const MeshStore::Record &RecordOf(const state::Scene &, state::Entity);
// Captures the record the entity's instances draw before a write.
MeshStore::Record &EditRecordOf(state::Scene &, state::Entity);
