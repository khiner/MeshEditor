#pragma once

#include "Mesh.h"
#include "MeshAttributes.h"
#include "MeshData.h"
#include "MorphTargetData.h"
#include "Range.h"
#include "TetBuffers.h"
#include "gpu/BoneDeformVertex.h"
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
#include "gpu/MorphTargetVertex.h"
#include "gpu/NormalSector.h"
#include "gpu/SelectionAggregate.h"
#include "gpu/SelectionUpdatePushConstants.h"
#include "gpu/SlotOffset.h"
#include "mesh/ElementArena.h"
#include "mesh/ElementAttribute.h"
#include "mesh/ElementAttributeView.h"
#include "mesh/CornerNormalView.h"
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

namespace mtl { struct ComputeChain; }
struct MeshPipelines;

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

// Corner-domain attribute layers in polygon loop order, empty where the source lacks the channel.
struct CornerLayers {
    std::vector<vec4> Tangents, Colors;
    std::array<std::vector<vec2>, 4> Uvs;
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
    // Derived aggregates mirroring each selectable domain's blocks, rebuilt from masks, positions, sharpness and connectivity.
    BufferArena<SelectionAggregate> VertexAggregates, EdgeAggregates, FaceAggregates;
    // Derived, each record's vertex, edge and face roots: its block aggregates reduced in ascending block order.
    BufferArena<SelectionAggregate> SelectionRoots;
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
    enum class ElementDomain { None, Vertex, Halfedge, Edge, Face, Triangle };

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
    struct Record {
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
    // Without a list the handles form one run, and otherwise a list of nonconsecutive handles goes to `list`, which needs a bindless slot.
    // New elements have their enabled attributes at defaults and no selection, and their value pages are captured.
    // After emitting values, refresh aggregates for affected selectable blocks before publishing the completed edit.
    ElementHandleRange InsertElements(uint32_t id, ElementDomain, uint32_t count, mtl::Buffer *list);
    // Clears the elements the work names, with their selection, and returns the blocks whose membership changed.
    // The caller refreshes those blocks' aggregates.
    std::vector<uint32_t> EraseElements(uint32_t id, ElementDomain, const BufferArena<uint32_t> &storage, ElementWork);
    // Clears the handles after the first `used` of an insertion made for a bound, listed in `list` when not a run, and shrinks `inserted` to the used ones.
    // The cleared handles never held selection.
    void TrimInsertedElements(uint32_t id, ElementDomain, ElementHandleRange &inserted, const mtl::Buffer &list, uint32_t used);

    void Track(store::History &);
    void FinishRestore();
    // Indexes the records written since the last index, so a restore maps its changed bytes to records in proportion to the change.
    void IndexHistory();
    std::vector<Change> TakeChanges();
    // Refreshes the selection aggregates of restored blocks and their incident blocks.
    void ReconcileSelection(state::Scene &, std::span<const Change>);
    // Records whose render data is stale, released or changed by a restore, for the render sync to drop.
    std::vector<uint32_t> TakeRenderStale() { return std::exchange(RenderStale, {}); }

    // Capture destination pages before dispatching GPU writes to Persistent mesh data.
    void CaptureVertexEdit(uint32_t id);
    void CaptureSelectionSummary(uint32_t id);
    void CaptureSelectionBlocks(Element, std::span<const uint32_t> blocks);
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
    void AllocateConnectivity(uint32_t id, uint32_t halfedge_count, uint32_t face_count, bool face_starts,
                              std::span<const uint32_t> face_offsets = {},
                              std::span<const std::array<uint32_t,2>> wire_edges = {});
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
    // A face-less line mesh changes its entire render domain when its first
    // face is created. Retire line incidence while preserving vertex handles.
    void RetireLineConnectivity(state::Scene &, uint32_t id);
    // Returns the clone's store ID.
    uint32_t CloneMesh(const Mesh &, const MeshPipelines &);
    // Returns a vertex-only store ID that must be released with Release.
    uint32_t AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &);
    void Release(uint32_t id);
    void Release(std::span<const uint32_t> ids);
    // Reset all arenas and the StoreId table to empty, keeping GPU allocations for reuse.
    // Requires a full scene clear without live StoreId references so allocation restarts deterministically.
    void Clear();

    const Record &Get(uint32_t id) const { return Records.at(id); }
    const DerivedRecord &GetDerived(uint32_t id) const { return DerivedRecords.at(id); }
    const MeshArenas &Arenas() const { return Buffers; }
    const MeshSlots &Slots() const { return SlotTable; }

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
    void ReleaseSoundVertices(Range);

    // Selection masks mirror canonical element blocks.
    // Only summaries, roots and gesture state are per mesh.
    // Creates cleared masks, a summary and the aggregates of every owned block, once per record.
    void EnsureSelectionState(state::Scene &, std::span<const uint32_t> ids);
    SelectionView GetSelectedElements(uint32_t id, Element) const;
    // Empty for a mesh without faces.
    BoundaryEdgeView GetBoundaryEdges(uint32_t id) const;
    const SelectionAggregate &GetSelectionRoot(uint32_t id, Element) const;
    // The vertex root, followed by the edge and face roots.
    SlotOffset GetSelectionRoots(uint32_t id) const;
    // Captures the listed ascending mask blocks and writes each with `write(block, words)`.
    void EditSelectionBlocks(Element element, std::span<const uint32_t> blocks, auto &&write) {
        auto &bits = element == Element::Vertex ? Buffers.VertexSelection : element == Element::Edge ? Buffers.EdgeSelection : Buffers.FaceSelection;
        bits.Buffer.CaptureWriteElements(blocks, sizeof(MeshArenas::SelectionBlock));
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            auto words = bits.GetMutable({blocks[first], uint32_t(count)});
            for (size_t j = 0u; j < count; ++j) write(blocks[first + j], words[j]);
        });
    }
    // Aggregates of `Blocks` refresh on the GPU, and each seed's incident blocks join them.
    // A valid Source first rewrites the other two domains' words in those blocks from its words.
    // The mesh's roots are then reduced again.
    struct SelectionUpdate {
        uint32_t StoreId;
        Element Source{Element::None};
        std::array<std::vector<uint32_t>, 3> Blocks{};
        std::vector<SelectionSeed> Seeds{};
    };
    void UpdateSelection(state::Scene &, std::span<const SelectionUpdate>);
    // Copies the root counts, sums and sharpness of the summary's mode into the summary.
    void PublishSelectionSummary(uint32_t id);
    EditSelectionSummary &WriteSelectionSummary(uint32_t id);
    void SetSelectionBaseline(uint32_t id, Element, std::vector<std::pair<uint32_t, MeshArenas::SelectionBlock>>, uint32_t active);
    // Writes the selected handles in ascending order.
    void GatherSelectedElements(state::Scene &, uint32_t id, Element, mtl::Buffer &output) const;
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
    CornerClassUpdate EncodeCornerClassification(state::Scene &, mtl::ComputeChain &, uint32_t id, ElementWork vertices = {},
                                                 uint32_t vertex_count = 0u, uint32_t incoming = 0u, bool complete = false);
    void PlanCornerClassification(state::Scene &, mtl::ComputeChain &, CornerClassUpdate &);
    void FinishCornerClassification(const mtl::ComputeChain &, const CornerClassUpdate &);
    // Classifies every corner of each listed mesh, through the three steps over one chain.
    void UpdateCornerClassification(state::Scene &, std::span<const uint32_t> ids);
    // Enumerates incident edges from canonical corner links, including boundaries.
    VertexEdgeIncidence GetVertexEdgeIncidence(uint32_t id) const;

private:
    MeshArenas Buffers;
    MeshSlots SlotTable;
    std::vector<Record> Records{};
    std::vector<DerivedRecord> DerivedRecords{};
    // Block lists, keyed by record and element domain, with the membership revision and restore epoch they describe.
    // Const reads fill a stale list under the lock, so concurrent reads between membership changes stay safe.
    struct BlockListEntry {
        ElementSetRef Set{};
        uint32_t Revision{};
        uint64_t Epoch{};
        Range Words{};
    };
    mutable std::mutex BlockListLock;
    uint64_t BlockListEpoch{}; // Advances with each restore
    mutable BufferArena<uint32_t> BlockLists;
    mutable std::vector<std::array<BlockListEntry, 5>> BlockListEntries{};
    bool FrameReadsBlockLists{};
    mutable std::vector<Range> RetiredBlockLists{}; // Replaced list words a submitted frame can still read
    BufferArena<uint32_t> SelectionWork; // Selection update scratch, retained so its capacity is reused
    BufferArena<uint32_t> SelectionDirty; // Dirty-block bits at SelectionDirtyWord, clear between updates
    uint64_t NextNormalRevision{};
    std::vector<uint32_t> FreeIds{};
    std::vector<uint32_t> RenderStale{};

    struct HistoryState;
    std::unique_ptr<HistoryState> Tracked;

    Record &WriteRecord(uint32_t id);
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
