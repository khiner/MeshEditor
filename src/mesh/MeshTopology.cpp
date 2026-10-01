#include "mesh/MeshTopology.h"
#include "mesh/MeshTopologyLayout.h"

#include "Profile.h"
#include "gpu/InsetVertexBasis.h"
#include "gpu/TiledJobPushConstants.h"
#include "mesh/NormalDeriveGpu.h"
#include "mesh/PageFootprint.h"
#include "mesh/ScratchChunks.h"
#include "mesh/SpatialFaceWork.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/ConnectivityBatch.h"
#include "mesh/ConnectivityWritePages.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/TopologyReadView.h"
#include "mesh/TopologyWritePages.h"
#include "mesh/VertexFanBuild.h"
#include "state/Scene.h"

MeshTopologyPushConstants TopologyPushConstants(const MeshStore &meshes) {
    const auto &a = meshes.Arenas();
    const MeshTopologyArenas arenas{
        .VertexSlot = a.Vertices.Buffer.Slot,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .FaceTriangleStartSlot = a.FaceTriangles.Buffer.Slot,
        .TriangleSlot = a.Triangles.Buffer.Slot,
        .EdgeSharpnessSlot = a.EdgeSharpness.Buffer.Slot,
        .FaceSharpnessSlot = a.FaceSharpness.Buffer.Slot,
        .FacePrimitives = a.FacePrimitives.Ref(),
        .Skin = a.Skin.Ref(),
        .Morph = a.Morph.Ref(),
        .CornerTangent = a.CornerTangents.Ref(),
        .CornerColor = a.CornerColors.Ref(),
        .VertexColor = a.VertexColors.Ref(),
        .CornerUvs = {a.CornerUvs[0].Ref(), a.CornerUvs[1].Ref(), a.CornerUvs[2].Ref(), a.CornerUvs[3].Ref()},
        .CustomNormals = a.CustomNormals.Ref(),
        .CornerSectors = a.CornerSectors.Ref(),
        .NormalSectors = a.NormalSectors.Ref(),
        .BaseVertexNormalSlot = a.BaseVertexNormals.Buffer.Slot,
        .BaseFaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
    };
    return {.Source = arenas, .Destination = arenas};
}

namespace {
enum Domain : uint32_t {
    TableEntries,
    SrcVertices,
    SrcHalfedges,
    SrcFaces,
    ListEntries,
    CountBlocks,
    Once,
    DstVertices,
    DstHalfedges,
    DstFaces,
    Collapse0, Collapse1, Collapse2, Collapse3, CollapseDigits,
    DomainCount
};
using Batch = TiledJobBatch<MeshTopologyJob, DomainCount>;

// Label rounds one submit encodes before the host checks convergence.
constexpr uint32_t LabelRounds{16};

constexpr std::array PreparePasses{
    TiledPass{MeshPass::TopologyZero, SrcVertices},
    TiledPass{MeshPass::TopologyMarkHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveLimitVertices, SrcVertices},
    TiledPass{MeshPass::TopologyMergeTable, TableEntries},
    TiledPass{MeshPass::TopologyMergeInsert, SrcVertices},
    TiledPass{MeshPass::TopologyMergeQuery, SrcVertices},
    TiledPass{MeshPass::TopologyListFill, ListEntries},
};
// One round of an iterating operator's label passes.
// The converge pass after each round zeroes the indirect arguments once a round changes nothing, so the rounds after it dispatch no threadgroups.
constexpr std::array LabelPasses{
    TiledPass{MeshPass::TopologyLink, SrcHalfedges, 0, true},
    TiledPass{MeshPass::TopologyJump, SrcFaces, 0, true},
    TiledPass{MeshPass::TopologyJoinBest, SrcFaces, 0, true},
    TiledPass{MeshPass::TopologyJoinMatch, SrcFaces, 0, true},
    TiledPass{MeshPass::TopologyJump, SrcVertices, 1, true},
};
// A joining line core records its distinct output lines in the emptied table once the targets are final.
constexpr std::array CountPasses{
    TiledPass{MeshPass::TopologyDissolveRegions, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveWalk, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveRevert, SrcHalfedges},
    TiledPass{MeshPass::TopologyMergeTable, TableEntries},
    TiledPass{MeshPass::TopologyLineKeys, SrcHalfedges},
    TiledPass{MeshPass::TopologyCountVertices, SrcVertices},
    TiledPass{MeshPass::TopologyCountHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyCountFaces, SrcFaces},
    TiledPass{MeshPass::TopologyScanBlockSum, CountBlocks},
    TiledPass{MeshPass::TopologyScanBlockPrefix, PerJob},
    TiledPass{MeshPass::TopologyScanOffsets, CountBlocks},
};
constexpr std::array OutputPasses{
    TiledPass{MeshPass::TopologyZeroVertices, DstVertices},
    TiledPass{MeshPass::TopologyScatterVertices, SrcVertices},
    TiledPass{MeshPass::TopologyScatterHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyScatterFaces, SrcFaces},
    TiledPass{MeshPass::TopologyFaceTables, DstFaces},
    TiledPass{MeshPass::TopologyGatherVertices, DstVertices},
    TiledPass{MeshPass::TopologyGatherCorners, DstHalfedges},
};

constexpr auto CollapseOutputPasses = [] {
    std::array<TiledPass, 1u + 8u * 3u + 4u + 3u + OutputPasses.size() + 1u> passes{};
    size_t next=0u;
    passes[next++]={MeshPass::TopologyCollapseKeys, Collapse0};
    for (uint32_t shift=0u; shift<32u; shift+=4u) {
        passes[next++]={MeshPass::TopologyCollapseHistogram, Collapse0, shift};
        passes[next++]={MeshPass::TopologyCollapsePrefix, CollapseDigits, shift};
        passes[next++]={MeshPass::TopologyCollapseScatter, Collapse0, shift};
    }
    for (uint32_t level=0u; level<4u; ++level)
        passes[next++]={MeshPass::TopologyCollapseReduce, Collapse0 + level, level};
    for (uint32_t level=3u; level--;)
        passes[next++]={MeshPass::TopologyCollapseCarry, Collapse0 + level, level};
    for (const auto pass:OutputPasses) passes[next++]=pass;
    passes[next]={MeshPass::TopologyCollapseCenters, Collapse0};
    return passes;
}();

std::span<const TiledPass> TopologyEmissionPasses(bool collapse) {
    return collapse ? std::span<const TiledPass>{CollapseOutputPasses} : std::span<const TiledPass>{OutputPasses};
}

struct FaceListReferences {
    std::vector<uint32_t> ExistingVertices;
    std::vector<uint32_t> Offsets;
    uint32_t FaceCount{};
};

FaceListReferences ParseFaceListReferences(const MeshTopologyTask &task) {
    // The existing bound parser validates polygon lengths and the terminal cursor.
    // These references are action metadata.
    // No geometry is read back.
    (void)TopologyOutputBounds(task,{});
    const auto &list=task.List;
    const auto new_vertices=list[0];
    const uint64_t attribute_source=1ull+3ull*new_vertices;
    if (list[attribute_source]>=task.AppendedBase) throw std::invalid_argument("Topology face list attribute source is not an existing vertex handle.");
    uint64_t cursor=attribute_source+1u;
    const auto faces=list[cursor++];
    FaceListReferences result;
    result.FaceCount=faces;
    result.Offsets.push_back(uint32_t(attribute_source));
    result.ExistingVertices.push_back(list[attribute_source]);
    for (uint32_t f=0u;f<faces;++f) {
        const auto count=list[cursor++];
        for (uint32_t i=0u;i<count;++i) {
            const auto handle=list[cursor+i];
            if (handle>=task.AppendedBase && uint64_t(handle)-task.AppendedBase>=new_vertices) {
                throw std::invalid_argument("Topology face list names a vertex outside its source and appended range.");
            }
            result.Offsets.push_back(uint32_t(cursor+i));
            if (handle<task.AppendedBase) result.ExistingVertices.push_back(handle);
        }
        cursor+=count;
    }
    return result;
}

MeshTopologyJob TopologyJob(const MeshStore &meshes, const MeshTopologyTask &task, const MeshClosure &source,
                            Range list, uint32_t collapse_count, const mtl::Buffer *collapse_vertices) {
    const auto &src = meshes.Get(task.SourceId);
    return {
        .Op = task.Op,
        .Flags = task.Flags,
        .Steps = task.Steps,
        .Param0 = task.Param0,
        .Param1 = task.Param1,
        .TargetVertex = task.TargetVertex,
        .TargetPosition = task.TargetPosition,
        .CopyRotation = task.CopyRotation,
        .CopyTranslation = task.CopyTranslation,
        .PlaneNormal = task.PlaneNormal,
        .PlaneOffset = task.PlaneOffset,
        .ScreenTransform = task.ScreenTransform,
        .Extent = task.Extent,
        .KnifeStart = task.KnifeStart,
        .KnifeEnd = task.KnifeEnd,
        .SrcVertexCount = source.Counts[0],
        .SrcHalfedgeCount = source.Counts[1],
        .SrcFaceCount = source.Counts[2],
        .SrcEdgeCount = source.Counts[3],
        .SrcVertexWork = source.Elements[0], .SrcHalfedgeWork = source.Elements[1],
        .SrcFaceWork = source.Elements[2], .SrcEdgeWork = source.Elements[3],
        .HasSkin = uint32_t(src.SkinBlocksReady),
        .CornerAttributes = src.CornerAttributes,
        .VertexAttributes = src.VertexAttributes,
        .ListOffset = OffsetOrInvalid(list),
        .MorphTargetCount = src.MorphBlocksReady ? src.MorphTargetCount : 0u,
        .CollapseCount = collapse_count,
        .CollapseVerticesSlot = collapse_vertices ? collapse_vertices->Slot : InvalidSlot,
    };
}

// Adds the laid-out job's source tiles, and no output yet.
void AddJob(Batch &batch, const MeshTopologyJob &job, const MeshTopologyTask &task) {
    const auto V = job.SrcVertexCount, H = job.SrcHalfedgeCount, F = job.SrcFaceCount;
    std::array<uint32_t, DomainCount> tiles{};
    tiles[TableEntries] = TileCount(TopologyTableWords(job.Op, {V, H, F}), TileElements);
    tiles[SrcVertices] = TileCount(V, TileElements);
    tiles[SrcHalfedges] = TileCount(H, TileElements);
    tiles[SrcFaces] = TileCount(F + 2u, TileElements);
    // A knife reads every source edge, and a selection or cut list reads each of its entries.
    const auto listed = (job.Flags & (TopologyFlagListSelects | TopologyFlagListCuts)) && !task.List.empty() ? task.List.front() : 0u;
    tiles[ListEntries] = TileCount((job.Flags & TopologyFlagScreenCuts) ? job.SrcEdgeCount : listed, TileElements);
    tiles[CountBlocks] = 3u * job.CountBlockCount;
    tiles[Once] = batch.Jobs.empty() ? 1u : 0u;
    for (uint32_t level = 0u, count = job.CollapseCount; count; ++level) {
        tiles[Collapse0 + level] = TileCount(count, TileElements);
        count = count > TileElements ? TileCount(count, TileElements) : 0u;
    }
    tiles[CollapseDigits] = job.CollapseCount ? 16u : 0u;
    batch.AddJob(job, tiles);
}

// Records the operator through its count scan. An iterating operator submits each label batch to check convergence.
void PrepareTopology(state::Scene &r, mtl::ComputeChain &chain, Batch &batch, const MeshTopologyPushConstants &pc, bool iterates) {
    const profile::CpuScope scope{"PrepareTopology"};
    const auto &pipelines = GetMeshPipelines(r);
    for (uint32_t rounds = LabelRounds;; rounds *= 2) {
        std::vector<TiledPass> passes(PreparePasses.begin(), PreparePasses.end());
        for (uint32_t round = 0; iterates && round < rounds; ++round) {
            passes.insert(passes.end(), LabelPasses.begin(), LabelPasses.end());
            passes.push_back({MeshPass::TopologyConverge, Once, (uint32_t(batch.Jobs.size()) << 8) | DomainCount});
        }
        passes.insert(passes.end(), CountPasses.begin(), CountPasses.end());
        batch.Encode(chain, pipelines, pc, passes);
        if (!iterates) return;
        chain.Submit();
        if (batch.IndirectGroups(SrcHalfedges) == 0) return;
    }
}

MeshStore::TopologyCounts ReadOutputCounts(const Batch &batch, const MeshTopologyJob &job, MeshStore::TopologyCounts bounds) {
    const auto scratch = batch.ScratchSpan();
    const auto total = [&](uint32_t q) { return scratch[job.CountsOffset + q * job.CountEntries + job.CountEntries - 1]; };
    const MeshStore::TopologyCounts counts{total(0), total(2), total(1)};
    if (counts.Vertices > bounds.Vertices || counts.Halfedges > bounds.Halfedges || counts.Faces > bounds.Faces || uint64_t(counts.Faces) * 3u > counts.Halfedges) {
        throw std::runtime_error("GPU topology counts exceed the reserved output bounds.");
    }
    return counts;
}

// Canonical handles become indices into sorted local source work.
// A stride of two keeps the cut parameter next to each edge handle.
Range RemapTopologyElementList(std::span<const uint32_t> list, BufferArena<uint32_t> &storage, ElementWork source_elements, uint32_t stride = 1u) {
    if ((stride!=1u && stride!=2u) || list.empty() || 1ull+uint64_t(list.front())*stride!=list.size()) throw std::invalid_argument("Invalid topology element list.");
    const auto range=storage.Allocate(uint32_t(list.size()));
    auto output=storage.GetMutable(range);
    std::ranges::copy(list,output.begin());
    const auto source=storage.Get(WorkStorageRange(source_elements));
    for (uint32_t i=0u;i<list.front();++i) {
        const auto at=1u+i*stride;
        output[at]=ElementWorkRank(source,source_elements,list[at]);
        if (output[at]==InvalidOffset) throw std::invalid_argument("Topology element list names a vertex or edge outside its local source.");
    }
    return range;
}

// Remaps only face-list vertex references, and position words and polygon lengths stay byte-for-byte intact.
// Appended vertex indices follow the local core.
Range RemapTopologyFaceList(std::span<const uint32_t> list, std::span<const uint32_t> reference_offsets, uint32_t appended_base,
                            uint32_t local_vertices, BufferArena<uint32_t> &storage, ElementWork source_vertices) {
    if (list.empty() || reference_offsets.empty()) throw std::invalid_argument("Invalid topology face list.");
    const auto range=storage.Allocate(uint32_t(list.size()));
    auto output=storage.GetMutable(range);
    std::ranges::copy(list,output.begin());
    const auto source=storage.Get(WorkStorageRange(source_vertices));
    for (const auto at:reference_offsets) {
        const auto handle=list[at];
        if (handle>=appended_base) {
            if (uint64_t(handle)-appended_base>=list[0]) throw std::invalid_argument("Topology face list names an invalid appended vertex.");
            output[at]=local_vertices+(handle-appended_base);
        } else {
            output[at]=ElementWorkRank(source,source_vertices,handle);
            if (output[at]==InvalidOffset) throw std::invalid_argument("Topology face list names a vertex outside its local source.");
        }
    }
    return range;
}

} // namespace

namespace {
enum TopologyRetirement : uint8_t { RetireNone, RetireVertices = 1u, RetireFaces = 2u, RetireBoth = 3u };
// Lines marks an operator with a rule for the lines of a mesh without faces.
struct LocalPolicy { uint8_t Retirement; Element Seed; bool Lines{}; };

// The operator owns its topology rule.
// The transaction owns publication and retirement.
// Keep every permission here so a new rule cannot be admitted to the local path with an inconsistent second allowlist.
std::optional<LocalPolicy> TopologyPolicy(MeshTopologyOp op) {
    switch (op) {
        case MeshTopologyOp::DeleteVertices:
        case MeshTopologyOp::DissolveVertices:
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse: return LocalPolicy{RetireBoth,Element::Vertex,true};
        case MeshTopologyOp::DissolveDegenerate: return LocalPolicy{RetireBoth,Element::Vertex};
        case MeshTopologyOp::DeleteEdges: return LocalPolicy{RetireBoth,Element::Edge,true};
        case MeshTopologyOp::DissolveEdges:
        case MeshTopologyOp::RotateEdges: return LocalPolicy{RetireBoth,Element::Edge};
        case MeshTopologyOp::DeleteFaces:
        case MeshTopologyOp::DissolveFaces:
        case MeshTopologyOp::DissolveLimited: return LocalPolicy{RetireBoth,Element::Face};
        case MeshTopologyOp::BevelVertices: return LocalPolicy{RetireVertices,Element::Vertex};
        case MeshTopologyOp::BevelEdges: return LocalPolicy{RetireVertices,Element::Edge};
        case MeshTopologyOp::DeleteOnlyEdgesFaces: return LocalPolicy{RetireFaces,Element::Edge};
        case MeshTopologyOp::DeleteOnlyFaces:
        case MeshTopologyOp::TrisToQuads: return LocalPolicy{RetireFaces,Element::Face};
        case MeshTopologyOp::ConnectVertices: return LocalPolicy{RetireNone,Element::Vertex};
        case MeshTopologyOp::AddFaces: return LocalPolicy{RetireNone,Element::Vertex,true};
        case MeshTopologyOp::ExtrudeEdges:
        case MeshTopologyOp::Subdivide: return LocalPolicy{RetireNone,Element::Edge,true};
        case MeshTopologyOp::EdgeSplit: return LocalPolicy{RetireNone,Element::Edge};
        case MeshTopologyOp::ExtrudeRegion:
        case MeshTopologyOp::ExtrudeFacesIndividual:
        case MeshTopologyOp::DuplicateFaces:
        case MeshTopologyOp::SplitFaces:
        case MeshTopologyOp::Triangulate:
        case MeshTopologyOp::Poke:
        case MeshTopologyOp::FlipNormals:
        case MeshTopologyOp::InsetRegion:
        case MeshTopologyOp::InsetIndividual:
        case MeshTopologyOp::Solidify: return LocalPolicy{RetireNone,Element::Face};
        case MeshTopologyOp::KeepSelectedFaces: return LocalPolicy{RetireNone,Element::Face};
        default: return std::nullopt;
    }
}

// Merges, face lists and vertex deletion keep their seed vertices without any face around them.
bool RetainsSeedVertices(MeshTopologyOp op) {
    return TopologyIsMerge(op) || op==MeshTopologyOp::AddFaces || op==MeshTopologyOp::DeleteVertices;
}

// Ascending unique blocks of both inputs.
std::vector<uint32_t> Union(std::vector<uint32_t> a, std::span<const uint32_t> b) {
    a.insert(a.end(), b.begin(), b.end());
    std::ranges::sort(a);
    a.erase(std::unique(a.begin(), a.end()), a.end());
    return a;
}

// Ascending blocks of inserted handles, a run or a list in `list`.
std::vector<uint32_t> InsertedBlocks(ElementHandleRange handles, const mtl::Buffer &list) {
    if (handles.Handles.Slot == InvalidSlot) return RunBlocks(handles.First, handles.Count);
    std::vector<uint32_t> blocks;
    for (const auto handle : list.GetSpan<uint32_t>({handles.Handles.Offset, handles.Count})) blocks.push_back(handle / MeshElementBlockSize);
    std::ranges::sort(blocks);
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    return blocks;
}
} // namespace

struct MeshTopologyEdit::Prepared {
    MeshStore::TopologyCounts Bounds;
    std::unique_ptr<mtl::Buffer> CollapseVertices;
    TopologyIdentityPolicy Identity;
    std::vector<uint32_t> OutputMaterials;
    MeshClosure Core, Neighborhood;
    std::optional<TopologyReadView> SourceView;
    Batch Jobs;
    MeshTopologyPushConstants Constants{};
    MeshStore::TopologyCounts Counts{};
    mtl::Buffer FaceList, EdgeList;
    ElementHandleRange NewFaces{}, NewEdges{};
    std::optional<ConnectivityBatch> Connectivity;
    MeshStore::CornerClassUpdate Classes{};

    Prepared(mtl::BufferContext &buffers, MeshStore::TopologyCounts bounds, std::unique_ptr<mtl::Buffer> collapse_vertices,
             TopologyIdentityPolicy identity, std::vector<uint32_t> materials, uint32_t scratch_words)
        : Bounds{bounds}, CollapseVertices{std::move(collapse_vertices)}, Identity{identity}, OutputMaterials{std::move(materials)}, Jobs{buffers,scratch_words,1u},
          FaceList{buffers,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace}, EdgeList{buffers,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace} {}
};

// The closures an edit records before construction's first submit.
// Listed marks a task naming seeds that must be live.
struct MeshTopologyEdit::Closures {
    std::optional<FaceListReferences> FaceList;
    std::optional<MeshClosure> Around;
    MeshClosure Core, Neighborhood;
    bool Listed;
    uint32_t CollapseCount{};
    std::unique_ptr<mtl::Buffer> CollapseVertices;
};

MeshTopologyEdit::MeshTopologyEdit(mtl::ComputeChain &chain, const MeshTopologyTask &task)
    : Chain{chain}, SourceId{task.SourceId}, StoreId{task.SourceId}, Op{task.Op},
      SourceTriangles{chain.Buffers, 0u, SlotType::Buffer,mtl::BufferLifetime::Workspace}, NewVertexList{chain.Buffers, 0u, SlotType::Buffer,mtl::BufferLifetime::Workspace},
      InsetBasis{chain.Buffers,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace} {}
MeshTopologyEdit::MeshTopologyEdit(MeshTopologyEdit &&) noexcept = default;
MeshTopologyEdit::~MeshTopologyEdit() = default;

std::optional<MeshTopologyEdit::Closures> MeshTopologyEdit::RecordClosures(state::Scene &r, const MeshTopologyTask &task) {
    auto &chain = Chain;
    // Every topology operator enters through this transaction's affected source closure.
    const bool fresh = task.Op == MeshTopologyOp::KeepSelectedFaces;
    const auto policy=TopologyPolicy(task.Op);
    if (!policy) throw std::invalid_argument("Topology source closure is not defined for this operator.");
    auto &meshes = r.Context.get<MeshStore>();
    const auto original = meshes.Get(StoreId);
    // A mesh without faces edits its lines through an edge core.
    // An operator without a line rule, or a subdivide that cuts by a list, a plane or a screen segment, has no source there.
    const bool lines = !Mesh{meshes,StoreId}.FaceCount();
    constexpr auto CutFlags = TopologyFlagListSelects | TopologyFlagListCuts | TopologyFlagPlaneCuts | TopologyFlagScreenCuts;
    if (lines && (!policy->Lines || (task.Op==MeshTopologyOp::Subdivide && (task.Flags & CutFlags)))) return std::nullopt;
    OriginalClassMode = original.Classification;
    {
        const profile::CpuScope stage{"TopologySelectionState"};
        meshes.EnsureSelectionState(r, std::array{StoreId});
    }
    uint32_t collapse_count{};
    std::unique_ptr<mtl::Buffer> collapse_vertices;
    std::optional<FaceListReferences> face_list;
    if (task.Op==MeshTopologyOp::AddFaces) {
        // Per-mesh action tasks may be prepared together.
        // An earlier mesh's publication can grow the shared arena before this task executes.
        if (task.AppendedBase>meshes.Arenas().Vertices.Capacity() ||
            task.AppendedBase%MeshElementBlockSize)
            throw std::invalid_argument("Topology face list has an invalid vertex append base.");
        face_list=ParseFaceListReferences(task);
        if (!face_list->FaceCount) return std::nullopt;
        if (!Mesh{meshes,StoreId}.FaceCount() && Mesh{meshes,StoreId}.EdgeCount()) meshes.RetireLineConnectivity(r,StoreId);
    }
    const bool select_all = (task.Flags & TopologyFlagSelectAll) != 0u;
    const bool spatial = ((task.Op==MeshTopologyOp::DeleteFaces || task.Op==MeshTopologyOp::DeleteOnlyFaces) &&
        (task.Flags & TopologyFlagPlaneSide)) || (task.Op==MeshTopologyOp::Subdivide &&
        (task.Flags & (TopologyFlagPlaneCuts | TopologyFlagScreenCuts)));
    // Every level is bounded on the host, so the closure records in one submit.
    // A vertex or edge seed's vertex level decides after it whether the edit has any source.
    ClosureSeed faces, edges, retained;
    std::optional<MeshClosure> around;
    const bool listed_cuts = task.Op==MeshTopologyOp::Subdivide && (task.Flags & (TopologyFlagListSelects | TopologyFlagListCuts));
    const bool listed_vertices = (task.Op==MeshTopologyOp::ConnectVertices || task.Op==MeshTopologyOp::DeleteVertices) && (task.Flags & TopologyFlagListSelects);
    const bool listed = (listed_vertices || listed_cuts) && !task.List.empty() && task.List.front();
    if (spatial) {
        const SpatialFaceWork spatial_faces{r, chain, task};
        if (!spatial_faces.Count) return std::nullopt;
        faces = FaceSeed(r, StoreId, chain.Scratch, spatial_faces.Faces);
    } else if (lines && policy->Seed==Element::Edge) {
        edges=EncodeSelectionSeed(r,chain,StoreId,Element::Edge,select_all);
    } else if (policy->Seed!=Element::Face) {
        ClosureSeed seed;
        if (task.Op==MeshTopologyOp::AddFaces) {
            seed=ListSeed(r,chain,StoreId,Element::Vertex,face_list->ExistingVertices);
        } else if (listed_cuts) {
            const auto stride=(task.Flags & TopologyFlagListCuts) ? 2u : 1u;
            if ((task.Flags & (TopologyFlagListSelects | TopologyFlagListCuts)) ==
                    (TopologyFlagListSelects | TopologyFlagListCuts) ||
                task.List.empty() || 1ull+uint64_t(task.List.front())*stride!=task.List.size())
                throw std::invalid_argument("Subdivide has an invalid edge list.");
            std::vector<uint32_t> handles;
            handles.reserve(task.List.front());
            for (uint32_t i=0u;i<task.List.front();++i) handles.push_back(task.List[1u+i*stride]);
            seed=EncodeEdgeVertices(r,chain,StoreId,ListSeed(r,chain,StoreId,Element::Edge,handles));
        } else if (listed_vertices) {
            if (task.List.empty() || task.List.front()!=task.List.size()-1u) throw std::invalid_argument("Topology vertex selection list is invalid.");
            seed=ListSeed(r,chain,StoreId,Element::Vertex,std::span<const uint32_t>{task.List}.subspan(1u));
        } else if (policy->Seed==Element::Vertex) seed=EncodeSelectionSeed(r,chain,StoreId,Element::Vertex,select_all);
        else seed=EncodeEdgeVertices(r,chain,StoreId,EncodeSelectionSeed(r,chain,StoreId,Element::Edge,select_all));
        if (task.Op==MeshTopologyOp::MergeCollapse) {
            collapse_count=seed.Count;
            if (collapse_count && !select_all) {
                collapse_vertices=std::make_unique<mtl::Buffer>(chain.Buffers,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace);
                meshes.GatherSelectedElements(r,StoreId,Element::Vertex,*collapse_vertices);
            }
        }
        if (!seed.Count) {
            if (listed) throw std::invalid_argument("Topology element list contains no live source elements.");
            return std::nullopt;
        }
        around=EncodeVertexClosure(r,chain,StoreId,seed);
        if (lines) edges=AroundVertices(r,StoreId,Element::Edge,around->Elements[3],seed);
        else faces=AroundVertices(r,StoreId,Element::Face,around->Elements[2],seed);
        if (RetainsSeedVertices(task.Op)) retained=std::move(seed);
    } else faces=EncodeSelectionSeed(r,chain,StoreId,Element::Face,select_all);
    // A line vertex reads whether it keeps a line from its fan, so a line core needs no widening.
    auto core=lines ? EncodeEdgeClosure(r,chain,StoreId,edges,retained) : EncodeFaceClosure(r,chain,StoreId,faces,retained);
    if (!lines && (task.Op==MeshTopologyOp::DeleteFaces || task.Op==MeshTopologyOp::DeleteEdges || task.Op==MeshTopologyOp::DissolveFaces ||
         task.Op==MeshTopologyOp::DissolveVertices || task.Op==MeshTopologyOp::DissolveLimited) && !select_all) {
        // A vertex can be retired only after every face incident to the
        // selected face core has marked whether it still keeps that vertex.
        const auto incident=EncodeVertexClosure(r,chain,StoreId,core.Seed(Element::Vertex));
        core=EncodeFaceClosure(r,chain,StoreId,AroundVertices(r,StoreId,Element::Face,incident.Elements[2],faces));
    }
    // An in-place edit's neighborhood is every fan around the core's vertices.
    MeshClosure neighborhood;
    if (!fresh) {
        neighborhood = EncodeVertexClosure(r, chain, StoreId, core.Seed(Element::Vertex));
        neighborhood.EncodeIncidence(r, chain, StoreId, Element::Face);
    }
    return Closures{.FaceList=std::move(face_list), .Around=std::move(around), .Core=core, .Neighborhood=neighborhood,
        .Listed=listed, .CollapseCount=collapse_count, .CollapseVertices=std::move(collapse_vertices)};
}

void MeshTopologyEdit::RecordCounts(state::Scene &r, const MeshTopologyTask &task, Closures &closures) {
    auto &chain = Chain;
    auto &meshes = r.Context.get<MeshStore>();
    const bool fresh = task.Op == MeshTopologyOp::KeepSelectedFaces;
    const auto identity = fresh ? TopologyIdentityPolicy::Fresh : TopologyIdentityPolicy::Preserve;
    auto &[face_list, around, core, neighborhood, listed, collapse_count, collapse_vertices] = closures;
    if (around) {
        around->Finish(chain);
        if (!around->Counts[0]) {
            if (listed) throw std::invalid_argument("Topology element list contains no live source elements.");
            return;
        }
        if (!around->Counts[1] && !RetainsSeedVertices(task.Op)) return;
    }
    core.Finish(chain);
    if (!fresh) neighborhood.Finish(chain);
    if (task.Op==MeshTopologyOp::RotateEdges &&
        ((task.Flags & TopologyFlagListSelects)==0u || task.List.empty() || task.List.front()!=task.List.size()-1u))
        throw std::invalid_argument("Rotate Edges has an invalid vertex selection list.");
    Range list_range{};
    if ((task.Op==MeshTopologyOp::ConnectVertices || task.Op==MeshTopologyOp::DeleteVertices || task.Op==MeshTopologyOp::RotateEdges) &&
        (task.Flags & TopologyFlagListSelects)) {
        list_range=RemapTopologyElementList(task.List,chain.Scratch,core.Elements[0]);
    } else if (task.Op==MeshTopologyOp::Subdivide && (task.Flags & (TopologyFlagListSelects | TopologyFlagListCuts))) {
        list_range=RemapTopologyElementList(task.List,chain.Scratch,core.Elements[3],(task.Flags & TopologyFlagListCuts) ? 2u : 1u);
    } else if (task.Op==MeshTopologyOp::AddFaces) {
        list_range=RemapTopologyFaceList(task.List,face_list->Offsets,task.AppendedBase,core.Counts[0],chain.Scratch,core.Elements[0]);
    }
    const MeshStore::TopologyCounts source_counts{core.Counts[0], core.Counts[1], core.Counts[2]};
    if (!source_counts.Halfedges && !RetainsSeedVertices(task.Op)) return;
    ElementWork primitive_work{};
    std::vector<uint32_t> output_materials;
    if (fresh) {
        const auto palette = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(StoreId).PrimitiveMaterials);
        std::vector<uint32_t> primitives;
        primitives.reserve(core.Counts[2]);
        ForEachWorkElement(chain.Scratch, core.Elements[2], [&](uint32_t face) {
            const auto primitive = meshes.Arenas().FacePrimitives.Get(face);
            if (primitive >= palette.size()) throw std::out_of_range("Selected face has no primitive material.");
            primitives.push_back(primitive);
        });
        std::ranges::sort(primitives);
        primitives.erase(std::unique(primitives.begin(), primitives.end()), primitives.end());
        primitive_work = SeedElementWorkHandles(chain.Scratch, uint32_t(palette.size()), primitives);
        for (const auto primitive : primitives) output_materials.push_back(palette[primitive]);
    }
    const auto bounds = TopologyOutputBounds(task,source_counts);
    auto initial_job = TopologyJob(meshes,task,core,list_range,collapse_count,collapse_vertices.get());
    const auto scratch_words = LayoutTopologyScratch(initial_job,source_counts,bounds,Batch::ArgumentWords);
    Plan=std::make_unique<Prepared>(chain.Buffers,bounds,std::move(collapse_vertices),identity,std::move(output_materials),scratch_words);
    auto &plan=*Plan;
    plan.Core = core;
    if (!fresh) {
        plan.Neighborhood = neighborhood;
        ForEachWorkBlock(chain.Scratch,neighborhood.Elements[1],[&](uint32_t block,auto) {
            const auto payload=meshes.Arenas().NormalSectors.PayloadBlock(block);
            if (payload) OldNormalPayloadBlocks.push_back(payload-1u);
        });
        std::ranges::sort(OldNormalPayloadBlocks);
        OldNormalPayloadBlocks.erase(std::unique(OldNormalPayloadBlocks.begin(),OldNormalPayloadBlocks.end()),OldNormalPayloadBlocks.end());
        ChangedTriangles = EncodeFaceTriangles(r, chain, StoreId, neighborhood.Seed(Element::Face));
        plan.SourceView.emplace(r, StoreId, neighborhood, chain.Scratch);
    }
    auto &batch = plan.Jobs;
    {
        const profile::CpuScope stage{"TopologyJobSetup"};
        batch.Begin();
        batch.AllocateScratch(scratch_words);
        AddJob(batch, initial_job, task);
    }
    auto &job = batch.Jobs[0];
    job.PrimitiveWork = primitive_work;
    if (task.Op==MeshTopologyOp::MergeAtTarget) {
        // The local job indexes the selected face closure in canonical handle order.
        const auto ordinal=ElementWorkRank(chain.Scratch.Get(WorkStorageRange(core.Elements[0])),core.Elements[0],task.TargetVertex);
        if (ordinal==InvalidOffset) throw std::invalid_argument("Merge target is outside the local source closure.");
        job.TargetVertex=ordinal;
    }
    const auto &source_view = plan.SourceView;
    job.SrcConnectivity = fresh ? meshes.GetConnectivityRef(SourceId) : source_view->Connectivity;
    job.SrcVertexBits = fresh ? SlotOffset{meshes.GetSelectionSlot(Element::Vertex),0u} : source_view->Selection[0];
    job.SrcEdgeBits = fresh ? SlotOffset{meshes.GetSelectionSlot(Element::Edge),0u} : source_view->Selection[1];
    job.SrcFaceBits = fresh ? SlotOffset{meshes.GetSelectionSlot(Element::Face),0u} : source_view->Selection[2];
    auto &pc = plan.Constants;
    pc = TopologyPushConstants(meshes);
    if (list_range.Count) pc.ListSlot=chain.Scratch.Buffer.Slot;
    if (!fresh) pc.Source = source_view->Arenas;
    // A line dissolve joins no faces, so it takes no label rounds.
    PrepareTopology(r, chain, batch, pc, TopologyIterates(task.Op) && !(TopologyIsDissolve(task.Op) && !source_counts.Faces));
    Output = std::make_unique<TopologyOutputHandles>(r, chain, job, bounds, batch.Scratch.Slot, pc.Source, identity);
}

void MeshTopologyEdit::ReadCounts(bool capture_inset_basis) {
    auto &plan = *Plan;
    auto &job = plan.Jobs.Jobs[0];
    const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
    const auto counts = plan.Counts = ReadOutputCounts(plan.Jobs, job, plan.Bounds);
    job.DstVertexCount = counts.Vertices;
    job.DstHalfedgeCount = counts.Halfedges;
    job.DstFaceCount = counts.Faces;
    Output->Finish(Chain);
    // A region without boundary edges has no inset effect; preserve its storage.
    if (Op == MeshTopologyOp::InsetRegion && Output->NewCounts == std::array{0u, 0u}) {
        Output.reset();
        return;
    }
    if (!fresh) ChangedTriangles.Finish(Chain);
    if (capture_inset_basis && (Op == MeshTopologyOp::InsetRegion || Op == MeshTopologyOp::InsetIndividual)) {
        InsetBasis.SetUsedSize(uint64_t(counts.Vertices) * sizeof(InsetVertexBasis));
        job.DstInsetBasisSlot = InsetBasis.Slot;
    }
    const auto retirement = TopologyPolicy(Op)->Retirement;
    if ((Output->RetiredCounts[0] && !(retirement & RetireVertices)) ||
        (Output->RetiredCounts[1] && !(retirement & RetireFaces)))
        throw std::logic_error("Local topology retired unsupported source identities: vertices="+
            std::to_string(Output->RetiredCounts[0])+", faces="+std::to_string(Output->RetiredCounts[1]));
}

std::vector<MeshTopologyEdit> MeshTopologyEdit::Construct(state::Scene &r, mtl::ComputeChain &chain, std::span<const MeshTopologyTask> tasks, bool capture_inset_basis) {
    const profile::CpuScope scope{"TopologyEditConstruct"};
    std::vector<MeshTopologyEdit> edits;
    std::vector<std::optional<Closures>> closures;
    edits.reserve(tasks.size());
    for (const auto &task : tasks) closures.push_back(edits.emplace_back(MeshTopologyEdit{chain, task}).RecordClosures(r, task));
    chain.Submit();
    for (uint32_t i = 0u; i < edits.size(); ++i) if (closures[i]) edits[i].RecordCounts(r, tasks[i], *closures[i]);
    // Identity planning reads the scanned counts on the GPU, so the host reads both after one submit.
    chain.Submit();
    for (auto &edit : edits) if (edit.Output) edit.ReadCounts(capture_inset_basis);
    return edits;
}

void MeshTopologyEdit::PublishAll(state::Scene &r, std::span<MeshTopologyEdit> all) {
    const profile::CpuScope scope{"TopologyEditPublish"};
    std::vector<MeshTopologyEdit *> edits;
    for (auto &edit : all) {
        if (!edit.Output) continue;
        if (edit.Published) throw std::logic_error("Topology edit was already published.");
        edits.push_back(&edit);
    }
    if (edits.empty()) return;
    auto &chain = edits.front()->Chain;
    auto &meshes = r.Context.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    using D = MeshStore::ElementDomain;

    // Every edit's output identities are inserted before any emission.
    for (auto *edit : edits) {
        const profile::CpuScope stage{"TopologyElementInsert"};
        auto &plan = *edit->Plan;
        if (plan.Identity == TopologyIdentityPolicy::Fresh) edit->StoreId = meshes.BeginTopologyOutput(edit->SourceId,plan.OutputMaterials);
        const auto counts = plan.Counts;
        auto &job = plan.Jobs.Jobs[0];
        // A face derives two triangles fewer than its corners, and a line corner derives none.
        edit->AddedTriangleCount = counts.Faces ? counts.Halfedges - 2u*counts.Faces : 0u;
        edit->SourceTriangles.SetUsedSize(uint64_t(edit->AddedTriangleCount)*sizeof(uint32_t));
        job.DstTriangleSourceSlot = edit->SourceTriangles.Slot;
        edit->NewVertices = meshes.InsertElements(edit->StoreId,D::Vertex,edit->Output->NewCounts[0],&edit->NewVertexList);
        plan.NewFaces = meshes.InsertElements(edit->StoreId,D::Face,edit->Output->NewCounts[1],&plan.FaceList);
        job.DstCornerOffset = meshes.InsertElements(edit->StoreId,D::Halfedge,counts.Halfedges,nullptr).First;
        job.DstTriangleOffset = edit->FirstTriangle = meshes.InsertElements(edit->StoreId,D::Triangle,edit->AddedTriangleCount,nullptr).First;
        edit->Output->Assign(r,chain,{edit->NewVertices,plan.NewFaces});
    }
    // Emission, incidence repair, edges, fans, edge attributes and the corner class counts share one submit.
    // New edges and fan items are inserted for bounds the host knows, then trimmed to the counts that submit reports.
    std::vector<Batch::Upload> emissions;
    std::vector<MeshConnectivityJob> fan_jobs;
    std::vector<uint32_t> fan_vertex_blocks;
    for (auto *edit : edits) {
        const profile::CpuScope stage{"TopologyEmit"};
        auto &plan = *edit->Plan;
        const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
        auto &batch = plan.Jobs;
        auto &job = batch.Jobs[0];
        const auto counts = plan.Counts;
        // Retained outputs keep core handles, and new outputs take the inserted ones.
        const auto inserted_vertices = InsertedBlocks(edit->NewVertices, edit->NewVertexList), inserted_faces = InsertedBlocks(plan.NewFaces, plan.FaceList);
        const auto vertex_blocks = fresh ? inserted_vertices : Union(WorkBlocks(chain.Scratch, plan.Core.Elements[0], 0u), inserted_vertices);
        const auto face_blocks = fresh ? inserted_faces : Union(WorkBlocks(chain.Scratch, plan.Core.Elements[2], 0u), inserted_faces);
        job.DstVertexHandles = {edit->Output->Vertices.Slot, 0u};
        job.DstFaceHandles = {edit->Output->Faces.Slot, 0u};
        job.DstConnectivity = meshes.GetConnectivityRef(edit->StoreId);
        job.DstVertexBits = {meshes.GetSelectionSlot(Element::Vertex), 0u};
        job.DstEdgeBits = {meshes.GetSelectionSlot(Element::Edge), 0u};
        job.DstFaceBits = {meshes.GetSelectionSlot(Element::Face), 0u};
        batch.SetDomainTiles(DstVertices, std::array{TileCount(counts.Vertices, TileElements)});
        batch.SetDomainTiles(DstHalfedges, std::array{TileCount(counts.Halfedges, TileElements)});
        batch.SetDomainTiles(DstFaces, std::array{TileCount(counts.Faces, TileElements)});
        CaptureTopologyEmitWrites(r, job, vertex_blocks, face_blocks);
        emissions.push_back(batch.Encode(chain, pipelines, plan.Constants, TopologyEmissionPasses(job.CollapseCount != 0u)));
        const auto &before = plan.Neighborhood;
        const std::array emitted{ElementHandleRange{.Handles = job.DstVertexHandles, .Count = counts.Vertices},
                                 ElementHandleRange{.First = job.DstCornerOffset, .Count = counts.Halfedges},
                                 ElementHandleRange{.Handles = job.DstFaceHandles, .Count = counts.Faces}};
        edit->Repair = std::make_unique<ConnectivityEditWork>(r, chain, before, std::array{plan.Core.Elements[1], plan.Core.Elements[2]}, emitted);
        // Emission replaces the core's loops inside the neighborhood, so the repaired counts are known before the pass runs.
        const std::array<uint32_t,3> repaired = fresh ? std::array{counts.Vertices, counts.Halfedges, counts.Faces} : std::array{
            before.Counts[0] + edit->Output->NewCounts[0], before.Counts[1] - plan.Core.Counts[1] + counts.Halfedges,
            before.Counts[2] - plan.Core.Counts[2] + counts.Faces};
        edit->RetiredEdges = AllocateElementWork(chain.Scratch, a.EdgeHalfedges.Capacity(), before.Counts[3]);
        // Every new edge holds an emitted halfedge, and a line holds two.
        plan.NewEdges = meshes.InsertElements(edit->StoreId, D::Edge, counts.Faces ? counts.Halfedges : counts.Halfedges / 2u, &plan.EdgeList);
        MeshConnectivityJob connectivity{
            .Corners = {a.FaceCorners.Buffer.Slot, 0u}, .Connectivity = job.DstConnectivity,
            .Vertices = edit->Repair->Elements[0], .Halfedges = edit->Repair->Elements[1], .Faces = edit->Repair->Elements[2],
            .EdgeHandles = plan.NewEdges,
            .SourceConnectivity = job.SrcConnectivity, .SourceCornerSlot = plan.Constants.Source.CornerSlot,
            .SourceEdges = before.Elements[3], .SourceEdgeCount = before.Counts[3],
            .VertexCount = repaired[0], .HalfedgeCount = repaired[1], .FaceCount = repaired[2], .FaceStarts = 1u,
            .RetiredEdgeWork = edit->RetiredEdges,
        };
        auto &rebuilt = plan.Connectivity.emplace(chain.Buffers, LayoutConnectivityScratch(connectivity), 1u);
        rebuilt.Begin();
        AddConnectivityJob(rebuilt, connectivity);
        const auto run_blocks = RunBlocks(job.DstCornerOffset, counts.Halfedges);
        // Retained edges are neighborhood edges.
        auto edge_blocks = Union(WorkBlocks(chain.Scratch, before.Elements[3], 0u), InsertedBlocks(plan.NewEdges, plan.EdgeList));
        auto repaired_face_blocks = Union(WorkBlocks(chain.Scratch, before.Elements[2], 0u), face_blocks);
        CaptureConnectivityPrepareWrites(r, rebuilt.Jobs[0], vertex_blocks, Union(WorkBlocks(chain.Scratch, before.Elements[1], 0u), run_blocks),
            repaired_face_blocks);
        CaptureTopologyEdgeWrites(r, edge_blocks);
        rebuilt.Encode(chain, pipelines, TiledJobPushConstants{}, ConnectivityPasses);
        fan_jobs.push_back(rebuilt.Jobs[0]);
        fan_vertex_blocks.insert(fan_vertex_blocks.end(), vertex_blocks.begin(), vertex_blocks.end());
        edit->RepairedBlocks = {std::move(edge_blocks), std::move(repaired_face_blocks)};
    }
    std::ranges::sort(fan_vertex_blocks);
    fan_vertex_blocks.erase(std::unique(fan_vertex_blocks.begin(), fan_vertex_blocks.end()), fan_vertex_blocks.end());
    EncodeVertexFans(r, chain, fan_jobs, fan_vertex_blocks);
    for (uint32_t i = 0u; auto *edit : edits) {
        auto &plan = *edit->Plan;
        const auto &connected = plan.Connectivity->Jobs[0];
        plan.Jobs.Record(chain, pipelines, emissions[i++], plan.Constants, std::array{TiledPass{MeshPass::TopologyEdgeAttributes, DstHalfedges}});
        EncodeSortElementWork(r, chain, std::span{&edit->RetiredEdges, 1u});
        // The rebuilt fans hold every corner at the repaired vertices.
        plan.Classes = meshes.EncodeCornerClassification(r, chain, edit->StoreId, edit->Repair->Elements[0], connected.VertexCount, connected.HalfedgeCount);
    }
    {
        const profile::CpuScope stage{"TopologyEmitSubmit"};
        chain.Submit();
    }

    // Corner classes, base normals and authored normals complete the canonical edit in the chain's next submit.
    for (auto *edit : edits) {
        const profile::CpuScope stage{"TopologyNormals"};
        auto &plan = *edit->Plan;
        const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
        auto &job = plan.Jobs.Jobs[0];
        const auto &connected = plan.Connectivity->Jobs[0];
        edit->Repair->Finish(chain);
        if (edit->Repair->Counts != std::array{connected.VertexCount, connected.HalfedgeCount, connected.FaceCount}) {
            throw std::logic_error("Connectivity repair disagrees with its emitted closure.");
        }
        meshes.TrimInsertedElements(edit->StoreId, D::Edge, plan.NewEdges, plan.EdgeList, plan.Connectivity->ScratchSpan()[connected.StateOffset]);
        CheckElementWork(chain.Scratch, edit->RetiredEdges);
        meshes.PlanCornerClassification(r, chain, plan.Classes);
        EncodeDeriveMeshNormals(r, chain, edit->StoreId, chain.Scratch, edit->Repair->Elements[0], edit->Repair->Counts[0],
            edit->Repair->Elements[2], edit->Repair->Counts[2]);
        if (job.CornerAttributes & MeshAttributeBit_Normal) {
            if (!fresh) {
                job.RetainedNormalCorners = plan.Neighborhood.Elements[1];
                job.RetainedNormalCornerCount = plan.Neighborhood.Counts[1];
            }
            plan.Jobs.SetDomainTiles(DstHalfedges, std::array{TileCount(std::max(job.DstHalfedgeCount, job.RetainedNormalCornerCount), TileElements)});
            CaptureTopologyNormalWrites(r, job, chain.Scratch);
            plan.Jobs.Encode(chain, pipelines, plan.Constants, std::array{TiledPass{MeshPass::TopologyCustomNormals, DstHalfedges}});
        }
        edit->AddedTriangles = AllocateElementWork(chain.Scratch, a.Triangles.Capacity(), 1u);
        SeedElementWorkRanges(chain.Scratch, edit->AddedTriangles, std::array{Range{job.DstTriangleOffset, edit->AddedTriangleCount}}, 0u, false);
        edit->Published = true;
    }
}

void MeshTopologyEdit::FinishAll(state::Scene &r, std::span<MeshTopologyEdit *const> edits) {
    const profile::CpuScope scope{"FinishTopologyEdits"};
    auto &meshes = r.Context.get<MeshStore>();
    std::vector<MeshStore::SelectionUpdate> updates;
    for (auto *edit : edits) {
        if (edit->Finished) throw std::logic_error("Topology edit was already finished.");
        edit->Finished = true;
        if (edit->Published && edit->Plan) {
            edit->Chain.Submit();
            meshes.FinishCornerClassification(edit->Chain, edit->Plan->Classes);
            edit->Plan.reset();
        }
        if (!edit->Output) continue;
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = edit->StoreId});
        if (edit->StoreId != edit->SourceId) {
            // A copied output emitted its masks, and every block's aggregate is new.
            for (uint32_t d = 0u; const auto domain : {MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face}) {
                const auto blocks = meshes.GetBlockList(edit->StoreId, domain).Blocks;
                update.Blocks[d++].assign(blocks.begin(), blocks.end());
            }
            continue;
        }
        // Publication changed masks, positions and incidence around the repaired vertices.
        // Their blocks, the blocks around them and the blocks that lost elements refresh their aggregates.
        const auto &storage = edit->Chain.Scratch;
        const auto &output = *edit->Output;
        update.Blocks[1] = meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Edge, storage, edit->RetiredEdges);
        meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Halfedge, storage, output.Replaced[0]);
        meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Triangle, storage, output.Replaced[1]);
        if (output.RetiredCounts[1]) update.Blocks[2] = meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Face, storage, output.Retired[1]);
        if (output.RetiredCounts[0]) update.Blocks[0] = meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Vertex, storage, output.Retired[0]);
        ForEachWorkBlock(storage, edit->Repair->Elements[0], [&](uint32_t block, auto) { update.Blocks[0].push_back(block); });
        update.Blocks[1].insert(update.Blocks[1].end(), edit->RepairedBlocks[0].begin(), edit->RepairedBlocks[0].end());
        update.Blocks[2].insert(update.Blocks[2].end(), edit->RepairedBlocks[1].begin(), edit->RepairedBlocks[1].end());
    }
    meshes.UpdateSelection(r, updates);
}
