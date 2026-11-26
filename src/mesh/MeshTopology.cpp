#include "mesh/MeshTopology.h"
#include "SortUnique.h"
#include "mesh/MeshTopologyLayout.h"

#include "Profile.h"
#include "gpu/ConnectivityEditJob.h"
#include "gpu/ElementRepairJob.h"
#include "gpu/LevelSetCountJob.h"
#include "gpu/MeshClosureJob.h"
#include "gpu/MeshTopologyPushConstants.h"
#include "gpu/MeshletBoundsJob.h"
#include "gpu/ShortEdgesJob.h"
#include "gpu/TiledJobPushConstants.h"
#include "gpu/TopologyIdentityJob.h"
#include "mesh/BooleanWork.h"
#include "mesh/ConnectivityBatch.h"
#include "mesh/ConnectivityWritePages.h"
#include "mesh/CornerClassBatch.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/Neighborhood.h"
#include "mesh/NormalDeriveGpu.h"
#include "mesh/ScratchPasses.h"
#include "mesh/SpatialFaceWork.h"
#include "mesh/TopologyReadView.h"
#include "mesh/TopologyWritePages.h"
#include "mesh/VertexFanBuild.h"
#include "state/Registry.h"

MeshTopologyPushConstants TopologyPushConstants(const MeshStore &meshes) {
    const auto &a = meshes.Arenas();
    const MeshTopologyArenas arenas{
        .VertexSlot = a.Vertices.Buffer.Slot,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .FaceTriangleStartSlot = a.FaceTriangles.Buffer.Slot,
        .TriangleSlot = a.Triangles.Buffer.Slot,
        .EdgeSharpnessSlot = a.EdgeSharpness.Buffer.Slot,
        .FaceSharpnessSlot = a.FaceSharpness.Buffer.Slot,
        .BaseVertexNormalSlot = a.BaseVertexNormals.Buffer.Slot,
        .BaseFaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
        .VertexHiddenSlot = a.VertexHidden.Buffer.Slot,
        .EdgeHiddenSlot = a.EdgeHidden.Buffer.Slot,
        .FaceHiddenSlot = a.FaceHidden.Buffer.Slot,
    };
    return {.Source = arenas, .Destination = arenas};
}

void PublishStoreSelection(const MeshStore &meshes, MeshTopologyTask &task) {
    const auto &a = meshes.Arenas();
    task.SelectionOutput = GeometryMasks{{&a.VertexSelection}, {&a.EdgeSelection}, {&a.FaceSelection}};
    if (task.Op == MeshTopologyOp::ExtrudeVertices) task.StoreSelectionSource = Element::Vertex;
}

struct PrimitiveListReferences {
    std::vector<uint32_t> ExistingVertices;
    std::vector<uint32_t> Offsets;
    std::vector<uint32_t> EdgeCorners;
    std::vector<uint32_t> Wires; // The edges of the corners in EdgeCorners that have no face
    uint32_t PrimitiveCount{}, FaceCount{};
};

namespace {
enum Domain : uint32_t {
    TableEntries,
    SrcVertices,
    SrcHalfedges,
    SrcFaces,
    RetainedEdgeVertices,
    ListEntries,
    CountBlocks,
    Once,
    DstVertices,
    DstHalfedges,
    DstFaces,
    Collapse0,
    Collapse1,
    Collapse2,
    Collapse3,
    CollapseDigits,
    DomainCount
};
using Batch = TiledJobBatch<MeshTopologyJob, DomainCount>;

constexpr std::array PreparePasses{
    TiledPass{MeshPass::TopologyZero, SrcVertices},
    TiledPass{MeshPass::TopologyListFill, ListEntries, 1},
    TiledPass{MeshPass::TopologySimplifyTargets, SrcVertices},
    TiledPass{MeshPass::TopologyMarkHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces},
    TiledPass{MeshPass::TopologyMarkRetainedEdges, RetainedEdgeVertices},
    TiledPass{MeshPass::TopologyFinalizeVertices, SrcVertices},
    TiledPass{MeshPass::TopologyMergeTable, TableEntries},
    TiledPass{MeshPass::TopologyMergeInsert, SrcVertices},
    TiledPass{MeshPass::TopologyMergeQuery, SrcVertices},
    TiledPass{MeshPass::TopologyListFill, ListEntries},
};
// The converge pass after each round zeroes the indirect arguments once a round changes nothing, so the rounds after it dispatch no threadgroups.
constexpr std::array LabelPasses{
    TiledPass{MeshPass::TopologyLink, SrcHalfedges, 0, true},
    TiledPass{MeshPass::TopologyJump, SrcFaces, 0, true},
    TiledPass{MeshPass::TopologyJoinBest, SrcFaces, 0, true},
    TiledPass{MeshPass::TopologyJoinMatch, SrcFaces, 0, true},
    TiledPass{MeshPass::TopologyJump, SrcVertices, 1, true},
};
// Once vertex targets are final, newly coincident polygons drop before surviving faces take precedence over loose edges.
constexpr std::array WeldPasses{
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces, 1},
    TiledPass{MeshPass::TopologyMergeTable, TableEntries},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces, 2},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces, 3},
};
constexpr std::array CountPasses{
    TiledPass{MeshPass::TopologyDissolveRegions, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveWalk, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveRevert, SrcHalfedges},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces, 4},
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
    TiledPass{MeshPass::TopologyGatherVertices, DstVertices},
    TiledPass{MeshPass::TopologyFaceTables, DstFaces},
    TiledPass{MeshPass::TopologyGatherCorners, DstHalfedges},
};
// A Bezier subdivide builds its tangents before emission and moves the gathered new vertices onto the patches.
constexpr std::array BezierOutputPasses{
    TiledPass{MeshPass::TopologyZeroVertices, DstVertices},
    TiledPass{MeshPass::TopologyTangents, SrcVertices},
    TiledPass{MeshPass::TopologyTangentsDistribute, SrcHalfedges},
    TiledPass{MeshPass::TopologyScatterVertices, SrcVertices},
    TiledPass{MeshPass::TopologyScatterHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyScatterFaces, SrcFaces},
    TiledPass{MeshPass::TopologyGatherVertices, DstVertices},
    TiledPass{MeshPass::TopologyBezierPositions, DstVertices},
    TiledPass{MeshPass::TopologyFaceTables, DstFaces},
    TiledPass{MeshPass::TopologyGatherCorners, DstHalfedges},
};

// Scans the boolean vertex, intersection and triangle counts once the intersection passes have sized them.
constexpr std::array BooleanCountPasses{
    TiledPass{MeshPass::TopologyBooleanCount, SrcVertices},
    TiledPass{MeshPass::TopologyScanBlockSum, CountBlocks},
    TiledPass{MeshPass::TopologyScanBlockPrefix, PerJob},
    TiledPass{MeshPass::TopologyScanOffsets, CountBlocks},
};
constexpr std::array BooleanOutputPasses{
    TiledPass{MeshPass::TopologyZeroVertices, DstVertices},
    TiledPass{MeshPass::TopologyBooleanScatter, SrcVertices},
    TiledPass{MeshPass::TopologyGatherVertices, DstVertices},
    TiledPass{MeshPass::TopologyFaceTables, DstFaces},
    TiledPass{MeshPass::TopologyGatherCorners, DstHalfedges},
};

constexpr auto CollapseOutputPasses = [] {
    std::array<TiledPass, 1u + 8u * 3u + 4u + 3u + OutputPasses.size() + 1u> passes{};
    size_t next = 0u;
    passes[next++] = {MeshPass::TopologyCollapseKeys, Collapse0};
    for (uint32_t shift = 0u; shift < 32u; shift += 4u) {
        passes[next++] = {MeshPass::TopologyCollapseHistogram, Collapse0, shift};
        passes[next++] = {MeshPass::TopologyCollapsePrefix, CollapseDigits, shift};
        passes[next++] = {MeshPass::TopologyCollapseScatter, Collapse0, shift};
    }
    for (uint32_t level = 0u; level < 4u; ++level) {
        passes[next++] = {MeshPass::TopologyCollapseReduce, Collapse0 + level, level};
    }
    for (uint32_t level = 3u; level--;) {
        passes[next++] = {MeshPass::TopologyCollapseCarry, Collapse0 + level, level};
    }
    for (const auto pass : OutputPasses) passes[next++] = pass;
    passes[next] = {MeshPass::TopologyCollapseCenters, Collapse0};
    return passes;
}();

std::span<const TiledPass> TopologyEmissionPasses(const MeshTopologyJob &job) {
    if (job.Op == MeshTopologyOp::Boolean) return BooleanOutputPasses;
    if (job.Op == MeshTopologyOp::Subdivide && (job.Flags & TopologyFlagBezier)) return BezierOutputPasses;
    return job.CollapseCount ? std::span<const TiledPass>{CollapseOutputPasses} : std::span<const TiledPass>{OutputPasses};
}

PrimitiveListReferences ParsePrimitiveListReferences(const MeshTopologyTask &task) {
    // TopologyOutputBounds validates polygon lengths and the terminal cursor.
    (void)TopologyOutputBounds(task, {});
    const auto &list = task.List;
    const auto new_vertices = list[0];
    const uint32_t attribute_source = 3u;
    if (list[attribute_source] >= task.AppendedBase) throw std::invalid_argument("Topology face list attribute source is not an existing vertex handle.");
    uint64_t cursor = TopologyPrimitiveHeaderWords + uint64_t(list[1]);
    const auto faces = list[cursor++];
    PrimitiveListReferences result;
    result.PrimitiveCount = faces;
    result.Offsets.push_back(uint32_t(attribute_source));
    result.ExistingVertices.push_back(list[attribute_source]);
    for (uint32_t i = 0u; i < list[1]; ++i) {
        const uint32_t offset = TopologyPrimitiveHeaderWords + i;
        if (list[offset] >= task.AppendedBase) throw std::invalid_argument("Grid boundary needs existing vertices.");
        result.Offsets.push_back(offset);
        result.ExistingVertices.push_back(list[offset]);
    }
    for (uint32_t f = 0u; f < faces; ++f) {
        const auto count = list[cursor++];
        result.FaceCount += count > 2u;
        for (uint32_t i = 0u; i < count; ++i) {
            const auto handle = list[cursor + 2u * i];
            if (handle >= task.AppendedBase && uint64_t(handle) - task.AppendedBase >= new_vertices) {
                throw std::invalid_argument("Topology face list names a vertex outside its source and appended range.");
            }
            result.Offsets.push_back(uint32_t(cursor + 2u * i));
            if (handle < task.AppendedBase) result.ExistingVertices.push_back(handle);
            const auto edge = list[cursor + 2u * i + 1u];
            if (edge != InvalidOffset) result.EdgeCorners.push_back(edge);
        }
        cursor += 2u * count;
    }
    return result;
}

MeshTopologyJob TopologyJob(const MeshStore &meshes, const MeshTopologyTask &task, const MeshClosure &source, Range list, uint32_t collapse_count, SlotOffset collapse_vertices) {
    const auto &src = meshes.Get(task.SourceId);
    return {
        .Op = task.Op,
        .Flags = task.Flags,
        .Steps = task.Steps,
        .Param0 = task.Param0,
        .Param1 = task.Param1,
        .Param0Attribute = task.Param0Attribute,
        .Param1Attribute = task.Param1Attribute,
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
        .LatticeSize = task.LatticeSize,
        .SrcVertexCount = source.Counts[0],
        .SrcHalfedgeCount = source.Counts[1],
        .SrcFaceCount = source.Counts[2],
        .SrcEdgeCount = source.Counts[3],
        .SrcVertexWork = source.Elements[0],
        .SrcHalfedgeWork = source.Elements[1],
        .SrcFaceWork = source.Elements[2],
        .SrcEdgeWork = source.Elements[3],
        .SourceLayers = src.CornerAttributes | src.VertexAttributes << 16u,
        .Simplify = task.Simplify,
        .SelectionElement = task.SelectionElement,
        .HasSkin = uint32_t(src.SkinBlocksReady),
        .HasVertexPrimitives = uint32_t(src.VertexPrimitivesReady),
        .CornerAttributes = src.CornerAttributes,
        .VertexAttributes = src.VertexAttributes,
        .ListOffset = OffsetOrInvalid(list),
        .MorphTargetCount = src.MorphBlocksReady ? src.MorphTargetCount : 0u,
        .CollapseCount = collapse_count,
        .CollapseVertices = collapse_vertices,
    };
}

void AddJob(Batch &batch, const MeshTopologyJob &job, const MeshTopologyTask &task) {
    const auto V = job.SrcVertexCount, H = job.SrcHalfedgeCount, F = job.SrcFaceCount;
    std::array<uint32_t, DomainCount> tiles{};
    if (job.Op == MeshTopologyOp::Boolean) {
        tiles[SrcVertices] = TileCount(job.CountEntries, TileElements);
        tiles[CountBlocks] = TopologyCountQuantities * job.CountBlockCount;
        tiles[Once] = batch.Jobs.empty() ? 1u : 0u;
        batch.AddJob(job, tiles);
        return;
    }
    tiles[TableEntries] = TileCount(TopologyTableWords(job.Op, {V, H, F}), TileElements);
    tiles[SrcVertices] = TileCount(V, TileElements);
    tiles[RetainedEdgeVertices] = TopologyClassifiesEdges(job.Op) ? TileCount(V, TileElements) : 0u;
    tiles[SrcHalfedges] = TileCount(H, TileElements);
    tiles[SrcFaces] = TileCount(F + 2u, TileElements);
    const auto listed = ((job.Flags & (TopologyFlagListSelects | TopologyFlagEdgeLists)) || job.Op == MeshTopologyOp::ReplaceFaces) && !task.List.empty() ? task.List.front() : 0u;
    tiles[ListEntries] = TileCount((job.Flags & TopologyFlagScreenCuts) ? job.SrcEdgeCount : listed, TileElements);
    tiles[CountBlocks] = TopologyCountQuantities * job.CountBlockCount;
    tiles[Once] = batch.Jobs.empty() ? 1u : 0u;
    for (uint32_t level = 0u, count = job.CollapseCount; count; ++level) {
        tiles[Collapse0 + level] = TileCount(count, TileElements);
        count = count > TileElements ? TileCount(count, TileElements) : 0u;
    }
    tiles[CollapseDigits] = job.CollapseCount ? 16u : 0u;
    batch.AddJob(job, tiles);
}

Batch::Upload EncodeTopologyCounts(state::Registry &r, mtl::ComputeChain &chain, Batch &batch, const MeshTopologyPushConstants &pc, uint32_t rounds) {
    const profile::CpuScope scope{"EncodeTopologyCounts"};
    if (batch.Jobs.front().Op == MeshTopologyOp::Boolean) return batch.Encode(chain, GetMeshPipelines(r), pc, BooleanCountPasses);
    std::vector<TiledPass> passes{
        {MeshPass::TopologySelection, SrcVertices, 0u},
        {MeshPass::TopologySelection, SrcHalfedges, 1u},
        {MeshPass::TopologySelection, SrcFaces, 2u}
    };
    const bool limited = std::ranges::any_of(batch.Jobs, [](const auto &job) { return job.Op == MeshTopologyOp::DissolveLimited; });
    passes.insert(passes.end(), PreparePasses.begin(), PreparePasses.end());
    for (uint32_t round = 0; round < rounds; ++round) {
        passes.insert(passes.end(), LabelPasses.begin(), LabelPasses.end());
        if (limited) passes.push_back({MeshPass::TopologyJump, SrcVertices, 2u, true});
        passes.push_back({MeshPass::TopologyConverge, Once, (uint32_t(batch.Jobs.size()) << 8) | DomainCount});
    }
    if (std::ranges::any_of(batch.Jobs, [](const auto &job) { return TopologyIsMerge(job.Op); })) {
        passes.insert(passes.end(), WeldPasses.begin(), WeldPasses.end());
    }
    if (limited) passes.push_back({MeshPass::TopologyFinalizeVertices, SrcVertices, 1u});
    passes.insert(passes.end(), CountPasses.begin(), CountPasses.end());
    return batch.Encode(chain, GetMeshPipelines(r), pc, passes);
}

struct OutputCounts {
    MeshStore::TopologyCounts Counts;
    uint32_t Triangles, ProvenancePairs;
};
OutputCounts ReadOutputCounts(const Batch &batch, const MeshTopologyJob &job, MeshStore::TopologyCounts bounds) {
    const auto scratch = batch.ScratchSpan();
    const auto total = [&](uint32_t q) { return scratch[job.CountsOffset + q * job.CountEntries + job.CountEntries - 1]; };
    const uint64_t halfedges = uint64_t(total(2)) + total(3);
    if (total(0) > bounds.Vertices || halfedges > bounds.Halfedges || total(1) > bounds.Faces || uint64_t(total(1)) * 3u > total(2) ||
        (total(3) & 1u) || (!total(1) && total(2))) {
        throw std::runtime_error("GPU topology counts exceed the reserved output bounds.");
    }
    return {{total(0), uint32_t(halfedges), total(1)}, total(2) - 2u * total(1), total(4)};
}

Range PrepareTopologyList(const MeshTopologyTask &task, const PrimitiveListReferences *primitives, const MeshClosure &core, BufferArena<uint32_t> &storage) {
    const auto op = task.Op;
    const bool vertices = op == MeshTopologyOp::DissolveVertices && (task.Flags & TopologyFlagListSelects);
    const bool edges = op == MeshTopologyOp::Subdivide && (task.Flags & TopologyFlagEdgeLists);
    if (!vertices && !edges && op != MeshTopologyOp::AddPrimitives && op != MeshTopologyOp::ReplaceFaces) return {};
    const auto range = storage.Allocate(task.List);
    const auto list = storage.GetMutable(range);
    const auto rank = [&](uint32_t domain, uint32_t handle) {
        const auto work = core.Elements[domain];
        const auto result = ElementWorkRank(storage.Get(WorkStorageRange(work)), work, handle);
        if (result == InvalidOffset) throw std::invalid_argument("Topology list names an element outside its source closure.");
        return result;
    };
    if (vertices || edges) {
        const auto stride = edges ? 2u : 1u;
        for (uint32_t i = 0u; i < list[0]; ++i) {
            auto &handle = list[1u + i * stride];
            handle = rank(edges ? 3u : 0u, handle);
            if (task.Flags & TopologyFlagListJunctions) list[2u + i * stride] = rank(0u, list[2u + i * stride]);
        }
    } else if (op == MeshTopologyOp::AddPrimitives) {
        for (const auto at : primitives->Offsets) {
            auto &handle = list[at];
            if (handle < task.AppendedBase) handle = rank(0u, handle);
            else handle = core.Counts[0] + handle - task.AppendedBase;
        }
    } else {
        for (uint32_t i = 1u; i <= list[0]; ++i) list[i] = rank(0u, list[i]);
        const auto offset = 2u + list[0], count = list[offset - 1u];
        if (count != core.Counts[2]) throw std::invalid_argument("Face replacement source faces are not live.");
    }
    return range;
}

enum TopologyRetirement : uint8_t { RetireNone,
                                    RetireVertices = 1u,
                                    RetireFaces = 2u,
                                    RetireBoth = 3u };
// Lines marks an operator with a rule for the lines of a mesh without faces.
struct LocalPolicy {
    uint8_t Retirement;
    Element Seed;
    bool Lines{};
};

// The local edit path's one allowlist, with each operator's retirement and seed domain.
std::optional<LocalPolicy> TopologyPolicy(MeshTopologyOp op) {
    switch (op) {
        case MeshTopologyOp::DeleteLoose:
        case MeshTopologyOp::DeleteVertices:
        case MeshTopologyOp::DissolveVertices:
        case MeshTopologyOp::DissolveLimited:
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse: return LocalPolicy{RetireBoth, Element::Vertex, true};
        case MeshTopologyOp::Decimate: return LocalPolicy{RetireBoth, Element::Vertex};
        case MeshTopologyOp::DissolveDegenerate: return LocalPolicy{RetireBoth, Element::Vertex};
        case MeshTopologyOp::DeleteEdges: return LocalPolicy{RetireBoth, Element::Edge, true};
        case MeshTopologyOp::DissolveEdges:
        case MeshTopologyOp::RotateEdges: return LocalPolicy{RetireBoth, Element::Edge};
        case MeshTopologyOp::DeleteFaces:
        case MeshTopologyOp::ReplaceFaces:
        case MeshTopologyOp::Wireframe:
        case MeshTopologyOp::DissolveFaces: return LocalPolicy{RetireBoth, Element::Face};
        case MeshTopologyOp::BevelVertices: return LocalPolicy{RetireVertices, Element::Vertex};
        case MeshTopologyOp::BevelEdges: return LocalPolicy{RetireVertices, Element::Edge};
        case MeshTopologyOp::DeleteOnlyEdgesFaces: return LocalPolicy{RetireFaces, Element::Edge, true};
        case MeshTopologyOp::DeleteOnlyFaces:
        case MeshTopologyOp::TrisToQuads: return LocalPolicy{RetireFaces, Element::Face};
        case MeshTopologyOp::ConnectVertices: return LocalPolicy{RetireNone, Element::Vertex};
        case MeshTopologyOp::SplitGeometry:
        case MeshTopologyOp::DuplicateGeometry:
        case MeshTopologyOp::AddPrimitives:
        case MeshTopologyOp::ExtrudeVertices: return LocalPolicy{RetireNone, Element::Vertex, true};
        case MeshTopologyOp::ExtrudeEdges:
        case MeshTopologyOp::Subdivide: return LocalPolicy{RetireNone, Element::Edge, true};
        case MeshTopologyOp::EdgeSplit: return LocalPolicy{RetireNone, Element::Edge};
        case MeshTopologyOp::ExtrudeRegion: return LocalPolicy{RetireVertices, Element::Vertex, true};
        case MeshTopologyOp::ExtrudeFacesIndividual:
        case MeshTopologyOp::Triangulate:
        case MeshTopologyOp::SplitNonplanarFaces:
        case MeshTopologyOp::SplitConcaveFaces:
        case MeshTopologyOp::Poke:
        case MeshTopologyOp::FlipNormals:
        case MeshTopologyOp::InsetRegion:
        case MeshTopologyOp::InsetIndividual:
        case MeshTopologyOp::Solidify: return LocalPolicy{RetireNone, Element::Face};
        case MeshTopologyOp::KeepSelectedFaces: return LocalPolicy{RetireNone, Element::Face, true};
        case MeshTopologyOp::Boolean: return LocalPolicy{RetireNone, Element::Face};
        case MeshTopologyOp::LevelSet: return LocalPolicy{RetireNone, Element::Vertex, true};
        default: return std::nullopt;
    }
}

bool RetainsSeedVertices(MeshTopologyOp op) {
    return TopologyIsMerge(op) || op == MeshTopologyOp::DissolveVertices || op == MeshTopologyOp::DissolveLimited || op == MeshTopologyOp::ExtrudeRegion || TopologyCopiesSelection(op) || op == MeshTopologyOp::AddPrimitives || op == MeshTopologyOp::ExtrudeVertices || op == MeshTopologyOp::DeleteVertices || TopologyDeletesEdges(op) || op == MeshTopologyOp::KeepSelectedFaces || op == MeshTopologyOp::LevelSet;
}

std::vector<uint32_t> Union(std::vector<uint32_t> a, std::span<const uint32_t> b) {
    a.insert(a.end(), b.begin(), b.end());
    SortUnique(a);
    return a;
}

// Each source attribute and the binding its values read through, which an in-place edit takes from the batch's clones before its insertions move any range.
std::vector<std::pair<uint32_t, NamedAttributeRef>> SourceAttributeRefs(const MeshStore &meshes, uint32_t source, const TopologyReadView *view) {
    std::vector<std::pair<uint32_t, NamedAttributeRef>> refs;
    meshes.ForEachNamedAttribute(source, [&](uint32_t attribute, const NamedAttribute &entry) {
        auto &ref = refs.emplace_back(attribute, meshes.AttributeRef(attribute)).second;
        if (view) ref.ValuesSlot = view->AttributeValues[uint32_t(entry.Type)];
    });
    return refs;
}

void UploadTopologyAttributes(const MeshStore &meshes, uint32_t source, uint32_t output, std::span<const std::pair<uint32_t, NamedAttributeRef>> sources, BufferArena<uint32_t> &scratch, MeshTopologyJob &job) {
    const auto read = [&](const NamedAttribute &entry) {
        const auto attribute = meshes.FindAttribute(source, entry.NameView());
        if (attribute == InvalidOffset || meshes.GetAttribute(attribute).Type != entry.Type || meshes.GetAttribute(attribute).Domain != entry.Domain) return NamedAttributeRef{};
        return std::ranges::find(sources, attribute, &std::pair<uint32_t, NamedAttributeRef>::first)->second;
    };
    std::vector<TopologyAttribute> attributes;
    for (uint32_t d = 0u; d < job.AttributeEnds.size(); ++d) {
        meshes.ForEachNamedAttribute(output, [&](uint32_t attribute, const NamedAttribute &entry) {
            if (uint32_t(entry.Domain) != d) return;
            attributes.push_back({.Type = entry.Type, .Source = read(entry), .Destination = meshes.AttributeRef(attribute), .Default = entry.Default});
        });
        job.AttributeEnds[d] = uint32_t(attributes.size());
    }
    if (attributes.empty()) return;
    const auto words = scratch.Allocate(std::span{reinterpret_cast<const uint32_t *>(attributes.data()), attributes.size() * sizeof(TopologyAttribute) / sizeof(uint32_t)});
    job.Attributes = {scratch.Buffer.Slot, words.Offset};
}

std::vector<uint32_t> InsertedBlocks(ElementHandleRange handles, const mtl::Buffer &list, std::span<const uint32_t> table) {
    std::vector<uint32_t> blocks;
    if (handles.Handles.Slot == InvalidSlot) blocks = RunBlocks(table, handles.First, handles.Count);
    else {
        std::vector<uint32_t> locals;
        for (const auto handle : list.GetSpan<uint32_t>({handles.Handles.Offset, handles.Count})) locals.push_back(handle / MeshElementBlockSize);
        SortUnique(locals);
        blocks = TableBlocks(table, locals);
    }
    SortUnique(blocks);
    return blocks;
}
} // namespace

// The jobs of a construction's edits of one operator, identity policy and pass list, whose passes record together.
struct MeshTopologyEdit::Group {
    Batch Jobs;
    MeshTopologyPushConstants Constants{};
    bool Iterates{};
    // The upload of the batch's last count passes, whose job table the identity passes read.
    Batch::Upload Counted{};
};

// Every workspace of a prepared edit is a range of its chain's scratch.
struct MeshTopologyEdit::Prepared {
    MeshStore::TopologyCounts Bounds;
    TopologyIdentityPolicy Identity;
    std::vector<uint32_t> OutputMaterials;
    MeshClosure Core, Neighborhood;
    // The batch's source clones, which an in-place edit's recorded passes read until it completes.
    std::shared_ptr<const TopologyReadView> SourceView;
    std::vector<std::pair<uint32_t, NamedAttributeRef>> SourceAttributes;
    MeshStore::TopologyCounts SourceCounts;
    // The edit's job, which joins its group's batch at index Job.
    MeshTopologyJob Staged;
    std::shared_ptr<Group> Jobs;
    uint32_t Job{};
    MeshStore::TopologyCounts Counts{};
    ElementHandleRange NewVertices{}, NewFaces{}, NewEdges{};
    uint32_t ProvenancePairs{}; // The provenance pairs the output blends, allocated at emission
    // The connectivity batch of the publication's edits, and the edit's job there.
    std::shared_ptr<ConnectivityBatch> Connectivity;
    uint32_t ConnectivityJob{};
    bool Iterates{};

    MeshTopologyJob &JobOf() { return Jobs->Jobs.Jobs[Job]; }
};

// The closures an edit records before construction's first submit.
// Listed marks a task whose seeds must be live.
// A boolean's Second contains its second source's faces, and its intersection passes run between the closures and the counts.
struct MeshTopologyEdit::Closures {
    std::optional<PrimitiveListReferences> FaceList;
    std::optional<MeshClosure> Around;
    MeshClosure Core, Neighborhood;
    bool Listed;
    uint32_t CollapseCount{};
    SlotOffset CollapseVertices{};
    std::array<ElementWork, 3> Selection{};
    std::optional<MeshClosure> Second{};
    std::optional<BooleanWork> Boolean{};
};

MeshTopologyEdit::MeshTopologyEdit(mtl::ComputeChain &chain, const MeshTopologyTask &task)
    : Chain{chain}, SourceId{task.SourceId}, StoreId{task.SourceId}, Op{task.Op}, SelectionOutput{task.SelectionOutput}, StoreSelectionSource{task.StoreSelectionSource} {}
MeshTopologyEdit::MeshTopologyEdit(MeshTopologyEdit &&) noexcept = default;
MeshTopologyEdit::~MeshTopologyEdit() = default;

namespace {
bool HasTopologySource(const MeshStore &meshes, const MeshTopologyTask &task) {
    const auto policy = TopologyPolicy(task.Op);
    if (!policy) throw std::invalid_argument("Topology source closure is not defined for this operator.");
    constexpr auto CutFlags = TopologyFlagEdgeLists | TopologyFlagSingleCut;
    return Mesh{meshes, task.SourceId}.FaceCount() || (policy->Lines && !(task.Op == MeshTopologyOp::Subdivide && (task.Flags & CutFlags)));
}

bool SelectsSpatialFaces(const MeshTopologyTask &task) {
    return task.Op == MeshTopologyOp::Subdivide && (task.Flags & (TopologyFlagPlaneCuts | TopologyFlagScreenCuts));
}

// The tasks with each level set's output counts, which one pass over every lattice measures, or none without a level set.
std::vector<MeshTopologyTask> MeasureLevelSets(state::Registry &r, std::span<const MeshTopologyTask> tasks) {
    if (std::ranges::none_of(tasks, [](const MeshTopologyTask &task) { return task.Op == MeshTopologyOp::LevelSet; })) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<LevelSetCountJob> jobs;
    std::vector<uint32_t> tiles;
    for (const auto &task : tasks) {
        if (task.Op != MeshTopologyOp::LevelSet) continue;
        const auto points = Mesh{meshes, task.SourceId}.VertexCount();
        jobs.push_back({.Values = task.Param0Attribute, .Size = task.LatticeSize, .Level = task.Param1, .Points = points, .Counts = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u).Offset}});
        tiles.push_back(TileCount(points, JobTileThreads));
    }
    mtl::GatherPass(chain, GetMeshPipelines(r)[MeshPass::LevelSetCount], std::span{jobs}, tiles, JobTileThreads);
    chain.Submit();
    std::vector<MeshTopologyTask> measured{tasks.begin(), tasks.end()};
    for (uint32_t k = 0u; auto &task : measured) {
        if (task.Op != MeshTopologyOp::LevelSet) continue;
        const auto counts = chain.Scratch.Get({jobs[k++].Counts.Offset, 2u});
        task.Outputs = uvec3{counts[0], 3u * counts[1], counts[1]};
    }
    return measured;
}
} // namespace

std::optional<MeshTopologyEdit::Closures> MeshTopologyEdit::RecordClosures(state::Registry &r, const MeshTopologyTask &task, std::optional<PrimitiveListReferences> face_list, const SpatialFaceWork *spatial_faces) {
    auto &chain = Chain;
    const bool fresh = TopologyCreatesRecord(task.Op);
    const auto &meshes = r.Resources.get<MeshStore>();
    if (!HasTopologySource(meshes, task)) return std::nullopt;
    const SelectedHandles handles{meshes, StoreId, task.Selection};
    if (!(task.Flags & TopologyFlagSelectAll) && task.Op != MeshTopologyOp::AddPrimitives &&
        handles->Vertices.empty() && handles->Edges.empty() && handles->Faces.empty()) return std::nullopt;
    const auto policy = TopologyPolicy(task.Op);
    const auto original = meshes.Get(StoreId);
    const bool lines = !Mesh{meshes, StoreId}.FaceCount();
    OriginalClassMode = original.Classification;
    uint32_t collapse_count{};
    SlotOffset collapse_vertices{};
    if (task.Op == MeshTopologyOp::AddPrimitives) {
        if (task.AppendedBase < meshes.Arenas().Vertices.HandleBound(original.Vertices) ||
            task.AppendedBase % MeshElementBlockSize) {
            throw std::invalid_argument("Topology face list has an invalid vertex append base.");
        }
        if (!face_list->PrimitiveCount) return std::nullopt;
    }
    const bool select_all = (task.Flags & TopologyFlagSelectAll) != 0u;
    const auto selected = [&](Element element) {
        return select_all ? EncodeAllSeed(r, chain, StoreId, element) :
                            ListSeed(r, chain, StoreId, element, handles->Get(element));
    };
    ClosureSeed faces, edges, retained, second_faces;
    std::optional<MeshClosure> around;
    const bool listed_cuts = task.Op == MeshTopologyOp::Subdivide && (task.Flags & TopologyFlagEdgeLists);
    const bool listed_vertices = task.Op == MeshTopologyOp::DissolveVertices && (task.Flags & TopologyFlagListSelects);
    const bool listed = (listed_vertices || listed_cuts) && !task.List.empty() && task.List.front();
    const auto expand_vertices = [&](ClosureSeed seed, bool incident = true) {
        if (incident) {
            around = EncodeVertexClosure(r, chain, StoreId, seed);
            if (lines && task.Op != MeshTopologyOp::AddPrimitives) edges = AroundVertices(r, StoreId, Element::Edge, around->Elements[3], seed);
            else faces = AroundVertices(r, StoreId, Element::Face, around->Elements[2], seed);
            if ((fresh || TopologyIsMerge(task.Op) || TopologyIsDissolve(task.Op) || TopologyDeletesEdges(task.Op) || task.Op == MeshTopologyOp::DeleteVertices || task.Op == MeshTopologyOp::ExtrudeEdges || task.Op == MeshTopologyOp::Subdivide) && !lines) {
                edges = AroundVertices(r, StoreId, Element::Edge, around->Elements[3], seed);
            }
        }
        if (task.Op == MeshTopologyOp::MergeCollapse) {
            collapse_count = seed.Count;
            if (collapse_count && !select_all) {
                const auto list = chain.Scratch.Allocate(uint32_t(seed.Vertices.size()));
                chain.Scratch.Buffer.Update(std::as_bytes(std::span{seed.Vertices}), uint64_t(list.Offset) * 4u);
                collapse_vertices = {chain.Scratch.Buffer.Slot, list.Offset};
            }
        }
        if (RetainsSeedVertices(task.Op)) retained = std::move(seed);
    };
    if ((TopologyCopiesSelection(task.Op) || task.Op == MeshTopologyOp::ExtrudeRegion) && task.SelectionElement == Element::None) {
        retained = selected(Element::Vertex);
        const auto selected_edges = selected(Element::Edge);
        const auto selected_faces = selected(Element::Face);
        if (!retained.Count && !selected_edges.Count && !selected_faces.Count) return std::nullopt;
        if (selected_faces.Count) {
            auto vertices = retained.Vertices;
            vertices.insert(vertices.end(), selected_edges.Vertices.begin(), selected_edges.Vertices.end());
            vertices.insert(vertices.end(), selected_faces.Vertices.begin(), selected_faces.Vertices.end());
            SortUnique(vertices);
            const auto seed = select_all ? selected(Element::Vertex) : ListSeed(r, chain, StoreId, Element::Vertex, vertices);
            around = EncodeVertexClosure(r, chain, StoreId, seed);
            edges = AroundVertices(r, StoreId, Element::Edge, around->Elements[3], seed);
            faces = AroundVertices(r, StoreId, Element::Face, around->Elements[2], seed);
        } else if (selected_edges.Count) {
            const auto seed = EncodeEdgeVertices(r, chain, StoreId, selected_edges);
            around = EncodeVertexClosure(r, chain, StoreId, seed);
            edges = AroundVertices(r, StoreId, Element::Edge, around->Elements[3], seed);
            if (!lines) faces = AroundVertices(r, StoreId, Element::Face, around->Elements[2], seed);
        }
    } else if (task.Op == MeshTopologyOp::DeleteLoose) {
        edges = selected(Element::Edge);
        retained = selected(Element::Vertex);
        if (!edges.Count && !retained.Count) return std::nullopt;
    } else if (task.Op == MeshTopologyOp::ExtrudeVertices) {
        if (task.SelectionElement != Element::None && task.SelectionElement != Element::Vertex) {
            throw std::invalid_argument("Extrude Vertices requires a vertex selection.");
        }
        retained = selected(Element::Vertex);
        if (!retained.Count) return std::nullopt;
    } else if (spatial_faces) {
        if (!spatial_faces->Count) return std::nullopt;
        faces = FaceSeed(r, StoreId, chain.Scratch, spatial_faces->Faces);
    } else if (task.SelectionElement != Element::None) {
        auto seed = selected(task.SelectionElement);
        if (!seed.Count) return std::nullopt;
        if (task.SelectionElement == Element::Face) faces = std::move(seed);
        else if (task.SelectionElement == Element::Edge && lines) edges = std::move(seed);
        else {
            if (task.SelectionElement == Element::Edge) seed = EncodeEdgeVertices(r, chain, StoreId, seed);
            expand_vertices(std::move(seed));
        }
    } else if (lines && policy->Seed == Element::Edge) {
        edges = selected(Element::Edge);
    } else if (policy->Seed != Element::Face) {
        ClosureSeed seed;
        if (task.Op == MeshTopologyOp::AddPrimitives) {
            seed = ListSeed(r, chain, StoreId, Element::Vertex, face_list->ExistingVertices);
        } else if (listed_cuts) {
            std::vector<uint32_t> handles;
            handles.reserve(task.List.front());
            for (uint32_t i = 0u; i < task.List.front(); ++i) handles.push_back(task.List[1u + 2u * i]);
            seed = EncodeEdgeVertices(r, chain, StoreId, ListSeed(r, chain, StoreId, Element::Edge, handles));
        } else if (listed_vertices) {
            seed = ListSeed(r, chain, StoreId, Element::Vertex, std::span<const uint32_t>{task.List}.subspan(1u));
        } else if (policy->Seed == Element::Vertex) {
            seed = selected(Element::Vertex);
        } else {
            seed = EncodeEdgeVertices(r, chain, StoreId, selected(Element::Edge));
        }
        if (!seed.Count) {
            if (listed) throw std::invalid_argument("Topology element list contains no live source elements.");
            return std::nullopt;
        }
        expand_vertices(std::move(seed), task.Op != MeshTopologyOp::AddPrimitives || face_list->FaceCount);
        if (task.Op == MeshTopologyOp::AddPrimitives && !face_list->EdgeCorners.empty()) edges = ListSeed(r, chain, StoreId, Element::Edge, face_list->Wires);
    } else if (task.Op == MeshTopologyOp::ReplaceFaces) {
        TopologyOutputBounds(task, {});
        const auto offset = 2u + task.List[0], count = task.List[offset - 1u];
        std::vector<uint32_t> handles;
        handles.reserve(count);
        for (uint32_t i = 0u; i < count; ++i) handles.push_back(task.List[task.List[offset + i]]);
        if (!std::ranges::is_sorted(handles) || std::ranges::adjacent_find(handles) != handles.end()) {
            throw std::invalid_argument("Face replacement face handles must be sorted and unique.");
        }
        faces = ListSeed(r, chain, StoreId, Element::Face, handles);
    } else if (task.Op == MeshTopologyOp::FlipNormals && !task.List.empty()) {
        if (task.List.front() != task.List.size() - 1u) throw std::invalid_argument("Flip Normals has an invalid face list.");
        faces = ListSeed(r, chain, StoreId, Element::Face, std::span{task.List}.subspan(1u));
    } else if (task.Op == MeshTopologyOp::Boolean) {
        // The second mesh's faces follow the first's, from the handle past the first's last face.
        const auto &a = meshes.Arenas();
        const auto &other = meshes.Get(task.SecondSourceId);
        const auto entries = a.FaceTriangles.Entries(original.FaceData);
        const auto split = a.FaceTriangles.LiveBound(original.FaceData) - a.FaceTriangles.LiveBound(other.FaceData);
        const std::array counts{a.FaceTriangles.Count(original.FaceData) - a.FaceTriangles.Count(other.FaceData), a.FaceTriangles.Count(other.FaceData)};
        const std::array corners{a.FaceCorners.Count(original.FaceCorners) - a.FaceCorners.Count(other.FaceCorners), a.FaceCorners.Count(other.FaceCorners)};
        std::array<ElementWorkSeedJob, 2> jobs;
        for (uint32_t side = 0u; side < 2u; ++side) {
            std::vector<uint32_t> blocks;
            for (uint32_t local = 0u; local < entries.size(); ++local) {
                if (entries[local] != InvalidOffset && (side ? (local + 1u) * MeshElementBlockSize > split : local * MeshElementBlockSize < split)) blocks.push_back(entries[local]);
            }
            jobs[side] = PrepareBlockMembershipWork(chain.Scratch, a.FaceTriangles, original.FaceData, blocks, AllocateElementWork(chain.Scratch, a.FaceTriangles.HandleBound(original.FaceData), uint32_t(blocks.size())));
            jobs[side].First = side ? split : 0u;
            jobs[side].End = side ? InvalidOffset : split;
            (side ? second_faces : faces) = {.Work = jobs[side].Work, .Count = counts[side], .Incidence = corners[side], .LoopFans = corners[side]};
        }
        EncodeElementMembershipWork(r, chain, jobs);
        EncodeSortElementWork(r, chain, std::array{faces.Work, second_faces.Work});
    } else {
        faces = selected(Element::Face);
    }
    if (task.Op == MeshTopologyOp::ExtrudeEdges && task.SelectionElement == Element::None && (faces.Count || edges.Count)) {
        retained = selected(Element::Vertex);
    }
    auto core = EncodePrimitiveClosure(r, chain, StoreId, faces, edges, retained, task.Op == MeshTopologyOp::DeleteLoose);
    if (faces.Count && !lines && (task.Op == MeshTopologyOp::DissolveFaces || task.Op == MeshTopologyOp::DissolveVertices || (task.Op == MeshTopologyOp::DissolveLimited && task.SelectionElement == Element::Face) || task.Op == MeshTopologyOp::Wireframe) && (!select_all || TopologyIsDissolve(task.Op))) {
        // A vertex can be retired only after every face incident to the selected face core has marked whether it still keeps that vertex.
        auto vertices = core.Seed(Element::Vertex);
        vertices.All = faces.All || edges.All || retained.All;
        if (!vertices.All) {
            vertices.Vertices = faces.Vertices;
            vertices.Vertices.insert(vertices.Vertices.end(), edges.Vertices.begin(), edges.Vertices.end());
            vertices.Vertices.insert(vertices.Vertices.end(), retained.Vertices.begin(), retained.Vertices.end());
            SortUnique(vertices.Vertices);
        }
        const auto incident = EncodeVertexClosure(r, chain, StoreId, vertices);
        core = EncodePrimitiveClosure(r, chain, StoreId, AroundVertices(r, StoreId, Element::Face, incident.Elements[2], vertices), TopologyIsDissolve(task.Op) ? AroundVertices(r, StoreId, Element::Edge, incident.Elements[3], vertices) : ClosureSeed{}, retained);
    }
    MeshClosure neighborhood;
    if (!fresh) {
        neighborhood = EncodeVertexClosure(r, chain, StoreId, core.Seed(Element::Vertex));
        neighborhood.EncodeIncidence(r, chain, StoreId, Element::Face);
    }
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(StoreId);
    const std::array selection{SeedElementWorkHandles(chain.Scratch, a.Vertices.HandleBound(record.Vertices), handles->Vertices), SeedElementWorkHandles(chain.Scratch, a.EdgeHalfedges.HandleBound(record.EdgeData), handles->Edges), SeedElementWorkHandles(chain.Scratch, a.FaceTriangles.HandleBound(record.FaceData), handles->Faces)};
    std::optional<MeshClosure> second;
    if (task.Op == MeshTopologyOp::Boolean) second = EncodePrimitiveClosure(r, chain, StoreId, second_faces);
    return Closures{.FaceList = std::move(face_list), .Around = std::move(around), .Core = core, .Neighborhood = neighborhood, .Listed = listed, .CollapseCount = collapse_count, .CollapseVertices = collapse_vertices, .Selection = selection, .Second = std::move(second)};
}

bool MeshTopologyEdit::FinishClosures(const MeshTopologyTask &task, Closures &closures) { // NOLINT(readability-make-member-function-const)
    auto &[face_list, around, core, neighborhood, listed, collapse_count, collapse_vertices, selection, second, boolean] = closures;
    if (around) {
        around->Finish(Chain);
        if (!around->Counts[0]) {
            if (listed) throw std::invalid_argument("Topology element list contains no live source elements.");
            return false;
        }
        if (!around->Counts[1] && !RetainsSeedVertices(task.Op)) return false;
    }
    core.Finish(Chain);
    if (second) second->Finish(Chain);
    if (task.Op == MeshTopologyOp::DeleteLoose && !core.Counts[0]) return false;
    if (!TopologyCreatesRecord(task.Op)) neighborhood.Finish(Chain);
    return core.Counts[1] || RetainsSeedVertices(task.Op);
}

void MeshTopologyEdit::PrepareCounts(state::Registry &r, const MeshTopologyTask &task, Closures &closures, const std::shared_ptr<const TopologyReadView> &view) {
    auto &chain = Chain;
    const auto &meshes = r.Resources.get<MeshStore>();
    const bool fresh = TopologyCreatesRecord(task.Op);
    const auto identity = fresh ? TopologyIdentityPolicy::Fresh : TopologyIdentityPolicy::Preserve;
    auto &[face_list, around, core, neighborhood, listed, collapse_count, collapse_vertices, selection, second, boolean] = closures;
    const auto list_range = PrepareTopologyList(task, face_list ? &*face_list : nullptr, core, chain.Scratch);
    const MeshStore::TopologyCounts source_counts = boolean ? MeshStore::TopologyCounts{boolean->Vertices, boolean->Intersections, boolean->Triangles} : MeshStore::TopologyCounts{core.Counts[0], core.Counts[1], core.Counts[2]};
    profile::RecordCounter("TopologySourceVertices", source_counts.Vertices);
    profile::RecordCounter("TopologySourceHalfedges", source_counts.Halfedges);
    profile::RecordCounter("TopologySourceFaces", source_counts.Faces);
    ElementWork primitive_work{};
    std::vector<uint32_t> output_materials;
    if (task.Op == MeshTopologyOp::LevelSet) {
        output_materials = {0u};
    } else if (fresh && !boolean) {
        const auto palette = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(StoreId).PrimitiveMaterials);
        std::vector<uint32_t> primitives;
        primitives.reserve(core.Counts[2] ? core.Counts[2] : core.Counts[0]);
        const bool faces = core.Counts[2] != 0u;
        const auto &record = meshes.Get(StoreId);
        const auto layer = faces ? meshes.Arenas().FacePrimitives.View(record.FacePrimitives) : meshes.Arenas().VertexPrimitives.View(record.VertexPrimitives);
        ForEachWorkElement(chain.Scratch, core.Elements[faces ? 2u : 0u], [&](uint32_t handle) {
            const auto primitive = layer.GetOr(handle);
            if (primitive >= palette.size()) throw std::out_of_range("Separated element has no primitive material.");
            primitives.push_back(primitive);
        });
        SortUnique(primitives);
        primitive_work = SeedElementWorkHandles(chain.Scratch, uint32_t(palette.size()), primitives);
        for (const auto primitive : primitives) output_materials.push_back(palette[primitive]);
    }
    const auto bounds = boolean ? MeshStore::TopologyCounts{boolean->OutputVertices, 3u * boolean->OutputTriangles, boolean->OutputTriangles} : TopologyOutputBounds(task, source_counts);
    auto initial_job = TopologyJob(meshes, task, core, list_range, collapse_count, collapse_vertices);
    if (boolean) {
        // The source's layers and material table cover both meshes, and its second side's elements follow the first's in each source domain.
        initial_job.Src2Counts = second->Counts;
        initial_job.Src2Elements = second->Elements;
        const auto materials = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(StoreId).PrimitiveMaterials);
        output_materials.assign(materials.begin(), materials.end());
        initial_job.Boolean = boolean->Job;
    }
    Plan = std::make_unique<Prepared>(Prepared{.Bounds = bounds, .Identity = identity, .OutputMaterials = std::move(output_materials), .SourceView = fresh ? nullptr : view, .SourceCounts = source_counts});
    auto &plan = *Plan;
    plan.Core = core;
    if (!fresh) {
        plan.Neighborhood = neighborhood;
        if (meshes.Get(StoreId).NormalSectors.Count) OldCornerBlocks = meshes.ArenaBlocks(StoreId, MeshStore::ElementDomain::Halfedge, chain.Scratch, neighborhood.Elements[1]);
        SortUnique(OldCornerBlocks);
        ChangedTriangles = EncodeFaceTriangles(r, chain, StoreId, neighborhood.Seed(Element::Face));
    }
    auto &job = plan.Staged = initial_job;
    job.PrimitiveWork = primitive_work;
    job.VertexSelection = selection[0];
    job.EdgeSelection = selection[1];
    job.FaceSelection = selection[2];
    if (task.Op == MeshTopologyOp::MergeAtTarget) {
        // The local job indexes the selected face closure in handle order.
        const auto ordinal = ElementWorkRank(chain.Scratch.Get(WorkStorageRange(core.Elements[0])), core.Elements[0], task.TargetVertex);
        if (ordinal == InvalidOffset) throw std::invalid_argument("Merge target is outside the local source closure.");
        job.TargetVertex = ordinal;
    }
    const auto &source_view = plan.SourceView;
    job.SrcConnectivity = fresh ? meshes.GetConnectivityRef(SourceId) : source_view->SourceConnectivity(meshes, SourceId);
    job.SrcLayers = fresh ? meshes.Layers(SourceId) : source_view->SourceLayers(meshes, SourceId);
    plan.SourceAttributes = SourceAttributeRefs(meshes, SourceId, fresh ? nullptr : source_view.get());
    plan.Iterates = TopologyIterates(task.Op) && !(TopologyIsDissolve(task.Op) && task.Op != MeshTopologyOp::DissolveLimited && !source_counts.Faces);
}

void MeshTopologyEdit::ReadCounts() {
    auto &plan = *Plan;
    auto &job = plan.JobOf();
    const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
    const auto output = ReadOutputCounts(plan.Jobs->Jobs, job, plan.Bounds);
    const auto counts = output.Counts;
    plan.Counts = counts;
    plan.ProvenancePairs = output.ProvenancePairs;
    AddedTriangleCount = output.Triangles;
    job.DstVertexCount = counts.Vertices;
    job.DstHalfedgeCount = counts.Halfedges;
    job.DstFaceCount = counts.Faces;
    Output->Finish(Chain);
    if ((Op == MeshTopologyOp::InsetRegion || Op == MeshTopologyOp::SplitNonplanarFaces || Op == MeshTopologyOp::SplitConcaveFaces) && Output->NewCounts == std::array{0u, 0u}) {
        Output.reset();
        return;
    }
    if (!fresh) ChangedTriangles.Finish(Chain);
    const auto retirement = TopologyPolicy(Op)->Retirement;
    if ((Output->RetiredCounts[0] && !(retirement & RetireVertices)) ||
        (Output->RetiredCounts[1] && !(retirement & RetireFaces))) {
        throw std::logic_error("Local topology retired unsupported source identities: vertices=" + std::to_string(Output->RetiredCounts[0]) + ", faces=" + std::to_string(Output->RetiredCounts[1]));
    }
}

std::vector<MeshTopologyEdit> MeshTopologyEdit::Construct(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshTopologyTask> requested) {
    const profile::CpuScope scope{"TopologyEditConstruct"};
    std::vector<MeshTopologyTask> tasks(requested.begin(), requested.end());
    // A Boolean task edits its first source joined with its second.
    if (std::ranges::any_of(tasks, [](const auto &task) { return task.Op == MeshTopologyOp::Boolean; })) {
        auto &meshes = r.Resources.get<MeshStore>();
        std::vector<std::vector<ConcatPart>> sources;
        for (const auto &task : tasks) {
            if (task.Op == MeshTopologyOp::Boolean) sources.push_back({{task.SourceId}, {task.SecondSourceId}});
        }
        mtl::ComputeChain joining{meshes.BufferContext()};
        const auto joined = meshes.Concatenate(r, joining, sources);
        joining.Submit();
        auto next = joined.begin();
        for (auto &task : tasks) {
            if (task.Op == MeshTopologyOp::Boolean) task.SourceId = *next++;
        }
    }
    std::vector<MeshTopologyEdit> edits;
    std::vector<std::optional<Closures>> closures;
    edits.reserve(tasks.size());
    std::vector<std::optional<PrimitiveListReferences>> face_lists(tasks.size());
    std::vector<std::optional<SpatialFaceWork>> spatial(tasks.size());
    {
        const profile::CpuScope stage{"TopologySourceState"};
        const auto &meshes = r.Resources.get<MeshStore>();
        std::vector<RecordElements> edge_corners;
        for (uint32_t i = 0u; i < tasks.size(); ++i) {
            const auto &task = tasks[i];
            if (!HasTopologySource(meshes, task)) continue;
            if (task.Op == MeshTopologyOp::AddPrimitives) {
                face_lists[i] = ParsePrimitiveListReferences(task);
                if (!face_lists[i]->EdgeCorners.empty()) edge_corners.push_back({task.SourceId, face_lists[i]->EdgeCorners});
            } else if (SelectsSpatialFaces(task)) {
                chain.Lane([&] { spatial[i].emplace(r, chain, task); });
            }
        }
        chain.Submit();
        // The corners of every task's face list read in one submit.
        const auto corners = edge_corners.empty() ? std::vector<std::vector<CornerFacts>>{} : Corners(meshes, GetMeshPipelines(r), edge_corners);
        for (uint32_t i = 0u, k = 0u; i < tasks.size(); ++i) {
            if (!face_lists[i] || face_lists[i]->EdgeCorners.empty()) continue;
            for (const auto &corner : corners[k++]) {
                if (corner.Face == InvalidOffset) face_lists[i]->Wires.push_back(corner.Edge);
            }
        }
    }
    if (std::ranges::any_of(spatial, [](const auto &work) { return work.has_value(); })) {
        for (auto &work : spatial) {
            if (work) chain.Lane([&] { work->RecordFaces(r, chain); });
        }
        chain.Submit();
    }
    for (uint32_t i = 0u; i < tasks.size(); ++i) {
        auto &edit = edits.emplace_back(MeshTopologyEdit{chain, tasks[i]});
        edit.OwnsSource = tasks[i].Op == MeshTopologyOp::Boolean;
    }
    // Each edit's closures record in its own lane, so the closure passes of every edit record together.
    closures.resize(tasks.size());
    for (uint32_t i = 0u; i < tasks.size(); ++i) chain.Lane([&] { closures[i] = edits[i].RecordClosures(r, tasks[i], std::move(face_lists[i]), spatial[i] ? &*spatial[i] : nullptr); });
    chain.Submit();
    const auto view = std::make_shared<TopologyReadView>();
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (closures[i] && !edits[i].FinishClosures(tasks[i], *closures[i])) closures[i].reset();
        if (closures[i] && !TopologyCreatesRecord(tasks[i].Op)) view->Add(r, edits[i].StoreId, closures[i]->Neighborhood, chain.Scratch);
    }
    view->Clone(r);
    std::vector<uint32_t> booleans;
    std::vector<BooleanSources> boolean_sources;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (!closures[i] || tasks[i].Op != MeshTopologyOp::Boolean) continue;
        booleans.push_back(i);
        boolean_sources.push_back({tasks[i], closures[i]->Core, *closures[i]->Second});
    }
    if (!booleans.empty()) {
        for (uint32_t k = 0u; const auto &work : RecordBooleanWork(r, chain, boolean_sources)) closures[booleans[k++]]->Boolean = work;
    }
    std::vector<MeshTopologyEdit *> counted;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (closures[i]) counted.push_back(&edits[i]);
    }
    for (auto *edit : counted) chain.Lane([&] { edit->PrepareCounts(r, tasks[edit - edits.data()], *closures[edit - edits.data()], view); });
    // Edits of one operator, identity policy and pass list share a batch, each job laid out in its own range of the batch's scratch.
    std::vector<std::shared_ptr<Group>> groups;
    {
        const profile::CpuScope stage{"TopologyJobSetup"};
        const auto &meshes = r.Resources.get<MeshStore>();
        using Key = std::tuple<bool, bool, MeshTopologyOp, const TiledPass *>;
        const auto key_of = [](const Prepared &plan) { return Key{plan.Identity == TopologyIdentityPolicy::Fresh, plan.Iterates, plan.Staged.Op, TopologyEmissionPasses(plan.Staged).data()}; };
        std::vector<std::pair<Key, std::vector<MeshTopologyEdit *>>> members;
        for (auto *edit : counted) {
            const auto key = key_of(*edit->Plan);
            auto found = std::ranges::find(members, key, &std::pair<Key, std::vector<MeshTopologyEdit *>>::first);
            if (found == members.end()) found = members.insert(members.end(), {key, {}});
            found->second.push_back(edit);
        }
        for (const auto &[key, group_edits] : members) {
            uint64_t words = 0u;
            for (auto *edit : group_edits) {
                auto &plan = *edit->Plan;
                words += LayoutTopologyScratch(plan.Staged, plan.SourceCounts, plan.Bounds, 0u);
            }
            if (words + Batch::ArgumentWords > UINT32_MAX) throw std::length_error("GPU topology scratch exceeds its address space.");
            auto &group = *groups.emplace_back(std::make_shared<Group>(Group{.Jobs = Batch{chain.Scratch, uint32_t(words)}, .Iterates = std::get<1>(key)}));
            group.Jobs.Begin();
            auto &pc = group.Constants = TopologyPushConstants(meshes);
            pc.ListSlot = chain.Scratch.Buffer.Slot;
            if (!std::get<0>(key)) pc.Source = view->Arenas;
            for (auto *edit : group_edits) {
                auto &plan = *edit->Plan;
                const auto job_words = LayoutTopologyScratch(plan.Staged, plan.SourceCounts, plan.Bounds, 0u);
                LayoutTopologyScratch(plan.Staged, plan.SourceCounts, plan.Bounds, group.Jobs.AllocateScratch(job_words));
                plan.Jobs = groups.back();
                plan.Job = uint32_t(group.Jobs.Jobs.size());
                AddJob(group.Jobs, plan.Staged, tasks[edit - edits.data()]);
            }
        }
    }
    // The groups whose label rounds have not converged record again from their prepare passes with twice the rounds.
    auto pending = groups;
    for (uint32_t rounds = LabelRounds; !pending.empty(); rounds *= 2u) {
        // Indirect label passes bind the scratch storage directly, so every upload of this submit is reserved before any records.
        uint64_t upload_words = 0u;
        for (const auto &group : pending) upload_words += group->Jobs.UploadWords();
        chain.Scratch.ReserveAdditional(upload_words);
        for (auto &group : pending) group->Counted = EncodeTopologyCounts(r, chain, group->Jobs, group->Constants, group->Iterates ? rounds : 0u);
        std::erase_if(pending, [](const auto &group) { return !group->Iterates; });
        if (pending.empty()) break;
        chain.Submit();
        std::erase_if(pending, [](const auto &group) { return group->Jobs.IndirectGroups(SrcHalfedges) == 0u; });
    }
    for (auto *edit : counted) {
        chain.Lane([&] {
            auto &plan = *edit->Plan;
            const auto record = plan.Jobs->Counted.JobsOffset + plan.Job * uint32_t(sizeof(MeshTopologyJob) / sizeof(uint32_t));
            edit->Output = std::make_unique<TopologyOutputHandles>(r, chain, edit->SourceId, plan.JobOf(), record, plan.Bounds, plan.Jobs->Jobs.ScratchBinding(), plan.Jobs->Constants.Source, plan.Identity);
        });
    }
    // Identity planning reads the scanned counts on the GPU, so the host reads both after one submit.
    chain.Submit();
    for (auto &edit : edits) {
        if (edit.Output) edit.ReadCounts();
    }
    // An owned source without output has no recorded reader left.
    for (auto &edit : edits) {
        if (!edit.Output && std::exchange(edit.OwnsSource, false)) r.Resources.get<MeshStore>().Release(edit.SourceId);
    }
    return edits;
}

std::vector<ExtrusionSource> MeshTopologyEdit::ExtrusionSources(state::Registry &r) const {
    std::vector<ExtrusionSource> sources;
    if (!Output) return sources;
    if (Published) throw std::logic_error("Extrusion sources must be read before publication.");
    const bool faces = Op == MeshTopologyOp::ExtrudeFacesIndividual;
    if (!faces && Op != MeshTopologyOp::ExtrudeRegion && Op != MeshTopologyOp::ExtrudeEdges && Op != MeshTopologyOp::ExtrudeVertices) return sources;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &job = Plan->JobOf();
    const auto counts = Chain.Scratch.Get({Plan->Jobs->Jobs.ScratchBinding().Offset + job.CountsOffset, job.CountEntries});
    const auto output = Chain.Scratch.Get(Output->Vertices);
    std::vector<uint32_t> elements;
    ForEachWorkElement(Chain.Scratch, faces ? job.SrcFaceWork : job.SrcVertexWork, [&](uint32_t element) { elements.push_back(element); });
    const auto normals = faces ? FaceNormals(meshes, SourceId, elements) : VertexNormals(meshes, SourceId, elements);
    // Face entries follow the vertex and halfedge entries.
    const uint32_t first = faces ? job.SrcVertexCount + job.SrcHalfedgeCount : 0u;
    for (uint32_t i = 0u; i < elements.size(); ++i) {
        for (auto v = counts[first + i]; v < counts[first + i + 1u]; ++v) {
            if (faces || output[v] == InvalidOffset) sources.push_back({v, elements[i], normals[i]});
        }
    }
    return sources;
}

void MeshTopologyEdit::PublishAll(state::Registry &r, std::span<MeshTopologyEdit> all) {
    const profile::CpuScope scope{"TopologyEditPublish"};
    std::vector<MeshTopologyEdit *> edits;
    for (auto &edit : all) {
        if (!edit.Output) continue;
        if (edit.Published) throw std::logic_error("Topology edit was already published.");
        edits.push_back(&edit);
    }
    if (edits.empty()) return;
    auto &chain = edits.front()->Chain;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    using D = MeshStore::ElementDomain;

    // Every edit's output identities are inserted before any emission.
    for (auto *edit : edits) {
        const profile::CpuScope stage{"TopologyElementInsert"};
        auto &plan = *edit->Plan;
        if (plan.Identity == TopologyIdentityPolicy::Fresh) edit->StoreId = meshes.BeginTopologyOutput(edit->SourceId, plan.OutputMaterials);
        const auto counts = plan.Counts;
        if (plan.Identity == TopologyIdentityPolicy::Fresh) {
            auto &record = meshes.WriteRecord(edit->StoreId);
            record.FacePrimitivesReady = counts.Faces != 0u;
            record.VertexPrimitivesReady = counts.Faces == 0u;
        }
        auto &job = plan.JobOf();
        edit->SourceTriangles = chain.Scratch.Allocate(edit->AddedTriangleCount);
        job.DstTriangleSources = {chain.Scratch.Buffer.Slot, edit->SourceTriangles.Offset};
        if (plan.ProvenancePairs) job.Provenance = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * plan.ProvenancePairs).Offset};
        plan.NewVertices = meshes.InsertElements(edit->StoreId, D::Vertex, edit->Output->NewCounts[0], &chain.Scratch);
        plan.NewFaces = meshes.InsertElements(edit->StoreId, D::Face, edit->Output->NewCounts[1], &chain.Scratch);
        job.DstCornerOffset = meshes.InsertElements(edit->StoreId, D::Halfedge, counts.Halfedges, nullptr).First;
        job.DstTriangleOffset = edit->FirstTriangle = meshes.InsertElements(edit->StoreId, D::Triangle, edit->AddedTriangleCount, nullptr).First;
        // Every new edge contains an emitted halfedge, and a line contains two.
        plan.NewEdges = meshes.InsertElements(edit->StoreId, D::Edge, counts.Faces ? counts.Halfedges : counts.Halfedges / 2u, &chain.Scratch);
        chain.Lane([&] { edit->Output->Assign(r, chain, {plan.NewVertices, plan.NewFaces}); });
    }
    // New edges and fan items take bounds the host knows, then trim to the counts that submit reports.
    std::vector<Group *> groups;
    // The blocks each edit's emission writes, of vertices, faces and corners.
    std::vector<std::array<std::vector<uint32_t>, 3>> written(edits.size());
    for (uint32_t i = 0u; auto *edit : edits) {
        const profile::CpuScope stage{"TopologyEmit"};
        auto &[vertex_blocks, face_blocks, corner_blocks] = written[i++];
        auto &plan = *edit->Plan;
        if (!std::ranges::contains(groups, plan.Jobs.get())) groups.push_back(plan.Jobs.get());
        const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
        auto &job = plan.JobOf();
        const auto counts = plan.Counts;
        const auto &record = meshes.Get(edit->StoreId);
        const auto vertex_table = a.Vertices.Entries(record.Vertices), corner_table = a.FaceCorners.Entries(record.FaceCorners), face_table = a.FaceTriangles.Entries(record.FaceData);
        const auto inserted_vertices = InsertedBlocks(plan.NewVertices, chain.Scratch.Buffer, vertex_table), inserted_faces = InsertedBlocks(plan.NewFaces, chain.Scratch.Buffer, face_table);
        vertex_blocks = fresh ? inserted_vertices : Union(WorkBlocks(chain.Scratch, plan.Core.Elements[0], 0u, vertex_table), inserted_vertices);
        face_blocks = fresh ? inserted_faces : Union(WorkBlocks(chain.Scratch, plan.Core.Elements[2], 0u, face_table), inserted_faces);
        job.HasVertexPrimitives |= uint32_t(meshes.Get(edit->StoreId).VertexPrimitivesReady) << 1u;
        job.DstVertexHandles = {chain.Scratch.Buffer.Slot, edit->Output->Vertices.Offset};
        job.DstFaceHandles = {chain.Scratch.Buffer.Slot, edit->Output->Faces.Offset};
        job.DstConnectivity = meshes.GetConnectivityRef(edit->StoreId);
        job.DstLayers = meshes.Layers(edit->StoreId);
        if (const auto &output = edit->SelectionOutput) {
            job.DstVertexBits = MaskBinding(output->Vertices);
            job.DstEdgeBits = MaskBinding(output->Edges);
            job.DstFaceBits = MaskBinding(output->Faces);
        }
        UploadTopologyAttributes(meshes, edit->SourceId, edit->StoreId, plan.SourceAttributes, chain.Scratch, job);
        CaptureTopologyEmitWrites(r, edit->StoreId, job, edit->AddedTriangleCount, vertex_blocks, face_blocks, edit->SelectionOutput);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Vertex, vertex_blocks);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Face, face_blocks);
        corner_blocks = RunBlocks(corner_table, job.DstCornerOffset, counts.Halfedges);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Corner, corner_blocks);
    }
    // Each group's jobs emit together, sized by their exact counts.
    const auto tiles = [&](Group &group, uint32_t domain, auto &&count) {
        group.Jobs.SetDomainTiles(domain, group.Jobs.Jobs | std::views::transform([&](const MeshTopologyJob &job) { return TileCount(std::invoke(count, job), TileElements); }) | std::ranges::to<std::vector>());
    };
    std::vector<Batch::Upload> emissions;
    for (auto *group : groups) {
        tiles(*group, DstVertices, &MeshTopologyJob::DstVertexCount);
        tiles(*group, DstHalfedges, &MeshTopologyJob::DstHalfedgeCount);
        tiles(*group, DstFaces, &MeshTopologyJob::DstFaceCount);
        emissions.push_back(group->Jobs.Encode(chain, pipelines, group->Constants, TopologyEmissionPasses(group->Jobs.Jobs.front())));
    }
    std::vector<MeshConnectivityJob> connectivity_jobs(edits.size());
    uint64_t connectivity_words = 0u;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        chain.Lane([&] {
            auto *edit = edits[i];
            auto &plan = *edit->Plan;
            const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
            const auto &job = plan.JobOf();
            const auto counts = plan.Counts;
            const auto &before = plan.Neighborhood;
            const std::array emitted{ElementHandleRange{.Handles = job.DstVertexHandles, .Count = counts.Vertices}, ElementHandleRange{.First = job.DstCornerOffset, .Count = counts.Halfedges}, ElementHandleRange{.Handles = job.DstFaceHandles, .Count = counts.Faces}};
            edit->Repair = std::make_unique<ConnectivityEditWork>(r, chain, edit->StoreId, before, std::array{plan.Core.Elements[1], plan.Core.Elements[2]}, emitted);
            // Emission replaces the core's loops inside the neighborhood, so the repaired counts are known before the pass runs.
            const std::array<uint32_t, 3> repaired = fresh ? std::array{counts.Vertices, counts.Halfedges, counts.Faces} : std::array{before.Counts[0] + edit->Output->NewCounts[0], before.Counts[1] - plan.Core.Counts[1] + counts.Halfedges, before.Counts[2] - plan.Core.Counts[2] + counts.Faces};
            const auto &record = meshes.Get(edit->StoreId);
            edit->RetiredEdges = AllocateElementWork(chain.Scratch, a.EdgeHalfedges.HandleBound(record.EdgeData), before.Counts[3]);
            auto &connectivity = connectivity_jobs[i] = MeshConnectivityJob{
                .Connectivity = job.DstConnectivity,
                .Vertices = edit->Repair->Elements[0],
                .Halfedges = edit->Repair->Elements[1],
                .Faces = edit->Repair->Elements[2],
                .EdgeHandles = plan.NewEdges,
                .SourceConnectivity = job.SrcConnectivity,
                .SourceEdges = before.Elements[3],
                .ConvertedEdges = job.Op == MeshTopologyOp::AddPrimitives || TopologyMapsWireEdges(job.Op) ? plan.Core.Elements[3] : ElementWork{},
                .ConvertedWirePairs = TopologyMapsWireEdges(job.Op) ? SlotOffset{chain.Scratch.Buffer.Slot, plan.Jobs->Jobs.ScratchBinding().Offset + job.WireEdgeMapOffset} : SlotOffset{},
                .SourceEdgeCount = before.Counts[3],
                .VertexCount = repaired[0],
                .HalfedgeCount = repaired[1],
                .FaceCount = repaired[2],
                .FaceStarts = 1u,
                .RetiredEdgeWork = edit->RetiredEdges,
            };
            connectivity_words += LayoutConnectivityScratch(connectivity);
        });
    }
    if (connectivity_words > UINT32_MAX) throw std::length_error("GPU connectivity scratch exceeds its address space.");
    // The edits' connectivity repairs run as one batch.
    const auto rebuilt = std::make_shared<ConnectivityBatch>(chain.Scratch, uint32_t(connectivity_words));
    rebuilt->Begin();
    std::vector<uint32_t> fan_vertex_blocks;
    for (uint32_t i = 0u; auto *edit : edits) {
        auto &plan = *edit->Plan;
        const auto &before = plan.Neighborhood;
        const auto &[vertex_blocks, face_blocks, corner_blocks] = written[i];
        const auto &record = meshes.Get(edit->StoreId);
        const auto corner_table = a.FaceCorners.Entries(record.FaceCorners), face_table = a.FaceTriangles.Entries(record.FaceData), edge_table = a.EdgeHalfedges.Entries(record.EdgeData);
        plan.Connectivity = rebuilt;
        plan.ConnectivityJob = uint32_t(rebuilt->Jobs.size());
        AddConnectivityJob(*rebuilt, connectivity_jobs[i++]);
        auto edge_blocks = Union(WorkBlocks(chain.Scratch, before.Elements[3], 0u, edge_table), InsertedBlocks(plan.NewEdges, chain.Scratch.Buffer, edge_table));
        auto repaired_face_blocks = Union(WorkBlocks(chain.Scratch, before.Elements[2], 0u, face_table), face_blocks);
        CaptureConnectivityPrepareWrites(r, rebuilt->Jobs.back(), vertex_blocks, Union(WorkBlocks(chain.Scratch, before.Elements[1], 0u, corner_table), corner_blocks), repaired_face_blocks);
        CaptureTopologyEdgeWrites(r, edge_blocks, edit->SelectionOutput);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Edge, edge_blocks);
        fan_vertex_blocks.insert(fan_vertex_blocks.end(), vertex_blocks.begin(), vertex_blocks.end());
        edit->RepairedBlocks = {std::move(edge_blocks), std::move(repaired_face_blocks)};
    }
    rebuilt->Encode(chain, pipelines, TiledJobPushConstants{}, ConnectivityPasses);
    SortUnique(fan_vertex_blocks);
    EncodeVertexFans(r, chain, rebuilt->Jobs, edits | std::views::transform([](const auto *edit) { return edit->StoreId; }) | std::ranges::to<std::vector>(), fan_vertex_blocks);
    for (uint32_t i = 0u; auto *group : groups) group->Jobs.Record(chain, pipelines, emissions[i++], group->Constants, std::array{TiledPass{MeshPass::TopologyEdgeAttributes, DstHalfedges}});
    std::vector<ElementWork> retired;
    std::vector<MeshStore::CornerClassRequest> requests;
    requests.reserve(edits.size());
    for (auto *edit : edits) {
        const auto &connected = rebuilt->Jobs[edit->Plan->ConnectivityJob];
        retired.push_back(edit->RetiredEdges);
        requests.push_back({edit->StoreId, edit->Repair->Elements[0], connected.VertexCount, connected.HalfedgeCount});
    }
    EncodeSortElementWork(r, chain, retired);
    auto classes = meshes.EncodeCornerClassification(r, chain, requests);
    {
        const profile::CpuScope stage{"TopologyEmitSubmit"};
        chain.Submit();
    }

    // The edits' corner classes plan as one batch before one derive records the base normals the authored normals rebase on.
    std::vector<LocalNormalWork> normals;
    normals.reserve(edits.size());
    for (auto *edit : edits) {
        const profile::CpuScope stage{"TopologyNormals"};
        auto &plan = *edit->Plan;
        const auto &connected = rebuilt->Jobs[plan.ConnectivityJob];
        edit->Repair->Finish(chain);
        if (edit->Repair->Counts != std::array{connected.VertexCount, connected.HalfedgeCount, connected.FaceCount}) {
            throw std::logic_error("Connectivity repair disagrees with its emitted closure.");
        }
        const auto state = rebuilt->ScratchSpan().subspan(connected.StateOffset, 4u);
        meshes.TrimInsertedElements(edit->StoreId, D::Edge, plan.NewEdges, chain.Scratch, state[0]);
        // An edge's corners are its unpaired ones plus the two its first halfedge pairs unless it is open, so the unpaired corners change with the halfedges, edges and open edges.
        meshes.WriteRecord(edit->StoreId).UnpairedCorners += connected.HalfedgeCount - plan.Neighborhood.Counts[1] - 2u * (state[0] - state[1]) + 2u * (state[2] - state[3]);
        CheckElementWork(chain.Scratch, edit->RetiredEdges);
        normals.push_back({edit->StoreId, edit->Repair->Elements[0], edit->Repair->Counts[0], edit->Repair->Elements[2], edit->Repair->Counts[2]});
    }
    meshes.PlanCornerClassification(r, chain, classes);
    EncodeDeriveMeshNormals(r, chain, chain.Scratch, normals);
    for (auto *edit : edits) {
        auto &plan = *edit->Plan;
        auto &job = plan.JobOf();
        if (job.CornerAttributes & MeshAttributeBit_Normal) {
            job.DstLayers = meshes.Layers(edit->StoreId);
            if (plan.Identity == TopologyIdentityPolicy::Preserve) {
                job.RetainedNormalCorners = plan.Neighborhood.Elements[1];
                job.RetainedNormalCornerCount = plan.Neighborhood.Counts[1];
            }
            CaptureTopologyNormalWrites(r, edit->StoreId, job, chain.Scratch);
        }
        edit->Published = true;
    }
    // The jobs with corner normals rebase their authored normals, one batch per group.
    for (auto *group : groups) {
        const auto normal = [](const MeshTopologyJob &job) { return (job.CornerAttributes & MeshAttributeBit_Normal) != 0u; };
        if (std::ranges::none_of(group->Jobs.Jobs, normal)) continue;
        tiles(*group, DstHalfedges, [&](const MeshTopologyJob &job) { return normal(job) ? std::max(job.DstHalfedgeCount, job.RetainedNormalCornerCount) : 0u; });
        group->Jobs.Encode(chain, pipelines, group->Constants, std::array{TiledPass{MeshPass::TopologyCustomNormals, DstHalfedges}});
    }
}

void MeshTopologyEdit::FinishAll(state::Registry &r, std::span<MeshTopologyEdit *const> edits) {
    const profile::CpuScope scope{"FinishTopologyEdits"};
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<MeshStore::SelectionUpdate> updates;
    std::vector<uint32_t> sources;
    for (auto *edit : edits) {
        if (edit->Finished) throw std::logic_error("Topology edit was already finished.");
        edit->Finished = true;
        if (edit->Published && edit->Plan) {
            edit->Chain.Submit();
            edit->Plan.reset();
        }
        if (std::exchange(edit->OwnsSource, false)) sources.push_back(edit->SourceId);
        if (!edit->Output) continue;
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = edit->StoreId, .Source = edit->StoreSelectionSource});
        if (edit->StoreId != edit->SourceId) {
            for (uint32_t d = 0u; const auto domain : SelectionDomains) {
                const auto blocks = meshes.GetBlockList(edit->StoreId, domain).Blocks;
                update.Blocks[d++].assign(blocks.begin(), blocks.end());
            }
            continue;
        }
        const auto &storage = edit->Chain.Scratch;
        const auto &output = *edit->Output;
        update.Blocks[1] = meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Edge, storage, edit->RetiredEdges);
        meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Halfedge, storage, output.Replaced[0]);
        meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Triangle, storage, output.Replaced[1]);
        if (output.RetiredCounts[1]) update.Blocks[2] = meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Face, storage, output.Retired[1]);
        if (output.RetiredCounts[0]) update.Blocks[0] = meshes.EraseElements(edit->StoreId, MeshStore::ElementDomain::Vertex, storage, output.Retired[0]);
        update.Blocks[0].append_range(meshes.ArenaBlocks(edit->StoreId, MeshStore::ElementDomain::Vertex, storage, edit->Repair->Elements[0]));
        update.Blocks[1].insert(update.Blocks[1].end(), edit->RepairedBlocks[0].begin(), edit->RepairedBlocks[0].end());
        update.Blocks[2].insert(update.Blocks[2].end(), edit->RepairedBlocks[1].begin(), edit->RepairedBlocks[1].end());
    }
    meshes.Release(sources);
    if (!updates.empty()) meshes.UpdateSelection(r, edits.front()->Chain, updates);
}

std::vector<GeometryTopologyResult> ExecuteGeometryTopology(state::Registry &r, std::span<const MeshTopologyTask> listed, const std::function<void(mtl::ComputeChain &, std::span<MeshTopologyEdit>, uint32_t first)> &finish, const std::function<void(std::span<const MeshTopologyEdit>, uint32_t first)> &constructed) {
    // Level sets bound their outputs by their measured counts.
    const auto measured = MeasureLevelSets(r, listed);
    const auto tasks = measured.empty() ? listed : std::span<const MeshTopologyTask>{measured};
    // A fresh copy emits before its source's one mutation overwrites the source's canonical values.
    std::set<uint32_t> mutated;
    for (const auto &task : tasks) {
        if (mutated.contains(task.SourceId)) {
            throw std::invalid_argument("Dependent topology tasks require ExecuteGeometryTopologyStages.");
        }
        if (!TopologyCreatesRecord(task.Op)) mutated.insert(task.SourceId);
    }
    const auto &meshes = r.Resources.get<MeshStore>();
    // An edit records at most 11 closure jobs, 5 identity jobs, a connectivity edit job, 4 element repair jobs and 2 meshlet bounds jobs.
    // Each job takes its record and its first tile in a job table, and each table its count.
    constexpr auto JobWords = [](size_t bytes) { return uint32_t(bytes / sizeof(uint32_t)) + 2u; };
    constexpr uint32_t JobTableWords{11u * JobWords(sizeof(MeshClosureJob)) + 5u * JobWords(sizeof(TopologyIdentityJob)) + JobWords(sizeof(ConnectivityEditJob)) + 4u * JobWords(sizeof(ElementRepairJob)) + 2u * JobWords(sizeof(MeshletBoundsJob))};
    // A task's scratch words, bounded before any GPU result: its fixed tables, its job tables, its job's layout at its output bounds with corner normals, its closures per source element, its publish per output element, and a Boolean's intersection work over the triangles of both sources with its job tables.
    const auto words = tasks | std::views::transform([&](const MeshTopologyTask &task) {
        const Mesh first{meshes, task.SourceId};
        const bool boolean = task.Op == MeshTopologyOp::Boolean;
        const uint64_t vertices = uint64_t(first.VertexCount()) + (boolean ? Mesh{meshes, task.SecondSourceId}.VertexCount() : 0u);
        const uint64_t triangles = boolean ? uint64_t(meshes.Get(task.SourceId).TriangleCount) + meshes.Get(task.SecondSourceId).TriangleCount : 0u;
        const auto clamped = [](uint64_t count) { return uint32_t(std::min<uint64_t>(count, UINT32_MAX)); };
        const MeshStore::TopologyCounts source = boolean ? MeshStore::TopologyCounts{clamped(vertices), clamped(3u * triangles), clamped(triangles)} : MeshStore::TopologyCounts{first.VertexCount(), first.HalfEdgeCount(), first.FaceCount()};
        MeshTopologyJob job{.Op = task.Op, .Flags = task.Flags, .Steps = task.Steps, .Param0 = task.Param0, .CornerAttributes = MeshAttributeBit_Normal};
        const auto bounds = TopologyOutputBounds(task, source);
        const uint64_t layout = LayoutTopologyScratch(job, source, bounds, 0u);
        const uint64_t elements = uint64_t(source.Vertices) + source.Halfedges + source.Faces, emitted = uint64_t(bounds.Vertices) + bounds.Halfedges + bounds.Faces;
        return clamped((3u << 10) + JobTableWords + layout + 16u * elements + 4u * emitted + (boolean ? 2u * vertices + 67u * triangles + BooleanTableWords : 0u));
    }) | std::ranges::to<std::vector>();
    // The tasks run in chunks whose scratch stays within the budget, in order.
    const auto split = ChunkByScratch(uint32_t(tasks.size()), ScratchWordBudget, [&](uint32_t i) { return words[i]; });
    std::vector<GeometryTopologyResult> results;
    results.reserve(tasks.size());
    // The chunks run on one chain in turn, each after the scratch of the one before it is released.
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    for (const auto chunk : split.Chunks) {
        if (chunk.Offset) chain.ReleaseScratch();
        auto edits = MeshTopologyEdit::Construct(r, chain, tasks.subspan(chunk.Offset, chunk.Count));
        if (constructed) constructed(edits, chunk.Offset);
        MeshTopologyEdit::PublishAll(r, edits);
        for (const auto &edit : edits) results.push_back({.GeometryId = edit.StoreId, .Changed = bool(edit.Output)});
        if (finish) finish(chain, edits, chunk.Offset);
        else MeshTopologyEdit::FinishAll(r, edits | std::views::transform([](MeshTopologyEdit &edit) { return &edit; }) | std::ranges::to<std::vector>());
        chain.Submit();
    }
    return results;
}

std::vector<GeometryTopologyResult> ExecuteGeometryTopologyStages(state::Registry &r, std::span<const MeshTopologyTask> tasks) {
    std::vector<GeometryTopologyResult> results;
    results.reserve(tasks.size());
    for (const auto &task : tasks) results.push_back(ExecuteGeometryTopology(r, std::span{&task, 1u}).front());
    return results;
}

namespace {
ShortEdgesJob ShortEdgesOf(const MeshStore &meshes, uint32_t id, float distance) {
    const auto &a = meshes.Arenas();
    return {
        .Connectivity = meshes.GetConnectivityRef(id),
        .Blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Halfedge),
        .VertexSlot = a.Vertices.Buffer.Slot,
        .Distance = distance,
    };
}
// Records `pass` over the halfedge blocks of every job.
void RecordShortEdges(state::Registry &r, mtl::ComputeChain &chain, MeshPass pass, std::span<const ShortEdgesJob> jobs) {
    mtl::GatherPass(chain, GetMeshPipelines(r)[pass], jobs, [](const ShortEdgesJob &job) { return job.Blocks.Count * MeshElementBlockSize; });
}
// Whether each record has an edge shorter than its distance.
std::vector<bool> HasShortEdges(state::Registry &r, std::span<const uint32_t> ids, std::span<const float> distances) {
    const auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto flags = chain.Scratch.Allocate(uint32_t(ids.size()));
    std::ranges::fill(chain.Scratch.GetMutable(flags), 0u);
    std::vector<ShortEdgesJob> jobs;
    for (uint32_t i = 0u; i < ids.size(); ++i) {
        auto &job = jobs.emplace_back(ShortEdgesOf(meshes, ids[i], distances[i]));
        job.Flag = {chain.Scratch.Buffer.Slot, flags.Offset + i};
    }
    RecordShortEdges(r, chain, MeshPass::FlagShortEdges, jobs);
    chain.Submit();
    return chain.Scratch.Get(flags) | std::views::transform([](uint32_t flag) { return flag != 0u; }) | std::ranges::to<std::vector>();
}
// Each record's edges with a vertex on their interior within its distance, each paired with that vertex, as a Subdivide junction list.
std::vector<std::vector<uint32_t>> Junctions(state::Registry &r, std::span<const uint32_t> ids, std::span<const float> distances) {
    const auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<ShortEdgesJob> jobs;
    jobs.reserve(ids.size());
    for (uint32_t i = 0u; i < ids.size(); ++i) {
        auto &job = jobs.emplace_back(ShortEdgesOf(meshes, ids[i], distances[i]));
        job.Capacity = meshes.Arenas().EdgeHalfedges.Count(meshes.Get(ids[i]).EdgeData);
        const auto list = chain.Scratch.Allocate(1u + 2u * job.Capacity);
        chain.Scratch.GetMutable({list.Offset, 1u})[0] = 0u;
        job.Junctions = {chain.Scratch.Buffer.Slot, list.Offset};
    }
    RecordShortEdges(r, chain, MeshPass::FindJunctions, jobs);
    chain.Submit();
    return jobs | std::views::transform([&](const ShortEdgesJob &job) {
               const auto words = chain.Scratch.Get({job.Junctions.Offset, 1u + 2u * job.Capacity});
               const auto count = std::min(words[0], job.Capacity);
               std::vector<uint32_t> junctions(words.begin(), words.begin() + 1u + 2u * count);
               junctions[0] = count;
               return junctions;
           }) |
        std::ranges::to<std::vector>();
}
// The record's edges with more than two face sides, ascending.
std::vector<uint32_t> NonmanifoldEdges(const Mesh &mesh) {
    const auto &c = mesh.GetConnectivity();
    std::vector<uint32_t> edges;
    for (const auto h : he::HandleRange<he::HH>{c.HalfedgeTable, c.HalfedgeBlocks}) {
        const auto e = c.Edge(h);
        const auto first = c.EdgeHalfedge(*e);
        if (h != first && h != c.Opposite(first)) edges.push_back(*e);
    }
    std::ranges::sort(edges);
    edges.erase(std::ranges::unique(edges).begin(), edges.end());
    return edges;
}
} // namespace

std::vector<std::optional<uint32_t>> ExecuteGeometryBooleans(state::Registry &r, std::span<const MeshTopologyTask> tasks) {
    std::vector<std::optional<uint32_t>> outputs(tasks.size());
    std::vector<uint32_t> ids;
    for (uint32_t i = 0u; const auto &result : ExecuteGeometryTopology(r, tasks)) {
        if (result.Changed) ids.push_back(*(outputs[i] = result.GeometryId));
        ++i;
    }
    if (ids.empty()) return outputs;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto tolerances = VertexExtents(meshes, GetMeshPipelines(r), ids) | std::views::transform([](float extent) { return RelativePrecision * extent; }) | std::ranges::to<std::vector>();
    // Welds the records at the indices, each within its tolerance.
    const auto weld = [&](std::span<const uint32_t> indices) {
        std::vector<MeshTopologyTask> merges, loose;
        for (const auto i : indices) {
            merges.push_back({.SourceId = ids[i], .Op = MeshTopologyOp::MergeByDistance, .Param0 = tolerances[i], .Flags = TopologyFlagSelectAll | TopologyFlagDropFlaps});
            loose.push_back({.SourceId = ids[i], .Op = MeshTopologyOp::DeleteLoose, .Flags = TopologyFlagSelectAll});
        }
        if (merges.empty()) return;
        ExecuteGeometryTopology(r, merges);
        ExecuteGeometryTopology(r, loose);
    };
    auto pending = std::views::iota(0u, uint32_t(ids.size())) | std::ranges::to<std::vector>();
    const auto pending_ids = [&] { return pending | std::views::transform([&](uint32_t i) { return ids[i]; }) | std::ranges::to<std::vector>(); };
    const auto pending_tolerances = [&] { return pending | std::views::transform([&](uint32_t i) { return tolerances[i]; }) | std::ranges::to<std::vector>(); };
    const auto short_edges = HasShortEdges(r, ids, tolerances);
    weld(pending | std::views::filter([&](uint32_t i) { return short_edges[i]; }) | std::ranges::to<std::vector>());
    while (!pending.empty()) {
        auto junctions = Junctions(r, pending_ids(), pending_tolerances());
        std::vector<MeshTopologyTask> splits;
        std::vector<uint32_t> split;
        for (uint32_t k = 0u; k < pending.size(); ++k) {
            if (!junctions[k][0]) continue;
            splits.push_back({.SourceId = ids[pending[k]], .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagListJunctions | TopologyFlagSelectAll, .List = std::move(junctions[k])});
            split.push_back(pending[k]);
        }
        if (!splits.empty()) ExecuteGeometryTopology(r, splits);
        weld(split);
        pending = std::move(split);
    }
    // The weld joins surface sheets that touch along an edge, and an edge split there gives each sheet its own copy of the edge.
    std::vector<MeshTopologyTask> separations;
    for (const auto id : ids) {
        if (const Mesh mesh{meshes, id}; !mesh.HasClosedSurface()) {
            if (auto edges = NonmanifoldEdges(mesh); !edges.empty()) separations.push_back({.SourceId = id, .Op = MeshTopologyOp::EdgeSplit, .SelectionElement = Element::Edge, .Selection = GeometrySelection{.Edges = std::move(edges)}});
        }
    }
    if (!separations.empty()) ExecuteGeometryTopology(r, separations);
    return outputs;
}
