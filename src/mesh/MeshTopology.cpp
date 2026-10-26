#include "mesh/MeshTopology.h"
#include "SortUnique.h"
#include "mesh/MeshTopologyLayout.h"

#include "Profile.h"
#include "gpu/ShortEdgesPushConstants.h"
#include "gpu/TiledJobPushConstants.h"
#include "mesh/BooleanWork.h"
#include "mesh/ConnectivityBatch.h"
#include "mesh/ConnectivityWritePages.h"
#include "mesh/CornerClassBatch.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/NormalDeriveGpu.h"
#include "mesh/ScratchChunks.h"
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

// Label rounds one submit encodes before the host checks convergence.
constexpr uint32_t LabelRounds{16};

constexpr std::array PreparePasses{
    TiledPass{MeshPass::TopologyZero, SrcVertices},
    TiledPass{MeshPass::TopologyListFill, ListEntries, 1},
    TiledPass{MeshPass::TopologyMarkHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces},
    TiledPass{MeshPass::TopologyMarkRetainedEdges, RetainedEdgeVertices},
    TiledPass{MeshPass::TopologyFinalizeVertices, SrcVertices},
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
// Once vertex targets are final, remove newly coincident polygons before surviving faces take precedence over loose edges.
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

// Scan boolean vertex, intersection, and triangle counts after the intersection passes determine their sizes.
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
    return job.CollapseCount ? std::span<const TiledPass>{CollapseOutputPasses} : std::span<const TiledPass>{OutputPasses};
}

PrimitiveListReferences ParsePrimitiveListReferences(const MeshTopologyTask &task) {
    // The existing bound parser validates polygon lengths and the terminal cursor.
    // These references are action metadata.
    // No geometry is read back.
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
        .SrcVertexCount = source.Counts[0],
        .SrcHalfedgeCount = source.Counts[1],
        .SrcFaceCount = source.Counts[2],
        .SrcEdgeCount = source.Counts[3],
        .SrcVertexWork = source.Elements[0],
        .SrcHalfedgeWork = source.Elements[1],
        .SrcFaceWork = source.Elements[2],
        .SrcEdgeWork = source.Elements[3],
        .SourceLayers = src.CornerAttributes | src.VertexAttributes << 16u,
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
        tiles[CountBlocks] = 4u * job.CountBlockCount;
        tiles[Once] = batch.Jobs.empty() ? 1u : 0u;
        batch.AddJob(job, tiles);
        return;
    }
    tiles[TableEntries] = TileCount(TopologyTableWords(job.Op, {V, H, F}), TileElements);
    tiles[SrcVertices] = TileCount(V, TileElements);
    tiles[RetainedEdgeVertices] = TopologyClassifiesEdges(job.Op) ? TileCount(V, TileElements) : 0u;
    tiles[SrcHalfedges] = TileCount(H, TileElements);
    tiles[SrcFaces] = TileCount(F + 2u, TileElements);
    // A knife processes every source edge; selection and cut-list operations process their listed entries.
    const auto listed = ((job.Flags & (TopologyFlagListSelects | TopologyFlagListCuts)) || job.Op == MeshTopologyOp::ReplaceFaces || job.Op == MeshTopologyOp::Decimate) && !task.List.empty() ? task.List.front() : 0u;
    tiles[ListEntries] = TileCount((job.Flags & TopologyFlagScreenCuts) ? job.SrcEdgeCount : listed, TileElements);
    tiles[CountBlocks] = 4u * job.CountBlockCount;
    tiles[Once] = batch.Jobs.empty() ? 1u : 0u;
    for (uint32_t level = 0u, count = job.CollapseCount; count; ++level) {
        tiles[Collapse0 + level] = TileCount(count, TileElements);
        count = count > TileElements ? TileCount(count, TileElements) : 0u;
    }
    tiles[CollapseDigits] = job.CollapseCount ? 16u : 0u;
    batch.AddJob(job, tiles);
}

void EncodeTopologyCounts(state::Registry &r, mtl::ComputeChain &chain, Batch &batch, const MeshTopologyPushConstants &pc, uint32_t rounds) {
    const profile::CpuScope scope{"EncodeTopologyCounts"};
    if (batch.Jobs.front().Op == MeshTopologyOp::Boolean) {
        batch.Encode(chain, GetMeshPipelines(r), pc, BooleanCountPasses);
        return;
    }
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
    batch.Encode(chain, GetMeshPipelines(r), pc, passes);
}

std::pair<MeshStore::TopologyCounts, uint32_t> ReadOutputCounts(const Batch &batch, const MeshTopologyJob &job, MeshStore::TopologyCounts bounds) {
    const auto scratch = batch.ScratchSpan();
    const auto total = [&](uint32_t q) { return scratch[job.CountsOffset + q * job.CountEntries + job.CountEntries - 1]; };
    const uint64_t halfedges = uint64_t(total(2)) + total(3);
    if (total(0) > bounds.Vertices || halfedges > bounds.Halfedges || total(1) > bounds.Faces || uint64_t(total(1)) * 3u > total(2) ||
        (total(3) & 1u) || (!total(1) && total(2))) {
        throw std::runtime_error("GPU topology counts exceed the reserved output bounds.");
    }
    return {{total(0), uint32_t(halfedges), total(1)}, total(2) - 2u * total(1)};
}

// Upload once, then remap only the handle words in the operator's payload.
// Geometry and corner provenance remain in canonical storage.
Range PrepareTopologyList(const MeshTopologyTask &task, const PrimitiveListReferences *primitives, const MeshClosure &core, BufferArena<uint32_t> &storage) {
    const auto op = task.Op;
    const bool vertices = (op == MeshTopologyOp::ConnectVertices || op == MeshTopologyOp::DeleteVertices ||
                           op == MeshTopologyOp::DissolveVertices || op == MeshTopologyOp::RotateEdges) &&
        (task.Flags & TopologyFlagListSelects);
    const bool edges = op == MeshTopologyOp::Subdivide && (task.Flags & (TopologyFlagListSelects | TopologyFlagListCuts));
    if (!vertices && !edges && op != MeshTopologyOp::AddPrimitives && op != MeshTopologyOp::Decimate && op != MeshTopologyOp::ReplaceFaces) return {};
    const auto range = storage.Allocate(std::span<const uint32_t>{task.List});
    const auto list = storage.GetMutable(range);
    const auto rank = [&](uint32_t domain, uint32_t handle) {
        const auto work = core.Elements[domain];
        const auto result = ElementWorkRank(storage.Get(WorkStorageRange(work)), work, handle);
        if (result == InvalidOffset) throw std::invalid_argument("Topology list names an element outside its source closure.");
        return result;
    };
    if (vertices || edges) {
        const auto stride = edges && (task.Flags & TopologyFlagListCuts) ? 2u : 1u;
        if (list.empty() || 1ull + uint64_t(list[0]) * stride != list.size()) throw std::invalid_argument("Invalid topology element list.");
        for (uint32_t i = 0u; i < list[0]; ++i) {
            auto &handle = list[1u + i * stride];
            handle = rank(edges ? 3u : 0u, handle);
        }
    } else if (op == MeshTopologyOp::AddPrimitives) {
        for (const auto at : primitives->Offsets) {
            auto &handle = list[at];
            if (handle < task.AppendedBase) handle = rank(0u, handle);
            else handle = core.Counts[0] + handle - task.AppendedBase;
        }
    } else if (op == MeshTopologyOp::Decimate) {
        for (uint32_t i = 0u; i < list[0]; ++i) {
            for (uint32_t k = 1u; k <= 2u; ++k) list[5u * i + k] = rank(0u, list[5u * i + k]);
        }
    } else {
        for (uint32_t i = 1u; i <= list[0]; ++i) list[i] = rank(0u, list[i]);
        const auto offset = 2u + list[0], count = list[offset - 1u];
        if (count != core.Counts[2]) throw std::invalid_argument("Face replacement source faces are not live.");
        for (uint32_t i = 0u; i < count; ++i) {
            auto cursor = list[offset + i] + 1u;
            const auto polygons = list[cursor++];
            for (uint32_t j = 0u; j < polygons; ++j) {
                const auto n = list[cursor++];
                for (uint32_t k = 0u; k < 2u * n; ++k) rank(1u, list[cursor++]);
            }
        }
    }
    return range;
}

} // namespace

namespace {
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

// The operator owns its topology rule.
// The transaction owns publication and retirement.
// Keep every permission here so a new rule cannot be admitted to the local path with an inconsistent second allowlist.
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
        default: return std::nullopt;
    }
}

// These rules also operate on seed vertices without any face around them.
bool RetainsSeedVertices(MeshTopologyOp op) {
    return TopologyIsMerge(op) || op == MeshTopologyOp::DissolveVertices || op == MeshTopologyOp::DissolveLimited || op == MeshTopologyOp::ExtrudeRegion || TopologyCopiesSelection(op) || op == MeshTopologyOp::AddPrimitives || op == MeshTopologyOp::ExtrudeVertices || op == MeshTopologyOp::DeleteVertices || TopologyDeletesEdges(op) || op == MeshTopologyOp::KeepSelectedFaces;
}

// Ascending unique blocks of both inputs.
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

// Uploads a record of each named attribute of the output, which carries each source's attribute of its name, type and domain to it, in runs by domain.
void UploadTopologyAttributes(const MeshStore &meshes, uint32_t source, uint32_t output, std::span<const std::pair<uint32_t, NamedAttributeRef>> sources, BufferArena<uint32_t> &scratch, MeshTopologyJob &job) {
    // Use the matching source attribute, or the output default when the source has none.
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

// Ascending arena blocks of inserted handles, a run or a list in `list`, through the record's table entries.
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

// Every workspace of a prepared edit is a range of its chain's scratch.
struct MeshTopologyEdit::Prepared {
    MeshStore::TopologyCounts Bounds;
    TopologyIdentityPolicy Identity;
    std::vector<uint32_t> OutputMaterials;
    MeshClosure Core, Neighborhood;
    // The batch's source clones, which an in-place edit's recorded passes read until it completes.
    std::shared_ptr<const TopologyReadView> SourceView;
    std::vector<std::pair<uint32_t, NamedAttributeRef>> SourceAttributes;
    Batch Jobs;
    MeshTopologyPushConstants Constants{};
    MeshStore::TopologyCounts Counts{};
    ElementHandleRange NewFaces{}, NewEdges{};
    std::optional<ConnectivityBatch> Connectivity;
    // Whether the operator joins through label rounds, which repeat until a round changes nothing.
    bool Iterates{};

    Prepared(BufferArena<uint32_t> &scratch, MeshStore::TopologyCounts bounds, TopologyIdentityPolicy identity, std::vector<uint32_t> materials, std::shared_ptr<const TopologyReadView> source_view, uint32_t scratch_words)
        : Bounds{bounds}, Identity{identity}, OutputMaterials{std::move(materials)}, SourceView{std::move(source_view)}, Jobs{scratch, scratch_words} {}
};

// The closures an edit records before construction's first submit.
// Listed marks a task naming seeds that must be live.
// Selection contains the task's selected vertices, edges and faces, which the job classifies over its source domains.
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
// A mesh without faces edits its lines through an edge core.
// An operator without a line rule, or a subdivide that cuts by a list, a plane or a screen segment, has no source there.
bool HasTopologySource(const MeshStore &meshes, const MeshTopologyTask &task) {
    const auto policy = TopologyPolicy(task.Op);
    if (!policy) throw std::invalid_argument("Topology source closure is not defined for this operator.");
    constexpr auto CutFlags = TopologyFlagListSelects | TopologyFlagListCuts | TopologyFlagPlaneCuts | TopologyFlagScreenCuts;
    return Mesh{meshes, task.SourceId}.FaceCount() || (policy->Lines && !(task.Op == MeshTopologyOp::Subdivide && (task.Flags & CutFlags)));
}

bool SelectsSpatialFaces(const MeshTopologyTask &task) {
    return ((task.Op == MeshTopologyOp::DeleteFaces || task.Op == MeshTopologyOp::DeleteOnlyFaces) && (task.Flags & TopologyFlagPlaneSide)) ||
        (task.Op == MeshTopologyOp::Subdivide && (task.Flags & (TopologyFlagPlaneCuts | TopologyFlagScreenCuts)));
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
        // Every existing vertex handle is below the append base.
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
    // Every level is bounded on the host, so the closure records in one submit.
    // A vertex or edge seed's vertex level decides after it whether the edit has any source.
    ClosureSeed faces, edges, retained, second_faces;
    std::optional<MeshClosure> around;
    const bool listed_cuts = task.Op == MeshTopologyOp::Subdivide && (task.Flags & (TopologyFlagListSelects | TopologyFlagListCuts));
    const bool listed_vertices = (task.Op == MeshTopologyOp::ConnectVertices || task.Op == MeshTopologyOp::DeleteVertices || task.Op == MeshTopologyOp::DissolveVertices) && (task.Flags & TopologyFlagListSelects);
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
        // Only selected loose edges and isolated vertices enter the core.
        // Faces stay in the neighborhood.
        edges = selected(Element::Edge);
        retained = selected(Element::Vertex);
        if (!edges.Count && !retained.Count) return std::nullopt;
    } else if (task.Op == MeshTopologyOp::ExtrudeVertices) {
        // The rule only adds vertices and edges.
        // Incident primitives belong to the repair neighborhood.
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
        if (task.Op == MeshTopologyOp::Decimate) {
            if (task.List.empty() || 1ull + 5ull * task.List.front() != task.List.size()) throw std::invalid_argument("Decimate has an invalid collapse plan.");
            std::vector<uint32_t> vertices;
            for (uint32_t i = 0u; i < task.List.front(); ++i) {
                vertices.push_back(task.List[1u + 5u * i]);
                vertices.push_back(task.List[2u + 5u * i]);
            }
            SortUnique(vertices);
            seed = ListSeed(r, chain, StoreId, Element::Vertex, vertices);
        } else if (task.Op == MeshTopologyOp::AddPrimitives) {
            seed = ListSeed(r, chain, StoreId, Element::Vertex, face_list->ExistingVertices);
        } else if (listed_cuts) {
            const auto stride = (task.Flags & TopologyFlagListCuts) ? 2u : 1u;
            if ((task.Flags & (TopologyFlagListSelects | TopologyFlagListCuts)) ==
                    (TopologyFlagListSelects | TopologyFlagListCuts) ||
                task.List.empty() || 1ull + uint64_t(task.List.front()) * stride != task.List.size()) {
                throw std::invalid_argument("Subdivide has an invalid edge list.");
            }
            std::vector<uint32_t> handles;
            handles.reserve(task.List.front());
            for (uint32_t i = 0u; i < task.List.front(); ++i) handles.push_back(task.List[1u + i * stride]);
            seed = EncodeEdgeVertices(r, chain, StoreId, ListSeed(r, chain, StoreId, Element::Edge, handles));
        } else if (listed_vertices) {
            if (task.List.empty() || task.List.front() != task.List.size() - 1u) throw std::invalid_argument("Topology vertex selection list is invalid.");
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
        if (task.Op == MeshTopologyOp::AddPrimitives && !face_list->EdgeCorners.empty()) {
            const Mesh mesh{meshes, StoreId};
            std::vector<uint32_t> wires;
            for (const auto corner : face_list->EdgeCorners) {
                const he::HH h{corner};
                if (!mesh.GetConnectivity().FaceOf(h)) wires.push_back(*mesh.GetEdge(h));
            }
            edges = ListSeed(r, chain, StoreId, Element::Edge, wires);
        }
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
        // The second mesh's faces fill the source's last local face blocks.
        const auto &a = meshes.Arenas();
        const auto &other = meshes.Get(task.SecondSourceId);
        const auto entries = a.FaceTriangles.Entries(original.FaceData);
        const auto split = uint32_t(entries.size() - a.FaceTriangles.Entries(other.FaceData).size());
        const std::array counts{a.FaceTriangles.Count(original.FaceData) - a.FaceTriangles.Count(other.FaceData), a.FaceTriangles.Count(other.FaceData)};
        const std::array corners{a.FaceCorners.Count(original.FaceCorners) - a.FaceCorners.Count(other.FaceCorners), a.FaceCorners.Count(other.FaceCorners)};
        std::array<ElementWorkSeedJob, 2> jobs;
        for (uint32_t side = 0u; side < 2u; ++side) {
            std::vector<uint32_t> blocks;
            for (auto local = side ? split : 0u; local < (side ? uint32_t(entries.size()) : split); ++local) {
                if (entries[local] != InvalidOffset) blocks.push_back(entries[local]);
            }
            jobs[side] = PrepareBlockMembershipWork(chain.Scratch, a.FaceTriangles, original.FaceData, blocks, AllocateElementWork(chain.Scratch, a.FaceTriangles.HandleBound(original.FaceData), uint32_t(blocks.size())));
            (side ? second_faces : faces) = {.Work = jobs[side].Work, .Count = counts[side], .Incidence = corners[side], .LoopFans = corners[side]};
        }
        EncodeElementMembershipWork(r, chain, jobs);
        EncodeSortElementWork(r, chain, std::array{faces.Work, second_faces.Work});
    } else {
        faces = selected(Element::Face);
    }
    // Edge extrusion deselects original vertices, including selected points outside the edge closure.
    if (task.Op == MeshTopologyOp::ExtrudeEdges && task.SelectionElement == Element::None && (faces.Count || edges.Count)) {
        retained = selected(Element::Vertex);
    }
    // A line vertex reads whether it keeps a line from its fan, so a line core needs no widening.
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
    // An in-place edit's neighborhood is every fan around the core's vertices.
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
    if (task.Op == MeshTopologyOp::RotateEdges &&
        ((task.Flags & TopologyFlagListSelects) == 0u || task.List.empty() || task.List.front() != task.List.size() - 1u)) {
        throw std::invalid_argument("Rotate Edges has an invalid vertex selection list.");
    }
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
    if (fresh && !boolean) {
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
    const auto scratch_words = LayoutTopologyScratch(initial_job, source_counts, bounds, Batch::ArgumentWords);
    Plan = std::make_unique<Prepared>(chain.Scratch, bounds, identity, std::move(output_materials), fresh ? nullptr : view, scratch_words);
    auto &plan = *Plan;
    plan.Core = core;
    if (!fresh) {
        plan.Neighborhood = neighborhood;
        if (meshes.Get(StoreId).NormalSectors.Count) OldCornerBlocks = meshes.ArenaBlocks(StoreId, MeshStore::ElementDomain::Halfedge, chain.Scratch, neighborhood.Elements[1]);
        SortUnique(OldCornerBlocks);
        ChangedTriangles = EncodeFaceTriangles(r, chain, StoreId, neighborhood.Seed(Element::Face));
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
    plan.SourceAttributes = SourceAttributeRefs(meshes, SourceId, fresh ? nullptr : source_view.get());
    auto &pc = plan.Constants;
    pc = TopologyPushConstants(meshes);
    if (list_range.Count) pc.ListSlot = chain.Scratch.Buffer.Slot;
    if (!fresh) pc.Source = source_view->Arenas;
    pc.Source.Layers = fresh ? meshes.Layers(SourceId) : source_view->SourceLayers(meshes, SourceId);
    // A line dissolve joins no faces, so it takes no label rounds.
    plan.Iterates = TopologyIterates(task.Op) && !(TopologyIsDissolve(task.Op) && task.Op != MeshTopologyOp::DissolveLimited && !source_counts.Faces);
}

void MeshTopologyEdit::ReadCounts() {
    auto &plan = *Plan;
    auto &job = plan.Jobs.Jobs[0];
    const bool fresh = plan.Identity == TopologyIdentityPolicy::Fresh;
    const auto [counts, triangles] = ReadOutputCounts(plan.Jobs, job, plan.Bounds);
    plan.Counts = counts;
    AddedTriangleCount = triangles;
    job.DstVertexCount = counts.Vertices;
    job.DstHalfedgeCount = counts.Halfedges;
    job.DstFaceCount = counts.Faces;
    Output->Finish(Chain);
    // A boundary-free inset or a polygon split below its threshold has no effect.
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
    // A boolean edits a combined copy of its meshes, which the edit owns: a copy of the first mesh with a copy of the second joined after it.
    std::vector<MeshTopologyTask> tasks(requested.begin(), requested.end());
    for (auto &task : tasks) {
        if (task.Op != MeshTopologyOp::Boolean) continue;
        auto &meshes = r.Resources.get<MeshStore>();
        const auto copied = CloneRecords(r, std::array{task.SourceId, task.SecondSourceId});
        meshes.Join(r, copied[0], copied[1]);
        meshes.Release(copied[1]);
        task.SourceId = copied[0];
    }
    std::vector<MeshTopologyEdit> edits;
    std::vector<std::optional<Closures>> closures;
    edits.reserve(tasks.size());
    // Face lists name their faces up front.
    // Spatial predicates record canonical candidate membership in the first submit.
    std::vector<std::optional<PrimitiveListReferences>> face_lists(tasks.size());
    std::vector<std::optional<SpatialFaceWork>> spatial(tasks.size());
    {
        const profile::CpuScope stage{"TopologySourceState"};
        const auto &meshes = r.Resources.get<MeshStore>();
        for (uint32_t i = 0u; i < tasks.size(); ++i) {
            const auto &task = tasks[i];
            if (const auto *handles = std::get_if<GeometrySelection>(&task.Selection)) ValidateGeometrySelection(meshes, task.SourceId, *handles);
            if (!HasTopologySource(meshes, task)) continue;
            if (task.Op == MeshTopologyOp::AddPrimitives) {
                face_lists[i] = ParsePrimitiveListReferences(task);
            } else if (SelectsSpatialFaces(task)) {
                spatial[i].emplace(r, chain, task);
            }
        }
        chain.Submit();
    }
    if (std::ranges::any_of(spatial, [](const auto &work) { return work.has_value(); })) {
        for (auto &work : spatial) {
            if (work) work->RecordFaces(r, chain);
        }
        chain.Submit();
    }
    for (uint32_t i = 0u; i < tasks.size(); ++i) {
        auto &edit = edits.emplace_back(MeshTopologyEdit{chain, tasks[i]});
        edit.OwnsSource = tasks[i].Op == MeshTopologyOp::Boolean;
        closures.push_back(edit.RecordClosures(r, tasks[i], std::move(face_lists[i]), spatial[i] ? &*spatial[i] : nullptr));
    }
    chain.Submit();
    // Every in-place edit reads its neighborhood through the batch's one set of source clones.
    const auto view = std::make_shared<TopologyReadView>();
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (closures[i] && !edits[i].FinishClosures(tasks[i], *closures[i])) closures[i].reset();
        if (closures[i] && !TopologyCreatesRecord(tasks[i].Op)) view->Add(r, edits[i].StoreId, closures[i]->Neighborhood, chain.Scratch);
    }
    view->Clone(r);
    // Each boolean's intersection passes size its count entries and output in submits of their own.
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (closures[i] && tasks[i].Op == MeshTopologyOp::Boolean) closures[i]->Boolean.emplace(r, chain, tasks[i], closures[i]->Core, *closures[i]->Second);
    }
    std::vector<MeshTopologyEdit *> counted;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (closures[i]) {
            edits[i].PrepareCounts(r, tasks[i], *closures[i], view);
            counted.push_back(&edits[i]);
        }
    }
    // Every edit records its counts beside the others, and the iterating ones submit together to check convergence.
    // The edits whose label rounds have not converged record again from their prepare passes with twice the rounds.
    auto pending = counted;
    for (uint32_t rounds = LabelRounds; !pending.empty(); rounds *= 2u) {
        // Indirect label passes bind the scratch storage directly, so every upload of this submit is reserved before any records.
        uint64_t upload_words = 0u;
        for (const auto *edit : pending) upload_words += edit->Plan->Jobs.UploadWords();
        chain.Scratch.ReserveAdditional(upload_words);
        for (auto *edit : pending) EncodeTopologyCounts(r, chain, edit->Plan->Jobs, edit->Plan->Constants, edit->Plan->Iterates ? rounds : 0u);
        std::erase_if(pending, [](const MeshTopologyEdit *edit) { return !edit->Plan->Iterates; });
        if (pending.empty()) break;
        chain.Submit();
        std::erase_if(pending, [](const MeshTopologyEdit *edit) { return edit->Plan->Jobs.IndirectGroups(SrcHalfedges) == 0u; });
    }
    for (auto *edit : counted) {
        auto &plan = *edit->Plan;
        edit->Output = std::make_unique<TopologyOutputHandles>(r, chain, edit->SourceId, plan.Jobs.Jobs[0], plan.Bounds, plan.Jobs.ScratchBinding(), plan.Constants.Source, plan.Identity);
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
        auto &job = plan.Jobs.Jobs[0];
        edit->SourceTriangles = chain.Scratch.Allocate(edit->AddedTriangleCount);
        job.DstTriangleSources = {chain.Scratch.Buffer.Slot, edit->SourceTriangles.Offset};
        edit->NewVertices = meshes.InsertElements(edit->StoreId, D::Vertex, edit->Output->NewCounts[0], &chain.Scratch);
        plan.NewFaces = meshes.InsertElements(edit->StoreId, D::Face, edit->Output->NewCounts[1], &chain.Scratch);
        job.DstCornerOffset = meshes.InsertElements(edit->StoreId, D::Halfedge, counts.Halfedges, nullptr).First;
        job.DstTriangleOffset = edit->FirstTriangle = meshes.InsertElements(edit->StoreId, D::Triangle, edit->AddedTriangleCount, nullptr).First;
        // Every new edge contains an emitted halfedge, and a line contains two.
        plan.NewEdges = meshes.InsertElements(edit->StoreId, D::Edge, counts.Faces ? counts.Halfedges : counts.Halfedges / 2u, &chain.Scratch);
        edit->Output->Assign(r, chain, {edit->NewVertices, plan.NewFaces});
    }
    // Emission, incidence repair, edges, fans, edge attributes and the corner class counts share one submit.
    // New edges and fan items take bounds the host knows, then trim to the counts that submit reports.
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
        const auto &record = meshes.Get(edit->StoreId);
        const auto vertex_table = a.Vertices.Entries(record.Vertices), corner_table = a.FaceCorners.Entries(record.FaceCorners);
        const auto face_table = a.FaceTriangles.Entries(record.FaceData), edge_table = a.EdgeHalfedges.Entries(record.EdgeData);
        const auto inserted_vertices = InsertedBlocks(edit->NewVertices, chain.Scratch.Buffer, vertex_table), inserted_faces = InsertedBlocks(plan.NewFaces, chain.Scratch.Buffer, face_table);
        const auto vertex_blocks = fresh ? inserted_vertices : Union(WorkBlocks(chain.Scratch, plan.Core.Elements[0], 0u, vertex_table), inserted_vertices);
        const auto face_blocks = fresh ? inserted_faces : Union(WorkBlocks(chain.Scratch, plan.Core.Elements[2], 0u, face_table), inserted_faces);
        job.HasVertexPrimitives |= uint32_t(meshes.Get(edit->StoreId).VertexPrimitivesReady) << 1u;
        job.DstVertexHandles = {chain.Scratch.Buffer.Slot, edit->Output->Vertices.Offset};
        job.DstFaceHandles = {chain.Scratch.Buffer.Slot, edit->Output->Faces.Offset};
        job.DstConnectivity = meshes.GetConnectivityRef(edit->StoreId);
        plan.Constants.Destination.Layers = meshes.Layers(edit->StoreId);
        if (const auto &output = edit->SelectionOutput) {
            job.DstVertexBits = MaskBinding(output->Vertices);
            job.DstEdgeBits = MaskBinding(output->Edges);
            job.DstFaceBits = MaskBinding(output->Faces);
        }
        batch.SetDomainTiles(DstVertices, std::array{TileCount(counts.Vertices, TileElements)});
        batch.SetDomainTiles(DstHalfedges, std::array{TileCount(counts.Halfedges, TileElements)});
        batch.SetDomainTiles(DstFaces, std::array{TileCount(counts.Faces, TileElements)});
        UploadTopologyAttributes(meshes, edit->SourceId, edit->StoreId, plan.SourceAttributes, chain.Scratch, job);
        CaptureTopologyEmitWrites(r, edit->StoreId, job, edit->AddedTriangleCount, vertex_blocks, face_blocks, edit->SelectionOutput);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Vertex, vertex_blocks);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Face, face_blocks);
        const auto run_blocks = RunBlocks(corner_table, job.DstCornerOffset, counts.Halfedges);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Corner, run_blocks);
        emissions.push_back(batch.Encode(chain, pipelines, plan.Constants, TopologyEmissionPasses(job)));
        const auto &before = plan.Neighborhood;
        const std::array emitted{ElementHandleRange{.Handles = job.DstVertexHandles, .Count = counts.Vertices}, ElementHandleRange{.First = job.DstCornerOffset, .Count = counts.Halfedges}, ElementHandleRange{.Handles = job.DstFaceHandles, .Count = counts.Faces}};
        edit->Repair = std::make_unique<ConnectivityEditWork>(r, chain, edit->StoreId, before, std::array{plan.Core.Elements[1], plan.Core.Elements[2]}, emitted);
        // Emission replaces the core's loops inside the neighborhood, so the repaired counts are known before the pass runs.
        const std::array<uint32_t, 3> repaired = fresh ? std::array{counts.Vertices, counts.Halfedges, counts.Faces} : std::array{before.Counts[0] + edit->Output->NewCounts[0], before.Counts[1] - plan.Core.Counts[1] + counts.Halfedges, before.Counts[2] - plan.Core.Counts[2] + counts.Faces};
        edit->RetiredEdges = AllocateElementWork(chain.Scratch, a.EdgeHalfedges.HandleBound(record.EdgeData), before.Counts[3]);
        MeshConnectivityJob connectivity{
            .CornersSlot = a.FaceCorners.Buffer.Slot,
            .Connectivity = job.DstConnectivity,
            .Vertices = edit->Repair->Elements[0],
            .Halfedges = edit->Repair->Elements[1],
            .Faces = edit->Repair->Elements[2],
            .EdgeHandles = plan.NewEdges,
            .SourceConnectivity = job.SrcConnectivity,
            .SourceCornerSlot = plan.Constants.Source.CornerSlot,
            .SourceEdges = before.Elements[3],
            .ConvertedEdges = job.Op == MeshTopologyOp::AddPrimitives || TopologyMapsWireEdges(job.Op) ? plan.Core.Elements[3] : ElementWork{},
            .ConvertedWirePairs = TopologyMapsWireEdges(job.Op) ? SlotOffset{chain.Scratch.Buffer.Slot, plan.Jobs.ScratchBinding().Offset + job.WireEdgeMapOffset} : SlotOffset{},
            .SourceEdgeCount = before.Counts[3],
            .VertexCount = repaired[0],
            .HalfedgeCount = repaired[1],
            .FaceCount = repaired[2],
            .FaceStarts = 1u,
            .RetiredEdgeWork = edit->RetiredEdges,
        };
        auto &rebuilt = plan.Connectivity.emplace(chain.Scratch, LayoutConnectivityScratch(connectivity));
        rebuilt.Begin();
        AddConnectivityJob(rebuilt, connectivity);
        // Retained edges are neighborhood edges.
        auto edge_blocks = Union(WorkBlocks(chain.Scratch, before.Elements[3], 0u, edge_table), InsertedBlocks(plan.NewEdges, chain.Scratch.Buffer, edge_table));
        auto repaired_face_blocks = Union(WorkBlocks(chain.Scratch, before.Elements[2], 0u, face_table), face_blocks);
        CaptureConnectivityPrepareWrites(r, rebuilt.Jobs[0], vertex_blocks, Union(WorkBlocks(chain.Scratch, before.Elements[1], 0u, corner_table), run_blocks), repaired_face_blocks);
        CaptureTopologyEdgeWrites(r, edge_blocks, edit->SelectionOutput);
        meshes.CaptureAttributeWrites(edit->StoreId, AttributeDomain::Edge, edge_blocks);
        rebuilt.Encode(chain, pipelines, TiledJobPushConstants{}, ConnectivityPasses);
        fan_jobs.push_back(rebuilt.Jobs[0]);
        fan_vertex_blocks.insert(fan_vertex_blocks.end(), vertex_blocks.begin(), vertex_blocks.end());
        edit->RepairedBlocks = {std::move(edge_blocks), std::move(repaired_face_blocks)};
    }
    SortUnique(fan_vertex_blocks);
    EncodeVertexFans(r, chain, fan_jobs, fan_vertex_blocks);
    std::vector<MeshStore::CornerClassRequest> requests;
    requests.reserve(edits.size());
    for (uint32_t i = 0u; auto *edit : edits) {
        auto &plan = *edit->Plan;
        const auto &connected = plan.Connectivity->Jobs[0];
        plan.Jobs.Record(chain, pipelines, emissions[i++], plan.Constants, std::array{TiledPass{MeshPass::TopologyEdgeAttributes, DstHalfedges}});
        EncodeSortElementWork(r, chain, std::span{&edit->RetiredEdges, 1u});
        // The rebuilt fans contain every corner at the repaired vertices.
        requests.push_back({edit->StoreId, edit->Repair->Elements[0], connected.VertexCount, connected.HalfedgeCount});
    }
    auto classes = meshes.EncodeCornerClassification(r, chain, requests);
    {
        const profile::CpuScope stage{"TopologyEmitSubmit"};
        chain.Submit();
    }

    // Corner classes, base normals and authored normals complete the canonical edit in the chain's next submit.
    // The edits' corner classes plan as one batch before one derive records the base normals the authored normals rebase on.
    std::vector<LocalNormalWork> normals;
    normals.reserve(edits.size());
    for (auto *edit : edits) {
        const profile::CpuScope stage{"TopologyNormals"};
        auto &plan = *edit->Plan;
        const auto &connected = plan.Connectivity->Jobs[0];
        edit->Repair->Finish(chain);
        if (edit->Repair->Counts != std::array{connected.VertexCount, connected.HalfedgeCount, connected.FaceCount}) {
            throw std::logic_error("Connectivity repair disagrees with its emitted closure.");
        }
        meshes.TrimInsertedElements(edit->StoreId, D::Edge, plan.NewEdges, chain.Scratch, plan.Connectivity->ScratchSpan()[connected.StateOffset]);
        CheckElementWork(chain.Scratch, edit->RetiredEdges);
        normals.push_back({edit->StoreId, edit->Repair->Elements[0], edit->Repair->Counts[0], edit->Repair->Elements[2], edit->Repair->Counts[2]});
    }
    meshes.PlanCornerClassification(r, chain, classes);
    EncodeDeriveMeshNormals(r, chain, chain.Scratch, normals);
    for (auto *edit : edits) {
        auto &plan = *edit->Plan;
        auto &job = plan.Jobs.Jobs[0];
        if (job.CornerAttributes & MeshAttributeBit_Normal) {
            // The corner classification's plan can give the output its sector roots.
            plan.Constants.Destination.Layers = meshes.Layers(edit->StoreId);
            if (plan.Identity == TopologyIdentityPolicy::Preserve) {
                job.RetainedNormalCorners = plan.Neighborhood.Elements[1];
                job.RetainedNormalCornerCount = plan.Neighborhood.Counts[1];
            }
            plan.Jobs.SetDomainTiles(DstHalfedges, std::array{TileCount(std::max(job.DstHalfedgeCount, job.RetainedNormalCornerCount), TileElements)});
            CaptureTopologyNormalWrites(r, edit->StoreId, job, chain.Scratch);
            plan.Jobs.Encode(chain, pipelines, plan.Constants, std::array{TiledPass{MeshPass::TopologyCustomNormals, DstHalfedges}});
        }
        edit->AddedTriangles = AllocateElementWork(chain.Scratch, a.Triangles.HandleBound(meshes.Get(edit->StoreId).TriangleData), 1u);
        SeedElementWorkRanges(chain.Scratch, edit->AddedTriangles, std::array{Range{job.DstTriangleOffset, edit->AddedTriangleCount}}, 0u, false);
        edit->Published = true;
    }
}

void MeshTopologyEdit::FinishAll(state::Registry &r, std::span<MeshTopologyEdit *const> edits) {
    const profile::CpuScope scope{"FinishTopologyEdits"};
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<MeshStore::SelectionUpdate> updates;
    for (auto *edit : edits) {
        if (edit->Finished) throw std::logic_error("Topology edit was already finished.");
        edit->Finished = true;
        if (edit->Published && edit->Plan) {
            edit->Chain.Submit();
            edit->Plan.reset();
        }
        if (std::exchange(edit->OwnsSource, false)) meshes.Release(edit->SourceId);
        if (!edit->Output) continue;
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = edit->StoreId, .Source = edit->StoreSelectionSource});
        if (edit->StoreId != edit->SourceId) {
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
        update.Blocks[0].append_range(meshes.ArenaBlocks(edit->StoreId, MeshStore::ElementDomain::Vertex, storage, edit->Repair->Elements[0]));
        update.Blocks[1].insert(update.Blocks[1].end(), edit->RepairedBlocks[0].begin(), edit->RepairedBlocks[0].end());
        update.Blocks[2].insert(update.Blocks[2].end(), edit->RepairedBlocks[1].begin(), edit->RepairedBlocks[1].end());
    }
    if (!updates.empty()) meshes.UpdateSelection(r, edits.front()->Chain, updates);
}

GeometryTopologyResult MeshTopologyEdit::Result(const MeshStore &meshes) const {
    GeometryTopologyResult result{.GeometryId = StoreId, .Changed = bool(Output)};
    if (!Output) return result;
    if (!Published || !Plan) throw std::logic_error("Topology results must be read after publication and before finishing.");
    const auto handles = [&](ElementHandleRange range, std::vector<uint32_t> &out) {
        if (range.Handles.Slot == InvalidSlot) {
            out.resize(range.Count);
            std::ranges::iota(out, range.First);
        } else {
            const auto listed = Chain.Scratch.Get({range.Handles.Offset, range.Count});
            out.assign(listed.begin(), listed.end());
            SortUnique(out);
        }
    };
    handles(NewVertices, result.Created.Vertices);
    handles(Plan->NewFaces, result.Created.Faces);
    handles(Plan->NewEdges, result.Created.Edges);
    if (TopologyCreatesRecord(Op)) return result;
    const auto retained = [&](std::vector<uint32_t> emitted, std::span<const uint32_t> created, std::vector<uint32_t> &out) {
        SortUnique(emitted);
        std::ranges::set_difference(emitted, created, std::back_inserter(out));
    };
    const auto emitted = [&](Range range) { return std::vector<uint32_t>(std::from_range, Chain.Scratch.Get(range)); };
    retained(emitted({Output->Vertices.Offset, Plan->Counts.Vertices}), result.Created.Vertices, result.Retained.Vertices);
    retained(emitted({Output->Faces.Offset, Plan->Counts.Faces}), result.Created.Faces, result.Retained.Faces);
    const auto &job = Plan->Jobs.Jobs[0];
    const auto edges = meshes.Arenas().HalfedgeEdges.Buffer.GetSpan<uint32_t>();
    std::vector<uint32_t> emitted_edges;
    for (uint32_t h = job.DstCornerOffset; h < job.DstCornerOffset + job.DstHalfedgeCount; ++h) {
        if (const auto edge = edges[meshes.HandleAddress(StoreId, MeshStore::ElementDomain::Halfedge, ElementHandle{h}).Value()]; edge != InvalidOffset) emitted_edges.push_back(edge);
    }
    retained(std::move(emitted_edges), result.Created.Edges, result.Retained.Edges);
    return result;
}

std::vector<GeometryTopologyResult> ExecuteGeometryTopology(state::Registry &r, std::span<const MeshTopologyTask> tasks) {
    // Closures and counts read one common input state.
    // Fresh copies must emit before its one mutation overwrites that source's canonical values.
    std::set<uint32_t> mutated;
    for (const auto &task : tasks) {
        if (mutated.contains(task.SourceId)) {
            throw std::invalid_argument("Dependent topology tasks require ExecuteGeometryTopologyStages.");
        }
        if (!TopologyCreatesRecord(task.Op)) mutated.insert(task.SourceId);
    }
    const auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, tasks);
    MeshTopologyEdit::PublishAll(r, edits);
    std::vector<GeometryTopologyResult> results;
    std::vector<MeshTopologyEdit *> finished;
    for (auto &edit : edits) {
        results.push_back(edit.Result(meshes));
        finished.push_back(&edit);
    }
    MeshTopologyEdit::FinishAll(r, finished);
    chain.Submit();
    return results;
}

std::vector<GeometryTopologyResult> ExecuteGeometryTopologyStages(state::Registry &r, std::span<const MeshTopologyTask> tasks) {
    std::vector<GeometryTopologyResult> results;
    results.reserve(tasks.size());
    for (const auto &task : tasks) results.push_back(ExecuteGeometryTopology(r, std::span{&task, 1u}).front());
    return results;
}

namespace {
bool HasShortEdge(state::Registry &r, uint32_t id, float distance) {
    const auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto flag = chain.Scratch.Allocate(std::span<const uint32_t>{std::array{0u}});
    const ShortEdgesPushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .Flag = {chain.Scratch.Buffer.Slot, flag.Offset},
        .VertexSlot = a.Vertices.Buffer.Slot,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .HalfedgeCount = a.FaceCorners.Count(record.FaceCorners),
        .FaceCount = a.FaceTriangles.Count(record.FaceData),
        .Distance = distance,
    };
    chain.Threads(GetMeshPipelines(r)[MeshPass::FlagShortEdges], pc, pc.HalfedgeCount);
    chain.Submit();
    return chain.Scratch.Get({flag.Offset, 1u})[0] != 0u;
}
} // namespace

std::optional<uint32_t> ExecuteGeometryBoolean(state::Registry &r, const MeshTopologyTask &task) {
    const auto result = ExecuteGeometryTopology(r, std::span{&task, 1u}).front();
    if (!result.Changed) return {};
    // Edges shorter than the least normal float have coincident ends, and a degenerate dissolve merges each run of them into one vertex.
    constexpr auto Coincident = std::numeric_limits<float>::min();
    if (HasShortEdge(r, result.GeometryId, Coincident)) {
        const MeshTopologyTask cleanup{.SourceId = result.GeometryId, .Op = MeshTopologyOp::DissolveDegenerate, .Param0 = Coincident, .Flags = TopologyFlagSelectAll};
        ExecuteGeometryTopology(r, std::span{&cleanup, 1u});
    }
    return result.GeometryId;
}
