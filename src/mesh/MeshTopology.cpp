#include "mesh/MeshTopology.h"

#include "numeric/VectorMath.h"

#include "Profile.h"
#include "Range.h"
#include "gpu/MeshTopologyJob.h"
#include "gpu/MeshTopologyPushConstants.h"
#include "mesh/Mesh.h"
#include "mesh/MeshConnectivityGpu.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TiledJobBatch.h"
#include "state/Scene.h"

#include <bit>
#include <unordered_map>

namespace {
// Scratch words per submit. A batch over this splits across submits.
constexpr uint32_t ScratchWordBudget{192u << 20};

enum Domain : uint32_t {
    TableEntries,
    SrcVertices,
    SrcHalfedges,
    SrcFaces,
    CountBlocks,
    Once,
    DstVertices,
    DstHalfedges,
    DstFaces,
    DstFanCorners,
    DstWords,
    CustomWords,
    CustomBlocks,
    DomainCount
};
using Batch = TiledJobBatch<MeshTopologyJob, DomainCount>;

// Label rounds one submit encodes before the host checks convergence.
constexpr uint32_t LabelRounds{16};

constexpr std::array PreparePasses{
    TiledPass{MeshPass::TopologyFaceIndex, SrcFaces},
    TiledPass{MeshPass::TopologyZero, SrcVertices},
    TiledPass{MeshPass::TopologyMarkHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyMarkFaces, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveLimitVertices, SrcVertices},
    TiledPass{MeshPass::TopologyMergeTable, TableEntries},
    TiledPass{MeshPass::TopologyMergeInsert, SrcVertices},
    TiledPass{MeshPass::TopologyMergeQuery, SrcVertices},
    TiledPass{MeshPass::TopologyListFill, SrcHalfedges},
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
constexpr std::array CountPasses{
    TiledPass{MeshPass::TopologyDissolveRegions, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveWalk, SrcFaces},
    TiledPass{MeshPass::TopologyDissolveRevert, SrcHalfedges},
    TiledPass{MeshPass::TopologyCountVertices, SrcVertices},
    TiledPass{MeshPass::TopologyCountHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyCountFaces, SrcFaces},
    TiledPass{MeshPass::TopologyScanBlockSum, CountBlocks, ScanCounts},
    TiledPass{MeshPass::TopologyScanBlockPrefix, PerJob, ScanCounts},
    TiledPass{MeshPass::TopologyScanOffsets, CountBlocks, ScanCounts},
};
constexpr std::array OutputPasses{
    TiledPass{MeshPass::TopologyZeroOutput, DstWords},
    TiledPass{MeshPass::TopologyZeroVertices, DstVertices},
    TiledPass{MeshPass::TopologyScatterVertices, SrcVertices},
    TiledPass{MeshPass::TopologyScatterHalfedges, SrcHalfedges},
    TiledPass{MeshPass::TopologyScatterFaces, SrcFaces},
    TiledPass{MeshPass::TopologyFaceTables, DstFaces},
    TiledPass{MeshPass::TopologyGatherVertices, DstVertices},
    TiledPass{MeshPass::TopologyGatherCorners, DstFanCorners},
    TiledPass{MeshPass::TopologyCustomPopcount, CustomWords},
    TiledPass{MeshPass::TopologyScanBlockSum, CustomBlocks, ScanCustomNormals},
    TiledPass{MeshPass::TopologyScanBlockPrefix, PerJob, ScanCustomNormals},
    TiledPass{MeshPass::TopologyScanOffsets, CustomBlocks, ScanCustomNormals},
    TiledPass{MeshPass::TopologyCustomPack, DstFanCorners},
};

struct Work {
    MeshTopologyTask Task;
    uint32_t OutputId{InvalidStoreId};
    MeshStore::TopologyCounts Bounds;
    bool HasCustomNormals;
    Range ListRange{};
};

// Upper bounds on the output elements an operator produces from its source counts, which size the scratch.
MeshStore::TopologyCounts Bounds(const MeshTopologyTask &task, const Mesh &mesh) {
    const auto V = mesh.VertexCount(), H = mesh.HalfEdgeCount(), F = mesh.FaceCount();
    switch (TopologyBaseOp(task.Op)) {
        case MeshTopologyOp::ExtrudeRegion: {
            const uint32_t s = task.Op == MeshTopologyOp::ExtrudeRegion ? task.Steps : 1u;
            return {V * (1 + s), H * (1 + s) + 4 * H * s, F * (1 + s) + H * s};
        }
        case MeshTopologyOp::DuplicateFaces:
        case MeshTopologyOp::SplitFaces: return {2 * V, 2 * H + 4 * H, 2 * F + H};
        case MeshTopologyOp::ExtrudeEdges: return {2 * V, H + 4 * H, F + H};
        case MeshTopologyOp::ExtrudeFacesIndividual: return {V + H, H + 4 * H, F + H};
        case MeshTopologyOp::Subdivide: {
            const uint32_t c = (task.Flags & (TopologyFlagPlaneCuts | TopologyFlagListCuts)) ? 1u : std::max(1u, uint32_t(task.Param0)), rows = (c + 1) * (c + 1);
            return {V + c * H + c * c * F, 4 * rows * F + H * (1 + 2 * c), rows * F + H * c + F};
        }
        case MeshTopologyOp::Triangulate:
        case MeshTopologyOp::Poke: return {V + F, 3 * H, H};
        case MeshTopologyOp::EdgeSplit: return {V + H, H, F};
        case MeshTopologyOp::AddFaces: {
            const uint32_t vertices = task.List[0], faces = task.List[1 + 3 * vertices];
            return {V + vertices, H + uint32_t(task.List.size()) - 2 - 3 * vertices - faces, F + faces};
        }
        case MeshTopologyOp::BevelEdges:
        case MeshTopologyOp::BevelVertices: {
            const uint32_t s = task.Op == MeshTopologyOp::BevelVertices ? 1u : std::clamp(uint32_t(task.Param1), 1u, 16u);
            return {V + H * (2 * s + 1), 6 * H + 4 * H * s, F + H * s + V};
        }
        case MeshTopologyOp::ConnectVertices: return {V, 3 * H, H};
        default: return {V, H, F};
    }
}

// Assigns the job's scratch runs from `base` and returns the words they span.
// Runs the operator does not use take no words.
uint32_t LayoutScratch(MeshTopologyJob &job, const Mesh &mesh, const MeshStore::TopologyCounts &bounds, bool custom, uint32_t base) {
    const auto op = job.Op;
    const auto V = mesh.VertexCount(), H = mesh.HalfEdgeCount(), F = mesh.FaceCount();
    uint32_t cursor = base;
    const auto take = [&](uint32_t words) { return std::exchange(cursor, cursor + words); };
    // Counts cover every source vertex, halfedge, and face, then the appended-face list, then the scan terminator.
    job.CountEntries = V + H + F + 2;
    job.CountBlockCount = TileCount(job.CountEntries, BlockElements);
    job.CustomWordCount = custom ? BitWords(bounds.Halfedges * 3) : 0u;
    job.CustomBlockCount = custom ? TileCount(job.CustomWordCount + 1, BlockElements) : 0u;
    // Size the table for a load factor under three quarters.
    const uint32_t table = op == MeshTopologyOp::MergeByDistance ? std::bit_ceil(V + V / 2 + 1) : 0u;
    job.TableMask = table > 0 ? table - 1 : 0u;
    job.StateOffset = take(2);
    job.SrcFaceOffset = mesh.GetConnectivity().Faces.empty() ? InvalidOffset : take(H);
    job.FlagVertexOffset = take(V);
    job.VertexTargetOffset = take(V);
    job.FlagHalfedgeOffset = take(H);
    job.FlagFaceOffset = take(F);
    job.CountsOffset = take(3 * job.CountEntries);
    job.CountBlockOffset = take(3 * job.CountBlockCount);
    job.VertexMapOffset = take(6 * bounds.Vertices);
    job.CornerMapOffset = take(8 * bounds.Halfedges);
    job.FaceMapOffset = take(bounds.Faces);
    job.LabelOffset = take(TopologyIterates(op) ? 4 * F + 2 * V : 0u);
    job.HalfedgeAuxOffset = take(op == MeshTopologyOp::EdgeSplit || (job.Flags & (TopologyFlagListCuts | TopologyFlagListSelects | TopologyFlagScreenCuts)) ? H : 0u);
    job.TableOffset = take(table);
    job.VertexOverrideOffset = take(op == MeshTopologyOp::MergeCollapse ? 4 * V : 0u);
    job.VertexInwardOffset = take(TopologyDisplaces(op, job.Flags) ? 6 * bounds.Vertices : 0u);
    job.CustomPopcountOffset = take(custom ? job.CustomWordCount + 1 : 0u);
    job.CustomBlockOffset = take(job.CustomBlockCount);
    return cursor - base;
}

uint32_t ScratchWords(const Work &work, const MeshStore &meshes) {
    MeshTopologyJob probe{.Op = work.Task.Op, .Flags = work.Task.Flags};
    return LayoutScratch(probe, Mesh{meshes, work.Task.SourceId}, work.Bounds, work.HasCustomNormals, 0);
}

MeshTopologyPushConstants PushConstants(const MeshStore &meshes) {
    const auto &a = meshes.Arenas();
    return {
        .VertexSlot = a.Vertices.Buffer.Slot,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .ConnectivitySlot = a.Connectivity.Buffer.Slot,
        .SelectionBitsSlot = a.SelectionBits.Buffer.Slot,
        .FaceFirstTriangleSlot = a.FaceFirstTriangles.Buffer.Slot,
        .TriangleFaceIdSlot = a.TriangleFaceIds.Buffer.Slot,
        .EdgeSharpnessSlot = a.EdgeSharpness.Buffer.Slot,
        .FaceSharpnessSlot = a.FaceSharpness.Buffer.Slot,
        .ElementPrimitiveSlot = a.ElementPrimitives.Buffer.Slot,
        .BoneDeformSlot = a.BoneDeform.Buffer.Slot,
        .MorphTargetSlot = a.MorphTargets.Buffer.Slot,
        .CornerTangentSlot = a.CornerTangents.Buffer.Slot,
        .CornerColorSlot = a.CornerColors.Buffer.Slot,
        .CornerUvSlot = a.CornerUvs.Buffer.Slot,
        .CustomCornerMaskSlot = a.CustomCornerMasks.Buffer.Slot,
        .CustomCornerNormalSlot = a.CustomCornerNormals.Buffer.Slot,
        .AdjacencySlot = a.Adjacency.Buffer.Slot,
        .BaseVertexNormalSlot = a.BaseVertexNormals.Buffer.Slot,
        .BaseFaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
        .ListSlot = a.Lists.Buffer.Slot,
    };
}

// Adds the job with its source and scratch, and no output yet.
void AddJob(Batch &batch, const MeshStore &meshes, const Work &work) {
    const Mesh mesh{meshes, work.Task.SourceId};
    const auto &src = meshes.Get(work.Task.SourceId);
    const auto V = mesh.VertexCount(), H = mesh.HalfEdgeCount(), F = mesh.FaceCount();
    MeshTopologyJob job{
        .Op = work.Task.Op,
        .Flags = work.Task.Flags,
        .Steps = work.Task.Steps,
        .Param0 = work.Task.Param0,
        .Param1 = work.Task.Param1,
        .TargetVertex = work.Task.TargetVertex,
        .TargetPosition = work.Task.TargetPosition,
        .CopyRotation = work.Task.CopyRotation,
        .CopyTranslation = work.Task.CopyTranslation,
        .PlaneNormal = work.Task.PlaneNormal,
        .PlaneOffset = work.Task.PlaneOffset,
        .ScreenTransform = work.Task.ScreenTransform,
        .Extent = work.Task.Extent,
        .KnifeStart = work.Task.KnifeStart,
        .KnifeEnd = work.Task.KnifeEnd,
        .SrcVertexOffset = src.Vertices.Offset,
        .SrcCornerOffset = src.FaceCorners.Offset,
        .SrcConnectivityOffset = src.Connectivity.Offset,
        .SrcVertexCount = V,
        .SrcHalfedgeCount = H,
        .SrcFaceCount = F,
        .SrcEdgeCount = mesh.EdgeCount(),
        .SrcFaceStarts = src.ConnectivityFaceStarts ? 1u : 0u,
        .SrcVertexBitsOffset = src.SelectionBits[0].Offset,
        .SrcEdgeBitsOffset = src.SelectionBits[1].Offset,
        .SrcFaceBitsOffset = src.SelectionBits[2].Offset,
        .SrcFaceFirstTriangleOffset = src.FaceData.Offset,
        .SrcEdgeSharpnessOffset = src.EdgeSharpness.Offset,
        .SrcElementPrimitiveOffset = src.ElementPrimitives.Offset,
        .SrcBoneDeformOffset = OffsetOrInvalid(src.BoneDeform),
        .SrcMorphTargetOffset = OffsetOrInvalid(src.MorphTargets),
        .SrcCornerTangentOffset = OffsetOrInvalid(src.CornerTangents),
        .SrcCornerColorOffset = OffsetOrInvalid(src.CornerColors),
        .SrcCornerUvOffsets = {OffsetOrInvalid(src.CornerUvs[0]), OffsetOrInvalid(src.CornerUvs[1]), OffsetOrInvalid(src.CornerUvs[2]), OffsetOrInvalid(src.CornerUvs[3])},
        .SrcCustomCornerMaskOffset = OffsetOrInvalid(src.CustomCornerMasks),
        .SrcCustomCornerNormalOffset = OffsetOrInvalid(src.CustomCornerNormals),
        .SrcFanAdjacencyOffset = OffsetOrInvalid(meshes.GetDerived(work.Task.SourceId).VertexFanAdjacency),
        .ListOffset = OffsetOrInvalid(work.ListRange),
        .MorphTargetCount = src.MorphTargetCount,
    };
    batch.AllocateScratch(LayoutScratch(job, mesh, work.Bounds, work.HasCustomNormals, batch.ScratchWords));
    batch.AddJob(job, {
                          TileCount(job.Op == MeshTopologyOp::MergeByDistance ? job.TableMask + 1 : 0u, TileElements),
                          TileCount(V, TileElements),
                          TileCount(H, TileElements),
                          TileCount(F + 2, TileElements),
                          3 * job.CountBlockCount,
                          batch.Jobs.empty() ? 1u : 0u,
                      });
}

// Points the job at its output, allocated at the scanned counts.
void SetOutput(MeshTopologyJob &job, const MeshStore::Record &dst) {
    job.DstVertexOffset = dst.Vertices.Offset;
    job.DstCornerOffset = dst.FaceCorners.Offset;
    job.DstConnectivityOffset = dst.Connectivity.Offset;
    job.DstVertexCount = dst.ConnectivityVertices;
    job.DstHalfedgeCount = dst.ConnectivityHalfedges;
    job.DstFaceCount = dst.ConnectivityFaces;
    job.DstFaceStarts = dst.ConnectivityFaceStarts ? 1u : 0u;
    job.DstVertexBitsOffset = dst.SelectionBits[0].Offset;
    job.DstEdgeBitsOffset = dst.SelectionBits[1].Offset;
    job.DstFaceBitsOffset = dst.SelectionBits[2].Offset;
    job.DstFaceFirstTriangleOffset = dst.FaceData.Offset;
    job.DstTriangleFaceIdOffset = dst.TriangleFaceIds.Offset;
    job.DstEdgeSharpnessOffset = dst.EdgeSharpness.Offset;
    job.DstElementPrimitiveOffset = dst.ElementPrimitives.Offset;
    job.DstBoneDeformOffset = OffsetOrInvalid(dst.BoneDeform);
    job.DstMorphTargetOffset = OffsetOrInvalid(dst.MorphTargets);
    job.DstCornerTangentOffset = OffsetOrInvalid(dst.CornerTangents);
    job.DstCornerColorOffset = OffsetOrInvalid(dst.CornerColors);
    job.DstCornerUvOffsets = {OffsetOrInvalid(dst.CornerUvs[0]), OffsetOrInvalid(dst.CornerUvs[1]), OffsetOrInvalid(dst.CornerUvs[2]), OffsetOrInvalid(dst.CornerUvs[3])};
    job.DstCustomCornerMaskOffset = OffsetOrInvalid(dst.CustomCornerMasks);
    job.DstCustomCornerNormalOffset = OffsetOrInvalid(dst.CustomCornerNormals);
    job.CustomWordCount = dst.CustomCornerMasks.Count > 0 ? BitWords(dst.TriangleCount * 3) : 0u;
    job.CustomBlockCount = dst.CustomCornerMasks.Count > 0 ? TileCount(job.CustomWordCount + 1, BlockElements) : 0u;
}

void SubmitChunk(state::Scene &r, std::span<Work> chunk, Batch &batch) {
    auto &meshes = r.ctx().get<MeshStore>();
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = GetMeshPipelines(r);
    for (auto &work : chunk) {
        if (!work.Task.List.empty()) work.ListRange = meshes.AllocateList(work.Task.List);
    }
    batch.Begin();
    for (const auto &work : chunk) AddJob(batch, meshes, work);
    const auto push_constants = PushConstants(meshes);
    const auto scratch = batch.ScratchSpan();
    const bool iterates = std::ranges::any_of(chunk, [](const Work &work) { return TopologyIterates(work.Task.Op); });
    // A chunk whose labels still change in the last round runs again from the start with twice the rounds.
    for (uint32_t rounds = LabelRounds;; rounds *= 2) {
        std::vector<TiledPass> passes(PreparePasses.begin(), PreparePasses.end());
        for (uint32_t round = 0; iterates && round < rounds; ++round) {
            passes.insert(passes.end(), LabelPasses.begin(), LabelPasses.end());
            passes.push_back({MeshPass::TopologyConverge, Once, (uint32_t(chunk.size()) << 8) | DomainCount});
        }
        passes.insert(passes.end(), CountPasses.begin(), CountPasses.end());
        batch.Submit(ctx, slots, pipelines, push_constants, passes);
        if (!iterates || batch.IndirectGroups(SrcHalfedges) == 0) break;
    }

    std::vector<uint32_t> outputs;
    for (uint32_t i = 0; i < chunk.size(); ++i) {
        auto &job = batch.Jobs[i];
        const auto total = [&](uint32_t quantity) { return scratch[job.CountsOffset + quantity * job.CountEntries + job.CountEntries - 1]; };
        chunk[i].OutputId = meshes.BeginTopologyOutput(chunk[i].Task.SourceId, {.Vertices = total(0), .Halfedges = total(2), .Faces = total(1)});
        SetOutput(job, meshes.Get(chunk[i].OutputId));
        outputs.push_back(chunk[i].OutputId);
    }
    // A collapse places each merged run at its center, summed here from the final targets.
    for (uint32_t i = 0; i < chunk.size(); ++i) {
        if (chunk[i].Task.Op != MeshTopologyOp::MergeCollapse) continue;
        const auto &job = batch.Jobs[i];
        const Mesh mesh{meshes, chunk[i].Task.SourceId};
        const auto bits = meshes.GetSelectionBits(chunk[i].Task.SourceId, Element::Vertex);
        std::unordered_map<uint32_t, std::pair<vec3, uint32_t>> sums;
        ForEachSelected(bits, mesh.VertexCount(), [&](uint32_t v) {
            auto &[sum, count] = sums[scratch[job.VertexTargetOffset + v]];
            sum += mesh.GetPosition(Mesh::VH{v});
            ++count;
        });
        for (const auto &[root, sum] : sums) {
            const auto center = sum.first / float(sum.second);
            auto *override = scratch.data() + job.VertexOverrideOffset + 4 * root;
            override[0] = 1u;
            override[1] = std::bit_cast<uint32_t>(center.x);
            override[2] = std::bit_cast<uint32_t>(center.y);
            override[3] = std::bit_cast<uint32_t>(center.z);
        }
    }
    const auto retile = [&](uint32_t domain, auto &&tiles_of) {
        std::vector<uint32_t> tiles;
        for (const auto &job : batch.Jobs) tiles.push_back(tiles_of(job));
        batch.SetDomainTiles(domain, tiles);
    };
    retile(DstVertices, [](const MeshTopologyJob &job) { return TileCount(job.DstVertexCount, TileElements); });
    retile(DstHalfedges, [](const MeshTopologyJob &job) { return TileCount(job.DstHalfedgeCount, TileElements); });
    retile(DstFaces, [](const MeshTopologyJob &job) { return TileCount(job.DstFaceCount, TileElements); });
    retile(DstFanCorners, [](const MeshTopologyJob &job) { return TileCount(3 * (job.DstHalfedgeCount - 2 * job.DstFaceCount), TileElements); });
    retile(DstWords, [](const MeshTopologyJob &job) {
        return TileCount(std::max({BitWords(job.DstVertexCount), BitWords(job.DstHalfedgeCount), BitWords(job.DstFaceCount), job.CustomWordCount}), TileElements);
    });
    retile(CustomWords, [](const MeshTopologyJob &job) { return job.CustomBlockCount > 0 ? TileCount(job.CustomWordCount + 1, TileElements) : 0u; });
    retile(CustomBlocks, [](const MeshTopologyJob &job) { return job.CustomBlockCount; });
    // The output passes, the outputs' connectivity builds, and the edge pass share one command buffer and one wait.
    // The edge pass reads the output connectivity's edge numbering through the same scratch maps.
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    batch.Encode(slots, pipelines, push_constants, OutputPasses, encoder);
    auto pending = EncodeConnectivity(r, outputs, encoder);
    batch.Encode(slots, pipelines, push_constants, std::array{TiledPass{MeshPass::TopologyEdgeAttributes, DstHalfedges}}, encoder);
    encoder->endEncoding();
    // The connectivity chunks' buffers are allocated during encoding, so residency commits after it.
    ctx.CommitResidency();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    FinishConnectivity(r, pending);
    for (uint32_t i = 0; i < chunk.size(); ++i) {
        const auto &job = batch.Jobs[i];
        meshes.CompleteTopologyOutput(chunk[i].OutputId, job.CustomBlockCount > 0 ? scratch[job.CustomPopcountOffset + job.CustomWordCount] : 0u);
        meshes.ReleaseList(chunk[i].ListRange);
    }
}
} // namespace

std::vector<uint32_t> RunMeshTopology(state::Scene &r, std::span<const MeshTopologyTask> tasks) {
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<uint32_t> outputs(tasks.size(), InvalidStoreId);
    std::vector<Work> work;
    std::vector<size_t> task_of;
    for (size_t i = 0; i < tasks.size(); ++i) {
        const Mesh mesh{meshes, tasks[i].SourceId};
        if (mesh.FaceCount() == 0) continue;
        meshes.EnsureSelectionBits(mesh);
        work.push_back({.Task = tasks[i], .Bounds = Bounds(tasks[i], mesh), .HasCustomNormals = meshes.Get(tasks[i].SourceId).CustomCornerMasks.Count > 0});
        task_of.push_back(i);
    }
    if (work.empty()) return outputs;
    const profile::CpuScope scope{"MeshTopology"};
    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) { return ScratchWords(work[i], meshes); });
    Batch batch{meshes.BufferContext(), split.WidestWords, split.MostJobs};
    for (const auto chunk : split.Chunks) SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count), batch);
    for (size_t i = 0; i < work.size(); ++i) outputs[task_of[i]] = work[i].OutputId;
    return outputs;
}
