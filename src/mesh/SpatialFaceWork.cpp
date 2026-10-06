#include "mesh/SpatialFaceWork.h"

#include "Profile.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "state/Scene.h"

namespace {
void CheckSpatialState(const mtl::ComputeChain &chain, Range state) {
    if (chain.Scratch.Get({state.Offset + 1u, 1u})[0]) throw std::runtime_error("Spatial face query found invalid meshlet ownership or topology.");
}
} // namespace

SpatialFaceWork::SpatialFaceWork(state::Scene &r, mtl::ComputeChain &chain, const MeshTopologyTask &task) {
    const profile::CpuScope scope{"SpatialFaceCount"};
    const auto &meshes = r.Context.get<const MeshStore>();
    auto &render = meshes.Render();
    const auto *owner = meshes.TryGet(task.SourceId);
    if (!owner || owner->MeshletRoot == InvalidOffset) throw std::invalid_argument("Spatial topology query needs current meshlet ownership.");
    if (owner->SpatialRoot == InvalidOffset) return;
    const auto &record = meshes.Get(task.SourceId);
    const auto &a = meshes.Arenas();
    const uint32_t mode = task.Flags & TopologyFlagScreenCuts ? 2u :
        task.Flags & TopologyFlagPlaneSide                    ? 1u :
                                                                0u;
    if (mode == 0u && !(task.Flags & TopologyFlagPlaneCuts)) throw std::invalid_argument("Spatial face query requires a plane or screen predicate.");
    const uint32_t group_bound = std::max(1u, owner->Level0Count / 256u + (owner->Level0Count % 256u != 0u));
    Groups = std::min(16u, std::bit_ceil(group_bound));
    profile::RecordCounter("SpatialFaceQueryGroups", Groups);
    FaceBlocks = record.FaceData ? a.FaceTriangles.Set(record.FaceData).BlockCount : 0u;
    auto &storage = chain.Scratch;
    State = storage.Allocate(4u);
    std::ranges::fill(storage.GetMutable(State), 0u);
    Pc = {
        .Meshlets = render.ActiveMeshlets.Ref(owner->MeshletRoot),
        .SpatialRoot = owner->SpatialRoot,
        .SpatialNodeSlot = render.MeshletSpatialNodes.Buffer.Slot,
        .SpatialNodeCapacity = render.MeshletSpatialNodes.Buffer.Count<MeshletSpatialNode>(),
        .SeedDepth = 8u + std::countr_zero(Groups),
        .MeshletSlot = render.Meshlets.Buffer.Slot,
        .TriangleIdSlot = render.MeshletTriangleIds.Buffer.Slot,
        .TriangleSlot = a.Triangles.Buffer.Slot,
        .HalfedgeFaceSlot = a.HalfedgeFaces.Buffer.Slot,
        .FaceRangeSlot = a.FaceRanges.Buffer.Slot,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .VertexSlot = a.Vertices.Buffer.Slot,
        .FaceBlockSlot = a.FaceTriangles.Blocks.Buffer.Slot,
        .FaceOwner = record.FaceData.Index,
        .FaceCapacity = a.FaceTriangles.Capacity(),
        .FaceRangeCapacity = a.FaceRanges.Buffer.Count<uvec2>(),
        .TriangleCapacity = a.Triangles.Capacity(),
        .CornerCapacity = a.FaceCorners.Capacity(),
        .VertexCapacity = a.Vertices.Capacity(),
        .TriangleIdCapacity = render.MeshletTriangleIds.Buffer.Count<uint32_t>(),
        .MeshletCapacity = render.Meshlets.Buffer.Count<MeshletRecord>(),
        .ResultSlot = storage.Buffer.Slot,
        .ResultOffset = State.Offset,
        .Mode = mode,
        .PlaneNormal = task.PlaneNormal,
        .PlaneOffset = task.PlaneOffset,
        .ScreenTransform = task.ScreenTransform,
        .Extent = task.Extent,
        .KnifeStart = task.KnifeStart,
        .KnifeEnd = task.KnifeEnd,
    };
    chain.Groups(GetMeshPipelines(r)[MeshPass::SpatialFaceQueryCount], Pc, Groups);
}

void SpatialFaceWork::RecordFaces(state::Scene &r, mtl::ComputeChain &chain) {
    const profile::CpuScope scope{"SpatialFaceGatherExpand"};
    auto &storage = chain.Scratch;
    CheckSpatialState(chain, State);
    const uint32_t triangle_bound = CandidateTriangles = storage.Get({State.Offset, 1u})[0];
    if (!triangle_bound) return;
    const uint32_t meshlet_count = CandidateMeshlets = storage.Get({State.Offset + 3u, 1u})[0];
    if (!meshlet_count) throw std::logic_error("Spatial face query counted triangles without meshlets.");
    std::ranges::fill(storage.GetMutable({State.Offset + 2u, 2u}), 0u);
    Pc.CandidateCount = triangle_bound;
    const auto meshlet_candidates = storage.Allocate(meshlet_count);
    Pc.MeshletCandidates = {storage.Buffer.Slot, meshlet_candidates.Offset};
    Pc.MeshletCandidateCount = meshlet_count;
    Faces = AllocateElementWork(storage, r.Context.get<const MeshStore>().Arenas().FaceTriangles.Capacity(), std::min<uint64_t>(triangle_bound, FaceBlocks));
    Pc.Faces = Faces;
    const auto &pipelines = GetMeshPipelines(r);
    chain.Groups(pipelines[MeshPass::SpatialFaceQueryGather], Pc, Groups);
    chain.Groups(pipelines[MeshPass::SpatialFaceQueryExpand], Pc, meshlet_count);
    EncodeSortElementWork(r, chain, std::span{&Faces, 1u});
    chain.AfterSubmit([this, &chain] {
        const auto &storage = chain.Scratch;
        CheckSpatialState(chain, State);
        if (storage.Get({State.Offset + 2u, 1u})[0] != CandidateTriangles ||
            storage.Get({State.Offset + 3u, 1u})[0] != CandidateMeshlets)
            throw std::runtime_error("Spatial face gather disagrees with its candidate count.");
        Count = ElementWorkCount(storage, Faces);
    });
}
