#include "mesh/SpatialFaceWork.h"

#include "Profile.h"
#include "gpu/SpatialFaceQueryPushConstants.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "state/Scene.h"

SpatialFaceWork::SpatialFaceWork(state::Scene &r, mtl::ComputeChain &chain, const MeshTopologyTask &task) {
    const profile::CpuScope scope{"SpatialFaceWork"};
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto *owner = buffers.TryMeshOf(task.SourceId);
    if (!owner || owner->MeshletRoot == InvalidOffset) throw std::invalid_argument("Spatial topology query needs current meshlet ownership.");
    if (owner->SpatialRoot == InvalidOffset) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &record = meshes.Get(task.SourceId);
    const auto &a = meshes.Arenas();
    const uint32_t mode = task.Flags & TopologyFlagScreenCuts ? 2u :
        task.Flags & TopologyFlagPlaneSide ? 1u : 0u;
    if (mode == 0u && !(task.Flags & TopologyFlagPlaneCuts)) throw std::invalid_argument("Spatial face query requires a plane or screen predicate.");
    const uint32_t group_bound = std::max(1u,owner->Level0Count/256u+(owner->Level0Count%256u!=0u));
    const uint32_t groups = std::min(16u,std::bit_ceil(group_bound));
    profile::RecordCounter("SpatialFaceQueryGroups", groups);
    auto &storage = chain.Scratch;
    const auto state = storage.Allocate(4u);
    std::ranges::fill(storage.GetMutable(state), 0u);
    SpatialFaceQueryPushConstants pc{
        .Meshlets = buffers.ActiveMeshlets.Ref(owner->MeshletRoot),
        .SpatialRoot = owner->SpatialRoot,
        .SpatialNodeSlot = buffers.MeshletSpatialNodes.Buffer.Slot,
        .SpatialNodeCapacity = buffers.MeshletSpatialNodes.Buffer.Count<MeshletSpatialNode>(),
        .SeedDepth = 8u+std::countr_zero(groups),
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .TriangleIdSlot = buffers.MeshletTriangleIds.Buffer.Slot,
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
        .TriangleIdCapacity = buffers.MeshletTriangleIds.Buffer.Count<uint32_t>(),
        .MeshletCapacity = buffers.Meshlets.Buffer.Count<MeshletRecord>(),
        .ResultSlot = storage.Buffer.Slot,
        .ResultOffset = state.Offset,
        .Mode = mode,
        .PlaneNormal = task.PlaneNormal,
        .PlaneOffset = task.PlaneOffset,
        .ScreenTransform = task.ScreenTransform,
        .Extent = task.Extent,
        .KnifeStart = task.KnifeStart,
        .KnifeEnd = task.KnifeEnd,
    };
    const auto &pipelines = GetMeshPipelines(r);
    const auto check = [&] {
        if (storage.Get({state.Offset + 1u, 1u})[0]) throw std::runtime_error("Spatial face query found invalid meshlet ownership or topology.");
    };
    {
        const profile::CpuScope stage{"SpatialFaceCount"};
        chain.Groups(pipelines[MeshPass::SpatialFaceQueryCount], pc, groups);
        chain.Submit();
        check();
    }
    const uint32_t triangle_bound = CandidateTriangles = storage.Get({state.Offset, 1u})[0];
    if (!triangle_bound) return;
    const uint32_t meshlet_count = CandidateMeshlets = storage.Get({state.Offset + 3u, 1u})[0];
    if (!meshlet_count) throw std::logic_error("Spatial face query counted triangles without meshlets.");
    std::ranges::fill(storage.GetMutable({state.Offset + 2u, 2u}), 0u);
    pc.CandidateCount = triangle_bound;
    const auto meshlet_candidates = storage.Allocate(meshlet_count);
    pc.MeshletCandidates = {storage.Buffer.Slot, meshlet_candidates.Offset};
    pc.MeshletCandidateCount = meshlet_count;
    Faces = AllocateElementWork(storage, a.FaceTriangles.Capacity(),
        std::min<uint64_t>(triangle_bound, record.FaceData ? a.FaceTriangles.Set(record.FaceData).BlockCount : 0u));
    pc.Faces = Faces;
    {
        const profile::CpuScope stage{"SpatialFaceGatherExpand"};
        chain.Groups(pipelines[MeshPass::SpatialFaceQueryGather], pc, groups);
        chain.Groups(pipelines[MeshPass::SpatialFaceQueryExpand], pc, meshlet_count);
        EncodeSortElementWork(r, chain, std::span{&Faces, 1u});
        chain.Submit();
        check();
    }
    if (storage.Get({state.Offset + 2u, 1u})[0] != triangle_bound ||
        storage.Get({state.Offset + 3u, 1u})[0] != meshlet_count)
        throw std::runtime_error("Spatial face gather disagrees with its candidate count.");
    Count = ElementWorkCount(storage, Faces);
}
