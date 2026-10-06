#include "mesh/SpatialFaceWork.h"

#include "Profile.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "metal/Dispatch.h"
#include "numeric/VectorMath.h"
#include "render/ElementWorkOps.h"
#include "state/Scene.h"

namespace {
void CheckSpatialState(const mtl::ComputeChain &chain, Range state) {
    if (chain.Scratch.Get({state.Offset + 1u, 1u})[0]) throw std::runtime_error("Spatial face query found invalid canonical topology.");
}
// Conservative AABB predicates; the GPU retains the exact face/edge test.
bool Overlaps(const MeshTopologyTask &task, const AABB &bounds) {
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        if (!(bounds.Min[axis] <= bounds.Max[axis])) throw std::logic_error("Canonical face bounds are missing.");
    if (!(task.Flags & TopologyFlagScreenCuts)) {
        const auto center = (bounds.Min + bounds.Max) * .5f;
        const auto extent = (bounds.Max - bounds.Min) * .5f;
        const auto distance = Dot(task.PlaneNormal, center) - task.PlaneOffset;
        float radius = 0.f;
        for (uint32_t axis = 0u; axis < 3u; ++axis) radius += std::abs(task.PlaneNormal[axis]) * extent[axis];
        const float margin = 1e-5f * (1.f + Length(extent));
        return task.Flags & TopologyFlagPlaneSide ? distance < radius + margin : std::abs(distance) <= radius + margin;
    }
    vec2 low{std::numeric_limits<float>::max()}, high{-std::numeric_limits<float>::max()};
    for (uint32_t corner = 0u; corner < 8u; ++corner) {
        const vec4 p{corner & 1u ? bounds.Max.x : bounds.Min.x, corner & 2u ? bounds.Max.y : bounds.Min.y, corner & 4u ? bounds.Max.z : bounds.Min.z, 1.f};
        vec4 clip{};
        for (uint32_t axis = 0u; axis < 4u; ++axis) clip += task.ScreenTransform.Columns[axis] * p[axis];
        if (clip.w <= 1e-5f) return true;
        const vec2 pixel{(clip.x / clip.w + 1.f) * .5f * task.Extent.x, (1.f - clip.y / clip.w) * .5f * task.Extent.y};
        low = Min(low, pixel);
        high = Max(high, pixel);
    }
    const auto delta = task.KnifeEnd - task.KnifeStart;
    float enter = 0.f, leave = 1.f;
    for (uint32_t axis = 0u; axis < 2u; ++axis) {
        if (std::abs(delta[axis]) < 1e-12f) {
            if (task.KnifeStart[axis] < low[axis] - 1e-3f || task.KnifeStart[axis] > high[axis] + 1e-3f) return false;
        } else {
            auto near = (low[axis] - 1e-3f - task.KnifeStart[axis]) / delta[axis];
            auto far = (high[axis] + 1e-3f - task.KnifeStart[axis]) / delta[axis];
            if (near > far) std::swap(near, far);
            enter = std::max(enter, near);
            leave = std::min(leave, far);
            if (enter > leave) return false;
        }
    }
    return true;
}
} // namespace

SpatialFaceWork::SpatialFaceWork(state::Scene &r, mtl::ComputeChain &chain, const MeshTopologyTask &task) {
    const profile::CpuScope scope{"SpatialFaceCount"};
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &record = meshes.Get(task.SourceId);
    const auto &a = meshes.Arenas();
    const uint32_t mode = task.Flags & TopologyFlagScreenCuts ? 2u :
        task.Flags & TopologyFlagPlaneSide                    ? 1u :
                                                                0u;
    if (mode == 0u && !(task.Flags & TopologyFlagPlaneCuts)) throw std::invalid_argument("Spatial face query requires a plane or screen predicate.");
    ElementWork candidates;
    if (!(task.Flags & TopologyFlagSelectAll)) {
        const auto selected = ListSeed(r, chain, task.SourceId, Element::Face, task.Selection.Faces);
        candidates = selected.Work;
        CandidateCount = selected.Count;
        CandidateBlocks = WorkBlockCount(chain.Scratch, candidates);
    } else if (record.FaceData && a.FaceTriangles.Count(record.FaceData)) {
        std::vector<uint32_t> blocks;
        VisitedNodes = a.SelectionTree.VisitBounds(3u * task.SourceId + 2u, a.FaceAggregates.Buffer.GetSpan<SelectionAggregate>(), [&](const AABB &bounds) { return Overlaps(task, bounds); }, [&](uint32_t block) { blocks.push_back(block); CandidateCount += a.FaceTriangles.Blocks.Get({block, 1u})[0].Count; });
        if (!VisitedNodes) throw std::logic_error("Spatial face query requires canonical face aggregates.");
        CandidateBlocks = uint32_t(blocks.size());
        candidates = AllocateElementWork(chain.Scratch, a.FaceTriangles.Capacity(), blocks.size());
        const auto seed = PrepareBlockMembershipWork(chain.Scratch, a.FaceTriangles, record.FaceData, blocks, candidates);
        EncodeElementMembershipWork(r, chain, std::span{&seed, 1u});
        EncodeSortElementWork(r, chain, std::span{&candidates, 1u});
    }
    profile::RecordCounter("SpatialFaceCandidateNodes", VisitedNodes);
    profile::RecordCounter("SpatialFaceCandidateBlocks", CandidateBlocks);
    profile::RecordCounter("SpatialFaceCandidates", CandidateCount);
    auto &storage = chain.Scratch;
    State = storage.Allocate(2u);
    std::ranges::fill(storage.GetMutable(State), 0u);
    Faces = AllocateElementWork(storage, a.FaceTriangles.Capacity(), std::min(CandidateCount, record.FaceData ? a.FaceTriangles.Set(record.FaceData).BlockCount : 0u));
    Pc = {
        .Candidates = candidates,
        .Faces = Faces,
        .CandidateCount = CandidateCount,
        .FaceRangeSlot = a.FaceRanges.Buffer.Slot,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .VertexSlot = a.Vertices.Buffer.Slot,
        .FaceRangeCapacity = a.FaceRanges.Buffer.Count<uvec2>(),
        .CornerCapacity = a.FaceCorners.Capacity(),
        .VertexCapacity = a.Vertices.Capacity(),
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
}

void SpatialFaceWork::RecordFaces(state::Scene &r, mtl::ComputeChain &chain) {
    if (!CandidateCount) return;
    chain.Groups(GetMeshPipelines(r)[MeshPass::SpatialFaceQueryExpand], Pc, (CandidateCount + 255u) / 256u);
    EncodeSortElementWork(r, chain, std::span{&Faces, 1u});
    chain.AfterSubmit([this, &chain] {
        CheckSpatialState(chain, State);
        Count = ElementWorkCount(chain.Scratch, Faces);
    });
}
