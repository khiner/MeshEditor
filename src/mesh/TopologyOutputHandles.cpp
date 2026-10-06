#include "mesh/TopologyOutputHandles.h"
#include "Profile.h"

#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "state/Scene.h"

TopologyOutputHandles::TopologyOutputHandles(state::Scene &r, mtl::ComputeChain &chain, const MeshTopologyJob &job, MeshStore::TopologyCounts bounds, SlotOffset scratch, const MeshTopologyArenas &source, TopologyIdentityPolicy policy) {
    const profile::CpuScope scope{"TopologyOutputHandles"};
    const bool preserve = policy == TopologyIdentityPolicy::Preserve;
    if (preserve && job.SrcFaceCount && source.FaceTriangleStartSlot == InvalidSlot) {
        throw std::invalid_argument("Topology identity planning requires source triangle ownership.");
    }
    const std::array output{bounds.Vertices, bounds.Faces};
    Vertices = chain.Scratch.Allocate(output[0]);
    Faces = chain.Scratch.Allocate(output[1]);
    const auto slot = chain.Scratch.Buffer.Slot;
    TopologyIdentityPushConstants pc{.Job = job, .Outputs = {SlotOffset{slot, Vertices.Offset}, SlotOffset{slot, Faces.Offset}}, .Error = {slot, 0u}};
    pc.Topology.StorageSlot = scratch.Slot;
    pc.Topology.ScratchOffset = scratch.Offset;
    pc.Topology.Source = source;
    for (uint32_t d = 0u; d < 2u; ++d) pc.NewElements[d] = New[d] = AllocateElementWork(chain.Scratch, output[d], WorkDomainBlocks(output[d]));
    if (preserve) {
        const auto &a = r.Context.get<MeshStore>().Arenas();
        pc.RetiredElements[0] = Retired[0] = AllocateElementWork(chain.Scratch, a.Vertices.Capacity(), job.SrcVertexCount);
        pc.RetiredElements[1] = Retired[1] = AllocateElementWork(chain.Scratch, a.FaceTriangles.Capacity(), job.SrcFaceCount);
        pc.ReplacedElements[0] = Replaced[0] = AllocateElementWork(chain.Scratch, a.FaceCorners.Capacity(), job.SrcHalfedgeCount);
        pc.ReplacedElements[1] = Replaced[1] = AllocateElementWork(chain.Scratch, a.Triangles.Capacity(), job.SrcHalfedgeCount);
    }
    const auto &pipelines = GetMeshPipelines(r);
    const auto groups = [](uint64_t count) { return uint32_t((count + 255u) / 256u); };
    const auto output_groups = groups(std::ranges::max(output));
    chain.Groups(pipelines[MeshPass::TopologyIdentityInit], pc, output_groups);
    if (preserve) {
        chain.Groups(pipelines[MeshPass::TopologyIdentityRetain], pc, groups(std::max(job.SrcVertexCount, job.SrcFaceCount)));
        chain.Groups(pipelines[MeshPass::TopologyIdentityReplaced], pc, groups(job.SrcHalfedgeCount));
    }
    chain.Groups(pipelines[MeshPass::TopologyIdentityNew], pc, output_groups, 256u, 2u);
    const std::array work{New[0], New[1], Retired[0], Retired[1], Replaced[0], Replaced[1]};
    EncodeSortElementWork(r, chain, std::span{work}.first(preserve ? work.size() : 2u));
}

void TopologyOutputHandles::Finish(const mtl::ComputeChain &chain) {
    for (uint32_t d = 0u; d < 2u; ++d) {
        NewCounts[d] = ElementWorkCount(chain.Scratch, New[d]);
        RetiredCounts[d] = ElementWorkCount(chain.Scratch, Retired[d]);
    }
}

void TopologyOutputHandles::Assign(state::Scene &r, mtl::ComputeChain &chain, std::array<ElementHandleRange, 2> inserted) const {
    for (uint32_t d = 0u; d < 2u; ++d)
        if (inserted[d].Count != NewCounts[d]) throw std::invalid_argument("Topology identity allocation count differs from its plan.");
    const auto slot = chain.Scratch.Buffer.Slot;
    const TopologyIdentityPushConstants pc{.Outputs = {SlotOffset{slot, Vertices.Offset}, SlotOffset{slot, Faces.Offset}}, .NewElements = {New[0], New[1]}, .Error = {slot, 0u}, .Inserted = {inserted[0], inserted[1]}};
    chain.Groups(GetMeshPipelines(r)[MeshPass::TopologyIdentityAssign], pc, (std::max(NewCounts[0], NewCounts[1]) + 255u) / 256u, 256u, 2u);
}
