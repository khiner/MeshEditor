#include "mesh/ConnectivityEditWork.h"
#include "Profile.h"

#include "gpu/ConnectivityEditPushConstants.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "state/Scene.h"

ConnectivityEditWork::ConnectivityEditWork(state::Scene &r, mtl::ComputeChain &chain, const MeshClosure &before, const std::array<ElementWork, 2> &replaced, const std::array<ElementHandleRange, 3> &emitted) {
    const profile::CpuScope scope{"ConnectivityEditWork"};
    const auto &a = r.Context.get<MeshStore>().Arenas();
    const std::array capacities{a.Vertices.Capacity(), a.FaceCorners.Capacity(), a.FaceTriangles.Capacity()};
    uint32_t count = 0u;
    for (uint32_t d = 0u; d < 3u; ++d) {
        const uint64_t bound = uint64_t(before.Counts[d]) + emitted[d].Count;
        if (bound > UINT32_MAX) throw std::length_error("Connectivity edit work exceeds its dispatch address space.");
        if (emitted[d].Count && emitted[d].Handles.Slot == InvalidSlot && uint64_t(emitted[d].First) + emitted[d].Count > capacities[d]) {
            throw std::out_of_range("Emitted connectivity run exceeds its canonical arena.");
        }
        if (emitted[d].Handles.Slot != InvalidSlot && uint64_t(emitted[d].Handles.Offset) + emitted[d].Count > UINT32_MAX) {
            throw std::length_error("Emitted connectivity handle list exceeds its address space.");
        }
        count = std::max(count, uint32_t(bound));
    }
    if (!count) return;
    ConnectivityEditPushConstants pc{
        .Before = {before.Elements[0], before.Elements[1], before.Elements[2]},
        .Counts = {before.Counts[0], before.Counts[1], before.Counts[2]},
        .Replaced = replaced,
        .Emitted = emitted,
        .ErrorSlot = chain.Scratch.Buffer.Slot,
    };
    for (uint32_t d = 0u; d < 3u; ++d)
        pc.After[d] = Elements[d] = AllocateElementWork(chain.Scratch, capacities[d], uint64_t(before.Counts[d]) + emitted[d].Count);
    chain.Groups(GetMeshPipelines(r)[MeshPass::ConnectivityEditWork], pc, (count + 255u) / 256u, 256u, 3u);
    EncodeSortElementWork(r, chain, Elements);
}

void ConnectivityEditWork::Finish(const mtl::ComputeChain &chain) {
    for (uint32_t d = 0u; d < 3u; ++d) Counts[d] = ElementWorkCount(chain.Scratch, Elements[d]);
}
