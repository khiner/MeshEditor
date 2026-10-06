#include "mesh/ElementWorkSort.h"
#include "mesh/ElementMembershipWork.h"

#include "gpu/ElementWorkSortPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"

namespace {
uint32_t TemporaryWords(uint32_t capacity) { return capacity + 16u * ((capacity + 255u) / 256u) + 16u; }
} // namespace

uint32_t SortElementWorkWords(uint32_t capacity) { return TemporaryWords(capacity) + sizeof(ElementWorkSortJob) / sizeof(uint32_t); }

void EncodeElementMembershipWork(state::Scene &r, mtl::ComputeChain &chain, std::span<const ElementWorkSeedJob> jobs) {
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::ElementWorkSeed];
    // Each seed fills its own work.
    chain.Concurrent([&] { for (const auto &job : jobs) chain.Groups(pipeline, job, job.BlockCount); });
}

void EncodeSortElementWork(state::Scene &r, mtl::ComputeChain &chain, std::span<const ElementWork> work) {
    if (work.empty()) return;
    auto &storage = chain.Scratch;
    std::vector<ElementWorkSortJob> jobs;
    uint32_t capacity = 0u;
    uint32_t largest_key = 0u;
    for (const auto domain : work) {
        const auto temporary = storage.Allocate(TemporaryWords(domain.Capacity));
        jobs.push_back({domain, temporary.Offset});
        capacity = std::max(capacity, domain.Capacity);
        // Work keys are one-based 256-element block numbers. Higher radix
        // digits are zero for every key in this domain and need no passes.
        largest_key = std::max(largest_key, uint32_t((uint64_t(domain.Count) + 255u) / 256u));
    }
    const auto at = storage.Allocate(uint32_t(jobs.size()) * sizeof(ElementWorkSortJob) / sizeof(uint32_t));
    storage.Buffer.Update(as_bytes(jobs), uint64_t(at.Offset) * sizeof(uint32_t));
    ElementWorkSortPushConstants pc{.Jobs = {storage.Buffer.Slot, at.Offset}};
    const auto &pipelines = GetMeshPipelines(r);
    const auto domains = uint32_t(work.size()), groups = (capacity + 255u) / 256u;
    // Each pass alternates the order and temporary arrays. An even number of
    // passes leaves the final order in its canonical array for FinishWork.
    const uint32_t passes = std::max(2u, 2u * ((std::bit_width(largest_key) + 7u) / 8u));
    for (pc.Shift = 0u; pc.Shift < 4u * passes; pc.Shift += 4u) {
        chain.Groups(pipelines[MeshPass::ElementWorkHistogram], pc, groups, 256u, domains);
        chain.Groups(pipelines[MeshPass::ElementWorkPrefix], pc, 16u, 256u, domains);
        chain.Groups(pipelines[MeshPass::ElementWorkScatter], pc, groups, 256u, domains);
    }
    chain.Groups(pipelines[MeshPass::ElementWorkFinish], pc, 1u, 256u, domains);
}
