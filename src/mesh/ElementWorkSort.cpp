#include "mesh/ElementWorkSort.h"
#include "mesh/ElementMembershipWork.h"

#include "gpu/ElementWorkSortPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"

#include <ranges>

namespace {
uint32_t TemporaryWords(uint32_t capacity) { return capacity + 16u * TileCount(capacity, TileElements) + 16u; }
} // namespace

uint32_t SortElementWorkWords(uint32_t capacity) { return TemporaryWords(capacity) + sizeof(ElementWorkSortJob) / sizeof(uint32_t); }

void EncodeElementMembershipWork(state::Registry &r, mtl::ComputeChain &chain, std::span<const ElementWorkSeedJob> jobs) {
    // A job takes a tile per listed block.
    mtl::GatherPass(chain, GetMeshPipelines(r)[MeshPass::ElementWorkSeed], jobs, jobs | std::views::transform(&ElementWorkSeedJob::BlockCount) | std::ranges::to<std::vector>(), MeshElementBlockSize);
}

namespace {
// Sorts works whose dispatches cover the largest one's tiles for each work.
void EncodeSortChunk(state::Registry &r, mtl::ComputeChain &chain, std::span<const ElementWork> work) {
    std::vector<ElementWorkSortJob> jobs;
    uint32_t capacity = 0u;
    uint32_t largest_key = 0u;
    for (const auto domain : work) {
        const auto temporary = chain.Scratch.Allocate(TemporaryWords(domain.Capacity));
        jobs.push_back({domain, temporary.Offset});
        capacity = std::max(capacity, domain.Capacity);
        // Work keys are one-based 256-element block numbers.
        // Higher radix digits are zero for every key in this domain and need no passes.
        largest_key = std::max(largest_key, TileCount(domain.Count, TileElements));
    }
    ElementWorkSortPushConstants pc{.Jobs = chain.Upload(BytesOf(jobs))};
    const auto &pipelines = GetMeshPipelines(r);
    const auto domains = uint32_t(work.size()), groups = TileCount(capacity, TileElements);
    // Each pass alternates the order and temporary arrays.
    // An even number of passes leaves the final order in its canonical array for FinishWork.
    const uint32_t passes = std::max(2u, 2u * ((std::bit_width(largest_key) + 7u) / 8u));
    for (pc.Shift = 0u; pc.Shift < 4u * passes; pc.Shift += 4u) {
        chain.Groups(pipelines[MeshPass::ElementWorkHistogram], pc, groups, 256u, domains);
        chain.Groups(pipelines[MeshPass::ElementWorkPrefix], pc, 16u, 256u, domains);
        chain.Groups(pipelines[MeshPass::ElementWorkScatter], pc, groups, 256u, domains);
    }
    chain.Groups(pipelines[MeshPass::ElementWorkFinish], pc, 1u, 256u, domains);
}

// Sorts the works in chunks whose covered tiles stay within twice their own tiles, each within a grid's depth.
void EncodeSortChunks(state::Registry &r, mtl::ComputeChain &chain, std::span<const ElementWork> work) {
    constexpr size_t MaxDepth{65535u};
    constexpr uint64_t SlackTiles{64u};
    const auto tiles = [](const ElementWork &w) { return uint64_t(TileCount(w.Capacity, TileElements)); };
    std::vector<ElementWork> sorted{work.begin(), work.end()};
    std::ranges::stable_sort(sorted, std::greater{}, tiles);
    for (size_t first = 0u; first < sorted.size();) {
        const auto largest = tiles(sorted[first]);
        auto sum = largest;
        auto last = first + 1u;
        for (; last < sorted.size() && last - first < MaxDepth && (last - first + 1u) * largest <= 2u * (sum + tiles(sorted[last])) + SlackTiles; ++last) sum += tiles(sorted[last]);
        EncodeSortChunk(r, chain, std::span{sorted}.subspan(first, last - first));
        first = last;
    }
}
constexpr char SortKind{};
} // namespace

void EncodeSortElementWork(state::Registry &r, mtl::ComputeChain &chain, std::span<const ElementWork> work) {
    const auto words = std::ranges::fold_left(work, uint64_t{}, [](uint64_t sum, const ElementWork &w) { return sum + SortElementWorkWords(w.Capacity); });
    chain.Gather(&SortKind, std::vector(work.begin(), work.end()), words, [&r, &chain](std::vector<ElementWork> gathered) { EncodeSortChunks(r, chain, gathered); });
}
