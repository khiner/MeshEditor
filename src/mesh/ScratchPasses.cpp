#include "mesh/ScratchPasses.h"

#include "gpu/ScratchPassPushConstants.h"

Range TakeScratch(mtl::ComputeChain &chain, uint64_t words) {
    if (words > UINT32_MAX) throw std::length_error("GPU scratch exceeds its 32-bit address space.");
    return chain.Scratch.Allocate(uint32_t(words));
}

void FillScratch(mtl::ComputeChain &chain, const MeshPipelines &pipelines, uint32_t run, uint32_t count, uint32_t value) {
    const ScratchPassPushConstants pc{.Slot = chain.Scratch.Buffer.Slot, .Values = run, .Count = count, .Value = value};
    chain.Threads(pipelines[MeshPass::ScratchFill], pc, count);
}

void RecordColliderBuild(mtl::ComputeChain &chain, const MeshPipelines &pipelines, SlotOffset runs, uint32_t leaf_groups, uint32_t layers) {
    JobPushConstants pc{.Job = runs};
    for (pc.Parameter = 0u; pc.Parameter < 32u; pc.Parameter += 4u) {
        chain.Groups(pipelines[MeshPass::ColliderSortHistogram], pc, leaf_groups, 256u, layers);
        chain.Groups(pipelines[MeshPass::ColliderSortPrefix], pc, 16u, 256u, layers);
        chain.Groups(pipelines[MeshPass::ColliderSortScatter], pc, leaf_groups, 256u, layers);
    }
    chain.Groups(pipelines[MeshPass::ColliderTree], pc, leaf_groups, 256u, layers);
}

void ScanScratch(mtl::ComputeChain &chain, const MeshPipelines &pipelines, uint32_t values, uint32_t length) {
    const auto blocks = TileCount(length, BlockElements);
    const ScratchPassPushConstants pc{.Slot = chain.Scratch.Buffer.Slot, .Values = values, .Count = length, .Blocks = chain.Scratch.Allocate(blocks).Offset};
    chain.Groups(pipelines[MeshPass::ScratchScanSum], pc, blocks);
    chain.Groups(pipelines[MeshPass::ScratchScanPrefix], pc, 1u);
    chain.Groups(pipelines[MeshPass::ScratchScanOffsets], pc, blocks);
}
