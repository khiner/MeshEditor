#pragma once
#include "gpu/ElementWork.h"
#include "gpu/ElementWorkSeedJob.h"
#include "mesh/ElementArena.h"
#include "mesh/SelectionView.h"
#include "metal/BufferArena.h"
#include "render/ElementWorkOps.h"
#include "state/Entity.h"

namespace mtl { struct ComputeChain; }

// Lists the blocks whose live elements seed `work`, masked by a selection when one is named.
// Only block IDs are uploaded, and live and selected membership stays on the GPU.
template<typename T>
ElementWorkSeedJob PrepareBlockMembershipWork(BufferArena<uint32_t> &storage, const ElementArena<T> &arena, ElementSetRef set,
                                              std::span<const uint32_t> blocks, ElementWork work, uint32_t mask_slot = InvalidSlot) {
    const auto ids = storage.Allocate(uint32_t(blocks.size()));
    storage.Buffer.Update(std::as_bytes(blocks),uint64_t(ids.Offset)*4u);
    return {work,{storage.Buffer.Slot,ids.Offset},ids.Count,arena.Blocks.Buffer.Slot,set.Index,mask_slot};
}

template<typename T>
ElementWorkSeedJob PrepareElementMembershipWork(BufferArena<uint32_t> &storage, const ElementArena<T> &arena, ElementSetRef set) {
    std::vector<uint32_t> blocks;
    arena.ForEachBlock(set, [&](uint32_t b, const auto &) { blocks.push_back(b); });
    return PrepareBlockMembershipWork(storage,arena,set,blocks,AllocateElementWork(storage,arena.Capacity(),blocks.size()));
}

// Only the blocks holding selected elements are listed.
template<typename T>
ElementWorkSeedJob PrepareSelectedMembershipWork(BufferArena<uint32_t> &storage, const ElementArena<T> &arena, ElementSetRef set,
                                                 const SelectionView &selection, uint32_t mask_slot) {
    std::vector<uint32_t> blocks;
    selection.ForEachBlock([&](uint32_t block, uint32_t) { blocks.push_back(block); });
    return PrepareBlockMembershipWork(storage,arena,set,blocks,AllocateElementWork(storage,arena.Capacity(),blocks.size()),mask_slot);
}

// Records membership gathering before the shared compact work sorter.
void EncodeElementMembershipWork(state::Scene &, mtl::ComputeChain &, std::span<const ElementWorkSeedJob>);
