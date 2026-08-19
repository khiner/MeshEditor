#pragma once

#include "RangeAllocator.h"
#include "metal/MetalContext.h"
#include "metal/Slots.h"

#include <array>

namespace mtl {
// Tier-2 argument buffer mirroring the shader BindlessSet.
struct BindlessSet {
    explicit BindlessSet(const Context &);

    uint32_t Allocate(SlotType);
    // Snapshot restore re-acquires slots baked into restored state.
    bool Reserve(SlotType, uint32_t slot);
    void Release(TypedSlot);

    void SetBuffer(TypedSlot, MTL::Buffer *, uint64_t offset = 0);
    void SetTexture(uint32_t slot, MTL::Texture *);
    void SetSampler(TypedSlot, MTL::Texture *, MTL::SamplerState *);
    void Clear(TypedSlot);

    MTL::Buffer *Table() const { return ArgumentBuffer.get(); }

private:
    size_t EntryOffset(SlotType, uint32_t slot) const;
    uint64_t *EntryAt(SlotType, uint32_t slot) const;

    const Context &Ctx;
    NS::SharedPtr<MTL::Buffer> ArgumentBuffer;
    std::array<RangeAllocator, SlotTypeCount> Allocators;
};
} // namespace mtl
