#pragma once

#include "RangeAllocator.h"
#include "metal/MetalContext.h"
#include "metal/MetalSchema.h"
#include "metal/Slots.h"

namespace MTL {
class Buffer;
class Texture;
class SamplerState;
} // namespace MTL

namespace mtl {
// Tier-2 argument buffer mirroring the shader BindlessSet.
// Every resource it references is resident on the queue.
struct BindlessSet {
    explicit BindlessSet(const Context &);
    ~BindlessSet();
    BindlessSet(const BindlessSet &);
    BindlessSet(BindlessSet &&) noexcept;

    uint32_t Allocate(SlotType);
    uint32_t TryAllocate(SlotType); // InvalidSlot on capacity exhaustion, and no reservation remains.
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

// Slots allocated from one set and released together when the owner goes away.
// A move hands the slots to the destination.
struct SlotOwner {
    explicit SlotOwner(BindlessSet &slots) : Slots(&slots) {}
    SlotOwner(SlotOwner &&other) noexcept : Slots(other.Slots), Owned(std::exchange(other.Owned, {})) {}
    SlotOwner &operator=(SlotOwner &&other) noexcept {
        if (this != &other) {
            Release();
            Slots = other.Slots;
            Owned = std::exchange(other.Owned, {});
        }
        return *this;
    }
    ~SlotOwner() { Release(); }

    uint32_t Allocate(SlotType type) {
        const auto slot = Slots->Allocate(type);
        Owned.push_back({type, slot});
        return slot;
    }

    BindlessSet *Slots;
    std::vector<TypedSlot> Owned;

private:
    void Release() {
        for (const auto slot : Owned) Slots->Release(slot);
        Owned.clear();
    }
};
} // namespace mtl
