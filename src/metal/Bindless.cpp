#include "metal/Bindless.h"
#include "metal/AutoreleaseScope.h"

#include "metal/MetalCpp.h"

#include <format>

namespace mtl {
BindlessSet::~BindlessSet() { AutoreleaseScope::Release(ArgumentBuffer); }
BindlessSet::BindlessSet(const BindlessSet &) = default;
BindlessSet::BindlessSet(BindlessSet &&) noexcept = default;

BindlessSet::BindlessSet(const Context &ctx) : Ctx(ctx) {
    const AutoreleaseScope pool;
    ArgumentBuffer = NS::TransferPtr(ctx.Device->newBuffer(BindlessTableSize, MTL::ResourceStorageModeShared));
    if (!ArgumentBuffer) throw std::runtime_error("Failed to allocate the bindless argument buffer.");
    std::memset(ArgumentBuffer->contents(), 0, BindlessTableSize);
    ctx.AddResident(ArgumentBuffer.get());
}

// Lowest-free allocation keeps scene replay byte-identical regardless of release order.
uint32_t BindlessSet::Allocate(SlotType type) {
    const auto slot = TryAllocate(type);
    if (slot == InvalidSlot) {
        throw std::runtime_error(std::format("Ran out of '{}' bindless slots ({})", BindingDefs[size_t(type)].Name, SlotCapacity(BindingDefs[size_t(type)].Kind)));
    }
    return slot;
}
uint32_t BindlessSet::TryAllocate(SlotType type) {
    auto &allocator = Allocators[size_t(type)];
    RangeAllocator::Transaction transaction{allocator};
    const auto slot = allocator.Allocate(1).Offset;
    if (slot >= SlotCapacity(BindingDefs[size_t(type)].Kind)) return InvalidSlot;
    transaction.Commit();
    return slot;
}
bool BindlessSet::Reserve(SlotType type, uint32_t slot) { return Allocators[size_t(type)].Reserve({slot, 1}); }
void BindlessSet::Release(TypedSlot slot) { Allocators[size_t(slot.Type)].Free({slot.Slot, 1}); }

size_t BindlessSet::EntryOffset(SlotType type, uint32_t slot) const {
    return BindlessOffset(type) + size_t(slot) * SlotStride(BindingDefs[size_t(type)].Kind);
}

uint64_t *BindlessSet::EntryAt(SlotType type, uint32_t slot) const {
    return reinterpret_cast<uint64_t *>(static_cast<std::byte *>(ArgumentBuffer->contents()) + EntryOffset(type, slot));
}

void BindlessSet::SetBuffer(TypedSlot slot, MTL::Buffer *buffer, uint64_t offset) {
    const auto kind = BindingDefs[size_t(slot.Type)].Kind;
    if (kind == BindKind::Uniform || kind == BindKind::UniformDynamic) return;
    if (!buffer) {
        *EntryAt(slot.Type, slot.Slot) = 0;
        return;
    }
    *EntryAt(slot.Type, slot.Slot) = buffer->gpuAddress() + offset;
    Ctx.AddResident(buffer);
}

void BindlessSet::SetTexture(uint32_t slot, MTL::Texture *texture) {
    *EntryAt(SlotType::Image, slot) = texture ? texture->gpuResourceID()._impl : 0;
    if (texture) Ctx.AddResident(texture);
}

void BindlessSet::SetSampler(TypedSlot slot, MTL::Texture *texture, MTL::SamplerState *sampler) {
    auto *entry = EntryAt(slot.Type, slot.Slot);
    entry[0] = texture ? texture->gpuResourceID()._impl : 0;
    entry[1] = sampler ? sampler->gpuResourceID()._impl : 0;
    if (texture) Ctx.AddResident(texture);
}

void BindlessSet::Clear(TypedSlot slot) {
    const auto stride = SlotStride(BindingDefs[size_t(slot.Type)].Kind);
    if (stride == 0) return;
    std::memset(EntryAt(slot.Type, slot.Slot), 0, stride);
}
} // namespace mtl
