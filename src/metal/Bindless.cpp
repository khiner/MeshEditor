#include "metal/Bindless.h"

#include <cstring>
#include <format>
#include <stdexcept>

namespace mtl {
namespace {
MTL::ResourceUsage UsageFor(SlotType type) {
    return BindingDefs[size_t(type)].Kind == BindKind::Sampler ? MTL::ResourceUsageRead : MTL::ResourceUsageRead | MTL::ResourceUsageWrite;
}

void ForEachResource(const auto &resources, auto use) {
    for (size_t i = 0; i < resources.size(); ++i) {
        for (auto *resource : resources[i]) {
            if (resource) use(resource, UsageFor(SlotType(i)));
        }
    }
}
} // namespace

BindlessSet::BindlessSet(const Context &ctx) : Ctx(ctx) {
    ArgumentBuffer = NS::TransferPtr(ctx.Device->newBuffer(BindlessTableSize, MTL::ResourceStorageModeShared));
    if (!ArgumentBuffer) throw std::runtime_error("Failed to allocate the bindless argument buffer.");
    std::memset(ArgumentBuffer->contents(), 0, BindlessTableSize);
    ctx.AddResident(ArgumentBuffer.get());
    if (!ctx.Residency) {
        for (size_t i = 0; i < Resources.size(); ++i) Resources[i].resize(BindlessLayout[i].Capacity);
    }
}

// Lowest-free allocation keeps scene replay byte-identical regardless of release order.
uint32_t BindlessSet::Allocate(SlotType type) {
    const auto slot = Allocators[size_t(type)].Allocate(1).Offset;
    if (slot >= BindlessLayout[size_t(type)].Capacity) {
        throw std::runtime_error(std::format("Ran out of '{}' bindless slots ({})", BindingDefs[size_t(type)].Name, slot));
    }
    return slot;
}
bool BindlessSet::Reserve(SlotType type, uint32_t slot) { return Allocators[size_t(type)].Reserve({slot, 1}); }
void BindlessSet::Release(TypedSlot slot) {
    Track(slot, nullptr);
    Allocators[size_t(slot.Type)].Free({slot.Slot, 1});
}

size_t BindlessSet::EntryOffset(SlotType type, uint32_t slot) const {
    const auto &layout = BindlessLayout[size_t(type)];
    return layout.Offset + size_t(slot) * layout.Stride;
}

uint64_t *BindlessSet::EntryAt(SlotType type, uint32_t slot) const {
    return reinterpret_cast<uint64_t *>(static_cast<std::byte *>(ArgumentBuffer->contents()) + EntryOffset(type, slot));
}

void BindlessSet::Track(TypedSlot slot, MTL::Resource *resource) {
    auto &resources = Resources[size_t(slot.Type)];
    if (!resources.empty()) resources[slot.Slot] = resource;
}

void BindlessSet::SetBuffer(TypedSlot slot, MTL::Buffer *buffer, uint64_t offset) {
    const auto kind = BindingDefs[size_t(slot.Type)].Kind;
    if (kind == BindKind::Uniform || kind == BindKind::UniformDynamic) return;
    Track(slot, buffer);
    if (!buffer) {
        *EntryAt(slot.Type, slot.Slot) = 0;
        return;
    }
    *EntryAt(slot.Type, slot.Slot) = buffer->gpuAddress() + offset;
    Ctx.AddResident(buffer);
}

void BindlessSet::SetTexture(uint32_t slot, MTL::Texture *texture) {
    Track({SlotType::Image, slot}, texture);
    *EntryAt(SlotType::Image, slot) = texture ? texture->gpuResourceID()._impl : 0;
    if (texture) Ctx.AddResident(texture);
}

void BindlessSet::SetSampler(TypedSlot slot, MTL::Texture *texture, MTL::SamplerState *sampler) {
    Track(slot, texture);
    auto *entry = EntryAt(slot.Type, slot.Slot);
    entry[0] = texture ? texture->gpuResourceID()._impl : 0;
    entry[1] = sampler ? sampler->gpuResourceID()._impl : 0;
    if (texture) Ctx.AddResident(texture);
}

void BindlessSet::Clear(TypedSlot slot) {
    const auto &layout = BindlessLayout[size_t(slot.Type)];
    if (layout.Stride == 0) return;
    Track(slot, nullptr);
    std::memset(EntryAt(slot.Type, slot.Slot), 0, layout.Stride);
}

void BindlessSet::UseResources(MTL::RenderCommandEncoder *encoder) const {
    if (Ctx.Residency) return;
    constexpr auto stages = MTL::RenderStageVertex | MTL::RenderStageMesh | MTL::RenderStageFragment;
    ForEachResource(Resources, [&](auto *resource, auto usage) { encoder->useResource(resource, usage, stages); });
}

void BindlessSet::UseResources(MTL::ComputeCommandEncoder *encoder) const {
    if (Ctx.Residency) return;
    ForEachResource(Resources, [&](auto *resource, auto usage) { encoder->useResource(resource, usage); });
}
} // namespace mtl
