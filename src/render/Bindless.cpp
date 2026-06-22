#include "render/Bindless.h"

#include <algorithm>
#include <ranges>

using std::ranges::count_if, std::views::transform, std::ranges::to;
using std::views::iota;

namespace {
constexpr std::pair<uint32_t, uint32_t> MaxUpdateAfterBindPair(const vk::PhysicalDeviceDescriptorIndexingProperties &props, BindKind kind) {
    switch (kind) {
        case BindKind::Buffer:
            return {props.maxDescriptorSetUpdateAfterBindStorageBuffers, props.maxPerStageDescriptorUpdateAfterBindStorageBuffers};
        case BindKind::Image:
            return {props.maxDescriptorSetUpdateAfterBindStorageImages, props.maxPerStageDescriptorUpdateAfterBindStorageImages};
        case BindKind::Uniform:
            return {props.maxDescriptorSetUpdateAfterBindUniformBuffers, props.maxPerStageDescriptorUpdateAfterBindUniformBuffers};
        case BindKind::Sampler:
            return {props.maxDescriptorSetUpdateAfterBindSampledImages, props.maxPerStageDescriptorUpdateAfterBindSamplers};
    }
    return {};
}

constexpr uint32_t ClampLimit(const vk::PhysicalDeviceDescriptorIndexingProperties &props, BindKind kind, uint32_t limit) {
    const auto [set_max, stage_max] = MaxUpdateAfterBindPair(props, kind);
    return std::clamp(std::min(set_max, stage_max), 1u, limit);
}

constexpr uint32_t LimitCapFor(BindKind kind) {
    switch (kind) {
        case BindKind::Buffer: return 32768u;
        case BindKind::Image: return 1024u;
        case BindKind::Uniform: return 64u;
        case BindKind::Sampler: return 256u;
    }
    return 0u;
}

constexpr uint32_t GetMaxDescriptors(const vk::PhysicalDeviceDescriptorIndexingProperties &props, BindKind kind) {
    return ClampLimit(props, kind, LimitCapFor(kind));
}

constexpr vk::DescriptorType DescriptorTypeFor(BindKind kind) {
    switch (kind) {
        case BindKind::Uniform: return vk::DescriptorType::eUniformBuffer;
        case BindKind::Image: return vk::DescriptorType::eStorageImage;
        case BindKind::Sampler: return vk::DescriptorType::eCombinedImageSampler;
        case BindKind::Buffer: return vk::DescriptorType::eStorageBuffer;
    }
    return vk::DescriptorType::eStorageBuffer;
}

constexpr vk::DescriptorBindingFlags FlagsFor(BindKind kind) {
    using enum vk::DescriptorBindingFlagBits;
    switch (kind) {
        case BindKind::Uniform: return ePartiallyBound;
        case BindKind::Image: return ePartiallyBound | eUpdateAfterBind;
        case BindKind::Sampler: return ePartiallyBound | eUpdateAfterBind;
        case BindKind::Buffer: return ePartiallyBound | eUpdateAfterBind;
    }
    return {};
}

struct SlotBinding {
    uint32_t Binding;
    vk::DescriptorType Descriptor;
};

constexpr SlotBinding BindingFor(SlotType type) { return {uint32_t(type), DescriptorTypeFor(BindingDefs[size_t(type)].Kind)}; }

} // namespace

DescriptorSlots::DescriptorSlots(vk::Device device, const vk::PhysicalDeviceDescriptorIndexingProperties &props)
    : Device(device) {
    const auto indices = iota(uint32_t{0}, uint32_t(BindingDefs.size()));
    const auto bindings = indices | transform([&](uint32_t i) {
                              const auto &kind = BindingDefs[i].Kind;
                              return vk::DescriptorSetLayoutBinding{i, DescriptorTypeFor(kind), GetMaxDescriptors(props, kind), vk::ShaderStageFlagBits::eAll};
                          }) |
        to<std::vector>();
    const auto binding_flags = indices | transform([&](uint32_t i) { return FlagsFor(BindingDefs[i].Kind); }) | to<std::vector>();
    vk::DescriptorSetLayoutCreateInfo layout_ci{vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, bindings};
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{binding_flags};
    layout_ci.pNext = &binding_flags_ci;
    SetLayout = Device.createDescriptorSetLayoutUnique(layout_ci);

    const auto pool_size_for = [&](BindKind kind) {
        return vk::DescriptorPoolSize{
            DescriptorTypeFor(kind),
            GetMaxDescriptors(props, kind) * uint32_t(count_if(BindingDefs, [kind](const auto &def) { return def.Kind == kind; })),
        };
    };
    const std::array pool_sizes{
        pool_size_for(BindKind::Uniform),
        pool_size_for(BindKind::Image),
        pool_size_for(BindKind::Sampler),
        pool_size_for(BindKind::Buffer),
    };
    DescriptorPool = Device.createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, uint32_t(pool_sizes.size()), pool_sizes.data()});
    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique({*DescriptorPool, 1, &*SetLayout}).front());
}

// A slot is a count-1 range. RangeAllocator hands out the lowest available offset deterministically (a function
// of the free set, not of release order), so allocations survive a scene clear + replay byte-identically — no
// free-list reset needed: releasing the scene's slots restores the canonical free set on its own.
uint32_t DescriptorSlots::Allocate(SlotType type) { return Allocators[size_t(type)].Allocate(1).Offset; }
bool DescriptorSlots::Reserve(SlotType type, uint32_t slot) { return Allocators[size_t(type)].Reserve({slot, 1}); }
void DescriptorSlots::Release(TypedSlot slot) { Allocators[size_t(slot.Type)].Free({slot.Slot, 1}); }

vk::WriteDescriptorSet DescriptorSlots::MakeBufferWrite(TypedSlot slot, const vk::DescriptorBufferInfo &info) const {
    const auto [binding, descriptor] = BindingFor(slot.Type);
    return {*DescriptorSet, binding, slot.Slot, 1, descriptor, nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    const auto [binding, descriptor] = BindingFor(SlotType::Image);
    return {*DescriptorSet, binding, slot, 1, descriptor, &info, nullptr};
}

vk::WriteDescriptorSet DescriptorSlots::MakeUniformWrite(TypedSlot slot, const vk::DescriptorBufferInfo &info) const {
    return MakeBufferWrite(slot, info);
}

vk::WriteDescriptorSet DescriptorSlots::MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    const auto [binding, descriptor] = BindingFor(SlotType::Sampler);
    return {*DescriptorSet, binding, slot, 1, descriptor, &info, nullptr};
}

vk::WriteDescriptorSet DescriptorSlots::MakeCubeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    const auto [binding, descriptor] = BindingFor(SlotType::CubeSampler);
    return {*DescriptorSet, binding, slot, 1, descriptor, &info, nullptr};
}
