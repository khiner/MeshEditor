#include "Bindless.h"

#include <algorithm>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

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

struct SlotInfo {
    std::string_view Name;
    vk::DescriptorType Descriptor;
    uint32_t Binding;
};

constexpr auto SlotInfos = []<size_t... I>(std::index_sequence<I...>) {
    return std::array{
        SlotInfo{BindingDefs[I].Name, DescriptorTypeFor(BindingDefs[I].Kind), static_cast<uint32_t>(I)}...
    };
}(std::make_index_sequence<SlotTypeCount>{});
constexpr auto GetSlotInfo(SlotType type) { return SlotInfos[size_t(type)]; }
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

    const uint32_t buffer_binding_count = count_if(BindingDefs, [](const auto &def) { return def.Kind == BindKind::Buffer; });
    const std::array pool_sizes{
        vk::DescriptorPoolSize{DescriptorTypeFor(BindKind::Uniform), GetMaxDescriptors(props, BindKind::Uniform)},
        vk::DescriptorPoolSize{DescriptorTypeFor(BindKind::Image), GetMaxDescriptors(props, BindKind::Image)},
        vk::DescriptorPoolSize{DescriptorTypeFor(BindKind::Sampler), GetMaxDescriptors(props, BindKind::Sampler)},
        vk::DescriptorPoolSize{DescriptorTypeFor(BindKind::Buffer), GetMaxDescriptors(props, BindKind::Buffer) * buffer_binding_count},
    };
    DescriptorPool = Device.createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()});
    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique({*DescriptorPool, 1, &*SetLayout}).front());

    // Initialize free slot lists
    std::ranges::transform(indices, FreeSlots.begin(), [&](uint32_t i) {
        return iota(uint32_t{0}, GetMaxDescriptors(props, BindingDefs[i].Kind)) | std::views::reverse | to<std::vector>();
    });
}

uint32_t DescriptorSlots::Allocate(SlotType type) {
    auto &free_list = FreeSlots[size_t(type)];
    if (free_list.empty()) throw std::runtime_error(std::format("Bindless {} slots exhausted", GetSlotInfo(type).Name));

    const auto slot = free_list.back();
    free_list.pop_back();
    return slot;
}

void DescriptorSlots::Release(TypedSlot slot) {
    FreeSlots[size_t(slot.Type)].emplace_back(slot.Slot);
}

vk::WriteDescriptorSet DescriptorSlots::MakeBufferWrite(TypedSlot slot, const vk::DescriptorBufferInfo &info) const {
    const auto &slot_info = GetSlotInfo(slot.Type);
    return {*DescriptorSet, slot_info.Binding, slot.Slot, 1, slot_info.Descriptor, nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    const auto &slot_info = GetSlotInfo(SlotType::Image);
    return {*DescriptorSet, slot_info.Binding, slot, 1, slot_info.Descriptor, &info, nullptr};
}

vk::WriteDescriptorSet DescriptorSlots::MakeUniformWrite(uint32_t slot, const vk::DescriptorBufferInfo &info) const {
    const auto &slot_info = GetSlotInfo(SlotType::Uniform);
    return {*DescriptorSet, slot_info.Binding, slot, 1, slot_info.Descriptor, nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    const auto &slot_info = GetSlotInfo(SlotType::Sampler);
    return {*DescriptorSet, slot_info.Binding, slot, 1, slot_info.Descriptor, &info, nullptr};
}
