#include "Bindless.h"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {
constexpr vk::DescriptorBindingFlags BindlessFlagsUpdateAfterBind = vk::DescriptorBindingFlagBits::ePartiallyBound |
    vk::DescriptorBindingFlagBits::eUpdateAfterBind;

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
    return {0u, 0u};
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
    switch (kind) {
        case BindKind::Uniform: return vk::DescriptorBindingFlagBits::ePartiallyBound;
        case BindKind::Image: return BindlessFlagsUpdateAfterBind;
        case BindKind::Sampler: return BindlessFlagsUpdateAfterBind;
        case BindKind::Buffer: return vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind;
    }
    return vk::DescriptorBindingFlagBits::ePartiallyBound;
}

constexpr uint32_t CountBufferBindings() {
    uint32_t count = 0;
    for (const auto &def : BindingDefs) {
        if (def.Kind == BindKind::Buffer) ++count;
    }
    return count;
}

struct SlotInfo {
    std::string_view Name;
    vk::DescriptorType Descriptor;
    uint32_t Binding;
};

constexpr std::array<SlotInfo, SlotTypeCount> MakeSlotInfos() {
    std::array<SlotInfo, SlotTypeCount> infos{};
    for (size_t i = 0; i < BindingDefs.size(); ++i) {
        const auto &def = BindingDefs[i];
        const auto type = static_cast<SlotType>(i);
        infos[i] = SlotInfo{def.Name, DescriptorTypeFor(def.Kind), static_cast<uint32_t>(type)};
    }
    return infos;
}

constexpr auto SlotInfos = MakeSlotInfos();
static_assert(SlotInfos.size() == SlotTypeCount, "SlotInfos must match SlotTypeCount.");
} // namespace

DescriptorSlots::DescriptorSlots(vk::Device device, const vk::PhysicalDeviceDescriptorIndexingProperties &props)
    : Device(device) {
    std::unordered_map<BindKind, uint32_t> limits_map{
        {BindKind::Buffer, GetMaxDescriptors(props, BindKind::Buffer)},
        {BindKind::Image, GetMaxDescriptors(props, BindKind::Image)},
        {BindKind::Uniform, GetMaxDescriptors(props, BindKind::Uniform)},
        {BindKind::Sampler, GetMaxDescriptors(props, BindKind::Sampler)},
    };
    std::array<vk::DescriptorSetLayoutBinding, SlotTypeCount> bindings{};
    std::array<vk::DescriptorBindingFlags, SlotTypeCount> binding_flags{};
    for (size_t i = 0; i < BindingDefs.size(); ++i) {
        const auto &def = BindingDefs[i];
        bindings[i] = vk::DescriptorSetLayoutBinding{
            static_cast<uint32_t>(i),
            DescriptorTypeFor(def.Kind),
            limits_map[def.Kind],
            vk::ShaderStageFlagBits::eAll
        };
        binding_flags[i] = FlagsFor(def.Kind);
    }
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{
        static_cast<uint32_t>(binding_flags.size()),
        binding_flags.data()
    };
    vk::DescriptorSetLayoutCreateInfo layout_ci{vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, bindings};
    layout_ci.pNext = &binding_flags_ci;
    SetLayout = Device.createDescriptorSetLayoutUnique(layout_ci);

    const std::array pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, limits_map.at(BindKind::Uniform)},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, limits_map.at(BindKind::Image)},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, limits_map.at(BindKind::Sampler)},
        // Storage-buffer bindings, each with MaxBuffers slots.
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, limits_map.at(BindKind::Buffer) * CountBufferBindings()},
    };
    DescriptorPool = Device.createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()});
    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique({*DescriptorPool, 1, &*SetLayout}).front());

    // Initialize free slot lists
    for (size_t i = 0; i < SlotTypeCount; ++i) {
        const auto max = limits_map.at(BindingDefs[i].Kind);
        FreeSlots[i].reserve(max);
        for (uint32_t j = max; j-- > 0;) FreeSlots[i].emplace_back(j);
    }
}

uint32_t DescriptorSlots::Allocate(SlotType type) {
    auto &free_list = FreeSlots[static_cast<size_t>(type)];
    if (free_list.empty()) {
        throw std::runtime_error(std::format("Bindless {} slots exhausted", SlotInfos[static_cast<size_t>(type)].Name));
    }
    const auto slot = free_list.back();
    free_list.pop_back();
    return slot;
}

void DescriptorSlots::Release(TypedSlot slot) {
    FreeSlots[static_cast<size_t>(slot.Type)].emplace_back(slot.Slot);
}

vk::WriteDescriptorSet DescriptorSlots::MakeBufferWrite(TypedSlot slot, const vk::DescriptorBufferInfo &info) const {
    const auto idx = static_cast<size_t>(slot.Type);
    return {GetSet(), SlotInfos[idx].Binding, slot.Slot, 1, SlotInfos[idx].Descriptor, nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    constexpr auto idx = static_cast<size_t>(SlotType::Image);
    return {GetSet(), SlotInfos[idx].Binding, slot, 1, SlotInfos[idx].Descriptor, &info, nullptr};
}

vk::WriteDescriptorSet DescriptorSlots::MakeUniformWrite(uint32_t slot, const vk::DescriptorBufferInfo &info) const {
    constexpr auto idx = static_cast<size_t>(SlotType::Uniform);
    return {GetSet(), SlotInfos[idx].Binding, slot, 1, SlotInfos[idx].Descriptor, nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    constexpr auto idx = static_cast<size_t>(SlotType::Sampler);
    return {GetSet(), SlotInfos[idx].Binding, slot, 1, SlotInfos[idx].Descriptor, &info, nullptr};
}
