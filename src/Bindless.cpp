#include "Bindless.h"

#include <format>
#include <stdexcept>

namespace {
constexpr vk::DescriptorBindingFlags BindlessFlagsUpdateAfterBind = vk::DescriptorBindingFlagBits::ePartiallyBound |
    vk::DescriptorBindingFlagBits::eUpdateAfterBind;

struct SlotInfo {
    const char *Name;
    vk::DescriptorType Descriptor;
    uint32_t Binding;
};

constexpr std::array SlotInfos{
    SlotInfo{"uniform", vk::DescriptorType::eUniformBuffer, 0u},
    SlotInfo{"image", vk::DescriptorType::eStorageImage, 1u},
    SlotInfo{"sampler", vk::DescriptorType::eCombinedImageSampler, 2u},
    SlotInfo{"buffer", vk::DescriptorType::eStorageBuffer, 6u}, // General storage buffers
    SlotInfo{"vertex buffer", vk::DescriptorType::eStorageBuffer, 3u},
    SlotInfo{"index buffer", vk::DescriptorType::eStorageBuffer, 4u},
    SlotInfo{"model buffer", vk::DescriptorType::eStorageBuffer, 5u},
    SlotInfo{"object id buffer", vk::DescriptorType::eStorageBuffer, 7u},
    SlotInfo{"face normal buffer", vk::DescriptorType::eStorageBuffer, 8u},
    SlotInfo{"draw data buffer", vk::DescriptorType::eStorageBuffer, 9u},
};
static_assert(SlotInfos.size() == SlotTypeCount, "SlotInfos must match SlotTypeCount.");
} // namespace

DescriptorSlots::DescriptorSlots(vk::Device device, const BindlessConfig &config)
    : Device(device), Config(config) {
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, Config.MaxUniforms, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageImage, Config.MaxImages, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eCombinedImageSampler, Config.MaxSamplers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{4, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{5, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{6, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{7, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{8, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
        vk::DescriptorSetLayoutBinding{9, vk::DescriptorType::eStorageBuffer, Config.MaxBuffers, vk::ShaderStageFlagBits::eAll},
    };
    const std::array<vk::DescriptorBindingFlags, bindings.size()> binding_flags{
        vk::DescriptorBindingFlagBits::ePartiallyBound, // Uniforms
        BindlessFlagsUpdateAfterBind, // Storage images
        BindlessFlagsUpdateAfterBind, // Samplers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Vertex buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Index buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Model buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // General buffers (selection nodes, counters, click/box results, element state buffers)
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Object ID buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Face normal buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind // Draw data buffers
    };
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{
        static_cast<uint32_t>(binding_flags.size()),
        binding_flags.data()
    };
    vk::DescriptorSetLayoutCreateInfo layout_ci{vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, bindings};
    layout_ci.pNext = &binding_flags_ci;
    SetLayout = Device.createDescriptorSetLayoutUnique(layout_ci);

    const std::array pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, Config.MaxUniforms},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, Config.MaxImages},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, Config.MaxSamplers},
        // Storage-buffer bindings (vertex, index, model, general, object id, face normals, draw data), each with MaxBuffers slots.
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, Config.MaxBuffers * 7},
    };
    DescriptorPool = Device.createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()});
    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique({*DescriptorPool, 1, &*SetLayout}).front());

    // Initialize free slot lists
    for (size_t i = 0; i < SlotTypeCount; ++i) {
        const uint32_t max = Config.Max(static_cast<SlotType>(i));
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
    const auto idx = static_cast<size_t>(slot.Type);
    if (const uint32_t max = Config.Max(slot.Type); slot.Slot >= max) {
        throw std::runtime_error(std::format("Bindless {} slot {} is out of range (max {}).", SlotInfos[idx].Name, slot.Slot, max));
    }
    FreeSlots[idx].emplace_back(slot.Slot);
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
