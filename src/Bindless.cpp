#include "Bindless.h"

#include <format>
#include <stdexcept>

namespace {
constexpr vk::DescriptorBindingFlags BindlessFlagsUpdateAfterBind = vk::DescriptorBindingFlagBits::ePartiallyBound |
    vk::DescriptorBindingFlagBits::eUpdateAfterBind;

constexpr std::array SlotTypeNames{"uniform", "image", "sampler", "buffer", "vertex buffer", "index buffer", "model buffer", "object id buffer"};
constexpr std::array SlotTypeDescriptors{
    vk::DescriptorType::eUniformBuffer,
    vk::DescriptorType::eStorageImage,
    vk::DescriptorType::eCombinedImageSampler,
    vk::DescriptorType::eStorageBuffer,
    vk::DescriptorType::eStorageBuffer,
    vk::DescriptorType::eStorageBuffer,
    vk::DescriptorType::eStorageBuffer,
    vk::DescriptorType::eStorageBuffer
};
constexpr std::array SlotTypeBindings{
    0u, // Uniform
    1u, // Image
    2u, // Sampler
    6u, // General storage buffers
    3u, // Vertex buffers
    4u, // Index buffers
    5u, // Model buffers
    7u // Object ID buffers
};
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
    };
    const std::array<vk::DescriptorBindingFlags, bindings.size()> binding_flags{
        vk::DescriptorBindingFlagBits::ePartiallyBound, // Uniforms
        BindlessFlagsUpdateAfterBind, // Storage images
        BindlessFlagsUpdateAfterBind, // Samplers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Vertex buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Index buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // Model buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind, // General buffers
        vk::DescriptorBindingFlagBits::ePartiallyBound | BindlessFlagsUpdateAfterBind // Object ID buffers
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
        // Storage-buffer bindings (vertex, index, model, general, object id), each with MaxBuffers slots.
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, Config.MaxBuffers * 5},
    };
    DescriptorPool = Device.createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()});

    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique({*DescriptorPool, 1, &*SetLayout}).front());

    // Initialize free slot lists
    for (size_t i = 0; i < SlotTypeCount; ++i) {
        const auto type = static_cast<SlotType>(i);
        const uint32_t max = Config.Max(type);
        FreeSlots[i].reserve(max);
        for (uint32_t j = max; j-- > 0;) FreeSlots[i].push_back(j);
    }
}

uint32_t DescriptorSlots::Allocate(SlotType type) {
    auto &free_list = FreeSlots[static_cast<size_t>(type)];
    if (free_list.empty()) {
        throw std::runtime_error(std::format("Bindless {} slots exhausted", SlotTypeNames[static_cast<size_t>(type)]));
    }
    const auto slot = free_list.back();
    free_list.pop_back();
    return slot;
}

void DescriptorSlots::Release(SlotType type, uint32_t slot) {
    const auto idx = static_cast<size_t>(type);
    const uint32_t max = Config.Max(type);
    if (slot >= max) {
        throw std::runtime_error(std::format("Bindless {} slot {} is out of range (max {}).", SlotTypeNames[idx], slot, max));
    }
    FreeSlots[idx].push_back(slot);
}

vk::WriteDescriptorSet DescriptorSlots::MakeBufferWrite(SlotType type, uint32_t slot, const vk::DescriptorBufferInfo &info) const {
    const auto idx = static_cast<size_t>(type);
    return {GetSet(), SlotTypeBindings[idx], slot, 1, SlotTypeDescriptors[idx], nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    constexpr auto idx = static_cast<size_t>(SlotType::Image);
    return {GetSet(), SlotTypeBindings[idx], slot, 1, SlotTypeDescriptors[idx], &info, nullptr};
}

vk::WriteDescriptorSet DescriptorSlots::MakeUniformWrite(uint32_t slot, const vk::DescriptorBufferInfo &info) const {
    constexpr auto idx = static_cast<size_t>(SlotType::Uniform);
    return {GetSet(), SlotTypeBindings[idx], slot, 1, SlotTypeDescriptors[idx], nullptr, &info};
}

vk::WriteDescriptorSet DescriptorSlots::MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &info) const {
    constexpr auto idx = static_cast<size_t>(SlotType::Sampler);
    return {GetSet(), SlotTypeBindings[idx], slot, 1, SlotTypeDescriptors[idx], &info, nullptr};
}
