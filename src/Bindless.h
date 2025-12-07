#pragma once

#include "Slots.h"

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdint>
#include <vector>

struct BindlessConfig {
    uint32_t MaxBuffers;
    uint32_t MaxImages;
    uint32_t MaxUniforms;
    uint32_t MaxSamplers;

    constexpr uint32_t Max(SlotType type) const {
        switch (type) {
            case SlotType::Uniform: return MaxUniforms;
            case SlotType::Image: return MaxImages;
            case SlotType::Sampler: return MaxSamplers;
            case SlotType::Buffer:
            case SlotType::VertexBuffer:
            case SlotType::IndexBuffer:
            case SlotType::ModelBuffer: return MaxBuffers;
        }
    }
};

struct BindlessResources {
    BindlessResources(vk::Device device, const BindlessConfig &config);

    vk::Device Device;
    BindlessConfig Config;
    vk::UniqueDescriptorSetLayout SetLayout;
    vk::UniqueDescriptorPool DescriptorPool;
    vk::UniqueDescriptorSet DescriptorSet;
};

struct BindlessAllocator {
    BindlessAllocator(const BindlessResources &resources);

    uint32_t Allocate(SlotType type);
    void Release(SlotType type, uint32_t slot);

    vk::WriteDescriptorSet MakeBufferWrite(SlotType type, uint32_t slot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;
    vk::WriteDescriptorSet MakeUniformWrite(uint32_t slot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;

    vk::DescriptorSet GetSet() const { return *Resources.DescriptorSet; }
    const BindlessConfig &GetConfig() const { return Resources.Config; }

private:
    const BindlessResources &Resources;
    std::array<std::vector<uint32_t>, SlotTypeCount> FreeSlots;
};
