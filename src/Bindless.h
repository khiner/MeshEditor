#pragma once

#include "vulkan/Slots.h"

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
            case SlotType::ModelBuffer:
            case SlotType::ObjectIdBuffer:
            case SlotType::FaceNormalBuffer: return MaxBuffers;
        }
    }
};

struct DescriptorSlots {
    DescriptorSlots(vk::Device, const BindlessConfig &);

    uint32_t Allocate(SlotType);
    void Release(TypedSlot);

    vk::WriteDescriptorSet MakeBufferWrite(TypedSlot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;
    vk::WriteDescriptorSet MakeUniformWrite(uint32_t slot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;

    vk::DescriptorSetLayout GetSetLayout() const { return *SetLayout; }
    vk::DescriptorSet GetSet() const { return *DescriptorSet; }
    const BindlessConfig &GetConfig() const { return Config; }

private:
    vk::Device Device;
    BindlessConfig Config;
    vk::UniqueDescriptorSetLayout SetLayout;
    vk::UniqueDescriptorPool DescriptorPool;
    vk::UniqueDescriptorSet DescriptorSet;
    std::array<std::vector<uint32_t>, SlotTypeCount> FreeSlots;
};
