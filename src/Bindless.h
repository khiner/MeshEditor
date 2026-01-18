#pragma once

#include "vulkan/Slots.h"

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdint>
#include <vector>

struct DescriptorSlots {
    DescriptorSlots(vk::Device, const vk::PhysicalDeviceDescriptorIndexingProperties &);

    uint32_t Allocate(SlotType);
    void Release(TypedSlot);

    vk::WriteDescriptorSet MakeBufferWrite(TypedSlot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;
    vk::WriteDescriptorSet MakeUniformWrite(uint32_t slot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;

    vk::DescriptorSetLayout GetSetLayout() const { return *SetLayout; }
    vk::DescriptorSet GetSet() const { return *DescriptorSet; }

private:
    vk::Device Device;
    vk::UniqueDescriptorSetLayout SetLayout;
    vk::UniqueDescriptorPool DescriptorPool;
    vk::UniqueDescriptorSet DescriptorSet;
    std::array<std::vector<uint32_t>, SlotTypeCount> FreeSlots;
};
