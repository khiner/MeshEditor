#pragma once

#include "RangeAllocator.h"
#include "vulkan/Slots.h"

#include <array>
#include <vulkan/vulkan.hpp>

struct DescriptorSlots {
    DescriptorSlots(vk::Device, const vk::PhysicalDeviceDescriptorIndexingProperties &);

    uint32_t Allocate(SlotType);
    // Reserve a specific slot: removes it from the free list if present. Returns true if it was free (now
    // reserved), false if it was already allocated. Used on snapshot restore to re-acquire the exact slots
    // baked into restored state, idempotently (a no-op when the slot is already live, e.g. during import).
    bool Reserve(SlotType, uint32_t slot);
    void Release(TypedSlot);

    vk::WriteDescriptorSet MakeBufferWrite(TypedSlot, const vk::DescriptorBufferInfo &) const;
    vk::WriteDescriptorSet MakeImageWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;
    vk::WriteDescriptorSet MakeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;
    vk::WriteDescriptorSet MakeCubeSamplerWrite(uint32_t slot, const vk::DescriptorImageInfo &) const;

    vk::DescriptorSetLayout GetSetLayout() const { return *SetLayout; }
    vk::DescriptorSet GetSet() const { return *DescriptorSet; }

private:
    vk::Device Device;
    vk::UniqueDescriptorSetLayout SetLayout;
    vk::UniqueDescriptorPool DescriptorPool;
    vk::UniqueDescriptorSet DescriptorSet;
    std::array<RangeAllocator, SlotTypeCount> Allocators; // one per slot type
};
