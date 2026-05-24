#include "FindMemoryType.h"

uint32_t FindMemoryType(vk::PhysicalDevice pd, uint32_t type_filter, vk::MemoryPropertyFlags prop_flags) {
    auto mem_props = pd.getMemoryProperties();
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) return i;
    }
    throw std::runtime_error{std::format("Failed to find suitable memory type for type_filter: {}, prop_flags: {}", type_filter, uint32_t(prop_flags))};
}
