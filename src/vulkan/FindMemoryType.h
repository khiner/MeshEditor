#pragma once

#include <vulkan/vulkan.hpp>

uint32_t FindMemoryType(vk::PhysicalDevice, uint32_t, vk::MemoryPropertyFlags);
