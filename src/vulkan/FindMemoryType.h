#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>

uint32_t FindMemoryType(vk::PhysicalDevice, uint32_t, vk::MemoryPropertyFlags);
