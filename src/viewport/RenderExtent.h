#pragma once

#include "numeric/vec2.h"

#include <vulkan/vulkan.hpp>

// Physical pixel render extent for a viewport's logical size, scaled by the current DPI framebuffer scale.
vk::Extent2D RenderExtentPx(uvec2 logical_extent);
