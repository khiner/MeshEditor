#pragma once

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Vulkan-free view of the texture store's slot/name pairs, so UI consumers that only
// populate texture pickers need not include the (vulkan-heavy) Textures.h.
struct TextureRef {
    uint32_t SamplerSlot;
    std::string Name;
};
std::vector<TextureRef> GetTextureRefs(entt::registry &);
