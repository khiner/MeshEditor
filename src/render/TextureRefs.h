#pragma once

#include <entt/entity/fwd.hpp>

#include <string>
#include <vector>

struct TextureRef {
    uint32_t SamplerSlot;
    std::string Name;
};
std::vector<TextureRef> GetTextureRefs(entt::registry &);

struct HdriRefs {
    std::vector<std::string> Names;
    uint32_t ActiveIndex;
};
HdriRefs GetHdriRefs(entt::registry &);
