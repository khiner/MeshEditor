#pragma once

#include "state/Entity.h"

#include <vector>

struct TextureRef {
    uint32_t SamplerSlot;
    std::string Name;
};
std::vector<TextureRef> GetTextureRefs(state::Registry &);

struct HdriRefs {
    std::vector<std::string> Names;
    uint32_t ActiveIndex; // The selected HDRI, or the first when no HDRI has its name.
};
HdriRefs GetHdriRefs(state::Registry &, std::string_view selected);
