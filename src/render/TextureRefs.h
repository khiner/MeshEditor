#pragma once

#include "state/Entity.h"

#include <string>
#include <vector>

struct TextureRef {
    uint32_t SamplerSlot;
    std::string Name;
};
std::vector<TextureRef> GetTextureRefs(state::Scene &);

struct HdriRefs {
    std::vector<std::string> Names;
    uint32_t ActiveIndex;
};
HdriRefs GetHdriRefs(state::Scene &);
