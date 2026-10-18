#pragma once

#include "gpu/TextureInfo.h"

struct Specular {
    float Factor DEFAULT(1);
    vec3 ColorFactor DEFAULT(1);
    TextureInfo Texture DEFAULT();
    TextureInfo ColorTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Specular> Specs<Specular>{
    {&Specular::Factor, {.Help = "Specular reflection strength, multiplied by the texture's alpha channel"}},
    {&Specular::ColorFactor, {.Help = "Tint of specular reflection at normal incidence, multiplied by the color texture"}},
    {&Specular::Texture, {.Help = "Specular strength in the alpha channel"}},
    {&Specular::ColorTexture, {.Help = "sRGB specular color"}},
};
#endif
