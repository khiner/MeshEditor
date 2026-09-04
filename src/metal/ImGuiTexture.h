#pragma once

#include "metal/MetalCpp.h"

#include <imgui.h>

namespace mtl {
inline ImTextureID ImGuiTextureId(MTL::Texture *texture) { return reinterpret_cast<ImTextureID>(texture); }
} // namespace mtl
