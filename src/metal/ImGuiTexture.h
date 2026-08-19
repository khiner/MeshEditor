#pragma once

#include "metal/MetalCpp.h"

#include <imgui.h>

namespace mtl {
// The ImGui Metal backend takes the texture itself as the identifier.
inline ImTextureID ImGuiTextureId(MTL::Texture *texture) { return reinterpret_cast<ImTextureID>(texture); }
} // namespace mtl
