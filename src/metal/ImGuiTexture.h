#pragma once

#include <imgui.h>

namespace MTL {
class Texture;
} // namespace MTL

namespace mtl {
inline ImTextureID ImGuiTextureId(MTL::Texture *texture) { return reinterpret_cast<ImTextureID>(texture); }
} // namespace mtl
