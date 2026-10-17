#pragma once

#include "numeric/vec4.h"
#include "state/Entity.h"

#include "imgui.h"

struct SvgResource;

namespace ui {
inline ImU32 U32(numeric::vec4 c) { return ImGui::ColorConvertFloat4ToU32({c.x, c.y, c.z, c.w}); }
// Fills an icon button's background over [min, max] inset by half a pixel, and centers `icon` at `icon_dim` pixels square.
void DrawIconButton(ImVec2 min, ImVec2 max, ImU32 color, float rounding, ImDrawFlags corners, const SvgResource *icon = nullptr, float icon_dim = 0);
void DrawInterfaceStyle(state::Scene &, state::Entity viewport);
} // namespace ui
