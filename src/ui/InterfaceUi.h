#pragma once

#include "numeric/vec4.h"
#include "state/Entity.h"

#include "imgui.h"
#include "imgui_internal.h"

struct SvgResource;

namespace ui {
inline ImU32 U32(numeric::vec4 c) { return ImGui::ColorConvertFloat4ToU32({c.x, c.y, c.z, c.w}); }
void StyleColors();
inline constexpr float IconButtonSeam{1};
// Lays out the next `count` icon buttons edge to edge along `axis` at `size` each, with an `IconButtonSeam` between neighbors and `rounding` on the group's outer corners.
void BeginIconButtonGroup(ImGuiAxis axis, ImVec2 size, int count, float rounding);
// Returns true when clicked. `length` overrides the button's extent along the group axis.
bool IconButton(const char *id, const SvgResource *icon, bool active = false, bool enabled = true, float length = 0);
void EndIconButtonGroup();
// Draws a dropdown arrow centered in the last item.
void DropdownArrow();
void DrawInterfaceStyle(state::Registry &);
} // namespace ui
