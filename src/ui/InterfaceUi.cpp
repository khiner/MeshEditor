#include "ui/InterfaceUi.h"
#include "action/View.h"
#include "render/SvgResource.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceTheme.h"

using namespace ImGui;

namespace ui {
// Applies ImGui's dark style with the window, frame, table and docking colors it had before ImGui 1.93.
void StyleColors() {
    StyleColorsDark();
    auto &colors = GetStyle().Colors;
    colors[ImGuiCol_TextDisabled] = {0.50f, 0.50f, 0.50f, 1.00f};
    colors[ImGuiCol_WindowBg] = {0.06f, 0.06f, 0.06f, 0.94f};
    colors[ImGuiCol_FrameBg] = {0.16f, 0.29f, 0.48f, 0.54f};
    colors[ImGuiCol_FrameBgHovered] = {0.26f, 0.59f, 0.98f, 0.40f};
    colors[ImGuiCol_FrameBgActive] = {0.26f, 0.59f, 0.98f, 0.67f};
    colors[ImGuiCol_DockingEmptyBg] = {0.20f, 0.20f, 0.20f, 1.00f};
    colors[ImGuiCol_TableHeaderBg] = {0.19f, 0.19f, 0.20f, 1.00f};
    colors[ImGuiCol_TableBorderStrong] = {0.31f, 0.31f, 0.35f, 1.00f};
    colors[ImGuiCol_TableBorderLight] = {0.23f, 0.23f, 0.25f, 1.00f};
}

void DrawIconButton(ImVec2 min, ImVec2 max, ImU32 color, float rounding, ImDrawFlags corners, const SvgResource *icon, float icon_dim) {
    // Leave a one-point seam between adjacent buttons in a group.
    GetWindowDrawList()->AddRectFilled(min, max - ImVec2{1, 1}, color, rounding, corners);
    if (!icon) return;
    const auto saved = GetCursorScreenPos();
    SetCursorScreenPos((min + max - ImVec2{icon_dim, icon_dim}) * 0.5f);
    icon->DrawIcon({icon_dim, icon_dim});
    SetCursorScreenPos(saved);
}

void DrawInterfaceStyle(state::Scene &r, state::Entity viewport) {
    if (Button("Reset")) action::Emit(action::view::ResetInterfaceTheme{});
    DrawThemeSizes<InterfaceTheme>(r, viewport);
}
} // namespace ui
