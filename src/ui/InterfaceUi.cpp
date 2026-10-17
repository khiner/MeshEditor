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

namespace {
struct {
    ImGuiAxis Axis;
    ImVec2 Size, Next;
    int Count, Index;
    float Rounding;
} Group;
} // namespace

void BeginIconButtonGroup(ImGuiAxis axis, ImVec2 size, int count, float rounding) {
    BeginGroup();
    Group = {axis, size, GetCursorScreenPos(), count, 0, rounding};
}

bool IconButton(const char *id, const SvgResource *icon, bool active, bool enabled, float length) {
    auto size = Group.Size;
    if (length > 0) size[Group.Axis] = length;
    SetCursorScreenPos(Group.Next);
    if (!enabled) BeginDisabled();
    const bool clicked = InvisibleButton(id, size);
    if (!enabled) EndDisabled();
    const auto min = GetItemRectMin(), max = GetItemRectMax();
    const auto color = GetColorU32(!enabled ? ImGuiCol_FrameBg : active ? ImGuiCol_ButtonActive :
                                       IsItemHovered()                  ? ImGuiCol_ButtonHovered :
                                                                          ImGuiCol_Button);
    const bool first = Group.Index == 0, last = Group.Index == Group.Count - 1;
    const bool horizontal = Group.Axis == ImGuiAxis_X;
    const ImDrawFlags corners = !first && !last ? ImDrawFlags_RoundNone :
                                                  (first ? (horizontal ? ImDrawFlags_RoundLeft : ImDrawFlags_RoundTop) : 0) | (last ? (horizontal ? ImDrawFlags_RoundRight : ImDrawFlags_RoundBottom) : 0);
    GetWindowDrawList()->AddRectFilled(min, max, color, Group.Rounding, corners);
    if (icon) {
        static constexpr float IconScale{0.75f};
        const auto center = (min + max) * 0.5f, half = ImVec2{size.y, size.y} * IconScale * 0.5f;
        icon->DrawIcon(std::bit_cast<vec2>(center - half), std::bit_cast<vec2>(center + half));
    }
    Group.Next[Group.Axis] += size[Group.Axis] + IconButtonSeam;
    ++Group.Index;
    return clicked;
}

void EndIconButtonGroup() {
    IM_ASSERT(Group.Index == Group.Count);
    EndGroup();
}

void DropdownArrow() {
    const auto center = (GetItemRectMin() + GetItemRectMax()) * 0.5f;
    constexpr float HalfWidth{3.5f};
    GetWindowDrawList()->AddTriangleFilled(center - ImVec2{HalfWidth, HalfWidth * 0.5f}, center + ImVec2{HalfWidth, -HalfWidth * 0.5f}, center + ImVec2{0.f, HalfWidth * 0.5f}, GetColorU32(ImGuiCol_Text));
}

void DrawInterfaceStyle(state::Scene &r, state::Entity viewport) {
    if (Button("Reset")) action::Emit(action::view::ResetInterfaceTheme{});
    SeparatorText("Sizes");
    DrawThemeSizes<InterfaceTheme>(r, viewport);
}
} // namespace ui
