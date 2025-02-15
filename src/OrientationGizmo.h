// Started with https://github.com/fknfilewalker/imoguizmo and modified/simplified heavily.

// imgui must be included before this header.

#pragma once

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <string_view>

// Axis order: [+x, +y, +z, -x, -y, -z]

namespace OrientationGizmo {
namespace internal {
// Scales are in relation to the rect size.
struct Scale {
    float LineWidth{.017};
    float PositiveRadius{.075};
    float NegativeRadius{.05};
    float AxisLength{.5f - PositiveRadius};
    float HoverCircleRadius{.5};
};
struct Color {
    ImU32 Axes[6]{
        IM_COL32(255, 54, 83, 255),
        IM_COL32(138, 219, 0, 255),
        IM_COL32(44, 143, 255, 255),
        IM_COL32(154, 57, 71, 255),
        IM_COL32(98, 138, 34, 255),
        IM_COL32(52, 100, 154, 255),
    };
    ImU32 Hover{IM_COL32(100, 100, 100, 130)};
};
} // namespace internal

static constexpr internal::Scale Scale;
static constexpr internal::Color Color;

// Returns orientation direction if clicked.
std::optional<vec3> Draw(vec2 pos, float size, const mat4 &view) {
    static const mat4 proj = glm::ortho(-1, 1, -1, 1, -1, 1);
    auto *draw_list = ImGui::GetWindowDrawList();

    const auto mouse_pos_imgui = ImGui::GetIO().MousePos;
    const vec2 mouse_pos{mouse_pos_imgui.x, mouse_pos_imgui.y};
    const auto MouseInCircle = [&mouse_pos](vec2 center, float r) { return glm::dot(mouse_pos - center, mouse_pos - center) <= r * r; };

    const auto hover_circle_r = size * Scale.HoverCircleRadius;
    const auto center = pos + vec2{size, size} * 0.5f;
    if (MouseInCircle(center, hover_circle_r)) {
        draw_list->AddCircleFilled({center.x, center.y}, hover_circle_r, Color.Hover);
    }

    // Flip Y: ImGui uses top-left origin, and glm is bottom-left.
    const auto view_proj = glm::scale(mat4{1}, vec3{1, -1, 1}) * (proj * view);
    const auto axes_proj = view_proj * glm::scale(mat4{1}, vec3{size * Scale.AxisLength});
    const vec3 axes[]{axes_proj[0], axes_proj[1], axes_proj[2], -axes_proj[0], -axes_proj[1], -axes_proj[2]};
    // Sort axis based on z-depth in clip space.
    size_t depth_order[]{0, 1, 2, 3, 4, 5};
    std::ranges::sort(depth_order, [&axes](auto i, auto j) { return axes[i].z < axes[j].z; });

    // Find first hovered.
    const auto hovered_it = std::ranges::find_if(depth_order, [&](size_t i) {
        return MouseInCircle(center + vec2{axes[i]}, size * (i < 3 ? Scale.PositiveRadius : Scale.NegativeRadius));
    });
    std::optional<size_t> hovered_i = hovered_it != std::ranges::end(depth_order) ? std::optional{*hovered_it} : std::nullopt;

    // Draw back to front
    for (auto i : std::views::reverse(depth_order)) {
        static constexpr auto ToImVec = [](vec2 v) { return ImVec2{v.x, v.y}; };
        const vec2 axis{axes[i]};
        const auto color = Color.Axes[i];
        const bool is_positive = i < 3;
        const bool selected = hovered_i && *hovered_i == i;
        const float radius = size * (is_positive ? Scale.PositiveRadius : Scale.NegativeRadius);
        const auto line_end = ToImVec(center + axis);
        if (is_positive) draw_list->AddLine(ToImVec(center), line_end, color, size * Scale.LineWidth);
        draw_list->AddCircleFilled(line_end, radius, color);
        if (selected) draw_list->AddCircle(line_end, radius, IM_COL32_WHITE, 0, 1.1f);
        if (is_positive) {
            static constexpr std::string_view AxisLabels[]{"X", "Y", "Z", "", "", ""};
            const auto *label = AxisLabels[i].data();
            const auto text_pos = line_end - ImGui::CalcTextSize(label) * 0.5f + ImVec2{0.5, 0.5};
            draw_list->AddText(text_pos, selected ? IM_COL32_WHITE : IM_COL32_BLACK, label);
        }
    }

    if (hovered_i && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        static constexpr vec3 Axes[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
        return Axes[*hovered_i];
    }

    return {};
}
} // namespace OrientationGizmo
