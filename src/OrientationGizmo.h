// Started with https://github.com/fknfilewalker/imoguizmo and modified/simplified heavily.

// imgui must be included before this header.

#pragma once

#include "Camera.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <string_view>

namespace OrientationGizmo {

// Axis order: [+x, +y, +z, -x, -y, -z]
static constexpr vec3 Axes[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};

namespace internal {
// Scales are in relation to the rect size.
struct Scale {
    float CircleRadius{.095};
    float AxisLength{.5f - CircleRadius};
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
struct Context {
    std::optional<vec2> MouseDownPos; // Only present if mouse was pressed in hover circle.
    std::optional<vec2> DragEndPos; // Only present if mouse was dragged past click threshold.
    bool Hovered;
};
} // namespace internal

static constexpr internal::Scale Scale;
static constexpr internal::Color Color;
static internal::Context Context;

bool IsActive() { return Context.Hovered || Context.MouseDownPos || Context.DragEndPos; }

void Draw(vec2 pos, float size, Camera &camera) {
    static const mat4 proj = glm::ortho(-1, 1, -1, 1, -1, 1);
    auto *draw_list = ImGui::GetWindowDrawList();

    const auto mouse_pos_imgui = ImGui::GetIO().MousePos;
    const vec2 mouse_pos{mouse_pos_imgui.x, mouse_pos_imgui.y};
    const auto MouseInCircle = [&mouse_pos](vec2 center, float r) {
        return glm::dot(mouse_pos - center, mouse_pos - center) <= r * r;
    };

    const auto hover_circle_r = size * Scale.HoverCircleRadius;
    const auto center = pos + vec2{size, size} * 0.5f;
    Context.Hovered = MouseInCircle(center, hover_circle_r);
    if (Context.Hovered) draw_list->AddCircleFilled({center.x, center.y}, hover_circle_r, Color.Hover);

    // Flip Y: ImGui uses top-left origin, and glm is bottom-left.
    const auto view = camera.GetView();
    const auto view_proj = glm::scale(mat4{1}, vec3{1, -1, 1}) * (proj * view);
    const auto axes_proj = view_proj * glm::scale(mat4{1}, vec3{size * Scale.AxisLength});
    const vec3 axes[]{axes_proj[0], axes_proj[1], axes_proj[2], -axes_proj[0], -axes_proj[1], -axes_proj[2]};
    // Sort axis based on z-depth in clip space.
    size_t depth_order[]{0, 1, 2, 3, 4, 5};
    std::ranges::sort(depth_order, [&axes](auto i, auto j) { return axes[i].z < axes[j].z; });

    // Find first hovered.
    std::optional<size_t> hovered_i;
    if (Context.Hovered) {
        const auto it = std::ranges::find_if(depth_order, [&](size_t i) {
            return MouseInCircle(center + vec2{axes[i]}, size * Scale.CircleRadius);
        });
        hovered_i = it != std::ranges::end(depth_order) ? std::optional{*it} : std::nullopt;
    }

    const bool is_aligned[3]{camera.IsAligned(Axes[0]), camera.IsAligned(Axes[1]), camera.IsAligned(Axes[2])};

    // Draw back to front
    for (auto i : std::views::reverse(depth_order)) {
        static constexpr auto ToImVec = [](vec2 v) { return ImVec2{v.x, v.y}; };
        const vec2 axis{axes[i]};
        const auto color = Color.Axes[i];
        const bool is_positive = i < 3;
        const float radius = size * Scale.CircleRadius;
        const auto line_end = ToImVec(center + axis);
        if (is_positive) draw_list->AddLine(ToImVec(center), line_end, color, 1.5);
        draw_list->AddCircleFilled(line_end, radius, color);
        if (hovered_i && *hovered_i == i) draw_list->AddCircle(line_end, radius, IM_COL32_WHITE, 20, 1.1f);
        else if (!is_positive) draw_list->AddCircle(line_end, radius, Color.Axes[i - 3], 20, 1.f);
        if (const bool selected = (hovered_i && *hovered_i == i);
            is_positive || selected || is_aligned[is_positive ? i : i - 3]) {
            static constexpr std::string_view AxisLabels[]{"X", "Y", "Z", "-X", "-Y", "-Z"};
            const auto *label = AxisLabels[i].data();
            const auto text_pos = line_end - ImGui::CalcTextSize(label) * 0.5f + ImVec2{0.5, 0};
            draw_list->AddText(text_pos, selected ? IM_COL32_WHITE : IM_COL32_BLACK, label);
        }
    }

    if (Context.Hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        Context.MouseDownPos = mouse_pos;
    } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && Context.MouseDownPos) {
        if (!Context.DragEndPos) {
            // Should we transition to dragging mode?
            // Click threshold is an arbitrary smaller amount than a hovered circle,
            // since we don't want to wait for that long of a drag to switch into drag behavior.
            const auto click_threshold = size * Scale.CircleRadius / 2;
            const auto mouse_delta = mouse_pos - *Context.MouseDownPos;
            if (glm::dot(mouse_delta, mouse_delta) > click_threshold * click_threshold) {
                Context.DragEndPos = mouse_pos;
            }
        } else { // Dragging
            const auto drag_delta = mouse_pos - *Context.DragEndPos;
            Context.DragEndPos = mouse_pos;
            camera.OrbitDelta(vec2{-drag_delta.x, drag_delta.y} * 0.01f);
        }
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        if (!Context.DragEndPos && hovered_i) { // Click
            // If selecting the same axis, select the opposite axis.
            if (is_aligned[*hovered_i > 2 ? *hovered_i - 3 : *hovered_i]) {
                hovered_i = *hovered_i < 3 ? *hovered_i + 3 : *hovered_i - 3;
            }
            camera.SetTargetDirection(Axes[*hovered_i]);
        }
        Context.MouseDownPos = std::nullopt;
        Context.DragEndPos = std::nullopt;
    }
}
} // namespace OrientationGizmo
