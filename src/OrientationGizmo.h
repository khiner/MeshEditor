// Designed to look and behave just like Blender's orientation gizmo.
// imgui.h must be included before this header.

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
    ImU32 Hover{IM_COL32(120, 120, 120, 130)};
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
    auto *draw_list = ImGui::GetWindowDrawList();

    const auto mouse_pos_imgui = ImGui::GetIO().MousePos;
    const vec2 mouse_pos{mouse_pos_imgui.x, mouse_pos_imgui.y};
    const auto hover_circle_r = size * Scale.HoverCircleRadius;
    const auto center = pos + vec2{size, size} * 0.5f;
    Context.Hovered = glm::dot(mouse_pos - center, mouse_pos - center) <= hover_circle_r * hover_circle_r;
    if (Context.Hovered) draw_list->AddCircleFilled({center.x, center.y}, hover_circle_r, Color.Hover);

    // Project camera-relative axes to screen space.
    const mat3 transform = glm::transpose(camera.Basis());
    static vec3 axes[6];
    for (size_t i = 0; i < 6; ++i) {
        axes[i] = vec3{i < 3 ? transform[i] : -transform[i - 3]} * size * (.5f - Scale.CircleRadius);
        axes[i].y = -axes[i].y; // Flip for ImGui
    }

    static size_t AxisIndices[]{0, 1, 2, 3, 4, 5};
    // Sort axis based on z-depth in clip space, with farthest first.
    std::ranges::sort(AxisIndices, [](auto i, auto j) { return axes[i].z > axes[j].z; });

    // Find closest hovered axis.
    std::optional<size_t> hovered_i;
    if (Context.Hovered && !Context.DragEndPos) {
        hovered_i = std::ranges::min(AxisIndices, {}, [&](size_t i) {
            const auto mouse_delta = mouse_pos - (center + vec2{axes[i]});
            // Add z to avoid hovering (covered) back axes.
            return glm::dot(mouse_delta, mouse_delta) + axes[i].z;
        });
    }

    const bool is_aligned[]{camera.IsAligned(Axes[0]), camera.IsAligned(Axes[1]), camera.IsAligned(Axes[2])};

    // Draw back to front
    for (auto i : AxisIndices) {
        static constexpr auto ToImVec = [](vec2 v) { return ImVec2{v.x, v.y}; };
        const vec2 axis{axes[i]};
        const auto color = Color.Axes[i];
        const bool positive = i < 3;
        const float radius = size * Scale.CircleRadius;
        const auto line_end = ToImVec(center + axis);
        if (positive) draw_list->AddLine(ToImVec(center), line_end, color, 1.5);
        draw_list->AddCircleFilled(line_end, radius, color);
        if (hovered_i && *hovered_i == i) draw_list->AddCircle(line_end, radius, IM_COL32_WHITE, 20, 1.1f);
        else if (!positive) draw_list->AddCircle(line_end, radius, Color.Axes[i - 3], 20, 1.f);
        if (const bool selected = (hovered_i && *hovered_i == i);
            positive || selected || is_aligned[positive ? i : i - 3]) {
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
            camera.AddYawPitch(drag_delta * 0.01f);
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
