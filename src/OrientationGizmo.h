// Designed to look and behave just like Blender's orientation gizmo.
// imgui.h must be included before this header.

#pragma once

#include "AxisColors.h"
#include "Camera.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <string_view>

namespace OrientationGizmo {

// [X, Y, Z, -X, -Y, -Z]
static constexpr vec3 Axes[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};

namespace internal {
// Scales are in relation to the rect size.
struct Scale {
    float CircleRadius{.095};
    float HoverCircleRadius{.5};
};
struct Color {
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
    auto &dl = *ImGui::GetWindowDrawList();

    const auto mouse_pos_imgui = ImGui::GetIO().MousePos;
    const vec2 mouse_pos{mouse_pos_imgui.x, mouse_pos_imgui.y};
    const auto hover_circle_r = size * Scale.HoverCircleRadius;
    const auto center = pos + vec2{size, size} * 0.5f;
    Context.Hovered = glm::dot(mouse_pos - center, mouse_pos - center) <= hover_circle_r * hover_circle_r;
    if (Context.Hovered || Context.DragEndPos) dl.AddCircleFilled({center.x, center.y}, hover_circle_r, Color.Hover);

    // Project camera-relative axes to screen space.
    const mat3 transform = glm::transpose(camera.Basis());

    // [-1, 1], -1 faces camera
    const auto AxisCam = [&transform](size_t i) -> vec3 { return i < 3 ? transform[i] : -transform[i - 3]; };

    const auto AxisScreen = [&](size_t i) -> vec2 {
        auto dir = AxisCam(i) * size * (0.5f - Scale.CircleRadius);
        dir.y = -dir.y; // Flip for ImGui
        return dir;
    };

    static size_t AxisIndices[]{0, 1, 2, 3, 4, 5};
    // Sort axis based on z-depth, with farthest first.
    std::ranges::sort(AxisIndices, [&](auto i, auto j) { return AxisCam(i).z > AxisCam(j).z; });

    // Find closest hovered axis.
    std::optional<size_t> hovered_i;
    if (Context.Hovered && !Context.DragEndPos) {
        hovered_i = std::ranges::min(AxisIndices, {}, [&](size_t i) {
            const auto mouse_delta = mouse_pos - (center + AxisScreen(i));
            // Add z to avoid hovering (covered) back axes.
            return glm::dot(mouse_delta, mouse_delta) + AxisCam(i).z;
        });
    }

    // const bool is_aligned[]{camera.IsAligned(Axes[0]), camera.IsAligned(Axes[1]), camera.IsAligned(Axes[2])};
    const auto IsAligned = [&camera](uint32_t i) { return camera.IsAligned(Axes[i]); };

    // Draw back to front
    for (auto i : AxisIndices) {
        static constexpr auto ToImVec = [](vec2 v) { return ImVec2{v.x, v.y}; };
        const auto facing = AxisCam(i).z; // [-1, 1], -1 faces camera
        const auto t = 1.f - .5f * (facing + 1); // map to [0, 1]
        const bool positive = i < 3;
        const bool aligned = IsAligned(i);
        const auto fill_color = positive ?
            colors::Blend(colors::Axes[i + 3], colors::Axes[i], t) :
            aligned ? colors::Axes[i - 3] :
                      colors::WithAlpha(colors::Axes[i], t);
        const float radius = size * Scale.CircleRadius;
        const auto line_end = ToImVec(center + AxisScreen(i));
        if (positive) dl.AddLine(ToImVec(center), line_end, fill_color, 1.5f);
        dl.AddCircleFilled(line_end, positive ? radius : radius - .5f, fill_color);
        // Outline
        if (!positive) {
            const auto color = aligned ? colors::Lighten(colors::Axes[i - 3], .8f) : colors::Blend(colors::Axes[i], colors::Axes[i - 3], t);
            dl.AddCircle(line_end, radius, color, 32, 1.f);
        }
        if (const bool hovered = hovered_i && *hovered_i == i;
            positive || hovered || aligned) {
            static constexpr std::string_view AxisLabels[]{"X", "Y", "Z", "-X", "-Y", "-Z"};
            const auto *label = AxisLabels[i].data();
            const auto text_pos = line_end - ImGui::CalcTextSize(label) * .5f + ImVec2{.5f, 0.f};
            dl.AddText(text_pos, hovered ? IM_COL32_WHITE : IM_COL32_BLACK, label);
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
            camera.SetTargetYawPitch(camera.GetYawPitch() + drag_delta * 0.02f);
        }
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        if (!Context.DragEndPos && hovered_i) { // Click
            // If selecting the same axis, select the opposite axis.
            if (IsAligned(*hovered_i)) hovered_i = *hovered_i < 3 ? *hovered_i + 3 : *hovered_i - 3;
            camera.SetTargetDirection(Axes[*hovered_i]);
        }
        Context.MouseDownPos = std::nullopt;
        Context.DragEndPos = std::nullopt;
    }
}
} // namespace OrientationGizmo
