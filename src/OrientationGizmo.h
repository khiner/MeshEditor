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
inline constexpr vec3 Signed(const mat3 &m, uint32_t i) { return i < 3 ? m[i] : -m[i - 3]; }

namespace internal {
struct Style {
    // Radii are relative to rect size
    float CircleRad{.095};
    float HoverCircleRad{.5};
};
struct Color {
    ImU32 Hover{IM_COL32(120, 120, 120, 130)};
};
struct Context {
    std::optional<vec2> MouseDownPos; // Present if mouse was pressed in hover circle.
    std::optional<vec2> DragEndPos; // Present if mouse was dragged past click threshold.
    bool Hovered;
};
} // namespace internal

static constexpr internal::Style Style;
static constexpr internal::Color Color;
static internal::Context Context;

bool IsActive() { return Context.Hovered || Context.MouseDownPos || Context.DragEndPos; }

void Draw(vec2 pos, float size, Camera &camera) {
    auto &dl = *ImGui::GetWindowDrawList();

    const auto mouse_pos = std::bit_cast<vec2>(ImGui::GetIO().MousePos);
    const auto hover_circle_r = size * Style.HoverCircleRad;
    const auto center = pos + vec2{size, size} * 0.5f;
    Context.Hovered = glm::dot(mouse_pos - center, mouse_pos - center) <= hover_circle_r * hover_circle_r;
    if (Context.Hovered || Context.DragEndPos) dl.AddCircleFilled({center.x, center.y}, hover_circle_r, Color.Hover);

    // Project camera-relative axes to screen space.
    const auto cam_transform = glm::transpose(camera.Basis());
    // [-1, 1], -1 faces camera
    const auto AxisCam = [&cam_transform](size_t i) -> vec3 { return Signed(cam_transform, i); };

    const auto AxisScreen = [&](size_t i) -> vec2 {
        auto dir = AxisCam(i) * size * (0.5f - Style.CircleRad);
        dir.y = -dir.y; // Flip for ImGui
        return dir;
    };

    // Sort axes by z-depth with farthest first.
    static size_t SortedAxisIndices[]{0, 1, 2, 3, 4, 5};
    std::ranges::sort(SortedAxisIndices, [&](auto i, auto j) { return AxisCam(i).z > AxisCam(j).z; });

    // Find closest hovered axis.
    std::optional<size_t> hovered_i;
    if (Context.Hovered && !Context.DragEndPos) {
        hovered_i = std::ranges::min(SortedAxisIndices, {}, [&](size_t i) {
            const auto mouse_delta = mouse_pos - (center + AxisScreen(i));
            // Add z to avoid hovering (covered) back axes.
            return glm::dot(mouse_delta, mouse_delta) + AxisCam(i).z;
        });
    }

    const auto IsAligned = [&camera](auto i) { return camera.IsAligned(Signed(I3, i)); };

    // Draw back to front
    for (auto i : SortedAxisIndices) {
        const auto t = 1.f - 0.5f * (AxisCam(i).z + 1); // [0, 1], 0 faces camera
        const bool positive = i < 3;
        const bool aligned = IsAligned(i);
        const auto fill_color = positive ?
            colors::Blend(colors::Axes[i + 3], colors::Axes[i], t) :
            aligned ? colors::Axes[i - 3] :
                      colors::WithAlpha(colors::Axes[i], t);
        const float r = size * Style.CircleRad;
        const auto line_end = std::bit_cast<ImVec2>(center + AxisScreen(i));
        if (positive) dl.AddLine(std::bit_cast<ImVec2>(center), line_end, fill_color, 1.5f);
        dl.AddCircleFilled(line_end, positive ? r : r - .5f, fill_color);
        if (!positive) { // Outline
            const auto color = aligned ? colors::Lighten(colors::Axes[i - 3], .8f) : colors::Blend(colors::Axes[i], colors::Axes[i - 3], t);
            dl.AddCircle(line_end, r, color, 32, 1.f);
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
            const auto click_threshold = 0.5f * size * Style.CircleRad;
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
            // If selecting the same axis, switch to the opposite axis.
            if (IsAligned(*hovered_i)) hovered_i = (*hovered_i + 3) % 6;
            camera.SetTargetDirection(Signed(I3, *hovered_i));
        }
        Context.MouseDownPos = std::nullopt;
        Context.DragEndPos = std::nullopt;
    }
}
} // namespace OrientationGizmo
