// Designed to look and behave just like Blender's orientation gizmo.
// imgui.h must be included before this header.

#pragma once

#include "AxisColors.h"
#include "ViewCamera.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <string_view>

namespace OrientationGizmo {
// Radii are relative to rect size.
static constexpr float CircleRad = .095f, HoverCircleRad = .5f;

inline constexpr vec3 SignedAxis(const mat3 &m, uint32_t i) { return i < 3 ? m[i] : -m[i - 3]; }

struct Context {
    std::optional<vec2> MouseDownPos; // Present if mouse was pressed in hover circle.
    std::optional<vec2> DragEndPos; // Present if mouse was dragged past click threshold.
    bool Hovered;
    std::optional<size_t> HoveredAxis;
    // Cached per-frame layout computed by Interact, consumed by Render.
    vec2 Center;
    float Size;
    vec3 AxisCam[6]; // Camera-space axis directions
    vec2 AxisScreen[6]; // Screen-space axis endpoints (relative to center)
    size_t SortedIndices[6]{0, 1, 2, 3, 4, 5};
    bool Aligned[6];
};
static Context Ctx;

bool IsActive() { return Ctx.Hovered || Ctx.MouseDownPos || Ctx.DragEndPos; }

void Interact(vec2 pos, float size, ViewCamera &camera, bool interactive = true) {
    const auto mouse_pos = std::bit_cast<vec2>(ImGui::GetMousePos());
    const auto center = pos + vec2{size, size} * 0.5f;
    Ctx.Center = center;
    Ctx.Size = size;
    Ctx.HoveredAxis = std::nullopt;

    if (!interactive) {
        Ctx.Hovered = false;
        Ctx.MouseDownPos = std::nullopt;
        Ctx.DragEndPos = std::nullopt;
    }
    const auto hover_r = size * HoverCircleRad;
    Ctx.Hovered = interactive && glm::dot(mouse_pos - center, mouse_pos - center) <= hover_r * hover_r;

    // Precompute per-axis layout for both Interact and Render.
    const auto cam_basis = glm::transpose(camera.Basis());
    for (size_t i = 0; i < 6; ++i) {
        Ctx.AxisCam[i] = SignedAxis(cam_basis, i);
        auto dir = Ctx.AxisCam[i] * size * (0.5f - CircleRad);
        dir.y = -dir.y; // Flip for ImGui
        Ctx.AxisScreen[i] = dir;
        Ctx.Aligned[i] = camera.IsAligned(SignedAxis(I3, i));
    }
    std::ranges::sort(Ctx.SortedIndices, [](auto i, auto j) { return Ctx.AxisCam[i].z > Ctx.AxisCam[j].z; });

    // Find closest hovered axis.
    if (interactive && Ctx.Hovered && !Ctx.DragEndPos) {
        Ctx.HoveredAxis = std::ranges::min(Ctx.SortedIndices, {}, [&](size_t i) {
            const auto mouse_delta = mouse_pos - (center + Ctx.AxisScreen[i]);
            return glm::dot(mouse_delta, mouse_delta) + Ctx.AxisCam[i].z;
        });
    }

    if (interactive) {
        if (Ctx.Hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            Ctx.MouseDownPos = mouse_pos;
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && Ctx.MouseDownPos) {
            if (!Ctx.DragEndPos) {
                // Click threshold is an arbitrary smaller amount than a hovered circle,
                // since we don't want to wait for that long of a drag to switch into drag behavior.
                const auto click_threshold = 0.5f * size * CircleRad;
                const auto mouse_delta = mouse_pos - *Ctx.MouseDownPos;
                if (glm::dot(mouse_delta, mouse_delta) > click_threshold * click_threshold) {
                    Ctx.DragEndPos = mouse_pos;
                }
            } else { // Dragging
                const auto drag_delta = mouse_pos - *Ctx.DragEndPos;
                Ctx.DragEndPos = mouse_pos;
                camera.SetTargetYawPitch(camera.YawPitch + drag_delta * 0.02f);
            }
        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            auto hovered_i = Ctx.HoveredAxis;
            if (!Ctx.DragEndPos && hovered_i) {
                // If selecting the same axis, switch to the opposite axis.
                if (Ctx.Aligned[*hovered_i]) hovered_i = (*hovered_i + 3) % 6;
                camera.SetTargetDirection(SignedAxis(I3, *hovered_i));
            }
            Ctx.MouseDownPos = std::nullopt;
            Ctx.DragEndPos = std::nullopt;
        }
    }
}

void Render(const colors::AxesArray &Axes) {
    auto &dl = *ImGui::GetWindowDrawList();
    const auto center = Ctx.Center;
    const auto size = Ctx.Size;

    if (Ctx.Hovered || Ctx.DragEndPos) {
        dl.AddCircleFilled({center.x, center.y}, size * HoverCircleRad, IM_COL32(120, 120, 120, 130));
    }

    // Draw back to front
    for (auto i : Ctx.SortedIndices) {
        const auto t = 1.f - 0.5f * (Ctx.AxisCam[i].z + 1); // [0, 1], 0 faces camera
        const bool positive = i < 3;
        const bool aligned = Ctx.Aligned[i];
        const auto fill_color = positive ? colors::Blend(Axes[i + 3], Axes[i], t) :
            aligned                      ? Axes[i - 3] :
                                           colors::WithAlpha(Axes[i], t);
        const float r = size * CircleRad;
        const auto line_end = std::bit_cast<ImVec2>(center + Ctx.AxisScreen[i]);
        if (positive) dl.AddLine(std::bit_cast<ImVec2>(center), line_end, fill_color, 2.f);
        dl.AddCircleFilled(line_end, positive ? r : r - .5f, fill_color);
        if (!positive) { // Outline
            const auto color = aligned ? colors::Lighten(Axes[i - 3], .8f) : colors::Blend(Axes[i], Axes[i - 3], t);
            dl.AddCircle(line_end, r, color, 32, 1.f);
        }
        if (const bool hovered = Ctx.HoveredAxis && *Ctx.HoveredAxis == i;
            positive || hovered || aligned) {
            static constexpr std::string_view AxisLabels[]{"X", "Y", "Z", "-X", "-Y", "-Z"};
            auto *const label = AxisLabels[i].data();
            auto *const font = ImGui::GetFont();
            const float font_size = r * 1.5f;
            const auto text_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, label);
            const auto text_pos = line_end - text_size * .5f + ImVec2{.5f, 0.f};
            dl.AddText(font, font_size, text_pos, hovered ? IM_COL32_WHITE : IM_COL32_BLACK, label);
        }
    }
}
} // namespace OrientationGizmo
