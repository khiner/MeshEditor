// Started with https://github.com/fknfilewalker/imoguizmo and modified/simplified heavily.

// imgui must be included before this header.

#pragma once

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <algorithm>
#include <ranges>
#include <vector>

namespace OrientationGizmo {
namespace internal {
// Scales are in relation to the rect size.
struct Scale {
    float LineThickness{.017};
    float PositiveRadius{.075};
    float NegativeRadius{.05};
    float AxisLength{.5f - PositiveRadius};
    float HoverCircleRadius{.5};
};
struct Color {
    ImU32 XFront{IM_COL32(255, 54, 83, 255)}, XBack{IM_COL32(154, 57, 71, 255)};
    ImU32 YFront{IM_COL32(138, 219, 0, 255)}, YBack{IM_COL32(98, 138, 34, 255)};
    ImU32 ZFront{IM_COL32(44, 143, 255, 255)}, ZBack{IM_COL32(52, 100, 154, 255)};
    ImU32 Hover{IM_COL32(100, 100, 100, 130)};
};
} // namespace internal

static constexpr internal::Scale Scale;
static constexpr internal::Color Color;

bool Draw(vec2 pos, float size, mat4 &view, float pivot_distance) {
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
    auto view_proj = glm::scale(mat4{1}, vec3{1, -1, 1}) * (proj * view);
    const auto axes = view_proj * glm::scale(mat4{1}, vec3{size * Scale.AxisLength});
    const vec3 x_axis{axes[0]}, y_axis{axes[1]}, z_axis{axes[2]};

    // sort axis based on z-depth in clip space
    // 0 : +x axis, 1 : +y axis, 2 : +z axis, 3 : -x axis, 4 : -y axis, 5 : -z axis
    std::vector<std::pair<uint32_t, float>> pairs{
        {0, x_axis.z}, {1, y_axis.z}, {2, z_axis.z}, {3, -x_axis.z}, {4, -y_axis.z}, {5, -z_axis.z}
    };
    std::ranges::sort(pairs, [](const auto &a, const auto &b) { return a.second > b.second; });

    const auto xa = vec2{x_axis}, ya = vec2{y_axis}, za = vec2{z_axis};
    const float positive_r = size * Scale.PositiveRadius;
    const float negative_r = size * Scale.NegativeRadius;

    // find selection, front to back
    int selection = -1;
    for (const auto &pair : pairs) {
        switch (pair.first) {
            case 0: // +x axis
                if (MouseInCircle(center + xa, positive_r)) selection = 0;
                break;
            case 1: // +y axis
                if (MouseInCircle(center + ya, positive_r)) selection = 1;
                break;
            case 2: // +z axis
                if (MouseInCircle(center + za, positive_r)) selection = 2;
                break;
            case 3: // -x axis
                if (MouseInCircle(center - xa, negative_r)) selection = 3;
                break;
            case 4: // -y axis
                if (MouseInCircle(center - ya, negative_r)) selection = 4;
                break;
            case 5: // -z axis
                if (MouseInCircle(center - za, negative_r)) selection = 5;
                break;
            default: break;
        }
    }

    const auto DrawPositiveLine = [&center, &draw_list](vec2 axis, ImU32 color, float radius, float thickness, const char *text, bool selected) {
        const auto line_end_glm = center + axis;
        const ImVec2 line_end{line_end_glm.x, line_end_glm.y};
        draw_list->AddLine({center.x, center.y}, line_end, color, thickness);
        draw_list->AddCircleFilled(line_end, radius, color);
        const auto text_pos = line_end - ImGui::CalcTextSize(text) * 0.5f + ImVec2{0.5, 0.5};
        if (selected) {
            draw_list->AddCircle(line_end, radius, IM_COL32_WHITE, 0, 1.1f);
            draw_list->AddText(text_pos, IM_COL32_WHITE, text);
        } else {
            draw_list->AddText(text_pos, IM_COL32_BLACK, text);
        }
    };
    const auto DrawNegativeLine = [&center, &draw_list](vec2 axis, ImU32 color, float radius, bool selected) {
        const auto line_end = center - axis;
        draw_list->AddCircleFilled({line_end.x, line_end.y}, radius, color);
        if (selected) draw_list->AddCircle({line_end.x, line_end.y}, radius, IM_COL32_WHITE, 0, 1.1f);
    };

    const bool x_positive_closer = x_axis.z <= 0;
    const bool y_positive_closer = y_axis.z <= 0;
    const bool z_positive_closer = z_axis.z <= 0;
    // draw back first
    const float weight = size * Scale.LineThickness;
    for (const auto &pair : pairs) {
        switch (pair.first) {
            case 0: // +x axis
                DrawPositiveLine(xa, x_positive_closer ? Color.XFront : Color.XBack, positive_r, weight, "X", selection == 0);
                continue;
            case 1: // +y axis
                DrawPositiveLine(ya, y_positive_closer ? Color.YFront : Color.YBack, positive_r, weight, "Y", selection == 1);
                continue;
            case 2: // +z axis
                DrawPositiveLine(za, z_positive_closer ? Color.ZFront : Color.ZBack, positive_r, weight, "Z", selection == 2);
                continue;
            case 3: // -x axis
                DrawNegativeLine(xa, !x_positive_closer ? Color.XFront : Color.XBack, negative_r, selection == 3);
                continue;
            case 4: // -y axis
                DrawNegativeLine(ya, !y_positive_closer ? Color.YFront : Color.YBack, negative_r, selection == 4);
                continue;
            case 5: // -z axis
                DrawNegativeLine(za, !z_positive_closer ? Color.ZFront : Color.ZBack, negative_r, selection == 5);
                continue;
            default: break;
        }
    }

    if (selection != -1 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        static constexpr auto ViewMatrix = [](vec3 pos, vec3 right, vec3 up, vec3 forward) {
            return glm::transpose(mat4{
                {right, -glm::dot(right, pos)},
                {up, -glm::dot(up, pos)},
                {forward, -glm::dot(forward, pos)},
                {0, 0, 0, 1},
            });
        };

        const auto model_matrix = glm::inverse(view);
        // Right-handed
        const auto pivot_pos = vec3{model_matrix[3]} - vec3{model_matrix[2]} * pivot_distance;
        // Left-handed
        // const auto pivot_pos = ImVec3{&model_matrix[12]} + ImVec3{&model_matrix[8]} * pivot_distance;

        // +x axis
        if (selection == 0) view = ViewMatrix(pivot_pos + vec3{pivot_distance, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 0, 0});
        // +y axis
        if (selection == 1) view = ViewMatrix(pivot_pos + vec3{0, pivot_distance, 0}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0});
        // +z axis
        if (selection == 2) view = ViewMatrix(pivot_pos + vec3{0, 0, pivot_distance}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
        // -x axis
        if (selection == 3) view = ViewMatrix(pivot_pos - vec3{pivot_distance, 0, 0}, {0, 0, 1}, {0, 1, 0}, {-1, 0, 0});
        // -y axis
        if (selection == 4) view = ViewMatrix(pivot_pos - vec3{0, pivot_distance, 0}, {1, 0, 0}, {0, 0, 1}, {0, -1, 0});
        // -z axis
        if (selection == 5) view = ViewMatrix(pivot_pos - vec3{0, 0, pivot_distance}, {-1, 0, 0}, {0, 1, 0}, {0, 0, -1});

        return true;
    }

    return false;
}
} // namespace OrientationGizmo
