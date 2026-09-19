#pragma once

#include <imgui.h>

#include <algorithm>
#include <cmath>

// A dashed line from `a` to `b`, stroked without ImDrawList::AddLine's half-pixel offset.
inline void DrawDashedLine(ImDrawList &dl, ImVec2 a, ImVec2 b, ImU32 color, float thickness = 1.f) {
    static constexpr float DashLen{4}, GapLen{3};

    const auto dir = b - a;
    const float len = sqrtf(dir.x * dir.x + dir.y * dir.y);
    if (len <= 1e-3f) return;

    const auto dir_unit = dir / len;
    for (float t = 0; t < len; t += DashLen + GapLen) {
        dl.PathLineTo(a + dir_unit * t);
        dl.PathLineTo(a + dir_unit * (t + std::min(DashLen, len - t)));
        dl.PathStroke(color, 0, thickness);
    }
}

// The dashed guide from a mouse-driven operator's center to the mouse, shadowed for contrast on any background.
inline void DrawDashedGuide(ImDrawList &dl, ImVec2 center_px, ImVec2 mouse_px) {
    static constexpr auto LineColor{IM_COL32(255, 255, 255, 255)}, ShadowColor{IM_COL32(90, 90, 90, 200)};
    static constexpr ImVec2 ShadowOffset{1.5, 1.5};
    DrawDashedLine(dl, center_px + ShadowOffset, mouse_px + ShadowOffset, ShadowColor);
    DrawDashedLine(dl, center_px, mouse_px, LineColor);
}
