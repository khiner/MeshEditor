#pragma once

// imgui.h must be included before this header.

namespace colors {

constexpr ImU32 Lighten(ImU32 color, float amount) {
    const ImVec4 rgba{ImColor{color}};
    return ImColor{
        std::lerp(rgba.x, 1.f, amount),
        std::lerp(rgba.y, 1.f, amount),
        std::lerp(rgba.z, 1.f, amount),
        rgba.w
    };
}

constexpr ImU32 Darken(ImU32 color, float amount) {
    const ImVec4 rgba{ImColor{color}};
    return ImColor{
        std::lerp(rgba.x, .0f, amount),
        std::lerp(rgba.y, .0f, amount),
        std::lerp(rgba.z, .0f, amount),
        rgba.w
    };
}

constexpr ImU32 WithAlpha(ImU32 color, float alpha) {
    const ImVec4 rgba{ImColor{color}};
    return ImColor{rgba.x, rgba.y, rgba.z, alpha};
}

constexpr ImU32 MultAlpha(ImU32 color, float factor) {
    const ImVec4 rgba{ImColor{color}};
    return ImColor{rgba.x, rgba.y, rgba.z, rgba.w * factor};
}

constexpr ImU32 Blend(ImU32 color1, ImU32 color2, float factor) {
    const ImVec4 rgba1{ImColor{color1}};
    const ImVec4 rgba2{ImColor{color2}};
    return ImColor{
        std::lerp(rgba1.x, rgba2.x, factor),
        std::lerp(rgba1.y, rgba2.y, factor),
        std::lerp(rgba1.z, rgba2.z, factor),
        std::lerp(rgba1.w, rgba2.w, factor)
    };
}

constexpr ImU32 RgbToU32(const vec3 in) { return ImGui::ColorConvertFloat4ToU32({in.x, in.y, in.z, 1.f}); }

// Blender:Themes:Axis & Gizmo Colors
constexpr ImU32 AxisX IM_COL32(255, 51, 82, 255);
constexpr ImU32 AxisY IM_COL32(139, 220, 0, 255);
constexpr ImU32 AxisZ IM_COL32(40, 144, 255, 255);
// [X, Y, Z, -X, -Y, -Z]
static const ImU32 Axes[]{AxisX, AxisY, AxisZ, Darken(AxisX, .3f), Darken(AxisY, .3f), Darken(AxisZ, .3f)};
} // namespace colors
