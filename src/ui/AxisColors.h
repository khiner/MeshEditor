#pragma once

// imgui.h must be included before this header.

#include "gpu/AxisThemeColors.h"

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

// Build [X, Y, Z, -X, -Y, -Z] ImU32 array from Axis theme colors.
struct AxesArray {
    ImU32 Values[6];
    const ImU32 &operator[](size_t i) const { return Values[i]; }
};

inline AxesArray MakeAxes(const AxisThemeColors &a) {
    const auto x = RgbToU32(a.X), y = RgbToU32(a.Y), z = RgbToU32(a.Z);
    return {x, y, z, Darken(x, .3f), Darken(y, .3f), Darken(z, .3f)};
}
} // namespace colors
