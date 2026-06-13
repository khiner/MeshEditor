#include "viewport/RenderExtent.h"

#include <imgui.h>

#include <algorithm>

uvec2 RenderExtentPx(uvec2 logical_extent) {
    const auto scale = ImGui::GetIO().DisplayFramebufferScale;
    const auto scaled = [](uint32_t l, float s) -> uint32_t {
        if (l == 0u) return 0u;
        return std::max(1u, uint32_t(float(l) * (s > 0 ? s : 1.f) + 0.5f));
    };
    return {scaled(logical_extent.x, scale.x), scaled(logical_extent.y, scale.y)};
}
