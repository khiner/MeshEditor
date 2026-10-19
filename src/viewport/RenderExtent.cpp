#include "viewport/RenderExtent.h"

#include "viewport/FrameState.h"
#include "viewport/Viewport.h"

uvec2 RenderExtentPx(const Viewport &viewport, const FrameState &frame) {
    const auto logical_extent = viewport.State.Extent;
    const auto scale = frame.DisplayFramebufferScale;
    const auto scaled = [](uint32_t l, float s) -> uint32_t {
        if (l == 0u) return 0u;
        return std::max(1u, uint32_t(std::lround(float(l) * (s > 0 ? s : 1.f))));
    };
    return {scaled(logical_extent.x, scale.x), scaled(logical_extent.y, scale.y)};
}
