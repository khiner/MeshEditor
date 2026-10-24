#include "viewport/RenderExtent.h"

#include "state/Registry.h"
#include "viewport/FrameState.h"
#include "viewport/ViewportComponents.h"

uvec2 RenderExtentPx(const state::Registry &r, state::Entity viewport) {
    const auto logical_extent = r.get<const ViewportPanel>(viewport).Extent;
    const auto scale = r.Resources.get<const FrameState>().DisplayFramebufferScale;
    const auto scaled = [](uint32_t l, float s) -> uint32_t {
        if (l == 0u) return 0u;
        return std::max(1u, uint32_t(std::lround(float(l) * (s > 0 ? s : 1.f))));
    };
    return {scaled(logical_extent.x, scale.x), scaled(logical_extent.y, scale.y)};
}

float AspectRatio(uvec2 extent) { return extent.x == 0 || extent.y == 0 ? 1.f : float(extent.x) / float(extent.y); }
