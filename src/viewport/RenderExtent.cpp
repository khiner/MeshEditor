#include "viewport/RenderExtent.h"

#include "viewport/FrameState.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

#include <algorithm>

uvec2 RenderExtentPx(const entt::registry &r) {
    const auto logical_extent = r.ctx().get<const ViewportExtent>().Value;
    const auto scale = r.ctx().get<const FrameState>().DisplayFramebufferScale;
    const auto scaled = [](uint32_t l, float s) -> uint32_t {
        if (l == 0u) return 0u;
        return std::max(1u, uint32_t(float(l) * (s > 0 ? s : 1.f) + 0.5f));
    };
    return {scaled(logical_extent.x, scale.x), scaled(logical_extent.y, scale.y)};
}
