#include "ViewportDisplay.h"

#include <entt/entity/registry.hpp>

const PBRViewportLighting &GetActivePbrLighting(const entt::registry &r, entt::entity viewport, ViewportShadingMode mode) {
    return mode == ViewportShadingMode::Rendered ? static_cast<const PBRViewportLighting &>(r.get<const RenderedLighting>(viewport)) : static_cast<const PBRViewportLighting &>(r.get<const MaterialPreviewLighting>(viewport));
}
