#include "viewport/ViewportDisplay.h"

#include "state/Scene.h"

const PBRViewportLighting &GetActivePbrLighting(const state::Scene &r, state::Entity viewport, ViewportShadingMode mode) {
    return mode == ViewportShadingMode::Rendered ? static_cast<const PBRViewportLighting &>(r.get<const RenderedLighting>(viewport)) : static_cast<const PBRViewportLighting &>(r.get<const MaterialPreviewLighting>(viewport));
}
