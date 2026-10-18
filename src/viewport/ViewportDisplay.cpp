#include "viewport/ViewportDisplay.h"

#include "state/Registry.h"

const PBRViewportLighting &GetActivePbrLighting(const state::Registry &r, state::Entity viewport, ViewportShadingMode mode) {
    return mode == ViewportShadingMode::Rendered ? r.get<const RenderedLighting>(viewport).Value : r.get<const MaterialPreviewLighting>(viewport).Value;
}
