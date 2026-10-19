#include "viewport/ViewCameraOps.h"
#include "scene/CameraLens.h"
#include "viewport/Viewport.h"

void SetLookThrough(const state::Registry &r, ViewportState &state, state::Entity target) {
    if (!HasLens(r, target) || state.LookThroughEntity == target) return;
    // Preserve the saved view across camera switches.
    // Only capture fresh on first entry.
    if (state.LookThroughEntity == state::Null) state.Camera.LookThroughSaved = state.Camera.Active;
    state.LookThroughEntity = target;
}

void ClearLookThrough(ViewportState &state) {
    if (state.LookThroughEntity == state::Null) return;
    if (state.Camera.LookThroughSaved) state.Camera.Active = *state.Camera.LookThroughSaved;
    state.LookThroughEntity = state::Null;
}
