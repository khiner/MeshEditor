#include "viewport/ViewCameraOps.h"
#include "scene/CameraLens.h"
#include "scene/SceneGraph.h"
#include "state/Registry.h"
#include "viewport/Viewport.h"

void SetLookThrough(const state::Registry &r, ViewCameraState &state, state::Entity target) {
    if (!HasLens(r, target) || state.LookThrough == target) return;
    // Preserve the saved view across camera switches.
    // Only capture fresh on first entry.
    if (state.LookThrough == state::Null) state.LookThroughSaved = state.Active;
    state.LookThrough = target;
}

void ClearLookThrough(ViewCameraState &state) {
    if (state.LookThrough == state::Null) return;
    if (state.LookThroughSaved) state.Active = *state.LookThroughSaved;
    state.LookThrough = state::Null;
}

state::Entity LookThroughCamera(const state::Registry &r, state::Entity viewport) {
    const auto camera = r.get<const ViewCameraState>(viewport).LookThrough;
    return camera != state::Null && HasLens(r, camera) && InScene(r, camera, SceneOf(r, viewport)) ? camera : state::Null;
}
