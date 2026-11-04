#include "viewport/ViewCameraOps.h"
#include "Camera.h"
#include "scene/CameraLens.h"
#include "scene/SceneGraph.h"
#include "viewport/RenderExtent.h"
#include "viewport/Viewport.h"

void SetLookThrough(const state::Registry &r, ViewCameraState &state, state::Entity target) {
    if (!HasLens(r, target) || state.LookThrough == target) return;
    if (state.LookThrough == state::Null) state.LookThroughSaved = state.Active;
    state.LookThrough = target;
}

void ClearLookThrough(ViewCameraState &state) {
    if (state.LookThrough == state::Null) return;
    if (state.LookThroughSaved) state.Active = *state.LookThroughSaved;
    state.LookThrough = state::Null;
}

void LeaveLookThrough(ViewCameraState &state, const ViewCamera &shown) {
    if (state.LookThrough == state::Null) return;
    const auto &previous = state.LookThroughSaved ? *state.LookThroughSaved : state.Active;
    const auto position = shown.Position(), away = shown.Forward();
    const float depth = Dot(position - previous.Target, away);
    state.Active = shown;
    state.Active.Distance = depth > std::numeric_limits<float>::epsilon() ? depth : previous.Distance;
    state.Active.Target = position - away * state.Active.Distance;
    state.LookThrough = state::Null;
    state.LookThroughSaved.reset();
}

state::Entity LookThroughCamera(const state::Registry &r, state::Entity viewport) {
    const auto camera = r.get<const ViewCameraState>(viewport).LookThrough;
    return camera != state::Null && HasLens(r, camera) && InScene(r, camera, SceneOf(r, viewport)) ? camera : state::Null;
}

ViewCamera ViewCameraOf(const state::Registry &r, state::Entity viewport) {
    const auto &active = r.get<const ViewCameraState>(viewport).Active;
    const auto camera = LookThroughCamera(r, viewport);
    if (camera == state::Null) return active;
    const auto lens = WidenForLookThrough(*LensOf(r, camera), AspectRatio(RenderExtentPx(r, viewport)));
    if (active.IsAnimating()) {
        auto view = active;
        view.Data = lens;
        return view;
    }
    const auto &world = *WorldTransformOf(r, camera);
    return ViewCamera{world.P, world.R, lens};
}
