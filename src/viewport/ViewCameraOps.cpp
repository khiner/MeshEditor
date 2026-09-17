#include "viewport/ViewCameraOps.h"
#include "scene/CameraLens.h"
#include "state/Scene.h"

state::Entity LookThroughCameraEntity(const state::Scene &r) {
    auto view = r.view<LookingThrough>();
    return view.empty() ? state::Null : *view.begin();
}

void SetLookThrough(state::Scene &r, state::Entity viewport, state::Entity target) {
    if (!HasLens(r, target)) return;
    const auto previous = LookThroughCameraEntity(r);
    if (previous == target) return;

    // Preserve the saved view across camera switches. Only capture fresh on first entry.
    auto saved = previous != state::Null ? r.get<LookingThrough>(previous).SavedViewCamera : r.get<ViewCamera>(viewport);
    if (previous != state::Null) r.remove<LookingThrough>(previous);
    r.emplace<LookingThrough>(target, std::move(saved));
}

void ClearLookThrough(state::Scene &r, state::Entity viewport) {
    if (const auto camera = LookThroughCameraEntity(r); camera != state::Null) {
        r.replace<ViewCamera>(viewport, r.get<LookingThrough>(camera).SavedViewCamera);
        r.remove<LookingThrough>(camera);
    }
}

ViewCameraState GetViewCameraState(const state::Scene &r, state::Entity viewport) {
    ViewCameraState state{r.get<ViewCamera>(viewport), std::nullopt};
    if (const auto e = LookThroughCameraEntity(r); e != state::Null) state.LookThroughSaved = r.get<LookingThrough>(e).SavedViewCamera;
    return state;
}

void SetViewCameraState(state::Scene &r, state::Entity viewport, ViewCameraState state) {
    r.emplace_or_replace<ViewCamera>(viewport, std::move(state.Active));
    if (state.LookThroughSaved) {
        if (const auto e = LookThroughCameraEntity(r); e != state::Null) r.replace<LookingThrough>(e, LookingThrough{std::move(*state.LookThroughSaved)});
    }
}
