#include "viewport/ViewCameraOps.h"
#include "scene/CameraLens.h"

state::Entity LookThroughCameraEntity(const state::Registry &r) {
    const auto view = r.view<LookingThrough>();
    return view.empty() ? state::Null : *view.begin();
}

void SetLookThrough(state::Registry &r, state::Entity viewport, state::Entity target) {
    if (!HasLens(r, target)) return;
    const auto previous = LookThroughCameraEntity(r);
    if (previous == target) return;

    // Preserve the saved view across camera switches.
    // Only capture fresh on first entry.
    if (previous == state::Null) r.emplace_or_replace<SavedViewCamera>(viewport, r.get<ViewCamera>(viewport));
    if (previous != state::Null) r.remove<LookingThrough>(previous);
    r.emplace<LookingThrough>(target);
}

void ClearLookThrough(state::Registry &r, state::Entity viewport) {
    if (const auto camera = LookThroughCameraEntity(r); camera != state::Null) {
        if (const auto *saved = r.try_get<SavedViewCamera>(viewport)) r.replace<ViewCamera>(viewport, saved->Value);
        r.remove<LookingThrough>(camera);
    }
}

ViewCameraState GetViewCameraState(const state::Registry &r, state::Entity viewport) {
    ViewCameraState state{r.get<ViewCamera>(viewport), std::nullopt};
    if (const auto *saved = r.try_get<SavedViewCamera>(viewport)) state.LookThroughSaved = saved->Value;
    return state;
}

void SetViewCameraState(state::Registry &r, state::Entity viewport, ViewCameraState state) {
    r.emplace_or_replace<ViewCamera>(viewport, std::move(state.Active));
    if (state.LookThroughSaved) r.emplace_or_replace<SavedViewCamera>(viewport, std::move(*state.LookThroughSaved));
    else r.remove<SavedViewCamera>(viewport);
}
