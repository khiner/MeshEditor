#include "viewport/ViewCameraOps.h"

#include <entt/entity/registry.hpp>

entt::entity LookThroughCameraEntity(const entt::registry &r) {
    auto view = r.view<LookingThrough>();
    return view.empty() ? entt::null : *view.begin();
}

void SetLookThrough(entt::registry &r, entt::entity viewport, entt::entity target) {
    if (!r.all_of<Camera>(target)) return;
    const auto previous = LookThroughCameraEntity(r);
    if (previous == target) return;

    // Preserve the saved view across camera switches. Only capture fresh on first entry.
    auto saved = previous != entt::null ? r.get<LookingThrough>(previous).SavedViewCamera : r.get<ViewCamera>(viewport);
    if (previous != entt::null) r.remove<LookingThrough>(previous);
    r.emplace<LookingThrough>(target, std::move(saved));
}

void ClearLookThrough(entt::registry &r, entt::entity viewport) {
    if (const auto camera = LookThroughCameraEntity(r); camera != entt::null) {
        r.replace<ViewCamera>(viewport, r.get<LookingThrough>(camera).SavedViewCamera);
        r.remove<LookingThrough>(camera);
    }
}

ViewCameraState GetViewCameraState(const entt::registry &r, entt::entity viewport) {
    ViewCameraState state{r.get<ViewCamera>(viewport), std::nullopt};
    if (const auto e = LookThroughCameraEntity(r); e != entt::null) state.LookThroughSaved = r.get<LookingThrough>(e).SavedViewCamera;
    return state;
}

void SetViewCameraState(entt::registry &r, entt::entity viewport, ViewCameraState state) {
    r.emplace_or_replace<ViewCamera>(viewport, std::move(state.Active));
    if (state.LookThroughSaved) {
        if (const auto e = LookThroughCameraEntity(r); e != entt::null) r.replace<LookingThrough>(e, LookingThrough{std::move(*state.LookThroughSaved)});
    }
}
