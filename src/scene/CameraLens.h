#pragma once

#include "CameraTypes.h"
#include "state/Scene.h"

#include <optional>

// A camera object's lens component, Perspective or Orthographic.
inline bool HasLens(const state::Scene &r, state::Entity e) { return r.any_of<Perspective, Orthographic>(e); }
inline std::optional<CameraLens> LensOf(const state::Scene &r, state::Entity e) {
    if (const auto *perspective = r.try_get<const Perspective>(e)) return *perspective;
    if (const auto *orthographic = r.try_get<const Orthographic>(e)) return *orthographic;
    return {};
}
// Installs `lens` as the entity's one lens component.
inline void SetLens(state::Scene &r, state::Entity e, const CameraLens &lens) {
    if (const auto *perspective = std::get_if<Perspective>(&lens)) {
        r.remove<Orthographic>(e);
        r.emplace_or_replace<Perspective>(e, *perspective);
    } else {
        r.remove<Perspective>(e);
        r.emplace_or_replace<Orthographic>(e, std::get<Orthographic>(lens));
    }
}
