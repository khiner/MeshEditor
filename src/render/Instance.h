#pragma once

#include <entt/entity/fwd.hpp>

// Link to the entity holding shared GPU buffers
struct Instance {
    entt::entity Entity;
};

// Canonical per-object visibility, present == hidden (sparse, since most objects are visible).
// RenderInstance is reactively created for an Instance without Hidden, and removed when Hidden appears.
struct Hidden {};
