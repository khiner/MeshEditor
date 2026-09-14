#pragma once

#include "state/Entity.h"

struct Instance {
    state::Entity Entity;
};

// Canonical per-object visibility, present == hidden (sparse, since most objects are visible).
// RenderInstance is reactively created for an Instance without Hidden, and removed when Hidden appears.
struct Hidden {};
